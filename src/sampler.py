import math
import time
from functools import partial
from typing import Callable, Dict, Iterator

import jax
import numpy as np
import torch
from flax.training.train_state import TrainState
from torch import Generator
from torch.utils import data
from tqdm import tqdm

from data_loader import DataLoader


class WeightedSampler(data.Sampler[int]):
    """PyTorch data sampler with weight-based sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        batch_size: int,
        inverse: bool,
        replacement: bool,
        classwise: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            batch_size (int): Batch size for loss computations.
            inverse (bool): If set, samples data items with lowest weight with highest probability.
            replacement (bool): If set, samples with replacement.
            classwise (bool): If set, samples class weight-based and intra-class uniformly.
            no_progress_bar (bool): Disables progress bar.
        """

        self.data_source = data_source
        self.rng = rng
        self.batch_size = batch_size
        self.inverse = inverse
        self.replacement = replacement
        self.classwise = classwise

        self.no_progress_bar = no_progress_bar

        self.data_len = len(self.data_source)  # type: ignore
        self.batch_indices = [
            list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.data_len)))
            for idx in range(math.ceil(self.data_len / self.batch_size))
        ]

        if self.classwise:
            self.class_assignments = self._get_class_assignments()
            self.weights = torch.ones((len(self.class_assignments),))
        else:
            self.weights = torch.ones((self.data_len,))

    def _get_updated_weights(self, state: TrainState) -> torch.Tensor:
        """
        Computes updated sampling weights.

        Args:
            state (TrainState): Current training state.

        Raises:
            NotImplementedError: Needs to be implemented for a concrete WeightedSampler.

        Returns:
            torch.Tensor: Updated sampling weights.
        """

        raise NotImplementedError

    def _get_class_assignments(self) -> Dict[int, torch.Tensor]:
        """
        Sweep over dataset to get data indices for each class index.

        Returns:
            Dict[int, torch.Tensor]: Class index -> data indices.
        """

        class_assignments = {class_idx: [] for class_idx in range(len(self.data_source.classes))}  # type: ignore
        for data_idx in range(self.data_len):
            _, class_idx = self.data_source[data_idx]
            class_assignments[class_idx].append(data_idx)

        return {
            class_idx: torch.tensor(data_indices, dtype=torch.int64)
            for class_idx, data_indices in class_assignments.items()
        }

    def update(self, state: TrainState) -> None:
        """
        Updates weights.

        Args:
            state (TrainState): Current training state.
        """

        weights = self._get_updated_weights(state)

        if self.classwise:
            for class_idx, data_indices in self.class_assignments.items():
                self.weights[class_idx] = torch.mean(weights[data_indices])
        else:
            self.weights = weights

        self.weights += 1e-8

        if self.inverse:
            self.weights = 1.0 / self.weights

        torch.save(self.weights, "../results/distr/gradnorm_class_inv_distr.pth")

    def __len__(self) -> int:
        """
        Returns number of samples, being equal to the dataset size.

        Returns:
            int: Number of samples.
        """

        return self.data_len

    def __iter__(self) -> Iterator[int]:
        """
        Builds iterator over sampled data items according to loss-based probability distribution.

        Yields:
            Iterator[int]: Iterator over sampled data items.
        """

        if self.classwise:
            perm_interclass = torch.multinomial(
                self.weights, self.data_len, True, generator=self.rng
            )
            _, class_counts = torch.unique(perm_interclass, return_counts=True)
            perm_intraclass = {
                class_idx: torch.multinomial(
                    torch.ones(
                        len(data_indices),
                    ),
                    class_counts[class_idx],
                    True,
                    generator=self.rng,
                )
                for class_idx, data_indices in self.class_assignments.items()
            }
            counter_intraclass = {class_idx: 0 for class_idx in self.class_assignments.keys()}
            perm = torch.zeros_like(perm_interclass)
            for data_idx in range(self.data_len):
                class_idx = int(perm_interclass[data_idx].item())
                perm[data_idx] = perm_intraclass[class_idx][counter_intraclass[class_idx]]
                counter_intraclass[class_idx] += 1
        else:
            perm = torch.multinomial(
                self.weights, self.data_len, self.replacement, generator=self.rng
            )
        yield from perm.tolist()


class LossSampler(WeightedSampler):
    """PyTorch data sampler with loss-based sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        step_fn: Callable,
        batch_size: int,
        inverse: bool,
        replacement: bool,
        classwise: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            step_fn (Callable): Jax function that takes a training state and batch and computes the loss.
            batch_size (int): Batch size for loss computations.
            inverse (bool): If set, samples data items with lowest loss with highest probability.
            replacement (bool): If set, samples with replacement.
            classwise (bool): If set, samples class loss-based and intra-class uniformly.
            no_progress_bar (bool): Disables progress bar.
        """

        super().__init__(
            data_source, rng, batch_size, inverse, replacement, classwise, no_progress_bar
        )
        self.step_fn = jax.jit(partial(step_fn, n_classes=len(data_source.classes)))  # type: ignore

    def _get_updated_weights(self, state: TrainState) -> torch.Tensor:
        """
        Updates weights by forward pass over the whole dataset.

        Args:
            state (TrainState): Current training state.

        Returns:
            torch.Tensor: Updated sampling weights.
        """

        weights = torch.zeros((self.data_len,))

        for indices in tqdm(
            self.batch_indices, desc="Sampler Update", disable=self.no_progress_bar
        ):
            batch = DataLoader.collate_fn([self.data_source[idx] for idx in indices])
            loss, _, _ = self.step_fn(state, batch)
            weights[indices] = torch.from_numpy(np.array(loss))

        return weights


class GradnormSampler(WeightedSampler):
    """PyTorch data sampler with gradient-norm-based sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        step_fn: Callable,
        batch_size: int,
        inverse: bool,
        replacement: bool,
        classwise: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            step_fn (Callable): Jax function that takes a training state and batch and computes the gradient.
            batch_size (int): Batch size for gradient computations.
            inverse (bool): If set, samples data items with lowest gradient-norm with highest probability.
            replacement (bool): If set, samples with replacement.
            classwise (bool): If set, samples class loss-based and intra-class uniformly.
            no_progress_bar (bool): Disables progress bar.
        """

        super().__init__(
            data_source, rng, batch_size, inverse, replacement, classwise, no_progress_bar
        )
        self.step_fn = jax.jit(partial(step_fn, n_classes=len(data_source.classes), return_grad=True))  # type: ignore

    def _get_updated_weights(self, state: TrainState) -> torch.Tensor:
        """
        Updates weights by forward and backward pass over the whole dataset.

        Args:
            state (TrainState): Current training state.

        Returns:
            torch.Tensor: Updated sampling weights.
        """

        weights = torch.zeros((self.data_len,))

        for indices in tqdm(
            self.batch_indices, desc="Sampler Update", disable=self.no_progress_bar
        ):
            batch = DataLoader.collate_fn([self.data_source[idx] for idx in indices])
            _, d_loss, _, _ = self.step_fn(state, batch)
            weights[indices] = torch.from_numpy(np.linalg.norm(np.array(d_loss), ord=2, axis=1))

        return weights
