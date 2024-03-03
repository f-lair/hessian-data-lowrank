import math
import time
from functools import partial
from typing import Callable, Dict, Iterator, Tuple

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
        replacement_stride: int,
        inverse: bool,
        classwise: bool,
        classeq: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            batch_size (int): Batch size for loss computations.
            replacement_stride (int): Number of consecutive sampled datapoints that can be identical.
            inverse (bool): If set, samples data items with lowest weight with highest probability.
            classwise (bool): If set, samples class weight-based and intra-class uniformly.
            classeq (bool): If set, samples class uniformly and intra-class weight-based.
            no_progress_bar (bool): Disables progress bar.
        """

        self.data_source = data_source
        self.rng = rng
        self.batch_size = batch_size
        self.replacement_stride = replacement_stride
        self.inverse = inverse
        self.classwise = classwise
        self.classeq = classeq

        assert not (self.classwise and self.classeq), "Invalid WeightedSampler configuration!"

        self.no_progress_bar = no_progress_bar

        self.data_len = len(self.data_source)  # type: ignore
        self.batch_indices = [
            list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.data_len)))
            for idx in range(math.ceil(self.data_len / self.batch_size))
        ]

        if self.classwise:
            self.class_assignments, self.class_counts = self._get_class_assignments()
            self.weights = torch.ones((len(self.class_assignments),))
        elif self.classeq:
            self.class_assignments, self.class_counts = self._get_class_assignments()
            self.weights = torch.ones((self.data_len,))
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

    def _get_class_assignments(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Sweep over dataset to get data indices for each class index.

        Returns:
            Tuple[Dict[int, torch.Tensor], torch.Tensor]: Class index -> data indices; counts per class.
        """

        class_assignments = {class_idx: [] for class_idx in range(len(self.data_source.classes))}  # type: ignore
        counts = torch.zeros((len(class_assignments),), dtype=torch.int)

        for data_idx in range(self.data_len):
            _, class_idx = self.data_source[data_idx]
            class_assignments[class_idx].append(data_idx)
            counts[class_idx] += 1

        return {
            class_idx: torch.tensor(data_indices, dtype=torch.int64)
            for class_idx, data_indices in class_assignments.items()
        }, counts

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

        # torch.save(self.weights, "../results/distr/gradnorm_class_inv_distr.pth")

    @staticmethod
    def multinomial_limited(
        weights: torch.Tensor, n_samples: int, limits: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        limits_ = limits.clone()
        limits_mask = limits_ > 0
        if torch.count_nonzero(limits_mask) > 1:
            limits_min = int(torch.amin(limits_[limits_mask]).item())
        else:
            limits_min = n_samples
        weights_limited = weights.clone()
        remaining_samples = n_samples
        samples_buffer = []

        while remaining_samples > 0:
            weights_limited[~limits_mask] = 0.0
            samples = torch.multinomial(weights_limited, limits_min, True, generator=generator)
            samples_buffer.append(samples)
            remaining_samples -= limits_min

            counts = torch.bincount(samples, minlength=len(limits_))
            limits_ -= counts
            limits_mask = limits_ > 0
            if torch.count_nonzero(limits_mask) > 1:
                limits_min = int(torch.amin(limits_[limits_mask]).item())
            else:
                limits_min = remaining_samples

        return torch.concat(samples_buffer)

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

        if self.classwise or self.classeq:
            perms = []
            for idx in range(self.replacement_stride):
                if self.classwise:
                    # Weighted inter-class distribution
                    perm_interclass = self.multinomial_limited(
                        self.weights, self.data_len, self.class_counts, self.rng
                    )
                    # Uniform intra-class distribution
                    perm_intraclass = {
                        class_idx: torch.randperm(len(data_indices), generator=self.rng)
                        for class_idx, data_indices in self.class_assignments.items()
                    }
                else:
                    # Uniform inter-class distribution
                    perm_interclass = self.multinomial_limited(
                        torch.ones((len(self.class_assignments),)),
                        self.data_len,
                        self.class_counts,
                        self.rng,
                    )
                    # Weighted intra-class distribution
                    perm_intraclass = {
                        class_idx: torch.multinomial(
                            self.weights[data_indices],
                            int(self.class_counts[class_idx].item()),
                            False,
                            generator=self.rng,
                        )
                        for class_idx, data_indices in self.class_assignments.items()
                    }
                # Map first from inter-class sampled indices to intra-class sampled indices, then to actual data indices
                counter_intraclass = torch.zeros_like(self.class_counts)
                perm = torch.zeros((self.data_len,), dtype=torch.int64)
                for data_idx in range(self.data_len):
                    class_idx = int(perm_interclass[data_idx].item())
                    intraclass_idx = int(
                        perm_intraclass[class_idx][
                            int(counter_intraclass[class_idx].item())
                        ].item()
                    )
                    perm[data_idx] = self.class_assignments[class_idx][intraclass_idx]
                    counter_intraclass[class_idx] += 1
                perms.append(perm)
        else:
            perms = [
                torch.multinomial(
                    self.weights,
                    self.data_len,
                    False,
                    generator=self.rng,
                )
                for idx in range(self.replacement_stride)
            ]

        perm = torch.flatten(torch.stack(perms, dim=1))
        yield from perm.tolist()


class LossSampler(WeightedSampler):
    """PyTorch data sampler with loss-based sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        step_fn: Callable,
        batch_size: int,
        replacement_stride: int,
        inverse: bool,
        classwise: bool,
        classeq: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            step_fn (Callable): Jax function that takes a training state and batch and computes the loss.
            batch_size (int): Batch size for loss computations.
            replacement_stride (int): Number of consecutive sampled datapoints that can be identical.
            inverse (bool): If set, samples data items with lowest loss with highest probability.
            classwise (bool): If set, samples class loss-based and intra-class uniformly.
            classeq (bool): If set, samples class uniformly and intra-class weight-based.
            no_progress_bar (bool): Disables progress bar.
        """

        super().__init__(
            data_source,
            rng,
            batch_size,
            replacement_stride,
            inverse,
            classwise,
            classeq,
            no_progress_bar,
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
        replacement_stride: int,
        inverse: bool,
        classwise: bool,
        classeq: bool,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            step_fn (Callable): Jax function that takes a training state and batch and computes the gradient.
            batch_size (int): Batch size for gradient computations.
            replacement_stride (int): Number of consecutive sampled datapoints that can be identical.
            inverse (bool): If set, samples data items with lowest gradient-norm with highest probability.
            classwise (bool): If set, samples class loss-based and intra-class uniformly.
            classeq (bool): If set, samples class uniformly and intra-class weight-based.
            no_progress_bar (bool): Disables progress bar.
        """

        super().__init__(
            data_source,
            rng,
            batch_size,
            replacement_stride,
            inverse,
            classwise,
            classeq,
            no_progress_bar,
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


class WeightedBinnedSampler(data.Sampler[int]):
    """PyTorch data sampler with weight-based binned sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        step_fn: Callable,
        batch_size: int,
        replacement_stride: int,
        num_bins: int,
        no_progress_bar: bool,
    ) -> None:
        """
        Initializes data sampler.

        Args:
            data_source (data.Dataset): Dataset.
            rng (Generator): Random number generator.
            step_fn (Callable): Jax function that takes a training state and batch and computes the loss.
            batch_size (int): Batch size for loss computations.
            replacement_stride (int): Number of consecutive sampled datapoints that can be identical.
            num_bins (int): Number of bins.
            no_progress_bar (bool): Disables progress bar.
        """

        self.data_source = data_source
        self.rng = rng
        self.step_fn = jax.jit(partial(step_fn, n_classes=len(data_source.classes)))
        self.batch_size = batch_size
        self.replacement_stride = replacement_stride
        self.num_bins = num_bins

        self.no_progress_bar = no_progress_bar

        self.data_len = len(self.data_source)  # type: ignore
        self.batch_indices = [
            list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.data_len)))
            for idx in range(math.ceil(self.data_len / self.batch_size))
        ]

        self.weights = torch.ones((self.data_len,))

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

    def update(self, state: TrainState) -> None:
        """
        Updates weights.

        Args:
            state (TrainState): Current training state.
        """

        self.weights = self._get_updated_weights(state) + 1e-8

    def update_binned_indices(self):
        bin_range = (
            torch.log10(torch.amin(self.weights)).item(),
            torch.log10(torch.amax(self.weights)).item(),
        )

        bins = torch.logspace(bin_range[0], bin_range[1], self.num_bins + 1)
        bins[-1] = torch.inf
        binned_weights = torch.bucketize(self.weights, bins, right=True)
        self.binned_indices = [
            torch.nonzero(binned_weights == bin_idx)[:, 0]
            for bin_idx in range(1, self.num_bins + 1)
        ]

    def __len__(self) -> int:
        """
        Returns number of samples, being equal to the number of bins.

        Returns:
            int: Number of samples.
        """

        return self.num_bins * self.replacement_stride

    def __iter__(self) -> Iterator[int]:
        """
        Builds iterator over sampled data items according to loss-based probability distribution.

        Yields:
            Iterator[int]: Iterator over sampled data items.
        """

        self.update_binned_indices()

        perms = [
            (
                self.binned_indices[bin_idx][
                    torch.multinomial(
                        torch.ones((len(self.binned_indices[bin_idx]),)),
                        self.replacement_stride,
                        True,
                    )
                ]
                if len(self.binned_indices[bin_idx]) > 0
                else torch.zeros((self.replacement_stride,), dtype=torch.int64)
            )
            for bin_idx in range(self.num_bins)
        ]

        perm = torch.cat(perms)
        yield from perm.tolist()
