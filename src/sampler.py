import math
from functools import partial
from typing import Any, Callable, Iterator, List

import jax
import numpy as np
import torch
from flax.training.train_state import TrainState
from torch import Generator
from torch.utils import data

from data_loader import DataLoader


class LossSampler(data.Sampler[int]):
    """PyTorch data sampler with loss-based sampling."""

    def __init__(
        self,
        data_source: data.Dataset,
        rng: Generator,
        step_fn: Callable,
        batch_size: int,
        inverse: bool,
        replacement: bool,
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
        """

        self.data_source = data_source
        self.rng = rng
        self.step_fn = jax.jit(partial(step_fn, n_classes=len(data_source.classes)))  # type: ignore
        self.batch_size = batch_size
        self.inverse = inverse
        self.replacement = replacement

        self.data_len = len(self.data_source)  # type: ignore
        self.loss_weights = torch.ones((self.data_len,))
        self.batch_indices = [
            list(range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.data_len)))
            for idx in range(math.ceil(self.data_len / self.batch_size))
        ]

    def update(self, state: TrainState) -> None:
        """
        Updates loss values by forward pass over the whole dataset.

        Args:
            state (TrainState): Current training state.
        """

        for indices in self.batch_indices:
            batch = DataLoader.collate_fn([self.data_source[idx] for idx in indices])
            loss, _, _ = self.step_fn(state, batch)
            self.loss_weights[indices] = torch.from_numpy(np.array(loss))

        self.loss_weights += 1e-8

        if self.inverse:
            self.loss_weights = 1.0 / self.loss_weights

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

        perm = torch.multinomial(
            self.loss_weights, self.data_len, self.replacement, generator=self.rng
        )
        yield from perm.tolist()
