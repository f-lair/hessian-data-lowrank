from typing import Any, List

import jax
import numpy as np
from torch.utils import data


class DataLoader(data.DataLoader):
    """PyTorch data loader with jax-compatible collate function."""

    def __init__(self, dataset: data.Dataset, batch_size: int, sampler: data.Sampler):
        """
        Initializes data loader.

        Args:
            dataset (data.Dataset): Dataset.
            batch_size (int): Batch size.
            sampler (data.Sampler): Data sampler.
        """

        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )

    @staticmethod
    def collate_fn(batch: List[Any]) -> jax.Array:
        """
        Collate function, mapping a list of PyTorch batch items to a numpy array.
        cf. https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html#data-loading-with-pytorch

        Args:
            batch (List[Any]): List of PyTorch batch items.

        Returns:
            jax.Array: Array with batch items stacked along a new dimension.
        """

        return jax.tree_map(np.asarray, data.default_collate(batch))
