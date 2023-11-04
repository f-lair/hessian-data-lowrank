import random
from typing import Any, List

import jax
import numpy as np
import torch
from jax.tree_util import tree_map
from torch import Tensor
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms.v2.functional import resize


def get_dataset(dataset: str, train: bool, px: int, path: str) -> data.Dataset:
    """
    Returns dataset specified by CLI arguments.

    Args:
        dataset (str): Dataset name.
        train (bool): Use train split.
        px (int): Downsampled image size per side.
        path (str): Data path.

    Raises:
        ValueError: Unsupported dataset.

    Returns:
        data.Dataset: Dataset specified by CLI arguments.
    """

    if dataset == "mnist":
        return MNIST(root=path, train=train, transform=MNISTTransform(px), download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


class DataLoader(data.DataLoader):
    """Deterministicly shuffled PyTorch data loader."""

    def __init__(self, dataset: data.Dataset, batch_size: int, rng_seed: int):
        """
        Initializes data loader.

        Args:
            dataset (data.Dataset): Dataset.
            batch_size (int): Batch size.
            rng_seed (int): RNG seed.
        """

        rng = torch.manual_seed(rng_seed)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
            generator=rng,
        )

    @staticmethod
    def worker_init(worker_id: int, rng_seed: int) -> None:
        """
        Initialization method for data loader workers, which fixes the random number generator seed.

        Args:
            worker_id (int): Worker ID.
            rng_seed (int): Random number generator seed.
        """

        np.random.seed(rng_seed)
        random.seed(rng_seed)

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

        return tree_map(np.asarray, data.default_collate(batch))


class MNISTTransform:
    """Pre-processing transform for MNIST data."""

    def __init__(self, px: int) -> None:
        """
        Initializes transform.

        Args:
            px (int): Downsampled image size per side.
        """

        self.px = px

    def __call__(self, pic: Tensor) -> np.ndarray:
        """
        Applies transform.
        cf. https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html#data-loading-with-pytorch

        Args:
            pic (Tensor): Input image.

        Returns:
            np.ndarray: Transformed image.
        """

        return np.ravel(np.array(resize(pic, [self.px, self.px]), dtype=np.float32))
