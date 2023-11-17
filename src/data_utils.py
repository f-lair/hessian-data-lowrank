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


def get_sampler(sampling: str, dataset: data.Dataset, rng_seed: int) -> data.Sampler:
    """
    Returns data sampler specified by CLI arguments.

    Args:
        sampling (str): Sampling method.
        dataset (data.Dataset): Dataset.
        rng_seed (int): RNG seed.

    Raises:
        ValueError: Unsupported sampling method.

    Returns:
        data.Sampler: Data sampler specified by CLI arguments.
    """

    rng = torch.manual_seed(rng_seed)

    if sampling == "uniform":
        return data.RandomSampler(dataset, generator=rng)  # type: ignore
    else:
        raise ValueError(f"Unsupported sampling: {sampling}")


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
