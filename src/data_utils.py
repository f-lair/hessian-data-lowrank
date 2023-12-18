from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms.v2.functional import resize

from sampler import GradnormSampler, LossSampler


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


def get_sampler(
    sampling: str,
    dataset: data.Dataset,
    rng_seed: int,
    test_step_fn: Callable,
    batch_size: int,
    no_progress_bar: bool,
) -> data.Sampler:
    """
    Returns data sampler specified by CLI arguments.

    Args:
        sampling (str): Sampling method.
        dataset (data.Dataset): Dataset.
        rng_seed (int): RNG seed.
        test_step_fn (Callable): Step function taking a train state and data batch and yielding the loss.
        batch_size (int): Batch size used for loss computations.
        no_progress_bar (bool): Disables progress bar.

    Raises:
        ValueError: Unsupported sampling method.

    Returns:
        data.Sampler: Data sampler specified by CLI arguments.
    """

    rng = torch.manual_seed(rng_seed)

    if sampling == "uniform":
        return data.RandomSampler(dataset, generator=rng)  # type: ignore
    elif sampling == "loss":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-inv":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-class":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=True,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-classeq":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=False,
            classeq=True,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-class-inv":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=True,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-classeq-inv":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=False,
            classeq=True,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-rep":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=True,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "loss-inv-rep":
        return LossSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=True,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-inv":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-class":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=True,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-classeq":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=False,
            classwise=False,
            classeq=True,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-class-inv":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=True,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-classeq-inv":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=False,
            classwise=False,
            classeq=True,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-rep":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=False,
            replacement=True,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    elif sampling == "gradnorm-inv-rep":
        return GradnormSampler(
            dataset,
            rng,
            test_step_fn,
            batch_size,
            inverse=True,
            replacement=True,
            classwise=False,
            classeq=False,
            no_progress_bar=no_progress_bar,
        )
    else:
        raise ValueError(f"Unsupported sampling: {sampling}")


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
