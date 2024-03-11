import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.linalg import eigh


def save_train_log(train_log: Dict[str, List[float]], checkpoint_path: str) -> None:
    """
    Saves train log as csv file in checkpoint path.

    Args:
        train_log (Dict[str, List[float]]): Train log, containing loss and accuracy progression.
        checkpoint_path (str): Checkpoint path.
    """

    # Create results dir, if not existing
    os.makedirs(checkpoint_path, exist_ok=True)

    df = pd.DataFrame.from_dict(train_log, orient='index').transpose()
    df.to_csv(str(Path(checkpoint_path, "train_log.csv")))


def get_save_measure(experiment_name: str, num_classes: int) -> Callable:
    """
    Returns function that saves measures for a particular experiment.

    Args:
        experiment_name (str): Experiment name.
        num_classes (int): Number of classes.

    Raises:
        ValueError: Unsupported experiment name for measure saving.

    Returns:
        Callable: Function that saves measures.
    """

    if experiment_name == "frobenius":
        return save_f_distance
    elif experiment_name == "eigen":
        return partial(save_eigen, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported experiment name for measure saving: {experiment_name}")


def save_f_distance(
    GGN_1: jax.Array,
    GGN_2: jax.Array,
    step_idx: int,
    results_path: str,
    sample_size: int,
) -> None:
    """
    Saves Frobenius distance between GGNs.
    N: Batch dimension.
    D: Parameter dimension.

    Args:
        GGN_1 (jax.Array): Sample-approximate GGN [N, D, D].
        GGN_2 (jax.Array): Total GGN [D, D].
        step_idx (int): Number of completed training steps across epochs.
        results_path (str): Results path.
        sample_size (int): Sample size used for sample-approximate GGN.

    Raises:
        ValueError: Unsupported GGN dimensionalities.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    # Enable broadcasting
    with jax.default_device(jax.devices("cpu")[0]):
        if GGN_1.ndim == 2 and GGN_2.ndim == 3:
            GGN_1 = GGN_1[None, ...]
        elif GGN_1.ndim == 3 and GGN_2.ndim == 2:
            GGN_2 = GGN_2[None, ...]
        elif GGN_1.ndim == 3 and GGN_2.ndim == 3:
            GGN_1 = GGN_1[None, ...]
            GGN_2 = GGN_2[:, None, ...]
        else:
            raise ValueError(f"Unsupported GGN dimensionalities: {GGN_1.shape}, {GGN_2.shape}")

        GGN_1 = jax.device_put(GGN_1, jax.devices('cpu')[0])
        GGN_2 = jax.device_put(GGN_2, jax.devices('cpu')[0])
        f_norm = jnp.linalg.norm(GGN_1 - GGN_2, ord="fro", axis=(-2, -1))
        f_norm_rel = f_norm / jnp.linalg.norm(GGN_2, ord="fro", axis=(-2, -1))

        jnp.save(
            str(Path(results_path, f"f_norm_{sample_size}_sampled_{step_idx}.npy")),
            f_norm,
        )
        jnp.save(
            str(Path(results_path, f"f_norm_rel_{sample_size}_sampled_{step_idx}.npy")),
            f_norm_rel,
        )


def compute_eigen(
    GGN_1: jax.Array, GGN_2: jax.Array, num_classes: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes eigenvalues and top-C eigenspace overlap.
    D: Parameter dimension.
    C: Class dimension.

    Args:
        GGN_1 (jax.Array): Sample-approximate GGN [D, D].
        GGN_2 (jax.Array): Total GGN [D, D].
        num_classes (int): Number of classes.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Top-C eigenspace overlap [1],
            Eigenvalues of sample-approximate GGN [D],
            Eigenvalues of total GGN [D],
    """

    eigw_1, eigv_1 = eigh(GGN_1)
    eigw_2, eigv_2 = eigh(GGN_2)

    # Eigenvalues are in ascending order
    eigv_1 = jnp.flip(eigv_1[:, -num_classes:], axis=1)
    eigv_2 = jnp.flip(eigv_2[:, -num_classes:], axis=1)

    eigw_1 = jnp.flip(eigw_1)
    eigw_2 = jnp.flip(eigw_2)

    return (
        jnp.einsum("ij,ij->", eigv_1 @ eigv_1.T, eigv_2 @ eigv_2.T)
        / jnp.sqrt(jnp.einsum("ij,ij->", eigv_1, eigv_1) * jnp.einsum("ij,ij->", eigv_2, eigv_2)),
        eigw_1,
        eigw_2,
    )


def save_eigen(
    GGN_1: jax.Array,
    GGN_2: jax.Array,
    step_idx: int,
    results_path: str,
    sample_size: int,
    num_classes: int,
) -> None:
    """
    Saves eigenvalues and top-C eigenspace overlap of/between GGNs.
    N: Batch dimension.
    D: Parameter dimension.

    Args:
        GGN_1 (jax.Array): Sample-approximate GGN [N, D, D].
        GGN_2 (jax.Array): Total GGN [D, D].
        step_idx (int): Number of completed training steps across epochs.
        results_path (str): Results path.
        sample_size (int): Sample size used for sample-approximate GGN.
        num_classes (int): Number of classes

    Raises:
        ValueError: Unsupported GGN dimensionalities.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    with jax.default_device(jax.devices("cpu")[0]):
        if GGN_1.ndim == 2 and GGN_2.ndim == 3:
            GGN_1 = GGN_1[None, ...]

        elif GGN_1.ndim == 3 and GGN_2.ndim == 2:
            GGN_2 = GGN_2[None, ...]
        elif GGN_1.ndim == 3 and GGN_2.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported GGN dimensionalities: {GGN_1.shape}, {GGN_2.shape}")

    compute_eigen_jit = jax.jit(
        partial(compute_eigen, num_classes=num_classes),
        device=jax.devices("cpu")[0],
    )
    compute_eigen_vmap = jax.vmap(
        jax.vmap(compute_eigen_jit, in_axes=(0, None)),
        in_axes=(None, 0),
    )

    topc_eigenspace_overlap, eigw_1, eigw_2 = compute_eigen_vmap(
        GGN_1,
        GGN_2,
    )

    jnp.save(
        str(Path(results_path, f"eig_overlap_{sample_size}_sampled_{step_idx}.npy")),
        topc_eigenspace_overlap,
    )
    jnp.save(
        str(Path(results_path, f"eigvals_{sample_size}_sampled_{step_idx}.npy")),
        eigw_1,
    )
    jnp.save(
        str(Path(results_path, f"eigvals_total_{step_idx}.npy")),
        eigw_2,
    )


def save_ltk(
    LTK: jax.Array,
    step_idx: int,
    results_path: str,
    sample_size: int | None = None,
) -> None:
    """
    Saves LTKs on disk.
    C: Class dim.
    N: Number of LTK samples.
    M: Number of test datapoints.

    Args:
        LTK(jax.Array): Total LTK ([M, C, C]) or LTK samples ([N, M, C, C]).
        step_idx (int): Training step index.
        results_path (str): Results path.
        sample_size (int | None, optional): Sample size (only needed, if LTK samples passed). Defaults to None.

    Raises:
        ValueError: No sample size passed for LTK samples.
        ValueError: Unsupported LTK dimensionality.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    # Total LTK ([M, C, C])
    if LTK.ndim == 3:
        jnp.save(
            str(Path(results_path, f"LTK_total_{step_idx}.npy")),
            LTK,
        )
    # LTK samples ([N, M, C, C])
    elif LTK.ndim == 4:
        if sample_size is None:
            raise ValueError(f"LTK samples requires sample size argument.")

        jnp.save(
            str(Path(results_path, f"LTK_{sample_size}_sampled_{step_idx}.npy")),
            LTK,
        )
    else:
        raise ValueError(f"Unsupported LTK dimensionality: {LTK.shape}")


def save_predictive_distribution(
    pred_distr: jax.Array,
    step_idx: int,
    results_path: str,
    sample_size: int | None = None,
) -> None:
    """
    Saves predictive distributions on disk.
    C: Class dim.
    N: Number of samples.
    M: Number of test datapoints.

    Args:
        pred_distr(jax.Array): Total predictive distribution ([M, C]) or predictive distribution samples ([N, M, C]).
        step_idx (int): Training step index.
        results_path (str): Results path.
        sample_size (int | None, optional): Sample size (only needed, if predictive distribution samples passed). Defaults to None.

    Raises:
        ValueError: No sample size passed for predictive distribution samples.
        ValueError: Unsupported predictive distribution dimensionality.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    # Total predictive distribution ([M, C])
    if pred_distr.ndim == 2:
        jnp.save(
            str(Path(results_path, f"pred_distr_total_{step_idx}.npy")),
            pred_distr,
        )
    # predictive distribution samples ([N, M, C])
    elif pred_distr.ndim == 3:
        if sample_size is None:
            raise ValueError(f"Predictive distribution samples requires sample size argument.")

        jnp.save(
            str(Path(results_path, f"pred_distr_{sample_size}_sampled_{step_idx}.npy")),
            pred_distr,
        )
    else:
        raise ValueError(f"Unsupported predictive distribution dimensionality: {pred_distr.shape}")


def save_ggn(
    GGN: jax.Array,
    step_idx: int,
    results_path: str,
    sample_size: int | None = None,
) -> None:
    """
    Saves GGN on disk.
    D: Parameter dim.
    N: Number of GGN samples.

    Args:
        GGN(jax.Array): Total GGN ([D, D]) or GGN samples ([N, D, D]).
        step_idx (int): Training step index.
        results_path (str): Results path.
        sample_size (int | None, optional): Sample size (only needed, if GGN samples passed). Defaults to None.

    Raises:
        ValueError: No sample size passed for GGN samples.
        ValueError: Unsupported GGN dimensionality.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    # Total GGN ([D, D])
    if GGN.ndim == 2:
        jnp.save(
            str(Path(results_path, f"GGN_total_{step_idx}.npy")),
            GGN,
        )
    # GGN samples ([N, D, D])
    elif GGN.ndim == 3:
        if sample_size is None:
            raise ValueError(f"GGN samples requires sample size argument.")

        jnp.save(
            str(Path(results_path, f"GGN_{sample_size}_sampled_{step_idx}.npy")),
            GGN,
        )
    else:
        raise ValueError(f"Unsupported GGN dimensionality: {GGN.shape}")


def load_ggn(
    step_idx: int,
    results_path: str,
    sample_size: int | None = None,
) -> jax.Array:
    """
    Loads GGN from disk.
    D: Parameter dim.
    N: Number of GGN samples.

    Args:
        step_idx (int): Training step index.
        results_path (str): Results path.
        sample_size (int | None, optional): Sample size (only needed, if GGN samples passed). Defaults to None.

    Returns:
        jax.Array: Loaded GGN [N, D, D] / [D, D].
    """

    with jax.default_device(jax.devices("cpu")[0]):
        if sample_size is None:
            return jnp.load(str(Path(results_path, f"GGN_total_{step_idx}.npy")))
        else:
            return jnp.load(str(Path(results_path, f"GGN_{sample_size}_sampled_{step_idx}.npy")))


def remove_ggn(
    step_idx: int,
    results_path: str,
    sample_size: int | None = None,
) -> None:
    """
    Removes GGN from disk.

    Args:
        step_idx (int): Training step index.
        results_path (str): Results path.
        sample_size (int | None, optional): Sample size (only needed, if GGN samples passed). Defaults to None.
    """

    if sample_size is None:
        filepath = Path(results_path, f"GGN_total_{step_idx}.npy")
    else:
        filepath = Path(results_path, f"GGN_{sample_size}_sampled_{step_idx}.npy")

    filepath.unlink(missing_ok=True)
