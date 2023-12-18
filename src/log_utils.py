import os
from pathlib import Path
from typing import Dict, List

import jax
import pandas as pd
from jax import numpy as jnp


def save_train_log(train_log: Dict[str, List[float]], results_path: str) -> None:
    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    df = pd.DataFrame.from_dict(train_log, orient='index').transpose()
    df.to_csv(str(Path(results_path, "train_log.csv")))


def save_norm(
    GGN_1: jax.Array, GGN_2: jax.Array, step_idx: int, results_path: str, batch_size: int
) -> None:
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
    jnp.save(
        str(Path(results_path, f"f_norm_{batch_size}_batched_{step_idx}.npy")),
        f_norm,
    )


def save_ggn(
    GGN: jax.Array,
    step_idx: int,
    results_path: str,
    batch_size: int | None = None,
) -> None:
    """
    Saves GGN on disk.
    D: Parameter dim.
    N: Number of GGN samples.

    Args:
        GGN(jax.Array): Total GGN ([D, D]) or GGN samples ([N, D, D]).
        step_idx (int): Training step index.
        results_path (str): Results path.
        batch_size (int | None, optional): Batch size (only needed, if GGN samples passed). Defaults to None.

    Raises:
        ValueError: No batch size passed for GGN samples.
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
        if batch_size is None:
            raise ValueError(f"GGN samples requires batch size argument.")

        jnp.save(
            str(Path(results_path, f"GGN_{batch_size}_batched_{step_idx}.npy")),
            GGN,
        )
    else:
        raise ValueError(f"Unsupported GGN dimensionality: {GGN.shape}")


def load_ggn(
    step_idx: int,
    results_path: str,
    batch_size: int | None = None,
) -> jax.Array:
    with jax.default_device(jax.devices("cpu")[0]):
        if batch_size is None:
            return jnp.load(str(Path(results_path, f"GGN_total_{step_idx}.npy")))
        else:
            return jnp.load(str(Path(results_path, f"GGN_{batch_size}_batched_{step_idx}.npy")))


def remove_ggn(
    step_idx: int,
    results_path: str,
    batch_size: int | None = None,
) -> None:
    if batch_size is None:
        filepath = Path(results_path, f"GGN_total_{step_idx}.npy")
    else:
        filepath = Path(results_path, f"GGN_{batch_size}_batched_{step_idx}.npy")

    filepath.unlink(missing_ok=True)
