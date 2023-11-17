import os
from pathlib import Path

import jax
from jax import numpy as jnp


def save_results(
    GGN: jax.Array,
    step_idx: int,
    results_path: str,
    batch_size: int | None = None,
) -> None:
    """
    Saves results on disk.
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
