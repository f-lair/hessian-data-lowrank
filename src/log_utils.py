import os
from pathlib import Path
from typing import List

import jax
from jax import numpy as jnp


def save_results(
    GGN_batched: List[jax.Array],
    GGN_total: jax.Array | None,
    batch_size: int,
    epoch_idx: int,
    results_path: str,
):
    """
    Saves results on disk.
    D: Parameter dim.

    Args:
        GGN_batched (List[jax.Array]): Batched GGNs ('n_ggn_samples' x [D, D]).
        GGN_total (jax.Array | None): Total GGN ([D, D]).
        batch_size (int): Batch size.
        epoch_idx (int): Epoch index.
        results_path (str): Results path.
    """

    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    # Save batched GGNs as one array, stacked along new axis.
    jnp.save(
        str(Path(results_path, f"GGN_{batch_size}_batched_{epoch_idx}.npy")),
        jnp.stack(GGN_batched, axis=0),
    )

    # Save total GGN
    if GGN_total is not None:
        jnp.save(
            str(Path(results_path, f"GGN_total_{epoch_idx}.npy")),
            GGN_total,
        )
