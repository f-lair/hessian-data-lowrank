import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import jax
import pandas as pd
from jax import numpy as jnp
from jax import scipy as jsp
from jax.experimental.sparse.linalg import lobpcg_standard


def save_train_log(train_log: Dict[str, List[float]], results_path: str) -> None:
    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    df = pd.DataFrame.from_dict(train_log, orient='index').transpose()
    df.to_csv(str(Path(results_path, "train_log.csv")))


def get_save_measure(measure: str, num_classes: int, compose_on_cpu: bool) -> Callable:
    if measure == "frobenius":
        return save_f_norm
    elif measure == "eig-overlap":
        return partial(
            save_topc_eigenspace_overlap, num_classes=num_classes, compose_on_cpu=compose_on_cpu
        )
    else:
        raise ValueError(f"Unsupported measure: {measure}")


def save_f_norm(
    GGN_1: jax.Array,
    GGN_2: jax.Array,
    prng_key: jax.random.KeyArray,
    step_idx: int,
    results_path: str,
    batch_size: int,
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


def compute_topc_eigenspace_overlap(
    GGN_1: jax.Array, GGN_2: jax.Array, prng_key: jax.random.KeyArray, num_classes: int
) -> jax.Array:
    D, _ = GGN_1.shape

    ### vvv ### Not implemented in JAX! ### vvv ###
    # _, eigv_1 = jsp.linalg.eigh(
    #     GGN_1, eigvals=(D - num_classes, D - 1)  # type: ignore
    # )  # P^{U}: [D, C]
    # _, eigv_2 = jsp.linalg.eigh(
    #     GGN_2, eigvals=(D - num_classes, D - 1)  # type: ignore
    # )  # P^{V}: [D, C]
    ### ^^^ ### Not implemented in JAX! ### ^^^ ###

    # Use approximative LOBPCG instead
    prng_key_1, prng_key_2 = jax.random.split(prng_key)
    eigv_1 = jax.random.normal(prng_key_1, (D, num_classes))
    eigv_2 = jax.random.normal(prng_key_2, (D, num_classes))
    _, eigv_1, _ = lobpcg_standard(GGN_1, eigv_1)
    _, eigv_2, _ = lobpcg_standard(GGN_2, eigv_2)

    # Tr(P^{U} @ P^{V}) / (Tr(P^{U}) * Tr(P^{V}))^(-0.5)
    return jnp.einsum("ij,ij->", eigv_1 @ eigv_1.T, eigv_2 @ eigv_2.T) / jnp.sqrt(
        jnp.einsum("ij,ij->", eigv_1, eigv_1) * jnp.einsum("ij,ij->", eigv_2, eigv_2)
    )


def save_topc_eigenspace_overlap(
    GGN_1: jax.Array,
    GGN_2: jax.Array,
    prng_key: jax.random.KeyArray,
    step_idx: int,
    results_path: str,
    batch_size: int,
    num_classes: int,
    compose_on_cpu: bool,
) -> None:
    # Create results dir, if not existing
    os.makedirs(results_path, exist_ok=True)

    device = jax.devices("cpu")[0] if compose_on_cpu else None
    with jax.default_device(device):
        if GGN_1.ndim == 2 and GGN_2.ndim == 3:
            GGN_1 = GGN_1[None, ...]

        elif GGN_1.ndim == 3 and GGN_2.ndim == 2:
            GGN_2 = GGN_2[None, ...]
        elif GGN_1.ndim == 3 and GGN_2.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported GGN dimensionalities: {GGN_1.shape}, {GGN_2.shape}")
    prng_key_vmap = jax.random.split(prng_key, (GGN_1.shape[0], GGN_2.shape[0]))

    compute_topc_eigenspace_overlap_jit = jax.jit(
        partial(compute_topc_eigenspace_overlap, num_classes=num_classes), device=device
    )
    compute_topc_eigenspace_overlap_vmap = jax.vmap(
        jax.vmap(compute_topc_eigenspace_overlap_jit, in_axes=(0, None, 0)),
        in_axes=(None, 0, 1),
    )

    topc_eigenspace_overlap = compute_topc_eigenspace_overlap_vmap(
        GGN_1,
        GGN_2,
        prng_key_vmap,
    )

    jnp.save(
        str(Path(results_path, f"eig_overlap_{batch_size}_batched_{step_idx}.npy")),
        topc_eigenspace_overlap,
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
