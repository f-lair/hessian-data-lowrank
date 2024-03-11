from functools import partial
from typing import Any, List, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import tree_util
from tqdm import tqdm

from data_loader import DataLoader
from log_utils import (
    get_save_measure,
    load_ggn,
    remove_ggn,
    save_ggn,
    save_ltk,
    save_predictive_distribution,
)
from sampler import WeightedSampler


@jax.jit
def compute_ggn_decomp(
    state: TrainState, batch: Tuple[jax.Array, jax.Array]
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes Jacobian of model w.r.t. parameters and Hessian of loss w.r.t. model prediction.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Per-item logit ([N, C]),
            per-item J_model ([N, C, D]),
            per-item H_loss ([N, C, C]).
    """

    def model_fn(params: FrozenDict[str, Any], x: jax.Array) -> jax.Array:
        """
        Performs model forward pass as a function of model parameters.

        Args:
            params (FrozenDict[str, Any]): Model parameters ([D], pytree in D).
            x (jax.Array): Model input ([N, ...]).

        Returns:
            jax.Array: Model output ([N, C]).
        """

        logits = state.apply_fn(params, x)  # [N, C]
        return logits

    def loss_fn(logits: jax.Array, y: jax.Array) -> jax.Array:
        """
        Computes per-item loss as a function of model output.

        Args:
            logits (jax.Array): Model output ([N, C]).
            y (jax.Array): Ground truth, integer-encoded ([N]).

        Returns:
            jax.Array: Per-item loss ([N]).
        """

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)  # [N]
        return loss  # type: ignore

    # Retrieve data
    x, y = batch

    # Forward pass
    logits = model_fn(state.params, x)  # [N, C]
    N, C = logits.shape

    # Define differentiating functions:
    #   - 'J_model_fn': Jacobian of model output w.r.t. model parameters
    #   - 'd_loss_fn': Gradient of loss w.r.t. model output
    #   - 'H_loss_fn': Hessian of loss w.r.t. model output
    J_model_fn = jax.jacrev(model_fn)  # [D]->[C, D], D>>C
    d_loss_fn = jax.grad(loss_fn)  # [C]->[C]
    H_loss_fn = jax.jacfwd(d_loss_fn)  # [C]->[C, C]

    # Compute differential quantities:
    #   - 'J_model': Per-item Jacobian of model output w.r.t. model parameters
    #   - 'H_loss': Per-item Hessian of loss w.r.t. model output
    J_model = J_model_fn(state.params, x)  # [N, C, D], pytree in D
    H_loss = jax.vmap(H_loss_fn)(logits, y)  # [N, C, C]

    # Transform 'J_model' from pytree representation into vector representation
    J_model = jnp.concatenate(
        [x.reshape(N, C, -1) for x in tree_util.tree_leaves(J_model)], axis=2
    )  # [N, C, D]

    return logits, J_model, H_loss


def compute_ggn(J_model: jax.Array, H_loss: jax.Array) -> jax.Array:
    """
    Computes GGN realization as product of Jacobians and Hessian.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        J_model (jax.Array): Per-item J_model ([N, C, D]).
        H_loss (jax.Array): Per-item H_loss ([N, C, C]).

    Returns:
        jax.Array: Per-item GGN ([N, D, D]).
    """

    # Compute per-item Generalized Gauss-Newton (GGN) matrix: J_model.T @ H_loss @ J_model
    return jnp.einsum("nax,nab,nby->nxy", J_model, H_loss, J_model)  # [N, D, D]


def compute_ggn_inv(GGN: jax.Array, prior_precision: float) -> jax.Array:
    """
    Computes inverse of GGN via eigendecomposition.
    D: Parameter dim.

    Args:
        GGN: GGN matrix ([D, D])
        prior_precision (float): Prior precision for GGN computation.

    Returns:
        jax.Array: GGN inverse matrix ([D, D]).
    """

    eigval, eigvec = jsp.linalg.eigh(GGN)

    return eigvec @ jnp.diag(1 / (eigval + prior_precision)) @ eigvec.T


@jax.jit
def compute_ltk(J_infer: jax.Array, GGN_inv: jax.Array) -> jax.Array:
    """
    Computes Laplace Tangent Kernel (LTK).
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.
    M: Test batch dim.

    Args:
        J_infer (jax.Array): Per-item J_model, evaluated at test data ([M, C, D]).
        GGN_inv: Prior for inverse GGN ([N, D, D])

    Returns:
        jax.Array: Per-item LTK ([N, M, C, C]).
    """

    # Compute LTK as simple matrix product: J_infer @ GGN_inv @ J_infer.T
    return jnp.einsum("mai,nij,mbj->nmab", J_infer, GGN_inv, J_infer)  # [N, M, C, C]


@jax.jit
def compute_predictive_distribution(y_infer: jax.Array, LTK: jax.Array) -> jax.Array:
    """
    Evaluates predictive distribution for classification task.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.
    M: Test batch dim.

    Args:
        y_infer (jax.Array): Predicted point estimates ([M, C]).
        LTK (jax.Array): Laplace Tangent Kernel ([N, M, C, C])

    Returns:
        jax.Array: Predictive distribution ([N, M, C]).
    """

    return jax.nn.softmax(
        y_infer[None, ...] / jnp.sqrt(1 + (jnp.pi / 8) * jnp.diagonal(LTK, axis1=2, axis2=3)),
        axis=2,
    )  # [N, M, C]


def aggregate_samples(
    average: jax.Array, samples: jax.Array, aggregated_sample_size: int
) -> jax.Array:
    """
    Aggregates samples as moving average.
    N: Batch dim.

    Args:
        average (jax.Array): Previous moving average ([N, ...]).
        samples (jax.Array): New samples ([N, ...]).
        aggregated_sample_size (int): Sample size after aggregation.

    Returns:
        jax.Array: Aggregated moving average ([N, ...]).
    """

    # Aggregates samples as moving average
    return average + (samples - average) / aggregated_sample_size  # [N, ...]


def aggregate_samples_total(
    average_total: jax.Array, samples: jax.Array, aggregated_sample_size: int
) -> jax.Array:
    """
    Aggregates total samples as moving average.
    N: Batch dim.

    Args:
        average_total (jax.Array): Previous total moving average ([...]).
        samples (jax.Array): New samples ([N, ...]).
        aggregated_sample_size (int): Total sample size after aggregation.

    Returns:
        jax.Array: Aggregated total moving average ([...]).
    """

    # Aggregates samples as total moving average
    return (
        average_total
        + jnp.sum(samples - average_total[None, ...], axis=0) / aggregated_sample_size
    )  # [...]


def start_experiment(
    state: TrainState,
    sample_dataloader: DataLoader,
    total_dataloader: DataLoader,
    test_dataloader: DataLoader,
    ggn_sample_sizes: List[int],
    prior_precision: float,
    experiment_name: str,
    n_steps: int,
    compose_on_cpu: bool,
    no_progress_bar: bool,
    results_path: str,
) -> None:
    """
    Starts an experiment.

    Args:
        state (TrainState): Current training state.
        sample_dataloader (DataLoader): Dataloader for sample-approximate GGNs.
        total_dataloader (DataLoader): Dataloader for total GGN.
        test_dataloader (DataLoader): Dataloader for Laplace approximation.
        ggn_sample_sizes (List[int]): Sample sizes for sample-approximate GGNs.
        prior_precision (float): Prior precision used for Laplace approximation.
        experiment_name (str): Name of the experiment.
        n_steps (int): Number of completed training steps across epochs.
        compose_on_cpu (bool): Whether GGN realization should be computed on CPU.
        no_progress_bar (bool): Disables progress bar.
        results_path (str): Results path.
    """

    if experiment_name == "laplace":
        save_measure = None
    else:
        save_measure = get_save_measure(experiment_name, len(sample_dataloader.dataset.classes))  # type: ignore

    # Init running statistics
    GGN_counter = 0  # Number of already computed per-item GGNs (for running average)
    GGN_total = None  # Total GGN, encompassing all per-item GGNs across the dataset
    GGN_samples = None  # GGN samples, aggregated over one/multiple data batches
    GGN_inv = None  # GGN inverse

    # Init UQ buffers when iterating over test dataset
    LTK_samples_buffer = {ggn_sample_size: [] for ggn_sample_size in ggn_sample_sizes}
    LTK_total_buffer = []
    pred_distr_samples_buffer = {ggn_sample_size: [] for ggn_sample_size in ggn_sample_sizes}
    pred_distr_total_buffer = []

    # Compute GGN realization on CPU, if needed
    device = jax.devices("cpu")[0] if compose_on_cpu else None
    device_cpu = jax.devices("cpu")[0]
    compute_ggn_jit = jax.jit(compute_ggn, device=device)
    aggregate_ggn_jit = jax.jit(aggregate_samples, device=device)
    aggregate_ggn_total_jit = jax.jit(aggregate_samples_total, device=device)
    compute_ggn_inv_jit = jax.vmap(
        jax.jit(partial(compute_ggn_inv, prior_precision=prior_precision), device=device)
    )

    # Update weights, if WeightedSampler is used
    if isinstance(sample_dataloader.sampler, WeightedSampler):
        sample_dataloader.sampler.update(state)

    # Iterate over dataset for GGN samples computation
    for ggn_step_idx, ggn_batch in enumerate(
        tqdm(
            sample_dataloader,
            desc="GGN-sample-step",
            disable=no_progress_bar,
        )
    ):
        # Compute GGN samples
        _, J_model, H_loss = compute_ggn_decomp(state, ggn_batch)  # [N, C, D], [N, C, C]
        if compose_on_cpu:
            J_model = jax.device_put(J_model, jax.devices('cpu')[0])
            H_loss = jax.device_put(H_loss, jax.devices('cpu')[0])
        GGN = compute_ggn_jit(J_model, H_loss)  # [N, D, D]

        # Aggregate GGN samples as running average to save memory
        aggregated_sample_size = ggn_step_idx + 1
        if GGN_samples is None:
            GGN_samples = GGN.copy()  # [N, D, D]
        else:
            GGN_samples = aggregate_ggn_jit(GGN_samples, GGN, aggregated_sample_size)  # [N, D, D]

        # Save GGN samples on disk, if needed aggregated sample size reached
        if aggregated_sample_size in ggn_sample_sizes:
            ggn_sample_size_idx = ggn_sample_sizes.index(aggregated_sample_size)

            # Save GGN samples
            save_ggn(
                GGN_samples,
                n_steps,
                results_path,
                sample_size=aggregated_sample_size,
            )

            # Last GGN samples: Stop further GGN computations
            if ggn_sample_size_idx + 1 == len(ggn_sample_sizes):
                GGN_samples = None
                break

    # Iterate over dataset for GGN total computation
    for ggn_step_idx, ggn_batch in enumerate(
        tqdm(
            total_dataloader,
            desc="GGN-total-step",
            disable=no_progress_bar,
        )
    ):
        ### DEBUG
        if ggn_step_idx >= 8:
            break

        # Compute GGN samples
        _, J_model, H_loss = compute_ggn_decomp(state, ggn_batch)  # [N, C, D], [N, C, C]
        if compose_on_cpu:
            J_model = jax.device_put(J_model, jax.devices('cpu')[0])
            H_loss = jax.device_put(H_loss, jax.devices('cpu')[0])
        GGN = compute_ggn_jit(J_model, H_loss)  # [N, D, D]

        # Compute total GGN as running average to save memory
        GGN_counter += GGN.shape[0]
        if GGN_total is None:
            GGN_total = jnp.mean(GGN, axis=0)  # [D, D]
        else:
            GGN_total = aggregate_ggn_total_jit(GGN_total, GGN, GGN_counter)  # [D, D]

    # Compute and save measure, then remove GGN
    if save_measure is not None:
        for ggn_sample_size in ggn_sample_sizes:
            GGN_samples = load_ggn(n_steps, results_path, sample_size=ggn_sample_size)
            save_measure(
                GGN_samples,
                GGN_total,  # type: ignore
                n_steps,
                results_path,
                ggn_sample_size,
            )
            remove_ggn(n_steps, results_path, sample_size=ggn_sample_size)
    # Do UQ
    else:
        for laplace_step_idx, laplace_batch in enumerate(
            tqdm(
                test_dataloader,
                desc="Laplace-step",
                disable=no_progress_bar,
            )
        ):
            ### DEBUG
            if laplace_step_idx >= 8:
                break

            # Compute Jacobians of the model for test data
            y_infer, J_infer, _ = compute_ggn_decomp(state, laplace_batch)  # [M, C], [M, C, D]

            # Load sampled GGNs, invert them and compute LTKs
            for ggn_sample_size in ggn_sample_sizes:
                GGN_inv = load_ggn(n_steps, results_path, sample_size=ggn_sample_size)  # [N, D, D]
                GGN_inv = jax.device_put(GGN_inv, device)  # [N, D, D]
                GGN_inv = compute_ggn_inv_jit(GGN_inv)  # [N, D, D]

                # Compute LTK and predictive distribution
                LTK = compute_ltk(J_infer, GGN_inv)  # [N1, M, C, C]  # type: ignore
                pred_distr = compute_predictive_distribution(y_infer, LTK)  # [N1, M, C]
                LTK = jax.device_put(LTK, device_cpu)
                pred_distr = jax.device_put(pred_distr, device_cpu)
                LTK_samples_buffer[ggn_sample_size].append(LTK.copy())  # [N1, M, C, C]
                pred_distr_samples_buffer[ggn_sample_size].append(pred_distr.copy())  # [N1, M, C]

            # Compute LTK and predictive distribution for total GGN
            GGN_inv = jax.device_put(GGN_total, device)  # [1, D, D]
            GGN_inv = compute_ggn_inv_jit(GGN_inv[None, :, :])  # [1, D, D]  # type: ignore
            LTK = compute_ltk(J_infer, GGN_inv)  # [N1, M, C, C]  # type: ignore
            pred_distr = compute_predictive_distribution(y_infer, LTK)  # [N1, M, C]
            LTK = jax.device_put(LTK, device_cpu)[0]  # [M, C, C]
            pred_distr = jax.device_put(pred_distr, device_cpu)[0]  # [M, C]
            LTK_total_buffer.append(LTK.copy())  # [M, C, C]
            pred_distr_total_buffer.append(pred_distr.copy())  # [M, C]

        # Save LTK and predictive distribution results on disk, if computed
        # Additionally, remove GGNs from disk
        for ggn_sample_size, LTK_samples in LTK_samples_buffer.items():
            save_ltk(jnp.concatenate(LTK_samples, axis=1), n_steps, results_path, ggn_sample_size)
            remove_ggn(n_steps, results_path, sample_size=ggn_sample_size)
        for pred_distr_sample_size, pred_distr_samples in pred_distr_samples_buffer.items():
            save_predictive_distribution(
                jnp.concatenate(pred_distr_samples, axis=1),
                n_steps,
                results_path,
                pred_distr_sample_size,
            )
        save_ltk(jnp.concatenate(LTK_total_buffer, axis=0), n_steps, results_path)
        save_predictive_distribution(
            jnp.concatenate(pred_distr_total_buffer, axis=0), n_steps, results_path
        )
