import os
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Callable, List, Tuple

import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import scipy as jsp
from jax import tree_util
from tqdm import tqdm

from data_loader import DataLoader
from log_utils import (
    load_ggn,
    remove_ggn,
    save_eigh_lobpcg_overlap,
    save_ggn,
    save_ltk,
    save_predictive_distribution,
)
from sampler import WeightedSampler


def train_step(
    state: TrainState,
    batch: Tuple[jax.Array, jax.Array],
    n_classes: int,
    l2_reg: float,
) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array]:
    """
    Performs a single training step without GGN computation.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.
        n_classes (int): Number of classes (equal to C).
        l2_reg (float): L2 regularizer weighting.

    Returns:
        Tuple[TrainState, jax.Array, jax.Array, jax.Array]:
            Updated training state,
            per-item loss ([N]),
            number of correct predictions per class ([C]),
            number of ground-truth items per class ([C]).
    """

    def model_loss_fn(
        params: FrozenDict[str, Any], x: jax.Array, y: jax.Array
    ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
        """
        Performs model forward pass and evaluates mean loss as a function of model parameters.

        Args:
            params (FrozenDict[str, Any]): Model parameters ([D], pytree in D).
            x (jax.Array): Model input ([N, ...]).
            y (jax.Array): Ground truth, integer-encoded ([N]).

        Returns:
            Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
                Mean loss ([1]),
                per-item loss ([N]),
                model output ([N, C]).
        """

        logits = state.apply_fn(params, x)  # [N, C]
        loss_unreduced = optax.softmax_cross_entropy_with_integer_labels(logits, y)  # [N]
        loss = jnp.mean(loss_unreduced)  # [1]
        l2_penalty = (
            l2_reg
            / 2
            * jnp.sum(
                jnp.concatenate([x.ravel() * x.ravel() for x in tree_util.tree_leaves(params)])
            )
        )  # [1]
        loss = loss + l2_penalty  # [1]
        return loss, (loss_unreduced, logits)  # type: ignore

    # Retrieve data
    x, y = batch

    # Forward pass + gradient
    (_, (loss, logits)), d_loss = jax.value_and_grad(model_loss_fn, has_aux=True)(
        state.params, x, y
    )  # [N]; [N, C]; [D], pytree in D

    # Compute number of correct predictions per class
    correct = (jnp.argmax(logits, -1) == y).astype(int)  # [N]
    # Add dummy false predictions to account for every class
    y_dummy = jnp.concatenate((y, jnp.arange(n_classes, dtype=int)))  # [N+C]
    correct_dummy = jnp.concatenate((correct, jnp.zeros((n_classes,), dtype=int)))  # [N+C]
    n_correct_per_class = jnp.bincount(y_dummy, correct_dummy, length=n_classes)  # [C]
    n_per_class = jnp.bincount(
        y_dummy,
        jnp.concatenate((jnp.ones_like(y), jnp.zeros((n_classes,), dtype=int))),
        length=n_classes,
    )  # [C]

    # Perform gradient step, update training state
    state = state.apply_gradients(grads=d_loss)

    return state, loss, n_correct_per_class, n_per_class


def test_step(
    state: TrainState,
    batch: Tuple[
        jax.Array,
        jax.Array,
    ],
    n_classes: int,
    l2_reg: float,
    return_grad: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array] | Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Performs a single test step without GGN computation.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.
        n_classes (int): Number of classes (equal to C).
        l2_reg (float): L2 regularizer weighting.
        return_grad (bool, optional): Whether to return loss gradients. Defaults to False.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Per-item loss ([N]),
            per-item gradient ([N, D], optional),
            number of correct predictions per class ([C]),
            number of ground-truth items per class ([C]).
    """

    def model_loss_fn(
        params: FrozenDict[str, Any], x: jax.Array, y: jax.Array
    ) -> Tuple[jax.Array, Tuple[jax.Array]]:
        """
        Performs model forward pass and evaluates mean loss as a function of model parameters.

        Args:
            params (FrozenDict[str, Any]): Model parameters ([D], pytree in D).
            x (jax.Array): Model input ([N, ...]).
            y (jax.Array): Ground truth, integer-encoded ([N]).

        Returns:
            Tuple[jax.Array, Tuple[jax.Array]]:
                Per-item loss ([N]),
                model output ([N, C]).
        """

        logits = state.apply_fn(params, x)  # [N, C]
        loss_unreduced = optax.softmax_cross_entropy_with_integer_labels(logits, y)  # [N]
        l2_penalty = (
            l2_reg
            / 2
            * jnp.sum(
                jnp.concatenate([x.ravel() * x.ravel() for x in tree_util.tree_leaves(params)])
            )
        )  # [1]
        loss_unreduced = loss_unreduced + l2_penalty  # [N]
        return loss_unreduced, (logits,)  # type: ignore

    # Retrieve data
    x, y = batch

    # Forward pass
    loss, (logits,) = model_loss_fn(state.params, x, y)  # [N]; [N, C]

    # Compute number of correct predictions per class
    correct = (jnp.argmax(logits, -1) == y).astype(int)  # [N]
    # Add dummy false predictions to account for every class
    y_dummy = jnp.concatenate((y, jnp.arange(n_classes, dtype=int)))  # [N+C]
    correct_dummy = jnp.concatenate((correct, jnp.zeros((n_classes,), dtype=int)))  # [N+C]
    n_correct_per_class = jnp.bincount(y_dummy, correct_dummy, length=n_classes)  # [C]
    n_per_class = jnp.bincount(
        y_dummy,
        jnp.concatenate((jnp.ones_like(y), jnp.zeros((n_classes,), dtype=int))),
        length=n_classes,
    )  # [C]

    if return_grad:
        d_loss, _ = jax.jacrev(model_loss_fn, has_aux=True)(
            state.params, x, y
        )  # [N, D], pytree in D
        N, _ = logits.shape

        # Transform 'd_loss' from pytree representation into vector representation
        d_loss = jnp.concatenate(
            [x.reshape(N, -1) for x in tree_util.tree_leaves(d_loss)], axis=1
        )  # [N, D]
        return loss, d_loss, n_correct_per_class, n_per_class
    else:
        return loss, n_correct_per_class, n_per_class


@jax.jit
def compute_ggn_decomp(
    state: TrainState, batch: Tuple[jax.Array, jax.Array]
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Performs a single training step with decomposed GGN computation.
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


def compute_ggn(J_model: jax.Array, H_loss: jax.Array, l2_reg: float) -> jax.Array:
    """
    Computes GGN realization as product of Jacobians and Hessian.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        J_model (jax.Array): Per-item J_model ([N, C, D]).
        H_loss (jax.Array): Per-item H_loss ([N, C, C]).
        l2_reg (float): L2 regularizer weighting.

    Returns:
        jax.Array: Per-item GGN ([N, D, D]).
    """

    # Realize L2 prior as diagonal matrix
    D = J_model.shape[2]
    L2_prior = -l2_reg * jnp.eye(D)  # do not forget the minus!

    # Compute per-item Generalized Gauss-Newton (GGN) matrix: J_model.T @ H_loss @ J_model + L2_prior
    return jnp.einsum("nax,nab,nby->nxy", J_model, H_loss, J_model) + L2_prior  # [N, D, D]


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


def woodbury(
    A_inv: jax.Array, UCV: Tuple[jax.Array, jax.Array, jax.Array]
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the inverse of the product U @ (C_inv)^(-1) @ V + (A_inv)^(-1) efficiently using the Woodbury matrix identity.

    Args:
        A_inv (jax.Array): Inverse small matrix ([B, B]).
        UCV (Tuple[jax.Array, jax.Array, jax.Array]):
            U (jax.Array): Tall matrix ([A, B]).
            C_inv (jax.Array): Inverse small matrix ([B, B]).
            V (jax.Array): Wide matrix ([B, A]).

    Returns:
        Tuple[jax.Array, jax.Array]: Inverse large matrix ([A, A]), dummy output needed for scan compatibility ([1]).
    """

    U, C_inv, V = UCV

    cho = C_inv + V @ A_inv @ U
    cho = jsp.linalg.cho_factor(cho)
    cho = jsp.linalg.cho_solve(cho, V)

    return A_inv - A_inv @ U @ cho @ A_inv, jnp.zeros((1,))


@jax.jit
def compute_ggn_inv(J_model: jax.Array, H_loss: jax.Array, GGN_inv: jax.Array) -> jax.Array:
    """
    Updates inverse of GGN.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        J_model (jax.Array): Per-item J_model ([N, C, D]).
        H_loss (jax.Array): Per-item H_loss ([N, C, C])
        GGN_inv: Prior for inverse GGN ([D, D])

    Returns:
        jax.Array: Updated GGN inverse ([D, D]).
    """

    # @jax.vmap
    # def vmap_scan(f, init, xs):
    #     return jax.lax.scan(f, init, xs)

    # Compute GGN inverse using recursive application of the woodbury matrix identity
    # Can be implemented efficiently using a scan-op, initialized by GGN inverse prior and scanning over (J_model, H_loss) pairs
    # Scale H_loss_inv by N beforehand to account for averaging
    N = J_model.shape[0]
    GGN_inv, _ = jax.lax.scan(
        woodbury, GGN_inv, (J_model.transpose((0, 2, 1)), N * jnp.linalg.inv(H_loss), J_model)
    )  # [D, D]

    return GGN_inv


@jax.jit
def compute_ltk(J_infer: jax.Array, GGN_inv: jax.Array) -> jax.Array:
    """
    Computes Laplace Tangent Kernel (LTK).
    C: Model output dim.
    D: Parameter dim.
    N1: Batch dim.
    N2: Sample dim.
    M: Test batch dim.

    Args:
        J_infer (jax.Array): Per-item J_model, evaluated at test data ([M, C, D]).
        GGN_inv: Prior for inverse GGN ([N1, D, D])

    Returns:
        jax.Array: Per-item LTK ([N1, M, C, C]).
    """

    # LTK: -J_infer @ GGN_inv @ J_infer.T

    # Compute LTK as simple matrix product
    return jnp.einsum("mai,nij,mbj->nmab", J_infer, GGN_inv, J_infer)  # [N1, M, C, C]


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


def train_epoch(
    state: TrainState,
    train_dataloader: DataLoader,
    l2_reg: float,
    ggn_dataloader: DataLoader,
    ggn_total_dataloader: DataLoader,
    ggn_batch_sizes: List[int],
    ggn_freq: int,
    n_ggn_iterations: int,
    n_steps: int,
    prng_key: jax.Array,
    save_measure: Callable,
    measure_saving: str,
    ggn_saving: str,
    compose_on_cpu: bool,
    no_progress_bar: bool,
    results_path: str,
) -> Tuple[TrainState, float, float, jax.Array, int, int]:
    """
    Performs a single training epoch.

    Args:
        state (TrainState): Current training state.
        train_dataloader (DataLoader): Data loader for model training.
        l2_reg (float): L2 regularizer weighting.
        ggn_dataloader (DataLoader): Data loader for GGN samples computation.
        ggn_total_dataloader (DataLoader): Data loader for total GGN computation.
        ggn_batch_sizes (List[int]): Batch sizes for which GGNs will be saved.
        ggn_freq (int): Frequency of GGN iterations.
        n_ggn_iterations (int): Number of GGN iterations.
        n_steps (int): Current number of completed training step across epochs.
        prng_key (jax.Array): Random key.
        save_measure (Callable): GGN error measure: GGN_1 [N_1, D, D], GGN_2 [N_2, D, D] -> measure [N_1, N_2] saved on disk.
        measure_saving (str): GGN error measure saving: disabled, total, next, last.
        ggn_saving (str): GGN saving: disabled, dense.
        compose_on_cpu (bool): Computes GGN realization on CPU instead of GPU (might exceed GPU memory otherwise).
        no_progress_bar (bool): Disables progress bar.
        results_path (str): Results path.

    Returns:
        Tuple[TrainState, float, float, jax.Array, int, int]:
            Updated training state,
            epoch loss,
            epoch accuracy,
            epoch accuracy per class,
            current number of completed training step across epochs,
            number of remaining ggn iterations after this epoch.
    """

    ggn_computations_disabled = ggn_saving == "disabled" and measure_saving == "disabled"

    n_classes = len(train_dataloader.dataset.classes)  # type: ignore
    train_step_jit = jax.jit(partial(train_step, n_classes=n_classes, l2_reg=l2_reg))

    # Compute GGN realization on CPU, if needed
    device = jax.devices("cpu")[0] if compose_on_cpu else None
    compute_ggn_jit = jax.jit(partial(compute_ggn, l2_reg=l2_reg), device=device)
    aggregate_ggn_jit = jax.jit(aggregate_samples, device=device)
    aggregate_ggn_total_jit = jax.jit(aggregate_samples_total, device=device)

    # Running statistics
    loss_epoch = []  # Per-item losses per training steps
    n_correct_epoch = 0  # Number of correct predictions across the epoch
    n_correct_per_class_epoch = jnp.zeros((n_classes,), dtype=int)
    n_per_class_epoch = jnp.zeros_like(n_correct_per_class_epoch)
    GGN_iter_counter = 0  # Number of already completed GGN iterations

    # Start epoch
    pbar_stats = {"loss": 0.0, "acc": 0.0}
    with tqdm(
        total=len(train_dataloader), desc="Step", disable=no_progress_bar, postfix=pbar_stats
    ) as pbar:
        # Iterate over dataset for training
        for batch in train_dataloader:
            # Compute GGN, if n_ggn_iterations not reached and current step is multiple of ggn_freq
            if GGN_iter_counter < n_ggn_iterations and n_steps % ggn_freq == 0:
                GGN_iter_counter += 1

                # Init running statistics for this GGN iteration
                GGN_counter = 0  # Number of already computed per-item GGNs (for running average)
                GGN_total = None  # Total GGN, encompassing all per-item GGNs across the dataset
                GGN_samples = None  # GGN samples, aggregated over one/multiple data batches

                # Update weights, if WeightedSampler is used
                if not ggn_computations_disabled and isinstance(
                    ggn_dataloader.sampler, WeightedSampler
                ):
                    ggn_dataloader.sampler.update(state)

                # Iterate over dataset for GGN samples computation
                for ggn_step_idx, ggn_batch in enumerate(
                    tqdm(
                        ggn_dataloader,
                        desc="GGN-sample-step",
                        disable=no_progress_bar or ggn_computations_disabled,
                    )
                ):
                    # No GGN computations during training needed
                    if ggn_computations_disabled:
                        break

                    # Compute GGN samples
                    _, J_model, H_loss = compute_ggn_decomp(
                        state, ggn_batch
                    )  # [N, C, D], [N, C, C]
                    if compose_on_cpu:
                        J_model = jax.device_put(J_model, jax.devices('cpu')[0])
                        H_loss = jax.device_put(H_loss, jax.devices('cpu')[0])
                    GGN = compute_ggn_jit(J_model, H_loss)  # [N, D, D]

                    # Aggregate GGN samples as running average to save memory
                    aggregated_batch_size = ggn_step_idx + 1
                    if GGN_samples is None:
                        GGN_samples = GGN.copy()  # [N, D, D]
                    else:
                        GGN_samples = aggregate_ggn_jit(
                            GGN_samples, GGN, aggregated_batch_size
                        )  # [N, D, D]

                    # Save GGN samples on disk, if needed aggregated batch size reached
                    if aggregated_batch_size in ggn_batch_sizes:
                        ggn_batch_size_idx = ggn_batch_sizes.index(aggregated_batch_size)

                        if ggn_batch_size_idx > 0:
                            prev_ggn_batch_size = ggn_batch_sizes[ggn_batch_size_idx - 1]
                            # Norm-saving "next": Load previous batched GGN samples, compute norm
                            if measure_saving == "next":
                                prev_GGN_samples = load_ggn(
                                    n_steps, results_path, batch_size=prev_ggn_batch_size
                                )
                                save_measure(
                                    prev_GGN_samples,
                                    GGN_samples,
                                    prng_key,
                                    n_steps,
                                    results_path,
                                    prev_ggn_batch_size,
                                )
                            # GGN-saving "disabled" and not norm-saving "total" or "last": Remove previous batched GGN samples
                            if ggn_saving == "disabled" and measure_saving not in {
                                "total",
                                "last",
                            }:
                                remove_ggn(n_steps, results_path, batch_size=prev_ggn_batch_size)

                        # Not norm-saving "next" or not ggn-saving "disabled" or not last GGN samples: Save GGN samples
                        if (
                            measure_saving != "next"
                            or ggn_saving != "disabled"
                            or ggn_batch_size_idx + 1 < len(ggn_batch_sizes)
                        ):
                            save_ggn(
                                GGN_samples,
                                n_steps,
                                results_path,
                                batch_size=aggregated_batch_size,
                            )

                        # Last GGN samples: Stop further GGN computations
                        if ggn_batch_size_idx + 1 == len(ggn_batch_sizes):
                            break
                # Norm-saving "last": Compute norm
                if measure_saving == "last":
                    GGN_samples_last = load_ggn(
                        n_steps, results_path, batch_size=ggn_batch_sizes[-1]
                    )
                    for ggn_batch_size in ggn_batch_sizes[:-1]:
                        GGN_samples = load_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                        save_measure(
                            GGN_samples,
                            GGN_samples_last,
                            prng_key,
                            n_steps,
                            results_path,
                            ggn_batch_size,
                        )
                        # GGN-saving "disabled" : Remove batched GGN samples
                        if ggn_saving == "disabled":
                            remove_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                    if ggn_saving == "disabled":
                        remove_ggn(n_steps, results_path, batch_size=ggn_batch_sizes[-1])

                # Norm-saving "disabled" or "total": Iterate over dataset for total GGN computation
                if measure_saving in {"disabled", "total"}:
                    for ggn_step_idx, ggn_batch in enumerate(
                        tqdm(
                            ggn_total_dataloader,
                            desc="GGN-total-step",
                            disable=no_progress_bar or ggn_computations_disabled,
                        )
                    ):
                        # No GGN computations during training needed
                        if ggn_computations_disabled:
                            break

                        # Compute GGN samples
                        _, J_model, H_loss = compute_ggn_decomp(
                            state, ggn_batch
                        )  # [N, C, D], [N, C, C]
                        if compose_on_cpu:
                            J_model = jax.device_put(J_model, jax.devices('cpu')[0])
                            H_loss = jax.device_put(H_loss, jax.devices('cpu')[0])
                        GGN = compute_ggn_jit(J_model, H_loss)  # [N, D, D]

                        # Compute total GGN as running average to save memory
                        GGN_counter += GGN.shape[0]
                        if GGN_total is None:
                            GGN_total = jnp.mean(GGN, axis=0)  # [D, D]
                        else:
                            GGN_total = aggregate_ggn_total_jit(
                                GGN_total, GGN, GGN_counter
                            )  # [D, D]

                    # GGN-saving "dense": Save total GGN on disk
                    if ggn_saving == "dense":
                        save_ggn(GGN_total, n_steps, results_path)  # type: ignore

                    # Norm-saving "total": Compute norm
                    if measure_saving == "total":
                        for ggn_batch_size in ggn_batch_sizes:
                            GGN_samples = load_ggn(
                                n_steps, results_path, batch_size=ggn_batch_size
                            )
                            save_measure(
                                GGN_samples,
                                GGN_total,  # type: ignore
                                prng_key,
                                n_steps,
                                results_path,
                                ggn_batch_size,
                            )
                            # LOBPCG-EIGH
                            save_eigh_lobpcg_overlap(
                                GGN_samples,
                                prng_key,
                                n_steps,
                                results_path,
                                10,
                                compose_on_cpu,
                                batch_size=ggn_batch_size,
                            )
                            # GGN-saving "disabled" : Remove batched GGN samples
                            if ggn_saving == "disabled":
                                remove_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                        # LOBPCG-EIGH
                        save_eigh_lobpcg_overlap(GGN_total, prng_key, n_steps, results_path, 10, compose_on_cpu)  # type: ignore

            # Perform training step
            state, loss, n_correct_per_class, n_per_class = train_step_jit(
                state,
                batch,
            )
            # Update running statistics
            loss_epoch.append(loss)
            n_correct_per_class_epoch = n_correct_per_class_epoch + n_correct_per_class
            n_per_class_epoch = n_per_class_epoch + n_per_class
            n_correct = n_correct_per_class.sum()
            n_correct_epoch += n_correct
            N = batch[0].shape[0]
            n_steps += 1
            # Update progress bar
            pbar.update()
            pbar_stats["loss"] = round(float(jnp.mean(loss)), 6)
            pbar_stats["acc"] = round(n_correct / N, 4)
            pbar.set_postfix(pbar_stats)

    # Compute final epoch statistics: Epoch loss, epoch accuracy (per class)
    loss = jnp.mean(jnp.concatenate(loss_epoch)).item()  # [1]
    accuracy = float(n_correct_epoch / len(train_dataloader.dataset))  # type: ignore
    accuracy_per_class = n_correct_per_class_epoch / n_per_class_epoch  # type: ignore

    return state, loss, accuracy, accuracy_per_class, n_steps, n_ggn_iterations - GGN_iter_counter


def test_epoch(
    state: TrainState,
    test_dataloader: DataLoader,
    l2_reg: float,
    ltk_dataloader: DataLoader,
    ltk_total_dataloader: DataLoader,
    ltk_batch_sizes: List[int],
    uncertainty_quantification: str,
    n_steps: int,
    prng_key: jax.Array,
    no_progress_bar: bool,
    results_path: str,
) -> Tuple[float, float, jax.Array]:
    """
    Performs a single training epoch.

    Args:
        state (TrainState): Current training state.
        test_dataloader (DataLoader): Data loader for model training.
        l2_reg (float): L2 regularizer weighting.
        ltk_dataloader (DataLoader): Data loader for LTK samples computation.
        ltk_total_dataloader (DataLoader): Data loader for total LTK computation.
        ltk_batch_sizes (List[int]): Batch sizes for which LTKs will be saved.
        uncertainty_quantification (str): Whether uncertainty is computed using Laplace Approximation (disabled, sampled, total).
        n_steps (int): Current number of completed training steps across epochs.
        prng_key (jax.Array): Random key.
        no_progress_bar (bool): Disables progress bar.
        results_path (str): Results path.

    Returns:
        Tuple[float, float, jax.Array]:
            Epoch loss,
            epoch accuracy,
            epoch accuracy per class.
    """

    n_classes = len(test_dataloader.dataset.classes)  # type: ignore
    test_step_jit = jax.jit(partial(test_step, n_classes=n_classes, l2_reg=l2_reg))

    # LTK buffers when iterating over test dataset
    device = jax.devices(jax.default_backend())[0]
    device_cpu = jax.devices("cpu")[0]
    J_model_aggregation_buffer = []
    H_loss_aggregation_buffer = []
    GGN_inv = None
    LTK_samples_buffer = {ltk_batch_size: [] for ltk_batch_size in ltk_batch_sizes}
    LTK_total_buffer = []
    pred_distr_samples_buffer = {ltk_batch_size: [] for ltk_batch_size in ltk_batch_sizes}
    pred_distr_total_buffer = []

    # Running statistics
    loss_epoch = []  # Per-item losses per test steps
    n_correct_epoch = 0  # Number of correct predictions across the epoch
    n_correct_per_class_epoch = jnp.zeros((n_classes,), dtype=int)
    n_per_class_epoch = jnp.zeros_like(n_correct_per_class_epoch)

    # Update weights for UQ, if WeightedSampler is used
    if uncertainty_quantification != "disabled":
        if isinstance(ltk_dataloader.sampler, WeightedSampler):
            ltk_dataloader.sampler.update(state)

    # Iterate over dataset to compute LTK samples
    continue_LTK_computation = uncertainty_quantification == "sampled"
    compute_ggn_inv_vmap = jax.vmap(compute_ggn_inv)
    for ltk_step_idx, ltk_batch in enumerate(
        tqdm(
            ltk_dataloader,
            desc="LTK-sample-step",
            disable=no_progress_bar or not continue_LTK_computation,
        )
    ):
        # Aggregate J_model, H_loss via concatenation
        if continue_LTK_computation:
            _, J_model, H_loss = compute_ggn_decomp(state, ltk_batch)  # [N, C, D], [N, C, C]
            J_model = jax.device_put(J_model, device_cpu)
            H_loss = jax.device_put(H_loss, device_cpu)
            J_model_aggregation_buffer.append(J_model.copy())
            H_loss_aggregation_buffer.append(H_loss.copy())

        # Start epoch
        pbar_stats = {"loss": 0.0, "acc": 0.0}
        with tqdm(
            total=len(test_dataloader),
            desc="Test-step",
            disable=no_progress_bar or ltk_step_idx > 0,
            postfix=pbar_stats,
        ) as pbar:
            # Iterate over dataset for testing
            for batch_idx, batch in enumerate(test_dataloader):
                # Perform test steps only once
                if ltk_step_idx == 0:
                    loss, n_correct_per_class, n_per_class = test_step_jit(state, batch)
                    # Update running statistics
                    loss_epoch.append(loss)
                    n_correct_per_class_epoch = n_correct_per_class_epoch + n_correct_per_class
                    n_per_class_epoch = n_per_class_epoch + n_per_class
                    n_correct = n_correct_per_class.sum()
                    n_correct_epoch += n_correct
                    N = batch[0].shape[0]
                    # Update progress bar
                    pbar.update()
                    pbar_stats["loss"] = round(float(jnp.mean(loss)), 6)
                    pbar_stats["acc"] = round(n_correct / N, 4)
                    pbar.set_postfix(pbar_stats)
                # Compute uncertainty, if needed
                if not continue_LTK_computation:
                    break
                aggregated_batch_size = ltk_step_idx + 1
                if aggregated_batch_size in ltk_batch_sizes:
                    # Compute Jacobians of the model for test data
                    y_infer, J_infer, _ = compute_ggn_decomp(state, batch)  # [M, C], [M, C, D]
                    # For first test batch, initialize GGN_inv with L2 prior, then update
                    if batch_idx == 0:
                        assert GGN_inv is None
                        J_model = jax.device_put(
                            jnp.stack(J_model_aggregation_buffer, axis=1), device
                        )  # [N1, N2, C, D]
                        H_loss = jax.device_put(
                            jnp.stack(H_loss_aggregation_buffer, axis=1), device
                        )  # [N1, N2, C, C]
                        N, _, _, D = J_model.shape
                        GGN_inv = jnp.broadcast_to(
                            (1 / l2_reg) * jnp.eye(D)[None, ...], (N, D, D)
                        )  # [N1, D, D]
                        GGN_inv = compute_ggn_inv_vmap(J_model, H_loss, GGN_inv)  # [N1, D, D]
                    # Compute LTK and predictive distribution
                    LTK = compute_ltk(J_infer, GGN_inv)  # [N1, M, C, C]  # type: ignore
                    pred_distr = compute_predictive_distribution(y_infer, LTK)  # [N1, M, C]
                    LTK = jax.device_put(LTK, device_cpu)
                    pred_distr = jax.device_put(pred_distr, device_cpu)
                    LTK_samples_buffer[aggregated_batch_size].append(LTK.copy())  # [N1, M, C, C]
                    pred_distr_samples_buffer[aggregated_batch_size].append(
                        pred_distr.copy()
                    )  # [N1, M, C]
                    # Reset GGN_inv after last test batch
                    if batch_idx == len(test_dataloader) - 1:
                        GGN_inv = None
                # Stop further LTK computations, if not needed (only after last sample size and last test datapoints)
                if (
                    aggregated_batch_size == ltk_batch_sizes[-1]
                    and batch_idx == len(test_dataloader) - 1
                ):
                    continue_LTK_computation = False

        # No redundant test epochs or LTK computations
        if not continue_LTK_computation:
            break

    # Iterate over train dataset to compute J_model and H_loss
    if uncertainty_quantification == "total":
        J_model_aggregation_buffer.clear()
        H_loss_aggregation_buffer.clear()
        GGN_inv = None
        for ltk_step_idx, ltk_batch in enumerate(
            tqdm(
                ltk_total_dataloader,
                desc="LTK-total-step",
                disable=no_progress_bar,
            )
        ):
            # Aggregate J_model, H_loss via concatenation
            _, J_model, H_loss = compute_ggn_decomp(state, ltk_batch)  # [N, C, D], [N, C, C]
            J_model = jax.device_put(J_model, device_cpu)
            H_loss = jax.device_put(H_loss, device_cpu)
            J_model_aggregation_buffer.append(J_model.copy())
            H_loss_aggregation_buffer.append(H_loss.copy())

        # Iterate over test dataset
        for batch_idx, batch in enumerate(test_dataloader):
            # Compute Jacobians of the model for test data
            y_infer, J_infer, _ = compute_ggn_decomp(state, batch)  # [M, C], [M, C, D]
            # For first test batch, initialize GGN_inv with L2 prior, then update
            if batch_idx == 0:
                assert GGN_inv is None
                J_model = jax.device_put(
                    jnp.concatenate(J_model_aggregation_buffer, axis=0), device
                )  # [N2, C, D]
                H_loss = jax.device_put(
                    jnp.concatenate(H_loss_aggregation_buffer, axis=0), device
                )  # [N2, C, C]
                D = J_model.shape[2]
                GGN_inv = (1 / l2_reg) * jnp.eye(D)  # [D, D]
                GGN_inv = compute_ggn_inv(J_model, H_loss, GGN_inv)  # [D, D]
            # Compute LTK
            LTK = compute_ltk(J_infer, GGN_inv[None, ...])  # [1, M, C, C]  # type: ignore
            pred_distr = compute_predictive_distribution(y_infer, LTK)  # [1, M, C]
            LTK = jax.device_put(LTK, device_cpu)
            pred_distr = jax.device_put(pred_distr, device_cpu)
            LTK_total_buffer.append(LTK[0].copy())  # [M, C, C]
            pred_distr_total_buffer.append(pred_distr[0].copy())  # [M, C]

    # Save LTK and predictive distribution results on disk, if computed
    if uncertainty_quantification == "sampled":
        for ltk_batch_size, LTK_samples in LTK_samples_buffer.items():
            save_ltk(jnp.concatenate(LTK_samples, axis=1), n_steps, results_path, ltk_batch_size)
        for pred_distr_batch_size, pred_distr_samples in pred_distr_samples_buffer.items():
            save_predictive_distribution(
                jnp.concatenate(pred_distr_samples, axis=1),
                n_steps,
                results_path,
                pred_distr_batch_size,
            )
    elif uncertainty_quantification == "total":
        save_ltk(jnp.concatenate(LTK_total_buffer, axis=0), n_steps, results_path)
        save_predictive_distribution(
            jnp.concatenate(pred_distr_total_buffer, axis=0), n_steps, results_path
        )

    # Compute final epoch statistics: Epoch loss, epoch accuracy (per class)
    loss = jnp.mean(jnp.concatenate(loss_epoch)).item()  # [1]
    accuracy = float(n_correct_epoch / len(test_dataloader.dataset))  # type: ignore
    accuracy_per_class = n_correct_per_class_epoch / n_per_class_epoch  # type: ignore

    return loss, accuracy, accuracy_per_class
