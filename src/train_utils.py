from functools import partial
from pathlib import Path
from time import time
from typing import Any, List, Tuple

import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import tree_util
from tqdm import tqdm

from data_loader import DataLoader
from log_utils import load_ggn, remove_ggn, save_ggn, save_norm
from sampler import WeightedSampler


def train_step(
    state: TrainState,
    batch: Tuple[jax.Array, jax.Array],
    n_classes: int,
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
) -> Tuple[jax.Array, jax.Array]:
    """
    Performs a single training step with decomposed GGN computation.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.

    Returns:
        Tuple[jax.Array, jax.Array]:
            Per-item J_model ([N, C, D]),
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

    return J_model, H_loss


def compute_ggn(J_model: jax.Array, H_loss: jax.Array) -> jax.Array:
    """
    Computes GGN realization as product of Jacobians and Hessian.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        J_model (jax.Array): Per-item J_model ([N, C, D]).
        H_loss (jax.Array): Per-item H_loss ([N, C, C])

    Returns:
        jax.Array: Per-item GGN ([N, D, D])
    """

    # Compute per-item Generalized Gauss-Newton (GGN) matrix: J_model.T @ H_loss @ J_model
    return jnp.einsum("nax,nab,nby->nxy", J_model, H_loss, J_model)  # [N, D, D]


def aggregate_ggn(GGN_samples: jax.Array, GGN: jax.Array, aggregated_batch_size: int) -> jax.Array:
    """
    Aggregates GGN as moving average.
    D: Parameter dim.
    N: Batch dim.

    Args:
        GGN_samples (jax.Array): Previous GGN moving average ([N, D, D]).
        GGN (jax.Array): New GGN sample ([N, D, D]).
        aggregated_batch_size (int): Sample size after aggregation.

    Returns:
        jax.Array: Aggregated GGN moving average ([N, D, D]).
    """

    # Aggregates GGN as moving average
    return GGN_samples + (GGN - GGN_samples) / aggregated_batch_size  # [N, D, D]


def aggregate_ggn_total(GGN_total: jax.Array, GGN: jax.Array, GGN_counter: int) -> jax.Array:
    """
    Aggregates total GGN as moving average.
    D: Parameter dim.
    N: Batch dim.

    Args:
        GGN_total (jax.Array): Previous total GGN moving average ([D, D]).
        GGN (jax.Array): New GGN sample ([N, D, D]).
        GGN_counter (int): Total sample size after aggregation.

    Returns:
        jax.Array: Aggregated total GGN moving average ([D, D]).
    """

    # Aggregates total GGN as moving average
    return GGN_total + jnp.sum(GGN - GGN_total[None, :, :], axis=0) / GGN_counter  # [D, D]


def train_epoch(
    state: TrainState,
    train_dataloader: DataLoader,
    ggn_dataloader: DataLoader,
    ggn_batch_sizes: List[int],
    ggn_freq: int,
    n_ggn_iterations: int,
    n_steps: int,
    norm_saving: str,
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
        ggn_dataloader (DataLoader): Data loader for GGN computation.
        ggn_batch_sizes (List[int]): Batch sizes for which GGNs will be saved.
        ggn_freq (int): Frequency of GGN iterations.
        n_ggn_iterations (int): Number of GGN iterations.
        n_steps (int): Current number of completed training step across epochs.
        norm_saving (str): GGN norm saving: disabled, total, next, last.
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

    n_classes = len(train_dataloader.dataset.classes)  # type: ignore
    train_step_jit = jax.jit(partial(train_step, n_classes=n_classes))

    # Compute GGN realization on CPU, if needed
    device = jax.devices("cpu")[0] if compose_on_cpu else None
    compute_ggn_jit = jax.jit(compute_ggn, device=device)
    aggregate_ggn_jit = jax.jit(aggregate_ggn, device=device)
    aggregate_ggn_total_jit = jax.jit(aggregate_ggn_total, device=device)

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
                if isinstance(ggn_dataloader.sampler, WeightedSampler):
                    ggn_dataloader.sampler.update(state)

                datapoints = []

                # Iterate over dataset for GGN computation
                for ggn_step_idx, ggn_batch in enumerate(
                    tqdm(ggn_dataloader, desc="GGN-sample-step", disable=no_progress_bar)
                ):
                    # CODE TO SAVE FIRST N DATA POINTS
                    # if ggn_step_idx < ggn_batch_sizes[-1]:
                    #     x, _ = ggn_batch

                    #     datapoints.append(x)
                    # elif ggn_step_idx == ggn_batch_sizes[-1]:
                    #     jnp.save(
                    #         str(
                    #             Path(
                    #                 results_path,
                    #                 "datapoints",
                    #                 f"uniform_{n_ggn_iterations - GGN_iter_counter + 1}.npy",
                    #             )
                    #         ),
                    #         jnp.stack(datapoints),
                    #     )

                    # Compute batch GGNs
                    # t1 = time()
                    J_model, H_loss = compute_ggn_decomp(state, ggn_batch)  # [N, C, D], [N, C, C]
                    # t2 = time()
                    J_model = jax.device_put(J_model, jax.devices('cpu')[0])
                    H_loss = jax.device_put(H_loss, jax.devices('cpu')[0])
                    # t3 = time()
                    GGN = compute_ggn_jit(J_model, H_loss)
                    # t4 = time()
                    # print("compute_ggn_decomp:", t2 - t1)
                    # print("jax.device_put:", t3 - t2)
                    # print("compute_ggn:", t4 - t3)
                    # print("----------")

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
                            if norm_saving == "next":
                                prev_GGN_samples = load_ggn(
                                    n_steps, results_path, batch_size=prev_ggn_batch_size
                                )
                                save_norm(
                                    prev_GGN_samples,
                                    GGN_samples,
                                    n_steps,
                                    results_path,
                                    batch_size=prev_ggn_batch_size,
                                )
                            # GGN-saving "disabled" and not norm-saving "total" or "last": Remove previous batched GGN samples
                            if ggn_saving == "disabled" and norm_saving not in {"total", "last"}:
                                remove_ggn(n_steps, results_path, batch_size=prev_ggn_batch_size)

                        # Not norm-saving "next" or not ggn-saving "disabled" or not last GGN samples: Save GGN samples
                        if (
                            norm_saving != "next"
                            or ggn_saving != "disabled"
                            or ggn_batch_size_idx + 1 < len(ggn_batch_sizes)
                        ):
                            save_ggn(
                                GGN_samples,
                                n_steps,
                                results_path,
                                batch_size=aggregated_batch_size,
                            )

                        # Norm-saving "next" or "last" and last GGN samples: Stop further GGN computations
                        if norm_saving in {"next", "last"} and ggn_batch_size_idx + 1 == len(
                            ggn_batch_sizes
                        ):
                            break

                    # Norm-saving "disabled" or "total": Compute total GGN as running average to save memory
                    if norm_saving in {"disabled", "total"}:
                        GGN_counter += GGN.shape[0]
                        if GGN_total is None:
                            GGN_total = jnp.mean(GGN, axis=0)  # [D, D]
                        else:
                            GGN_total = aggregate_ggn_total_jit(
                                GGN_total, GGN, GGN_counter
                            )  # [D, D]

                # GGN-saving "dense" and not Norm-saving "disabled" or "total": Save total GGN on disk
                if ggn_saving == "dense" and norm_saving in {"disabled", "total"}:
                    save_ggn(GGN_total, n_steps, results_path)  # type: ignore

                # Norm-saving "total": Compute norm
                if norm_saving == "total":
                    for ggn_batch_size in ggn_batch_sizes:
                        GGN_samples = load_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                        save_norm(
                            GGN_samples,
                            GGN_total,  # type: ignore
                            n_steps,
                            results_path,
                            batch_size=ggn_batch_size,
                        )
                        # GGN-saving "disabled" : Remove batched GGN samples
                        if ggn_saving == "disabled":
                            remove_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                # Norm-saving "last": Compute norm
                elif norm_saving == "last":
                    GGN_samples_last = load_ggn(
                        n_steps, results_path, batch_size=ggn_batch_sizes[-1]
                    )
                    for ggn_batch_size in ggn_batch_sizes[:-1]:
                        GGN_samples = load_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                        save_norm(
                            GGN_samples,
                            GGN_samples_last,
                            n_steps,
                            results_path,
                            batch_size=ggn_batch_size,
                        )
                        # GGN-saving "disabled" : Remove batched GGN samples
                        if ggn_saving == "disabled":
                            remove_ggn(n_steps, results_path, batch_size=ggn_batch_size)
                    if ggn_saving == "disabled":
                        remove_ggn(n_steps, results_path, batch_size=ggn_batch_sizes[-1])

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
    no_progress_bar: bool,
) -> Tuple[float, float, jax.Array]:
    """
    Performs a single training epoch.

    Args:
        state (TrainState): Current training state.
        test_dataloader (DataLoader): Data loader for model training.
        no_progress_bar (bool): Disables progress bar.

    Returns:
        Tuple[float, float, jax.Array]:
            Epoch loss,
            epoch accuracy,
            epoch accuracy per class.
    """

    n_classes = len(test_dataloader.dataset.classes)  # type: ignore
    test_step_jit = jax.jit(partial(test_step, n_classes=n_classes))

    # Running statistics
    loss_epoch = []  # Per-item losses per test steps
    n_correct_epoch = 0  # Number of correct predictions across the epoch
    n_correct_per_class_epoch = jnp.zeros((n_classes,), dtype=int)
    n_per_class_epoch = jnp.zeros_like(n_correct_per_class_epoch)

    # Start epoch
    pbar_stats = {"loss": 0.0, "acc": 0.0}
    with tqdm(
        total=len(test_dataloader), desc="Test-step", disable=no_progress_bar, postfix=pbar_stats
    ) as pbar:
        # Iterate over dataset for testing
        for batch in test_dataloader:
            # Perform test step
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

    # Compute final epoch statistics: Epoch loss, epoch accuracy (per class)
    loss = jnp.mean(jnp.concatenate(loss_epoch)).item()  # [1]
    accuracy = float(n_correct_epoch / len(test_dataloader.dataset))  # type: ignore
    accuracy_per_class = n_correct_per_class_epoch / n_per_class_epoch  # type: ignore

    return loss, accuracy, accuracy_per_class
