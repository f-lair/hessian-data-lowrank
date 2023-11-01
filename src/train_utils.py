from typing import Any, List, Tuple

import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import tree_util
from tqdm import tqdm

from data_utils import DataLoader


@jax.jit
def train_step(
    state: TrainState, batch: Tuple[jax.Array, jax.Array]
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """
    Performs a single training step without GGN computation.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.

    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.

    Returns:
        Tuple[TrainState, jax.Array, jax.Array]:
            Updated training state,
            per-item loss ([N]),
            number of correct predictions ([1]).
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
    )  # [1]; [N, C]; [D], pytree in D
    n_correct = jnp.sum(jnp.argmax(logits, -1) == y)  # [1]

    # Perform gradient step, update training state
    state = state.apply_gradients(grads=d_loss)

    return state, loss, n_correct


@jax.jit
def train_step_ggn(
    state: TrainState, batch: Tuple[jax.Array, jax.Array]
) -> Tuple[TrainState, jax.Array, jax.Array, jax.Array]:
    """
    Performs a single training step with GGN computation.
    C: Model output dim.
    D: Parameter dim.
    N: Batch dim.


    Args:
        state (TrainState): Current training state.
        batch (Tuple[jax.Array, jax.Array]): Batched input data.

    Returns:
        Tuple[TrainState, jax.Array, jax.Array, jax.Array]:
            Updated training state,
            per-item loss ([N]),
            per-item GGN ([N, D, D]),
            number of correct predictions ([1]).
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
    loss = loss_fn(logits, y)  # [N]
    n_correct = jnp.sum(jnp.argmax(logits, -1) == y)  # [1]
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
    #   - 'J_loss': Per-item gradient (=Jacobian) of loss w.r.t. model output
    #   - 'H_loss': Per-item Hessian of loss w.r.t. model output
    J_model = J_model_fn(state.params, x)  # [N, C, D], pytree in D
    J_loss = jax.vmap(d_loss_fn)(logits, y)  # [N, C]
    H_loss = jax.vmap(H_loss_fn)(logits, y)  # [N, C, C]

    # Compute regular gradient of loss w.r.t. model parameters as product of 'J_loss' and 'J_model'
    d_loss = tree_util.tree_map(
        lambda pytree: jnp.mean(jnp.einsum("nc...,nc->n...", pytree, J_loss), axis=0), J_model
    )  # [D], pytree in D

    # Transform 'J_model' from pytree representation into vector representation
    J_model = jnp.concatenate(
        [x.reshape(N, C, -1) for x in tree_util.tree_leaves(J_model)], axis=2
    )  # [N, C, D]

    # Compute per-item Generalized Gauss-Newton (GGN) matrix: J_model.T @ H_loss @ J_model
    GGN = jnp.einsum("nax,nab,nby->nxy", J_model, H_loss, J_model)  # [N, D, D]

    # Perform gradient step, update training state
    state = state.apply_gradients(grads=d_loss)

    return state, loss, GGN, n_correct


def train_epoch(
    state: TrainState,
    dataloader: DataLoader,
    n_ggn_samples: int,
    no_total_ggn: bool,
    no_progress_bar: bool,
) -> Tuple[TrainState, float, float, List[jax.Array], jax.Array | None]:
    """
    Performs a single training epoch.

    Args:
        state (TrainState): Current training state.
        dataloader (DataLoader): Data loader.
        n_ggn_samples (int): Max number of GGN samples per epoch.
        no_total_ggn (bool): Disables computation of total GGN.
        no_progress_bar (bool): Disables progress bar.

    Returns:
        Tuple[TrainState, float, float, List[jax.Array], jax.Array | None]:
            Updated training state,
            epoch loss,
            epoch accuracy,
            batched GGNs ('n_ggn_samples' x [D, D]),
            total GGN ([D, D]).
    """

    # Running statistics
    loss_epoch = []  # Per-item losses per training steps
    GGN_epoch = []  # Per-item GGNs per training steps
    n_correct_epoch = 0  # Number of correct predictions across the epoch
    GGN_total = None  # Total GGN, encompassing all per-item GGNs across the epoch
    GGN_counter = 0  # Number of already computed per-item GGNs, needed for running average

    # Start epoch
    pbar_stats = {"loss": 0.0, "acc": 0.0}
    with tqdm(
        total=len(dataloader), desc="Step", disable=no_progress_bar, postfix=pbar_stats
    ) as pbar:
        for step_idx, batch in enumerate(dataloader):
            # Perform training step
            # If no total GGN is needed and needed samples are already computed, omit GGN
            # computation to speed up training
            if no_total_ggn and step_idx > n_ggn_samples:
                state, loss, n_correct = train_step(state, batch)
            else:
                state, loss, GGN, n_correct = train_step_ggn(state, batch)

                # Update running statistics
                if step_idx < n_ggn_samples:
                    GGN_epoch.append(GGN)

                # Compute total GGN as running average to save memory
                if not no_total_ggn:
                    GGN_counter += GGN.shape[0]
                    if GGN_total is None:
                        GGN_total = jnp.mean(GGN, axis=0)  # [D, D]
                    else:
                        GGN_total = (
                            GGN_total + jnp.sum(GGN - GGN_total[None, :, :], axis=0) / GGN_counter
                        )  # [D, D]

            # Update running statistics
            loss_epoch.append(loss)
            n_correct_epoch += n_correct
            N = batch[0].shape[0]

            # Update progress bar
            pbar.update()
            pbar_stats["loss"] = round(float(jnp.mean(loss)), 6)
            pbar_stats["acc"] = round(n_correct / N, 4)
            pbar.set_postfix(pbar_stats)

    # Compute final epoch statistics: Epoch loss, epoch accuracy, batched GGNs (averaged per batch)
    loss = jnp.mean(jnp.concatenate(loss_epoch)).item()  # [1]
    accuracy = n_correct_epoch / len(dataloader.dataset)  # type: ignore
    GGN_batched = [jnp.mean(GGN, axis=0) for GGN in GGN_epoch]  # n_ggn_samples x [D, D]

    return state, loss, accuracy, GGN_batched, GGN_total
