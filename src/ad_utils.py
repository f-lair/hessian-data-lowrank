from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


def get_param_split(state: TrainState) -> Tuple[int, ...]:
    param_leaves = jax.tree_util.tree_leaves(state.params)
    # cf. https://github.com/google/jax/issues/13769#issuecomment-1363171469
    return tuple(np.cumsum(np.array([p.size for p in param_leaves]))[:-1].tolist())


def get_param_count(state: TrainState) -> int:
    return sum([p.size for p in jax.tree_util.tree_leaves(state.params)])


@partial(jax.jit, static_argnums=(3, 4))
def model_jvp(
    state: TrainState, x: jax.Array, v: jax.Array, model_fn: Callable, param_split: Tuple[int, ...]
) -> Tuple[jax.Array, jax.Array]:
    treedef = jax.tree_util.tree_structure(state.params)
    param_leaves = jax.tree_util.tree_leaves(state.params)
    _model_fn = model_fn(state, x)

    v_leaves = jnp.split(v, param_split, axis=0)  # type: ignore
    v_leaves = [v_leaf.reshape(*p.shape) for (v_leaf, p) in zip(v_leaves, param_leaves)]
    v_tree = jax.tree_util.tree_unflatten(treedef, v_leaves)

    return jax.jvp(_model_fn, (state.params,), (v_tree,))  # type: ignore


@partial(jax.jit, static_argnums=(3,))
def model_jvp_pytree(
    state: TrainState, x: jax.Array, v: jax.Array, model_fn: Callable
) -> Tuple[jax.Array, jax.Array]:
    _model_fn = model_fn(state, x)

    return jax.jvp(_model_fn, (state.params,), (v,))  # type: ignore


@partial(jax.jit, static_argnums=(3,))
def model_vjp(
    state: TrainState, x: jax.Array, v: jax.Array, model_fn: Callable
) -> Tuple[Any, Any]:
    _model_fn = model_fn(state, x)
    y, _vjp = jax.vjp(_model_fn, state.params)

    return y, _vjp(v)  # type: ignore


@partial(jax.jit, static_argnums=(3,))
def loss_hvp(
    y: jax.Array, y_pred: jax.Array, v: jax.Array, loss_fn: Callable
) -> Tuple[jax.Array, jax.Array]:
    _loss_fn = loss_fn(y)

    return jax.jvp(jax.grad(_loss_fn), (y_pred,), (v,))


@partial(jax.jit, static_argnums=(2,))
def loss_hessian(y: jax.Array, y_pred: jax.Array, loss_fn: Callable) -> jax.Array:
    _loss_fn = loss_fn(y)

    return jax.jacfwd(jax.grad(_loss_fn))(y_pred)
