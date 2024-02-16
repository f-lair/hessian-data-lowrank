import math
import pathlib
import sys
from functools import partial
from typing import Any, Callable, Tuple

sys_path = str(pathlib.Path(__file__).parent.parent.resolve())
if sys_path not in sys.path:
    sys.path.append(sys_path)

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
from flax.training.train_state import TrainState
from jax.experimental.sparse.linalg import lobpcg_standard
from matfree import hutchinson, lanczos
from tqdm import tqdm

from src.ad_utils import (
    get_param_count,
    get_param_split,
    loss_hvp,
    model_jvp,
    model_jvp_pytree,
    model_vjp,
)
from src.data_loader import DataLoader


def ggn_matfree(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
) -> Callable:
    model_jvp_batched = jax.vmap(model_jvp, in_axes=(None, 0, 0, None, None))
    loss_hvp_batched = jax.vmap(loss_hvp, in_axes=(0, 0, 0, None))
    model_vjp_batched = jax.vmap(model_vjp, in_axes=(None, 0, 0, None))

    param_split = get_param_split(state)

    num_iterations = num_data_samples // batch_size
    num_data_samples = num_iterations * batch_size

    x_buffer, y_buffer = [], []

    for idx, (x, y) in enumerate(data_loader):
        if idx >= num_iterations:
            break
        x_buffer.append(x)
        y_buffer.append(y)

    x_buffer = jnp.concatenate(x_buffer, axis=0)
    y_buffer = jnp.concatenate(y_buffer, axis=0)

    @jax.jit
    def _ggn_matfree(v: jax.Array) -> jax.Array:
        data_iterator = iter(data_loader)
        carry = l2_reg * v

        def __ggn_matfree(idx: int, carry: jax.Array) -> jax.Array:
            low = idx * batch_size
            x = jax.lax.dynamic_slice_in_dim(x_buffer, low, batch_size)
            y = jax.lax.dynamic_slice_in_dim(y_buffer, low, batch_size)

            N = x.shape[0]
            v_batched = jnp.broadcast_to(v[None, ...], (N,) + v.shape)
            y_pred, v_batched = model_jvp_batched(
                state, x, v_batched, model_fn, param_split
            )  # [N, C]
            _, v_batched = loss_hvp_batched(y, y_pred, v_batched, loss_fn)  # [N, C]
            v_batched = 1 / num_data_samples * v_batched  # [N, C]
            _, v_batched = model_vjp_batched(state, x, v_batched, model_fn)  # [N, D]
            v_batched = jax.tree_util.tree_map(lambda _v_: jnp.sum(_v_, axis=0), v_batched)  # [D]
            v_batched = jax.flatten_util.ravel_pytree(v_batched)[0]  # [D]
            return v_batched + carry  # [D]

        return jax.lax.fori_loop(
            0,
            num_iterations,
            __ggn_matfree,
            carry,
        )

    return _ggn_matfree


def ggn_matfree_nd(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
) -> Callable:
    model_jmp_batched = jax.vmap(
        jax.vmap(model_jvp, in_axes=(None, 0, 0, None, None)),
        in_axes=(None, None, 2, None, None),
        out_axes=(None, 2),
    )
    loss_hmp_batched = jax.vmap(
        jax.vmap(loss_hvp, in_axes=(0, 0, 0, None)),
        in_axes=(None, None, 2, None),
        out_axes=(None, 2),
    )
    model_mjp_batched = jax.vmap(
        jax.vmap(model_vjp, in_axes=(None, 0, 0, None)),
        in_axes=(None, None, 2, None),
        out_axes=(None, 1),
    )

    param_split = get_param_split(state)

    num_iterations = num_data_samples // batch_size
    num_data_samples = num_iterations * batch_size

    x_buffer, y_buffer = [], []

    for idx, (x, y) in enumerate(data_loader):
        if idx >= num_iterations:
            break
        x_buffer.append(x)
        y_buffer.append(y)

    x_buffer = jnp.concatenate(x_buffer, axis=0)
    y_buffer = jnp.concatenate(y_buffer, axis=0)

    @jax.jit
    def _ggn_matfree_nd(m: jax.Array) -> jax.Array:
        data_iterator = iter(data_loader)
        m_shape = m.shape
        carry = l2_reg * m

        def __ggn_matfree_nd(idx: int, carry: jax.Array) -> jax.Array:
            low = idx * batch_size
            x = jax.lax.dynamic_slice_in_dim(x_buffer, low, batch_size)
            y = jax.lax.dynamic_slice_in_dim(y_buffer, low, batch_size)

            N = x.shape[0]
            m_batched = jnp.broadcast_to(m[None, ...], (N,) + m.shape)  # [N, D, M]
            y_pred, m_batched = model_jmp_batched(
                state, x, m_batched, model_fn, param_split
            )  # [N, C, M]
            _, m_batched = loss_hmp_batched(y, y_pred, m_batched, loss_fn)  # [N, C, M]
            m_batched = 1 / num_data_samples * m_batched  # [N, C, M]
            _, m_batched = model_mjp_batched(state, x, m_batched, model_fn)  # [N, M, D]
            m_batched = jax.tree_util.tree_map(
                lambda _m_: jnp.sum(_m_, axis=0), m_batched
            )  # [M, D]
            m_batched = jnp.concatenate(
                [_m_.reshape(*(m_shape[1:]), -1) for _m_ in jax.tree_util.tree_leaves(m_batched)],
                axis=-1,
            ).T  # [D, M]
            return m_batched + carry  # [D, M]

        return jax.lax.fori_loop(
            0,
            num_iterations,
            __ggn_matfree_nd,
            carry,
        )

    return _ggn_matfree_nd


def ggn_matfree_pytree(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
) -> Callable:
    model_jvp_pytree_batched = jax.vmap(model_jvp_pytree, in_axes=(None, 0, 0, None))
    loss_hvp_batched = jax.vmap(loss_hvp, in_axes=(0, 0, 0, None))
    model_vjp_batched = jax.vmap(model_vjp, in_axes=(None, 0, 0, None))

    num_iterations = num_data_samples // batch_size
    num_data_samples = num_iterations * batch_size

    x_buffer, y_buffer = [], []

    for idx, (x, y) in enumerate(data_loader):
        if idx >= num_iterations:
            break
        x_buffer.append(x)
        y_buffer.append(y)

    x_buffer = jnp.concatenate(x_buffer, axis=0)
    y_buffer = jnp.concatenate(y_buffer, axis=0)

    @jax.jit
    def _ggn_matfree_pytree(v: Any) -> Any:
        # data_iterator = iter(data_loader)
        carry = jax.tree_util.tree_map(lambda _v_: l2_reg * _v_, v)

        def __ggn_matfree_pytree(idx: int, carry: Any) -> Any:
            # x, y = next(data_iterator)
            low = idx * batch_size
            x = jax.lax.dynamic_slice_in_dim(x_buffer, low, batch_size)
            y = jax.lax.dynamic_slice_in_dim(y_buffer, low, batch_size)

            N = x.shape[0]
            v_batched = jax.tree_util.tree_map(
                lambda _v_: jnp.broadcast_to(_v_[None, ...], (N,) + _v_.shape), v
            )[
                0
            ]  # [N, D]
            y_pred, v_batched = model_jvp_pytree_batched(state, x, v_batched, model_fn)  # [N, C]
            _, v_batched = loss_hvp_batched(y, y_pred, v_batched, loss_fn)  # [N, C]
            v_batched = 1 / num_data_samples * v_batched  # [N, C]
            _, v_batched = model_vjp_batched(state, x, v_batched, model_fn)  # [N, D]
            return jax.tree_util.tree_map(
                lambda _v_, __v__: jnp.sum(_v_, axis=0) + __v__, v_batched, carry
            )  # [D]

        return jax.lax.fori_loop(
            0,
            num_iterations,
            __ggn_matfree_pytree,
            carry,
        )

    return _ggn_matfree_pytree


def frobenius_matfree(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
    num_hutchinson_samples: int,
    prng_key: jax.Array,
) -> jax.Array:
    param_count = get_param_count(state)
    v = jnp.empty((param_count,))

    _ggn_matfree = ggn_matfree(
        state, data_loader, model_fn, loss_fn, num_data_samples, batch_size, l2_reg
    )

    sampler = hutchinson.sampler_rademacher(v, num=num_hutchinson_samples)
    integrand = hutchinson.integrand_frobeniusnorm_squared(_ggn_matfree)
    estimator = hutchinson.hutchinson(integrand, sampler)

    return estimator(prng_key)


def frobenius_inv_matfree(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
    num_hutchinson_samples: int,
    lanczos_order: int,
    prng_key: jax.Array,
) -> jax.Array:
    param_count = get_param_count(state)
    v = jnp.empty((param_count,))

    _ggn_matfree = ggn_matfree(
        state, data_loader, model_fn, loss_fn, num_data_samples, batch_size, l2_reg
    )

    sampler = hutchinson.sampler_rademacher(v, num=num_hutchinson_samples)
    integrand = lanczos.integrand_product(
        lambda x: 1 / x, lanczos_order, _ggn_matfree, _ggn_matfree
    )
    estimator = hutchinson.hutchinson(integrand, sampler)

    return estimator(prng_key)


def eigen_matfree(
    state: TrainState,
    data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
    num_eigvals: int,
    num_lobpcg_iterations: int,
    prng_key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    param_count = get_param_count(state)

    _ggn_matfree = ggn_matfree_nd(
        state, data_loader, model_fn, loss_fn, num_data_samples, batch_size, l2_reg
    )

    U = jax.random.normal(prng_key, (param_count, num_eigvals))
    w, U, _ = lobpcg_standard(_ggn_matfree, U, m=num_lobpcg_iterations)  # type: ignore
    return w, U


def ltk_matfree(
    state: TrainState,
    _ggn_matfree_pytree: Callable,
    model_fn: Callable,
    num_cg_iterations: int,
) -> Callable:

    @jax.jit
    def _ltk_matfree(x: jax.Array, v: jax.Array) -> jax.Array:
        _, v = model_vjp(state, x, v, model_fn)  # [D]
        v = jsp.sparse.linalg.cg(_ggn_matfree_pytree, v, maxiter=num_cg_iterations)[0][0]  # [D]
        _, v = model_jvp_pytree(state, x, v, model_fn)  # [C]
        return v

    return _ltk_matfree


def laplace_matfree(
    state: TrainState,
    data_loader: DataLoader,
    test_data_loader: DataLoader,
    model_fn: Callable,
    loss_fn: Callable,
    num_data_samples: int,
    batch_size: int,
    l2_reg: float,
    num_classes: int,
    num_laplace_samples: int,
    num_cg_iterations: int,
    num_hutchinson_samples: int,
    prng_key: jax.Array,
    no_progress_bar: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    v = jnp.empty((num_classes,))
    num_datapoints = len(test_data_loader.dataset)  # type: ignore

    _ggn_matfree_pytree = ggn_matfree_pytree(
        state, data_loader, model_fn, loss_fn, num_data_samples, batch_size, l2_reg
    )

    _ltk_matfree = ltk_matfree(
        state,
        _ggn_matfree_pytree,
        model_fn,
        num_cg_iterations,
    )

    sampler = hutchinson.sampler_rademacher(v, num=num_hutchinson_samples)
    prng_keys = jax.random.split(prng_key, (num_datapoints,))
    laplace_trace_results = []
    laplace_diagonal_results = []
    laplace_logits_results = []

    for idx, (x, _) in enumerate(
        tqdm(test_data_loader, desc="Laplace", total=num_laplace_samples, disable=no_progress_bar)
    ):
        if idx >= num_laplace_samples:
            break
        laplace_result = hutchinson.hutchinson(
            hutchinson.integrand_trace_and_diagonal(partial(_ltk_matfree, x[0])),
            sampler,
        )(prng_keys[idx])
        laplace_logits_result = model_fn(state, x)(state.params)
        laplace_trace_results.append(laplace_result["trace"])
        laplace_diagonal_results.append(laplace_result["diagonal"])
        laplace_logits_results.append(laplace_logits_result[0])

    laplace_trace_results = jnp.stack(laplace_trace_results, axis=0)
    laplace_diagonal_results = jnp.stack(laplace_diagonal_results, axis=0)
    laplace_logits_results = jnp.stack(laplace_logits_results, axis=0)

    return laplace_trace_results, laplace_diagonal_results, laplace_logits_results
