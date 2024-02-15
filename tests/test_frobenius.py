import sys

sys.path.append("../")

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from src.ad_utils import get_param_count
from src.matfree_utils import frobenius_inv_matfree, frobenius_matfree, ggn_matfree
from tests.conftest import dummy_material
from tests.dummy.dummy_model import DummyModel, loss_fn, model_fn


def test_ggn_matfree():
    state, dataloader, _ = dummy_material()

    num_params = get_param_count(state)
    num_classes = dataloader.dataset.num_classes  # type: ignore
    v = jnp.ones((num_params,))

    x, y = next(iter(dataloader))
    y_pred = model_fn(state, x)(state.params)

    l2_reg = 1e-2

    _ggn_matfree = ggn_matfree(
        state, dataloader, model_fn, loss_fn, x.shape[0], x.shape[0], l2_reg
    )

    def ggn_naive(_v):
        J = jax.jacrev(model_fn(state, x))(state.params)
        J = jnp.concatenate(
            [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
            axis=-1,
        )  # [N, C, D]
        H = (
            1
            / x.shape[0]
            * jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)
        )
        G = jnp.einsum("ijk,ijl,ilm->km", J, H, J)  # [D, D]
        G = G + l2_reg * jnp.eye(num_params)
        return G @ _v  # [D]

    out_ggn = _ggn_matfree(v)
    out_ggn_expected = ggn_naive(v)

    assert jnp.allclose(out_ggn, out_ggn_expected, rtol=1e-6, atol=1e-6)


def test_frobenius():
    state, dataloader, _ = dummy_material()

    num_params = get_param_count(state)
    num_classes = dataloader.dataset.num_classes  # type: ignore

    x, y = next(iter(dataloader))
    y_pred = model_fn(state, x)(state.params)

    l2_reg = 1e-2
    num_hutchinson_samples = 5_000
    prng_key = jax.random.PRNGKey(7)

    def frobenius_naive():
        J = jax.jacrev(model_fn(state, x))(state.params)
        J = jnp.concatenate(
            [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
            axis=-1,
        )  # [N, C, D]
        H = (
            1
            / x.shape[0]
            * jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)
        )
        G = jnp.einsum("ijk,ijl,ilm->km", J, H, J)  # [D, D]
        G = G + l2_reg * jnp.eye(num_params)
        return jnp.linalg.norm(G, ord="fro") ** 2  # [1]

    out_frobenius = frobenius_matfree(
        state,
        dataloader,
        model_fn,
        loss_fn,
        x.shape[0],
        x.shape[0],
        l2_reg,
        num_hutchinson_samples,
        prng_key,
    )
    out_frobenius_expected = frobenius_naive()

    assert jnp.allclose(out_frobenius, out_frobenius_expected, rtol=1e-2, atol=1e-2)


def test_frobenius_inv():
    state, dataloader, _ = dummy_material()

    num_params = get_param_count(state)
    num_classes = dataloader.dataset.num_classes  # type: ignore

    x, y = next(iter(dataloader))
    y_pred = model_fn(state, x)(state.params)

    l2_reg = 1e-2
    num_hutchinson_samples = 5_000
    lanczos_order = 3
    prng_key = jax.random.PRNGKey(7)

    def frobenius_inv_naive():
        J = jax.jacrev(model_fn(state, x))(state.params)
        J = jnp.concatenate(
            [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
            axis=-1,
        )  # [N, C, D]
        H = (
            1
            / x.shape[0]
            * jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)
        )
        G = jnp.einsum("ijk,ijl,ilm->km", J, H, J)  # [D, D]
        G = G + l2_reg * jnp.eye(num_params)
        U, s, V = jsp.linalg.svd(G)  # [D, D], [D], [D, D]
        G_inv = U @ jnp.multiply(1 / (s[:, None] + 1e-8), V)  # [D, D]
        return jnp.linalg.norm(G_inv, ord="fro") ** 2  # [1]

    out_frobenius_inv = frobenius_inv_matfree(
        state,
        dataloader,
        model_fn,
        loss_fn,
        x.shape[0],
        x.shape[0],
        l2_reg,
        num_hutchinson_samples,
        lanczos_order,
        prng_key,
    )
    out_frobenius_inv_expected = frobenius_inv_naive()

    assert jnp.allclose(out_frobenius_inv, out_frobenius_inv_expected, rtol=1e2, atol=1e2)
