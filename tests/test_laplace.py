import sys

sys.path.append("../")

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from src.ad_utils import get_param_count
from src.matfree_utils import laplace_matfree
from tests.conftest import dummy_material
from tests.dummy.dummy_model import loss_fn, model_fn


def test_laplace():
    state, train_dataloader, test_dataloader = dummy_material()

    num_params = get_param_count(state)
    num_classes = test_dataloader.dataset.num_classes  # type: ignore

    x, y = next(iter(train_dataloader))
    y_pred = model_fn(state, x)(state.params)

    l2_reg = 1e-2
    num_hutchinson_samples = 5_000
    num_cg_iterations = 20
    prng_key = jax.random.PRNGKey(7)

    def laplace_naive():
        logits = model_fn(state, x)(state.params)
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
        G = G + l2_reg * jnp.eye(num_params)  # [D, D]
        G_chol = jsp.linalg.cho_factor(G)  # [D, D]
        LTK = jax.vmap(jsp.linalg.cho_solve, in_axes=(None, 0))(
            G_chol, J.transpose(0, 2, 1)
        )  # [N, D, C]
        LTK = jnp.einsum("ijk,ikl->ijl", J, LTK)  # [N, C, C]

        return (
            jnp.einsum("ijj->i", LTK),
            jnp.diagonal(LTK, axis1=1, axis2=2),
            logits,
        )  # [N], [N, C], [N, C]

    out_laplace_trace, out_laplace_diagonal, out_laplace_logits = laplace_matfree(
        state,
        train_dataloader,
        test_dataloader,
        model_fn,
        loss_fn,
        x.shape[0],
        x.shape[0],
        l2_reg,
        num_classes,
        x.shape[0],
        num_cg_iterations,
        num_hutchinson_samples,
        prng_key,
        True,
    )
    out_laplace_trace_expected, out_laplace_diagonal_expected, out_laplace_logits_expected = (
        laplace_naive()
    )

    assert jnp.allclose(out_laplace_trace, out_laplace_trace_expected, rtol=1e1, atol=1e1)
    assert jnp.allclose(out_laplace_diagonal, out_laplace_diagonal_expected, rtol=1e-1, atol=1e-1)
    assert jnp.allclose(out_laplace_logits, out_laplace_logits_expected)
