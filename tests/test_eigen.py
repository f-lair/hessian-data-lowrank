import sys

sys.path.append("../")

import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest

from src.matfree_utils import eigen_matfree, ggn_matfree_nd
from tests.conftest import dummy_material
from tests.dummy.dummy_model import loss_fn, model_fn


def test_eigen():
    state, dataloader, _ = dummy_material()

    num_classes = dataloader.dataset.num_classes  # type: ignore

    x, _ = next(iter(dataloader))

    l2_reg = 1e-2
    num_lobpcg_iterations = 100
    prng_key = jax.random.PRNGKey(7)

    out_eigvals, out_eigvecs = eigen_matfree(
        state,
        dataloader,
        model_fn,
        loss_fn,
        x.shape[0],
        x.shape[0],
        l2_reg,
        num_classes,
        num_lobpcg_iterations,
        prng_key,
    )

    _ggn_matfree_nd = ggn_matfree_nd(
        state, dataloader, model_fn, loss_fn, x.shape[0], x.shape[0], l2_reg
    )

    lhs = _ggn_matfree_nd(out_eigvecs)
    rhs = out_eigvals[None, :] * out_eigvecs

    assert jnp.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)
