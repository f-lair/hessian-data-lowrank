import sys

sys.path.append("../")

import jax
import jax.numpy as jnp
import pytest

from src.ad_utils import (
    get_param_split,
    loss_hessian,
    loss_hvp,
    model_jvp,
    model_jvp_pytree,
    model_vjp,
)
from tests.conftest import dummy_material
from tests.dummy.dummy_model import DummyModel, loss_fn, model_fn


def test_model_jvp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))

    num_params = sum([p.size for p in jax.tree_util.tree_leaves(state.params)])
    num_classes = dataloader.dataset.num_classes  # type: ignore
    param_split = get_param_split(state)
    v = jnp.ones((x.shape[0], num_params))

    _model_jvp = jax.vmap(model_jvp, in_axes=(None, 0, 0, None, None))
    out_y, out_jvp = _model_jvp(state, x, v, model_fn, param_split)  # type: ignore

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_jvp_expected = jnp.einsum("ijk,ik->ij", J, v)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_jvp, out_jvp_expected)


def test_model_jmp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))

    num_params = sum([p.size for p in jax.tree_util.tree_leaves(state.params)])
    num_classes = dataloader.dataset.num_classes  # type: ignore
    param_split = get_param_split(state)
    M = 8
    m = jnp.ones((x.shape[0], num_params, M))

    _model_jmp = jax.vmap(
        jax.vmap(model_jvp, in_axes=(None, 0, 0, None, None)),
        in_axes=(None, None, 2, None, None),
        out_axes=(None, 2),
    )
    out_y, out_jmp = _model_jmp(state, x, m, model_fn, param_split)  # type: ignore

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_jmp_expected = jnp.einsum("ijk,ikl->ijl", J, m)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_jmp, out_jmp_expected)


def test_model_jvp_pytree():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))

    num_params = sum([p.size for p in jax.tree_util.tree_leaves(state.params)])
    num_classes = dataloader.dataset.num_classes  # type: ignore
    v = jnp.ones((x.shape[0], num_params))

    treedef = jax.tree_util.tree_structure(state.params)
    param_leaves = jax.tree_util.tree_leaves(state.params)
    param_split = jnp.cumsum(jnp.array([p.size for p in param_leaves]))[:-1]
    v_leaves = jnp.split(v, param_split, axis=1)  # type: ignore
    v_leaves = [
        v_leaf.reshape(*((x.shape[0],) + p.shape)) for (v_leaf, p) in zip(v_leaves, param_leaves)
    ]
    v_tree = jax.tree_util.tree_unflatten(treedef, v_leaves)

    _model_jvp_pytree = jax.vmap(model_jvp_pytree, in_axes=(None, 0, 0, None))
    out_y, out_jvp = _model_jvp_pytree(state, x, v_tree, model_fn)  # type: ignore

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_jvp_expected = jnp.einsum("ijk,ik->ij", J, v)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_jvp, out_jvp_expected)


def test_model_jmp_pytree():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))

    num_params = sum([p.size for p in jax.tree_util.tree_leaves(state.params)])
    num_classes = dataloader.dataset.num_classes  # type: ignore
    param_split = get_param_split(state)
    M = 8
    m = jnp.ones((x.shape[0], num_params, M))

    treedef = jax.tree_util.tree_structure(state.params)
    param_leaves = jax.tree_util.tree_leaves(state.params)
    m_leaves = jnp.split(m, param_split, axis=1)  # type: ignore
    m_leaves = [
        m_leaf.reshape(*((x.shape[0],) + p.shape + (M,)))
        for (m_leaf, p) in zip(m_leaves, param_leaves)
    ]
    m_tree = jax.tree_util.tree_unflatten(treedef, m_leaves)

    _model_jmp_pytree = jax.vmap(
        jax.vmap(model_jvp_pytree, in_axes=(None, 0, 0, None)),
        in_axes=(None, None, -1, None),
        out_axes=(None, -1),
    )
    out_y, out_jmp = _model_jmp_pytree(state, x, m_tree, model_fn)  # type: ignore

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_jmp_expected = jnp.einsum("ijk,ikl->ijl", J, m)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_jmp, out_jmp_expected)


def test_model_vjp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))
    num_classes = dataloader.dataset.num_classes  # type: ignore

    v = jnp.ones((x.shape[0], num_classes))

    _model_vjp = jax.vmap(model_vjp, in_axes=(None, 0, 0, None))
    out_y, out_vjp = _model_vjp(state, x, v, model_fn)  # type: ignore
    out_vjp = jnp.concatenate(
        [x.reshape(x.shape[0], -1) for x in jax.tree_util.tree_leaves(out_vjp)], axis=-1
    )

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_vjp_expected = jnp.einsum("ijk,ij->ik", J, v)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_vjp, out_vjp_expected)


def test_model_vmp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))
    num_classes = dataloader.dataset.num_classes  # type: ignore

    M = 8
    m = jnp.ones((x.shape[0], num_classes, M))

    _model_mjp = jax.vmap(
        jax.vmap(model_vjp, in_axes=(None, 0, 0, None)),
        in_axes=(None, None, 2, None),
        out_axes=(None, 1),
    )
    out_y, out_mjp = _model_mjp(state, x, m, model_fn)  # type: ignore
    out_mjp = jnp.concatenate(
        [x.reshape(x.shape[0], M, -1) for x in jax.tree_util.tree_leaves(out_mjp)],
        axis=-1,
    ).transpose((0, 2, 1))

    out_y_expected = model_fn(state, x)(state.params)
    J = jax.jacrev(model_fn(state, x))(state.params)
    J = jnp.concatenate(
        [x.reshape(x.shape[0], num_classes, -1) for x in jax.tree_util.tree_leaves(J)],
        axis=-1,
    )
    out_mjp_expected = jnp.einsum("ijk,ijl->ikl", J, m)

    assert jnp.allclose(out_y, out_y_expected)
    assert jnp.allclose(out_mjp, out_mjp_expected, rtol=1e-6, atol=1e-6)


def test_loss_hvp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))
    num_classes = dataloader.dataset.num_classes  # type: ignore

    y = jnp.ones((x.shape[0],), dtype=int)
    v = jnp.ones((x.shape[0], num_classes))
    y_pred = model_fn(state, x)(state.params)

    _loss_hvp = jax.vmap(loss_hvp, in_axes=(0, 0, 0, None))
    out_grad, out_hvp = _loss_hvp(y, y_pred, v, loss_fn)  # type: ignore

    out_grad_expected = jax.vmap(lambda _y, _y_pred: jax.grad(loss_fn(_y))(_y_pred))(y, y_pred)
    H = jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)
    out_hvp_expected = jnp.einsum("ijk,ik->ij", H, v)

    assert jnp.allclose(out_grad, out_grad_expected)
    assert jnp.allclose(out_hvp, out_hvp_expected, rtol=1e-6, atol=1e-6)


def test_loss_hmp():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))
    num_classes = dataloader.dataset.num_classes  # type: ignore

    y = jnp.ones((x.shape[0],), dtype=int)
    M = 8
    m = jnp.ones((x.shape[0], num_classes, M))
    y_pred = model_fn(state, x)(state.params)

    _loss_hmp = jax.vmap(
        jax.vmap(loss_hvp, in_axes=(0, 0, 0, None)),
        in_axes=(None, None, 2, None),
        out_axes=(None, 2),
    )
    out_grad, out_hmp = _loss_hmp(y, y_pred, m, loss_fn)  # type: ignore

    out_grad_expected = jax.vmap(lambda _y, _y_pred: jax.grad(loss_fn(_y))(_y_pred))(y, y_pred)
    H = jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)
    out_hmp_expected = jnp.einsum("ijk,ikl->ijl", H, m)

    assert jnp.allclose(out_grad, out_grad_expected)
    assert jnp.allclose(out_hmp, out_hmp_expected, rtol=1e-6, atol=1e-6)


def test_loss_hessian():
    state, dataloader, _ = dummy_material()
    x, _ = next(iter(dataloader))

    y = jnp.ones((x.shape[0],), dtype=int)
    y_pred = model_fn(state, x)(state.params)

    _loss_hessian = jax.vmap(loss_hessian, in_axes=(0, 0, None))
    out_H = _loss_hessian(y, y_pred, loss_fn)  # type: ignore

    out_H_expected = jax.vmap(lambda _y, _y_pred: jax.hessian(loss_fn(_y))(_y_pred))(y, y_pred)

    assert jnp.allclose(out_H, out_H_expected, rtol=1e-6, atol=1e-6)
