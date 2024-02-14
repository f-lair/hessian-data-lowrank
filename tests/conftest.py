import sys
from typing import Tuple

sys.path.append("../")

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from torch.utils.data import SequentialSampler

from src.data_loader import DataLoader
from tests.dummy.dummy_dataset import DummyDataset
from tests.dummy.dummy_model import DummyModel


def dummy_material() -> Tuple[TrainState, DataLoader, DataLoader]:
    N = 4
    D = 10
    num_classes = 5
    prng_key = jax.random.PRNGKey(7)

    dummy_model = DummyModel(num_classes)
    dummy_params = dummy_model.init(prng_key, jnp.empty((1, D)))

    dummy_dataset = DummyDataset(N, D, num_classes)
    dummy_train_dataloader = DataLoader(dummy_dataset, N, SequentialSampler(dummy_dataset))
    dummy_test_dataloader = DataLoader(dummy_dataset, 1, SequentialSampler(dummy_dataset))
    # for k1 in dummy_params["params"]:
    #     for k2 in dummy_params["params"][k1]:
    #         dummy_params["params"][k1][k2] = jnp.ones_like(dummy_params["params"][k1][k2])  # type: ignore
    dummy_tx = optax.sgd(0.001)
    dummy_state = TrainState.create(apply_fn=dummy_model.apply, params=dummy_params, tx=dummy_tx)

    return dummy_state, dummy_train_dataloader, dummy_test_dataloader
