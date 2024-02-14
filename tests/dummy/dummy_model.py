from typing import Any, Callable

import jax
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState


class DummyModel(nn.Module):

    num_classes: int

    def setup(self) -> None:
        self.lin1 = nn.Dense(10)
        self.lin2 = nn.Dense(self.num_classes)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lin1(x)
        x = nn.relu(x)
        x = self.lin2(x)

        return x

    # @staticmethod
    # def num_classes() -> int:
    #     return 5


def model_fn(state: TrainState, x: jax.Array) -> Callable:
    def _model_fn(params: FrozenDict[str, Any]) -> jax.Array:
        logits = state.apply_fn(params, x)  # [N, C]
        return logits

    return _model_fn


def loss_fn(y: jax.Array) -> Callable:
    def _loss_fn(logits: jax.Array):
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)  # [N]
        return loss  # type: ignore

    return _loss_fn
