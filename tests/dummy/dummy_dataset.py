from typing import Tuple

import jax
import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self, N: int, D: int, num_classes: int) -> None:
        super().__init__()

        prng_key = jax.random.split(jax.random.PRNGKey(7), (2,))
        self.N = N
        self.D = D
        self.num_classes = num_classes

        self.x = np.array(jax.random.normal(prng_key[0], (N, D)), copy=True)
        self.y = np.array(jax.random.randint(prng_key[0], (N,), 0, 2), copy=True)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]
