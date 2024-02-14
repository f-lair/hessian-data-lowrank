import jax
from flax import linen as nn


def get_model(dataset: str, hidden_dim: int) -> nn.Module:
    """
    Returns model specified by CLI arguments.

    Args:
        dataset (str): Dataset name.
        hidden_dim (int): Hidden layer width.

    Raises:
        ValueError: Unsupported dataset.

    Returns:
        nn.Module: Model specified by CLI arguments.
    """

    if dataset == "mnist":
        return MNIST_MLP(hidden_dim)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


class MNIST_MLP(nn.Module):
    """MLP model for MNIST classification."""

    hidden_dim: int

    def setup(self) -> None:
        """
        Initializes model.

        Args:
            hidden_dim (int): Hidden layer width.
        """

        self.lin1 = nn.Dense(self.hidden_dim)
        self.lin2 = nn.Dense(10)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Performs forward pass.

        Args:
            x (jax.Array): Model input.

        Returns:
            jax.Array: Model output.
        """

        x = self.lin1(x)
        x = nn.relu(x)
        x = self.lin2(x)

        return x
