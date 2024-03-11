import pathlib
from argparse import ArgumentParser

import jax
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer

from data_loader import DataLoader
from data_utils import get_dataset, get_sampler
from experiment_utils import start_experiment
from model import get_model
from train_utils import test_step


def main() -> None:
    parser = ArgumentParser("GGN Sample-Approximation Experiment Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="frobenius",
        help="Experiment: frobenius (default), eigen, laplace.",
    )
    parser.add_argument(
        "--sample-size-min-exp",
        type=int,
        default=0,
        help="Min exponent of base-2 variable sample size for GGN computation.",
    )
    parser.add_argument(
        "--sample-size-max-exp",
        type=int,
        default=12,
        help="Max exponent of base-2 variable batch size for GGN computation.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        help="Sampling method for GGN computation: uniform (default), loss(-x), gradnorm(-x); x={inv, inter, intra, inter-inv, intra-inv}.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of independent samples per experiment.",
    )
    parser.add_argument(
        "--prior-precision",
        type=float,
        default=100.0,
        help="Prior precision for Laplace approximation.",
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument("--checkpoint-step", type=int, default=0, help="Checkpoint step.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="../checkpoints/", help="Checkpoint path."
    )
    parser.add_argument("--results-path", type=str, default="../results/", help="Results path.")
    parser.add_argument(
        "--compose-on-cpu",
        default=False,
        action="store_true",
        help="Computes GGN realization on CPU instead of GPU (might exceed GPU memory otherwise).",
    )
    parser.add_argument(
        "--no-progress-bar", default=False, action="store_true", help="Disables progress bar."
    )
    args = parser.parse_args()

    # Load data
    train_dataset = get_dataset(args.dataset, train=True, px=args.px, path=args.data_path)
    test_dataset = get_dataset(args.dataset, train=False, px=args.px, path=args.data_path)
    sample_sampler = get_sampler(
        args.sampling,
        train_dataset,
        args.rng_seed + 1,
        test_step,
        args.batch_size,
        args.num_samples,
        args.no_progress_bar,
    )
    total_sampler = get_sampler(
        "sequential",
        train_dataset,
        args.rng_seed,
        test_step,
        args.batch_size,
        0,
        args.no_progress_bar,
    )
    test_sampler = get_sampler(
        "sequential",
        test_dataset,
        args.rng_seed,
        test_step,
        args.batch_size,
        0,
        args.no_progress_bar,
    )
    sample_dataloader = DataLoader(train_dataset, args.num_samples, sample_sampler)
    total_dataloader = DataLoader(train_dataset, args.num_samples, total_sampler)
    test_dataloader = DataLoader(test_dataset, args.batch_size, test_sampler)

    # Setup checkpointing
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(
        str(pathlib.Path(args.checkpoint_path).resolve()), checkpointer
    )

    # Setup model
    model = get_model(args.dataset, args.hidden_dim)
    params = checkpoint_manager.restore(args.checkpoint_step)["params"]
    tx = optax.sgd(1.0)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Setup experiment
    ggn_sample_sizes = [
        2**exp for exp in range(args.sample_size_min_exp, args.sample_size_max_exp + 1)
    ]

    # Start experiment
    start_experiment(
        train_state,
        sample_dataloader,
        total_dataloader,
        test_dataloader,
        ggn_sample_sizes,
        args.prior_precision,
        args.experiment,
        args.checkpoint_step,
        args.compose_on_cpu,
        args.no_progress_bar,
        args.results_path,
    )


if __name__ == "__main__":
    main()
