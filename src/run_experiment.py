import copy
import pathlib
import sys
from argparse import ArgumentParser
from functools import partial

sys_path = str(pathlib.Path(__file__).parent.parent.resolve())
if sys_path not in sys.path:
    sys.path.append(sys_path)

import jax
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
from tqdm import tqdm

from src.data_loader import DataLoader
from src.data_utils import get_dataset, get_sampler
from src.log_utils import save_experiment_results
from src.matfree_utils import (
    eigen_matfree,
    frobenius_inv_matfree,
    frobenius_matfree,
    laplace_matfree,
)
from src.model import get_model, loss_fn, model_fn
from src.sampler import WeightedSampler
from src.train_utils import test_step


def main() -> None:
    parser = ArgumentParser("Hessian Data Low-Rank Experiment Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for experiments.")
    parser.add_argument("--l2-reg", type=float, default=1e-3, help="L2 regularizer weighting.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="frobenius",
        help="Experiment: frobenius (default), frobenius-inv, eigen, laplace.",
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
        help="Sampling method for GGN computation: uniform (default), loss(-x), gradnorm(-x); x={inv, class, classeq, class-inv, classeq-inv}.",
    )
    parser.add_argument(
        "--full-ggn",
        default=False,
        action="store_true",
        help="Uses full training dataset to compute GGN.",
    )
    parser.add_argument(
        "--num-hutchinson-samples",
        type=int,
        default=5_000,
        help="Number of samples used for Hutchinson estimator.",
    )
    parser.add_argument(
        "--lanczos-order",
        type=int,
        default=3,
        help="Order used for Lanczos quadrature.",
    )
    parser.add_argument(
        "--num-eigvals",
        type=int,
        default=10,
        help="Number of computed top-eigenvalues and eigenvectors.",
    )
    parser.add_argument(
        "--num-lobpcg-iterations",
        type=int,
        default=100,
        help="Number of LOBPCG iterations.",
    )
    parser.add_argument(
        "--num-laplace-samples",
        type=int,
        default=200,
        help="Number of test datapoints used for laplace experiment.",
    )
    parser.add_argument(
        "--num-cg-iterations",
        type=int,
        default=20,
        help="Number of CG iterations.",
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument("--checkpoint-step", type=int, default=0, help="Checkpoint step.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="../checkpoints/", help="Checkpoint path."
    )
    parser.add_argument(
        "--no-progress-bar", default=False, action="store_true", help="Disables progress bar."
    )
    args = parser.parse_args()
    prng_key = jax.random.key(args.rng_seed)

    # Load data
    train_dataset = get_dataset(args.dataset, train=True, px=args.px, path=args.data_path)
    test_dataset = get_dataset(args.dataset, train=False, px=args.px, path=args.data_path)

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

    # Select experiment
    if args.experiment == "frobenius":
        experiment_fn = partial(
            frobenius_matfree,
            state=train_state,
            model_fn=model_fn,
            loss_fn=loss_fn,
            l2_reg=args.l2_reg,
            num_hutchinson_samples=args.num_hutchinson_samples,
        )
    elif args.experiment == "frobenius-inv":
        experiment_fn = partial(
            frobenius_inv_matfree,
            state=train_state,
            model_fn=model_fn,
            loss_fn=loss_fn,
            l2_reg=args.l2_reg,
            num_hutchinson_samples=args.num_hutchinson_samples,
            lanczos_order=args.lanczos_order,
        )
    elif args.experiment == "eigen":
        experiment_fn = partial(
            eigen_matfree,
            state=train_state,
            model_fn=model_fn,
            loss_fn=loss_fn,
            l2_reg=args.l2_reg,
            num_eigvals=args.num_eigvals,
            num_lobpcg_iterations=args.num_lobpcg_iterations,
        )
    elif args.experiment == "laplace":
        experiment_fn = partial(
            laplace_matfree,
            state=train_state,
            model_fn=model_fn,
            loss_fn=loss_fn,
            l2_reg=args.l2_reg,
            num_classes=len(test_dataset.classes),  # type: ignore
            num_laplace_samples=args.num_laplace_samples,
            num_cg_iterations=args.num_cg_iterations,
            num_hutchinson_samples=args.num_hutchinson_samples,
            no_progress_bar=args.no_progress_bar,
        )
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")

    # dataloader, num_samples, prng_key
    if args.full_ggn:
        sample_size = len(train_dataset)  # type: ignore
        train_sampler = get_sampler(
            "sequential",
            train_dataset,
            args.rng_seed,
            test_step,
            args.batch_size,
            0,
            args.no_progress_bar,
        )
        test_sampler = get_sampler(
            "uniform",
            test_dataset,
            args.rng_seed,
            partial(test_step, l2_reg=args.l2_reg),
            1,
            0,
            args.no_progress_bar,
        )
        train_dataloader = DataLoader(train_dataset, args.batch_size, train_sampler)
        test_dataloader = DataLoader(test_dataset, 1, test_sampler)
        if args.experiment == "laplace":
            results = experiment_fn(
                data_loader=train_dataloader,
                test_data_loader=test_dataloader,  # type: ignore
                batch_size=args.batch_size,
                num_data_samples=sample_size,
                prng_key=prng_key,
            )
        else:
            results = experiment_fn(
                data_loader=train_dataloader,
                batch_size=args.batch_size,
                num_data_samples=sample_size,
                prng_key=prng_key,
            )  # type: ignore
        save_experiment_results(
            results,
            args.experiment,
            "full",
            sample_size,
            args.checkpoint_step,
            args.checkpoint_path,
        )
    else:
        sample_sizes = [
            2**exp for exp in range(args.sample_size_min_exp, args.sample_size_max_exp + 1)
        ]

        train_sampler = get_sampler(
            args.sampling,
            train_dataset,
            args.rng_seed,
            test_step,
            args.batch_size,
            1,
            args.no_progress_bar,
        )
        # Update weights, if WeightedSampler is used
        if isinstance(train_sampler, WeightedSampler):
            train_sampler.update(train_state)
        train_samplers = {
            sample_size: copy.deepcopy(train_sampler) for sample_size in sample_sizes
        }

        for sample_size in tqdm(sample_sizes, desc="Experiment", disable=args.no_progress_bar):
            batch_size = min(args.batch_size, sample_size)

            test_sampler = get_sampler(
                "uniform",
                test_dataset,
                args.rng_seed,
                partial(test_step, l2_reg=args.l2_reg),
                1,
                0,
                args.no_progress_bar,
            )

            train_dataloader = DataLoader(train_dataset, batch_size, train_samplers[sample_size])
            test_dataloader = DataLoader(test_dataset, 1, test_sampler)
            if args.experiment == "laplace":
                results = experiment_fn(
                    data_loader=train_dataloader,
                    test_data_loader=test_dataloader,  # type: ignore
                    batch_size=batch_size,
                    num_data_samples=sample_size,
                    prng_key=prng_key,
                )
            else:
                results = experiment_fn(
                    data_loader=train_dataloader,
                    batch_size=batch_size,
                    num_data_samples=sample_size,
                    prng_key=prng_key,
                )  # type: ignore
            save_experiment_results(
                results,
                args.experiment,
                args.sampling,
                sample_size,
                args.checkpoint_step,
                args.checkpoint_path,
            )


if __name__ == "__main__":
    main()
