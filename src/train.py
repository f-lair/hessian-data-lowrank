from argparse import ArgumentParser

import jax
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from data_utils import DataLoader, get_dataset, get_sampler
from model import get_model
from train_utils import test_epoch, train_epoch


def main() -> None:
    parser = ArgumentParser("Hessian Data Low-Rank Training Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--ggn-batch-size-min-exp",
        type=int,
        default=0,
        help="Min exponent of base-2 variable batch size for GGN computation.",
    )
    parser.add_argument(
        "--ggn-batch-size-max-exp",
        type=int,
        default=12,
        help="Max exponent of base-2 variable batch size for GGN computation.",
    )
    parser.add_argument("--ggn-freq", type=int, default=4000, help="Frequency of GGN iterations.")
    parser.add_argument("--ggn-iterations", type=int, default=3, help="Number of GGN iterations.")
    parser.add_argument(
        "--ggn-sampling",
        type=str,
        default="uniform",
        help="Sampling method for GGN computation: uniform (default).",
    )
    parser.add_argument(
        "--ggn-samples",
        type=int,
        default=8,
        help="Number of GGN samples per GGN iteration. Equivalent to the batch size per forward pass in GGN computation.",
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument("--results-path", type=str, default="../results/", help="Results path.")
    parser.add_argument(
        "--no-testing",
        default=False,
        action="store_true",
        help="Disables testing on holdout data.",
    )
    parser.add_argument(
        "--no-progress-bar", default=False, action="store_true", help="Disables progress bar."
    )
    args = parser.parse_args()
    prng_key = jax.random.key(args.rng_seed)

    # Load data
    train_dataset = get_dataset(args.dataset, train=True, px=args.px, path=args.data_path)
    test_dataset = get_dataset(args.dataset, train=False, px=args.px, path=args.data_path)
    train_sampler = get_sampler("uniform", train_dataset, args.rng_seed)
    test_sampler = get_sampler("uniform", test_dataset, args.rng_seed)
    ggn_sampler = get_sampler(args.ggn_sampling, train_dataset, args.rng_seed + 1)
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, train_sampler)
    test_dataloader = DataLoader(test_dataset, args.train_batch_size, test_sampler)
    ggn_dataloader = DataLoader(train_dataset, args.ggn_samples, ggn_sampler)

    # Setup model
    model = get_model(args.dataset, args.hidden_dim)
    prng_key, model_init_key = jax.random.split(prng_key)
    params = model.init(model_init_key, train_dataset[0][0])
    tx = optax.sgd(args.lr)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    pbar_stats = {"loss": 0.0, "acc": 0.0, "test-loss": 0.0, "test-acc": 0.0}
    pbar = trange(args.epochs, desc="Epoch", disable=args.no_progress_bar, postfix=pbar_stats)
    ggn_batch_sizes = [
        2**exp for exp in range(args.ggn_batch_size_min_exp, args.ggn_batch_size_max_exp + 1)
    ]
    n_ggn_iterations = args.ggn_iterations
    n_steps = 0

    # Start training
    for epoch_idx in pbar:
        # Perform training epoch
        train_state, loss, accuracy, n_steps, n_ggn_iterations = train_epoch(
            train_state,
            train_dataloader,
            ggn_dataloader,
            ggn_batch_sizes,
            args.ggn_freq,
            n_ggn_iterations,
            n_steps,
            args.no_progress_bar,
            args.results_path,
        )
        pbar_stats["loss"] = round(loss, 6)
        pbar_stats["acc"] = round(accuracy, 4)

        # Perform testing epoch
        if not args.no_testing:
            test_loss, test_accuracy = test_epoch(
                train_state, test_dataloader, args.no_progress_bar
            )
            pbar_stats["test-loss"] = round(test_loss, 6)
            pbar_stats["test-acc"] = round(test_accuracy, 4)

        # Update progress bar
        pbar.set_postfix(pbar_stats)


if __name__ == "__main__":
    main()
