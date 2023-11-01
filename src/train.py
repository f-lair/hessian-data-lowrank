from argparse import ArgumentParser

import jax
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from data_utils import DataLoader, get_dataset
from log_utils import save_results
from model import get_model
from train_utils import train_epoch


def main() -> None:
    parser = ArgumentParser("Hessian Data Low-Rank Training Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--ggn-samples", type=int, default=10, help="Max number of GGN samples per epoch."
    )
    parser.add_argument(
        "--no-total-ggn",
        default=False,
        action="store_true",
        help="Disables computation of total GGN.",
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument("--results-path", type=str, default="../results/", help="Results path.")
    parser.add_argument(
        "--no-progress-bar", default=False, action="store_true", help="Disables progress bar."
    )
    args = parser.parse_args()
    prng_key = jax.random.key(args.rng_seed)

    # Load data
    dataset = get_dataset(args.dataset, train=True, px=args.px, path=args.data_path)
    dataloader = DataLoader(dataset, args.batch_size, args.rng_seed)

    # Setup model
    model = get_model(args.dataset, args.hidden_dim)
    prng_key, model_init_key = jax.random.split(prng_key)
    params = model.init(model_init_key, dataset[0][0])
    tx = optax.sgd(args.lr)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    pbar_stats = {"loss": 0.0, "acc": 0.0}
    pbar = trange(args.epochs, desc="Epoch", disable=args.no_progress_bar, postfix=pbar_stats)

    # Start training
    for epoch_idx in pbar:
        # Perform training epoch
        train_state, loss, accuracy, GGN_batched, GGN_total = train_epoch(
            train_state, dataloader, args.ggn_samples, args.no_total_ggn, args.no_progress_bar
        )

        # Save results on disk
        save_results(GGN_batched, GGN_total, args.batch_size, epoch_idx, args.results_path)

        # Update progress bar
        pbar_stats["loss"] = round(loss, 6)
        pbar_stats["acc"] = round(accuracy, 4)
        pbar.set_postfix(pbar_stats)


if __name__ == "__main__":
    main()
