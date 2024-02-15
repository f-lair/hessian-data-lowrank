import pathlib
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

sys_path = str(pathlib.Path(__file__).parent.parent.resolve())
if sys_path not in sys.path:
    sys.path.append(sys_path)

import jax
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from tqdm import trange

from src.data_loader import DataLoader
from src.data_utils import get_dataset, get_sampler
from src.log_utils import save_train_log
from src.model import get_model
from src.train_utils import test_epoch, test_step, train_epoch


def main() -> None:
    parser = ArgumentParser("Hessian Data Low-Rank Training Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2-reg", type=float, default=1e-3, help="L2 regularizer weighting.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=4000, help="Checkpoint interval."
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="../checkpoints/", help="Checkpoint path."
    )
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
    train_sampler = get_sampler(
        "uniform",
        train_dataset,
        args.rng_seed,
        test_step,
        args.train_batch_size,
        0,
        args.no_progress_bar,
    )
    test_sampler = get_sampler(
        "sequential",
        test_dataset,
        args.rng_seed,
        test_step,
        args.train_batch_size,
        0,
        args.no_progress_bar,
    )

    train_dataloader = DataLoader(train_dataset, args.train_batch_size, train_sampler)
    test_dataloader = DataLoader(test_dataset, args.train_batch_size, test_sampler)

    # Setup model
    model = get_model(args.dataset, args.hidden_dim)
    prng_key, model_init_key = jax.random.split(prng_key)
    params = model.init(model_init_key, train_dataset[0][0])
    tx = optax.sgd(args.lr)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Setup checkpointing
    if Path(args.checkpoint_path).exists():
        shutil.rmtree(args.checkpoint_path)
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        save_interval_steps=args.checkpoint_interval, create=True
    )
    checkpoint_manager = CheckpointManager(args.checkpoint_path, checkpointer, checkpoint_options)

    pbar_stats = {"loss": 0.0, "acc": 0.0, "test-loss": 0.0, "test-acc": 0.0}
    pbar = trange(args.epochs, desc="Epoch", disable=args.no_progress_bar, postfix=pbar_stats)
    n_steps = 0
    train_log = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
    train_log.update({f"train_acc_c{idx}": [] for idx in range(len(train_dataset.classes))})  # type: ignore
    train_log.update({f"test_acc_c{idx}": [] for idx in range(len(test_dataset.classes))})  # type: ignore

    # Start training
    for epoch_idx in pbar:
        # Perform training epoch
        train_state, loss, accuracy, accuracy_per_class, n_steps = train_epoch(
            train_state,
            train_dataloader,
            args.l2_reg,
            n_steps,
            args.no_progress_bar,
            checkpoint_manager,
        )
        pbar_stats["loss"] = round(loss, 6)
        pbar_stats["acc"] = round(accuracy, 4)
        # Update train log
        train_log["train_loss"].append(loss)
        train_log["train_acc"].append(accuracy)
        for idx, val in enumerate(accuracy_per_class):
            train_log[f"train_acc_c{idx}"].append(val)

        # Perform testing epoch
        if not args.no_testing:
            test_loss, test_accuracy, test_accuracy_per_class = test_epoch(
                train_state,
                test_dataloader,
                args.l2_reg,
                args.no_progress_bar,
            )
            pbar_stats["test-loss"] = round(test_loss, 6)
            pbar_stats["test-acc"] = round(test_accuracy, 4)
            # Update train log
            train_log["test_loss"].append(test_loss)
            train_log["test_acc"].append(test_accuracy)
            for idx, val in enumerate(test_accuracy_per_class):
                train_log[f"test_acc_c{idx}"].append(val)

        # Update progress bar
        pbar.set_postfix(pbar_stats)

    # Save train log
    save_train_log(train_log, args.checkpoint_path)


if __name__ == "__main__":
    main()
