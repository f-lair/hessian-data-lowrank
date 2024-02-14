from argparse import ArgumentParser

import jax
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from data_loader import DataLoader
from data_utils import get_dataset, get_sampler
from log_utils import get_save_measure, save_train_log
from model import get_model
from train_utils import test_epoch, test_step, train_epoch, train_step


def main() -> None:
    parser = ArgumentParser("Hessian Data Low-Rank Training Script")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist (default).")
    parser.add_argument("--px", type=int, default=7, help="Downsampled image size per side.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2-reg", type=float, default=1e-6, help="L2 regularizer weighting.")
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
        help="Sampling method for GGN computation: uniform (default), loss(-x), gradnorm(-x); x={inv, class, classeq, class-inv, classeq-inv}.",
    )
    parser.add_argument(
        "--ggn-samples",
        type=int,
        default=8,
        help="Number of GGN samples per GGN iteration. Equivalent to the batch size per forward pass in GGN computation.",
    )
    parser.add_argument(
        "--measure",
        type=str,
        default="frobenius",
        help="Error measure: frobenius (default), eig-overlap.",
    )
    parser.add_argument(
        "--measure-saving",
        type=str,
        default="disabled",
        help="GGN error measure saving: disabled (default), total, next, last.",
    )
    parser.add_argument(
        "--ggn-saving",
        type=str,
        default="dense",
        help="GGN saving: disabled, dense (default).",
    )
    parser.add_argument(
        "--uq",
        type=str,
        default="disabled",
        help="Computes uncertainty for test data using Laplace Approximations: disabled (default), sampled, total.",
    )
    parser.add_argument("--rng-seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--data-path", type=str, default="../data/", help="Data path.")
    parser.add_argument("--results-path", type=str, default="../results/", help="Results path.")
    parser.add_argument(
        "--compose-on-cpu",
        default=False,
        action="store_true",
        help="Computes GGN realization on CPU instead of GPU (might exceed GPU memory otherwise).",
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
    ggn_sampler = get_sampler(
        args.ggn_sampling,
        train_dataset,
        args.rng_seed + 1,
        test_step,
        args.train_batch_size,
        args.ggn_samples,
        args.no_progress_bar,
    )
    ggn_total_sampler = get_sampler(
        "sequential",
        train_dataset,
        args.rng_seed,
        test_step,
        args.train_batch_size,
        0,
        args.no_progress_bar,
    )
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, train_sampler)
    test_dataloader = DataLoader(test_dataset, args.train_batch_size, test_sampler)
    ggn_dataloader = DataLoader(train_dataset, args.ggn_samples, ggn_sampler)
    ggn_total_dataloader = DataLoader(train_dataset, args.ggn_samples, ggn_total_sampler)

    # Get measure
    save_measure = get_save_measure(args.measure, len(train_dataset.classes), args.compose_on_cpu)  # type: ignore

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
    train_log = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
    train_log.update({f"train_acc_c{idx}": [] for idx in range(len(train_dataset.classes))})  # type: ignore
    train_log.update({f"test_acc_c{idx}": [] for idx in range(len(test_dataset.classes))})  # type: ignore

    # Start training
    for epoch_idx in pbar:
        # Perform training epoch
        train_state, loss, accuracy, accuracy_per_class, n_steps, n_ggn_iterations = train_epoch(
            train_state,
            train_dataloader,
            args.l2_reg,
            ggn_dataloader,
            ggn_total_dataloader,
            ggn_batch_sizes,
            args.ggn_freq,
            n_ggn_iterations,
            n_steps,
            prng_key,
            save_measure,
            args.measure_saving,
            args.ggn_saving,
            args.compose_on_cpu,
            args.no_progress_bar,
            args.results_path,
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
                ggn_dataloader,
                ggn_total_dataloader,
                ggn_batch_sizes,
                (
                    args.uq if epoch_idx == args.epochs - 1 else "disabled"
                ),  # UQ only after last epoch
                n_steps,
                prng_key,
                args.no_progress_bar,
                args.results_path,
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
    save_train_log(train_log, args.results_path)


if __name__ == "__main__":
    main()
