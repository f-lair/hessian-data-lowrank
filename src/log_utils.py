import os
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import pandas as pd


def save_train_log(train_log: Dict[str, List[float]], checkpoint_path: str) -> None:
    # Create results dir, if not existing
    os.makedirs(checkpoint_path, exist_ok=True)

    df = pd.DataFrame.from_dict(train_log, orient='index').transpose()
    df.to_csv(str(Path(checkpoint_path, "train_log.csv")))


def save_experiment_results(
    results: Any,
    experiment: str,
    sampling: str,
    sample_size: int,
    checkpoint_step: int,
    checkpoint_path: str,
) -> None:
    sampling = sampling.replace('-', '_')
    os.makedirs(str(Path(checkpoint_path, str(checkpoint_step), sampling)), exist_ok=True)

    if experiment == "frobenius":
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"frobenius_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results,
        )
    elif experiment == "frobenius-inv":
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"frobenius_inv_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results,
        )
    elif experiment == "eigen":
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"eigvals_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results[0],
        )
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"eigvecs_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results[1],
        )
    elif experiment == "laplace":
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"laplace_trace_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results[0],
        )
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"laplace_diagonal_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results[1],
        )
        jnp.save(
            str(
                Path(
                    checkpoint_path,
                    str(checkpoint_step),
                    sampling,
                    f"laplace_logits_{sample_size}_{checkpoint_step}.npy",
                )
            ),
            results[2],
        )
