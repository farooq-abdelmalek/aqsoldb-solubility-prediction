"""
Benchmark molecular solubility regression models using MLflow.

Compares five regression algorithms across three feature sets
(15 experiments in total).

Scientific safeguards
---------------------
- Molecular grouping prevents identical grouping keys across splits.
- All learned preprocessing is fitted using training data only.
- Model comparison and selection use validation metrics.
- The held-out test set is NOT evaluated in this script.
"""

from pathlib import Path
from time import perf_counter
import hashlib

import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from src.preprocessing import canonicalize_smiles
from src.splitting import make_group_key, split_by_group
from src.features import build_feature_matrices
from src.models import build_models


DATA_PATH = Path("data/curated-solubility-dataset.csv")
TRACKING_DIR = Path("tracking")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


def prepare_dataset():
    """Load, canonicalize, and split the AqSolDB dataset."""

    df = pd.read_csv(DATA_PATH)

    # Preserve salts, counterions, charges, and stereochemistry.
    df["canonical_smiles"] = df["SMILES"].apply(
        canonicalize_smiles
    )

    df = df.dropna(subset=["canonical_smiles"]).copy()

    # Group keys are used for splitting, not model features.
    df["group_key"] = df["canonical_smiles"].apply(
        make_group_key
    )

    if df["group_key"].isna().any():
        raise ValueError("Some molecular grouping keys are missing.")

    df = split_by_group(df, random_state=RANDOM_STATE)

    # Confirm that all target values are usable.
    if not np.isfinite(df["Solubility"].to_numpy()).all():
        raise ValueError("Non-finite solubility values found.")

    return df


def configure_mlflow():
    """Configure persistent SQLite and artifact storage."""

    artifacts_dir = TRACKING_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    experiment_name = "aqsoldb-model-benchmark"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        return mlflow.create_experiment(
            experiment_name,
            artifact_location=artifacts_dir.resolve().as_uri()
        )

    return experiment.experiment_id


def main():
    """Train, validate, and track all model-feature combinations."""

    df = prepare_dataset()

    # Calculate the three feature sets only once.
    feature_sets = build_feature_matrices(
        df["canonical_smiles"].tolist()
    )

    targets = df["Solubility"].to_numpy(dtype=np.float64)

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("validation").to_numpy()

    print("Training samples:", train_mask.sum())
    print("Validation samples:", val_mask.sum())
    print("Held-out test samples:", df["split"].eq("test").sum())

    # Identify the exact input dataset for reproducibility.
    dataset_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    experiment_id = configure_mlflow()
    results = []

    for feature_name, X in feature_sets.items():

        X_train = X[train_mask]
        X_val = X[val_mask]

        y_train = targets[train_mask]
        y_val = targets[val_mask]

        for model_name, model in build_models().items():

            run_name = (
                model_name.lower().replace(" ", "-")
                + "-"
                + feature_name
            )

            print(f"\nTraining: {model_name} | {feature_name}")

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name
            ) as run:

                # Record the experimental configuration.
                mlflow.log_params({
                    "algorithm": model_name,
                    "feature_set": feature_name,
                    "n_features": X.shape[1],
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "random_state": RANDOM_STATE,
                    "split_method": "group-aware",
                    "dataset_sha256": dataset_hash,
                })

                # Log estimator hyperparameters.
                for key, value in model.named_steps[
                    "model"
                ].get_params(deep=False).items():

                    if value is not None:
                        mlflow.log_param(
                            f"estimator_{key}",
                            str(value)
                        )

                mlflow.set_tag(
                    "evaluation_split", "validation-only"
                )

                # Fit the entire preprocessing/model pipeline
                # exclusively on the training subset.
                start = perf_counter()
                model.fit(X_train, y_train)
                training_seconds = perf_counter() - start

                predictions = model.predict(X_val)

                metrics = {
                    "val_r2": float(
                        r2_score(y_val, predictions)
                    ),
                    "val_rmse": float(
                        np.sqrt(
                            mean_squared_error(y_val, predictions)
                        )
                    ),
                    "val_mae": float(
                        mean_absolute_error(y_val, predictions)
                    ),
                    "training_seconds": training_seconds,
                }

                mlflow.log_metrics(metrics)

                results.append({
                    "model": model_name,
                    "feature_set": feature_name,
                    "run_id": run.info.run_id,
                    **metrics,
                })

                print(
                    f"R2={metrics['val_r2']:.4f} | "
                    f"RMSE={metrics['val_rmse']:.4f} | "
                    f"MAE={metrics['val_mae']:.4f}"
                )

    # Rank all experiments by validation R2.
    # No test scores are used for model selection.
    leaderboard = pd.DataFrame(results).sort_values(
        "val_r2", ascending=False
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    leaderboard.to_csv(
        RESULTS_DIR / "validation_leaderboard.csv",
        index=False
    )

    print("\n--- Top 5 validation experiments ---")
    print(leaderboard.head(5).to_string(index=False))

    print("\nHeld-out test set was NOT evaluated.")
    print("PASS: Benchmark completed successfully!")


if __name__ == "__main__":
    main()
