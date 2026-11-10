"""
Group-aware cross-validation benchmark for AqSolDB.

Evaluates five regression models across three molecular feature sets
using three-fold GroupKFold cross-validation.

Scientific methodology
----------------------
1. Use only the development dataset (train + validation).
2. Keep the original held-out test dataset excluded.
3. Ensure molecular grouping keys never cross folds.
4. Fit preprocessing and models on each fold's training subset.
5. Evaluate on the corresponding validation subset.
6. Record individual runs using MLflow.
7. Summarize performance using mean and standard deviation.

Model selection will use mean cross-validation R2.
"""

from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from scripts.benchmark_models import prepare_dataset
from src.features import build_feature_matrices
from src.models import build_models


RANDOM_STATE = 42
N_FOLDS = 3
RESULTS_DIR = Path("results")


def configure_mlflow():
    """Create or retrieve the cross-validation experiment."""

    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    name = "aqsoldb-grouped-3fold-cv"

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        return experiment.experiment_id

    artifacts = Path("/app/tracking/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    return mlflow.create_experiment(
        name,
        artifact_location=artifacts.as_uri()
    )


def main():
    """Train, validate, and track 45 cross-validation runs."""

    # Load and group the original molecular dataset.
    df = prepare_dataset()

    # Use development data only.
    # The original test set is reserved for final evaluation.
    dev = df[
        df["split"].isin(["train", "validation"])
    ].reset_index(drop=True)

    groups = dev["group_key"].to_numpy()
    y = dev["Solubility"].to_numpy(dtype=np.float64)

    # Generate all molecular representations once.
    # Feature extraction is deterministic and does not learn
    # statistics from validation targets.
    feature_sets = build_feature_matrices(
        dev["canonical_smiles"].tolist()
    )

    # Reuse the exact same folds for every algorithm.
    cv = GroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(cv.split(dev, y, groups))

    experiment_id = configure_mlflow()
    results = []

    print("Development samples:", len(dev))
    print("Held-out test samples:", len(df) - len(dev))
    print("Cross-validation folds:", N_FOLDS)

    for feature_name, X in feature_sets.items():

        for model_name in build_models():

            print(
                f"\n--- {model_name} | {feature_name} ---",
                flush=True
            )

            for fold_number, (train_idx, val_idx) in enumerate(
                folds, start=1
            ):

                # Verify no grouping key appears on both sides.
                train_groups = set(groups[train_idx])
                val_groups = set(groups[val_idx])

                assert train_groups.isdisjoint(val_groups)

                # Instantiate a fresh, untrained pipeline.
                # This prevents fitted parameters from leaking
                # between cross-validation folds.
                model = build_models()[model_name]

                X_train = X[train_idx]
                X_val = X[val_idx]

                y_train = y[train_idx]
                y_val = y[val_idx]

                run_name = (
                    f"{model_name.lower().replace(' ', '-')}"
                    f"-{feature_name}-fold{fold_number}"
                )

                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=run_name
                ) as run:

                    mlflow.log_params({
                        "algorithm": model_name,
                        "feature_set": feature_name,
                        "fold": fold_number,
                        "n_folds": N_FOLDS,
                        "n_features": X.shape[1],
                        "n_train": len(train_idx),
                        "n_validation": len(val_idx),
                        "random_state": RANDOM_STATE,
                        "cv_method": "GroupKFold",
                    })

                    mlflow.set_tag(
                        "evaluation_split", "cv-validation"
                    )

                    # Fit only on this fold's training samples.
                    start = perf_counter()
                    model.fit(X_train, y_train)
                    duration = perf_counter() - start

                    predictions = model.predict(X_val)

                    metrics = {
                        "cv_r2": float(
                            r2_score(y_val, predictions)
                        ),
                        "cv_rmse": float(
                            np.sqrt(
                                mean_squared_error(
                                    y_val, predictions
                                )
                            )
                        ),
                        "cv_mae": float(
                            mean_absolute_error(
                                y_val, predictions
                            )
                        ),
                        "training_seconds": duration,
                    }

                    mlflow.log_metrics(metrics)

                    results.append({
                        "model": model_name,
                        "feature_set": feature_name,
                        "fold": fold_number,
                        "run_id": run.info.run_id,
                        **metrics,
                    })

                print(
                    f"Fold {fold_number}: "
                    f"R2={metrics['cv_r2']:.4f}, "
                    f"RMSE={metrics['cv_rmse']:.4f}, "
                    f"MAE={metrics['cv_mae']:.4f}",
                    flush=True
                )

    # Save the individual fold results.
    RESULTS_DIR.mkdir(exist_ok=True)

    fold_results = pd.DataFrame(results)

    assert len(fold_results) == 45

    fold_results.to_csv(
        RESULTS_DIR / "cv_fold_scores.csv",
        index=False
    )

    # Calculate mean and sample standard deviation across folds.
    summary = fold_results.groupby(
        ["model", "feature_set"],
        as_index=False
    ).agg(
        mean_r2=("cv_r2", "mean"),
        std_r2=("cv_r2", "std"),
        mean_rmse=("cv_rmse", "mean"),
        std_rmse=("cv_rmse", "std"),
        mean_mae=("cv_mae", "mean"),
        std_mae=("cv_mae", "std"),
    )

    # Rank models by mean cross-validation R2.
    summary = summary.sort_values(
        "mean_r2", ascending=False
    )

    summary.to_csv(
        RESULTS_DIR / "cv_summary.csv",
        index=False
    )

    print("\n--- Cross-validation leaderboard ---")
    print(summary.to_string(index=False))

    # Compare LightGBM and XGBoost on identical folds.
    hybrid = fold_results[
        fold_results["feature_set"] == "hybrid"
    ]

    paired = hybrid.pivot(
        index="fold",
        columns="model",
        values="cv_r2"
    )

    differences = (
        paired["LightGBM"] - paired["XGBoost"]
    )

    print("\n--- LightGBM vs XGBoost: Hybrid ---")
    print("Paired R2 differences by fold:")
    print(differences.to_string())
    print("Mean difference:", round(differences.mean(), 5))

    print("\nHeld-out test set was NOT evaluated.")
    print("PASS: All 45 cross-validation runs completed!")


if __name__ == "__main__":
    main()
