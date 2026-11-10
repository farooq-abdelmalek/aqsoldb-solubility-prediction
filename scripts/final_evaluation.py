"""
Final model training and independent held-out test evaluation.

The model and hyperparameters must already have been selected
using development-set cross-validation.

This script:
1. Verifies the frozen model configuration and dataset checksum.
2. Trains on the combined training and validation samples.
3. Evaluates the held-out test samples once.
4. Saves the fitted model, predictions, and evaluation metrics.
5. Logs the final experiment to MLflow.

IMPORTANT:
Never use the test results to change model selection or tuning.
"""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scripts.benchmark_models import prepare_dataset
from src.features import build_feature_matrices
from src.models import build_models


DATA_PATH = Path("data/curated-solubility-dataset.csv")
SELECTION_PATH = Path("results/selected_model.json")
METRICS_PATH = Path("results/final_test_metrics.json")
PREDICTIONS_PATH = Path("results/final_test_predictions.csv")
MODEL_PATH = Path("artifacts/final_model.joblib")


def load_selection():
    """Verify the frozen configuration and source dataset."""

    if METRICS_PATH.exists():
        raise FileExistsError(
            "Final test metrics already exist. "
            "Do not repeat the held-out evaluation."
        )

    selection = json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )

    if (
        selection["model"] != "XGBoost"
        or selection["feature_set"] != "hybrid"
        or selection["selection_split"]
        != "development_cv_after_optuna"
    ):
        raise ValueError("Unexpected final model selection.")

    current_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    if current_hash != selection["dataset_sha256"]:
        raise ValueError(
            "Dataset checksum mismatch. "
            "The dataset has changed since model selection."
        )

    return selection


def prepare_final_data():
    """Generate features and preserve the independent test split."""

    df = prepare_dataset()

    development_mask = df["split"].isin(
        ["train", "validation"]
    ).to_numpy()

    test_mask = df["split"].eq("test").to_numpy()

    assert development_mask.sum() == 8967
    assert test_mask.sum() == 1013
    assert (development_mask | test_mask).all()

    # Verify group isolation before final fitting.
    development_groups = set(
        df.loc[development_mask, "group_key"]
    )
    test_groups = set(
        df.loc[test_mask, "group_key"]
    )

    assert development_groups.isdisjoint(test_groups)

    # The RDKit feature calculations are deterministic.
    # No model preprocessing is fitted on test samples.
    X = build_feature_matrices(
        df["canonical_smiles"].tolist()
    )["hybrid"]

    y = df["Solubility"].to_numpy(dtype=np.float64)

    assert X.shape[1] == 2063
    assert np.isfinite(y).all()

    return (
        X[development_mask],
        y[development_mask],
        X[test_mask],
        y[test_mask],
        df.loc[test_mask, "canonical_smiles"].to_numpy(),
    )


def configure_mlflow():
    """Use the persistent MLflow database and artifact storage."""

    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    name = "aqsoldb-final-model"

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        return experiment.experiment_id

    artifact_dir = Path("/app/tracking/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    return mlflow.create_experiment(
        name,
        artifact_location=artifact_dir.as_uri(),
    )


def main():
    """Train the selected model and record its test performance."""

    selection = load_selection()

    X_train, y_train, X_test, y_test, smiles_test = (
        prepare_final_data()
    )

    # Rebuild the same pipeline used during optimization.
    # The imputer is fitted only on development samples.
    model = build_models()[selection["model"]]

    model.set_params(**{
        f"model__{key}": value
        for key, value in selection["hyperparameters"].items()
    })

    experiment_id = configure_mlflow()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="final-xgboost-hybrid",
    ) as run:

        mlflow.log_params({
            "algorithm": selection["model"],
            "feature_set": selection["feature_set"],
            "n_features": X_train.shape[1],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "random_state": selection["random_state"],
            "selection_method": selection["selection_split"],
            "dataset_sha256": selection["dataset_sha256"],
            "optuna_trial": selection["optuna_trial"],
        })

        mlflow.log_params({
            f"tuned_{key}": value
            for key, value in selection["hyperparameters"].items()
        })

        print(
            "Training final XGBoost on",
            len(y_train),
            "development molecules...",
            flush=True,
        )

        start = perf_counter()
        model.fit(X_train, y_train)
        training_seconds = perf_counter() - start

        # Independent test evaluation.
        predictions = model.predict(X_test)

        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite test predictions.")

        metrics = {
            "test_r2": float(
                r2_score(y_test, predictions)
            ),
            "test_rmse": float(
                np.sqrt(mean_squared_error(
                    y_test, predictions
                ))
            ),
            "test_mae": float(
                mean_absolute_error(
                    y_test, predictions
                )
            ),
            "training_seconds": training_seconds,
        }

        mlflow.log_metrics(metrics)

        # Save the fitted pipeline together with its metadata.
        # The inference script will reconstruct molecular features.
        model_bundle = {
            "pipeline": model,
            "model_name": selection["model"],
            "feature_set": selection["feature_set"],
            "n_features": X_train.shape[1],
            "target": "Solubility",
            "selection": selection,
        }

        joblib.dump(model_bundle, MODEL_PATH)

        # Save predictions for later error analysis.
        predictions_df = pd.DataFrame({
            "canonical_smiles": smiles_test,
            "observed_logs": y_test,
            "predicted_logs": predictions,
            "residual": y_test - predictions,
        })

        predictions_df.to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

        report = {
            "model": selection["model"],
            "feature_set": selection["feature_set"],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "selection_cv_mean_r2": selection["cv_mean_r2"],
            "mlflow_run_id": run.info.run_id,
            "dataset_sha256": selection["dataset_sha256"],
            **metrics,
        }

        # Persist model and predictions in MLflow.
        mlflow.log_artifact(
            str(MODEL_PATH), artifact_path="model"
        )
        mlflow.log_artifact(
            str(SELECTION_PATH), artifact_path="selection"
        )
        mlflow.log_artifact(
            str(PREDICTIONS_PATH), artifact_path="evaluation"
        )

        # Write final metrics last. Their existence prevents
        # accidental repeated held-out evaluation.
        METRICS_PATH.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(METRICS_PATH), artifact_path="evaluation"
        )

    print("\n--- FINAL HELD-OUT TEST RESULTS ---")
    print(json.dumps(report, indent=4))
    print("\nSaved model:", MODEL_PATH)
    print("Saved predictions:", PREDICTIONS_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("\nPASS: Final evaluation completed!")


if __name__ == "__main__":
    main()
