"""
Run a pilot AqSolDB regression experiment with MLflow.

This script verifies the complete training workflow:
1. Load and canonicalize molecular structures.
2. Create group-aware train/validation/test splits.
3. Calculate RDKit molecular descriptors.
4. Train a Ridge regression pipeline.
5. Evaluate on validation data only.
6. Record the experiment in persistent MLflow storage.

The held-out test set is never used during this pilot.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import mlflow

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from src.preprocessing import canonicalize_smiles
from src.splitting import make_group_key, split_by_group
from src.features import calculate_descriptors
from src.models import build_models


def main():
    """Train and track one Ridge regression pilot experiment."""

    # --------------------------------------------------
    # 1. Load the original dataset
    # --------------------------------------------------
    data_path = Path("data/curated-solubility-dataset.csv")
    df = pd.read_csv(data_path)

    # Preserve complete molecular structures.
    df["canonical_smiles"] = df["SMILES"].apply(
        canonicalize_smiles
    )

    df = df.dropna(subset=["canonical_smiles"]).copy()

    # Grouping keys are used for splitting, not features.
    df["group_key"] = df["canonical_smiles"].apply(
        make_group_key
    )

    # --------------------------------------------------
    # 2. Create reproducible dataset splits
    # --------------------------------------------------
    df = split_by_group(df, random_state=42)

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]

    # The test set remains untouched.
    print("Training molecules:", len(train))
    print("Validation molecules:", len(validation))

    # --------------------------------------------------
    # 3. Generate molecular descriptors
    # --------------------------------------------------
    X_train = calculate_descriptors(
        train["canonical_smiles"]
    ).to_numpy(dtype=np.float32)

    X_val = calculate_descriptors(
        validation["canonical_smiles"]
    ).to_numpy(dtype=np.float32)

    y_train = train["Solubility"].to_numpy()
    y_val = validation["Solubility"].to_numpy()

    # --------------------------------------------------
    # 4. Train Ridge regression
    # --------------------------------------------------
    # The pipeline fits its imputer and scaler
    # exclusively on the training dataset.
    model = build_models()["Ridge Regression"]

    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    # --------------------------------------------------
    # 5. Evaluate on validation data only
    # --------------------------------------------------
    predictions = model.predict(X_val)

    metrics = {
        "val_r2": float(r2_score(y_val, predictions)),
        "val_rmse": float(
            np.sqrt(mean_squared_error(y_val, predictions))
        ),
        "val_mae": float(
            mean_absolute_error(y_val, predictions)
        ),
        "training_seconds": training_seconds,
    }

    # --------------------------------------------------
    # 6. Configure persistent MLflow tracking
    # --------------------------------------------------
    tracking_dir = Path("/app/tracking")
    artifacts_dir = tracking_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    # Keep pilot runs separate from final experiments.
    experiment_name = "aqsoldb-pilot"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifacts_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    # --------------------------------------------------
    # 7. Log model configuration and results
    # --------------------------------------------------
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="ridge-descriptors-pilot"
    ) as run:

        mlflow.log_params({
            "model": "Ridge Regression",
            "features": "RDKit descriptors",
            "n_features": X_train.shape[1],
            "n_train": len(train),
            "n_validation": len(validation),
            "random_state": 42,
            "split_strategy": "group-aware",
        })

        mlflow.log_metrics(metrics)

        print("\nMLflow run ID:", run.info.run_id)

    # --------------------------------------------------
    # 8. Display validation results
    # --------------------------------------------------
    print("\n--- Validation Results ---")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nPASS: MLflow pilot experiment completed!")


if __name__ == "__main__":
    main()
