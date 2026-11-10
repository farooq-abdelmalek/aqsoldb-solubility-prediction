"""
Reconstruct the frozen AqSolDB model for reproducible inference.

Uses the original development split and frozen hyperparameters.

This script NEVER evaluates the held-out test set.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import canonicalize_smiles
from src.splitting import make_group_key, split_by_group
from src.features import build_feature_matrices
from src.models import build_models


def main():
    """Train the selected model and save its inference bundle."""

    parser = argparse.ArgumentParser(
        description="Reconstruct the frozen AqSolDB model."
    )

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    # Load the frozen model-selection configuration.
    selection = json.loads(
        args.selection.read_text(encoding="utf-8")
    )

    if (
        selection["model"] != "XGBoost"
        or selection["feature_set"] != "hybrid"
        or selection["selection_split"]
        != "development_cv_after_optuna"
    ):
        raise ValueError("Unexpected model-selection configuration.")

    # Ensure this is exactly the dataset used during development.
    dataset_hash = hashlib.sha256(
        args.data.read_bytes()
    ).hexdigest()

    if dataset_hash != selection["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 checksum mismatch.")

    # Reuse the original preprocessing and group-aware split.
    df = pd.read_csv(args.data)

    df["canonical_smiles"] = df["SMILES"].apply(
        canonicalize_smiles
    )

    df = df.dropna(subset=["canonical_smiles"]).copy()

    df["group_key"] = df["canonical_smiles"].apply(
        make_group_key
    )

    if df["group_key"].isna().any():
        raise ValueError("Missing molecular grouping keys.")

    df = split_by_group(
        df,
        random_state=selection["random_state"]
    )

    # Train on development molecules only.
    development = df[
        df["split"].isin(["train", "validation"])
    ].copy()

    test = df[df["split"] == "test"]

    assert len(development) == 8967
    assert len(test) == 1013

    assert set(development["group_key"]).isdisjoint(
        set(test["group_key"])
    )

    # Calculate the same hybrid features used for optimization.
    X_train = build_feature_matrices(
        development["canonical_smiles"].tolist()
    )["hybrid"]

    y_train = development["Solubility"].to_numpy(
        dtype=np.float64
    )

    assert X_train.shape == (8967, 2063)
    assert np.isfinite(X_train).all()
    assert np.isfinite(y_train).all()

    # Rebuild the frozen model without any additional tuning.
    model = build_models()[selection["model"]]

    model.set_params(**{
        f"model__{key}": value
        for key, value in selection["hyperparameters"].items()
    })

    print(
        f"Training {selection['model']} on "
        f"{len(y_train)} development molecules...",
        flush=True
    )

    model.fit(X_train, y_train)

    # Save the same bundle format used by scripts.predict.
    bundle = {
        "pipeline": model,
        "model_name": selection["model"],
        "feature_set": selection["feature_set"],
        "n_features": X_train.shape[1],
        "target": "Solubility",
        "selection": selection,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(bundle, args.output)

    print("Saved model:", args.output)
    print("Held-out test predictions: NOT GENERATED")
    print("Held-out test metrics: NOT COMPUTED")
    print("PASS: Frozen model reconstructed successfully!")


if __name__ == "__main__":
    main()
