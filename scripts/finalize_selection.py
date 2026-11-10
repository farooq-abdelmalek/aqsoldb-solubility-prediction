"""
Freeze the final AqSolDB model selection before test evaluation.

Compares original CV baselines with completed Optuna studies.
Uses development-set cross-validation scores only.

The held-out test set is never accessed.
"""

import hashlib
import json
import shutil
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState


RESULTS = Path("results")
DATA_PATH = Path("data/curated-solubility-dataset.csv")
OPTUNA_STORAGE = "sqlite:////app/tracking/optuna.db"


def main():
    """Compare four candidates and save the selected configuration."""

    summary = pd.read_csv(RESULTS / "cv_summary.csv")

    assert len(summary) == 15

    dataset_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    candidates = []

    for model_name in ["LightGBM", "XGBoost"]:

        # Original, untuned three-fold CV baseline.
        baseline = summary[
            (summary["model"] == model_name)
            & (summary["feature_set"] == "hybrid")
        ]

        assert len(baseline) == 1

        candidates.append({
            "model": model_name,
            "variant": "baseline",
            "mean_cv_r2": float(baseline.iloc[0]["mean_r2"]),
            "hyperparameters": {},
            "optuna_trial": None,
            "mlflow_run_id": None,
        })

        # Retrieve the persistent, completed Optuna study.
        study = optuna.load_study(
            study_name=f"aqsoldb-{model_name.lower()}-hybrid-v1",
            storage=OPTUNA_STORAGE,
        )

        completed = [
            trial for trial in study.trials
            if trial.state == TrialState.COMPLETE
        ]

        if len(study.trials) != 25 or len(completed) != 25:
            raise ValueError(
                f"{model_name}: expected exactly 25 completed trials."
            )

        if study.user_attrs.get("dataset_sha256") != dataset_hash:
            raise ValueError(
                f"{model_name}: dataset checksum mismatch."
            )

        if study.user_attrs.get("cv_folds") != 3:
            raise ValueError(
                f"{model_name}: unexpected CV configuration."
            )

        best = study.best_trial

        candidates.append({
            "model": model_name,
            "variant": "optuna_tuned",
            "mean_cv_r2": float(best.value),
            "hyperparameters": best.params,
            "optuna_trial": best.number,
            "mlflow_run_id": best.user_attrs.get("mlflow_run_id"),
        })

    # Select using mean development-set CV R2 only.
    candidates.sort(
        key=lambda candidate: candidate["mean_cv_r2"],
        reverse=True,
    )

    winner = candidates[0]

    # Preserve the previous selection before replacing it.
    selection_path = RESULTS / "selected_model.json"
    archive_path = RESULTS / "cv_baseline_selection.json"

    if selection_path.exists() and not archive_path.exists():
        previous = json.loads(
            selection_path.read_text(encoding="utf-8")
        )

        if previous.get("selection_split") == "development_cross_validation":
            shutil.copy2(selection_path, archive_path)

    selection = {
        "model": winner["model"],
        "feature_set": "hybrid",
        "variant": winner["variant"],
        "hyperparameters": winner["hyperparameters"],
        "selection_metric": "mean_cv_r2",
        "cv_mean_r2": winner["mean_cv_r2"],
        "cv_method": "GroupKFold",
        "cv_folds": 3,
        "random_state": 42,
        "selection_split": "development_cv_after_optuna",
        "dataset_sha256": dataset_hash,
        "optuna_trial": winner["optuna_trial"],
        "mlflow_run_id": winner["mlflow_run_id"],
        "test_evaluated": False,
    }

    selection_path.write_text(
        json.dumps(selection, indent=4),
        encoding="utf-8",
    )

    print("--- Final candidate comparison ---")

    for candidate in candidates:
        print(
            f"{candidate['model']:8s} "
            f"{candidate['variant']:12s} "
            f"mean CV R2 = {candidate['mean_cv_r2']:.6f}"
        )

    print("\n--- Frozen model selection ---")
    print(json.dumps(selection, indent=4))

    print("\nHeld-out test set was NOT evaluated.")
    print("PASS: Final model selection frozen!")


if __name__ == "__main__":
    main()
