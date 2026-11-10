"""
Select the final model using group-aware cross-validation results.

Selection criterion: highest mean validation R2 across three folds.

The previous single-split selection is archived for traceability.
The held-out test set is never accessed by this script.
"""

import json
import shutil
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def main():
    """Validate CV results and save the selected configuration."""

    summary = pd.read_csv(RESULTS_DIR / "cv_summary.csv")
    folds = pd.read_csv(RESULTS_DIR / "cv_fold_scores.csv")

    # Require the complete 15-combination, three-fold benchmark.
    assert len(summary) == 15
    assert len(folds) == 45

    fold_counts = folds.groupby(
        ["model", "feature_set"]
    )["fold"].nunique()

    assert (fold_counts == 3).all()

    # Select using mean CV R2, never test performance.
    winner = summary.loc[summary["mean_r2"].idxmax()]

    winning_folds = folds[
        (folds["model"] == winner["model"]) &
        (folds["feature_set"] == winner["feature_set"])
    ].sort_values("fold")

    selection_path = RESULTS_DIR / "selected_model.json"

    # Keep the original single-split decision for auditability.
    archive_path = RESULTS_DIR / "single_split_selection.json"

    if selection_path.exists() and not archive_path.exists():
        previous = json.loads(
            selection_path.read_text(encoding="utf-8")
        )

        if previous.get("selection_split") == "validation":
            shutil.copy2(selection_path, archive_path)

    selection = {
        "model": str(winner["model"]),
        "feature_set": str(winner["feature_set"]),
        "selection_metric": "mean_r2",
        "cv_mean_r2": float(winner["mean_r2"]),
        "cv_std_r2": float(winner["std_r2"]),
        "cv_mean_rmse": float(winner["mean_rmse"]),
        "cv_mean_mae": float(winner["mean_mae"]),
        "cv_method": "GroupKFold",
        "cv_folds": 3,
        "random_state": 42,
        "selection_split": "development_cross_validation",
        "mlflow_fold_run_ids": winning_folds["run_id"].tolist(),
        "test_evaluated": False
    }

    selection_path.write_text(
        json.dumps(selection, indent=4),
        encoding="utf-8"
    )

    print(json.dumps(selection, indent=4))
    print("\nPASS: CV-based model selection frozen!")


if __name__ == "__main__":
    main()
