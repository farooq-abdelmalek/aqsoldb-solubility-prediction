"""
Reproducible hyperparameter optimization for AqSolDB.

Tune LightGBM or XGBoost with Optuna TPE and three-fold
group-aware cross-validation. Every trial is logged to MLflow.

Only development data are used. The held-out test set is
excluded from optimization and model selection.

The Optuna study persists in tracking/optuna.db, allowing
interrupted searches to be resumed.
"""

import argparse
import hashlib
from pathlib import Path

import mlflow
import numpy as np
import optuna

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from scripts.benchmark_models import prepare_dataset
from src.features import build_feature_matrices
from src.models import build_models


RANDOM_STATE = 42
N_FOLDS = 3
DATA_PATH = Path("data/curated-solubility-dataset.csv")
TRACKING_DIR = Path("/app/tracking")


def suggest_parameters(trial, model_name):
    """Suggest model hyperparameters within predefined ranges."""

    # Parameters common to both gradient-boosting algorithms.
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 800, step=100
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.001, 10.0, log=True
        ),
    }

    # Model-specific complexity and regularization controls.
    if model_name == "LightGBM":
        params["num_leaves"] = trial.suggest_int(
            "num_leaves", 15, 63
        )
        params["min_child_samples"] = trial.suggest_int(
            "min_child_samples", 10, 50, step=10
        )
    else:
        params["min_child_weight"] = trial.suggest_int(
            "min_child_weight", 1, 10
        )

    return params


def prepare_development_data():
    """Load development molecules and verify test isolation."""

    df = prepare_dataset()

    dev = df[
        df["split"].isin(["train", "validation"])
    ].reset_index(drop=True)

    test = df[df["split"] == "test"]

    # Original molecular-group separation must be preserved.
    assert set(dev["group_key"]).isdisjoint(
        set(test["group_key"])
    )

    # Deterministic molecular features require no fitting.
    X = build_feature_matrices(
        dev["canonical_smiles"].tolist()
    )["hybrid"]

    y = dev["Solubility"].to_numpy(dtype=np.float64)
    groups = dev["group_key"].to_numpy()

    cv = GroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(cv.split(X, y, groups))

    for train_idx, val_idx in folds:
        assert set(groups[train_idx]).isdisjoint(
            set(groups[val_idx])
        )

    return X, y, folds, len(test)


def configure_mlflow():
    """Configure persistent MLflow experiment tracking."""

    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    experiment_name = "aqsoldb-optuna-tuning"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is not None:
        return experiment.experiment_id

    artifacts = TRACKING_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    return mlflow.create_experiment(
        experiment_name,
        artifact_location=artifacts.as_uri(),
    )


def optimize_model(model_name, total_trials):
    """Optimize CV R2 and persist Optuna and MLflow results."""

    X, y, folds, test_count = prepare_development_data()

    dataset_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    experiment_id = configure_mlflow()

    # A separate study for each model avoids mixing search spaces.
    study = optuna.create_study(
        study_name=f"aqsoldb-{model_name.lower()}-hybrid-v1",
        storage="sqlite:////app/tracking/optuna.db",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        load_if_exists=True,
    )

    # Prevent an existing study from silently using changed data.
    previous_hash = study.user_attrs.get("dataset_sha256")

    if previous_hash is not None and previous_hash != dataset_hash:
        raise ValueError(
            "Dataset changed: do not reuse the existing study."
        )

    study.set_user_attr("dataset_sha256", dataset_hash)
    study.set_user_attr("cv_folds", N_FOLDS)
    study.set_user_attr("random_state", RANDOM_STATE)

    def objective(trial):
        """Fit the proposed parameters on all three CV folds."""

        params = suggest_parameters(trial, model_name)

        r2_scores = []
        rmse_scores = []
        mae_scores = []

        run_name = (
            f"optuna-{model_name.lower()}-trial{trial.number}"
        )

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ) as run:

            mlflow.log_params({
                "algorithm": model_name,
                "feature_set": "hybrid",
                "trial_number": trial.number,
                "cv_folds": N_FOLDS,
                "cv_method": "GroupKFold",
                "random_state": RANDOM_STATE,
                "dataset_sha256": dataset_hash,
            })

            mlflow.log_params({
                f"tuned_{key}": value
                for key, value in params.items()
            })

            mlflow.set_tag(
                "evaluation_split", "development_cv"
            )

            for fold_number, (train_idx, val_idx) in enumerate(
                folds, start=1
            ):
                # A new pipeline is fitted independently per fold.
                # Its imputer learns only from training molecules.
                model = build_models()[model_name]

                model.set_params(**{
                    f"model__{key}": value
                    for key, value in params.items()
                })

                model.fit(
                    X[train_idx], y[train_idx]
                )

                predictions = model.predict(X[val_idx])

                r2 = float(
                    r2_score(y[val_idx], predictions)
                )
                rmse = float(
                    np.sqrt(mean_squared_error(
                        y[val_idx], predictions
                    ))
                )
                mae = float(
                    mean_absolute_error(
                        y[val_idx], predictions
                    )
                )

                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mae_scores.append(mae)

                mlflow.log_metrics({
                    f"fold_{fold_number}_r2": r2,
                    f"fold_{fold_number}_rmse": rmse,
                    f"fold_{fold_number}_mae": mae,
                })

            mean_r2 = float(np.mean(r2_scores))

            mlflow.log_metrics({
                "mean_cv_r2": mean_r2,
                "std_cv_r2": float(
                    np.std(r2_scores, ddof=1)
                ),
                "mean_cv_rmse": float(np.mean(rmse_scores)),
                "mean_cv_mae": float(np.mean(mae_scores)),
            })

            trial.set_user_attr(
                "mlflow_run_id", run.info.run_id
            )

        return mean_r2

    completed_or_attempted = len(study.trials)
    remaining = max(0, total_trials - completed_or_attempted)

    print("Model:", model_name)
    print("Development samples:", len(y))
    print("Held-out test samples:", test_count)
    print("Existing trials:", completed_or_attempted)
    print("New trials requested:", remaining, flush=True)

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
        )

    print("\nBest trial:", study.best_trial.number)
    print("Best mean CV R2:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    print("Held-out test set was NOT evaluated.")
    print("PASS: Optuna optimization completed!")


def main():
    """Read CLI arguments and launch one model's Optuna study."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["LightGBM", "XGBoost"],
    )

    parser.add_argument(
        "--total-trials",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    if args.total_trials < 1:
        parser.error("--total-trials must be positive")

    optimize_model(args.model, args.total_trials)


if __name__ == "__main__":
    main()
