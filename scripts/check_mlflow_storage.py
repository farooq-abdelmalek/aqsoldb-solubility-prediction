"""
Verify that MLflow experiment data persists outside Docker.

Both the SQLite database and experiment artifacts are stored
inside the project's tracking/ directory, which is mounted
from Windows into the container.
"""

from pathlib import Path
import mlflow


def main():
    """Create a test run and verify persistent storage."""

    tracking_dir = Path("/app/tracking")
    tracking_dir.mkdir(parents=True, exist_ok=True)

    database = tracking_dir / "mlflow.db"
    artifacts = tracking_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)

    # Explicitly configure persistent metadata storage.
    mlflow.set_tracking_uri(
        "sqlite:////app/tracking/mlflow.db"
    )

    experiment_name = "aqsoldb-storage-check"

    # Create an experiment with a persistent artifact location.
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifacts.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    # Record one small test run.
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="persistence-test"
    ) as run:

        mlflow.log_param("purpose", "storage-verification")
        mlflow.log_metric("sample_count", 9980)
        mlflow.log_text(
            "MLflow persistent storage test successful.",
            "verification.txt"
        )

        run_id = run.info.run_id

    # Verify that both storage components exist.
    assert database.is_file()
    assert list(artifacts.rglob("verification.txt"))

    print("Experiment:", experiment_name)
    print("Run ID:", run_id)
    print("SQLite database:", database)
    print("Artifact directory:", artifacts)
    print("PASS: Persistent MLflow storage verified!")


if __name__ == "__main__":
    main()
