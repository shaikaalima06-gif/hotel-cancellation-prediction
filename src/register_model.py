"""Register the best tracked model in the local MLflow Model Registry."""
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"
MODEL_NAME = "HotelCancellationModel"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")

    experiment = mlflow.set_experiment("Hotel_Cancellation_Prediction")

    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found. Run Experiment 4 first.")

    best = runs[0]
    run_id = best.info.run_id

    print(f"Selected run for registration: {run_id}")
    print(f"F1: {best.data.metrics.get('f1')}")

    # Register an MLflow model logged by run_lab4_tracking.py.
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"Registered {MODEL_NAME} version {result.version}")
    print("Use the MLflow UI to inspect/promote/compare model versions.")


if __name__ == "__main__":
    main()
