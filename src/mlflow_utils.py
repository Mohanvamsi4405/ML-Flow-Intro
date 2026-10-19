"""MLflow helpers shared by the training and evaluation stages."""
import json
import os

import mlflow
from dotenv import load_dotenv

from src.config import MLFLOW_RUN_PATH
from src.utils import load_params, save_json


def setup_mlflow() -> dict:
    """Point MLflow at the tracking server and experiment.

    Credentials come from environment variables (or a .env file), never from code:
      MLFLOW_TRACKING_USERNAME  - DagsHub username
      MLFLOW_TRACKING_PASSWORD  - DagsHub access token
      MLFLOW_TRACKING_URI       - optional, overrides params.yaml
    """
    load_dotenv()
    params = load_params()["mlflow"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or params["tracking_uri"]
    if "<username>" in tracking_uri:
        raise ValueError(
            "Set your DagsHub tracking URI in params.yaml (mlflow.tracking_uri) "
            "or as MLFLOW_TRACKING_URI in .env"
        )
    if "dagshub.com" in tracking_uri and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise ValueError("Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD (DagsHub token) in .env")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment_name"])
    return params


def save_run_id(run_id: str) -> None:
    """Training and evaluation are separate processes, so the run id is passed through a file."""
    save_json({"run_id": run_id}, MLFLOW_RUN_PATH)


def load_run_id() -> str:
    with open(MLFLOW_RUN_PATH) as f:
        return json.load(f)["run_id"]
