"""Stage 4: Model evaluation - score the saved model on the held-out test set.

Resumes the MLflow training run, logs the test metrics, and logs + registers the model.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import FEATURES, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH, TARGET, TEST_DATA_PATH
from src.logger import get_logger
from src.mlflow_utils import load_run_id, setup_mlflow
from src.utils import load_object, save_json

logger = get_logger("model_evaluation")


def evaluate(y_true, y_pred) -> dict:
    rounded = np.clip(np.round(y_pred), 3, 8)
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        # Quality is a whole number, so also check how often the rounded prediction is exactly right
        "rounded_accuracy": round(float(np.mean(rounded == y_true)), 4),
        "within_1_point": round(float(np.mean(np.abs(rounded - y_true) <= 1)), 4),
    }


def run():
    mlflow_params = setup_mlflow()

    test_df = pd.read_csv(TEST_DATA_PATH)
    preprocessor = load_object(PREPROCESSOR_PATH)
    model = load_object(MODEL_PATH)

    X_test = preprocessor.transform(test_df[FEATURES])
    y_test = test_df[TARGET].to_numpy()
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    save_json(metrics, METRICS_PATH)
    logger.info("Test metrics: %s", metrics)

    with mlflow.start_run(run_id=load_run_id()):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(METRICS_PATH)

        # Log preprocessor + model together so the MLflow model takes raw feature columns
        full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        input_example = test_df[FEATURES].head(5)
        mlflow.sklearn.log_model(
            full_pipeline,
            name="model",
            signature=infer_signature(input_example, full_pipeline.predict(input_example)),
            input_example=input_example,
            registered_model_name=mlflow_params["registered_model_name"],
            # MLflow saves sklearn models with skops, which only loads types you approve.
            # Tree models (RandomForest / GradientBoosting) need these two; we trained the model ourselves.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        logger.info("Logged and registered model '%s'", mlflow_params["registered_model_name"])


if __name__ == "__main__":
    run()
