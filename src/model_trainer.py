"""Stage 3: Model training - compare several regressors with cross-validation, save the best.

Every candidate is logged to MLflow as a nested run under one parent "training" run.
"""
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

from src.config import FEATURES, MODEL_PATH, PREPROCESSOR_PATH, TARGET, TRAIN_DATA_PATH, TRAINING_REPORT_PATH
from src.logger import get_logger
from src.mlflow_utils import save_run_id, setup_mlflow
from src.utils import load_object, load_params, save_json, save_object

logger = get_logger("model_trainer")

MODEL_CLASSES = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_models(params: dict) -> dict:
    models = {}
    for name, model_params in params["models"].items():
        model_params = dict(model_params or {})
        if name in ("random_forest", "gradient_boosting"):
            model_params["random_state"] = params["random_state"]
        models[name] = MODEL_CLASSES[name](**model_params)
    return models


def run():
    all_params = load_params()
    params = all_params["model_trainer"]
    setup_mlflow()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    preprocessor = load_object(PREPROCESSOR_PATH)
    X_train = preprocessor.transform(train_df[FEATURES])
    y_train = train_df[TARGET]

    with mlflow.start_run(run_name="training") as parent_run:
        mlflow.log_params({
            "cv_folds": params["cv_folds"],
            "test_size": all_params["data_transformation"]["test_size"],
            "n_train_rows": len(train_df),
            "n_features": len(FEATURES),
        })

        # Cross-validated R2 for each model, each one in its own nested run
        report = {}
        for name, model in build_models(params).items():
            with mlflow.start_run(run_name=name, nested=True):
                scores = cross_val_score(model, X_train, y_train, cv=params["cv_folds"], scoring="r2")
                mlflow.log_param("model", name)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({"cv_r2_mean": scores.mean(), "cv_r2_std": scores.std()})

            report[name] = {"cv_r2_mean": round(scores.mean(), 4), "cv_r2_std": round(scores.std(), 4)}
            logger.info("%-18s CV R2 = %.4f (+/- %.4f)", name, scores.mean(), scores.std())

        best_name = max(report, key=lambda n: report[n]["cv_r2_mean"])
        logger.info("Best model: %s", best_name)

        # Refit the best model on the full training set
        best_model = build_models(params)[best_name]
        best_model.fit(X_train, y_train)
        save_object(best_model, MODEL_PATH)
        logger.info("Saved model -> %s", MODEL_PATH)

        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_cv_r2", report[best_name]["cv_r2_mean"])
        save_json({"best_model": best_name, "cv_results": report}, TRAINING_REPORT_PATH)
        mlflow.log_artifact(TRAINING_REPORT_PATH)

        save_run_id(parent_run.info.run_id)
        logger.info("MLflow run id: %s", parent_run.info.run_id)


if __name__ == "__main__":
    run()
