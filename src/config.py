"""All file paths and column names in one place."""
import os

# Data
RAW_DATA_PATH = os.path.join("data", "raw", "winequality_red.csv")
TRAIN_DATA_PATH = os.path.join("data", "processed", "train.csv")
TEST_DATA_PATH = os.path.join("data", "processed", "test.csv")

# Artifacts
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")
MODEL_PATH = os.path.join("models", "model.pkl")

# Reports
TRAINING_REPORT_PATH = os.path.join("reports", "training_report.json")
METRICS_PATH = os.path.join("reports", "metrics.json")
MLFLOW_RUN_PATH = os.path.join("reports", "mlflow_run.json")   # links training and evaluation runs

# Columns
TARGET = "quality"
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
    # engineered
    "total_acidity",
    "free_so2_ratio",
    "alcohol_x_sulphates",
]
