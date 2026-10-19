"""Stage 2: Data transformation - clean, engineer features, split, build the preprocessor."""
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    TARGET,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.logger import get_logger
from src.utils import load_params, save_object

logger = get_logger("data_transformation")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    logger.info("Cleaning removed %d rows (duplicates / missing target)", before - len(df))
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. Also used at prediction time."""
    df = df.copy()
    df["total_acidity"] = df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    df["free_so2_ratio"] = df["free_sulfur_dioxide"] / df["total_sulfur_dioxide"]
    df["alcohol_x_sulphates"] = df["alcohol"] * df["sulphates"]
    return df


def build_preprocessor() -> Pipeline:
    """Fill any missing values with the median, then scale every feature."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def run():
    params = load_params()["data_transformation"]

    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_data(df)
    df = add_features(df)

    # Stratify on quality so rare scores (3 and 8) appear in both train and test
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET],
    )
    os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info("Train %s, Test %s", train_df.shape, test_df.shape)

    # Fit the preprocessor on train only, so no information leaks from the test set
    preprocessor = build_preprocessor()
    preprocessor.fit(train_df[FEATURES])
    save_object(preprocessor, PREPROCESSOR_PATH)
    logger.info("Saved preprocessor -> %s", PREPROCESSOR_PATH)


if __name__ == "__main__":
    run()
