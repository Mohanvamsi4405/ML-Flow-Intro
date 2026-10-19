"""Stage 1: Data ingestion - download the red wine quality dataset."""
import os

import pandas as pd

from src.config import RAW_DATA_PATH
from src.logger import get_logger
from src.utils import load_params

logger = get_logger("data_ingestion")


def download_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep=";")
    # "fixed acidity" -> "fixed_acidity", "pH" -> "ph"
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def run():
    params = load_params()["data_ingestion"]
    logger.info("Downloading data from %s", params["source_url"])

    df = download_data(params["source_url"])

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Saved raw data %s -> %s", df.shape, RAW_DATA_PATH)
    logger.info("Quality distribution: %s", df["quality"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    run()
