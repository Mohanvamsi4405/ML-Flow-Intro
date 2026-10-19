import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console and to logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
    file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file)
    return logger
