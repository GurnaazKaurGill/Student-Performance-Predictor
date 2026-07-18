import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"run_{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger