import logging
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "anomaly_detection",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name
        log_file: Path to log file (if None, logs to console only)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Add file handler if specified
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger