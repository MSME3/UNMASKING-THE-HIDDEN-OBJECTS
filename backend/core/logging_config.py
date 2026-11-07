"""
Structured Logging Configuration for AI Vision Search Platform.
Provides colorized and timestamped console logging suitable for Edge AI execution.
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured standard logger with consistent formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

