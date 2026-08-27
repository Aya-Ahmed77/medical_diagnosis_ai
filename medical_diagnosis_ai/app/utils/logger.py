"""Centralized logging configuration used across the whole application."""
import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    Log level is controlled via the LOG_LEVEL environment variable
    (defaults to INFO). Logs go to stdout so they are visible whether
    the app runs locally, in a container, or under a test runner.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated imports

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
