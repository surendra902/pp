from __future__ import annotations

import logging
import sys

try:
    import colorlog  # type: ignore
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _HAS_COLOR:
        h = colorlog.StreamHandler(sys.stdout)
        h.setFormatter(
            colorlog.ColoredFormatter(
                "%(asctime)s.%(msecs)03d %(log_color)s[%(name)-14s]%(reset)s "
                "%(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d [%(name)-14s] %(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(h)
    logger.propagate = False
    return logger
