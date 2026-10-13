"""Logging to console and to <run_dir>/run.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "qacr", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_qacr_console", False) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._qacr_console = True  # type: ignore[attr-defined]
        logger.addHandler(sh)
    if log_file is not None:
        log_file = str(Path(log_file).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
