"""Structured logging utilities."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    project_name: str = "wound-segmentation",
) -> logging.Logger:
    """Configure logging with both console and file handlers.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file. If provided, creates file handler.
        project_name: Project identifier for log entries.

    Returns:
        Root logger configured for the project.
    """
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(fmt="%(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a structured logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class AuditLogger:
    """FDA-compliant audit trail logger for clinical data processing."""

    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger("audit")
        self.log_file = log_file

    def log_action(
        self,
        action: str,
        file_path: Optional[Path] = None,
        result: str = "SUCCESS",
        details: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now().isoformat()
        parts = [f"[{timestamp}]", f"ACTION={action}", f"RESULT={result}"]
        if file_path:
            parts.append(f"PATH={file_path}")
        if details:
            parts.append(f"DETAILS={details}")
        self.logger.info(" | ".join(parts))

    def log_validation(
        self,
        check_name: str,
        passed: bool,
        value: Optional[str] = None,
        threshold: Optional[str] = None,
    ) -> None:
        status = "PASS" if passed else "FAIL"
        parts = [f"CHECK={check_name}", f"STATUS={status}"]
        if value is not None:
            parts.append(f"VALUE={value}")
        if threshold is not None:
            parts.append(f"THRESHOLD={threshold}")
        self.logger.info(" | ".join(parts))
