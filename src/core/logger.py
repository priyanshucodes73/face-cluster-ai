import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(name: str = "FaceClusterAI", level: str = "INFO"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_dir / "project.log",
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True
    )

    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)

    return logger


logger = setup_logger()