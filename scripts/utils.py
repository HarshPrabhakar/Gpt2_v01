"""
=========================================================
Project : MyGPT2
File    : utils.py
Purpose : Common helper functions
=========================================================
"""

from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime

from config import LOG_DIR


# ==========================================================
# Logging
# ==========================================================

LOG_FILE = LOG_DIR / "dataset_manager.log"


def log(message: str) -> None:
    """
    Prints a message to the console and appends it to the log file.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    formatted = f"[{timestamp}] {message}"

    print(formatted)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(formatted + "\n")


# ==========================================================
# Directory Helpers
# ==========================================================

def ensure_directory(path: Path) -> None:
    """
    Creates a directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)


def is_dataset_downloaded(path: Path) -> bool:
    """
    Determines whether a dataset has already been saved.

    A Hugging Face dataset saved with save_to_disk()
    always contains a dataset_info.json file.
    """

    return (path / "dataset_info.json").exists()


# ==========================================================
# Size Helpers
# ==========================================================

def folder_size(path: Path) -> int:
    """
    Returns folder size in bytes.
    """

    total = 0

    if not path.exists():
        return total

    for file in path.rglob("*"):

        if file.is_file():

            total += file.stat().st_size

    return total


def human_readable_size(size: int) -> str:
    """
    Converts bytes into KB / MB / GB.
    """

    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:

        if value < 1024:

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


# ==========================================================
# Disk Information
# ==========================================================

def free_disk_space(path: Path) -> int:
    """
    Returns free disk space in bytes.
    """

    usage = shutil.disk_usage(path)

    return usage.free


def total_disk_space(path: Path) -> int:
    """
    Returns total disk size in bytes.
    """

    usage = shutil.disk_usage(path)

    return usage.total


# ==========================================================
# Pretty Printing
# ==========================================================

def divider(length: int = 60) -> None:
    """
    Prints a horizontal divider.
    """

    print("=" * length)


def title(text: str) -> None:
    """
    Prints a formatted section title.
    """

    divider()

    print(text.center(60))

    divider()