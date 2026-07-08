"""
=========================================================
Project : MyGPT2
File    : file_utils.py
Purpose : Common file and directory helper functions.
=========================================================
"""

from __future__ import annotations

import json
from pathlib import Path


# ==========================================================
# Directory Helpers
# ==========================================================

def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """

    path.mkdir(parents=True, exist_ok=True)


# ==========================================================
# JSON Helpers
# ==========================================================

def save_json(data: dict, output_file: Path) -> None:
    """
    Save a dictionary as a JSON file.
    """

    ensure_directory(output_file.parent)

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# Markdown Helpers
# ==========================================================

def save_markdown(content: str, output_file: Path) -> None:
    """
    Save markdown text to disk.
    """

    ensure_directory(output_file.parent)

    output_file.write_text(
        content,
        encoding="utf-8"
    )


# ==========================================================
# Folder Size
# ==========================================================

def calculate_folder_size(folder: Path) -> int:
    """
    Calculate folder size in bytes.
    """

    total_size = 0

    for file in folder.rglob("*"):

        if file.is_file():

            total_size += file.stat().st_size

    return total_size


# ==========================================================
# Human Readable Size
# ==========================================================

def human_readable_size(size: int) -> str:
    """
    Convert bytes into KB, MB, GB...
    """

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    value = float(size)

    for unit in units:

        if value < 1024:

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"