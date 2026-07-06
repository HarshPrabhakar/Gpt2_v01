"""
=========================================================
Project : MyGPT2
File    : download.py
Purpose : Download datasets from Hugging Face
=========================================================
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------
# IMPORTANT:
# Set the Hugging Face cache BEFORE importing datasets.
# ---------------------------------------------------------

from config import HF_CACHE_DIR

os.environ["HF_HOME"] = str(HF_CACHE_DIR)

from datasets import load_dataset

from scripts.dataset_registry import DatasetInfo
from scripts.utils import (
    ensure_directory,
    log,
)

# ==========================================================
# Helper Functions
# ==========================================================

def already_downloaded(path: Path) -> bool:
    """
    Returns True if the dataset folder already contains files.
    """

    if not path.exists():
        return False

    return any(path.iterdir())


# ==========================================================
# Download One Dataset
# ==========================================================

def download_dataset(dataset: DatasetInfo) -> bool:
    """
    Downloads one dataset.

    Returns
    -------
    bool
        True if download succeeded.
        False otherwise.
    """

    save_path = dataset.save_path

    ensure_directory(save_path)

    if already_downloaded(save_path):

        log(f"[SKIP] {dataset.name} already exists.")

        return True

    log(f"Downloading {dataset.name}...")

    try:

        if dataset.config is None:

            ds = load_dataset(

                dataset.hf_repo,

                split=dataset.split,

            )

        else:

            ds = load_dataset(

                dataset.hf_repo,

                name=dataset.config,

                split=dataset.split,

            )

        ds.save_to_disk(save_path)

        log(f"[SUCCESS] {dataset.name}")

        return True

    except Exception as e:

        log(f"[FAILED] {dataset.name}")

        log(str(e))

        return False


# ==========================================================
# Download Multiple Datasets
# ==========================================================

def download_all(datasets: list[DatasetInfo]) -> None:
    """
    Downloads every dataset in the registry.
    """

    success = 0

    failed = 0

    skipped = 0

    log("")

    log("===================================")

    log("Starting Dataset Download")

    log("===================================")

    for dataset in datasets:

        if already_downloaded(dataset.save_path):

            skipped += 1

            log(f"[SKIP] {dataset.name}")

            continue

        ok = download_dataset(dataset)

        if ok:

            success += 1

        else:

            failed += 1

    log("")

    log("===================================")

    log("Download Summary")

    log("===================================")

    log(f"Downloaded : {success}")

    log(f"Skipped    : {skipped}")

    log(f"Failed     : {failed}")

    log("")