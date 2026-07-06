"""
=========================================================
Project : MyGPT2
File    : menu.py
Purpose : Dataset Manager User Interface
=========================================================
"""

from scripts.dataset_registry import get_all_datasets
from scripts.utils import (
    divider,
    title,
    folder_size,
    human_readable_size,
    free_disk_space,
)

from config import DATASETS_DIR


# ==========================================================
# Main Menu
# ==========================================================

def show_main_menu() -> str:
    """
    Displays the main menu and returns the user's choice.
    """

    title("GPT2 DATASET MANAGER")

    datasets = get_all_datasets()

    print("Datasets\n")

    for index, dataset in enumerate(datasets, start=1):

        print(f"[{index}] Download {dataset.name}")

    print()

    print("[A] Download ALL datasets")

    print("[S] Dataset Status")

    print("[R] Storage Report")

    print("[Q] Quit")

    divider()

    return input("Select an option: ").strip().upper()


# ==========================================================
# Dataset Status
# ==========================================================

def show_dataset_status() -> None:

    title("DATASET STATUS")

    datasets = get_all_datasets()

    for dataset in datasets:

        size = folder_size(dataset.save_path)

        if size > 0:

            status = "Downloaded"

            size_text = human_readable_size(size)

        else:

            status = "Missing"

            size_text = "-"

        print(f"{dataset.name}")

        print(f"Status : {status}")

        print(f"Size   : {size_text}")

        divider()


# ==========================================================
# Storage Report
# ==========================================================

def show_storage_report() -> None:

    title("STORAGE REPORT")

    datasets = get_all_datasets()

    total = 0

    for dataset in datasets:

        size = folder_size(dataset.save_path)

        total += size

        print(f"{dataset.name:<20}{human_readable_size(size)}")

    divider()

    print(f"{'Total Dataset Size':<20}{human_readable_size(total)}")

    print()

    free = free_disk_space(DATASETS_DIR)

    print(f"{'Free Disk Space':<20}{human_readable_size(free)}")

    divider()


# ==========================================================
# Pause
# ==========================================================

def wait_for_user() -> None:

    input("\nPress ENTER to continue...")