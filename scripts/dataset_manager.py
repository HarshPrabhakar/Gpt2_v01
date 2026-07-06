"""
=========================================================
Project : MyGPT2
File    : dataset_manager.py
Purpose : Main controller for Dataset Management System
Author  : Harsh Prabhakar
=========================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow executing this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_registry import (
    get_all_datasets,
)

from scripts.download import (
    download_dataset,
    download_all,
)

from scripts.menu import (
    show_main_menu,
    show_dataset_status,
    show_storage_report,
    wait_for_user,
)

from scripts.utils import (
    log,
)

# ==========================================================
# Main Loop
# ==========================================================

def main():

    datasets = get_all_datasets()

    while True:

        choice = show_main_menu()

        # ---------------------------------------------
        # Quit
        # ---------------------------------------------

        if choice == "Q":

            log("Closing Dataset Manager.")

            print("\nGoodbye!\n")

            break

        # ---------------------------------------------
        # Download All
        # ---------------------------------------------

        elif choice == "A":

            download_all(datasets)

            wait_for_user()

        # ---------------------------------------------
        # Dataset Status
        # ---------------------------------------------

        elif choice == "S":

            show_dataset_status()

            wait_for_user()

        # ---------------------------------------------
        # Storage Report
        # ---------------------------------------------

        elif choice == "R":

            show_storage_report()

            wait_for_user()

        # ---------------------------------------------
        # Download Individual Dataset
        # ---------------------------------------------

        elif choice.isdigit():

            index = int(choice) - 1

            if 0 <= index < len(datasets):

                dataset = datasets[index]

                download_dataset(dataset)

            else:

                print("\nInvalid dataset number.\n")

            wait_for_user()

        # ---------------------------------------------
        # Invalid Input
        # ---------------------------------------------

        else:

            print("\nInvalid option.\n")

            wait_for_user()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()