import os
from pathlib import Path

# ----------------------------------------
# Set Hugging Face Cache BEFORE imports
# ----------------------------------------

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

from datasets import load_dataset

from config import *

# ----------------------------------------
# Dataset Registry
# ----------------------------------------

DATASETS = [

    {
        "name": "TinyStories",
        "hf_id": "roneneldan/TinyStories",
        "config": None,
        "split": None,
        "save_path": TINYSTORIES_PATH
    },

    {
        "name": "WikiText103",
        "hf_id": "Salesforce/wikitext",
        "config": "wikitext-103-v1",
        "split": None,
        "save_path": WIKITEXT_PATH
    },

    {
        "name": "OpenWebText",
        "hf_id": "Skylion007/openwebtext",
        "config": None,
        "split": "train[:10%]",
        "save_path": OPENWEBTEXT_PATH
    },

    {
        "name": "FineWeb",
        "hf_id": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "split": "train[:2%]",
        "save_path": FINEWEB_PATH
    }

]

# ----------------------------------------
# Download Function
# ----------------------------------------

def download_dataset(dataset):

    save_path = Path(dataset["save_path"])

    if save_path.exists():

        print(f"✓ {dataset['name']} already downloaded.")

        return

    print(f"\nDownloading {dataset['name']}...")

    ds = load_dataset(

        dataset["hf_id"],

        name=dataset["config"],

        split=dataset["split"]

    )

    ds.save_to_disk(save_path)

    print(f"Saved to {save_path}")


# ----------------------------------------
# Main
# ----------------------------------------

def main():

    for dataset in DATASETS:

        download_dataset(dataset)


if __name__ == "__main__":

    main()