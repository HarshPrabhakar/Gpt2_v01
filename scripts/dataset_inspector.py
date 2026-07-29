"""
=========================================================
Project : MyGPT2
File    : dataset_inspector.py
Purpose : Inspect downloaded Hugging Face datasets.
=========================================================
"""

from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk


DATASETS = {
    "TinyStories": Path("datasets/TinyStories/raw"),
    "WikiText103": Path("datasets/WikiText103/raw"),
    "OpenWebText": Path("datasets/OpenWebText/raw"),
    "FineWeb": Path("datasets/FineWeb/raw"),
}


def inspect_dataset(name: str, path: Path):

    print("=" * 70)
    print(name)
    print("=" * 70)

    dataset = load_from_disk(str(path))

    if isinstance(dataset, DatasetDict):

        print(f"Splits : {list(dataset.keys())}")

        split_name = "train"

        if split_name not in dataset:

            split_name = list(dataset.keys())[0]

        dataset = dataset[split_name]

    assert isinstance(dataset, Dataset)

    print(f"Rows : {len(dataset):,}")

    print("\nColumns:")

    for column in dataset.column_names:
        print(f"  • {column}")

    print("\nSample Record:\n")

    sample = dataset[0]

    for key, value in sample.items():

        value = str(value)

        if len(value) > 300:
            value = value[:300] + "..."

        print(f"{key}:")
        print(value)
        print()


def main():

    for name, path in DATASETS.items():

        inspect_dataset(name, path)


if __name__ == "__main__":
    main()