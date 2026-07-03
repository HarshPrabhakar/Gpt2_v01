from datasets import load_dataset
import os


ROOT = "datasets"


def download_dataset(dataset_name, save_path, split=None, name=None):
    """
    Downloads a Hugging Face dataset and saves it to disk.
    """

    print(f"\nDownloading {dataset_name}...")

    if name:
        dataset = load_dataset(
            dataset_name,
            name=name,
            split=split
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split
        )

    dataset.save_to_disk(save_path)

    print(f"Saved to: {save_path}")


def main():

    os.makedirs(ROOT, exist_ok=True)

    # -----------------------------
    # TinyStories
    # -----------------------------
    download_dataset(
        dataset_name="roneneldan/TinyStories",
        save_path="datasets/TinyStories/raw"
    )

    # -----------------------------
    # WikiText-103
    # -----------------------------
    download_dataset(
        dataset_name="Salesforce/wikitext",
        name="wikitext-103-v1",
        save_path="datasets/WikiText103/raw"
    )

    # -----------------------------
    # OpenWebText
    # -----------------------------
    download_dataset(
        dataset_name="Skylion007/openwebtext",
        save_path="datasets/OpenWebText/raw"
    )

    # -----------------------------
    # FineWeb (small subset)
    # -----------------------------
    download_dataset(
        dataset_name="HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train[:5%]",
        save_path="datasets/FineWeb/raw"
    )


if __name__ == "__main__":
    main()