"""
=========================================================
Project : MyGPT2
File    : dataset_registry.py
Purpose : Stores all supported datasets.
=========================================================
"""

from dataclasses import dataclass
from pathlib import Path

from config import (
    TINYSTORIES_RAW,
    WIKITEXT_RAW,
    OPENWEBTEXT_RAW,
    FINEWEB_RAW,
    GUTENBERG_RAW,
    TINYSTORIES_SPLIT,
    WIKITEXT_SPLIT,
    OPENWEBTEXT_SPLIT,
    FINEWEB_SPLIT,
)

# =========================================================
# Dataset Definition
# =========================================================

@dataclass(frozen=True)
class DatasetInfo:
    """
    Stores metadata about a dataset.

    Attributes
    ----------
    name:
        Display name shown in the Dataset Manager.

    hf_repo:
        Hugging Face repository ID.

    config:
        Optional Hugging Face configuration.

    split:
        Dataset split to download.

    save_path:
        Local directory where the dataset will be stored.

    description:
        Human-readable description.
    """

    name: str
    hf_repo: str
    config: str | None
    split: str | None
    save_path: Path
    description: str


# =========================================================
# Dataset Registry
# =========================================================

DATASETS = {

    "TinyStories": DatasetInfo(

        name="TinyStories",

        hf_repo="roneneldan/TinyStories",

        config=None,

        split=TINYSTORIES_SPLIT,

        save_path=TINYSTORIES_RAW,

        description="Simple short stories for language model pretraining."

    ),

    "WikiText103": DatasetInfo(

        name="WikiText103",

        hf_repo="Salesforce/wikitext",

        config="wikitext-103-v1",

        split=WIKITEXT_SPLIT,

        save_path=WIKITEXT_RAW,

        description="Wikipedia articles with long coherent documents."

    ),

    "OpenWebText": DatasetInfo(

        name="OpenWebText",

        hf_repo="Skylion007/openwebtext",

        config=None,

        split=OPENWEBTEXT_SPLIT,

        save_path=OPENWEBTEXT_RAW,

        description="Open-source recreation of OpenAI's WebText."

    ),

    "FineWeb": DatasetInfo(

        name="FineWeb",

        hf_repo="HuggingFaceFW/fineweb",

        config="sample-10BT",

        split=FINEWEB_SPLIT,

        save_path=FINEWEB_RAW,

        description="High-quality filtered web corpus."

    ),

    "Gutenberg": DatasetInfo(

        name="Gutenberg",

        hf_repo="",

        config=None,

        split=None,

        save_path=GUTENBERG_RAW,

        description="Classic books downloaded manually from Project Gutenberg."

    ),
}

# =========================================================
# Helper Functions
# =========================================================

def get_dataset(name: str) -> DatasetInfo:
    """
    Returns one dataset from the registry.

    Raises:
        KeyError if the dataset name does not exist.
    """
    return DATASETS[name]


def get_all_datasets() -> list[DatasetInfo]:
    """
    Returns all registered datasets.
    """
    return list(DATASETS.values())


def dataset_exists(name: str) -> bool:
    """
    Checks whether a dataset is registered.
    """
    return name in DATASETS