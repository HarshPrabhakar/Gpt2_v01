"""
=========================================================
Project : MyGPT2
File    : dataset_analyzer.py
Purpose : Analyze Hugging Face datasets and return a
          DatasetAnalysis object.
=========================================================
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from preprocessing.models import (
    DatasetAnalysis,
    DatasetInfo,
    StorageInfo,
    SchemaInfo,
    QualityInfo,
    TextStatistics,
    SampleInfo,
)


class DatasetAnalyzer:
    """
    Performs Stage 01 dataset analysis.

    This class is responsible only for analyzing datasets.
    It does not generate reports or metadata files.
    """

    def __init__(
        self,
        dataset_path: Path,
        dataset_name: str,
        huggingface_repo: str,
        split: str = "train",
        sample_size: int = 10_000,
    ) -> None:

        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.huggingface_repo = huggingface_repo
        self.split = split
        self.sample_size = sample_size

        self.dataset: Dataset | None = None

    # =====================================================
    # Public API
    # =====================================================

    def analyze(self) -> DatasetAnalysis:
        """
        Analyze the dataset and return a DatasetAnalysis object.
        """

        self._load_dataset()

        raise NotImplementedError("Next step: implement analysis pipeline.")

    # =====================================================
    # Private Methods
    # =====================================================

    def _load_dataset(self) -> None:
        """
        Load the dataset from disk.
        """

        loaded = load_from_disk(str(self.dataset_path))

        if isinstance(loaded, DatasetDict):

            if self.split not in loaded:

                raise ValueError(
                    f"Split '{self.split}' not found."
                )

            self.dataset = loaded[self.split]

        else:

            self.dataset = loaded