"""
=========================================================
Project : MyGPT2
File    : models.py
Purpose : Data models used throughout the preprocessing
          pipeline.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ==========================================================
# Storage Information
# ==========================================================

@dataclass(slots=True)
class StorageInfo:
    """
    Information about the dataset stored on disk.
    """

    dataset_format: str
    disk_size_bytes: int
    disk_size_human: str


# ==========================================================
# Schema Information
# ==========================================================

@dataclass(slots=True)
class SchemaInfo:
    """
    Describes the dataset schema.
    """

    columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    text_column: str | None = None


# ==========================================================
# Quality Information
# ==========================================================

@dataclass(slots=True)
class QualityInfo:
    """
    Basic quality statistics for the dataset.
    """

    missing_values: dict[str, int] = field(default_factory=dict)
    empty_documents: int = 0
    duplicate_documents: int | None = None


# ==========================================================
# Text Statistics
# ==========================================================

@dataclass(slots=True)
class TextStatistics:
    """
    Statistics computed from the text column.
    """

    total_characters: int = 0
    total_words: int = 0

    average_characters: float = 0.0
    average_words: float = 0.0

    minimum_characters: int = 0
    maximum_characters: int = 0

    minimum_words: int = 0
    maximum_words: int = 0


# ==========================================================
# Sample Information
# ==========================================================

@dataclass(slots=True)
class SampleInfo:
    """
    Stores a small preview of the dataset.
    """

    preview: str = ""


# ==========================================================
# Dataset Information
# ==========================================================

@dataclass(slots=True)
class DatasetInfo:
    """
    General information about a dataset.
    """

    name: str
    huggingface_repo: str
    split: str

    total_rows: int = 0

    storage: StorageInfo | None = None
    schema: SchemaInfo | None = None
    quality: QualityInfo | None = None
    text_statistics: TextStatistics | None = None
    sample: SampleInfo | None = None


# ==========================================================
# Complete Analysis Result
# ==========================================================

@dataclass(slots=True)
class DatasetAnalysis:
    """
    Complete output produced by Stage 01.

    This object is passed between the analyzer,
    metadata generator, and report generator.
    """

    dataset: DatasetInfo
    success: bool = True
    errors: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """
        Add an error message and mark the analysis as failed.
        """

        self.success = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the analysis into a dictionary suitable for
        JSON serialization.
        """

        from dataclasses import asdict

        return asdict(self)