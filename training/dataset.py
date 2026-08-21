"""
============================================================
MyGPT2 - Training Dataset
============================================================

Purpose
-------
Converts the project's text datasets into training samples
for the MyGPT2 language model.

Pipeline:

    Raw Dataset
        |
        v
    Training Documents ONLY
        |
        v
    MyGPTTokenizer
        |
        v
    Token IDs
        |
        v
    Fixed-Length Sequences
        |
        v
    Input IDs + Target IDs
        |
        v
    PyTorch IterableDataset
        |
        v
    DataLoader
        |
        v
    GPU
        |
        v
    MyGPT2

IMPORTANT
---------
This dataset is intended ONLY for model training.

Evaluation splits such as:

    - validation
    - test

are NEVER used for training.

For autoregressive language modeling:

    Input:
        [t1, t2, t3, t4]

    Target:
        [t2, t3, t4, t5]

The model learns to predict the next token.

============================================================
"""

from __future__ import annotations


# ============================================================
# Standard Library
# ============================================================

import sys

from pathlib import Path

from typing import (
    Iterator,
    Optional,
    Sequence,
)


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Third-Party
# ============================================================

import torch

from torch.utils.data import IterableDataset


# ============================================================
# Project Imports
# ============================================================

from tokenizer.my_tokenizer import MyGPTTokenizer


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)


# ============================================================
# Dataset Paths
# ============================================================

DEFAULT_DATASETS = {

    "TinyStories": (
        PROJECT_ROOT
        / "datasets"
        / "TinyStories"
        / "raw"
    ),

    "WikiText103": (
        PROJECT_ROOT
        / "datasets"
        / "WikiText103"
        / "raw"
    ),

    "OpenWebText": (
        PROJECT_ROOT
        / "datasets"
        / "OpenWebText"
        / "raw"
    ),

    "FineWeb": (
        PROJECT_ROOT
        / "datasets"
        / "FineWeb"
        / "raw"
    ),
}


# ============================================================
# Training Split Configuration
# ============================================================

# These splits are NEVER used for training.
#
# This is deliberately explicit so that a DatasetDict such as:
#
#     train
#     validation
#     test
#
# will only use:
#
#     train
#
# This prevents accidental data leakage.

EXCLUDED_SPLITS = {
    "test",
    "validation",
    "valid",
    "dev",
    "eval",
    "evaluation",
}


# ============================================================
# GPT Text Dataset
# ============================================================

class GPTTextDataset(IterableDataset):
    """
    Iterable dataset used for GPT training.

    Documents are loaded one at a time.

    Each document is:

        text
          ↓
        tokenizer
          ↓
        token IDs
          ↓
        fixed-length chunks
          ↓
        input / target tensors

    Parameters
    ----------
    dataset_paths:
        Paths to Hugging Face datasets saved with
        `save_to_disk()`.

    tokenizer:
        Instance of MyGPTTokenizer.

    sequence_length:
        Number of tokens used as model input.

    max_documents:
        Optional limit for testing.

        None means process all documents.

    stride:
        Distance between consecutive windows.

        If None, sequence_length is used.

    Notes
    -----
    This dataset intentionally uses only training data.

    Evaluation splits such as validation/test are skipped.
    """

    def __init__(
        self,
        dataset_paths: Optional[
            Sequence[Path]
        ] = None,
        tokenizer: Optional[
            MyGPTTokenizer
        ] = None,
        sequence_length: int = 512,
        max_documents: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if sequence_length < 2:

            raise ValueError(
                "sequence_length must be at least 2."
            )

        if tokenizer is None:

            raise ValueError(
                "A MyGPTTokenizer instance must be provided."
            )

        if not tokenizer.is_loaded:

            raise RuntimeError(
                "The tokenizer is not loaded."
            )

        if max_documents is not None:

            if max_documents < 1:

                raise ValueError(
                    "max_documents must be positive "
                    "when specified."
                )

        if stride is not None:

            if stride < 1:

                raise ValueError(
                    "stride must be positive "
                    "when specified."
                )

        # ----------------------------------------------------
        # Dataset paths
        # ----------------------------------------------------

        if dataset_paths is None:

            dataset_paths = list(
                DEFAULT_DATASETS.values()
            )

        self.dataset_paths = [
            Path(path)
            for path in dataset_paths
        ]

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.tokenizer = tokenizer

        self.sequence_length = (
            sequence_length
        )

        self.max_documents = (
            max_documents
        )

        self.stride = (
            stride
            if stride is not None
            else sequence_length
        )

        # ----------------------------------------------------
        # Runtime statistics
        # ----------------------------------------------------

        self.documents_seen = 0

        self.documents_processed = 0

        self.documents_skipped = 0

        self.samples_generated = 0

        self.tokens_generated = 0

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        self.dataset_statistics = {}

    # ========================================================
    # Reset Statistics
    # ========================================================

    def reset_statistics(self) -> None:
        """
        Reset runtime statistics before a new iteration.
        """

        self.documents_seen = 0

        self.documents_processed = 0

        self.documents_skipped = 0

        self.samples_generated = 0

        self.tokens_generated = 0

        self.dataset_statistics = {}

    # ========================================================
    # Load Dataset
    # ========================================================

    def _load_dataset(
        self,
        dataset_path: Path,
    ):
        """
        Load a dataset saved with Hugging Face
        `save_to_disk()`.
        """

        from datasets import load_from_disk

        if not dataset_path.exists():

            raise FileNotFoundError(
                "Dataset path does not exist:\n"
                f"{dataset_path}"
            )

        return load_from_disk(
            str(dataset_path)
        )

    # ========================================================
    # Determine Training Splits
    # ========================================================

    def _get_training_splits(
        self,
        dataset,
        dataset_name: str,
    ):
        """
        Determine which splits should be used for training.

        Supported cases:

        1. DatasetDict

            {
                "train": ...,
                "validation": ...,
                "test": ...
            }

            Result:

                train only

        2. Dataset

            A dataset without named splits.

            Result:

                dataset itself

        IMPORTANT
        ---------
        Validation/test/evaluation splits are never used.
        """

        # ----------------------------------------------------
        # DatasetDict
        # ----------------------------------------------------

        if hasattr(dataset, "items"):

            available_splits = [
                str(name)
                for name, _ in dataset.items()
            ]

            print(
                "  Available Splits : "
                f"{', '.join(available_splits)}"
            )

            training_splits = []

            # ------------------------------------------------
            # Prefer explicit train split
            # ------------------------------------------------

            for split_name, split in dataset.items():

                normalized_name = (
                    str(split_name)
                    .strip()
                    .lower()
                )

                # --------------------------------------------
                # Never use evaluation splits
                # --------------------------------------------

                if normalized_name in EXCLUDED_SPLITS:

                    print(
                        f"  Skipping Split     : "
                        f"{split_name} "
                        f"(evaluation split)"
                    )

                    continue

                # --------------------------------------------
                # Only use train-like split
                # --------------------------------------------

                if normalized_name == "train":

                    training_splits.append(
                        (
                            str(split_name),
                            split,
                        )
                    )

            # ------------------------------------------------
            # If explicit train split exists
            # ------------------------------------------------

            if training_splits:

                return training_splits

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------
            #
            # Some datasets may have an unusual name.
            #
            # We DO NOT automatically consume arbitrary
            # evaluation-looking data.
            #
            # If there is no train split, look for a split
            # that is not explicitly evaluation data.
            # ------------------------------------------------

            for split_name, split in dataset.items():

                normalized_name = (
                    str(split_name)
                    .strip()
                    .lower()
                )

                if normalized_name in EXCLUDED_SPLITS:

                    continue

                print(
                    f"  Using Split        : "
                    f"{split_name}"
                )

                return [
                    (
                        str(split_name),
                        split,
                    )
                ]

            # ------------------------------------------------
            # No valid training split
            # ------------------------------------------------

            print(
                f"  WARNING            : "
                f"No training split found for "
                f"{dataset_name}"
            )

            return []

        # ----------------------------------------------------
        # Normal Dataset
        # ----------------------------------------------------

        print(
            "  Split              : dataset"
        )

        return [
            (
                "dataset",
                dataset,
            )
        ]

    # ========================================================
    # Extract Text
    # ========================================================

    @staticmethod
    def _extract_text(
        record: dict,
    ) -> str:
        """
        Extract the `text` field from a dataset record.

        Returns
        -------
        str
            Clean text or empty string.
        """

        if not isinstance(
            record,
            dict,
        ):

            return ""

        text = record.get(
            "text",
            "",
        )

        if text is None:

            return ""

        if not isinstance(
            text,
            str,
        ):

            text = str(text)

        return text.strip()

    # ========================================================
    # Tokenize
    # ========================================================

    def _tokenize(
        self,
        text: str,
    ) -> list[int]:
        """
        Convert text to token IDs using MyGPTTokenizer.
        """

        token_ids = (
            self.tokenizer.encode(
                text
            )
        )

        if not isinstance(
            token_ids,
            list,
        ):

            raise TypeError(
                "MyGPTTokenizer.encode() "
                "must return list[int]."
            )

        return [
            int(token)
            for token in token_ids
        ]

    # ========================================================
    # Create Samples
    # ========================================================

    def _create_samples(
        self,
        token_ids: list[int],
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor
        ]
    ]:
        """
        Convert token IDs into input/target pairs.

        Example:

            Tokens:

                [10, 20, 30, 40, 50]

            sequence_length = 4

            Input:

                [10, 20, 30, 40]

            Target:

                [20, 30, 40, 50]
        """

        required_tokens = (
            self.sequence_length + 1
        )

        # ----------------------------------------------------
        # Document too short
        # ----------------------------------------------------

        if len(token_ids) < required_tokens:

            return

        # ----------------------------------------------------
        # Sliding window
        # ----------------------------------------------------

        for start in range(
            0,
            len(token_ids)
            - self.sequence_length,
            self.stride,
        ):

            end = (
                start
                + self.sequence_length
                + 1
            )

            chunk = token_ids[
                start:end
            ]

            # ------------------------------------------------
            # Ignore incomplete sequence
            # ------------------------------------------------

            if len(chunk) < required_tokens:

                continue

            # ------------------------------------------------
            # Shift by one token
            # ------------------------------------------------

            input_ids = chunk[:-1]

            target_ids = chunk[1:]

            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
            )

            target_tensor = torch.tensor(
                target_ids,
                dtype=torch.long,
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            self.samples_generated += 1

            self.tokens_generated += (
                len(input_ids)
            )

            yield (
                input_tensor,
                target_tensor,
            )

    # ========================================================
    # Iterator
    # ========================================================

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor
        ]
    ]:
        """
        Iterate over all training datasets.

        IMPORTANT
        ---------
        Only training splits are consumed.

        Evaluation splits are skipped.
        """

        # ----------------------------------------------------
        # Reset statistics
        # ----------------------------------------------------

        self.reset_statistics()

        # ----------------------------------------------------
        # Dataset loop
        # ----------------------------------------------------

        for dataset_name, dataset_path in (
            self._dataset_items()
        ):

            print()

            print(
                f"Loading Dataset: "
                f"{dataset_name}"
            )

            print(
                f"  Path               : "
                f"{dataset_path}"
            )

            # ------------------------------------------------
            # Load dataset
            # ------------------------------------------------

            try:

                dataset = (
                    self._load_dataset(
                        dataset_path
                    )
                )

            except Exception as exc:

                print(
                    f"  ERROR loading dataset: "
                    f"{exc}"
                )

                raise

            # ------------------------------------------------
            # Determine training splits
            # ------------------------------------------------

            training_splits = (
                self._get_training_splits(
                    dataset,
                    dataset_name,
                )
            )

            # ------------------------------------------------
            # No usable training split
            # ------------------------------------------------

            if not training_splits:

                print(
                    f"  WARNING: No usable "
                    f"training split."
                )

                continue

            # ------------------------------------------------
            # Split loop
            # ------------------------------------------------

            for (
                split_name,
                split,
            ) in training_splits:

                print(
                    f"  Training Split      : "
                    f"{split_name}"
                )

                # --------------------------------------------
                # Record loop
                # --------------------------------------------

                for record in split:

                    # ----------------------------------------
                    # Document limit
                    # ----------------------------------------

                    if (
                        self.max_documents
                        is not None
                        and self.documents_seen
                        >= self.max_documents
                    ):

                        print()

                        print(
                            "Maximum document limit "
                            "reached."
                        )

                        return

                    # ----------------------------------------
                    # Document counter
                    # ----------------------------------------

                    self.documents_seen += 1

                    # ----------------------------------------
                    # Extract text
                    # ----------------------------------------

                    text = (
                        self._extract_text(
                            record
                        )
                    )

                    # ----------------------------------------
                    # Empty document
                    # ----------------------------------------

                    if not text:

                        self.documents_skipped += 1

                        continue

                    # ----------------------------------------
                    # Tokenization
                    # ----------------------------------------

                    try:

                        token_ids = (
                            self._tokenize(
                                text
                            )
                        )

                    except Exception as exc:

                        self.documents_skipped += 1

                        print()

                        print(
                            "WARNING: tokenizer "
                            "failed."
                        )

                        print(
                            f"Dataset : "
                            f"{dataset_name}"
                        )

                        print(
                            f"Split   : "
                            f"{split_name}"
                        )

                        print(
                            f"Document: "
                            f"{self.documents_seen}"
                        )

                        print(
                            f"Error   : "
                            f"{exc}"
                        )

                        continue

                    # ----------------------------------------
                    # Empty token sequence
                    # ----------------------------------------

                    if not token_ids:

                        self.documents_skipped += 1

                        continue

                    # ----------------------------------------
                    # Document processed
                    # ----------------------------------------

                    self.documents_processed += 1

                    # ----------------------------------------
                    # Generate samples
                    # ----------------------------------------

                    yield from (
                        self._create_samples(
                            token_ids
                        )
                    )

    # ========================================================
    # Dataset Items
    # ========================================================

    def _dataset_items(self):

        for (
            name,
            path,
        ) in DEFAULT_DATASETS.items():

            if path.exists():

                yield (
                    name,
                    path,
                )

            else:

                print()

                print(
                    "WARNING: dataset not found:"
                )

                print(
                    f"  {name}:"
                )

                print(
                    f"  {path}"
                )

    # ========================================================
    # Statistics
    # ========================================================

    def get_statistics(self) -> dict:
        """
        Return current runtime dataset statistics.
        """

        return {

            "documents_seen": (
                self.documents_seen
            ),

            "documents_processed": (
                self.documents_processed
            ),

            "documents_skipped": (
                self.documents_skipped
            ),

            "samples_generated": (
                self.samples_generated
            ),

            "tokens_generated": (
                self.tokens_generated
            ),
        }


# ============================================================
# Dataset Factory
# ============================================================

def create_training_dataset(
    tokenizer: MyGPTTokenizer,
    sequence_length: int = 512,
    max_documents: Optional[int] = None,
    stride: Optional[int] = None,
) -> GPTTextDataset:
    """
    Create the default MyGPT2 training dataset.
    """

    return GPTTextDataset(

        dataset_paths=list(
            DEFAULT_DATASETS.values()
        ),

        tokenizer=tokenizer,

        sequence_length=sequence_length,

        max_documents=max_documents,

        stride=stride,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "MyGPT2 Training Dataset Test"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Verify tokenizer file
    # --------------------------------------------------------

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer file not found:\n"
            f"{TOKENIZER_PATH}"
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        MyGPTTokenizer.load(
            TOKENIZER_PATH
        )
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocabulary_size:,}"
    )

    print()

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    sequence_length = 32

    batch_size = 4

    max_documents = 20

    print(
        "Dataset configuration:"
    )

    print(
        f"  Sequence Length : "
        f"{sequence_length}"
    )

    print(
        f"  Batch Size      : "
        f"{batch_size}"
    )

    print(
        f"  Max Documents   : "
        f"{max_documents}"
    )

    print()

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = create_training_dataset(

        tokenizer=tokenizer,

        sequence_length=sequence_length,

        max_documents=max_documents,
    )

    print(
        "Dataset created successfully."
    )

    print()

    # --------------------------------------------------------
    # Read samples
    # --------------------------------------------------------

    sample_count = 0

    for (
        input_ids,
        target_ids,
    ) in dataset:

        sample_count += 1

        print(
            f"Sample {sample_count}"
        )

        print(
            f"Input Shape  : "
            f"{tuple(input_ids.shape)}"
        )

        print(
            f"Target Shape : "
            f"{tuple(target_ids.shape)}"
        )

        print(
            f"Input IDs    : "
            f"{input_ids.tolist()}"
        )

        print(
            f"Target IDs   : "
            f"{target_ids.tolist()}"
        )

        print(
            "-" * 70
        )

        if sample_count >= 3:

            break

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Dataset Test Summary"
    )

    print("=" * 70)

    stats = dataset.get_statistics()

    for (
        key,
        value,
    ) in stats.items():

        print(
            f"{key:25}: "
            f"{value:,}"
        )

    print()

    print(
        "Dataset test completed successfully."
    )

    print("=" * 70)