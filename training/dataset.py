"""
============================================================
MyGPT2 - Training Dataset
============================================================

This module converts raw text documents into GPT training
samples.

Pipeline:

Raw Documents
      |
      v
MyGPTTokenizer
      |
      v
Token IDs
      |
      v
Fixed-length sequences
      |
      v
Input IDs + Target IDs
      |
      v
PyTorch Dataset

For autoregressive language modeling:

Input:
    [t1, t2, t3, t4, t5]

Target:
    [t2, t3, t4, t5, t6]

The model therefore learns to predict the next token.

============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import IterableDataset


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)


# ============================================================
# Dataset Configuration
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
# GPT Training Dataset
# ============================================================

class GPTTextDataset(IterableDataset):
    """
    Iterable dataset for GPT language-model training.

    Documents are read one at a time and converted into
    fixed-length token sequences.

    Parameters
    ----------
    dataset_paths:
        Dataset directories.

    tokenizer:
        MyGPTTokenizer instance.

    sequence_length:
        Number of tokens provided to the model.

    max_documents:
        Optional maximum number of documents to process.

        None means process all available documents.

    stride:
        Distance between consecutive windows.

        If None, defaults to sequence_length.

    add_special_tokens:
        Whether tokenizer should add BOS/EOS tokens.

    """

    def __init__(
        self,
        dataset_paths: Optional[
            Sequence[Path]
        ] = None,
        tokenizer=None,
        sequence_length: int = 512,
        max_documents: Optional[int] = None,
        stride: Optional[int] = None,
        add_special_tokens: bool = True,
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
                "A tokenizer must be provided."
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

        self.add_special_tokens = (
            add_special_tokens
        )

        # ----------------------------------------------------
        # Runtime statistics
        # ----------------------------------------------------

        self.documents_seen = 0

        self.documents_processed = 0

        self.documents_skipped = 0

        self.samples_generated = 0

        self.tokens_generated = 0

    # ========================================================
    # Reset Statistics
    # ========================================================

    def reset_statistics(self) -> None:

        self.documents_seen = 0

        self.documents_processed = 0

        self.documents_skipped = 0

        self.samples_generated = 0

        self.tokens_generated = 0

    # ========================================================
    # Read Dataset
    # ========================================================

    def _load_dataset(self, dataset_path: Path):

        """
        Load a Hugging Face dataset from disk.

        The dataset was previously saved using
        datasets.save_to_disk().
        """

        from datasets import load_from_disk

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset path does not exist:\n"
                f"{dataset_path}"
            )

        return load_from_disk(
            str(dataset_path)
        )

    # ========================================================
    # Extract Text
    # ========================================================

    @staticmethod
    def _extract_text(
        record: dict,
    ) -> str:

        """
        Extract the text field from a dataset record.
        """

        text = record.get(
            "text",
            ""
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
    # Tokenize Document
    # ========================================================

    def _tokenize(
        self,
        text: str,
    ) -> list[int]:

        """
        Convert text into token IDs.

        This function supports the custom MyGPTTokenizer
        wrapper.

        Expected tokenizer API:

            tokenizer.encode(text)

        It may return:

            list[int]

        or an object containing:

            .ids
        """

        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=(
                self.add_special_tokens
            ),
        )

        # ----------------------------------------------------
        # Direct list output
        # ----------------------------------------------------

        if isinstance(
            encoded,
            list,
        ):

            return [
                int(token)
                for token in encoded
            ]

        # ----------------------------------------------------
        # Hugging Face Encoding object
        # ----------------------------------------------------

        if hasattr(
            encoded,
            "ids",
        ):

            return [
                int(token)
                for token in encoded.ids
            ]

        # ----------------------------------------------------
        # Tensor output
        # ----------------------------------------------------

        if torch.is_tensor(
            encoded
        ):

            return [
                int(token)
                for token in encoded.tolist()
            ]

        raise TypeError(
            "Unsupported tokenizer output type: "
            f"{type(encoded)}"
        )

    # ========================================================
    # Create Training Samples
    # ========================================================

    def _create_samples(
        self,
        token_ids: list[int],
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor]
    ]:

        """
        Convert token IDs into input/target pairs.

        Example:

        Tokens:

            [1, 2, 3, 4, 5, 6]

        sequence_length = 4

        Input:

            [1, 2, 3, 4]

        Target:

            [2, 3, 4, 5]

        """

        required_tokens = (
            self.sequence_length + 1
        )

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

            if len(chunk) < required_tokens:

                continue

            input_ids = chunk[
                :-1
            ]

            target_ids = chunk[
                1:
            ]

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
            )

            target_tensor = torch.tensor(
                target_ids,
                dtype=torch.long,
            )

            self.samples_generated += 1

            self.tokens_generated += (
                len(input_ids)
            )

            yield (
                input_tensor,
                target_tensor,
            )

    # ========================================================
    # Iterate Dataset
    # ========================================================

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor]
    ]:

        self.reset_statistics()

        # ----------------------------------------------------
        # Iterate through datasets
        # ----------------------------------------------------

        for dataset_path in (
            self.dataset_paths
        ):

            dataset = self._load_dataset(
                dataset_path
            )

            # ------------------------------------------------
            # Handle DatasetDict
            # ------------------------------------------------

            if hasattr(
                dataset,
                "items",
            ):

                splits = dataset.items()

            else:

                splits = [
                    (
                        "dataset",
                        dataset,
                    )
                ]

            # ------------------------------------------------
            # Iterate splits
            # ------------------------------------------------

            for split_name, split in splits:

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

                        return

                    self.documents_seen += 1

                    # ----------------------------------------
                    # Extract text
                    # ----------------------------------------

                    text = (
                        self._extract_text(
                            record
                        )
                    )

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

                        print(
                            f"\nWarning: tokenizer "
                            f"failed for document "
                            f"{self.documents_seen}: "
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

                    yield from self._create_samples(
                        token_ids
                    )

    # ========================================================
    # Statistics
    # ========================================================

    def get_statistics(self) -> dict:

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
    tokenizer,
    sequence_length: int = 512,
    max_documents: Optional[int] = None,
    stride: Optional[int] = None,
) -> GPTTextDataset:

    """
    Convenience function for creating the GPT dataset.
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
# Module Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "MyGPT2 Training Dataset Test"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Import tokenizer
    # --------------------------------------------------------

    try:

        from tokenizer.tokenizer import (
            MyGPTTokenizer
        )

    except ImportError as exc:

        print(
            "Could not import MyGPTTokenizer."
        )

        print(
            f"Error: {exc}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "Loading tokenizer..."
    )

    tokenizer = MyGPTTokenizer(
        str(TOKENIZER_PATH)
    )

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.vocab_size:,}"
    )

    print()

    # --------------------------------------------------------
    # Create small test dataset
    # --------------------------------------------------------

    dataset = create_training_dataset(
        tokenizer=tokenizer,
        sequence_length=32,
        max_documents=20,
    )

    print(
        "Testing dataset with:"
    )

    print(
        "  Sequence length : 32"
    )

    print(
        "  Maximum documents : 20"
    )

    print()

    # --------------------------------------------------------
    # Read samples
    # --------------------------------------------------------

    sample_count = 0

    for input_ids, target_ids in dataset:

        sample_count += 1

        print(
            f"Sample {sample_count}"
        )

        print(
            f"Input shape  : "
            f"{tuple(input_ids.shape)}"
        )

        print(
            f"Target shape : "
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

    print(
        "=" * 70
    )

    print(
        "Dataset Test Summary"
    )

    print(
        "=" * 70
    )

    stats = (
        dataset.get_statistics()
    )

    for key, value in stats.items():

        print(
            f"{key:25}: {value:,}"
        )

    print()

    print(
        "Dataset test completed."
    )

    print(
        "=" * 70
    )