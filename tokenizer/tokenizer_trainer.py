"""
=========================================================
Project : MyGPT2
File    : tokenizer_trainer.py
Purpose : Train a Byte Pair Encoding (BPE) tokenizer
          using Hugging Face Tokenizers.
=========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_from_disk

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from tokenizer.my_tokenizer import MyGPTTokenizer
from tokenizer.tokenizer_config import TokenizerConfig


class TokenizerTrainer:
    """
    Trains a Byte Pair Encoding tokenizer.

    This class is responsible only for training.

    It does NOT:
        - encode text
        - decode text
        - load existing tokenizers
    """

    def __init__(
        self,
        config: TokenizerConfig
    ) -> None:

        self.config = config

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def train(
        self,
        dataset_paths: list[Path],
    ) -> MyGPTTokenizer:
        """
        Train tokenizer from multiple datasets.
        """

        tokenizer = Tokenizer(BPE(unk_token=self.config.unk_token))

        tokenizer.normalizer = Sequence([
            NFC()
        ])

        tokenizer.pre_tokenizer = ByteLevel()

        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,

            min_frequency=self.config.min_frequency,

            show_progress=self.config.show_progress,

            special_tokens=[
                self.config.pad_token,
                self.config.unk_token,
                self.config.bos_token,
                self.config.eos_token,
            ],
        )

        tokenizer.train_from_iterator(
            iterator=self._text_iterator(dataset_paths),
            trainer=trainer,
        )

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",

            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",

            special_tokens=[
                (
                    self.config.bos_token,
                    tokenizer.token_to_id(self.config.bos_token),
                ),
                (
                    self.config.eos_token,
                    tokenizer.token_to_id(self.config.eos_token),
                ),
            ],
        )

        return MyGPTTokenizer(tokenizer)

    # --------------------------------------------------
    # Dataset Iterator
    # --------------------------------------------------

    def _text_iterator(
        self,
        dataset_paths: list[Path],
    ) -> Iterable[str]:
        """
        Streams text from every dataset.
        """

        for dataset_path in dataset_paths:

            print(f"\nLoading {dataset_path.name}...")

            dataset = load_from_disk(str(dataset_path))

            if isinstance(dataset, DatasetDict):

                if "train" in dataset:

                    dataset = dataset["train"]

                else:

                    dataset = next(iter(dataset.values()))

            assert isinstance(dataset, Dataset)

            text_column = self._detect_text_column(dataset)

            print(
                f"Found {len(dataset):,} samples "
                f"using '{text_column}' column."
            )

            for sample in dataset:

                text = sample.get(text_column)

                if not isinstance(text, str):
                    continue

                text = text.strip()

                if text:

                    yield text

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def _detect_text_column(
        dataset: Dataset,
    ) -> str:
        """
        Automatically detect text column.
        """

        candidates = [

            "text",

            "content",

            "body",

            "article",

            "story",

            "document",

        ]

        columns = dataset.column_names

        for candidate in candidates:

            if candidate in columns:

                return candidate

        for column in columns:

            lower = column.lower()

            for candidate in candidates:

                if candidate in lower:

                    return column

        raise RuntimeError(
            "Unable to detect text column."
        )