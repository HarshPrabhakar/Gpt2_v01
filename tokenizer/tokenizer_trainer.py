"""
=========================================================
Project : MyGPT2
File    : tokenizer_trainer.py
Purpose : Train a Byte-Level BPE tokenizer.
=========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
from datetime import datetime
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk

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
    Train a Byte-Level BPE tokenizer.
    """

    def __init__(self, config: TokenizerConfig):

        self.config = config

        self.dataset_statistics = {}

    # =====================================================
    # PUBLIC
    # =====================================================

    def train(
        self,
        dataset_paths: list[Path],
    ) -> MyGPTTokenizer:

        tokenizer = Tokenizer(
            BPE(
                unk_token=self.config.unk_token
            )
        )

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

        print("\nStarting BPE Training...\n")

        tokenizer.train_from_iterator(

            iterator=self._text_iterator(dataset_paths),

            trainer=trainer,

        )

        tokenizer.post_processor = TemplateProcessing(

            single=f"{self.config.bos_token} $A {self.config.eos_token}",

            pair=f"{self.config.bos_token} $A {self.config.eos_token} "
                 f"$B:1 {self.config.eos_token}:1",

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

        self._print_summary()
        
        def save_metadata(self, output_dir):

            total_rows = sum(
                stats["rows"]
                for stats in self.dataset_statistics.values()
            )

            total_processed = sum(
                stats["processed"]
                for stats in self.dataset_statistics.values()
            )

            total_skipped = sum(
                stats["skipped"]
                for stats in self.dataset_statistics.values()
            )

            metadata = {
                "training_date": datetime.now().isoformat(),

                "vocab_size": self.config.vocab_size,

                "datasets": list(self.dataset_statistics.keys()),

                "dataset_statistics": self.dataset_statistics,

                "total_rows": total_rows,

                "processed_documents": total_processed,

                "skipped_documents": total_skipped,

                "max_documents_per_dataset":
                    self.config.max_documents_per_dataset,

                "special_tokens": {
                    "pad": self.config.pad_token,
                    "unk": self.config.unk_token,
                    "bos": self.config.bos_token,
                    "eos": self.config.eos_token,
                }
            }

            metadata_path = output_dir / "metadata.json"

            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(f"Metadata Saved : {metadata_path}")
        
        return MyGPTTokenizer(tokenizer)

    # =====================================================
    # DATASET ITERATOR
    # =====================================================

    def _text_iterator(
        self,
        dataset_paths: list[Path],
    ) -> Iterable[str]:

        for dataset_path in dataset_paths:

            dataset_name = dataset_path.parent.name

            print("=" * 70)
            print(f"Loading Dataset : {dataset_name}")
            print("=" * 70)

            dataset = load_from_disk(str(dataset_path))

            if isinstance(dataset, DatasetDict):

                if "train" in dataset:

                    dataset = dataset["train"]

                else:

                    dataset = next(iter(dataset.values()))

            assert isinstance(dataset, Dataset)

            text_column = self._detect_text_column(dataset)

            print(f"Rows        : {len(dataset):,}")
            print(f"Text Column : {text_column}")

            processed = 0
            skipped = 0

            limit = self.config.max_documents_per_dataset

            for sample in dataset:

                if limit is not None and processed >= limit:
                    break

                text = sample.get(text_column)

                if not isinstance(text, str):

                    skipped += 1
                    continue

                text = text.strip()

                if text == "":

                    skipped += 1
                    continue

                processed += 1

                yield text

            self.dataset_statistics[dataset_name] = {

                "rows": len(dataset),

                "processed": processed,

                "skipped": skipped,

                "text_column": text_column,

            }

    # =====================================================
    # UTILITIES
    # =====================================================

    @staticmethod
    def _detect_text_column(
        dataset: Dataset,
    ) -> str:

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

    # =====================================================
    # SUMMARY
    # =====================================================

    def _print_summary(self):

        print("\n")
        print("=" * 70)
        print("Tokenizer Training Summary")
        print("=" * 70)

        total_rows = 0
        total_processed = 0
        total_skipped = 0

        for dataset_name, stats in self.dataset_statistics.items():

            total_rows += stats["rows"]
            total_processed += stats["processed"]
            total_skipped += stats["skipped"]

            print(f"\n{dataset_name}")
            print("-" * 45)

            print(f"Rows        : {stats['rows']:,}")
            print(f"Text Column : {stats['text_column']}")
            print(f"Processed   : {stats['processed']:,}")
            print(f"Skipped     : {stats['skipped']:,}")

        print("\n" + "=" * 70)

        print(f"Total Rows        : {total_rows:,}")
        print(f"Total Processed   : {total_processed:,}")
        print(f"Total Skipped     : {total_skipped:,}")

        print("=" * 70)