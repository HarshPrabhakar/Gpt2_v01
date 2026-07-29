"""
tokenizer_trainer.py
Minimal production-ready TokenizerTrainer with metadata support.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from tokenizer.my_tokenizer import MyGPTTokenizer
from tokenizer.tokenizer_config import TokenizerConfig


class TokenizerTrainer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.dataset_statistics = {}

    def train(self, dataset_paths: list[Path]) -> MyGPTTokenizer:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(BPE(unk_token=self.config.unk_token))
        tokenizer.normalizer = Sequence([NFC()])
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
            self._text_iterator(dataset_paths),
            trainer=trainer,
        )

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
            special_tokens=[
                (self.config.bos_token, tokenizer.token_to_id(self.config.bos_token)),
                (self.config.eos_token, tokenizer.token_to_id(self.config.eos_token)),
            ],
        )

        self._print_summary()
        self.save_metadata(self.config.output_dir)

        return MyGPTTokenizer(tokenizer)

    def _text_iterator(self, dataset_paths: list[Path]) -> Iterable[str]:
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.parent.name
            ds = load_from_disk(str(dataset_path))
            if isinstance(ds, DatasetDict):
                ds = ds["train"] if "train" in ds else next(iter(ds.values()))
            assert isinstance(ds, Dataset)
            text_col = self._detect_text_column(ds)

            processed = skipped = 0
            limit = self.config.max_documents_per_dataset

            print(f"Loading Dataset: {dataset_name}")

            for row in ds:
                if limit is not None and processed >= limit:
                    break
                txt = row.get(text_col)
                if not isinstance(txt, str):
                    skipped += 1
                    continue
                txt = txt.strip()
                if not txt:
                    skipped += 1
                    continue
                processed += 1
                yield txt

            self.dataset_statistics[dataset_name] = {
                "rows": len(ds),
                "processed": processed,
                "skipped": skipped,
                "text_column": text_col,
            }

    @staticmethod
    def _detect_text_column(dataset: Dataset) -> str:
        for c in ("text","content","body","article","story","document"):
            if c in dataset.column_names:
                return c
        raise RuntimeError("No text column found.")

    def save_metadata(self, outdir: Path):
        meta = {
            "training_date": datetime.now().isoformat(),
            "vocab_size": self.config.vocab_size,
            "max_documents_per_dataset": self.config.max_documents_per_dataset,
            "datasets": self.dataset_statistics,
            "total_rows": sum(v["rows"] for v in self.dataset_statistics.values()),
            "processed_documents": sum(v["processed"] for v in self.dataset_statistics.values()),
            "skipped_documents": sum(v["skipped"] for v in self.dataset_statistics.values()),
        }
        with open(outdir / "metadata.json","w",encoding="utf-8") as f:
            json.dump(meta,f,indent=4)

    def _print_summary(self):
        print("\nTokenizer Training Summary")
        total_p=total_s=0
        for name,st in self.dataset_statistics.items():
            print(f"\n{name}")
            print(f"Rows: {st['rows']:,}")
            print(f"Processed: {st['processed']:,}")
            print(f"Skipped: {st['skipped']:,}")
            total_p += st["processed"]
            total_s += st["skipped"]
        print(f"\nTotal Processed: {total_p:,}")
        print(f"Total Skipped: {total_s:,}")
