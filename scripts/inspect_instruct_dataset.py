from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from tokenizer.my_tokenizer import MyGPTTokenizer

from training.instruct.dataset import (
    IGNORE_INDEX,
    InstructionDataset,
)


TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)


DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "instruct_datasets"
    / "train.jsonl"
)


SEQUENCE_LENGTH = 512


def separator():

    print()
    print("=" * 80)
    print()


def main():

    print("=" * 80)
    print(
        "MYGPT2 INSTRUCTION DATASET INSPECTION"
    )
    print("=" * 80)

    # ========================================================
    # Load tokenizer
    # ========================================================

    tokenizer = MyGPTTokenizer.load(
        TOKENIZER_PATH
    )

    print()
    print(
        f"Tokenizer loaded : "
        f"{TOKENIZER_PATH}"
    )

    print(
        f"Vocabulary       : "
        f"{tokenizer.vocabulary_size:,}"
    )

    # ========================================================
    # Load dataset
    # ========================================================

    dataset = InstructionDataset(
        file_path=DATASET_PATH,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
    )

    print()
    print(
        f"Conversations    : "
        f"{dataset.conversation_count:,}"
    )

    print(
        f"SFT exchanges    : "
        f"{len(dataset):,}"
    )

    print(
        f"Context length   : "
        f"{SEQUENCE_LENGTH:,}"
    )

    # ========================================================
    # Inspect examples
    # ========================================================

    inspect_indices = [
        0,
        1,
        2,
        10,
        100,
    ]

    inspect_indices = [
        index
        for index in inspect_indices
        if index < len(dataset)
    ]

    for index in inspect_indices:

        sample = dataset[
            index
        ]

        input_ids = sample[
            "input_ids"
        ]

        labels = sample[
            "labels"
        ]

        # ----------------------------------------------------
        # Structural checks
        # ----------------------------------------------------

        if not isinstance(
            input_ids,
            torch.Tensor,
        ):
            raise RuntimeError(
                "input_ids is not a tensor."
            )

        if not isinstance(
            labels,
            torch.Tensor,
        ):
            raise RuntimeError(
                "labels is not a tensor."
            )

        if input_ids.dtype != torch.long:
            raise RuntimeError(
                "input_ids must use torch.long."
            )

        if labels.dtype != torch.long:
            raise RuntimeError(
                "labels must use torch.long."
            )

        if len(input_ids) != len(labels):
            raise RuntimeError(
                "input_ids and labels "
                "have different lengths."
            )

        if len(input_ids) > SEQUENCE_LENGTH:
            raise RuntimeError(
                "Sequence exceeds context length."
            )

        # ----------------------------------------------------
        # Loss mask
        # ----------------------------------------------------

        train_mask = (
            labels != IGNORE_INDEX
        )

        trained_count = int(
            train_mask.sum().item()
        )

        ignored_count = int(
            (~train_mask).sum().item()
        )

        if trained_count <= 1:
            raise RuntimeError(
                f"Sample {index} contains "
                f"no meaningful assistant target."
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        full_text = tokenizer.decode(
            input_ids.tolist()
        )

        trained_token_ids = labels[
            train_mask
        ].tolist()

        target_text = tokenizer.decode(
            trained_token_ids
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        separator()

        print(
            f"SAMPLE {index}"
        )

        print("-" * 80)

        print(
            f"Sequence tokens     : "
            f"{len(input_ids):,}"
        )

        print(
            f"Ignored tokens      : "
            f"{ignored_count:,}"
        )

        print(
            f"Training tokens     : "
            f"{trained_count:,}"
        )

        print(
            f"Conversation index  : "
            f"{sample['conversation_index']:,}"
        )

        print(
            f"Assistant turn      : "
            f"{sample['assistant_turn']:,}"
        )

        print(
            f"Source              : "
            f"{sample.get('source')}"
        )

        print()
        print("-" * 80)
        print("FULL MODEL INPUT")
        print("-" * 80)
        print()

        print(
            full_text[:4000]
        )

        print()
        print("-" * 80)
        print(
            "ASSISTANT TRAINING TARGET"
        )
        print("-" * 80)
        print()

        print(
            target_text[:4000]
        )

    # ========================================================
    # Dataset-wide sanity scan
    # ========================================================

    separator()

    print(
        "RUNNING DATASET SANITY SCAN..."
    )

    print()

    scan_count = min(
        1000,
        len(dataset),
    )

    maximum_length = 0
    minimum_length = SEQUENCE_LENGTH

    total_tokens = 0
    total_training_tokens = 0

    full_context_samples = 0

    for index in range(
        scan_count
    ):

        sample = dataset[
            index
        ]

        input_ids = sample[
            "input_ids"
        ]

        labels = sample[
            "labels"
        ]

        length = len(
            input_ids
        )

        training_tokens = int(
            (
                labels
                != IGNORE_INDEX
            )
            .sum()
            .item()
        )

        if length > SEQUENCE_LENGTH:
            raise RuntimeError(
                f"Sample {index} exceeds "
                f"context length."
            )

        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"Sample {index} has "
                f"input/label mismatch."
            )

        if training_tokens <= 1:
            raise RuntimeError(
                f"Sample {index} contains "
                f"no assistant supervision."
            )

        maximum_length = max(
            maximum_length,
            length,
        )

        minimum_length = min(
            minimum_length,
            length,
        )

        total_tokens += length

        total_training_tokens += (
            training_tokens
        )

        if length == SEQUENCE_LENGTH:
            full_context_samples += 1

    average_length = (
        total_tokens
        /
        scan_count
    )

    average_training_tokens = (
        total_training_tokens
        /
        scan_count
    )

    print(
        f"Scanned samples         : "
        f"{scan_count:,}"
    )

    print(
        f"Minimum length          : "
        f"{minimum_length:,}"
    )

    print(
        f"Maximum length          : "
        f"{maximum_length:,}"
    )

    print(
        f"Average length          : "
        f"{average_length:.2f}"
    )

    print(
        f"Average training tokens : "
        f"{average_training_tokens:.2f}"
    )

    print(
        f"Full 512-token samples  : "
        f"{full_context_samples:,}"
    )

    separator()

    print(
        "DATASET INSPECTION PASSED"
    )

    print()

    print(
        "The dataset is structurally ready "
        "for the SFT collator/trainer."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()