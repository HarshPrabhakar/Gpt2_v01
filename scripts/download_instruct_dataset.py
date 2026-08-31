from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "instruct_datasets"
)

RAW_DATASET_DIR = (
    OUTPUT_ROOT
    / "smol_smoltalk_raw"
)

TRAIN_JSONL = (
    OUTPUT_ROOT
    / "train.jsonl"
)

VALIDATION_JSONL = (
    OUTPUT_ROOT
    / "validation.jsonl"
)

TEST_JSONL = (
    OUTPUT_ROOT
    / "test.jsonl"
)

MANIFEST_PATH = (
    OUTPUT_ROOT
    / "manifest.json"
)


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"

DEFAULT_TRAIN_SAMPLES = 80000
DEFAULT_VALIDATION_SAMPLES = 4000
DEFAULT_TEST_SAMPLES = 4000

DEFAULT_SEED = 42


# ============================================================
# Validation Helpers
# ============================================================

def clean_messages(messages):
    """
    Normalize and validate one conversation.

    Output format:

    [
        {
            "role": "user",
            "content": "..."
        },
        {
            "role": "assistant",
            "content": "..."
        }
    ]
    """

    if not isinstance(messages, list):
        return None

    cleaned = []

    allowed_roles = {
        "user",
        "assistant",
        "system",
    }

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in allowed_roles:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    if len(cleaned) < 2:
        return None

    # Require at least one user and one assistant message.
    has_user = any(
        message["role"] == "user"
        for message in cleaned
    )

    has_assistant = any(
        message["role"] == "assistant"
        for message in cleaned
    )

    if not has_user or not has_assistant:
        return None

    return cleaned


# ============================================================
# JSONL Writer
# ============================================================

def write_jsonl(
    dataset: Dataset,
    output_path: Path,
) -> int:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in dataset:

            messages = clean_messages(
                row.get("messages")
            )

            if messages is None:
                continue

            record = {
                "messages": messages,
                "source": row.get(
                    "source",
                    DATASET_NAME,
                ),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    return written


# ============================================================
# Main Download / Preparation
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare the Smol-SmolTalk "
            "instruction dataset for MyGPT2."
        )
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLES,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--download-full",
        action="store_true",
        help=(
            "Store the complete Hugging Face dataset "
            "locally with save_to_disk()."
        ),
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("MYGPT2 INSTRUCTION DATASET PREPARATION")
    print("=" * 80)

    print()
    print(f"Dataset : {DATASET_NAME}")
    print(f"Output  : {OUTPUT_ROOT}")
    print(f"Seed    : {args.seed}")

    # ========================================================
    # Download
    # ========================================================

    print()
    print("=" * 80)
    print("DOWNLOADING DATASET")
    print("=" * 80)

    dataset = load_dataset(
        DATASET_NAME
    )

    if not isinstance(
        dataset,
        DatasetDict,
    ):
        raise RuntimeError(
            "Expected Hugging Face DatasetDict."
        )

    print()
    print("Available splits:")

    for split_name, split_data in dataset.items():

        print(
            f"  {split_name:<12} "
            f"{len(split_data):,} rows"
        )

    # ========================================================
    # Optionally preserve exact raw HF dataset
    # ========================================================

    if args.download_full:

        print()
        print("=" * 80)
        print("SAVING RAW DATASET")
        print("=" * 80)

        dataset.save_to_disk(
            str(RAW_DATASET_DIR)
        )

        print(
            f"Saved raw dataset:\n"
            f"{RAW_DATASET_DIR}"
        )

    # ========================================================
    # Prepare Training Source
    # ========================================================

    if "train" not in dataset:

        raise RuntimeError(
            "Dataset does not contain train split."
        )

    train_source = dataset[
        "train"
    ]

    # Fixed shuffle makes the dataset reproducible.
    train_source = train_source.shuffle(
        seed=args.seed
    )

    requested_from_train = (
        args.train_samples
        +
        args.validation_samples
    )

    if requested_from_train > len(
        train_source
    ):

        raise ValueError(
            "Requested train + validation samples "
            "exceed available train examples."
        )

    train_end = args.train_samples

    validation_end = (
        args.train_samples
        +
        args.validation_samples
    )

    train_dataset = train_source.select(
        range(
            0,
            train_end,
        )
    )

    validation_dataset = train_source.select(
        range(
            train_end,
            validation_end,
        )
    )

    # ========================================================
    # Prepare Test Dataset
    # ========================================================

    if "test" in dataset:

        test_source = dataset[
            "test"
        ].shuffle(
            seed=args.seed
        )

        test_count = min(
            args.test_samples,
            len(test_source),
        )

        test_dataset = test_source.select(
            range(test_count)
        )

    else:

        # Fallback in case dataset changes in future.
        remaining_start = validation_end

        remaining_end = min(
            remaining_start
            + args.test_samples,
            len(train_source),
        )

        test_dataset = train_source.select(
            range(
                remaining_start,
                remaining_end,
            )
        )

    # ========================================================
    # Save Normalized JSONL
    # ========================================================

    print()
    print("=" * 80)
    print("WRITING MYGPT2 JSONL DATASET")
    print("=" * 80)

    train_written = write_jsonl(
        train_dataset,
        TRAIN_JSONL,
    )

    validation_written = write_jsonl(
        validation_dataset,
        VALIDATION_JSONL,
    )

    test_written = write_jsonl(
        test_dataset,
        TEST_JSONL,
    )

    print()
    print(
        f"Train      : "
        f"{train_written:,} examples"
    )

    print(
        f"Validation : "
        f"{validation_written:,} examples"
    )

    print(
        f"Test       : "
        f"{test_written:,} examples"
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {

        "dataset_name": DATASET_NAME,

        "seed": args.seed,

        "requested": {
            "train": args.train_samples,
            "validation": args.validation_samples,
            "test": args.test_samples,
        },

        "written": {
            "train": train_written,
            "validation": validation_written,
            "test": test_written,
        },

        "files": {
            "train": str(
                TRAIN_JSONL
            ),
            "validation": str(
                VALIDATION_JSONL
            ),
            "test": str(
                TEST_JSONL
            ),
        },

        "raw_dataset_saved": bool(
            args.download_full
        ),

        "raw_dataset_directory": (
            str(RAW_DATASET_DIR)
            if args.download_full
            else None
        ),

    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # Preview
    # ========================================================

    print()
    print("=" * 80)
    print("DATASET PREVIEW")
    print("=" * 80)

    if train_written > 0:

        with TRAIN_JSONL.open(
            "r",
            encoding="utf-8",
        ) as file:

            first_line = file.readline()

        sample = json.loads(
            first_line
        )

        for message in sample[
            "messages"
        ]:

            role = message[
                "role"
            ].upper()

            content = message[
                "content"
            ]

            preview = content[
                :500
            ]

            print()
            print(f"{role}:")
            print(preview)

            if len(content) > 500:
                print("...")

    print()
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print()
    print("Files:")

    print(
        f"  {TRAIN_JSONL}"
    )

    print(
        f"  {VALIDATION_JSONL}"
    )

    print(
        f"  {TEST_JSONL}"
    )

    print(
        f"  {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()