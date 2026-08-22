"""
======================================================================
MyGPT2 - FINAL Training Entry Point
======================================================================

Purpose
-------
Final, reproducible GPT-2 training pipeline.

This version explicitly locks:

    TinyStories
    WikiText103
    OpenWebText
    FineWeb

The dataset mixture, order, tokenizer, model configuration and
training configuration are written into every checkpoint.

Resume is SAFE by default:

    - New checkpoints must contain the training manifest.
    - Dataset mixture must match exactly.
    - Dataset paths must match.
    - Dataset signatures must match.
    - Tokenizer must match.
    - Model configuration must match.
    - Batch size / sequence length must match.

Legacy checkpoints created by the old train.py are rejected unless:

    --allow-legacy-resume

is explicitly supplied.

Examples
--------

Fresh final training:

    python train.py --max-steps 50000

Resume:

    python train.py \
        --resume artifacts/checkpoints/step_00020000.pt \
        --max-steps 50000

Legacy checkpoint:

    python train.py \
        --resume artifacts/checkpoints/step_00020000.pt \
        --max-steps 50000 \
        --allow-legacy-resume

CPU:

    python train.py --device cpu --max-steps 100

Test:

    python train.py --max-steps 10 --max-documents 1000

======================================================================
"""

from __future__ import annotations

# ======================================================================
# Standard Library
# ======================================================================

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


# ======================================================================
# Third Party
# ======================================================================

import numpy as np
import torch
from torch.utils.data import DataLoader


# ======================================================================
# Project Root
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# Project Imports
# ======================================================================

from model.config import GPTConfig
from model.model import MyGPTModel

from tokenizer.my_tokenizer import MyGPTTokenizer

from training.trainer import Trainer
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


# ======================================================================
# Constants
# ======================================================================

TRAINING_VERSION = "2.0-final-locked-mixture"

DEFAULT_TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)

DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
)

DEFAULT_NUM_WORKERS = 0

DEFAULT_WARMUP_RATIO = 0.10

CHECKPOINT_INTERVAL = 1000


# ======================================================================
# LOCKED DATASET MIXTURE
# ======================================================================
#
# IMPORTANT:
#
# Do NOT casually change this after final training starts.
#
# The exact order is:
#
#     1. TinyStories
#     2. WikiText103
#     3. OpenWebText
#     4. FineWeb
#
# training/dataset.py previously relied on DEFAULT_DATASETS.
# We explicitly replace that configuration before creating the
# dataset so that this train.py owns the dataset mixture.
# ======================================================================

LOCKED_DATASETS = OrderedDict(
    [
        (
            "TinyStories",
            PROJECT_ROOT
            / "datasets"
            / "TinyStories"
            / "raw",
        ),
        (
            "WikiText103",
            PROJECT_ROOT
            / "datasets"
            / "WikiText103"
            / "raw",
        ),
        (
            "OpenWebText",
            PROJECT_ROOT
            / "datasets"
            / "OpenWebText"
            / "raw",
        ),
        (
            "FineWeb",
            PROJECT_ROOT
            / "datasets"
            / "FineWeb"
            / "raw",
        ),
    ]
)


# ======================================================================
# Argument Parser
# ======================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Final MyGPT2 training pipeline "
            "with locked dataset mixture."
        )
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum optimizer steps. "
            "Required for step-limited final training."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs for non-step-limited training.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help=(
            "Maximum documents across the locked dataset mixture. "
            "Useful only for testing."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "DataLoader workers. "
            "Use 0 on Windows."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        help="Training device.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from.",
    )

    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help=(
            "Allow loading an old checkpoint without the "
            "new locked-mixture manifest."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable checkpoint saving.",
    )

    return parser.parse_args()


# ======================================================================
# Reproducibility
# ======================================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        # Performance-oriented setting.
        torch.backends.cudnn.benchmark = True


# ======================================================================
# Device
# ======================================================================

def get_device(
    requested_device: str,
) -> torch.device:

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was explicitly requested, "
                "but CUDA is not available."
            )

        return torch.device("cuda")

    if requested_device == "cpu":

        return torch.device("cpu")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ======================================================================
# Device Information
# ======================================================================

def print_device_information(
    device: torch.device,
) -> None:

    print()
    print("=" * 75)
    print("Device Information")
    print("=" * 75)

    print(
        f"PyTorch Version : {torch.__version__}"
    )

    print(
        f"CUDA Available  : "
        f"{'YES' if torch.cuda.is_available() else 'NO'}"
    )

    print(
        f"Selected Device : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        properties = (
            torch.cuda.get_device_properties(0)
        )

        memory_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU Memory      : "
            f"{memory_gb:.2f} GB"
        )

        print(
            f"CUDA Version    : "
            f"{torch.version.cuda}"
        )

    print("=" * 75)


# ======================================================================
# Build Model Configuration
# ======================================================================

def build_config(
    args: argparse.Namespace,
    device: torch.device,
) -> GPTConfig:

    config = GPTConfig()

    if args.epochs is not None:

        if args.epochs <= 0:

            raise ValueError(
                "--epochs must be greater than zero."
            )

        config.max_epochs = args.epochs

    if args.batch_size is not None:

        if args.batch_size <= 0:

            raise ValueError(
                "--batch-size must be greater than zero."
            )

        config.batch_size = args.batch_size

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers cannot be negative."
        )

    # Some existing versions of GPTConfig expose device,
    # while others do not. Set it only when available.
    if hasattr(config, "device"):

        config.device = device.type

    return config


# ======================================================================
# Convert Configuration to Dictionary
# ======================================================================

def config_to_dict(
    config: GPTConfig,
) -> dict[str, Any]:

    result = {}

    if hasattr(config, "__dict__"):

        for key, value in vars(config).items():

            try:

                json.dumps(value)

                result[key] = value

            except TypeError:

                result[key] = str(value)

    else:

        for key in dir(config):

            if key.startswith("_"):

                continue

            try:

                value = getattr(config, key)

            except Exception:

                continue

            if callable(value):

                continue

            try:

                json.dumps(value)

                result[key] = value

            except TypeError:

                continue

    return result


# ======================================================================
# Parameter Count
# ======================================================================

def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ======================================================================
# Print Model Information
# ======================================================================

def print_model_information(
    model: torch.nn.Module,
    config: GPTConfig,
) -> None:

    total, trainable = count_parameters(model)

    print()
    print("=" * 75)
    print("Model Configuration")
    print("=" * 75)

    print(
        f"Vocabulary Size      : "
        f"{config.vocab_size:,}"
    )

    print(
        f"Context Length       : "
        f"{config.max_position_embeddings:,}"
    )

    print(
        f"Hidden Size          : "
        f"{config.hidden_size:,}"
    )

    print(
        f"Transformer Layers   : "
        f"{config.num_layers}"
    )

    print(
        f"Attention Heads      : "
        f"{config.num_attention_heads}"
    )

    print(
        f"Intermediate Size    : "
        f"{config.intermediate_size:,}"
    )

    print(
        f"Total Parameters     : "
        f"{total:,}"
    )

    print(
        f"Total Parameters     : "
        f"{total / 1_000_000:.2f}M"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable:,}"
    )

    print("=" * 75)


# ======================================================================
# Verify Tokenizer
# ======================================================================

def load_tokenizer() -> MyGPTTokenizer:

    print()
    print("=" * 75)
    print("Tokenizer")
    print("=" * 75)

    if not DEFAULT_TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer not found:\n"
            f"{DEFAULT_TOKENIZER_PATH}"
        )

    size_mb = (
        DEFAULT_TOKENIZER_PATH.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Path            : "
        f"{DEFAULT_TOKENIZER_PATH}"
    )

    print(
        f"Size            : "
        f"{size_mb:.2f} MB"
    )

    tokenizer = (
        MyGPTTokenizer.load(
            DEFAULT_TOKENIZER_PATH
        )
    )

    print(
        "Tokenizer       : ✅ LOADED"
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocabulary_size:,}"
    )

    print("=" * 75)

    return tokenizer


# ======================================================================
# File Hash
# ======================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(chunk_size)

            if not chunk:

                break

            digest.update(chunk)

    return digest.hexdigest()


# ======================================================================
# Dataset Signature
# ======================================================================
#
# We deliberately do NOT hash every Arrow byte.
#
# Dataset files can be extremely large.
#
# Instead we create a deterministic signature from:
#
#     relative path
#     file size
#     modification timestamp
#
# This catches accidental replacement/modification of the local
# dataset tree without forcing a multi-GB hashing operation.
# ======================================================================

def dataset_signature(
    path: Path,
) -> str:

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset path does not exist:\n{path}"
        )

    entries = []

    if path.is_file():

        stat = path.stat()

        entries.append(
            (
                path.name,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )

    else:

        for file_path in sorted(
            path.rglob("*")
        ):

            if not file_path.is_file():

                continue

            stat = file_path.stat()

            relative = (
                file_path
                .relative_to(path)
                .as_posix()
            )

            entries.append(
                (
                    relative,
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )

    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


# ======================================================================
# Locked Dataset Manifest
# ======================================================================

def build_dataset_manifest() -> dict[str, Any]:

    datasets = []

    print()
    print("=" * 75)
    print("LOCKED DATASET MIXTURE")
    print("=" * 75)

    print(
        "Dataset order is FIXED:"
    )

    for index, (
        name,
        path,
    ) in enumerate(
        LOCKED_DATASETS.items(),
        start=1,
    ):

        path = path.resolve()

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired training dataset "
                "was not found.\n\n"
                f"Dataset : {name}\n"
                f"Path    : {path}\n"
            )

        signature = dataset_signature(
            path
        )

        item = {
            "index": index,
            "name": name,
            "path": str(path),
            "signature": signature,
        }

        datasets.append(item)

        print()
        print(
            f"{index}. {name}"
        )

        print(
            f"   Path      : {path}"
        )

        print(
            f"   Signature : {signature}"
        )

    print()
    print(
        "Dataset mixture : "
        "TinyStories + WikiText103 + "
        "OpenWebText + FineWeb"
    )

    print("=" * 75)

    return {
        "locked": True,
        "order": [
            item["name"]
            for item in datasets
        ],
        "datasets": datasets,
    }


# ======================================================================
# Lock Dataset Module
# ======================================================================
#
# training/dataset.py currently resolves DEFAULT_DATASETS at runtime
# from the module global. We replace that global with our locked
# configuration BEFORE creating GPTTextDataset.
# ======================================================================

def lock_dataset_module() -> None:

    import training.dataset as dataset_module

    locked = OrderedDict()

    for name, path in LOCKED_DATASETS.items():

        locked[name] = path.resolve()

    dataset_module.DEFAULT_DATASETS = locked

    print()
    print(
        "Dataset module lock : ✅ ENABLED"
    )


# ======================================================================
# Create Training Dataset
# ======================================================================

def create_locked_dataset(
    tokenizer: MyGPTTokenizer,
    config: GPTConfig,
    max_documents: int | None,
):
    """
    Create GPTTextDataset after locking DEFAULT_DATASETS.
    """

    lock_dataset_module()

    from training.dataset import GPTTextDataset

    dataset = GPTTextDataset(
        dataset_paths=list(
            LOCKED_DATASETS.values()
        ),
        tokenizer=tokenizer,
        sequence_length=(
            config.max_position_embeddings
        ),
        max_documents=max_documents,
    )

    return dataset


# ======================================================================
# Create DataLoader
# ======================================================================

def create_train_loader(
    dataset,
    config: GPTConfig,
    num_workers: int,
    device: torch.device,
):

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=True,
    )

    return loader


# ======================================================================
# Dataset Statistics
# ======================================================================

def get_dataset_statistics(
    loader,
) -> dict[str, Any] | None:

    dataset = getattr(
        loader,
        "dataset",
        None,
    )

    if dataset is None:

        return None

    function = getattr(
        dataset,
        "get_statistics",
        None,
    )

    if callable(function):

        try:

            return function()

        except Exception:

            return None

    return None


# ======================================================================
# Validate Loader
# ======================================================================

def validate_dataloader(
    loader,
    config: GPTConfig,
) -> None:

    try:

        batches = len(loader)

    except TypeError:

        batches = None

    if batches is not None:

        if batches <= 0:

            raise RuntimeError(
                "Training DataLoader contains "
                "zero batches."
            )

    iterator = iter(loader)

    try:

        batch = next(iterator)

    except StopIteration:

        raise RuntimeError(
            "Training DataLoader produced "
            "no batches."
        )

    if not isinstance(
        batch,
        (tuple, list),
    ):

        raise RuntimeError(
            "DataLoader must return "
            "(input_ids, labels)."
        )

    if len(batch) != 2:

        raise RuntimeError(
            "DataLoader batch must contain "
            "exactly two tensors."
        )

    input_ids, labels = batch

    expected_shape = (
        config.batch_size,
        config.max_position_embeddings,
    )

    if tuple(input_ids.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected input shape.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {tuple(input_ids.shape)}"
        )

    if tuple(labels.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected label shape.\n"
            f"Expected : {expected_shape}\n"
            f"Actual   : {tuple(labels.shape)}"
        )

    if input_ids.dtype != torch.long:

        raise RuntimeError(
            "input_ids must be torch.long."
        )

    if labels.dtype != torch.long:

        raise RuntimeError(
            "labels must be torch.long."
        )

    if input_ids.numel():

        minimum = int(
            input_ids.min().item()
        )

        maximum = int(
            input_ids.max().item()
        )

        if minimum < 0:

            raise RuntimeError(
                "Negative token IDs detected."
            )

        if maximum >= config.vocab_size:

            raise RuntimeError(
                "Token ID exceeds vocabulary.\n"
                f"Maximum : {maximum}\n"
                f"Vocab   : {config.vocab_size}"
            )

    # GPT next-token shift:
    #
    # input[1:] == labels[:-1]

    if input_ids.shape[1] >= 2:

        if not torch.equal(
            input_ids[0, 1:],
            labels[0, :-1],
        ):

            raise RuntimeError(
                "Dataset token shifting is invalid."
            )

    print()
    print(
        "DataLoader validation : ✅ PASSED"
    )

    print(
        f"Input Shape           : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Target Shape          : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Input DType           : "
        f"{input_ids.dtype}"
    )

    print(
        f"Target DType          : "
        f"{labels.dtype}"
    )


# ======================================================================
# Print Loader Information
# ======================================================================

def print_loader_information(
    loader,
    config: GPTConfig,
    max_documents: int | None,
) -> int | None:

    try:

        batches = len(loader)

    except TypeError:

        batches = None

    print()
    print("=" * 75)
    print("Training DataLoader")
    print("=" * 75)

    print(
        f"Batch Size      : "
        f"{config.batch_size}"
    )

    print(
        f"Sequence Length : "
        f"{config.max_position_embeddings}"
    )

    print(
        f"Max Documents   : "
        f"{max_documents if max_documents is not None else 'ALL'}"
    )

    print(
        f"Workers         : "
        f"{getattr(loader, 'num_workers', 0)}"
    )

    if batches is not None:

        print(
            f"Batches / Epoch : "
            f"{batches:,}"
        )

    else:

        print(
            "Batches / Epoch : Unknown"
        )

    statistics = (
        get_dataset_statistics(loader)
    )

    if statistics:

        print()
        print(
            "Current Dataset Statistics"
        )

        print("-" * 75)

        for key, value in statistics.items():

            print(
                f"{key:25}: {value}"
            )

    print("=" * 75)

    return batches


# ======================================================================
# Training Step Calculation
# ======================================================================

def calculate_total_steps(
    batches_per_epoch: int | None,
    config: GPTConfig,
    max_steps: int | None,
) -> int:

    if max_steps is not None:

        if max_steps <= 0:

            raise ValueError(
                "--max-steps must be greater than zero."
            )

        return max_steps

    if batches_per_epoch is None:

        raise RuntimeError(
            "Cannot determine total training steps."
        )

    if batches_per_epoch <= 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    if config.max_epochs <= 0:

        raise ValueError(
            "Training epochs must be greater than zero."
        )

    return (
        batches_per_epoch
        * config.max_epochs
    )


# ======================================================================
# Scheduler
# ======================================================================

def create_training_scheduler(
    optimizer,
    total_steps: int,
):

    if total_steps <= 0:

        raise ValueError(
            "total_steps must be greater than zero."
        )

    warmup_steps = max(
        1,
        int(
            total_steps
            * DEFAULT_WARMUP_RATIO
        ),
    )

    if total_steps > 1:

        warmup_steps = min(
            warmup_steps,
            total_steps - 1,
        )

    print()
    print("=" * 75)
    print("Scheduler")
    print("=" * 75)

    print(
        f"Total Steps     : "
        f"{total_steps:,}"
    )

    print(
        f"Warmup Steps    : "
        f"{warmup_steps:,}"
    )

    print(
        f"Warmup Ratio    : "
        f"{warmup_steps / total_steps * 100:.2f}%"
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    print(
        "Scheduler       : ✅ CREATED"
    )

    print("=" * 75)

    return scheduler


# ======================================================================
# Checkpoint Utilities
# ======================================================================

def ensure_checkpoint_directory() -> None:

    DEFAULT_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def resolve_resume_path(
    argument: str | None,
) -> Path | None:

    if argument is None:

        return None

    path = Path(argument)

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    path = path.resolve()

    if not path.exists():

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{path}"
        )

    return path


# ======================================================================
# Read Checkpoint Safely
# ======================================================================

def read_checkpoint(
    path: Path,
) -> dict[str, Any]:

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    return checkpoint


# ======================================================================
# Build Training Manifest
# ======================================================================

def build_training_manifest(
    config: GPTConfig,
    tokenizer: MyGPTTokenizer,
    dataset_manifest: dict[str, Any],
    total_steps: int,
) -> dict[str, Any]:

    total_params = None

    # The model parameter count is filled later.

    return {
        "training_version": TRAINING_VERSION,

        "dataset": dataset_manifest,

        "tokenizer": {
            "path": str(
                DEFAULT_TOKENIZER_PATH.resolve()
            ),
            "size_bytes": (
                DEFAULT_TOKENIZER_PATH.stat()
                .st_size
            ),
            "sha256": sha256_file(
                DEFAULT_TOKENIZER_PATH
            ),
            "vocab_size": (
                tokenizer.vocabulary_size
            ),
        },

        "model_config": config_to_dict(
            config
        ),

        "training": {
            "batch_size": config.batch_size,
            "sequence_length": (
                config.max_position_embeddings
            ),
            "learning_rate": (
                config.learning_rate
            ),
            "weight_decay": (
                config.weight_decay
            ),
            "seed": config.seed,
            "total_steps": total_steps,
            "warmup_ratio": (
                DEFAULT_WARMUP_RATIO
            ),
            "checkpoint_interval": (
                CHECKPOINT_INTERVAL
            ),
        },
    }


# ======================================================================
# Validate Resume Manifest
# ======================================================================

def validate_resume_manifest(
    checkpoint: dict[str, Any],
    current_manifest: dict[str, Any],
    allow_legacy: bool,
) -> None:

    saved_manifest = checkpoint.get(
        "mygpt2_training_manifest"
    )

    if saved_manifest is None:

        if allow_legacy:

            print()
            print(
                "⚠️ LEGACY CHECKPOINT RESUME"
            )

            print(
                "Checkpoint does not contain "
                "the locked-mixture manifest."
            )

            print(
                "Dataset safety cannot be fully "
                "verified."
            )

            return

        raise RuntimeError(
            "\nUNSAFE RESUME BLOCKED.\n\n"
            "This checkpoint was created by an "
            "older train.py and does not contain "
            "the locked dataset manifest.\n\n"
            "This is exactly the type of checkpoint "
            "that can cause the dataset-mixture problem "
            "we are fixing.\n\n"
            "Start a fresh final training run, or "
            "explicitly use:\n\n"
            "--allow-legacy-resume\n"
        )

    # --------------------------------------------------------------
    # Dataset mixture
    # --------------------------------------------------------------

    saved_dataset = (
        saved_manifest.get("dataset", {})
    )

    current_dataset = (
        current_manifest.get("dataset", {})
    )

    if saved_dataset != current_dataset:

        raise RuntimeError(
            "\nDATASET MIXTURE MISMATCH.\n\n"
            "The checkpoint was created using a "
            "different dataset configuration.\n\n"
            f"Saved:\n"
            f"{json.dumps(saved_dataset, indent=2)}\n\n"
            f"Current:\n"
            f"{json.dumps(current_dataset, indent=2)}"
        )

    # --------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------

    saved_tokenizer = (
        saved_manifest.get(
            "tokenizer",
            {},
        )
    )

    current_tokenizer = (
        current_manifest.get(
            "tokenizer",
            {},
        )
    )

    if saved_tokenizer != current_tokenizer:

        raise RuntimeError(
            "\nTOKENIZER MISMATCH.\n\n"
            "The tokenizer used by the checkpoint "
            "does not match the current tokenizer."
        )

    # --------------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------------

    saved_model = (
        saved_manifest.get(
            "model_config",
            {},
        )
    )

    current_model = (
        current_manifest.get(
            "model_config",
            {},
        )
    )

    # Device can legitimately differ between runs.
    saved_model = dict(saved_model)
    current_model = dict(current_model)

    saved_model.pop("device", None)
    current_model.pop("device", None)

    if saved_model != current_model:

        raise RuntimeError(
            "\nMODEL CONFIGURATION MISMATCH.\n\n"
            "The checkpoint model configuration "
            "does not match the current model."
        )

    # --------------------------------------------------------------
    # Critical training parameters
    # --------------------------------------------------------------

    saved_training = (
        saved_manifest.get(
            "training",
            {},
        )
    )

    current_training = (
        current_manifest.get(
            "training",
            {},
        )
    )

    critical_keys = [
        "batch_size",
        "sequence_length",
        "learning_rate",
        "weight_decay",
        "seed",
        "warmup_ratio",
    ]

    for key in critical_keys:

        if (
            saved_training.get(key)
            != current_training.get(key)
        ):

            raise RuntimeError(
                "\nTRAINING CONFIGURATION MISMATCH.\n\n"
                f"Parameter : {key}\n"
                f"Saved     : "
                f"{saved_training.get(key)}\n"
                f"Current   : "
                f"{current_training.get(key)}"
            )

    print()
    print(
        "Resume manifest validation : ✅ PASSED"
    )


# ======================================================================
# Add Manifest to Checkpoint
# ======================================================================

def save_checkpoint_with_manifest(
    trainer,
    filename: str,
    manifest: dict[str, Any],
) -> Path:

    ensure_checkpoint_directory()

    # First let the existing Trainer create the checkpoint.
    #
    # This preserves the project's existing checkpoint format.

    created = trainer.save(
        filename
    )

    path = Path(created)

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    path = path.resolve()

    checkpoint = read_checkpoint(
        path
    )

    checkpoint[
        "mygpt2_training_manifest"
    ] = manifest

    checkpoint[
        "mygpt2_checkpoint_version"
    ] = TRAINING_VERSION

    checkpoint[
        "mygpt2_checkpoint_saved_at"
    ] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------------
    # Atomic replacement
    # --------------------------------------------------------------

    temp_path = path.with_suffix(
        ".tmp"
    )

    try:

        torch.save(
            checkpoint,
            temp_path,
        )

        os.replace(
            temp_path,
            path,
        )

    finally:

        if temp_path.exists():

            try:

                temp_path.unlink()

            except Exception:

                pass

    return path


# ======================================================================
# Print GPU Memory
# ======================================================================

def print_gpu_memory(
    device: torch.device,
) -> None:

    if device.type != "cuda":

        return

    allocated = (
        torch.cuda.memory_allocated(
            device
        )
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved(
            device
        )
        / (1024 ** 2)
    )

    print(
        f"GPU Memory Allocated : "
        f"{allocated:.2f} MB"
    )

    print(
        f"GPU Memory Reserved  : "
        f"{reserved:.2f} MB"
    )


# ======================================================================
# Create Trainer
# ======================================================================

def create_training_trainer(
    model,
    train_loader,
    config,
    optimizer,
    scheduler,
    device,
):

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    return trainer


# ======================================================================
# Print Training Plan
# ======================================================================

def print_training_plan(
    config: GPTConfig,
    batches_per_epoch: int | None,
    total_steps: int,
    max_steps: int | None,
    max_documents: int | None,
) -> None:

    print()
    print("=" * 75)
    print("FINAL TRAINING PLAN")
    print("=" * 75)

    print(
        "Dataset Mixture : "
        "LOCKED"
    )

    print(
        "Datasets        : "
        "TinyStories, WikiText103, "
        "OpenWebText, FineWeb"
    )

    print(
        f"Epochs          : "
        f"{config.max_epochs}"
    )

    print(
        f"Batch Size      : "
        f"{config.batch_size}"
    )

    print(
        f"Sequence Length : "
        f"{config.max_position_embeddings}"
    )

    print(
        f"Max Documents   : "
        f"{max_documents if max_documents is not None else 'ALL'}"
    )

    if batches_per_epoch is not None:

        print(
            f"Batches / Epoch : "
            f"{batches_per_epoch:,}"
        )

    else:

        print(
            "Batches / Epoch : Unknown"
        )

    print(
        f"Target Steps    : "
        f"{total_steps:,}"
    )

    if max_steps is not None:

        print(
            "Training Mode   : "
            "STEP-LIMITED"
        )

    else:

        print(
            "Training Mode   : "
            "EPOCH-BASED"
        )

    print(
        "Checkpointing   : "
        "EVERY 1000 STEPS"
    )

    print(
        "Resume Safety   : "
        "ENABLED"
    )

    print("=" * 75)


# ======================================================================
# Step-Limited Training
# ======================================================================

def run_step_limited_training(
    trainer,
    train_loader,
    target_steps: int,
    no_save: bool,
    manifest: dict[str, Any],
    device: torch.device,
) -> None:

    print()
    print("=" * 75)
    print("Starting FINAL Step-Limited Training")
    print("=" * 75)

    print(
        f"Starting Step   : "
        f"{trainer.global_step:,}"
    )

    print(
        f"Target Step     : "
        f"{target_steps:,}"
    )

    print(
        "Dataset Mixture : "
        "LOCKED"
    )

    print(
        "Checkpointing   : "
        f"{'ENABLED' if not no_save else 'DISABLED'}"
    )

    print("=" * 75)

    start_time = time.time()

    data_iterator = iter(
        train_loader
    )

    dataset_pass = 1

    while (
        trainer.global_step
        < target_steps
    ):

        try:

            batch = next(
                data_iterator
            )

        except StopIteration:

            dataset_pass += 1

            print()
            print(
                "=" * 75
            )

            print(
                "Dataset pass completed."
            )

            print(
                f"Starting pass    : "
                f"{dataset_pass}"
            )

            print(
                f"Current Step     : "
                f"{trainer.global_step:,}"
            )

            print(
                "=" * 75
            )

            data_iterator = iter(
                train_loader
            )

            continue

        if not isinstance(
            batch,
            (tuple, list),
        ):

            raise RuntimeError(
                "DataLoader must return "
                "(input_ids, labels)."
            )

        if len(batch) != 2:

            raise RuntimeError(
                "DataLoader batch must contain "
                "exactly two tensors."
            )

        input_ids, labels = batch

        loss = trainer.train_step(
            input_ids,
            labels,
        )

        print(
            f"Step "
            f"{trainer.global_step:>6} | "
            f"Loss "
            f"{loss:.6f} | "
            f"LR "
            f"{trainer.get_learning_rate():.8f}"
        )

        # ----------------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------------

        if (
            not no_save
            and trainer.global_step
            % CHECKPOINT_INTERVAL
            == 0
        ):

            checkpoint_name = (
                f"step_"
                f"{trainer.global_step:08d}"
                f".pt"
            )

            try:

                path = (
                    save_checkpoint_with_manifest(
                        trainer,
                        checkpoint_name,
                        manifest,
                    )
                )

                print()
                print(
                    "Checkpoint saved : "
                    f"{path}"
                )

            except Exception as exc:

                print()
                print(
                    "WARNING: checkpoint save failed."
                )

                print(
                    f"Reason : {exc}"
                )

                print(
                    "Training will continue."
                )

    # --------------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------------

    if not no_save:

        final_name = (
            f"final_step_"
            f"{trainer.global_step:08d}"
            f".pt"
        )

        path = (
            save_checkpoint_with_manifest(
                trainer,
                final_name,
                manifest,
            )
        )

        print()
        print(
            "Final checkpoint : "
            f"{path}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 75)
    print("FINAL TRAINING COMPLETED")
    print("=" * 75)

    print(
        f"Global Step     : "
        f"{trainer.global_step:,}"
    )

    print(
        f"Target Step     : "
        f"{target_steps:,}"
    )

    print(
        f"Dataset Passes  : "
        f"{dataset_pass}"
    )

    print(
        f"Final Loss      : "
        f"{trainer.current_train_loss:.6f}"
    )

    print(
        f"Elapsed Time    : "
        f"{elapsed / 3600:.2f} hours"
    )

    if trainer.global_step >= target_steps:

        print(
            "Status          : ✅ PASSED"
        )

    else:

        print(
            "Status          : ⚠️ INCOMPLETE"
        )

    if device.type == "cuda":

        print()

        print_gpu_memory(
            device
        )

    print("=" * 75)


# ======================================================================
# Full Epoch Training
# ======================================================================

def run_epoch_training(
    trainer,
    config: GPTConfig,
    no_save: bool,
) -> None:

    print()
    print("=" * 75)
    print("Starting Epoch-Based Training")
    print("=" * 75)

    trainer.train(
        save_every_epoch=(
            not no_save
        ),
        validate_every_epoch=False,
    )

    print()
    print("=" * 75)
    print("Epoch Training Completed")
    print("=" * 75)

    print(
        f"Global Step     : "
        f"{trainer.global_step:,}"
    )

    current_loss = getattr(
        trainer,
        "current_train_loss",
        None,
    )

    if current_loss is not None:

        print(
            f"Final Loss      : "
            f"{current_loss:.6f}"
        )

    print("=" * 75)


# ======================================================================
# Main
# ======================================================================

def main() -> None:

    args = parse_arguments()

    print()
    print("=" * 75)
    print("MyGPT2 FINAL TRAINING")
    print("=" * 75)

    print(
        f"Training Version : "
        f"{TRAINING_VERSION}"
    )

    print(
        f"Project Root     : "
        f"{PROJECT_ROOT}"
    )

    print("=" * 75)

    # ==================================================================
    # Device
    # ==================================================================

    device = get_device(
        args.device
    )

    print_device_information(
        device
    )

    # ==================================================================
    # Configuration
    # ==================================================================

    config = build_config(
        args,
        device,
    )

    # ==================================================================
    # Seed
    # ==================================================================

    set_seed(
        config.seed
    )

    print(
        f"Random Seed     : "
        f"{config.seed}"
    )

    # ==================================================================
    # Tokenizer
    # ==================================================================

    tokenizer = load_tokenizer()

    if (
        tokenizer.vocabulary_size
        != config.vocab_size
    ):

        raise RuntimeError(
            "Tokenizer vocabulary does not match "
            "GPT configuration.\n\n"
            f"Tokenizer : "
            f"{tokenizer.vocabulary_size}\n"
            f"Config    : "
            f"{config.vocab_size}"
        )

    # ==================================================================
    # Dataset Manifest
    # ==================================================================

    dataset_manifest = (
        build_dataset_manifest()
    )

    # ==================================================================
    # Dataset
    # ==================================================================

    print()
    print("=" * 75)
    print("Creating LOCKED Training Dataset")
    print("=" * 75)

    train_dataset = (
        create_locked_dataset(
            tokenizer=tokenizer,
            config=config,
            max_documents=args.max_documents,
        )
    )

    print(
        "Training Dataset : ✅ CREATED"
    )

    # ==================================================================
    # DataLoader
    # ==================================================================

    train_loader = (
        create_train_loader(
            dataset=train_dataset,
            config=config,
            num_workers=args.num_workers,
            device=device,
        )
    )

    batches_per_epoch = (
        print_loader_information(
            train_loader,
            config,
            args.max_documents,
        )
    )

    # ==================================================================
    # DataLoader Validation
    # ==================================================================

    validate_dataloader(
        train_loader,
        config,
    )

    # ==================================================================
    # Training Steps
    # ==================================================================

    total_steps = (
        calculate_total_steps(
            batches_per_epoch,
            config,
            args.max_steps,
        )
    )

    # ==================================================================
    # Manifest
    # ==================================================================

    manifest = (
        build_training_manifest(
            config=config,
            tokenizer=tokenizer,
            dataset_manifest=dataset_manifest,
            total_steps=total_steps,
        )
    )

    # ==================================================================
    # Training Plan
    # ==================================================================

    print_training_plan(
        config=config,
        batches_per_epoch=batches_per_epoch,
        total_steps=total_steps,
        max_steps=args.max_steps,
        max_documents=args.max_documents,
    )

    # ==================================================================
    # Model
    # ==================================================================

    print()
    print(
        "Creating model..."
    )

    model = MyGPTModel(
        config
    )

    model = model.to(
        device
    )

    print(
        "Model created successfully."
    )

    print_model_information(
        model,
        config,
    )

    # ==================================================================
    # Add parameter count to manifest
    # ==================================================================

    total_params, trainable_params = (
        count_parameters(model)
    )

    manifest[
        "model_parameters"
    ] = {
        "total": total_params,
        "trainable": trainable_params,
    }

    # ==================================================================
    # Optimizer
    # ==================================================================

    print()
    print(
        "Creating optimizer..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(
        "Optimizer       : ✅ CREATED"
    )

    # ==================================================================
    # Scheduler
    # ==================================================================

    scheduler = (
        create_training_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
        )
    )

    # ==================================================================
    # Trainer
    # ==================================================================

    print()
    print(
        "Creating Trainer..."
    )

    trainer = create_training_trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print(
        "Trainer         : ✅ CREATED"
    )

    # ==================================================================
    # Resume
    # ==================================================================

    resume_path = (
        resolve_resume_path(
            args.resume
        )
    )

    if resume_path is not None:

        print()
        print("=" * 75)
        print("RESUME REQUEST")
        print("=" * 75)

        print(
            f"Checkpoint : "
            f"{resume_path}"
        )

        # --------------------------------------------------------------
        # Inspect manifest BEFORE loading trainer state.
        # --------------------------------------------------------------

        checkpoint = (
            read_checkpoint(
                resume_path
            )
        )

        validate_resume_manifest(
            checkpoint=checkpoint,
            current_manifest=manifest,
            allow_legacy=(
                args.allow_legacy_resume
            ),
        )

        # --------------------------------------------------------------
        # Restore model / optimizer / scheduler / RNG.
        # --------------------------------------------------------------

        print()
        print(
            "Loading checkpoint state..."
        )

        trainer.load(
            path=resume_path,
            restore_rng=True,
        )

        print(
            f"Restored Epoch  : "
            f"{trainer.current_epoch}"
        )

        print(
            f"Restored Step   : "
            f"{trainer.global_step:,}"
        )

        print(
            "Checkpoint      : ✅ LOADED"
        )

        print("=" * 75)

        # --------------------------------------------------------------
        # Never allow a target below the restored step.
        # --------------------------------------------------------------

        if (
            args.max_steps is not None
            and args.max_steps
            <= trainer.global_step
        ):

            raise RuntimeError(
                "\nINVALID RESUME TARGET.\n\n"
                f"Checkpoint step : "
                f"{trainer.global_step:,}\n"
                f"Requested target: "
                f"{args.max_steps:,}\n\n"
                "The requested target must be greater "
                "than the restored checkpoint step."
            )

    # ==================================================================
    # Step-Limited Training
    # ==================================================================

    if args.max_steps is not None:

        run_step_limited_training(
            trainer=trainer,
            train_loader=train_loader,
            target_steps=args.max_steps,
            no_save=args.no_save,
            manifest=manifest,
            device=device,
        )

        return

    # ==================================================================
    # Epoch Training
    # ==================================================================

    run_epoch_training(
        trainer=trainer,
        config=config,
        no_save=args.no_save,
    )


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":

    main()