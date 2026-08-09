"""
============================================================
MyGPT2 - Main Training Entry Point
============================================================

Project
-------
MyGPT2

Purpose
-------
Main entry point for training the MyGPT2 GPT model.

Pipeline
--------
Configuration
      ↓
Tokenizer
      ↓
Dataset
      ↓
DataLoader
      ↓
GPT Model
      ↓
AdamW Optimizer
      ↓
Learning Rate Scheduler
      ↓
Trainer
      ↓
Checkpoint Manager

Examples
--------
Full training:

    python train.py

Short pipeline test:

    python train.py --max-steps 10 --max-documents 10000

Longer training test:

    python train.py --max-steps 100 --max-documents 10000

Override batch size:

    python train.py --batch-size 4

CPU training:

    python train.py --device cpu

Resume training:

    python train.py --resume artifacts/checkpoints/latest.pt

Disable checkpoint saving:

    python train.py --no-save

Disable validation:

    python train.py --no-validation
============================================================
"""

from __future__ import annotations

# ============================================================
# Standard Library
# ============================================================

import argparse
import random
import sys
import time
from pathlib import Path


# ============================================================
# Third Party
# ============================================================

import numpy as np
import torch


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from model.config import GPTConfig
from model.model import MyGPTModel

from tokenizer.my_tokenizer import MyGPTTokenizer

from training.trainer import Trainer
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


# ============================================================
# Constants
# ============================================================

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

DEFAULT_TEST_CHECKPOINT = (
    DEFAULT_CHECKPOINT_DIR
    / "pipeline_test.pt"
)

DEFAULT_NUM_WORKERS = 0

DEFAULT_WARMUP_RATIO = 0.10


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train MyGPT2."
    )

    # --------------------------------------------------------
    # Training steps
    # --------------------------------------------------------

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum optimizer steps. "
            "Useful for testing."
        ),
    )

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the number of training epochs."
        ),
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override training batch size."
        ),
    )

    # --------------------------------------------------------
    # Maximum documents
    # --------------------------------------------------------

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help=(
            "Maximum number of documents to process. "
            "Useful for testing. Default: all documents."
        ),
    )

    # --------------------------------------------------------
    # DataLoader workers
    # --------------------------------------------------------

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of DataLoader workers. "
            "Use 0 on Windows for reliability."
        ),
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint to resume from."
        ),
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Disable checkpoint saving."
        ),
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help=(
            "Disable validation."
        ),
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # CUDA deterministic settings
    #
    # We deliberately keep these conservative because exact
    # deterministic CUDA execution can reduce performance.
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.backends.cudnn.benchmark = True


# ============================================================
# Device Selection
# ============================================================

def get_device(
    requested_device: str,
) -> torch.device:
    """
    Select the training device.
    """

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was explicitly requested, "
                "but CUDA is not available."
            )

        return torch.device("cuda")

    if requested_device == "cpu":

        return torch.device("cpu")

    # --------------------------------------------------------
    # Automatic selection
    # --------------------------------------------------------

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Device Information
# ============================================================

def print_device_information(
    device: torch.device,
) -> None:
    """
    Display device information.
    """

    print()
    print("=" * 75)
    print("Device Information")
    print("=" * 75)

    print(
        f"PyTorch Version : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA Available  : "
        f"{'YES' if torch.cuda.is_available() else 'NO'}"
    )

    print(
        f"Selected Device : "
        f"{device}"
    )

    if device.type == "cuda":

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        print(
            f"GPU             : "
            f"{gpu_name}"
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


# ============================================================
# Build Configuration
# ============================================================

def build_config(
    args: argparse.Namespace,
) -> GPTConfig:
    """
    Build GPT configuration and apply command-line overrides.
    """

    config = GPTConfig()

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    if args.epochs is not None:

        if args.epochs <= 0:

            raise ValueError(
                "--epochs must be greater than zero."
            )

        config.max_epochs = args.epochs

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    if args.batch_size is not None:

        if args.batch_size <= 0:

            raise ValueError(
                "--batch-size must be greater than zero."
            )

        config.batch_size = args.batch_size

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers cannot be negative."
        )

    # --------------------------------------------------------
    # Maximum documents
    # --------------------------------------------------------

    if args.max_documents is not None:

        if args.max_documents <= 0:

            raise ValueError(
                "--max-documents must be greater than zero."
            )

    return config


# ============================================================
# Parameter Counting
# ============================================================

def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

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


# ============================================================
# Model Information
# ============================================================

def print_model_information(
    model: torch.nn.Module,
    config: GPTConfig,
) -> None:
    """
    Print model configuration and parameter information.
    """

    total, trainable = (
        count_parameters(model)
    )

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


# ============================================================
# Tokenizer Verification
# ============================================================

def verify_tokenizer_path() -> Path:
    """
    Verify that the trained tokenizer exists.
    """

    print()
    print("=" * 75)
    print("Tokenizer")
    print("=" * 75)

    if not DEFAULT_TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer was not found.\n\n"
            f"Expected path:\n"
            f"{DEFAULT_TOKENIZER_PATH}\n\n"
            "Train the tokenizer before starting GPT training."
        )

    size_mb = (
        DEFAULT_TOKENIZER_PATH.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Tokenizer       : "
        f"{DEFAULT_TOKENIZER_PATH}"
    )

    print(
        f"Size            : "
        f"{size_mb:.2f} MB"
    )

    print(
        "Tokenizer       : ✅ FOUND"
    )

    return DEFAULT_TOKENIZER_PATH


# ============================================================
# Load Tokenizer
# ============================================================

def load_tokenizer() -> MyGPTTokenizer:
    """
    Load the project's trained tokenizer.
    """

    tokenizer_path = (
        verify_tokenizer_path()
    )

    print()
    print("Loading tokenizer...")

    tokenizer = (
        MyGPTTokenizer.load(
            tokenizer_path
        )
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocabulary_size:,}"
    )

    return tokenizer


# ============================================================
# DataLoader Factory Import
# ============================================================

def import_dataloader_factory():
    """
    Import create_dataloader from training.dataloader.
    """

    import training.dataloader as dataloader_module

    factory = getattr(
        dataloader_module,
        "create_dataloader",
        None,
    )

    if not callable(factory):

        raise ImportError(
            "training.dataloader.create_dataloader "
            "was not found."
        )

    return factory


# ============================================================
# Create Training DataLoader
# ============================================================

def create_train_loader(
    *,
    config: GPTConfig,
    tokenizer: MyGPTTokenizer,
    max_documents: int | None,
    num_workers: int,
):
    """
    Create the real training DataLoader.

    This function matches the actual DataLoader API:

        create_dataloader(
            tokenizer=...,
            sequence_length=...,
            batch_size=...,
            max_documents=...,
            num_workers=...,
            pin_memory=...,
            drop_last=...
        )
    """

    factory = (
        import_dataloader_factory()
    )

    print()
    print(
        "Creating training DataLoader..."
    )

    loader = factory(
        tokenizer=tokenizer,

        sequence_length=(
            config.max_position_embeddings
        ),

        batch_size=(
            config.batch_size
        ),

        max_documents=max_documents,

        num_workers=num_workers,

        pin_memory=(
            torch.cuda.is_available()
        ),

        drop_last=True,
    )

    if loader is None:

        raise RuntimeError(
            "DataLoader factory returned None."
        )

    print(
        "Training DataLoader "
        "created successfully."
    )

    return loader


# ============================================================
# DataLoader Length
# ============================================================

def get_loader_length(
    loader,
) -> int | None:
    """
    Safely determine the number of batches.
    """

    try:

        return len(loader)

    except TypeError:

        return None


# ============================================================
# DataLoader Statistics
# ============================================================

def get_dataset_statistics(
    loader,
) -> dict | None:
    """
    Obtain dataset statistics if supported.
    """

    dataset = getattr(
        loader,
        "dataset",
        None,
    )

    if dataset is None:

        return None

    statistics_function = getattr(
        dataset,
        "get_statistics",
        None,
    )

    if callable(statistics_function):

        try:

            return statistics_function()

        except Exception:

            return None

    return None


# ============================================================
# Print DataLoader Information
# ============================================================

def print_dataloader_information(
    loader,
    config: GPTConfig,
    max_documents: int | None,
) -> int | None:
    """
    Display DataLoader and dataset information.
    """

    num_batches = (
        get_loader_length(loader)
    )

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
        f"{getattr(loader, 'num_workers', 'Unknown')}"
    )

    if num_batches is not None:

        print(
            f"Batches / Epoch : "
            f"{num_batches:,}"
        )

    else:

        print(
            "Batches / Epoch : "
            "Unknown"
        )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    statistics = (
        get_dataset_statistics(loader)
    )

    if statistics:

        print()
        print(
            "Dataset Statistics"
        )

        print("-" * 75)

        for key, value in statistics.items():

            print(
                f"{key:25}: {value}"
            )

    print("=" * 75)

    return num_batches


# ============================================================
# Validate DataLoader
# ============================================================

def validate_dataloader(
    loader,
    config: GPTConfig,
) -> None:
    """
    Verify that the DataLoader contains usable batches.
    """

    num_batches = (
        get_loader_length(loader)
    )

    if num_batches is not None:

        if num_batches <= 0:

            statistics = (
                get_dataset_statistics(loader)
            )

            message = (
                "\nTraining DataLoader contains "
                "ZERO usable batches.\n\n"
                f"Sequence Length : "
                f"{config.max_position_embeddings}\n"
                f"Batch Size      : "
                f"{config.batch_size}\n"
            )

            if statistics:

                message += (
                    "\nDataset Statistics:\n"
                    f"{statistics}\n"
                )

            message += (
                "\nIncrease --max-documents or "
                "reduce the test dataset size."
            )

            raise RuntimeError(
                message
            )

    # --------------------------------------------------------
    # Validate one batch.
    # --------------------------------------------------------

    try:

        batch = next(iter(loader))

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
            "Training DataLoader must return "
            "(input_ids, labels)."
        )

    if len(batch) != 2:

        raise RuntimeError(
            "Training DataLoader batch must "
            "contain exactly two tensors."
        )

    input_ids, labels = batch

    if not isinstance(
        input_ids,
        torch.Tensor,
    ):

        raise RuntimeError(
            "input_ids must be a torch.Tensor."
        )

    if not isinstance(
        labels,
        torch.Tensor,
    ):

        raise RuntimeError(
            "labels must be a torch.Tensor."
        )

    expected_shape = (
        config.batch_size,
        config.max_position_embeddings,
    )

    actual_input_shape = (
        tuple(input_ids.shape)
    )

    actual_label_shape = (
        tuple(labels.shape)
    )

    # --------------------------------------------------------
    # Because drop_last=True is used, a complete batch is
    # expected.
    # --------------------------------------------------------

    if actual_input_shape != expected_shape:

        raise RuntimeError(
            "Unexpected input batch shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_input_shape}"
        )

    if actual_label_shape != expected_shape:

        raise RuntimeError(
            "Unexpected label batch shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_label_shape}"
        )

    # --------------------------------------------------------
    # Token dtype
    # --------------------------------------------------------

    if input_ids.dtype != torch.long:

        raise RuntimeError(
            "input_ids must use torch.int64 / torch.long."
        )

    if labels.dtype != torch.long:

        raise RuntimeError(
            "labels must use torch.int64 / torch.long."
        )

    # --------------------------------------------------------
    # Token range
    # --------------------------------------------------------

    if input_ids.numel() > 0:

        minimum = (
            int(input_ids.min().item())
        )

        maximum = (
            int(input_ids.max().item())
        )

        if minimum < 0:

            raise RuntimeError(
                "Input token IDs contain "
                "negative values."
            )

        if maximum >= config.vocab_size:

            raise RuntimeError(
                "Input token ID exceeds "
                "configured vocabulary size.\n"
                f"Maximum token: {maximum}\n"
                f"Vocabulary:    {config.vocab_size}"
            )

    # --------------------------------------------------------
    # Token shift verification
    # --------------------------------------------------------

    if input_ids.shape[1] >= 2:

        shifted_inputs = (
            input_ids[0, 1:]
        )

        shifted_targets = (
            labels[0, :-1]
        )

        if not torch.equal(
            shifted_inputs,
            shifted_targets,
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
        f"{actual_input_shape}"
    )

    print(
        f"Target Shape          : "
        f"{actual_label_shape}"
    )

    print(
        f"Input DType           : "
        f"{input_ids.dtype}"
    )

    print(
        f"Target DType          : "
        f"{labels.dtype}"
    )


# ============================================================
# Calculate Training Steps
# ============================================================

def calculate_training_steps(
    *,
    batches_per_epoch: int | None,
    epochs: int,
    max_steps: int | None,
) -> int:
    """
    Calculate total optimizer steps.

    For a test run:

        --max-steps

    takes priority.

    For full training:

        batches_per_epoch × epochs

    is used.
    """

    if max_steps is not None:

        if max_steps <= 0:

            raise ValueError(
                "--max-steps must be greater than zero."
            )

        return max_steps

    if batches_per_epoch is None:

        raise RuntimeError(
            "Unable to determine batches per epoch "
            "for full training."
        )

    if batches_per_epoch <= 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    if epochs <= 0:

        raise ValueError(
            "Training epochs must be greater than zero."
        )

    return (
        batches_per_epoch
        * epochs
    )


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    total_steps: int,
):
    """
    Create the project's scheduler.

    Uses 10% warmup.
    """

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

    # --------------------------------------------------------
    # Avoid warmup being equal to the entire schedule.
    # --------------------------------------------------------

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


# ============================================================
# Resume Path
# ============================================================

def resolve_resume_path(
    resume_argument: str | None,
) -> Path | None:
    """
    Resolve a resume checkpoint path.
    """

    if resume_argument is None:

        return None

    path = Path(
        resume_argument
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    path = path.resolve()

    if not path.exists():

        raise FileNotFoundError(
            "Resume checkpoint does not exist:\n"
            f"{path}"
        )

    return path


# ============================================================
# Checkpoint Directory
# ============================================================

def prepare_checkpoint_directory() -> None:
    """
    Create checkpoint directory if needed.
    """

    DEFAULT_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# GPU Memory Information
# ============================================================

def print_gpu_memory(
    device: torch.device,
) -> None:
    """
    Print current GPU memory usage.
    """

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


# ============================================================
# Print Training Plan
# ============================================================

def print_training_plan(
    *,
    config: GPTConfig,
    batches_per_epoch: int | None,
    total_steps: int,
    max_steps: int | None,
    max_documents: int | None,
) -> None:
    """
    Print the complete training plan.
    """

    print()
    print("=" * 75)
    print("Training Plan")
    print("=" * 75)

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

    if batches_per_epoch is not None:

        print(
            f"Batches / Epoch : "
            f"{batches_per_epoch:,}"
        )

    else:

        print(
            "Batches / Epoch : "
            "Unknown"
        )

    print(
        f"Total Steps     : "
        f"{total_steps:,}"
    )

    print(
        f"Max Documents   : "
        f"{max_documents if max_documents is not None else 'ALL'}"
    )

    if max_steps is not None:

        print(
            f"Test Mode       : "
            f"MAX {max_steps:,} STEPS"
        )

    else:

        print(
            "Test Mode       : NO"
        )

    print("=" * 75)


# ============================================================
# Create Trainer
# ============================================================

def create_training_trainer(
    *,
    model,
    train_loader,
    config,
    optimizer,
    scheduler,
    device,
):
    """
    Construct the existing Trainer.
    """

    print()
    print(
        "Creating Trainer..."
    )

    trainer = Trainer(
        model=model,

        train_loader=train_loader,

        config=config,

        val_loader=None,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,
    )

    print(
        "Trainer created successfully."
    )

    return trainer


# ============================================================
# Run Short Test
# ============================================================

def run_short_test(
    *,
    trainer,
    train_loader,
    max_steps: int,
    no_save: bool,
    device: torch.device,
) -> None:
    """
    Run an exact optimizer-step test.

    This deliberately uses the existing Trainer.train_step()
    so the already-tested Trainer remains the owner of:

        forward
        loss
        backward
        optimizer.step
        scheduler.step
    """

    print()
    print("=" * 75)
    print("Starting Training")
    print("=" * 75)

    print(
        f"Device          : "
        f"{device}"
    )

    print(
        f"Max Steps       : "
        f"{max_steps}"
    )

    print("=" * 75)

    start_time = time.time()

    completed_steps = 0

    data_iterator = iter(
        train_loader
    )

    while (
        completed_steps < max_steps
    ):

        try:

            batch = next(
                data_iterator
            )

        except StopIteration:

            print()
            print(
                "WARNING: DataLoader ended "
                f"before {max_steps} steps."
            )

            break

        if not isinstance(
            batch,
            (tuple, list),
        ):

            raise RuntimeError(
                "Training DataLoader must "
                "return (input_ids, labels)."
            )

        if len(batch) != 2:

            raise RuntimeError(
                "Training DataLoader batch "
                "must contain exactly two tensors."
            )

        input_ids, labels = batch

        loss = trainer.train_step(
            input_ids,
            labels,
        )

        completed_steps += 1

        print(
            f"Step "
            f"{trainer.global_step:>6} | "
            f"Loss "
            f"{loss:.6f} | "
            f"LR "
            f"{trainer.get_learning_rate():.8f}"
        )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = None

    if not no_save:

        prepare_checkpoint_directory()

        checkpoint_path = (
            trainer.save(
                "pipeline_test.pt"
            )
        )

        print()
        print(
            f"Test checkpoint : "
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 75)
    print("Pipeline Test Completed")
    print("=" * 75)

    print(
        f"Steps           : "
        f"{trainer.global_step}"
    )

    current_loss = getattr(
        trainer,
        "current_train_loss",
        float("inf"),
    )

    print(
        f"Final Loss      : "
        f"{current_loss:.6f}"
    )

    print(
        f"Elapsed Time    : "
        f"{elapsed:.2f}s"
    )

    if (
        trainer.global_step
        >= max_steps
    ):

        print(
            "Status          : "
            "✅ PASSED"
        )

    else:

        print(
            "Status          : "
            "⚠️ INCOMPLETE"
        )

    if device.type == "cuda":

        print()

        print_gpu_memory(
            device
        )

    print("=" * 75)


# ============================================================
# Run Full Training
# ============================================================

def run_full_training(
    *,
    trainer,
    config: GPTConfig,
    no_save: bool,
    no_validation: bool,
) -> None:
    """
    Run full training through the existing Trainer.

    The Trainer remains responsible for the actual epoch
    training logic.
    """

    print()
    print("=" * 75)
    print("Starting Full Training")
    print("=" * 75)

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
        f"Learning Rate   : "
        f"{config.learning_rate}"
    )

    print("=" * 75)

    start_time = time.time()

    trainer.train(
        save_every_epoch=(
            not no_save
        ),

        validate_every_epoch=(
            not no_validation
        ),
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 75)
    print("Training Process Finished")
    print("=" * 75)

    print(
        f"Total Time      : "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Global Steps    : "
        f"{trainer.global_step:,}"
    )

    total_tokens = getattr(
        trainer,
        "total_tokens",
        0,
    )

    print(
        f"Tokens Seen     : "
        f"{total_tokens:,}"
    )

    current_loss = getattr(
        trainer,
        "current_train_loss",
        None,
    )

    if current_loss is not None:

        print(
            f"Final Train Loss: "
            f"{current_loss:.6f}"
        )

    print("=" * 75)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Main MyGPT2 training entry point.
    """

    args = parse_arguments()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("MyGPT2 Training")
    print("=" * 75)

    print(
        f"Project Root    : "
        f"{PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device(
        args.device
    )

    print_device_information(
        device
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = build_config(
        args
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        config.seed
    )

    print(
        f"Random Seed     : "
        f"{config.seed}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Vocabulary verification
    # --------------------------------------------------------

    if (
        tokenizer.vocabulary_size
        != config.vocab_size
    ):

        raise RuntimeError(
            "Tokenizer vocabulary size does not "
            "match GPT configuration.\n\n"
            f"Tokenizer: "
            f"{tokenizer.vocabulary_size}\n"
            f"Config:    "
            f"{config.vocab_size}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = (
        create_train_loader(
            config=config,
            tokenizer=tokenizer,
            max_documents=args.max_documents,
            num_workers=args.num_workers,
        )
    )

    batches_per_epoch = (
        print_dataloader_information(
            train_loader,
            config,
            args.max_documents,
        )
    )

    # --------------------------------------------------------
    # Validate DataLoader
    # --------------------------------------------------------

    validate_dataloader(
        train_loader,
        config,
    )

    # --------------------------------------------------------
    # Calculate training steps
    # --------------------------------------------------------

    total_steps = (
        calculate_training_steps(
            batches_per_epoch=(
                batches_per_epoch
            ),

            epochs=(
                config.max_epochs
            ),

            max_steps=(
                args.max_steps
            ),
        )
    )

    print_training_plan(
        config=config,
        batches_per_epoch=(
            batches_per_epoch
        ),
        total_steps=(
            total_steps
        ),
        max_steps=(
            args.max_steps
        ),
        max_documents=(
            args.max_documents
        ),
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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
        "Optimizer created successfully."
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = create_training_trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    resume_path = (
        resolve_resume_path(
            args.resume
        )
    )

    if resume_path is not None:

        print()
        print("=" * 75)
        print("Resuming Training")
        print("=" * 75)

        print(
            f"Checkpoint      : "
            f"{resume_path}"
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
            f"{trainer.global_step}"
        )

        print(
            "Checkpoint      : "
            "✅ LOADED"
        )

        print("=" * 75)

    # --------------------------------------------------------
    # Short test mode
    # --------------------------------------------------------

    if args.max_steps is not None:

        run_short_test(
            trainer=trainer,

            train_loader=train_loader,

            max_steps=args.max_steps,

            no_save=args.no_save,

            device=device,
        )

        return

    # --------------------------------------------------------
    # Full training
    # --------------------------------------------------------

    run_full_training(
        trainer=trainer,

        config=config,

        no_save=args.no_save,

        no_validation=args.no_validation,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()