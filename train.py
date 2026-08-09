"""
============================================================
MyGPT2 - Main Training Entry Point
============================================================

Project
-------
MyGPT2

Purpose
-------
Main entry point for training the MyGPT2 model.

Pipeline
--------

    Configuration
          ↓
    Tokenizer
          ↓
    Dataset / DataLoader
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

Usage
-----

Full training:

    python train.py

Short pipeline test:

    python train.py --max-steps 10

Short test with limited documents:

    python train.py --max-steps 10 --max-documents 100

Resume training:

    python train.py --resume artifacts/checkpoints/latest.pt

CPU training:

    python train.py --device cpu

Override batch size:

    python train.py --batch-size 4

Disable checkpoint saving:

    python train.py --no-save

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
# Third-Party
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

from training.dataloader import create_dataloader

from training.trainer import Trainer
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


# ============================================================
# Paths
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
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of optimizer steps. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the number of training epochs."
        ),
    )

    # --------------------------------------------------------
    # Batch Size
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
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help=(
            "Maximum number of documents per dataset. "
            "Useful for short tests."
        ),
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        help=(
            "Training device. Default: auto."
        ),
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
# Random Seed
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set all supported random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device Selection
# ============================================================

def get_device(
    requested_device: str | None,
) -> torch.device:
    """
    Select the training device.
    """

    if requested_device is None:
        requested_device = "auto"

    # --------------------------------------------------------
    # Explicit CUDA
    # --------------------------------------------------------

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was explicitly requested, "
                "but CUDA is not available."
            )

        return torch.device("cuda")

    # --------------------------------------------------------
    # Explicit CPU
    # --------------------------------------------------------

    if requested_device == "cpu":

        return torch.device("cpu")

    # --------------------------------------------------------
    # Automatic
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
    Print PyTorch and GPU information.
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


# ============================================================
# Configuration
# ============================================================

def build_config(
    args: argparse.Namespace,
) -> GPTConfig:
    """
    Build GPT configuration and apply
    command-line overrides.
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
    # Batch Size
    # --------------------------------------------------------

    if args.batch_size is not None:

        if args.batch_size <= 0:

            raise ValueError(
                "--batch-size must be greater than zero."
            )

        config.batch_size = args.batch_size

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
    Print model configuration.
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
# Tokenizer
# ============================================================

def load_tokenizer() -> MyGPTTokenizer:
    """
    Verify and load the trained tokenizer.
    """

    print()
    print("=" * 75)
    print("Tokenizer")
    print("=" * 75)

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    if not DEFAULT_TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer was not found.\n\n"
            f"Expected path:\n"
            f"{DEFAULT_TOKENIZER_PATH}\n\n"
            "Train the tokenizer before "
            "starting GPT training."
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = MyGPTTokenizer.load(
        DEFAULT_TOKENIZER_PATH
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocabulary_size:,}"
    )

    # --------------------------------------------------------
    # Verify vocabulary
    # --------------------------------------------------------

    if tokenizer.vocabulary_size != 32000:

        print(
            "WARNING: Tokenizer vocabulary size "
            f"is {tokenizer.vocabulary_size}, "
            "while GPTConfig expects 32000."
        )

    print("=" * 75)

    return tokenizer


# ============================================================
# DataLoader
# ============================================================

def create_train_loader(
    config: GPTConfig,
    tokenizer: MyGPTTokenizer,
    max_documents: int | None = None,
):
    """
    Create the project's real training DataLoader.

    This directly matches training/dataloader.py.
    """

    print()
    print(
        "Creating training DataLoader..."
    )

    loader = create_dataloader(
        tokenizer=tokenizer,

        sequence_length=(
            config.max_position_embeddings
        ),

        batch_size=config.batch_size,

        max_documents=max_documents,

        num_workers=0,

        pin_memory=torch.cuda.is_available(),

        drop_last=True,
    )

    print(
        "Training DataLoader "
        "created successfully."
    )

    return loader


# ============================================================
# DataLoader Information
# ============================================================

def get_loader_length(
    loader,
) -> int | None:
    """
    Safely obtain DataLoader length.
    """

    try:

        return len(loader)

    except TypeError:

        return None


def print_dataloader_information(
    loader,
    config: GPTConfig,
) -> int | None:
    """
    Print DataLoader configuration.
    """

    batches = (
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

    if batches is not None:

        print(
            f"Batches / Epoch : "
            f"{batches:,}"
        )

    else:

        print(
            "Batches / Epoch : Unknown"
        )

    print("=" * 75)

    return batches


# ============================================================
# Training Steps
# ============================================================

def calculate_training_steps(
    *,
    batches_per_epoch: int | None,
    epochs: int,
    max_steps: int | None,
) -> int:
    """
    Calculate total optimizer steps.

    --max-steps takes priority.
    """

    if max_steps is not None:

        if max_steps <= 0:

            raise ValueError(
                "--max-steps must be greater than zero."
            )

        return max_steps

    if batches_per_epoch is None:

        raise RuntimeError(
            "Unable to determine the number "
            "of batches per epoch."
        )

    if batches_per_epoch <= 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
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
    """

    warmup_steps = max(
        1,
        int(total_steps * 0.10),
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
    Resolve a checkpoint path.
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
# Max-Step Training Test
# ============================================================

def run_max_steps_test(
    trainer: Trainer,
    train_loader,
    max_steps: int,
    save_checkpoint: bool,
) -> None:
    """
    Run exactly max_steps optimizer updates.

    This is intentionally handled here instead of calling
    Trainer.train(), because the existing Trainer's normal
    training method operates at epoch level.
    """

    start_time = time.time()

    print()
    print("=" * 75)
    print("Starting Training")
    print("=" * 75)

    print(
        f"Device          : "
        f"{trainer.device}"
    )

    print(
        f"Max Steps       : "
        f"{max_steps:,}"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for batch in train_loader:

        if trainer.global_step >= max_steps:
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

        print(
            f"Step "
            f"{trainer.global_step:>6} | "
            f"Loss "
            f"{loss:.6f} | "
            f"LR "
            f"{trainer.get_learning_rate():.8f}"
        )

    # --------------------------------------------------------
    # Check exact step count
    # --------------------------------------------------------

    if trainer.global_step < max_steps:

        print()
        print(
            "WARNING: DataLoader ended before "
            f"{max_steps} steps."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save_checkpoint:

        checkpoint_path = trainer.save(
            "pipeline_test.pt"
        )

        print()
        print(
            f"Test checkpoint : "
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Summary
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

    print(
        f"Final Loss      : "
        f"{trainer.current_train_loss:.6f}"
    )

    print(
        f"Elapsed Time    : "
        f"{elapsed:.2f}s"
    )

    if trainer.global_step >= max_steps:

        print(
            "Status          : ✅ PASSED"
        )

    else:

        print(
            "Status          : ⚠️ INCOMPLETE"
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

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 75)
    print("MyGPT2 Training")
    print("=" * 75)

    print(
        f"Project Root    : "
        f"{PROJECT_ROOT}"
    )

    # ========================================================
    # Device
    # ========================================================

    device = get_device(
        args.device
    )

    print_device_information(
        device
    )

    # ========================================================
    # Configuration
    # ========================================================

    config = build_config(
        args
    )

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        config.seed
    )

    print(
        f"Random Seed     : "
        f"{config.seed}"
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = load_tokenizer()

    # ========================================================
    # Model
    # ========================================================

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

    model.train()

    print(
        "Model created successfully."
    )

    print_model_information(
        model,
        config,
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = create_train_loader(
        config=config,
        tokenizer=tokenizer,
        max_documents=args.max_documents,
    )

    batches_per_epoch = (
        print_dataloader_information(
            train_loader,
            config,
        )
    )

    # ========================================================
    # Training Steps
    # ========================================================

    total_steps = (
        calculate_training_steps(
            batches_per_epoch=batches_per_epoch,
            epochs=config.max_epochs,
            max_steps=args.max_steps,
        )
    )

    print()
    print("=" * 75)
    print("Training Plan")
    print("=" * 75)

    print(
        f"Epochs          : "
        f"{config.max_epochs}"
    )

    if batches_per_epoch is not None:

        print(
            f"Batches / Epoch : "
            f"{batches_per_epoch:,}"
        )

    print(
        f"Total Steps     : "
        f"{total_steps:,}"
    )

    if args.max_steps is not None:

        print(
            f"Test Mode       : "
            f"MAX {args.max_steps:,} STEPS"
        )

    else:

        print(
            "Test Mode       : NO"
        )

    if args.max_documents is not None:

        print(
            f"Max Documents   : "
            f"{args.max_documents:,}"
        )

    else:

        print(
            "Max Documents   : ALL"
        )

    print("=" * 75)

    # ========================================================
    # Optimizer
    # ========================================================

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

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
    )

    # ========================================================
    # Trainer
    # ========================================================

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

    # ========================================================
    # Resume
    # ========================================================

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
            "Checkpoint      : ✅ LOADED"
        )

        print("=" * 75)

    # ========================================================
    # Short Test Mode
    # ========================================================

    if args.max_steps is not None:

        run_max_steps_test(
            trainer=trainer,
            train_loader=train_loader,
            max_steps=args.max_steps,
            save_checkpoint=(
                not args.no_save
            ),
        )

        return

    # ========================================================
    # Full Training
    # ========================================================

    print()
    print("=" * 75)
    print("Starting Full Training")
    print("=" * 75)

    print(
        f"Device          : "
        f"{device}"
    )

    print(
        f"Model           : "
        f"{config.model_size}"
    )

    total_parameters = (
        count_parameters(model)[0]
    )

    print(
        f"Parameters      : "
        f"{total_parameters / 1_000_000:.2f}M"
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

    print(
        f"Total Steps     : "
        f"{total_steps:,}"
    )

    print("=" * 75)

    start_time = time.time()

    # --------------------------------------------------------
    # Important:
    #
    # No validation loader has been implemented yet.
    # Therefore validation is disabled at this stage.
    # --------------------------------------------------------

    trainer.train(
        save_every_epoch=(
            not args.no_save
        ),
        validate_every_epoch=False,
    )

    # ========================================================
    # Final Summary
    # ========================================================

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

    print(
        f"Tokens Seen     : "
        f"{trainer.total_tokens:,}"
    )

    print(
        f"Final Train Loss: "
        f"{trainer.current_train_loss:.6f}"
    )

    print("=" * 75)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()