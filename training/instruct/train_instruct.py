from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (
    DataLoader,
    Subset,
)


# =============================================================
# Project import path
# =============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from model.config import GPTConfig
from model.model import MyGPTModel

from tokenizer.my_tokenizer import MyGPTTokenizer

from training.instruct.collator import (
    InstructionCollator,
)

from training.instruct.config import (
    InstructionTrainingConfig,
)

from training.instruct.dataset import (
    IGNORE_INDEX,
    InstructionDataset,
)


# =============================================================
# Utilities
# =============================================================


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng_state() -> dict[str, Any]:

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict[str, Any] | None,
) -> None:

    if not state:
        return

    if "python" in state:
        random.setstate(
            state["python"]
        )

    if "numpy" in state:
        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:
        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def resolve_device(
    requested_device: str,
) -> torch.device:

    if (
        requested_device == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "WARNING: CUDA requested but unavailable. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    return torch.device(
        requested_device
    )


# =============================================================
# Model-output compatibility helper
# =============================================================


def extract_logits(
    model_output: Any,
) -> torch.Tensor:
    """
    Handle several common custom-model output styles.

    Supported:

        Tensor
        tuple/list with Tensor first
        dict containing "logits"
        object with .logits
    """

    if isinstance(
        model_output,
        torch.Tensor,
    ):
        return model_output

    if isinstance(
        model_output,
        (tuple, list),
    ):
        if (
            model_output
            and isinstance(
                model_output[0],
                torch.Tensor,
            )
        ):
            return model_output[0]

    if isinstance(
        model_output,
        dict,
    ):

        logits = model_output.get(
            "logits"
        )

        if isinstance(
            logits,
            torch.Tensor,
        ):
            return logits

    logits = getattr(
        model_output,
        "logits",
        None,
    )

    if isinstance(
        logits,
        torch.Tensor,
    ):
        return logits

    raise TypeError(
        "Unable to extract logits from MyGPTModel output. "
        f"Received type: {type(model_output)}"
    )


# =============================================================
# Checkpoint config helpers
# =============================================================


def config_to_dict(
    config: Any,
) -> dict[str, Any]:

    if isinstance(
        config,
        dict,
    ):
        return dict(config)

    if hasattr(
        config,
        "to_dict",
    ):
        return config.to_dict()

    if hasattr(
        config,
        "__dict__",
    ):
        return dict(
            vars(config)
        )

    raise TypeError(
        "Unable to convert model config to dictionary."
    )


def build_model_config(
    checkpoint: dict[str, Any],
) -> GPTConfig:

    checkpoint_config = checkpoint.get(
        "config"
    )

    if checkpoint_config is None:
        raise KeyError(
            "Base checkpoint does not contain a "
            "'config' field."
        )

    if isinstance(
        checkpoint_config,
        GPTConfig,
    ):
        return checkpoint_config

    config_dict = config_to_dict(
        checkpoint_config
    )

    try:
        return GPTConfig(
            **config_dict
        )

    except TypeError as error:

        raise RuntimeError(
            "Could not rebuild GPTConfig from the "
            "checkpoint configuration.\n"
            f"Config keys: {sorted(config_dict.keys())}"
        ) from error


# =============================================================
# Base model loading
# =============================================================


def load_base_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    MyGPTModel,
    GPTConfig,
    dict[str, Any],
]:

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Base checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print("=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)

    print()
    print(
        f"Checkpoint : {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model_config = build_model_config(
        checkpoint
    )

    model = MyGPTModel(
        model_config
    )

    load_result = model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading unexpectedly "
            "reported key differences."
        )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Parameters : "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable  : "
        f"{trainable_count:,}"
    )

    print(
        "Checkpoint load: STRICT PASS"
    )

    return (
        model,
        model_config,
        checkpoint,
    )


# =============================================================
# Loss
# =============================================================


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[
    torch.Tensor,
    int,
]:
    """
    Standard causal next-token loss.

    Important:

        logits[:, :-1]
        predicts
        labels[:, 1:]

    Labels are -100 outside assistant responses, therefore
    user/system/padding tokens contribute zero loss.
    """

    if logits.ndim != 3:

        raise RuntimeError(
            "Expected logits with shape "
            "[batch, sequence, vocabulary]. "
            f"Received: {tuple(logits.shape)}"
        )

    if labels.ndim != 2:

        raise RuntimeError(
            "Expected labels with shape "
            "[batch, sequence]. "
            f"Received: {tuple(labels.shape)}"
        )

    if (
        logits.shape[0]
        != labels.shape[0]
        or logits.shape[1]
        != labels.shape[1]
    ):

        raise RuntimeError(
            "Logits/labels batch or sequence "
            "dimensions do not match.\n"
            f"logits={tuple(logits.shape)} "
            f"labels={tuple(labels.shape)}"
        )

    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[:, 1:]
        .contiguous()
    )

    supervised_tokens = int(
        (
            shift_labels
            != ignore_index
        )
        .sum()
        .item()
    )

    if supervised_tokens == 0:

        raise RuntimeError(
            "Batch contains zero supervised "
            "assistant tokens."
        )

    loss = F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return (
        loss,
        supervised_tokens,
    )


# =============================================================
# Learning-rate schedule
# =============================================================


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be > 0."
        )

    warmup_steps = max(
        1,
        warmup_steps,
    )

    def lr_lambda(
        current_step: int,
    ) -> float:

        # Warmup
        if current_step < warmup_steps:

            return max(
                1e-8,
                float(current_step + 1)
                / float(warmup_steps),
            )

        # Cosine decay
        progress = (
            float(
                current_step
                - warmup_steps
            )
            /
            float(
                max(
                    1,
                    total_steps
                    - warmup_steps,
                )
            )
        )

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        cosine = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

        return (
            min_lr_ratio
            +
            (
                1.0
                - min_lr_ratio
            )
            * cosine
        )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


# =============================================================
# Data
# =============================================================


def create_datasets(
    config: InstructionTrainingConfig,
    tokenizer: MyGPTTokenizer,
    debug: bool,
):

    print()
    print("=" * 80)
    print("LOADING SFT DATASETS")
    print("=" * 80)

    train_dataset = InstructionDataset(
        file_path=config.train_dataset,
        tokenizer=tokenizer,
        sequence_length=(
            config.sequence_length
        ),
    )

    validation_dataset = (
        InstructionDataset(
            file_path=(
                config.validation_dataset
            ),
            tokenizer=tokenizer,
            sequence_length=(
                config.sequence_length
            ),
        )
    )

    print()
    print(
        f"Train exchanges      : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation exchanges : "
        f"{len(validation_dataset):,}"
    )

    if debug:

        train_count = min(
            config.debug_train_samples,
            len(train_dataset),
        )

        validation_count = min(
            config.debug_validation_samples,
            len(validation_dataset),
        )

        train_dataset = Subset(
            train_dataset,
            range(train_count),
        )

        validation_dataset = Subset(
            validation_dataset,
            range(validation_count),
        )

        print()
        print("DEBUG MODE DATASET LIMITS")

        print(
            f"  Train      : "
            f"{train_count:,}"
        )

        print(
            f"  Validation : "
            f"{validation_count:,}"
        )

    return (
        train_dataset,
        validation_dataset,
    )


def create_train_loader(
    dataset,
    collator,
    batch_size: int,
    seed: int,
    epoch: int,
    num_workers: int,
) -> DataLoader:
    """
    Epoch-specific generator makes shuffle ordering deterministic.

    This allows safe resume by recreating the same epoch ordering
    and skipping already-completed batches.
    """

    generator = torch.Generator()

    generator.manual_seed(
        seed + epoch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def create_validation_loader(
    dataset,
    collator,
    batch_size: int,
    num_workers: int,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# =============================================================
# Validation
# =============================================================


@torch.no_grad()
def validate(
    model: MyGPTModel,
    validation_loader: DataLoader,
    device: torch.device,
    ignore_index: int,
    max_batches: int | None,
) -> dict[str, float]:

    model.eval()

    total_loss_times_tokens = 0.0
    total_tokens = 0
    total_batches = 0

    start_time = time.time()

    for batch_index, batch in enumerate(
        validation_loader
    ):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=True,
        )

        output = model(
            input_ids=input_ids
        )

        logits = extract_logits(
            output
        )

        loss, supervised_tokens = (
            compute_sft_loss(
                logits=logits,
                labels=labels,
                ignore_index=ignore_index,
            )
        )

        total_loss_times_tokens += (
            float(loss.item())
            * supervised_tokens
        )

        total_tokens += (
            supervised_tokens
        )

        total_batches += 1

    if total_tokens == 0:

        raise RuntimeError(
            "Validation produced zero "
            "supervised tokens."
        )

    average_loss = (
        total_loss_times_tokens
        /
        total_tokens
    )

    perplexity = math.exp(
        min(
            average_loss,
            20.0,
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    model.train()

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "batches": total_batches,
        "seconds": elapsed,
    }


# =============================================================
# Checkpoint saving
# =============================================================


def save_checkpoint(
    path: Path,
    model: MyGPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    training_config: InstructionTrainingConfig,
    model_config: GPTConfig,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    best_validation_loss: float,
    last_train_loss: float | None,
    last_validation_loss: float | None,
    debug: bool,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "checkpoint_type": (
            "mygpt2_instruction_sft"
        ),
        "checkpoint_version": "1.0",
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "config": config_to_dict(
            model_config
        ),
        "instruction_training_config": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value
            in asdict(
                training_config
            ).items()
        },
        "base_checkpoint": str(
            training_config.base_checkpoint
        ),
        "global_step": int(
            global_step
        ),
        "epoch": int(
            epoch
        ),
        "batch_in_epoch": int(
            batch_in_epoch
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "train_loss": (
            None
            if last_train_loss is None
            else float(last_train_loss)
        ),
        "validation_loss": (
            None
            if last_validation_loss is None
            else float(last_validation_loss)
        ),
        "rng_state": save_rng_state(),
        "debug": bool(
            debug
        ),
    }

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


# =============================================================
# Resume
# =============================================================


def load_resume_checkpoint(
    resume_path: Path,
    model: MyGPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> dict[str, Any]:

    if not resume_path.exists():

        raise FileNotFoundError(
            f"Resume checkpoint not found:\n"
            f"{resume_path}"
        )

    print()
    print("=" * 80)
    print("RESUMING SFT TRAINING")
    print("=" * 80)

    print()
    print(
        f"Resume checkpoint : {resume_path}"
    )

    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get(
        "checkpoint_type"
    ) != "mygpt2_instruction_sft":

        raise RuntimeError(
            "The requested resume file is not "
            "an SFT checkpoint."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    # Move optimizer tensors to current device.
    for optimizer_state in (
        optimizer.state.values()
    ):

        for key, value in (
            optimizer_state.items()
        ):

            if isinstance(
                value,
                torch.Tensor,
            ):

                optimizer_state[
                    key
                ] = value.to(
                    device
                )

    restore_rng_state(
        checkpoint.get(
            "rng_state"
        )
    )

    print(
        f"Restored global step : "
        f"{checkpoint.get('global_step', 0):,}"
    )

    print(
        f"Restored epoch       : "
        f"{checkpoint.get('epoch', 0):,}"
    )

    print(
        f"Restored batch       : "
        f"{checkpoint.get('batch_in_epoch', 0):,}"
    )

    print(
        f"Restored LR          : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    return checkpoint


# =============================================================
# Training
# =============================================================


def train(
    config: InstructionTrainingConfig,
    debug: bool,
    resume_path: Path | None,
    max_steps_override: int | None,
) -> None:

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    set_seed(
        config.seed
    )

    device = resolve_device(
        config.device
    )

    config.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("MYGPT2 SUPERVISED INSTRUCTION FINE-TUNING")
    print("=" * 80)

    print()
    print(
        f"Mode                 : "
        f"{'DEBUG' if debug else 'FULL SFT'}"
    )

    print(
        f"Device               : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU                  : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Base checkpoint      : "
        f"{config.base_checkpoint}"
    )

    print(
        f"Context length       : "
        f"{config.sequence_length}"
    )

    print(
        f"Batch size           : "
        f"{config.batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{config.gradient_accumulation_steps}"
    )

    print(
        f"Effective batch size : "
        f"{config.batch_size * config.gradient_accumulation_steps}"
    )

    print(
        f"Learning rate        : "
        f"{config.learning_rate:.8f}"
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = MyGPTTokenizer.load(
        config.tokenizer_path
    )

    print(
        f"Tokenizer vocabulary : "
        f"{tokenizer.vocabulary_size:,}"
    )

    # ---------------------------------------------------------
    # Base model
    # ---------------------------------------------------------

    (
        model,
        model_config,
        base_checkpoint,
    ) = load_base_model(
        checkpoint_path=(
            config.base_checkpoint
        ),
        device=device,
    )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
    ) = create_datasets(
        config=config,
        tokenizer=tokenizer,
        debug=debug,
    )

    collator = InstructionCollator(
        pad_token_id=config.pad_token_id,
        ignore_index=config.ignore_index,
        max_length=config.sequence_length,
    )

    validation_loader = (
        create_validation_loader(
            dataset=validation_dataset,
            collator=collator,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    )

    # ---------------------------------------------------------
    # Training length
    # ---------------------------------------------------------

    epochs = (
        config.debug_epochs
        if debug
        else config.epochs
    )

    batches_per_epoch = math.ceil(
        len(train_dataset)
        /
        config.batch_size
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        /
        config.gradient_accumulation_steps
    )

    calculated_total_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    if debug:

        max_steps = (
            config.debug_max_steps
        )

    else:

        max_steps = (
            calculated_total_steps
        )

    if max_steps_override is not None:

        max_steps = min(
            max_steps,
            max_steps_override,
        )

    total_scheduler_steps = max(
        1,
        max_steps,
    )

    warmup_steps = max(
        1,
        int(
            total_scheduler_steps
            * config.warmup_ratio
        ),
    )

    print()
    print("=" * 80)
    print("TRAINING PLAN")
    print("=" * 80)

    print()
    print(
        f"Epochs               : "
        f"{epochs}"
    )

    print(
        f"Batches / epoch      : "
        f"{batches_per_epoch:,}"
    )

    print(
        f"Optimizer steps/epoch: "
        f"{optimizer_steps_per_epoch:,}"
    )

    print(
        f"Maximum steps        : "
        f"{max_steps:,}"
    )

    print(
        f"Warmup steps         : "
        f"{warmup_steps:,}"
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    min_lr_ratio = (
        config.min_learning_rate
        /
        config.learning_rate
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=(
            total_scheduler_steps
        ),
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # ---------------------------------------------------------
    # Resume state
    # ---------------------------------------------------------

    global_step = 0

    start_epoch = 0

    resume_batch_in_epoch = 0

    best_validation_loss = float(
        "inf"
    )

    last_train_loss = None

    last_validation_loss = None

    if resume_path is not None:

        resume_checkpoint = (
            load_resume_checkpoint(
                resume_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
        )

        global_step = int(
            resume_checkpoint.get(
                "global_step",
                0,
            )
        )

        start_epoch = int(
            resume_checkpoint.get(
                "epoch",
                0,
            )
        )

        resume_batch_in_epoch = int(
            resume_checkpoint.get(
                "batch_in_epoch",
                0,
            )
        )

        best_validation_loss = float(
            resume_checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        last_train_loss = (
            resume_checkpoint.get(
                "train_loss"
            )
        )

        last_validation_loss = (
            resume_checkpoint.get(
                "validation_loss"
            )
        )

    # ---------------------------------------------------------
    # Initial validation
    # ---------------------------------------------------------

    if global_step == 0:

        print()
        print("=" * 80)
        print("INITIAL VALIDATION")
        print("=" * 80)

        initial_validation = validate(
            model=model,
            validation_loader=(
                validation_loader
            ),
            device=device,
            ignore_index=(
                config.ignore_index
            ),
            max_batches=(
                config.max_validation_batches
            ),
        )

        print()
        print(
            f"Initial validation loss : "
            f"{initial_validation['loss']:.6f}"
        )

        print(
            f"Initial validation PPL  : "
            f"{initial_validation['perplexity']:.4f}"
        )

        last_validation_loss = (
            initial_validation[
                "loss"
            ]
        )

        best_validation_loss = min(
            best_validation_loss,
            initial_validation["loss"],
        )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    training_start_time = (
        time.time()
    )

    running_loss = 0.0
    running_tokens = 0
    running_micro_batches = 0
    accumulation_counter = 0

    stop_training = False

    for epoch in range(
        start_epoch,
        epochs,
    ):

        train_loader = create_train_loader(
            dataset=train_dataset,
            collator=collator,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
            num_workers=config.num_workers,
        )

        # Only the first resumed epoch needs skipping.
        batch_to_resume_from = (
            resume_batch_in_epoch
            if epoch == start_epoch
            else 0
        )

        print()
        print(
            f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_index, batch in enumerate(
            train_loader
        ):

            if (
                batch_index
                < batch_to_resume_from
            ):
                continue

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ].to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            output = model(
                input_ids=input_ids
            )

            logits = extract_logits(
                output
            )

            loss, supervised_tokens = (
                compute_sft_loss(
                    logits=logits,
                    labels=labels,
                    ignore_index=(
                        config.ignore_index
                    ),
                )
            )

            unscaled_loss = float(
                loss.item()
            )

            # -------------------------------------------------
            # Gradient accumulation
            # -------------------------------------------------

            scaled_loss = (
                loss
                /
                config.gradient_accumulation_steps
            )

            scaled_loss.backward()

            running_loss += (
                unscaled_loss
                * supervised_tokens
            )

            running_tokens += (
                supervised_tokens
            )

            running_micro_batches += 1
            accumulation_counter += 1

            is_accumulation_boundary = (
                accumulation_counter
                >= config.gradient_accumulation_steps
            )

            is_last_batch = (
                batch_index
                == len(train_loader) - 1
            )

            if (
                not is_accumulation_boundary
                and not is_last_batch
            ):
                continue

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        config.max_grad_norm
                    ),
                )
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )
            accumulation_counter = 0

            global_step += 1

            last_train_loss = (
                running_loss
                /
                max(
                    1,
                    running_tokens,
                )
            )

            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if (
                global_step == 1
                or global_step
                % config.log_every
                == 0
            ):

                elapsed = (
                    time.time()
                    - training_start_time
                )

                tokens_per_second = (
                    running_tokens
                    /
                    max(
                        elapsed,
                        1e-8,
                    )
                )

                print(
                    f"Step {global_step:06d} | "
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_index + 1:,}"
                    f"/{len(train_loader):,} | "
                    f"Loss {last_train_loss:.6f} | "
                    f"LR {current_lr:.8f} | "
                    f"Grad {float(grad_norm):.4f} | "
                    f"Tok/s {tokens_per_second:.1f}"
                )

                # reset logging window
                training_start_time = (
                    time.time()
                )

                running_loss = 0.0
                running_tokens = 0
                running_micro_batches = 0

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            should_validate = (
                global_step
                % config.validate_every
                == 0
            )

            if should_validate:

                validation_result = validate(
                    model=model,
                    validation_loader=(
                        validation_loader
                    ),
                    device=device,
                    ignore_index=(
                        config.ignore_index
                    ),
                    max_batches=(
                        config.max_validation_batches
                    ),
                )

                last_validation_loss = (
                    validation_result[
                        "loss"
                    ]
                )

                print()
                print(
                    f"[Validation] "
                    f"Step {global_step:,} | "
                    f"Loss "
                    f"{validation_result['loss']:.6f} | "
                    f"PPL "
                    f"{validation_result['perplexity']:.4f}"
                )

                if (
                    validation_result[
                        "loss"
                    ]
                    < best_validation_loss
                ):

                    best_validation_loss = (
                        validation_result[
                            "loss"
                        ]
                    )

                    best_path = (
                        config.checkpoint_dir
                        / "best_instruct.pt"
                    )

                    save_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_config=config,
                        model_config=model_config,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=(
                            batch_index + 1
                        ),
                        best_validation_loss=(
                            best_validation_loss
                        ),
                        last_train_loss=(
                            last_train_loss
                        ),
                        last_validation_loss=(
                            last_validation_loss
                        ),
                        debug=debug,
                    )

                    print(
                        f"Saved new best checkpoint: "
                        f"{best_path}"
                    )

            # -------------------------------------------------
            # Periodic checkpoint
            # -------------------------------------------------

            if (
                global_step
                % config.save_every
                == 0
            ):

                checkpoint_path = (
                    config.checkpoint_dir
                    /
                    (
                        "instruct_step_"
                        f"{global_step:08d}.pt"
                    )
                )

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_config=config,
                    model_config=model_config,
                    global_step=global_step,
                    epoch=epoch,
                    batch_in_epoch=(
                        batch_index + 1
                    ),
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    last_train_loss=(
                        last_train_loss
                    ),
                    last_validation_loss=(
                        last_validation_loss
                    ),
                    debug=debug,
                )

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_path}"
                )

            # -------------------------------------------------
            # Stop condition
            # -------------------------------------------------

            if global_step >= max_steps:

                stop_training = True
                break

        # Resume skipping applies only once.
        resume_batch_in_epoch = 0

        if stop_training:
            break

    # =========================================================
    # Final validation
    # =========================================================

# =========================================================
# Final validation
# =========================================================

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    final_validation = validate(
        model=model,
        validation_loader=validation_loader,
        device=device,
        ignore_index=config.ignore_index,
        max_batches=None,
    )

    last_validation_loss = final_validation["loss"]

    # ---------------------------------------------------------
    # Check whether the final model is the best model
    # ---------------------------------------------------------

    final_is_best = (
        last_validation_loss
        < best_validation_loss
    )

    if final_is_best:
        best_validation_loss = last_validation_loss

    print()
    print(
        f"Final validation loss : "
        f"{final_validation['loss']:.6f}"
    )

    print(
        f"Final validation PPL  : "
        f"{final_validation['perplexity']:.4f}"
    )

    print(
        f"Validation tokens     : "
        f"{final_validation['tokens']:,}"
    )


    # =========================================================
    # Determine final checkpoint path
    # =========================================================

    if debug:
        final_name = "debug_instruct_final.pt"
    else:
        final_name = "final_instruct.pt"

    final_path = (
        config.checkpoint_dir
        / final_name
    )


    # =========================================================
    # Save final checkpoint
    # =========================================================

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=config,
        model_config=model_config,
        global_step=global_step,
        epoch=epoch,
        batch_in_epoch=0,
        best_validation_loss=best_validation_loss,
        last_train_loss=last_train_loss,
        last_validation_loss=last_validation_loss,
        debug=debug,
    )

    print(
        f"Saved final checkpoint: "
        f"{final_path}"
    )


    # =========================================================
    # Save best checkpoint if final model is best
    # =========================================================

    if final_is_best:

        if debug:
            best_path = (
                config.checkpoint_dir
                / "debug_best_instruct.pt"
            )
        else:
            best_path = (
                config.checkpoint_dir
                / "best_instruct.pt"
            )

        save_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=config,
            model_config=model_config,
            global_step=global_step,
            epoch=epoch,
            batch_in_epoch=0,
            best_validation_loss=best_validation_loss,
            last_train_loss=last_train_loss,
            last_validation_loss=last_validation_loss,
            debug=debug,
        )

        print(
            f"Saved best checkpoint : "
            f"{best_path}"
        )


    # =========================================================
    # Training complete
    # =========================================================

    print()
    print("=" * 80)
    print("SFT TRAINING COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Global steps          : "
        f"{global_step:,}"
    )

    print(
        f"Last train loss       : "
        f"{last_train_loss}"
    )

    print(
        f"Final validation loss : "
        f"{last_validation_loss:.6f}"
    )

    print(
        f"Best validation loss  : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Final checkpoint      : "
        f"{final_path}"
    )


# =============================================================
# CLI
# =============================================================


# =============================================================
# CLI
# =============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "MyGPT2 supervised instruction "
            "fine-tuning trainer."
        )
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Use a small dataset subset and stop "
            "after the configured debug steps."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from an existing SFT checkpoint."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional optimizer-step limit."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = InstructionTrainingConfig()

    resume_path = (
        Path(args.resume)
        if args.resume
        else None
    )

    train(
        config=config,
        debug=args.debug,
        resume_path=resume_path,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()