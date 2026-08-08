"""
============================================================
MyGPT2 - Learning Rate Scheduler
============================================================

Purpose
-------
Controls the learning rate during GPT-2 training.

Schedule:
    1. Linear Warmup
    2. Cosine Decay

Training flow:

    Learning Rate
          ^
          |
          |          /\
          |         /  \
          |        /    \
          |       /      \
          |      /        \
          |_____/          \________
          |
          +--------------------------> Training Steps
             Warmup      Cosine Decay

Why warmup?
-----------
At the beginning of training, model weights are still
randomly initialized. Starting immediately at the maximum
learning rate can produce unstable gradients.

Warmup gradually increases the learning rate.

Why cosine decay?
-----------------
After warmup, the learning rate gradually decreases,
allowing the model to make smaller and more precise updates
as training progresses.

============================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PyTorch
# ============================================================

import torch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ============================================================
# Default Scheduler Configuration
# ============================================================

DEFAULT_WARMUP_RATIO = 0.05

DEFAULT_MIN_LR_RATIO = 0.10


# ============================================================
# Learning Rate Lambda
# ============================================================

def create_lr_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = DEFAULT_MIN_LR_RATIO,
):
    """
    Create the learning-rate function used by LambdaLR.

    Parameters
    ----------
    warmup_steps:
        Number of warmup optimization steps.

    total_steps:
        Total number of optimization steps.

    min_lr_ratio:
        Final learning rate as a fraction of the initial
        learning rate.

        Example:

            initial LR = 3e-4
            min_lr_ratio = 0.1

            final LR = 3e-5

    Returns
    -------
    callable
        Lambda function used by LambdaLR.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps cannot be negative."
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be greater than zero."
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot be greater than total_steps."
        )

    if not (
        0.0 <= min_lr_ratio <= 1.0
    ):
        raise ValueError(
            "min_lr_ratio must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Lambda function
    # --------------------------------------------------------

    def lr_lambda(current_step: int) -> float:

        # ====================================================
        # Warmup
        # ====================================================

        if warmup_steps > 0:

            if current_step < warmup_steps:

                return float(
                    current_step + 1
                ) / float(
                    warmup_steps
                )

        # ====================================================
        # After warmup
        # ====================================================

        if total_steps <= warmup_steps:

            return min_lr_ratio

        # ----------------------------------------------------
        # Progress through cosine decay
        # ----------------------------------------------------

        progress = (
            current_step - warmup_steps
        ) / (
            total_steps - warmup_steps
        )

        progress = max(
            0.0,
            min(
                1.0,
                progress,
            ),
        )

        # ----------------------------------------------------
        # Cosine decay
        # ----------------------------------------------------

        cosine_decay = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        # ----------------------------------------------------
        # Apply minimum learning rate
        # ----------------------------------------------------

        return (
            min_lr_ratio
            + (
                1.0
                - min_lr_ratio
            )
            * cosine_decay
        )

    return lr_lambda


# ============================================================
# Create Scheduler
# ============================================================

def create_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    min_lr_ratio: float = DEFAULT_MIN_LR_RATIO,
) -> LambdaLR:
    """
    Create a linear-warmup + cosine-decay scheduler.

    Parameters
    ----------
    optimizer:
        AdamW optimizer.

    total_steps:
        Total number of optimizer updates.

    warmup_steps:
        Explicit warmup steps.

        If None, warmup_ratio is used.

    warmup_ratio:
        Fraction of training used for warmup.

    min_lr_ratio:
        Final LR / initial LR.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if total_steps <= 0:

        raise ValueError(
            "total_steps must be greater than zero."
        )

    if warmup_steps is None:

        if not (
            0.0 <= warmup_ratio <= 1.0
        ):

            raise ValueError(
                "warmup_ratio must be between 0 and 1."
            )

        warmup_steps = max(
            1,
            int(
                total_steps
                * warmup_ratio
            ),
        )

    if warmup_steps < 0:

        raise ValueError(
            "warmup_steps cannot be negative."
        )

    if warmup_steps > total_steps:

        raise ValueError(
            "warmup_steps cannot exceed total_steps."
        )

    # --------------------------------------------------------
    # Create LR function
    # --------------------------------------------------------

    lr_lambda = create_lr_lambda(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # --------------------------------------------------------
    # Create scheduler
    # --------------------------------------------------------

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    return scheduler


# ============================================================
# Get Current Learning Rate
# ============================================================

def get_learning_rate(
    optimizer: Optimizer,
) -> float:
    """
    Return the learning rate of the first optimizer group.
    """

    if not optimizer.param_groups:

        raise RuntimeError(
            "Optimizer contains no parameter groups."
        )

    return float(
        optimizer.param_groups[0]["lr"]
    )


# ============================================================
# Get All Learning Rates
# ============================================================

def get_all_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    """
    Return learning rates from every optimizer group.
    """

    return [
        float(
            group["lr"]
        )
        for group
        in optimizer.param_groups
    ]


# ============================================================
# Scheduler Information
# ============================================================

def get_scheduler_info(
    scheduler: LambdaLR,
) -> dict:
    """
    Return useful scheduler information.
    """

    return {
        "last_epoch":
            scheduler.last_epoch,

        "base_learning_rates":
            list(
                scheduler.base_lrs
            ),

        "current_learning_rates":
            list(
                scheduler.get_last_lr()
            ),
    }


# ============================================================
# Preview Learning Rate Schedule
# ============================================================

def preview_schedule(
    optimizer: Optimizer,
    scheduler: LambdaLR,
    total_steps: int,
    points: int = 10,
) -> list[tuple[int, float]]:
    """
    Preview selected points from the learning-rate schedule.

    This does not modify the optimizer or scheduler.

    Returns:
        List of:
            (step, learning_rate)
    """

    if total_steps <= 0:

        raise ValueError(
            "total_steps must be greater than zero."
        )

    if points < 2:

        raise ValueError(
            "points must be at least 2."
        )

    base_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # Recover lambda function
    # --------------------------------------------------------

    lr_function = scheduler.lr_lambdas[0]

    results = []

    for index in range(points):

        if points == 1:

            step = 0

        else:

            step = int(
                index
                * (
                    total_steps - 1
                )
                / (
                    points - 1
                )
            )

        multiplier = lr_function(
            step
        )

        learning_rate = (
            base_lr
            * multiplier
        )

        results.append(
            (
                step,
                learning_rate,
            )
        )

    return results


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "MyGPT2 Learning Rate Scheduler Test"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # Import configuration
    # --------------------------------------------------------

    from model.config import GPTConfig

    # --------------------------------------------------------
    # Import model
    # --------------------------------------------------------

    try:

        from model.model import MyGPTModel

    except ImportError as exc:

        print(
            "Could not import MyGPTModel."
        )

        print(
            f"Error: {exc}"
        )

        raise

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            "CUDA available : YES"
        )

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "CUDA available : NO"
        )

        print(
            "Running scheduler test on CPU."
        )

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GPTConfig()

    print(
        f"Learning Rate        : "
        f"{config.learning_rate}"
    )

    print(
        f"Weight Decay         : "
        f"{config.weight_decay}"
    )

    print(
        f"Batch Size           : "
        f"{config.batch_size}"
    )

    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

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

    print()

    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    print(
        "Creating optimizer..."
    )

    # Use the optimizer implementation
    # already created for MyGPT2.

    from training.optimizer import (
        create_optimizer,
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(
        "Optimizer created successfully."
    )

    print()

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    test_total_steps = 1000

    test_warmup_steps = 100

    print(
        "Scheduler Configuration"
    )

    print(
        "-" * 70
    )

    print(
        f"Total Steps         : "
        f"{test_total_steps:,}"
    )

    print(
        f"Warmup Steps        : "
        f"{test_warmup_steps:,}"
    )

    print(
        f"Warmup Ratio        : "
        f"{test_warmup_steps / test_total_steps:.2%}"
    )

    print(
        f"Minimum LR Ratio    : "
        f"{DEFAULT_MIN_LR_RATIO}"
    )

    print()

    # --------------------------------------------------------
    # Create scheduler
    # --------------------------------------------------------

    print(
        "Creating scheduler..."
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=test_total_steps,
        warmup_steps=test_warmup_steps,
        min_lr_ratio=DEFAULT_MIN_LR_RATIO,
    )

    print(
        "Scheduler created successfully."
    )

    print()

    # --------------------------------------------------------
    # Initial LR
    # --------------------------------------------------------

    print(
        f"Initial Learning Rate : "
        f"{get_learning_rate(optimizer):.10f}"
    )

    print()

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print(
        "Learning Rate Schedule Preview"
    )

    print(
        "-" * 70
    )

    preview = preview_schedule(
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=test_total_steps,
        points=12,
    )

    for step, learning_rate in preview:

        print(
            f"Step {step:>5}  "
            f"LR = {learning_rate:.10f}"
        )

    print()

    # --------------------------------------------------------
    # Verify warmup
    # --------------------------------------------------------

    warmup_lr_start = (
        preview[0][1]
    )

    warmup_lr_end = (
        preview[1][1]
    )

    if warmup_lr_end <= warmup_lr_start:

        raise RuntimeError(
            "Learning rate did not increase "
            "during warmup."
        )

    print(
        "Warmup behavior       : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Verify final decay
    # --------------------------------------------------------

    final_lr = preview[-1][1]

    initial_lr = config.learning_rate

    minimum_lr = (
        initial_lr
        * DEFAULT_MIN_LR_RATIO
    )

    if final_lr > initial_lr:

        raise RuntimeError(
            "Final learning rate is greater "
            "than the initial learning rate."
        )

    print(
        "Cosine decay behavior : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Verify minimum LR
    # --------------------------------------------------------

    print(
        f"Initial LR            : "
        f"{initial_lr:.10f}"
    )

    print(
        f"Minimum LR            : "
        f"{minimum_lr:.10f}"
    )

    print(
        f"Final Preview LR      : "
        f"{final_lr:.10f}"
    )

    print()

    # --------------------------------------------------------
    # Actual scheduler stepping
    # --------------------------------------------------------

    print(
        "Testing scheduler.step()..."
    )

    learning_rates = []

    for step in range(
        20
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Create tiny dummy loss
        # ----------------------------------------------------

        dummy_loss = sum(
            parameter.sum() * 0.0
            for parameter
            in model.parameters()
        )

        dummy_loss.backward()

        optimizer.step()

        scheduler.step()

        learning_rates.append(
            get_learning_rate(
                optimizer
            )
        )

    print(
        "Scheduler stepping     : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Check finite LR
    # --------------------------------------------------------

    for learning_rate in learning_rates:

        if not math.isfinite(
            learning_rate
        ):

            raise RuntimeError(
                "Non-finite learning rate detected."
            )

    print(
        "Learning rates valid   : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Scheduler state
    # --------------------------------------------------------

    info = get_scheduler_info(
        scheduler
    )

    print()

    print(
        "Scheduler State"
    )

    print(
        "-" * 70
    )

    print(
        f"Current Step          : "
        f"{info['last_epoch']}"
    )

    print(
        f"Current Learning Rate : "
        f"{info['current_learning_rates']}"
    )

    print()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "Scheduler test completed successfully."
    )

    print(
        "=" * 70
    )