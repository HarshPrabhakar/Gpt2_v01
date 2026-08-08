"""
============================================================
MyGPT2 - Learning Rate Scheduler
============================================================

Learning-rate schedule:

    Linear Warmup
          ↓
    Maximum Learning Rate
          ↓
    Cosine Decay
          ↓
    Minimum Learning Rate

Default configuration:

    Base LR        : 3e-4
    Warmup Ratio   : 5%
    Minimum LR     : 10% of base LR

============================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Defaults
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
    Create the learning-rate multiplier.

    The returned value is a MULTIPLIER.

    Example:

        base LR = 0.0003
        multiplier = 1.0

        actual LR = 0.0003
    """

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
            "warmup_steps cannot exceed total_steps."
        )

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            "min_lr_ratio must be between 0 and 1."
        )

    def lr_lambda(
        current_step: int,
    ) -> float:

        # ====================================================
        # Warmup
        # ====================================================

        if (
            warmup_steps > 0
            and current_step < warmup_steps
        ):

            return (
                float(current_step + 1)
                / float(warmup_steps)
            )

        # ====================================================
        # No decay region
        # ====================================================

        if total_steps <= warmup_steps:

            return min_lr_ratio

        # ====================================================
        # Cosine decay progress
        # ====================================================

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

        # ====================================================
        # Cosine decay
        # ====================================================

        cosine_decay = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        # ====================================================
        # Minimum LR
        # ====================================================

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
    Create a Linear Warmup + Cosine Decay scheduler.
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be greater than zero."
        )

    # --------------------------------------------------------
    # Calculate warmup steps
    # --------------------------------------------------------

    if warmup_steps is None:

        if not 0.0 <= warmup_ratio <= 1.0:
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
    # IMPORTANT:
    #
    # Save the original/base LR BEFORE LambdaLR changes it.
    # --------------------------------------------------------

    base_learning_rates = [
        float(group["lr"])
        for group in optimizer.param_groups
    ]

    # --------------------------------------------------------
    # Create lambda
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

    # --------------------------------------------------------
    # Restore base learning rates.
    #
    # This makes the configured learning rate the actual
    # maximum learning rate.
    # --------------------------------------------------------

    scheduler.base_lrs = base_learning_rates

    return scheduler


# ============================================================
# Get Current Learning Rate
# ============================================================

def get_learning_rate(
    optimizer: Optimizer,
) -> float:

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

    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


# ============================================================
# Get Scheduler Information
# ============================================================

def get_scheduler_info(
    scheduler: LambdaLR,
) -> dict:

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
# Preview Schedule
# ============================================================

def preview_schedule(
    optimizer: Optimizer,
    scheduler: LambdaLR,
    total_steps: int,
    points: int = 12,
) -> list[tuple[int, float]]:

    if total_steps <= 0:

        raise ValueError(
            "total_steps must be greater than zero."
        )

    if points < 2:

        raise ValueError(
            "points must be at least 2."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use scheduler.base_lrs, NOT the optimizer's current LR.
    #
    # The optimizer LR may already contain the warmup
    # multiplier.
    # --------------------------------------------------------

    base_learning_rate = (
        scheduler.base_lrs[0]
    )

    lr_function = (
        scheduler.lr_lambdas[0]
    )

    results = []

    for index in range(points):

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
            base_learning_rate
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

    print("=" * 70)

    print(
        "MyGPT2 Learning Rate Scheduler Test"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Imports
    # --------------------------------------------------------

    from model.config import GPTConfig

    from model.model import MyGPTModel

    from training.optimizer import (
        create_optimizer,
    )

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
    # Config
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
    # Model
    # --------------------------------------------------------

    print(
        "Creating model..."
    )

    model = MyGPTModel(
        config
    ).to(device)

    model.train()

    print(
        "Model created successfully."
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

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

    print()

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    test_total_steps = 1000

    test_warmup_steps = 100

    min_lr_ratio = 0.10

    print(
        "Scheduler Configuration"
    )

    print("-" * 70)

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
        f"{min_lr_ratio}"
    )

    print()

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    print(
        "Creating scheduler..."
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=test_total_steps,
        warmup_steps=test_warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    print(
        "Scheduler created successfully."
    )

    print()

    # --------------------------------------------------------
    # Base LR
    # --------------------------------------------------------

    base_lr = scheduler.base_lrs[0]

    print(
        f"Base Learning Rate   : "
        f"{base_lr:.10f}"
    )

    print(
        f"Current Learning Rate : "
        f"{get_learning_rate(optimizer):.10f}"
    )

    print()

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print(
        "Learning Rate Schedule Preview"
    )

    print("-" * 70)

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
    # Expected values
    # --------------------------------------------------------

    expected_max_lr = (
        config.learning_rate
    )

    expected_min_lr = (
        config.learning_rate
        * min_lr_ratio
    )

    final_lr = preview[-1][1]

    print(
        f"Expected Maximum LR  : "
        f"{expected_max_lr:.10f}"
    )

    print(
        f"Expected Minimum LR  : "
        f"{expected_min_lr:.10f}"
    )

    print(
        f"Final Preview LR     : "
        f"{final_lr:.10f}"
    )

    print()

    # --------------------------------------------------------
    # Verify maximum LR
    # --------------------------------------------------------

    if not math.isclose(
        base_lr,
        expected_max_lr,
        rel_tol=1e-6,
    ):

        raise RuntimeError(
            "Base learning rate is incorrect."
        )

    print(
        "Base LR verification : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Verify warmup
    # --------------------------------------------------------

    warmup_start = preview[0][1]

    warmup_end = (
        preview[1][1]
    )

    if warmup_end <= warmup_start:

        raise RuntimeError(
            "Learning rate did not increase "
            "during warmup."
        )

    print(
        "Warmup behavior      : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Verify final LR
    # --------------------------------------------------------

    if final_lr > expected_max_lr:

        raise RuntimeError(
            "Final LR is greater than maximum LR."
        )

    if final_lr < expected_min_lr * 0.999:

        raise RuntimeError(
            "Final LR is below configured minimum."
        )

    print(
        "Cosine decay         : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Test scheduler stepping
    # --------------------------------------------------------

    print()

    print(
        "Testing scheduler.step()..."
    )

    learning_rates = []

    for _ in range(20):

        optimizer.zero_grad(
            set_to_none=True
        )

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
        "Scheduler stepping    : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Validate learning rates
    # --------------------------------------------------------

    for learning_rate in learning_rates:

        if not math.isfinite(
            learning_rate
        ):

            raise RuntimeError(
                "Non-finite learning rate detected."
            )

    print(
        "Learning rates valid  : ✅ PASSED"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    info = get_scheduler_info(
        scheduler
    )

    print()

    print(
        "Scheduler State"
    )

    print("-" * 70)

    print(
        f"Current Step          : "
        f"{info['last_epoch']}"
    )

    print(
        f"Base Learning Rates   : "
        f"{info['base_learning_rates']}"
    )

    print(
        f"Current Learning Rate : "
        f"{info['current_learning_rates']}"
    )

    print()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "Scheduler test completed successfully."
    )

    print("=" * 70)