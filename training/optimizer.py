"""
============================================================
MyGPT2 - Optimizer
============================================================

Purpose
-------
Creates and configures the optimizer used to train MyGPT2.

Optimizer:
    AdamW

The implementation separates model parameters into two groups:

    1. Parameters WITH weight decay
    2. Parameters WITHOUT weight decay

Weight decay is applied to normal weight matrices, while
biases and normalization parameters are excluded.

This is the standard approach for GPT-style models.

============================================================
"""

from __future__ import annotations


# ============================================================
# Standard Library
# ============================================================

import sys
from pathlib import Path
from typing import Iterable


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Third-Party
# ============================================================

import torch
from torch import nn
from torch.optim import AdamW


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_LEARNING_RATE = 3e-4

DEFAULT_WEIGHT_DECAY = 0.1

DEFAULT_BETAS = (
    0.9,
    0.95,
)

DEFAULT_EPS = 1e-8


# ============================================================
# Parameter Classification
# ============================================================

def classify_parameters(
    model: nn.Module,
) -> tuple[
    list[tuple[str, nn.Parameter]],
    list[tuple[str, nn.Parameter]],
]:
    """
    Separate model parameters into:

        decay_parameters
        no_decay_parameters

    Weight decay is applied to normal weight matrices.

    Biases and normalization parameters are excluded from
    weight decay.

    Returns
    -------
    decay_parameters:
        Parameters receiving weight decay.

    no_decay_parameters:
        Parameters receiving zero weight decay.
    """

    decay_parameters = []

    no_decay_parameters = []

    # --------------------------------------------------------
    # Iterate through model parameters
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        # ----------------------------------------------------
        # Ignore frozen parameters
        # ----------------------------------------------------

        if not parameter.requires_grad:

            continue

        # ----------------------------------------------------
        # Bias
        # ----------------------------------------------------

        if name.endswith(
            ".bias"
        ):

            no_decay_parameters.append(
                (
                    name,
                    parameter,
                )
            )

            continue

        # ----------------------------------------------------
        # Normalization parameters
        # ----------------------------------------------------

        name_lower = name.lower()

        if (
            "norm" in name_lower
            or "layernorm" in name_lower
            or "ln_" in name_lower
        ):

            no_decay_parameters.append(
                (
                    name,
                    parameter,
                )
            )

            continue

        # ----------------------------------------------------
        # Everything else
        # ----------------------------------------------------

        decay_parameters.append(
            (
                name,
                parameter,
            )
        )

    return (
        decay_parameters,
        no_decay_parameters,
    )


# ============================================================
# Create Optimizer
# ============================================================

def create_optimizer(
    model: nn.Module,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
) -> AdamW:
    """
    Create AdamW optimizer for MyGPT2.

    Parameters
    ----------
    model:
        MyGPT2 model.

    learning_rate:
        Initial learning rate.

    weight_decay:
        Weight decay applied to eligible parameters.

    betas:
        AdamW beta values.

    eps:
        Numerical stability value.

    Returns
    -------
    torch.optim.AdamW
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if learning_rate <= 0:

        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if weight_decay < 0:

        raise ValueError(
            "weight_decay cannot be negative."
        )

    if len(betas) != 2:

        raise ValueError(
            "betas must contain exactly two values."
        )

    beta1, beta2 = betas

    if not (
        0 <= beta1 < 1
        and 0 <= beta2 < 1
    ):

        raise ValueError(
            "AdamW beta values must be in [0, 1)."
        )

    if eps <= 0:

        raise ValueError(
            "eps must be greater than zero."
        )

    # --------------------------------------------------------
    # Classify parameters
    # --------------------------------------------------------

    decay_parameters, no_decay_parameters = (
        classify_parameters(model)
    )

    if len(decay_parameters) == 0:

        raise RuntimeError(
            "No parameters were assigned to "
            "the weight-decay group."
        )

    if len(no_decay_parameters) == 0:

        raise RuntimeError(
            "No parameters were assigned to "
            "the no-weight-decay group."
        )

    # --------------------------------------------------------
    # Parameter groups
    # --------------------------------------------------------

    parameter_groups = [
        {
            "params": [
                parameter
                for _, parameter
                in decay_parameters
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for _, parameter
                in no_decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    # --------------------------------------------------------
    # Create AdamW
    # --------------------------------------------------------

    optimizer = AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer


# ============================================================
# Optimizer Statistics
# ============================================================

def get_optimizer_statistics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """
    Return useful optimizer statistics.
    """

    decay_parameters, no_decay_parameters = (
        classify_parameters(model)
    )

    decay_count = sum(
        parameter.numel()
        for _, parameter
        in decay_parameters
    )

    no_decay_count = sum(
        parameter.numel()
        for _, parameter
        in no_decay_parameters
    )

    total_parameters = (
        decay_count
        + no_decay_count
    )

    return {
        "total_parameters": total_parameters,
        "decay_parameters": decay_count,
        "no_decay_parameters": no_decay_count,
        "parameter_groups": len(
            optimizer.param_groups
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }


# ============================================================
# Print Optimizer Summary
# ============================================================

def print_optimizer_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Print optimizer configuration and parameter statistics.
    """

    stats = get_optimizer_statistics(
        model,
        optimizer,
    )

    print(
        "=" * 70
    )

    print(
        "MyGPT2 Optimizer"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Optimizer            : "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Learning Rate        : "
        f"{stats['learning_rate']}"
    )

    print(
        f"Weight Decay         : "
        f"{stats['weight_decay']}"
    )

    print(
        f"Parameter Groups     : "
        f"{stats['parameter_groups']}"
    )

    print()

    print(
        f"Total Parameters     : "
        f"{stats['total_parameters']:,}"
    )

    print(
        f"Weight Decay Params  : "
        f"{stats['decay_parameters']:,}"
    )

    print(
        f"No Decay Params     : "
        f"{stats['no_decay_parameters']:,}"
    )

    print()

    print(
        "AdamW Betas          : "
        f"{optimizer.defaults['betas']}"
    )

    print(
        "AdamW Epsilon        : "
        f"{optimizer.defaults['eps']}"
    )

    print()

    print(
        "=" * 70
    )


# ============================================================
# Verify Optimizer
# ============================================================

def verify_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Perform consistency checks on the optimizer.

    Raises an error if a trainable parameter is missing
    from the optimizer.
    """

    # --------------------------------------------------------
    # Collect model parameters
    # --------------------------------------------------------

    model_parameters = {
        id(parameter): parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    }

    # --------------------------------------------------------
    # Collect optimizer parameters
    # --------------------------------------------------------

    optimizer_parameters = {}

    for group in optimizer.param_groups:

        for parameter in group["params"]:

            optimizer_parameters[
                id(parameter)
            ] = parameter

    # --------------------------------------------------------
    # Check missing parameters
    # --------------------------------------------------------

    missing_parameters = (
        set(model_parameters.keys())
        - set(optimizer_parameters.keys())
    )

    if missing_parameters:

        raise RuntimeError(
            f"{len(missing_parameters)} trainable "
            "parameters are missing from the optimizer."
        )

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    parameter_ids = []

    for group in optimizer.param_groups:

        for parameter in group["params"]:

            parameter_ids.append(
                id(parameter)
            )

    if len(parameter_ids) != len(
        set(parameter_ids)
    ):

        raise RuntimeError(
            "A parameter appears more than once "
            "in the optimizer."
        )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "MyGPT2 Optimizer Test"
    )

    print(
        "=" * 70
    )

    print()

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

        print()

        print(
            "Make sure your model package is available."
        )

        raise

    # --------------------------------------------------------
    # Import configuration
    # --------------------------------------------------------

    try:

        from model.config import GPTConfig

    except ImportError as exc:

        print(
            "Could not import GPTConfig."
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
            "Running optimizer test on CPU."
        )

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GPTConfig()

    print(
        f"Vocabulary Size : "
        f"{config.vocab_size:,}"
    )

    print(
        f"Hidden Size     : "
        f"{config.hidden_size}"
    )

    print(
        f"Layers          : "
        f"{config.num_layers}"
    )

    print(
        f"Attention Heads : "
        f"{config.num_heads}"
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

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"Model Parameters : "
        f"{total_parameters:,}"
    )

    print(
        f"Model Parameters : "
        f"{total_parameters / 1e6:.2f}M"
    )

    print()

    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    print(
        "Creating AdamW optimizer..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=DEFAULT_BETAS,
        eps=DEFAULT_EPS,
    )

    print(
        "Optimizer created successfully."
    )

    print()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_optimizer(
        model,
        optimizer,
    )

    print(
        "Optimizer verification : ✅ PASSED"
    )

    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_optimizer_summary(
        model,
        optimizer,
    )

    # --------------------------------------------------------
    # Test optimizer step
    # --------------------------------------------------------

    print(
        "Testing optimizer step..."
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Create tiny test batch
    # --------------------------------------------------------

    batch_size = 2

    sequence_length = 32

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    labels = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    output = model(
        input_ids,
        labels=labels,
    )

    # --------------------------------------------------------
    # Extract loss
    # --------------------------------------------------------

    if isinstance(
        output,
        dict,
    ):

        loss = output.get(
            "loss"
        )

    elif hasattr(
        output,
        "loss",
    ):

        loss = output.loss

    else:

        raise RuntimeError(
            "Model output does not contain a loss."
        )

    if loss is None:

        raise RuntimeError(
            "Model returned None for loss."
        )

    print(
        f"Test Loss : "
        f"{loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    gradient_count = 0

    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            if not torch.isfinite(
                parameter.grad
            ).all():

                invalid_gradients += 1

    print(
        f"Parameters with gradients : "
        f"{gradient_count}"
    )

    print(
        f"Invalid gradients         : "
        f"{invalid_gradients}"
    )

    if invalid_gradients != 0:

        raise RuntimeError(
            "Invalid gradients detected."
        )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    optimizer.step()

    print(
        "Optimizer step            : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    print()

    print(
        "=" * 70
    )

    print(
        "Optimizer test completed successfully."
    )

    print(
        "=" * 70
    )