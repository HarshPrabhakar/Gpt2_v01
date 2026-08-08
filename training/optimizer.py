"""
============================================================
MyGPT2 - Optimizer
============================================================

Purpose
-------
Creates and validates the optimizer used to train MyGPT2.

Optimizer:
    AdamW

The optimizer separates parameters into two groups:

    1. Parameters WITH weight decay
    2. Parameters WITHOUT weight decay

Biases and LayerNorm parameters do not receive weight decay.

============================================================
"""

from __future__ import annotations

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
from torch import nn
from torch.optim import AdamW


# ============================================================
# Default Optimizer Configuration
# ============================================================

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01

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
):
    """
    Separate trainable parameters into decay and no-decay
    groups.

    Weight decay:
        Normal weight parameters.

    No weight decay:
        Biases.
        LayerNorm parameters.
        Other normalization parameters.
    """

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        name_lower = name.lower()

        # ----------------------------------------------------
        # Bias
        # ----------------------------------------------------

        if name.endswith(".bias"):

            no_decay_parameters.append(
                (name, parameter)
            )

            continue

        # ----------------------------------------------------
        # Normalization parameters
        # ----------------------------------------------------

        if (
            "norm" in name_lower
            or "layernorm" in name_lower
            or "ln_" in name_lower
        ):

            no_decay_parameters.append(
                (name, parameter)
            )

            continue

        # ----------------------------------------------------
        # Everything else
        # ----------------------------------------------------

        decay_parameters.append(
            (name, parameter)
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
    Create an AdamW optimizer for MyGPT2.
    """

    # --------------------------------------------------------
    # Validation
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

    if not (0 <= beta1 < 1):
        raise ValueError(
            "beta1 must be in [0, 1)."
        )

    if not (0 <= beta2 < 1):
        raise ValueError(
            "beta2 must be in [0, 1)."
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

    if not decay_parameters:
        raise RuntimeError(
            "No parameters were assigned to "
            "the weight-decay group."
        )

    if not no_decay_parameters:
        raise RuntimeError(
            "No parameters were assigned to "
            "the no-weight-decay group."
        )

    # --------------------------------------------------------
    # Create parameter groups
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
    # AdamW
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
    Return optimizer statistics.
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

        "total_parameters":
            total_parameters,

        "decay_parameters":
            decay_count,

        "no_decay_parameters":
            no_decay_count,

        "parameter_groups":
            len(optimizer.param_groups),

        "learning_rate":
            optimizer.param_groups[0]["lr"],

        "weight_decay":
            optimizer.param_groups[0]["weight_decay"],

    }


# ============================================================
# Print Summary
# ============================================================

def print_optimizer_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:

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
        f"No Decay Params      : "
        f"{stats['no_decay_parameters']:,}"
    )

    print()

    print(
        f"AdamW Betas          : "
        f"{optimizer.defaults['betas']}"
    )

    print(
        f"AdamW Epsilon        : "
        f"{optimizer.defaults['eps']}"
    )

    print()


# ============================================================
# Verify Optimizer
# ============================================================

def verify_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    model_parameters = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    # --------------------------------------------------------
    # Optimizer parameters
    # --------------------------------------------------------

    optimizer_parameters = {}

    for group in optimizer.param_groups:

        for parameter in group["params"]:

            optimizer_parameters[
                id(parameter)
            ] = parameter

    # --------------------------------------------------------
    # Missing parameters
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
    # Duplicate parameters
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
# Main Test
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
            "Running optimizer test on CPU."
        )

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GPTConfig()

    print(
        f"Vocabulary Size      : "
        f"{config.vocab_size:,}"
    )

    print(
        f"Context Length       : "
        f"{config.max_position_embeddings}"
    )

    print(
        f"Hidden Size          : "
        f"{config.hidden_size}"
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
        f"{config.intermediate_size}"
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

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total Parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Total Parameters     : "
        f"{total_parameters / 1e6:.2f}M"
    )

    print(
        f"Trainable Parameters  : "
        f"{trainable_parameters:,}"
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

        learning_rate=config.learning_rate,

        weight_decay=config.weight_decay,

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
    # Test forward/backward
    # --------------------------------------------------------

    print(
        "Testing optimizer update..."
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    batch_size = 2

    sequence_length = 32

    # --------------------------------------------------------
    # Random input
    # --------------------------------------------------------

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

    elif isinstance(
        output,
        tuple,
    ):

        loss = output[1]

    else:

        raise RuntimeError(
            "Unable to find loss in model output."
        )

    if loss is None:

        raise RuntimeError(
            "Model returned None for loss."
        )

    print(
        f"Test Loss             : "
        f"{loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()

    print(
        "Backward pass         : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    gradient_count = 0

    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        gradient_count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            invalid_gradients += 1

    print(
        f"Parameters with grads : "
        f"{gradient_count}"
    )

    print(
        f"Invalid gradients     : "
        f"{invalid_gradients}"
    )

    if invalid_gradients != 0:

        raise RuntimeError(
            "Invalid gradients detected."
        )

    print(
        "Gradient verification : ✅ PASSED"
    )

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.gradient_clip,
    )

    print(
        f"Gradient Norm         : "
        f"{gradient_norm.item():.6f}"
    )

    print(
        f"Gradient Clip         : "
        f"{config.gradient_clip}"
    )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    optimizer.step()

    print(
        "Optimizer step        : ✅ PASSED"
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