"""
============================================================
MyGPT2 - Model Verification Test
============================================================

This script verifies that the complete GPT model can:

1. Load the GPT configuration
2. Construct the model
3. Count parameters
4. Create dummy input
5. Perform a forward pass
6. Calculate training loss
7. Perform backpropagation
8. Generate a next token

This is a MODEL TEST only.

It does NOT train the model on the real dataset.

============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


# ============================================================
# Make project root available
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Import Model
# ============================================================

from model.config import GPTConfig
from model.model import MyGPTModel


# ============================================================
# Utility
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main Test
# ============================================================

def main():

    print_section(
        "MyGPT2 MODEL VERIFICATION"
    )

    # ========================================================
    # Device
    # ========================================================

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"CUDA available : YES"
        )

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "CUDA available : NO"
        )

        print(
            "Running model test on CPU."
        )

    # ========================================================
    # Configuration
    # ========================================================

    print_section(
        "Loading GPT Configuration"
    )

    config = GPTConfig()

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

    # ========================================================
    # Build Model
    # ========================================================

    print_section(
        "Building MyGPTModel"
    )

    model = MyGPTModel(
        config
    )

    model = model.to(
        device
    )

    # ========================================================
    # Model Summary
    # ========================================================

    model.print_model_summary()

    total_parameters = (
        model.get_num_parameters()
    )

    print(
        f"Total Parameters : "
        f"{total_parameters:,}"
    )

    print(
        f"Total Parameters : "
        f"{total_parameters / 1_000_000:.2f}M"
    )

    # ========================================================
    # Dummy Input
    # ========================================================

    print_section(
        "Creating Dummy Input"
    )

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

    print(
        f"Input Shape  : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Labels Shape : "
        f"{tuple(labels.shape)}"
    )

    # ========================================================
    # Forward Pass
    # ========================================================

    print_section(
        "Testing Forward Pass"
    )

    model.train()

    logits, loss = model(
        input_ids=input_ids,
        labels=labels,
    )

    print(
        f"Logits Shape : "
        f"{tuple(logits.shape)}"
    )

    if loss is not None:

        print(
            f"Loss         : "
            f"{loss.item():.6f}"
        )

    # ========================================================
    # Verify Output Shape
    # ========================================================

    expected_shape = (
        batch_size,
        sequence_length,
        config.vocab_size,
    )

    print()

    print(
        f"Expected Shape : "
        f"{expected_shape}"
    )

    print(
        f"Actual Shape   : "
        f"{tuple(logits.shape)}"
    )

    if tuple(logits.shape) != expected_shape:

        raise RuntimeError(
            "❌ Logits shape is incorrect."
        )

    print(
        "✅ Logits shape is correct."
    )

    # ========================================================
    # Verify Loss
    # ========================================================

    if loss is None:

        raise RuntimeError(
            "❌ Loss was not calculated."
        )

    if not torch.isfinite(loss):

        raise RuntimeError(
            "❌ Loss contains NaN or Inf."
        )

    print(
        "✅ Loss is valid."
    )

    # ========================================================
    # Backward Pass
    # ========================================================

    print_section(
        "Testing Backward Pass"
    )

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    print(
        "Backward pass completed."
    )

    # ========================================================
    # Check Gradients
    # ========================================================

    parameters_with_grad = 0

    parameters_without_grad = 0

    invalid_gradients = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:

            continue

        if parameter.grad is None:

            parameters_without_grad += 1

            print(
                f"⚠ No gradient: {name}"
            )

            continue

        parameters_with_grad += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            invalid_gradients += 1

            print(
                f"❌ Invalid gradient: {name}"
            )

    print()

    print(
        f"Parameters with gradients    : "
        f"{parameters_with_grad}"
    )

    print(
        f"Parameters without gradients : "
        f"{parameters_without_grad}"
    )

    print(
        f"Invalid gradients            : "
        f"{invalid_gradients}"
    )

    if invalid_gradients > 0:

        raise RuntimeError(
            "❌ Invalid gradients detected."
        )

    print(
        "✅ Backward pass and gradients "
        "look valid."
    )

    # ========================================================
    # Test Weight Tying
    # ========================================================

    print_section(
        "Testing Weight Tying"
    )

    embedding_weight = (
        model.embeddings
        .token_embeddings
        .weight
    )

    lm_head_weight = (
        model.lm_head.weight
    )

    same_storage = (
        embedding_weight.data_ptr()
        ==
        lm_head_weight.data_ptr()
    )

    if not same_storage:

        raise RuntimeError(
            "❌ Weight tying is not working."
        )

    print(
        "Embedding weight address:"
    )

    print(
        embedding_weight.data_ptr()
    )

    print()

    print(
        "LM head weight address:"
    )

    print(
        lm_head_weight.data_ptr()
    )

    print()

    print(
        "✅ Token embedding and LM head "
        "weights are tied."
    )

    # ========================================================
    # Test Next Token Prediction
    # ========================================================

    print_section(
        "Testing Next Token Prediction"
    )

    model.eval()

    test_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 8),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        next_token = (
            model.generate_next_token(
                test_input,
                temperature=1.0,
            )
        )

    print(
        f"Input Shape       : "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Next Token Shape  : "
        f"{tuple(next_token.shape)}"
    )

    print(
        f"Input IDs         : "
        f"{test_input.tolist()[0]}"
    )

    print(
        f"Predicted Token   : "
        f"{next_token.item()}"
    )

    if tuple(next_token.shape) != (1, 1):

        raise RuntimeError(
            "❌ Next-token output has incorrect shape."
        )

    print(
        "✅ Next-token prediction works."
    )

    # ========================================================
    # Test Basic Generation
    # ========================================================

    print_section(
        "Testing Basic Generation"
    )

    generation_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 5),
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        generation_input,
        max_new_tokens=5,
        temperature=1.0,
    )

    print(
        f"Original Length  : "
        f"{generation_input.shape[1]}"
    )

    print(
        f"Generated Length  : "
        f"{generated.shape[1]}"
    )

    print(
        f"Generated IDs     : "
        f"{generated.tolist()[0]}"
    )

    expected_length = 10

    if generated.shape[1] != expected_length:

        raise RuntimeError(
            "❌ Generation produced an "
            "unexpected sequence length."
        )

    print(
        "✅ Basic generation works."
    )

    # ========================================================
    # Final Result
    # ========================================================

    print_section(
        "MODEL TEST COMPLETED SUCCESSFULLY"
    )

    print(
        "✅ Configuration loaded"
    )

    print(
        "✅ Model constructed"
    )

    print(
        "✅ Parameter count calculated"
    )

    print(
        "✅ Forward pass works"
    )

    print(
        "✅ Logits shape is correct"
    )

    print(
        "✅ Loss calculation works"
    )

    print(
        "✅ Backward pass works"
    )

    print(
        "✅ Gradients are valid"
    )

    print(
        "✅ Weight tying works"
    )

    print(
        "✅ Next-token prediction works"
    )

    print(
        "✅ Basic generation works"
    )

    print()

    print(
        "The GPT model is ready for the "
        "training pipeline."
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()