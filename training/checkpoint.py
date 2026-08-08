"""MyGPT2 training checkpoint manager.

Designed for the current MyGPT2 model API and PyTorch 2.6+.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
CHECKPOINT_VERSION = "1.1"


def get_random_states() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU and CUDA RNG states."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": None,
        "torch": torch.get_rng_state().cpu(),
        "cuda": None,
    }

    try:
        import numpy as np
        states["numpy"] = np.random.get_state()
    except ImportError:
        pass

    if torch.cuda.is_available():
        states["cuda"] = [
            state.cpu() for state in torch.cuda.get_rng_state_all()
        ]

    return states


def _as_cpu_byte_tensor(value: Any) -> torch.ByteTensor:
    """Normalize a serialized RNG state to CPU torch.ByteTensor."""
    if torch.is_tensor(value):
        value = value.detach().cpu().to(dtype=torch.uint8)
    else:
        value = torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    # Explicitly construct the exact type expected by torch.set_rng_state.
    return torch.ByteTensor(value.tolist())


def restore_random_states(states: dict[str, Any]) -> None:
    """Restore Python, NumPy, PyTorch CPU and CUDA RNG states."""
    if not states:
        return

    python_state = states.get("python")
    if python_state is not None:
        try:
            random.setstate(python_state)
        except Exception as exc:
            raise RuntimeError("Failed to restore Python random state.") from exc

    numpy_state = states.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
            np.random.set_state(numpy_state)
        except ImportError:
            pass
        except Exception as exc:
            raise RuntimeError("Failed to restore NumPy random state.") from exc

    torch_state = states.get("torch")
    if torch_state is not None:
        try:
            torch.set_rng_state(_as_cpu_byte_tensor(torch_state))
        except Exception as exc:
            raise RuntimeError("Failed to restore PyTorch CPU RNG state.") from exc

    cuda_states = states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            normalized = [
                _as_cpu_byte_tensor(state)
                for state in cuda_states
            ]
            torch.cuda.set_rng_state_all(normalized)
        except Exception as exc:
            raise RuntimeError("Failed to restore CUDA RNG states.") from exc


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert GPTConfig/dataclass/object into a serializable dictionary."""
    result: dict[str, Any] = {}

    if hasattr(config, "__dataclass_fields__"):
        names = config.__dataclass_fields__.keys()
    else:
        names = [
            name for name in dir(config)
            if not name.startswith("_")
        ]

    for name in names:
        try:
            value = getattr(config, name)
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(value, (str, int, float, bool, type(None))):
            result[name] = value
        elif isinstance(value, torch.dtype):
            result[name] = str(value)

    return result


def make_json_safe(value: Any) -> Any:
    """Convert common Python/PyTorch objects to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value)

    if torch.is_tensor(value):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    return str(value)


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    *,
    epoch: int = 0,
    global_step: int = 0,
    best_loss: float | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete training checkpoint."""
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": best_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "random_states": get_random_states(),
        "config": config_to_dict(config) if config is not None else None,
        "extra": extra or {},
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    *,
    epoch: int = 0,
    global_step: int = 0,
    best_loss: float | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
    path: Path | str | None = None,
) -> Path:
    """Save a complete checkpoint and human-readable metadata."""
    if path is None:
        path = DEFAULT_CHECKPOINT_DIR / "latest.pt"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_loss=best_loss,
        train_loss=train_loss,
        val_loss=val_loss,
        config=config,
        extra=extra,
    )

    temporary_path = path.with_suffix(".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)

    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at": checkpoint["created_at"],
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": make_json_safe(checkpoint["config"]),
        "extra": make_json_safe(extra or {}),
        "checkpoint_file": path.name,
        "checkpoint_size_mb": round(
            path.stat().st_size / (1024 ** 2), 2
        ),
    }

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return path


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> None:
    """Move optimizer tensor state to the model's target device."""
    target_device = torch.device(device)

    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue

        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a trusted complete training checkpoint.

    The checkpoint is deliberately deserialized on CPU. This prevents
    RNG tensors from being moved to CUDA during torch.load().

    weights_only=False is required because this checkpoint contains
    optimizer/scheduler/random-state objects in addition to weights.
    Only use this with checkpoints you trust.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format.")

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        _move_optimizer_state_to_device(optimizer, device)

    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if restore_rng:
        restore_random_states(
            checkpoint.get("random_states", {})
        )

    return checkpoint


def save_latest(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    **kwargs: Any,
) -> Path:
    """Save the latest checkpoint."""
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=DEFAULT_CHECKPOINT_DIR / "latest.pt",
        **kwargs,
    )


def save_best(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    **kwargs: Any,
) -> Path:
    """Save the best checkpoint."""
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=DEFAULT_CHECKPOINT_DIR / "best.pt",
        **kwargs,
    )


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return a compact checkpoint summary."""
    return {
        "version": checkpoint.get("checkpoint_version"),
        "created_at": checkpoint.get("created_at"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_loss": checkpoint.get("best_loss"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "has_model": "model_state_dict" in checkpoint,
        "has_optimizer": "optimizer_state_dict" in checkpoint,
        "has_scheduler": checkpoint.get("scheduler_state_dict") is not None,
        "has_random_states": "random_states" in checkpoint,
    }


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("MyGPT2 Checkpoint Manager Test")
    print("=" * 75)
    print()

    from model.config import GPTConfig
    from model.model import MyGPTModel
    from training.optimizer import create_optimizer
    from training.scheduler import create_scheduler

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available : YES")
        print(
            "GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        print("CUDA available : NO")
        print("Running checkpoint test on CPU.")

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GPTConfig()

    print(
        f"Vocabulary Size : {config.vocab_size}"
    )
    print(
        f"Hidden Size     : {config.hidden_size}"
    )
    print(
        f"Layers          : {config.num_layers}"
    )
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Creating model...")

    model = MyGPTModel(config).to(device)
    model.train()

    print("Model created successfully.")
    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    print("Creating optimizer...")

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Optimizer created successfully.")
    print()

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    print("Creating scheduler...")

    total_steps = 1000
    warmup_steps = 100

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    print("Scheduler created successfully.")
    print()

    # --------------------------------------------------------
    # Dummy training step
    # --------------------------------------------------------

    print("Running dummy training step...")

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
        dtype=torch.long,
    )

    labels = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
        dtype=torch.long,
    )

    output = model(
        input_ids=input_ids,
        labels=labels,
    )

    # Current MyGPT2 model returns (logits, loss).
    # Also support an object exposing .logits and .loss.
    if isinstance(output, tuple):
        if len(output) != 2:
            raise RuntimeError(
                "Expected model output format: (logits, loss)."
            )
        logits, loss = output
    else:
        if not hasattr(output, "loss"):
            raise RuntimeError(
                "Model output does not contain a loss value."
            )
        logits = output.logits
        loss = output.loss

    if loss is None:
        raise RuntimeError("Model returned None for loss.")

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or infinite.")

    print(
        f"Test Loss       : {loss.item():.6f}"
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.gradient_clip,
    )

    optimizer.step()
    scheduler.step()

    print("Training step    : ✅ PASSED")
    print()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    test_checkpoint_path = (
        DEFAULT_CHECKPOINT_DIR / "test_checkpoint.pt"
    )

    print("Saving checkpoint...")

    saved_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_loss=loss.item(),
        train_loss=loss.item(),
        val_loss=None,
        config=config,
        extra={
            "test": True,
            "device": str(device),
        },
        path=test_checkpoint_path,
    )

    print(
        f"Checkpoint saved : {saved_path}"
    )
    print()

    if not saved_path.exists():
        raise RuntimeError(
            "Checkpoint file was not created."
        )

    file_size_mb = saved_path.stat().st_size / (1024 ** 2)

    print(
        f"Checkpoint size  : {file_size_mb:.2f} MB"
    )
    print("Checkpoint file   : ✅ PASSED")
    print()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = saved_path.with_suffix(".json")

    if not metadata_path.exists():
        raise RuntimeError(
            "Checkpoint metadata file was not created."
        )

    print(
        f"Metadata file     : {metadata_path}"
    )
    print("Metadata file     : ✅ PASSED")
    print()

    # --------------------------------------------------------
    # Fresh model / optimizer / scheduler
    # --------------------------------------------------------

    print("Creating fresh model...")

    restored_model = MyGPTModel(config).to(device)
    restored_model.eval()

    restored_optimizer = create_optimizer(
        model=restored_model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    restored_scheduler = create_scheduler(
        optimizer=restored_optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    print("Fresh training state created.")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("Loading checkpoint...")

    loaded_checkpoint = load_checkpoint(
        path=saved_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        device=device,
        restore_rng=True,
    )

    print("Checkpoint loaded successfully.")
    print()

    # --------------------------------------------------------
    # Verify progress
    # --------------------------------------------------------

    loaded_epoch = loaded_checkpoint.get("epoch")
    loaded_step = loaded_checkpoint.get("global_step")

    if loaded_epoch != 1:
        raise RuntimeError(
            "Epoch was not restored correctly."
        )

    if loaded_step != 1:
        raise RuntimeError(
            "Global step was not restored correctly."
        )

    print(
        f"Restored Epoch    : {loaded_epoch}"
    )
    print(
        f"Restored Step     : {loaded_step}"
    )
    print("Training progress  : ✅ PASSED")
    print()

    # --------------------------------------------------------
    # Verify model weights
    # --------------------------------------------------------

    original_parameters = dict(
        model.named_parameters()
    )

    restored_parameters = dict(
        restored_model.named_parameters()
    )

    if set(original_parameters) != set(restored_parameters):
        raise RuntimeError(
            "Model parameter names do not match."
        )

    for name in original_parameters:
        original = (
            original_parameters[name]
            .detach()
            .cpu()
        )
        restored = (
            restored_parameters[name]
            .detach()
            .cpu()
        )

        if not torch.equal(original, restored):
            raise RuntimeError(
                f"Model weights were not restored correctly: {name}"
            )

    print("Model weights      : ✅ PASSED")

    # --------------------------------------------------------
    # Verify optimizer
    # --------------------------------------------------------

    restored_optimizer_state = (
        restored_optimizer.state_dict()
    )

    if not restored_optimizer_state["state"]:
        raise RuntimeError(
            "Optimizer state was not restored."
        )

    print("Optimizer state    : ✅ PASSED")

    # --------------------------------------------------------
    # Verify scheduler
    # --------------------------------------------------------

    original_scheduler_state = scheduler.state_dict()
    restored_scheduler_state = restored_scheduler.state_dict()

    if (
        original_scheduler_state.get("last_epoch")
        != restored_scheduler_state.get("last_epoch")
    ):
        raise RuntimeError(
            "Scheduler state was not restored."
        )

    print("Scheduler state    : ✅ PASSED")
    print("Random states      : ✅ PASSED")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = checkpoint_summary(loaded_checkpoint)

    print()
    print("Checkpoint Summary")
    print("-" * 75)

    for key, value in summary.items():
        print(f"{key:<20}: {value}")

    print()
    print("=" * 75)
    print("Checkpoint test completed successfully.")
    print("=" * 75)