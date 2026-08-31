from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


@dataclass
class InstructionTrainingConfig:
    """
    Configuration for MyGPT2 supervised instruction tuning.
    """

    # =========================================================
    # Paths
    # =========================================================

    project_root: Path = PROJECT_ROOT

    base_checkpoint: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "checkpoints"
        / "mygpt2_base_v1.pt"
    )

    tokenizer_path: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "tokenizer"
        / "tokenizer.json"
    )

    train_dataset: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "instruct_datasets"
        / "train.jsonl"
    )

    validation_dataset: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "instruct_datasets"
        / "validation.jsonl"
    )

    checkpoint_dir: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "instruct_checkpoints"
    )

    # =========================================================
    # Dataset/model
    # =========================================================

    sequence_length: int = 512

    pad_token_id: int = 0

    ignore_index: int = -100

    # =========================================================
    # Optimization
    # =========================================================

    batch_size: int = 8

    gradient_accumulation_steps: int = 4

    learning_rate: float = 2.0e-5

    min_learning_rate: float = 2.0e-6

    weight_decay: float = 0.01

    max_grad_norm: float = 1.0

    warmup_ratio: float = 0.05

    epochs: int = 2

    # =========================================================
    # Runtime
    # =========================================================

    seed: int = 42

    num_workers: int = 0

    device: str = "cuda"

    # FP32 first.
    use_amp: bool = False

    # =========================================================
    # Logging / evaluation
    # =========================================================

    log_every: int = 10

    validate_every: int = 250

    save_every: int = 250

    max_validation_batches: int = 100

    # =========================================================
    # Debug mode
    # =========================================================

    debug_train_samples: int = 500

    debug_validation_samples: int = 200

    debug_max_steps: int = 100

    debug_epochs: int = 10