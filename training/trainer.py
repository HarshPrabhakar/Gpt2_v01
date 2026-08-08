"""
============================================================
MyGPT2 - Training Engine
============================================================

Purpose
-------
Main training loop for MyGPT2.

This module connects:

    Dataset
        ↓
    DataLoader
        ↓
    GPT Model
        ↓
    Forward Pass
        ↓
    Loss
        ↓
    Backward Pass
        ↓
    Gradient Clipping
        ↓
    AdamW
        ↓
    Learning Rate Scheduler
        ↓
    Checkpointing

The trainer is responsible for the actual optimization
process. Model architecture, dataset creation, optimizer,
scheduler and checkpoint implementation remain in their
respective modules.

============================================================
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from model.config import GPTConfig
from model.model import MyGPTModel

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


# ============================================================
# Trainer
# ============================================================

class Trainer:
    """
    Main MyGPT2 training engine.

    Responsibilities
    ----------------
    • Model training
    • Forward pass
    • Loss calculation
    • Backward pass
    • Gradient clipping
    • Optimizer updates
    • Scheduler updates
    • Validation
    • Checkpoint saving
    • Checkpoint loading
    • Training statistics
    """

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        config: GPTConfig,
        *,
        val_loader: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
    ) -> None:

        self.config = config

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:

            if torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")

        else:
            self.device = torch.device(device)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = model.to(self.device)

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        self.train_loader = train_loader
        self.val_loader = val_loader

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        if optimizer is None:

            self.optimizer = create_optimizer(
                model=self.model,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        else:

            self.optimizer = optimizer

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if scheduler is None:

            self.scheduler = None

        else:

            self.scheduler = scheduler

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        self.current_epoch = 0

        self.global_step = 0

        self.best_val_loss = math.inf

        self.current_train_loss = math.inf

        self.current_val_loss = math.inf

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_tokens = 0

        self.total_samples = 0

        self.training_start_time: float | None = None

        # ----------------------------------------------------
        # Checkpoint directory
        # ----------------------------------------------------

        self.checkpoint_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Training Step
    # ========================================================

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Execute one complete optimization step.

        Steps:

            1. Zero gradients
            2. Forward pass
            3. Calculate loss
            4. Backward pass
            5. Gradient clipping
            6. Optimizer update
            7. Scheduler update
        """

        # ----------------------------------------------------
        # Training mode
        # ----------------------------------------------------

        self.model.train()

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        labels = labels.to(
            self.device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = self.model(
            input_ids=input_ids,
            labels=labels,
        )

        # ----------------------------------------------------
        # Current MyGPT2 API:
        #
        #     logits, loss
        #
        # Also support an object with .loss.
        # ----------------------------------------------------

        if isinstance(output, tuple):

            if len(output) != 2:

                raise RuntimeError(
                    "Expected model output "
                    "format: (logits, loss)."
                )

            logits, loss = output

        else:

            if not hasattr(output, "loss"):

                raise RuntimeError(
                    "Model output does not contain "
                    "a loss value."
                )

            logits = output.logits
            loss = output.loss

        # ----------------------------------------------------
        # Validate loss
        # ----------------------------------------------------

        if loss is None:

            raise RuntimeError(
                "Model returned None for loss."
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Loss became NaN or infinite."
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer.step()

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if self.scheduler is not None:

            self.scheduler.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_tokens = (
            input_ids.numel()
        )

        self.total_tokens += (
            batch_tokens
        )

        self.total_samples += (
            input_ids.shape[0]
        )

        self.global_step += 1

        self.current_train_loss = (
            loss.item()
        )

        return loss.item()

    # ========================================================
    # Validation Step
    # ========================================================

    @torch.no_grad()
    def validation_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Execute one validation step.

        No gradients are calculated.
        """

        self.model.eval()

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        labels = labels.to(
            self.device,
            non_blocking=True,
        )

        output = self.model(
            input_ids=input_ids,
            labels=labels,
        )

        if isinstance(output, tuple):

            if len(output) != 2:

                raise RuntimeError(
                    "Expected model output "
                    "format: (logits, loss)."
                )

            _, loss = output

        else:

            if not hasattr(output, "loss"):

                raise RuntimeError(
                    "Model output does not contain "
                    "a loss value."
                )

            loss = output.loss

        if loss is None:

            raise RuntimeError(
                "Validation returned None loss."
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Validation loss became "
                "NaN or infinite."
            )

        return loss.item()

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate(
        self,
    ) -> float | None:
        """
        Run validation across the entire validation loader.
        """

        if self.val_loader is None:

            return None

        self.model.eval()

        total_loss = 0.0

        batches = 0

        start_time = time.time()

        for batch in self.val_loader:

            # ------------------------------------------------
            # Support tuple/list batches.
            # ------------------------------------------------

            if not isinstance(
                batch,
                (tuple, list),
            ):

                raise RuntimeError(
                    "DataLoader batch must be "
                    "(input_ids, labels)."
                )

            if len(batch) != 2:

                raise RuntimeError(
                    "Expected DataLoader batch "
                    "to contain input_ids and labels."
                )

            input_ids, labels = batch

            loss = self.validation_step(
                input_ids,
                labels,
            )

            total_loss += loss

            batches += 1

        if batches == 0:

            return None

        average_loss = (
            total_loss / batches
        )

        self.current_val_loss = (
            average_loss
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Validation completed | "
            f"Loss: {average_loss:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        return average_loss

    # ========================================================
    # Train Epoch
    # ========================================================

    def train_epoch(
        self,
        epoch: int,
    ) -> float:
        """
        Train the model for one complete epoch.
        """

        self.model.train()

        epoch_start = time.time()

        total_loss = 0.0

        batches = 0

        print()
        print(
            "=" * 75
        )

        print(
            f"Epoch {epoch + 1} "
            f"/ {self.config.max_epochs}"
        )

        print(
            "=" * 75
        )

        for batch_index, batch in enumerate(
            self.train_loader
        ):

            # ------------------------------------------------
            # Validate batch
            # ------------------------------------------------

            if not isinstance(
                batch,
                (tuple, list),
            ):

                raise RuntimeError(
                    "DataLoader batch must "
                    "be (input_ids, labels)."
                )

            if len(batch) != 2:

                raise RuntimeError(
                    "Expected exactly two "
                    "batch tensors."
                )

            input_ids, labels = batch

            # ------------------------------------------------
            # Training step
            # ------------------------------------------------

            loss = self.train_step(
                input_ids,
                labels,
            )

            total_loss += loss

            batches += 1

            # ------------------------------------------------
            # Progress logging
            # ------------------------------------------------

            if (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
            ):

                current_lr = (
                    self.get_learning_rate()
                )

                elapsed = (
                    time.time()
                    - epoch_start
                )

                print(
                    f"Step {self.global_step:>7} | "
                    f"Batch {batch_index + 1:>6} | "
                    f"Loss {loss:.6f} | "
                    f"LR {current_lr:.8f} | "
                    f"Time {elapsed:.1f}s"
                )

        if batches == 0:

            raise RuntimeError(
                "Training DataLoader produced "
                "zero batches."
            )

        average_loss = (
            total_loss / batches
        )

        self.current_train_loss = (
            average_loss
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch + 1} completed."
        )

        print(
            f"Training Loss : "
            f"{average_loss:.6f}"
        )

        print(
            f"Epoch Time    : "
            f"{epoch_time:.2f}s"
        )

        return average_loss

    # ========================================================
    # Learning Rate
    # ========================================================

    def get_learning_rate(
        self,
    ) -> float:
        """
        Return the current learning rate.
        """

        if len(
            self.optimizer.param_groups
        ) == 0:

            return 0.0

        return float(
            self.optimizer
            .param_groups[0]
            ["lr"]
        )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    def save(
        self,
        filename: str = "latest.pt",
    ) -> Path:
        """
        Save current training state.
        """

        path = (
            self.checkpoint_dir
            / filename
        )

        saved_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_loss=(
                self.best_val_loss
                if math.isfinite(
                    self.best_val_loss
                )
                else None
            ),
            train_loss=(
                self.current_train_loss
                if math.isfinite(
                    self.current_train_loss
                )
                else None
            ),
            val_loss=(
                self.current_val_loss
                if math.isfinite(
                    self.current_val_loss
                )
                else None
            ),
            config=self.config,
            extra={
                "total_tokens":
                    self.total_tokens,

                "total_samples":
                    self.total_samples,
            },
            path=path,
        )

        return saved_path

    # ========================================================
    # Load Checkpoint
    # ========================================================

    def load(
        self,
        path: str | Path,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """
        Resume training from a checkpoint.
        """

        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            restore_rng=restore_rng,
        )

        self.current_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        self.global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )

        best_loss = checkpoint.get(
            "best_loss"
        )

        if best_loss is not None:

            self.best_val_loss = float(
                best_loss
            )

        train_loss = checkpoint.get(
            "train_loss"
        )

        if train_loss is not None:

            self.current_train_loss = (
                float(train_loss)
            )

        val_loss = checkpoint.get(
            "val_loss"
        )

        if val_loss is not None:

            self.current_val_loss = (
                float(val_loss)
            )

        extra = checkpoint.get(
            "extra",
            {},
        )

        if isinstance(extra, dict):

            self.total_tokens = int(
                extra.get(
                    "total_tokens",
                    self.total_tokens,
                )
            )

            self.total_samples = int(
                extra.get(
                    "total_samples",
                    self.total_samples,
                )
            )

        return checkpoint

    # ========================================================
    # Full Training
    # ========================================================

    def train(
        self,
        *,
        save_every_epoch: bool = True,
        validate_every_epoch: bool = True,
    ) -> None:
        """
        Run the complete training process.
        """

        self.training_start_time = (
            time.time()
        )

        print()
        print(
            "=" * 75
        )

        print(
            "MyGPT2 Training Started"
        )

        print(
            "=" * 75
        )

        print(
            f"Device          : "
            f"{self.device}"
        )

        print(
            f"Epochs          : "
            f"{self.config.max_epochs}"
        )

        print(
            f"Batch Size      : "
            f"{self.config.batch_size}"
        )

        print(
            f"Learning Rate   : "
            f"{self.config.learning_rate}"
        )

        print()

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(
            self.current_epoch,
            self.config.max_epochs,
        ):

            self.current_epoch = (
                epoch
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            train_loss = self.train_epoch(
                epoch
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_loss = None

            if validate_every_epoch:

                val_loss = self.validate()

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if (
                val_loss is not None
                and val_loss
                < self.best_val_loss
            ):

                self.best_val_loss = (
                    val_loss
                )

                best_path = self.save(
                    "best.pt"
                )

                print(
                    f"New best model saved: "
                    f"{best_path}"
                )

            # ------------------------------------------------
            # Latest checkpoint
            # ------------------------------------------------

            if save_every_epoch:

                latest_path = self.save(
                    "latest.pt"
                )

                print(
                    f"Latest checkpoint saved: "
                    f"{latest_path}"
                )

            # ------------------------------------------------
            # Epoch summary
            # ------------------------------------------------

            print()
            print(
                "-" * 75
            )

            print(
                f"Epoch {epoch + 1} Summary"
            )

            print(
                f"Train Loss      : "
                f"{train_loss:.6f}"
            )

            if val_loss is not None:

                print(
                    f"Validation Loss : "
                    f"{val_loss:.6f}"
                )

            else:

                print(
                    "Validation Loss : N/A"
                )

            print(
                f"Learning Rate   : "
                f"{self.get_learning_rate():.8f}"
            )

            print(
                f"Global Step     : "
                f"{self.global_step}"
            )

            print(
                f"Tokens Seen     : "
                f"{self.total_tokens:,}"
            )

            print(
                "-" * 75
            )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        total_time = (
            time.time()
            - self.training_start_time
        )

        print()
        print(
            "=" * 75
        )

        print(
            "MyGPT2 Training Completed"
        )

        print(
            "=" * 75
        )

        print(
            f"Final Epoch     : "
            f"{self.current_epoch + 1}"
        )

        print(
            f"Global Steps    : "
            f"{self.global_step}"
        )

        print(
            f"Tokens Seen     : "
            f"{self.total_tokens:,}"
        )

        print(
            f"Final Train Loss: "
            f"{self.current_train_loss:.6f}"
        )

        if math.isfinite(
            self.best_val_loss
        ):

            print(
                f"Best Val Loss   : "
                f"{self.best_val_loss:.6f}"
            )

        print(
            f"Total Time      : "
            f"{total_time / 3600:.2f} hours"
        )

        print(
            "=" * 75
        )


# ============================================================
# Standalone Trainer Test
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("MyGPT2 Trainer Test")
    print("=" * 75)
    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "CUDA available : YES"
        )

        print(
            "GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "CUDA available : NO"
        )

        print(
            "Running trainer test on CPU."
        )

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = GPTConfig()

    # Use a tiny test batch rather than loading the complete
    # dataset just to verify Trainer functionality.
    # --------------------------------------------------------

    print(
        "Creating test model..."
    )

    model = MyGPTModel(
        config
    ).to(device)

    print(
        "Model created successfully."
    )

    print()

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_batches = []

    for _ in range(2):

        input_ids = torch.randint(
            0,
            config.vocab_size,
            (
                config.batch_size,
                32,
            ),
            dtype=torch.long,
        )

        labels = torch.randint(
            0,
            config.vocab_size,
            (
                config.batch_size,
                32,
            ),
            dtype=torch.long,
        )

        test_batches.append(
            (
                input_ids,
                labels,
            )
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=test_batches,
        config=config,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print(
        "Trainer created successfully."
    )

    print()

    # --------------------------------------------------------
    # One training step
    # --------------------------------------------------------

    print(
        "Running trainer step..."
    )

    test_input, test_labels = (
        test_batches[0]
    )

    loss = trainer.train_step(
        test_input,
        test_labels,
    )

    print(
        f"Test Loss       : "
        f"{loss:.6f}"
    )

    print(
        f"Global Step     : "
        f"{trainer.global_step}"
    )

    print(
        f"Learning Rate   : "
        f"{trainer.get_learning_rate():.8f}"
    )

    print(
        "Trainer step     : ✅ PASSED"
    )

    print()

    # --------------------------------------------------------
    # Epoch test
    # --------------------------------------------------------

    print(
        "Running one epoch test..."
    )

    epoch_loss = trainer.train_epoch(
        epoch=0
    )

    print(
        f"Epoch Loss      : "
        f"{epoch_loss:.6f}"
    )

    print(
        "Epoch training   : ✅ PASSED"
    )

    print()

    # --------------------------------------------------------
    # Checkpoint test
    # --------------------------------------------------------

    print(
        "Saving trainer checkpoint..."
    )

    checkpoint_path = trainer.save(
        "trainer_test.pt"
    )

    print(
        f"Checkpoint      : "
        f"{checkpoint_path}"
    )

    print(
        "Checkpoint save  : ✅ PASSED"
    )

    print()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("=" * 75)
    print(
        "Trainer test completed successfully."
    )
    print("=" * 75)