"""
============================================================
MyGPT2 - Training Engine
============================================================

Purpose
-------
Main training loop for MyGPT2.

Supports:

    • Step-based training
    • Epoch-based training
    • IterableDataset / DataLoader
    • Automatic DataLoader restart
    • Gradient clipping
    • AdamW optimizer
    • Learning-rate scheduler
    • Checkpoint saving/loading
    • Resume training
    • NaN / Inf protection
    • CUDA support
    • Training statistics

Important
---------
MyGPT2 uses an IterableDataset. Therefore a DataLoader can
naturally reach StopIteration even though training should
continue.

For step-based training, the trainer automatically creates a
new iterator when the current DataLoader iterator is exhausted.

This allows:

    --max-steps 10000

to actually reach 10,000 optimization steps instead of
stopping at the number of batches produced by one pass through
the dataset.

============================================================
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

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

    Supports both:

        1. Epoch-based training
        2. Step-based training

    The step-based API is the preferred API for GPT-style
    pretraining.
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

        self.last_gradient_norm: float | None = None

        # ----------------------------------------------------
        # Step-based iterator
        # ----------------------------------------------------

        self._train_iterator: Iterator[Any] | None = None

        self._train_iterator_restarts = 0

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
    # Internal: Get New Training Iterator
    # ========================================================

    def _create_train_iterator(self) -> Iterator[Any]:
        """
        Create a fresh iterator from the training DataLoader.
        """

        if self.train_loader is None:

            raise RuntimeError(
                "Training DataLoader is not available."
            )

        iterator = iter(self.train_loader)

        self._train_iterator_restarts += 1

        return iterator

    # ========================================================
    # Internal: Next Training Batch
    # ========================================================

    def _next_train_batch(self) -> Any:
        """
        Return the next training batch.

        If an IterableDataset is exhausted, automatically
        restart the DataLoader.

        This is critical for long step-based training.
        """

        if self._train_iterator is None:

            self._train_iterator = (
                self._create_train_iterator()
            )

        try:

            return next(
                self._train_iterator
            )

        except StopIteration:

            # ------------------------------------------------
            # Dataset exhausted.
            #
            # Start another pass.
            # ------------------------------------------------

            self.current_epoch += 1

            self._train_iterator = (
                self._create_train_iterator()
            )

            try:

                return next(
                    self._train_iterator
                )

            except StopIteration as exc:

                raise RuntimeError(
                    "Training DataLoader produced "
                    "zero batches even after restarting."
                ) from exc

    # ========================================================
    # Internal: Validate Batch
    # ========================================================

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and unpack a DataLoader batch.
        """

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
                "Expected exactly two batch tensors: "
                "(input_ids, labels)."
            )

        input_ids, labels = batch

        if not torch.is_tensor(input_ids):

            raise TypeError(
                "input_ids must be a torch.Tensor."
            )

        if not torch.is_tensor(labels):

            raise TypeError(
                "labels must be a torch.Tensor."
            )

        if input_ids.ndim != 2:

            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence_length]. "
                f"Received: {tuple(input_ids.shape)}"
            )

        if labels.ndim != 2:

            raise ValueError(
                "labels must have shape "
                "[batch, sequence_length]. "
                f"Received: {tuple(labels.shape)}"
            )

        if input_ids.shape != labels.shape:

            raise ValueError(
                "input_ids and labels must have "
                "the same shape. "
                f"Input: {tuple(input_ids.shape)}, "
                f"Labels: {tuple(labels.shape)}"
            )

        return input_ids, labels

    # ========================================================
    # Internal: Extract Model Loss
    # ========================================================

    @staticmethod
    def _extract_loss(
        output: Any,
    ) -> torch.Tensor:
        """
        Extract loss from the current MyGPT2 model API.

        Supported:

            (logits, loss)

        or:

            object.logits
            object.loss
        """

        if isinstance(output, tuple):

            if len(output) != 2:

                raise RuntimeError(
                    "Expected model output format: "
                    "(logits, loss)."
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
                "Model returned None for loss."
            )

        if not torch.is_tensor(loss):

            raise TypeError(
                "Model loss must be a torch.Tensor."
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "Loss became NaN or infinite."
            )

        return loss

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
        """

        self.model.train()

        # ----------------------------------------------------
        # Move data
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

        loss = self._extract_loss(
            output
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

        if torch.is_tensor(
            gradient_norm
        ):

            self.last_gradient_norm = (
                float(
                    gradient_norm.detach().cpu()
                )
            )

        else:

            self.last_gradient_norm = (
                float(gradient_norm)
            )

        # ----------------------------------------------------
        # Optimizer update
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

        batch_tokens = int(
            input_ids.numel()
        )

        batch_samples = int(
            input_ids.shape[0]
        )

        self.total_tokens += (
            batch_tokens
        )

        self.total_samples += (
            batch_samples
        )

        self.global_step += 1

        loss_value = float(
            loss.detach().item()
        )

        self.current_train_loss = (
            loss_value
        )

        return loss_value

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

        loss = self._extract_loss(
            output
        )

        return float(
            loss.detach().item()
        )

    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
        max_batches: int | None = None,
    ) -> float | None:
        """
        Run validation.

        Parameters
        ----------
        max_batches:
            Optional limit for validation batches.

            Useful for large validation datasets.
        """

        if self.val_loader is None:

            return None

        self.model.eval()

        total_loss = 0.0

        batches = 0

        start_time = time.time()

        for batch in self.val_loader:

            input_ids, labels = (
                self._unpack_batch(batch)
            )

            loss = self.validation_step(
                input_ids,
                labels,
            )

            total_loss += loss

            batches += 1

            if (
                max_batches is not None
                and batches >= max_batches
            ):

                break

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
            f"Batches: {batches} | "
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
        Train for one complete pass through the DataLoader.
        """

        self.model.train()

        epoch_start = time.time()

        total_loss = 0.0

        batches = 0

        print()
        print("=" * 75)

        print(
            f"Epoch {epoch + 1} "
            f"/ {self.config.max_epochs}"
        )

        print("=" * 75)

        for batch_index, batch in enumerate(
            self.train_loader
        ):

            input_ids, labels = (
                self._unpack_batch(batch)
            )

            loss = self.train_step(
                input_ids,
                labels,
            )

            total_loss += loss

            batches += 1

            if (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
            ):

                print(
                    f"Step {self.global_step:>7} | "
                    f"Batch {batch_index + 1:>6} | "
                    f"Loss {loss:.6f} | "
                    f"LR {self.get_learning_rate():.8f}"
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
    # Step-Based Training
    # ========================================================

    def train_steps(
        self,
        max_steps: int,
        *,
        log_every: int = 1,
        save_every: int | None = None,
        save_filename: str = "latest.pt",
        stop_on_dataloader_exhaustion: bool = False,
    ) -> dict[str, Any]:
        """
        Train for exactly max_steps optimization steps.

        This is the preferred training method for GPT-style
        pretraining.

        Important
        ---------
        IterableDataset exhaustion is NOT considered a failure.

        The DataLoader is automatically restarted.

        Parameters
        ----------
        max_steps:
            Number of optimization steps to execute.

        log_every:
            Print progress every N steps.

        save_every:
            Save checkpoint every N steps.

        save_filename:
            Checkpoint filename.

        stop_on_dataloader_exhaustion:
            If True, stop when one DataLoader pass ends.

            Normally this should remain False.
        """

        if max_steps < 1:

            raise ValueError(
                "max_steps must be at least 1."
            )

        if log_every < 1:

            raise ValueError(
                "log_every must be at least 1."
            )

        if (
            save_every is not None
            and save_every < 1
        ):

            raise ValueError(
                "save_every must be at least 1."
            )

        self.model.train()

        if self.training_start_time is None:

            self.training_start_time = (
                time.time()
            )

        start_step = self.global_step

        target_step = (
            start_step + max_steps
        )

        running_loss = 0.0

        steps_completed = 0

        iterator_restarts_before = (
            self._train_iterator_restarts
        )

        print()
        print("=" * 75)

        print(
            "Step-Based Training"
        )

        print("=" * 75)

        print(
            f"Starting Step   : "
            f"{self.global_step}"
        )

        print(
            f"Target Step     : "
            f"{target_step}"
        )

        print(
            f"Remaining Steps : "
            f"{max_steps}"
        )

        print(
            f"Device          : "
            f"{self.device}"
        )

        print("=" * 75)

        while self.global_step < target_step:

            try:

                batch = (
                    self._next_train_batch()
                )

            except RuntimeError:

                if stop_on_dataloader_exhaustion:

                    print(
                        "WARNING: DataLoader "
                        "ended."
                    )

                    break

                raise

            input_ids, labels = (
                self._unpack_batch(batch)
            )

            loss = self.train_step(
                input_ids,
                labels,
            )

            running_loss += loss

            steps_completed += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            should_log = (
                self.global_step % log_every == 0
                or self.global_step == 1
                or self.global_step == target_step
            )

            if should_log:

                elapsed = (
                    time.time()
                    - self.training_start_time
                )

                avg_loss = (
                    running_loss
                    / max(
                        steps_completed,
                        1,
                    )
                )

                print(
                    f"Step "
                    f"{self.global_step:>7} | "
                    f"Loss {loss:.6f} | "
                    f"Avg {avg_loss:.6f} | "
                    f"LR "
                    f"{self.get_learning_rate():.8f} | "
                    f"Time {elapsed:.1f}s"
                )

            # ------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------

            if (
                save_every is not None
                and self.global_step % save_every == 0
            ):

                checkpoint_path = self.save(
                    save_filename
                )

                print(
                    f"Checkpoint saved: "
                    f"{checkpoint_path}"
                )

        average_loss = (
            running_loss
            / max(
                steps_completed,
                1,
            )
        )

        self.current_train_loss = (
            average_loss
        )

        restarts_used = (
            self._train_iterator_restarts
            - iterator_restarts_before
        )

        elapsed = (
            time.time()
            - self.training_start_time
        )

        completed = (
            self.global_step >= target_step
        )

        print()
        print("=" * 75)

        print(
            "Step-Based Training Completed"
        )

        print("=" * 75)

        print(
            f"Steps Completed : "
            f"{steps_completed}"
        )

        print(
            f"Global Step     : "
            f"{self.global_step}"
        )

        print(
            f"Average Loss    : "
            f"{average_loss:.6f}"
        )

        print(
            f"Final Loss      : "
            f"{self.current_train_loss:.6f}"
        )

        print(
            f"LR              : "
            f"{self.get_learning_rate():.8f}"
        )

        print(
            f"Tokens Seen     : "
            f"{self.total_tokens:,}"
        )

        print(
            f"Samples Seen    : "
            f"{self.total_samples:,}"
        )

        print(
            f"DataLoader Restarts : "
            f"{restarts_used}"
        )

        print(
            f"Elapsed Time    : "
            f"{elapsed:.2f}s"
        )

        print(
            f"Status          : "
            f"{'PASSED' if completed else 'INCOMPLETE'}"
        )

        print("=" * 75)

        return {
            "steps": steps_completed,
            "global_step": self.global_step,
            "average_loss": average_loss,
            "final_loss": self.current_train_loss,
            "learning_rate": self.get_learning_rate(),
            "tokens_seen": self.total_tokens,
            "samples_seen": self.total_samples,
            "dataloader_restarts": restarts_used,
            "elapsed_seconds": elapsed,
            "completed": completed,
        }

    # ========================================================
    # Learning Rate
    # ========================================================

    def get_learning_rate(
        self,
    ) -> float:
        """
        Return current learning rate.
        """

        if len(
            self.optimizer.param_groups
        ) == 0:

            return 0.0

        return float(
            self.optimizer
            .param_groups[0]["lr"]
        )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    def save(
        self,
        filename: str = "latest.pt",
    ) -> Path:
        """
        Save current model/training state.
        """

        path = (
            self.checkpoint_dir
            / filename
        )

        best_loss = (
            self.best_val_loss
            if math.isfinite(
                self.best_val_loss
            )
            else None
        )

        train_loss = (
            self.current_train_loss
            if math.isfinite(
                self.current_train_loss
            )
            else None
        )

        val_loss = (
            self.current_val_loss
            if math.isfinite(
                self.current_val_loss
            )
            else None
        )

        return save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_loss=best_loss,
            train_loss=train_loss,
            val_loss=val_loss,
            config=self.config,
            extra={
                "total_tokens": self.total_tokens,
                "total_samples": self.total_samples,
                "last_gradient_norm":
                    self.last_gradient_norm,
                "dataloader_iterator_restarts":
                    self._train_iterator_restarts,
            },
            path=path,
        )

    # ========================================================
    # Load Checkpoint
    # ========================================================

    def load(
        self,
        path: str | Path,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """
        Load and restore a training checkpoint.
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

            self.current_train_loss = float(
                train_loss
            )

        val_loss = checkpoint.get(
            "val_loss"
        )

        if val_loss is not None:

            self.current_val_loss = float(
                val_loss
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

            gradient_norm = extra.get(
                "last_gradient_norm"
            )

            if gradient_norm is not None:

                self.last_gradient_norm = float(
                    gradient_norm
                )

            self._train_iterator_restarts = int(
                extra.get(
                    "dataloader_iterator_restarts",
                    self._train_iterator_restarts,
                )
            )

        # ----------------------------------------------------
        # Never reuse an old DataLoader iterator after loading.
        # ----------------------------------------------------

        self._train_iterator = None

        print()
        print("=" * 75)

        print(
            "Checkpoint Loaded"
        )

        print("=" * 75)

        print(
            f"Checkpoint      : {path}"
        )

        print(
            f"Epoch           : "
            f"{self.current_epoch}"
        )

        print(
            f"Global Step     : "
            f"{self.global_step}"
        )

        print(
            f"Train Loss      : "
            f"{self.current_train_loss}"
        )

        print(
            f"Learning Rate   : "
            f"{self.get_learning_rate():.8f}"
        )

        print(
            f"Tokens Seen     : "
            f"{self.total_tokens:,}"
        )

        print("=" * 75)

        return checkpoint

    # ========================================================
    # Full Epoch Training
    # ========================================================

    def train(
        self,
        *,
        save_every_epoch: bool = True,
        validate_every_epoch: bool = True,
    ) -> None:
        """
        Run traditional epoch-based training.

        For GPT pretraining with a large IterableDataset,
        train_steps() is preferred.
        """

        self.training_start_time = (
            time.time()
        )

        print()
        print("=" * 75)

        print(
            "MyGPT2 Training Started"
        )

        print("=" * 75)

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

        for epoch in range(
            self.current_epoch,
            self.config.max_epochs,
        ):

            self.current_epoch = epoch

            train_loss = self.train_epoch(
                epoch
            )

            val_loss = None

            if validate_every_epoch:

                val_loss = self.validate()

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if (
                val_loss is not None
                and val_loss < self.best_val_loss
            ):

                self.best_val_loss = val_loss

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
            print("-" * 75)

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

            print("-" * 75)

        total_time = (
            time.time()
            - self.training_start_time
        )

        print()
        print("=" * 75)

        print(
            "MyGPT2 Training Completed"
        )

        print("=" * 75)

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

        print("=" * 75)


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

    # --------------------------------------------------------
    # Model
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
    # Test batches
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
        "Trainer step     : PASSED"
    )

    print()

    # --------------------------------------------------------
    # Step-based restart test
    # --------------------------------------------------------

    print(
        "Testing DataLoader restart..."
    )

    result = trainer.train_steps(
        max_steps=5,
        log_every=1,
    )

    if result["completed"]:

        print(
            "Step training    : PASSED"
        )

    else:

        raise RuntimeError(
            "Step training did not reach target."
        )

    print()

    # --------------------------------------------------------
    # Checkpoint
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

    if not checkpoint_path.exists():

        raise RuntimeError(
            "Checkpoint file was not created."
        )

    print(
        "Checkpoint save  : PASSED"
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