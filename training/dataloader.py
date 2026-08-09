"""
============================================================
MyGPT2 - DataLoader
============================================================

Purpose
-------
Creates PyTorch DataLoaders for the MyGPT2 training pipeline.

Pipeline:

    Raw Dataset
        |
        v
    GPTTextDataset
        |
        v
    Individual Samples
        |
        v
    PyTorch DataLoader
        |
        v
    Batches
        |
        v
    GPU
        |
        v
    MyGPT2 Model

Example:

    Individual sample:

        Input  ->  [512]
        Target ->  [512]

    Batch:

        Input  ->  [4, 512]
        Target ->  [4, 512]

============================================================
"""

from __future__ import annotations


# ============================================================
# Standard Library
# ============================================================

import sys
from pathlib import Path
from typing import Optional


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

from torch.utils.data import DataLoader


# ============================================================
# Project Imports
# ============================================================

from tokenizer.my_tokenizer import MyGPTTokenizer

from training.dataset import (
    GPTTextDataset,
    DEFAULT_DATASETS,
)


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_SEQUENCE_LENGTH = 512

DEFAULT_BATCH_SIZE = 4

DEFAULT_NUM_WORKERS = 0


# ============================================================
# MyGPT DataLoader
# ============================================================

class MyGPTDataLoader:
    """
    Wrapper around PyTorch DataLoader.

    This class creates the DataLoader used by the
    MyGPT2 training pipeline.

    Parameters
    ----------
    tokenizer:
        Loaded MyGPTTokenizer.

    sequence_length:
        Number of tokens in each training sequence.

    batch_size:
        Number of training sequences per batch.

    max_documents:
        Optional document limit for testing.

    stride:
        Distance between consecutive token windows.

    num_workers:
        Number of worker processes.

        Start with 0 on Windows for reliability.

    pin_memory:
        Enables faster CPU -> GPU transfer.

    drop_last:
        Drops incomplete final batch.

    """

    def __init__(
        self,
        tokenizer: MyGPTTokenizer,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_documents: Optional[int] = None,
        stride: Optional[int] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not tokenizer.is_loaded:

            raise RuntimeError(
                "Tokenizer must be loaded before "
                "creating the DataLoader."
            )

        if sequence_length < 2:

            raise ValueError(
                "sequence_length must be at least 2."
            )

        if batch_size < 1:

            raise ValueError(
                "batch_size must be at least 1."
            )

        if num_workers < 0:

            raise ValueError(
                "num_workers cannot be negative."
            )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        self.tokenizer = tokenizer

        self.sequence_length = (
            sequence_length
        )

        self.batch_size = batch_size

        self.max_documents = (
            max_documents
        )

        self.stride = stride

        self.num_workers = num_workers

        self.pin_memory = (
            pin_memory
        )

        self.drop_last = (
            drop_last
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        self.dataset = GPTTextDataset(
            dataset_paths=list(
                DEFAULT_DATASETS.values()
            ),
            tokenizer=self.tokenizer,
            sequence_length=self.sequence_length,
            max_documents=self.max_documents,
            stride=self.stride,
        )

        # ----------------------------------------------------
        # DataLoader
        # ----------------------------------------------------

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    # ========================================================
    # Iterator
    # ========================================================

    def __iter__(self):

        return iter(self.loader)

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.loader)

    # ========================================================
    # Get Loader
    # ========================================================

    def get_loader(self) -> DataLoader:

        return self.loader

    # ========================================================
    # Get Dataset
    # ========================================================

    def get_dataset(self) -> GPTTextDataset:

        return self.dataset

    # ========================================================
    # Statistics
    # ========================================================

    def get_statistics(self) -> dict:

        return self.dataset.get_statistics()


# ============================================================
# Factory Function
# ============================================================

def create_dataloader(
    tokenizer: MyGPTTokenizer,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_documents: Optional[int] = None,
    stride: Optional[int] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create and return a PyTorch DataLoader.

    This is the simplest interface for the training
    pipeline.
    """

    dataset = GPTTextDataset(
        dataset_paths=list(
            DEFAULT_DATASETS.values()
        ),
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_documents=max_documents,
        stride=stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader


# ============================================================
# GPU Transfer Helper
# ============================================================

def move_batch_to_device(
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Move a batch to the selected device.

    Parameters
    ----------
    batch:
        Tuple containing:

            input_ids
            target_ids

    device:
        CPU or CUDA device.

    Returns
    -------
    input_ids:
        Tensor on selected device.

    target_ids:
        Tensor on selected device.
    """

    input_ids, target_ids = batch

    input_ids = input_ids.to(
        device,
        non_blocking=True,
    )

    target_ids = target_ids.to(
        device,
        non_blocking=True,
    )

    return (
        input_ids,
        target_ids,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "MyGPT2 DataLoader Test"
    )

    print("=" * 70)

    print()

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
            "Running DataLoader test on CPU."
        )

    print()

    # --------------------------------------------------------
    # Tokenizer path
    # --------------------------------------------------------

    tokenizer_path = (
        PROJECT_ROOT
        / "artifacts"
        / "tokenizer"
        / "tokenizer.json"
    )

    if not tokenizer_path.exists():

        raise FileNotFoundError(
            "Tokenizer file not found:\n"
            f"{tokenizer_path}"
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        MyGPTTokenizer.load(
            tokenizer_path
        )
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocabulary_size:,}"
    )

    print()

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    sequence_length = 32

    batch_size = 4

    max_documents = 100

    print(
        "DataLoader configuration:"
    )

    print(
        f"  Sequence Length : "
        f"{sequence_length}"
    )

    print(
        f"  Batch Size      : "
        f"{batch_size}"
    )

    print(
        f"  Max Documents   : "
        f"{max_documents}"
    )

    print(
        f"  Workers         : "
        f"{DEFAULT_NUM_WORKERS}"
    )

    print()

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    loader = create_dataloader(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_documents=max_documents,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    print(
        "DataLoader created successfully."
    )

    print()

    # --------------------------------------------------------
    # Get first batch
    # --------------------------------------------------------

    print(
        "Loading first batch..."
    )

    batch = next(
        iter(loader)
    )

    input_ids, target_ids = (
        move_batch_to_device(
            batch,
            device,
        )
    )

    # --------------------------------------------------------
    # Batch information
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "First Batch"
    )

    print("=" * 70)

    print(
        f"Input Shape      : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Target Shape     : "
        f"{tuple(target_ids.shape)}"
    )

    print(
        f"Input Device     : "
        f"{input_ids.device}"
    )

    print(
        f"Target Device    : "
        f"{target_ids.device}"
    )

    print(
        f"Input DType      : "
        f"{input_ids.dtype}"
    )

    print(
        f"Target DType     : "
        f"{target_ids.dtype}"
    )

    print()

    # --------------------------------------------------------
    # Display first sequence
    # --------------------------------------------------------

    print(
        "First Input Sequence:"
    )

    print(
        input_ids[0].tolist()
    )

    print()

    print(
        "First Target Sequence:"
    )

    print(
        target_ids[0].tolist()
    )

    print()

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
    )

    actual_input_shape = (
        tuple(input_ids.shape)
    )

    actual_target_shape = (
        tuple(target_ids.shape)
    )

    print("=" * 70)

    print(
        "Validation"
    )

    print("=" * 70)

    if actual_input_shape == expected_shape:

        print(
            "Input Shape  : ✅ Correct"
        )

    else:

        print(
            "Input Shape  : ❌ Incorrect"
        )

    if actual_target_shape == expected_shape:

        print(
            "Target Shape : ✅ Correct"
        )

    else:

        print(
            "Target Shape : ❌ Incorrect"
        )

    # --------------------------------------------------------
    # Verify device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        if input_ids.device.type == "cuda":

            print(
                "GPU Transfer : ✅ Successful"
            )

        else:

            print(
                "GPU Transfer : ❌ Failed"
            )

    else:

        print(
            "GPU Transfer : ⚠️ CUDA unavailable"
        )

    # --------------------------------------------------------
    # Verify token shifting
    # --------------------------------------------------------

    shift_valid = torch.equal(
        input_ids[0, 1:],
        target_ids[0, :-1],
    )

    if shift_valid:

        print(
            "Token Shift  : ✅ Correct"
        )

    else:

        print(
            "Token Shift  : ❌ Incorrect"
        )

    print()

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        )

        print(
            f"GPU Memory Allocated : "
            f"{allocated:.2f} MB"
        )

        print(
            f"GPU Memory Reserved  : "
            f"{reserved:.2f} MB"
        )

        print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "Dataset Statistics"
    )

    print("=" * 70)

    stats = (
        loader.dataset.get_statistics()
    )

    for key, value in stats.items():

        print(
            f"{key:25}: {value:,}"
        )

    print()

    print("=" * 70)

    print(
        "DataLoader test completed successfully."
    )

    print("=" * 70)