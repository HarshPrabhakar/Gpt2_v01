"""
============================================================
MyGPT2 - Checkpoint Validation Evaluator
============================================================

Purpose
-------
Evaluate a trained MyGPT2 checkpoint on held-out dataset
splits that were not used during training.

Features
--------
1. Automatically finds the highest-step checkpoint.
2. Restores checkpoint model configuration.
3. Loads the project's trained tokenizer.
4. Loads exact checkpoint model weights.
5. Verifies every model tensor against the checkpoint.
6. Discovers safe validation/test splits.
7. Balances evaluation across available datasets.
8. Uses the same autoregressive next-token objective as training.
9. Computes:
       - Combined validation loss
       - Combined perplexity
       - Per-dataset loss
       - Per-dataset perplexity
       - Evaluation throughput
       - Peak GPU memory
10. Saves evaluation results to JSON.

Important
---------
The evaluator never intentionally evaluates unnamed/plain datasets
because they may contain training data.

============================================================
"""

from __future__ import annotations


# ============================================================
# Standard Library
# ============================================================

import argparse
import json
import math
import re
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ============================================================
# Third-Party
# ============================================================

import torch
import torch.nn.functional as F

from datasets import (
    Dataset,
    DatasetDict,
    load_from_disk,
)


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Project Imports
# ============================================================

from model.config import GPTConfig
from model.model import MyGPTModel
from tokenizer.my_tokenizer import MyGPTTokenizer


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)


CHECKPOINT_DIRECTORIES = [

    PROJECT_ROOT
    / "artifacts"
    / "checkpoints",

    PROJECT_ROOT
    / "checkpoints",

]


RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


# ============================================================
# Dataset Paths
# ============================================================

DATASET_PATHS = {

    "TinyStories": (
        PROJECT_ROOT
        / "datasets"
        / "TinyStories"
        / "raw"
    ),

    "WikiText103": (
        PROJECT_ROOT
        / "datasets"
        / "WikiText103"
        / "raw"
    ),

    "OpenWebText": (
        PROJECT_ROOT
        / "datasets"
        / "OpenWebText"
        / "raw"
    ),

    "FineWeb": (
        PROJECT_ROOT
        / "datasets"
        / "FineWeb"
        / "raw"
    ),

}


# ============================================================
# Evaluation Defaults
# ============================================================

DEFAULT_BATCH_SIZE = 8

DEFAULT_MAX_EVAL_TOKENS = 100_000

DEFAULT_SEED = 42


# ============================================================
# Safe Evaluation Split Names
# ============================================================

VALIDATION_SPLIT_NAMES = (

    "validation",
    "valid",
    "dev",
    "eval",
    "evaluation",

)


TEST_SPLIT_NAMES = (

    "test",

)


# ============================================================
# Printing Utility
# ============================================================

def print_header(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Checkpoint Step Extraction
# ============================================================

def extract_checkpoint_step(
    checkpoint_path: Path,
) -> int:

    """
    Determine checkpoint step.

    Priority:
        1. JSON sidecar global_step
        2. Numeric value in filename
        3. -1 when unavailable
    """

    metadata_path = (
        checkpoint_path.with_suffix(
            ".json"
        )
    )


    # --------------------------------------------------------
    # JSON sidecar
    # --------------------------------------------------------

    if metadata_path.exists():

        try:

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            step = metadata.get(
                "global_step"
            )

            if step is not None:
                return int(step)

        except Exception:
            pass


    # --------------------------------------------------------
    # Filename fallback
    # --------------------------------------------------------

    numbers = re.findall(
        r"\d+",
        checkpoint_path.stem,
    )

    if numbers:

        try:
            return int(
                numbers[-1]
            )

        except ValueError:
            pass


    return -1


# ============================================================
# Locate Latest Checkpoint
# ============================================================

def locate_latest_checkpoint() -> Path:

    """
    Find the checkpoint with the highest discovered global step.
    """

    candidates: list[Path] = []


    for directory in CHECKPOINT_DIRECTORIES:

        if not directory.exists():
            continue

        candidates.extend(
            directory.glob("*.pt")
        )


    if not candidates:

        searched = "\n".join(
            str(path)
            for path in CHECKPOINT_DIRECTORIES
        )

        raise FileNotFoundError(
            "No checkpoint files were found.\n\n"
            "Searched:\n"
            f"{searched}"
        )


    ranked = [

        (
            extract_checkpoint_step(path),
            path.stat().st_mtime,
            path,
        )

        for path in candidates

    ]


    ranked.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )


    return ranked[0][2]


# ============================================================
# Load Checkpoint
# ============================================================

def load_raw_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:

    """
    Load checkpoint on CPU.
    """

    try:

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

            weights_only=False,

        )

    except TypeError:

        # Compatibility fallback for older PyTorch versions.

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu",

        )


    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint root must be a dictionary."
        )


    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    return checkpoint


# ============================================================
# Restore Configuration
# ============================================================

def restore_config(
    checkpoint: dict[str, Any],
) -> GPTConfig:

    """
    Restore configuration saved inside checkpoint.
    """

    config = GPTConfig()

    saved_config = checkpoint.get(
        "config"
    )


    if saved_config is None:

        print(
            "Warning: checkpoint contains no saved config."
        )

        return config


    # --------------------------------------------------------
    # Saved dictionary
    # --------------------------------------------------------

    if isinstance(
        saved_config,
        dict,
    ):

        for key, value in (
            saved_config.items()
        ):

            if hasattr(
                config,
                key,
            ):

                setattr(
                    config,
                    key,
                    value,
                )


        return config


    # --------------------------------------------------------
    # Saved config object
    # --------------------------------------------------------

    for key in vars(config):

        if hasattr(
            saved_config,
            key,
        ):

            setattr(
                config,
                key,
                getattr(
                    saved_config,
                    key,
                ),
            )


    return config


# ============================================================
# Load Tokenizer
# ============================================================

def load_tokenizer(
    config: GPTConfig,
) -> MyGPTTokenizer:

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Tokenizer file not found:\n"
            f"{TOKENIZER_PATH}"
        )


    tokenizer = MyGPTTokenizer.load(
        TOKENIZER_PATH
    )


    tokenizer_vocab_size = int(
        tokenizer.vocabulary_size
    )

    model_vocab_size = int(
        config.vocab_size
    )


    if (
        tokenizer_vocab_size
        !=
        model_vocab_size
    ):

        raise RuntimeError(
            "Tokenizer/model vocabulary mismatch.\n\n"
            f"Tokenizer vocabulary : "
            f"{tokenizer_vocab_size:,}\n"
            f"Model vocabulary     : "
            f"{model_vocab_size:,}"
        )


    return tokenizer


# ============================================================
# Load Model
# ============================================================

def load_model(
    checkpoint: dict[str, Any],
    config: GPTConfig,
    device: torch.device,
) -> MyGPTModel:

    """
    Construct model and load exact trained checkpoint weights.
    """

    model = MyGPTModel(
        config
    )


    checkpoint_state = checkpoint[
        "model_state_dict"
    ]


    load_result = model.load_state_dict(

        checkpoint_state,

        strict=True,

    )


    if load_result.missing_keys:

        raise RuntimeError(
            "Missing model keys:\n"
            +
            "\n".join(
                load_result.missing_keys
            )
        )


    if load_result.unexpected_keys:

        raise RuntimeError(
            "Unexpected model keys:\n"
            +
            "\n".join(
                load_result.unexpected_keys
            )
        )


    model = model.to(
        device
    )


    model.eval()


    return model


# ============================================================
# Verify Loaded Model
# ============================================================

def verify_model_weights(
    model: MyGPTModel,
    checkpoint: dict[str, Any],
) -> int:

    """
    Verify every loaded model tensor exactly equals checkpoint.
    """

    checkpoint_state = checkpoint[
        "model_state_dict"
    ]

    model_state = model.state_dict()


    checkpoint_keys = set(
        checkpoint_state.keys()
    )

    model_keys = set(
        model_state.keys()
    )


    if checkpoint_keys != model_keys:

        checkpoint_only = sorted(
            checkpoint_keys
            -
            model_keys
        )

        model_only = sorted(
            model_keys
            -
            checkpoint_keys
        )


        raise RuntimeError(
            "Checkpoint and model parameter names differ.\n\n"
            f"Checkpoint-only keys: "
            f"{checkpoint_only}\n\n"
            f"Model-only keys: "
            f"{model_only}"
        )


    matching_tensors = 0


    for name in checkpoint_state:

        checkpoint_tensor = (
            checkpoint_state[name]
            .detach()
            .cpu()
        )

        model_tensor = (
            model_state[name]
            .detach()
            .cpu()
        )


        if checkpoint_tensor.shape != model_tensor.shape:

            raise RuntimeError(
                f"Tensor shape mismatch: {name}\n"
                f"Checkpoint: "
                f"{tuple(checkpoint_tensor.shape)}\n"
                f"Model     : "
                f"{tuple(model_tensor.shape)}"
            )


        if not torch.equal(
            checkpoint_tensor,
            model_tensor,
        ):

            maximum_difference = (

                checkpoint_tensor.float()
                -
                model_tensor.float()

            ).abs().max().item()


            raise RuntimeError(
                "Loaded model tensor does not exactly "
                "match checkpoint.\n\n"
                f"Tensor             : {name}\n"
                f"Maximum difference : "
                f"{maximum_difference:.10f}"
            )


        matching_tensors += 1


    return matching_tensors


# ============================================================
# Detect Dataset Text Column
# ============================================================

def detect_text_column(
    dataset: Dataset,
) -> str:

    possible_columns = (

        "text",
        "content",
        "body",
        "article",
        "story",
        "document",

    )


    for column in possible_columns:

        if column in dataset.column_names:

            return column


    raise RuntimeError(
        "Unable to identify a text column.\n"
        f"Columns: {dataset.column_names}"
    )


# ============================================================
# Choose Safe Evaluation Split
# ============================================================

def choose_evaluation_split(
    dataset: Dataset | DatasetDict,
    dataset_name: str,
) -> tuple[str, Dataset] | None:

    """
    Prefer validation split.

    Test split is used only when no validation split exists.

    Plain Dataset objects are deliberately skipped because
    they cannot be proven to be held-out from training.
    """

    if not isinstance(
        dataset,
        DatasetDict,
    ):

        print(
            f"  {dataset_name:<16}: "
            "SKIPPED "
            "(no named validation/test split)"
        )

        return None


    available = {

        str(name).lower(): name

        for name in dataset.keys()

    }


    # --------------------------------------------------------
    # Validation preferred
    # --------------------------------------------------------

    for wanted in VALIDATION_SPLIT_NAMES:

        if wanted in available:

            actual_name = available[
                wanted
            ]

            return (
                str(actual_name),
                dataset[actual_name],
            )


    # --------------------------------------------------------
    # Test fallback
    # --------------------------------------------------------

    for wanted in TEST_SPLIT_NAMES:

        if wanted in available:

            actual_name = available[
                wanted
            ]

            return (
                str(actual_name),
                dataset[actual_name],
            )


    available_string = ", ".join(
        str(name)
        for name in dataset.keys()
    )


    print(
        f"  {dataset_name:<16}: "
        "SKIPPED "
        f"(available splits: "
        f"{available_string})"
    )


    return None


# ============================================================
# Discover Validation Sources
# ============================================================

def load_validation_sources() -> list[dict[str, Any]]:

    sources: list[
        dict[str, Any]
    ] = []


    print_header(
        "DISCOVERING VALIDATION DATA"
    )


    for (
        dataset_name,
        dataset_path,
    ) in DATASET_PATHS.items():

        if not dataset_path.exists():

            print(
                f"  {dataset_name:<16}: "
                "NOT FOUND"
            )

            continue


        try:

            dataset = load_from_disk(
                str(dataset_path)
            )

        except Exception as exc:

            print(
                f"  {dataset_name:<16}: "
                f"FAILED TO LOAD ({exc})"
            )

            continue


        selected = choose_evaluation_split(

            dataset=dataset,

            dataset_name=dataset_name,

        )


        if selected is None:
            continue


        split_name, split = selected


        try:

            text_column = detect_text_column(
                split
            )

        except RuntimeError as exc:

            print(
                f"  {dataset_name:<16}: "
                f"SKIPPED ({exc})"
            )

            continue


        print(
            f"  {dataset_name:<16}: "
            f"{split_name:<12} "
            f"rows={len(split):,} "
            f"text={text_column}"
        )


        sources.append({

            "dataset_name": (
                dataset_name
            ),

            "split_name": (
                split_name
            ),

            "dataset": (
                split
            ),

            "text_column": (
                text_column
            ),

        })


    if not sources:

        raise RuntimeError(
            "\nNo safe validation/test splits were found.\n\n"
            "The evaluator refuses to use unnamed datasets "
            "because they may contain training data."
        )


    return sources


# ============================================================
# Balanced Evaluation Sample Generator
# ============================================================

# ============================================================
# Balanced Evaluation Sample Generator
# ============================================================

def evaluation_samples(
    tokenizer: MyGPTTokenizer,
    sources: list[dict[str, Any]],
    sequence_length: int,
    max_eval_tokens: int,
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        str,
    ]
]:
    """
    Build balanced held-out validation sequences.

    Documents from each dataset are tokenized and accumulated
    into a corpus-level token stream. This is necessary for
    datasets such as WikiText-103 where individual rows are
    often shorter than the model's 512-token context.

    Objective:

        chunk  = sequence_length + 1
        input  = chunk[:-1]
        target = chunk[1:]
    """

    if not sources:
        raise RuntimeError(
            "No validation sources supplied."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    if max_eval_tokens < 1:
        raise ValueError(
            "max_eval_tokens must be >= 1."
        )

    required_tokens = (
        sequence_length + 1
    )

    # --------------------------------------------------------
    # Total number of full evaluation sequences
    # --------------------------------------------------------

    total_target_sequences = max(
        1,
        math.ceil(
            max_eval_tokens
            / sequence_length
        ),
    )

    source_count = len(
        sources
    )

    base_sequences = (
        total_target_sequences
        // source_count
    )

    remainder_sequences = (
        total_target_sequences
        % source_count
    )

    # --------------------------------------------------------
    # Balanced quota
    # --------------------------------------------------------

    source_sequence_limits = {}

    for index, source in enumerate(
        sources
    ):

        source_key = (
            f"{source['dataset_name']}:"
            f"{source['split_name']}"
        )

        source_sequence_limits[
            source_key
        ] = (
            base_sequences
            +
            (
                1
                if index < remainder_sequences
                else 0
            )
        )

    print()
    print("Validation allocation:")

    for (
        source_key,
        sequence_limit,
    ) in source_sequence_limits.items():

        print(
            f"  {source_key:<30} "
            f"{sequence_limit:,} sequences "
            f"(~{sequence_limit * sequence_length:,} tokens)"
        )

    # --------------------------------------------------------
    # Process each dataset independently
    # --------------------------------------------------------

    for source in sources:

        dataset_name = source[
            "dataset_name"
        ]

        split_name = source[
            "split_name"
        ]

        dataset = source[
            "dataset"
        ]

        text_column = source[
            "text_column"
        ]

        source_key = (
            f"{dataset_name}:"
            f"{split_name}"
        )

        sequence_limit = (
            source_sequence_limits[
                source_key
            ]
        )

        sequences_generated = 0

        # This is the critical difference from the old version.
        token_buffer: list[int] = []

        # ----------------------------------------------------
        # Read rows and accumulate tokens
        # ----------------------------------------------------

        for record in dataset:

            if (
                sequences_generated
                >= sequence_limit
            ):
                break

            text = record.get(
                text_column
            )

            if not isinstance(
                text,
                str,
            ):
                continue

            text = text.strip()

            if not text:
                continue

            try:
                token_ids = tokenizer.encode(
                    text
                )

            except Exception:
                continue

            if not token_ids:
                continue

            token_buffer.extend(
                token_ids
            )

            # ------------------------------------------------
            # Produce as many full sequences as possible
            # ------------------------------------------------

            while (
                len(token_buffer)
                >= required_tokens
                and
                sequences_generated
                < sequence_limit
            ):

                chunk = token_buffer[
                    :required_tokens
                ]

                input_ids = torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                )

                target_ids = torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                )

                yield (
                    input_ids,
                    target_ids,
                    dataset_name,
                    split_name,
                )

                sequences_generated += 1

                # Keep the final token as the first input token
                # of the next autoregressive window.
                del token_buffer[
                    :sequence_length
                ]

        # ----------------------------------------------------
        # Report actual contribution
        # ----------------------------------------------------

        actual_tokens = (
            sequences_generated
            * sequence_length
        )

        print(
            f"  {source_key:<30} "
            f"generated "
            f"{sequences_generated:,} sequences "
            f"({actual_tokens:,} tokens)"
        )

        if (
            sequences_generated
            <
            sequence_limit
        ):

            print(
                f"  WARNING: {source_key} requested "
                f"{sequence_limit:,} sequences but "
                f"generated only "
                f"{sequences_generated:,}."
            )

# ============================================================
# Model Output -> Logits
# ============================================================

def extract_logits(
    output: Any,
) -> torch.Tensor:

    """
    Support the output formats currently used by MyGPT2
    and common model return styles.
    """

    if isinstance(
        output,
        tuple,
    ):

        if len(output) == 0:

            raise RuntimeError(
                "Model returned an empty tuple."
            )

        logits = output[0]


    elif hasattr(
        output,
        "logits",
    ):

        logits = output.logits


    elif isinstance(
        output,
        dict,
    ):

        if "logits" not in output:

            raise RuntimeError(
                "Model output dictionary "
                "does not contain 'logits'."
            )

        logits = output[
            "logits"
        ]


    elif torch.is_tensor(
        output
    ):

        logits = output


    else:

        raise RuntimeError(
            "Unsupported model output type: "
            f"{type(output)}"
        )


    if not torch.is_tensor(
        logits
    ):

        raise RuntimeError(
            "Extracted logits are not a tensor."
        )


    return logits


# ============================================================
# Evaluate
# ============================================================

@torch.inference_mode()
def evaluate(
    model: MyGPTModel,
    tokenizer: MyGPTTokenizer,
    sources: list[dict[str, Any]],
    *,
    device: torch.device,
    sequence_length: int,
    batch_size: int,
    max_eval_tokens: int,
) -> dict[str, Any]:

    """
    Run balanced validation evaluation.
    """

    model.eval()


    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_negative_log_likelihood = 0.0

    total_tokens = 0

    total_sequences = 0


    # --------------------------------------------------------
    # Per-dataset statistics
    # --------------------------------------------------------

    dataset_token_counts: dict[
        str,
        int,
    ] = {}


    dataset_loss_sums: dict[
        str,
        float,
    ] = {}


    dataset_sequence_counts: dict[
        str,
        int,
    ] = {}


    # --------------------------------------------------------
    # Batch buffers
    # --------------------------------------------------------

    batch_inputs: list[
        torch.Tensor
    ] = []


    batch_targets: list[
        torch.Tensor
    ] = []


    batch_sources: list[
        tuple[str, str]
    ] = []


    # --------------------------------------------------------
    # GPU statistics
    # --------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            device
        )


    start_time = time.perf_counter()


    # ========================================================
    # Process Buffered Batch
    # ========================================================

    def process_batch() -> None:

        nonlocal total_negative_log_likelihood
        nonlocal total_tokens
        nonlocal total_sequences


        if not batch_inputs:
            return


        # ----------------------------------------------------
        # Stack tensors
        # ----------------------------------------------------

        inputs = torch.stack(
            batch_inputs
        ).to(

            device,

            non_blocking=True,

        )


        targets = torch.stack(
            batch_targets
        ).to(

            device,

            non_blocking=True,

        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(
            input_ids=inputs
        )


        logits = extract_logits(
            output
        )


        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if logits.ndim != 3:

            raise RuntimeError(
                "Expected logits shape "
                "[batch, sequence, vocabulary].\n"
                f"Received: {tuple(logits.shape)}"
            )


        if (
            logits.shape[0]
            !=
            targets.shape[0]
        ):

            raise RuntimeError(
                "Logit/target batch-size mismatch."
            )


        if (
            logits.shape[1]
            !=
            targets.shape[1]
        ):

            raise RuntimeError(
                "Logit/target sequence-length mismatch.\n"
                f"Logits : {tuple(logits.shape)}\n"
                f"Targets: {tuple(targets.shape)}"
            )


        # ----------------------------------------------------
        # Token-level cross entropy
        # ----------------------------------------------------

        flat_logits = logits.reshape(

            -1,

            logits.size(-1),

        )


        flat_targets = targets.reshape(
            -1
        )


        token_losses = F.cross_entropy(

            flat_logits,

            flat_targets,

            reduction="none",

        )


        if not torch.isfinite(
            token_losses
        ).all():

            raise RuntimeError(
                "Validation produced NaN or "
                "infinite token loss."
            )


        token_losses = token_losses.view(
            targets.shape
        )


        # ----------------------------------------------------
        # Global metrics
        # ----------------------------------------------------

        batch_loss_sum = float(
            token_losses.sum().item()
        )


        predicted_tokens = int(
            targets.numel()
        )


        total_negative_log_likelihood += (
            batch_loss_sum
        )


        total_tokens += (
            predicted_tokens
        )


        total_sequences += int(
            inputs.shape[0]
        )


        # ----------------------------------------------------
        # Dataset metrics
        # ----------------------------------------------------

        for (
            index,
            source_info,
        ) in enumerate(batch_sources):

            (
                dataset_name,
                split_name,
            ) = source_info


            source_key = (

                f"{dataset_name}:"
                f"{split_name}"

            )


            sequence_loss_sum = float(

                token_losses[
                    index
                ]
                .sum()
                .item()

            )


            sequence_tokens = int(

                targets[
                    index
                ]
                .numel()

            )


            dataset_token_counts[
                source_key
            ] = (

                dataset_token_counts.get(
                    source_key,
                    0,
                )

                +
                sequence_tokens

            )


            dataset_loss_sums[
                source_key
            ] = (

                dataset_loss_sums.get(
                    source_key,
                    0.0,
                )

                +
                sequence_loss_sum

            )


            dataset_sequence_counts[
                source_key
            ] = (

                dataset_sequence_counts.get(
                    source_key,
                    0,
                )

                +
                1

            )


        # ----------------------------------------------------
        # Clear batch
        # ----------------------------------------------------

        batch_inputs.clear()

        batch_targets.clear()

        batch_sources.clear()


    # ========================================================
    # Evaluation Loop
    # ========================================================

    for (
        input_ids,
        target_ids,
        dataset_name,
        split_name,
    ) in evaluation_samples(

        tokenizer=tokenizer,

        sources=sources,

        sequence_length=sequence_length,

        max_eval_tokens=max_eval_tokens,

    ):

        batch_inputs.append(
            input_ids
        )

        batch_targets.append(
            target_ids
        )

        batch_sources.append(
            (
                dataset_name,
                split_name,
            )
        )


        if (
            len(batch_inputs)
            >=
            batch_size
        ):

            process_batch()


    # --------------------------------------------------------
    # Final partial batch
    # --------------------------------------------------------

    process_batch()


    if device.type == "cuda":

        torch.cuda.synchronize(
            device
        )


    elapsed_seconds = (

        time.perf_counter()
        -
        start_time

    )


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if total_tokens == 0:

        raise RuntimeError(
            "Validation generated zero usable tokens.\n"
            f"A complete sequence requires "
            f"{sequence_length + 1} source tokens."
        )


    # ========================================================
    # Combined Metrics
    # ========================================================

    validation_loss = (

        total_negative_log_likelihood
        /
        total_tokens

    )


    if validation_loss < 100:

        validation_perplexity = (
            math.exp(
                validation_loss
            )
        )

    else:

        validation_perplexity = (
            float("inf")
        )


    tokens_per_second = (

        total_tokens
        /
        elapsed_seconds

        if elapsed_seconds > 0

        else 0.0

    )


    if device.type == "cuda":

        peak_gpu_memory_mb = (

            torch.cuda.max_memory_allocated(
                device
            )

            /
            (1024 ** 2)

        )

    else:

        peak_gpu_memory_mb = 0.0


    # ========================================================
    # Per-Dataset Metrics
    # ========================================================

    per_dataset_metrics: dict[
        str,
        dict[str, Any],
    ] = {}


    for source_key in (
        dataset_token_counts
    ):

        token_count = (
            dataset_token_counts[
                source_key
            ]
        )


        loss_sum = (
            dataset_loss_sums[
                source_key
            ]
        )


        sequence_count = (
            dataset_sequence_counts[
                source_key
            ]
        )


        dataset_loss = (

            loss_sum
            /
            token_count

        )


        if dataset_loss < 100:

            dataset_perplexity = (
                math.exp(
                    dataset_loss
                )
            )

        else:

            dataset_perplexity = (
                float("inf")
            )


        per_dataset_metrics[
            source_key
        ] = {

            "sequences": (
                sequence_count
            ),

            "tokens": (
                token_count
            ),

            "loss": (
                dataset_loss
            ),

            "perplexity": (
                dataset_perplexity
            ),

        }


    return {

        "validation_loss": (
            validation_loss
        ),

        "validation_perplexity": (
            validation_perplexity
        ),

        "evaluated_tokens": (
            total_tokens
        ),

        "evaluated_sequences": (
            total_sequences
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),

        "tokens_per_second": (
            tokens_per_second
        ),

        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),

        "dataset_token_counts": (
            dataset_token_counts
        ),

        "per_dataset": (
            per_dataset_metrics
        ),

    }


# ============================================================
# Save Results
# ============================================================

def save_results(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config: GPTConfig,
    model: MyGPTModel,
    matching_tensors: int,
    metrics: dict[str, Any],
    sources: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_eval_tokens: int,
) -> Path:

    """
    Save complete evaluation result as JSON.
    """

    RESULTS_DIRECTORY.mkdir(

        parents=True,

        exist_ok=True,

    )


    global_step_raw = checkpoint.get(
        "global_step"
    )


    if global_step_raw is None:

        global_step = (
            extract_checkpoint_step(
                checkpoint_path
            )
        )

    else:

        global_step = int(
            global_step_raw
        )


    if global_step >= 0:

        result_name = (
            f"checkpoint_step_"
            f"{global_step:08d}_validation.json"
        )

    else:

        result_name = (
            f"{checkpoint_path.stem}"
            "_validation.json"
        )


    result_path = (
        RESULTS_DIRECTORY
        /
        result_name
    )


    # --------------------------------------------------------
    # Dataset source metadata
    # --------------------------------------------------------

    source_metadata = []


    for source in sources:

        source_metadata.append({

            "dataset": (
                source[
                    "dataset_name"
                ]
            ),

            "split": (
                source[
                    "split_name"
                ]
            ),

            "rows": int(
                len(
                    source[
                        "dataset"
                    ]
                )
            ),

            "text_column": (
                source[
                    "text_column"
                ]
            ),

        })


    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    parameter_count = sum(

        parameter.numel()

        for parameter
        in model.parameters()

    )


    trainable_parameter_count = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad

    )


    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    if device.type == "cuda":

        gpu_name = (
            torch.cuda.get_device_name(
                device
            )
        )

    else:

        gpu_name = None


    # --------------------------------------------------------
    # Serializable result
    # --------------------------------------------------------

    result = {

        "evaluation_version": 2,

        "evaluated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "checkpoint": {

            "file": str(
                checkpoint_path
            ),

            "name": (
                checkpoint_path.name
            ),

            "global_step": (
                checkpoint.get(
                    "global_step"
                )
            ),

            "epoch": (
                checkpoint.get(
                    "epoch"
                )
            ),

            "saved_train_loss": (
                checkpoint.get(
                    "train_loss"
                )
            ),

            "saved_val_loss": (
                checkpoint.get(
                    "val_loss"
                )
            ),

            "checkpoint_version": (
                checkpoint.get(
                    "checkpoint_version"
                )
            ),

        },

        "model": {

            "vocab_size": int(
                config.vocab_size
            ),

            "max_position_embeddings": int(
                config.max_position_embeddings
            ),

            "hidden_size": int(
                config.hidden_size
            ),

            "num_layers": int(
                config.num_layers
            ),

            "num_attention_heads": int(
                config.num_attention_heads
            ),

            "intermediate_size": int(
                config.intermediate_size
            ),

            "parameter_count": int(
                parameter_count
            ),

            "trainable_parameter_count": int(
                trainable_parameter_count
            ),

            "matching_checkpoint_tensors": int(
                matching_tensors
            ),

        },

        "evaluation": {

            **metrics,

            "batch_size": int(
                batch_size
            ),

            "max_eval_tokens_requested": int(
                max_eval_tokens
            ),

            "sequence_length": int(
                config.max_position_embeddings
            ),

        },

        "sources": (
            source_metadata
        ),

        "device": {

            "type": (
                device.type
            ),

            "name": (
                gpu_name
                if gpu_name is not None
                else "CPU"
            ),

        },

    }


    # --------------------------------------------------------
    # JSON cannot serialize NaN/Infinity cleanly across tools.
    # Current values should be finite, but keep serialization
    # strict so an invalid metric cannot silently enter results.
    # --------------------------------------------------------

    result_path.write_text(

        json.dumps(

            result,

            indent=4,

            allow_nan=False,

        ),

        encoding="utf-8",

    )


    return result_path


# ============================================================
# Command Line Parser
# ============================================================

def create_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate a trained MyGPT2 checkpoint "
            "on held-out validation data."
        )

    )


    parser.add_argument(

        "--checkpoint",

        type=str,

        default=None,

        help=(
            "Explicit checkpoint path. "
            "If omitted, the highest-step checkpoint "
            "is selected automatically."
        ),

    )


    parser.add_argument(

        "--max-eval-tokens",

        type=int,

        default=(
            DEFAULT_MAX_EVAL_TOKENS
        ),

        help=(
            "Approximate total number of predicted "
            "validation tokens."
        ),

    )


    parser.add_argument(

        "--batch-size",

        type=int,

        default=(
            DEFAULT_BATCH_SIZE
        ),

        help=(
            "Evaluation batch size."
        ),

    )


    parser.add_argument(

        "--seed",

        type=int,

        default=(
            DEFAULT_SEED
        ),

        help=(
            "Random seed."
        ),

    )


    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = create_argument_parser()

    args = parser.parse_args()


    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1."
        )


    if args.max_eval_tokens < 1:

        raise ValueError(
            "--max-eval-tokens must be >= 1."
        )


    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    torch.manual_seed(
        args.seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    # ========================================================
    # Header
    # ========================================================

    print_header(
        "MYGPT2 VALIDATION EVALUATION"
    )


    # ========================================================
    # Checkpoint Selection
    # ========================================================

    if args.checkpoint is None:

        checkpoint_path = (
            locate_latest_checkpoint()
        )

    else:

        checkpoint_path = Path(
            args.checkpoint
        )


        if not checkpoint_path.is_absolute():

            checkpoint_path = (
                PROJECT_ROOT
                /
                checkpoint_path
            )


        checkpoint_path = (
            checkpoint_path.resolve()
        )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )


    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )


    print(
        f"Device     : "
        f"{device}"
    )


    if device.type == "cuda":

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(device)}"
        )


    # ========================================================
    # Load Checkpoint
    # ========================================================

    checkpoint = load_raw_checkpoint(
        checkpoint_path
    )


    print()

    print(
        f"Global Step : "
        f"{checkpoint.get('global_step')}"
    )


    print(
        f"Train Loss  : "
        f"{checkpoint.get('train_loss')}"
    )


    # ========================================================
    # Restore Config
    # ========================================================

    config = restore_config(
        checkpoint
    )


    print()

    print(
        f"Context     : "
        f"{config.max_position_embeddings}"
    )


    print(
        f"Vocabulary  : "
        f"{config.vocab_size:,}"
    )


    # ========================================================
    # Tokenizer
    # ========================================================

    print_header(
        "LOADING TOKENIZER"
    )


    tokenizer = load_tokenizer(
        config
    )


    print(
        f"Vocabulary : "
        f"{tokenizer.vocabulary_size:,}"
    )


    print(
        "Tokenizer : PASS"
    )


    # ========================================================
    # Model
    # ========================================================

    print_header(
        "LOADING TRAINED MODEL"
    )


    model = load_model(

        checkpoint=checkpoint,

        config=config,

        device=device,

    )


    matching_tensors = verify_model_weights(

        model=model,

        checkpoint=checkpoint,

    )


    total_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()

    )


    print(
        f"Parameters       : "
        f"{total_parameters:,}"
    )


    print(
        f"Matching tensors : "
        f"{matching_tensors}"
    )


    print(
        "Checkpoint load  : PASS"
    )


    # ========================================================
    # Validation Data
    # ========================================================

    sources = load_validation_sources()


    # ========================================================
    # Run Evaluation
    # ========================================================

    print_header(
        "RUNNING VALIDATION"
    )


    print(
        f"Sequence length : "
        f"{config.max_position_embeddings}"
    )


    print(
        f"Batch size      : "
        f"{args.batch_size}"
    )


    print(
        f"Target tokens   : "
        f"{args.max_eval_tokens:,}"
    )


    metrics = evaluate(

        model=model,

        tokenizer=tokenizer,

        sources=sources,

        device=device,

        sequence_length=(
            config.max_position_embeddings
        ),

        batch_size=(
            args.batch_size
        ),

        max_eval_tokens=(
            args.max_eval_tokens
        ),

    )


    # ========================================================
    # Combined Results
    # ========================================================

    print_header(
        "VALIDATION RESULTS"
    )


    print(
        f"Evaluated Sequences : "
        f"{metrics['evaluated_sequences']:,}"
    )


    print(
        f"Evaluated Tokens    : "
        f"{metrics['evaluated_tokens']:,}"
    )


    print()

    print(
        f"Validation Loss     : "
        f"{metrics['validation_loss']:.6f}"
    )


    print(
        f"Validation PPL      : "
        f"{metrics['validation_perplexity']:.4f}"
    )


    print()

    print(
        f"Evaluation Time     : "
        f"{metrics['elapsed_seconds']:.2f} sec"
    )


    print(
        f"Tokens / second     : "
        f"{metrics['tokens_per_second']:.2f}"
    )


    if device.type == "cuda":

        print(
            f"Peak GPU Memory     : "
            f"{metrics['peak_gpu_memory_mb']:.2f} MB"
        )


    # ========================================================
    # Per-Dataset Results
    # ========================================================

    print()

    print(
        "Per-Dataset Results:"
    )

    print(
        "-" * 80
    )


    for (
        dataset_name,
        dataset_metrics,
    ) in metrics[
        "per_dataset"
    ].items():

        print(
            dataset_name
        )


        print(
            f"  Sequences : "
            f"{dataset_metrics['sequences']:,}"
        )


        print(
            f"  Tokens    : "
            f"{dataset_metrics['tokens']:,}"
        )


        print(
            f"  Loss      : "
            f"{dataset_metrics['loss']:.6f}"
        )


        print(
            f"  PPL       : "
            f"{dataset_metrics['perplexity']:.4f}"
        )


        print()


    # ========================================================
    # Train vs Validation Comparison
    # ========================================================

    saved_train_loss = checkpoint.get(
        "train_loss"
    )


    if saved_train_loss is not None:

        try:

            saved_train_loss = float(
                saved_train_loss
            )

        except (
            TypeError,
            ValueError,
        ):

            saved_train_loss = None


    if (
        saved_train_loss is not None
        and
        math.isfinite(
            saved_train_loss
        )
    ):

        saved_train_perplexity = (
            math.exp(
                saved_train_loss
            )

            if saved_train_loss < 100

            else float("inf")
        )


        generalization_gap = (

            metrics[
                "validation_loss"
            ]

            -

            saved_train_loss

        )


        print(
            "-" * 80
        )


        print(
            f"Saved Train Loss    : "
            f"{saved_train_loss:.6f}"
        )


        print(
            f"Saved Train PPL     : "
            f"{saved_train_perplexity:.4f}"
        )


        print(
            f"Validation Loss     : "
            f"{metrics['validation_loss']:.6f}"
        )


        print(
            f"Validation PPL      : "
            f"{metrics['validation_perplexity']:.4f}"
        )


        print(
            f"Generalization Gap  : "
            f"{generalization_gap:+.6f}"
        )


        print(
            "-" * 80
        )


    # ========================================================
    # Save JSON
    # ========================================================

    result_path = save_results(

        checkpoint_path=(
            checkpoint_path
        ),

        checkpoint=(
            checkpoint
        ),

        config=(
            config
        ),

        model=(
            model
        ),

        matching_tensors=(
            matching_tensors
        ),

        metrics=(
            metrics
        ),

        sources=(
            sources
        ),

        device=(
            device
        ),

        batch_size=(
            args.batch_size
        ),

        max_eval_tokens=(
            args.max_eval_tokens
        ),

    )


    print()

    print(
        "Results saved:"
    )


    print(
        result_path
    )


    print_header(
        "VALIDATION COMPLETE"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()