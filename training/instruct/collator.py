from __future__ import annotations

from typing import Any

import torch


IGNORE_INDEX = -100


class InstructionCollator:
    """
    Dynamic-padding collator for MyGPT2 supervised instruction tuning.

    Padding policy:

        input_ids -> pad_token_id
        labels    -> -100

    Because the model is causal and padding is appended to the RIGHT,
    real tokens cannot attend to future padding tokens.

    The custom MyGPT2 model therefore does not require an attention mask
    for this batching strategy.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
        max_length: int = 512,
    ) -> None:

        self.pad_token_id = int(
            pad_token_id
        )

        self.ignore_index = int(
            ignore_index
        )

        self.max_length = int(
            max_length
        )

        if self.max_length < 2:
            raise ValueError(
                "max_length must be >= 2."
            )

    def __call__(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not batch:
            raise ValueError(
                "InstructionCollator received an empty batch."
            )

        # -----------------------------------------------------
        # Determine batch padding length
        # -----------------------------------------------------

        batch_max_length = max(
            len(sample["input_ids"])
            for sample in batch
        )

        batch_max_length = min(
            batch_max_length,
            self.max_length,
        )

        batch_size = len(batch)

        # -----------------------------------------------------
        # Allocate padded tensors
        # -----------------------------------------------------

        input_ids = torch.full(
            (
                batch_size,
                batch_max_length,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )

        labels = torch.full(
            (
                batch_size,
                batch_max_length,
            ),
            fill_value=self.ignore_index,
            dtype=torch.long,
        )

        lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
        )

        # -----------------------------------------------------
        # Copy samples
        # -----------------------------------------------------

        sources: list[Any] = []
        conversation_indices: list[Any] = []
        assistant_turns: list[Any] = []

        for batch_index, sample in enumerate(
            batch
        ):

            sample_input_ids = sample[
                "input_ids"
            ][:batch_max_length]

            sample_labels = sample[
                "labels"
            ][:batch_max_length]

            sequence_length = len(
                sample_input_ids
            )

            if sequence_length != len(
                sample_labels
            ):
                raise RuntimeError(
                    "input_ids and labels length mismatch "
                    f"inside batch item {batch_index}."
                )

            input_ids[
                batch_index,
                :sequence_length,
            ] = sample_input_ids

            labels[
                batch_index,
                :sequence_length,
            ] = sample_labels

            lengths[
                batch_index
            ] = sequence_length

            sources.append(
                sample.get("source")
            )

            conversation_indices.append(
                sample.get(
                    "conversation_index"
                )
            )

            assistant_turns.append(
                sample.get(
                    "assistant_turn"
                )
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "lengths": lengths,
            "sources": sources,
            "conversation_indices": (
                conversation_indices
            ),
            "assistant_turns": assistant_turns,
        }