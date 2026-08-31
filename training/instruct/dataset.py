from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tokenizer.my_tokenizer import MyGPTTokenizer

from training.instruct.formatter import (
    normalize_messages,
    role_prefix,
)


IGNORE_INDEX = -100


class InstructionDataset(Dataset):
    """
    Supervised fine-tuning dataset.

    The model sees the entire conversation as input, but loss is
    calculated only on assistant-response tokens.

    User/system text:
        input_ids -> included
        labels    -> -100

    Assistant response:
        input_ids -> included
        labels    -> actual token IDs
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: MyGPTTokenizer,
        sequence_length: int = 512,
    ) -> None:

        self.file_path = Path(
            file_path
        )

        self.tokenizer = tokenizer

        self.sequence_length = int(
            sequence_length
        )

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"Instruction dataset not found:\n"
                f"{self.file_path}"
            )

        if self.sequence_length < 2:

            raise ValueError(
                "sequence_length must be >= 2."
            )

        self.samples = self._load_jsonl()

        if not self.samples:

            raise RuntimeError(
                f"No valid conversations found in:\n"
                f"{self.file_path}"
            )


    # ========================================================
    # JSONL loading
    # ========================================================

    def _load_jsonl(
        self,
    ) -> list[dict[str, Any]]:

        samples = []

        with self.file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    print(
                        f"WARNING: invalid JSON on line "
                        f"{line_number:,}"
                    )

                    continue

                messages = normalize_messages(
                    record.get(
                        "messages",
                        []
                    )
                )

                has_user = any(
                    message["role"] == "user"
                    for message in messages
                )

                has_assistant = any(
                    message["role"] == "assistant"
                    for message in messages
                )

                if (
                    not has_user
                    or
                    not has_assistant
                ):
                    continue

                samples.append(
                    {
                        "messages": messages,
                        "source": record.get(
                            "source"
                        ),
                    }
                )

        return samples


    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )


    # ========================================================
    # Tokenization
    # ========================================================

    def _encode_text(
        self,
        text: str,
    ) -> list[int]:

        token_ids = self.tokenizer.encode(
            text
        )

        return [
            int(token_id)
            for token_id in token_ids
        ]


    # ========================================================
    # Sample construction
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        sample = self.samples[
            index
        ]

        messages = sample[
            "messages"
        ]


        input_ids: list[int] = []
        labels: list[int] = []


        # ----------------------------------------------------
        # Optional BOS token
        # ----------------------------------------------------

        bos_token_id = getattr(
            self.tokenizer,
            "bos_token_id",
            None,
        )

        if bos_token_id is None:

            bos_token_id = getattr(
                self.tokenizer,
                "bos_id",
                None,
            )


        if bos_token_id is not None:

            input_ids.append(
                int(bos_token_id)
            )

            labels.append(
                IGNORE_INDEX
            )


        # ----------------------------------------------------
        # Encode messages one by one
        # ----------------------------------------------------

        for message_index, message in enumerate(
            messages
        ):

            role = message[
                "role"
            ]

            content = message[
                "content"
            ]


            prefix = role_prefix(
                role
            )


            # Put a blank line between conversation turns.
            if message_index > 0:

                separator_ids = self._encode_text(
                    "\n\n"
                )

                input_ids.extend(
                    separator_ids
                )

                labels.extend(
                    [IGNORE_INDEX]
                    * len(separator_ids)
                )


            # ------------------------------------------------
            # Role prefix
            # ------------------------------------------------

            prefix_ids = self._encode_text(
                prefix
            )

            input_ids.extend(
                prefix_ids
            )

            # Never train on "User:", "Assistant:", etc.
            labels.extend(
                [IGNORE_INDEX]
                * len(prefix_ids)
            )


            # ------------------------------------------------
            # Message content
            # ------------------------------------------------

            content_ids = self._encode_text(
                content
            )

            input_ids.extend(
                content_ids
            )


            if role == "assistant":

                labels.extend(
                    content_ids
                )

            else:

                labels.extend(
                    [IGNORE_INDEX]
                    * len(content_ids)
                )


        # ----------------------------------------------------
        # EOS token
        # ----------------------------------------------------

        eos_token_id = getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

        if eos_token_id is None:

            eos_token_id = getattr(
                self.tokenizer,
                "eos_id",
                None,
            )


        if eos_token_id is not None:

            input_ids.append(
                int(eos_token_id)
            )

            # Train EOS only when we have assistant targets.
            if any(
                label != IGNORE_INDEX
                for label in labels
            ):

                labels.append(
                    int(eos_token_id)
                )

            else:

                labels.append(
                    IGNORE_INDEX
                )


        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        if len(input_ids) > self.sequence_length:

            # Keep the most recent part of the conversation.
            input_ids = input_ids[
                -self.sequence_length:
            ]

            labels = labels[
                -self.sequence_length:
            ]


        return {

            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),

            "source": sample.get(
                "source"
            ),

        }