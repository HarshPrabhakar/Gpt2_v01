from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from training.instruct.formatter import (
    normalize_messages,
    role_prefix,
)


IGNORE_INDEX = -100


class InstructionDataset(Dataset):
    """
    Context-aware supervised instruction dataset.

    For every assistant response, create one training example.

    Previous conversation turns are provided as context but are masked
    from the loss.

    Only the CURRENT assistant response and EOS are supervised.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer,
        sequence_length: int = 512,
        minimum_answer_tokens: int = 64,
    ) -> None:

        self.file_path = Path(file_path)

        self.tokenizer = tokenizer

        self.sequence_length = int(
            sequence_length
        )

        self.minimum_answer_tokens = int(
            minimum_answer_tokens
        )

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Instruction dataset not found:\n"
                f"{self.file_path}"
            )

        if self.sequence_length < 8:
            raise ValueError(
                "sequence_length is too small."
            )

        # -----------------------------------------------------
        # Special tokens
        # -----------------------------------------------------

        self.bos_token_id = self._resolve_token_id(
            names=[
                "bos_token_id",
                "bos_id",
            ],
            fallback=2,
        )

        self.eos_token_id = self._resolve_token_id(
            names=[
                "eos_token_id",
                "eos_id",
            ],
            fallback=3,
        )

        # -----------------------------------------------------
        # Load conversations
        # -----------------------------------------------------

        self.conversations: list[
            dict[str, Any]
        ] = []

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
                    record = json.loads(line)

                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON at "
                        f"{self.file_path}:"
                        f"{line_number}"
                    ) from error

                messages = normalize_messages(
                    record.get(
                        "messages",
                        []
                    )
                )

                if not messages:
                    continue

                self.conversations.append(
                    {
                        "messages": messages,
                        "source": record.get(
                            "source",
                            "unknown",
                        ),
                    }
                )

        # -----------------------------------------------------
        # Build assistant-response index
        # -----------------------------------------------------

        self.examples: list[
            dict[str, Any]
        ] = []

        for conversation_index, conversation in enumerate(
            self.conversations
        ):

            messages = conversation[
                "messages"
            ]

            assistant_turn = 0

            for message_index, message in enumerate(
                messages
            ):

                if message["role"] != "assistant":
                    continue

                assistant_turn += 1

                # Require some context before assistant.
                if message_index == 0:
                    continue

                # Most SFT data should have a user request before
                # the assistant response.
                previous_messages = messages[
                    :message_index
                ]

                current_answer = message[
                    "content"
                ]

                if not current_answer.strip():
                    continue

                self.examples.append(
                    {
                        "conversation_index": (
                            conversation_index
                        ),
                        "assistant_turn": (
                            assistant_turn
                        ),
                        "context_messages": (
                            previous_messages
                        ),
                        "assistant_content": (
                            current_answer
                        ),
                        "source": conversation[
                            "source"
                        ],
                    }
                )

    # =========================================================
    # Helpers
    # =========================================================

    def _resolve_token_id(
        self,
        names: list[str],
        fallback: int,
    ) -> int:

        for name in names:

            value = getattr(
                self.tokenizer,
                name,
                None,
            )

            if value is not None:
                return int(value)

        return int(fallback)

    def _encode_text(
        self,
        text: str,
    ) -> list[int]:

        token_ids = self.tokenizer.encode(
            text
        )

        if isinstance(
            token_ids,
            torch.Tensor,
        ):
            token_ids = (
                token_ids
                .detach()
                .cpu()
                .tolist()
            )

        token_ids = [
            int(token_id)
            for token_id in token_ids
        ]

        # We control BOS/EOS manually.
        if (
            token_ids
            and token_ids[0]
            == self.bos_token_id
        ):
            token_ids = token_ids[1:]

        if (
            token_ids
            and token_ids[-1]
            == self.eos_token_id
        ):
            token_ids = token_ids[:-1]

        return token_ids

    def _encode_message(
        self,
        message: dict[str, str],
    ) -> list[int]:

        prefix = role_prefix(
            message["role"]
        )

        text = (
            prefix
            + message["content"].strip()
            + "\n\n"
        )

        return self._encode_text(
            text
        )

    # =========================================================
    # Context construction
    # =========================================================

    def _build_example(
        self,
        example: dict[str, Any],
    ) -> tuple[
        list[int],
        list[int],
    ]:

        context_messages = example[
            "context_messages"
        ]

        assistant_content = example[
            "assistant_content"
        ].strip()

        # -----------------------------------------------------
        # Current assistant target
        # -----------------------------------------------------

        assistant_prefix_ids = self._encode_text(
            role_prefix(
                "assistant"
            )
        )

        answer_ids = self._encode_text(
            assistant_content
        )

        if not answer_ids:
            raise RuntimeError(
                "Assistant response encoded to "
                "zero tokens."
            )

        # -----------------------------------------------------
        # Locate current user
        # -----------------------------------------------------

        current_user_index = None

        for index in range(
            len(context_messages) - 1,
            -1,
            -1,
        ):

            if (
                context_messages[index]["role"]
                == "user"
            ):
                current_user_index = index
                break

        if current_user_index is None:
            raise RuntimeError(
                "Assistant response has no "
                "preceding user message."
            )

        current_user_message = (
            context_messages[
                current_user_index
            ]
        )

        current_user_ids = (
            self._encode_message(
                current_user_message
            )
        )

        # Everything before the current user becomes optional
        # historical context.
        history_messages = (
            context_messages[
                :current_user_index
            ]
        )

        # -----------------------------------------------------
        # Fixed token budget
        # -----------------------------------------------------

        # BOS + EOS
        fixed_special_tokens = 2

        available = (
            self.sequence_length
            - fixed_special_tokens
            - len(assistant_prefix_ids)
        )

        if available <= 1:
            raise RuntimeError(
                "No usable token budget."
            )

        # -----------------------------------------------------
        # Reserve assistant answer capacity
        # -----------------------------------------------------

        desired_answer_budget = min(
            len(answer_ids),
            max(
                self.minimum_answer_tokens,
                1,
            ),
        )

        desired_answer_budget = min(
            desired_answer_budget,
            available,
        )

        # -----------------------------------------------------
        # Current user gets priority
        # -----------------------------------------------------

        user_budget = max(
            1,
            available
            - desired_answer_budget,
        )

        if len(current_user_ids) > user_budget:

            # Preserve the beginning of the user instruction.
            current_user_ids = (
                current_user_ids[
                    :user_budget
                ]
            )

        used_without_history = (
            len(current_user_ids)
            + len(assistant_prefix_ids)
            + fixed_special_tokens
        )

        answer_budget = (
            self.sequence_length
            - used_without_history
        )

        answer_ids = answer_ids[
            :answer_budget
        ]

        if not answer_ids:
            raise RuntimeError(
                "No room remains for assistant response."
            )

        # -----------------------------------------------------
        # Historical context budget
        # -----------------------------------------------------

        history_budget = (
            self.sequence_length
            - (
                fixed_special_tokens
                + len(current_user_ids)
                + len(assistant_prefix_ids)
                + len(answer_ids)
            )
        )

        selected_history: list[
            list[int]
        ] = []

        # -----------------------------------------------------
        # Add newest history first.
        #
        # Whole messages only: no mid-message historical
        # truncation.
        # -----------------------------------------------------

        for message in reversed(
            history_messages
        ):

            message_ids = (
                self._encode_message(
                    message
                )
            )

            if not message_ids:
                continue

            if (
                len(message_ids)
                <= history_budget
            ):

                selected_history.append(
                    message_ids
                )

                history_budget -= (
                    len(message_ids)
                )

        selected_history.reverse()

        # -----------------------------------------------------
        # Construct input + labels
        # -----------------------------------------------------

        input_ids: list[int] = [
            self.bos_token_id
        ]

        labels: list[int] = [
            IGNORE_INDEX
        ]

        # Historical conversation:
        # context only, never supervised.
        for history_ids in selected_history:

            input_ids.extend(
                history_ids
            )

            labels.extend(
                [
                    IGNORE_INDEX
                ]
                * len(history_ids)
            )

        # Current user:
        # context only.
        input_ids.extend(
            current_user_ids
        )

        labels.extend(
            [
                IGNORE_INDEX
            ]
            * len(current_user_ids)
        )

        # Assistant role prefix:
        # context only.
        input_ids.extend(
            assistant_prefix_ids
        )

        labels.extend(
            [
                IGNORE_INDEX
            ]
            * len(
                assistant_prefix_ids
            )
        )

        # Current assistant answer:
        # supervised.
        input_ids.extend(
            answer_ids
        )

        labels.extend(
            answer_ids
        )

        # Train EOS.
        input_ids.append(
            self.eos_token_id
        )

        labels.append(
            self.eos_token_id
        )

        # -----------------------------------------------------
        # Final safety checks
        # -----------------------------------------------------

        if len(input_ids) != len(labels):
            raise RuntimeError(
                "input_ids/labels length mismatch."
            )

        if (
            len(input_ids)
            > self.sequence_length
        ):
            raise RuntimeError(
                "Constructed example exceeds "
                f"context length: "
                f"{len(input_ids)} > "
                f"{self.sequence_length}"
            )

        supervised_tokens = sum(
            label != IGNORE_INDEX
            for label in labels
        )

        if supervised_tokens <= 1:
            raise RuntimeError(
                "Example has no meaningful "
                "assistant supervision."
            )

        return (
            input_ids,
            labels,
        )

    # =========================================================
    # Dataset API
    # =========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.examples
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        example = self.examples[
            index
        ]

        input_ids, labels = (
            self._build_example(
                example
            )
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "source": example[
                "source"
            ],
            "conversation_index": (
                example[
                    "conversation_index"
                ]
            ),
            "assistant_turn": (
                example[
                    "assistant_turn"
                ]
            ),
        }