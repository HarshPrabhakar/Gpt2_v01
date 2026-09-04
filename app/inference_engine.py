# D:\Gpt2_v01\app\inference_engine.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model.config import GPTConfig
from model.model import MyGPTModel
from tokenizer.my_tokenizer import MyGPTTokenizer


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "tokenizer"
    / "tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "instruct_checkpoints"
    / "best_instruct.pt"
)


# ============================================================
# Inference Engine
# ============================================================

class MyGPTInferenceEngine:

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("=" * 72)
        print("ARCLM INSTRUCT INFERENCE")
        print("=" * 72)
        print(f"Device          : {self.device}")
        print(f"Checkpoint      : {CHECKPOINT_PATH}")
        print(f"Tokenizer       : {TOKENIZER_PATH}")

        self.checkpoint = self._load_checkpoint()
        self.config = self._restore_config()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        self.global_step = self.checkpoint.get(
            "global_step",
            None,
        )

        self.parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        print()
        print("Model loaded successfully.")
        print(f"Checkpoint      : {CHECKPOINT_PATH.name}")
        print(f"Checkpoint step : {self.global_step}")
        print(f"Parameters      : {self.parameter_count:,}")
        print(
            f"Context         : "
            f"{self.config.max_position_embeddings}"
        )
        print(
            f"Vocabulary      : "
            f"{self.config.vocab_size:,}"
        )
        print(f"Device          : {self.device}")

        if self.device.type == "cuda":
            print(
                f"GPU             : "
                f"{torch.cuda.get_device_name(self.device)}"
            )

        print("=" * 72)


    # ========================================================
    # Checkpoint
    # ========================================================

    def _load_checkpoint(self) -> dict[str, Any]:

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Instruction checkpoint not found:\n"
                f"{CHECKPOINT_PATH}"
            )

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location="cpu",
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        return checkpoint


    # ========================================================
    # Config
    # ========================================================

    def _restore_config(self) -> GPTConfig:

        config = GPTConfig()

        saved_config = self.checkpoint.get(
            "config"
        )

        if isinstance(saved_config, dict):

            for key, value in saved_config.items():

                if hasattr(config, key):
                    setattr(
                        config,
                        key,
                        value,
                    )

        return config


    # ========================================================
    # Tokenizer
    # ========================================================

    def _load_tokenizer(self) -> MyGPTTokenizer:

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer not found:\n"
                f"{TOKENIZER_PATH}"
            )

        tokenizer = MyGPTTokenizer.load(
            TOKENIZER_PATH
        )

        if (
            tokenizer.vocabulary_size
            != self.config.vocab_size
        ):
            raise RuntimeError(
                "Tokenizer vocabulary does not "
                "match model vocabulary."
            )

        return tokenizer


    # ========================================================
    # Model
    # ========================================================

    def _load_model(self) -> MyGPTModel:

        model = MyGPTModel(
            self.config
        )

        load_result = model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        print("Strict checkpoint load: PASS")
        print(f"Load result      : {load_result}")

        model = model.to(
            self.device
        )

        model.eval()

        return model


    # ========================================================
    # Model Output
    # ========================================================

    @staticmethod
    def _extract_logits(
        output: Any,
    ) -> torch.Tensor:

        if isinstance(output, tuple):
            return output[0]

        if hasattr(output, "logits"):
            return output.logits

        if isinstance(output, dict):

            if "logits" not in output:
                raise KeyError(
                    "Model output dictionary "
                    "does not contain 'logits'."
                )

            return output["logits"]

        if torch.is_tensor(output):
            return output

        raise TypeError(
            "Could not extract logits from "
            f"model output type: {type(output)}"
        )


    # ========================================================
    # History Helpers
    # ========================================================

    @staticmethod
    def _normalize_history(
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:

        if not history:
            return []

        normalized: list[dict[str, str]] = []

        for message in history:

            if not isinstance(message, dict):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
            ).strip().lower()

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if (
                role
                not in {
                    "user",
                    "assistant",
                    "system",
                }
            ):
                continue

            if not content:
                continue

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized


    @staticmethod
    def _history_to_turns(
        history: list[dict[str, str]],
    ) -> list[str]:
        """
        Convert message history into whole conversation turns.

        A normal turn becomes:

            User: ...
            
            Assistant: ...

        Keeping turns whole prevents context truncation from
        chopping through the middle of an old message.
        """

        turns: list[str] = []

        index = 0

        while index < len(history):

            message = history[index]

            role = message["role"]
            content = message["content"]

            # ------------------------------------------------
            # System message
            # ------------------------------------------------

            if role == "system":

                turns.append(
                    f"System: {content}"
                )

                index += 1
                continue

            # ------------------------------------------------
            # User + Assistant pair
            # ------------------------------------------------

            if role == "user":

                turn = (
                    f"User: {content}"
                )

                if (
                    index + 1 < len(history)
                    and
                    history[index + 1]["role"]
                    == "assistant"
                ):

                    assistant_content = (
                        history[index + 1][
                            "content"
                        ]
                    )

                    turn += (
                        "\n\n"
                        f"Assistant: "
                        f"{assistant_content}"
                    )

                    index += 2

                else:
                    index += 1

                turns.append(turn)
                continue

            # ------------------------------------------------
            # Standalone assistant message
            # ------------------------------------------------

            if role == "assistant":

                turns.append(
                    f"Assistant: {content}"
                )

            index += 1

        return turns


    # ========================================================
    # Prompt Construction
    # ========================================================

    def _build_prompt(
        self,
        user_prompt: str,
        history: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        """
        Build a turn-aware prompt while reserving context space
        for the new answer.

        We DO NOT simply keep the last 511 raw tokens anymore.

        Instead:
            1. Reserve generation space.
            2. Always keep current User/Assistant prefix.
            3. Keep newest complete conversation turns.
            4. Drop oldest turns first.
        """

        max_context = int(
            self.config.max_position_embeddings
        )

        # ----------------------------------------------------
        # Reserve room for generation.
        #
        # With the normal 120-token generation:
        #
        # 512 total
        # -120 reserved
        # =392 prompt/history budget
        #
        # Even if max_new_tokens is set very high, we cap the
        # explicit reservation at 160 so the prompt still has
        # useful room.
        # ----------------------------------------------------

        reserved_generation_tokens = min(
            max(
                int(max_new_tokens),
                32,
            ),
            160,
        )

        prompt_budget = (
            max_context
            -
            reserved_generation_tokens
        )

        # Never make prompt budget unreasonably tiny.
        prompt_budget = max(
            prompt_budget,
            128,
        )

        normalized_history = (
            self._normalize_history(
                history
            )
        )

        history_turns = (
            self._history_to_turns(
                normalized_history
            )
        )

        user_prompt = user_prompt.strip()

        current_block = (
            f"User: {user_prompt}"
            "\n\n"
            "Assistant: "
        )

        current_ids = self.tokenizer.encode(
            current_block
        )

        # ----------------------------------------------------
        # Extremely long current user message
        # ----------------------------------------------------

        if len(current_ids) > prompt_budget:

            prefix = "User: "
            suffix = "\n\nAssistant: "

            prefix_ids = self.tokenizer.encode(
                prefix
            )

            suffix_ids = self.tokenizer.encode(
                suffix
            )

            user_ids = self.tokenizer.encode(
                user_prompt
            )

            available_user_tokens = max(
                prompt_budget
                -
                len(prefix_ids)
                -
                len(suffix_ids),
                1,
            )

            # Keep the most recent portion of an oversized
            # current message.
            user_ids = user_ids[
                -available_user_tokens:
            ]

            shortened_user = (
                self.tokenizer.decode(
                    user_ids
                )
            )

            current_block = (
                f"User: {shortened_user.strip()}"
                "\n\n"
                "Assistant: "
            )

        # ----------------------------------------------------
        # Start with all history. Remove oldest whole turns
        # until the complete prompt fits.
        # ----------------------------------------------------

        selected_turns = list(
            history_turns
        )

        def compose_prompt(
            turns: list[str],
        ) -> str:

            if turns:

                return (
                    "\n\n".join(turns)
                    +
                    "\n\n"
                    +
                    current_block
                )

            return current_block

        formatted_prompt = compose_prompt(
            selected_turns
        )

        prompt_ids = self.tokenizer.encode(
            formatted_prompt
        )

        dropped_turns = 0

        while (
            len(prompt_ids) > prompt_budget
            and selected_turns
        ):

            selected_turns.pop(0)
            dropped_turns += 1

            formatted_prompt = compose_prompt(
                selected_turns
            )

            prompt_ids = self.tokenizer.encode(
                formatted_prompt
            )

        # Final safety guard.
        if len(prompt_ids) > prompt_budget:

            prompt_ids = prompt_ids[
                -prompt_budget:
            ]

            formatted_prompt = (
                self.tokenizer.decode(
                    prompt_ids
                )
            )

        stats = {

            "prompt_budget": (
                prompt_budget
            ),

            "reserved_generation_tokens": (
                reserved_generation_tokens
            ),

            "history_turns_total": (
                len(history_turns)
            ),

            "history_turns_used": (
                len(selected_turns)
            ),

            "history_turns_dropped": (
                dropped_turns
            ),

            "prompt_tokens": (
                len(
                    self.tokenizer.encode(
                        formatted_prompt
                    )
                )
            ),
        }

        return (
            formatted_prompt,
            stats,
        )


    # ========================================================
    # Sampling Helpers
    # ========================================================

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        continuation: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Apply repetition penalty ONLY to tokens generated by
        the assistant in the current response.

        The user's prompt and previous history are deliberately
        excluded.
        """

        penalty = float(
            repetition_penalty
        )

        if penalty <= 1.0:
            return logits

        if continuation.numel() == 0:
            return logits

        unique_tokens = torch.unique(
            continuation
        )

        for token_id in unique_tokens.tolist():

            token_id = int(token_id)

            token_logits = logits[
                :,
                token_id,
            ]

            logits[
                :,
                token_id,
            ] = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )

        return logits


    @staticmethod
    def _get_banned_ngram_tokens(
        continuation_ids: list[int],
        ngram_size: int,
    ) -> set[int]:

        n = int(
            ngram_size
        )

        if n <= 0:
            return set()

        if not continuation_ids:
            return set()

        # no-repeat unigram
        if n == 1:
            return set(
                continuation_ids
            )

        if (
            len(continuation_ids)
            <
            n - 1
        ):
            return set()

        ngram_map: dict[
            tuple[int, ...],
            set[int],
        ] = {}

        for index in range(
            len(continuation_ids)
            -
            n
            +
            1
        ):

            ngram = continuation_ids[
                index:
                index + n
            ]

            prefix = tuple(
                ngram[:-1]
            )

            next_token = int(
                ngram[-1]
            )

            if prefix not in ngram_map:
                ngram_map[prefix] = set()

            ngram_map[prefix].add(
                next_token
            )

        current_prefix = tuple(
            continuation_ids[
                -(n - 1):
            ]
        )

        return ngram_map.get(
            current_prefix,
            set(),
        )


    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        continuation: torch.Tensor,
        ngram_size: int,
    ) -> torch.Tensor:
        """
        Block repeated n-grams only inside the current assistant
        completion.
        """

        if int(ngram_size) <= 0:
            return logits

        if continuation.numel() == 0:
            return logits

        continuation_ids = (
            continuation[
                0
            ]
            .detach()
            .cpu()
            .tolist()
        )

        banned_tokens = (
            self._get_banned_ngram_tokens(
                continuation_ids,
                int(ngram_size),
            )
        )

        if banned_tokens:

            banned_tensor = torch.tensor(
                list(banned_tokens),
                dtype=torch.long,
                device=logits.device,
            )

            logits[
                :,
                banned_tensor,
            ] = float("-inf")

        return logits


    @staticmethod
    def _apply_top_k(
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:

        top_k = int(
            top_k
        )

        if (
            top_k <= 0
            or
            top_k >= logits.size(-1)
        ):
            return logits

        values, _ = torch.topk(
            logits,
            top_k,
        )

        threshold = values[
            ...,
            -1,
            None,
        ]

        return torch.where(
            logits < threshold,
            torch.full_like(
                logits,
                float("-inf"),
            ),
            logits,
        )


    @staticmethod
    def _apply_top_p(
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:

        top_p = float(
            top_p
        )

        if (
            top_p <= 0.0
            or
            top_p >= 1.0
        ):
            return logits

        sorted_logits, sorted_indices = (
            torch.sort(
                logits,
                descending=True,
            )
        )

        sorted_probabilities = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = (
            torch.cumsum(
                sorted_probabilities,
                dim=-1,
            )
        )

        sorted_mask = (
            cumulative_probabilities
            >
            top_p
        )

        # Keep at least one token.
        sorted_mask[
            ...,
            1:
        ] = sorted_mask[
            ...,
            :-1
        ].clone()

        sorted_mask[
            ...,
            0
        ] = False

        mask = torch.zeros_like(
            sorted_mask,
            dtype=torch.bool,
        )

        mask.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_mask,
        )

        return logits.masked_fill(
            mask,
            float("-inf"),
        )


    # ========================================================
    # Generation
    # ========================================================

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        history: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 120,
        temperature: float = 0.70,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
    ) -> dict[str, Any]:

        prompt = str(
            prompt
        ).strip()

        if not prompt:

            return {
                "text": "",
                "generated_tokens": 0,
                "elapsed_seconds": 0.0,
                "tokens_per_second": 0.0,
                "input_tokens": 0,
                "stopped_on_eos": False,
                "formatted_prompt": "",
                "history_turns_used": 0,
                "history_turns_dropped": 0,
                "prompt_budget": 0,
                "reserved_generation_tokens": 0,
            }

        max_new_tokens = max(
            int(max_new_tokens),
            1,
        )

        temperature = max(
            float(temperature),
            1e-5,
        )

        # ----------------------------------------------------
        # Build clean turn-aware prompt
        # ----------------------------------------------------

        formatted_prompt, prompt_stats = (
            self._build_prompt(
                user_prompt=prompt,
                history=history,
                max_new_tokens=max_new_tokens,
            )
        )

        token_ids = self.tokenizer.encode(
            formatted_prompt
        )

        if not token_ids:

            return {
                "text": "",
                "generated_tokens": 0,
                "elapsed_seconds": 0.0,
                "tokens_per_second": 0.0,
                "input_tokens": 0,
                "stopped_on_eos": False,
                "formatted_prompt": (
                    formatted_prompt
                ),
                **prompt_stats,
            }

        input_tokens = len(
            token_ids
        )

        generated = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        prompt_tensor_length = int(
            generated.size(1)
        )

        eos_token_id = getattr(
            self.config,
            "eos_token_id",
            None,
        )

        stopped_on_eos = False

        actual_generated_tokens = 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Autoregressive generation
        # ----------------------------------------------------

        for _ in range(
            max_new_tokens
        ):

            # Model itself still sees no more than its
            # 512-position limit.
            context = generated[
                :,
                -self.config.max_position_embeddings:
            ]

            output = self.model(
                input_ids=context
            )

            logits = self._extract_logits(
                output
            )

            next_token_logits = (
                logits[
                    :,
                    -1,
                    :
                ]
                .clone()
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Only current assistant continuation is used for
            # repetition/no-repeat penalties.
            #
            # The old implementation could penalize words from
            # the USER prompt/history too.
            # ------------------------------------------------

            continuation = generated[
                :,
                prompt_tensor_length:
            ]

            next_token_logits = (
                self._apply_repetition_penalty(
                    logits=next_token_logits,
                    continuation=continuation,
                    repetition_penalty=(
                        repetition_penalty
                    ),
                )
            )

            next_token_logits = (
                self._apply_no_repeat_ngram(
                    logits=next_token_logits,
                    continuation=continuation,
                    ngram_size=(
                        no_repeat_ngram_size
                    ),
                )
            )

            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            next_token_logits = (
                next_token_logits
                /
                temperature
            )

            # ------------------------------------------------
            # Top-K
            # ------------------------------------------------

            next_token_logits = (
                self._apply_top_k(
                    next_token_logits,
                    top_k,
                )
            )

            # ------------------------------------------------
            # Top-P
            # ------------------------------------------------

            next_token_logits = (
                self._apply_top_p(
                    next_token_logits,
                    top_p,
                )
            )

            # ------------------------------------------------
            # Numerical safety
            # ------------------------------------------------

            finite_values = torch.isfinite(
                next_token_logits
            )

            if not finite_values.any():

                # Extremely defensive fallback.
                next_token_logits = (
                    logits[
                        :,
                        -1,
                        :
                    ]
                    /
                    temperature
                )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            if torch.isnan(
                probabilities
            ).any():

                raise RuntimeError(
                    "NaN probabilities encountered "
                    "during generation."
                )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            generated = torch.cat(
                [
                    generated,
                    next_token,
                ],
                dim=1,
            )

            actual_generated_tokens += 1

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if (
                eos_token_id is not None
                and
                int(next_token.item())
                ==
                int(eos_token_id)
            ):

                stopped_on_eos = True
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds = (
            time.perf_counter()
            -
            start_time
        )

        # ----------------------------------------------------
        # Decode ONLY newly generated tokens
        # ----------------------------------------------------

        completion_ids = (
            generated[
                0,
                prompt_tensor_length:
            ]
            .detach()
            .cpu()
            .tolist()
        )

        # Do not expose EOS in visible response.
        if (
            eos_token_id is not None
            and completion_ids
            and completion_ids[-1]
            == int(eos_token_id)
        ):
            visible_completion_ids = (
                completion_ids[:-1]
            )

        else:
            visible_completion_ids = (
                completion_ids
            )

        generated_text = (
            self.tokenizer.decode(
                visible_completion_ids
            )
            if visible_completion_ids
            else ""
        )

        generated_text = (
            generated_text.strip()
        )

        tokens_per_second = (
            actual_generated_tokens
            /
            elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        return {

            "text": generated_text,

            "generated_tokens": (
                actual_generated_tokens
            ),

            "elapsed_seconds": (
                elapsed_seconds
            ),

            "tokens_per_second": (
                tokens_per_second
            ),

            "input_tokens": (
                input_tokens
            ),

            "stopped_on_eos": (
                stopped_on_eos
            ),

            "formatted_prompt": (
                formatted_prompt
            ),

            "history_turns_used": (
                prompt_stats[
                    "history_turns_used"
                ]
            ),

            "history_turns_dropped": (
                prompt_stats[
                    "history_turns_dropped"
                ]
            ),

            "history_turns_total": (
                prompt_stats[
                    "history_turns_total"
                ]
            ),

            "prompt_budget": (
                prompt_stats[
                    "prompt_budget"
                ]
            ),

            "reserved_generation_tokens": (
                prompt_stats[
                    "reserved_generation_tokens"
                ]
            ),
        }


    # ========================================================
    # Status
    # ========================================================

    def status(self) -> dict[str, Any]:

        gpu_name = (
            torch.cuda.get_device_name(
                self.device
            )
            if self.device.type == "cuda"
            else "CPU"
        )

        return {

            "device": str(
                self.device
            ),

            "gpu": gpu_name,

            "step": (
                self.global_step
            ),

            "parameters": (
                self.parameter_count
            ),

            "context_length": (
                self.config.max_position_embeddings
            ),

            "vocab_size": (
                self.config.vocab_size
            ),

            "checkpoint": (
                CHECKPOINT_PATH.name
            ),

            "checkpoint_path": str(
                CHECKPOINT_PATH
            ),
        }