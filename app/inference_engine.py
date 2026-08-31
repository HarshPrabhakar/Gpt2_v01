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
    / "checkpoints"
    / "final_step_00213751.pt"
)


class MyGPTInferenceEngine:

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.checkpoint = self._load_checkpoint()

        self.config = self._restore_config()

        self.tokenizer = self._load_tokenizer()

        self.model = self._load_model()

        self.global_step = self.checkpoint.get(
            "global_step",
            None,
        )

        self.parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )


    # ========================================================
    # Checkpoint
    # ========================================================

    def _load_checkpoint(self) -> dict[str, Any]:

        if not CHECKPOINT_PATH.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n{CHECKPOINT_PATH}"
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

        return checkpoint


    # ========================================================
    # Config
    # ========================================================

    def _restore_config(self) -> GPTConfig:

        config = GPTConfig()

        saved_config = self.checkpoint.get(
            "config"
        )

        if isinstance(
            saved_config,
            dict,
        ):

            for key, value in saved_config.items():

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


    # ========================================================
    # Tokenizer
    # ========================================================

    def _load_tokenizer(self) -> MyGPTTokenizer:

        if not TOKENIZER_PATH.exists():

            raise FileNotFoundError(
                f"Tokenizer not found:\n{TOKENIZER_PATH}"
            )

        tokenizer = MyGPTTokenizer.load(
            TOKENIZER_PATH
        )

        if (
            tokenizer.vocabulary_size
            != self.config.vocab_size
        ):

            raise RuntimeError(
                "Tokenizer vocabulary does not match model vocabulary."
            )

        return tokenizer


    # ========================================================
    # Model
    # ========================================================

    def _load_model(self) -> MyGPTModel:

        model = MyGPTModel(
            self.config
        )

        model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        model = model.to(
            self.device
        )

        model.eval()

        return model


    # ========================================================
    # Sampling Helpers
    # ========================================================

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:

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

        logits = torch.where(
            logits < threshold,
            torch.full_like(
                logits,
                float("-inf"),
            ),
            logits,
        )

        return logits


    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:

        if (
            top_p <= 0.0
            or
            top_p >= 1.0
        ):
            return logits

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probs = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        sorted_mask = (
            cumulative_probs
            >
            top_p
        )

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
            sorted_mask
        )

        mask.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_mask,
        )

        logits = logits.masked_fill(
            mask,
            float("-inf"),
        )

        return logits


    # ========================================================
    # Generation
    # ========================================================

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> dict[str, Any]:

        prompt = prompt.strip()

        if not prompt:

            return {
                "text": "",
                "generated_tokens": 0,
                "elapsed_seconds": 0.0,
                "tokens_per_second": 0.0,
                "input_tokens": 0,
            }


        token_ids = self.tokenizer.encode(
            prompt
        )


        if not token_ids:

            return {
                "text": "",
                "generated_tokens": 0,
                "elapsed_seconds": 0.0,
                "tokens_per_second": 0.0,
                "input_tokens": 0,
            }


        input_tokens = len(
            token_ids
        )


        generated = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )


        eos_token_id = getattr(
            self.config,
            "eos_token_id",
            None,
        )


        start_time = time.perf_counter()


        if self.device.type == "cuda":

            torch.cuda.synchronize()


        actual_generated_tokens = 0


        for _ in range(
            max_new_tokens
        ):

            context = generated[
                :,
                -self.config.max_position_embeddings:
            ]


            output = self.model(
                input_ids=context
            )


            if isinstance(
                output,
                tuple,
            ):

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

                logits = output[
                    "logits"
                ]

            else:

                logits = output


            next_token_logits = logits[
                :,
                -1,
                :
            ]


            temperature = max(
                float(temperature),
                1e-5,
            )


            next_token_logits = (
                next_token_logits
                /
                temperature
            )


            next_token_logits = (
                self._apply_top_k(
                    next_token_logits,
                    top_k,
                )
            )


            next_token_logits = (
                self._apply_top_p(
                    next_token_logits,
                    top_p,
                )
            )


            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
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


            if (
                eos_token_id is not None
                and
                int(
                    next_token.item()
                )
                ==
                int(
                    eos_token_id
                )
            ):

                break


        if self.device.type == "cuda":

            torch.cuda.synchronize()


        elapsed_seconds = (
            time.perf_counter()
            -
            start_time
        )


        full_token_ids = (
            generated[
                0
            ]
            .detach()
            .cpu()
            .tolist()
        )


        full_text = self.tokenizer.decode(
            full_token_ids
        )


        generated_text = full_text


        if (
            full_text.startswith(
                prompt
            )
        ):

            generated_text = (
                full_text[
                    len(prompt):
                ]
            )


        tokens_per_second = (

            actual_generated_tokens
            /
            elapsed_seconds

            if elapsed_seconds > 0

            else 0.0

        )


        return {

            "text": (
                generated_text.strip()
            ),

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

        }


    # ========================================================
    # Status
    # ========================================================

    def status(self) -> dict[str, Any]:

        gpu_name = (

            torch.cuda.get_device_name(
                self.device
            )

            if self.device.type
            ==
            "cuda"

            else "CPU"

        )

        return {

            "device": str(
                self.device
            ),

            "gpu": gpu_name,

            "step": self.global_step,

            "parameters": self.parameter_count,

            "context_length": (
                self.config.max_position_embeddings
            ),

            "vocab_size": (
                self.config.vocab_size
            ),

        }