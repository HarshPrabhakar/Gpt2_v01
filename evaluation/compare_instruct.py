from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "mygpt2_base_v1.pt"
)

SFT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "instruct_checkpoints"
    / "best_instruct.pt"
)


# ============================================================
# Generation settings
# ============================================================

MAX_NEW_TOKENS = 120

TEMPERATURE = 0.7

TOP_K = 40

TOP_P = 0.90

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


PROMPTS = [
    "Hi",
    "What is RAM?",
    "Explain gravity to a beginner.",
    "What is machine learning?",
    "What is the capital of France?",
    "Write a Python function that adds two numbers.",
]


# ============================================================
# Helpers
# ============================================================

def config_to_dict(config):

    if isinstance(config, dict):
        return dict(config)

    if hasattr(config, "to_dict"):
        return config.to_dict()

    if hasattr(config, "__dict__"):
        return dict(vars(config))

    raise TypeError(
        f"Unsupported config type: {type(config)}"
    )


def extract_logits(output):

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        if output and isinstance(
            output[0],
            torch.Tensor,
        ):
            return output[0]

    if isinstance(output, dict):

        logits = output.get("logits")

        if isinstance(logits, torch.Tensor):
            return logits

    logits = getattr(
        output,
        "logits",
        None,
    )

    if isinstance(logits, torch.Tensor):
        return logits

    raise TypeError(
        f"Unable to extract logits from "
        f"{type(output)}"
    )


def load_model(checkpoint_path):

    print()
    print(
        f"Loading: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_config = checkpoint[
        "config"
    ]

    if isinstance(
        checkpoint_config,
        GPTConfig,
    ):
        config = checkpoint_config

    else:
        config = GPTConfig(
            **config_to_dict(
                checkpoint_config
            )
        )

    model = MyGPTModel(
        config
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    print("Strict load: PASS")

    return model, config


def top_k_top_p_filter(
    logits,
    top_k,
    top_p,
):

    # --------------------------------------------------------
    # Top-k
    # --------------------------------------------------------

    if top_k > 0:

        top_k = min(
            top_k,
            logits.size(-1),
        )

        threshold = torch.topk(
            logits,
            top_k,
        ).values[..., -1, None]

        logits = torch.where(
            logits < threshold,
            torch.full_like(
                logits,
                float("-inf"),
            ),
            logits,
        )

    # --------------------------------------------------------
    # Top-p
    # --------------------------------------------------------

    if (
        top_p > 0.0
        and top_p < 1.0
    ):

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

        sorted_remove = (
            cumulative_probabilities
            > top_p
        )

        # Keep the first token above the threshold.
        sorted_remove[..., 1:] = (
            sorted_remove[
                ..., :-1
            ].clone()
        )

        sorted_remove[..., 0] = False

        remove_mask = torch.zeros_like(
            logits,
            dtype=torch.bool,
        )

        remove_mask.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_remove,
        )

        logits = logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_position_embeddings,
):

    # Use the exact format used during SFT.
    formatted_prompt = (
        f"User: {prompt}\n\n"
        f"Assistant: "
    )

    encoded = tokenizer.encode(
        formatted_prompt
    )

    if isinstance(
        encoded,
        torch.Tensor,
    ):
        encoded = (
            encoded
            .detach()
            .cpu()
            .tolist()
        )

    encoded = [
        int(token)
        for token in encoded
    ]

    bos_token_id = getattr(
        tokenizer,
        "bos_token_id",
        2,
    )

    eos_token_id = getattr(
        tokenizer,
        "eos_token_id",
        3,
    )

    # Avoid duplicate BOS if tokenizer already inserts one.
    if (
        not encoded
        or encoded[0] != bos_token_id
    ):
        encoded = [
            bos_token_id
        ] + encoded

    # If tokenizer automatically appends EOS to encode(),
    # remove it because generation must continue.
    if (
        encoded
        and encoded[-1] == eos_token_id
    ):
        encoded = encoded[:-1]

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE,
    )

    generated_tokens = []

    for _ in range(
        MAX_NEW_TOKENS
    ):

        model_input = input_ids[
            :,
            -max_position_embeddings:,
        ]

        output = model(
            input_ids=model_input
        )

        logits = extract_logits(
            output
        )

        next_token_logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        next_token_logits = (
            top_k_top_p_filter(
                next_token_logits,
                top_k=TOP_K,
                top_p=TOP_P,
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

        token_id = int(
            next_token.item()
        )

        if token_id == eos_token_id:
            break

        generated_tokens.append(
            token_id
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    if not generated_tokens:
        return "<EOS>"

    return tokenizer.decode(
        generated_tokens
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("MYGPT2 BASE vs SFT COMPARISON")
    print("=" * 80)

    print()
    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Temperature : {TEMPERATURE}"
    )

    print(
        f"Top-k       : {TOP_K}"
    )

    print(
        f"Top-p       : {TOP_P}"
    )

    tokenizer = MyGPTTokenizer.load(
        TOKENIZER_PATH
    )

    print(
        f"Vocabulary  : "
        f"{tokenizer.vocabulary_size:,}"
    )

    base_model, base_config = (
        load_model(
            BASE_CHECKPOINT
        )
    )

    sft_model, sft_config = (
        load_model(
            SFT_CHECKPOINT
        )
    )

    base_context = int(
        base_config.max_position_embeddings
    )

    sft_context = int(
        sft_config.max_position_embeddings
    )

    if base_context != sft_context:
        raise RuntimeError(
            "Base and SFT context lengths differ."
        )

    # --------------------------------------------------------
    # Deterministic comparison
    # --------------------------------------------------------

    for prompt_index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):

        print()
        print("=" * 80)

        print(
            f"PROMPT {prompt_index}"
        )

        print("=" * 80)

        print()
        print(
            f"User: {prompt}"
        )

        # Same random seed means sampling differences are
        # primarily caused by the model distributions.
        torch.manual_seed(
            1000 + prompt_index
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                1000 + prompt_index
            )

        base_response = generate(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_position_embeddings=(
                base_context
            ),
        )

        torch.manual_seed(
            1000 + prompt_index
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                1000 + prompt_index
            )

        sft_response = generate(
            model=sft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_position_embeddings=(
                sft_context
            ),
        )

        print()
        print("--- BASE MODEL ---")
        print()
        print(base_response)

        print()
        print("--- SFT MODEL ---")
        print()
        print(sft_response)

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()