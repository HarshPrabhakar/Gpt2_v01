"""
============================================================
MyGPT2 - Main GPT Language Model
============================================================

A GPT-2 style decoder-only Transformer language model.

Architecture:

    Input Token IDs
          │
          ▼
    Token + Position Embeddings
          │
          ▼
    Transformer Blocks × N
          │
          ▼
    Final LayerNorm
          │
          ▼
    Language Modeling Head
          │
          ▼
    Vocabulary Logits

Project : MyGPT2
============================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig
from model.embeddings import GPTEmbeddings
from model.transformer_block import TransformerBlock


class MyGPTModel(nn.Module):
    """
    GPT-2 style decoder-only Transformer.

    Parameters
    ----------
    config : GPTConfig
        Configuration containing model dimensions and hyperparameters.

    Input
    -----
    input_ids:
        Tensor of token IDs.

        Shape:
            (batch_size, sequence_length)

    labels:
        Optional target token IDs.

        Shape:
            (batch_size, sequence_length)

    Output
    ------
    logits:
        Tensor containing vocabulary scores.

        Shape:
            (batch_size, sequence_length, vocab_size)

    loss:
        Cross-entropy loss when labels are supplied.
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # ==================================================
        # Token + Position Embeddings
        # ==================================================

        self.embeddings = GPTEmbeddings(config)

        # ==================================================
        # Transformer Blocks
        # ==================================================

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # ==================================================
        # Final Layer Normalization
        # ==================================================

        self.final_layer_norm = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # ==================================================
        # Language Modeling Head
        # ==================================================

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # ==================================================
        # Weight Tying
        # ==================================================
        #
        # The output projection shares its weights with
        # the token embedding matrix.
        #
        # This is the same general idea used by GPT-style
        # language models and reduces parameter count.
        # ==================================================

        self.lm_head.weight = (
            self.embeddings.token_embeddings.weight
        )

        # ==================================================
        # Initialize Parameters
        # ==================================================

        self._initialize_weights()

    # ======================================================
    # Weight Initialization
    # ======================================================

    def _initialize_weights(
        self,
    ) -> None:
        """
        Initialize model parameters.

        Linear and embedding weights use a normal
        distribution.

        LayerNorm weights are initialized to 1 and
        biases to 0.

        GPT-2 style residual scaling is applied to
        projection layers inside Transformer blocks.
        """

        # --------------------------------------------------
        # Standard GPT-style initialization
        # --------------------------------------------------

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )

            elif isinstance(module, nn.LayerNorm):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        # --------------------------------------------------
        # GPT-2 residual projection scaling
        # --------------------------------------------------
        #
        # As the number of Transformer layers increases,
        # residual branches can become increasingly large.
        #
        # Scaling the residual projection weights helps
        # maintain stable activations.
        # --------------------------------------------------

        residual_std = (
            self.config.initializer_range
            /
            (
                2.0
                *
                self.config.num_layers
            ) ** 0.5
        )

        for name, module in self.named_modules():

            if not isinstance(module, nn.Linear):
                continue

            # Attention output projection
            if name.endswith("attention.out_proj"):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=residual_std,
                )

            # MLP output projection
            elif name.endswith("mlp.fc2"):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=residual_std,
                )

        # --------------------------------------------------
        # Re-establish weight tying
        # --------------------------------------------------

        self.lm_head.weight = (
            self.embeddings.token_embeddings.weight
        )

    # ======================================================
    # Forward Pass
    # ======================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run a forward pass through the GPT model.

        Parameters
        ----------
        input_ids:
            Token IDs.

            Shape:
                (batch_size, sequence_length)

        labels:
            Optional target token IDs.

            Shape:
                (batch_size, sequence_length)

            If provided, cross-entropy loss is calculated.

        Returns
        -------
        logits:
            Vocabulary logits.

            Shape:
                (batch_size, sequence_length, vocab_size)

        loss:
            Cross-entropy loss if labels are provided.
            Otherwise None.
        """

        # ==================================================
        # Validate Input
        # ==================================================

        if input_ids.dim() != 2:

            raise ValueError(
                "input_ids must have shape "
                "(batch_size, sequence_length). "
                f"Received shape: {tuple(input_ids.shape)}"
            )

        batch_size, sequence_length = input_ids.shape

        # ==================================================
        # Context Length Check
        # ==================================================

        if (
            sequence_length
            >
            self.config.max_position_embeddings
        ):

            raise ValueError(
                "Sequence length exceeds model context "
                f"length.\n"
                f"Sequence length: {sequence_length}\n"
                f"Maximum length: "
                f"{self.config.max_position_embeddings}"
            )

        # ==================================================
        # Embeddings
        # ==================================================

        hidden_states = self.embeddings(
            input_ids
        )

        # ==================================================
        # Transformer Stack
        # ==================================================

        for block in self.blocks:

            hidden_states = block(
                hidden_states
            )

        # ==================================================
        # Final LayerNorm
        # ==================================================

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        # ==================================================
        # Language Modeling Head
        # ==================================================

        logits = self.lm_head(
            hidden_states
        )

        # ==================================================
        # Loss
        # ==================================================

        loss = None

        if labels is not None:

            if labels.shape != input_ids.shape:

                raise ValueError(
                    "labels must have the same shape "
                    "as input_ids.\n"
                    f"input_ids: {tuple(input_ids.shape)}\n"
                    f"labels:    {tuple(labels.shape)}"
                )

            # ------------------------------------------------
            # Flatten batch and sequence dimensions
            # ------------------------------------------------

            logits_flat = logits.reshape(
                -1,
                self.config.vocab_size,
            )

            labels_flat = labels.reshape(
                -1
            )

            # ------------------------------------------------
            # Cross Entropy
            # ------------------------------------------------

            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
            )

        return logits, loss

    # ======================================================
    # Parameter Count
    # ======================================================

    def get_num_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        """
        Return the number of model parameters.

        Parameters
        ----------
        trainable_only:
            If True, only parameters with
            requires_grad=True are counted.
        """

        if trainable_only:

            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    # ======================================================
    # Model Summary
    # ======================================================

    def print_model_summary(
        self,
    ) -> None:
        """
        Print basic model information.
        """

        total_parameters = (
            self.get_num_parameters()
        )

        print()
        print("=" * 70)
        print("                    MyGPT2 Model")
        print("=" * 70)

        print(
            f"Vocabulary Size       : "
            f"{self.config.vocab_size:,}"
        )

        print(
            f"Context Length        : "
            f"{self.config.max_position_embeddings:,}"
        )

        print(
            f"Hidden Size           : "
            f"{self.config.hidden_size:,}"
        )

        print(
            f"Transformer Layers    : "
            f"{self.config.num_layers}"
        )

        print(
            f"Attention Heads       : "
            f"{self.config.num_attention_heads}"
        )

        print(
            f"Parameters            : "
            f"{total_parameters:,}"
        )

        print(
            f"Parameters (Millions) : "
            f"{total_parameters / 1_000_000:.2f}M"
        )

        print("=" * 70)
        print()

    # ======================================================
    # Device Helper
    # ======================================================

    def get_device(
        self,
    ) -> torch.device:
        """
        Return the device currently used by the model.
        """

        return next(
            self.parameters()
        ).device

    # ======================================================
    # Generate Next Token
    # ======================================================

    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate one next-token prediction.

        This is a simple baseline implementation.

        KV caching and advanced sampling will be added
        later in generation.py.
        """

        if temperature <= 0:

            raise ValueError(
                "temperature must be greater than 0."
            )

        logits, _ = self(
            input_ids
        )

        # --------------------------------------------------
        # Only use logits from the final position
        # --------------------------------------------------

        next_token_logits = logits[:, -1, :]

        # --------------------------------------------------
        # Temperature scaling
        # --------------------------------------------------

        next_token_logits = (
            next_token_logits
            /
            temperature
        )

        # --------------------------------------------------
        # Convert logits to probabilities
        # --------------------------------------------------

        probabilities = F.softmax(
            next_token_logits,
            dim=-1,
        )

        # --------------------------------------------------
        # Sample next token
        # --------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        return next_token

    # ======================================================
    # Basic Generation
    # ======================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Basic autoregressive text generation.

        Parameters
        ----------
        input_ids:
            Starting token IDs.

        max_new_tokens:
            Number of new tokens to generate.

        temperature:
            Sampling temperature.

        Returns
        -------
        Tensor containing original + generated tokens.
        """

        if max_new_tokens < 0:

            raise ValueError(
                "max_new_tokens must be >= 0."
            )

        for _ in range(
            max_new_tokens
        ):

            # ------------------------------------------------
            # Keep sequence inside context window
            # ------------------------------------------------

            if (
                input_ids.shape[1]
                >
                self.config.max_position_embeddings
            ):

                input_ids_for_model = (
                    input_ids[
                        :,
                        -self.config.max_position_embeddings:
                    ]
                )

            else:

                input_ids_for_model = (
                    input_ids
                )

            # ------------------------------------------------
            # Predict next token
            # ------------------------------------------------

            next_token = (
                self.generate_next_token(
                    input_ids_for_model,
                    temperature=temperature,
                )
            )

            # ------------------------------------------------
            # Append token
            # ------------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )

        return input_ids