"""
============================================================
MyGPTModel
============================================================

Complete GPT-2 Style Language Model

Architecture:

Input Tokens
      ↓
Embeddings
      ↓
Transformer Blocks (12x)
      ↓
Final LayerNorm
      ↓
Language Modeling Head
      ↓
Vocabulary Logits

Author : Harsh Prabhakar
Project : MyGPT2
============================================================
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig
from model.embeddings import GPTEmbeddings
from model.transformer_block import TransformerBlock


class MyGPTModel(nn.Module):
    """
    GPT-2 Style Language Model.
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # ====================================================
        # Embedding Layer
        # ====================================================

        self.embeddings = GPTEmbeddings(
            config
        )

        # ====================================================
        # Transformer Stack
        # ====================================================

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # ====================================================
        # Final Layer Normalization
        # ====================================================

        self.final_layer_norm = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # ====================================================
        # Language Modeling Head
        # ====================================================

        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

        # ====================================================
        # Weight Tying
        # ====================================================

        self.lm_head.weight = (
            self.embeddings.token_embeddings.weight
        )

        # ====================================================
        # Initialize Weights
        # ====================================================

        self.apply(
            self._init_weights
        )

        # ====================================================
        # Print Model Information
        # ====================================================

        print("\n" + "=" * 60)
        print("MyGPTModel Initialized")
        print("=" * 60)

        print(
            f"Vocabulary Size : "
            f"{config.vocab_size:,}"
        )

        print(
            f"Hidden Size     : "
            f"{config.hidden_size}"
        )

        print(
            f"Layers          : "
            f"{config.num_layers}"
        )

        print(
            f"Heads           : "
            f"{config.num_attention_heads}"
        )

        print(
            f"Context Length  : "
            f"{config.max_position_embeddings}"
        )

        print(
            f"Device          : "
            f"{config.device}"
        )

        print("=" * 60)

    # ========================================================
    # Weight Initialization Placeholder
    # ========================================================

    def _init_weights(
        self,
        module: nn.Module
    ):
        """
        GPT-style initialization.

        Full implementation added in Part 2.
        """
        pass