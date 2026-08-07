"""
=============================================================
MyGPT2 Language Model
=============================================================

This module defines the main GPT architecture.

Responsibilities
----------------
• Build Embedding Layer
• Build Transformer Stack
• Final LayerNorm
• Language Modeling Head
• Weight Tying

Forward propagation and utility methods are implemented
inside separate modules.

Project : MyGPT2
Author  : Harsh Prabhakar
=============================================================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.config import GPTConfig
from model.embeddings import GPTEmbeddings
from model.transformer_block import TransformerBlock

# Helper Modules
from model.forward import forward
from model.initialization import initialize_weights
from model.optimizer import configure_optimizer
from model.checkpoint import save_checkpoint
from model.checkpoint import load_checkpoint
from model.parameter_counter import count_parameters


class MyGPTModel(nn.Module):
    """
    GPT-2 Style Decoder Only Transformer
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # -------------------------------------------------
        # Embedding Layer
        # -------------------------------------------------

        self.embeddings = GPTEmbeddings(config)

        # -------------------------------------------------
        # Transformer Blocks
        # -------------------------------------------------

        self.blocks = nn.ModuleList(

            [
                TransformerBlock(config)

                for _ in range(config.num_layers)

            ]

        )

        # -------------------------------------------------
        # Final Layer Normalization
        # -------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(

            normalized_shape=config.hidden_size,

            eps=config.layer_norm_epsilon

        )

        # -------------------------------------------------
        # Language Modeling Head
        # -------------------------------------------------

        self.lm_head = nn.Linear(

            in_features=config.hidden_size,

            out_features=config.vocab_size,

            bias=False

        )

        # -------------------------------------------------
        # Weight Tying
        # -------------------------------------------------

        self.lm_head.weight = (

            self.embeddings.token_embeddings.weight

        )

        # -------------------------------------------------
        # Weight Initialization
        # -------------------------------------------------

        initialize_weights(self)

        # -------------------------------------------------
        # Print Summary
        # -------------------------------------------------

        total = count_parameters(self)

        print()

        print("=" * 60)
        print("MyGPT2 Successfully Built")
        print("=" * 60)

        print(f"Vocabulary Size : {config.vocab_size:,}")
        print(f"Context Length  : {config.max_position_embeddings}")
        print(f"Hidden Size     : {config.hidden_size}")
        print(f"Layers          : {config.num_layers}")
        print(f"Heads           : {config.num_attention_heads}")
        print(f"Parameters      : {total:,}")

        print("=" * 60)

    # =====================================================
    # Forward
    # =====================================================

    forward = forward

    # =====================================================
    # Optimizer
    # =====================================================

    configure_optimizer = configure_optimizer

    # =====================================================
    # Save
    # =====================================================

    save = save_checkpoint

    # =====================================================
    # Load
    # =====================================================

    load = load_checkpoint