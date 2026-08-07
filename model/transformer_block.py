"""
============================================================
GPT-2 Transformer Block
============================================================

One complete Transformer Block consisting of:

    LayerNorm
        ↓
    Multi-Head Self Attention
        ↓
    Residual Connection
        ↓
    LayerNorm
        ↓
    Feed Forward Network
        ↓
    Residual Connection

Author : Harsh Prabhakar
Project : MyGPT2
============================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import GPTConfig
from model.attention import MultiHeadSelfAttention
from model.mlp import GPTMLP


class TransformerBlock(nn.Module):
    """
    One GPT-2 Transformer Block.
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # -----------------------------------------
        # Layer Normalization 1
        # -----------------------------------------

        self.ln1 = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # -----------------------------------------
        # Multi-Head Attention
        # -----------------------------------------

        self.attention = MultiHeadSelfAttention(
            config
        )

        # -----------------------------------------
        # Layer Normalization 2
        # -----------------------------------------

        self.ln2 = nn.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        # -----------------------------------------
        # Feed Forward Network
        # -----------------------------------------

        self.mlp = GPTMLP(
            config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states

            Shape

            (batch_size,
             sequence_length,
             hidden_size)
        """

        # -----------------------------------------
        # Residual Connection 1
        # -----------------------------------------

        residual = hidden_states

        hidden_states = self.ln1(
            hidden_states
        )

        hidden_states = self.attention(
            hidden_states
        )

        hidden_states = residual + hidden_states

        # -----------------------------------------
        # Residual Connection 2
        # -----------------------------------------

        residual = hidden_states

        hidden_states = self.ln2(
            hidden_states
        )

        hidden_states = self.mlp(
            hidden_states
        )

        hidden_states = residual + hidden_states

        return hidden_states