"""
============================================================
GPT-2 Feed Forward Network (MLP)
============================================================

This module implements the position-wise feed-forward
network used inside every Transformer block.

Architecture

Input
   │
   ▼
Linear (768 → 3072)
   │
   ▼
GELU
   │
   ▼
Linear (3072 → 768)
   │
   ▼
Dropout

Author : Harsh Prabhakar
Project: MyGPT2
============================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import GPTConfig


class GPTMLP(nn.Module):
    """
    Position-wise Feed Forward Network.

    Input Shape
        (batch_size,
         sequence_length,
         hidden_size)

    Output Shape
        (batch_size,
         sequence_length,
         hidden_size)
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # -------------------------------------------------
        # First Linear Layer
        # -------------------------------------------------

        self.fc1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=config.use_bias,
        )

        # -------------------------------------------------
        # GELU Activation
        # -------------------------------------------------

        self.activation = nn.GELU()

        # -------------------------------------------------
        # Second Linear Layer
        # -------------------------------------------------

        self.fc2 = nn.Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=config.use_bias,
        )

        # -------------------------------------------------
        # Dropout
        # -------------------------------------------------

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward Pass
        """

        hidden_states = self.fc1(hidden_states)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout(hidden_states)

        return hidden_states