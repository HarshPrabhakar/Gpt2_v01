"""
============================================================
GPT-2 Multi-Head Self Attention
============================================================

Implements masked multi-head self-attention from scratch.

Architecture

Input
   │
   ▼
Q K V Projection
   │
Split Heads
   │
Scaled Dot Product Attention
   │
Merge Heads
   │
Output Projection

============================================================
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig


class MultiHeadSelfAttention(nn.Module):
    """
    GPT-2 Masked Multi-Head Self Attention.
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        # --------------------------------------------------
        # QKV Projection
        # --------------------------------------------------

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size * 3,
            bias=config.use_bias,
        )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        self.out_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.use_bias,
        )

        self.attn_dropout = nn.Dropout(
            config.attention_dropout
        )

        self.resid_dropout = nn.Dropout(
            config.dropout
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

        batch_size, sequence_length, _ = hidden_states.shape

        # --------------------------------------------------
        # Compute QKV
        # --------------------------------------------------

        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv.chunk(3, dim=-1)

        # --------------------------------------------------
        # Split into attention heads
        # --------------------------------------------------

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Shape:
        # (batch,
        #  heads,
        #  seq_len,
        #  head_dim)

        # --------------------------------------------------
        # Attention Scores
        # --------------------------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # --------------------------------------------------
        # Causal Mask
        # --------------------------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=hidden_states.device,
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------

        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention = self.attn_dropout(
            attention
        )

        # --------------------------------------------------
        # Weighted Sum
        # --------------------------------------------------

        context = attention @ v

        # --------------------------------------------------
        # Merge Heads
        # --------------------------------------------------

        context = context.transpose(
            1,
            2,
        ).contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        # --------------------------------------------------
        # Final Projection
        # --------------------------------------------------

        output = self.out_proj(
            context
        )

        output = self.resid_dropout(
            output
        )

        return output