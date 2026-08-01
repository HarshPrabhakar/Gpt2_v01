"""
============================================================
GPT-2 Embedding Layer
============================================================

This module converts token IDs into dense vectors and
adds learned positional embeddings.

Final Embedding = Token Embedding + Position Embedding

Author : Harsh Prabhakar
Project: MyGPT2
============================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import GPTConfig


class GPTEmbeddings(nn.Module):
    """
    GPT-2 Embedding Layer.

    Combines:

        • Token Embedding
        • Position Embedding

    Output Shape

        (batch_size,
         sequence_length,
         hidden_size)
    """

    def __init__(self, config: GPTConfig):

        super().__init__()

        self.config = config

        # -------------------------------------------------
        # Token Embedding
        # -------------------------------------------------

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        # -------------------------------------------------
        # Position Embedding
        # -------------------------------------------------

        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )

        # -------------------------------------------------
        # Dropout
        # -------------------------------------------------

        self.dropout = nn.Dropout(
            config.embedding_dropout
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids

            Shape

            (batch_size,
             sequence_length)

        Returns
        -------
        embeddings

            Shape

            (batch_size,
             sequence_length,
             hidden_size)
        """

        batch_size, sequence_length = input_ids.shape

        device = input_ids.device

        # ---------------------------------------------
        # Create Position IDs
        # ---------------------------------------------

        position_ids = torch.arange(
            sequence_length,
            device=device,
        )

        position_ids = position_ids.unsqueeze(0)

        # Shape

        # (1, sequence_length)

        # ---------------------------------------------
        # Lookup Embeddings
        # ---------------------------------------------

        token_embeddings = self.token_embeddings(
            input_ids
        )

        position_embeddings = self.position_embeddings(
            position_ids
        )

        # ---------------------------------------------
        # Combine
        # ---------------------------------------------

        embeddings = (
            token_embeddings
            + position_embeddings
        )

        embeddings = self.dropout(
            embeddings
        )

        return embeddings