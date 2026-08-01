"""
============================================================
GPT-2 Configuration
============================================================

This module contains all configuration parameters required
to build and train the GPT model.

Every component of the model receives the same configuration
object, ensuring consistency across the entire architecture.

Author : Harsh Prabhakar
Project: MyGPT2
============================================================
"""

from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    """
    Configuration for GPT-2 Small.

    All dimensions and hyperparameters are stored here so the
    architecture can easily be scaled to Medium, Large, or XL
    by modifying this file only.
    """

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    vocab_size: int = 32000

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    max_position_embeddings: int = 512

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    hidden_size: int = 768

    num_layers: int = 12

    num_attention_heads: int = 12

    intermediate_size: int = 3072

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout: float = 0.1

    attention_dropout: float = 0.1

    embedding_dropout: float = 0.1

    # --------------------------------------------------------
    # Layer Normalization
    # --------------------------------------------------------

    layer_norm_epsilon: float = 1e-5

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    initializer_range: float = 0.02

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    batch_size: int = 8

    learning_rate: float = 3e-4

    weight_decay: float = 0.01

    max_epochs: int = 10

    gradient_clip: float = 1.0

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    temperature: float = 1.0

    top_k: int = 50

    top_p: float = 0.95

    # --------------------------------------------------------
    # Special Tokens
    # --------------------------------------------------------

    pad_token_id: int = 0

    unk_token_id: int = 1

    bos_token_id: int = 2

    eos_token_id: int = 3

    # --------------------------------------------------------
    # Miscellaneous
    # --------------------------------------------------------

    use_bias: bool = True

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = torch.float32

    seed: int = 42

    # --------------------------------------------------------
    # Derived Values
    # --------------------------------------------------------

    @property
    def head_dim(self) -> int:
        """
        Dimension of one attention head.
        """

        return self.hidden_size // self.num_attention_heads

    @property
    def model_size(self) -> str:
        return "GPT-2 Small"

    @property
    def total_attention_dimensions(self) -> int:
        return self.head_dim * self.num_attention_heads