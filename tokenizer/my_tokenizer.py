"""
=========================================================
Project : MyGPT2
File    : my_tokenizer.py
Purpose : Wrapper around the Hugging Face Tokenizers
          library used throughout the project.

          This class does not train anything . It simply provides a clean interface
          to the Tokenizer
=========================================================
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class MyGPTTokenizer:
    """
    Wrapper around Hugging Face Tokenizer.

    This class provides a stable API for the rest of the
    project. Internally it uses the Hugging Face
    `tokenizers` library, but the rest of the project
    never depends on that implementation directly.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None
    ) -> None:

        self._tokenizer = tokenizer

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if a tokenizer is loaded.
        """

        return self._tokenizer is not None

    @property
    def vocabulary_size(self) -> int:
        """
        Number of learned vocabulary tokens.
        """

        self._ensure_loaded()

        return self._tokenizer.get_vocab_size()

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------

    def encode(
        self,
        text: str
    ) -> list[int]:
        """
        Convert text into token IDs.
        """

        self._ensure_loaded()

        encoding = self._tokenizer.encode(text)

        return encoding.ids

    # --------------------------------------------------
    # Decode
    # --------------------------------------------------

    def decode(
        self,
        token_ids: list[int]
    ) -> str:
        """
        Convert token IDs back into text.
        """

        self._ensure_loaded()

        return self._tokenizer.decode(token_ids)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(
        self,
        output_path: Path
    ) -> None:
        """
        Save tokenizer to disk.
        """

        self._ensure_loaded()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._tokenizer.save(str(output_path))

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    @classmethod
    def load(
        cls,
        tokenizer_path: Path
    ) -> "MyGPTTokenizer":
        """
        Load tokenizer from disk.
        """

        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        return cls(tokenizer)

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------

    def _ensure_loaded(self) -> None:
        """
        Ensure tokenizer has been initialized.
        """

        if self._tokenizer is None:

            raise RuntimeError(
                "Tokenizer has not been loaded or trained."
            )