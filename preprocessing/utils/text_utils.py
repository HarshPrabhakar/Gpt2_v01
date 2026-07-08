"""
=========================================================
Project : MyGPT2
File    : text_utils.py
Purpose : Text analysis utilities used throughout the
          preprocessing pipeline.
=========================================================
"""

from __future__ import annotations

import re


class TextAnalyzer:
    """
    Utility class for analyzing a single text document.

    This class provides lightweight text statistics that
    are reused across multiple preprocessing stages.
    """

    def __init__(self, text: str | None):

        self.text = text or ""

    # --------------------------------------------------
    # Basic Properties
    # --------------------------------------------------

    @property
    def normalized_text(self) -> str:
        """
        Returns the text with normalized whitespace.
        """

        return re.sub(r"\s+", " ", self.text).strip()

    @property
    def is_empty(self) -> bool:
        """
        Returns True if the text is empty after stripping
        whitespace.
        """

        return len(self.normalized_text) == 0

    # --------------------------------------------------
    # Character Statistics
    # --------------------------------------------------

    @property
    def character_count(self) -> int:

        return len(self.normalized_text)

    # --------------------------------------------------
    # Word Statistics
    # --------------------------------------------------

    @property
    def words(self) -> list[str]:

        if self.is_empty:
            return []

        return self.normalized_text.split()

    @property
    def word_count(self) -> int:

        return len(self.words)

    # --------------------------------------------------
    # Line Statistics
    # --------------------------------------------------

    @property
    def line_count(self) -> int:

        if self.is_empty:
            return 0

        return len(self.text.splitlines())

    @property
    def paragraph_count(self) -> int:
        """
        Paragraphs are separated by one or more blank lines.
        """

        if self.is_empty:
            return 0

        paragraphs = re.split(r"\n\s*\n", self.text)

        return len(
            [p for p in paragraphs if p.strip()]
        )

    # --------------------------------------------------
    # Sentence Statistics
    # --------------------------------------------------

    @property
    def sentence_count(self) -> int:
        """
        Estimates the number of sentences using punctuation.

        Version 1 uses a lightweight regex approach.
        """

        if self.is_empty:
            return 0

        sentences = re.split(
            r"[.!?]+",
            self.normalized_text
        )

        return len(
            [s for s in sentences if s.strip()]
        )

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    def preview(
        self,
        length: int = 300
    ) -> str:
        """
        Returns the first N characters of the text.
        """

        text = self.normalized_text

        if len(text) <= length:
            return text

        return text[:length] + "..."

    # --------------------------------------------------
    # Dictionary Export
    # --------------------------------------------------

    def to_dict(self) -> dict:
        """
        Export all available statistics.
        """

        return {

            "characters": self.character_count,

            "words": self.word_count,

            "lines": self.line_count,

            "paragraphs": self.paragraph_count,

            "sentences": self.sentence_count,

            "empty": self.is_empty,

            "preview": self.preview()

        }