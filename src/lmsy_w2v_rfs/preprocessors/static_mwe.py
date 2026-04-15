"""Static-MWE-only preprocessor.

No parser, no NER. Splits on sentences, tokenizes, applies NLTK's
``MWETokenizer`` with a curated finance-flavored MWE list. Fast, fully
deterministic, and trivially extensible (users edit a text file).

This is also the implementation of the ``mwe_list`` post-pass applied after
the other preprocessors. See ``base.apply_mwe_list``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import apply_mwe_list, load_mwe_list


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class StaticMWEPreprocessor:
    """Whitespace tokenize + NLTK MWE concatenation.

    Attributes:
        mwe_list: Loaded MWE tuples used for tokenization.
        lowercase: Whether to lowercase tokens before matching.
    """

    def __init__(
        self,
        mwe_source: str | Path = "finance",
        lowercase: bool = True,
    ) -> None:
        """Initialize.

        Args:
            mwe_source: ``"finance"`` for the packaged list, or a path.
            lowercase: Whether to lowercase tokens before matching.
        """
        self.mwe_list = load_mwe_list(mwe_source)
        self.lowercase = lowercase

    def process(self, text: str) -> list[list[str]]:
        """Tokenize and apply the MWE list.

        Args:
            text: Raw document.

        Returns:
            List of sentences, each a list of tokens (MWEs joined by ``_``).
        """
        text = text.replace("\n", " ")
        sentences = _SENT_SPLIT.split(text.strip()) or [text.strip()]
        raw: list[list[str]] = []
        for s in sentences:
            toks = s.split()
            toks = [self._strip(t) for t in toks]
            toks = [t for t in toks if t]
            if toks:
                raw.append(toks)
        return apply_mwe_list(raw, self.mwe_list)

    def _strip(self, tok: str) -> str:
        t = tok.lower() if self.lowercase else tok
        # Strip trailing and leading punctuation; keep internal hyphens/apostrophes.
        return t.strip(".,;:!?()[]{}\"'`")
