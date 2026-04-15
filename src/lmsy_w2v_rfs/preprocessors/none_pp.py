"""Passthrough preprocessor: whitespace split, lowercase, no parse, no NER."""

from __future__ import annotations

import re


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class NoOpPreprocessor:
    """Trivial preprocessor: split on sentence-ending punctuation, lowercase.

    Fastest possible path. Useful for quick iteration, for tests, and for
    users who only want the gensim ``Phrases`` + Word2Vec half of the
    pipeline with a curated static MWE list applied afterwards.
    """

    def process(self, text: str) -> list[list[str]]:
        """Split a document into whitespace-tokenized, lowercased sentences.

        Args:
            text: Raw document.

        Returns:
            List of sentences, each a list of tokens.
        """
        text = text.replace("\n", " ")
        sentences = _SENT_SPLIT.split(text.strip()) or [text.strip()]
        out: list[list[str]] = []
        for s in sentences:
            toks = s.lower().split()
            if toks:
                out.append(toks)
        return out
