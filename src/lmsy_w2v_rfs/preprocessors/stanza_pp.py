"""stanza.Pipeline preprocessor (no Java required).

Uses the native neural pipeline for tokenize/pos/lemma/depparse/ner. The
MWE join uses the UD v2 labels ``fixed`` / ``flat`` / ``compound``. NER
spans are replaced with ``[NER:TYPE]`` placeholders.

This preprocessor is the slowest of the three parser-based backends (CPU,
no batching) and produces the largest vocabulary because stanza's NER
model is more type-fine-grained than CoreNLP's default setting. Prefer
``preprocessor="spacy"`` for speed, or ``preprocessor="corenlp"`` for
paper-faithful syntactic MWE coverage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


log = logging.getLogger(__name__)


_MWE_DEPS = {"fixed", "flat", "compound"}


class StanzaPreprocessor:
    """stanza.Pipeline-based preprocessor.

    Attributes:
        nlp: Loaded stanza Pipeline.
    """

    def __init__(self, config: "Config") -> None:
        """Load the stanza pipeline.

        Args:
            config: Pipeline config.

        Raises:
            ImportError: If stanza is not installed.
        """
        import stanza

        # Download the English model on first use; stanza caches to
        # ~/.stanza_resources by default.
        stanza.download("en", verbose=False, processors="tokenize,pos,lemma,depparse,ner")
        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,lemma,depparse,ner",
            use_gpu=False,
            verbose=False,
            tokenize_no_ssplit=False,
        )
        log.info("StanzaPreprocessor loaded")

    def process(self, text: str) -> list[list[str]]:
        """Parse one document.

        Args:
            text: Raw document.

        Returns:
            List of sentences, each a list of tokens.
        """
        doc = self.nlp(text)
        out: list[list[str]] = []
        for s in doc.sentences:
            out.append(self._sentence_tokens(s))
        return out

    def _sentence_tokens(self, sent) -> list[str]:
        # stanza.word.id is 1-based within the sentence.
        # sent.ents contains Span objects with .tokens; each token has .id.

        # NER: word id -> (type, is_start). Only first word gets the placeholder.
        ner_at: dict[int, tuple[str, bool]] = {}
        for ent in sent.ents:
            for j, tok in enumerate(ent.tokens):
                tid = tok.id[0] if isinstance(tok.id, tuple) else tok.id
                ner_at[tid] = (ent.type, j == 0)

        # Build id -> word map for fast lookup.
        by_id = {w.id: w for w in sent.words}

        # MWE join pairs. Only join adjacent pairs (the paper did the same).
        mwe_after: set[int] = set()
        for w in sent.words:
            if w.deprel in _MWE_DEPS or w.deprel.startswith("compound:"):
                head = w.head
                if head == 0:
                    continue
                lo, hi = sorted((w.id, head))
                if hi - lo == 1:
                    mwe_after.add(lo)

        words_sorted = sorted(sent.words, key=lambda w: w.id)
        out: list[str] = []
        i = 0
        while i < len(words_sorted):
            w = words_sorted[i]
            if w.id in ner_at:
                etype, is_start = ner_at[w.id]
                if is_start:
                    out.append(f"[NER:{etype}]")
                i += 1
                continue
            parts = [(w.lemma or w.text).lower()]
            # Absorb an adjacent MWE-linked token, but not across NER boundaries.
            while w.id in mwe_after and i + 1 < len(words_sorted):
                i += 1
                w = words_sorted[i]
                if w.id in ner_at:
                    break
                parts.append((w.lemma or w.text).lower())
            joined = "_".join(parts)
            if joined.strip():
                out.append(joined)
            i += 1
        return out
