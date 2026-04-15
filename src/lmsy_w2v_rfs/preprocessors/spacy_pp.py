"""spaCy preprocessor (fast, Python-native alternative to CoreNLP).

Parses, lemmatizes, masks named entities as ``[NER:TYPE]`` placeholders, and
joins tokens linked by ``fixed`` / ``flat`` / ``compound`` dependencies.

On the 150-doc benchmark this was 9x faster than stanza and 2x faster than
the CoreNLP server, produced the cleanest NER output (18 types, zero
punctuation contamination), and the smallest Word2Vec-ready vocabulary. It
loses on syntactic MWE coverage (0% on the grammaticalized-fixed test set)
because spaCy's English UD converter does not emit ``fixed`` or
``compound:prt``. Pair with a curated ``mwe_list`` or switch to
``preprocessor="corenlp"`` when syntactic MWE matters.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


log = logging.getLogger(__name__)


_MWE_DEPS = {"fixed", "flat", "compound"}


class SpacyPreprocessor:
    """spaCy-based preprocessor.

    Attributes:
        nlp: Loaded spaCy ``Language``.
        model_name: Name of the loaded model.
    """

    def __init__(self, config: "Config") -> None:
        """Load the spaCy model named in ``config.spacy_model``.

        Args:
            config: Pipeline config.

        Raises:
            ImportError: If spaCy is not installed.
            OSError: If the requested model is not downloaded.
        """
        import os

        # Pin all BLAS libraries to 1 thread per worker BEFORE spaCy workers
        # fork. Without this, N workers * N BLAS threads = severe oversubscription
        # on CPU. See docs/how-to/run-on-hpc.md.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass

        import spacy

        self._n_cores = config.n_cores
        self.model_name = config.spacy_model
        try:
            self.nlp = spacy.load(config.spacy_model)
        except OSError as e:  # pragma: no cover
            raise OSError(
                f"spaCy model '{config.spacy_model}' is not installed. "
                f"Run: python -m spacy download {config.spacy_model}"
            ) from e
        log.info("SpacyPreprocessor loaded model %s", config.spacy_model)

    def process(self, text: str) -> list[list[str]]:
        """Parse one document.

        Args:
            text: Raw document.

        Returns:
            List of sentences, each a list of tokens.
        """
        doc = self.nlp(text)
        out: list[list[str]] = []
        for sent in doc.sents:
            out.append(self._sentence_tokens(sent))
        return out

    def process_documents(
        self, texts: Iterable[str]
    ) -> Iterator[list[list[str]]]:
        """Fan out via ``nlp.pipe(n_process=n_cores)``.

        spaCy's native pipe is the right way to process many documents:
        it batches on the C side and, with ``n_process>1``, uses Python
        multiprocessing. We apply the thread-oversubscription fix
        (``torch.set_num_threads(1)``) in ``__init__`` so PyTorch under
        each worker does not fight for cores.

        Args:
            texts: Iterable of raw documents.

        Yields:
            Preprocessed documents in input order.
        """
        n_proc = max(1, self._n_cores)
        for doc in self.nlp.pipe(texts, batch_size=50, n_process=n_proc):
            out: list[list[str]] = []
            for sent in doc.sents:
                out.append(self._sentence_tokens(sent))
            yield out

    def _sentence_tokens(self, sent) -> list[str]:
        # Mark NER tokens: {token_index: (entity_type, is_start)}
        ner_at: dict[int, tuple[str, bool]] = {}
        for ent in sent.ents:
            for j, tok in enumerate(ent):
                ner_at[tok.i] = (ent.label_, j == 0)

        # MWE join indices: for each MWE edge, mark the lower index.
        # A join of index i means token i concatenates to token i+1.
        mwe_after: set[int] = set()
        for tok in sent:
            if tok.dep_ in _MWE_DEPS or tok.dep_.startswith("compound:"):
                lo = min(tok.i, tok.head.i)
                hi = max(tok.i, tok.head.i)
                if hi - lo == 1:  # only join adjacent pairs; paper did the same
                    mwe_after.add(lo)

        # Emit tokens.
        out: list[str] = []
        i = 0
        sent_tokens = list(sent)
        while i < len(sent_tokens):
            tok = sent_tokens[i]
            if tok.i in ner_at:
                etype, is_start = ner_at[tok.i]
                # Skip all tokens in the entity; emit one placeholder at start.
                if is_start:
                    out.append(f"[NER:{etype}]")
                i += 1
                continue
            # Build an MWE run: while the current global index is in mwe_after,
            # concatenate with the next token.
            parts = [tok.lemma_.lower()]
            while tok.i in mwe_after and i + 1 < len(sent_tokens):
                i += 1
                tok = sent_tokens[i]
                if tok.i in ner_at:  # do not cross an NER boundary
                    break
                parts.append(tok.lemma_.lower())
            joined = "_".join(parts)
            if joined.strip():
                out.append(joined)
            i += 1
        return out
