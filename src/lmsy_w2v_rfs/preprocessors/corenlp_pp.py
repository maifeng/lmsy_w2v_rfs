"""Stanford CoreNLP preprocessor (paper-exact, requires Java).

Starts one CoreNLP JVM via ``stanza.server.CoreNLPClient`` and holds it
open for the lifetime of the preprocessor instance. The ``ner`` and
``entitymentions`` annotators are both enabled so ``sentence.mentions`` is
populated (the 2021 paper uses this, and the bakeoff showed that omitting
``entitymentions`` produces almost no NER output).

Use this for paper-exact reproduction. Modern CoreNLP emits UD v2 labels,
which differ slightly from the 2021 paper's UD v1 output (the ``mwe``
relation was renamed ``fixed`` in UD v2, among other changes). See
``docs/explanation/mwe-comparison.md`` for the details.
"""

from __future__ import annotations

import logging
import os
import pathlib
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from ..config import default_cache_dir

if TYPE_CHECKING:
    from ..config import Config


log = logging.getLogger(__name__)


_MWE_DEPS = {"fixed", "flat", "compound"}


class CoreNLPPreprocessor:
    """CoreNLP-server-based preprocessor.

    The client is started when the first document is processed and kept
    warm until ``close()`` is called or the instance is garbage collected.
    Use as a context manager (``with CoreNLPPreprocessor(cfg) as pp:``) to
    guarantee clean shutdown.
    """

    def __init__(self, config: "Config") -> None:
        """Stand up the CoreNLP client.

        Args:
            config: Pipeline config.

        Raises:
            ImportError: If the corenlp extra is not installed.
            RuntimeError: If Java is not on PATH.
        """
        import stanza

        home = default_cache_dir() / "corenlp"
        home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CORENLP_HOME", str(home))

        jars = list(pathlib.Path(home).glob("*stanford-corenlp-*.jar"))
        if not jars:
            log.info("Installing CoreNLP at %s", home)
            stanza.install_corenlp(dir=str(home))

        self._cfg = config
        self._stanza = stanza
        self._client = stanza.server.CoreNLPClient(
            annotators=[
                "tokenize", "ssplit", "pos", "lemma", "ner",
                "entitymentions", "depparse",
            ],
            memory=config.corenlp_memory,
            threads=config.n_cores,
            timeout=config.corenlp_timeout_ms,
            be_quiet=True,
            properties={"ner.applyFineGrained": "false"},
        )
        self._client.start()
        log.info("CoreNLPPreprocessor ready")

    def __enter__(self) -> "CoreNLPPreprocessor":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the CoreNLP server."""
        try:
            self._client.stop()
        except Exception as e:  # pragma: no cover
            log.warning("Error stopping CoreNLP: %s", e)

    def process(self, text: str) -> list[list[str]]:
        """Parse one document.

        Args:
            text: Raw document.

        Returns:
            List of sentences, each a list of tokens.
        """
        ann = self._client.annotate(text)
        out: list[list[str]] = []
        for s in ann.sentence:
            out.append(self._sentence_tokens(s))
        return out

    def process_documents(
        self, texts: Iterable[str]
    ) -> Iterator[list[list[str]]]:
        """Fan out annotate requests across the JVM thread pool.

        The CoreNLP server has a thread pool of size ``config.n_cores``.
        Sending requests serially leaves those threads idle. This override
        submits ``n_cores`` requests in flight via a Python
        ``ThreadPoolExecutor``; the JVM processes them concurrently.

        Args:
            texts: Iterable of raw documents.

        Yields:
            Preprocessed documents in input order.
        """
        import concurrent.futures as cf

        texts = list(texts)
        n_workers = max(1, self._cfg.n_cores)
        if n_workers == 1:
            yield from (self.process(t) for t in texts)
            return
        # Map preserves input order.
        with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            for ann in ex.map(self._client.annotate, texts):
                out: list[list[str]] = []
                for s in ann.sentence:
                    out.append(self._sentence_tokens(s))
                yield out

    def _sentence_tokens(self, sent) -> list[str]:
        if not sent.token:
            return []
        base = sent.token[0].tokenBeginIndex

        # NER by entitymentions: doc_token_index -> (type, is_start)
        ner_at: dict[int, tuple[str, bool]] = {}
        for m in sent.mentions:
            t_start = m.tokenStartInSentenceInclusive + base
            t_end = m.tokenEndInSentenceExclusive + base
            for j in range(t_start, t_end):
                ner_at[j] = (m.entityType, j == t_start)

        # MWE pairs on UD v2 deps. Only adjacent pairs.
        mwe_after: set[int] = set()
        for e in sent.enhancedPlusPlusDependencies.edge:
            if e.dep in _MWE_DEPS or e.dep.startswith("compound:"):
                lo = min(e.source, e.target) - 1 + base
                hi = max(e.source, e.target) - 1 + base
                if hi - lo == 1:
                    mwe_after.add(lo)

        tokens = list(sent.token)
        out: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            idx = t.tokenBeginIndex
            if idx in ner_at:
                etype, is_start = ner_at[idx]
                if is_start:
                    out.append(f"[NER:{etype}]")
                i += 1
                continue
            parts = [(t.lemma or t.word).lower()]
            while idx in mwe_after and i + 1 < len(tokens):
                i += 1
                t = tokens[i]
                idx = t.tokenBeginIndex
                if idx in ner_at:
                    break
                parts.append((t.lemma or t.word).lower())
            joined = "_".join(parts)
            if joined.strip():
                out.append(joined)
            i += 1
        return out
