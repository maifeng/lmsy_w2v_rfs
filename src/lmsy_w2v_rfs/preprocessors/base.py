"""Preprocessor protocol and the optional static-MWE post-pass."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Preprocessor(Protocol):
    """Phase-1 preprocessor contract.

    Implementations turn a raw document into a list of preprocessed sentences.
    Each sentence is a list of tokens. Tokens may contain underscores for
    multi-word expressions and the literal string ``[NER:TYPE]`` as an entity
    placeholder. The pipeline's cleaner drops punctuation-only and stopword
    tokens downstream; the preprocessor does not have to.

    Implementations supply ``process`` (one doc) and optionally override
    ``process_documents`` (batch) for concurrent processing. The default
    batch implementation just loops over ``process``.
    """

    def process(self, text: str) -> list[list[str]]:
        """Parse one document.

        Args:
            text: A raw document string. May contain newlines.

        Returns:
            A list of sentences, each a list of tokens.
        """

    def process_documents(
        self, texts: Iterable[str]
    ) -> Iterator[list[list[str]]]:
        """Parse a stream of documents, possibly concurrently.

        The default implementation loops serially. Backends that benefit
        from concurrency (CoreNLP with a JVM thread pool, spaCy with
        ``nlp.pipe(n_process=N)``) override this to unlock real throughput.

        Args:
            texts: Iterable of raw document strings.

        Yields:
            One preprocessed document at a time, in input order.
        """
        for t in texts:
            yield self.process(t)


def load_mwe_list(source: str | Path) -> list[tuple[str, ...]]:
    """Load a curated MWE list from the packaged ``data/`` or a file path.

    Args:
        source: Either ``"finance"`` for the packaged finance list, or a
            filesystem path to a newline-delimited list.

    Returns:
        List of tuples, each a token sequence for NLTK ``MWETokenizer``.

    Raises:
        FileNotFoundError: If the source cannot be resolved.
    """
    if isinstance(source, str) and source in {"finance"}:
        from importlib import resources

        text = (
            resources.files("lmsy_w2v_rfs.data")
            .joinpath(f"mwes_{source}.txt")
            .read_text(encoding="utf-8")
        )
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"MWE list not found: {source}")
        text = path.read_text(encoding="utf-8")
    out: list[tuple[str, ...]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = tuple(w.lower() for w in line.replace("_", " ").split())
        if len(parts) >= 2:
            out.append(parts)
    return out


def apply_mwe_list(
    sentences: Iterable[list[str]],
    mwe_list: list[tuple[str, ...]] | None,
) -> list[list[str]]:
    """Apply NLTK ``MWETokenizer`` to each sentence as a post-pass.

    Designed to run AFTER the main preprocessor, so MWE patterns the parser
    missed (for example, ``customer_commitment``) still get joined.

    Args:
        sentences: Sentences of tokens.
        mwe_list: Loaded MWE tuples. If ``None`` or empty, return input
            unchanged.

    Returns:
        Sentences with matching MWEs joined by ``_``.
    """
    if not mwe_list:
        return [list(s) for s in sentences]
    from nltk.tokenize import MWETokenizer

    tokenizer = MWETokenizer(mwe_list, separator="_")
    out: list[list[str]] = []
    for s in sentences:
        # Split NER placeholders out so MWETokenizer does not match across them.
        # A contiguous run of non-placeholder tokens is tokenized as a group.
        new_sent: list[str] = []
        buf: list[str] = []
        for t in s:
            if t.startswith("[NER:"):
                if buf:
                    new_sent.extend(tokenizer.tokenize(buf))
                    buf = []
                new_sent.append(t)
            else:
                buf.append(t)
        if buf:
            new_sent.extend(tokenizer.tokenize(buf))
        out.append(new_sent)
    return out
