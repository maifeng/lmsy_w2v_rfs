"""Regex-based cleaner for CoreNLP-annotated text, plus a fallback cleaner.

The CoreNLP preprocessor emits tokens shaped like ``lemma[pos:TAG]``
with optional ``[NER:TYPE]`` prefixes and ``_``-joined multi-word
expressions. ``clean_corenlp_line`` removes the brackets while keeping
the lemma and the MWE underscores.

``clean_plain_line`` is the no-CoreNLP fallback: lowercase, drop
punctuation-only tokens, drop 1-letter words, drop stopwords. Useful
when the user only wants the gensim Phrases + Word2Vec half of the
pipeline.
"""

from __future__ import annotations

import re

_NER_TAG_RE = re.compile(r"\[NER:\w+\]")
_POS_TAG_RE = re.compile(r"\[pos:.*?\]")
_BRACKET_TOKENS = {"-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"}


def clean_corenlp_line(line: str, stopwords: set[str]) -> str:
    """Strip POS and NER tags, drop punctuation, numerics, and stopwords.

    Args:
        line: A line produced by the CoreNLP preprocessor.
        stopwords: Lowercased stopword set.

    Returns:
        A cleaned space-joined token string.
    """
    line = _NER_TAG_RE.sub("", line)
    tokens = line.strip().lower().split()
    tokens = [_POS_TAG_RE.sub("", t) for t in tokens]
    drop = _BRACKET_TOKENS | stopwords
    out = [
        t
        for t in tokens
        if any(c.isalpha() for c in t) and len(t) > 1 and t not in drop
    ]
    return " ".join(out)


def clean_plain_line(line: str, stopwords: set[str]) -> str:
    """Post-preprocessor cleaner.

    Runs AFTER the preprocessor has emitted lemmatized tokens with
    ``[NER:TYPE]`` placeholders and ``_``-joined MWEs. Steps:

    * lowercase
    * preserve ``[ner:*]`` placeholders verbatim
    * preserve ``_``-joined tokens verbatim (trimmed)
    * strip leading and trailing punctuation from plain tokens
    * drop plain tokens that are stopwords, 1-letter, or have no alpha

    Args:
        line: A space-joined preprocessor line.
        stopwords: Lowercased stopword set.

    Returns:
        A cleaned space-joined token string.
    """
    tokens = line.lower().split()
    out: list[str] = []
    for t in tokens:
        if t.startswith("[ner:") and t.endswith("]"):
            out.append(t)
            continue
        # Trim outer punctuation; keep internal hyphens / apostrophes / underscores.
        t = t.strip(".,;:!?()[]{}\"'`")
        if not t or len(t) < 2:
            continue
        if not any(c.isalpha() for c in t):
            continue
        # For underscore-joined multi-word tokens, do NOT test against the
        # stopword list (a MWE like "because_of" is legitimate even though
        # both parts are stopwords).
        if "_" not in t and t in stopwords:
            continue
        out.append(t)
    return " ".join(out)
