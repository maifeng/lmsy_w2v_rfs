"""Expand seed words into per-dimension dictionaries.

Ported to gensim 4 (``model.wv.key_to_index`` instead of
``model.wv.vocab``). Logic mirrors the 2021 replication repo's
``culture/culture_dictionary.py`` step by step so results match.
"""

from __future__ import annotations

import math
from collections import Counter
from operator import itemgetter
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec


def expand_words_dimension_mean(
    model: Word2Vec,
    seeds: dict[str, list[str]],
    n: int = 500,
    restrict_vocab: float | None = None,
    min_similarity: float = 0.0,
    filter_words: set[str] | None = None,
) -> dict[str, set[str]]:
    """Expand each dimension's seed list by mean-vector nearest neighbors.

    For each dimension: average the in-vocab seed vectors, find the
    top-``n`` words by cosine similarity, filter out NER tokens, cross-
    dimension seeds, and any user-supplied stop set, then combine with
    the seeds.

    Args:
        model: Trained Word2Vec model.
        seeds: Mapping of dimension name to seed words.
        n: Top-k expansion per dimension.
        restrict_vocab: Restrict to the top fraction of vocab by
            frequency, or ``None`` to use the full vocab.
        min_similarity: Discard candidates below this cosine.
        filter_words: Additional words to drop from expansion results.

    Returns:
        Mapping of dimension name to expanded word set.
    """
    vocab = model.wv.key_to_index
    all_seeds = {w for ws in seeds.values() for w in ws}

    restrict_k: int | None = None
    if restrict_vocab is not None:
        restrict_k = int(len(vocab) * restrict_vocab)

    out: dict[str, set[str]] = {}
    for dim, words in seeds.items():
        in_vocab = [w for w in words if w in vocab]
        if in_vocab:
            candidates = [
                w
                for w, sim in model.wv.most_similar(
                    in_vocab, topn=n, restrict_vocab=restrict_k
                )
                if sim >= min_similarity and w not in all_seeds
            ]
        else:
            candidates = []
        if filter_words:
            candidates = [w for w in candidates if w not in filter_words]
        candidates = [w for w in candidates if "[ner:" not in w]
        out[dim] = set(candidates) | set(in_vocab)
    return out


def deduplicate_keywords(
    model: Word2Vec,
    expanded: dict[str, set[str]],
    seeds: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Assign cross-loading words to their most similar dimension.

    Args:
        model: Trained Word2Vec model.
        expanded: Output of ``expand_words_dimension_mean``.
        seeds: Original seed lists (in-vocab only entries are used).

    Returns:
        Deduplicated expansion mapping.
    """
    vocab = model.wv.key_to_index
    counter: Counter[str] = Counter()
    for words in expanded.values():
        counter.update(words)

    seeds_in_vocab = {
        dim: [w for w in ws if w in vocab] for dim, ws in seeds.items()
    }
    duplicates = {w for w, c in counter.items() if c > 1}

    deduped = {dim: set(words) - duplicates for dim, words in expanded.items()}
    for word in duplicates:
        best_dim = max(
            deduped.keys(),
            key=lambda d: model.wv.n_similarity(seeds_in_vocab[d], [word])
            if seeds_in_vocab[d] and word in vocab
            else -1.0,
        )
        deduped[best_dim].add(word)
    return deduped


def rank_by_similarity(
    expanded: dict[str, set[str]],
    seeds: dict[str, list[str]],
    model: Word2Vec,
) -> dict[str, list[str]]:
    """Sort each dimension's words by similarity to the seed mean.

    Args:
        expanded: Deduplicated expansion.
        seeds: Original seed lists.
        model: Trained Word2Vec model.

    Returns:
        Mapping of dimension to sorted word list.
    """
    vocab = model.wv.key_to_index
    ranked: dict[str, list[str]] = {}
    for dim, words in expanded.items():
        dim_seeds = [w for w in seeds[dim] if w in vocab]
        scores: dict[str, float] = {}
        for w in words:
            if w in vocab:
                scores[w] = float(model.wv.n_similarity(dim_seeds, [w]))
        ranked[dim] = [w for w, _ in sorted(scores.items(), key=itemgetter(1), reverse=True)]
    return ranked


def write_dict_csv(culture_dict: dict[str, list[str]], path: Path | str) -> Path:
    """Write an expanded dictionary to CSV (one column per dimension).

    Args:
        culture_dict: Mapping of dimension to word list.
        path: Destination CSV.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(culture_dict, orient="index").transpose().to_csv(
        path, index=False
    )
    return path


def read_dict_csv(path: Path | str) -> tuple[dict[str, list[str]], set[str]]:
    """Read an expanded dictionary CSV.

    Args:
        path: CSV produced by ``write_dict_csv``.

    Returns:
        ``(dimension_to_words, all_words)``.
    """
    df = pd.read_csv(path, index_col=None)
    culture = {k: [x for x in v if isinstance(x, str)] for k, v in df.to_dict("list").items()}
    all_words: set[str] = set()
    for v in culture.values():
        all_words |= set(v)
    return culture, all_words


def similarity_weights(culture_dict: dict[str, list[str]]) -> dict[str, float]:
    """Compute the 1 / ln(2 + rank) word weights used for SIMWEIGHT scoring.

    Args:
        culture_dict: Mapping of dimension to rank-sorted word list.

    Returns:
        Mapping of word to weight.
    """
    weights: dict[str, float] = {}
    for words in culture_dict.values():
        for rank, w in enumerate(words):
            weights[w] = 1.0 / math.log(2 + rank)
    return weights
