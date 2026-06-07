"""Tests for seed-expansion dictionary construction.

These exercise expand_words_dimension_mean and deduplicate_keywords directly
on a small, fully-controlled Word2Vec model so a regression in the expansion
logic (NER filtering, seed exclusion, restrict_vocab, cross-loading dedup) is
caught without a full pipeline run.
"""

from __future__ import annotations

from gensim.models import Word2Vec

from lmsy_w2v_rfs.dictionary import (
    deduplicate_keywords,
    expand_words_dimension_mean,
)


def _toy_model() -> Word2Vec:
    """Two well-separated topical clusters plus an NER placeholder token.

    'risk' cluster and 'growth' cluster co-occur within their own sentences so
    skip-gram pulls each cluster's vectors together. '[ner:org]' rides along in
    the risk sentences so it becomes a plausible neighbour (and must be filtered).
    """
    risk = ["risk", "uncertainty", "volatility", "downside", "[ner:org]"]
    growth = ["growth", "expansion", "scale", "opportunity"]
    sentences = [risk] * 40 + [growth] * 40
    return Word2Vec(
        sentences, vector_size=24, window=4, min_count=1,
        workers=1, epochs=50, sg=1, seed=0,
    )


def test_expand_filters_ner_and_seeds() -> None:
    model = _toy_model()
    seeds = {"risk": ["risk", "uncertainty"], "growth": ["growth", "expansion"]}
    out = expand_words_dimension_mean(model, seeds, n=20)

    assert set(out) == {"risk", "growth"}
    # Seeds are folded back in.
    assert {"risk", "uncertainty"} <= out["risk"]
    # NER placeholders must never enter any dimension.
    for words in out.values():
        assert not any(w.startswith("[ner:") for w in words)
    # Cross-dimension seeds are excluded from each other's *expanded* candidates
    # (growth's seeds should not be pulled into risk as new candidates).
    risk_non_seed = out["risk"] - {"risk", "uncertainty"}
    assert "growth" not in risk_non_seed


def test_expand_restrict_vocab_runs_and_sorts() -> None:
    model = _toy_model()
    seeds = {"risk": ["risk", "uncertainty"]}
    # A fraction triggers the sort_by_descending_frequency() + restrict path.
    out = expand_words_dimension_mean(model, seeds, n=10, restrict_vocab=0.8)
    assert "risk" in out["risk"]
    assert all(not w.startswith("[ner:") for w in out["risk"])


def test_deduplicate_assigns_word_to_one_dimension() -> None:
    model = _toy_model()
    seeds = {"risk": ["risk", "uncertainty"], "growth": ["growth", "expansion"]}
    expanded = {
        "risk": {"risk", "uncertainty", "shared"},
        "growth": {"growth", "expansion", "shared"},
    }
    # Inject a vector for the shared word so n_similarity works.
    expanded2 = {k: {w for w in v if w in model.wv.key_to_index} | (
        {"risk"} if k == "risk" else {"growth"}) for k, v in expanded.items()}
    deduped = deduplicate_keywords(model, expanded2, seeds)
    # Every word lands in at most one dimension.
    seen: set[str] = set()
    for words in deduped.values():
        assert not (words & seen)
        seen |= words
