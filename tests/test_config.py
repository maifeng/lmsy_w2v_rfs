"""Tests for seeds, stopwords, and config."""

from __future__ import annotations

from dataclasses import replace

from lmsy_w2v_rfs import CULTURE_DIMS, CULTURE_SEEDS, STOPWORDS_SRAF, Config


def test_five_dimensions_loaded() -> None:
    assert set(CULTURE_SEEDS.keys()) == set(CULTURE_DIMS)
    assert CULTURE_DIMS == ["integrity", "teamwork", "innovation", "respect", "quality"]


def test_total_seed_count() -> None:
    n = sum(len(v) for v in CULTURE_SEEDS.values())
    assert n == 47, "Paper reports 47 seed words across the five dimensions."


def test_stopwords_size_and_lowercase() -> None:
    assert 100 <= len(STOPWORDS_SRAF) <= 130
    assert all(w == w.lower() for w in STOPWORDS_SRAF)


def test_config_defaults_enforce_two_phase_pipeline() -> None:
    cfg = Config()
    # Phase 1a: parser-based MWE + NER. Default = corenlp (paper-faithful,
    # best syntactic MWE coverage, scales near-linearly on JVM threads).
    # Python-only users can switch to "spacy", "stanza", "static", or "none".
    assert cfg.preprocessor == "corenlp"
    # Phase 1b: optional curated list. Default off; it is opt-in only.
    assert cfg.mwe_list is None
    # Phase 2: statistical MWE learning. On by default.
    assert cfg.use_gensim_phrases is True
    assert cfg.phrase_passes == 2
    # Default n_cores = 4 (safe on an 8-core laptop).
    assert cfg.n_cores == 4


def test_config_with_overrides() -> None:
    cfg = Config().with_(w2v_dim=50, w2v_epochs=3)
    assert cfg.w2v_dim == 50
    assert cfg.w2v_epochs == 3


def test_config_is_frozen() -> None:
    cfg = Config()
    try:
        cfg.w2v_dim = 100  # type: ignore[misc]
    except Exception as e:  # dataclasses.FrozenInstanceError subclasses AttributeError
        assert "FrozenInstanceError" in type(e).__name__ or isinstance(e, AttributeError)
    else:
        raise AssertionError("Config should be frozen.")


def test_replace_produces_new_instance() -> None:
    cfg = Config()
    cfg2 = replace(cfg, w2v_dim=123)
    assert cfg.w2v_dim == 300 and cfg2.w2v_dim == 123
