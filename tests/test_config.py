"""Tests for Config, stopwords, and the named example loader."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lmsy_w2v_rfs import STOPWORDS_SRAF, Config, load_example_seeds


_SEEDS = {"risk": ["risk", "uncertainty"], "growth": ["growth", "scale"]}


# ----- example seeds (opt-in reproducer) --------------------------------


def test_load_example_seeds_culture_2021() -> None:
    seeds = load_example_seeds("culture_2021")
    assert set(seeds.keys()) == {
        "integrity", "teamwork", "innovation", "respect", "quality",
    }
    assert sum(len(v) for v in seeds.values()) == 47


def test_load_example_seeds_returns_fresh_copy() -> None:
    s1 = load_example_seeds("culture_2021")
    s2 = load_example_seeds("culture_2021")
    s1["integrity"].append("foo")
    assert "foo" not in s2["integrity"]


def test_load_example_seeds_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown example"):
        load_example_seeds("nonexistent")


# ----- stopwords --------------------------------------------------------


def test_stopwords_size_and_lowercase() -> None:
    assert 100 <= len(STOPWORDS_SRAF) <= 130
    assert all(w == w.lower() for w in STOPWORDS_SRAF)


# ----- Config requires seeds -------------------------------------------


def test_config_requires_seeds() -> None:
    with pytest.raises(TypeError):
        Config()  # type: ignore[call-arg]


def test_config_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Config(seeds={})


def test_config_rejects_empty_dimension() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Config(seeds={"risk": []})


def test_config_defaults_enforce_two_phase_pipeline() -> None:
    cfg = Config(seeds=_SEEDS)
    # Phase 1a: parser-based MWE + NER. Default = corenlp (paper-faithful).
    assert cfg.preprocessor == "corenlp"
    # Phase 1b: optional curated list. Default off.
    assert cfg.mwe_list is None
    # Phase 2: statistical MWE learning. On by default.
    assert cfg.use_gensim_phrases is True
    assert cfg.phrase_passes == 2
    assert cfg.n_cores == 4


def test_config_dims_property_reflects_seeds() -> None:
    cfg = Config(seeds=_SEEDS)
    assert cfg.dims == ["risk", "growth"]


def test_config_with_overrides() -> None:
    cfg = Config(seeds=_SEEDS).with_(w2v_dim=50, w2v_epochs=3)
    assert cfg.w2v_dim == 50
    assert cfg.w2v_epochs == 3


def test_config_is_frozen() -> None:
    cfg = Config(seeds=_SEEDS)
    try:
        cfg.w2v_dim = 100  # type: ignore[misc]
    except Exception as e:
        assert "FrozenInstanceError" in type(e).__name__ or isinstance(e, AttributeError)
    else:
        raise AssertionError("Config should be frozen.")


def test_replace_produces_new_instance() -> None:
    cfg = Config(seeds=_SEEDS)
    cfg2 = replace(cfg, w2v_dim=123)
    assert cfg.w2v_dim == 300 and cfg2.w2v_dim == 123
