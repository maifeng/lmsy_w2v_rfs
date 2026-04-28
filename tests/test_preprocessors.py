"""Offline tests for the five preprocessor backends.

The ``none`` and ``static`` backends are always tested (no external models).
The ``stanza``, ``corenlp``, and ``spacy`` backends are tested only if the
corresponding import succeeds AND the required model / jar / Java is in
place. Otherwise each test skips with a clear message.
"""

from __future__ import annotations

import importlib.util
import os
import shutil

import pytest

from lmsy_w2v_rfs import Config
from lmsy_w2v_rfs.preprocessors import build_preprocessor
from lmsy_w2v_rfs.preprocessors.base import apply_mwe_list, load_mwe_list


SAMPLE = (
    "Apple CEO Tim Cook said revenues grew in the third quarter of 2024. "
    "Our customer commitment is strong, with respect to long term value creation."
)

# Minimal stub seeds for tests that only construct Config, not run the pipeline.
_SEEDS = {"demo": ["customer", "commitment"]}


def test_no_op_preprocessor() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="none", mwe_list=None)
    pp = build_preprocessor(cfg)
    sents = pp.process(SAMPLE)
    assert len(sents) >= 1
    flat = " ".join(tok for sent in sents for tok in sent)
    assert "apple" in flat
    assert "customer" in flat
    # no NER mask, no MWE
    assert "[NER:" not in flat
    assert "customer_commitment" not in flat


def test_static_preprocessor_joins_curated_mwes() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="static", mwe_list="finance")
    pp = build_preprocessor(cfg)
    sents = pp.process(SAMPLE)
    flat = " ".join(tok for sent in sents for tok in sent)
    assert "customer_commitment" in flat
    assert "with_respect_to" in flat
    assert "long_term" in flat


def test_load_mwe_list_packaged() -> None:
    mwes = load_mwe_list("finance")
    assert len(mwes) >= 100
    # Some known entries
    assert ("customer", "commitment") in mwes
    assert ("with", "respect", "to") in mwes
    assert ("balance", "sheet") in mwes


def test_apply_mwe_list_respects_ner_boundaries() -> None:
    mwes = [("with", "respect", "to")]
    input_sents = [["we", "act", "[NER:ORG]", "with", "respect", "to", "rules"]]
    out = apply_mwe_list(input_sents, mwes)
    # [NER:ORG] should not be inside the join
    assert "[NER:ORG]" in out[0]
    assert "with_respect_to" in out[0]


def test_apply_mwe_list_with_none_returns_input() -> None:
    sents = [["a", "b", "c"]]
    assert apply_mwe_list(sents, None) == [["a", "b", "c"]]


def test_unknown_preprocessor_name_raises() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="none", mwe_list=None).with_(preprocessor="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown preprocessor"):
        build_preprocessor(cfg)


# ---------------- optional backend tests ------------------------------------

_HAS_STANZA = importlib.util.find_spec("stanza") is not None
_HAS_SPACY = importlib.util.find_spec("spacy") is not None


def _has_spacy_model(name: str) -> bool:
    if not _HAS_SPACY:
        return False
    import spacy

    try:
        spacy.load(name)
        return True
    except OSError:
        return False


def _has_java() -> bool:
    return shutil.which("java") is not None


def _has_corenlp_jars() -> bool:
    import pathlib

    home = os.environ.get("CORENLP_HOME") or str(
        pathlib.Path.home() / ".cache" / "lmsy_w2v_rfs" / "corenlp"
    )
    return len(list(pathlib.Path(home).glob("*stanford-corenlp-*.jar"))) > 0


@pytest.mark.skipif(not _HAS_STANZA, reason="stanza not installed")
def test_stanza_preprocessor_smoke() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="stanza", mwe_list=None)
    pp = build_preprocessor(cfg)
    sents = pp.process(SAMPLE)
    flat = " ".join(tok for sent in sents for tok in sent)
    # stanza should mask "Apple" and "Tim Cook" as NER placeholders
    assert "[NER:" in flat


@pytest.mark.skipif(not _has_spacy_model("en_core_web_sm"), reason="spaCy model en_core_web_sm not installed")
def test_spacy_preprocessor_smoke() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="spacy", spacy_model="en_core_web_sm", mwe_list=None)
    pp = build_preprocessor(cfg)
    sents = pp.process(SAMPLE)
    flat = " ".join(tok for sent in sents for tok in sent)
    assert "[NER:" in flat


@pytest.mark.corenlp
@pytest.mark.skipif(not (_HAS_STANZA and _has_java() and _has_corenlp_jars()),
                    reason="corenlp extra / Java / jars not available")
def test_corenlp_preprocessor_smoke() -> None:
    cfg = Config(seeds=_SEEDS, preprocessor="corenlp", mwe_list=None, corenlp_memory="3G")
    pp = build_preprocessor(cfg)
    try:
        sents = pp.process(SAMPLE)
    finally:
        close = getattr(pp, "close", None)
        if callable(close):
            close()
    flat = " ".join(tok for sent in sents for tok in sent)
    assert "[NER:" in flat
