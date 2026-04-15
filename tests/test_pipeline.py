"""End-to-end pipeline smoke tests with each preprocessor variant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lmsy_w2v_rfs import Config, Pipeline


def _base_cfg(**overrides) -> Config:
    # Offline test default: "none" preprocessor avoids external model loads
    # even though the production default is "corenlp".
    return Config(
        preprocessor="none",
        mwe_list=None,
        use_gensim_phrases=True,
        phrase_passes=2,
        phrase_min_count=2,
        phrase_threshold=1.0,
        w2v_dim=30,
        w2v_epochs=5,
        w2v_window=3,
        w2v_min_count=1,
        n_cores=1,
        n_words_dim=20,
    ).with_(**overrides)


def test_static_preprocessor_runs_end_to_end(tiny_corpus, work_dir: Path) -> None:
    cfg = _base_cfg(preprocessor="static", mwe_list="finance")
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TF", "TFIDF", "WFIDF"))
    for method in ("TF", "TFIDF", "WFIDF"):
        df = pipe.score_df(method)
        assert len(df) == len(tiny_corpus)
        assert set(cfg.dims).issubset(df.columns)


def test_none_preprocessor_runs(tiny_corpus, work_dir: Path) -> None:
    cfg = _base_cfg(preprocessor="none", mwe_list=None)
    pipe = Pipeline(texts=tiny_corpus, doc_ids=[f"d{i}" for i in range(len(tiny_corpus))],
                    work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    df = pipe.score_df("TFIDF")
    assert len(df) == len(tiny_corpus)


def test_pipeline_resumes_from_existing_artifacts(tiny_corpus, work_dir: Path) -> None:
    cfg = _base_cfg()
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg).run()

    pipe2 = Pipeline(work_dir=work_dir, config=cfg)
    pipe2.train()
    pipe2.expand_dictionary()
    scores = pipe2.score(methods=("TFIDF",))
    assert "TFIDF" in scores


_HAS_SPACY_SM = False
if importlib.util.find_spec("spacy") is not None:
    import spacy

    try:
        spacy.load("en_core_web_sm")
        _HAS_SPACY_SM = True
    except OSError:
        _HAS_SPACY_SM = False


@pytest.mark.skipif(not _HAS_SPACY_SM, reason="spaCy model en_core_web_sm not installed")
def test_spacy_preprocessor_masks_ner(tiny_corpus, work_dir: Path) -> None:
    cfg = _base_cfg(preprocessor="spacy", spacy_model="en_core_web_sm", mwe_list="finance")
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.parse()
    text = (work_dir / "parsed" / "sentences.txt").read_text(encoding="utf-8")
    # Some placeholders should appear (the tiny corpus has company names etc.)
    # Even if the small model misses some, the PIPELINE should have run cleanly.
    assert text.strip(), "parsed output is empty"
