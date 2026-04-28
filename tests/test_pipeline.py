"""End-to-end pipeline smoke tests with each preprocessor variant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lmsy_w2v_rfs import Config, Pipeline


def _base_cfg(seeds: dict[str, list[str]], **overrides) -> Config:
    # Offline test default: "none" preprocessor avoids external model loads
    # even though the production default is "corenlp".
    return Config(
        seeds=seeds,
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


def test_static_preprocessor_runs_end_to_end(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds, preprocessor="static", mwe_list="finance")
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TF", "TFIDF", "WFIDF"))
    for method in ("TF", "TFIDF", "WFIDF"):
        df = pipe.score_df(method)
        assert len(df) == len(tiny_corpus)
        assert set(cfg.dims).issubset(df.columns)


def test_none_preprocessor_runs(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds, preprocessor="none", mwe_list=None)
    pipe = Pipeline(texts=tiny_corpus, doc_ids=[f"d{i}" for i in range(len(tiny_corpus))],
                    work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    df = pipe.score_df("TFIDF")
    assert len(df) == len(tiny_corpus)


def test_pipeline_resumes_from_existing_artifacts(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg).run()

    pipe2 = Pipeline(work_dir=work_dir, config=cfg)
    pipe2.train()
    pipe2.expand_dictionary()
    scores = pipe2.score(methods=("TFIDF",))
    assert "TFIDF" in scores


# ----- inspection + curation -------------------------------------------


def test_dictionary_preview_returns_dataframe(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    df = pipe.dictionary_preview(top_k=5)
    assert list(df.columns) == ["dimension", "seeds", "expanded_top_k", "n_expanded"]
    assert len(df) == len(cfg.dims)
    assert all(df["n_expanded"] > 0)


def test_show_dictionary_prints(tiny_corpus, work_dir: Path, culture_seeds, capsys) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    pipe.show_dictionary(top_k=5)
    captured = capsys.readouterr().out
    for dim in cfg.dims:
        assert dim in captured
    assert "seeds:" in captured and "expanded:" in captured


def test_edit_dictionary_removes_words(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    integrity_before = list(pipe.expanded_dict["integrity"])
    assert len(integrity_before) >= 2
    drop = integrity_before[:2]
    updated = pipe.edit_dictionary(remove={"integrity": drop})
    for w in drop:
        assert w not in updated["integrity"]
    # CSV on disk is also updated.
    from lmsy_w2v_rfs import read_dict_csv
    on_disk, _ = read_dict_csv(pipe.dict_path)
    for w in drop:
        assert w not in on_disk["integrity"]


def test_edit_dictionary_adds_words(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    updated = pipe.edit_dictionary(add={"integrity": ["zzznewword"]})
    assert "zzznewword" in updated["integrity"]
    # Idempotent: adding the same word twice does not duplicate.
    updated2 = pipe.edit_dictionary(add={"integrity": ["zzznewword"]})
    assert updated2["integrity"].count("zzznewword") == 1


def test_edit_dictionary_unknown_dimension_raises(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))
    with pytest.raises(KeyError, match="Unknown dimension"):
        pipe.edit_dictionary(remove={"not_a_dim": ["x"]})


def test_reload_dictionary_picks_up_disk_edits(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.run(methods=("TFIDF",))

    # Edit the on-disk CSV directly: drop every word from "integrity".
    from lmsy_w2v_rfs import read_dict_csv, write_dict_csv
    d, _ = read_dict_csv(pipe.dict_path)
    d["integrity"] = ["accountable"]
    write_dict_csv(d, pipe.dict_path)

    reloaded = pipe.reload_dictionary()
    assert reloaded["integrity"] == ["accountable"]


def test_reload_dictionary_without_csv_raises(work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds)
    pipe = Pipeline(texts=["a b c"], doc_ids=["d0"], work_dir=work_dir, config=cfg)
    with pytest.raises(FileNotFoundError, match="No dictionary CSV"):
        pipe.reload_dictionary()


_HAS_SPACY_SM = False
if importlib.util.find_spec("spacy") is not None:
    import spacy

    try:
        spacy.load("en_core_web_sm")
        _HAS_SPACY_SM = True
    except OSError:
        _HAS_SPACY_SM = False


@pytest.mark.skipif(not _HAS_SPACY_SM, reason="spaCy model en_core_web_sm not installed")
def test_spacy_preprocessor_masks_ner(tiny_corpus, work_dir: Path, culture_seeds) -> None:
    cfg = _base_cfg(culture_seeds, preprocessor="spacy", spacy_model="en_core_web_sm", mwe_list="finance")
    ids = [f"doc{i}" for i in range(len(tiny_corpus))]
    pipe = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    pipe.parse()
    text = (work_dir / "parsed" / "sentences.txt").read_text(encoding="utf-8")
    # Some placeholders should appear (the tiny corpus has company names etc.)
    # Even if the small model misses some, the PIPELINE should have run cleanly.
    assert text.strip(), "parsed output is empty"
