"""Regression checks for scoring IDs, dictionary persistence, and phrase helpers.

Inputs are controlled corpora, real trained embeddings, and temporary files.
Outputs are verified research observations and unchanged caller-owned artifacts.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from gensim.models import Word2Vec

from lmsy_w2v_rfs import Config, Pipeline
from lmsy_w2v_rfs.dictionary import (
    deduplicate_keywords,
    expand_words_dimension_mean,
    rank_by_similarity,
    read_dict_csv,
    write_dict_csv,
)
from lmsy_w2v_rfs.phrases import apply_phrase_model, train_phrase_model
from lmsy_w2v_rfs.scoring import (
    ScoringMethod,
    iter_doc_level_corpus,
    score_document,
    word_contributions,
)


def _controlled_model() -> Word2Vec:
    """Create an unsorted vocabulary with known counts and vectors.

    Returns:
        Model whose rare token is the closest seed neighbor.
    """
    model = Word2Vec(vector_size=2, min_count=1, workers=1, sorted_vocab=0)
    model.build_vocab([["seed", "rare"] + ["common"] * 20 + ["second"] * 10])
    for word, vector in {
        "seed": [100, 0], "rare": [2, 0],
        "common": [1, 1], "second": [1, 2],
    }.items():
        model.wv[word] = np.asarray(vector, dtype=np.float32)
    return model


def test_underscore_ids_score_and_aggregate_end_to_end(tmp_path: Path) -> None:
    """Keep quarterly transcript observations and their firm-year joins.

    Args:
        tmp_path: Isolated pipeline output directory.
    """
    cfg = Config(seeds={"topic": ["alpha"]}, stopwords=set(),
                 use_gensim_phrases=False, n_cores=1, w2v_dim=8,
                 w2v_epochs=2, w2v_min_count=1, n_words_dim=3)
    ids = ["AAPL_2021Q1", "AAPL_2021Q2", "MSFT_2021Q1"]
    pipe = Pipeline(texts=["alpha beta", "alpha gamma gamma", "beta delta"],
                    doc_ids=ids, work_dir=tmp_path, config=cfg)
    pipe.run(methods=("TF", "TFIDF"))
    scores = pipe.score_df("TFIDF")
    assert scores.Doc_ID.tolist() == ids
    assert scores.document_length.tolist() == [2, 3, 2]
    mapping = pd.DataFrame({"document_id": ids, "firm_id": ["AAPL", "AAPL", "MSFT"],
                            "time": [2021] * 3})
    firms = pipe.firm_year(mapping, method="TFIDF")
    assert firms.firm_id.tolist() == ["AAPL", "MSFT"]


@pytest.mark.parametrize("longer_file", ["corpus", "ids"])
def test_sentence_file_alignment_is_strict(tmp_path: Path, longer_file: str) -> None:
    """Reject either direction of malformed paired sentence files.

    Args:
        tmp_path: Isolated file directory.
        longer_file: Which file contains an extra line.
    """
    corpus, ids = tmp_path / "sentences.txt", tmp_path / "ids.txt"
    corpus.write_text("alpha\n" + ("beta\n" if longer_file == "corpus" else ""))
    ids.write_text("d_0\n" + ("d_1\n" if longer_file == "ids" else ""))
    with pytest.raises(ValueError, match="zip"):
        list(iter_doc_level_corpus(corpus, ids))


@pytest.mark.parametrize("method", ["TFIDF+SIMWEIGHT", "WFIDF+SIMWEIGHT"])
def test_simweight_contributions_skip_nondictionary_tokens(method: ScoringMethod) -> None:
    """Match weighted document scores after per-document length normalization.

    Args:
        method: Similarity-weighted scoring method.
    """
    dictionary = {"topic": {"seed"}}
    kwargs = {"df_dict": {"seed": 1, "ordinary": 1}, "n_docs": 2,
              "word_weights": {"seed": 0.5}}
    scores, length = score_document("seed seed ordinary", dictionary, method=method, **kwargs)
    out = word_contributions([("d", "seed seed ordinary")], dictionary,
                             method=method, show_progress=False, **kwargs)
    assert out.word.tolist() == ["seed"]
    assert out.contribution.iloc[0] == pytest.approx(scores[0] / length)


def test_simweight_still_rejects_missing_dictionary_weight() -> None:
    """Keep errors visible when an actual dictionary token has no weight."""
    with pytest.raises(KeyError, match="seed"):
        word_contributions([("d", "seed ordinary")], {"topic": {"seed"}},
                           method="TFIDF+SIMWEIGHT", df_dict={"seed": 1},
                           n_docs=2, word_weights={}, show_progress=False)


@pytest.mark.parametrize("restriction", [None, 0.5, 0.01])
def test_repeat_expansion_preserves_real_model(tmp_path: Path, restriction: float | None) -> None:
    """Repeat forced expansion without changing model indices or cached norms.

    Args:
        tmp_path: Isolated pipeline output directory.
        restriction: Candidate frequency fraction, including no restriction.
    """
    cfg = Config(seeds={"risk": ["risk"], "growth": ["growth"]},
                 w2v_dim=12, w2v_min_count=1, w2v_epochs=8, n_cores=1,
                 n_words_dim=5, dict_restrict_vocab=restriction)
    corpus = ([["risk", "uncertainty", "volatility"]] * 30
              + [["growth", "expansion", "scale"]] * 20)
    model = Word2Vec(corpus, vector_size=12, min_count=1, workers=1,
                     epochs=8, seed=42, sorted_vocab=0)
    model.wv.fill_norms()
    before_keys = list(model.wv.index_to_key)
    before_indices = dict(model.wv.key_to_index)
    before_vectors = model.wv.vectors.copy()
    before_norms = model.wv.norms.copy()
    before_output = model.syn1neg.copy()
    before_counts = model.wv.expandos["count"].copy()
    pipe = Pipeline(config=cfg, work_dir=tmp_path)
    pipe._w2v_model = model
    first = pipe.expand_dictionary(force=True)
    assert first == pipe.expand_dictionary(force=True)
    assert first == pipe.expand_dictionary(force=True)
    assert model.wv.index_to_key == before_keys
    assert model.wv.key_to_index == before_indices
    np.testing.assert_array_equal(model.wv.vectors, before_vectors)
    np.testing.assert_array_equal(model.wv.norms, before_norms)
    np.testing.assert_array_equal(model.syn1neg, before_output)
    np.testing.assert_array_equal(model.wv.expandos["count"], before_counts)
    np.testing.assert_allclose(model.wv.norms, np.linalg.norm(model.wv.vectors, axis=1))


def test_expansion_does_not_populate_callers_empty_norm_cache() -> None:
    """Treat the supplied embedding model as caller-owned state."""
    model = _controlled_model()
    assert model.wv.norms is None
    expand_words_dimension_mean(model, {"topic": ["seed"]}, n=3)
    assert model.wv.norms is None


def test_restrict_vocab_uses_frequency_in_unsorted_model() -> None:
    """Limit search by token frequency even when indices are insertion-ordered."""
    model = _controlled_model()
    assert model.wv.index_to_key[:2] == ["seed", "rare"]
    out = expand_words_dimension_mean(model, {"topic": ["seed"]}, n=10,
                                      restrict_vocab=0.5, min_similarity=-1)
    assert out == {"topic": {"seed", "common", "second"}}
    tiny = expand_words_dimension_mean(model, {"topic": ["seed"]}, n=10,
                                       restrict_vocab=0.01, min_similarity=-1)
    assert tiny == {"topic": {"seed", "common"}}


def test_expansion_matches_gensim_without_restriction() -> None:
    """Retain the historical gensim normalized-seed neighbor formula."""
    model = _controlled_model()
    expected = {w for w, _ in model.wv.most_similar(["seed", "common"], topn=1)}
    actual = expand_words_dimension_mean(model, {"topic": ["seed", "common"]}, n=1)
    assert actual["topic"] == expected | {"seed", "common"}


def test_historical_deduplication_and_ranking_remain_unchanged() -> None:
    """Keep all-dimension dedup eligibility and raw-vector seed-mean ranking."""
    model = Word2Vec(vector_size=2, min_count=1, workers=1)
    model.build_vocab([["a", "b", "c", "shared", "extra"]])
    for word, vector in {"a": [100, 0], "b": [0, 1], "c": [1, 1],
                         "shared": [1, 1], "extra": [100, 1]}.items():
        model.wv[word] = np.asarray(vector, dtype=np.float32)
    result = deduplicate_keywords(model, {"x": {"a", "shared"}, "y": {"b", "shared"}, "z": {"c"}},
                                  {"x": ["a"], "y": ["b"], "z": ["c"]})
    assert "shared" in result["z"]
    ranked = rank_by_similarity({"x": {"c", "extra"}}, {"x": ["a", "b"]}, model)
    assert ranked["x"] == ["extra", "c"]
    scores, length = score_document("a a ordinary", {"x": {"a"}}, method="TF")
    assert scores == [2.0] and length == 3
    weighted, _ = score_document("a a", {"x": {"a"}}, method="WFIDF", df_dict={"a": 1}, n_docs=2)
    assert weighted == pytest.approx([(1 + math.log(2)) * math.log(2)])


def test_csv_roundtrip_preserves_literals_and_padding(tmp_path: Path) -> None:
    """Preserve numeric and missing-value-like tokens in ragged dimensions.

    Args:
        tmp_path: Isolated dictionary output directory.
    """
    expected = {"numbers": ["2021", "001"], "literal": ["nan", "null", "NA", "N/A"], "empty": []}
    path = write_dict_csv(expected, tmp_path / "dictionary.csv")
    actual, words = read_dict_csv(path)
    assert actual == expected
    assert words == {"2021", "001", "nan", "null", "NA", "N/A"}


def test_failed_dictionary_write_preserves_original_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the published CSV unchanged if writing the replacement fails.

    Args:
        tmp_path: Isolated dictionary output directory.
        monkeypatch: Scoped injection of a partial CSV write failure.
    """
    path = write_dict_csv({"topic": ["original"]}, tmp_path / "dictionary.csv")
    before = path.read_bytes()

    def partial_write(frame: pd.DataFrame, stream: Any, **kwargs: Any) -> None:
        """Simulate a disk failure after partial output.

        Args:
            frame: DataFrame whose output is being injected.
            stream: Writable temporary file.
            **kwargs: CSV options passed by the production writer.

        Raises:
            OSError: Always, after writing partial content.
        """
        stream.write("partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write)
    with pytest.raises(OSError, match="injected"):
        write_dict_csv({"topic": ["replacement"]}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_phrase_application_rejects_same_file(tmp_path: Path, alias: str) -> None:
    """Protect sentence inputs from overwriting through path aliases.

    Args:
        tmp_path: Isolated phrase output directory.
        alias: File identity alias to exercise.
    """
    path = tmp_path / "sentences.txt"
    path.write_text("new york office\n" * 10)
    cfg = Config(seeds={"topic": ["office"]}, phrase_min_count=1, phrase_threshold=0.1)
    model = tmp_path / "phrases.mod"
    train_phrase_model(path, model, cfg)
    output = path
    if alias == "symlink":
        output = tmp_path / "linked.txt"
        output.symlink_to(path)
    elif alias == "hardlink":
        output = tmp_path / "linked.txt"
        os.link(path, output)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different files"):
        apply_phrase_model(path, output, model)
    assert path.read_bytes() == before


def test_phrase_application_accepts_string_model_path(tmp_path: Path) -> None:
    """Exercise advertised string paths with a real saved phrase model.

    Args:
        tmp_path: Isolated phrase output directory.
    """
    path = tmp_path / "sentences.txt"
    path.write_text("new york office\n" * 20 + "\n")
    cfg = Config(seeds={"topic": ["office"]}, phrase_min_count=1,
                 phrase_threshold=0.1, stopwords=set())
    model = tmp_path / "phrases.mod"
    train_phrase_model(path, model, cfg)
    output = apply_phrase_model(str(path), str(tmp_path / "output.txt"), str(model))
    assert "new_york" in output.read_text()
    assert len(output.read_text().splitlines()) == 21
