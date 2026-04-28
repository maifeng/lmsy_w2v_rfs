"""Tests for the ZCA whitening post-process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lmsy_w2v_rfs import zca_whiten


def test_zca_whiten_decorrelates_and_unit_variance() -> None:
    rng = np.random.default_rng(0)
    # Synthetic correlated scores: 5 dims, correlated via a random matrix.
    n = 200
    X = rng.normal(size=(n, 5)) @ rng.normal(size=(5, 5))
    df = pd.DataFrame(X, columns=list("abcde"))
    df["Doc_ID"] = [f"d{i}" for i in range(n)]
    df["document_length"] = 1000

    out = zca_whiten(df, dims=list("abcde"))

    # Off-diagonals of the covariance should be ~0, diagonals ~1.
    cov = np.cov(out[list("abcde")].to_numpy(), rowvar=False)
    assert np.allclose(np.diag(cov), 1.0, atol=1e-2), cov
    off_diag = cov - np.diag(np.diag(cov))
    assert np.max(np.abs(off_diag)) < 1e-2

    # Non-dimension columns are preserved.
    assert list(out["Doc_ID"]) == list(df["Doc_ID"])
    assert (out["document_length"] == 1000).all()


def test_zca_whiten_preserves_column_names_and_row_count() -> None:
    df = pd.DataFrame({
        "Doc_ID": ["a", "b", "c", "d"],
        "integrity": [1.0, 2.0, 3.0, 4.0],
        "quality": [4.0, 3.0, 2.0, 1.0],
        "document_length": [100, 200, 150, 175],
    })
    out = zca_whiten(df, dims=["integrity", "quality"])
    assert list(out.columns) == list(df.columns)
    assert len(out) == len(df)


def test_zca_whiten_handles_tiny_sample() -> None:
    # One row: cannot estimate covariance; should return input unchanged.
    df = pd.DataFrame({"Doc_ID": ["a"], "integrity": [1.0], "quality": [2.0],
                       "document_length": [100]})
    out = zca_whiten(df, dims=["integrity", "quality"])
    pd.testing.assert_frame_equal(out, df)


def test_pipeline_applies_zca_when_configured(tiny_corpus, work_dir, culture_seeds) -> None:
    from lmsy_w2v_rfs import Config, Pipeline

    ids = [f"d{i}" for i in range(len(tiny_corpus))]
    cfg = Config(
        seeds=culture_seeds,
        preprocessor="none", mwe_list=None,
        w2v_dim=20, w2v_epochs=3, w2v_min_count=1,
        phrase_min_count=2, phrase_threshold=1.0,
        n_words_dim=10, n_cores=1,
        zca_whiten=True,
    )
    p = Pipeline(texts=tiny_corpus, doc_ids=ids, work_dir=work_dir, config=cfg)
    p.run(methods=("TFIDF",))
    df = p.score_df("TFIDF")

    # After ZCA the dim columns should have approximately unit variance.
    dim_cov = np.cov(df[cfg.dims].to_numpy(), rowvar=False)
    assert np.allclose(np.diag(dim_cov), 1.0, atol=0.2)
