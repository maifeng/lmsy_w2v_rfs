"""Tests for scoring kernels."""

from __future__ import annotations

import math

import pandas as pd

from lmsy_w2v_rfs.scoring import (
    aggregate_to_firm_year,
    document_frequencies,
    score_document,
    score_documents,
    word_contributions,
)


def test_score_document_tf_counts() -> None:
    doc = "integrity integrity ethics innovation"
    expanded = {"integrity": {"integrity", "ethics"}, "innovation": {"innovation"}}
    scores, length = score_document(doc, expanded, method="TF")
    # dims are sorted alphabetically: innovation, integrity
    assert length == 4
    assert scores == [1.0, 3.0]


def test_score_document_tfidf_matches_formula() -> None:
    doc = "integrity integrity ethics"
    expanded = {"integrity": {"integrity", "ethics"}}
    df = {"integrity": 1, "ethics": 2}
    n = 10
    scores, length = score_document(
        doc, expanded, method="TFIDF", df_dict=df, n_docs=n
    )
    assert length == 3
    expected = 2 * math.log(n / 1) + 1 * math.log(n / 2)
    assert abs(scores[0] - expected) < 1e-9


def test_score_document_wfidf_matches_formula() -> None:
    doc = "integrity integrity ethics"
    expanded = {"integrity": {"integrity", "ethics"}}
    df = {"integrity": 1, "ethics": 2}
    n = 10
    scores, length = score_document(
        doc, expanded, method="WFIDF", df_dict=df, n_docs=n
    )
    assert length == 3
    # WFIDF weight per hit = (1 + log(tf)) * log(N/df)
    expected = (1 + math.log(2)) * math.log(n / 1) + (1 + math.log(1)) * math.log(n / 2)
    assert abs(scores[0] - expected) < 1e-9


def test_score_document_tfidf_missing_df_falls_back() -> None:
    # A dictionary word absent from df_dict must not raise; df defaults to 1.
    doc = "integrity newword"
    expanded = {"integrity": {"integrity", "newword"}}
    df = {"integrity": 2}  # 'newword' deliberately absent
    scores, _ = score_document(doc, expanded, method="TFIDF", df_dict=df, n_docs=10)
    expected = 1 * math.log(10 / 2) + 1 * math.log(10 / 1)
    assert abs(scores[0] - expected) < 1e-9


def test_aggregate_zero_length_document_is_zero_not_nan() -> None:
    scores = pd.DataFrame(
        {
            "Doc_ID": ["d1", "d2"],
            "integrity": [2.0, 0.0],
            "document_length": [100, 0],  # d2 is an empty document
        }
    )
    id2firm = pd.DataFrame(
        {"document_id": ["d1", "d2"], "firm_id": ["F1", "F2"], "time": [2020, 2020]}
    )
    out = aggregate_to_firm_year(scores, id2firm, dims=["integrity"])
    assert out["integrity"].notna().all()
    assert out.loc[out.firm_id == "F2", "integrity"].iloc[0] == 0.0


def test_score_documents_returns_dataframe() -> None:
    docs = [("a", "integrity ethics"), ("b", "innovation passion")]
    expanded = {
        "integrity": {"integrity", "ethics"},
        "innovation": {"innovation", "passion"},
    }
    df = score_documents(docs, expanded, method="TF")
    assert list(df.columns) == ["Doc_ID", "innovation", "integrity", "document_length"]
    assert df.loc[df.Doc_ID == "a", "integrity"].iloc[0] == 2.0
    assert df.loc[df.Doc_ID == "b", "innovation"].iloc[0] == 2.0


def test_word_contributions_shares_sum_to_one() -> None:
    docs = [
        ("a", "integrity integrity ethics"),
        ("b", "ethics innovation"),
        ("c", "innovation innovation passion"),
    ]
    expanded = {
        "integrity": {"integrity", "ethics"},
        "innovation": {"innovation", "passion"},
    }
    df, n = document_frequencies([t for _, t in docs])
    out = word_contributions(docs, expanded, method="TFIDF", df_dict=df, n_docs=n)
    assert list(out.columns) == [
        "dimension", "word", "contribution", "relative", "cumulative",
    ]
    # Relative shares within each dimension sum to 1, cumulative ends at ~1.
    for dim, grp in out.groupby("dimension"):
        assert abs(grp["relative"].sum() - 1.0) < 1e-9
        assert abs(grp["cumulative"].iloc[-1] - 1.0) < 1e-9
    # Within a dimension, contribution is sorted descending.
    integ = out[out.dimension == "integrity"]["contribution"].tolist()
    assert integ == sorted(integ, reverse=True)


def test_document_frequencies() -> None:
    df, n = document_frequencies(["a b c", "a b", "a"])
    assert n == 3
    assert df == {"a": 3, "b": 2, "c": 1}


def test_aggregate_to_firm_year() -> None:
    scores = pd.DataFrame(
        {
            "Doc_ID": ["d1", "d2"],
            "integrity": [2.0, 4.0],
            "document_length": [100, 200],
        }
    )
    id2firm = pd.DataFrame(
        {"document_id": ["d1", "d2"], "firm_id": ["F1", "F1"], "time": [2020, 2020]}
    )
    out = aggregate_to_firm_year(scores, id2firm, dims=["integrity"])
    # adjusted: 100*2/100 = 2.0 and 100*4/200 = 2.0 -> mean 2.0
    assert out.loc[0, "integrity"] == 2.0
