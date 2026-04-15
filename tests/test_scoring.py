"""Tests for scoring kernels."""

from __future__ import annotations

import math

import pandas as pd

from lmsy_w2v_rfs.scoring import (
    aggregate_to_firm_year,
    document_frequencies,
    score_document,
    score_documents,
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
