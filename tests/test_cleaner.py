"""Tests for ``cleaner`` module."""

from __future__ import annotations

from lmsy_w2v_rfs.cleaner import clean_corenlp_line, clean_plain_line
from lmsy_w2v_rfs.config import STOPWORDS_SRAF


def test_clean_corenlp_strips_tags() -> None:
    line = "[NER:ORGANIZATION]apple[pos:NNP] announce[pos:VBZ] iphone[pos:NN] ."
    out = clean_corenlp_line(line, STOPWORDS_SRAF)
    assert "[NER:" not in out
    assert "[pos:" not in out
    assert "iphone" in out


def test_clean_corenlp_drops_stopwords() -> None:
    line = "the[pos:DT] and[pos:CC] innovation[pos:NN] ."
    out = clean_corenlp_line(line, STOPWORDS_SRAF)
    assert "innovation" in out
    assert "the" not in out and "and" not in out


def test_clean_plain_line() -> None:
    out = clean_plain_line("The Customers demand QUALITY, and innovation!", STOPWORDS_SRAF)
    tokens = out.split()
    assert "customers" in tokens
    assert "quality" in tokens
    assert "the" not in tokens and "and" not in tokens


def test_clean_plain_drops_single_letters_and_pure_numbers() -> None:
    out = clean_plain_line("a 12 innovation", STOPWORDS_SRAF)
    assert out == "innovation"
