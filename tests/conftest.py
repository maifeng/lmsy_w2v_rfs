"""Shared fixtures for the offline test suite."""

from __future__ import annotations

import random
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_corpus() -> list[str]:
    """Return a deterministic toy corpus that covers every culture dimension."""
    random.seed(0)
    dims_sentences = {
        "integrity": [
            "integrity and ethics are core to our accountability and trust.",
            "we act with honesty and transparency in all responsibilities.",
            "our ethical standards demand fairness and accountability.",
            "trust is built on honesty transparency and responsibility.",
        ],
        "teamwork": [
            "teamwork and collaboration drive every project we deliver.",
            "we cooperate across teams through a collaborative cooperative culture.",
            "cooperation and teamwork unlock better cooperation between departments.",
            "collaboration is how we grow: teamwork cooperation collaborative work.",
        ],
        "innovation": [
            "innovation and creativity define our product roadmap.",
            "we innovate with passion and creative excellence.",
            "our innovative culture drives creativity passion and pride.",
            "efficiency innovation and passion make our teams excel.",
        ],
        "respect": [
            "we respect every employee and foster dignity and empowerment.",
            "respectful treatment and empowerment are part of our talent strategy.",
            "our talented employees are empowered with dignity and respect.",
            "we empower respectful and talented employees across the company.",
        ],
        "quality": [
            "quality and customer dedication are our top priorities.",
            "dedicated to customer expectations with high quality service.",
            "our customers expect quality and dedication from every team.",
            "we dedicate our quality efforts to customer commitment.",
        ],
    }
    corpus: list[str] = []
    # Build 40 documents (8 per dimension) so gensim Phrases has material
    # and Word2Vec vocab is large enough to be useful.
    for dim, sents in dims_sentences.items():
        for _ in range(8):
            doc = " ".join(random.sample(sents, k=len(sents)))
            # add filler with dimension-flavored words
            doc += " " + " ".join(random.sample(sents, k=2)) + " " + dim
            corpus.append(doc)
    random.shuffle(corpus)
    return corpus


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """A per-test scratch directory."""
    return tmp_path / "run"
