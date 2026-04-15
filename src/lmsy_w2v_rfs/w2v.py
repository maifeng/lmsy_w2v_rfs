"""Word2Vec training wrapper (gensim 4 API).

Uses streaming ``PathLineSentences`` so the corpus never has to fit in
memory. All hyperparameters come from the ``Config`` dataclass.
"""

from __future__ import annotations

from pathlib import Path

import gensim
from gensim.models import Word2Vec

from .config import Config


def train_word2vec(
    sentences_path: Path | str,
    model_path: Path | str,
    config: Config,
) -> Word2Vec:
    """Train a Word2Vec model and save it to disk.

    Args:
        sentences_path: Input corpus, one sentence per line.
        model_path: Destination ``.mod`` path.
        config: Pipeline config.

    Returns:
        The trained Word2Vec model.
    """
    sentences_path = Path(sentences_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = gensim.models.word2vec.PathLineSentences(
        str(sentences_path), max_sentence_length=10_000_000
    )
    model = Word2Vec(
        corpus,
        vector_size=config.w2v_dim,
        window=config.w2v_window,
        min_count=config.w2v_min_count,
        workers=config.n_cores,
        epochs=config.w2v_epochs,
        seed=config.random_state,
    )
    model.save(str(model_path))
    return model


def load_word2vec(model_path: Path | str) -> Word2Vec:
    """Load a saved Word2Vec model.

    Args:
        model_path: Path produced by ``train_word2vec``.

    Returns:
        The loaded model.
    """
    return Word2Vec.load(str(model_path))
