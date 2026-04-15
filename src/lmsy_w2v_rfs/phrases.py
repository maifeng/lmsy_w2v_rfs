"""gensim Phrases wrappers (Phase 2 of the construction procedure).

Two passes by default: a bigram pass then a trigram pass. Each pass
trains a ``gensim.models.Phrases`` on the sentence file and writes a
transformed file with ``_``-joined phrases.
"""

from __future__ import annotations

from pathlib import Path

import gensim
import tqdm
from gensim import models

from .config import Config


def train_phrase_model(
    sentences_path: Path | str,
    model_path: Path | str,
    config: Config,
) -> models.phrases.Phrases:
    """Train a gensim Phrases model on a line-sentence corpus.

    Args:
        sentences_path: Input file, one sentence per line, whitespace tokens.
        model_path: Where to save the trained Phrases model.
        config: Pipeline config.

    Returns:
        The trained Phrases model.
    """
    sentences_path = Path(sentences_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = gensim.models.word2vec.PathLineSentences(
        str(sentences_path), max_sentence_length=10_000_000
    )
    n_lines = _count_lines(sentences_path)
    phraser = models.phrases.Phrases(
        tqdm.tqdm(corpus, total=n_lines, desc=f"phrases {sentences_path.name}"),
        min_count=config.phrase_min_count,
        threshold=config.phrase_threshold,
        scoring="default",
        connector_words=config.stopwords,
    )
    phraser.save(str(model_path))
    return phraser


def apply_phrase_model(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str,
) -> Path:
    """Rewrite a sentence file with phrase tokens joined by ``_``.

    Args:
        input_path: Source sentence file.
        output_path: Destination sentence file.
        model_path: Saved Phrases model.

    Returns:
        The output path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phraser = models.phrases.Phrases.load(str(model_path))
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        for line in tqdm.tqdm(f_in, desc=f"apply {model_path.name}"):
            tokens = line.strip().split()
            if not tokens:
                f_out.write("\n")
                continue
            f_out.write(" ".join(phraser[tokens]) + "\n")
    return output_path


def learn_phrases(
    sentences_path: Path | str,
    work_dir: Path | str,
    config: Config,
) -> Path:
    """Run ``config.phrase_passes`` passes of Phrases training + apply.

    Pass 1 learns bigrams, pass 2 learns trigrams on top of the bigram
    output, and so on. The final phrase-expanded sentence file's path
    is returned.

    Args:
        sentences_path: Input unigram corpus.
        work_dir: Directory to write intermediate corpora and models.
        config: Pipeline config.

    Returns:
        Path to the last sentence file produced.
    """
    work_dir = Path(work_dir)
    (work_dir / "models").mkdir(parents=True, exist_ok=True)
    (work_dir / "corpora").mkdir(parents=True, exist_ok=True)

    current = Path(sentences_path)
    for i in range(1, max(1, config.phrase_passes) + 1):
        model_path = work_dir / "models" / f"phrases_pass{i}.mod"
        out_path = work_dir / "corpora" / f"pass{i}.txt"
        train_phrase_model(current, model_path, config)
        apply_phrase_model(current, out_path, model_path)
        current = out_path
    return current


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n
