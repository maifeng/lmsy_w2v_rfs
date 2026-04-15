"""Word2Vec corporate-culture measurement, from Li et al. (2021, RFS).

Public API:
    Pipeline                        - end-to-end orchestrator
    Config                          - hyperparameter dataclass
    Preprocessor, build_preprocessor - Phase 1a preprocessor protocol and factory
    CULTURE_SEEDS, CULTURE_DIMS     - five-dimension seed dictionary
    STOPWORDS_SRAF                  - 120-token SRAF generic stopword list
    load_mwe_list, apply_mwe_list   - optional static-MWE post-pass helpers
    download_corenlp                - one-call CoreNLP install (optional extra)

Minimal example::

    from lmsy_w2v_rfs import Pipeline, Config

    p = Pipeline(
        texts=["innovation is our core value ...", "we value our customers ..."],
        doc_ids=["doc1", "doc2"],
        work_dir="runs/demo",
        config=Config(preprocessor="none"),
    )
    p.run()
    print(p.score_df("TFIDF"))

Paper:
    Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021).
    "Measuring Corporate Culture Using Machine Learning."
    Review of Financial Studies 34(7):3265-3315.
    https://doi.org/10.1093/rfs/hhaa079
"""

from __future__ import annotations

from .config import (
    CULTURE_DIMS,
    CULTURE_SEEDS,
    STOPWORDS_SRAF,
    Config,
    PreprocessorName,
    default_cache_dir,
    load_seeds,
)
from .preprocessors import Preprocessor, build_preprocessor
from .preprocessors.base import apply_mwe_list, load_mwe_list
from .dictionary import (
    expand_words_dimension_mean,
    read_dict_csv,
    write_dict_csv,
)
from .pipeline import Pipeline
from .scoring import (
    aggregate_to_firm_year,
    document_frequencies,
    iter_doc_level_corpus,
    score_document,
    score_documents,
    zca_whiten,
)
from .w2v import load_word2vec, train_word2vec

__version__ = "0.1.0a1"

__paper__ = (
    "Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan. 2021. "
    "'Measuring Corporate Culture Using Machine Learning.' "
    "Review of Financial Studies 34(7):3265-3315. "
    "https://doi.org/10.1093/rfs/hhaa079"
)


def download_corenlp(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Install Stanford CoreNLP into the local cache directory.

    Imported lazily so the base install does not require ``stanza``.
    Requires the ``[corenlp]`` extra.
    """
    import os
    import pathlib

    import stanza

    home = kwargs.pop("install_dir", None) or default_cache_dir() / "corenlp"
    home = pathlib.Path(home).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    stanza.install_corenlp(dir=str(home), *args, **kwargs)
    os.environ["CORENLP_HOME"] = str(home)
    return home


__all__ = [
    "Pipeline",
    "Config",
    "PreprocessorName",
    "Preprocessor",
    "build_preprocessor",
    "load_seeds",
    "load_mwe_list",
    "apply_mwe_list",
    "CULTURE_SEEDS",
    "CULTURE_DIMS",
    "STOPWORDS_SRAF",
    "default_cache_dir",
    "download_corenlp",
    "train_word2vec",
    "load_word2vec",
    "expand_words_dimension_mean",
    "read_dict_csv",
    "write_dict_csv",
    "score_document",
    "score_documents",
    "document_frequencies",
    "iter_doc_level_corpus",
    "aggregate_to_firm_year",
    "zca_whiten",
    "__version__",
    "__paper__",
]
