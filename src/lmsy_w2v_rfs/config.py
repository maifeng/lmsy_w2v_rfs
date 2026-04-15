"""Config dataclass, seed dictionary, and stopword list.

Everything hyperparameter-shaped lives here. Values match the 2021
replication repo's ``global_options.py`` where applicable.

Phase 1 preprocessing is pluggable. See ``Config.preprocessor`` below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any, Literal


PreprocessorName = Literal["none", "static", "stanza", "corenlp", "spacy"]
"""Valid values for ``Config.preprocessor``."""


def _load_seed_json() -> dict[str, Any]:
    """Load the packaged culture seed dictionary.

    Returns:
        The parsed ``data/seeds_culture.json`` contents.
    """
    with resources.files("lmsy_w2v_rfs.data").joinpath("seeds_culture.json").open("r") as f:
        return json.load(f)


def _load_stopwords() -> set[str]:
    """Load the packaged SRAF generic stopword list.

    Returns:
        Lowercased stopword tokens.
    """
    text = resources.files("lmsy_w2v_rfs.data").joinpath("stopwords_sraf.txt").read_text()
    return {w.strip().lower() for w in text.split() if w.strip()}


_SEED_JSON: dict[str, Any] = _load_seed_json()

CULTURE_DIMS: list[str] = list(_SEED_JSON["dims"])
"""Five culture dimensions from Li et al. (2021, RFS)."""

CULTURE_SEEDS: dict[str, list[str]] = {k: list(v) for k, v in _SEED_JSON["seeds"].items()}
"""Five-dimension seed dictionary. 47 seed words total."""

STOPWORDS_SRAF: set[str] = _load_stopwords()
"""120-token generic stopword list from Notre Dame SRAF."""


@dataclass(frozen=True)
class Config:
    """Hyperparameters for the RFS 2021 word2vec culture pipeline.

    Two construction phases are composed:

    * **Phase 1** (preprocessor + optional static MWE post-pass). Lemmatize,
      mask named entities as ``[NER:TYPE]``, join multi-word expressions.
      Pick a preprocessor with ``preprocessor``; optionally apply a curated
      static MWE list as a second pass with ``mwe_list``.
    * **Phase 2** (gensim Phrases). Learns corpus-specific bigrams and
      trigrams via co-occurrence statistics. On by default.

    Attributes:
        seeds: Mapping of dimension name to seed word list.
        stopwords: Lowercased stopwords removed during cleaning.
        preprocessor: Backend to use for Phase 1. One of
            ``none`` (whitespace split only; fastest),
            ``static`` (no parse; NLTK MWETokenizer with the packaged list;
            Java-free; zero-ML),
            ``stanza`` (stanza.Pipeline; Python-native),
            ``corenlp`` (CoreNLP server via stanza.server; Java; paper-exact),
            ``spacy`` (spaCy; recommended default; fastest parser and best
            NER on the benchmark).
        mwe_list: Curated MWE list applied AFTER the main preprocessor as a
            second pass. Set to ``"finance"`` for the packaged list, to a
            path for your own, or to ``None`` to skip.
        spacy_model: Name of the spaCy model to load when
            ``preprocessor="spacy"``.
        corenlp_memory: JVM heap for the CoreNLP server.
        corenlp_port: TCP port the CoreNLP server listens on.
        corenlp_timeout_ms: Per-request CoreNLP timeout.
        n_cores: Parallel workers for parsing and training.
        use_gensim_phrases: Whether to run gensim Phrases.
        phrase_passes: Number of phrase passes (1 bigram, 2 bigram+trigram).
        phrase_threshold: gensim Phrases score threshold.
        phrase_min_count: Minimum bigram count.
        w2v_dim: Word2Vec vector dimension.
        w2v_window: Word2Vec context window.
        w2v_min_count: Word2Vec minimum token count.
        w2v_epochs: Word2Vec training epochs.
        n_words_dim: Top-k expanded words per dimension.
        dict_restrict_vocab: Restrict expansion to the top fraction of vocab.
        min_similarity: Discard expansion candidates below this cosine.
        tfidf_normalize: L2-normalize the tf-idf vector per document.
        random_state: Seed for Word2Vec.
    """

    seeds: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in CULTURE_SEEDS.items()}
    )
    stopwords: set[str] = field(default_factory=lambda: set(STOPWORDS_SRAF))

    # Phase 1: preprocessing.
    #
    # The pipeline has two construction phases, both on by default:
    #   Phase 1a: parser-based syntactic MWE joining + NER masking.
    #   Phase 2:  gensim ``Phrases`` statistical bigram / trigram learning.
    #
    # Default ``preprocessor="corenlp"``: Stanford CoreNLP via
    # ``stanza.server.CoreNLPClient``. Matches the 2021 paper's pipeline,
    # gives the best syntactic MWE coverage (76% of the test set), and
    # scales near-linearly with JVM threads (5.7x at 8 threads on the
    # benchmark). Requires a one-time Java install plus
    # ``lmsy-w2v-rfs download-corenlp``.
    #
    # Backup options:
    #   ``"spacy"``     Python-native; fastest (3.9 min on 1,393 docs);
    #                    lose syntactic MWE coverage.
    #   ``"stanza"``    Python-native; Pythonic successor to CoreNLP;
    #                    slowest on CPU.
    #   ``"static"``    Curated-list-only pass; no parser, no NER.
    #   ``"none"``      Skip Phase 1a; rely on gensim ``Phrases`` alone.
    #
    # ``mwe_list`` is an optional SECOND pass that runs after the main
    # preprocessor. It takes a newline-delimited list of MWEs and joins
    # them with NLTK. ``"finance"`` loads the packaged earnings-call
    # example list; a path loads your own; ``None`` (default) skips it.
    preprocessor: PreprocessorName = "corenlp"
    mwe_list: str | Path | None = None
    spacy_model: str = "en_core_web_sm"

    # CoreNLP-specific (used only when preprocessor == "corenlp")
    corenlp_memory: str = "6G"
    corenlp_port: int = 9002
    corenlp_timeout_ms: int = 120_000
    # ``n_cores`` is the JVM thread pool size (also reused by stanza and spaCy
    # workers). 4 works well on an 8-core laptop without saturating the machine;
    # bump to 8 on workstations.
    n_cores: int = 4

    # Phase 2: gensim Phrases
    use_gensim_phrases: bool = True
    phrase_passes: int = 2
    phrase_threshold: float = 10.0
    phrase_min_count: int = 10

    # Word2Vec
    w2v_dim: int = 300
    w2v_window: int = 5
    w2v_min_count: int = 5
    w2v_epochs: int = 20

    # Dictionary expansion
    n_words_dim: int = 500
    dict_restrict_vocab: float | None = None
    min_similarity: float = 0.0

    # Scoring
    tfidf_normalize: bool = False
    # Optional ZCA whitening applied to every scored DataFrame.
    # Decorrelates the dimension columns while preserving column names and
    # approximate axis orientation. Off by default. Similar to the
    # post-processing in Marketing-Measures/marketing-measures.
    zca_whiten: bool = False
    zca_epsilon: float = 1e-6

    random_state: int = 42

    def with_(self, **kwargs: Any) -> Config:
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: Fields to override.

        Returns:
            A new ``Config``.
        """
        return replace(self, **kwargs)

    @property
    def dims(self) -> list[str]:
        """Dimension names in insertion order."""
        return list(self.seeds.keys())


def default_cache_dir() -> Path:
    """Return the default on-disk cache root for CoreNLP and phrase models.

    Returns:
        ``$LMSY_W2V_RFS_HOME`` if set, otherwise ``~/.cache/lmsy_w2v_rfs``.
    """
    env = os.environ.get("LMSY_W2V_RFS_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "lmsy_w2v_rfs"


def load_seeds(source: str | Path | dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Load a seed dictionary from a dict, a JSON file, or a text file.

    The framework is domain-agnostic: the seed dictionary is the *only*
    place where the user declares what concepts to measure. This helper
    accepts the three formats researchers typically have:

    **Python dict** (pass through)::

        {"integrity": ["integrity", "ethic", "honest"],
         "teamwork":  ["teamwork", "collaborate", "cooperate"]}

    **JSON file** (path ending in ``.json``)::

        {
          "integrity": ["integrity", "ethic", "honest"],
          "teamwork":  ["teamwork", "collaborate", "cooperate"]
        }

    **Text file** (anything else). One dimension per line, name and words
    separated by a colon, words by whitespace or commas. Blank lines and
    ``#`` comments are skipped::

        # culture dimensions
        integrity: integrity ethic ethical honest
        teamwork:  teamwork, collaboration, cooperate
        innovation: innovation innovate creative

    Args:
        source: A dict, a path to a ``.json`` or ``.txt`` file, or ``None``
            to return the packaged 2021-paper five-dimension default.

    Returns:
        Mapping of dimension name to seed word list.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the JSON file is not a ``dict[str, list[str]]``.
    """
    import json

    if source is None:
        return {k: list(v) for k, v in CULTURE_SEEDS.items()}
    if isinstance(source, dict):
        return {k: list(v) for k, v in source.items()}

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, list) for k, v in data.items()
        ):
            raise ValueError(
                f"Seeds JSON file {path} must be a dict of "
                f"str -> list[str]. Got: {type(data).__name__}"
            )
        return {k: [str(x) for x in v] for k, v in data.items()}

    # Plain text: "dim: word1 word2 word3" per line.
    out: dict[str, list[str]] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"{path}:{lineno}: expected 'dim: word1 word2 ...', got {raw!r}"
            )
        dim, rest = line.split(":", 1)
        words = [w for w in rest.replace(",", " ").split() if w]
        if not words:
            continue
        out[dim.strip()] = words
    return out
