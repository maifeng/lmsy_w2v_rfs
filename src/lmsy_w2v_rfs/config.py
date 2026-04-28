"""Config dataclass, seed loading, and stopword list.

The package is theory-agnostic: every run requires a user-supplied seed
dictionary. The 2021 paper's 5-dimension culture seeds are shipped only as
a named example via :func:`load_example_seeds`, never as a default.
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


_EXAMPLE_SEEDS: dict[str, str] = {
    "culture_2021": "seeds_culture.json",
}
"""Named example seed dictionaries shipped as package data.

Each entry maps a short example name to a JSON file under
``lmsy_w2v_rfs/data/``. Add new entries here to expose more reproducible
example dictionaries.
"""


def _load_stopwords() -> set[str]:
    """Load the packaged SRAF generic stopword list.

    Returns:
        Lowercased stopword tokens.
    """
    text = resources.files("lmsy_w2v_rfs.data").joinpath("stopwords_sraf.txt").read_text()
    return {w.strip().lower() for w in text.split() if w.strip()}


STOPWORDS_SRAF: set[str] = _load_stopwords()
"""120-token generic stopword list from Notre Dame SRAF."""


def load_example_seeds(name: str) -> dict[str, list[str]]:
    """Load a named example seed dictionary shipped with the package.

    The package itself is seed-agnostic. This helper is provided so users
    who want to reproduce a specific paper can opt in by name. The 2021
    RFS paper's five-dimension culture dictionary is currently the only
    example.

    Args:
        name: Short identifier of the example. Currently:
            ``"culture_2021"`` (Li, Mai, Shen, Yan 2021, RFS).

    Returns:
        A fresh ``dict[str, list[str]]`` copy of the example seeds.

    Raises:
        KeyError: If ``name`` is not a known example.

    Example::

        from lmsy_w2v_rfs import Config, load_example_seeds

        seeds = load_example_seeds("culture_2021")
        cfg = Config(seeds=seeds, preprocessor="none")
    """
    if name not in _EXAMPLE_SEEDS:
        known = ", ".join(sorted(_EXAMPLE_SEEDS)) or "(none)"
        raise KeyError(
            f"Unknown example seed dictionary {name!r}. Known: {known}."
        )
    filename = _EXAMPLE_SEEDS[name]
    with resources.files("lmsy_w2v_rfs.data").joinpath(filename).open("r") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data["seeds"].items()}


@dataclass(frozen=True)
class Config:
    """Hyperparameters for the seed-expansion pipeline.

    Two construction phases compose to build the training corpus:

    * **Phase 1** (preprocessor + optional static MWE post-pass). Lemmatize,
      mask named entities as ``[NER:TYPE]``, join multi-word expressions.
      Pick a preprocessor with ``preprocessor``; optionally apply a curated
      static MWE list as a second pass with ``mwe_list``.
    * **Phase 2** (gensim Phrases). Learns corpus-specific bigrams and
      trigrams via co-occurrence statistics. On by default.

    Seeds are required and have no default. Pass any mapping of
    dimension name to seed words; the package is theory-agnostic.

    Attributes:
        seeds: Mapping of dimension name to seed word list. Required.
        stopwords: Lowercased stopwords removed during cleaning.
        preprocessor: Backend to use for Phase 1. One of
            ``corenlp`` (default; CoreNLP server via stanza.server; needs Java
            and the ``[corenlp]`` extra; paper-exact),
            ``spacy`` (faster on modern hardware; needs ``pip install spacy``),
            ``stanza`` (stanza.Pipeline; Python-native),
            ``static`` (NLTK MWETokenizer with the packaged list; Java-free),
            ``none`` (whitespace split only; Java-free, for already-tokenized
            input).
        mwe_list: Curated MWE list applied AFTER the main preprocessor as
            a second pass. ``"finance"`` for the packaged list, a path for
            your own, or ``None`` (default) to skip.
        spacy_model: Name of the spaCy model when ``preprocessor="spacy"``.
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
        zca_whiten: Apply ZCA whitening to the dimension columns.
        zca_epsilon: Numerical stabilizer for ZCA.
        random_state: Seed for Word2Vec.
    """

    seeds: dict[str, list[str]]
    stopwords: set[str] = field(default_factory=lambda: set(STOPWORDS_SRAF))

    # Phase 1: preprocessing.
    preprocessor: PreprocessorName = "corenlp"
    mwe_list: str | Path | None = None
    spacy_model: str = "en_core_web_sm"

    # CoreNLP-specific (used only when preprocessor == "corenlp")
    corenlp_memory: str = "6G"
    corenlp_port: int = 9002
    corenlp_timeout_ms: int = 120_000
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
    zca_whiten: bool = False
    zca_epsilon: float = 1e-6

    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError(
                "Config.seeds is required and must be non-empty. "
                "Pass a mapping of dimension name to seed word list, e.g. "
                'Config(seeds={"risk": ["risk", "uncertainty"], "growth": [...]}). '
                "To reproduce the 2021 paper, use "
                'load_example_seeds("culture_2021").'
            )
        for dim, words in self.seeds.items():
            if not isinstance(dim, str) or not dim:
                raise ValueError(f"Seed dimension names must be non-empty strings. Got: {dim!r}")
            if not isinstance(words, list) or not words:
                raise ValueError(
                    f"Seeds for dimension {dim!r} must be a non-empty list of strings."
                )

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


def load_seeds(source: str | Path | dict[str, list[str]]) -> dict[str, list[str]]:
    """Load a seed dictionary from a dict, a JSON file, or a text file.

    The package is domain-agnostic: the seed dictionary is the *only*
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
        source: A dict, or a path to a ``.json`` or ``.txt`` file.

    Returns:
        Mapping of dimension name to seed word list.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If the JSON file is not a ``dict[str, list[str]]``.
        TypeError: If ``source`` is ``None``.
    """
    import json

    if source is None:
        raise TypeError(
            "load_seeds(source) requires a dict, a JSON path, or a text path. "
            "There is no built-in default. To reproduce the 2021 paper, use "
            'load_example_seeds("culture_2021").'
        )
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
