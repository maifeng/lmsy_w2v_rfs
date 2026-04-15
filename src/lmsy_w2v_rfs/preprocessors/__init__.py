"""Pluggable preprocessors for Phase 1 (MWE join + NER mask).

Each backend implements a single method ``process(text) -> list[list[str]]``:
a list of sentences, each a list of lemmatized tokens with MWE groups joined
by ``_`` and named-entity spans replaced by a ``[NER:TYPE]`` placeholder.

Backends, selected by ``Config.preprocessor``:

- ``"none"``: whitespace split, no parse, no NER. Fastest, no external models.
- ``"static"``: whitespace split, then NLTK ``MWETokenizer`` with the packaged
  ``mwes_finance.txt`` list. No NER, no Java, no ML.
- ``"stanza"``: stanza.Pipeline (UD v2). Needs the ``[stanza]`` extra.
- ``"corenlp"``: CoreNLP server via stanza.server. Needs the ``[corenlp]``
  extra and Java. Paper-exact reproduction path.
- ``"spacy"``: spaCy ``en_core_web_trf`` by default. Needs the ``[spacy]``
  extra. **Recommended default**: fastest, best NER, cleanest vocab.

Independent of the backend, ``Config.mwe_list`` names a curated MWE file
that is applied AFTER the main preprocessor as a second pass. Default is
``None`` (skip). The packaged ``"finance"`` list is an OPT-IN example for
earnings-call text, not a default. Pass a path to use your own list.

The default pipeline (``preprocessor="corenlp"``, ``mwe_list=None``)
matches the 2021 paper's Phase 1 behavior: CoreNLP handles the syntactic
MWE joins, and gensim ``Phrases`` in Phase 2 learns high-frequency
collocations (like ``forward_looking_statement``) statistically from
your corpus.

Backend choice is a trade-off the user makes at runtime. See the
MWE benchmark comparison in the docs for the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Preprocessor

if TYPE_CHECKING:
    from ..config import Config


_IMPORT_ERRORS = {
    "stanza": (
        "The stanza preprocessor needs the '[stanza]' extra. "
        "Install with: pip install 'lmsy_w2v_rfs[stanza]'."
    ),
    "corenlp": (
        "The corenlp preprocessor needs the '[corenlp]' extra AND Java 8+. "
        "Install with: pip install 'lmsy_w2v_rfs[corenlp]' and run "
        "'lmsy-w2v-rfs download-corenlp' once."
    ),
    "spacy": (
        "The spacy preprocessor needs the '[spacy]' extra plus a spaCy model. "
        "Install with: pip install 'lmsy_w2v_rfs[spacy]' and "
        "'python -m spacy download en_core_web_trf' (or en_core_web_sm for a "
        "smaller, faster alternative)."
    ),
}


def build_preprocessor(config: "Config") -> Preprocessor:
    """Instantiate the preprocessor named in ``config.preprocessor``.

    Args:
        config: Pipeline config.

    Returns:
        A preprocessor instance.

    Raises:
        ImportError: If the chosen backend's optional extra is not installed.
        ValueError: If ``config.preprocessor`` is not a known name.
    """
    name = config.preprocessor
    if name == "none":
        from .none_pp import NoOpPreprocessor

        return NoOpPreprocessor()
    if name == "static":
        from .static_mwe import StaticMWEPreprocessor

        if config.mwe_list is None:
            raise ValueError(
                "preprocessor='static' needs config.mwe_list to point at an "
                "MWE file. Set it to a path, to 'finance' for the packaged "
                "earnings-call example, or switch preprocessor to 'none'."
            )
        return StaticMWEPreprocessor(mwe_source=config.mwe_list, lowercase=True)
    if name == "stanza":
        try:
            from .stanza_pp import StanzaPreprocessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(_IMPORT_ERRORS["stanza"]) from e
        return StanzaPreprocessor(config)
    if name == "corenlp":
        try:
            from .corenlp_pp import CoreNLPPreprocessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(_IMPORT_ERRORS["corenlp"]) from e
        return CoreNLPPreprocessor(config)
    if name == "spacy":
        try:
            from .spacy_pp import SpacyPreprocessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(_IMPORT_ERRORS["spacy"]) from e
        return SpacyPreprocessor(config)
    raise ValueError(
        f"Unknown preprocessor: {name!r}. "
        f"Choose one of: none, static, stanza, corenlp, spacy."
    )


__all__ = ["Preprocessor", "build_preprocessor"]
