"""Pluggable preprocessors for Phase 1 (MWE join + NER mask).

Each backend implements a single method ``process(text) -> list[list[str]]``:
a list of sentences, each a list of lemmatized tokens with MWE groups joined
by ``_`` and named-entity spans replaced by a ``[NER:TYPE]`` placeholder.

Backends, selected by ``Config.preprocessor``:

- ``"none"``: **default.** Whitespace split + lowercase, no parse, no NER.
  No external dependencies, so a bare ``pip install`` runs out of the box.
- ``"static"``: whitespace split, then NLTK ``MWETokenizer`` with a curated
  list (e.g. the packaged ``mwes_finance.txt``). No NER, no Java, no ML.
- ``"spacy"``: spaCy. Lemmatization, NER masking, dependency MWEs; fast and
  Python-native. Needs the ``[spacy]`` extra plus a model. Recommended when
  you want richer parsing than ``none``.
- ``"corenlp"``: CoreNLP server via stanza.server. Needs the ``[corenlp]``
  extra and Java. The paper-exact reproduction path.
- ``"stanza"``: stanza.Pipeline (UD v2). Needs the ``[stanza]`` extra; slowest.

Independent of the backend, ``Config.mwe_list`` names a curated MWE file
that is applied AFTER the main preprocessor as a second pass. Default is
``None`` (skip). The packaged ``"finance"`` list is an OPT-IN example for
earnings-call text, not a default. Pass a path to use your own list.

The default (``preprocessor="none"``) is chosen for zero-friction first runs.
For paper-faithful Phase 1 behavior use ``preprocessor="corenlp"``: CoreNLP
handles the syntactic MWE joins and lemmatization, and gensim ``Phrases`` in
Phase 2 learns high-frequency collocations (like ``forward_looking_statement``)
statistically from your corpus regardless of the Phase 1 backend.

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
        "The stanza preprocessor could not be loaded. It needs the '[stanza]' "
        "extra: pip install 'lmsy_w2v_rfs[stanza]'. "
        "If stanza is installed but still fails to import (e.g. an OSError about "
        "GLIBC on an older Linux/HPC node), the torch/stanza wheels are "
        "incompatible with that system — use preprocessor='spacy' or 'none'."
    ),
    "corenlp": (
        "The corenlp preprocessor could not be loaded. It needs the '[corenlp]' "
        "extra AND Java 8+: pip install 'lmsy_w2v_rfs[corenlp]', then run "
        "'lmsy-w2v-rfs download-corenlp' once. "
        "If the import itself fails with a GLIBC OSError on an older Linux/HPC "
        "node, use preprocessor='spacy' or 'none' instead."
    ),
    "spacy": (
        "The spacy preprocessor needs the '[spacy]' extra plus a spaCy model. "
        "Install with: pip install 'lmsy_w2v_rfs[spacy]' and "
        "'python -m spacy download en_core_web_sm'."
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
        # OSError catches the GLIBC wheel-incompatibility crash on old Linux;
        # ImportError catches the missing extra. Both happen at the module
        # import (stanza is imported at the top of stanza_pp).
        try:
            from .stanza_pp import StanzaPreprocessor
        except (ImportError, OSError) as e:
            raise ImportError(_IMPORT_ERRORS["stanza"]) from e
        return StanzaPreprocessor(config)
    if name == "corenlp":
        try:
            from .corenlp_pp import CoreNLPPreprocessor
        except (ImportError, OSError) as e:
            raise ImportError(_IMPORT_ERRORS["corenlp"]) from e
        return CoreNLPPreprocessor(config)
    if name == "spacy":
        # spaCy is imported inside SpacyPreprocessor.__init__ (BLAS env vars
        # must be set before the import), so the missing-package ImportError
        # surfaces at construction. A missing-model OSError is allowed to
        # propagate — it already carries its own actionable message.
        try:
            from .spacy_pp import SpacyPreprocessor

            return SpacyPreprocessor(config)
        except ImportError as e:
            raise ImportError(_IMPORT_ERRORS["spacy"]) from e
    raise ValueError(
        f"Unknown preprocessor: {name!r}. "
        f"Choose one of: none, static, stanza, corenlp, spacy."
    )


__all__ = ["Preprocessor", "build_preprocessor"]
