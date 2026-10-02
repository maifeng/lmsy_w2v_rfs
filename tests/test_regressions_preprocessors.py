"""Test parser fixes with synthetic annotations and optional real spaCy output.

Inputs are isolated fake parser annotations, a fake CoreNLP client, and an
installed spaCy model when available. Output is deterministic pytest results;
no Java server, model download, or external service is required.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lmsy_w2v_rfs import Config
from lmsy_w2v_rfs.preprocessors.base import Preprocessor
from lmsy_w2v_rfs.preprocessors.none_pp import NoOpPreprocessor
from lmsy_w2v_rfs.preprocessors.spacy_pp import SpacyPreprocessor
from lmsy_w2v_rfs.preprocessors.static_mwe import StaticMWEPreprocessor


@pytest.fixture
def parser_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    """Load optional parser adapters independently of installed NLP extras.

    Args:
        monkeypatch: Scoped module substitution helper.

    Returns:
        Adapter modules using a fake stanza dependency without model loading.
    """
    dependency = ModuleType("stanza")
    dependency.server = SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stanza", dependency)
    directory = Path(__file__).resolve().parents[1] / "src/lmsy_w2v_rfs/preprocessors"
    modules: dict[str, ModuleType] = {}
    for name in ("stanza", "corenlp"):
        spec = importlib.util.spec_from_file_location(
            f"lmsy_w2v_rfs.preprocessors._regression_{name}", directory / f"{name}_pp.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def sentence(backend: str, entity_length: int, offset: int) -> Any:
    """Create adjacent lexical MWEs followed by an entity and a trailing word.

    Args:
        backend: Parser annotation format.
        entity_length: Number of tokens in the entity span, or zero for no entity.
        offset: Global sentence token offset for backends that support it.

    Returns:
        A backend-shaped annotation with MWE edges entering the entity span.
    """
    words = ["chief", "technology", *(["Apple", "Inc"][:entity_length]), "announce"]
    entity_indexes = list(range(2, 2 + entity_length))
    if backend == "spacy":
        tokens = [SimpleNamespace(i=offset + i, lemma_=word,
                                  dep_="compound" if i < len(words) - 1 else "ROOT")
                  for i, word in enumerate(words)]
        for i, token in enumerate(tokens):
            token.head = tokens[min(i + 1, len(tokens) - 1)]

        class Entity(list[Any]):
            """Minimal iterable entity annotation."""

            label_ = "ORG"

        class Sentence(list[Any]):
            """Minimal iterable sentence annotation."""

            ents: list[Entity]

        sent = Sentence(tokens)
        sent.ents = [Entity(tokens[2:2 + entity_length])] if entity_length else []
        return sent
    if backend == "stanza":
        return SimpleNamespace(
            words=[SimpleNamespace(id=i + 1, lemma=word, text=word,
                                   deprel="compound" if i < len(words) - 1 else "root",
                                   head=i + 2 if i < len(words) - 1 else 0)
                   for i, word in enumerate(words)],
            ents=[SimpleNamespace(type="ORG", tokens=[SimpleNamespace(id=(i + 1,))
                                                       for i in entity_indexes])]
            if entity_length else [],
        )
    return SimpleNamespace(
        token=[SimpleNamespace(tokenBeginIndex=offset + i, lemma=word, word=word)
               for i, word in enumerate(words)],
        mentions=[SimpleNamespace(tokenStartInSentenceInclusive=2,
                                  tokenEndInSentenceExclusive=2 + entity_length,
                                  entityType="ORG")] if entity_length else [],
        enhancedPlusPlusDependencies=SimpleNamespace(
            edge=[SimpleNamespace(dep="compound", source=i + 1, target=i + 2)
                  for i in range(len(words) - 1)]),
    )


@pytest.mark.parametrize("backend", ["spacy", "stanza", "corenlp"])
@pytest.mark.parametrize("entity_length", [0, 1, 2])
@pytest.mark.parametrize("offset", [0, 17])
def test_entity_boundary_preserves_placeholder_and_mwes(
    parser_modules: dict[str, ModuleType], backend: str, entity_length: int, offset: int
) -> None:
    """Preserve entity starts without breaking lexical joins or trailing tokens.

    Args:
        parser_modules: Isolated optional adapter modules.
        backend: Parser under test.
        entity_length: Entity span size.
        offset: Global sentence token offset.
    """
    classes = {"spacy": SpacyPreprocessor,
               "stanza": parser_modules["stanza"].StanzaPreprocessor,
               "corenlp": parser_modules["corenlp"].CoreNLPPreprocessor}
    parser = object.__new__(classes[backend])
    actual = parser._sentence_tokens(sentence(backend, entity_length, offset))
    expected = (["chief_technology", "[NER:ORG]", "announce"] if entity_length
                else ["chief_technology_announce"])
    assert actual == expected


@pytest.mark.parametrize("workers", [1, 2])
def test_corenlp_nondefault_port_and_lifecycle(
    parser_modules: dict[str, ModuleType], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """Verify configured endpoint, ordered batch execution, and context cleanup.

    Args:
        parser_modules: Isolated optional adapter modules.
        tmp_path: Temporary installation directory.
        monkeypatch: Scoped dependency substitution helper.
        workers: Number of request workers.
    """
    module = parser_modules["corenlp"]
    home = tmp_path / "corenlp"
    home.mkdir()
    (home / "stanford-corenlp-test.jar").touch()
    settings: dict[str, Any] = {}
    events: list[str] = []

    class Client:
        """Record endpoint and lifecycle while supplying deterministic annotations."""

        def __init__(self, **kwargs: Any) -> None:
            """Capture constructor settings.

            Args:
                **kwargs: CoreNLP client configuration.
            """
            settings.update(kwargs)

        def start(self) -> None:
            """Record client startup."""
            events.append("start")

        def stop(self) -> None:
            """Record client shutdown."""
            events.append("stop")

        def annotate(self, text: str) -> SimpleNamespace:
            """Produce one token without any parser-specific transformations.

            Args:
                text: Document text to return as its token.

            Returns:
                Fake CoreNLP document annotation.
            """
            sent = SimpleNamespace(
                token=[SimpleNamespace(tokenBeginIndex=0, lemma=text, word=text)],
                mentions=[], enhancedPlusPlusDependencies=SimpleNamespace(edge=[]))
            return SimpleNamespace(sentence=[sent])

    monkeypatch.setattr(module, "default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(module.stanza.server, "CoreNLPClient", Client, raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/mock/java")
    monkeypatch.setenv("CORENLP_HOME", str(home))
    cfg = Config(seeds={"demo": ["alpha"]}, corenlp_port=9123, n_cores=workers,
                 corenlp_properties={"ner.applyFineGrained": "true"})
    with module.CoreNLPPreprocessor(cfg) as parser:
        assert isinstance(parser, Preprocessor)
        assert parser.process("alpha") == [["alpha"]]
        assert list(parser.process_documents(iter(["beta", "alpha"]))) == [
            [["beta"]], [["alpha"]]]
    assert settings["endpoint"] == "http://localhost:9123"
    assert settings["threads"] == workers
    assert settings["properties"]["ner.applyFineGrained"] == "true"
    assert events == ["start", "stop"]


def test_corenlp_context_closes_on_exception(parser_modules: dict[str, ModuleType]) -> None:
    """Ensure a failed operation still closes the CoreNLP client.

    Args:
        parser_modules: Isolated optional adapter modules.
    """
    parser = object.__new__(parser_modules["corenlp"].CoreNLPPreprocessor)
    events: list[str] = []
    parser._client = SimpleNamespace(stop=lambda: events.append("stop"))
    with pytest.raises(ValueError, match="parse failure"), parser:
        raise ValueError("parse failure")
    assert events == ["stop"]


@pytest.mark.parametrize("backend", ["none", "static", "stanza"])
def test_backends_inherit_serial_batch_protocol(
    parser_modules: dict[str, ModuleType], backend: str
) -> None:
    """Require concrete backends to expose the protocol's ordered default batch.

    Args:
        parser_modules: Isolated optional adapter modules.
        backend: Serial backend under test.
    """
    parser: Any
    if backend == "none":
        parser = NoOpPreprocessor()
    elif backend == "static":
        parser = StaticMWEPreprocessor("finance")
    else:
        parser = object.__new__(parser_modules["stanza"].StanzaPreprocessor)
        parser.nlp = lambda text: SimpleNamespace(sentences=[SimpleNamespace(
            ents=[], words=[SimpleNamespace(id=1, lemma=text.lower(), text=text,
                                           deprel="root", head=0)])])
    texts = ["Alpha", "Beta"]
    assert isinstance(parser, Preprocessor)
    batch: Iterator[list[list[str]]] = parser.process_documents(iter(texts))
    assert list(batch) == [parser.process(text) for text in texts]


def test_spacy_natural_entity_boundary() -> None:
    """Check the real Microsoft entity construction when its optional model exists."""
    try:
        import spacy
    except (ImportError, OSError):
        pytest.skip("spaCy extra unavailable")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy en_core_web_sm model unavailable")
    text = "Technology giant Microsoft announced a new product."
    doc = nlp(text)
    assert any(ent.text == "Microsoft" and ent.label_ == "ORG" for ent in doc.ents)
    parser = object.__new__(SpacyPreprocessor)
    parser.nlp = nlp
    tokens = [token for sent in parser.process(text) for token in sent]
    assert tokens.count("[NER:ORG]") == 1
    assert "announce" in tokens
