"""Check pipeline persistence and input contracts with isolated synthetic corpora.

Input: pytest temporary paths and small in-memory documents. Output: regression
results for cached identifiers, curation, configuration, and malformed input.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lmsy_w2v_rfs import Config, Pipeline, load_seeds
from lmsy_w2v_rfs.dictionary import write_dict_csv


def _config(**overrides: Any) -> Config:
    """Create an inexpensive offline configuration.

    Args:
        **overrides: Values to replace in the base configuration.

    Returns:
        A configuration requiring no external NLP models.
    """
    return Config(seeds={"risk": ["risk"]}, stopwords=set(),
                  use_gensim_phrases=False, n_cores=1).with_(**overrides)


def _prepared(path: Path, identifiers: list[str]) -> Pipeline:
    """Parse documents and install a fixed dictionary to isolate persistence.

    Args:
        path: Work directory for this pipeline.
        identifiers: Document identifiers to preserve.

    Returns:
        A pipeline ready for scoring without training embeddings.
    """
    pipe = Pipeline(texts=["risk risk ordinary"] * len(identifiers),
                    doc_ids=identifiers, work_dir=path, config=_config())
    pipe.parse()
    pipe.clean()
    write_dict_csv({"risk": ["risk"]}, pipe.dict_path)
    return pipe


@pytest.mark.parametrize("identifiers", [["0", "1"], ["001", "002"], ["NA", "null", "nan", "007"]])
@pytest.mark.parametrize("reader", ["score", "score_df"])
def test_score_reload_preserves_ids_and_aggregation(
    tmp_path: Path, identifiers: list[str], reader: str,
) -> None:
    """Require both score readers to preserve IDs and existing mapping joins.

    Args:
        tmp_path: Isolated run directory.
        identifiers: Literal identifiers, including numbers and NA-like strings.
        reader: Cached score loading entry point.
    """
    pipe = _prepared(tmp_path, identifiers)
    fresh = pipe.score(("TF",))["TF"]
    mapping = pd.DataFrame({"document_id": identifiers, "firm_id": ["firm"] * len(identifiers),
                            "time": [2024] * len(identifiers)})
    expected = pipe.firm_year(mapping, "TF")
    resumed = Pipeline(work_dir=tmp_path, config=_config())
    restored = resumed.score(("TF",))["TF"] if reader == "score" else resumed.score_df("TF")
    assert restored.Doc_ID.tolist() == identifiers
    pd.testing.assert_frame_equal(restored, fresh, check_dtype=False)
    pd.testing.assert_frame_equal(resumed.firm_year(mapping, "TF"), expected)


def test_score_id_converter_keeps_numeric_missing_values(tmp_path: Path) -> None:
    """Preserve identifier text without changing numeric missing-value parsing.

    Args:
        tmp_path: Isolated run directory.
    """
    pipe = Pipeline(work_dir=tmp_path, config=_config())
    pipe.scores_path("TF").parent.mkdir()
    pipe.scores_path("TF").write_text("Doc_ID,risk,document_length\nNA,,3\n001,2,3\n")
    frame = pipe.score_df("TF")
    assert frame.Doc_ID.tolist() == ["NA", "001"]
    assert pd.isna(frame.risk.iloc[0])
    assert pd.api.types.is_numeric_dtype(frame.risk)


@pytest.mark.parametrize("reload_from_disk", [False, True])
def test_curation_invalidates_contribution_cache(tmp_path: Path, reload_from_disk: bool) -> None:
    """Require contribution diagnostics to reflect either curation entry point.

    Args:
        tmp_path: Isolated run directory.
        reload_from_disk: Whether to edit the CSV and invoke reload_dictionary.
    """
    pipe = _prepared(tmp_path, ["a"])
    assert pipe.word_contributions("TF").word.tolist() == ["risk"]
    if reload_from_disk:
        write_dict_csv({"risk": ["ordinary"]}, pipe.dict_path)
        pipe.reload_dictionary()
    else:
        pipe.edit_dictionary(remove={"risk": ["risk"]}, add={"risk": ["ordinary"]})
    assert pipe.word_contributions("TF").word.tolist() == ["ordinary"]


def test_failed_curation_preserves_all_prior_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Require a failed dictionary save to preserve memory, disk, and cached scores.

    Args:
        tmp_path: Isolated run directory.
        monkeypatch: Dependency substitution fixture.
    """
    pipe = _prepared(tmp_path, ["a"])
    scores = pipe.score(("TF",))["TF"]
    dictionary = pipe.expanded_dict
    disk = pipe.dict_path.read_bytes()

    def fail_save(words: dict[str, list[str]], path: Path) -> Path:
        """Raise a controlled write failure.

        Args:
            words: Attempted dictionary contents.
            path: Intended destination.

        Raises:
            OSError: Always, simulating a failed save.
        """
        raise OSError("simulated write failure")

    monkeypatch.setattr("lmsy_w2v_rfs.pipeline.write_dict_csv", fail_save)
    with pytest.raises(OSError, match="simulated"):
        pipe.edit_dictionary(remove={"risk": ["risk"]})
    assert pipe.expanded_dict is dictionary
    assert pipe.expanded_dict == {"risk": ["risk"]}
    assert pipe.dict_path.read_bytes() == disk
    pd.testing.assert_frame_equal(pipe.score_df("TF"), scores)


def test_dump_config_serializes_path_and_sets(tmp_path: Path) -> None:
    """Save supported Path and set-valued configuration fields as readable JSON.

    Args:
        tmp_path: Isolated run directory.
    """
    mwe_path = tmp_path / "mwe.txt"
    pipe = Pipeline(work_dir=tmp_path, config=_config(
        mwe_list=mwe_path, phrase_extra={"connector_words": {"of", "the"}}))
    pipe._dump_config()
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["mwe_list"] == str(mwe_path)
    assert saved["phrase_extra"]["connector_words"] == ["of", "the"]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("source", ["dataframe", "csv", "jsonl"])
def test_missing_text_is_blank_and_id_spelling_survives(tmp_path: Path, source: str) -> None:
    """Exclude missing text without introducing a nan/None document or losing IDs.

    Args:
        tmp_path: Isolated input and work directory.
        source: Factory entry point.
    """
    frame = pd.DataFrame({"id": ["001", "NA", "null"], "text": [None, "risk", "ordinary"]})
    kwargs = {"work_dir": tmp_path / "run", "config": _config()}
    if source == "dataframe":
        pipe = Pipeline.from_dataframe(frame, **kwargs)
    elif source == "csv":
        path = tmp_path / "input.csv"
        frame.to_csv(path, index=False)
        pipe = Pipeline.from_csv(path, **kwargs)
    else:
        path = tmp_path / "input.jsonl"
        frame.to_json(path, orient="records", lines=True)
        pipe = Pipeline.from_jsonl(path, **kwargs)
    pipe.parse()
    assert pipe._doc_ids == ["001", "NA", "null"]
    assert pipe.parsed_sents_path.read_text().splitlines() == ["risk", "ordinary"]
    assert pipe.parsed_ids_path.read_text().splitlines() == ["NA_0", "null_0"]


@pytest.mark.parametrize("identifier", [None, "", "  "])
def test_missing_explicit_ids_are_rejected(tmp_path: Path, identifier: str | None) -> None:
    """Reject missing explicit identifiers before a parsed cache can be created.

    Args:
        tmp_path: Isolated directory.
        identifier: Invalid identifier value.
    """
    with pytest.raises(ValueError, match="IDs"):
        Pipeline.from_dataframe(pd.DataFrame({"id": [identifier], "text": ["risk"]}),
                                work_dir=tmp_path / "df", config=_config())
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"id": identifier, "text": "risk"}) + "\n")
    with pytest.raises(ValueError, match="document ID"):
        Pipeline.from_jsonl(path, work_dir=tmp_path / "json", config=_config())


def test_csv_respects_explicit_identifier_dtype(tmp_path: Path) -> None:
    """Keep explicitly requested pandas conversion under caller control.

    Args:
        tmp_path: Isolated input and work directory.
    """
    path = tmp_path / "input.csv"
    path.write_text("id,text\n001,risk\n")
    pipe = Pipeline.from_csv(path, dtype={"id": int}, work_dir=tmp_path / "run", config=_config())
    assert pipe._doc_ids == ["1"]


@pytest.mark.parametrize("count", [1, 3])
def test_backend_cardinality_failure_does_not_promote_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    """Reject too few or too many backend results and retain previous outputs.

    Args:
        tmp_path: Isolated run directory.
        monkeypatch: Dependency substitution fixture.
        count: Backend output count for two requested inputs.
    """
    pipe = _prepared(tmp_path, ["a", "b"])
    sentences, ids = pipe.parsed_sents_path.read_bytes(), pipe.parsed_ids_path.read_bytes()

    class Backend:
        """Simulate an incorrectly sized backend response."""

        def process_documents(self, texts: list[str]) -> Iterator[list[list[str]]]:
            """Yield the configured number of results.

            Args:
                texts: Requested input documents.

            Yields:
                Synthetic one-token documents.
            """
            for _ in range(count):
                yield [["changed"]]

    def build(config: Config) -> Backend:
        """Supply the controlled backend.

        Args:
            config: Pipeline configuration.

        Returns:
            A backend with a deliberately wrong result count.
        """
        return Backend()

    monkeypatch.setattr("lmsy_w2v_rfs.pipeline.build_preprocessor", build)
    with pytest.raises(ValueError):
        pipe.parse(force=True)
    assert pipe.parsed_sents_path.read_bytes() == sentences
    assert pipe.parsed_ids_path.read_bytes() == ids
    assert not list((tmp_path / "parsed").glob("*.tmp"))


@pytest.mark.parametrize("invalid", ["innovation", [42], [""], ["  "], []])
@pytest.mark.parametrize("source", ["config", "dict", "json"])
def test_seed_validation_is_consistent(tmp_path: Path, invalid: Any, source: str) -> None:
    """Reject malformed seed values consistently across constructors and loaders.

    Args:
        tmp_path: Temporary seed file directory.
        invalid: Malformed seed-list value.
        source: Validation entry point.
    """
    seeds = {"demo": invalid}
    with pytest.raises(ValueError):
        if source == "config":
            Config(seeds=seeds)
        elif source == "dict":
            load_seeds(seeds)
        else:
            path = tmp_path / "seeds.json"
            path.write_text(json.dumps(seeds))
            load_seeds(path)
