"""Tests for load_seeds and the Pipeline.from_* factories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lmsy_w2v_rfs import Config, Pipeline, load_seeds
from lmsy_w2v_rfs.config import CULTURE_SEEDS


# ----- load_seeds -------------------------------------------------------


def test_load_seeds_none_returns_default() -> None:
    seeds = load_seeds(None)
    assert set(seeds.keys()) == set(CULTURE_SEEDS.keys())
    # Copy, not alias
    seeds["integrity"].append("foo")
    assert "foo" not in CULTURE_SEEDS["integrity"]


def test_load_seeds_dict_passthrough() -> None:
    seeds = load_seeds({"risk": ["risk", "hedge"], "growth": ["growth", "scale"]})
    assert list(seeds.keys()) == ["risk", "growth"]
    assert seeds["risk"] == ["risk", "hedge"]


def test_load_seeds_from_json(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps({"risk": ["risk", "uncertainty"], "growth": ["growth", "expand"]}),
        encoding="utf-8",
    )
    seeds = load_seeds(path)
    assert seeds == {"risk": ["risk", "uncertainty"], "growth": ["growth", "expand"]}


def test_load_seeds_from_json_rejects_bad_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="dict of"):
        load_seeds(path)


def test_load_seeds_from_txt(tmp_path: Path) -> None:
    path = tmp_path / "seeds.txt"
    path.write_text(
        "# my culture dimensions\n"
        "\n"
        "integrity: integrity ethic honest\n"
        "teamwork: teamwork, collaborate, cooperate\n"
        "innovation : innovation innovate creative\n",
        encoding="utf-8",
    )
    seeds = load_seeds(path)
    assert seeds == {
        "integrity": ["integrity", "ethic", "honest"],
        "teamwork": ["teamwork", "collaborate", "cooperate"],
        "innovation": ["innovation", "innovate", "creative"],
    }


def test_load_seeds_from_txt_rejects_missing_colon(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("integrity integrity ethic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_seeds(path)


def test_load_seeds_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_seeds("/nonexistent/seeds.json")


# ----- Pipeline factories -----------------------------------------------


def _cfg() -> Config:
    return Config(
        preprocessor="none", mwe_list=None,
        w2v_dim=10, w2v_epochs=2, w2v_min_count=1,
        phrase_min_count=2, phrase_threshold=1.0,
        n_words_dim=5, n_cores=1,
    )


def test_from_dataframe_with_id_col(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "doc_id": ["a", "b"],
        "text": ["innovation and creativity", "respect and dignity"],
    })
    p = Pipeline.from_dataframe(df, text_col="text", id_col="doc_id",
                                work_dir=tmp_path / "run", config=_cfg())
    assert p._texts == ["innovation and creativity", "respect and dignity"]
    assert p._doc_ids == ["a", "b"]


def test_from_dataframe_with_index(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["a b c", "d e f"]}, index=[101, 102])
    p = Pipeline.from_dataframe(df, text_col="text", id_col=None,
                                work_dir=tmp_path / "run", config=_cfg())
    assert p._doc_ids == ["101", "102"]


def test_from_csv(tmp_path: Path) -> None:
    csv = tmp_path / "docs.csv"
    pd.DataFrame({"id": ["x", "y"], "text": ["alpha", "beta"]}).to_csv(csv, index=False)
    p = Pipeline.from_csv(csv, work_dir=tmp_path / "run", config=_cfg())
    assert p._doc_ids == ["x", "y"]
    assert p._texts == ["alpha", "beta"]


def test_from_directory(tmp_path: Path) -> None:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "AAPL_2024.txt").write_text("apple revenue grew", encoding="utf-8")
    (d / "MSFT_2024.txt").write_text("microsoft revenue flat", encoding="utf-8")
    p = Pipeline.from_directory(d, work_dir=tmp_path / "run", config=_cfg())
    assert set(p._doc_ids) == {"AAPL_2024", "MSFT_2024"}


def test_from_jsonl(tmp_path: Path) -> None:
    jp = tmp_path / "docs.jsonl"
    jp.write_text(
        '{"id": "d1", "text": "alpha beta"}\n'
        '{"id": "d2", "text": "gamma delta"}\n',
        encoding="utf-8",
    )
    p = Pipeline.from_jsonl(jp, work_dir=tmp_path / "run", config=_cfg())
    assert p._doc_ids == ["d1", "d2"]
    assert p._texts == ["alpha beta", "gamma delta"]
