# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-04-28

### Added

- `docs/reference/seeds.md`: reference page for `load_seeds`,
  `load_example_seeds`, `STOPWORDS_SRAF`, and `download_corenlp`. Added to
  MkDocs nav under Reference.
- `CHANGELOG.md`: this file, covering v0.1.0a1 through v0.1.3.

### Changed

- `mkdocs.yml` `paths`: changed from `src/lmsy_w2v_rfs` to `src` so
  mkdocstrings/griffe can resolve `lmsy_w2v_rfs.*` identifiers correctly
  (it expects a package directory named `lmsy_w2v_rfs` inside the path).
- `mkdocs.yml` `site_description`: rewritten as theory-agnostic: "Word2Vec
  dictionary expansion and scoring for any seed-based vocabulary."
- `docs/reference/scoring.md`: appended `zca_whiten` reference entry, which
  was exported in `__all__` and used in the how-to guide but had no reference
  page.
- `CLAUDE.md`: replaced stale "Not-yet-done" section with accurate "Current
  status (as of v0.1.3)" section reflecting the shipped Colab notebook, full
  MkDocs site, and PyPI publication.

## [0.1.2] - 2026-04-28

### Fixed

- `pyproject.toml`: corrected `Repository` URL to
  `github.com/maifeng/lmsy_w2v_rfs` (v0.1.1 shipped with a wrong URL).
  (`0d25001`)
- `CITATION.cff`: updated URL, version, and release date. (`0d25001`)

### Added

- README badges: Open in Colab, PyPI version, License. (`0d25001`)
- `notebooks/01_quickstart_colab.ipynb`: Colab-ready notebook, force-tracked
  in git. (`75e8990`)
- `notebooks/data/glassdoor_culture_2000.csv`: 2,000-review workshop corpus
  shipped in the repo for Colab access. (`75e8990`)
- `Pipeline._invalidate_scores()`: clears both the in-memory `_scores` cache
  and score CSVs on disk. Called by `edit_dictionary` and
  `reload_dictionary`. (`a81a0bd`)
- Tests: `test_edit_dictionary_invalidates_score_csvs`,
  `test_reload_dictionary_invalidates_score_csvs`. (`a81a0bd`)

## [0.1.1] - 2026-04-15

### Fixed

- `Pipeline.score_df`: merge now detects `Doc_ID` (capitalized) column name
  dynamically instead of assuming `doc_id`. (`9790f42`)
- `Config`: `preprocessor="none"` replaces the removed `use_corenlp` boolean
  flag. Updated notebook and CLAUDE.md to reflect the correct field.
  (`ff71244`)

### Added

- Non-zero score rate reporting: prints the fraction of documents with a
  non-zero score per dimension, helping users understand sparsity in short
  corpora. (`9790f42`)

## [0.1.0] - 2026-04-27

### Added

- Theory-agnostic API: `Config(seeds=...)` is now required with no default.
  The 2021 culture dictionary is available via
  `load_example_seeds("culture_2021")`. (`0f51edc`)
- Built-in dictionary curation methods on `Pipeline`:
  `show_dictionary`, `dictionary_preview`, `edit_dictionary`,
  `reload_dictionary`. (`0f51edc`)
- Unified Glassdoor demo corpus (2,000 reviews) wired into the quickstart
  notebook. (`9c9b460`)

### Changed

- `CULTURE_SEEDS` and `CULTURE_DIMS` removed from the public API. (`0f51edc`)
- CLI `--seeds` is now required. (`0f51edc`)
- README, MkDocs pages, and CLAUDE.md reframed to reflect the theory-agnostic
  scope. (`0f51edc`)

## [0.1.0a1] - 2026-04-15

### Added

- Initial package release: frozen `Config` dataclass, idempotent `Pipeline`
  stages (`parse`, `clean`, `phrase`, `train`, `expand_dictionary`, `score`),
  streaming scoring, gensim 4 throughout, Java-free default parse path,
  optional `[corenlp]` extra, 7 test files, MkDocs site scaffold.
  (`1019bfb`)

[0.1.3]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.0a1...v0.1.0
[0.1.0a1]: https://github.com/maifeng/lmsy_w2v_rfs/releases/tag/v0.1.0a1
