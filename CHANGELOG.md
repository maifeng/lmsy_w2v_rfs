# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-04-28

### Added

- Documentation site at https://maifeng.github.io/lmsy_w2v_rfs/ (gh-pages deploy via GitHub Actions).
- `Documentation` and `Changelog` entries in `[project.urls]` in `pyproject.toml`.
- `load_seeds` now accepts both flat `{dim: [seeds]}` and wrapped `{"seeds": {dim: [seeds]}}` JSON formats; bundled `seeds_culture.json` is now loadable via `load_seeds` directly.
- 2 regression tests for `load_seeds` wrapped format (test count: 59 -> 61).
- `ScoringMethod` re-exported from top-level package (`from lmsy_w2v_rfs import ScoringMethod`).
- `corenlp_timeout_ms` documented in README "All knobs".
- Cross-links from `whiten-scores.md` (was orphan: no inbound links).

### Changed

- README title now theory-agnostic: "Word2Vec dictionary expansion and scoring for any seed-based vocabulary."
- README citation moved from near-top to dedicated bottom section (with one-line "Cite as:" pointer near top).
- README "Large corpora" recipe replaced with link to HPC how-to (was contradicting it).
- `mkdocs.yml` `paths`: `[src/lmsy_w2v_rfs]` -> `[src]` (mkdocstrings could not resolve `lmsy_w2v_rfs.*` identifiers in clean CI builds).
- `mkdocs.yml` `site_description` rewritten as theory-agnostic.
- `mkdocs.yml` `site_url` set for canonical URL generation.
- How-to nav reordered: goal-oriented (Load documents, Use seeds, Switch preprocessor) first; install and troubleshooting later.
- `docs/how-to/aggregate-firm-year.md`: title and nav entry renamed to "Aggregate document scores" (filename kept for back-compat).
- `docs/how-to/troubleshooting.md`: moved from Explanation to How-to (Diataxis-correct).
- `docs/explanation/mwe-comparison.md`: nav label renamed to "Preprocessor comparison" (matches page h1).
- `reference/w2v.md`: expanded from 17-line stub to ~40 lines with descriptions and examples.
- Install steps de-duplicated 3 places -> 1 canonical (`how-to/install-corenlp.md`).
- Preprocessor decision tables consolidated; `mwe-comparison.md` is the deep authoritative source.
- Culture-specific framing replaced with theory-agnostic language across `concepts/scoring.md`, `concepts/word2vec-dictionary.md`, `how-to/aggregate-firm-year.md`, `how-to/load-documents.md`, `reference/pipeline.md`, `reference/scoring.md`.
- `reference/seeds.md` heading style aligned with other reference pages.
- `reference/scoring.md`: `ScoringMethod` directive uses canonical top-level path.
- spaCy no longer called "recommended" in `Config` docstring or `reference/preprocessors.md`; CoreNLP described as the default with neutral language.
- README `Config(...)` snippets in Step 1b and Step 2 now include `seeds=`.
- `__all__` in `__init__.py`: `__version__` and `__paper__` removed (still accessible as module attributes).

### Fixed

- 16 `Config(...)` code blocks across 6 how-to pages omitted required `seeds=` (every example raised `ValueError` on copy-paste).
- `STOPWORDS_SRAF`: type was documented as `frozenset`, actually `set`; count "120" updated to "121" in 6 places.
- Integrity seed list in `concepts/word2vec-dictionary.md`: was 10 entries with 4 not in JSON; now 14 entries matching `seeds_culture.json`.
- `docs/how-to/resume-after-crash.md` stage table: `score` reads `corpora/pass{N}.txt` (when `use_gensim_phrases=True`), not `cleaned/sentences.txt`.
- `docs/how-to/troubleshooting.md`: parameter `memory=` -> `Config(corenlp_memory=)`; default `4 GB` -> `6G`; port `9000` -> `9002`; 3 `Config()` snippets missing `seeds=`.
- Stale `use_corenlp=False` reference in package CLAUDE.md (replaced by `preprocessor="none"` in v0.1.1).
- Dead "Limits" cross-reference in `concepts/scoring.md`.
- CHANGELOG: `a81a0bd` (invalidate stale score CSVs on dictionary curation) moved from `[0.1.2]` to `[0.1.1]` (where it actually shipped).
- `docs/index.md` footer: 3 missing how-to pages added (`run-from-cli.md`, `whiten-scores.md`, `run-on-hpc.md`).

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
- `Pipeline._invalidate_scores()`: clears both the in-memory `_scores` cache
  and score CSVs on disk. Called by `edit_dictionary` and
  `reload_dictionary`. (`a81a0bd`)
- Tests: `test_edit_dictionary_invalidates_score_csvs`,
  `test_reload_dictionary_invalidates_score_csvs`. (`a81a0bd`)

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

[0.1.4]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maifeng/lmsy_w2v_rfs/compare/v0.1.0a1...v0.1.0
[0.1.0a1]: https://github.com/maifeng/lmsy_w2v_rfs/releases/tag/v0.1.0a1
