# CLAUDE.md — lmsy_w2v_rfs

Project-local instructions for future Claude sessions. Read before touching any code.

## What this package does

`lmsy_w2v_rfs` is a generic word-embedding seed-expansion text scorer.
You bring a corpus and a seed-word dictionary (one short list per
concept); it trains Word2Vec on the corpus, expands each seed list via
nearest-neighbor search, and scores every document with TF / TF-IDF /
WFIDF variants.

Originally a port of Li, Mai, Shen, Yan (2021), "Measuring Corporate
Culture Using Machine Learning," *RFS* 34(7):3265-3315,
[doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079). The
package itself is **theory-agnostic** — `Config(seeds=...)` is required,
no defaults. The 2021 paper's 5-dim culture dictionary is shipped only
as a named example via `load_example_seeds("culture_2021")`.

## Public API (stable surface)

```python
from lmsy_w2v_rfs import (
    Pipeline,            # end-to-end orchestrator
    Config,              # frozen dataclass of hyperparameters; seeds= required
    load_seeds,          # read dict / .json / .txt -> dict[str, list[str]]
    load_example_seeds,  # opt-in named examples ("culture_2021")
    STOPWORDS_SRAF,      # 121-token SRAF generic stopword list
    download_corenlp,    # optional, needs the [corenlp] extra
)
```

`Pipeline` has stage methods `parse / clean / phrase / train / expand_dictionary / score`
and a one-shot `run`. Stages are idempotent: rerunning does not redo work unless
`force=True` is passed or the artifact is missing.

Inspection + manual curation methods (between `expand_dictionary` and `score`):

- `Pipeline.show_dictionary(top_k=10)` — pretty-print per-dim seeds + expansion.
- `Pipeline.dictionary_preview(top_k=10)` — DataFrame view for notebook display.
- `Pipeline.edit_dictionary(remove=, add=)` — programmatic curation; updates both in-memory dict and CSV.
- `Pipeline.reload_dictionary()` — reread CSV after editing it in a spreadsheet.

The 2021 paper's authors did manual curation (mostly removals) before scoring; both methods support that workflow.

## Two-phase construction, both configurable

This was the explicit ask when building this package:

1. **Phase 1** (preprocessor-based MWE joining + NER masking). Select with
   `Config(preprocessor=...)`. Options: `"corenlp"` (default, paper-exact, needs
   Java), `"spacy"` (fastest, Python-native), `"stanza"`, `"static"` (curated
   list only), `"none"` (skip Phase 1).
2. **Phase 2** (gensim `Phrases`, corpus-learned bigrams/trigrams). Toggle with
   `Config.use_gensim_phrases=True` and `Config.phrase_passes=N`. Default: on,
   2 passes (bigram + trigram).

The package has to work with Phase 1 off (`preprocessor="none"`). The offline
test suite proves this.

## CoreNLP design decisions

- `download_corenlp()` calls `stanza.install_corenlp()`. Caches under
  `~/.cache/lmsy_w2v_rfs/corenlp/` by default (override via
  `$LMSY_W2V_RFS_HOME` env var).
- `corenlp_server()` is a context manager that starts one JVM and yields the
  endpoint. Inside the block, `parse_documents` spawns Python workers that
  attach to the server with `start_server=False`. This matches the original
  `parse_parallel.py` pattern: one JVM, many client processes, parse work is
  amortized over the JVM warm-up cost.
- All CoreNLP imports live in `lmsy_w2v_rfs.corenlp` and are guarded by a
  lazy `_import_stanza()` that raises a clear `ImportError` with install
  instructions. The base install does not pull `stanza` or `protobuf`.

## Design decisions worth remembering

- **One `Config` dataclass.** Frozen, with a `with_(...)` helper. No scattered
  globals (the original `global_options.py` mutated `os.environ` at import).
- **Seeds are package data**, loaded via `importlib.resources`. The stopword list
  is also package data.
- **Streaming scoring.** `iter_doc_level_corpus` and `document_frequencies`
  yield one document at a time. The original pickled the whole corpus to
  `corpus_doc_level.pickle`; we never materialize that.
- **gensim 4 throughout.** `vector_size=` / `epochs=` on Word2Vec,
  `connector_words=` on Phrases, `model.wv.key_to_index` for vocab membership.
- **The plain-parse path.** With `preprocessor="none"`, `_parse_plain` just writes
  each input line as a sentence with a `doc_id_0` ID. The clean stage then
  uses `clean_plain_line` (lowercase, strip punctuation, drop stopwords). This
  is the Java-free happy path.

## Package data

```
src/lmsy_w2v_rfs/data/
├── seeds_culture.json       # 5 dims, 47 seeds, opt-in via load_example_seeds
└── stopwords_sraf.txt       # 121-token SRAF list
```

Both are shipped via `[tool.setuptools.package-data]` in `pyproject.toml`.

## How to run tests

```bash
pip install -e ".[dev]"
pytest                                  # offline, no Java, ~10-20 s
pytest -m corenlp                       # requires Java + [corenlp] extra
```

## Writing conventions

Inherited from the parent project `CLAUDE.md`:

- Python 3.10+ with `from __future__ import annotations`.
- Google-style docstrings on every public function/class, type hints on every
  signature.
- No em-dashes in prose or docstrings. Colons, commas, periods only.
- No emojis.
- No "X, not Y" rhetorical constructions.
- Files under ~300 lines where possible.
- Pin deps in `pyproject.toml` with both lower and upper bounds.

## Current status (as of v0.1.4)

- Colab notebook: `notebooks/01_quickstart_colab.ipynb` ships in the repo (force-tracked). Jupytext source is `notebooks/01_quickstart.py`.
- MkDocs site: 23 substantive pages deployed; `mkdocs build --strict` passes. Docs live at https://maifeng.github.io/lmsy_w2v_rfs/ (gh-pages, deployed automatically).
- PyPI: published at `lmsy-w2v-rfs` (latest: v0.1.4). Install with `pip install lmsy_w2v_rfs`.
- Tests: 61 passing, 1 skipped (CoreNLP/Java).
