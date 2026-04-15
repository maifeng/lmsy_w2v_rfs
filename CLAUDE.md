# CLAUDE.md — lmsy_w2v_rfs

Project-local instructions for future Claude sessions. Read before touching any code.

## What this package does

`lmsy_w2v_rfs` is the PyPI packaging of Li, Mai, Shen & Yan (2021),
"Measuring Corporate Culture Using Machine Learning," *RFS* 34(7):3265-3315,
[doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079).

It turns a list of earnings-call transcripts into per-document scores on
five culture dimensions (integrity, teamwork, innovation, respect, quality)
by training a Word2Vec model, expanding seed words with nearest-neighbor
search, and scoring documents with tf-idf variants.

## Public API (stable surface)

```python
from lmsy_w2v_rfs import (
    Pipeline,           # end-to-end orchestrator
    Config,             # frozen dataclass of hyperparameters
    CULTURE_SEEDS,      # 5 dims x 47 seed words
    CULTURE_DIMS,
    STOPWORDS_SRAF,
    download_corenlp,   # optional, needs the [corenlp] extra
    corenlp_server,     # context manager
)
```

`Pipeline` has stage methods `parse / clean / phrase / train / expand_dictionary / score`
and a one-shot `run`. Stages are idempotent: rerunning does not redo work unless
`force=True` is passed or the artifact is missing.

## Two-phase construction, both configurable

This was the explicit ask when building this package:

1. **Phase 1** (Stanford CoreNLP, syntactic multi-word expressions). Toggle with
   `Config.use_corenlp=True`. Default: off, because CoreNLP needs Java and a ~1 GB
   download.
2. **Phase 2** (gensim `Phrases`, corpus-learned bigrams/trigrams). Toggle with
   `Config.use_gensim_phrases=True` and `Config.phrase_passes=N`. Default: on,
   2 passes (bigram + trigram).

The package has to work with Phase 1 off. The offline test suite proves this.

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
- **The plain-parse path.** With `use_corenlp=False`, `_parse_plain` just writes
  each input line as a sentence with a `doc_id_0` ID. The clean stage then
  uses `clean_plain_line` (lowercase, strip punctuation, drop stopwords). This
  is the Java-free happy path.

## Package data

```
src/lmsy_w2v_rfs/data/
├── seeds_culture.json       # 5 dims, 47 seeds, loaded at import
└── stopwords_sraf.txt       # 120-token SRAF list
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

## Not-yet-done

- Colab quickstart `.ipynb` (jupytext source lives in `notebooks/`).
- MkDocs site (low priority for the RFS 2021 package per `plan/02_packages.md`).
- PyPI publish (pending Feng's sign-off + GitHub repo creation).
