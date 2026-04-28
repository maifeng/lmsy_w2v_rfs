# Switch the Phase 1a preprocessor

## Problem

The pipeline has five Phase 1a backends and the right choice depends on three
things: whether Java is available, whether you need named-entity masking, and
whether syntactic multi-word expressions matter for your concept. Picking the
wrong one silently degrades results. CoreNLP is paper-faithful but slow and
Java-bound. spaCy is fast and Python-only but drops every UD `fixed` and
`compound:prt` pattern. Stanza is Python-native but 5x slower on CPU. Static
and none bypass the parser entirely.

## Solution

Set `Config(preprocessor=...)` to one of `"corenlp"`, `"spacy"`, `"stanza"`,
`"static"`, `"none"`. The decision table below tells you which one to pick.

### Decision table

| Your situation | Recommended preprocessor |
|---|---|
| Java available, want paper-faithful results | `"corenlp"` (default) |
| No Java, want speed, NER matters | `"spacy"` |
| No Java, want modern UD parser, ~5 hours is fine | `"stanza"` |
| No parser dependencies, have a curated MWE list | `"static"` |
| Already-lemmatized input, whitespace tokenize only | `"none"` |

### Syntactic MWE recall (from the 60-phrase benchmark)

| Backend | Fixed phrases (13) | Phrasal verbs (8) | Syntactic total (21) | Compound nouns (10) |
|---|---|---|---|---|
| `"corenlp"` | 8/13 | 8/8 | **16/21 (76%)** | 6/10 |
| `"stanza"` | 4/13 | 8/8 | 12/21 (57%) | 7/10 |
| `"spacy"` | 0/13 | 0/8 | 0/21 (0%) | 7/10 |
| `"static"` | depends on your list | depends | depends | depends |
| `"none"` | 0 | 0 | 0 | 0 |

spaCy's English model does not emit UD `fixed` or `compound:prt` at all. Its
strengths are NER (96% type accuracy) and compound nouns. If your concept
relies on phrases like `with_respect_to`, `as_well_as`, `in_addition_to`,
`roll_out`, or `pay_off`, CoreNLP or stanza is the right call.

### Config snippets per backend

**CoreNLP (default, paper-faithful):**

```python
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds

seeds = load_example_seeds("culture_2021")
cfg = Config(
    seeds=seeds,
    preprocessor="corenlp",
    n_cores=8,                   # JVM thread pool size
    corenlp_memory="6G",
    corenlp_port=9002,
)
```

Needs `pip install "lmsy_w2v_rfs[corenlp]"` and
`lmsy-w2v-rfs download-corenlp`. See [Install the CoreNLP backend](install-corenlp.md).

**spaCy (fastest, no Java):**

```python
cfg = Config(
    seeds=seeds,
    preprocessor="spacy",
    spacy_model="en_core_web_sm",   # or "_md" / "_trf"
    n_cores=8,                       # Python process count
)
```

Needs `pip install "lmsy_w2v_rfs[spacy]"` and
`python -m spacy download en_core_web_sm`. Runtime: ~4 min on 1,393 earnings
transcripts at `n_cores=8`.

**Stanza (Python-native, slow on CPU):**

```python
cfg = Config(
    seeds=seeds,
    preprocessor="stanza",
    n_cores=4,
)
```

Needs `pip install "lmsy_w2v_rfs[stanza]"`. First run auto-downloads the
English UD model. Expect ~5 hours for 1,393 docs on CPU; GPU is not yet
supported on Apple Silicon and is optional on CUDA.

**Static (no parser, deterministic):**

```python
cfg = Config(
    seeds={"integrity": ["integrity", "ethic"]},
    preprocessor="static",
    mwe_list="finance",              # packaged earnings-call list
)
```

```python
# or with your own list
cfg = Config(
    seeds={"integrity": ["integrity", "ethic"]},
    preprocessor="static",
    mwe_list="path/to/my_mwes.txt",   # one MWE per line, space-separated tokens
)
```

Zero-ML; NLTK's `MWETokenizer` replaces each match with an underscored token.
No lemmatization, no NER masking.

**None (pure whitespace split):**

```python
cfg = Config(seeds={"integrity": ["integrity", "ethic"]}, preprocessor="none")
```

For pre-lemmatized corpora or when you want gensim `Phrases` (Phase 2) to do
all MWE work alone.

### Applying a second-pass static MWE list

`mwe_list=` is an optional post-pass that runs after the main preprocessor.
It is independent of `preprocessor=` (except for `"static"`, where the list
IS the preprocessor).

```python
cfg = Config(
    seeds=seeds,
    preprocessor="spacy",
    mwe_list="finance",              # rescue MWEs spaCy's UD converter drops
)
```

The packaged `"finance"` list is an earnings-call example, not a default.
Pass a path to your own list for any other domain.

## Gotchas

- Switching preprocessors after a run has already produced `work_dir/parsed/`
  does NOT trigger a re-parse. The stage detects existing output and reuses it.
  Delete `work_dir/parsed/` or pass `force=True` to redo Phase 1a. See
  [Resume after a crash](resume-after-crash.md).
- `n_cores` means "JVM threads" for CoreNLP and "Python worker processes" for
  spaCy and stanza. On macOS, Python multiprocessing defaults to `spawn`,
  which reloads the spaCy model per worker. See [Run on HPC](run-on-hpc.md).
- The `corenlp` and `stanza` backends both emit UD v2 labels, but on different
  models trained on different data. They disagree on roughly a third of `fixed`
  patterns. CoreNLP's PTB-to-UD converter memorizes more of them by rule.

## Related

- [Install the CoreNLP backend](install-corenlp.md)
- [Run on HPC](run-on-hpc.md)
- [Use your own seeds](use-your-own-seeds.md)
