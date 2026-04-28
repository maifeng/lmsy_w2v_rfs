# Command-line interface

The package ships a `lmsy-w2v-rfs` console script with two subcommands.

- [`run`](#run): end-to-end pipeline from documents on disk to scores on disk.
- [`download-corenlp`](#download-corenlp): one-time Stanford CoreNLP install.

For an end-to-end walkthrough, see [Run from the command line](../how-to/run-from-cli.md).

---

## `run`

Reads documents from `--input`, parses + cleans + trains Word2Vec + expands
the seed dictionary + scores every document on every dimension, and writes
all artifacts under `--out`.

Exit code 0 on success. 1 on argparse usage errors. Python exceptions bubble
up with their native traceback and non-zero code.

### Input flags

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--input`, `-i` | path | required | Input path; interpretation set by `--input-format`. |
| `--input-format` | choice | `text` | One of `text`, `csv`, `jsonl`, `directory`. |
| `--ids` | path | `None` | [text] Optional matching IDs file, one ID per line. If omitted, line numbers are used. |
| `--text-col` | str | `text` | [csv] Column holding document text. |
| `--id-col` | str | `id` | [csv] Column holding document IDs. Set to `""` to use the DataFrame row index. |
| `--text-key` | str | `text` | [jsonl] JSON key holding document text. |
| `--id-key` | str | `id` | [jsonl] JSON key holding document IDs. Set to `""` to use 1-based line numbers. |
| `--glob-pattern` | str | `*.txt` | [directory] Glob for files under `--input`. Document IDs come from file stems. |

### Output flags

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--out`, `-o` | path | `runs/out` | Output directory. Stages are idempotent: rerunning skips stages whose outputs already exist. |
| `--force` | flag | off | Rerun every stage regardless of existing outputs. |

### Seeds flag

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--seeds` | path | (required) | Seed dictionary file (`.json` or `.txt`). The package is theory-agnostic and has no built-in default. To reproduce the 2021 paper, dump `load_example_seeds("culture_2021")` to a JSON file. See [Use your own seed dictionary](../how-to/use-your-own-seeds.md). |

### Phase 1 (preprocessor) flags

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--preprocessor` | choice | `corenlp` | `corenlp` (paper-faithful, needs Java) / `spacy` (fastest) / `stanza` (Python-native) / `static` (list-only) / `none`. |
| `--mwe-list` | str | `none` | Optional static MWE list post-pass. `none` skips. `finance` uses the packaged earnings-call example. Path loads a custom list. |
| `--spacy-model` | str | `en_core_web_sm` | spaCy model name when `--preprocessor=spacy`. `en_core_web_trf` is the best-NER slower option. |
| `--n-cores` | int | `4` | JVM threads for CoreNLP, `n_process` for spaCy / stanza. 4 is safe on an 8-core laptop; 8 on a workstation. |

### Phase 2 (gensim Phrases) flags

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--no-phrases` | flag | off | Skip the gensim `Phrases` pass entirely. |
| `--phrase-passes` | int | `2` | 1 = bigram only. 2 = bigram then trigram. |
| `--phrase-min-count` | int | `10` | Minimum bigram count. Lower on small corpora. |
| `--phrase-threshold` | float | `10.0` | gensim `Phrases` score threshold. |

### Word2Vec flags

| Flag | Type | Default |
| --- | --- | --- |
| `--w2v-dim` | int | `300` |
| `--w2v-window` | int | `5` |
| `--w2v-min-count` | int | `5` |
| `--w2v-epochs` | int | `20` |

### Dictionary and scoring flags

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--n-words-dim` | int | `500` | Top-k expanded words per dimension. |
| `--methods` | list | `TF TFIDF WFIDF` | Any subset of `TF`, `TFIDF`, `WFIDF`, `TFIDF+SIMWEIGHT`, `WFIDF+SIMWEIGHT`. |
| `--zca-whiten` | flag | off | Decorrelate dimension columns. See [Whiten the dimension scores](../how-to/whiten-scores.md). |

### Output layout

After a successful `run`, `--out` contains:

```
runs/out/
├── config.json              # frozen snapshot of the Config used
├── parsed/
│   ├── sentences.txt        # preprocessor output, one sentence per line
│   └── sentence_ids.txt     # parallel IDs, one per sentence
├── cleaned/
│   └── sentences.txt        # stopwords and punctuation dropped
├── corpora/                 # only if gensim Phrases is enabled
│   ├── pass1.txt            # bigram-joined sentences
│   └── pass2.txt            # trigram-joined sentences
├── models/
│   ├── phrases_pass1.mod    # saved gensim Phrases models
│   ├── phrases_pass2.mod
│   └── w2v.mod              # saved Word2Vec model
└── outputs/
    ├── expanded_dict.csv    # one column per dimension, ranked words
    ├── scores_TF.csv        # document-level scores, one file per method
    ├── scores_TFIDF.csv
    └── scores_WFIDF.csv
```

Every `scores_*.csv` has columns: `Doc_ID`, one column per dimension
(alphabetical), `document_length`.

### Common invocations

**Quickest: use a CSV with paper defaults.**

```bash
lmsy-w2v-rfs run --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id --out runs/rfs2021
```

**Paper-faithful CoreNLP, 8 threads, all three scoring methods:**

```bash
lmsy-w2v-rfs download-corenlp             # one time
lmsy-w2v-rfs run \
  --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --out runs/corenlp8 \
  --preprocessor corenlp --n-cores 8
```

**Fast spaCy path with custom seeds:**

```bash
lmsy-w2v-rfs run \
  --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --seeds my_seeds.txt \
  --preprocessor spacy --spacy-model en_core_web_sm --n-cores 8 \
  --out runs/spacy_custom
```

**Directory of SEC filings, static MWE list post-pass, ZCA whitening:**

```bash
lmsy-w2v-rfs run \
  --input ./10k_filings/ --input-format directory --glob-pattern "*.txt" \
  --mwe-list finance --zca-whiten \
  --out runs/10k
```

**JSON Lines input, no parser, rely on gensim Phrases only:**

```bash
lmsy-w2v-rfs run \
  --input dump.jsonl --input-format jsonl \
  --text-key body --id-key id \
  --preprocessor none --phrase-min-count 5 \
  --out runs/jsonl_demo
```

---

## `download-corenlp`

Installs Stanford CoreNLP into `$LMSY_W2V_RFS_HOME/corenlp` (or
`~/.cache/lmsy_w2v_rfs/corenlp` if the env var is unset). Delegates to
`stanza.install_corenlp`. Prints the install path on completion.

Required once before the first `--preprocessor corenlp` run. Takes no
flags. Requires network access and ~1 GB free disk.

```bash
lmsy-w2v-rfs download-corenlp
# CoreNLP installed at: /Users/you/.cache/lmsy_w2v_rfs/corenlp
```

---

## Getting help

Every subcommand prints its own help with `--help`:

```bash
lmsy-w2v-rfs --help
lmsy-w2v-rfs run --help
lmsy-w2v-rfs download-corenlp --help
```

The grouped output (input / output / seeds / phase 1 / phase 2 / word2vec /
scoring) is the same structure used above.
