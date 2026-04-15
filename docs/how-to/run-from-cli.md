# Run from the command line

## Problem

You want to run the full pipeline without writing any Python. You have a
CSV file of documents and a text file of seed words. You want culture
scores out.

## Solution

`lmsy-w2v-rfs` is a single command that takes documents and seeds in,
writes scores and an expanded dictionary out. This page walks through a
complete run from nothing to a `scores_TFIDF.csv` you can open in Excel.

### 1. Install

```bash
pip install "lmsy_w2v_rfs[corenlp]"        # or [spacy] if you cannot use Java
lmsy-w2v-rfs download-corenlp              # one-time ~1 GB download, skip if using spaCy
```

Verify:

```bash
lmsy-w2v-rfs --help
# usage: lmsy-w2v-rfs [-h] {run,download-corenlp} ...
```

### 2. Prepare your input

Suppose you have `transcripts.csv` with two columns:

```csv
call_id,transcript
AAPL_2024Q1,"Thank you, operator. Ladies and gentlemen..."
MSFT_2024Q1,"Good morning. This quarter our customer..."
WFC_2024Q1,"We are pleased to report record integrity..."
```

Supported input formats are covered in the [Load your documents](load-documents.md)
how-to: CSV, JSONL, directory of files, one-document-per-line text.

### 3. Prepare your seed dictionary (optional)

Omit this and the CLI uses the 2021 paper's five culture dimensions. If
you want your own concepts, write a plain text file (easiest for hand
editing):

```
# my_seeds.txt
# one dimension per line; words separated by whitespace or commas
# lines starting with # are ignored

integrity:  integrity ethic honest accountable trust transparent
quality:    quality customer dedicated reliable excellence
innovation: innovation innovate creative pioneer breakthrough
```

Or JSON, if you prefer structured formats:

```json
// my_seeds.json
{
  "integrity":  ["integrity", "ethic", "honest", "accountable", "trust"],
  "quality":    ["quality", "customer", "dedicated", "reliable"],
  "innovation": ["innovation", "innovate", "creative", "pioneer"]
}
```

### 4. Run

```bash
lmsy-w2v-rfs run \
  --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --seeds my_seeds.txt \
  --out runs/my_experiment \
  --n-cores 8
```

The command will:

1. Read every row of the CSV.
2. Parse each transcript with CoreNLP (8 JVM threads).
3. Mask named entities, join multi-word expressions.
4. Learn bigrams and trigrams with gensim `Phrases`.
5. Train a 300-dim Word2Vec model on the corpus.
6. Expand each seed list to 500 words via nearest-neighbor search.
7. Score every document on each dimension under TF, TFIDF, and WFIDF.

On a 1,393-document earnings-call corpus at `--n-cores 8`, total runtime
is ~13 minutes on an Apple Silicon M-series laptop.

### 5. Look at the outputs

The run prints a summary at the end:

```
Done. Outputs under: runs/my_experiment
  scores:     runs/my_experiment/outputs/scores_*.csv
  dictionary: runs/my_experiment/outputs/expanded_dict.csv
  w2v model:  runs/my_experiment/models/w2v.mod
```

Open the expanded dictionary in Excel or a text editor:

```csv
# runs/my_experiment/outputs/expanded_dict.csv
integrity,quality,innovation
integrity,customer,innovation
ethic,quality,creative
honest,dedicated,innovate
accountable,customer_experience,pioneer
trust,customer_service,breakthrough
transparent,reliable,inventive
...
```

Each column is one dimension; rows are words ranked by similarity to the
seed mean. Seeds appear at the top; their nearest neighbors follow.

Open the scores CSV:

```csv
# runs/my_experiment/outputs/scores_TFIDF.csv
Doc_ID,innovation,integrity,quality,document_length
AAPL_2024Q1,82.3,5.4,11.1,15240
MSFT_2024Q1,34.2,17.8,42.6,12105
WFC_2024Q1,4.1,73.6,18.2,13980
```

### 6. Resume or rerun

Every stage is idempotent. Run the same command again and the pipeline
skips completed stages. To redo a specific stage, delete its output and
rerun; to redo everything, pass `--force`:

```bash
lmsy-w2v-rfs run --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --seeds my_seeds.txt \
  --out runs/my_experiment --n-cores 8 \
  --force
```

## Different input formats

### JSONL

```bash
lmsy-w2v-rfs run \
  --input dump.jsonl --input-format jsonl \
  --text-key body --id-key doc_id \
  --out runs/x
```

### Directory of text files

```bash
lmsy-w2v-rfs run \
  --input ./10k_filings/ --input-format directory --glob-pattern "*.txt" \
  --out runs/x
```

Each file's stem becomes the doc ID (for example, `AAPL_2024.txt` → `AAPL_2024`).

### Plain text, one document per line

The format the 2021 paper ships with its sample corpus:

```bash
lmsy-w2v-rfs run \
  --input documents.txt --ids document_ids.txt \
  --out runs/x
```

No `--input-format` flag needed; this is the default.

## Different preprocessor backends

### spaCy (fastest, no Java)

```bash
pip install "lmsy_w2v_rfs[spacy]"
python -m spacy download en_core_web_sm

lmsy-w2v-rfs run \
  --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --preprocessor spacy --spacy-model en_core_web_sm \
  --out runs/x --n-cores 8
```

~4 min for the 1,393-doc sample; loses syntactic MWE coverage vs CoreNLP.

### No parser, just statistics

For a rough-and-ready run without installing any parser:

```bash
lmsy-w2v-rfs run --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --preprocessor none \
  --phrase-min-count 5 --phrase-threshold 5.0 \
  --out runs/x
```

Only gensim `Phrases` handles MWE discovery; no lemmatization, no NER
masking. Fast, but less accurate on small corpora.

## Getting help

Each subcommand prints its own help with `--help`:

```bash
lmsy-w2v-rfs run --help
```

## Related

- [Load your documents](load-documents.md) - all supported input formats.
- [Use your own seed dictionary](use-your-own-seeds.md) - the seeds file
  format.
- [Run on an HPC compute node](run-on-hpc.md) - SLURM and SGE templates,
  thread oversubscription fix.
- [Reference: CLI flags](../reference/cli.md) - every flag with types and
  defaults.
