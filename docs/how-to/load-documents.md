# Load your documents

## Problem

You have a corpus on disk in some format (CSV, directory of txt files, pandas
DataFrame, JSONL, a single big text file) and you want to feed it into
`Pipeline` without writing a loop that reads files and parses them by hand.

## Solution

Use the factory classmethod that matches your input format. Every factory
returns a ready-to-run `Pipeline` with `texts` and `doc_ids` attached. You
then call `p.run()` the same way regardless of how you loaded the data.

All factories produce the same internal representation: `texts` is a
`list[str]` where each string is one document (may contain multiple
sentences, newlines, headers, etc.), and `doc_ids` is a matching
`list[str]` of identifiers for later joins.

---

### A. You already have a list of strings

The simplest case. If you built `texts` and `doc_ids` in your own script or
notebook:

```python
from lmsy_w2v_rfs import Pipeline

texts = [
    "Thank you, operator. Ladies and gentlemen ... innovation ...",
    "Good morning. This quarter our customer ...",
    "We are pleased to report record integrity and trust ...",
]
doc_ids = ["AAPL_2024Q1", "WFC_2024Q1", "TSLA_2024Q1"]

p = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/demo")
p.run()
```

---

### B. CSV file with `id` and `text` columns

The most common research format. Pulling from WRDS, Compustat, or your own
database usually lands here.

```python
# transcripts.csv
# call_id,firm_ticker,transcript
# AAPL_2024Q1,AAPL,"Thank you operator. ..."
# WFC_2024Q1,WFC,"Good morning. ..."

from lmsy_w2v_rfs import Pipeline

p = Pipeline.from_csv(
    "transcripts.csv",
    text_col="transcript",
    id_col="call_id",
    work_dir="runs/demo",
)
p.run()
```

Extra kwargs flow through to `pandas.read_csv` if your file has unusual
settings:

```python
p = Pipeline.from_csv(
    "transcripts.tsv",
    text_col="body", id_col="doc_id",
    work_dir="runs/demo",
    sep="\t", encoding="latin-1",
)
```

---

### C. Pandas DataFrame you already built

Useful when you loaded from Parquet, merged several sources, or filtered
rows before scoring.

```python
import pandas as pd
from lmsy_w2v_rfs import Pipeline

df = pd.read_parquet("transcripts.parquet")
df = df[df["year"] >= 2020]          # filter however you like
df = df[df["transcript"].str.len() > 1000]

p = Pipeline.from_dataframe(
    df, text_col="transcript", id_col="call_id", work_dir="runs/demo",
)
p.run()
```

If you have no explicit ID column, set `id_col=None` and the DataFrame's
row index is used:

```python
p = Pipeline.from_dataframe(df, text_col="transcript", id_col=None, work_dir="runs/x")
```

---

### D. Directory of one-file-per-document text files

The SEC's EDGAR 10-K filings come this way. So do many ad-hoc scraping
pipelines.

```
10k_filings/
├── AAPL_2024.txt
├── MSFT_2024.txt
├── WFC_2024.txt
└── ...
```

```python
p = Pipeline.from_directory(
    "./10k_filings/",
    pattern="*.txt",
    work_dir="runs/sec_10k",
)
p.run()
```

Document IDs come from each file's stem (filename without extension).
Subdirectories: use `pattern="**/*.txt"` for recursive globbing.

---

### E. A single large text file, one document per line

This is the format the 2021 paper ships with its sample corpus. One file
`documents.txt` where each line is a full earnings-call transcript, plus a
parallel `document_ids.txt` with matching IDs.

```python
p = Pipeline.from_text_file(
    "data/input/documents.txt",
    id_path="data/input/document_ids.txt",
    work_dir="runs/rfs2021",
)
p.run()
```

If you only have the texts file and no IDs file, the factory auto-assigns
1-based line numbers as IDs.

---

### F. JSON Lines (JSONL)

One JSON object per line. Common when the data came from an API dump or a
streaming database export.

```
transcripts.jsonl
{"call_id": "AAPL_2024Q1", "firm": "AAPL", "transcript": "Thank you ..."}
{"call_id": "WFC_2024Q1",  "firm": "WFC",  "transcript": "Good morning ..."}
```

```python
p = Pipeline.from_jsonl(
    "transcripts.jsonl",
    text_key="transcript",
    id_key="call_id",
    work_dir="runs/demo",
)
p.run()
```

If your records do not have an explicit ID field, pass `id_key=None` and
1-based line numbers are used.

---

## What counts as "one document"?

Whatever you put in one string. The pipeline's preprocessor handles
sentence segmentation internally. So you can pass:

- a full earnings-call transcript as one string (tens of KB, hundreds of
  sentences),
- a single paragraph (typical for news articles),
- a single sentence (typical for SEC risk-factor sentences pre-chunked).

Document-length normalization (in `firm_year`) divides by token count, so
you can mix very long and very short documents and the scale stays
comparable across them.

---

## Memory cost

All factories load the full corpus into RAM as a Python list. Earnings-call
sample: 1,393 documents at ~30 KB each = ~40 MB. Fine on any laptop. If
your corpus is millions of long documents:

- Shard into 10k-document files.
- Run the pipeline once per shard with a shared `seeds` dictionary.
- Concatenate the resulting scores DataFrames across shards.
- Word2Vec training needs the full corpus once; use a sharded training run
  only after preprocessing all shards.

For small to mid-sized corpora (thousands to tens of thousands of documents),
memory is rarely the issue.
