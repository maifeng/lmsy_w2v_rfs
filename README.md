# lmsy_w2v_rfs

**Word-embedding seed expansion and document scoring for text analysis.**

You bring a corpus and a seed-word dictionary, one short list per concept you want to measure. The package trains Word2Vec on your corpus, expands each seed list via nearest-neighbor search in the embedding space, and scores every document with TF / TF-IDF / WFIDF variants.

Originally a port of Li, Mai, Shen, Yan (2021), *RFS* corporate-culture method. The package itself is theory-agnostic: any concept that fits "a list of words" works.

---

## The key pipeline

Five stages; you run them with one call, then optionally inspect and curate the expanded dictionary before scoring.

### 1. Define your seeds

The only place you tell the package what to measure. Any mapping of dimension name to a list of seed words works. Three equivalent ways:

```python
# (a) Python dict
seeds = {
    "risk":   ["risk", "uncertainty", "volatility", "downside"],
    "growth": ["growth", "expansion", "scale", "opportunity"],
}

# (b) JSON file
from lmsy_w2v_rfs import load_seeds
seeds = load_seeds("my_seeds.json")

# (c) Plain text file: one dim per line, "name: word1 word2 ..."
seeds = load_seeds("my_seeds.txt")
```

To reproduce the 2021 paper's 5-dim culture dictionary:

```python
from lmsy_w2v_rfs import load_example_seeds
seeds = load_example_seeds("culture_2021")
```

### 2. Run the pipeline

```python
from lmsy_w2v_rfs import Pipeline, Config

p = Pipeline.from_csv(
    "transcripts.csv",
    text_col="text", id_col="call_id",
    work_dir="runs/my_experiment",
    config=Config(seeds=seeds, preprocessor="none"),  # "none" = no Java required
)
p.run()
```

`run()` parses, cleans, learns phrases, trains Word2Vec, expands the seed dictionary, and scores every document. Other input formats (DataFrame, directory of files, JSONL, in-memory list) are listed [below](#how-do-i-load-my-documents).

### 3. Inspect the expanded dictionary

`show_dictionary` prints seeds and the top-K expanded words per dimension. `dictionary_preview` returns the same content as a DataFrame for notebook display.

```python
p.show_dictionary(top_k=10)
p.dictionary_preview(top_k=10)   # DataFrame
```

The expansion is what you trust the pipeline for. Without it, this is just a keyword counter.

### 4. Curate (optional, recommended)

Word2Vec expansion picks up corpus-specific terms but also surfaces noise. The 2021 paper's authors manually inspected and edited the dictionary before scoring. Two paths:

**Programmatic** (replicable, in notebook):

```python
p.edit_dictionary(
    remove={"risk": ["fantastic", "build"]},
    add={"risk": ["liability"]},
)
```

**Spreadsheet** (faster for big dicts): open `p.dict_path` (a CSV) in Excel, edit, save, then call `p.reload_dictionary()`.

Both paths update both the in-memory dict and the on-disk CSV. Cached scores are dropped automatically; rerun `p.score()` to rescore against the curated dictionary.

### 5. Read the scores

```python
p.score_df("TFIDF")                  # one row per document
```

| Doc_ID | risk | growth | document_length |
|---|---|---|---|
| doc_1 | 0.05 | 0.82 | 15,240 |
| doc_2 | 0.73 | 0.04 | 12,105 |

---

## How do I load my documents?

Pick whichever matches what you have on disk:

```python
Pipeline(texts=[...], doc_ids=[...], work_dir=..., config=cfg)        # in-memory list
Pipeline.from_csv("transcripts.csv", text_col="text", id_col="id", ...) # CSV
Pipeline.from_dataframe(df, text_col="text", id_col="id", ...)          # DataFrame
Pipeline.from_directory("./docs/", pattern="*.txt", ...)                # one file per doc
Pipeline.from_text_file("docs.txt", id_path="ids.txt", ...)             # one doc per line
Pipeline.from_jsonl("transcripts.jsonl", text_key="t", id_key="i", ...) # JSONL
```

CLI: `lmsy-w2v-rfs run --seeds my_seeds.txt --input docs.csv --input-format csv ...`

---

## Install

The default preprocessor is `corenlp` (paper-faithful, needs Java):

```bash
pip install "lmsy_w2v_rfs[corenlp]"
lmsy-w2v-rfs download-corenlp           # one-time ~1 GB CoreNLP archive
```

Java-free alternatives:

```bash
pip install "lmsy_w2v_rfs[spacy]"       # spaCy: fastest, no Java
python -m spacy download en_core_web_sm
pip install "lmsy_w2v_rfs[stanza]"      # Python-native parser
pip install lmsy_w2v_rfs                # bare install: use preprocessor="static" or "none"
```

Then `Config(seeds=..., preprocessor="spacy")` (or `"stanza"` / `"static"` / `"none"`).

---

## Two construction phases

Both on by default; both configurable.

| Phase | What it does | Default |
|---|---|---|
| 1 | Parser-based lemmatization, NER masking, multi-word-expression (MWE) joining (e.g., `interest_rate`) (`preprocessor=`) | `"corenlp"` |
| 2 | gensim `Phrases` learns corpus-specific bigrams and trigrams | `True`, 2 passes |

Preprocessor options: `"corenlp"` (paper-exact, Java), `"spacy"` (fastest, no Java), `"stanza"` (Python-native), `"static"` (curated list only), `"none"` (whitespace tokenize only).

Full `Config` knob list and benchmark notes: see [docs/](docs/).

---

## Scoring methods

| Method | Weight per dictionary hit |
|---|---|
| `TF` | `tf` |
| `TFIDF` | `tf · log(N/df)` |
| `WFIDF` | `(1 + log tf) · log(N/df)` |
| `TFIDF+SIMWEIGHT`, `WFIDF+SIMWEIGHT` | × `1/ln(2 + rank)` |

---

## Citation

> Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021).
> "Measuring Corporate Culture Using Machine Learning."
> *Review of Financial Studies* 34(7):3265-3315.
> [doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079)

```bibtex
@article{li2021measuring,
  title={Measuring Corporate Culture Using Machine Learning},
  author={Li, Kai and Mai, Feng and Shen, Rui and Yan, Xinyan},
  journal={The Review of Financial Studies},
  volume={34}, number={7}, pages={3265--3315}, year={2021},
  doi={10.1093/rfs/hhaa079}
}
```

## License

MIT.
