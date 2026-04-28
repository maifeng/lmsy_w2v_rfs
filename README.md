# `lmsy_w2v_rfs` — Word2Vec dictionary expansion and scoring for any seed-based vocabulary

[![Open in Colab](https://img.shields.io/badge/Colab-quickstart-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/maifeng/lmsy_w2v_rfs/blob/main/notebooks/01_quickstart_colab.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/lmsy_w2v_rfs.svg)](https://pypi.org/project/lmsy_w2v_rfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Builds a **corpus-specific measurement dictionary with Word2Vec**. For each concept you want to measure in your corpus:

- **You provide** a short seed-word list per concept.
- **The package builds** a ranked dictionary of the words and multi-word phrases your corpus uses to express that concept.
- **You curate** the dictionary: inspect, drop noise, add domain words.
- **The package scores** every document by weighted hits against the curated dictionary.

Cite as: Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021), *RFS* 34(7):3265-3315. Full citation at the bottom.

---

## Install

```bash
pip install -U lmsy_w2v_rfs
```

The default preprocessor (`corenlp`) needs Java and a one-time CoreNLP archive download:

```bash
pip install -U "lmsy_w2v_rfs[corenlp]"
lmsy-w2v-rfs download-corenlp                 # one-time, ~1 GB
```

Java-free alternatives:

```bash
pip install -U "lmsy_w2v_rfs[spacy]" && python -m spacy download en_core_web_sm
pip install -U "lmsy_w2v_rfs[stanza]"
pip install -U lmsy_w2v_rfs                   # bare; use preprocessor="static" or "none"
```

---

## Quickstart

Two concepts, a few seed words each, four lines of pipeline:

```python
from lmsy_w2v_rfs import Pipeline, Config

seeds = {
    "risk":   ["risk", "uncertainty", "volatility", "downside"],
    "growth": ["growth", "expansion", "scale", "opportunity"],
}
texts = [
    "Macro uncertainty and rising rates weighed on margins this quarter.",
    "Strong customer demand drove double-digit revenue expansion across segments.",
    "We hedged commodity exposure to limit downside from price volatility.",
    "Investments in new markets are scaling our growth opportunity.",
    # ... thousands more rows in practice
]

p = Pipeline(
    texts=texts, doc_ids=[f"d{i}" for i in range(len(texts))],
    work_dir="runs/quickstart",
    config=Config(seeds=seeds, preprocessor="none"),
)
p.run()                     # phrase + train + expand + score
p.show_dictionary(top_k=10) # inspect the expanded dictionary
print(p.score_df("TFIDF"))  # per-document scores
```

```text
=== risk (12 words) ===
  seeds:    risk, uncertainty, volatility, downside
  expanded: risk, uncertainty, volatility, downside, exposure,
            commodity_exposure, rising_rates, hedge, macro_uncertainty
=== growth (14 words) ===
  seeds:    growth, expansion, scale, opportunity
  expanded: growth, expansion, scale, opportunity, customer_demand,
            new_markets, revenue_expansion, double_digit, scaling
```

| Doc_ID | risk | growth | document_length |
|---|---|---|---|
| d0 | 0.41 | 0.00 | 13 |
| d1 | 0.00 | 0.55 | 12 |
| d2 | 0.62 | 0.00 | 12 |
| d3 | 0.00 | 0.49 | 11 |

To reproduce the 2021 paper exactly:

```python
from lmsy_w2v_rfs import load_example_seeds
seeds = load_example_seeds("culture_2021")    # 47 seeds, 5 dimensions
```

---

## The construction procedure

The package implements the four-step construction procedure of Li et al. (2021). Each step is a method on `Pipeline`; calling `.run()` executes them in order and saves intermediate artifacts under `work_dir/` so any step can be redone without redoing the others.

### Step 1: Two-step phrase construction

Phrases carry meaning that single words cannot. The package extracts them in two complementary steps targeting different kinds of phrases.

**Step 1a, parser-based (general-English phrases).** A dependency parser identifies fixed multiword expressions (`with_respect_to`, `rather_than`) and compound words (`intellectual_property`, `healthcare_provider`). The parser also lemmatizes (`stocks` → `stock`) and masks named entities as `[NER:ORG]` placeholders so proper nouns do not bias the vector space. The 121-token SRAF generic stopword list is removed in the cleaning pass that follows.

| `Config(preprocessor=...)` | Backend | Needs |
|---|---|---|
| `"corenlp"` *(default, paper-faithful)* | Stanford CoreNLP via `stanza.server` | `[corenlp]` extra + Java |
| `"spacy"` | spaCy | `[spacy]` extra + a model |
| `"stanza"` | stanza `Pipeline` | `[stanza]` extra |
| `"static"` | NLTK `MWETokenizer` over a curated list | base install |
| `"none"` | whitespace tokenize, lowercase only | base install |

**Step 1b, statistical (corpus-specific phrases).** After Step 1a, gensim's `Phrases` scans the parsed corpus for statistically significant adjacent-token co-occurrences and joins them with `_`. A second pass over the bigram-joined corpus learns trigrams. This step identifies recurring collocations specific to the corpus: an earnings-call corpus surfaces `forward_looking_statement` and `cost_of_capital`; a product-review corpus surfaces `customer_service` and `delivery_time`; a Glassdoor corpus surfaces `work_life_balance` and `growth_opportunity`.

```python
from lmsy_w2v_rfs import Config, load_example_seeds

seeds = load_example_seeds("culture_2021")  # or any dict[str, list[str]]
Config(
    seeds=seeds,
    use_gensim_phrases=True,
    phrase_passes=2,            # 1 = bigrams; 2 = bigrams + trigrams
    phrase_min_count=10,        # works on a ~270k-doc corpus
    phrase_threshold=10.0,      # for smaller corpora try 3 / 5.0
)
```

The phrase-tagged corpus is written to `work_dir/corpora/pass2.txt` and can be opened directly to inspect the joined phrases.

### Step 2: Word2Vec

`Pipeline.train()` fits a `gensim.models.Word2Vec` on the phrase-tagged corpus. Every word and phrase receives a 300-dimensional vector. Defaults match the 2021 paper:

```python
from lmsy_w2v_rfs import Config, load_example_seeds

seeds = load_example_seeds("culture_2021")  # or any dict[str, list[str]]
Config(seeds=seeds, w2v_dim=300, w2v_window=5, w2v_min_count=5, w2v_epochs=20)
```

The model is saved at `work_dir/models/w2v.mod` and is available as `p.w2v` for ad-hoc queries.

### Step 3: Seed expansion

`Pipeline.expand_dictionary()` builds the per-concept dictionary by:

1. Averaging the in-vocabulary seed vectors for the concept.
2. Taking the top `n_words_dim` (default 500) tokens by cosine similarity to that mean.
3. Resolving cross-loadings: a token close to multiple concepts is assigned to the one whose seed mean it is closest to.
4. Dropping `[NER:*]` placeholders so named entities never enter the dictionary.

The result is written to `work_dir/outputs/expanded_dict.csv`, one column per concept, sorted by descending similarity to the seed mean.

```python
p.show_dictionary(top_k=10)         # prints per-concept seeds + top expansions
p.dictionary_preview(top_k=10)      # DataFrame for notebook display
```

### Step 4: Manual dictionary inspection

Nearest-neighbor expansion surfaces noise: off-topic terms, industry-specific outliers, words too general to be informative. Two ways to remove them, both atomic across the in-memory dictionary and the on-disk CSV:

```python
# Programmatic, replicable in a notebook:
p.edit_dictionary(
    remove={"risk": ["fantastic", "build"]},
    add={"risk": ["liability"]},
)

# Spreadsheet-driven, faster on a big dictionary:
#   1. open p.dict_path in Excel or any text editor
#   2. edit, save
#   3. p.reload_dictionary()
```

Cached scores are dropped after curation. Call `p.score()` to rescore against the curated dictionary.

---

## Scoring

A document's score on a concept is the sum of TF-IDF weights for every dictionary token present in the document, divided by total document length.

| Method | Weight per dictionary hit | Source |
|---|---|---|
| `TFIDF` | `tf · log(N/df)` | 2021 paper |
| `TF` | `tf` | extension |
| `WFIDF` | `(1 + log tf) · log(N/df)` | extension |
| `TFIDF+SIMWEIGHT`, `WFIDF+SIMWEIGHT` | × `1/ln(2 + rank)` | extension |

`SIMWEIGHT` variants additionally down-weight tokens further from the seed mean (rank in the expanded dictionary).

```python
p.score(methods=("TFIDF",))
p.score_df("TFIDF")
```

Outputs land at `work_dir/outputs/scores_<METHOD>.csv`.

---

## Loading documents and seeds

```python
Pipeline(texts=[...], doc_ids=[...], work_dir=..., config=cfg)              # in-memory list
Pipeline.from_csv("docs.csv", text_col="text", id_col="id", ...)            # CSV
Pipeline.from_dataframe(df, text_col="text", id_col="id", ...)              # DataFrame
Pipeline.from_directory("./docs/", pattern="*.txt", ...)                    # one file per doc
Pipeline.from_text_file("docs.txt", id_path="ids.txt", ...)                 # one doc per line
Pipeline.from_jsonl("docs.jsonl", text_key="text", id_key="id", ...)        # JSONL
```

Seeds accept a Python dict, a JSON file, or a plain text file:

```python
from lmsy_w2v_rfs import load_seeds
Config(seeds=load_seeds("my_seeds.json"))     # or .txt, or pass a dict directly
```

CLI: `lmsy-w2v-rfs run --seeds my_seeds.txt --input docs.csv --input-format csv --out runs/x`.

---

## Large corpora

Once parsing finishes, downstream stages stream through disk: `clean` reads parsed sentences line by line; `phrase` and `train` use gensim's `PathLineSentences` so the training corpus is never fully materialized. The bottleneck is the **input stage**: the document loader holds the corpus in a Python list before parsing begins.

For corpora beyond a few hundred thousand documents, or when running on a cluster, see the [Run on HPC how-to](docs/how-to/run-on-hpc.md) for the multi-shard workflow, SLURM and SGE templates, and BLAS thread-cap instructions.

---

## All knobs

```python
Config(
    seeds=...,                         # required: dict[str, list[str]]

    # Step 1a
    preprocessor="corenlp",            # "corenlp" | "spacy" | "stanza" | "static" | "none"
    mwe_list=None,                     # None | "finance" | path to a curated list
    spacy_model="en_core_web_sm",
    n_cores=4,
    corenlp_memory="6G",
    corenlp_port=9002,
    corenlp_timeout_ms=120_000,        # per-request CoreNLP timeout (ms)

    # Step 1b
    use_gensim_phrases=True,
    phrase_passes=2,
    phrase_threshold=10.0,
    phrase_min_count=10,

    # Step 2
    w2v_dim=300,
    w2v_window=5,
    w2v_min_count=5,
    w2v_epochs=20,

    # Step 3
    n_words_dim=500,                   # paper's threshold for the dictionary cutoff
    dict_restrict_vocab=None,
    min_similarity=0.0,

    # Scoring (extensions beyond the 2021 paper)
    tfidf_normalize=False,
    zca_whiten=False,                  # ZCA-decorrelate the concept columns; see docs/how-to/whiten-scores.md
    zca_epsilon=1e-6,

    random_state=42,
)
```

---

## Documentation

Full docs (concepts, how-to guides, API reference): https://maifeng.github.io/lmsy_w2v_rfs/

## Citation

If you use this package in your research, please cite the paper this implementation is based on:

Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021), "Measuring Corporate Culture Using Machine Learning," *Review of Financial Studies* 34(7):3265-3315, [doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079).

```bibtex
@article{li2021measuring,
  title={Measuring Corporate Culture Using Machine Learning},
  author={Li, Kai and Mai, Feng and Shen, Rui and Yan, Xinyan},
  journal={The Review of Financial Studies},
  volume={34}, number={7}, pages={3265--3315}, year={2021},
  doi={10.1093/rfs/hhaa079}
}
```

## Links

- GitHub: [github.com/maifeng/lmsy_w2v_rfs](https://github.com/maifeng/lmsy_w2v_rfs)
- PyPI: [pypi.org/project/lmsy_w2v_rfs](https://pypi.org/project/lmsy_w2v_rfs/)

## License

MIT.
