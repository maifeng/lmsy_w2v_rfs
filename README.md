# lmsy_w2v_rfs

Build a **corpus-specific measurement dictionary with Word2Vec**, ported from Li, Mai, Shen, Yan (2021). Give the package a corpus and a short seed list per concept; it learns the multi-word phrases your corpus uses, expands each seed list into a ranked dictionary of words and phrases, lets you curate that dictionary by hand, and scores every document against it. The package is theory-agnostic: any concept expressible as a list of seed words works.

If you find this library useful in your research, please cite:

Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021), "Measuring Corporate Culture Using Machine Learning," *Review of Financial Studies* 34(7):3265-3315, [doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079).

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

The 2021 paper, footnote 5: phrases convey cultural meaning that single words cannot. The package extracts phrases in two steps that target different kinds of phrases.

**Step 1a, parser-based (general-English phrases).** A dependency parser identifies fixed multiword expressions (`with_respect_to`, `rather_than`) and compound words (`intellectual_property`, `healthcare_provider`). The parser also lemmatizes (`stocks` → `stock`) and masks named entities as `[NER:ORG]` placeholders so proper nouns do not bias the vector space. The 120-token SRAF generic stopword list is removed in the cleaning pass that follows.

| `Config(preprocessor=...)` | Backend | Needs |
|---|---|---|
| `"corenlp"` *(default, paper-faithful)* | Stanford CoreNLP via `stanza.server` | `[corenlp]` extra + Java |
| `"spacy"` | spaCy | `[spacy]` extra + a model |
| `"stanza"` | stanza `Pipeline` | `[stanza]` extra |
| `"static"` | NLTK `MWETokenizer` over a curated list | base install |
| `"none"` | whitespace tokenize, lowercase only | base install |

**Step 1b, statistical (corpus-specific phrases).** After Step 1a, gensim's `Phrases` scans the parsed corpus for statistically significant adjacent-token co-occurrences and joins them with `_`. A second pass over the bigram-joined corpus learns trigrams. This step identifies recurring collocations specific to the corpus. On the paper's earnings-call corpus it surfaces `forward_looking_statement` and `cost_of_capital`; on a Glassdoor-review corpus it surfaces `work_life_balance` and `toxic_environment`.

```python
Config(
    use_gensim_phrases=True,
    phrase_passes=2,            # 1 = bigrams; 2 = bigrams + trigrams
    phrase_min_count=10,        # paper's threshold on a ~270k-doc corpus
    phrase_threshold=10.0,      # for smaller corpora try 3 / 5.0
)
```

The phrase-tagged corpus is written to `work_dir/corpora/pass2.txt` and can be opened directly to inspect the joined phrases.

### Step 2: Word2Vec

`Pipeline.train()` fits a `gensim.models.Word2Vec` on the phrase-tagged corpus. Every word and phrase receives a 300-dimensional vector. The 2021 paper's defaults are baked in:

```python
Config(w2v_dim=300, w2v_window=5, w2v_min_count=5, w2v_epochs=20)
```

The model is saved at `work_dir/models/w2v.mod` and is available as `p.w2v` for ad-hoc queries.

### Step 3: Seed expansion

`Pipeline.expand_dictionary()` builds the per-concept dictionary by:

1. Averaging the in-vocabulary seed vectors for the concept.
2. Taking the top `n_words_dim` (paper's default: 500) tokens by cosine similarity to that mean.
3. Resolving cross-loadings: from the paper, "if a word appears in dictionaries for multiple cultural values, only include it in the value with the highest cosine similarity to the average of seed word vectors for that value."
4. Dropping `[NER:*]` placeholders (the paper does not consider named entities).

The result is written to `work_dir/outputs/expanded_dict.csv`, one column per concept, sorted by descending similarity to the seed mean.

```python
p.show_dictionary(top_k=10)         # prints per-concept seeds + top expansions
p.dictionary_preview(top_k=10)      # DataFrame for notebook display
```

### Step 4: Manual dictionary inspection

The 2021 paper, Section 3.2: *"we manually inspect all the words in the auto-generated dictionary and exclude words that do not fit."* Nearest-neighbor expansion surfaces noise: off-topic terms, industry-specific outliers, words too general to be informative. The paper's authors removed them by hand. The package supports two ways to do this; both update the in-memory dictionary and the on-disk CSV in one call.

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

A document's score on a concept is the sum of TF-IDF weights for every dictionary token present in the document, divided by total document length (the 2021 paper, Section 3.3).

| Method | Weight per dictionary hit | In paper |
|---|---|---|
| `TFIDF` | `tf · log(N/df)` | yes (the paper's method) |
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
    zca_whiten=False,                  # ZCA-decorrelate the concept columns
    zca_epsilon=1e-6,

    random_state=42,
)
```

---

## Citation

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
