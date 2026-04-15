# lmsy_w2v_rfs

**Word2Vec-based corporate-culture measurement, ported from Li, Mai, Shen, Yan (2021).**

Code derived from Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021), "Measuring Corporate Culture Using Machine Learning," *Review of Financial Studies* 34(7):3265-3315, [doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079).

The framework itself is domain-agnostic. You supply a corpus and a per-concept seed-word dictionary; it trains Word2Vec, expands each seed set by nearest-neighbor search, and scores every document on each concept.

---

## The idea in one example

The pipeline's job is to take a **small seed word list per concept** and a **corpus of documents**, then train Word2Vec on the corpus, use nearest-neighbor search in the embedding space to **expand** each seed list into a full vocabulary, and score every document by counting weighted hits from the expanded vocabulary.

**Step 1. You provide a seed dictionary.** 47 words across five dimensions, from the 2021 paper:

```python
seeds = {
    "integrity":  ["integrity", "ethic", "ethical", "accountable", "trust",
                   "honesty", "honest", "transparency", "transparent"],
    "teamwork":   ["teamwork", "collaboration", "collaborate", "cooperation"],
    "innovation": ["innovation", "innovate", "innovative", "creativity",
                   "creative", "passion", "excellence", "pride"],
    "respect":    ["respectful", "talent", "talented", "employee", "dignity",
                   "empowerment", "empower"],
    "quality":    ["quality", "customer", "dedicated", "dedication"],
}
```

**Step 2. You provide a corpus.** Here a small one; in practice, tens of thousands of earnings-call transcripts. See [how to load your documents](#how-do-i-load-my-documents) below for the many common formats.

```python
from lmsy_w2v_rfs import Pipeline, Config

p = Pipeline(
    texts=earnings_call_texts,   # list of strings (other loaders: CSV, JSONL, directory, DataFrame)
    doc_ids=firm_quarter_ids,
    work_dir="runs/my_experiment",
    config=Config(seeds=seeds),
)
p.run()
```

**Step 3. The pipeline expands the 47 seeds into a few thousand words.** These are the words whose Word2Vec vectors are closest to the mean of each dimension's seed vectors:

```python
print(p.culture_dict["integrity"][:15])
# ['integrity', 'ethic', 'ethical', 'accountable', 'honest', 'honesty',
#  'trust', 'transparency', 'accountability', 'fairness', 'responsibility',
#  'win_win', 'pay_close_attention', 'stay_close_to', 'balancing_act']

print(p.culture_dict["quality"][:15])
# ['customer', 'quality', 'customer_experience', 'customer_service',
#  'high_quality', 'cost_effective', 'satisfy_need', 'dedicated',
#  'superior_value', 'customer_focused', 'knowledge', 'functionality', ...]
```

The expanded dictionary is what you trust the pipeline for. Seeds are the hand-curated input; the expansion is the data-driven output. Without the expansion step, this is just a keyword counter.

**Step 4. Documents get scored.** Per document, per dimension, tf-idf weighted count of hits from the expanded dictionary, divided by document length.

```python
print(p.score_df("TFIDF"))
```

| Doc_ID | innovation | integrity | quality | respect | teamwork | document_length |
|---|---|---|---|---|---|---|
| AAPL_2024Q1 | 0.82 | 0.05 | 0.11 | 0.03 | 0.02 | 15,240 |
| WFC_2024Q1  | 0.04 | 0.73 | 0.12 | 0.08 | 0.03 | 12,105 |
| TSLA_2024Q1 | 0.91 | 0.04 | 0.22 | 0.03 | 0.05 | 19,850 |

**Optional Step 5.** Aggregate to firm-year with `p.firm_year(id_to_firm)`.

The 47 input seeds grew into thousands of culture-related words because Word2Vec learned that `customer_experience` sits near `customer` in vector space, `win_win` sits near `integrity`, `cost_effective` sits near `quality`. That is the whole point.

The pipeline has two construction phases, both on by default:

1. **Phase 1a: parser-based syntactic MWE joining + NER masking.** Default `preprocessor="corenlp"`, matching the 2021 paper. Stanford CoreNLP via `stanza.server.CoreNLPClient` gives the best syntactic MWE coverage (76% of our benchmark) and scales near-linearly with JVM threads (5.7x speedup at 8 threads on the measured run).
2. **Phase 2: gensim `Phrases` statistical MWE learning.** Learns high-frequency bigrams and trigrams from your corpus.

Set `preprocessor="spacy"` for the fastest Java-free path (spaCy sm with `n_process=8` finishes the 1,393-doc sample in ~4 min vs CoreNLP's ~12 min, at the cost of 0% syntactic MWE recall). `preprocessor="stanza"` is the Python-native middle ground. `preprocessor="static"` runs a curated list only. `preprocessor="none"` skips Phase 1a entirely.

---

## How do I load my documents?

Whatever your raw data format, there is a one-line way in. Pick whichever matches what you have on disk:

**(a) You already have a list of strings in memory.**

```python
p = Pipeline(
    texts=["text of doc 1", "text of doc 2", ...],
    doc_ids=["id1", "id2", ...],
    work_dir="runs/x",
)
```

**(b) A CSV file with `id` and `text` columns** (Compustat, WRDS, your own pull):

```python
p = Pipeline.from_csv("transcripts.csv", text_col="text", id_col="id", work_dir="runs/x")
```

**(c) A pandas DataFrame you already built:**

```python
import pandas as pd
df = pd.read_parquet("transcripts.parquet")
p = Pipeline.from_dataframe(df, text_col="transcript", id_col="call_id", work_dir="runs/x")
```

**(d) A directory of one-file-per-document `.txt` files** (SEC filings, earnings calls you downloaded one at a time):

```python
p = Pipeline.from_directory("./10k_filings/", pattern="*.txt", work_dir="runs/x")
# Document IDs become the file stems: 10k_AAPL_2024.txt -> "10k_AAPL_2024"
```

**(e) A single large text file, one document per line** (what the 2021 paper shipped):

```python
p = Pipeline.from_text_file(
    "data/input/documents.txt",
    id_path="data/input/document_ids.txt",
    work_dir="runs/x",
)
```

**(f) A JSON Lines file, one JSON record per line:**

```python
p = Pipeline.from_jsonl("transcripts.jsonl", text_key="transcript", id_key="call_id",
                        work_dir="runs/x")
```

Each factory materializes the documents into a Python list in memory (fine for hundreds of thousands of earnings calls on a laptop; not fine for hundreds of millions of documents). For truly streaming ingestion over corpora larger than RAM, shard your input into 10k-document files and run the pipeline once per shard. See [how-to/load-documents.md](docs/how-to/load-documents.md) for worked examples of every format.

## How do I change the seed dictionary?

Three ways, pick whichever is most convenient for you:

**(a) Pass a Python dict** (what the example above shows):

```python
Config(seeds={"risk": ["risk", "uncertainty", "volatility"], ...})
```

**(b) Load a JSON file:**

```python
# my_seeds.json
# {
#   "risk":   ["risk", "uncertainty", "volatility"],
#   "growth": ["growth", "expand", "expansion"]
# }
from lmsy_w2v_rfs import load_seeds
Config(seeds=load_seeds("my_seeds.json"))
```

**(c) Load a plain text file** (one dimension per line, easiest to edit in Excel or a text editor):

```text
# my_seeds.txt
risk:   risk uncertainty volatility hedge downside
growth: growth expand expansion scale upside
people: employee workforce talent hire retain
```

```python
Config(seeds=load_seeds("my_seeds.txt"))
```

Or from the CLI: `lmsy-w2v-rfs run --seeds my_seeds.txt ...`.

---

## Install

The default preprocessor is `corenlp`. You need the `[corenlp]` extra AND a Java 8+ runtime (`brew install openjdk@21` / `apt install default-jre`):

```bash
pip install "lmsy_w2v_rfs[corenlp]"     # recommended default (paper-faithful)
lmsy-w2v-rfs download-corenlp           # one-time ~1 GB CoreNLP archive download
```

If Java is unavailable, the Python-only alternatives are:

```bash
pip install "lmsy_w2v_rfs[spacy]"       # spaCy: fastest, lowest friction (no Java)
python -m spacy download en_core_web_sm # small model; trf/md are also options

pip install "lmsy_w2v_rfs[stanza]"      # stanza: Python-native parser, modern UD

pip install "lmsy_w2v_rfs[all]"         # all three backends at once
```

Then set `Config(preprocessor="spacy")` or the CLI flag `--preprocessor spacy`.

Bare `pip install lmsy_w2v_rfs` is fine if you plan to use `preprocessor="static"` or `preprocessor="none"` (no parser needed).

For spaCy:

```bash
python -m spacy download en_core_web_trf    # best NER, slower
python -m spacy download en_core_web_sm     # smaller and faster
```

For CoreNLP (one-time archive download, ~1 GB):

```bash
lmsy-w2v-rfs download-corenlp
```

---

## Two-phase construction, both configurable

| Phase | What it does | Knob | Default |
|---|---|---|---|
| 1a | Parser-based lemmatization, NER masking, MWE joining | `preprocessor` | `"corenlp"` |
| 1b | Optional static MWE list as a post-pass | `mwe_list` | `None` |
| 2 | gensim `Phrases` learns corpus-specific bigrams and trigrams | `use_gensim_phrases`, `phrase_passes` | `True`, `2` |

### Preprocessor options (Phase 1a)

| value | What it does | Needs | Strength |
|---|---|---|---|
| `"corenlp"` (default) | parse, NER mask, UD v2 MWE join | `[corenlp]` extra + Java | paper-exact reproduction, best syntactic MWE |
| `"spacy"` | parse, NER mask, UD compound join | `[spacy]` extra + model | fastest parser, best NER, no Java |
| `"stanza"` | parse, NER mask, UD v2 MWE join | `[stanza]` extra | Python-native, no Java |
| `"static"` | NLTK `MWETokenizer` on your list | just `nltk` | deterministic, no parser |
| `"none"` | whitespace tokenize, lowercase | nothing | fastest path, no lemmas or NER |

See [MWE benchmark comparison](docs/explanation/mwe-comparison.md) for the benchmark that motivated this shape.

### Static MWE list (Phase 1b)

Completely optional. Applied AFTER the main preprocessor as a post-pass, so MWEs the parser missed can still be joined. Three ways to supply one:

```python
Config(mwe_list=None)             # default, skip this pass
Config(mwe_list="finance")        # packaged earnings-call example list
Config(mwe_list="my_mwes.txt")    # your own newline-delimited file
```

**About the packaged `"finance"` list**: it is a hand-curated example file the author assembled from (i) UD v2 `fixed` prepositional phrases, (ii) earnings-call jargon listed from general knowledge, (iii) business-culture MWEs from the RFS 2021 paper's dictionary appendix. **It is not a default.** It is not derived from any corpus-driven process. If you are not working with earnings-call text, ignore it or pass your own list. The file lives at `src/lmsy_w2v_rfs/data/mwes_finance.txt` and is readable.

### Full `Config` reference

Every knob, with defaults. The three sections above cover `preprocessor` and `mwe_list`; everything else here is standard Word2Vec and scoring machinery.

```python
Config(
    # Phase 1 preprocessing
    preprocessor="corenlp",            # "corenlp" | "spacy" | "stanza" | "static" | "none"
    mwe_list=None,                     # None | "finance" | path to a txt file
    spacy_model="en_core_web_sm",      # used when preprocessor == "spacy"
    n_cores=4,                         # JVM threads / spaCy n_process / stanza workers
    corenlp_memory="6G",
    corenlp_port=9002,

    # Phase 2: gensim Phrases
    use_gensim_phrases=True,
    phrase_passes=2,                   # 1 = bigram, 2 = bigram + trigram
    phrase_threshold=10.0,
    phrase_min_count=10,

    # Word2Vec
    w2v_dim=300,
    w2v_window=5,
    w2v_min_count=5,
    w2v_epochs=20,

    # Dictionary expansion
    n_words_dim=500,                   # top-k expanded words per dimension
    min_similarity=0.0,
    dict_restrict_vocab=None,          # fraction in (0,1] to restrict expansion to top-freq vocab

    # Scoring
    tfidf_normalize=False,             # L2-normalize tf-idf vectors per document
    zca_whiten=False,                  # decorrelate dimension columns post-scoring
    zca_epsilon=1e-6,

    random_state=42,
)
```

---

## Scoring methods

| Method | Weight on one dictionary hit |
|---|---|
| `TF` | `tf` |
| `TFIDF` | `tf * log(N / df)` |
| `WFIDF` | `(1 + log(tf)) * log(N / df)` |
| `TFIDF+SIMWEIGHT` | `tf * log(N / df) * 1 / ln(2 + rank)` |
| `WFIDF+SIMWEIGHT` | `(1 + log(tf)) * log(N / df) * 1 / ln(2 + rank)` |

Similarity weights come from the expansion rank (seed at rank 0 gets weight 1.44, rank 100 gets 0.21). Matches the 2021 paper's Appendix.

---

## Firm-year aggregation

```python
firm_year = p.firm_year(id_to_firm_df, method="TFIDF")
```

Scores are divided by document length, scaled to per-100-tokens, and averaged within `(firm_id, time)` groups.

---

## CLI

A full walkthrough for command-line users lives at
[docs/how-to/run-from-cli.md](docs/how-to/run-from-cli.md). Quick examples:

```bash
# CSV in, scores out, default CoreNLP preprocessor, 8 threads
lmsy-w2v-rfs run --input transcripts.csv --input-format csv \
  --text-col transcript --id-col call_id \
  --seeds my_seeds.txt \
  --out runs/my_experiment --n-cores 8

# Directory of .txt files, fast spaCy path (no Java needed)
lmsy-w2v-rfs run --input ./10k_filings/ --input-format directory \
  --preprocessor spacy --spacy-model en_core_web_sm \
  --out runs/10k

# One document per line (the 2021 paper's sample format)
lmsy-w2v-rfs run --input documents.txt --ids document_ids.txt \
  --out runs/rfs2021

# First-time CoreNLP setup
lmsy-w2v-rfs download-corenlp
```

Every flag is documented in [docs/reference/cli.md](docs/reference/cli.md).
Supported input formats: plain text (one doc per line), CSV, JSONL, directory
of files. Seed dictionaries: Python dict, JSON, or plain text.

---

## Limits

The scorer is a bag-of-words counter over a learned vocabulary. It cannot read context within a sentence, cannot handle negation, and cannot judge whether a passage describes practice or aspiration. A transcript that says "we do not tolerate a lack of integrity" scores identically to a genuine integrity claim.

The dictionary is frozen at training time. Adding "AI" or "freelancer" as culture terms means retraining.

For context-aware scoring, see the companion package [`lmsyz_genai_ie_rfs`](https://github.com/feng-mai/lmsyz_genai_ie_rfs), which replaces the weighted count with an LLM prompt.

---

## Paper

> Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan (2021).
> "Measuring Corporate Culture Using Machine Learning."
> *Review of Financial Studies* 34(7):3265-3315.
> [doi.org/10.1093/rfs/hhaa079](https://doi.org/10.1093/rfs/hhaa079)

```bibtex
@article{li2021measuring,
  title={Measuring Corporate Culture Using Machine Learning},
  author={Li, Kai and Mai, Feng and Shen, Rui and Yan, Xinyan},
  journal={The Review of Financial Studies},
  volume={34},
  number={7},
  pages={3265--3315},
  year={2021},
  doi={10.1093/rfs/hhaa079}
}
```

## License

MIT.
