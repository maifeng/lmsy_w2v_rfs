# Use your own seed dictionary

## Problem

The packaged seeds reproduce the 2021 paper's five corporate-culture
dimensions: integrity, teamwork, innovation, respect, quality. That is useful
for culture research on earnings-call transcripts. For anything else, you need
your own seeds. The framework itself is domain-agnostic: anything expressible
as "a list of words per concept" works, whether the concepts are finance risk
factors, policy topics, or a single sentiment axis. The trick is knowing how
to plug your seeds in and which pitfalls break expansion silently.

## Solution

Seeds can come from three sources: a Python dict, a JSON file, or a plain text
file. Pick whichever matches your workflow. All three resolve to the same
`dict[str, list[str]]` internally.

### Python dict

```python
from lmsy_w2v_rfs import Pipeline, Config

cfg = Config(seeds={
    "risk":   ["risk", "uncertainty", "volatility", "hedge"],
    "growth": ["growth", "expand", "expansion", "scale"],
})
```

### JSON file

```json
// my_seeds.json
{
  "risk":   ["risk", "uncertainty", "volatility", "hedge"],
  "growth": ["growth", "expand", "expansion", "scale"]
}
```

```python
from lmsy_w2v_rfs import Config, load_seeds
cfg = Config(seeds=load_seeds("my_seeds.json"))
```

### Plain text file (easiest for hand-editing in Excel or any text editor)

```
# my_seeds.txt - one dimension per line
# format: dim_name: word1 word2 word3 ...
# words can be separated by whitespace or commas
# blank lines and # comments are ignored

risk:   risk uncertainty volatility hedge downside
growth: growth, expand, expansion, scale, upside
people: employee workforce talent hire retain
```

```python
from lmsy_w2v_rfs import Config, load_seeds
cfg = Config(seeds=load_seeds("my_seeds.txt"))
```

### CLI

```bash
lmsy-w2v-rfs run --input docs.txt --seeds my_seeds.txt --out runs/x
lmsy-w2v-rfs run --input docs.txt --seeds my_seeds.json --out runs/x
```

The CLI understands both file extensions.

---

Once the seeds are loaded, the pipeline trains Word2Vec on your corpus, then
for each concept averages its seed vectors and returns the `n_words_dim`
nearest neighbors.

### Example 1: two-dimension risk / growth dictionary for finance text

```python
from lmsy_w2v_rfs import Pipeline, Config

cfg = Config(
    seeds={
        "risk":   ["risk", "uncertainty", "volatility", "hedge", "exposure"],
        "growth": ["growth", "expand", "expansion", "scale", "revenue"],
    },
    preprocessor="spacy",
    spacy_model="en_core_web_sm",
)

p = Pipeline(
    texts=[...],           # your 10-K excerpts, MD&A text, etc.
    doc_ids=[...],
    work_dir="runs/risk_growth",
    config=cfg,
)
p.run()
print(p.score_df("TFIDF").head())
```

Output columns will be `Doc_ID`, `growth`, `risk`, `document_length`. Dimension
names are sorted alphabetically in the output DataFrame.

### Example 2: a single-concept sentiment dictionary

Nothing stops you from shipping one concept. The scoring code treats concepts
independently.

```python
from lmsy_w2v_rfs import Pipeline, Config

cfg = Config(
    seeds={"positivity": ["good", "positive", "strong", "success", "gain"]},
    preprocessor="none",    # skip the parser; pure bag-of-words pipeline
)

p = Pipeline(
    texts=[...],
    doc_ids=[...],
    work_dir="runs/sentiment",
    config=cfg,
)
p.run(methods=("TFIDF",))
```

A single-concept dictionary is still useful: TFIDF weighting gives you a
corpus-calibrated score that you can threshold or rank.

### Example 3: extend the culture dictionary with a new dimension

To keep the paper's five dimensions and add a sixth (say, `sustainability`):

```python
from lmsy_w2v_rfs import Pipeline, Config, CULTURE_SEEDS

extra = {"sustainability": ["sustainability", "emission", "carbon", "renewable"]}
cfg = Config(seeds={**CULTURE_SEEDS, **extra})

p = Pipeline(
    texts=[...],
    doc_ids=[...],
    work_dir="runs/six_dims",
    config=cfg,
)
p.run()
```

`CULTURE_SEEDS` is a plain dict, so you can merge into it, drop keys from it,
or copy-paste its contents into a JSON file and hand-edit.

## Gotcha: seeds must be in the Word2Vec vocabulary

Seed expansion works by averaging seed vectors, so every seed must survive
Word2Vec's `min_count` filter (default 5). If a seed appears fewer than 5
times in your corpus, it is dropped from the vocab and silently skipped during
expansion. A concept whose seeds are all below threshold produces an empty
expanded word list and a score of zero for every document.

Three fixes:

1. **Lower `w2v_min_count`**. Pass `Config(w2v_min_count=2)` to keep rare
   words. Only safe when your corpus is small or your vocabulary is narrow.

2. **Check seed coverage before full scoring**. After running `p.train()`,
   inspect `p.w2v.wv.key_to_index` to see which seeds made it in.

   ```python
   p.parse()
   p.clean()
   p.phrase()
   p.train()

   vocab = p.w2v.wv.key_to_index
   for dim, words in p.config.seeds.items():
       missing = [w for w in words if w not in vocab]
       print(f"{dim}: {len(words) - len(missing)}/{len(words)} in vocab; missing={missing}")
   ```

3. **Use more, more common seed words**. Five seeds per concept is the floor.
   The 2021 paper uses 7 to 12 per dimension. Redundancy matters: if one
   seed is rare, the average still lands near the concept centroid.

## Gotcha: multi-word seeds need the matching preprocessor

If you include a seed like `customer_service`, that token has to exist in the
training corpus as a single underscored token. The CoreNLP backend produces
those via UD `compound` joins. The spaCy backend does not. If you rely on
statistical phrases only (`use_gensim_phrases=True`), co-occurrence has to be
high enough for gensim `Phrases` to discover the bigram. When in doubt,
inspect the final training corpus at `work_dir/corpora/pass2.txt` and grep for
your multi-word seeds before debugging further.

## Related

- [Load your documents](load-documents.md) - the many common input formats
- [Switch the preprocessor](switch-preprocessor.md)
- [Resume after a crash](resume-after-crash.md)
