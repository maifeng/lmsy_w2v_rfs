# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # lmsy_w2v_rfs: Quickstart
#
# This notebook reproduces the five-dimension corporate-culture measurement
# from Li, Mai, Shen, Yan (2021, *Review of Financial Studies*) on 2,000
# Glassdoor culture reviews. No Java required.
#
# **Paper:** https://doi.org/10.1093/rfs/hhaa079
#
# **Corpus**: 2,000 Glassdoor "pros" reviews about corporate culture,
# sampled from the RFS 2026 validation dataset. The same corpus is used
# across all three workshop notebooks for comparability.

# %% [markdown]
# ## 1. Install and download the corpus

# %%
# !pip install -q lmsy_w2v_rfs    # uncomment and run in Colab

# %%
# Download the shared workshop corpus (uncomment in Colab):
# !wget -q https://raw.githubusercontent.com/maifeng/culture-llm-workshop/main/data/glassdoor_culture_2000.csv

# %% [markdown]
# ## 2. Load the corpus

# %%
import os
import pandas as pd
from lmsy_w2v_rfs import Config, Pipeline, CULTURE_DIMS

CORPUS_PATH = "glassdoor_culture_2000.csv"
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = "../../../data/glassdoor_culture_2000.csv"

corpus = pd.read_csv(CORPUS_PATH)
texts = corpus["text"].tolist()
doc_ids = corpus["review_id"].astype(str).tolist()
print(f"Corpus: {len(texts)} reviews, {corpus['firm_id'].nunique()} firms")
corpus[["review_id", "text"]].head(3)

# %% [markdown]
# ## 3. Configure the pipeline
#
# Setting `preprocessor="none"` skips Java/CoreNLP entirely. The gensim
# Phrases pass still learns corpus-specific bigrams and trigrams.
#
# With 2,000 reviews (~385 chars each) we can train a useful Word2Vec
# model. We use 100-dim vectors and expand to 50 words per dimension,
# which works well for a corpus of this size. For a full earnings-call
# corpus (100k+ documents), use the defaults: 300-dim, 500 expanded words.

# %%
cfg = Config(
    preprocessor="none",
    use_gensim_phrases=True,
    phrase_passes=2,
    phrase_min_count=3,
    phrase_threshold=5.0,
    w2v_dim=100,
    w2v_epochs=15,
    w2v_min_count=2,
    n_words_dim=50,
    n_cores=2,
)
cfg

# %% [markdown]
# ## 4. Run the pipeline
#
# `Pipeline.run()` executes five stages in order:
#
# 1. **Parse**: tokenize and (optionally) lemmatize + NER-mask the corpus.
# 2. **Clean**: lowercase, remove stopwords and short tokens.
# 3. **Phrase**: learn bigrams/trigrams via gensim Phrases.
# 4. **Train**: fit a Word2Vec model on the cleaned corpus.
# 5. **Expand + Score**: for each dimension, find the nearest neighbors of
#    the seed words in vector space, then score every document by weighted
#    term frequency against the expanded dictionary.
#
# On 2,000 reviews with `preprocessor="none"`, this takes about 15-30
# seconds on a laptop CPU or Colab instance.

# %%
p = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/glassdoor", config=cfg)
p.run()

# %% [markdown]
# ## 5. Inspect the expanded dictionary
#
# Starting from the seed words (e.g., "integrity", "honesty", "ethic"),
# Word2Vec nearest-neighbor search surfaces corpus-specific terms that the
# researcher did not hand-pick. This is the core contribution of the method.
# Look for terms that capture how Glassdoor reviewers actually talk about
# each culture dimension.

# %%
for dim in CULTURE_DIMS:
    seeds = cfg.seeds[dim][:3]
    expanded = p.culture_dict[dim][:10]
    print(f"\n--- {dim} ---")
    print(f"  Seeds:    {seeds}")
    print(f"  Expanded: {expanded}")

# %% [markdown]
# ## 6. Inspect scores
#
# Each column is a culture dimension. Higher values mean the document uses
# more words from that dimension's expanded dictionary. The scoring variant
# below is TF-IDF weighted.

# %%
scores = p.score_df("TFIDF")
scores.head(10)

# %%
scores.to_csv("w2v_glassdoor_scores.csv", index=False)
print("Saved to w2v_glassdoor_scores.csv")

# %% [markdown]
# ## 7. Merge scores with metadata
#
# Since we scored the same corpus used by the other two notebooks, we can
# merge the five culture scores with Glassdoor metadata and look for
# patterns. For example, do reviews with higher "integrity" scores also
# have higher culture ratings?

# %%
merged = corpus[["review_id", "firm_id", "year", "rating_culture"]].copy()
merged["review_id"] = merged["review_id"].astype(str)
merged = merged.merge(scores, left_on="review_id", right_on="doc_id", how="inner")

print("Correlation of each dimension with rating_culture:")
for dim in CULTURE_DIMS:
    if dim in merged.columns:
        r = merged[dim].corr(merged["rating_culture"])
        print(f"  {dim:12s}: {r:+.3f}")

# %% [markdown]
# ## 8. Compare scoring methods
#
# The pipeline supports three scoring variants: raw term frequency (TF),
# TF-IDF, and weighted-frequency IDF (WFIDF). The choice rarely changes
# the ranking but affects the scale of the scores.

# %%
for method in ("TF", "TFIDF", "WFIDF"):
    s = p.score_df(method)
    print(f"{method:8s} mean integrity = {s['integrity'].mean():.4f}")

# %% [markdown]
# ## 9. Bring your own seeds
#
# Any concept that fits "a list of words" works. Here is a two-dimension
# "risk vs growth" dictionary. You can define any dimensions relevant to
# your research question.

# %%
my_cfg = cfg.with_(seeds={
    "risk":   ["risk", "uncertainty", "volatility", "downside", "compliance"],
    "growth": ["growth", "expansion", "scale", "upside", "opportunity"],
})
print("Custom dimensions:", my_cfg.dims)

# %%
p_custom = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/custom", config=my_cfg)
p_custom.run(methods=("TFIDF",))
p_custom.score_df("TFIDF").head()

# %% [markdown]
# ## 10. Citation
#
# If you use this package in research, please cite:
#
# ```
# Li, Kai, Feng Mai, Rui Shen, and Xinyan Yan. 2021.
# "Measuring Corporate Culture Using Machine Learning."
# Review of Financial Studies 34(7):3265-3315.
# https://doi.org/10.1093/rfs/hhaa079
# ```

# %%
import lmsy_w2v_rfs
print(lmsy_w2v_rfs.__paper__)

# %% [markdown]
# ## 11. Related packages
#
# This workshop covers three tools on the **same 2,000 Glassdoor reviews**.
# Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** (this notebook) | Historical, deterministic 5-dimension culture scores from word2vec | Local CPU, no API key |
