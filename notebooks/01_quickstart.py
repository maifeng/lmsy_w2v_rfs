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
# Word-embedding seed expansion and document scoring. Bring your own seed
# words; the package trains Word2Vec on your corpus, expands each
# dimension's seed list via nearest-neighbor search, and scores every
# document on every dimension.
#
# Originally a port of the corporate-culture method in Li, Mai, Shen,
# Yan (2021, *RFS*). The package is theory-agnostic: any
# concept that fits "a list of seed words" works.
#
# **Paper:** https://doi.org/10.1093/rfs/hhaa079
#
# **Demo corpus**: 2,000 Glassdoor "pros" reviews (sampled from the RFS
# 2026 validation dataset). The same corpus is used across all three
# workshop notebooks for comparability.

# %% [markdown]
# ## 1. Install and download the corpus
#
# **Colab quick test (TestPyPI build):** run the install cell below, then upload
# `glassdoor_culture_2000.csv` via Colab's file pane (or `files.upload()`).

# %%
# Colab: install from TestPyPI. Uncomment and run.
# !pip install -q --no-cache-dir --index-url https://test.pypi.org/simple/ \
#     --extra-index-url https://pypi.org/simple/ -U lmsy_w2v_rfs

# %%
# Colab: upload glassdoor_culture_2000.csv from your laptop. Uncomment and run.
# from google.colab import files
# files.upload()

# %% [markdown]
# ## 2. Define your seeds
#
# This is the only place where you tell the package what to measure. Any
# mapping of dimension name to a list of seed words works. The cell below
# uses the 2021 paper's five culture dimensions; replace with your own
# constructs (CVF, ESG, risk vs growth, anything word-list-shaped).

# %%
seeds = {
    "integrity":  ["integrity", "ethic", "ethical", "honest", "honesty",
                   "accountable", "trust", "responsibility", "transparency"],
    "teamwork":   ["teamwork", "collaboration", "collaborate", "cooperate",
                   "cooperation", "collaborative"],
    "innovation": ["innovation", "innovate", "innovative", "creative",
                   "creativity", "passion", "excellence"],
    "respect":    ["respect", "respectful", "dignity", "empower",
                   "empowerment", "talent"],
    "quality":    ["quality", "customer", "dedication", "dedicate"],
}

# %% [markdown]
# **Reproducing the 2021 paper exactly?** The original 47-seed dictionary
# is shipped as a named example:
#
# ```python
# from lmsy_w2v_rfs import load_example_seeds
# seeds = load_example_seeds("culture_2021")
# ```

# %% [markdown]
# ## 3. Load the corpus

# %%
import os
import pandas as pd
from lmsy_w2v_rfs import Config, Pipeline

CORPUS_PATH = "glassdoor_culture_2000.csv"
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = "../../../data/glassdoor_culture_2000.csv"

corpus = pd.read_csv(CORPUS_PATH)
texts = corpus["text"].tolist()
doc_ids = corpus["review_id"].astype(str).tolist()
print(f"Corpus: {len(texts)} reviews, {corpus['firm_id'].nunique()} firms")
corpus[["review_id", "text"]].head(3)

# %% [markdown]
# ## 4. Configure the pipeline
#
# `preprocessor="none"` skips Java/CoreNLP entirely. The gensim Phrases
# pass still learns corpus-specific bigrams and trigrams.
#
# With 2,000 short reviews we use 100-dim vectors and expand to 50 words
# per dimension. For a full earnings-call corpus (100k+ documents), the
# defaults (300-dim, 500 expanded words) work better.

# %%
cfg = Config(
    seeds=seeds,
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
# ## 5. Run the pipeline
#
# `Pipeline.run()` executes five stages: parse, clean, phrase, train,
# expand+score. On 2,000 reviews this takes 15-30 seconds on a laptop or
# Colab CPU.

# %%
p = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/glassdoor", config=cfg)
p.run()

# %% [markdown]
# ## 6. Inspect the expanded dictionary
#
# `show_dictionary` prints seeds and the top-K expanded words per
# dimension. `dictionary_preview` returns the same content as a DataFrame
# for notebook display.

# %%
p.show_dictionary(top_k=10)

# %%
p.dictionary_preview(top_k=10)

# %% [markdown]
# ## 7. Curate the dictionary (Step 4 of the paper's procedure)
#
# Word2Vec expansion surfaces noise alongside the signal: generic
# positives (`excellent`, `overall`, `positive`), corpus-specific
# outliers (`store` if many reviewers work in retail), and words too
# vague to be informative (`everything`, `terms`). The 2021 paper,
# Section 3.2: *"we manually inspect all the words in the auto-generated
# dictionary and exclude words that do not fit."*
#
# The package supports two paths:
#
# - **Programmatic**: `p.edit_dictionary(remove={...}, add={...})`
# - **Spreadsheet**: edit `p.dict_path` in Excel or any text editor, save, `p.reload_dictionary()`.
#
# Both update the in-memory dictionary and the on-disk CSV in one call.
# Cached scores are dropped automatically; rerun `p.score()` afterward.

# %%
def correlations(scores_df: pd.DataFrame) -> dict[str, float]:
    """Correlate each dimension column with rating_culture.

    Args:
        scores_df: DataFrame returned by ``p.score_df(...)``.

    Returns:
        Mapping of dimension name to Pearson correlation with rating_culture.
    """
    m = corpus[["review_id", "rating_culture"]].copy()
    m["review_id"] = m["review_id"].astype(str)
    sd = scores_df.copy()
    id_col = [c for c in sd.columns if c.lower() == "doc_id"][0]
    sd[id_col] = sd[id_col].astype(str)
    m = m.merge(sd, left_on="review_id", right_on=id_col)
    return {dim: float(m[dim].corr(m["rating_culture"])) for dim in cfg.dims if dim in m.columns}


initial_corrs = correlations(p.score_df("TFIDF"))

# Concrete noise candidates spotted in this run's expanded dictionary.
# Words not present in a dimension are silently ignored, so this list can
# be generous. In your own work, scan p.show_dictionary(top_k=20) and pick
# words that are not concept-specific.
p.edit_dictionary(remove={
    "integrity":  ["inclusion", "boldness", "employee_engagement"],
    "teamwork":   ["store", "excellent", "overall", "positive", "promotes", "encourages"],
    "innovation": ["individual", "based", "safety", "incredible"],
    "quality":    ["huge", "everything", "terms", "tech", "digital", "model",
                   "space", "used", "short", "voice", "conversations",
                   "approach", "thought", "works"],
})

p.score(methods=("TFIDF",))
curated_corrs = correlations(p.score_df("TFIDF"))

print(f"{'dimension':12s}  {'initial':>8s}  {'curated':>8s}  {'change':>8s}")
for dim in cfg.dims:
    i, c = initial_corrs.get(dim, 0.0), curated_corrs.get(dim, 0.0)
    print(f"  {dim:10s}  {i:+8.3f}  {c:+8.3f}  {c - i:+8.3f}")

# %%
# Inspect the curated dictionary
p.show_dictionary(top_k=10)

# %% [markdown]
# ## 8. Inspect scores

# %%
scores = p.score_df("TFIDF")
scores.head(10)

# %%
scores.to_csv("w2v_glassdoor_scores.csv", index=False)
print("Saved to w2v_glassdoor_scores.csv")

# %% [markdown]
# ## 9. Coverage check
#
# Short reviews have sparse dictionaries; some reviews score zero on
# concepts they do not mention by any dictionary term. Inspect the
# coverage to decide whether the corpus is dense enough for the
# concepts you care about.

# %%
merged = corpus[["review_id", "firm_id", "year", "rating_culture"]].copy()
merged["review_id"] = merged["review_id"].astype(str)
scores_str = scores.copy()
id_col = [c for c in scores_str.columns if c.lower() == "doc_id"][0]
scores_str[id_col] = scores_str[id_col].astype(str)
merged = merged.merge(scores_str, left_on="review_id", right_on=id_col, how="inner")

print("Non-zero score rates (after curation):")
for dim in cfg.dims:
    if dim in merged.columns:
        nz = (merged[dim] > 0).mean()
        print(f"  {dim:12s}: {nz:.0%} of reviews")

# %% [markdown]
# ## 10. Compare scoring methods
#
# Three variants are supported: raw term frequency (TF), TF-IDF (the
# 2021 paper's choice), and weighted-frequency IDF (WFIDF). The choice
# rarely changes the ranking but affects the scale.

# %%
p.score(methods=("TF", "TFIDF", "WFIDF"))
for method in ("TF", "TFIDF", "WFIDF"):
    s = p.score_df(method)
    print(f"{method:8s} mean {cfg.dims[0]} = {s[cfg.dims[0]].mean():.4f}")

# %% [markdown]
# ## 11. Bring your own seeds (custom dimensions)
#
# Same pipeline, different seeds. Below: a two-dimension "risk vs
# growth" dictionary. Define any dimensions relevant to your research
# question and rerun.

# %%
my_cfg = cfg.with_(seeds={
    "risk":   ["risk", "uncertainty", "volatility", "downside", "compliance"],
    "growth": ["growth", "expansion", "scale", "upside", "opportunity"],
})
print("Custom dimensions:", my_cfg.dims)

# %%
p_custom = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/custom", config=my_cfg)
p_custom.run(methods=("TFIDF",))
p_custom.show_dictionary(top_k=10)
p_custom.score_df("TFIDF").head()

# %% [markdown]
# ## 12. Citation
#
# If you use this package in research, please cite the origin paper:
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
# ## 13. Related packages
#
# This workshop covers three tools on the **same 2,000 Glassdoor reviews**.
# Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** (this notebook) | Word-list-driven scoring with corpus-trained Word2Vec expansion | Local CPU, no API key |
