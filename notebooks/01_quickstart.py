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
# # lmsy_w2v_rfs quickstart
#
# This notebook reproduces the five-dimension corporate-culture measurement
# from Li, Mai, Shen, Yan (2021, *Review of Financial Studies*) on a tiny
# toy corpus. No Java required.
#
# **Paper:** https://doi.org/10.1093/rfs/hhaa079

# %%
# !pip install -q lmsy_w2v_rfs    # uncomment and run in Colab

# %%
from lmsy_w2v_rfs import Config, Pipeline, CULTURE_DIMS

# %% [markdown]
# ## 1. Build a toy corpus
#
# 25 short "earnings calls" sprinkled with culture language.

# %%
texts = [
    "Our integrity and transparency are core to every transaction.",
    "Innovation and creative passion drive our product roadmap.",
    "We respect every employee and empower our talented workforce.",
    "Customer quality and dedication are the top priority this year.",
    "Our teamwork and cross-functional collaboration build better products.",
    "Ethical accountability and trust guide every decision we make.",
    "We innovate with passion excellence and pride in our craft.",
    "Our culture values integrity honesty and fairness above all else.",
    "A collaborative cooperative team delivers the best innovation outcomes.",
    "Dedicated customer service and quality excellence set us apart.",
    "We respect our talented employees and foster dignity and empowerment.",
    "Innovation creativity and passion drive our long term growth.",
    "Integrity and responsibility are the foundation of our company culture.",
    "Teamwork collaboration and cooperation fuel every major product launch.",
    "Customer expectations are met through quality dedication and service.",
    "Innovation passion and creative excellence drive our new product.",
    "Trust transparency and ethical behavior define our leadership team.",
    "Our employees are empowered with respect dignity and talent recognition.",
    "Quality and dedication to customer commitment are everything to us.",
    "We collaborate across teams through a cooperative collaborative culture.",
    "Accountability and integrity guide our response to every client concern.",
    "Innovative creative passionate employees are our strongest competitive advantage.",
    "Our customer first quality focused culture earns repeat business.",
    "Respectful empowerment of talented employees drives our retention rate.",
    "Teamwork cooperation and collaboration across functions unlock innovation.",
]
doc_ids = [f"call_{i:03d}" for i in range(len(texts))]
len(texts)

# %% [markdown]
# ## 2. Configure the pipeline
#
# Setting `preprocessor="none"` skips Java/CoreNLP entirely. The gensim
# Phrases pass still learns corpus-specific bigrams and trigrams.
#
# These settings are tuned for a tiny 25-document demo. For a real corpus,
# use the defaults: 300-dim vectors, 500 expanded words per dimension,
# `preprocessor="spacy"` (or `"corenlp"` for paper-exact replication).

# %%
cfg = Config(
    preprocessor="none",
    use_gensim_phrases=True,
    phrase_passes=2,
    phrase_min_count=2,
    phrase_threshold=1.0,
    w2v_dim=50,
    w2v_epochs=10,
    w2v_min_count=1,
    n_words_dim=15,
    n_cores=2,
)
cfg

# %% [markdown]
# ## 3. Run the pipeline
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

# %%
p = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/quickstart", config=cfg)
p.run()

# %% [markdown]
# ## 4. Inspect the expanded dictionary
#
# Starting from the seed words (e.g., "integrity", "honesty", "ethic"),
# Word2Vec nearest-neighbor search surfaces corpus-specific terms that the
# researcher did not hand-pick. This is the core contribution of the method.

# %%
for dim in CULTURE_DIMS:
    print(f"\n--- {dim} (top 10) ---")
    print(p.culture_dict[dim][:10])

# %% [markdown]
# ## 5. Inspect scores
#
# Each column is a culture dimension. Higher values mean the document uses
# more words from that dimension's expanded dictionary. The scoring variant
# below is TF-IDF weighted.

# %%
scores = p.score_df("TFIDF")
scores.head()

# %%
scores.to_csv("culture_scores.csv", index=False)
print("Saved to culture_scores.csv")

# %% [markdown]
# ## 6. Bring your own seeds
#
# Any concept that fits "a list of words" works. Here is a two-dimension
# "risk vs growth" dictionary.

# %%
my_cfg = cfg.with_(seeds={
    "risk":   ["risk", "uncertainty", "volatility", "downside"],
    "growth": ["growth", "expansion", "scale", "upside"],
})
print("Custom dimensions:", my_cfg.dims)

# %%
p_custom = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/custom", config=my_cfg)
p_custom.run(methods=("TFIDF",))
p_custom.score_df("TFIDF").head()

# %% [markdown]
# ## 7. Compare scoring methods
#
# The pipeline supports three scoring variants: raw term frequency (TF),
# TF-IDF, and weighted-frequency IDF (WFIDF). The choice rarely changes
# the ranking but affects the scale of the scores.

# %%
scores = p.score_df("TFIDF")
for method in ("TF", "TFIDF", "WFIDF"):
    s = p.score_df(method)
    print(f"{method:8s} mean integrity = {s['integrity'].mean():.4f}")

# %% [markdown]
# ## 8. Citation
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
# ## 9. Related packages
#
# This workshop covers three tools. Pick the one that fits your research question:
#
# | Package | Best for | Runtime |
# |---|---|---|
# | **`lmsyz_genai_ie_rfs`** | Structured extraction: culture type, causes, consequences, causal triples | Requires an LLM API key |
# | **`spar_measure`** | Scoring short texts on a custom semantic scale (e.g., CVF dimensions) | Local CPU/GPU, no API key |
# | **`lmsy_w2v_rfs`** (this notebook) | Historical, deterministic 5-dimension culture scores from word2vec | Local CPU, no API key |
