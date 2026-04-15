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
# !pip install -q lmsy_w2v_rfs

# %%
from lmsy_w2v_rfs import Config, Pipeline, CULTURE_SEEDS, CULTURE_DIMS

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
# With `use_corenlp=False`, no Java is needed. The gensim Phrases pass still
# learns corpus-specific bigrams and trigrams.

# %%
cfg = Config(
    use_corenlp=False,
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
# ## 3. Run all five stages

# %%
p = Pipeline(texts=texts, doc_ids=doc_ids, work_dir="runs/quickstart", config=cfg)
p.run(methods=("TF", "TFIDF", "WFIDF"))

# %% [markdown]
# ## 4. Inspect the expanded dictionary

# %%
for dim in CULTURE_DIMS:
    print(f"\n--- {dim} (top 10) ---")
    print(p.culture_dict[dim][:10])

# %% [markdown]
# ## 5. Inspect scores

# %%
p.score_df("TFIDF").head()

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
my_cfg.dims
