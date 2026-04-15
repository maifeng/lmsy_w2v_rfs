# Pipeline

`Pipeline` is the end-to-end orchestrator for the RFS 2021 culture-measurement
workflow. It wraps six stages behind a single class and writes every intermediate
artifact under `work_dir`, so reruns skip any stage whose outputs already exist
unless `force=True` is passed.

The six stages:

- `parse`: Phase 1 preprocessing (MWE join, NER mask) via the configured backend.
- `clean`: lowercase, strip punctuation, drop stopwords, keep `[NER:*]` placeholders.
- `phrase`: gensim `Phrases` statistical bigram / trigram learning (Phase 2).
- `train`: Word2Vec training on the phrase-expanded corpus.
- `expand_dictionary`: grow each seed list into a per-dimension dictionary via
  nearest-neighbor search on the trained vectors.
- `score`: compute document-level TF, TFIDF, WFIDF, and optional SIMWEIGHT variants.

The two construction phases are covered in
[Two-phase preprocessing](../concepts/two-phase-preprocessing.md). The individual
stage knobs live on [Config](config.md).

::: lmsy_w2v_rfs.Pipeline
