# Config

`Config` is a frozen dataclass that holds every hyperparameter the pipeline
needs. It composes the two construction phases: Phase 1 (parser-based MWE
joining and NER masking, with an optional static MWE post-pass) and Phase 2
(gensim `Phrases` bigram / trigram learning). Field defaults mirror the RFS
2021 replication repo's `global_options.py` where applicable. Use `Config.with_(...)`
to copy-and-override individual fields without rebuilding the whole object.

For the rationale behind the two-phase split see
[Two-phase preprocessing](../concepts/two-phase-preprocessing.md). For the
per-backend trade-offs see [Preprocessors](preprocessors.md).

::: lmsy_w2v_rfs.Config

## PreprocessorName

The string literal type accepted by `Config.preprocessor`. Valid values are
`"none"`, `"static"`, `"stanza"`, `"corenlp"`, and `"spacy"`.

::: lmsy_w2v_rfs.PreprocessorName

## default_cache_dir

Returns the on-disk cache root used by `download_corenlp` and by the CoreNLP
preprocessor's auto-install fallback. Honours the `LMSY_W2V_RFS_HOME`
environment variable.

::: lmsy_w2v_rfs.default_cache_dir
