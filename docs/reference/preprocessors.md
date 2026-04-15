# Preprocessors

Phase 1 is pluggable. Each backend turns a raw document into a list of sentences,
where each sentence is a list of lemmatized tokens with multi-word expressions
joined by `_` and named-entity spans replaced with `[NER:TYPE]` placeholders.
The `Pipeline.parse` stage selects a backend through `Config.preprocessor` and
instantiates it via [`build_preprocessor`](#build_preprocessor). An optional
curated MWE list (`Config.mwe_list`) is applied as a second pass via
[`apply_mwe_list`](#apply_mwe_list).

Design background and benchmark data live in
[Two-phase preprocessing](../concepts/two-phase-preprocessing.md).

## The Preprocessor protocol

Any backend that implements `process(text) -> list[list[str]]` is a valid
preprocessor. Backends that benefit from concurrency can also override
`process_documents`.

::: lmsy_w2v_rfs.Preprocessor

## build_preprocessor

::: lmsy_w2v_rfs.build_preprocessor

## NoOpPreprocessor

Whitespace split, lowercase, no parse, no NER. Fastest path. Useful for tests
and for users who only want the gensim `Phrases` + Word2Vec half of the pipeline.

::: lmsy_w2v_rfs.preprocessors.none_pp.NoOpPreprocessor

## StaticMWEPreprocessor

No parser, no NER. Splits on sentences and applies NLTK `MWETokenizer` with a
curated MWE list (the packaged `"finance"` list or a user-supplied path).
Deterministic, Java-free, zero-ML.

::: lmsy_w2v_rfs.preprocessors.static_mwe.StaticMWEPreprocessor

## StanzaPreprocessor

Neural stanza pipeline (tokenize, pos, lemma, depparse, ner) without Java.
Slowest of the three parser-based backends on CPU. Produces the largest
vocabulary because stanza's NER model is more type-fine-grained than CoreNLP.

::: lmsy_w2v_rfs.preprocessors.stanza_pp.StanzaPreprocessor

## CoreNLPPreprocessor

Paper-exact reproduction path. Holds a warm Stanford CoreNLP JVM open via
`stanza.server.CoreNLPClient` and fans requests across its thread pool. Use as
a context manager to guarantee server shutdown. Requires Java 8+ and the
`[corenlp]` extra.

::: lmsy_w2v_rfs.preprocessors.corenlp_pp.CoreNLPPreprocessor

## SpacyPreprocessor

Recommended default. Parses, lemmatizes, masks entities, and joins
`fixed` / `flat` / `compound` dependency pairs. On the 150-document bakeoff it
was 9x faster than stanza and 2x faster than CoreNLP, with the cleanest NER
output and the smallest Word2Vec-ready vocabulary.

::: lmsy_w2v_rfs.preprocessors.spacy_pp.SpacyPreprocessor

## load_mwe_list

Loads a curated MWE list, either the packaged `"finance"` list or a
newline-delimited file at a user-supplied path.

::: lmsy_w2v_rfs.load_mwe_list

## apply_mwe_list

Applies NLTK `MWETokenizer` to each sentence as a post-pass. Splits around
`[NER:*]` placeholders so MWE patterns do not match across entity boundaries.

::: lmsy_w2v_rfs.apply_mwe_list
