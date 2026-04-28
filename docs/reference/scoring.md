# Scoring

The scoring kernel turns a doc-level corpus and an expanded
[dictionary](dictionary.md) into document-level scores. Three base
methods are supported: `TF` (raw counts), `TFIDF` (`tf * log(N/df)`), and
`WFIDF` (`(1 + log tf) * log(N/df)`). Each can be combined with the
`similarity_weights` kernel to produce `TFIDF+SIMWEIGHT` and `WFIDF+SIMWEIGHT`.
The streaming design avoids materializing the full corpus in memory: document
frequencies and per-document text are built in a single pass over the sentence
file.

`Pipeline.score` chains these primitives together; the same functions are
exported for users who want to score a corpus directly.

## ScoringMethod

The string literal accepted by `score_document` and `score_documents`. Valid
values are `"TF"`, `"TFIDF"`, `"WFIDF"`, `"TFIDF+SIMWEIGHT"`, and
`"WFIDF+SIMWEIGHT"`.

::: lmsy_w2v_rfs.ScoringMethod

## iter_doc_level_corpus

Folds a sentence-level corpus back to document level by grouping consecutive
sentence lines whose IDs share the same `docID_` prefix.

::: lmsy_w2v_rfs.iter_doc_level_corpus

## document_frequencies

Computes the document-frequency dictionary and total document count needed by
every non-TF scoring method.

::: lmsy_w2v_rfs.document_frequencies

## score_document

Scores one document across all dimensions.

::: lmsy_w2v_rfs.score_document

## score_documents

Scores an iterable of `(doc_id, text)` pairs and returns a DataFrame with one
row per document.

::: lmsy_w2v_rfs.score_documents

## aggregate_to_firm_year

Joins document-level scores to a firm-year mapping, normalizes each dimension
by document length (per 100 tokens), and averages within each firm-year cell.

::: lmsy_w2v_rfs.aggregate_to_firm_year

## zca_whiten

Applies ZCA whitening to a matrix of document-level scores so that each
dimension has unit variance and the dimensions are decorrelated. Useful
when the seed dimensions overlap in embedding space.

::: lmsy_w2v_rfs.zca_whiten
