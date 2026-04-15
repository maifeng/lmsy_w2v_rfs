# Dictionary expansion

The seed-expansion kernel grows each dimension's seed word list into a ranked
dictionary using the trained Word2Vec model. The logic mirrors the RFS 2021
replication repo's `culture/culture_dictionary.py` step by step, ported to the
gensim 4 API (`model.wv.key_to_index`). `Pipeline.expand_dictionary` wires
these functions together; the same primitives are exported for researchers
who want to run the expansion against their own Word2Vec model.

## expand_words_dimension_mean

Averages in-vocab seed vectors for each dimension, takes the top-k
nearest neighbors, and filters out NER tokens and cross-dimension seeds.

::: lmsy_w2v_rfs.expand_words_dimension_mean

## deduplicate_keywords

Assigns words that loaded onto multiple dimensions to their single most similar
dimension.

::: lmsy_w2v_rfs.dictionary.deduplicate_keywords

## rank_by_similarity

Sorts each dimension's words by cosine similarity to its seed mean.

::: lmsy_w2v_rfs.dictionary.rank_by_similarity

## similarity_weights

Computes the `1 / ln(2 + rank)` per-word weights used by the `TFIDF+SIMWEIGHT`
and `WFIDF+SIMWEIGHT` scoring methods. See [Scoring](scoring.md).

::: lmsy_w2v_rfs.dictionary.similarity_weights

## read_dict_csv

Reads a dictionary CSV produced by `write_dict_csv` back into a
`(dimension_to_words, all_words)` tuple.

::: lmsy_w2v_rfs.read_dict_csv

## write_dict_csv

Writes an expanded dictionary to CSV with one column per dimension.

::: lmsy_w2v_rfs.write_dict_csv
