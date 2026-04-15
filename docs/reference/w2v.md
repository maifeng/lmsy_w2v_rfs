# Word2Vec

Thin wrappers around gensim 4's `Word2Vec`. `train_word2vec` streams the corpus
from disk via `PathLineSentences` so the full text never has to fit in RAM.
All hyperparameters (vector size, window, min count, epochs, worker count,
random seed) come from the [`Config`](config.md) dataclass. `Pipeline.train`
calls `train_word2vec` when no saved model exists and `load_word2vec`
otherwise.

## train_word2vec

::: lmsy_w2v_rfs.train_word2vec

## load_word2vec

::: lmsy_w2v_rfs.load_word2vec
