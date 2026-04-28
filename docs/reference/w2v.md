# Word2Vec

Thin wrappers around gensim 4's `Word2Vec`. `train_word2vec` streams the
corpus from disk via `PathLineSentences` so the full text never has to fit
in RAM. All hyperparameters (vector size, window, min count, epochs, worker
count, random seed) come from the [`Config`](config.md) dataclass.
`Pipeline.train` calls `train_word2vec` when no saved model exists and
`load_word2vec` otherwise.

---

## train_word2vec

Fits a `gensim.models.Word2Vec` on a sentence file and saves the result to
disk. The corpus is streamed line by line, so the full training set never
has to fit in RAM.

Example:

```python
from pathlib import Path
from lmsy_w2v_rfs import Config, load_example_seeds
from lmsy_w2v_rfs.w2v import train_word2vec

seeds = load_example_seeds("culture_2021")
cfg = Config(seeds=seeds, preprocessor="none", w2v_dim=100, w2v_epochs=5)
model = train_word2vec(
    sentences_path=Path("runs/demo/corpora/pass2.txt"),
    model_path=Path("runs/demo/models/w2v.mod"),
    config=cfg,
)
print(model.wv.most_similar("innovation", topn=5))
```

::: lmsy_w2v_rfs.train_word2vec

---

## load_word2vec

Loads a model saved by `train_word2vec`. Use this to restore a model from a
previous run without retraining.

Example:

```python
from pathlib import Path
from lmsy_w2v_rfs.w2v import load_word2vec

model = load_word2vec(Path("runs/demo/models/w2v.mod"))
print(model.wv["innovation"])   # 100-dim vector
```

::: lmsy_w2v_rfs.load_word2vec
