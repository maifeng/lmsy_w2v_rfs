# Whiten the dimension scores

## Problem

Your five culture dimensions are correlated. A firm that scores high on
integrity tends to score high on teamwork, because both pull on similar
vocabulary. This inflates the apparent signal when you use the scores as
regressors together. You want dimensions that are decorrelated, while
keeping column names interpretable.

## Solution

Apply ZCA (zero-phase component analysis) whitening as a post-scoring
step. ZCA is the whitening transform that makes the columns uncorrelated
with unit variance while staying as close as possible to the original
axes. Unlike PCA whitening, ZCA does not rotate the data into a new
basis: the column named ``integrity`` still measures something close to
integrity, not "principal component 1".

This is the same post-processing idea the
[Marketing-Measures/marketing-measures](https://github.com/Marketing-Measures/marketing-measures)
package uses for their firm-level measures.

### Turn it on in `Config`

```python
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds

seeds = load_example_seeds("culture_2021")
cfg = Config(seeds=seeds, zca_whiten=True)
p = Pipeline.from_csv("transcripts.csv", work_dir="runs/x", config=cfg)
p.run()
df = p.score_df("TFIDF")   # columns are already whitened
```

### Or whiten an existing scores DataFrame manually

```python
from lmsy_w2v_rfs import zca_whiten

df_whitened = zca_whiten(
    df, dims=["integrity", "teamwork", "innovation", "respect", "quality"],
)
```

### CLI

```bash
lmsy-w2v-rfs run --input docs.txt --out runs/x --zca-whiten
```

## Notes

- **In-sample fit**: the transform is computed from `scores[dims]`
  itself, so the decorrelation is exact on the data you pass in. If you
  want new documents to land on the same whitened scale, compute the
  whitening matrix on a reference corpus and cache it; apply the cached
  matrix to future scores. The current implementation does not split fit
  and transform; for now, fit on the full corpus you plan to analyze.
- **`epsilon`**: eigenvalue floor for numerical stability. Default
  `1e-6`. Raise to `1e-4` if the covariance is near-singular (small
  corpus, or dimensions that are nearly-degenerate).
- **Order of operations**: whitening runs after tf-idf weighting and
  after L2 normalization (if `Config(tfidf_normalize=True)` is set) but
  before `firm_year` aggregation. Firm-year means are over whitened
  document scores.
- **Interaction with firm-year aggregation**: the per-document divide-by-
  length and scale-to-per-100-tokens steps in `aggregate_to_firm_year`
  are applied to the whitened columns. If that is not what you want,
  skip `firm_year` and aggregate yourself.

## Related

- [Scoring](../concepts/scoring.md): the tf-idf formulas that produce
  the pre-whitening columns.
- [Reference: scoring](../reference/scoring.md): API docstring for
  `zca_whiten`.
