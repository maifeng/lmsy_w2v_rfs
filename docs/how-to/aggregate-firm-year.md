# Aggregate document scores to firm-year

## Problem

The pipeline scores every document independently: one row per transcript, one
column per culture dimension. Corporate-finance research usually wants
firm-year panels, not document panels. You need to (a) map each document to
its firm and fiscal year, (b) normalize by document length so long transcripts
do not dominate, (c) average across all documents within a firm-year, and (d)
merge the resulting panel with external firm covariates (total assets,
industry code, returns, whatever your regression calls for).

## Solution

Build an `id_to_firm` DataFrame with columns `document_id`, `firm_id`, and
`time`, then call `p.firm_year(id_to_firm, method="TFIDF")`. The method
normalizes scores to "per 100 tokens," averages within `(firm_id, time)`
groups, and returns a sorted panel.

### End-to-end example

```python
import pandas as pd
from lmsy_w2v_rfs import Pipeline, Config

# 1. Run the pipeline as usual.
p = Pipeline(
    texts=transcripts,
    doc_ids=transcript_ids,          # e.g., ["AAPL_2021Q1", "AAPL_2021Q2", ...]
    work_dir="runs/firm_panel",
    config=Config(preprocessor="corenlp", n_cores=8),
)
p.run(methods=("TFIDF",))

# 2. Build the document-to-firm-year mapping.
id_to_firm = pd.DataFrame({
    "document_id": ["AAPL_2021Q1", "AAPL_2021Q2", "AAPL_2021Q3", "AAPL_2021Q4",
                    "MSFT_2021Q1", "MSFT_2021Q2"],
    "firm_id":     ["AAPL",        "AAPL",        "AAPL",        "AAPL",
                    "MSFT",        "MSFT"],
    "time":        [2021,          2021,          2021,          2021,
                    2021,          2021],
})

# 3. Aggregate.
panel = p.firm_year(id_to_firm, method="TFIDF")
print(panel)
```

Expected shape: one row per `(firm_id, time)` combination, with the five
culture dimensions as columns. For the example above, two rows total (AAPL
2021, MSFT 2021), each the mean of the four or two document-level scores.

### What per-100-tokens normalization means

Raw scores scale with document length: a 5,000-word transcript mentions
`innovation` more often than a 500-word one, all else equal. To put all
documents on a comparable footing, `firm_year` divides each dimension by
`document_length` and multiplies by 100:

```
score_per_100_tokens = 100 * raw_score / document_length
```

The mean over `(firm_id, time)` is then taken on the normalized scores. If you
want the raw document scores for your own aggregation, pull them directly
from `p.score_df("TFIDF")` and skip `firm_year`.

### Merging with external firm covariates

A realistic research flow: take `panel`, merge a Compustat extract, run a
panel regression.

```python
import pandas as pd

covariates = pd.read_csv("compustat_firm_year.csv")
# covariates has columns: firm_id, time, at (total assets), roa, ind2

merged = panel.merge(covariates, on=["firm_id", "time"], how="inner")

# Standardize culture scores within year for readability of coefficients.
for dim in ["innovation", "integrity", "quality", "respect", "teamwork"]:
    merged[dim + "_z"] = merged.groupby("time")[dim].transform(
        lambda s: (s - s.mean()) / s.std()
    )

merged.to_parquet("panel_with_covariates.parquet")
```

From here you run your preferred fixed-effects regression in statsmodels,
linearmodels, or R. The pipeline's job ends at the panel export.

### Using a different aggregation window

`time` is an opaque label. Pass fiscal quarter (`2021Q1`), fiscal year
(`2021`), or decade (`"2020s"`) strings or integers; `firm_year` groups on
whatever is in the column. The method name is historical: the aggregation is
really "group by firm and time, whatever time means to you."

For a firm-quarter panel:

```python
id_to_firm = pd.DataFrame({
    "document_id": [...],
    "firm_id":     [...],
    "time":        ["2021Q1", "2021Q2", "2021Q3", ...],
})
panel = p.firm_year(id_to_firm, method="TFIDF")
```

## Gotcha: document IDs that do not match

`p.firm_year` does a left merge of document scores onto `id_to_firm`. Any
`doc_id` that is not in `id_to_firm["document_id"]` drops out at aggregation
(NaN firm_id and time produce a dropped group after `groupby`). Any
`document_id` in `id_to_firm` that has no matching score in `scores_TFIDF.csv`
is silently ignored. Sanity-check counts before running regressions:

```python
scores = p.score_df("TFIDF")
expected = set(scores["Doc_ID"])
mapped   = set(id_to_firm["document_id"])
print(f"scored but unmapped: {len(expected - mapped)}")
print(f"mapped but unscored: {len(mapped - expected)}")
```

## Gotcha: expected column names

The aggregator reads specific column names from `id_to_firm`. If your
DataFrame uses `gvkey` / `fyear`, rename before passing:

```python
id_to_firm = firm_data.rename(columns={
    "transcript_id": "document_id",
    "gvkey":         "firm_id",
    "fyear":         "time",
})
```

The lower-level `aggregate_to_firm_year` function takes `id_col=`,
`firm_col=`, `time_col=` kwargs if renaming in place is inconvenient, but
`Pipeline.firm_year` does not expose those and always expects the canonical
three names.

## Related

- [Use your own seeds](use-your-own-seeds.md)
- [Resume after a crash](resume-after-crash.md)
