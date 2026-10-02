# Resume after a crash

## Problem

Your pipeline crashed three hours into the parse stage and you do not want
to redo the CoreNLP parse. Or the train stage OOM-killed halfway through and
you are unsure which artifacts are salvageable. Or you changed one Config
field and do not know which stages will rerun. The pipeline persists every
stage's output to `work_dir/`, and every stage is idempotent: rerunning the
exact same command picks up where you left off, skipping stages whose
artifacts already exist.

## Solution

Rerun the same command. To force re-execution of a specific stage, delete its
output file (or pass `force=True` for the whole pipeline).

```python
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds

seeds = load_example_seeds("culture_2021")
p = Pipeline(
    texts=my_texts,
    doc_ids=my_ids,
    work_dir="runs/my_experiment",
    config=Config(seeds=seeds, preprocessor="corenlp", n_cores=8),
)
p.run()            # first time: runs every stage
p.run()            # after a crash: skips stages whose outputs exist
```

Each stage logs either `stage: reusing path/to/output` (skipped) or starts a
tqdm bar (executing). No code change between runs.

### Rebuild a stage and selected downstream outputs

Each stage checks its own output files independently. Deleting `w2v.mod` reruns training, but an existing dictionary and existing score files are still reused. Use a new work directory when changing the corpus or configuration.

If you intend to rebuild the dictionary from a retrained model, explicitly force each affected stage:

```python
p.train(force=True)
p.expand_dictionary(force=True)
p.score(force=True)
p.word_contributions("TFIDF", force=True)
```

`expand_dictionary(force=True)` replaces the saved dictionary, including manual curation. Back up the curated CSV first if you need to retain it. If you intentionally keep the existing curated dictionary, skip re-expansion and explicitly rescore:

```python
p.train(force=True)
p.reload_dictionary()
p.score(force=True)
```

Changing parsing or phrase settings similarly requires explicit forcing of all affected stages. `p.run(force=True)` rebuilds every stage and replaces the dictionary. There is no automatic downstream invalidation after retraining or deleting a stage artifact.

### Force re-execution of the whole pipeline

```python
p.run(force=True)           # redo every stage regardless of existing outputs
```

Or just delete the entire `work_dir/` and start fresh.

## The work_dir layout

```
runs/my_experiment/
├── config.json                           dumped Config for audit
├── parsed/
│   ├── sentences.txt                     one lemmatized sentence per line,
│   │                                     NER masked, MWEs joined by underscore
│   └── sentence_ids.txt                  matching IDs shaped doc_id_sentN
├── cleaned/
│   └── sentences.txt                     stopwords and punctuation dropped
├── corpora/
│   ├── pass1.txt                         after gensim bigram Phrases
│   └── pass2.txt                         after two Phrases joining passes
├── models/
│   ├── w2v.mod                           trained Word2Vec (gensim format)
│   └── phrases_pass1.mod / pass2.mod     fitted Phrases models
└── outputs/
    ├── expanded_dict.csv                 per-dimension ranked word lists
    ├── scores_TF.csv                     document-level TF scores
    ├── scores_TFIDF.csv                  document-level TFIDF scores
    └── scores_WFIDF.csv                  document-level WFIDF scores
```

One sentence per file:

- `parsed/sentences.txt`: Phase 1a output. Token streams for every sentence.
- `parsed/sentence_ids.txt`: parallel file with `doc_id_sentN` IDs so scoring
  can reassemble documents.
- `cleaned/sentences.txt`: Phase 1a output with stopwords, punctuation, and
  1-letter tokens removed. Input to Phase 2.
- `corpora/pass{1,2}.txt`: Phase 2 output from gensim `Phrases`. The file
  suffix matches `Config.phrase_passes`.
- `models/w2v.mod`: trained Word2Vec model; load with `gensim.models.Word2Vec.load`.
- `outputs/expanded_dict.csv`: the per-dimension dictionary after nearest-
  neighbor expansion. The CSV used on a rerun to skip re-expansion.
- `outputs/scores_{METHOD}.csv`: one CSV per scoring method requested.

### Which stage wrote what

| Stage | Reads | Writes |
|---|---|---|
| `parse` | `texts`, `doc_ids` | `parsed/sentences.txt`, `parsed/sentence_ids.txt` |
| `clean` | `parsed/sentences.txt` | `cleaned/sentences.txt` |
| `phrase` | `cleaned/sentences.txt` | `corpora/pass{1,2}.txt`, `models/phrases_pass{1,2}.mod` |
| `train` | `corpora/pass{N}.txt` (or `cleaned/sentences.txt`) | `models/w2v.mod` |
| `expand_dictionary` | `models/w2v.mod` | `outputs/expanded_dict.csv` |
| `score` | `corpora/pass{N}.txt` (or `cleaned/sentences.txt` when `use_gensim_phrases=False`), `parsed/sentence_ids.txt`, `expanded_dict.csv` | `outputs/scores_{METHOD}.csv` |

Deleting a file reruns only the stage that checks that file. Explicitly force any dependent stages whose outputs must change.

## Gotcha: config changes do not invalidate artifacts

The pipeline checks for file existence, not for "did the Config that produced
this file match the current Config." If you change `w2v_epochs` from 20 to
40 and rerun, `train` will skip because `w2v.mod` exists. Delete the model
file (or pass `force=True` to `train` or `run`) to pick up config changes.

The dumped `config.json` in `work_dir/` is an audit trail; the pipeline never reads it back to decide what to rerun.

## Gotcha: partial writes

The `parse` and `clean` stages write to a temporary file and atomically rename
it on success, so an interruption while writing leaves temporary files instead of truncated published output. Parse publishes its sentence and ID files separately; if publication itself is interrupted, delete both parsed outputs before rerunning. The `phrase` and `train` stages write their
model files directly, so a crash mid-write *can* leave a corrupt `models/w2v.mod`
or `models/phrases_pass*.mod`. If a run crashed during those stages, delete the
suspect model file (or pass `force=True`) and rerun.

## Related

- [Run on HPC](run-on-hpc.md)
- [Switch the preprocessor](switch-preprocessor.md)
