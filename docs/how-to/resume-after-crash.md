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

### Force re-execution of one stage

Delete its output file. The pipeline will detect the missing artifact and
rerun only that stage plus everything downstream.

```bash
# Redo just the Word2Vec training, keeping parse / clean / phrase outputs.
rm runs/my_experiment/models/w2v.mod

python -c "
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds
seeds = load_example_seeds('culture_2021')
p = Pipeline(texts=..., doc_ids=..., work_dir='runs/my_experiment',
             config=Config(seeds=seeds, preprocessor='corenlp'))
p.run()
"
```

Equivalent stage-by-stage calls if you want finer control:

```python
p.parse()                   # skips if runs/.../parsed/sentences.txt exists
p.clean()                   # skips if runs/.../cleaned/sentences.txt exists
p.phrase()                  # skips if runs/.../corpora/pass2.txt exists
p.train(force=True)         # always retrains Word2Vec
p.expand_dictionary(force=True)
p.score(force=True)
```

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
│   └── pass2.txt                         after bigram + trigram Phrases
├── models/
│   ├── w2v.mod                           trained Word2Vec (gensim format)
│   └── phrases_pass1.pkl / pass2.pkl     fitted Phrases models
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
| `phrase` | `cleaned/sentences.txt` | `corpora/pass{1,2}.txt`, `models/phrases_pass{1,2}.pkl` |
| `train` | `corpora/pass{N}.txt` (or `cleaned/sentences.txt`) | `models/w2v.mod` |
| `expand_dictionary` | `models/w2v.mod` | `outputs/expanded_dict.csv` |
| `score` | `corpora/pass{N}.txt` (or `cleaned/sentences.txt` when `use_gensim_phrases=False`), `parsed/sentence_ids.txt`, `expanded_dict.csv` | `outputs/scores_{METHOD}.csv` |

Deleting a file forces that stage and all downstream stages to rerun.

## Gotcha: config changes do not invalidate artifacts

The pipeline checks for file existence, not for "did the Config that produced
this file match the current Config." If you change `w2v_epochs` from 20 to
40 and rerun, `train` will skip because `w2v.mod` exists. Delete the model
file (or pass `force=True` to `train` or `run`) to pick up config changes.

The dumped `config.json` in `work_dir/` is an audit trail, not a cache key.

## Gotcha: partial writes

A crash during `parse` can leave `parsed/sentences.txt` truncated. The next
run will see the partial file and try to reuse it, skipping the parse stage
and producing silently-wrong downstream output. When in doubt after a hard
crash, delete the suspect file and rerun.

## Related

- [Run on HPC](run-on-hpc.md)
- [Switch the preprocessor](switch-preprocessor.md)
