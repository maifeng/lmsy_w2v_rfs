# Design decisions

This page explains the "why" behind the load-bearing choices in `lmsy_w2v_rfs`. The package packages a research pipeline from Li, Mai, Shen, Yan (2021), so several decisions are driven by paper-faithfulness. Others are driven by an empirical benchmark on 60 multi-word expressions, 50 NER sentences, and 300 earnings-call documents. See [MWE benchmark comparison](mwe-comparison.md) for the numbers.

---

## Why CoreNLP is the default preprocessor

CoreNLP is the default because it is the only backend that gets both high syntactic MWE coverage and strong multi-worker scaling on the benchmark we ran.

On the 60-phrase test set, CoreNLP 4.5 catches 16 of 21 syntactic MWEs (76%), compared with 12 of 21 for stanza (57%) and 0 of 21 for spaCy. The gap is in UD v2 `fixed` patterns like `with_respect_to`, `in_spite_of`, `due_to`, which CoreNLP's PTB-to-UD converter encodes as hand-written rules. Stanza and spaCy have to predict these from treebank training data where the patterns are sparse.

On throughput, CoreNLP's JVM thread pool scales 5.74x from 1 to 8 threads because all threads share one loaded model. That gets the full 1,393-document RFS 2021 sample corpus down to 13 minutes at `n_cores=8`. spaCy is faster in wall time (~4 minutes) but pays with 0% syntactic MWE recall. CoreNLP wins on the cost-benefit curve we care about.

The friction is real: you need Java 8+ on `$PATH` and a ~1 GB one-time archive download. We accept that friction because this is a research package, not a production tool. Users who cannot install Java have four other preprocessors to pick from.

---

## Why the two-phase design (syntactic + statistical)

The 2021 paper runs two MWE passes because neither alone is sufficient. Phase 1a (parser-based, syntactic) catches rare-but-meaningful MWEs like `with_respect_to` or `roll_out` that occur too infrequently for a statistical method to notice. Phase 2 (gensim `Phrases`) catches high-frequency domain-specific collocations like `forward_looking_statement` or `fourth_quarter` that no English parser will emit as a single MWE.

The paper's expanded culture dictionary contains tokens like `customer_commitment`, `shoulder_to_shoulder`, `hand_in_glove`, `world_class`. If Phase 1 fails to join them, the seed words never match anything in the Word2Vec vocabulary and the affected dimensions get systematically under-scored. If Phase 2 is skipped, corpus-specific jargon like `earnings_release` stays split across two tokens and loses its vector meaning. We ship both phases on by default and let users turn either off.

---

## Why the static MWE list is optional, not default

The package ships a hand-curated `finance` MWE list at `src/lmsy_w2v_rfs/data/mwes_finance.txt`. It is NOT loaded by default.

The list was assembled from three sources: UD v2 `fixed` prepositional phrases, earnings-call jargon from general knowledge, and the RFS 2021 paper's dictionary appendix. It is useful precisely because it is hand-curated for earnings-call text. That same property makes it a poor default for any other domain. A medical-notes pipeline with `finance` loaded silently would join `per_share` and `year_over_year` into opaque tokens with no corpus support.

gensim `Phrases` is the principled way to discover domain-specific MWEs: it looks at your corpus and picks up the collocations that actually appear. The static list is a deterministic override for cases where you already know the phrases you care about. Users opt in with `mwe_list="finance"` or by passing a path to their own file.

---

## Why we ship five preprocessor backends

Preprocessor selection is a trade-off surface with three axes: Java vs Python-only, fast vs faithful, deterministic vs learned. No single backend wins on all three.

| value | Java? | Syntactic MWE | Throughput at 8 workers |
|---|---|---|---|
| `none` | no | 0% | fastest |
| `static` | no | 100% on the list, 0% off the list | fast |
| `spacy` | no | 0% | 3.9 min on 1,393 docs |
| `stanza` | no | 57% | ~5 hours on 1,393 docs on CPU |
| `corenlp` | yes | 76% | 13 min on 1,393 docs |

Shipping all five lets users pick based on their actual constraint. Classroom Colab notebooks pick `spacy` because Java is absent. Paper-exact reproducers pick `corenlp`. Users with pre-lemmatized text pick `none`. The underlying Word2Vec, expansion, and scoring code is identical across backends, so switching one flag changes the upstream parser without touching the downstream analysis.

---

## Why `Pipeline` stages are idempotent and resumable

`Pipeline` exposes six stages (`parse`, `clean`, `phrase`, `train`, `expand_dictionary`, `score`) and writes each stage's output under `work_dir/`. A rerun of the same `work_dir` resumes from the latest stage that has complete artifacts; stages with `force=True` redo from scratch.

Researchers iterate. A Word2Vec run with the wrong `w2v_dim` should not force redo of the 13-minute CoreNLP parse. A prompt-engineering pass on the seed dictionary should not re-tokenize 1,393 documents. Forcing every stage to redo on every invocation is the single fastest way to make a research package unusable.

The idempotence is implemented by writing each stage's output to a stable path under `work_dir` and checking for its existence at stage entry. Simple, visible, easy to debug. Users can delete one artifact to rerun exactly one stage.

---

## Why seeds are a `dict[str, list[str]]` (and required)

Seeds are passed to `Config` as a plain Python dict; there is no built-in default. The package is theory-agnostic and refuses to assume what you are measuring.

```python
cfg = Config(seeds={
    "risk":   ["risk", "uncertainty", "volatility", "hedge"],
    "growth": ["growth", "expand", "expansion", "scale"],
})
```

No `SeedDictionary` class, no `Dimension` dataclass, no registry. Researchers working on domain-agnostic concepts should not have to import and subclass anything to experiment with a new seed set. Editing a dict in a notebook is the minimum possible friction.

The 2021 paper's 5-dim culture dictionary is shipped as a named example via `load_example_seeds("culture_2021")`. It is opt-in and clearly tagged as a reproduction artifact, never the default. The package stays useful across both that paper and the 2026 follow-up's 6-type taxonomy because the seed shape is the same plain dict.

---

## Why we use gensim 4

The 2021 research code was written against gensim 3.x, which does not run on Python 3.10 or later. The gensim 4 migration was forced: `Word2Vec.size` became `vector_size`, `iter` became `epochs`, `model.wv.vocab` became `model.wv.key_to_index`, `Phrases.common_terms` became `connector_words`.

We do the migration in the package so users do not have to. The public API is gensim-4-native throughout. If you have gensim-3 seed code or intermediate artifacts, they will not load; retrain from the corpus.
