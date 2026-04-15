# Preprocessor comparison

This page summarizes the empirical study behind the five preprocessor backends. The point is to let a researcher pick a backend based on numbers rather than guesswork.

---

## What we measured

Three complementary tests, all run on the same hardware (Apple Silicon M-series CPU, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`):

1. **60-phrase MWE test**. Hand-labeled gold MWEs across six categories: idiomatic (10), grammaticalized fixed (13), compound nouns (10), business jargon (11), phrasal verbs (8), named entities (8). A backend "catches" a phrase if it proposes a join whose endpoints both fall inside the gold MWE.
2. **50-sentence NER test**. Hand-labeled entity spans and types across person, organization, location, money, date, and other categories. Scored on both span recall and type accuracy.
3. **300-document end-to-end bakeoff**. Real earnings-call transcripts, 6.7 MB, parsed with each backend at `n_cores=8`. Measured wall time, CPU utilization, entity count, and sentence count.

---

## Results

### Syntactic MWE recall (21-phrase subset: fixed + phrasal)

| Backend | fixed (13) | phrasal (8) | syntactic total (21) |
|---|---|---|---|
| CoreNLP 4.5 | 8/13 | 8/8 | **16/21 (76%)** |
| stanza (EWT) | 4/13 | 8/8 | 12/21 (57%) |
| spaCy-trf | 0/13 | 0/8 | 0/21 (0%) |
| static (NLTK `MWETokenizer` + finance list) | 13/13 on list | 8/8 on list | 21/21 on list, 0 off list |

### NER quality

| Backend | Span recall | Type accuracy |
|---|---|---|
| spaCy-trf | 100% | 96% |
| stanza | 98% | 88% |
| CoreNLP 4.5 | 98% | 78% |

### Throughput on 1,393-doc corpus at 8 workers

| Backend | Wall time | Speedup over 1 worker | Docs/s |
|---|---|---|---|
| spaCy sm (`n_process=8`) | 3.9 min | 3.47x | 5.92 |
| spaCy md (`n_process=8`) | 5.0 min | 2.83x | 4.67 |
| CoreNLP 4.5 (`threads=8`) | 11.7 min | 5.74x | 1.98 |
| stanza (CPU) | ~5 hours | n/a | very low |

CoreNLP scales best because its JVM threads share one loaded model; spaCy's Python multiprocessing pays for per-worker `Doc` pickling. spaCy is faster in absolute terms because its per-thread throughput is 3x higher; CoreNLP wins on MWE quality.

---

## Why UD v2 dropped `mwe` and split into `fixed` / `flat` / `compound`

The old UD v1 `mwe` relation was renamed, not replaced. UD v2's [thematic report](https://universaldependencies.org/v2/mwe.html) makes the reasoning explicit: `mwe` was being misread as a catch-all for any multi-word expression, when it was only ever intended for grammaticalized fixed phrases like `because_of` or `as_well_as`. Three labels now cover disjoint MWE families:

- `fixed`: fully grammaticalized, morphosyntactically rigid phrases that behave like function words or short adverbials. `with_respect_to`, `in_spite_of`, `because_of`. The inventory is small and closed.
- `flat`: head-less MWEs, mostly proper names. `New_York_City`, `Barack_Obama`. Absorbs the old `name` and `foreign` labels.
- `compound`: compound nouns (`phone_book`), phrasal verbs via `compound:prt` (`look_up`, `roll_out`), and serial verbs.

Idioms like `beat_a_dead_horse` or `piece_of_cake` get no UD edge at all under v2. That is by design: UD is a dependency scheme, not an idiom inventory. The [PARSEME project](https://parsemefr.lis-lab.fr/parseme-st-guidelines/1.1/) publishes a separate layer for verbal MWEs, but no production pip package implements it.

Practical consequence: any parser-based preprocessor inherits this ceiling. If you want idiom-level MWE recall, pair the parser with a static list.

---

## What each backend is best at

**CoreNLP 4.5** is best for paper-faithful reproduction and maximum syntactic MWE coverage. Its PTB-to-UD rule converter encodes more `fixed` patterns than any treebank-trained model has seen. Its JVM thread pool scales near-linearly. Its NER type accuracy is the weakest of the three parsers (78%), but our downstream use only needs entity spans for masking, so this matters less than it looks. Worst at: install friction. Needs Java 8+ and a 1 GB download.

**stanza (EWT)** is the Python-native middle ground. Strong on compounds and phrasal verbs, weaker on `fixed` patterns (4/13). Best when you need POS tags and a modern neural parser without Java. Worst at: CPU throughput. Stanza on CPU takes ~5 hours on the full 1,393-doc corpus because PyTorch neural parsing is slow without GPU acceleration, and stanza does not yet support Apple Silicon MPS.

**spaCy** is the fastest backend by a wide margin (3.9 min on 1,393 docs with `en_core_web_sm`). Best NER by span recall (100%) and type accuracy (96%). Worst at: syntactic MWE. The English model's ClearNLP-to-UD converter does not emit `fixed` or `compound:prt` at all, so syntactic MWE recall is 0%. spaCy is a good default for workshop and classroom settings where Java is unavailable and where Phase 2 (gensim `Phrases`) plus an optional static list can pick up the MWE slack.

**static** is a precision tool backed by NLTK's `MWETokenizer`. 100% recall on the list, 0% off the list. Best when you know the exact set of MWEs you care about. Worst at: discovery. It cannot find MWEs you did not already think of.

**none** is whitespace tokenize plus lowercase, no parser, no NER masking. Best when your text is already pre-lemmatized and entity-masked. Worst at: everything else. Included for pipelines that want to bring their own tokenization.

---

## When to switch from the default

Switch from `corenlp` (the default) to:

- **`spacy`** when Java is unavailable (Colab classroom, slim Docker image, CI runner), when wall time matters more than MWE quality, or when the downstream analysis needs the strongest NER masking. Pair with `use_gensim_phrases=True` and optionally a curated `mwe_list` to recover some MWE coverage.
- **`stanza`** when you need a Python-native pipeline with modern UD labels and do not need the JVM. Only viable on GPU or on small corpora (< 100 docs) if you are on CPU. Stanza is actively developed at Stanford NLP and receives new neural capabilities first; CoreNLP is in maintenance mode.
- **`static`** when your domain has a known set of MWEs and you want deterministic, debuggable behavior. Combine with `preprocessor="static"` and pass `mwe_list="finance"` or your own path.
- **`none`** when your text is already tokenized and lemmatized upstream, or when you are iterating on the Word2Vec stage and want to eliminate parsing time.

