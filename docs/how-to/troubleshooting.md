# Troubleshooting

Common errors encountered when running `lmsy_w2v_rfs` and the exact resolution for each.

---

## `ImportError: cannot import name 'StanzaPreprocessor'`

**Symptoms:**

```
ImportError: cannot import name 'StanzaPreprocessor' from 'lmsy_w2v_rfs.preprocess'
```

**Cause:** The base install does not pull `stanza` or its transitive dependencies. Stanza-specific modules are gated behind the `[stanza]` extra so that a minimal install stays small.

**Fix:** Install the extra and rerun:

```bash
pip install "lmsy_w2v_rfs[stanza]"
```

The same pattern applies to `CoreNLPPreprocessor` (`[corenlp]` extra) and `SpacyPreprocessor` (`[spacy]` extra plus `python -m spacy download en_core_web_sm`). Install all three with `pip install "lmsy_w2v_rfs[all]"`.

---

## `Java not found on PATH`

**Symptoms:**

```
RuntimeError: Java runtime not found on PATH. CoreNLP requires Java 8 or later.
```

**Cause:** The `corenlp` preprocessor spawns a JVM subprocess via `stanza.server.CoreNLPClient`. Without `java` on `$PATH`, the client cannot launch.

**Fix:** Install a JRE or switch to a Python-only backend:

```bash
brew install openjdk@21          # macOS
apt install default-jre           # Debian/Ubuntu
java -version                     # verify

# Or switch to a Java-free backend:
pip install "lmsy_w2v_rfs[spacy]" && python -m spacy download en_core_web_sm
# then set Config(preprocessor="spacy")
```

---

## `CoreNLP server failed to start`

**Symptoms:**

```
stanza.server.client.StartServerError: CoreNLP server failed to start on port 9002
```

**Cause:** Three common causes, in order of frequency: (1) port 9002 is already in use by another CoreNLP instance or an unrelated process, (2) insufficient JVM heap memory, (3) the CoreNLP archive was never downloaded so the JVM cannot find the model jars.

**Fix:**

```bash
lsof -i :9002                           # who has the port? (default is 9002; adjust if you set corenlp_port)
lmsy-w2v-rfs download-corenlp           # (re)download the ~1 GB archive
ls ~/.cache/lmsy_w2v_rfs/corenlp/       # verify jars are present
```

Pass `Config(corenlp_memory="8G")` if the default `"6G"` is too small for long documents.

---

## `ValueError: preprocessor='static' needs config.mwe_list`

**Symptoms:**

```
ValueError: preprocessor='static' needs config.mwe_list to be set
```

**Cause:** The `static` preprocessor applies NLTK's `MWETokenizer` against a user-supplied list. With no list, it has nothing to match.

**Fix:** Pass either the packaged `finance` list or a path to your own newline-delimited file:

```python
from lmsy_w2v_rfs import Config, load_example_seeds

seeds = load_example_seeds("culture_2021")  # or any dict[str, list[str]]
Config(seeds=seeds, preprocessor="static", mwe_list="finance")            # packaged example
Config(seeds=seeds, preprocessor="static", mwe_list="/path/mwes.txt")     # your own list
```

The packaged `finance` list is hand-curated for earnings-call text; it is not appropriate for other domains. Write your own list or switch to a parser-based backend for domain-agnostic MWE detection.

---

## `KeyError` in `expand_words_dimension_mean`

**Symptoms:**

```
KeyError: "word 'shoulder_to_shoulder' not in Word2Vec vocabulary"
```

**Cause:** A seed word is missing from the trained Word2Vec vocabulary. This usually means the word was dropped by `min_count` filtering, or the MWE variant the seed expects (e.g., `shoulder_to_shoulder`) was never joined during preprocessing.

**Fix:** Lower `w2v_min_count` or enlarge the corpus:

```python
from lmsy_w2v_rfs import Config, load_example_seeds

seeds = load_example_seeds("culture_2021")  # or any dict[str, list[str]]
Config(seeds=seeds, w2v_min_count=3)           # default is 5; lowering keeps rare words
```

If the missing token is an MWE, verify that Phase 1 actually joined it. `preprocessor="corenlp"` has the highest MWE recall; switching from `spacy` or `none` often resolves this. You can also add the phrase to a static list and enable the static post-pass with `mwe_list="/path/mwes.txt"`.

---

## `My spaCy run is slower than expected`

**Symptoms:** spaCy at `n_process=8` takes much longer than the 3.9 min benchmark on a 1,393-doc corpus. `htop` shows CPU utilization well above 800%.

**Cause:** BLAS thread oversubscription. spaCy spawns N worker processes, each of which initializes PyTorch or NumPy, each of which spawns its own BLAS thread pool. Eight workers times eight BLAS threads equals 64 threads contending for 8 cores.

**Fix:** Pin BLAS to one thread per worker before the spaCy call:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python your_script.py
```

The `SpacyPreprocessor` sets `torch.set_num_threads(1)` at construction time, but the environment variables must be exported before the Python process starts for NumPy's BLAS pool to honor them.

---

## `Pipeline.parse took much longer than the benchmark suggested`

**Symptoms:** An 8-worker CoreNLP run takes 70+ minutes on 1,393 documents instead of the expected 13 minutes. CPU utilization stays near 100% instead of 700-800%.

**Cause:** Older versions of `Pipeline.parse()` submitted documents to the preprocessor one at a time, even when the backend supported concurrent submission. The JVM had eight threads allocated but only one was ever busy.

**Fix:** Upgrade to v0.1.0a1 or later, which adds a `Preprocessor.process_documents` method that CoreNLP and spaCy override with their native concurrency models (ThreadPoolExecutor around `client.annotate` for CoreNLP; `nlp.pipe(n_process=N)` for spaCy):

```bash
pip install --upgrade "lmsy_w2v_rfs[corenlp]"
python -c "import lmsy_w2v_rfs; print(lmsy_w2v_rfs.__version__)"
```

Verify your CPU utilization climbs above 500% within a minute of starting `pipeline.parse()`. If it stays below 200%, the old serial path is still being used.
