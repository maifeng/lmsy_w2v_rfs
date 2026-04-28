# Install the CoreNLP backend

## Problem

The default preprocessor is `"corenlp"` because it reproduces the 2021 paper's
Phase 1 behavior exactly and gives the best syntactic MWE coverage on the
benchmark (76%, versus 57% for stanza and 0% for spaCy). Getting it running
takes three things working at once: a Java 8+ runtime on `$PATH`, the
`[corenlp]` extra installed, and the ~1 GB CoreNLP archive expanded into the
cache directory. If any of the three is missing, the first call to the parser
fails with a cryptic JVM or import error.

## Solution

Install Java, install the extra, run the one-time downloader, smoke-test.

### 1. Install a Java runtime

```bash
# macOS
brew install openjdk@21

# Debian / Ubuntu
sudo apt install default-jre

# Verify
java -version
```

Any Java 8 or newer works. Stanford publishes CoreNLP against Java 8 minimum.

### 2. Install the package with the `[corenlp]` extra

```bash
pip install "lmsy_w2v_rfs[corenlp]"
```

The extra pulls in `stanza` and `protobuf`. The base install does not, so
`preprocessor="none"` and `preprocessor="static"` work without any of this.

### 3. Download the CoreNLP archive

```bash
lmsy-w2v-rfs download-corenlp
```

This calls `stanza.install_corenlp()` under the hood. The archive lands in
`~/.cache/lmsy_w2v_rfs/corenlp/` by default. Override the location with the
`LMSY_W2V_RFS_HOME` environment variable:

```bash
export LMSY_W2V_RFS_HOME=/scratch/$USER/lmsy_cache
lmsy-w2v-rfs download-corenlp
```

Disk footprint: ~1 GB for the zip, ~1.5 GB expanded. The downloader also sets
`CORENLP_HOME` in the current process, but you should not rely on that across
shells.

### 4. Smoke test

```python
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds

seeds = load_example_seeds("culture_2021")
p = Pipeline(
    texts=["Innovation and teamwork drive our roadmap at Apple Inc."],
    doc_ids=["doc1"],
    work_dir="runs/smoke",
    config=Config(seeds=seeds, preprocessor="corenlp", n_cores=2, use_gensim_phrases=False),
)
p.parse()
print((p.work_dir / "parsed" / "sentences.txt").read_text())
```

Expected output is one line of lemmatized, NER-masked tokens where `Apple Inc.`
has been replaced by a `[NER:ORGANIZATION]` placeholder. First call is slow:
the JVM loads pretrained models on startup (several seconds). Subsequent calls
on the same server are fast.

## What if Java is not available

Switch to a Python-only backend. Install the corresponding extra and set
`preprocessor=` accordingly.

```python
# Python-native, fastest parser, best NER. Drops syntactic MWE coverage.
from lmsy_w2v_rfs import Pipeline, Config, load_example_seeds

seeds = load_example_seeds("culture_2021")
p = Pipeline(
    texts=[...],
    doc_ids=[...],
    work_dir="runs/nojava",
    config=Config(seeds=seeds, preprocessor="spacy", spacy_model="en_core_web_sm", n_cores=8),
)
```

```bash
pip install "lmsy_w2v_rfs[spacy]"
python -m spacy download en_core_web_sm
```

Stanza is the middle ground (Python-native, keeps 57% of syntactic MWEs, slow
on CPU):

```bash
pip install "lmsy_w2v_rfs[stanza]"
```

See [Switch the preprocessor](switch-preprocessor.md) for the full trade-off
matrix.

## Notes

- The downloader is idempotent. Rerunning it refreshes the cache but does not
  re-download files that are already present and valid.
- Port 9002 is the default for the embedded JVM server. Change it via
  `Config(corenlp_port=...)` if another service is listening there.
- `Config(corenlp_memory="6G")` sets the JVM heap. Lower to `"2G"` on laptops
  with tight memory budgets. The parser still works, it just caches fewer
  pretrained models at once.
