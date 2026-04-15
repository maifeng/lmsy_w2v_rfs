# Run on an HPC compute node

## Problem

You have a 100k-document corpus and an HPC allocation with 16, 32, or 64 CPU
cores. Running the pipeline unchanged wastes most of that hardware. The CoreNLP
backend defaults to 4 JVM threads. spaCy defaults to 4 Python workers. Even
worse, scientific-Python stacks fork with BLAS thread pools that oversubscribe
the machine and slow everything down. A 13-minute job can silently become
74 minutes when parallelism is set up wrong. This page walks through the fix.

## Solution

Three levers, in order: set `n_cores` correctly, cap BLAS threads before any
worker forks, shard very large corpora into 10k-doc files. SLURM and SGE
templates tie it all together.

### 1. Set `n_cores` to the allocation

```python
import os
from lmsy_w2v_rfs import Pipeline, Config

N = int(os.environ.get("SLURM_CPUS_PER_TASK",
         os.environ.get("NSLOTS",                 # SGE
         os.cpu_count())))

cfg = Config(
    preprocessor="corenlp",
    n_cores=N,           # JVM threads for CoreNLP; process count for spaCy / stanza
    corenlp_memory="12G" if N >= 16 else "6G",
)

p = Pipeline.from_text_file("shard_000.txt", work_dir="runs/shard_000", config=cfg)
p.run()
```

Measured numbers on the 1,393-doc RFS 2021 sample corpus:

| Backend | `n_cores=1` | `n_cores=4` | `n_cores=8` | Speedup |
|---|---|---|---|---|
| CoreNLP | 67.4 min | 18.3 min | **11.7 min** | 5.74x |
| spaCy sm | 13.6 min | 5.9 min | **3.9 min** | 3.47x |

CoreNLP scales nearly linearly to 8 threads because the JVM shares one model
across the pool. spaCy plateaus around 8 workers because of `Doc`-pickling
IPC overhead. Beyond 8, gains are small. For 32-core nodes, shard the corpus
and run multiple pipelines in parallel (see step 3).

### 2. Cap BLAS threads before Python imports anything

Scientific Python pulls in NumPy, scikit-learn, and (for stanza) PyTorch, all
of which spawn their own BLAS thread pools at import time. When 8 worker
processes each spawn 8 BLAS threads on an 8-core machine, the kernel schedules
64 threads on 8 cores and the machine thrashes.

Set these BEFORE any Python import:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

The spaCy backend also sets `torch.set_num_threads(1)` inside the worker
bootstrap; the others do not. Set these env vars anyway. They are cheap and
the failure mode is silent slowdown.

### 3. Shard large corpora

The pipeline holds the full text list in memory and streams sentences to disk
stage by stage. RAM grows linearly with corpus size. For corpora beyond
~100k documents, shard the input by 10k and run one pipeline per shard.
Aggregate at scoring time with `pandas.concat`.

```python
import math
from pathlib import Path
from lmsy_w2v_rfs import Pipeline, Config

ALL = Path("corpus.txt").read_text().splitlines()
SHARD = 10_000
for i in range(math.ceil(len(ALL) / SHARD)):
    chunk = ALL[i * SHARD : (i + 1) * SHARD]
    Path(f"shards/shard_{i:03d}.txt").write_text("\n".join(chunk))
```

Each shard gets its own `work_dir`. Train Word2Vec on the concatenated
training corpus rather than per-shard: scoring does not benefit from sharding,
dictionary expansion needs the global vocab. See the SLURM template below.

### 4. SLURM template

```bash
#!/bin/bash
#SBATCH --job-name=lmsy_w2v_parse
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/parse_%A_%a.out

# Cap BLAS threads before Python starts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Point CoreNLP cache at scratch so the shared home quota is not burned.
export LMSY_W2V_RFS_HOME=/scratch/$USER/lmsy_cache

# Run one shard per array task. Each task parses, cleans, and writes a
# per-shard parsed/ directory. A second job concatenates and trains Word2Vec.
module load java/21
source .venv/bin/activate

python -c "
import os
from lmsy_w2v_rfs import Pipeline, Config
i = int(os.environ['SLURM_ARRAY_TASK_ID'])
cfg = Config(preprocessor='corenlp', n_cores=8, corenlp_memory='12G',
             corenlp_port=9002 + i)
p = Pipeline.from_text_file(f'shards/shard_{i:03d}.txt',
                            work_dir=f'runs/shard_{i:03d}', config=cfg)
p.parse()
p.clean()
"
```

`corenlp_port` is offset by `i` so each array task gets its own JVM server.
Ports collide if multiple tasks land on the same node; this avoids that.

### 5. SGE template (University of Iowa Argon and similar)

SGE uses `qsub` and `NSLOTS` instead of SLURM's `srun` and `SLURM_CPUS_PER_TASK`.
Same pipeline, different job scheduler.

```bash
#!/bin/bash
#$ -N lmsy_w2v_parse
#$ -t 1-10                      # 10-task array (SGE is 1-indexed, unlike SLURM)
#$ -pe smp 8                    # 8 cores per task
#$ -l mem_free=24G
#$ -l h_rt=04:00:00
#$ -q all.q                     # or UI-GPU / UI-MPI / UI-HM on Argon
#$ -o logs/parse_$JOB_ID_$TASK_ID.out
#$ -j y                         # merge stdout and stderr
#$ -cwd                         # run from submit directory
#$ -V                           # inherit environment variables

# Cap BLAS threads before Python starts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Point CoreNLP cache at scratch so the home quota is not burned.
export LMSY_W2V_RFS_HOME=/nfsscratch/$USER/lmsy_cache

# Load modules. On Argon adjust module names as needed.
module load stack/legacy
module load java/21
module load python/3.12

source .venv/bin/activate

# SGE task IDs are 1-indexed; shift to 0-index to match 000-099 filenames.
i=$((SGE_TASK_ID - 1))
port=$((9002 + i))

python -c "
import os
from lmsy_w2v_rfs import Pipeline, Config
i = int(os.environ['SGE_TASK_ID']) - 1
N = int(os.environ.get('NSLOTS', 8))
cfg = Config(preprocessor='corenlp', n_cores=N, corenlp_memory='12G',
             corenlp_port=9002 + i)
p = Pipeline.from_text_file(f'shards/shard_{i:03d}.txt',
                            work_dir=f'runs/shard_{i:03d}', config=cfg)
p.parse()
p.clean()
"
```

Submit with `qsub parse.sh`. Monitor with `qstat -u $USER`.

On the University of Iowa **Argon** cluster specifically:

- Shared home quota is small; use `/nfsscratch/$USER/` for the CoreNLP
  cache, phrase models, and intermediate `work_dir` output. Copy only the
  final `outputs/*.csv` to your project directory at the end.
- `-q all.q` is the default queue; `UI-HM` is the high-memory queue if
  you need `>64 GB` per task.
- Argon does not preload `java` or Python 3.12 by default. The `module
  load` lines above are the reproducible way to pin versions; check
  `module avail java` for the current module name (Argon renames
  occasionally).
- Argon is Linux, so fork-based multiprocessing works and spaCy workers
  share the model via copy-on-write.

## Gotcha: fork vs spawn on macOS

macOS defaults Python multiprocessing to `spawn`, which reloads the spaCy
model in every worker. Fork-based platforms (Linux) share the loaded model
via copy-on-write. Consequences:

- On Linux: `n_cores=8` with spaCy is memory-cheap because workers share the
  parent's loaded model. Memory per worker is small.
- On macOS: `n_cores=8` with spaCy allocates a full model copy per worker.
  Memory balloons (en_core_web_sm is ~50 MB per worker; en_core_web_trf is
  ~500 MB). On an M-series laptop with 16 GB, 8 trf workers can OOM.

Fixes:

- Use `en_core_web_sm` on macOS unless you have a workstation.
- Cap `n_cores` to 4 on Apple Silicon with 16 GB RAM.
- On Linux HPC, set `multiprocessing.set_start_method("fork")` explicitly if
  the cluster's Python was compiled with `spawn` as the default.

## Gotcha: CoreNLP JVM warm-up cost

The first request to a fresh JVM takes 5 to 15 seconds while it loads
pretrained models. Subsequent requests are fast. This matters for tiny
corpora (< 100 docs) where startup dominates wall time. Not an issue for HPC
workloads.

## Related

- [Install the CoreNLP backend](install-corenlp.md)
- [Switch the preprocessor](switch-preprocessor.md)
- [Resume after a crash](resume-after-crash.md)
