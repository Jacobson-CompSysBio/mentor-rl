# MENTOR-RL

MENTOR-RL is a research codebase for training tool-using language models to do
mechanistic interpretation over biological multiplex networks. The goal is to
teach an agent to explore graph evidence, recover or refine coherent gene
modules, explain likely mechanisms, and abstain when a gene set does not support
one shared mechanism.

The current implementation is designed for ORNL Frontier/Slurm workflows and
local full-brain multiplex data. It is not packaged as a pip-installable
library.

## Pipeline Overview

The proposed training pipeline has five stages.

1. Build module targets from the full-brain multiplex.
   The primary corpus uses MENTOR-derived modules from a genome-wide
   dendrogram. A complementary corpus uses RWR-LOE seed expansions: one module
   per gene, with membership selected by the RWR++ geometric elbow rule.

2. Convert modules into hidden-target tasks.
   Each module can produce four task types:
   `explanation`, `recovery`, `refinement`, and `none`. The model sees only the
   visible input genes and optional graph query specification; hidden module
   membership is used only for scoring.

3. Generate tool-using trajectories.
   The actor proposes reasoning steps and optional tool calls. Deterministic
   runtime tools execute graph operations. A verifier updates the structured
   interpretation and decides whether to continue, revise, or stop.

4. Mine supervised and preference data.
   Shared-prefix trajectory branches are scored by deterministic reward
   functions. Better branches become preference pairs for DPO, while completed
   trajectories provide SFT examples and audit artifacts.

5. Train and evaluate.
   The intended sequence is warm-start SFT, DPO over shared-prefix branch
   preferences, then GRPO or another policy-gradient method over terminal and
   intermediate verifiable rewards.

## Current Source Of Truth

Active full-brain runs use:

- Multiplex store: `data/runtime/full_brain_multiplex_store`
- Full-brain RWR++ flist:
  `/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/data/full_brain_flist.tsv`
- MENTOR dendrogram corpus: `data/gw_dendrogram_corpus_full_brain`
- RWR-LOE corpus: `data/rwr_loe_corpus_full_brain`
- Mixed corpus: `data/module_corpus_full_brain_mixed`

CORUM remains useful for legacy tests and proposal history, but it is no longer
the active source of truth for production task generation.

## Important Concepts

**Hidden targets.**
Task rows include hidden module membership for scoring, but actor and verifier
prompts must not expose those targets.

**Difficulty.**
For MENTOR dendrogram tasks, close distractor genes are hard and far distractor
genes are easy. The current dendrogram schema is `gw-dendrogram-corpus-v2`; old
rows with reversed easy/hard labels should be rebuilt.

**RWR-LOE modules.**
RWR-LOE rank caches are prewarmed with the MPI-capable RWR++ `rwr` app. The
corpus builder then creates each gene-centered module by retaining ranked genes
on the high-score side of the geometric elbow cutoff, using
`rank < elbow_rank_cutoff`.

**Runtime tools.**
Model-facing graph tools are schema-wrapped and deterministic. RWR++ tools cover
RWR ranking, distances, layer summaries, shortest paths, perturbation, ablation,
and related diagnostics. Native runtime tools still cover operations such as
neighborhood lookup and induced subgraphs.

## Main Commands

Build the MENTOR dendrogram corpus:

```bash
python scripts/build_gw_dendrogram_corpus.py \
  --dendrogram-path data/gw_dendrogram.txt \
  --store-dir data/runtime/full_brain_multiplex_store \
  --out-dir data/gw_dendrogram_corpus_full_brain
```

Prewarm RWR-LOE rank caches on Slurm:

```bash
LOE_MODE=prewarm LOE_SHARD_COUNT=64 \
  sbatch -p batch --array=0-63%4 scripts/build_rwr_loe_corpus.slurm
```

Materialize the RWR-LOE corpus after all shards finish:

```bash
LOE_MODE=materialize \
  sbatch -p batch scripts/build_rwr_loe_corpus.slurm
```

Mix the MENTOR and RWR-LOE corpora:

```bash
python scripts/mix_module_corpora.py --json
```

Select a stratified verification pilot:

```bash
python scripts/select_verification_tasks.py \
  --tasks-path data/module_corpus_full_brain_mixed/tasks.train.jsonl \
  --source-strata \
  --pilot-size 60 \
  --pilot-out data/module_corpus_full_brain_mixed/pilot.tasks.jsonl \
  --smoke-task-ids-out data/module_corpus_full_brain_mixed/smoke_task_ids.txt
```

Generate trajectories on Frontier:

```bash
TASKS_PATH=data/module_corpus_full_brain_mixed/pilot.tasks.jsonl \
OUT_DIR=data/module_corpus_trajectories/pilot \
sbatch generate_trajectories.slurm
```

Audit a generated run:

```bash
python scripts/audit_trajectory_run.py \
  --run-dir data/module_corpus_trajectories/pilot/<run> \
  --dpo-pair-gate
```

## Repo Layout

- `scripts/`: corpus builders, trajectory generation, audit utilities, and
  Slurm launchers.
- `runtime/`: deterministic state, schemas, validators, scoring, and RWR++
  structured backend.
- `cpp_runtime/`: compiled local graph runtime components.
- `external/rwr_hpc/`: vendored RWR++ apps and libraries used on Frontier.
- `tests/`: unit and integration tests for corpus building, runtime tools,
  verification gates, and launch-adjacent behavior.
- `agents/`: proposal and planning text.

Generated corpora, trajectories, caches, and logs live under `data/` and
`logs/`. Many large artifacts are intentionally gitignored.

## Training Entry Points

The current training scripts are:

```bash
python scripts/train_sft.py --help
python scripts/train_grpo.py --help
```

Cluster launchers exist for the main training paths:

```bash
sbatch train_sft.slurm
sbatch train_grpo.slurm
```

The DPO data path is built from generated trajectory artifacts and preference
pairs. In practice, run the smoke and audit gates before using a trajectory run
for training.

## Testing

Use the shared project environment on Frontier, then run:

```bash
python -m pytest
```

For focused validation of the current corpus and verification path:

```bash
python -m pytest \
  tests/test_build_gw_dendrogram_corpus.py \
  tests/test_build_rwr_loe_corpus.py \
  tests/test_rwr_hpc_structured_backend.py \
  tests/test_verification_gate.py
```

Most scripts also support `--help` and can run small local smoke fixtures with
reduced task counts or `--max-genes`.

## Notes For Contributors

- Prefer Slurm for full-brain RWR and trajectory jobs; use local commands only
  for tests and small smoke runs.
- Keep `AGENTS.md` aligned with current implementation decisions. It is the
  repo's living guidance for agents and future work.
- Keep proposal text separate from operational truth. The implementation has
  moved from CORUM/HumanNet framing to MENTOR and RWR-LOE modules over the
  full-brain multiplex.
