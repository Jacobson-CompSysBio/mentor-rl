# Mentor-RL Current Status

## Data Source Update (2026-05-13)

The training-data source is changing from CORUM complexes to a genome-wide dendrogram of MENTOR-derived modules. New corpus-building work should treat dendrogram modules/subtrees as the fundamental supervised units, not curated CORUM complexes.

The raw dendrogram input currently present in the repo is `data/gw_dendrogram.txt` (the `data/gw_dendrogram` datatype). It is a tab-delimited tree file with columns `node_id`, `left_id`, `right_id`, `height`, and `label`. Internal tree nodes use `label=NA`; leaf nodes use Ensembl gene IDs in `label` and `left_id=-1`, `right_id=-1`. Module/task construction should parse this tree structure and derive gene groups from dendrogram cuts or subtrees.

Existing CORUM corpus and trajectory artifacts are now legacy/debug context unless explicitly regenerated or ported to the dendrogram-module data model.

## Snapshot (2026-04-30)

The repo implements the main trajectory-generation path described in `shared_memory/methods_proposal.tex`: legacy CORUM task construction, structured runtime state, deterministic graph/runtime tools, shared-prefix branch generation, scoring, preference-pair mining, and Frontier/vLLM launch plumbing.

Gene overlap across the legacy CORUM corpus splits was intentional for that evaluation framing. Any new dendrogram-module split policy should be redefined around tree/module heldout behavior rather than inheriting CORUM complex/context assumptions. The SFT/GRPO/DPO training scripts are being handled on other branches and are not the blocker for the next trajectory-generation pass.

## Progress

- Legacy CORUM corpus is built in `data/corum_corpus/`.
  - `4,914` retained deduplicated complexes.
  - `214,368` canonical tasks total.
  - Split counts: `171,408` train, `21,480` val, `21,480` test.
  - Task types: `19,656` explanation, `28,560` recovery, `58,968` refinement, `107,184` none.
- HumanNet multiplex binary store is built in `data/humannet_multiplex_store/`.
  - Format: `mentor-rl-multiplex-store-v1`.
  - `44,491` genes, `384` layers, `24,211,378` aggregate undirected edges.
  - Store size is about `1.8G`.
- Runtime and generator stack are in place.
  - `runtime/schemas.py`, `state.py`, `validators.py`, and `scoring.py` cover state, validation, local branch scoring, terminal scoring, trajectory turns, and preference pairs.
  - `runtime/environment.py` can use either the Python reference graph tools or the compiled C++ backend.
  - `scripts/generate_trajectories.py` supports heuristic and model-backed generation, free-form actor reasoning, runtime tool calls, verifier JSON updates, branch scoring, selected turns, final summaries, and preference-pair artifacts.
- Frontier launcher work is in place in `generate_trajectories.slurm`.
  - It resolves model directories, stages models to NVMe, bootstraps tiktoken files for gpt-oss, starts vLLM/Ray, waits on health, records `/v1/models`, exports `PYTHONPATH`, runs preflight smoke checks, and passes generator timeout settings.

## Frontier Run State

- `data/corum_trajectories/gpt_oss_120b_job_4499998/` is the first non-empty model-backed trajectory run.
  - It wrote branch pools, selected turns, final summaries, and preference pairs.
  - It should be treated as debug-only because the artifact contract still included terminal score metadata plus raw model response/token payloads at the time it was generated.
- `data/corum_trajectories/gpt_oss_120b_job_4500259/` launched after that and failed before a clean manifest was written.
  - The local error log shows an HTTP 400 from the vLLM completions endpoint; the run also occurred during a reported Frontier instability/crash window, so this is treated as a rerun/infrastructure issue rather than a corpus-design blocker.
- Earlier run directories remain debugging records only.
  - `4440835` produced empty trajectory artifacts.
  - Smoke-only and malformed-output attempts before `4499998` should not be used for training data.

## Current Blockers

- The dendrogram-module corpus builder has not yet replaced the legacy CORUM corpus builder in the runtime/training data path.
- Future trajectory artifacts must exclude model-visible leakage/debug payloads.
  - `final_summaries.jsonl` should not write hidden terminal score metadata.
  - Branch pools, turns, finding records, and preference pairs should not write raw actor/verifier responses or token ID arrays.
- Model-emitted all-layer aliases must be canonicalized before storage and scoring.
  - `layers: ["all"]`, empty lists, and null all-layer requests should be stored as omitted layer fields so valid all-layer calls are not penalized as schema failures.
- A fresh clean model-backed generation run is needed before treating artifacts as training/evaluation material.

## Next Steps

1. Add a dendrogram parser/corpus builder for `data/gw_dendrogram.txt`.
2. Define the module extraction and split policy for tree-derived training examples.
3. Port runtime initialization/scoring assumptions from CORUM complexes to dendrogram-derived target modules.
4. Keep the artifact-sanitization and all-layer normalization requirements when regenerating model-backed trajectories.
5. Run a fresh small model-backed generation job only after the dendrogram-derived task path is wired in.

## Assumptions To Avoid

- Do not use CORUM complexes as the fundamental supervised unit for new training data.
- Do not flatten `data/gw_dendrogram.txt` into independent rows without preserving parent/child tree structure.
- Do not train on old trajectory directories unless they are explicitly marked debug-only and excluded from training inputs.
- Do not treat scalar reward/score fields as leakage by themselves; they are needed for ranking and auditability.
- Do not block trajectory generation on generic SFT/GRPO script cleanup in this branch.
