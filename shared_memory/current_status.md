# Mentor-RL Current Status

## Data Source Update (2026-05-13)

The training-data source is changing from CORUM complexes to a genome-wide dendrogram of MENTOR-derived modules. New corpus-building work should treat dendrogram modules/subtrees as the fundamental supervised units, not curated CORUM complexes.

The raw dendrogram input currently present in the repo is `data/gw_dendrogram.txt` (the `data/gw_dendrogram` datatype). It is a tab-delimited tree file with columns `node_id`, `left_id`, `right_id`, `height`, and `label`. Internal tree nodes use `label=NA`; leaf nodes use Ensembl gene IDs in `label` and `left_id=-1`, `right_id=-1`. Module/task construction should parse this tree structure and derive gene groups from dendrogram cuts or subtrees.

Existing CORUM corpus and trajectory artifacts are now legacy/debug context unless explicitly regenerated or ported to the dendrogram-module data model.

## Genome-Wide Dendrogram Sampling Procedure (2026-05-13)

The dendrogram corpus builder is `scripts/build_gw_dendrogram_corpus.py`. It writes the new corpus under `data/gw_dendrogram_corpus/` with `manifest.json`, `progress.json`, `split_report.json`, `modules.jsonl`, `prototypes.jsonl`, and `tasks.{train,val,test}.jsonl`.

Sampling starts by parsing `data/gw_dendrogram.txt` as a parent/child tree, then filtering leaf genes to the HumanNet compiled store in `data/humannet_multiplex_store/`. Internal tree nodes become candidate modules only after graph-gene filtering. Eligible modules are all deduplicated internal subtrees with filtered sizes in these bins: `small=5-10`, `medium=11-15`, and `large=16-30`. Exact duplicate filtered gene sets are collapsed to one retained module, with duplicate source node IDs recorded.

Splits are module-stratified 80/10/10 train/val/test within each size bin. This first dendrogram version accepts nested-module overlap across splits; it is not a gene-heldout or tree-block-heldout split.

For each retained module, the builder creates runtime-compatible task rows using only `minimal` and `graph` evidence modes. `mechanism_labels` are `null` for now, and Ensembl IDs are used as display symbols. The four task families are:

- `explanation`: input is the full module; hidden target is the same module; difficulty is `complete`.
- `recovery`: input is the module after dropping genes; hidden target is the full module.
- `refinement`: input is the full module plus dendrogram-distance-constrained noise genes; hidden target is the full module.
- `none`: input is an unrelated gene set; hidden target is null with `relationship_status=insufficient_support`.

Recovery/refinement/none difficulties are assigned deterministically and evenly within each split and size bin. Difficulty controls both the number of dropped/noise genes and how close negative genes may be in the dendrogram: `easy` drops/adds `1` gene, `medium` uses `round(0.20 * module_size)` with minimum `1`, and `hard` uses `round(0.33 * module_size)` with minimum `2`. Recovery always leaves at least two real seed genes.

Negative sampling now uses the dendrogram itself rather than repeated network-diffusion calls. Candidate negatives are dendrogram leaf genes present in the HumanNet store, excluding the true module. For a module rooted at node `m` and candidate leaf `g`, distance is `height(LCA(m, g)) - height(m)`, so genes that join the module at a nearby ancestor are harder negatives and genes that only join near the root are easier negatives.

Distance bands are chosen by per-module candidate percentiles after sorting outside-module leaves from nearest to farthest: `easy=0-25%`, `medium=25-50%`, and `hard=50-75%`. The fallback bands currently match those same ranges, so failed samples are skipped rather than expanded outside the requested percentile window. Skipped prototypes are counted in `split_report.json`. `none` tasks additionally require sampled genes to be conflict-free with respect to eligible module co-membership.

After prototype construction, each `(split, size_bin)` group is downsampled so `explanation`, `recovery`, `refinement`, and `none` have equal raw row counts after evidence-mode expansion. This enforces the intended 25% task-family balance in the materialized corpus.

Preflight on the real inputs succeeded: the HumanNet store has `44,491` genes and `384` layers; `32,603` dendrogram leaves are present in that store; and dendrogram extraction yields `10,623` deduplicated eligible modules after store filtering (`6,700` small, `1,838` medium, `2,085` large). The full dendrogram-distance corpus build has not been run yet.

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
- Genome-wide dendrogram corpus builder is implemented in `scripts/build_gw_dendrogram_corpus.py`.
  - Unit-style direct checks passed in the current environment; `pytest` is unavailable on the default Python path.
  - The CLI preflight and real dendrogram extraction pass, but the full dendrogram-distance corpus has not yet been materialized.
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

- The full dendrogram corpus still needs to be built and inspected before it replaces legacy CORUM training inputs.
- Future trajectory artifacts must exclude model-visible leakage/debug payloads.
  - `final_summaries.jsonl` should not write hidden terminal score metadata.
  - Branch pools, turns, finding records, and preference pairs should not write raw actor/verifier responses or token ID arrays.
- Model-emitted all-layer aliases must be canonicalized before storage and scoring.
  - `layers: ["all"]`, empty lists, and null all-layer requests should be stored as omitted layer fields so valid all-layer calls are not penalized as schema failures.
- A fresh clean model-backed generation run is needed before treating artifacts as training/evaluation material.

## Next Steps

1. Run `python scripts/build_gw_dendrogram_corpus.py --dendrogram-path data/gw_dendrogram.txt --store-dir data/humannet_multiplex_store --out-dir data/gw_dendrogram_corpus --seed 42`.
2. Inspect `data/gw_dendrogram_corpus/progress.json`, `manifest.json`, `split_report.json`, and a sample from each `tasks.{train,val,test}.jsonl`.
3. Confirm task-family balance, skipped prototype counts, and dendrogram-distance negative bands before treating the corpus as training material.
4. Keep the artifact-sanitization and all-layer normalization requirements when regenerating model-backed trajectories.
5. Run a fresh small model-backed generation job using `data/gw_dendrogram_corpus/` after the corpus build is accepted.

## Assumptions To Avoid

- Do not use CORUM complexes as the fundamental supervised unit for new training data.
- Do not flatten `data/gw_dendrogram.txt` into independent rows without preserving parent/child tree structure.
- Do not train on old trajectory directories unless they are explicitly marked debug-only and excluded from training inputs.
- Do not treat scalar reward/score fields as leakage by themselves; they are needed for ranking and auditability.
- Do not block trajectory generation on generic SFT/GRPO script cleanup in this branch.
