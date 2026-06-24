# MENTOR-RL Agent Notes

Update this file as implementation progress is made and project decisions
change, so future agents inherit the current repo direction.

## Project Goal

MENTOR-RL is a reasoning environment for scalable biological mechanistic
interpretation. It trains and evaluates tool-using agents that can explore
large biological multiplexes, recover or refine mechanistic groups, explain
coherent groups, determine when evidence is insufficient, and recognize when a
query contains multiple mechanistic groups.

The proposal frames this as a sequential decision process with one model used
in two modes:

- Actor mode proposes the next reasoning step and optional tool call.
- Verifier mode updates the working interpretation, structured state, and
  continuation decision: `continue`, `revise`, or `stop`.

The user-facing output is a human-readable interpretation. The scoring-facing
output is a schema-enforced structured state containing anchors, relationship
status, predicted group or groups, evidence with provenance, mechanistic
labels, remaining budget, and termination state. Keep these two views aligned:
the interpretation should explain the same claim the structured state records.

Model prompts should use a Markov-style prompt state, not an ever-growing
trajectory transcript. Full prior tool observations and evidence payloads stay
in trajectory artifacts for scoring, audit, DPO mining, and expert review.
Actor and verifier prompts should receive compact task inputs, the current
interpretation, relationship status, predicted groups, mechanistic labels with
evidence handles, tool/evidence counters, gene-set handles, and only the
current deterministic observation in richer form.

Current trajectory-generation work should build on the merged RWR++
infrastructure while preserving native tools where RWR++ does not provide a
direct replacement.

## Current Baseline

As of 2026-06-22, the RWR-HPC structured-tool infrastructure is no longer the
main blocker. PR #11 from `rwr-hpc-structured-tools` was merged into `main` and
`trajectory-gen` at commit `7259981`; keep that as the historical infrastructure
baseline unless the user points to a newer branch. That merge provided the
structured RWR-HPC backend, smoke-cache precompute profile split, Markov-style
compact prompt state, launcher updates, audit metrics, and tests.

Current trajectory-generation work is about exact membership quality. Recovery
and refinement positives now require exact gene-membership recovery/refinement
under the scorer. Partial recovery/refinement is useful as a rejected/contrastive
DPO signal, but it must not be treated as a chosen positive for exact recovery.

The generator now explicitly prompts actor/verifier mode to optimize exact
membership for recovery/refinement, and the controller enforces that partial
recovery/refinement states are nonterminal while budget remains. If a
recovery/refinement branch tries to stop with
`relationship_status=partially_observed_group` or hidden-score
`task_success_level=partial`, the branch is rewritten to `continue`, with
override provenance in `branch.metadata["exact_membership_nonterminal_override"]`.
Hidden targets remain scoring-only and are not exposed to prompts.

The next blocker is producing exact-positive recovery/refinement branches
reliably enough to mine exact-vs-partial DPO pairs. Mechanism quality is still
required to avoid weak `validated_group` overclaims, but exact membership is the
first acceptance target for recovery/refinement.

## Proposal Methodology Context

The methods proposal in `shared_memory/methods_proposal.tex` is conceptual
background for the larger project, but it is not the current data source of
truth. The key ideas to preserve are:

- Mechanistic interpretation is treated as long-horizon graph exploration over
  genes, proteins, molecular components, pathways, and regulatory context.
- The runtime is deterministic and schema-grounded. Tools execute through
  structured interfaces, and rewards should come from verifiable state changes
  rather than proprietary judge models.
- Warm-start SFT teaches schema-valid biological tool use, entity
  normalization, annotation retrieval, literature-grounded extraction, and
  evidence-to-finding generation.
- MENTOR-derived modules from a genome-wide dendrogram over the brain multiplex
  are the primary target source for current task and trajectory generation.
- CORUM-grounded tasks were part of the proposal's original supervision plan,
  but CORUM is no longer the repo's source of truth.
- DPO examples should prefer shared-prefix trajectory branches so the compared
  continuations differ after the same query and evidence context.
- GRPO and later online RL should use terminal trajectory rewards plus
  deterministic intermediate signals such as membership recovery, mechanistic
  label quality, schema validity, evidence provenance, and tool efficiency.
- Evaluation should prioritize held-out MENTOR-derived brain-multiplex modules,
  expert-preference evaluation, and ablations for verifier mode, graph tools,
  retrieval, and curriculum settings. CORUM is historical proposal context only
  and should not be used as an active task, reward, enrichment, label, or
  evaluation source.

Hidden targets must remain hidden from the actor and verifier. They are for
scoring and validation only.

## Current Data Source Of Truth

Use MENTOR-derived modules from the genome-wide dendrogram sourced from the
brain multiplex as the source of truth for large-scale trajectory generation.
This means recovery, refinement, explanation, insufficient-support, and
multiple-group tasks should be sampled and scored against dendrogram-derived
modules unless a run explicitly opts into another benchmark.

The full-brain multiplex flist default for launches from this repo is:

`/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/data/full_brain_flist.tsv`

There is also a relative local copy:

`data/full_brain_flist.tsv`

Use the local copy for `RWR_HPC_FLIST`. It should contain 358 full-brain layer
entries and should match `data/runtime/full_brain_multiplex_store`.
Do not use the old HumanNet/brain multiplex flist with the full-brain store.

The active full-brain runtime store is:

`data/runtime/full_brain_multiplex_store`

Current exact-membership pilots should use the mixed full-brain corpus:

`data/module_corpus_full_brain_mixed`

This corpus combines `data/gw_dendrogram_corpus_full_brain` and
`data/rwr_loe_corpus_full_brain`, balanced by source. Its standard pilot files
under `data/module_corpus_full_brain_mixed/pilots` are source-stratified; the
24-task pilot currently covers both GW dendrogram and RWR-LOE modules and
includes explanation, recovery, refinement, and none tasks. When using a pilot
file as `TASKS_PATH`, prefer `SMOKE_TASK_INDEX=0`; do not pass the broader
`smoke_task_ids.txt` unless those ids are known to be present in the same
`TASKS_PATH`.

The full-brain dendrogram corpus remains an underlying source corpus:

`data/gw_dendrogram_corpus_full_brain`

This corpus is built from `data/gw_dendrogram.txt` paired with
`data/runtime/full_brain_multiplex_store`. Do not use
`data/gw_dendrogram_corpus` for current full-brain trajectory pilots; that
older corpus was built against `data/humannet_multiplex_store` and can include
task genes that are absent from the active full-brain runtime graph.

As of the RWR-LOE corpus implementation, dendrogram corpus schema
`gw-dendrogram-corpus-v2` is the valid MENTOR dendrogram schema. Its distance
band semantics are: lower dendrogram-distance percentile means nearer to the
target module and therefore harder; high percentile means farther and easier.
Any previously generated full-brain dendrogram rows with reversed
easy/hard labels should be treated as stale and rebuilt before training or
evaluation.

The complementary full-brain RWR-LOE module corpus is produced by
`scripts/build_rwr_loe_corpus.py` and written to:

`data/rwr_loe_corpus_full_brain`

This corpus creates one LOE module per full-brain store gene. Module membership
is selected by the RWR++ geometric elbow rule over each seed's rank/score
curve: genes with numerically lower rank than the elbow cutoff (`rank <
elbow_rank_cutoff`) are retained, and lower-scoring genes are excluded. Full
cache prewarm should run through
`scripts/build_rwr_loe_corpus.slurm` using the MPI-capable RWR++ `rwr` app with
recorded encodings, then postprocess those encodings into per-seed LOE-style
min-rank files. The persistent rank cache is kept separate from the
model-facing RWR-HPC cache:

`data/runtime/rwr_loe_full_brain_rank_cache`

Use direct `rwr_loe` calls only for small validation samples, because the
standalone `rwr_loe` app flattens all seed rows into one vector and therefore
cannot batch one module per seed gene. Use `scripts/mix_module_corpora.py` to
build the combined full-brain training/testing corpus at
`data/module_corpus_full_brain_mixed` from the dendrogram and LOE corpora.

CORUM should not be treated as authoritative for current production work. It
can remain in legacy tests and proposal-history utilities, but it should not
drive active corpora, reward targets, source-of-truth language, enrichment
queries, mechanistic labels, sanity checks, or auxiliary evaluation.

## Proposal Divergences In This Repo

Several implementation decisions have intentionally moved beyond or away from
the original proposal. Preserve these decisions unless the user explicitly
changes direction.

- Tool names have migrated to RWR++ names. The proposal names graph tools as
  `shortest_path`, `rwr_multiplex`, and `rwr_monoplex`; production model-facing
  tools should now use the structured RWR++ names and wrappers listed in the
  RWR++ Tool Contract. Legacy names may remain as compatibility aliases.
- The graph backend and task source have shifted from the proposal's in-house
  HumanNet/CORUM framing to Ken's full-brain multiplex flist plus
  MENTOR-derived genome-wide dendrogram modules. These modules are now the
  primary source of truth for trajectory targets.
- The runtime binding is not the proposal's pybind11 interface. The current
  compiled backend uses `ctypes` around `libmentor_runtime.so`.
- The binary store choice is raw CSR directories with binary metadata, not
  HDF5. Text input is allowed at ingest boundaries, but production tool calls
  should move toward binary-only graph and metadata access.
- RWR++ is external infrastructure. The repo currently supports app-backed
  RWR++ calls with hidden scratch files, and the long-term target is C++
  adapters or RWR++ library bindings that avoid per-call text I/O.
- The model-serving path is not fixed to the proposal's example base model.
  Current trajectory generation supports OpenAI-compatible vLLM style serving
  and deterministic heuristic paths for tests.
- The corpus is no longer CORUM-centered. Scripts such as
  `scripts/build_gw_dendrogram_corpus.py` support MENTOR genome-wide
  dendrogram modules while preserving task shapes for recovery, explanation,
  refinement, insufficient support, and multiple groups. For current
  full-brain runs, build this corpus with `--store-dir
  data/runtime/full_brain_multiplex_store` and write it to
  `data/gw_dendrogram_corpus_full_brain`.
- Multiple-group recognition is an explicit repo goal and appears in the
  structured schema. The proposal mentions multiple groups in structured state,
  but its four main task cases focus on recovery, explanation, refinement, and
  no-shared-mechanism detection. Treat multiple-group support as important but
  still less mature than the four original task families.
- Query paraphrasing is not a required separate pipeline yet. The repo can use
  deterministic/generated task rows and model-backed trajectory generation
  without a dedicated paraphrasing model.

## RWR++ Tool Contract

- Model-facing tool names should use the RWR++ names and structured wrappers:
  `rwr`, `rwr_loe`, `shortest_paths`, `get_rank`, `get_distance`,
  `get_spearman`, `get_pearson`, `get_dot_similarity`,
  `get_rank_vector_summary`, `get_encoding_summary`, `get_gene_layers`,
  `get_nodes_by_layer`, `get_layer_stats`, `get_path_layer_counts`,
  `get_component_summary`, `get_seed_essentiality`, `get_layer_ablation`, and
  `get_node_perturbation`.
- `shortest_path` maps to the RWR++ `shortest_paths` app when the structured
  RWR-HPC backend is enabled.
- `rwr` maps to the RWR++ `rwr` app.
- `get_rank` maps to the RWR++ `rwr` app with rank recording enabled. It is a
  single-source target rank, not LoE.
- `get_distance` maps to the RWR++ `rwr` app with two one-gene seed vectors and
  parses the requested pairwise rank-vector distance.
- `get_spearman` derives Spearman correlation from the cached Spearman distance
  result returned by `get_distance`.
- `get_pearson` derives Pearson correlation from cached Pearson distance.
- `get_dot_similarity` uses the RWR++ `rwr` app's `dot` metric and reports the
  returned value as similarity rather than dissimilarity.
- Single-layer or subset-of-layers RWR is represented by the same RWR++ tools
  with one selected layer or a selected layer list.
- `get_neighbors` and `induce_subgraph` remain native mentor-rl tools unless
  an RWR++ equivalent is added later.
- Legacy names such as `rwr_multiplex`, `rwr_monoplex`, and `shortest_path`
  may remain as compatibility paths for old tests or local debugging, but
  large trajectory runs should prefer the RWR++ names.
- Ken's full-brain edge lists include headers. Full-brain launches should keep
  header-aware RWR++ input handling enabled; use no-header mode only for toy
  fixtures or explicitly headerless edge lists.

## Implemented RWR++ Tool Wrappers

The current structured backend implements compact, cacheable wrappers for the
RWR++ source and app functionality below. Keep model-facing payloads small and
avoid returning raw matrices or full dense vectors.

Pairwise and vector-summary tools:

- `get_gene_layers` or `get_nodes_by_layer`: use `gene_layer_map` or the
  `rwr` app's node-by-layer output to report which multiplex layers contain a
  gene.
- `get_layer_stats`: parse `gene_layer_map` network statistics for layer sizes
  and edge counts.
- `get_rank_vector_summary` or `get_encoding_summary`: use `rwr` recorded
  ranks or encodings, but return top entries or compact summaries rather than
  full vectors.
- `get_pearson` and `get_dot_similarity`: use the `rwr` app's existing
  distance metrics. Treat dot as a similarity score, not a dissimilarity.
- `get_path_layer_counts`: parse the existing `shortest_paths` layer-count
  output for a path-level layer-support summary.

Perturbation, ablation, and component tools:

- `get_component_summary`: wrap `disconnected_components` output into
  component membership and component-size summaries.
- `get_seed_essentiality`: wrap GRIN leave-one-out and null-rank outputs for
  seed-set sensitivity.
- `get_layer_ablation`: wrap `rwr_ablation` to report how layer removal
  changes RWR distances or ranks.
- `get_node_perturbation`: wrap `rwr_perturbation` to report how perturbing a
  gene or edge set changes RWR distances.

Do not expose `clean_edge_list` as a model tool. It is ingest and maintenance
infrastructure. Kendall tau exists in the RWR++ correlation code, but the
current `rwr` CLI does not expose it as a direct distance metric, so it would
require an app or library-binding change before becoming a normal wrapper.

## Current Backend Split

- `runtime/rwr_hpc_app_backend.py` is the CLI/developer bridge to standalone
  RWR++ apps discovered from a build directory or app manifest.
- `runtime/rwr_hpc_requests.py` defines structured biological request objects
  for model-facing calls and rejects low-level file/CLI arguments.
- `runtime/rwr_hpc_structured_backend.py` keeps RWR++ logically in memory from
  the model perspective: it accepts structured requests, uses cache first,
  creates hidden scratch inputs only for the current app fallback, invokes the
  app backend, parses results, and stores provenance.
- `runtime/rwr_hpc_cache.py` is the shared cache for structured RWR++ calls.
  Cache keys should include the app/tool name, logical request payload, flist
  hash, build id, and parser/runtime schema details.
- `runtime/environment.py` is the dispatch point. RWR++ production runs should
  initialize the structured backend and require it so `rwr`, `rwr_loe`,
  `shortest_paths`, and all `get_*` RWR++ wrappers cannot silently fall back to
  legacy Python/C++ behavior.

## Large-Scale Trajectory Target

The intended production goal is large-scale trajectory generation where RWR++
is the real graph backend for all RWR++ model tools, using Ken's updated full
brain multiplex as the exploration graph:

`/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/data/full_brain_flist.tsv`

Cache as much RWR++ work as possible. RWR++ calls are deterministic for a
given request, flist, build id, and parser schema, so repeated seed/query/path
calls should hit cache across rollout workers.

External annotation and enrichment calls are also cacheable and should be
precomputed for smoke runs when the visible input genes are known. Model/vLLM
responses are not trajectory evidence caches and should not be part of the
precompute plan.

The I/O target is binary after ingest. Text `.tsv`/`.txt` files are acceptable
only at ingestion boundaries or as temporary compatibility scratch for current
standalone RWR++ apps. The long-term target is C++ adapters or RWR++ library
bindings that consume binary graph metadata and return structured results
without per-call text files.

## Smoke-Test Cache Plan

For RWR++ smoke and pilot runs, run `scripts/precompute_smoke_caches.py`
through `generate_trajectories.slurm` with the default
`PRECOMPUTE_SMOKE_CACHES=1`. The precompute step must use the same
`STORE_DIR`, `RWR_HPC_FLIST`, `RWR_HPC_BUILD_DIR`, `RWR_HPC_CACHE_DIR`,
annotation cache paths, and enrichment background as the generation path so
cache keys and runtime metadata match.

For the smoke task, precompute only visible-input work:

- MyGene and g:Profiler annotation caches through the existing mechanism
  prefetch path.
- The default smoke cache profile is `core`. It precomputes stable RWR++ cache
  entries for `rwr`, the first visible seed-pair `shortest_paths` request,
  first seed-pair `get_rank`, `get_distance`, `get_spearman`, `get_pearson`,
  `get_dot_similarity`, and `get_path_layer_counts` requests, plus
  visible-seed `get_rank_vector_summary`, `get_encoding_summary`,
  `get_layer_stats`, `get_component_summary`, `get_gene_layers`,
  `get_nodes_by_layer`, and `get_node_perturbation` requests.
- `rwr_loe`, `get_seed_essentiality`, and `get_layer_ablation` are implemented
  model-facing tools, but they are excluded from the default smoke cache gate.
  On the full-brain graph, `rwr_loe` and `get_layer_ablation` can exceed the
  short per-app smoke timeout, and GRIN-backed `get_seed_essentiality` has
  shown app-level dictionary failures for at least one first-smoke seed set.
  Run `SMOKE_RWR_PRECOMPUTE_PROFILE=extended` only when intentionally
  diagnosing those heavyweight tools before a larger run.
- Native full-brain binary-store touches for `get_neighbors` and
  `induce_subgraph`, which verify the compiled store is readable before the
  model-facing smoke step.

A historical full-brain smoke launch failed before generation because RWR++ was
invoked in no-header mode against headered full-brain edge lists. Keep
`RWR_HPC_EDGELIST_HAS_HEADERS=1` for full-brain runs unless the input flist is
explicitly known to reference headerless edge lists.

Do not add vLLM/model-response caching to this plan. Model startup, tokenizer,
and kernel/runtime warmup are separate serving concerns and should not be
treated as reusable trajectory evidence caches.

Cold Frontier smoke-only runs can still need generous allocations because the
launcher allows up to 90 minutes for vLLM health alone, then still needs model
staging, annotation prefetch, RWR++ app-backed cache fills, native-store
touches, and one model actor/verifier smoke step. Set walltime from observed
`logs/` and `smoke_cache_precompute_*.json` timings; do not drop below 2 hours
unless the model startup path is already proven warm and stable.

For large multi-shard generation, move annotation and RWR++ precomputation into
a separate prewarm job that writes to stable shared cache paths before launching
many generation shards. Native binary-store warming only benefits the same
allocation/node, so keep that as an in-job smoke preflight.

## Full-Scale Readiness Gates

Older pilot details are mostly historical. Preserve only durable lessons:

- A functional smoke is not a production gate pass. It verifies launch, cache,
  RWR-HPC, native-store, annotation, and vLLM plumbing only.
- Full-brain smoke and pilot runs should keep
  `RWR_HPC_APP_TIMEOUT_SECONDS=1800` unless measured timings justify a smaller
  value.
- Use stratified full-brain pilot task files, never the first N rows and never
  stale `data/gw_dendrogram_corpus` pilot files.
- Recovery/refinement generation should retry when sampled actor candidates
  lack any RWR++ model-facing tool, and preference mining should prioritize
  exact-positive recovery/refinement over partial or negative branches.
- Partial recovery/refinement is not a chosen-positive training target. Use it
  as rejected near-miss signal in `exact_over_partial`, `exact_recovery`, or
  `exact_refinement` pairs.

Recent evidence:

- Job `4866438` was a successful mixed-corpus smoke-only run. The core
  smoke-cache precompute passed with zero RWR-HPC errors, vLLM started, and the
  smoke model step completed. The selected branch was biologically negative,
  which is acceptable for smoke because smoke checks infrastructure, not exact
  recovery performance.
- Job `4867440` was the first mixed-corpus exact-performance 24-task pilot. It
  timed out at the 6-hour Slurm limit after completing 18/24 tasks. The timeout
  occurred during full trajectory generation, not smoke preflight: core RWR-HPC
  precompute had zero errors, the full-brain runtime had 358 layers, and vLLM
  was still serving requests near cancellation.
- The completed portion of `4867440` is diagnostic only, not a completed
  training/audit artifact. Slurm killed the process mid-write, leaving truncated
  JSONL tails and no real `final_summaries.jsonl`.
- A completed-only review was reconstructed for inspection at
  `data/module_corpus_full_brain_mixed_trajectories/exact_perf24_20260619_090435_completed_review`.
  It includes 18 trajectories and 71 selected steps reconstructed from valid
  branch-pool rows. The reconstructed summaries are for visualization only.
- On the completed portion of `4867440`, strict positive success was 6/18
  overall, but exact recovery/refinement was 0/4 recovery and 0/4 refinement.
  Recovery/refinement mostly produced partial near-misses. This confirms that
  exact-positive branch generation, not infrastructure, is the current blocker.

Exact-performance acceptance criteria before full DPO generation:

1. Smoke-cache precompute passes with `SMOKE_RWR_PRECOMPUTE_PROFILE=core`.
2. No RWR-HPC observation errors.
3. At least one exact-positive recovery and at least one exact-positive
   refinement in a small pilot.
4. Exact-membership DPO pairs are present, especially exact-positive chosen
   branches over partial or negative rejected branches.
5. Existing schema, pair, tool-coverage, and weak-evidence audit gates still
   pass.

Recommended next pilot is a fresh `OUT_DIR` with longer walltime. Prefer a
recovery/refinement-only task subset when available; the mixed 24-task pilot is
acceptable for a broad check but spends time on explanation and none tasks.

Use the stronger exact-search settings for recovery/refinement pilots:

```bash
MAX_STEPS=6
N_ACT=6
N_VER=3
TASK_CONCURRENCY=1
SELECTION_POLICY=task_quality
SELECTION_SCORE_EPSILON=0.10
PAIR_MINING_STRATEGY=quality_balanced
TOOL_COVERAGE_RETRY_COUNT=4
RECOVERY_RWR_TOP_K=3000
RUN_SMOKE_TEST=1
SMOKE_TEST_ONLY=0
SMOKE_RWR_PRECOMPUTE_PROFILE=core
```

If a clean smoke has already passed in the same software/cache configuration,
`RUN_SMOKE_TEST=0` can be used for a rerun to save startup time, but use a
fresh output directory and do not reuse a timed-out directory with truncated
JSONL artifacts.

Audit an exact-performance pilot with:

```bash
python scripts/audit_trajectory_run.py \
  --run-dir <run_dir> \
  --dpo-pair-gate \
  --min-balanced-pair-bins 6 \
  --max-selected-no-tool-rate 0.35 \
  --max-positive-selected-no-tool-rate 0.15 \
  --min-recovery-expansion-pair-rate 0.20 \
  --min-tool-supported-pair-rate 0.30 \
  --max-mechanism-label-only-pair-rate 0.20 \
  --max-step0-pair-rate 0.70 \
  --min-none-success-rate 0.80 \
  --min-explanation-success-rate 0.80 \
  --min-recovery-refinement-partial-rate 0.60 \
  --min-recovery-exact-success-rate 0.125 \
  --min-refinement-exact-success-rate 0.25 \
  --min-exact-membership-pair-rate 0.01 \
  --min-selected-rwr-hpc-tool-rate 0.10 \
  --min-rwr-hpc-candidate-rate 0.25 \
  --min-rwr-hpc-supported-pair-rate 0.10 \
  --max-rwr-hpc-observation-error-rate 0.00 \
  --max-validated-weak-evidence-rate 0.10
```

For the 8:1 topology pilot, request 9 nodes and set
`RWR_HPC_SERVICE_NODE_COUNT=1`. The launcher reserves the final node for
`scripts/rwr_hpc_worker_service.py`, uses the first nodes for vLLM/Ray, and
routes model-facing RWR++ tools through `RWR_HPC_SERVICE_URL`. Native
`get_neighbors` and `induce_subgraph` remain local unless explicitly migrated.
Check `rwr_hpc_service_metrics.json` for per-tool request counts, queue wait,
service time, cache hit/miss counts, and errors.

## Scaling Guidance

Use a dedicated RWR++ worker pool for production trajectory generation. The
default topology to try first is 8 vLLM nodes feeding 1 RWR++ node with 8 GPUs.
This keeps model serving and graph serving isolated, lets RWR++ batch/cache
requests centrally, and avoids tying every vLLM node to a local RWR GPU.

Do not start with a 7 vLLM GPUs plus 1 RWR GPU per node layout unless profiling
shows network latency or RWR queueing dominates. That layout fragments RWR++
cache locality and can reduce vLLM batching efficiency.

Track at minimum:

- vLLM tokens/sec and request queue latency.
- RWR++ queue latency and service time by tool.
- RWR++ cache hit rate by tool.
- GPU utilization for both vLLM and RWR++ workers.
- Cache size and filesystem pressure.

## Implementation Priorities

1. Keep all model-facing RWR++ tools working through the structured RWR-HPC
   backend with cache-first behavior.
2. Add fail-fast launch options for large runs so missing RWR++ configuration
   is caught before trajectory generation starts.
3. Build a binary full-brain graph store from Ken's flist for native
   `get_neighbors` and `induce_subgraph`, so all graph tools operate on the
   same multiplex.
4. Replace app-fallback scratch files with C++ adapters or library bindings for
   RWR++, especially for `rwr` and `shortest_paths`.
5. Extend binary metadata support so gene/layer names are not read from text
   after ingest.
6. Add future RWR++ wrappers only after defining compact model-facing payloads,
   cache keys, and tests. Prefer summaries over raw matrices or full vectors.

## Verification Expectations

- Run focused unit tests for changed runtime modules before broad trajectory
  work.
- For Frontier RWR++ builds, verify both Slurm state and binary artifacts.
  The known working build path is `external/rwr_hpc/build_frontier`, with app
  binaries under `external/rwr_hpc/build_frontier/apps`.
- When judging whether a large run is RWR++ backed, check launch arguments and
  runtime manifest metadata, not just model tool names.
