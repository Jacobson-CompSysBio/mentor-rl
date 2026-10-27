# S1: Coordinate System and Conventions

This document explains the S1 design. The executable contract is versioned
separately so prose cannot silently change the training wire format.

## Current files (2026-10-27)

S1 v10 is the built training corpus. Its overlay is
`runtime/world_model/s1_v10.yaml`; its corpus and evaluator are under
`data/world_model_v4/corpora/s1_v10/` and `data/world_model_v4/eval/s1_v10/`.
It changes only S1.1.6, S1.3.8, and S1.8.4 training text. The original
system prompt, 4,096-token budget, allocations, S0 replay, and held-out panel
remain fixed. V10 training `5768596` and dependent S1 gate `5768597` and
S0 retention `5768598` were submitted on 2026-10-26; their launch is recorded
below. Training and S1 evaluation completed. V10 passes `structured_95_v1`
with 6,542/6,875 correct structured answers (95.156%). S0 retention remains
queued for priority; no v10 S0 retention result is available yet.
V9 remains the trained baseline (`5733083`, 1,728 updates). Its completed
S1 rerun `5742784` scores 6,526/6,875 (94.924%) and fails the 95% gate.
S0 retention passed. The earlier incomplete S1 result and parser repair
remain in the dated history below.
S1 v8 is the frozen decorrelation comparison, trained as job `5726246`.
S1 v6 remains the frozen reasoning/retry baseline.
S1 v7 is the prepared,
not-yet-submitted prompt experiment in `../mentor-rl-s1-prompt-v7/`, with an
8,192-token training limit and the same reviewed task content and recipe.

- [Current question catalog](S1-question-classes.md): all 66 active
  types, actual split counts, tool-call counts, statuses, decisions, full gold
  traces and the actual system prompt from v10.
- [V7 experiment and commands](S1-prompt-v7.md): prompt, files, rebuild,
  preflight, comparison constraints, and launch recipe.
- Frozen v6: `runtime/world_model/s1_v6.yaml`,
  `data/world_model_v4/corpora/s1_v6/`, `data/world_model_v4/eval/s1_v6/`,
  and `config/world_model_s1_120b_training_v6_32n.json`.
- V7 (isolated checkout): `runtime/world_model/s1_v7.yaml`,
  `runtime/world_model/prompts/task_scope_v1.txt`,
  `data/world_model_v4/corpora/s1_v7/`, `data/world_model_v4/eval/s1_v7/`,
  and `config/world_model_s1_120b_training_v7_32n.json`.

The dated entries below preserve the review history. Earlier plans and
examples do not override the current overlay, published rows, or this catalog.

## V10 evaluation results (2026-10-27)

Training `5768596` and frozen S1 evaluation `5768597` both completed with
exit code 0. The selected final checkpoint passes `structured_95_v1`:
6,542 of 6,875 structured answers are correct (95.156%), above the 95%
point-estimate threshold. The completed v9 rerun scored 6,526/6,875
(94.924%); v10 gains 16 correct answers on the same frozen panel.

All 6,875 record IDs are present exactly once, and every row reports no
network use. The report, metrics, checkpoint identity, and panel identity
agree. This headline scores structured answers; free-text summary wording
is not part of the promotion criterion. Per-type results remain diagnostics.

S0 retention job `5768598` is still pending for priority as of this check.
The S1 pass does not establish v10 S0 retention. No additional jobs were
submitted while preparing this PR.

The compact evidence receipt is
[`evaluation_summary.json`](../../diagnostics/s1_v10_results/evaluation_summary.json).
It records source-report SHA-256 values, coverage, the gate, and per-type
structured counts. Full predictions and reports remain under
`data/world_model_v4/eval_runs/s1_v10_120b_32n_test_structured95/oss120b-plain-base-tokenizer-lora-r32/slurm-5768597/`.
The [training and evaluation run](https://wandb.ai/jail-ai/mentor-sft/runs/oss120b-plain-base-tokenizer-lora-r32-slurm-5768596)
contains the logged metrics.

## V10 targeted scope fixes (2026-10-26)

The user requested all three fixes identified in the completed v9 evaluation.
Six net additional correct answers would cross its 95% overall gate. These
data checks establish corpus readiness; they do not establish a model gain.

| Type | V9 evidence | V10 training change |
| --- | --- | --- |
| S1.1.6 | 21/100 failures, all correct lookups with stop instead of continue | Replace 800 final thoughts. Bind the exact user request, confirmed ID, and symbol; explain the requested standalone or later-workflow scope. |
| S1.3.8 | 77/125 failures, including 48 structured-output failures; every standalone training row carried one fixed scope sentence | Remove that sentence from 750 user messages across 500 standalone rows. Replace all 1,000 final thoughts with visible-scope reasoning, resolved status, the exact stop/continue decision, and the three required result keys. |
| S1.8.4 | 22 failures have correct results and only the wrong continuation decision | Replace 666 final thoughts. Quote the actual outer request wrapper and distinguish it from workflow language inside the metric question. |

All 500 S1.3.8 direct/workflow pairs retain their identities, chosen IDs,
candidate order, 16 selection replies, and scope-placement assignments.
Workflow prompts stay unchanged. Standalone prompts use the existing
training wrappers without the fixed extra sentence; neither user turn
requests later work. No held-out wording or names enter training.
S1.1.6 and S1.8.4 retain all prompts. Every type retains its answers, calls,
tool outputs, evidence, metadata, row IDs, and ordering.

The aggregate has 55,276 training rows, with exactly 2,466 changed rows and
52,810 byte-identical rows. Validation and test each retain 6,875 rows.
All 22 held-out component/aggregate/panel files and 287 snapshotted historical
artifacts are unchanged. The test panel remains
`eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904`.
The shared contract and historical overlays remain unchanged. V10 uses
overlay schema v9; the schema number and corpus version are separate.

All component builds validate all splits. Aggregate training semantic
validation reports zero violations. Frozen validation/test aggregate reports
remain byte-identical to v9, including the existing 251/148 cross-component
evidence-ID collisions and no other violations. Keep those frozen IDs.
The actual training tokenizer and supervision audit covers all 69,026 rows,
including segment and structured-field masks. Maximum lengths are 4,025
training, 1,640 validation, and 1,372 test tokens, below 4,096.
Validator selftest rejects all 53 known bad fixtures with no false positives.

Build logs, historical reproduction checks, scoped mutation tests, exact
change audit, and tokenizer receipts are under `diagnostics/s1_v10_build/`.
`build_verification.json` is the consolidated readiness record. The generated
question catalog now uses v10 and includes complete examples of both scopes.
Version registration allows the existing generation and contract validators
to recognize v10. The training recipe remains the v9 32-node, batch-32,
1,728-update recipe with segment telemetry and the same S0 parent. Dedicated
launch configs and the real training preflight were completed for the
submission recorded below.

Rebuild from the repository root with the `pytorch-rocm` Python:

```bash
for section in 1 2 3 4 6 7 8 9; do
  python -m runtime.world_model.s1_${section}_corpus \
    --overlay runtime/world_model/s1_v10.yaml
done
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v10.yaml
python diagnostics/s1_v10_build/audit_corpus_changes.py
python scripts/audit_world_model_s1_decorrelation.py \
  --candidate data/world_model_v4/corpora/s1_v10 \
  --overlay runtime/world_model/s1_v10.yaml \
  --baseline-pins diagnostics/s1_v10_build/baseline_pins.json \
  --segment-loss-telemetry --workers 8 \
  --output diagnostics/s1_v10_build/final_audit.json
```

The v10 training SHA-256 is
`af7097e1780a91f7c82207483e344107a9c8ce6142d53fc85a4e76149b989d6c`.
The frozen source index remains the build input. The previously recorded
historical raw dendrogram/live-flist limitations are separate from these
passing corpus checks; no source snapshot was replaced. The old v9 paired
robustness config also retains its original implementation pins and correctly
rejects the revised builder; a future v10 diagnostic needs new pins.

### V10 training and evaluation submission (2026-10-26)

The user authorized training and evaluation after the corpus rebuild. All
three jobs were accepted and their resources/dependencies verified by Slurm:

| Job | Work | Resources | Dependency | State at verification |
| --- | --- | --- | --- | --- |
| 5768596 | V10 training | 32 nodes, 256 tasks, `extended`, 16h | None | Pending resources |
| 5768597 | Frozen S1 test, `structured_95_v1` | 512 nodes/tasks, `batch`, 4h | `afterok:5768596` | Pending dependency |
| 5768598 | S0 retention | 512 nodes/tasks, `batch`, 4h | `afterok:5768596,afterany:5768597` | Pending dependency |

Training uses `config/world_model_s1_120b_training_v10_32n.json`. The recipe
is byte-equivalent to v9 after normalizing versioned corpus/output identity:
one epoch, batch 32, 1,728 updates, learning rate 1e-4, r32 LoRA, 4,096-token
context, and S0 parent `slurm-5631883`. Segment loss telemetry and immutable
adapter snapshots at updates 400/800/1200/1600 remain enabled.

The S1 config is
`config/world_model_s1_120b_32n_test_v10_structured95_512n.json`; the retention
config is `config/world_model_s1_120b_s0_forgetting_test_v10_512n.json`.
The 6,875-row S1 and 120,766-row S0 panels, generation settings, and gates
remain unchanged. Evaluation writes to the training W&B run in sequence;
S0 still runs if the S1 job fails, provided training succeeds. Exactly these
three jobs were queued.

Both evaluations target the final checkpoint:

```text
data/world_model_v4/checkpoints/s1_v10_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5768596
```

The actual training launcher CPU preflight passed, as did five evaluation
config readiness checks and the final static input audit. Evaluation tests
use explicit future-checkpoint fixtures/stubs; the production launchers
validate the real completed checkpoint and exposure before generation.
GPU execution and an accuracy gain are not established by these checks.

Preparation receipts are under `diagnostics/s1_v10_preparation/`. The launch
record is `data/world_model_v4/launch_reports/s1_v10_120b_32n/`, including
`submission_receipt.json`, `launch_verification.json`, all three config
copies, readiness/build evidence, and 128 verified source snapshots under
`submitted_code/`. Preserve the corpus, configs, overlay, and live
runtime/launcher code while these jobs are queued or running. The earlier
build-only receipt retains its historical no-submission state; the launch
receipt is the current submission record.

## V9 results and compact W&B logging (2026-10-23)

Training completed, but there is no complete V9 S1 promotion result.
Slurm reports these outcomes:

| Job | Work | Outcome |
| --- | --- | --- |
| 5733083 | Training | Completed, 1,728 updates |
| 5733084 | Frozen S1 test | Aborted in generation shard 444; five rows missing |
| 5733085 | S0 retention | Completed; every gate check passed |
| 5733086 | Paired robustness diagnostic | Completed; diagnostic only |
| 5733087 | Checkpoint validation diagnostic | Aborted on the same chat-template error |

### Saved S1 predictions

The 511 complete shards and the failed shard's eight saved rows contain
6,870 of 6,875 questions. Their structured answer accuracy is
**6,539/6,870 (95.18%)**. Rescoring V8 on exactly these rows with the same
current scorer gives **6,177/6,870 (89.91%)**. V9 improves 550 rows and
regresses 188, a net gain of 362 correct answers or 5.27 percentage points.
This metric scores structured fields and excludes free-text summary wording.

| Family | Rows available | V8 structured accuracy | V9 structured accuracy |
| --- | ---: | ---: | ---: |
| Symbol to identifier | 1,250 | 94.88% | 98.40% |
| Identifier to symbol | 1,375 | 82.40% | 99.56% |
| Ambiguous symbol | 1,125 | 90.31% | 93.16% |
| Cross-context alignment | 999 | 99.50% | 99.80% |
| Layer tag parsing | 624 | 92.79% | 94.71% |
| Layer family claims | 748 | 90.78% | 89.30% |
| Metric direction conventions | 499 | 68.94% | 81.56% |
| Module ID parsing | 250 | 98.40% | 91.60% |

Direction conventions account for 92 of the 331 incorrect saved rows;
layer claims account for 80 and ambiguous symbols for 77. Module parsing
regresses by 17 correct answers despite the aggregate improvement. Tool
selection and arguments each score 99.96% on available rows. Structured
JSON validity is 99.13% and valid trajectories score 99.23%.

These are incomplete diagnostics, not promotion scores. The missing rows
are excluded from both models' denominators. No missing prediction is
invented, and no partial score is published to the training W&B run.
The audit verifies shard hashes, shared generation identity, resume
receipts, original panel coordinates for evidence IDs, and frozen input
pins. Reproduce it from the repository root:

```bash
/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python \
  diagnostics/s1_v9_results/review_results.py
```

The machine-readable result and input receipts are in
`diagnostics/s1_v9_results/partial_s1_results.json`. Slurm status and
verification logs are in the same directory. The full V8 S1 evaluation
scored 6,182/6,875 (89.92%) and failed its promotion gate.

All 65 task types with test rows are broken out in
`diagnostics/s1_v9_results/task_type_results.csv`, including matched V8
accuracy, result accuracy, continuation accuracy, and missing-row counts.
S1.6.7 has no held-out rows. The largest remaining weaknesses include:

| Task type | Case | Correct / available | Structured accuracy |
| --- | --- | ---: | ---: |
| S1.3.8 | User chooses an ambiguous identifier | 50 / 125 | 40.00% |
| S1.7.8 | Claims for the other layer family | 17 / 37 | 45.95% |
| S1.8.4 | Workflow-style metric truth | 52 / 83 | 62.65% |
| S1.6.4 | Layer list filtering | 68 / 94 | 72.34% |
| S1.7.4 | Layer claim checking | 139 / 187 | 74.33% |

### Retention, robustness, and training loss

V9 S0 retention passes every check. Result accuracy is
**120,681/120,766 (99.9296%)**; continuation accuracy is 99.7955%.
V8 result accuracy was 99.8857%, and its ambiguous-deferral check failed.
The authoritative V9 report is
`data/world_model_v4/eval_runs/s1_v9_120b_s0_forgetting/oss120b-plain-base-tokenizer-lora-r32/slurm-5733085/report/test_metrics.json`.

The separate 344-row robustness panel scores **296/344 (86.05%)**.
Both variants are correct in **119/160 pairs (74.38%)**. Identifier
notation/case is weakest: 48/64 rows and 17/32 pairs are correct. Task
scope and wording each score 82/96 rows. This is a training-owned
diagnostic and must remain separate from the frozen promotion panel.
Its report is
`data/world_model_v4/eval_runs/s1_v9_120b_32n_robustness/oss120b-plain-base-tokenizer-lora-r32/slurm-5733086/report/robustness_metrics.json`.

Validation loss falls from 1.87021 to 0.49227. The last logged training
window, at update 1,720, has loss 0.000535. Reasoning validation NLL is
1.79146; structured-final NLL is 0.13464 and tool-call NLL is 0.0000148.
The train/validation gap and robustness errors show that near-zero
training loss does not establish reliable held-out behavior. The failed
checkpoint diagnostic provides no complete checkpoint-selection result.

### Parser repair

Both failed jobs reached a chat-template exception because the model's
parsed thinking contained a nested `<|channel|>` tag. The parser now
returns `analysis_contains_channel_tag` before any tool execution or
prompt rendering. The existing row-failure path preserves the raw
generation and allows other rows to continue. Unexpected harness errors
still raise normally. Frozen questions, answers, generation limits, and
scoring definitions are unchanged. The later user-selected gate is recorded
below. No replacement GPU jobs were submitted.

### User-selected 95% overall gate (2026-10-23)

The user selected 95% overall structured accuracy, with task-type scores
reported separately. `structured_95_v1` checks the point estimate across
all 6,875 frozen panel rows, including metric-name variants. It excludes
free-text summary wording and checks blocked-reason presence, not wording.
It requires offline execution and preserves the existing complete-coverage,
checkpoint, tokenizer, panel, evidence, and publication identity checks.
It replaces the model-performance thresholds for this new evaluation only;
it does not require 95% per family or use a confidence-interval bound.

The separate config is
`config/world_model_s1_120b_32n_test_v9_structured95_512n.json`.
It changes only the gate, evaluation ID, and output root. The original
`tiered_v3` config, historical gate behavior, training recipe, panel bytes,
and generation settings remain intact. S0 retention criteria are unchanged.

The completed test needs at least 6,532 correct rows to pass. If the 6,870
saved outcomes stay unchanged, the five missing rows bound the final score
between 6,539/6,875 (95.1127%) and 6,544/6,875 (95.1855%). The two saved
nested-channel final actions already fail structured accuracy, so rejecting
them does not directly lower that count. A fresh GPU generation may still
change other answers despite greedy decoding and a fixed seed. In the
five-miss case, eight additional net regressions would drop below 95%.
The new gate has no completed result until the full rerun finishes.

### Evaluation recovery

Run a fresh frozen S1 test on the completed `slurm-5733083` checkpoint.
The launcher can resume saved shards, but two saved final actions contain
nested channel tags too: `wm3_s1_7_70449430aec57e5b94357504` and
`wm3_s1_6_c1af612eac57af73f6129e5e`. Both already fail structured accuracy,
but the old parser counted their trajectories as valid. A fresh run applies
the repaired parser to every row and gives one consistent gate result.
Keep the 6,875-row panel and retained reasoning; use `structured_95_v1`.
The actual completed-checkpoint launcher preflight passes; its receipt is
`diagnostics/s1_v9_structured95/contract_preflight.log`.

From the repository root, activate `pytorch-rocm` and submit:

```bash
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
export CHECKPOINT_PATH=data/world_model_v4/checkpoints/s1_v9_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5733083
export RUN_CONFIG=config/world_model_s1_120b_32n_test_v9_structured95_512n.json
export S1_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32
export CONTRACT_ONLY=0 PUBLISH_WANDB=1 S1_RESUME_SHARDS_ROOT=
sbatch --nodes=512 --ntasks=512 --partition=batch --time=04:00:00 \
  --job-name=s1_v9_eval95 --export=ALL world_model_s1_eval.slurm
```

The launcher writes to a new job directory, merges all shards, checks full
coverage, scores the gate, and publishes only the S1 headline accuracy and
gate. An incomplete generation must not publish a gate result. S0 retention
and robustness already completed; their results remain valid for this
checkpoint. Retraining is unnecessary for evaluation recovery.

Rerun the separate checkpoint diagnostic on 32 nodes to compare updates
400/800/1200/1600/1728. It needs a fresh output directory and code manifest:
the original submission hashes predate the parser and logging fixes.
The replacement manifest pins the same 127 source paths at
`diagnostics/s1_v9_structured95/diagnostic_code.sha256`; the original
submission artifacts are preserved. With the environment above active:

```bash
export RUN_CONFIG=config/world_model_s1_120b_checkpoint_diagnostics_v9_32n.json
export RUN_CONFIG_SHA256="$(sha256sum "$RUN_CONFIG" | awk '{print $1}')"
export DIAGNOSTIC_CODE_MANIFEST=diagnostics/s1_v9_structured95/diagnostic_code.sha256
export DIAGNOSTIC_CODE_MANIFEST_SHA256="$(sha256sum "$DIAGNOSTIC_CODE_MANIFEST" | awk '{print $1}')"
export OUTPUT_DIR=data/world_model_v4/eval_runs/s1_v9_checkpoint_diagnostics/structured95-20261023
sha256sum --check --strict --quiet "$DIAGNOSTIC_CODE_MANIFEST"
sbatch --nodes=32 --ntasks=32 --partition=extended --time=16:00:00 \
  --job-name=s1_v9_checkpoints_fixed --export=ALL world_model_s1_checkpoint_diagnostics.slurm
```

These commands are prepared, not submitted. The checkpoint diagnostic
publishes to its own child run and does not determine promotion eligibility.
Its CPU preparation also passes against all five real adapters and the
frozen validation panels; the plan is under
`diagnostics/s1_v9_structured95/checkpoint_preflight/`.

### Compact W&B metrics with segment losses

The training telemetry expanded into 3,585 training and 3,185 validation
diagnostic metric names, before ordinary scalar metrics. The user requested
segment losses after the initial six-metric reduction. Training now publishes
eight curves: total loss plus reasoning, tool-call, and structured-answer
NLL, each for training and validation. Four post-training metrics bring the
full training/test lifecycle to twelve. This keeps training below ten curves.

| W&B key | Meaning |
| --- | --- |
| `train/loss` | Training objective loss |
| `eval/loss` | Validation objective loss |
| `train/nll/reasoning`, `eval/nll/reasoning` | Mean reasoning token NLL |
| `train/nll/tool_call`, `eval/nll/tool_call` | Mean tool-call token NLL |
| `train/nll/structured_final`, `eval/nll/structured_final` | Mean structured-answer token NLL |
| `test/s1/structured_answer_accuracy` | Structured S1 fields, excluding summary wording |
| `test/s1/gate_passed` | Complete S1 promotion gate, 0 or 1 |
| `s0_forgetting/result_exact_rate` | S0 retained result accuracy |
| `s0_forgetting/gate_passed` | Complete S0 retention gate, 0 or 1 |

`global_step` remains a hidden chart axis. Tracking metadata and W&B system
telemetry are separate from these application metrics. Standalone S0
tests retain the `test/` namespace with only result accuracy and gate status.
An evaluation that has not finished produces no gate value; absence is not
reported as a pass or a model failure.

Full token, segment, field, family, question-type, throughput, and learning
rate diagnostics remain in the checkpoint directory's `throughput.jsonl`
and `validation_metrics.jsonl`. Complete evaluation breakdowns remain in
each evaluation's `report/test_metrics.json` and exact per-row report.
New training runs include `detailed_metrics_dir` in W&B metadata. This
change affects future logging; previously uploaded W&B history is retained.

Segment NLL uses pooled token sums/counts. It is teacher-forced loss, not
generated-answer accuracy. Empty segments are omitted rather than logged
as zero. No field, family, or question-type cross-products are sent to W&B.
The existing V9 telemetry already contains all five segments:

| Segment | Last training window (update 1720) | Final validation (update 1728) |
| --- | ---: | ---: |
| Reasoning | 0.00146930 | 1.79145587 |
| Tool calls | 0.00001475 | 0.00001477 |
| Structured answer | 0.00001121 | 0.13463588 |
| Free text | 0.00008345 | 0.17891310 |
| Formatting | 0.00002646 | 0.27276411 |

Free-text and formatting breakdowns remain local. The input files and their
hashes are recorded in `diagnostics/s1_v9_structured95/loss_snapshot.json`.

The separate checkpoint diagnostic now publishes eight scalar curves:
S1 structured accuracy, S0 result accuracy, and reasoning/tool-call/structured-
final NLL for each of S1 and S0. Its step axis is hidden. All other accuracy
and field breakdowns remain in `report.json`; both full diagnostic plot
images remain beside that report. The child run no longer publishes those
images or expands field metrics into additional W&B series.

Current verification passes 80 targeted tests for compact segment logging,
the exact 95% boundary, independent network checks, preserved historical
profiles, unchanged rerun inputs, scoring, and training telemetry. The log
is `diagnostics/s1_v9_structured95/regression_tests.log`.

Earlier recovery verification passed 58 targeted tests for both W&B metric limits,
local detail preservation, loss telemetry, malformed reasoning, shard
resumption, and checkpoint diagnostic accounting. The log is
`diagnostics/s1_v9_eval_recovery/regression_tests.log`.

The earlier six-metric verification passed 80 tests across compact publication, loss telemetry,
S1 generation/scoring, and S0 scoring. One legacy S0 bundle test cannot run
because `data/world_model_v2/eval/s0_human_identifier_trajectories_v6/manifest.json`
is absent. The new tests verify the six-metric limit, independent gate
failures, structured accuracy instead of summary exactness, retained local
payloads, and malformed reasoning under both history policies.

## V9 targeted corpus build (2026-10-22)

The user approved one full training run after corpus preparation, without
pilots. V9 combines v6 S1.1–S1.4 with selective v8 S1.6–S1.9 coverage and
the targeted changes below. The separate v7 prompt experiment stays separate.
The shared contract and historical overlays remain byte-identical.

- **Identity:** S1.1, S1.2, and S1.4 reproduce v6 training rows exactly.
  This protects resolved IDs without symbols and the reviewed continuation
  reasoning. All S1.3 forms except S1.3.8 also retain their v6 rows.
- **S1.3.8:** Its 500 direct/workflow pairs put task scope in both user
  turns for 250 pairs, only the earlier turn for 125, and only the latest
  turn for 125. Sixteen content-pinned selection replies vary independently
  of identity, scope, and placement. Earlier ambiguity summaries no longer
  say "to continue" when merely asking for an ID. Final reasoning separates
  the user's identity choice from task completion. Results, candidates,
  evidence, and tool calls remain unchanged.
- **S1.6.1–4/6:** Reasoning binds all seven parsed fields to concrete
  values. Filters explain matching and rejected entries in input order,
  with zero, partial, and complete matches under both scopes. The task's
  scope determines stop/continue independently of the filter outcome.
  Forms 5 and 7 retain their existing training bytes.
- **S1.7.2/4/8:** Ordered family-filter partitions and native HumanNet
  mappings are explicit. All eight canonical claims receive 187 or 188
  examples, totaling 1,500. Supportable claims cover positive and negative
  verdicts, and every applicable paraphrase covers both scopes. Negative
  verdicts remain resolved. The three existing unmapped HumanNet names
  cover both scopes, all wrappers, and all requests; no new native names
  or held-out facts are introduced.
- **S1.8:** Match polarity and task scope across names while preserving
  name-category allocations. Variant opposite-scope pairs increase from
  94 to 176, the maximum within the existing budget. Shared-alias coverage
  increases from 113 to all 128 polarity/scope cells. The 312 canonical,
  64 ambiguous, and 160 unknown cells stay covered. Reviewed reasoning is
  retained; form 6 moves the undetermined truth statement before the result
  schema, keeping name, convention, truth, schema, then continuation order.
- **S1.9:** Retain v8 request/wrapper decorrelation exactly.
- **S0 replay:** Select 92 resolved symbols, 56 named IDs, 56 unnamed IDs,
  48 ambiguous symbols, 16 retired IDs, and 8 historical symbols from
  existing training rows without replacement. Each category splits evenly
  between direct and workflow framing. Ambiguous rows cover candidate
  counts and canonical/casefold presentation. Only existing aggregate
  evidence normalization and metadata normalization alter source copies.

The aggregate contains 55,276 training rows, 6,875 validation rows, and
6,875 test rows. Training semantic validation reports zero violations.
All 22 held-out files match v6; 244 snapshotted historical files retain
their hashes. Frozen held-out aggregate semantic reports reproduce the
existing 251/148 cross-component evidence-ID collisions, with no other
violations. Component validation is clean. Keep those frozen IDs unchanged.

The actual tokenizer audit covers all 69,026 rows and their supervision
masks. Maximum lengths are 4,025 training, 1,640 validation, and 1,372 test
tokens, below 4,096. Evidence, hashes, coverage, and test logs are under
`diagnostics/s1_v9_build/`; the generated question catalog contains full
v9 gold traces and the actual original system prompt.

Regression suites pass 206 S1.3 tests, 192 S1.6/S1.7 tests, and 96 S1.8
tests; suites overlap, so these are not unique-test totals. The shared
corpus integration suite passes 88 tests, skips one unavailable-source
case, and excludes the failing live-flist check described below. Repeated
aggregate builds reproduce all seven corpus/panel files for v6, v8, and
v9. `diagnostics/s1_v9_build/build_verification.json` records the commands,
results, input pins, and verification artifact hashes.

The separate live raw-source checks remain unavailable for this frozen
contract: its historical dendrogram is absent, and the live flist has 213
rows where the frozen source expects 353. These are external source changes,
not revised corpus labels. The builds verify and use the pinned source
index. Keep the historical source pins unchanged; see
`diagnostics/s1_v9_build/source_availability.json`.

### V9 training and dependent evaluations

The user authorized submission after corpus verification. Training job
`5733083` uses `config/world_model_s1_120b_training_v9_32n.json`, S0 parent
`slurm-5631883`, and the unchanged 32-node recipe: batch 32, learning rate
1e-4, one epoch, and 1,728 updates. No pilot runs were submitted.

The existing opt-in loss telemetry from the context-audit checkout is now
integrated and enabled by this config. It reports reasoning, tool calls,
structured final JSON, free text, and formatting, plus subordinate value
losses for status, result, continuation, and provenance. Overall, S0/S1,
family, and source-trace metrics use raw token sums/counts before division.
These additive diagnostics preserve training tokens, masks, objective values,
and gradients; historical configs keep their defaults. The actual tokenizer
audit with telemetry enabled checks all 69,026 rows and preserves the same
4,025/1,640/1,372 token maxima. Telemetry tests pass 67 checks; launcher
regressions pass three, launch config suites pass 208 with one skipped, and
checkpoint diagnostic tests pass ten. Gold diagnostic scoring checks pass
all 6,875 S1 and 4,096 S0 validation rows.

| Job | Purpose | Resources | Dependency |
|---|---|---|---|
| 5733083 | Full v9 training | 32 nodes, extended, 16 hours | None |
| 5733084 | Official S1 tiered_v3 gate | 512 nodes, batch, 4 hours | Training succeeds |
| 5733085 | Official S0 retention gate | 512 nodes, batch, 4 hours | Training succeeds; S1 gate finishes |
| 5733086 | Separate 344-row robustness diagnostic | 32 nodes, extended, 4 hours | Training succeeds |
| 5733087 | Checkpoint loss/accuracy diagnostics | 32 nodes, extended, 16 hours | Training succeeds |

All submissions and scheduler dependencies were verified. Training was pending
for priority at recording time; evaluations were pending on dependencies.
The final checkpoint output is
`data/world_model_v4/checkpoints/s1_v9_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5733083/`.

The trainer captures immutable inference adapters at updates 400, 800, 1,200,
and 1,600 before rolling resume checkpoints can be pruned. Partial exposure
receipts count the actual consumed rows and cannot qualify as a final model.
The diagnostic evaluates these four adapters and the final update 1,728 on
full S1 validation and S0 behavioral validation. It measures S0 reference
loss separately on the existing 622-row teacher-forced panel. Exact checkpoint,
panel, row, and shard identities bind loss/accuracy joins; generated answers
precede reference-loss reads. Plots and descriptive correlations go to a
separate W&B run in `jail-ai/mentor-sft`, grouped by the training run.

The robustness rows, questions, and answer keys remain byte-identical to the
existing 344-row diagnostic. Its new v9 manifest pins the current implementation
without changing the frozen v1 manifest. Official gates keep their frozen
panels, criteria, generation settings, and final-checkpoint requirements.

Readiness, static input checks, telemetry tokenizer checks, and runtime-source
pins are under `diagnostics/s1_v9_preparation/`. Submission receipts, all five
configs, scheduler records, and exact submitted code copies are under
`data/world_model_v4/launch_reports/s1_v9_120b_32n/`. Preserve these submitted
inputs and code while the jobs are queued or running. This submission is not
a completed training result or a gate pass.

## Retained reasoning in inference and evaluation (2026-10-21)

The user authorized the inference fix and a full v6 S1 gate rerun. The
matched 260-row validation comparison found v6 structured accuracy improved
from 180/260 (69.23%) to 244/260 (93.85%) when current-turn reasoning stayed
available after tools. Retained-context v5 scored 231/260 (88.85%). The v6
change fixed 64 rows with no regressions. This diagnostic did not establish
a promotion pass. Detailed evidence is in the isolated context-audit
checkout's stage document and its `s1-context-ab-20261020T184455Z` snapshot.

The shared S1 inference runner now defaults to `retain_current_turn` in its
single-trajectory API, batched API, production generator, and CLI. It keeps
the model's own generated analysis through the current turn's tool calls.
Reasoning from earlier user turns remains excluded. Explicit
`--history-policy strip` reproduces the historical context behavior.
New generation plans and complete/partial manifests record the policy;
resume and shard merging reject mismatched policies. Historical manifests
without a policy field retain their old strip interpretation.

The full evaluator validates and exports `S1_HISTORY_POLICY`, passes it to
generation, and records it in `run_contract/launch.env`. The split-aware
question validator used by the diagnostic is included in the primary
checkout; full gate generation still requires the test split by default.
The independent S0 runner already retains current-turn reasoning. The
separate DPO actor/verifier and one-shot QA paths do not use this S1 loop.

The new config is
`config/world_model_s1_120b_32n_test_v6_retained_512n.json`. Relative to the
frozen original v6 test config, only its evaluation ID, output root, and
explicit history policy differ. It uses the same completed `slurm-5712804`
checkpoint, all 6,875 questions and answer keys, `tiered_v3` criteria,
tokenizer, seed, batch size, token limits, and 512-node resource layout.
No corpus, original run config, contract, or scoring rule was changed.

The prepared read-only launch snapshot is
`../mentor-rl-s1-v6-retained-gate-20261021T152645Z/`.
Its `snapshot_manifest.json` pins all 136 code/config/input files. The
checkpoint and frozen inputs are exact copies at their canonical paths;
later edits in either working checkout cannot change this run. The real
launcher contract preflight passes, and the generation bundle contains
all 6,875 frozen questions. The runner, full-gate setup, scorer, and retry
behavior checks pass: 166 passed, one skipped. The separate focused runner
and real-tokenizer parity suite passed 25 tests in the audit checkout.
Twenty-four additional retry-artifact integration checks reject the frozen
diagnostics' old implementation pins after this code change. Those original
artifacts remain frozen and are not inputs to this promotion gate. A future
retry rerun needs separately prepared artifacts for the updated code.

Submitted at 11:33 ET on 2026-10-21 as **Slurm job 5724278**,
`wm_s1_v6_retained`, on `batch`: 512 nodes, eight GPUs per node, four-hour
limit. The scheduler confirmed it pending for priority. The snapshot's
`launch_receipt.json` records submission, exact config and snapshot hashes,
checkpoint, history policy, and checks. The standard evaluator will append
its metrics to the originating v6 W&B run when complete.

Results will be written inside the snapshot at
`data/world_model_v4/eval_runs/s1_v6_120b_32n_test_retained/oss120b-plain-base-tokenizer-lora-r32/slurm-5724278/`.
The same run path in the primary checkout links to that directory. The final
gate files will be `report/test_metrics.json` and
`report/exact_s1_report.json`. Submission does not establish a gate result.

## Training coverage audit (2026-10-21)

The audit compares the frozen v6 training rows with the retained-context
260-row validation diagnostic and the earlier full test (`slurm-5712872`).
The corrected full test, `5724278`, is still pending for scheduler priority.
The counts below are structured-answer failures from the earlier full test,
not results from that pending run. Summary wording is not scored.

The focus is questions whose expected answer uses no new tool call. The
retained-context diagnostic passed all 124 tool-requiring questions, but its
four examples per type miss important weaknesses. For example, its four
S1.7.4 questions all check the same supported claim. Their success does not
cover canonicalization of the other claims or unsupported verdicts.

| Question type | Earlier full-test failures | Coverage finding | Proposed correction |
| --- | ---: | --- | --- |
| HumanNet catch-all claim cards, S1.7.8 | 32/37 | 600 rows repeat only three unmapped network names, 200 times each. All targets correctly map them to `other` with no supported claims. | Contrast mapped exceptions with the default `other` rule using training-owned inputs. Vary wording and context while requiring the same family and card. More copies of the same names alone are unlikely to help. |
| Canonical claim checking, S1.7.4 | 44/187 | All 44 failures have the wrong or missing canonical `claim`: 25 invented labels, 14 wrong allowed labels, five missing. The 1,500 training rows contain all 256 reviewed paraphrases. | Emphasize phrase-to-canonical-name normalization and distinctions between nearby claim meanings, before looking up family support. Keep the existing positive/negative contrasts. |
| Layer-list filtering, S1.6.4 | 33/94 | All 288 homogeneous scPEN lists require `continue`; all 22 `bulk_pen` zero-match cases require `stop`. One failed standalone pattern has 62 semantic training matches, all `continue`. | Decouple list composition and filter outcome from framing. Cross feasible family mixtures and zero/partial/all-match outcomes with both task scopes. Preserve every parsed field when only the filter changes. |
| Canonical metric questions, S1.8.1-4 | 98/280 | 1,600 canonical examples; forms 2-4 cover 306 of 312 metric/form/polarity/framing cells. Only one failure falls in an empty cell. Of 98 errors, 48 use the wrong result fields and 29 have only the wrong continuation. | Test the prepared task-scope prompt separately. Use training-only contrasts that hold the metric and proposition fixed while changing scope; reinforce the invariant result schema. Fill the six holes, but do not treat them as the main cause. |
| Ambiguous/unknown metrics, S1.8.5-6 | 27/50 and 17/50 | 400 correct training examples of each category. Failures invent field names, assume an unsupported convention, or confuse `not_found` with `invalid_input`. | Contrast known, ambiguous, and unsupported names, binding each to its exact fields, status, and decision. Keep name-category schema selection separate from direction truth. |
| Module parsing, S1.9.2 | 7/125 | 1,000 correct unique IDs, but each of 64 wrappers appears with exactly one of eight request variants: 64 of 512 possible combinations. | Schedule wrappers and request variants independently. Preserve all five fields and `existence_asserted:false` across wording changes. |
| User-selected candidate, S1.3.8 | 14/125 | Already has 500 exact direct/workflow pairs; all 14 failures get continuation wrong. Both user turns repeat the same framing, and replies use only four stems. | First test explicit task-scope instructions. If data work remains necessary, vary placement of the scope cue and selection wording while preserving candidate facts. Do not add duplicate pairs as though they were absent. |

The direction totals above exclude the 80 metric-variant and 40 shared-alias
test rows, which remain report-only under the existing gate. Ambiguous and
unknown metrics still contribute to the gate. Six retained-validation alias
failures receive an unclassified label from the scorer's test-vocabulary
classifier; that label alone must not be used to infer test gate inclusion.

Independent parsing found no incorrect gold layer objects, filter matches,
or module parses in the inspected 750 filter and 1,000 module examples.
Every filter target includes all seven LayerRef fields, including `network`;
every module target includes all five required fields. The 1,500 claim-check
targets and 600 catch-all cards are also consistent with their rules.
All 5,000 direction component rows reproduce from the frozen builder. The
evidence supports weak transfer and correlated sampling, not corrupted gold
schemas. Some generated reasoning states a correct rule and then contradicts
it in concrete assignments. That is a plausible mechanism for bad answers,
but it does not establish that reasoning length caused the regression.

The filter coupling comes from occurrence parity in
`runtime/world_model/s1_6_corpus.py` (`_diverse_list_filter_parts`), which is
also tied to direct/workflow assignment. The module coupling comes from the
wrapper and request indices sharing the same counter in
`runtime/world_model/s1_9_corpus.py` (`select_scenarios`). These are the first
concrete builder defects to fix in a new training-only version. Add coverage
checks for these joint combinations rather than checking marginal wrapper
and outcome counts alone. Keep the current training row budget where
possible and retain the baseline recipe and optimizer-step count.

For S1.7.4, 127 of the 128 supportable paraphrases already occur in both
supported and unsupported family contexts. The gap is not a general absence
of positive/negative pairs. Unsupported-only concepts have only three or
four examples per phrase, versus seven to eleven for supportable concepts.
Training should strengthen canonical semantic distinctions without copying
held-out wording. For HumanNet catch-all generalization, the frozen training
source provides only three unmapped names. Use source-backed contrastive
views; any wider native-name bank requires a separately reviewed, split-safe
source extension. Do not import validation or test network names.

The inference repair alone is unlikely to pass the gate. Under the explicit
assumption that earlier outputs with both expected and actual zero calls
stay unchanged, while every other gate-included answer becomes correct,
249 result errors remain among 6,755 rows. Result accuracy would be at most
96.31% (95% Wilson interval 95.84-96.74%), below the existing 98% decision
threshold. This is a conditional bound, not a new model score. The new full
run is still required to measure actual behavior.

Next steps are to fix the two demonstrated sampling couplings, then review
targeted claim/schema contrasts. Keep that data experiment separate from
the already prepared v7 task-scope prompt and any loss-weight experiment.
Retain separate reasoning/structured loss reporting and judge progress by
generated structured accuracy. Do not lengthen every trace or increase
epochs based on this audit. Before training, verify frozen held-out bytes,
passing-type preservation, joint coverage, semantic labels, tokenizer
budgets, and reproduction of historical corpus versions. No corpus,
training config, shared contract, model checkpoint, or job submission was
changed by this audit.

Evidence, input hashes, representative rows, and detailed counts are in
`diagnostics/s1_training_coverage_audit_20261021/`: `overview.json`,
`claims_decisions.json`, `parsers.json`, `directions.json`, and
`conditional_gate_ceiling.json`.

## Additional correlation audit (2026-10-21)

The user asked whether the layer-filter sampling problem occurs elsewhere.
The audit inspected all eight frozen v6 component builders and their actual
training rows. It found several additional associations between incidental
wording, presentation, and required behavior. These are verified corpus
properties. No counterfactual model generation was run, so they do not by
themselves establish that the model uses each shortcut or that it caused a
remaining gate failure. The corrected full gate is still pending.

| Association | Evidence in published training rows | Corrective direction |
| --- | --- | --- |
| S1.7.2 wrapper predicts whether a family filter matches | A majority lookup using only framing/wrapper identity fits 894/900 labels (99.33%); 814 rows belong to cells with only one outcome. For example, direct wrapper 2 has 14 matches and no misses, while wrapper 3 has 14 misses and no matches. | Schedule wrapper and match/no-match independently, within family and framing. |
| S1.7.4 request wording predicts claim support | Ten request variants (5-14) occur only in supported cases, covering 168 rows. Variant 0 has 207 supported and 652 unsupported rows. | Select generic request wording independently of family/claim frequency and support. Preserve the actual claim meaning and verdict. |
| Selectively restored wording banks reveal policy or outcome | S1.2 workflow registry cases use disjoint 64-wrapper banks for matching evidence/no call (562 rows) and unrelated evidence/lookup (281). Ordinary S1.1 workflow lookups likewise separate resolved (1,600) and not-found (600) by bank. S1.3 also has bank-exclusive wrappers that separate tool-use groups. | Reuse compatible reviewed wording across outcomes and exact/near-miss evidence cases. Keep success/failure prompts paired. First measure sensitivity under retained reasoning for types now looking healthy. |
| S1.4 wrapper determines mention order | All 8,000 rows tie even/odd wrapper index to which source is mentioned first. Every one of 64 framing/wrapper cells has only one order, although both orders are balanced overall. | Vary mention order independently within each wrapper and task scope. |
| S1.4 relevant evidence always appears first | All 7,000 module harnesses and 1,000 seed-set harnesses put the queried ID first and the distractor second. | Vary relevant-entry position independently of outcome and user mention order, using valid training facts. |
| S1.3 case style determines scope within perturbed inputs | In S1.3.5, all 96 lowercase perturbations are direct/stop and all 93 mixed-case perturbations are workflow/continue. | Assign case style independently of framing. Preserve case-insensitive identity semantics. |
| S1.9 wrapper fixes request phrasing | Each module form realizes only 64 of 512 wrapper/request combinations. Every wrapper uses exactly one request variant. | Independently cross wrappers with request phrasing; prioritize the failing module form. |

The majority-lookup rates above describe the corpus cross-tabs. They are
not model evaluation scores or out-of-sample classifier performance. The
S1.7.2 source cycles ordered match/no-match specifications and wrapper
indices together (`s1_7_corpus.py`, `_list_specs` and `select_scenarios`).
S1.7.4 advances request wording separately for each family/input/claim
combination; frequent supported combinations therefore reach wording that
rare unsupported combinations never see. These schedules can look balanced
when wrappers, outcomes, and families are checked only one dimension at a
time. The S1.4 order and S1.3 case problems also share occurrence parity
between otherwise independent choices.

Some further findings are coverage gaps or source restrictions, requiring
different treatment from an accidental scheduling defect:

- S1.8 ambiguous and unknown names each occur with only one of four
  statement polarities: 8/32 and 20/80 name/polarity combinations are present.
  Request variants also track polarity. All polarities should still defer
  for these names, so this is missing invariance coverage rather than an
  answer-label shortcut. Rotate polarities across repeated training names.
- S1.4 ID-only comparisons always use version-suffixed/lowercase forms for
  equal IDs (1,000 rows), and plain uppercase forms for unequal IDs (1,000).
  Input notation can substitute for actual normalization and comparison.
  Both types currently pass the frozen full test. A separate diagnostic
  should cross equality with notation before changing these passing types.
- S1.4 context pairs also predict outcomes: all abstract/seed-set examples
  are unequal; all MyGene/module examples match. Cross-context generalization
  needs source-backed opposite outcomes within those same context pairs.
- In S1.3.5, the sole surviving candidate is first in 2,030/2,070 training
  rows, and in all 259 validation and 259 test rows. This reflects the source
  data and canonical ordering. Reweight existing genuine nonfirst-survivor
  examples or add a separate diagnostic; do not falsify membership or
  reorder a contractually canonical list.
- Symbol/ID retry recovery occurs only in standalone requests, so every
  recovery stops. Workflow recovery/continue is absent. This is an untested
  combination; terminal and exhausted failures correctly defer.
- Selective wrapper restoration also changes user wording in 64/400 symbol
  and 110/800 ID success/failure pairs. Their first calls and inputs match.
  A future correction should keep user prompts identical across paired
  runtime outcomes.

Several shortcuts persist in the frozen held-out builders, including S1.4
mention order, first-position evidence, and notation/equality, plus S1.3
candidate position. A good frozen score alone cannot rule out those
heuristics. Keep the existing held-out files and gate unchanged; assess
these risks with a separately named diagnostic using training-owned inputs.

Negative checks limit the scope of the findings. S1.8 truth labels show no
comparable deterministic generic-wording shortcut in the inspected training
cross-tabs. S1.3 membership location is balanced between harness and user
request for each outcome. S1.4.6 candidate/noncandidate cases cover both
outcomes in every framing/wrapper cell. S1.1 exact and unrelated registry
blocks share their wrapper pools. Symbol/ID retry outcomes are independently
hashed and show ordinary sampling variation under permutation controls.
Species, identifier grammar, matching evidence, actual tool status, and
explicit retry allowance remain legitimate predictors.

Prioritize the demonstrated S1.6 and S1.7 sampling defects and the S1.9.2
wrapper/request lock in a new data-only experiment. Add joint coverage
checks that cross neutral wording with feasible outcomes, and independently
schedule wording, input presentation, framing, and runtime fault assignment.
For currently passing types, preserve their training rows while first
measuring sensitivity to wording swaps, evidence order, case, and notation.
All 92 identity and 36 S1.4 retained-validation questions passed, so the old
stripped-history full-test errors do not establish new causal evidence
against those types. Preserve source facts, closed vocabularies, frozen
held-out bytes, historical corpora, and the baseline training recipe.

The four detailed artifacts are
`diagnostics/s1_training_coverage_audit_20261021/correlations_s1_1_s1_2.json`,
`correlations_s1_3_s1_4.json`, `correlations_s1_6_s1_9.json`, and
`correlations_s1_7_s1_8.json`. They record actual counts, source references,
input hashes, negative checks, and interpretation limits. This turn changes
only audit artifacts and this stage record; no builder, corpus, config, or
job submission was changed.

## Training decorrelation sweep: v8 (2026-10-21)

The user authorized the overall sweep after the two audits above, including
the identified presentation problems in types that previously passed. This
supersedes the earlier recommendation to freeze those types while measuring
sensitivity. The new data overlay is `runtime/world_model/s1_v8.yaml`, with
component and aggregate outputs under `data/world_model_v4/corpora/s1_v8`
and evaluator outputs under `data/world_model_v4/eval/s1_v8`. This experiment
uses the v6 system prompt, reviewed reasoning, source facts, row allocations,
S0 replay, and training recipe. The separate v7 prompt experiment is unchanged.
No training or evaluation job is submitted by this data-fix step.

Each builder enables the changes only for training when `train_decorrelate`
is true. Facts and input choices are selected independently of wrapper-bank
size. The v8 overlay uses overlay schema v7: schema v6 was already reserved
by the separate prompt experiment. This schema label does not change the
v8 row settings. All historical overlays and the shared contract stay frozen.

| Targeted association | v8 correction and measured coverage |
| --- | --- |
| S1.1–S1.3 form-specific wording reveals outcomes or tool use | Reuse the reviewed common v5 banks across compatible forms. Select failure partners and distractors independently of wrapper identity. All 1,200 S1.1/S1.2 success/failure user prompts and first calls are identical within pairs. |
| S1.3 case style reveals task scope | Choose lower/mixed case independently of direct/workflow framing, retaining case-insensitive identity semantics. |
| S1.3.5 first-candidate shortcut | Reweight genuine nonfirst-survivor facts from 40/2,070 rows (1.93%) to 518/2,070 (25.02%). Keep canonical candidate order, actual membership, and all 299 original source symbols. Cross survivor position with wrapper, framing, and membership-evidence location. |
| S1.4 mention order and evidence position | Every form/framing/wrapper cell for the eight nonfailure forms covers both mention orders and both relevant-entry positions. Failure copies retain the same visible input as their success partner. |
| S1.4 ID notation reveals equality | Cross four item-relative notation styles with equal and unequal IDs: plain, case-only, and both versioned alternatives. Use actual source versions. The style assigned to the module versus seed item is also crossed with equality. |
| S1.6.4 list composition/filter/task scope | Independently schedule query, list composition, scope, and wording. Homogeneous scPEN lists now have 107 direct and 109 workflow rows. The formerly absent homogeneous-scPEN/bulk_pen-empty case has three of each scope. All 64 framing/wrapper cells contain match and no-match outcomes. Fine content-stratum scope counts differ by at most one; 283 of 391 strata are singletons, so this is not a claim that every exact content tuple appears in both scopes. |
| S1.7.2 wording predicts a filter match | Wrapper-only majority lookup falls from 894/900 (99.33%) to 456/900 (50.67%), close to the 50% class baseline. Every wrapper has both outcomes. |
| S1.7.4 request wording predicts claim support | Request-only majority lookup falls from 1,139/1,500 (75.93%) to 900/1,500 (60%), exactly the class-frequency baseline. All 24 request variants have both outcomes. Preserve the selected evidence, claim, and verdict. |
| S1.8 name/request locks to statement polarity | Every unknown/ambiguous name and request covers all four polarities in both framings. Fill six missing canonical metric/scope/polarity combinations by swapping scope in 12 same-truth canonical rows; canonical coverage is 312/312 cells. |
| S1.9 wrapper locks to request phrasing | Each module form now covers all 512 wrapper/request combinations instead of 64, preserving module IDs, task scope, and the absence of an existence claim. |

The lookup percentages describe training cross-tabs, not model accuracy.
The sweep does not establish which shortcuts the model used or whether a
retrained checkpoint passes the gate. The separate source-constrained S1.4
context-pair and workflow-retry coverage gaps remain follow-up curriculum
work. Frozen held-out rows retain their historical distributions; changing
them would invalidate the existing comparison.

Component checks cover semantic correctness, duplicate prompts, joint
coverage, exact v6 reproduction, historical no-overlay/v4/v5 reproduction,
and content invariance when wrapper banks change size. The validator
selftest accepts every clean fixture and rejects all 53 known-bad fixtures.
Published component reports and test logs are under
`diagnostics/s1_v8_decorrelation/`. The final aggregate audit is generated by:

```bash
PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 TOKENIZERS_PARALLELISM=false \
  /lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python \
  scripts/audit_world_model_s1_decorrelation.py --workers 8 \
  --output diagnostics/s1_v8_decorrelation/final_audit.json
```

This checks the real training serializer, pinned local tokenizer, supervision
masks, 4,096-token context limit, original system prompt, frozen baseline
hashes, and byte-identical held-out corpus and evaluator files. Aggregate
row counts remain 55,276 training, 6,875 validation, and 6,875 test examples.
The final audit records each split's actual token maximum and supervised-token
count, the tokenizer and serializer hashes, and all frozen-file comparisons.

## V8 comparison preparation (2026-10-21)

The user accepted two independent next steps while the retained-reasoning v6
gate is queued: a separate robustness diagnostic and the v8 training config
with CPU preflight. The published v8 corpus and the promotion panel remain
fixed. This preparation does not submit GPU jobs.

`config/world_model_s1_120b_training_v8_32n.json` changes only the versioned
run identity, corpus/evaluator paths and hashes, and checkpoint output path
from the v6 config. It starts from the same S0 checkpoint `slurm-5631883`,
with 32 nodes, global batch 32, uniform completion loss, learning rate 1e-4,
one epoch, and 1,728 optimizer steps. It does not initialize from the trained
v6 adapter. The existing training validator and actual shared launcher both
pass CPU contract preflight; 89 focused tests passed. Exact commands, recipe
parity, input hashes, and launch-time GPU checks are recorded in
`diagnostics/s1_v8_preparation/training_preflight_report.json`.

```bash
USE_TORCH=0 USE_TF=0 \
  PATH="/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin:$PATH" \
  RUN_CONFIG=config/world_model_s1_120b_training_v8_32n.json \
  S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 CONTRACT_ONLY=1 \
  bash world_model_v2_s0.slurm
```

The diagnostic builder is
`scripts/build_world_model_s1_robustness_panel.py`. Its separate output is
`data/world_model_v4/diagnostics/s1_robustness_v1/`, containing canonical
traces, answer-separated questions and keys, explicit pair/coverage metadata,
and a content-pinned manifest. It uses training-owned S1.3/S1.4/S1.6/S1.7/S1.9
facts and native builders to derive the expected answers. It is a controlled
behavior diagnostic, not an unseen-fact generalization test or promotion
score.

The 344 questions form 172 groups:

| Axis | Groups | Required relationship |
| --- | ---: | --- |
| Equivalent wording | 48 pairs | Structured answer and tool calls remain invariant. Include both filter outcomes, both claim-support outcomes, and module requests. |
| Task scope | 48 pairs | Identical facts under direct/workflow requests require stop/continue respectively; other structured semantics remain invariant. |
| Evidence and mention order | 32 pairs | Sixteen pairs move the relevant evidence entry without changing the answer; sixteen swap mention order and must swap the first/second answer fields. |
| ID notation and case | 32 pairs | Preserve normalized identities and equality while correctly echoing the actual literal inputs. |
| Survivor position | 12 coverage groups | Compare genuine first/nonfirst survivors under both scopes, preserving canonical order and actual membership. These use different facts and are not same-fact counterfactual pairs. |

Survivor coverage uses all six training-owned nonfirst-survivor symbols.
Matching candidate counts is possible only for part of this small source
pool; unmatched fields are explicit in the group metadata. The S1.7 probe
also constructs homogeneous-scPEN positive filters from existing training
lists by querying their actual family, deriving gold through the native
builder and recording the query override. This checks a missing combination
without changing the v8 training corpus.

`scripts/evaluate_world_model_s1_robustness.py` reuses native structured
scoring and reports accuracy by axis/stratum, worst strata, pair both-correct
rates, pair consistency, and stable-wrong pairs separately. A constant wrong
answer cannot count as both-correct. Scope consistency requires the expected
decision change; evidence references must be valid before nonce spelling is
normalized. Coverage groups have accuracy/all-correct results, without an
invariance-consistency rate. Summary prose is not scored token for token.

The native generation bundle includes questions only. Public loading does
not open diagnostic gold rows or answer keys. The v6 and future v8 diagnostic
configs use the same panel and generation settings, explicit
`retain_current_turn` reasoning history, 32 evaluation nodes, separate
diagnostic output paths, and no promotion gate or training-run W&B writer:

- `config/world_model_s1_120b_32n_robustness_v6_32n.json`
- `config/world_model_s1_120b_32n_robustness_v8_32n.json`

The existing `world_model_s1_eval.slurm` launcher recognizes these isolated
diagnostics and invokes the robustness scorer with pinned run/model/tokenizer
identities. Its v8 completed-checkpoint validation must run after v8 training;
CPU config tests cannot replace that receipt. Artifact, tokenizer, oracle,
regression, frozen-input, and launch-readiness results are collected under
`diagnostics/s1_v8_preparation/`. Scripted gold predictions verify the scorer
only and must never be reported as model performance.

The consolidated preparation receipt is
`diagnostics/s1_v8_preparation/readiness.json`. All 344 diagnostic traces pass
semantic validation and actual-tokenizer supervision checks; the longest
trace is 1,761 tokens against the 4,096-token training limit. The real v6
diagnostic launcher passes CPU contract preflight. Native generation still
requires the model identity receipt produced by node-local weight staging;
distributed GPU execution remains a launch-time check.

## V8 training submission (2026-10-21)

The user authorized training on the already built v8 corpus. Slurm accepted
job **5726246**, `s1_v8_32n`, at **16:50:11 ET on 2026-10-21**. It requests
32 nodes with eight GPUs per node on `extended`, with a 16-hour limit. Its
initial state was pending for priority. Submission is not completed training
or evidence of a model gate pass.

The run uses `config/world_model_s1_120b_training_v8_32n.json` unchanged.
It starts from S0 `slurm-5631883` and retains global batch 32, one epoch,
1,728 optimizer steps, learning rate 1e-4, r32 LoRA, uniform completion loss,
and the 4,096-token context limit. The corpus has 55,276 training rows,
including the unchanged S0 replay allocation. No evaluation job was
submitted in this step.

Before submission, the validator selftest caught all 53 known-bad fixtures
with no false positives. Aggregate training validation found zero violations.
The actual launcher passed CPU contract preflight. All 114 checked file
identities matched preparation, and all 22 held-out corpus and evaluator
files remained byte-identical to v6. The completed tokenizer audit remains
applicable because its corpus, tokenizer, overlay, and serializer hashes
match. Its training and validation maxima are 4,034 and 1,640 tokens.
Node-local full model verification and distributed GPU preflight run when
the allocation starts.

The checkpoint output will be:

`data/world_model_v4/checkpoints/s1_v8_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5726246/`.

Launch evidence, the exact submission command, config, input verification,
and copies of the submitted training code are under
`data/world_model_v4/launch_reports/s1_v8_120b_32n/slurm-5726246/`.
Keep the submitted training code unchanged while the job is queued or
running. Freeze the v8 corpus, overlay, manifests, and run config.

```bash
squeue -j 5726246
tail -F logs/s1_v8_32n-5726246.out logs/s1_v8_32n-5726246.err
```

The no-tool format was also checked across the training aggregate: all
27,000 current turns with zero tool calls contain exactly one assistant
action, with nonempty `thinking` and final `content`, and no `tool_call`
field. Tool definitions may remain in the prompt. The serializer emits
one reasoning block followed by the final JSON and supervises both. Earlier
conversation turns, when present, are context rather than extra supervised
thinking steps for the current answer.

## V8 post-training gates queued (2026-10-21)

The user authorized the S1 evaluation and S0 forgetting gate after v8
training. Slurm accepted both at **16:57:28 ET on 2026-10-21**:

| Evaluation | Job | Dependency | Frozen questions |
| --- | --- | --- | ---: |
| S1 promotion gate | `5726300` (`s1_v8_eval`) | `afterok:5726246` | 6,875 |
| S0 forgetting gate | `5726301` (`s1_v8_s0_retention`) | `afterok:5726246,afterany:5726300` | 120,766 |

Both request 512 nodes on `batch`, eight GPUs per node, one task per node,
and a four-hour limit. They target the final v8 training checkpoint at
`data/world_model_v4/checkpoints/s1_v8_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5726246/`.
Forgetting waits for the S1 job to end even if evaluation fails, so both can
append their separate metrics to the originating W&B run without concurrent
writers. Both still require successful training.

The S1 config is
`config/world_model_s1_120b_32n_test_v8_retained_512n.json`. It preserves the
retained-context v6 generation settings, `tiered_v3` criteria, tokenizer,
question bytes, and answer keys. Its changes bind the v8 training config,
corpus and evaluator manifests, checkpoint root, and evaluation output.
It explicitly sets `history_policy` to `retain_current_turn`.

The forgetting config is
`config/world_model_s1_120b_s0_forgetting_test_v8_512n.json`. Only its v8
evaluation identity, training-corpus identity, and output root differ from
v6. It preserves the full frozen S0 panel, generation settings, and gate
thresholds. The independent S0 runner already retains current-turn reasoning.

Static checks verified the exact corpus, panel, config, and tokenizer
bindings. Setup and parent/forgetting tests passed **67 tests**, with one
skipped. V8 setup tests explicitly use temporary checkpoint fixtures or a
future-checkpoint stub; they do not establish a completed v8 checkpoint.
Both unmodified production launchers validate the real completed checkpoint
and exposure receipt after their dependencies release, before generation.
The submitted training config and all recorded training code hashes remain
unchanged. Slurm confirmed both evaluations pending for dependency.

Submission commands, config copies, static checks, test output, and scheduler
receipts are in
`data/world_model_v4/launch_reports/s1_v8_120b_32n/slurm-5726246/`:
`evaluation_submissions.json`, `evaluation_static_audit.json`, and
`evaluation_launch_verification.json` summarize the evidence. Preserve the
two configs and evaluation code while the jobs are queued or running.

Outputs will be under `data/world_model_v4/eval_runs/`:

- `s1_v8_120b_32n_test_retained/oss120b-plain-base-tokenizer-lora-r32/slurm-5726300/`
- `s1_v8_120b_s0_forgetting/oss120b-plain-base-tokenizer-lora-r32/slurm-5726301/`

```bash
squeue -j 5726246,5726300,5726301
```

## Sources of truth

Use these sources, in this order:

1. `runtime/world_model/contract.yaml` plus the selected versioned S1 overlay:
   the executable wire format, grammars, tool rules, allocations, and reviewed
   reasoning configuration. The shared contract stays byte-identical.
2. Published component and aggregate manifests/rows for the selected corpus:
   exact question content, split counts, reasoning, and input hashes.
3. [Question classes, gold traces, and system prompt](S1-question-classes.md):
   all 66 current types, actual corpus counts and outcomes, complete worked
   traces with the reviewed v6 reasoning, and the approved v7 prompt.
4. `runtime/world_model/golden_traces.jsonl`: 73 shared wire examples
   (7 S0 and 66 S1), with v5 answer semantics.
5. This document: design rationale and dated review history.

The 2026-10-20 documentation correction restores 114 complete gold traces
covering all 66 types alongside the question-classes tables and prompt.
The renderer selects published training rows for each type and distinct call-count,
status, decision, and blocked-reason combination. It preserves every
non-system message, including earlier turns and retry history, and records
the source row ID and line. This keeps the current reviewed reasoning
visible without changing the frozen corpus or older wire fixtures.
All 114 rendered traces match their v6 and v7 source rows in full, and
both checkouts pass the generated-document and golden-fixture checks.

The validation kit remains in `s1_kit/`, but it reads the single runtime
contract rather than maintaining a second contract copy.

   Run these commands from the repo root, with the `pytorch-rocm`
   environment:

   ```bash
   K=agents/world_model_v3_curriculum/s1_kit
   PYTHON=python $K/selftest.sh       # run this first. It must print PASS.
   PYTHON=python $K/mutate.sh         # after any change to validate.py
   python -m runtime.world_model.contract check
   python scripts/render_world_model_s1_catalog.py --check
   python $K/validate.py --overlay runtime/world_model/s1_v5.yaml runtime/world_model/golden_traces.jsonl
   python $K/validate.py --overlay <selected-overlay> <corpus>/train.jsonl
   ```

   Exit codes: 0 clean, 1 violations found, 2 or more could not run. Run
   `selftest.sh` after any change to the contract or validator.

If prose and the runtime contract disagree, the runtime contract wins and the
prose must be corrected in the same change.

## Historical status (2026-10-08)

- **S1 v2 is the current corpus** (`data/world_model_v4/corpora/s1/`, built
  2026-10-08). The overlay `runtime/world_model/s1_v2.yaml` changes only two
  things from v1. The shared contract stays unchanged.
  - S1.2 has more genes without a symbol (see "S1.2 Ensembl to symbol").
  - The S0 replay comes from the S0 v4 corpus (see "S0 rebuild").
- S1 v2 has the same row counts as v1: 55,276 train rows (55,000 S1 rows and
  276 S0 replay rows), 6,875 validation rows, and 6,875 test rows. Its
  evaluator panels are in `data/world_model_v4/eval/s1/`. S1.5 is disabled.
- The S1 v1 corpus stays in `data/world_model_v3/corpora/s1/` for the 20B
  qualification and smoke runs of 2026-10-06.
- **Selected S0 parent:** the r32 uniform arm (`slurm-5631883`). It passed
  the S0 gate (see `S0-human-gene-identifiers.md`).
- **Training configs.** Both start from the S0 parent and use the uniform
  loss. The v2 training schema accepts only these two run profiles.
  - `config/world_model_s1_120b_training_v2_184n.json` (the selected run):
    184 nodes, global batch 184 (1 row per replica, no gradient
    accumulation), 2 epochs (602 steps), learning rate 2e-4, and 5% warmup.
    The larger batch takes about 6 times fewer steps for each epoch, so this
    profile trains 2 epochs at a higher learning rate. Set
    `MODEL_STAGE_WAVE_SIZE=184` so that all nodes copy the base model at the
    same time. The default copies 4 nodes at a time.
  - `config/world_model_s1_120b_training_v2.json` (fallback): 16 nodes, the
    v1 run settings (1 epoch, 1,728 steps, learning rate 1e-4).
- **Jobs (2026-10-08).** The 184-node run is job 5640408 on `batch`. The
  16-node fallback is job 5638673 on `extended`. Cancel the fallback when the
  184-node run finishes.
  - The first 184-node run (job 5638718) stopped after the first step. The
    launcher exports a snapshot receipt inbox for every run, and the trainer
    polled it although snapshots were off. The trainer now ignores the inbox
    when `snapshot_steps` is 0. A 20B S1 smoke run (job 5640381) completed
    all 124 steps with the fix.
- Open items:
  - Change the S1 gate to the two tiers (see "Stage gate").
  - The semantic validator reports the same cross-stage `evidence_id` repeats
    in v1 and v2 (for example, S1.1 and S1.3 rows share IDs). Each stage
    builder starts its own ID sequence. The 276 replay rows also carry the S0
    v4 metadata additions, which the validator reports as extra fields.

Rebuild S1 v2:

```bash
python -m runtime.world_model.s1_2_corpus --overlay runtime/world_model/s1_v2.yaml
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v2.yaml
```

The other seven stages are read unchanged from the v1 stage directories.

S1 v6 (finalized and submitted 2026-10-19):

All originally failing question types have reviewed changes. Passing types
retain their frozen content. The dated review entries below record the
v4/v5 results, proposed edits, implementation, and verification.

- **Training submitted.** Job `5712804` (`s1_v6_32n`) requests 32 nodes
  on `extended`, with a 16-hour limit. Slurm accepted it at 17:55:52 ET
  on 2026-10-19; its initial state was pending for priority. The run uses
  `config/world_model_s1_120b_training_v6_32n.json`. Preserve the v5
  recipe: global batch 32, one epoch, 1,728 optimizer steps, learning
  rate 1e-4, r32 LoRA, and S0 uniform parent `slurm-5631883`.
  The checkpoint directory will be
  `data/world_model_v4/checkpoints/s1_v6_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5712804`.
  Monitor from the repository root:

  ```bash
  squeue -j 5712804
  tail -F logs/s1_v6_32n-5712804.out logs/s1_v6_32n-5712804.err
  ```

- **Final build.** Rebuilt all eight components with the final v6 overlay,
  then assembled the aggregate and refreshed all three retry manifests.
  All 24 reviewed component row files reproduced byte for byte. The
  aggregate contains 55,000 S1 training rows and 276 S0 replay rows,
  with 6,875 rows in each held-out split. Aggregate validation/test rows
  and all four evaluator question/key files are byte-identical to v5.
  The test panel stays `eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904`.
  Freeze the submitted v6 corpus, overlay, and run configs. Reproduce
  trained versions in separate directories; do not repin their manifests
  when adding later results to this document.
- **Launch validation.** Aggregate semantic validation checks all 55,276
  rows with zero violations. The actual tokenizer verifies every training
  and validation row at 4,096 tokens: maxima are 4,025 and 1,640. All
  100,627 assistant thoughts, 38,476 calls, and 62,151 final answers are
  intact and supervised. Tool responses and prior turns remain masked.
  The selftest detects all 53 bad fixtures without false positives;
  check-doc confirms 73 traces. Training validation passes 111 tests.
  Evaluation integration passes 76 tests with one legacy skip. The real
  training launcher contract-only preflight passes. An independent audit
  checks 245 preexisting artifacts: 233 are unchanged, and only eleven
  v6 manifests and two overlay comment lines changed.
- **Frozen identities.** Training SHA-256:
  `4cd93d601e42f5e158fb76be734db44e2f4ada77c16b7cb0cd7ff7d3f9590d5f`.
  Corpus manifest SHA-256:
  `efe70ce238cc4db14aecf7e2801ead6c3947de5d3a2f6b6911d726dda9917d9d`.
  Evaluator manifest SHA-256:
  `c6eeea2bd9dc269e6d723f1423afba8992896b874fdada42cd0bb32c0673f0a7`.
  Training config SHA-256:
  `34dae5bf60775890321f7e9198f1a9cc5b55b9f59e3008722740dbc42113b638`.
  The final overlay hash is
  `fa5bc283b194cc628b305be69b9e3ff1619321d467b2cf626e66caf76f55dbe7`;
  only its stale introductory comments changed during final assembly.
- **Evaluation queued (2026-10-19).** The frozen S1 gate config is
  `config/world_model_s1_120b_32n_test_v6_512n.json`. The S0 retention
  config is `config/world_model_s1_120b_s0_forgetting_test_v6_512n.json`;
  it preserves the frozen 120,766-row panel and selects the r32 method.
  Both request 512 nodes. Separate retry configs use
  `config/world_model_s1_120b_32n_{lookup_retries,id_lookup_retries,membership_retries}_v6_32n.json`
  and request 32 nodes each. They retain 150/150/111 diagnostic rows and
  2/2/3 call limits. Their reports do not enter the promotion gate or
  resume the training W&B run. Answer-free bundle loading and scripted
  scoring pass for all 411 diagnostic rows. Training/evaluation launchers
  now register v6; diagnostic generation and scoring use an isolated path.
  S1 gate job `5712872` requires `afterok:5712804`. S0 retention job
  `5712873` requires both `afterok:5712804` and `afterany:5712872`;
  it runs after the S1 evaluation ends, even if that gate fails, while
  requiring successful training. This serializes the two W&B writers.
  Symbol, ID, and membership retry jobs are `5712874`, `5712875`, and
  `5712876`; each requires `afterok:5712804` and can run alongside the
  main evaluation. Gate and retention use `batch`; retries use `extended`.
  All five jobs have four-hour limits and one task per node, with eight
  GPUs and 56 CPUs per task. Retention preserves staging wave 256 and
  two generation attempts from the successful v5 run. Every job targets
  the final `slurm-5712804` checkpoint. Slurm verified all five as pending
  for their dependencies. Actual checkpoint preflight runs inside each
  job after training; model accuracy remains unmeasured.
- **Source limitations.** The pinned source index passes its integrity
  check and retains all 353 historical layers. A broader core suite has
  39 passes, one skip, and one failure because the live HPC flist now has
  213 entries instead of 353. The historical raw dendrogram is also
  unavailable. Neither is an input to this snapshot-based corpus rebuild.
  Two older S0 trajectory tests requiring absent legacy corpora were
  excluded. Preserve all historical source pins and record these external
  data limitations separately from the passing launch checks.
- **Evidence.** Build logs, token counts by trace, frozen-artifact audit,
  config identities, test logs, submission command, and Slurm receipt are
  archived under
  `data/world_model_v4/launch_reports/s1_v6_120b_32n/slurm-5712804/`.
  These establish launch readiness, not a model gate pass.

- **S1.1.1 results.** On the same 150 test questions and current scorer,
  v4 gets 150/150 (100%) and v5 gets 148/150 (98.67%). Both gene results
  and lookup arguments remain correct on every row. C7 and TOPBP1 choose
  `continue` although the direct requests require `stop`. This small
  discrepancy does not establish that wrapper changes caused the errors.
- **Approved edit.** Restore v4's 64 direct wrappers at the existing
  S1.1.1 template indices. Exactly 168 of 1,200 training user messages
  change; the other 1,032 are identical. Preserve all genes, calls,
  responses, thinking, targets, metadata, and source ordering. Every
  S1.1.1 target remains a resolved lookup followed by `stop`.
- **S1.1.2 results.** V4 gets 197/200 (98.5%) and v5 gets 198/200 (99%)
  on the same test questions, scorer, and keys. V4's three errors choose
  `stop` instead of `continue`. V5's MTND3P9 and OLA1P3 lookups resolve,
  then invent multiplex M1, call membership, and return `not_found/defer`.
  Both use the held-out "next workflow action" wording. These results do
  not establish which training change caused either failure mode.
- **S1.1.2 approved edit.** Restore v4's 64 step wrappers at their existing
  indices in all 1,600 training rows. Replace only final thinking with:
  "The registry resolves {symbol} to {gene_id}, completing the requested
  lookup. Return this mapping with continue so the workflow can use it."
  Bind both slots to the resolved tool result. Keep pre-call thinking,
  tool calls and responses, final answers, source order, and metadata.
  Each target stays one symbol lookup, `resolved/continue`, registry
  provenance, null graph provenance, and no blocked reason.
- **S1.1.3 results.** V4 gets 147/150 (98%) and v5 gets 148/150 (98.67%)
  on the same questions, scorer, and keys. V5's remaining errors are an
  inconsistent `stop` with a blocked reason and a trailing-period copy
  error. The latter question joins the symbol's period to wrapper
  punctuation. Its frozen wording remains unchanged.
- **S1.1.3 approved edit.** Keep v5's corpus wording and construction,
  since accuracy improved and exceeded 98%. Do not restore v4 wrappers
  for this form. Replace only final thinking in its 1,200 training rows:
  "The registry reports not_found for {symbol}; this release provides no
  mapping. Preserve the queried symbol and registry evidence, then defer
  with a blocked reason." Bind the symbol to the returned lookup value.
  Preserve pre-call thinking, calls, responses, final answers, metadata,
  ordering, and both held-out splits. Direct and step counts stay 600 each.
- **Version boundary.** `runtime/world_model/s1_v6.yaml` uses overlay
  schema v5. Its S1.1 `train_prompt_banks` mapping pins forms 1, 2, 9, 10, and 11 to
  the frozen v4 bank. Form 3 keeps the default v5 bank.
  `train_final_reasoning` carries forms 2 and 3's reviewed templates in
  the pinned overlay. S1.1.8 adds the pinned retry policy described below.
  `train_membership_reasoning` carries S1.1.9's post-lookup thought and
  framing-specific final thoughts, all frozen in the pinned overlay.
  `train_absent_membership_reasoning` does the same for S1.1.10, retaining
  its v5 `resolved/present:false` targets.
  `train_unknown_multiplex_reasoning` supplies S1.1.11's post-lookup thought
  and a shared final rule for both framings: `not_found/present:null/defer`.
  Its final templates have no interpolation slots.
  S1.2's `train_prompt_banks` mapping pins forms 1, 3, 4, 6, 8, and 9 to v4.
  Its `train_final_reasoning` mapping carries form 1's reviewed
  completion template, with slots for the returned gene ID and symbol,
  form 5's deferral template, binding only the queried gene ID,
  and form 6's invalid-input template, binding only the returned input.
  `train_version_suffix_reasoning` carries form 3's direct and workflow
  completion templates, with the same two returned-value slots.
  `train_no_symbol_reasoning` carries form 4's direct and workflow
  templates, binding only the returned gene ID. The symbol stays null.
  `train_near_miss_reasoning` carries form 8's direct and workflow
  templates, binding the returned gene ID and symbol. The workflow
  templates for S1.1.2 and S1.2.3/S1.2.4/S1.2.8 explicitly connect
  the user's stated need for the lookup result to the continue decision.
  `train_symbol_input_reasoning` carries form 9's direct and workflow
  templates, binding the returned symbol and ID. They exclude an
  unrequested membership check and tie the decision to user scope.
  S1.2.11's train_lookup_retries pins a separate ID policy with 800
  unchanged inputs and 400/200/200 terminal/recovery/exhaustion cases.
  S1.3's train_prompt_banks pins forms 1, 4, 5, 6, 7, and 10 to the frozen v4
  identity_s1_3 bank. Its train_final_reasoning binds forms 1 and 4's canonical
  symbol and candidate count; train_single_membership_reasoning carries
  form 5's intermediate and framing-specific final thoughts. Forms 6 and 7
  use train_multiple_membership_reasoning and train_no_membership_reasoning
  for their intermediate and shared deferral conclusions. Form 8 uses
  train_user_choice_reasoning for its final training thought, binding the
  selected ID and earlier symbol with separate direct and workflow endings.
  Its v5 prompts and answers stay fixed. Form 10 uses train_membership_retries
  for 300 terminal failures, 150 recoveries, and 150 exhausted retries while
  retaining the successful registry lookup. Other forms retain v5 content. Template validators enforce the required evidence
  slots and formatting. Training reasoning has no hard sentence or word
  limits; use the length needed for the evidence and decision, subject to
  semantic checks and the actual tokenizer/context budget.
  The S1.1, S1.2, and S1.3 components are built now, under
  `data/world_model_v4/corpora/s1_v6/s1_1/`, `s1_2/`, and `s1_3/`. All eight component paths
  point to new v6 locations to protect frozen versions during later work.
- **Failure pairs.** S1.1.8 retains its original v5 prompts and source
  pairs. The audit checks each pair member against its configured wrapper
  and preserves the same input, framing, and template index. Only declared
  training overrides may make paired wrapper text differ. Legacy builds
  retain their exact-prompt requirement.

Rebuild these approved components from the pinned source index:

```bash
python -m runtime.world_model.s1_1_corpus --overlay runtime/world_model/s1_v6.yaml
python -m runtime.world_model.s1_2_corpus --overlay runtime/world_model/s1_v6.yaml
python -m runtime.world_model.s1_3_corpus --overlay runtime/world_model/s1_v6.yaml
```

The focused regression suite compares every training row against v5,
substituting S1.1 forms 1, 2, 9, 10, and 11's v4 user text and the reviewed thinking.
It also checks form 8's bounded retry trajectories and preserved inputs.
S1.2 changes only forms 1, 3, 4, 6, and 8's configured wrappers and final thinking.
Form 5 changes only final thinking, preserving its v5 wrappers and targets.
It checks held-out bytes, repeated builds, pinned form scope, and
reproduction of no-overlay, v4, and v5 splits.
No v6 training or evaluation jobs have been submitted.

- **First increment (S1.1.1 only).** Semantic validation passed all three
  splits: 10,000 training, 1,250 validation, and 1,250 test rows. Exactly
  168 user messages change. The other 1,032 form-1 rows and all 8,800
  other training rows are byte-identical to v5. Both held-out files are
  byte-identical to v5. Each of the 64 restored direct wrappers appears
  18 or 19 times. All 42 snapshotted historical files retain their hashes.
- The first-increment component train SHA-256 was
  `3f43e22ef3376b8d09f33d4a8726bc40fc3f8d7339857a495f9a9d724391a443`.
  This is an incremental component, not a finalized aggregate corpus.
- **First-increment checks.** The focused corpus, shared-contract, overlay, and prompt-bank
  suites pass 79 tests. Three additional checks pass for unchanged family
  policies and rejection of corrupted pair prompts. Validator selftest
  passes all 53 known-bad fixtures with zero false positives.
- The separate full-source contract check cannot finish: its pinned
  `full_brain_all_genes_dendrogram.txt` is not mounted. Keep that historical
  source pin unchanged. The S1.1 component builds from its pinned source
  index and passes semantic validation for every split.
- **Second increment (S1.1.2).** All 1,600 form-2 user messages and final
  thinking messages change as reviewed. Every other field is unchanged.
  The other 8,400 training rows, including all 1,200 S1.1.1 rows from the
  first increment, are byte-identical to that increment. Both complete
  held-out files remain byte-identical. All 64 restored step wrappers
  occur exactly 25 times. Semantic validation passes all three splits,
  and all 42 snapshotted historical files retain their hashes.
- The second-increment component train SHA-256 was
  `828202562bba15fce849a0fbda71edd2f7610e6af3e531df81d5ed7f652896ac`.
  The aggregate corpus remains unbuilt until the later question reviews.
- **Second-increment checks.** All 93 focused tests pass, including
  repeated-build equality, legacy split reproduction, unchanged targets,
  and rejection of unreviewed forms or invalid reasoning templates.
  Validator selftest again passes all 53 known-bad fixtures with zero
  false positives.
- **Third increment (S1.1.3).** Exactly 1,200 final-thinking messages change.
  No user messages change. All other fields, the other 8,800 training
  rows, and both full held-out files are byte-identical to the second
  increment. This preserves all 2,800 S1.1.1 and S1.1.2 rows and fixes.
  Semantic validation passes all three splits. All 47 snapshotted
  historical files, including the v5 evaluator files, retain their hashes.
- The third-increment component train SHA-256 was
  `6ea3230b3c0412dbea7c4449cc0ce03d1c4fba0a77153eb25977e146bca1936f`.
- **Third-increment checks.** All 96 focused tests pass, including v5
  prompt preservation, exact thinking-only changes for form 3, repeated
  builds, and legacy split reproduction. Validator selftest passes all
  53 known-bad fixtures with zero false positives.

- **S1.1.4 through S1.1.7.** Retain v5 wording, thinking, tool evidence,
  and targets. Current-key structured results are, respectively: v4
  150/150, 99/100, 97/100, and 100/100; v5 150/150, 100/100, 99/100,
  and 100/100. These forms provide no evidence for a corpus rewrite.
- **S1.1.8 results.** V4 gets 47/50 (94%) and v5 gets 41/50 (82%)
  against the frozen one-attempt keys. Eight v5 failures retry the same
  lookup and obtain the correct mapping. SPSB4 still incorrectly chooses
  `continue`, so recovery alone is not full task correctness. The ninth
  failure returns an inconsistent continuation envelope. The old harness
  fails the first call only, making a second call succeed by construction.
  These traces support teaching a bounded retry, but do not demonstrate
  that v5 stops correctly during a persistent outage.
- **S1.1.8 approved behavior.** Teach one permitted retry now. It repeats
  an existing read-only action, rather than introducing another tool or
  planning task. Preserve the 400 training identities, original wrapper
  assignments, user prompts, and first lookup arguments. Assign outcomes
  by identity hashes, independently of wording and wrapper counts:
  200 terminal failures without a retry allowance; 100 first failures
  followed by successful recovery; and 100 first failures followed by a
  failed retry. Keep the ordinary failure payload in the terminal cases.
- **Visible allowance.** The existing error `message` carries
  `registry unreachable; retries_remaining=1` when a retry is permitted.
  A failed retry returns `registry unreachable; retries_remaining=0`.
  No tool-result fields or shared contract bytes change. A plain legacy
  error supplies no retry allowance and still requires deferral. This is
  an explicit runtime policy, not a claim that every outage is transient.
  The normal runtime default remains the legacy policy; the separate
  diagnostic enables the new allowances. A deployment adopting this
  policy must supply and enforce the same budget.
- **Targets and thinking.** Retry the identical tool and arguments once.
  A recovery returns exactly the resolved mapping, the successful call's
  registry evidence, `resolved/stop`, and no blocked reason. Terminal and
  exhausted cases return `error/defer`, the original queried symbol,
  `failed_tool`, `error_code`, and null provenance. Their summaries no
  longer instruct another retry. Frozen reasoning and summary templates
  live in `runtime/world_model/s1_v6_lookup_retries.json`, pinned by the
  v6 overlay and component manifest. Recovery evidence is allocated after
  baseline generation so other rows and held-out evidence IDs stay fixed.
- **Validation and diagnostics.** The semantic validator permits recovery
  only for one explicitly allowed retry of an identical symbol lookup,
  with a final mapping and provenance copied from the successful result.
  Other failure checks remain in force. The frozen promotion panel,
  questions, answer keys, and both corpus held-out files stay unchanged.
  A separate diagnostic uses the same 50 held-out S1.1.8 genes in all
  three fault schedules: 150 rows with a distinct panel ID. Its fault
  schedule stays in record IDs that the model does not see. Report these
  cases separately from the old policy score; never recompute historical
  accuracy using the new targets. No learned retry accuracy is available
  until a model runs this diagnostic.

Rebuild the additional diagnostic:

```bash
python scripts/build_world_model_s1_lookup_retry_panel.py
```

It writes `data/world_model_v4/eval/s1_v6/lookup_retries/`. The native
trajectory runner and structured scorer support its cases. The manifest
uses a diagnostic schema, rather than the promotion-panel schema. Wire
its job configuration when the final v6 aggregate is assembled. Keep the
then-current increment limited to the S1.1 component; no jobs are submitted.

- **Fourth-increment verification (S1.1.8).** Exactly 400 training rows
  change relative to the third increment. The other 9,600 rows remain
  byte-identical, preserving the earlier S1.1.1 through S1.1.3 fixes.
  Every form-8 user prompt, first assistant call, source identity, and
  tool inventory is unchanged. The complete validation and test files
  remain byte-identical to v5. All 52 snapshotted historical files retain
  their hashes, including frozen evaluator questions and answer keys.
  Semantic validation passes all 10,000 training, 1,250 validation,
  and 1,250 test rows. No-allowance, recovery, and exhaustion cases use
  63, 49, and 51 distinct existing wrappers, respectively.
- The fourth-increment S1.1 component train SHA-256 was
  `b9b1d846c9890cdca84f92b3a91133b96f1ffffc86f81661287f3eb371ab4eef`.
  The separate diagnostic panel ID is
  `e74b2b7e72651bae168b086f316d7491d68aeaa152a51bafa04038088a379f96`.
- **Fourth-increment checks.** The corpus, overlay, prompt-bank,
  contract, native-generation, structured-scoring, and retry suites pass
  162 tests, with one skip and one known source check deselected. They
  verify repeated builds, reproduction of legacy splits, successful
  recovery, exhaustion, premature deferral, changed retry arguments,
  rejection of a third call, and rejection of false recovery by the
  semantic validator. Validator selftest passes all 53 known-bad
  fixtures with zero false positives. These are scripted harness checks,
  not a measurement of learned model accuracy.
- The broader live-source check fails independently: the current flist
  has 213 entries, while its legacy grammar test expects 353. That one
  test was excluded from the final regression rerun. The earlier missing
  pinned dendrogram source also remains unresolved. The component uses
  the verified pinned source index; keep the historical contract and
  source pins unchanged. Resolve live-source preflight separately before
  a final submission. No v6 training or model evaluation has run.

- **S1.1.9 results.** Fresh rescoring on the same 125 test questions,
  current scorer, and v5 answer keys gives v4 119/125 (95.2%) and v5
  65/125 (52%) structured accuracy. Correct tools and arguments fall
  from 120/125 to 116/125. Valid trajectories fall from 122/125 to 79/125.
- **S1.1.9 failure interpretation.** Forty-five v5 outputs contain correct
  six-field JSON, both actual evidence channels, and the right decision,
  but omit required final thinking. The parser rejects them. Nine other
  rows omit the membership call, five choose `stop` instead of `continue`,
  and one produces malformed JSON with tool lists instead of `present`.
  V4 has one otherwise correct raw answer rejected for missing thinking,
  and five failures that omit membership. Inspecting rejected raw JSON
  gives 120/125 v4 and 110/125 v5 correct structured contents. That is
  a diagnostic; the official scores remain 119/125 and 65/125. Most of
  the strict decline concerns response format, though task completion
  and continuation also worsen.
- **S1.1.9 approved wording.** Restore the 64 direct and 64 workflow v4
  wrappers at their original row assignments. Exactly 571 of the 1,000
  training user messages change: 71 direct and all 500 workflow requests.
  These user messages are the only v4/v5 differences in this form. The
  comparison supports a conservative restoration, not a causal conclusion.
- **S1.1.9 approved reasoning.** Replace thinking after the symbol lookup
  in every training example: "The lookup supplies {gene_id} for {symbol},
  and the membership request remains unanswered. Check this ID in the
  requested multiplex." Bind both fields to the successful lookup output.
  Replace final thinking in all 1,000 rows, starting with: "The membership
  result lists {gene_id} as present, so return resolved with present:true
  and both evidence channels." Direct framing adds: "The request is
  complete, so stop." Workflow framing adds: "The next workflow step needs
  this membership result, so continue." All templates fit the existing
  two-sentence, 25-words-per-sentence, 40-words-total limits.
- **S1.1.9 preserved targets.** Keep 1,000 rows, 500 per framing, with the
  same identities, IDs, tool arguments and responses, final answers,
  metadata, and first-call thinking. Every target remains a symbol lookup
  followed by one membership check, `resolved`, exactly `gene_id`,
  `gene_symbol`, `multiplex_id`, and `present:true` in the result, both
  provenance channels, and no blocked reason. The final decision follows
  framing. Preserve both entire held-out files, other question types,
  and the S1.1.8 retry corpus and diagnostic questions and keys. Refresh
  only the retry diagnostic's manifest for the updated v6 overlay pin.
- **Fifth-increment verification (S1.1.9).** Exactly 1,000 training rows
  change relative to the fourth increment, with 571 user-message changes
  and 2,000 thinking-message changes. All other content in these rows is
  unchanged. The other 9,000 training rows, including every S1.1.8 retry
  example, remain byte-identical. Both complete held-out files remain
  byte-identical to v5. All 55 snapshotted files retain their hashes,
  including the shared contract, historical corpora and evaluator files,
  and the retry policy, diagnostic questions, and diagnostic answer key.
  Each framing uses all 64 restored wrappers, seven or eight times each.
  Semantic validation passes all 10,000 training, 1,250 validation, and
  1,250 test rows.
- The fifth-increment component train SHA-256 was
  `71695c3db490e41ffb4bce07466599e6db3cc686518563409801c60696ef57b6`.
- **Fifth-increment checks.** All 117 focused corpus, overlay,
  prompt-bank, and retry tests pass. These include deterministic repeated
  builds and reproduction of no-overlay, v4, and v5 split bytes. Validator
  selftest passes all 53 known-bad fixtures with zero false positives.
  The actual uniform-SFT tokenizer checks all 1,000 revised rows and
  confirms supervision of both revised thoughts in each row. Lengths
  range from 894 to 961 tokens, below the 4,096-token limit. This CPU
  check uses `USE_TORCH=0 USE_TF=0` to avoid importing unavailable ROCm
  libraries on the login node. No training recipe, frozen scorer policy,
  model predictions, or model scores change. No jobs are submitted.

- **S1.1.10 results.** Fresh rescoring on the same 88 held-out questions,
  current scorer, and v5 answer keys gives v4 87/88 (98.86%) and v5
  42/88 (47.73%) structured accuracy. Correct tools and arguments fall
  from 87/88 to 78/88; valid trajectories fall from 88/88 to 73/88;
  correct continuation falls from 88/88 to 53/88. The old v4 training
  `not_found` target is not restored: both models are compared with the
  current `resolved/present:false` key.
- **S1.1.10 failures.** Ten v5 rows fail to execute membership: eight
  finish after symbol lookup, and two format the membership call in the
  wrong native channel. Fourteen other rows have the correct result but
  choose `stop` or `defer` instead of `continue`, seven each. Nine invent
  result fields: three substitute `is_member`, three add tool lists,
  and three place evidence inside `result`. Thirteen responses are
  rejected after membership: five omit final thinking, four attach a
  blocked reason to a non-defer answer, two invent a status, and two
  attempt unnecessary extra calls. Only four rejected raw answers are
  otherwise correct; including them gives a raw-content diagnostic of
  46/88 (52.27%), not a replacement for the official score. V4's only
  error finishes after lookup without checking membership.
- **S1.1.10 approved wording.** Restore the 64 direct and 64 workflow v4
  wrappers at the existing assignments. Exactly 397 user messages change:
  47 direct and all 350 workflow examples. Keep the v5 answer semantics.
  V4/v5 also differ in all 700 final statuses and final thoughts, so this
  restoration copies user wording alone.
- **S1.1.10 approved reasoning.** Replace post-lookup thinking in all 700
  training rows: "The lookup supplies {gene_id} for {symbol}, and the
  membership request remains unanswered. Check this ID in the requested
  multiplex." Replace final thinking, beginning: "The membership result
  lists {gene_id} as absent, so return resolved with present:false, both
  evidence channels, and no blocked reason." Direct requests add: "The
  request is complete, so stop." Workflow requests add: "The next workflow
  step needs this membership result, so continue." The fields bind to
  the lookup result and confirmed membership absence. A negative result
  answers the question; the workflow can use `false` without requiring
  the gene to be a multiplex member. The templates stay within the
  existing two-sentence, 25-word-per-sentence, 40-word-total limits.
- **S1.1.10 preserved scope.** Keep 700 examples, 350 per framing, with
  the same genes, IDs, tool calls and responses, final answers, metadata,
  and first-call thinking. Each target stays two calls, `resolved`,
  exactly `gene_id`, `gene_symbol`, `multiplex_id`, and `present:false`
  in the result, both provenance channels, and no blocked reason.
  Preserve both complete held-out files, every other training form,
  and the retry diagnostic questions and keys. Refresh only that
  diagnostic's manifest for the updated v6 overlay pin.
- **Sixth-increment verification (S1.1.10).** Exactly 700 training rows
  change relative to the fifth increment: 397 user messages, 47 direct
  and 350 workflow, plus both post-tool thoughts in every row, 1,400
  thinking messages total. All other content in these rows is unchanged.
  The other 9,300 training rows, including all S1.1.8 and S1.1.9 changes,
  remain byte-identical. Both complete held-out files remain byte-identical
  to v5, preserving all 88 validation and 88 test examples of this form.
  All 55 snapshotted files retain their hashes, including the shared
  contract, historical corpora and evaluator files, and the retry policy,
  diagnostic questions, and diagnostic answer key. Each framing uses all
  64 restored wrappers, five or six times each. All 700 targets retain
  their two calls, resolved negative result, both evidence channels,
  framing-specific decision, and null blocked reason. Semantic validation
  passes all 10,000 training, 1,250 validation, and 1,250 test rows.
- The sixth-increment component train SHA-256 was
  `fa5390254c723eb506b8fa06dac5cadf7b06c4bd423fece99a2585b509725080`.
- **Sixth-increment checks.** All 127 focused corpus, overlay,
  prompt-bank, and retry tests pass, including deterministic repeated
  builds and reproduction of no-overlay, v4, and v5 split bytes.
  Validator selftest passes all 53 known-bad fixtures with zero false
  positives. The actual uniform-SFT tokenizer checks all 700 revised
  examples and confirms that all 1,400 revised thoughts receive
  supervision. Lengths range from 907 to 972 tokens, below the 4,096-token
  limit. The CPU tokenizer check uses `USE_TORCH=0 USE_TF=0`. Training
  recipes, frozen scorer policy, and saved model predictions are
  unchanged. No training or model evaluation jobs are submitted.

- **S1.1.11 results.** Fresh rescoring on the same 37 held-out questions,
  current scorer, and v5 keys gives v4 0/37 (0%) and v5 5/37 (13.51%)
  structured accuracy. V4 was trained without `present`; all 36 accepted
  v4 finals omit that newly required field. Of those, 28 are otherwise
  fully correct. V4 gets 28/37 (75.68%) against its original frozen key
  with the current scorer. This historical-schema diagnostic is not a
  same-key comparison with v5. Correct tools and arguments fall from
  36/37 to 33/37, status from 33/37 to 19/37, and continuation from
  32/37 to 14/37. The headline increase masks weaker deferral behavior.
- **S1.1.11 failures.** V5 has 14 accepted finals with incorrect results:
  six use `present:false`, and eight substitute `membership` or `member`.
  Another 14 final responses are rejected: ten violate continuation and
  blocked-reason requirements, and four omit final thinking while also
  returning incorrect membership and continuation. Two rows attempt an
  unnecessary third membership call, and two finish after symbol lookup.
  No otherwise-correct answer is recovered by relaxing final thinking.
  The central error confuses an unregistered multiplex with confirmed
  absence from a registered multiplex. The former gives no membership
  answer and requires `present:null`, `not_found`, and `defer`.
- **S1.1.11 approved wording.** Restore the 64 direct and 64 workflow v4
  wrappers at their existing assignments. Exactly 170 user messages
  change: 20 direct and all 150 workflow examples. Keep v5's nullable
  membership target. V4's stronger tool and deferral results support
  this conservative restoration, without establishing a wording cause.
- **S1.1.11 approved reasoning.** Replace post-lookup thinking in all 300
  training rows: "The lookup supplies {gene_id} for {symbol}, and the
  membership request remains unanswered. Check this ID in the requested
  multiplex." Bind both fields to the lookup result. Replace final
  thinking in both framings: "The requested multiplex is unregistered,
  so the gene's membership is unknown; return not_found with present:null.
  Preserve both evidence channels and defer with a blocked reason until
  a registered multiplex is supplied." Both templates stay within two
  sentences, 25 words per sentence, and 40 total.
- **S1.1.11 preserved scope.** Keep 300 examples, 150 per framing, with
  the same genes, query spelling, source order, calls, responses, final
  answers, metadata, and first-call thinking. Each target stays exactly
  symbol lookup, membership check, then `not_found/defer`, with a nonempty
  blocked reason. The result retains exactly `gene_id`, `gene_symbol`,
  `multiplex_id`, and `present:null`. Preserve both registry and graph
  evidence; the unknown-multiplex tool response supplies graph provenance
  despite providing no presence or absence list. Do not retry this
  deterministic outcome or apply S1.1.8's transient-error retry policy.
- **Seventh-increment verification (S1.1.11).** Exactly 300 training rows
  change from the sixth increment: 170 user messages and 600 post-tool
  thoughts. All other content in these rows and the other 9,700 training
  rows remain unchanged. Both complete held-out files remain byte-identical,
  including all 37 validation and 37 test examples of this form. All 55
  snapshotted files retain their hashes, including the shared contract,
  historical corpora and evaluator files, and retry diagnostic questions
  and keys. Refresh only the retry diagnostic manifest's v6 overlay pin;
  its panel ID stays fixed. Each framing uses all 64 restored wrappers,
  two or three times each. Semantic validation passes all 10,000 training,
  1,250 validation, and 1,250 test rows.
- The seventh-increment component train SHA-256 was
  `9894f9479fc07ba827e1291c4c6db54155afb6ec8ca6fa65ca08598ef3fb4af1`.
- **Seventh-increment checks.** All 136 focused corpus, overlay,
  prompt-bank, and retry tests pass, including deterministic repeated
  builds and reproduction of no-overlay, v4, and v5 split bytes.
  Validator selftest passes all 53 known-bad fixtures with zero false
  positives. The actual uniform-SFT tokenizer checks all 300 revised
  examples and confirms supervision of all 600 revised thoughts.
  Lengths range from 892 to 932 tokens, below the 4,096-token limit.
  Training recipes, frozen scorer policy, and saved predictions remain
  unchanged. No training or model evaluation jobs are submitted.

- **S1.2.1 results.** Fresh rescoring on the same 63 held-out questions,
  current scorer, and identical answer keys gives v4 63/63 (100%) and v5
  57/63 (90.48%) structured accuracy. Both models return the correct gene
  result, tool and arguments, status, and provenance on every row.
  V5's six errors choose `continue` instead of `stop`: RNF133, CPEB3,
  MTND3P9, GAPDHP55, RPL13, and FAM90A6P. Their thinking invents a next
  computation or analysis despite each request asking only for a symbol.
  This is a 9.52-point continuation decline, without a lookup regression.
- **S1.2.1 approved wording.** Restore the 64 v4 direct wrappers at the
  existing template assignments in all 800 v5 training examples. Exactly
  110 user messages change; 690 are already identical. Keep all 800
  identities, including the 300 added beyond v4's 500. Among the 500
  shared v4/v5 row IDs, 71 users and every evidence ID differ; other
  message content is identical. Copy wrapper wording only, preserving
  the v5 evidence IDs and the additional identities. The results support
  clearer standalone wording without establishing which v5 change caused
  the errors. They do not justify reducing the allocation.
- **S1.2.1 approved reasoning.** Replace final thinking in all 800 rows:
  "The registry resolves {gene_id} to {symbol}, completing the requested
  lookup. Return this mapping with registry evidence and stop." Bind both
  fields to the actual tool response. Preserve pre-call thinking and every
  final answer. Each target retains one `lookup_human_ensembl_gene_id`
  call, `resolved/stop`, exactly `gene_id` and `gene_symbol` in the result,
  registry provenance, null graph provenance, and `blocked_reason:null`.
  The template passes the existing two-sentence, 25-word-per-sentence,
  and 40-word-total limits.
- **S1.2 failure pairs.** Keep all 800 S1.2.11 training rows unchanged.
  The pair audit checks both members against their declared training
  wrappers and retains identical inputs, framing, and template indices.
  All 800 source pairs remain valid; 690 also retain identical user text.
  The other 110 pairs differ only by the approved form-1 wrapper change.
  The duplicate audit still rejects unintended prompt/outcome duplicates.
  Without overrides, the pair audit still requires identical prompt text.
- **Eighth-increment verification (S1.2.1).** Exactly 800 training rows
  change relative to v5: 110 user messages and 800 final thoughts. All
  remaining content, evidence IDs, metadata, and source order remain
  unchanged. The other 10,200 S1.2 training rows remain byte-identical.
  Both complete held-out files remain byte-identical, including all 63
  validation and 63 test examples of this form. All 70 snapshotted files
  retain their hashes, including historical corpora and evaluator files,
  the shared contract, retry diagnostic questions and keys, and all three
  current S1.1 row files. Refresh only S1.1's and the retry diagnostic's
  manifests to carry the new overlay pin. The retry panel ID stays fixed.
  Every restored wrapper occurs 12 or 13 times. Semantic validation passes
  all 11,000 training, 1,375 validation, and 1,375 test rows. The wording,
  thinking, duplicate, and pair audits pass without changing their gates.
- The eighth-increment S1.2 component train SHA-256 was
  `69bac1b47d560e9312a29f670c685b36a45aee69d6e2a5e23a8aab57a5ef64c4`.
- **Eighth-increment checks.** All 173 focused corpus, overlay,
  prompt-bank, and retry tests pass: 37 S1.2 checks and 136 S1.1/shared
  checks. They include deterministic repeated builds and reproduction
  of no-overlay, v4, and v5 split bytes for both components.
  Validator selftest passes all 53 known-bad
  fixtures with zero false positives. The actual uniform-SFT tokenizer
  checks all 800 revised examples and confirms supervision of all 800
  revised thoughts. Lengths range from 590 to 643 tokens, below 4,096.
  Training recipes, frozen scorer policy, and saved predictions remain
  unchanged. Later forms await review; no aggregate v6 bundle is built
  and no training or model evaluation jobs are submitted.

- **S1.2.2 retained.** Both models get 63/63 (100%) on the same held-out
  questions, current scorer, and keys. Every result, call and argument,
  status, provenance, and continuation decision is correct. Retain all
  500 v5 training rows unchanged, including their wrappers, thinking,
  tool evidence, and resolved/continue targets. No rollback is supported.
- **S1.2.3 results.** Fresh rescoring on the same 125 held-out questions,
  current scorer, and identical keys gives v4 125/125 (100%) and v5
  117/125 (93.60%) structured accuracy. Every initial lookup is correct,
  including passing the versioned input unchanged and returning its
  canonical gene ID. V5 gets 121 correct final gene results, 122 correct
  complete tool sequences and arguments, and 118 correct decisions.
  Direct requests score 55/60; workflow requests score 62/65.
- **S1.2.3 failures.** Four errors change only continuation: ADAD1P2,
  SNORA50A, and MAP2K4P1 continue on direct requests; WFDC21P stops on
  a workflow request. Four others attempt unrequested membership checks
  with invented multiplex IDs: LONRF2, TMBIM7P, CTSL3P, and RPL7P19.
  Three calls execute; CTSL3P's native call uses the wrong channel and
  is rejected. RPL7P19 also emits the invalid decision `reject`.
  The failures concern completion and task scope, rather than suffix
  handling. They do not establish which v5 change caused the decline.
- **S1.2.3 approved wording.** Keep 1,000 training identities and the
  current 438 direct/562 workflow assignments. Restore the full v4 bank
  at the current v5 template indices, changing exactly 624 users: 62
  direct and all 562 workflow. V4 had 479 direct and 521 workflow rows;
  v5 moved 41 of these inputs to workflow framing and reassigned wrappers.
  Do not copy whole v4 rows or restore their targets and metadata.
- **S1.2.3 approved reasoning.** Replace all 1,000 final thoughts. Both
  framings start: "The registry ignores the numeric version suffix and
  returns {gene_id} with symbol {symbol}, completing the requested lookup."
  Direct requests add: "Return this mapping with registry evidence and
  stop." Workflow requests add: "Return this mapping with registry
  evidence and continue so the next workflow step can use it." Bind
  both slots to the actual tool response. The templates satisfy the
  two-sentence, 25-word-per-sentence, and 40-word-total prose limits.
- **S1.2.3 preserved scope.** Keep pre-call thinking, every versioned
  argument, tool response, evidence ID, final answer, metadata field,
  row ID, and source ordering. Each target remains one ID lookup, then
  resolved with exactly `gene_id` and `gene_symbol`, registry evidence,
  null graph provenance, and no blocked reason. Preserve the current
  framing-specific stop/continue decision. Keep the other 10,000 training
  rows unchanged, including all S1.2.1 edits and retained S1.2.2 rows.
  Both whole held-out files and the completed S1.1 rows stay frozen.
- **Ninth-increment verification (S1.2.3).** Exactly 1,000 training rows
  change from the eighth increment: 624 user messages and 1,000 final
  thoughts. The user changes comprise 62 direct and 562 workflow rows.
  All other fields and the other 10,000 training rows remain unchanged.
  The framing counts stay 438 direct and 562 workflow. Each framing uses
  all 64 restored wrappers, six or seven times for direct requests and
  eight or nine for workflow requests. Both complete held-out files stay
  byte-identical, including all 125 validation and 125 test versioned-ID
  examples. All 70 snapshotted files retain their hashes, including the
  shared contract, historical corpora and evaluator files, retry questions
  and keys, and all current S1.1 row files. Refresh only S1.1's and the
  retry diagnostic's manifests for the new overlay pin; its panel ID
  stays fixed. Semantic validation passes all 11,000 training, 1,375
  validation, and 1,375 test rows. The wording, thinking, duplicate, and
  pair audits pass without changing their gates.
- The ninth-increment S1.2 component train SHA-256 was
  `d04afbc9eeed1ffadafa251a8200c136b03d86c58ede568bad59beb4d62cf5cd`.
- **Ninth-increment checks.** All 189 focused corpus, overlay,
  prompt-bank, and retry tests pass: 53 S1.2 checks and 136 S1.1/shared
  checks. They include deterministic repeated builds and reproduction
  of no-overlay, v4, and v5 split bytes for both components.
  Validator selftest passes all 53 known-bad
  fixtures with zero false positives. The actual uniform-SFT tokenizer
  checks all 1,000 revised rows and confirms supervision of all 1,000
  revised thoughts. Lengths range from 616 to 675 tokens, below 4,096.
  Training recipes, frozen scorer policy, and saved predictions remain
  unchanged. Later forms await review; no aggregate v6 bundle is built
  and no training or model evaluation jobs are submitted.

- **S1.2.4 results.** Fresh rescoring on the same 388 held-out questions,
  current scorer, and identical keys gives v4 283/388 (72.94%) and v5
  74/388 (19.07%) structured accuracy. All 388 initial calls in both
  models return the correct canonical ID, resolved status, and null
  symbol. Exact recorded tool sequences and arguments fall from 388
  to 240; correct final gene results from 322 to 217; correct decisions
  from 283 to 108. V5 scores 32/186 on direct requests and 42/202 on
  workflow requests. The 53.87-point decline follows successful lookups.
- **S1.2.4 failures.** V5's 314 structured failures comprise 137 rows
  with extra recorded calls, nine with rejected extra call attempts,
  99 with wrong decisions after one correct call, 19 with wrong statuses
  after one call, and 50 with invalid final responses after one call.
  Of the 99 decision errors, 84 defer and 15 swap stop/continue. The
  19 status errors are 13 not_found and six ambiguous. Extra calls
  include repeated ID lookups, symbol searches, invented tools, and
  membership checks. The 45 execution errors comprise 31 empty-symbol
  calls and 14 uses of the nonexistent `lookup_human_gene_id` tool.
  Eleven additional rows return correct structured answers after extra
  calls; only 63 rows combine structured correctness with exact calls.
  V4 also needs a clearer completion rule: its 105 failures are 81
  inappropriate deferrals and 24 stop/continue swaps. These results
  support explicit null-result reasoning alongside wrapper restoration;
  they do not isolate which v5 training change caused the regression.
- **S1.2.4 approved wording.** Restore the v4 bank at the current v5
  framing and template assignments. Exactly 1,931 user messages change:
  190 direct and all 1,741 workflow requests. Keep all 3,100 identities
  and the current 1,359 direct/1,741 workflow allocation. V4 had 1,485
  direct and 1,615 workflow rows; preserve the 126 inputs that v5 moved
  to workflow framing. Do not copy whole v4 rows or restore old targets.
- **S1.2.4 approved reasoning.** Replace all 3,100 final thoughts. Both
  framings begin: "The registry reports resolved for {gene_id} with
  gene_symbol:null, completing the lookup without an assigned symbol."
  Direct requests add: "Return the ID and null symbol with registry
  evidence, no blocked reason, and stop." Workflow requests add: "Return
  the ID and null symbol with registry evidence, no blocked reason, and
  continue using the ID." Bind only the canonical ID from the actual
  tool response. The templates satisfy two sentences, 25 words per
  sentence, and 40 words total. The overlay's reviewed map replaces
  the v5 bank's generic unnamed-gene reasoning for this form only.
- **S1.2.4 preserved scope.** Keep pre-call thinking, arguments, calls,
  responses, evidence IDs, final answers, metadata, row IDs, and ordering.
  Each target remains one ID lookup, then resolved with exactly gene_id
  and gene_symbol:null, registry evidence, null graph provenance, and
  blocked_reason:null. Keep the framing-specific stop/continue decision.
  A resolved null-symbol result completes the lookup and supplies no
  retry allowance. The other 7,900 training rows, both complete held-out
  files, and the completed S1.1 row files stay frozen.
- **Tenth-increment verification (S1.2.4).** Exactly 3,100 training rows
  change from the ninth increment: 1,931 user messages and 3,100 final
  thoughts. The user changes comprise 190 direct and 1,741 workflow
  rows. Every other field and the other 7,900 training rows remain
  unchanged, including the earlier S1.2.1 and S1.2.3 edits. Framing
  counts stay 1,359 direct and 1,741 workflow. Each framing uses all 64
  restored wrappers, 21 or 22 times for direct requests and 27 or 28
  for workflow requests. Both complete held-out files stay byte-identical,
  preserving all 388 validation and 388 test examples of this form.
  A fresh 55-file snapshot protects historical corpora and evaluator
  files, frozen banks and overlays, training configs, the shared contract,
  both current S1.2 held-out files, retry questions and keys, and all
  S1.1 row files. Every snapshotted hash matches after the rebuild.
  Refresh only S1.1's and the retry diagnostic's manifests for the new
  overlay pin; the retry panel ID stays fixed. Semantic validation
  passes all 11,000 training, 1,375 validation, and 1,375 test rows.
  The wording, thinking, duplicate, and pair audits retain their gates.
- The tenth-increment S1.2 component train SHA-256 was
  `1f2472c03479e22442676d87b286eadc92577c40c017f120201a3dc7a1fde2d2`.
- **Tenth-increment checks.** All 205 focused corpus, overlay,
  prompt-bank, and retry tests pass: 69 S1.2 checks and 136 S1.1/shared
  checks. They include deterministic repeated builds and reproduction
  of no-overlay, v4, and v5 split bytes for both components.
  Validator selftest passes all 53 known-bad
  fixtures with zero false positives. The actual uniform-SFT tokenizer
  checks all 3,100 revised rows and confirms supervision of all 3,100
  revised thoughts. Lengths range from 598 to 628 tokens, below 4,096.
  The no-symbol templates bind only the returned canonical ID and keep
  the symbol literal null. Training recipes, frozen scorer policy, and
  saved predictions remain unchanged. Later forms await review; no
  aggregate v6 bundle is built and no jobs are submitted.

- **S1.2.5 results.** Fresh rescoring on the same 125 held-out questions,
  current scorer, and identical keys gives both v4 and v5 119/125
  (95.2%) structured accuracy. All initial ID lookups and arguments
  are correct in both runs. V4's historical report gave 114/125 because
  five acceptable blocked-reason paraphrases were counted as failures.
  Both models have five stop decisions with a nonempty blocked reason,
  violating the defer contract and causing parser rejection. V4's sixth
  error adds gene_symbol:null to the result; v5's sixth invents multiplex
  M1 and attempts membership after the negative registry lookup.
  V5's complete call sequence is correct on 124/125 rows versus v4's
  125/125. These results show no structured-accuracy regression and
  provide no clear reason to restore v4 wrappers or increase allocation.
- **S1.2.5 approved edit.** Preserve all 1,000 examples, exact input
  identities, v5 prompts, and 438 direct/562 workflow assignments.
  Replace only final thinking with: "The registry reports not_found for
  {gene_id} in release 116, so the requested symbol is unavailable.
  Return only the queried gene_id with registry evidence and defer with
  a blocked reason until a current ID is supplied. Do not retry this
  negative result, guess a successor, or invent a graph lookup."
  Form 5 of S1.2's pinned train_final_reasoning supplies the template
  for both framings and binds only the exact queried ID in the tool
  response, preserving case and suffixes. The form-specific validator
  permits three sentences, 25 words per sentence, and 60 words total;
  other templates retain their existing prose limits.
- **S1.2.5 preserved scope.** Keep pre-call thinking, user messages,
  one ID lookup, exact arguments, tool responses and evidence IDs,
  final answers, metadata, row IDs, and ordering. Each target remains
  not_found/defer, with only gene_id in result, registry evidence,
  null graph provenance, and a nonempty blocked reason. A deterministic
  negative result supplies no retry allowance. The other 10,000 S1.2
  training rows, both complete held-out files (including 125 examples
  of this type each), and the completed S1.1 row files stay frozen.
- **Eleventh-increment verification (S1.2.5).** Exactly 1,000 final
  thoughts change from the tenth increment. Every other field and all
  other 10,000 S1.2 training rows remain byte-identical, including the
  earlier S1.2.1, S1.2.3, and S1.2.4 edits. User wording is unchanged
  throughout all 11,000 rows. Form 5 retains 438 direct and 562 workflow
  examples, one lookup per row, and the existing not_found/defer targets.
  Both complete held-out files remain byte-identical. All 73 snapshotted
  files retain their hashes, protecting historical corpora and evaluator
  files, frozen banks and overlays, training configs, the shared contract,
  current held-out splits, all S1.1 row files, and retry questions and keys.
  Only S1.1's and the retry diagnostic's manifests refresh for the new
  overlay pin; their row files and retry panel ID stay fixed. Semantic
  validation passes all 13,750 S1.2 rows and all 12,500 S1.1 rows.
  The wording, thinking, duplicate, and pair audits retain their gates.
- The eleventh-increment S1.2 component train SHA-256 was
  `489b23b9717c63e67dc7fe7dc8bf21f22a348c656d3681c410a5c7fd9a13d1df`.
- **S1.2.5 tokenizer check.** The actual uniform-SFT tokenizer checks
  all 1,000 revised rows and confirms supervision of all revised final
  thoughts. Lengths range from 643 to 668 tokens, below 4,096. Validator
  selftest passes all 53 known-bad fixtures with zero false positives.
  Later question types await review; no aggregate v6 bundle is built and
  no training or model evaluation jobs are submitted.
- **Eleventh-increment checks.** All 212 focused tests pass: 76 S1.2
  checks and 136 S1.1/shared corpus, overlay, prompt-bank, and retry
  checks. They verify the exact reviewed edits, deterministic repeated
  builds, unchanged held-out bytes, scoped template slots and prose
  limits, and reproduction of no-overlay, v4, and v5 split bytes for
  both components. The final-reasoning map remains limited to S1.2.1
  and S1.2.5; wrapper overrides still exclude S1.2.5.

- **S1.2.6 results.** Fresh rescoring on the same 62 held-out questions,
  current scorer, and identical keys gives v4 61/62 (98.39%) and v5
  42/62 (67.74%) structured accuracy, a decline of 30.65 points.
  Both models make the correct lookup with exact arguments and tool
  payloads on all 62 rows. Direct accuracy falls from 29/30 to 23/30;
  workflow accuracy falls from 32/32 to 19/32. Every raw final response
  recognizes invalid_input, so the failures follow correct tool evidence.
- **S1.2.6 failures.** V5 has 13 accepted finals with wrong result fields:
  12 return only gene_id and one adds gene_id to input and expected.
  Seven other rows fail final-response parsing because their decision
  is wait (four), await_input (two), or stop (one), rather than defer.
  The stop response also adds gene_id. V4's only failure chooses stop
  with a blocked reason. Some v5 reasoning miscounts digits; use the
  tool's invalid_input result without recounting or repairing the ID.
  Wrapper causality remains uncertain, but the errors directly support
  explicit schema and decision reasoning. Allocation need not increase.
- **S1.2.6 approved wording.** Restore the frozen v4 bank at the existing
  v5 framing and template indices. Exactly 310 user messages change:
  29 direct and all 281 workflow requests; 190 users already match.
  Keep all 500 examples, exact malformed inputs, and the 219 direct/281
  workflow allocation. Do not restore whole v4 rows or reassign examples.
- **S1.2.6 approved reasoning.** Replace all 500 final thoughts with:
  "The registry rejects {input}; return invalid_input with exactly input
  and expected:human_gene_id, preserving the original text. Keep registry
  evidence and set continuation_decision:defer with a blocked reason;
  do not repair the input or call again." Form 6 of S1.2's pinned
  train_final_reasoning supplies the same template for both framings.
  Bind only the exact input from the tool's invalid_input response;
  preserve its case and digits. The template has two sentences of
  19 and 20 words, within the existing 25-per-sentence and 40-total caps.
  The input slot is restricted to this reviewed form; resolved lookup
  and not_found templates retain their existing slot restrictions.
  The original thinking filter rejects continuation_decision as schema
  discussion. Permit only the literal continuation_decision:defer when
  the full thought exactly matches this pinned final training template
  for form 6. All other schema/provenance phrases stay forbidden, as
  do the same phrase in changed thoughts, pre-call messages, other
  forms, or held-out rows. Identifier visibility, status agreement,
  and decision agreement checks still run on the approved final thought.
- **S1.2.6 preserved scope.** Keep pre-call thinking, one ID lookup,
  exact arguments, tool responses and evidence IDs, final answers,
  metadata, row IDs, and ordering. Each target stays invalid_input/defer,
  with exactly input and expected:human_gene_id in result, registry
  evidence, null graph provenance, and a nonempty blocked reason.
  Keep all other 10,500 S1.2 training rows and both full held-out files,
  including all 62 validation and 62 test examples of this type.
  Completed S1.1 row files stay frozen. Review later forms before
  assembling the v6 aggregate or submitting training or evaluation jobs.
- **Twelfth-increment verification (S1.2.6).** Exactly 500 training
  rows change from the eleventh increment: 310 user messages and all
  500 final thoughts. User changes comprise 29 direct and 281 workflow
  messages. Every other field and the other 10,500 S1.2 training rows
  remain byte-identical, including all earlier reviewed changes. Framing
  remains 219 direct and 281 workflow examples. Each framing retains
  all 64 frozen v4 wrappers, used three or four times in direct requests
  and four or five times in workflow requests. Both full held-out files
  stay byte-identical, including all 62 examples of this type per split.
  All 73 protected file hashes match, preserving historical corpora,
  evaluator files, frozen banks and overlays, training configs, the
  shared contract, current held-out splits, all S1.1 rows, and retry
  questions and keys. Refresh only S1.1's and the retry diagnostic's
  manifests for the new overlay pin; their rows and panel ID stay fixed.
  Semantic validation passes all 13,750 S1.2 and 12,500 S1.1 rows.
  The wording, duplicate, and failure-pair audits retain their gates;
  the thinking audit admits only the reviewed decision-field phrase.
- The twelfth-increment S1.2 component train SHA-256 was
  `a15a33def1e346abd969134359f6e87c1c6dadee9f9fc4fbf3fce5c15ea6e8c8`.
- **S1.2.6 tokenizer check.** The actual uniform-SFT tokenizer verifies
  all 500 revised rows and supervision of all revised final thoughts.
  Lengths range from 653 to 678 tokens, below 4,096. Validator selftest
  rejects all 53 known-bad fixtures with zero false positives. Later
  forms await review; the aggregate v6 bundle remains unbuilt and no
  training or model evaluation jobs are submitted.
- **Twelfth-increment checks.** All 220 focused tests pass: 84 S1.2
  checks and 136 S1.1/shared corpus, overlay, prompt-bank, and retry
  checks. They verify exact message changes, unchanged held-out bytes,
  deterministic repeated builds, historical no-overlay/v4/v5 split
  reproduction for both components, and scoped template bindings.
  The thinking-filter regression accepts the pinned final training
  thought and rejects use without its pin, changed text, pre-call use,
  held-out use, other forms, and additional schema phrases. Status and
  decision contradictions still fail inside the approved thought.

- **S1.2.7 retained.** Both v4 and v5 get all 125 held-out questions
  right under the current scorer and keys. Keep every v5 row unchanged.
- **S1.2.8 results.** Fresh rescoring on the same 62 held-out questions,
  current scorer, and identical keys gives v4 62/62 (100%) and v5
  60/62 (96.77%) structured accuracy. Both runs have correct tools,
  arguments, payloads, mappings, statuses, and evidence on all 62 rows.
  Direct requests stay 30/30; workflow requests fall from 32/32 to 30/32.
  CTSL3P and GAPDHP55 return correct mappings but choose stop instead
  of continue. Both use the held-out downstream-use wrapper. Their
  reasoning equates a finished lookup with a finished workflow.
  These two failures do not establish wrapper causality, but support
  explicit user-purpose reasoning with the stronger v4 wording baseline.
- **S1.2.8 approved wording.** Restore the frozen v4 bank at the current
  v5 framing and template assignments. Exactly 310 user messages change:
  29 direct and all 281 workflow requests. Keep all 500 examples, exact
  identities, harness blocks and distractors, and the 219 direct/281
  workflow allocation. Do not copy whole v4 rows or reassign framings.
- **S1.2.8 approved reasoning.** Replace all 500 final thoughts. Both
  framings start: "The registry resolves {gene_id} to {symbol}."
  Direct requests add: "Because the user requested only this lookup,
  return the mapping with registry evidence and stop." Workflow requests
  add: "Because the user needs this mapping for a later workflow step,
  return it with registry evidence and continue." Bind the canonical ID
  and symbol from the actual lookup, not the unrelated harness entry.
  The pinned train_near_miss_reasoning map is train-only and contains
  exactly direct and step templates with the existing two-sentence,
  25-word-per-sentence, and 40-word-total limits.
- **Approved workflow backfill.** Refine final thinking only in all
  1,600 S1.1.2 rows, the 562 workflow S1.2.3 rows, and the 1,741 workflow
  S1.2.4 rows. S1.1.2 and S1.2.3 retain their first evidence sentence
  and replace the second with: "Because the user needs this mapping
  for a later workflow step, return it with registry evidence and
  continue." S1.2.4 retains its resolved/null-symbol evidence sentence
  and adds: "Because the user's workflow needs this result, return the
  ID and null symbol with registry evidence, no blocked reason, and
  continue." Direct thoughts in S1.2.3 and S1.2.4 remain unchanged.
  S1.1.9 and S1.1.10 already state that the next workflow step needs
  the membership result; preserve those reviewed thoughts. S1.1.1
  already explains its standalone scope. All blocked rows keep defer:
  user workflow wording alone cannot supply missing evidence. Continue
  passes a completed result back to the requested workflow and does not
  authorize invented downstream operations or extra tool calls.
- **S1.2.8 and backfill scope.** Preserve pre-call thinking, calls,
  exact arguments, tool payloads and evidence IDs, final answers,
  provenance, statuses, decisions, blocked reasons, metadata, row IDs,
  and ordering. The only user-message edits are the 310 restored
  S1.2.8 wrappers. Expected final-thought changes from the preceding
  increment are 4,403: 1,600 in S1.1.2, 562 in S1.2.3, 1,741 in S1.2.4,
  and 500 in S1.2.8. All other rows and both complete held-out splits
  per component stay fixed, along with all historical v1-v5 artifacts.
  No aggregate v6 bundle or training/evaluation jobs are part of this increment.
- **Thirteenth-increment verification (S1.2.8 and workflow backfill).**
  Independent comparison against the preceding v6 components confirms
  exactly 4,403 revised final thoughts and 310 revised user messages,
  with the per-form and framing counts above. Every other message field,
  target, fact, evidence ID, and metadata field stays identical. All
  8,400 other S1.1 training rows and 8,197 other S1.2 training rows stay
  unchanged. Direct S1.2.3/S1.2.4 thoughts, all blocked rows, and all
  S1.2.7 rows remain fixed. Both complete held-out files per component
  remain byte-identical; all 72 protected file hashes match. Each
  S1.2.8 framing retains all 64 v4 wrappers, used three or four times
  for direct requests and four or five times for workflow requests.
  Semantic validation passes all 12,500 S1.1 and 13,750 S1.2 rows.
  Refresh the component manifests and only the retry diagnostic's
  overlay pin; its questions, keys, policy, and panel ID remain fixed.
- The thirteenth-increment S1.1 component train SHA-256 was
  `f57422d99da309fb1a4c84758686c364ca384bd2795c8b545c4be02f2e351c89`.
  The thirteenth-increment S1.2 component train SHA-256 was
  `6df6ff92955ac738b0a6e9538059a5d1f1ca9d33707d97b0b434014a07c42d27`.
- **Thirteenth-increment tokenizer check.** The actual uniform-SFT
  tokenizer verifies all 4,403 revised rows and confirms that every
  revised final thought appears once and remains fully supervised.
  Lengths range from 590 to 731 tokens, below the 4,096-token limit.
  Validator selftest rejects all 53 known-bad fixtures with zero false
  positives.
- **Thirteenth-increment checks.** All 248 focused tests pass: 112
  S1.2 checks and 136 S1.1/shared corpus, overlay, prompt-bank, and retry
  checks. They verify the exact message changes, frozen held-out bytes,
  deterministic repeated builds, historical no-overlay/v4/v5 split
  reproduction for both components, and scoped template bindings.
  Matched standalone/workflow fixtures retain identical tool calls,
  evidence, results, statuses, and blocked reasons. Resolved requests
  select stop or continue by user scope; blocked requests retain defer.
  The new reasoning templates remain training-only. The aggregate v6
  bundle remains unbuilt, and no training or model evaluation jobs are
  submitted.
- **S1.2.9 results.** Rescoring the same 62 held-out questions with
  the current scorer and v5 keys gives v4 58/62 (93.55%) and v5 56/62
  (90.32%) structured accuracy. Direct requests score 27/29 versus
  26/29; workflow requests score 31/33 versus 30/33. Both models select
  the right initial symbol lookup and exact argument on all 62 rows.
  V4's four failures are stop/continue mistakes. V5 stops on workflow
  requests for IFT140, THY1, and C2, and continues on the direct CIROZ
  request. GARIN5B and RNU2-20P resolve correctly, then v5 invents
  multiplex M1 and an unrequested membership check before deferring.
  Complete tool-sequence accuracy falls from 62/62 to 60/62.
- **S1.2.9 approved wording.** Restore the frozen v4 bank at the
  existing v5 indices. Exactly 310 user messages change: 29 direct
  and 281 workflow requests. Keep all 500 examples and the current
  219 direct/281 workflow assignments. Both versions show scope errors;
  the score difference does not establish wrapper causality.
- **S1.2.9 approved reasoning.** Replace all 500 final thoughts with
  the pinned train_symbol_input_reasoning map. Both framings begin:
  "The registry confirms {symbol} as the symbol for {gene_id}; no
  membership check is requested." Direct requests add: "Because the
  user requested only this lookup, return the mapping with registry
  evidence and stop." Workflow requests add: "Because the user needs
  this mapping for a later workflow step, return it with registry
  evidence and continue." Bind the canonical values from the symbol
  tool response. Keep exactly direct and step templates, applied only
  in training, with two sentences, 25 words per sentence, and 40 total.
- **S1.2.9 preserved scope.** Keep pre-call reasoning, one
  lookup_human_gene_symbol call, original argument spelling, tool
  responses, final answers, evidence IDs, metadata, row IDs, and
  ordering. Targets remain resolved with exactly gene_id and
  gene_symbol, registry provenance, null graph provenance, and no
  blocked reason. Preserve the other 10,500 S1.2 training rows,
  every S1.1 row, and both complete held-out files per component,
  including all 62 validation and 62 test examples of this form.
  Historical corpora, banks, overlays, configs, evaluator questions
  and keys, and the shared contract remain frozen. Build only the
  S1.1/S1.2 components and refresh the retry diagnostic's manifest
  overlay pin. Its questions, keys, retry policy, and panel ID stay fixed.
  Review later types before assembling the v6 aggregate or submitting jobs.
- **Fourteenth-increment verification (S1.2.9).** Comparison against
  the preceding v6 component confirms exactly 500 changed training
  rows: 310 user messages (29 direct and 281 workflow) and all 500
  final thoughts. Every other field and the other 10,500 S1.2 training
  rows stay byte-identical, preserving all earlier reviewed fixes.
  Framing stays 219 direct and 281 workflow examples. Each framing
  retains all 64 v4 wrappers, used three or four times for direct
  requests and four or five times for workflow requests. Both full
  held-out files stay byte-identical, including all 62 examples of
  this type per split. All 73 protected hashes match, preserving the
  shared contract, historical artifacts, current held-out splits,
  all S1.1 row files, and retry questions and keys. S1.1's manifest
  changes only its overlay pin; the retry panel ID remains fixed. Semantic
  validation passes all 13,750 S1.2 and 12,500 S1.1 rows.
- The fourteenth-increment S1.1 component train SHA-256 remained
  `f57422d99da309fb1a4c84758686c364ca384bd2795c8b545c4be02f2e351c89`.
  The fourteenth-increment S1.2 component train SHA-256 was
  `3301b946c595ada93637b4cfde82499590d4dcb43f09bcdbce1cd37f5dd21196`.
- **S1.2.9 tokenizer check.** The actual uniform-SFT tokenizer verifies
  all 500 revised rows. Every revised final thought appears once and
  remains fully supervised. Lengths range from 602 to 659 tokens,
  below the 4,096-token limit. Validator selftest rejects all 53
  known-bad fixtures with zero false positives.
- **Fourteenth-increment checks.** All 268 focused tests pass: 132
  S1.2 checks and 136 S1.1/shared corpus, overlay, prompt-bank, and retry
  checks. They verify exact approved edits, unchanged held-out bytes,
  deterministic repeated builds, historical no-overlay/v4/v5 split
  reproduction, pinned bank bytes, and scoped template bindings.
  Matched standalone/workflow fixtures preserve the original lowercase
  symbol argument, perform one symbol lookup, and return its canonical
  ID and symbol. Only resolved continuation changes with user scope;
  blocked cases still defer. Held-out reasoning retains its original
  content even when the new templates are supplied to the builder.
  The aggregate v6 bundle remains unbuilt, and no training or model
  evaluation jobs are submitted.
- **S1.2.10 retained.** Both models score 125/125 under the current
  scorer and keys. Keep every row unchanged.
- **S1.2.11 results.** On the same 50 held-out questions, current scorer,
  and v5 keys, v4 scores 14/50 (28%) and v5 scores 40/50 (80%). Both
  initial ID lookups and arguments are correct on all 50 rows. Complete
  tool sequences score 49/50 in both runs; continuation scores 47/50
  versus 48/50. Eight v5 failures use the wrong error-result schema:
  seven substitute message for failed_tool and one returns only gene_id.
  Another returns stop with a blocked reason, so the parser rejects its
  envelope. The last retries ENSG00000236432 without a visible allowance,
  obtains MFF-DT, and correctly stops. The legacy harness makes a second
  attempt succeed by construction; this is not evidence of outage recovery
  under a persistent fault or authorization to retry a plain legacy error.
- **S1.2.11 approved allocation.** Keep v5's 800 identities, user
  messages, wrapper assignments, first-call reasoning and arguments,
  paired successes, row IDs, and ordering. Preserve all 46 lowercase
  inputs. Teach the already approved explicit allowance with 400 terminal
  failures, 200 successful retries, and 200 exhausted retries. Assign
  cases by case-folded ID and split-unit hashes, independently of wording
  and row IDs. All 800 final thoughts change; 400 rows gain exactly one
  identical call and tool response. First failures in these 400 rows carry
  registry unreachable; retries_remaining=1. Exhausted retries return
  registry unreachable; retries_remaining=0. Other error-result fields
  and the shared contract stay fixed. The new content-pinned policy is
  runtime/world_model/s1_v6_id_lookup_retries.json; keep the original
  S1.1.8 policy bytes and allocation unchanged.
- **S1.2.11 failure targets and reasoning.** The 600 terminal or
  exhausted cases retain error/defer, exactly failed_tool and error_code
  plus the original queried gene_id, null provenance, and a nonempty
  blocked reason. Bind the failed lookup's input and BACKEND_UNAVAILABLE
  in the first sentence, explaining the absent or exhausted allowance.
  Both cases conclude: "Return error/defer with exactly gene_id,
  failed_tool, and error_code, no evidence, and a blocked reason; do not
  infer absence." Replace summaries that previously instructed another
  retry with the explicit need to restore registry access.
- **S1.2.11 recovery targets and reasoning.** The 200 recoveries return
  resolved/stop, exactly the canonical gene_id and gene_symbol, only the
  successful call's registry evidence, null graph provenance, and no
  blocked reason. Their first sentence is: "The retry resolves {input}
  to {symbol}, returning canonical ID {gene_id}." The second is:
  "Because the user requested only this lookup, return the mapping with
  registry evidence and stop." Retry reasoning explicitly names one
  remaining attempt and repeats the same ID tool and argument. All new
  thinking stays within two sentences, 25 words per sentence, and 40 total.
  Allocate new recovery evidence after all baseline rows so other rows
  and held-out evidence nonces remain fixed.
- **S1.2.11 validation and diagnostic.** Extend the shared retry helper,
  semantic validator, native fault harness, and structured scorer to ID
  lookups with the same visible allowance. Reject changed arguments,
  changed tools, extra attempts, unbudgeted recovery, invented evidence,
  and incorrect final schemas. Preserve legacy no-overlay/v4/v5 split
  bytes and frozen promotion questions, keys, and panel ID. Add a separate
  150-row diagnostic at data/world_model_v4/eval/s1_v6/id_lookup_retries,
  using the same 50 frozen ID failure genes in all three fault schedules.
  Keep the original 150-row symbol diagnostic's questions, keys, and panel
  ID fixed; refresh only its manifest pins. Keep the other 10,200 S1.2
  training rows, all S1.1 rows, and both full held-out files per component
  byte-identical. Do not combine the diagnostics with frozen promotion
  scores or treat scripted trajectories as measured model accuracy.

```bash
python -m runtime.world_model.s1_1_corpus --overlay runtime/world_model/s1_v6.yaml
python -m runtime.world_model.s1_2_corpus --overlay runtime/world_model/s1_v6.yaml
python scripts/build_world_model_s1_lookup_retry_panel.py
python scripts/build_world_model_s1_lookup_retry_panel.py --section S1.2
```

- **Fifteenth-increment verification (S1.2.11).** An independent full-row
  comparison confirms exactly 800 revised training rows, with zero user
  or first-call reasoning changes. All 46 lowercase inputs remain intact.
  The 400/200/200 allocation produces 400 additional identical calls,
  400 visible first-error allowances, and 200 successful evidence blocks.
  The other 10,200 S1.2 training rows and every S1.1 row stay byte-identical,
  along with both complete held-out files per component. All 73 protected
  file hashes match. New evidence is appended after the complete original
  evidence ledger, with no collisions or changes to earlier nonces.
  S1.1's manifest changes only its overlay and shared retry implementation
  pins; the symbol diagnostic changes only its manifest pins.
- The current S1.1 component train SHA-256 remains
  `f57422d99da309fb1a4c84758686c364ca384bd2795c8b545c4be02f2e351c89`.
  The current S1.2 component train SHA-256 is
  `11ba548eebb598c3ba6f7b56a1d33fadf6095298fe5cdf88eaac56d727afeec5`.
- **ID-retry diagnostic verification.** Its panel ID is
  `19667fa9ff942125239b86457e731a31a93892effa6d5e6ead9ed529d16a15cb`.
  All 150 questions retain the original held-out user and system messages;
  the three fault cases share each of the 50 genes. Scripted actions pass
  the native runner and current scorer for all 150 questions, with exact
  behavior, structured answers, and fault counts. Semantic validation of
  all 150 resulting traces reports zero violations. The original symbol
  panel ID stays `e74b2b7e72651bae168b086f316d7491d68aeaa152a51bafa04038088a379f96`.
  Fresh scoring of all 50 saved S1.2.11 predictions per version reproduces
  every prior per-row score, including v4 14/50 and v5 40/50. These checks
  establish harness correctness, not learned retry accuracy.
- **S1.2.11 tokenizer and semantic checks.** The actual uniform-SFT
  tokenizer verifies all 800 revised rows and full supervision of all
  1,200 post-tool thoughts: 800 final thoughts plus 400 retry thoughts.
  Lengths range from 599 to 761 tokens, below 4,096. Corpus semantic
  validation passes all 13,750 S1.2 and 12,500 S1.1 rows. Validator
  selftest rejects all 53 known-bad fixtures with zero false positives.
- **Fifteenth-increment test results.** All 379 focused tests pass:
  133 S1.2 corpus checks, 119 S1.1/shared corpus and overlay checks,
  37 symbol/ID retry checks, and 90 native generation and evaluator
  checks. One historical v4 checkpoint check is skipped because its
  training-state file is absent. The suites reproduce no-overlay,
  v4, and v5 row bytes, deterministic v6 rebuilds, and both diagnostic
  panels. Retry corruption tests reject absent or zero allowance,
  changed arguments or tools, third calls, wrong mappings, premature
  deferral, and unsupported policy changes. The aggregate v6 bundle
  remains unbuilt; no training or model evaluation jobs are submitted.

- **S1.2.12 review.** Keep v5 unchanged: both versions score 125/125
  on the same held-out transcript-ID questions, current scorer, and keys.
- **S1.3.1 results.** On the same 125 held-out questions, current scorer,
  and v5 keys, v4 scores 125/125 (100%) and v5 scores 120/125 (96%).
  Both initial symbol lookups are correct on all 125 rows. V5 completes
  the correct tool sequence on 124, returns the correct candidate result
  on 121, and chooses the correct continuation on 122. One UBBP4 trace
  invents MXL-m000001, checks its first candidate, and attempts a third
  call before hitting the tool limit. GTF2IP1, POLR2J3, and another UBBP4
  trace invent choose-first or lowest-ID conventions and add a singular
  gene_id. TDH incorrectly stops without a blocked reason; POLR2J3 also
  continues. These five errors do not establish wrapper causation.
- **S1.3.1 approved edit.** Retain all 1,000 examples, their direct
  framing, case variants, source order, and V5 evidence nonces. Restore
  the frozen v4 identity_s1_3 direct bank at the existing indices:
  exactly 155 user messages change and 845 stay identical. All 32
  wrappers retain 31 or 32 uses. Replace all 1,000 final thoughts with:
  "{symbol} matches {count} candidates; the user supplied no choice or
  multiplex to disambiguate them. Return ambiguous/defer with all
  candidates and details, registry evidence, and a blocked reason; omit
  gene_id because order alone cannot resolve ambiguity." Bind the
  canonical symbol and candidate count from the lookup output.
- **S1.3.1 preserved behavior.** Keep pre-call thinking, one symbol
  lookup, exact arguments, full tool output, and final answer unchanged.
  Return exactly gene_symbol, candidate_gene_ids, and candidate_details,
  retaining their order, registry evidence, null graph provenance,
  ambiguous/defer, and the existing nonempty blocked reason. Preserve
  all other 8,000 S1.3 training rows, both complete 1,125-row held-out
  files (including 125 form-1 rows each), all S1.1/S1.2 rows, both retry
  diagnostic questions and keys, and historical artifacts. Refresh
  only the existing components' and diagnostics' manifest overlay pins.
  No new retry policy or thinking-filter exception is introduced.
- **S1.3.1 build scope.** Add only integer form 1's train_prompt_banks
  and train_final_reasoning under S1.3 in overlay schema v5. Enforce
  the existing two-sentence, 25-word-per-sentence, 40-word-total limits.
  The wrapper audit checks each row against its configured bank; the
  classifier's gates and held-out templates stay unchanged. Rebuild
  S1.3 with the component command above, refresh S1.1/S1.2, and rebuild
  both retry diagnostic manifests with their existing commands. Later
  S1.3 types await review; no aggregate bundle or jobs are authorized.

- **Sixteenth-increment verification (S1.3.1).** Independent full-row
  comparison confirms exactly 155 changed user messages and 1,000 changed
  final thoughts. Replacing those fields in V5 reproduces every new row
  byte, including candidate ordering and evidence nonces. All other
  8,000 training rows and both complete held-out files are byte-identical.
  Every target keeps one lookup and ambiguous/defer; no singular gene_id
  is added. All 81 protected artifacts retain their hashes. S1.1/S1.2
  component and both retry diagnostic manifests change only overlay pins;
  all of their row files and both diagnostic panel IDs stay fixed.
  The S1.3 train SHA-256 is
  `cd89e66a779fb6c2b23052908cac0fbf0a2ca0044f5137b89612c5425a1853e3`.
  The overlay SHA-256 is
  `be3d898369575f67f09ffae52a52599d43f5d68aa6396721a4ea414f2a3503fe`.
- **S1.3.1 tokenizer and semantic checks.** The actual uniform-SFT
  tokenizer checks all 1,000 revised rows, fully supervising every new
  final thought. Lengths range from 759 to 2,758 tokens, below 4,096.
  All three component builds pass semantic validation. S1.3's thinking,
  duplicate, false-defer, and masked-wording audits pass without relaxing
  their gates. Validator selftest rejects all 53 known-bad fixtures with
  zero false positives. No aggregate v6 bundle or model jobs are produced.
- **Sixteenth-increment regression results.** All 102 S1.3 corpus,
  prompt-bank, and overlay checks pass, including deterministic v6
  rebuilds and byte-for-byte reproduction of the no-overlay, v4, and v5
  S1.3 splits. All 60 S1.1 v6 regressions also pass. Invalid per-form
  overrides, wrong bank hashes, missing or repeated symbol/count slots,
  unsupported slots, and overlong reasoning remain rejected.

- **S1.3.4 review (2026-10-19).** Fresh rescoring of the same 125 held-out
  questions with the current scorer and v5 keys gives v4 125/125 (100%)
  and v5 116/125 (92.8%) structured accuracy. Direct requests fall from
  62/62 to 57/62; workflow requests fall from 63/63 to 59/63. V5 uses the
  correct tool sequence and arguments on 123/125, returns the correct
  result on 120/125, and chooses the correct continuation on 117/125.
  All failures follow the initial lookup of the requested symbol.
- **S1.3.4 failures.** GTF2IP1, CDIPTOSP, and POLR1HASP retain candidates
  but continue without a blocked reason. FAM197Y4 and ADAM3A invent
  multiplex IDs and membership checks. FAM197Y4 first checks the unrelated
  harness candidates, then hits the tool-call limit. ADAM3A also continues
  and adds graph provenance. PDPR2P invents candidate_gene_id and
  canonical_gene_id. LRRC77P and COL6A4P1 loop without a final answer.
  DDX11L16 leaves its summary string unterminated, invalidating the JSON
  envelope. These nine failures do not establish wrapper causation.
- **S1.3.4 approved wording.** Keep all 1,000 examples, exact inputs,
  distractors, case variants, order, and 500 direct/500 workflow assignments.
  Restore the frozen v4 identity_s1_3 bank at the existing indices.
  Exactly 576 user messages change: 76 direct and all 500 workflow.
  The other 424 messages already match. Each framing keeps 32 wrappers,
  used 15 or 16 times each. Do not copy entire v4 rows or reassign cases.
- **S1.3.4 approved reasoning.** Replace all 1,000 final thoughts with:
  "The lookup returns {count} candidates for {symbol}; unrelated harness
  entries supply no choice or multiplex. Keep only gene_symbol,
  candidate_gene_ids, and candidate_details in result; preserve lookup
  evidence and return ambiguous/defer with a blocked reason."
  Bind both slots to the actual lookup output. Use this same template for
  both framings: workflow wording cannot resolve an ambiguous identity.
  Form 4 of S1.3's pinned train_final_reasoning owns the template.
  Keep two sentences, at most 25 words per sentence and 40 words total;
  no thinking-filter exception is needed.
- **S1.3.4 preserved scope.** Keep harness blocks, pre-call thinking, one
  symbol lookup, exact arguments, tool responses, final answers, evidence
  IDs, candidate ordering, metadata, and row ordering. Targets remain
  ambiguous/defer with exactly gene_symbol, candidate_gene_ids, and
  candidate_details, lookup registry evidence, null graph provenance, and
  the existing blocked reason. Keep all other 8,000 S1.3 training rows,
  including reviewed S1.3.1, and both complete 1,125-row held-out files.
  S1.3.2 and S1.3.3 remain unchanged and are not marked reviewed here.
  Preserve all S1.1/S1.2 rows, retry diagnostic questions, keys, and panel
  IDs, and historical artifacts. Refresh only their manifest overlay pins.
  Rebuild S1.3; the aggregate v6 bundle and model jobs remain deferred.
- **Seventeenth-increment verification (S1.3.4).** Independent comparison
  against the preceding v6 component confirms exactly 576 changed user
  messages and 1,000 changed final thoughts. The user changes comprise
  76 direct and 500 workflow messages. Replacing those fields reproduces
  every new row byte. All other 8,000 training rows are byte-identical,
  including S1.3.1. Both complete held-out files retain their hashes.
  All 227 protected artifacts retain their hashes, including historical
  corpora, evaluator questions and keys, frozen banks and overlays,
  training configs, the shared contract, S1.1/S1.2 rows, and retry panels.
  The S1.1/S1.2 and both retry diagnostic manifests change only their
  overlay pins; both diagnostic panel IDs stay fixed.
  The S1.3 train SHA-256 is
  `bce1cc18a4122a46e73da727dcba06086c90926b6a55596314672cfd61e09c61`.
  The overlay SHA-256 is
  `a1a6335e8ce279a6c09433a6270c57fe233508c0247c095889b6cc1541636bd5`.
- **S1.3.4 tokenizer and semantic checks.** The actual uniform-SFT
  tokenizer checks all 1,000 revised rows and fully supervises every new
  final thought. Lengths range from 900 to 2,913 tokens, below 4,096.
  The S1.3 build passes semantic validation on all 9,000 training and
  both 1,125-row held-out splits. Thinking, duplicate, false-defer, and
  masked-wording audits pass with their existing gates. Validator
  selftest rejects all 53 known-bad fixtures with zero false positives.
  The aggregate v6 bundle remains unbuilt; no model jobs are submitted.
- **Seventeenth-increment regression results.** All 302 checks pass across
  the S1.3 corpus and v6 suites, shared overlay and prompt-bank suites,
  and S1.1/S1.2 v6 regressions. They verify deterministic rebuilding,
  byte-for-byte reproduction of historical no-overlay/v4/v5 corpora,
  exact approved row changes, frozen held-out splits, and rejection of
  unreviewed forms, invalid bank pins, missing or repeated bindings,
  unsupported slots, and overlong reasoning. Run from the repository root:

```bash
python -m pytest -q tests/test_world_model_s1_3_v6_corpus.py \
  tests/test_world_model_s1_3_corpus.py tests/test_world_model_s1_overlay.py \
  tests/test_world_model_s1_prompt_banks.py tests/test_world_model_s1_1_v6_corpus.py \
  tests/test_world_model_s1_2_v6_corpus.py
```

- **S1.3.5 review (2026-10-19).** Fresh scoring of the same 259 held-out
  questions with the current scorer and v5 keys gives v4 16/259 (6.18%)
  and v5 74/259 (28.57%) structured accuracy. Correct tool sequences fall
  from 223/259 to 153/259; exact arguments fall from 223/259 to 151/259.
  V4 often omits fields required by the approved v5 answer schema, so keep
  that schema. These results do not establish wrapper causation.
- **S1.3.5 failures.** Partition the 185 v5 failures by their first relevant
  failure: 106 never reach membership (including three malformed turns),
  48 use the correct calls but still label a sole survivor ambiguous,
  23 use incorrect result fields, two use incorrect arguments, and six
  have only the wrong continuation decision. Skipped membership affects
  44 harness-named and 62 task-named multiplexes. CDIPTOSP checks both
  candidates, sees exactly one present, and still defers. PDPR2P replaces
  the full candidate evidence with present_in_multiplex:true. One CSN1S2AP
  row checks only the first candidate, inventing an ordering convention.
- **S1.3.5 approved wording and reasoning.** Keep all 2,070 examples and
  the 1,035 direct/1,035 workflow assignments. Restore the v4 identity_s1_3
  bank at the current indices: 1,195 user messages change (160 direct and
  1,035 workflow). Each framing keeps 32 wrappers, used 32 to 34 times.
  Multiplex placement stays balanced: direct has 518 harness/517 task;
  workflow has 517 harness/518 task. Replace both post-tool thoughts with
  the pinned train_single_membership_reasoning templates:
  - After lookup: "The lookup returns {count} candidates for {symbol}, and
    the {source} names {multiplex_id}. Check all candidate_gene_ids in that
    multiplex before selecting an ID." Bind source to harness or task from
    the actual prompt, and symbol/count to the lookup output.
  - Direct conclusion: "Only {gene_id} is present in {multiplex_id}, resolving
    this request; the user requested only this result, so stop. Return
    resolved with gene_id, gene_symbol, candidate_gene_ids, candidate_details,
    multiplex_id, present_gene_ids, absent_gene_ids, both evidence channels,
    and no blocked reason."
  - Workflow conclusion: "Only {gene_id} is present in {multiplex_id}, resolving
    this request; the user needs this result for a later workflow step, so
    continue. Return resolved with gene_id, gene_symbol, candidate_gene_ids,
    candidate_details, multiplex_id, present_gene_ids, absent_gene_ids, both
    evidence channels, and no blocked reason."
  Bind the final ID to the sole present_gene_ids entry and the multiplex
  ID to the membership output. Preserve the complete v5 seven-field result.
- **Reasoning length policy (2026-10-19).** The user removed hard sentence
  and word limits for training reasoning, allowing longer explanations
  when needed. Remove the shared identity-template caps, the special
  S1.2.5 allowance, and the bank reasoning word cap. Retry templates use
  the same uncapped validator. Retain formatting, required bindings,
  evidence visibility, semantic and status/decision checks, and actual
  tokenizer/context-budget validation. Wrapper limits remain unchanged.
  Earlier length rules in this document record historical checks and do
  not constrain future reasoning reviews. Existing reviewed traces stay fixed.
- **S1.3.5 preserved scope.** Keep initial thinking, inputs, calls, tool
  responses, candidate order, final answers, evidence IDs, metadata, and
  ordering unchanged. Preserve all other 6,930 S1.3 training rows and
  both full 1,125-row held-out splits, including 259 form-5 rows each.
  Keep all 600 S1.3.10 failure rows. Permit only the pinned wrapper
  difference in 265 success/failure pairs (35 direct and 230 workflow),
  checking both original wrappers and all remaining context and assignments.
  The original duplicate audit still rejects duplicate prompt/outcome rows.
  The membership cue audit validates the exact pinned form-5 user message
  before normalizing only that wrapper to the shared bank for comparison.
  Normalized training cues retain 128 shared contexts for all three outcomes;
  each held-out split retains 16. Report raw training cues separately:
  keeps_one has 165, keeps_several and keeps_none have 128, and 91 are shared.
  These differences reflect the partial review's mixed wrapper banks; they
  are not a raw cue gate pass. Reconcile raw wrapper coverage as remaining
  forms are reviewed before assembling the training bundle. The existing
  held-out classifier gate passes unchanged: 232/1,125 validation and
  208/1,125 test predictions, below the 25.02% threshold.
  Preserve all S1.1/S1.2 rows, retry diagnostic questions/keys/panel IDs,
  frozen promotion data, and historical corpora, banks, overlays, and configs.
  Refresh only existing S1.1/S1.2 and retry diagnostic overlay manifest pins.
  No aggregate bundle or model jobs are part of this increment.

- **Eighteenth-increment verification (S1.3.5).** Comparison against the
  preceding v6 component confirms exactly 1,195 changed user messages and
  4,140 changed thoughts across 2,070 rows. Replacing just those fields
  reproduces every new row byte. The other 6,930 training rows, both full
  held-out splits, and all 227 protected artifacts retain their bytes.
  This includes historical corpora, frozen banks/overlays/configs, the
  shared contract, promotion panels, S1.1/S1.2 rows, and retry diagnostics.
  The four existing component/diagnostic manifests change only their overlay
  pins; diagnostic panel IDs remain fixed. S1.3 train SHA-256 is
  `c3d577dee29326c129a9612ad45605f908ab8bb70a0fbfcfa397a0906d12fc2c`.
  Overlay SHA-256 is
  `007cc13f0d929d04cf450177d58fae08acf0f50bc7741848766f2769839e1487`.
- **S1.3.5 semantic and tokenizer checks.** The build validates all 9,000
  training and both 1,125-row held-out splits. Thinking, duplicate, false-defer,
  normalized membership-context, and held-out wording checks pass. The actual
  uniform-SFT tokenizer checks all 2,070 revised rows at 1,106 to 1,760 tokens
  and fully supervises all 6,210 thoughts, including the 4,140 replacements.
  All rows fit the unchanged 4,096-token context budget. Validator selftest
  rejects all 53 known-bad fixtures with zero false positives. The updated
  S1.3 v6 suite passes 72 tests, including deterministic rebuilding,
  no-overlay/v4/v5 split reproduction, frozen held-out bytes, exact changed
  fields, uncapped templates, binding rejection, and restricted wrapper
  normalization. The raw training wrapper difference is reported above;
  the aggregate v6 bundle remains unbuilt and no model jobs were submitted.

- **Eighteenth-increment regression results.** All 333 collected cases
  pass after targeted reruns of the corrected S1.3 membership-context audit
  and the policy-isolation expectation for the newly approved reasoning key.
  Coverage includes the S1.3 corpus and v6 suites, shared overlay and bank
  validation, S1.1/S1.2 v6 regressions, and both lookup-retry diagnostic paths.
  The broad run passed 326 cases; the final S1.3 rerun passed all 72 cases
  (including the added cue-normalization test), and the corrected policy
  check passed separately. No regression failures remain. Reproduce with:

```bash
python -m pytest -q tests/test_world_model_s1_3_v6_corpus.py \
  tests/test_world_model_s1_3_corpus.py tests/test_world_model_s1_overlay.py \
  tests/test_world_model_s1_prompt_banks.py tests/test_world_model_s1_1_v6_corpus.py \
  tests/test_world_model_s1_2_v6_corpus.py tests/test_world_model_s1_lookup_retry.py
```

- **S1.3.6 review (2026-10-19).** Fresh scoring of the same 27 held-out
  questions with the current scorer and v5 keys gives v4 0/27 and v5 8/27
  (29.63%) structured accuracy. Correct tool sequences and arguments fall
  from 20/27 to 17/27. V5 result accuracy is 9/27, status 25/27, and
  continuation decision 24/27. The panel has one identity, POLR2J3, with
  two candidates both present. Broader evaluation coverage is deferred;
  see Future evaluation work. These results do not establish wrapper causation.
- **S1.3.6 failures.** Of the 19 v5 failures, ten skip membership, six
  make the correct calls but omit multiplex_id, present_gene_ids, and
  absent_gene_ids, two emit invalid final answers, and one returns the
  complete result but incorrectly continues without a blocked reason.
  All six missing-field cases name the multiplex in the harness. In v4,
  19 correct-call rows omit absent_gene_ids, one has an invalid final
  answer, and seven skip membership. Keep the current answer schema.
- **S1.3.6 approved wording and reasoning.** Keep all 218 examples and
  their 109 direct/109 workflow assignments. Restore 121 user messages
  with the frozen v4 identity_s1_3 bank at the current indices: 12 direct
  and 109 workflow. Keep 32 wrappers per framing, each used two to four
  times. Replace both post-tool thoughts with train_multiple_membership_reasoning:
  - After lookup: "The lookup returns {count} candidates for {symbol}, and
    the {source} names {multiplex_id}. Check all candidate_gene_ids in that
    multiplex before selecting an ID."
  - Both conclusions: "The membership check leaves {present_count} of {count}
    candidates for {symbol} in {multiplex_id}, so it does not identify one
    gene. Return ambiguous/defer with a blocked reason asking the user to
    choose one of the present IDs. Keep gene_symbol, candidate_gene_ids,
    candidate_details, multiplex_id, present_gene_ids, and absent_gene_ids
    exactly as returned, retaining absent_gene_ids even when empty and
    preserving both registry and graph evidence. Omit a singular gene_id,
    and wait for that choice even when a later workflow step needs the result."
  Bind symbol/count to the lookup, source to the actual harness/task cue,
  and final present_count/multiplex_id to membership. Keep all six result
  fields, both evidence channels, and the existing ambiguous/defer targets.
- **S1.3.6 preserved scope.** Keep pre-call thinking, input spelling, calls,
  responses, candidate order, final answers, evidence IDs, metadata, and row
  order unchanged. Preserve the other 8,782 training rows, both complete
  held-out splits, historical artifacts, S1.1/S1.2 rows, and retry diagnostic
  questions/keys/panel IDs. The existing scoped wrapper normalization now
  also validates form 6; 25 of its 48 success/failure pairs change user
  wording, while all 600 failure rows stay frozen. Continue reporting raw
  membership cues separately until forms 7 and 10 are reviewed. Refresh
  only the S1.1/S1.2 and retry diagnostic manifest overlay pins. No aggregate
  bundle or model jobs are part of this increment.

- **Nineteenth-increment verification (S1.3.6).** Comparison against the
  preceding v6 component confirms exactly 121 changed user messages and
  436 changed thoughts across 218 rows. Replacing only these fields
  reproduces every new row byte. The other 8,782 training rows, both full
  held-out splits, and all 227 protected artifacts retain their bytes.
  The four existing S1.1/S1.2 and retry diagnostic manifests change only
  overlay pins; diagnostic panel IDs stay fixed. S1.3 train SHA-256 is
  `42d0b12e551590a6edc8148f934f1a4e20971ea761d5cf7390791eecb198aad8`.
  Overlay SHA-256 is
  `4d3d295f54400c216f5d284aa89fdb2bce412fbd979fb68adfdaa7251fef0c16`.
- **S1.3.6 semantic and tokenizer checks.** The build validates all 9,000
  training rows and both 1,125-row held-out splits. Thinking, duplicate,
  false-defer, normalized membership-context, and held-out wording gates
  pass. Raw training cues are 165 for keeps_one, 153 for keeps_several,
  and 128 for keeps_none, with 79 shared; this is not a raw cue gate pass.
  The normalized comparison has 128 shared cues per training outcome and
  16 per held-out outcome. The actual uniform-SFT tokenizer checks all
  218 revised rows at 1,217 to 4,025 tokens and fully supervises all 654
  thoughts, including the 436 replacements, within the 4,096-token budget.
  Validator selftest rejects all 53 known-bad fixtures with zero false
  positives. The aggregate bundle remains unbuilt; no model jobs were submitted.

- **Nineteenth-increment regression results.** All 382 originally collected
  cases have passing results after targeted reruns. The broad run passed
  350 cases; the corrected raw-cue-count expectation passed separately.
  Concurrent S1.3.7 edits changed the overlay after that process loaded
  the S1.3.6 schema, causing 31 setup errors in S1.2 and retry diagnostics.
  A consistent private copy of the updated code passed all 42 targeted
  checks, including those 31 cases and S1.2 reproducibility. Historical
  no-overlay/v4/v5 split checks passed in the original run. The concurrent
  update preserves every revised S1.3.6 row and all 227 protected files.
  These checks cover the S1.3.6 increment; they do not substitute for the
  additional S1.3.7 regression cases. Verification logs and the preserved
  S1.3.6 component are under `/var/tmp/s1_3_6_vtcd2zwl/`.

- **S1.3.7 review (2026-10-19).** Fresh scoring of the same 52 held-out
  questions with the current scorer and v5 keys gives v4 28/52 (53.85%)
  and v5 10/52 (19.23%) structured accuracy. Correct tool sequences and
  arguments fall from 40/52 to 24/52; exact results fall from 30/52 to
  10/52. The test panel covers nine identities with two or five candidates.
  These results do not establish that wrapper changes caused the regression.
- **S1.3.7 failures.** Of the 42 v5 failures, 28 skip membership: 17 with
  the multiplex in the task and 11 in the harness. Eight make both calls
  correctly but omit multiplex_id, present_gene_ids, and absent_gene_ids;
  all eight name the multiplex in the harness. Six emit an invalid final
  answer with stop and a blocked reason. All six retain the full result,
  and one also says resolved. V4 has 12 skipped checks, 11 wrong-status
  answers, and one invalid final answer. Keep the v5 ambiguous/defer target.
- **S1.3.7 approved wording and reasoning.** Keep all 412 examples and
  their 206 direct/206 workflow assignments. Restore 236 user messages
  with the frozen v4 identity_s1_3 bank at the existing template indices:
  30 direct and 206 workflow. Each framing retains 32 wrappers, used six
  to eight times, and 103 harness/103 task multiplex placements. Replace
  both post-tool thoughts using train_no_membership_reasoning:
  - After lookup: "The lookup returns {count} candidates for {symbol}, and
    the {source} names {multiplex_id}. Check all candidate_gene_ids in that
    multiplex before selecting an ID."
  - Both conclusions: "The membership check finds no present candidates for
    {symbol} in {multiplex_id}; all {count} registry candidates are absent
    from this multiplex. The lookup remains ambiguous, and choosing one of
    these IDs cannot supply a usable node for the requested multiplex.
    Return ambiguous/defer with a blocked reason explaining that another
    gene or multiplex is needed. Keep exactly gene_symbol, candidate_gene_ids,
    candidate_details, multiplex_id, present_gene_ids, and absent_gene_ids,
    including the empty present_gene_ids list and every returned absent ID.
    Preserve both registry and graph evidence and omit a singular gene_id.
    Both direct and workflow requests remain blocked, so defer until the
    request changes."
  Bind symbol/count to the lookup, source to the actual harness/task cue,
  and the final multiplex ID to membership. Preserve the six-field result,
  both evidence channels, and all existing final answers.
- **S1.3.7 preserved scope.** Keep initial thinking, inputs, calls, tool
  responses, candidate order, evidence IDs, metadata, and row order fixed.
  Preserve all other 8,588 training rows, both complete held-out splits,
  all S1.1/S1.2 rows, retry diagnostic questions/keys/panel IDs, and historical
  artifacts. Extend pinned-wrapper pair and membership-context checks only
  to form 7; 52 of its 92 failure pairs change wrapper text. All 600 S1.3.10
  failure rows stay unchanged. Check raw successful
  membership wrapper coverage across forms 5, 6, and 7; report the remaining
  mixed-bank failure-pair cues separately until S1.3.10 is reviewed. Refresh
  only S1.1/S1.2 and retry diagnostic manifest overlay pins. The aggregate
  v6 bundle and model jobs remain deferred.

- **Twentieth-increment row and tokenizer verification (S1.3.7).** The
  rebuilt component changes exactly 236 user messages and 824 thoughts
  across 412 rows. Replacing only these fields reproduces every new row
  byte. All other 8,588 training rows and both full held-out splits retain
  their bytes. Training SHA-256 is
  `cfcad4eb7d4d661107071946680e9eb763859ac4e3179b25c620d85e39c08e48`.
  Semantic validation passes for 9,000 training and two 1,125-row held-out
  splits. The actual uniform-SFT tokenizer checks all 412 revised rows at
  1,193 to 1,608 tokens within the 4,096-token budget. All 1,236 thoughts,
  including the 824 replacements, are present and fully supervised.
  Validator selftest rejects all 53 known-bad fixtures with zero false positives.
- **S1.3.7 artifact verification.** The published component matches the
  independently checked build. All 214 protected artifacts retain their
  hashes, including historical corpora, banks, overlays and run configs,
  the shared contract, frozen promotion panels, S1.1/S1.2 rows, and retry
  diagnostic questions/keys. The four S1.1/S1.2 and retry diagnostic
  manifests change only their overlay hash; diagnostic panel IDs stay fixed.
  The S1.3 manifest pins the revised builder and approved templates.
  Overlay SHA-256 is
  `9b6022faccaec93f08ce23a033572fdef02242fc727e60d79d85ebc60599bd51`.
  The aggregate v6 bundle remains unbuilt and no model jobs were submitted.
- **S1.3.7 wording coverage.** The raw successful membership outcomes now
  share exactly 128 training contexts and 16 per held-out split. The audit
  enforces this when all three successful forms use the same pinned bank.
  Including the unchanged form-10 failures, raw training cue counts remain
  165 for keeps_one, 153 for keeps_several, and 165 for keeps_none, with 153
  shared. These mixed-bank failure cues still await S1.3.10 review; they
  are not a combined raw cue gate pass. Normalized comparisons retain 128
  shared training contexts and 16 held-out contexts. The existing held-out
  wording classifier passes at 203/1,125 validation and 136/1,125 test,
  below its unchanged 25.02% threshold. No thinking-filter exception is used.

- **S1.3.7 regression verification.** All 427 collected cases pass after
  correcting one test expectation and rerunning that case. The full run
  passes 426 cases; the row-delta test initially expected wrapper use counts
  of six, seven, and eight, but the preserved assignments use six or eight.
  Its corrected rerun passes. The seven suites cover S1.3 corpus building
  and v6 edits, shared overlays and prompt banks, S1.1/S1.2 v6 regressions,
  and lookup-retry diagnostics. They verify deterministic repeated builds,
  no-overlay/v4/v5 split reproduction, frozen held-out bytes, exact approved
  row changes, evidence-bound templates, and restricted wrapper normalization.
  No regression failures remain. Use the seven-suite command recorded in
  Eighteenth-increment regression results above.

- **S1.3.8 review (2026-10-19).** Fresh scoring of the same 125 frozen
  questions with current v5 keys gives v4 structured accuracy 78/125
  (62.4%) and v5 94/125 (75.2%). Both models correctly make no new calls
  on all 125 rows. Continuation accuracy is 125/125 for v4 and 123/125
  for v5. Summary and blocked-reason prose are excluded from the headline.
  V4's 47 failures concern the result schema: 42 omit the candidate list
  and five add candidate_details. Every failed v4 row selects the right ID.
- **S1.3.8 v5 failures.** The 31 failures comprise 18 extra-field results,
  nine correct selected IDs under candidate_gene_id or chosen_gene_id
  instead of gene_id, two missing candidate lists replaced by details,
  one renamed symbol with a null candidate list, and one invalid final
  envelope using continuation_decision:complete. One extra-field result
  also incorrectly continues a standalone request. These failures support
  clearer result construction and an explicit task-scope explanation.
  Retain the v5 prompts; this review does not establish wrapper causation.
- **S1.3.8 approved reasoning.** Keep all 1,000 training examples, with
  500 direct and 500 workflow cases. Replace only their final thoughts
  using train_user_choice_reasoning. The shared approved body is:

  > The reply selects {gene_id}, which appears in the earlier candidate_gene_ids list for {symbol}, so the user has resolved the choice. Build the result with exactly three fields: gene_id for the selected ID, gene_symbol for the earlier canonical symbol, and candidate_gene_ids for the entire earlier list in its original order. Keep the full list after selection; candidate_details and other fields do not belong in this resolved result. Preserve the earlier registry evidence and null graph provenance, and return resolved with no blocked reason. No new lookup or membership check is needed.

  Direct ending:

  > This completes the user's standalone gene-ID request, so stop.

  Workflow ending:

  > The user's later workflow step needs this selected mapping, so continue.

- **S1.3.8 preserved scope.** Keep both user turns, the masked earlier
  answer, final answers, full candidate order, evidence IDs, metadata, and
  ordering. All other 8,000 S1.3 training rows and both complete held-out
  splits remain frozen, as do S1.1/S1.2 rows, retry diagnostic questions
  and keys, shared contract, and historical artifacts. Refresh only the
  other v6 component and retry diagnostic manifest overlay pins. The
  reference example chooses by attribute ("Use the lncRNA"), while the
  current training and frozen test replies explicitly name IDs. Attribute
  selection needs a separate future diagnostic; these scores do not
  measure that capability. Keep it outside this reasoning-only increment.
- **S1.3.8 thinking filter.** The approved text contains "null graph
  provenance", which the old schema-word filter rejects. Permit only
  that phrase in the exact configured final thought of form-8 training
  rows. Keep the original filter for changed thoughts, other schema
  phrases, other forms, and held-out rows. Identifier visibility and
  status/decision agreement checks remain active. Regression mutations
  cover those boundaries in both framings.
- **Twenty-first-increment verification (S1.3.8).** The rebuilt component
  changes exactly 1,000 final thoughts: 500 direct and 500 workflow.
  Replacing only that field reproduces every new row byte. All other
  8,000 training rows and both full 1,125-row held-out splits retain their
  bytes. Both user turns, earlier answers, final answers, evidence, and
  metadata are unchanged. Training SHA-256 is
  `199e6885f694b07ecb5bf57fd27fcb455b3334dcd60ef2b52fdd1a0b980914e9`.
  Semantic validation passes for all 11,250 rows. The thinking audit
  checks 18,600 training thoughts and 2,250 per held-out split, with zero
  violations. Wording and duplicate audits pass their existing gates.
  The actual uniform-SFT tokenizer checks all 1,000 revised rows at
  780 to 1,946 tokens within the 4,096-token budget. Every revised thought
  is fully supervised, and every earlier answer remains masked.
  Validator selftest rejects all 53 known-bad fixtures with zero false positives.
- **S1.3.8 artifact and regression verification.** The published component
  matches the independently checked build. All 214 protected artifacts
  retain their hashes, including frozen evaluation panels, historical
  corpora and configurations, the shared contract, S1.1/S1.2 rows, and
  retry diagnostic questions and keys. The four dependent manifests
  change only their overlay hash. The S1.3 manifest changes only its
  overlay and builder pins, training-file hash, and new reasoning entry;
  allocations, evidence, tool counts, and wording audits are unchanged.
  Overlay SHA-256 is
  `147677789976e46680b1f00cd15deab67fb13c8c246bec62f097e79081a1ebbe`.
  All 265 regression cases pass. They cover exact row deltas, frozen
  held-out bytes, repeated builds, no-overlay/v4/v5 reproduction, template
  bindings, the narrow thinking-filter exception, shared prompt banks,
  and unchanged policies outside the reviewed forms. Verification logs,
  review scores, and the preceding v6 component are preserved under
  `/tmp/s1_3_8_change_pe4xmcba/`. The aggregate v6 bundle remains unbuilt;
  no model jobs were submitted.

```bash
python -m pytest -q tests/test_world_model_s1_3_v6_corpus.py \
  tests/test_world_model_s1_3_corpus.py tests/test_world_model_s1_overlay.py \
  tests/test_world_model_s1_prompt_banks.py \
  tests/test_world_model_s1_1_v6_corpus.py::test_unreviewed_policies_and_default_bank_remain_v5
```

- **S1.3.9 review (2026-10-19).** Both v4 and v5 score 125/125 on the
  frozen test questions with the current scorer and v5 keys. Both preserve
  ambiguous/defer, candidates and details, registry evidence, a blocked
  reason, and no new calls. Keep all 850 v5 training examples unchanged,
  with 425 direct and 425 workflow cases. Current v6 form-9 bytes match v5.
  These rows cover four generic nonselecting replies, rather than the
  reference example's unmatched biotype; broader attribute-choice coverage
  remains separate from this review.
- **S1.3.10 review (2026-10-19).** Current scoring of the same 37 frozen
  questions gives v4 structured accuracy 17/37 (45.9%) and v5 3/37 (8.1%).
  Correct tool sequences fall from 34/37 to 19/37. V5's 34 structured
  failures comprise 18 skipped membership calls, 11 results dropping
  multiplex_id/failed_tool/error_code after the check fails, two missing
  failed_tool, one error answer discarding the successful registry lookup,
  and two invalid final responses. Nine skipped checks have a harness
  multiplex and nine have a task multiplex. One of the three correct
  structured answers also uppercases the lookup input, so exact tool
  arguments score 18/37. Keep literal-input targets. Scores are preserved
  in `/tmp/s1_3_9_10_review.json`; summary prose is excluded.
- **S1.3.10 implementation (2026-10-19).** Implemented the approved
  terminal, recovery, and exhausted-retry policy within all 600 existing
  examples. Keep the 300 direct/300 workflow assignments, identities,
  literal inputs, retained success pairs, ordering, and harness/task
  multiplex placement. Restored exactly 342 user messages from the v4
  bank at their existing indices: 42 direct and 300 workflow. The first
  lookup thought, call, output, and evidence IDs stay unchanged.
  Every first post-lookup thought now uses the reviewed S1.3.5-7 template,
  bound to symbol, candidate count, harness/task source, and multiplex.
  It states no future failure or membership outcome.
- **Retry allocation.** The source-backed counterparts give these counts:

  | Successful counterpart | No allowance | Recovery | Exhausted | Total |
  |---|---:|---:|---:|---:|
  | One present candidate (S1.3.5) | 230 | 115 | 115 | 460 |
  | Several present candidates (S1.3.6) | 24 | 12 | 12 | 48 |
  | No present candidates (S1.3.7) | 46 | 23 | 23 | 92 |
  | Total | 300 | 150 | 150 | 600 |

  The assignment hashes source identity, occurrence, framing, and placement
  in the fixed S1.3.10 namespace. It is independent of wording, template
  indices, row IDs, and iteration order. All 36 outcome/case/framing/placement
  combinations remain covered. The assignment hash is
  `8b24b36e16fa7fb3dbfa5a817ea284984413074297f59c366f7dcffb5574694d`.
- **Allowance and calls.** No allowance means one plain graph failure and
  immediate deferral after the successful lookup: two calls total. The
  other 300 rows receive `graph service unreachable; retries_remaining=1`
  in the first membership error. They repeat check_multiplex_membership
  exactly once with identical multiplex_id and the entire gene_ids list
  in its original order. They never repeat the lookup. The retry succeeds
  from the pinned source or fails with
  `graph service unreachable; retries_remaining=0`. Both branches have
  three total calls. The approved retry thought preserves the registry
  lookup and names the one remaining attempt.
- **Terminal and exhausted answers.** Both branches retain ambiguous/defer,
  all original candidates/details, successful registry evidence, null graph
  evidence, and a nonempty text blocked reason. Their six result fields are
  gene_symbol, candidate_gene_ids, candidate_details, multiplex_id,
  failed_tool, and error_code. They omit singular gene_id and membership
  lists because a transport failure establishes no membership outcome.
  Final reasoning distinguishes no allowance from exhaustion and names the
  required fields. The summary ends with "Graph access must be restored to
  continue." It no longer requests an unqualified retry.
- **Recovery answers.** Recovery uses the actual membership result and
  the reviewed S1.3.5-7 final reasoning. Original registry evidence remains;
  only the successful retry supplies new graph evidence. The 115 unique
  outcomes return resolved with all seven fields, no blocked reason, and
  stop for direct requests or continue for workflow requests. The 12
  multiple outcomes retain six fields, ambiguous/defer, and ask for one
  present ID. The 23 empty outcomes retain six fields, ambiguous/defer,
  and ask for another gene or multiplex. No recovered answer retains stale
  failed_tool/error_code fields. Every row retains form 10 and its failure
  history; blocked-reason metadata reflects the final outcome.
- **Policy and evidence.** The overlay pins
  `runtime/world_model/s1_v6_membership_retries.json`; its SHA-256 is
  `ce83f0e1338d2c1241f7342c3b7ac12026006b492dc7780cd5766cf7e1199c1e`.
  The membership helper validates budgets, exact arguments, candidate
  partitions, final schemas, evidence, and decisions. New graph blocks are
  allocated after the complete original evidence sequence. All old evidence
  entries remain unchanged; exactly 150 graph entries are appended.
  All 600 failure prompts now match their retained success prompts. Raw
  membership context coverage is identical across source outcomes: 128
  training cues and 16 per held-out split, including forms 5-7 and 10.
- **Separate diagnostic.** Built 111 rows at
  `data/world_model_v4/eval/s1_v6/membership_retries/` using all three fault
  schedules for the 37 frozen questions from 27 split groups. Recovery
  includes 28 unique, three multiple, and six empty membership outcomes.
  User/harness prompts stay frozen, and each schedule receives distinct
  registry evidence. All 111 complete traces fit the actual tokenizer budget
  at 1,130-1,662 tokens. The new panel ID is
  `50327e3ca3330e17ca83cc3991fd44125d7bc924f5f75dffaf2d91e4185dc1cb`.
  Build with `python scripts/build_world_model_s1_membership_retry_panel.py`.
  Report this diagnostic separately from promotion and the two lookup
  diagnostics. Its scripted pass does not establish model retry accuracy.
- **Native runner and scorer.** Only recovery/exhaustion records in the
  membership diagnostic permit three calls and four assistant turns.
  No-allowance records and older panels retain the two-call limit. New
  evidence coordinates allocate below the namespace midpoint; legacy
  coordinates remain unchanged. The harness counts the first membership
  attempt after lookup and exposes allowance 1, then 0 on exhaustion.
  Semantic validation accepts recovery only after the permitted identical
  retry, including recovered ambiguous outcomes. Scoring still excludes
  summary and blocked-reason wording from structured accuracy. Job
  configuration remains deferred until the final v6 bundle exists.
- **Twenty-second-increment verification (S1.3.10).** All 600 revised rows
  pass semantic validation and the actual SFT tokenizer at 4,096 tokens.
  Their range is 1,101-3,387 tokens, with all 2,100 assistant thoughts fully
  supervised. The fixed hash allocation places the 37-candidate U6 row in
  the no-allowance branch, preserving full evidence and approved reasoning
  within the context limit. The other 8,400 training rows and both complete
  1,125-row held-out splits remain byte-identical to the preceding v6 build.
  The train SHA-256 is
  `5bff4cd10f8b008a5f13954e1955f1ceaccd52c5a54373b4332b5bf4c0eaa7fa`.
  The overlay SHA-256 is
  `c8696c28dd1459b1f8fce20fc3e31a91413fc53811cf0de0992bc03a9aa10131`.
  Frozen v4/v5 prediction scores are unchanged: S1.3.9 is 125/125 for both;
  S1.3.10 is 17/37 for v4 and 3/37 for v5. These are baseline results.
  All 368 regression tests pass, including deterministic v6 rebuilds,
  no-overlay/v4/v5 split reproduction, all 111 scripted native trajectories,
  mixed-batch fourth-turn generation, and invalid-retry rejection. The
  semantic-validator selftest catches all 53 known-bad fixtures with zero
  false positives. Existing lookup diagnostic question/answer bytes and panel
  IDs remain unchanged. S1.1/S1.2 row files, the shared contract, historical corpora,
  promotion panels, and trained configurations stay frozen. Refresh only
  the affected v6 overlay/implementation manifest pins. Verification reports
  are in `/tmp/s1_3_10_change_5oc_u03z/`. No aggregate v6 bundle or model jobs
  are created by this increment.

- **S1.4.1 results (2026-10-19).** Rescoring frozen v4 and v5 predictions
  with the current scorer and v5 answer keys gives 93/94 (98.94%) for v4
  and 82/94 (87.23%) for v5. Standalone accuracy is 48/48 versus 44/48;
  workflow accuracy is 45/46 versus 38/46. V5 gets the lookup, evidence,
  status, and same_gene:true right on all 94 rows. Eight workflow requests
  incorrectly stop. Four standalone answers use wrong item labels, with
  one also omitting both input fields. V4's single failure treats a module
  ID as a multiplex ID, calls membership, and defers after not_found.
  All 750 S1.4.1 training rows are byte-identical between v4 and v5, so
  changes within these examples do not explain the regression.
- **Approved S1.4.1 edit.** Keep all 750 identities, with 375 direct and
  375 workflow requests and 375 examples in each mention order. Replace
  only final training thinking with the exact approved train_match_reasoning
  templates in s1_v6.yaml. Bind the returned canonical symbol once and
  gene ID twice. Explain the match, first/second order, the exact item,
  input, and gene_id fields on each side, literal input preservation,
  same_gene:true, registry evidence, no graph evidence, resolved status,
  and no blocked reason. The module context suffices for the comparison;
  make no further call. The direct ending is: "The user asked only for
  this identity comparison. It is complete, so stop." The workflow ending
  is: "The user’s later workflow step needs this identity comparison, so
  continue with the resolved result."
  Preserve prompts, pre-call thinking, calls, responses, targets, evidence,
  metadata, ordering, the other 7,250 training rows (including all 250
  failure partners), and both complete 1,000-row held-out splits.
- **Implementation.** The v6 overlay alone supplies train_match_reasoning.
  The builder applies it only to form 1 in training, using actual tool
  output and the existing framing. The audit checks the exact rendered
  template and retains the existing visibility, status, decision, schema,
  and alignment checks. No thinking-filter exception is needed. Shared
  template validation permits repeated slots only when their occurrence
  counts are explicitly declared; earlier templates retain their original
  binding counts. The component manifest records both approved templates.
- **Twenty-third-increment verification (S1.4.1).** The verified component
  is published at data/world_model_v4/corpora/s1_v6/s1_4/. Exactly 750
  final training thoughts change; all other fields, 7,250 other training
  rows, and both complete 1,000-row held-out splits match frozen v5 bytes.
  All evidence records are unchanged. All 100 focused regression tests pass,
  including deterministic v6 rebuilds, byte-for-byte no-overlay/v4/v5
  reproduction, invalid-template rejection, and retained semantic checks.
  The actual SFT tokenizer places the
  revised traces at 798-859 tokens within the 4,096-token context, with
  both assistant thoughts supervised in every row (1,500 thoughts total).
  The validator selftest passes all 53 known-bad fixtures with zero false
  positives, and contract check-doc confirms the 73-trace catalog is current. The separate
  full-source check still cannot finish because the pinned MENTOR-EV
  dendrogram is not mounted, as recorded in the first increment. Keep its
  historical pin unchanged; S1.4 builds from the verified source index.
  All 225 preexisting corpus, evaluator, configuration, and
  pinned-policy artifacts were checked: only s1_v6.yaml and six existing
  v6 manifest overlay pins change. S1.1-S1.3 row files and all three retry
  diagnostic panels remain unchanged. The train SHA-256 is
  `00f7daa0c196051544b7ed25306750e8f7b93f7a17497622fc59affd0690eba4`;
  overlay SHA-256 is
  `45b7dfd272a289bc16d224ee986464f8bbdf846ced4f066a21beb473fc819279`.
  Verification reports are in `/tmp/s1_4_1_change_om277bde/`.
  Reproduce with `python -m runtime.world_model.s1_4_corpus --overlay
  runtime/world_model/s1_v6.yaml --output-directory <new-directory>`.
  This increment creates no aggregate v6 bundle or model jobs.
- **S1.4.2 results (2026-10-19).** Rescoring both models against the same
  frozen v5 questions and current answer keys gives 119/125 (95.20%) for
  v4 and 117/125 (93.60%) for v5. Standalone accuracy is 58/63 versus
  61/63; workflow accuracy is 61/62 versus 56/62. Both models make the
  correct initial symbol lookup on all 125 questions. V4's six failures
  all add unnecessary membership calls. V5's eight structured failures
  comprise five membership checks using invented multiplex IDs, one
  malformed membership-like action with no final answer, one workflow
  answer that stops, and one reversed first/second ordering. The five
  membership checks add graph evidence; three also change status to
  not_found, including two that defer. Another v5 question has a correct
  structured answer but unnecessarily looks up the supplied seed ID.
  All 1,000 training rows are byte-identical in v4, v5, and the preceding
  v6 increment; there is no wrapper change to reverse.
- **Approved S1.4.2 edit.** Keep all 1,000 examples, with 500 direct and
  500 workflow requests and 500 examples in each mention order. Replace
  only final training thinking with the exact approved train_mismatch_reasoning
  templates in s1_v6.yaml. Bind the returned canonical symbol once and
  returned ID and supplied seed ID twice each. Different IDs resolve the
  comparison with same_gene:false. Both IDs are available, so no further
  lookup or multiplex membership check is needed. Build first/second in
  user order with exactly item/input/gene_id in each. Preserve the abstract
  and seed_gene_set item labels, literal inputs, and each side's own ID.
  Retain registry evidence, no graph evidence, resolved status, and no
  blocked reason. Use the same exact direct/workflow endings as S1.4.1.
  Preserve all other fields and the other 7,000 training rows, including
  reviewed S1.4.1 and all 250 failure partners. Both full held-out splits
  remain frozen. The builder applies the template only to training form 2;
  failure partners retain form 9 even when their source form is 2. The
  manifest records both mismatch templates. Exact template, visibility,
  status, decision, schema, and alignment checks remain active; no filter
  exception is needed.
- **Twenty-fourth-increment verification (S1.4.2).** The validated component
  is published at data/world_model_v4/corpora/s1_v6/s1_4/. Compared with
  the preceding v6 increment, exactly 1,000 final training thoughts change.
  All other fields, the other 7,000 training rows, all evidence records,
  and both complete 1,000-row held-out splits remain byte-identical.
  All 117 focused tests pass, covering deterministic v6 rebuilds,
  byte-for-byte no-overlay/v4/v5 reproduction, invalid-template rejection,
  repeated seed-ID bindings, and retained visibility and semantic checks.
  The actual uniform-SFT tokenizer checks all 1,000 revised traces at
  859-937 tokens within the 4,096-token limit. Both assistant thoughts are
  fully supervised in every row, totaling 2,000 thoughts. Validator selftest
  passes all 53 known-bad fixtures with zero false positives; check-doc
  confirms the 73-trace catalog is current. The separate full-source check
  still fails on the previously documented unmounted MENTOR-EV dendrogram.
  Keep its historical pin; the corpus builds from the verified source index.
  The artifact audit checks 208 preexisting corpus, evaluator, configuration,
  and pinned-policy files. Only the v6 overlay, S1.4 train and manifest files,
  and six earlier v6 manifest overlay pins change. S1.1-S1.3 row files and
  all three retry diagnostic panels remain unchanged. The train SHA-256 is
  `dc88b8624ce9a1fa4f18dac5ddb0ad89a48644ef51a71984e9551af60f9bb3a4`;
  overlay SHA-256 is
  `73a8b8d88ed9806bbdcba03a861de6cbf452a5ace7d3fea2e9c1110c572ff2d5`.
  Verification reports are in `/tmp/s1_4_2_change_iatjnizp/`.
  Reproduce with `python -m runtime.world_model.s1_4_corpus --overlay
  runtime/world_model/s1_v6.yaml --output-directory <new-directory>`.
  Run the regression suite with `python -m pytest -q
  tests/test_world_model_s1_4_v6_corpus.py tests/test_world_model_s1_4_corpus.py
  tests/test_world_model_s1_overlay.py tests/test_world_model_s1_prompt_banks.py
  tests/test_world_model_s1_1_v6_corpus.py::test_unreviewed_policies_and_default_bank_remain_v5`.
  This increment creates no aggregate v6 bundle or model jobs.
- **S1.4.5 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 123/125 (98.40%) for v4 and 111/125 (88.80%) for v5.
  Standalone accuracy is 61/62 versus 52/62; workflow accuracy is 62/63
  versus 59/63. Both models make the correct initial symbol lookup on all
  125 questions. V5 has seven standalone answers rejected for stop with a
  blocked reason; six also assert same_gene:false. Three workflow answers
  assert false and continue. Two answers use wrong result fields, one
  repeats the lookup and breaks evidence binding, and one emits a malformed
  ID lookup for the unresolved symbol. In all nine false comparisons,
  the model still emits the abstract gene_id as null. V4's two failures
  treat module IDs as multiplex IDs and add membership calls; one also
  asserts false and stops. All 1,000 training rows are byte-identical in
  v4, v5, and the preceding v6 increment, so no wrappers need restoration.
- **Approved S1.4.5 edit.** Keep all 1,000 examples, with 500 direct and
  500 workflow requests and 500 examples in each mention order. Replace
  only final training thinking with the exact train_unresolved_symbol_reasoning
  templates in s1_v6.yaml. Bind the literal input echoed by the not_found
  lookup and supplied module ID once each. An empty candidate list leaves
  identity unknown and cannot justify same_gene:false. Preserve exactly
  first/second/same_gene and item/input/gene_id on each side, user order,
  literal inputs, abstract/module labels, null abstract gene_id, null
  same_gene, and the module's supplied ID. Keep not_found, registry evidence,
  null graph evidence, and a nonempty blocked reason. Both framings defer
  until the user supplies a current symbol or Ensembl gene ID. No retry,
  symbol-as-ID reinterpretation, or membership call is permitted. Preserve
  all other fields, the other 7,000 training rows including reviewed
  S1.4.1-2 and all 250 failure partners, and both complete held-out splits.
  The builder changes training form 5 only. The manifest records both
  templates. Exact binding and all existing thinking checks remain active;
  the alignment check also recognizes the explicit instruction to set
  same_gene to null. This tightens the check without an exemption.
  The user deferred explicit missing-value states across S0/S1 until v6
  is complete; preserve all current null values and schemas in this increment.
- **Twenty-fifth-increment verification (S1.4.5).** The validated component
  is published at data/world_model_v4/corpora/s1_v6/s1_4/. Compared with
  the preceding v6 increment, exactly 1,000 final training thoughts change.
  The other 7,000 training rows, every other field, all evidence records,
  and both complete 1,000-row held-out splits remain byte-identical.
  All 129 focused tests pass: 126 regression checks and three separate
  no-overlay/v4/v5 reproduction checks. They cover exact approved wording,
  both framings and mention orders, deterministic v6 builds, invalid
  bindings, and retained thinking semantics. The explicit null instruction
  is checked against the final comparison instead of bypassing the filter.
  The actual uniform-SFT tokenizer checks all 1,000 revised traces at
  902-1,010 tokens within the 4,096-token limit. Both assistant thoughts
  remain fully supervised in every row, totaling 2,000 thoughts. Validator
  selftest passes all 53 known-bad fixtures with zero false positives;
  check-doc confirms the 73-trace catalog is current. The separate full-source
  check still fails on the previously documented unmounted MENTOR-EV
  dendrogram. Keep its historical pin; builds use the verified source index.
  The artifact audit checks 208 preexisting corpus, evaluator, configuration,
  and pinned-policy files. Only the v6 overlay, S1.4 train and manifest files,
  and six earlier v6 manifest overlay pins change. S1.1-S1.3 rows and all
  three retry diagnostic panels remain unchanged. The train SHA-256 is
  `8eb81c0d145495507cedbec7402b9b40890dc681e0be5d78cd6ce14d844e046d`;
  overlay SHA-256 is
  `8a4e37c7b9665c0f5982b0d2e80f13192acda9ad71f6995f1358938cad01e5a9`.
  Verification reports are in `/tmp/s1_4_5_change_xk1zoz6r/`.
  Reproduce with the S1.4 build and regression commands recorded in the
  preceding increment. This increment creates no aggregate v6 bundle
  or model jobs. The explicit missing-value-state experiment is recorded
  under Future evaluation work and remains deferred until v6 is complete.
- **S1.4.6 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 27/125 (21.60%) for v4 and 44/125 (35.20%) for v5. When the
  module ID is a candidate, structured accuracy is 16/63 versus 25/63;
  when it is outside the candidate list, accuracy is 11/62 versus 19/62.
  Both models make the correct initial symbol lookup on all 125 questions.
  V5's 81 failures comprise 31 extra-call trajectories (26 membership
  calls using module IDs as multiplex IDs and five ID lookups), 45 wrong
  structured answers after one lookup, and five invalid final responses.
  Frequent answer errors select a candidate without support, defer despite
  a proven mismatch, or use incorrect result fields. V4 makes 57 invented
  membership calls. V5 changed all 1,000 final training thoughts, but its
  sentence "The module only lists that ID" is misleading because the module
  list also contains a distractor. V4 and v5 training rows otherwise match;
  the preceding v6 form-6 rows are byte-identical to v5. Review reports are
  in `/tmp/s1_4_6_review_zhi9851k/`.
- **Approved S1.4.6 edit.** Keep all 1,000 examples, with 250 per outcome
  and framing combination and 500 in each mention order. Replace only
  final training thinking using train_ambiguous_candidate_reasoning and
  train_ambiguous_noncandidate_reasoning. Bind the returned symbol and count
  once each and the supplied comparison ID twice. A candidate match leaves
  the comparison unknown; a nonempty candidate list excluding the target
  proves a mismatch without selecting the seed's identity. The module
  identifies the comparison target and does not choose the seed's meaning.
  Preserve exactly first/second/same_gene in user order. The typed_seed_list
  side keeps item/input/gene_id/candidate_gene_ids/candidate_details, its
  gene_id stays null, and all candidates retain their original order and
  details. The module side keeps only item/input/gene_id and the supplied
  ID. Preserve literal inputs and registry evidence with no graph evidence.
  No further lookup or membership call is needed. Candidate outcomes keep
  same_gene:null and ambiguous/defer with a blocked reason explaining the
  multiple candidates; both framings ask the user to choose an ID.
  Noncandidate outcomes keep same_gene:false and resolved with no blocked
  reason, stopping for standalone requests or continuing for later workflow
  use. Keep the exact approved wording, every other field, the other 7,000
  training rows including reviewed S1.4.1-2/5 and all 250 failure partners,
  and both complete held-out splits. The v5 switch and inline historical
  reasoning remain available for legacy builds; the new templates apply
  only to training form 6. All binding, visibility, status, decision, and
  alignment checks remain active without a thinking-filter exception.
- **Twenty-sixth-increment verification (S1.4.6).** The validated component
  is published at data/world_model_v4/corpora/s1_v6/s1_4/. Exactly 1,000
  final training thoughts change from the preceding v6 increment. All
  other fields, the other 7,000 training rows, all evidence records, and
  both complete 1,000-row held-out splits remain byte-identical.
  All 161 focused tests pass: 158 regression checks and three separate
  no-overlay/v4/v5 reproduction checks. They cover deterministic v6
  rebuilds, exact approved wording, both outcomes and framings, mention
  order, candidate preservation, invalid template bindings, evidence
  visibility, and retained status/decision/alignment checks. The actual
  uniform-SFT tokenizer checks all 1,000 revised traces at 1,057-3,087
  tokens within the 4,096-token limit; all 2,000 assistant thoughts remain
  fully supervised. Validator selftest passes all 53 known-bad fixtures
  with zero false positives, and check-doc confirms the 73-trace catalog
  is current. The separate full-source check still fails on the previously
  documented unmounted MENTOR-EV dendrogram. Keep its historical pin;
  builds use the verified source index. The artifact audit checks 208
  preexisting files; only the v6 overlay, S1.4 train and manifest files,
  and six earlier v6 manifest overlay pins change. S1.1-S1.3 rows, all
  retry diagnostic panels, historical corpora, evaluator panels, and job
  configurations remain unchanged. The train SHA-256 is
  `61ba976423e17241983b0f3f0466ff6e5f88ba3bf2132b1f5f2aee75a35f57dc`;
  overlay SHA-256 is
  `1b04daa009bc6d3b436f651f8396e258d64120af11fc7c4b295b1eb096345a5e`.
  Verification reports are in `/tmp/s1_4_6_change_e9n68psk/`. Reproduce
  with the S1.4 build and regression commands recorded above. This
  increment creates no aggregate v6 bundle or model jobs. Explicit
  missing-value states remain deferred until v6 is complete.
- **S1.4.8 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 123/125 (98.40%) for v4 and 69/125 (55.20%) for v5.
  Standalone accuracy is 60/62 versus 37/62; workflow accuracy is 63/63
  versus 32/63. Both models make the correct initial ID lookup on all
  125 questions. V5's 56 failures comprise 49 answers adding gene_symbol
  to both comparison items (three also stop prematurely), three further
  workflow answers stopping instead of continuing, and four membership
  calls using module IDs as multiplex IDs and corrupting the final status
  or evidence. V4's two failures replace the original MyGene query, BBC1
  or BAF57, with the returned gene ID in input. All 1,000 training examples
  are identical across v4, v5, and the preceding v6 increment. Review
  reports are in `/tmp/s1_4_8_review_g4x195fi/`.
- **Approved S1.4.8 edit.** Keep all 1,000 examples, split equally between
  direct and workflow framing and between mention orders. Replace only
  final training thinking using train_mygene_match_reasoning. Bind the
  returned symbol once, returned ID twice, and original MyGene query twice.
  The registry confirms the external mapping and the matching module ID.
  Build exactly first/second/same_gene in user order, with exactly
  item/input/gene_id on each side and query_mygene/module item labels.
  Preserve the external query and literal module input, use the returned
  canonical ID for both gene_id values, and keep same_gene:true. Explicitly
  omit gene_symbol from both items. Preserve registry evidence, null graph
  evidence, resolved status, and no blocked reason. No further lookup or
  membership check is needed. Stop for the standalone comparison or
  continue with the result for the user's later workflow step. Keep the
  exact approved wording, prompts, pre-call thoughts, calls, tool results,
  answers, evidence IDs, metadata, ordering, the other 7,000 training rows,
  all 250 failure partners, and both full held-out splits. No wrapper
  restoration or thinking-filter exception is needed. Apply the templates
  only to training form 8; preserve every binding, visibility, and semantic
  check. This completes the requested S1.4 reasoning rework. Passing forms
  3, 4, 7, and 9 stay unchanged; the next failing type to review is S1.6.1.
- **Twenty-seventh-increment verification (S1.4.8).** The validated
  component is published at data/world_model_v4/corpora/s1_v6/s1_4/.
  Exactly 1,000 final training thoughts change from the preceding v6
  increment. All other fields, the other 7,000 training rows, all evidence
  records, and both complete 1,000-row held-out splits remain byte-identical.
  All 176 focused tests pass: 173 regression checks and three separate
  no-overlay/v4/v5 reproduction checks. Coverage includes deterministic
  v6 rebuilds, exact approved wording, both framings and mention orders,
  original-query preservation, the exact three-field comparison items,
  invalid repeated-query/ID bindings, and retained visibility and semantic
  checks. The actual uniform-SFT tokenizer checks all 1,000 revised traces
  at 921-981 tokens within the 4,096-token limit; all 2,000 assistant
  thoughts remain fully supervised. Validator selftest passes all 53
  known-bad fixtures with zero false positives, and check-doc confirms
  the 73-trace catalog is current. The separate full-source check still
  fails on the previously documented unmounted MENTOR-EV dendrogram.
  Keep its historical pin; builds use the verified source index. The
  artifact audit checks 208 preexisting files; only the v6 overlay, S1.4
  train and manifest files, and six earlier v6 manifest overlay pins
  change. All other corpus rows, retry panels, evaluator panels, trained
  artifacts, and job configurations remain unchanged. The train SHA-256 is
  `dbe2233e064c2b66704fd3ac91964e78a6f0b6bee9ca9fad1ad96e407fea7247`;
  overlay SHA-256 is
  `d09508ecbcd39858b1db29948d507a27aaafc205aca86bedfd055d0107b9b316`.
  Verification reports are in `/tmp/s1_4_8_change_v0yczo49/`. Reproduce
  with the S1.4 build and regression commands recorded above. This
  increment creates no aggregate v6 bundle or model jobs. Explicit
  missing-value states remain deferred until v6 is complete.
- **S1.6.1 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 152/167 (91.02%) for v4 and 156/167 (93.41%) for v5.
  Standalone accuracy is 83/84 versus 84/84; workflow accuracy is 69/83
  versus 72/83. V4 has 14 premature workflow stops under "Answer this
  layer request for the next action" and one standalone rejection of
  the valid bulkPEN:brain:frontal_cortex_ba9 tag. V5 fixes all 15 failures
  but introduces 11 premature workflow stops, all for that tag under
  "The workflow first needs this layer request parsed". Each ends its
  thinking with "This request is complete after the parse, so I stop."
  V5 parses all 167 results correctly and preserves status, evidence,
  and zero calls. V4 parses 166/167 results correctly and also makes
  zero calls throughout. V5 changed 1,170 of the 1,250 training prompts
  and all thoughts relative to v4; the remaining row content is identical.
  Retain the v5 bank and strengthen the task-purpose explanation. Review
  reports are in `/tmp/s1_6_1_review_whevr3si/`.
- **Approved S1.6.1 edit.** Keep all 1,250 examples: 625 direct and 625
  workflow requests, using ten bulkPEN tags. Replace only training
  thinking using train_bulk_pen_reasoning. Bind the exact supplied tag,
  organ, and JSON region list once each. The prefix gives bulk_pen, the
  next component gives organ, and the remaining components give the
  ordered region_path. Preserve underscores, hyphens, and original
  spelling. Return exactly layer_id, layer_family, source, network,
  organ, region_path, and cell_type_path. Source, network, and the
  cell-type path remain null, as expected for a valid bulkPEN tag.
  The supplied tag suffices to finish the parse: return resolved with
  no blocked reason. Stop for a standalone parse or continue with its
  fields for the user's later workflow step. Keep the exact approved
  wording, v5 prompts, answers, zero calls, null provenance, metadata,
  row IDs, ordering, the other 3,750 training rows, and both complete
  625-row held-out splits. Preserve the 754 one-component and 496
  two-component region paths. Apply the templates only to training
  form 1, retaining every binding, visibility, and semantic check.
  Keep null unchanged throughout v6; explicit missing-value states
  across S0/S1 remain a future experiment.
- **Twenty-eighth-increment verification (S1.6.1).** The validated
  component is published at data/world_model_v4/corpora/s1_v6/s1_6/.
  Exactly 1,250 training thoughts change relative to v5. All other
  fields, the other 3,750 training rows, zero-call evidence records,
  and both complete 625-row held-out splits remain byte-identical.
  All 114 focused tests pass: 111 regression checks and three separate
  no-overlay/v4/v5 reproduction checks. Coverage includes deterministic
  v6 builds, exact approved wording, both framings and region depths,
  literal tag/path preservation, null fields, the seven-key result,
  invalid template bindings, and retained visibility and semantic checks.
  The actual uniform-SFT tokenizer checks all 1,250 revised traces at
  588-661 tokens within the 4,096-token limit; all 1,250 assistant
  thoughts remain fully supervised. Validator selftest passes all 53
  known-bad fixtures with zero false positives; check-doc confirms
  the 73-trace catalog is current. The separate full-source check still
  fails on the previously documented unmounted MENTOR-EV dendrogram.
  Keep its historical pin; builds use the verified source index.
  The artifact audit checks 208 preexisting files. Only the v6 overlay
  and seven earlier v6 manifest overlay pins change; the S1.6 component
  adds four files. All earlier corpus rows, retry panels, evaluator
  panels, trained artifacts, and job configurations remain unchanged.
  The train SHA-256 is
  `a7edad16c6b419783eab32e6af8aa2c9130eb94913e39b125030419a39b41f87`;
  overlay SHA-256 is
  `69d854949237e8d79b806e7cda8cda30ee8c26328e7358072a8be339206f8716`.
  Verification reports are in `/tmp/s1_6_1_change_xrd4j7t6/`.
  Reproduce with `python -m runtime.world_model.s1_6_corpus --overlay
  runtime/world_model/s1_v6.yaml --output-directory <new-directory>`.
  Run the regression suite with `python -m pytest -q
  tests/test_world_model_s1_6_v6_corpus.py tests/test_world_model_s1_6_corpus.py
  tests/test_world_model_s1_6_v5_corpus.py tests/test_world_model_s1_overlay.py
  tests/test_world_model_s1_prompt_banks.py
  tests/test_world_model_s1_1_v6_corpus.py::test_unreviewed_policies_and_default_bank_remain_v5`.
  Use the pytorch-rocm environment with USE_TORCH=0, USE_TF=0, and PYTHONPATH=.
  This increment creates no aggregate v6 bundle or model jobs.
- **S1.6.2 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 137/156 (87.82%) for v4 and 136/156 (87.18%) for v5.
  Standalone accuracy is 78/78 versus 77/78; workflow accuracy is 59/78
  for both. V4's 19 failures stop prematurely under "Answer this layer
  request for the next action": seven cell_type_path, six source, and
  six organ requests. V5 fixes those rows but introduces 19 premature
  stops under "The workflow first needs this layer request parsed":
  seven layer_family, six region_path, and six network requests. It also
  swaps region_path and cell_type_path in one standalone request for
  scPEN:brain:myelencephalon:ependymal. All 20 v5 failures end their
  thinking with "This request is complete after the parse, so I stop."
  V4 parses 156/156 results correctly and v5 parses 155/156. Both retain
  correct status, evidence, and zero calls throughout. V5 changed 1,170
  training prompts and all 1,250 thoughts relative to v4; the remaining
  content is identical. S1.6.2 stayed unchanged in the previous v6
  increment. Review reports are in `/tmp/s1_6_2_review_6mq1mzkm/`.
- **Approved S1.6.2 edit.** Keep all 1,250 examples and v5 prompts,
  split equally between standalone and workflow requests. Replace only
  training thinking using train_scpen_reasoning. Bind the supplied tag,
  organ, JSON region list, JSON cell-type list, requested field, and its
  typed JSON value once each. The second component supplies the organ;
  the third alone supplies the region; all remaining components supply
  the cell-type path. Preserve their separate ordered lists, second
  cell-type components when present, and exact spelling, underscores,
  and hyphens. Identify the requested field and value while returning
  exactly layer_id, layer_family, source, network, organ, region_path,
  and cell_type_path. Source and network remain null, including when
  requested, as expected for a valid scPEN tag. The supplied tag suffices
  to answer the request: return resolved with no blocked reason.
  Stop for a standalone field request; continue with the parse when
  its requested value supplies the user's later workflow step. Keep
  the exact approved wording, all six field categories, 209 tags across
  12 regions, 1,130 one-part and 120 two-part cell-type paths, answers,
  zero calls, null provenance, metadata, row IDs, and ordering. Preserve
  the other 3,750 training rows, including S1.6.1, and both complete
  625-row held-out splits. Templates apply only to training form 2;
  all binding, visibility, and semantic checks remain active. Keep
  null throughout v6; explicit missing-value states remain deferred.
- **Twenty-ninth-increment verification (S1.6.2).** The validated
  component is published at data/world_model_v4/corpora/s1_v6/s1_6/.
  Exactly 1,250 training thoughts change from the preceding v6 increment.
  All other fields, the other 3,750 training rows (including all 1,250
  reviewed S1.6.1 rows), zero-call evidence records, and both complete
  625-row held-out splits remain byte-identical. All 140 focused tests
  pass: 137 regression checks and three separate no-overlay/v4/v5
  reproduction checks. Coverage includes deterministic builds, the exact
  approved wording, both framings, all six requested-field categories,
  both cell-type path depths, separate region/cell-type lists, typed JSON
  values including null, invalid template bindings, and retained visibility
  and semantic checks. The actual uniform-SFT tokenizer checks all 1,250
  revised traces at 630-772 tokens within the 4,096-token limit; all
  1,250 assistant thoughts remain fully supervised. Validator selftest
  passes all 53 known-bad fixtures with zero false positives; check-doc
  confirms the 73-trace catalog is current. The separate full-source
  check still fails on the previously documented unmounted MENTOR-EV
  dendrogram. Keep its historical pin; builds use the verified source index.
  The artifact audit checks 212 preexisting files. Only the v6 overlay,
  S1.6 train and manifest files, and seven earlier v6 manifest overlay
  pins change. All other corpus rows, retry panels, evaluator panels,
  trained artifacts, and job configurations remain unchanged. The train
  SHA-256 is
  `5e71c6f7a0192bd6da22dc83a12be415528ecbe1e126da67157763e5bc56cf32`;
  overlay SHA-256 is
  `44f8100cda417ad768e992ad65672b18ec5f4e4e5193aee4680d9d254d3c93c4`.
  Verification reports are in `/tmp/s1_6_2_change_5eblga0d/`. Reproduce
  with the S1.6 build and regression commands recorded above. This
  increment creates no aggregate v6 bundle or model jobs.
- **S1.6.3 results (2026-10-19).** The current scorer and the same frozen
  v5 keys give 80/83 (96.39%) for v4 and 68/83 (81.93%) for v5.
  Standalone accuracy is 41/42 versus 40/42; workflow accuracy is 39/41
  versus 28/41. All 83 questions use HumanNetV3:phylogenetic, the single
  held-out test network. V4 has two premature stops under "Answer this
  layer request for the next action" and one answer inventing result
  fields and the family tag. V5 fixes those three rows but introduces
  15 failures: 11 premature stops under "The workflow first needs this
  layer request parsed", two workflow answers omitting source, one
  standalone answer using network as the family, and one standalone
  answer replacing the parse with a null family and invented support
  and evidence fields. All 11 premature-stop thoughts conclude "This
  request is complete after the parse, so I stop." V4 parses 82/83
  results correctly and v5 parses 79/83. Both preserve correct status,
  provenance, and zero calls throughout. V5 changed 585 training prompts
  and all 625 thoughts relative to v4; every other field is identical.
  S1.6.3 stayed unchanged in earlier v6 increments. Review reports are
  in `/tmp/s1_6_3_review_oye79xt9/`.
- **Approved S1.6.3 edit.** Keep all 625 examples, v5 prompts, and the
  existing 313 direct/312 workflow assignment. Replace only training
  thinking using train_humannet_reasoning. Bind the exact supplied tag,
  network, and layer family once each. State the two-component grammar:
  the prefix gives source HumanNetV3 and the second component gives network.
  Explain the existing contract mapping: string_ppi and biogrid_ppi map
  to ppi, coexpression maps to coexpression, and other network names map
  to other. Treat other as a valid family classification. Copy tag and
  network spelling exactly, including underscores. Return exactly layer_id,
  layer_family, source, network, organ, region_path, and cell_type_path.
  Organ and both paths remain null, as expected for a valid HumanNet tag.
  The tag suffices to complete a resolved parse with no blocked reason.
  Use the reviewed S1.6.1 endings: stop for the standalone parse or
  continue with its fields for the user's later workflow step. Keep the
  exact approved wording and all five training networks: string_ppi 121,
  coexpression 128, gene_neighbor 120, cocitation 128, and gene_interaction
  128. Their family counts remain ppi 121, coexpression 128, and other 376.
  The mapping mentions biogrid_ppi from the contract; no native training
  tag is added. Preserve answers, zero calls, null provenance, metadata,
  row IDs, ordering, the other 4,375 training rows (including S1.6.1-2),
  and both complete 625-row held-out splits. Templates apply only to
  training form 3, with every binding, visibility, and semantic check
  retained. Keep null throughout v6; explicit missing-value states remain
  deferred until the corpus revision is complete.
- **Thirtieth-increment verification (S1.6.3).** The validated component
  is published at data/world_model_v4/corpora/s1_v6/s1_6/. Exactly 625
  training thoughts change from the preceding v6 increment. All other
  fields, the other 4,375 training rows (including reviewed S1.6.1-2),
  zero-call evidence records, and both complete 625-row held-out splits
  remain byte-identical. All 160 focused tests pass: 157 regression
  checks and three separate no-overlay/v4/v5 reproduction checks.
  Coverage includes deterministic builds, exact approved wording, both
  framings, all five training networks and three families, the required
  source and seven-key parse, null context fields, invalid template
  bindings, and retained visibility and semantic checks. The actual
  uniform-SFT tokenizer checks all 625 revised traces at 627-652 tokens
  within the 4,096-token limit; all 625 assistant thoughts remain fully
  supervised. Validator selftest passes all 53 known-bad fixtures with
  zero false positives; check-doc confirms the 73-trace catalog is current.
  The separate full-source check still fails on the previously documented
  unmounted MENTOR-EV dendrogram. Keep its historical pin; builds use
  the verified source index. The artifact audit checks 212 preexisting
  files. Only the v6 overlay, S1.6 train and manifest files, and seven
  earlier v6 manifest overlay pins change. All other corpus rows, retry
  panels, evaluator panels, trained artifacts, and job configurations
  remain unchanged. The train SHA-256 is
  `23b7a93cee9180fcdcd643e435db8fc27994d5684b7c14e1d68e2d3553e7dfd3`;
  overlay SHA-256 is
  `fd39526987152ea5c0d351e370dcb3fa49936482470bd81c4d916a1d7e3377d2`.
  Verification reports are in `/tmp/s1_6_3_change_ourj08y4/`. Reproduce
  with the S1.6 build and regression commands recorded above. This
  increment creates no aggregate v6 bundle or model jobs.
- **Final reasoning batch review (2026-10-19).** The user approved batching
  the remaining 13 failing types. All have usable v5 prompts, targets, and
  case coverage for a reasoning-only experiment. Existing v5 reasoning
  already explains many rules; the increment binds their application to
  exact result construction and task scope. The frozen v4 and v5
  predictions were scored with the same current scorer and v5 keys.

  | Type | V4 structured correct | V5 structured correct | Revised train thoughts |
  | --- | ---: | ---: | ---: |
  | S1.6.4 | 60/94 | 62/94 | 750 |
  | S1.6.6 | 61/62 | 58/62 | 500 |
  | S1.7.2 | 110/113 | 104/113 | 900 |
  | S1.7.4 | 179/187 | 163/187 | 1,500 |
  | S1.7.8 | 21/37 | 7/37 | 600 |
  | S1.8.1 | 105/105 | 91/105 | 1,200 |
  | S1.8.2 | 53/59 | 47/59 | 667 |
  | S1.8.3 | 53/58 | 49/58 | 667 |
  | S1.8.4 | 58/58 | 40/58 | 666 |
  | S1.8.5 | 50/50 | 47/50 | 400 |
  | S1.8.6 | 49/50 | 34/50 | 400 |
  | S1.9.1 | 121/125 | 89/125 | 1,000 |
  | S1.9.2 | 117/125 | 94/125 | 1,000 |

  S1.8 figures use canonical/in-scope rows; metric variants and shared
  aliases remain separate diagnostics. The prior review artifacts are
  `/tmp/s1_remaining_review_20261019/`. S1.6.4 needs independent parses,
  complete ordered results, literal filtering, and task-scope decisions;
  S1.6.6 needs explicit grammar rejection and literal input preservation.
  S1.7's failures are principally result failures: every S1.7.2/8 v5
  continuation decision is correct. S1.7.4 needs canonical claim meaning
  separated from family support, and S1.7.8 needs the closed mapping's
  other default rather than inference from the network name. Canonical
  S1.8 truth checks are already correct on 174/175 v5 rows; missing or
  invented fields and continuation decisions dominate. S1.8.5's three
  failures explain ambiguity correctly but construct the wrong result.
  S1.8.6 has twelve schema failures and four false resolutions. S1.9
  imports unrelated schemas or rejects module inputs under another task's
  grammar. Both models make zero calls on every reviewed row.
- **Final reasoning batch implementation (2026-10-19).** S1.6-S1.9 now
  read pinned `train_final_reasoning` templates by form and direct/step
  framing. They apply only during training. Exact binding counts and
  template matching remain enforced, together with visible-identifier,
  status, decision, and semantic checks. Historical builds retain their
  old reasoning paths. The final batch changes only 10,250 assistant
  thoughts; all prompts, inputs, source selections, final answers, metadata,
  row ordering, zero-call behavior, null values, and case distributions
  remain fixed. The other 6,750 training rows across these four components
  and all 4,250 held-out rows stay byte-identical.
  - S1.6.4 binds each grammar's complete values and per-tag comparison,
    keeps all three parses, and selects ordered matching IDs by exact
    scalar equality or path-component membership. Empty matches resolve.
    S1.6.6 derives the invalid prefix, component count, or empty path from
    the visible input, preserves that input, and defers in both framings.
  - S1.7.2 constructs the exhaustive ordered family partition. S1.7.4
    binds the requested canonical claim independently of the family card,
    retains the complete ordered claim partition, and resolves unsupported
    claims with `claim_supported:false`. S1.7.8 retains the valid other
    family, null evidence type, empty supported list, and all eight
    unsupported claims. The HumanNet mapping includes the contract's
    biogrid_ppi rule without adding a native training tag.
  - S1.8 constructs exactly the metric result, separates the convention
    from statement truth, and derives continuation from the enclosing
    request. Agreeing aliases remain resolved with null metric and ordered
    candidates. Opposing percentile candidates remain ambiguous/defer;
    unknown names remain not_found/defer, each retaining required nulls.
    Preserve all v5 metric-name and polarity coverage. Workflow wording
    inside a statement still poses a prompt-generalization uncertainty;
    any later prompt ablation is separate from this reasoning experiment.
  - S1.9 selects the requested module grammar before decoding the ID,
    binds the exact components and result fields, preserves leading zeros,
    and resolves syntax with `existence_asserted:false`, null provenance,
    and no blocked reason. The filter permits only the exact reviewed
    null-provenance sentence after exact-template matching. Other runtime
    claims and invisible identifiers remain rejected. Preserve v5's
    balanced inner requests and all 2,000 selected IDs.
  Remaining sentence-length gates in S1.6, S1.7, and S1.9 no longer cap
  reasoning, as previously directed by the user. They still report its
  observed length and enforce prompt, summary, and blocked-reason limits.
  The actual tokenizer and supervision checks govern reasoning size.
- **Final reasoning batch verification (2026-10-19).** All 510 focused
  tests pass: 175 layer/overlay/prompt-bank tests, 120 claim tests, 120
  metric tests, 94 module tests, and the cross-family policy guard. These
  include deterministic builds, full thought-only comparisons, frozen
  held-out rows, and no-overlay/v4/v5 byte reproduction for all four
  builders. The actual uniform-SFT tokenizer checks every revised thought
  with full supervision and no truncation: S1.6 spans 565-1,505 tokens,
  S1.7 spans 708-1,011, S1.8 spans 598-760, and S1.9 spans 702-788, all
  within 4,096. Builder semantic validation passes every split. Validator
  selftest passes all 53 known-bad fixtures with zero false positives;
  check-doc confirms the 73-trace catalog. The full-source check retains
  its known failure on the unmounted MENTOR-EV dendrogram; keep its
  historical pin and build from the verified source index.
  The artifact audit checks 212 preexisting artifacts. Only the v6 overlay,
  S1.6 training rows and manifest, and seven prior v6 manifest overlay pins
  change among them. S1.7-S1.9 gain new v6 component directories. All trained
  corpora, evaluator panels, retry rows, and run configurations remain
  frozen. No aggregate v6 bundle or training/evaluation job is created.
  Final train hashes are:
  - S1.6: `1448ce1a365be342b4b5532d24aeb401ed9e8043fd1d35b35d1b84e0fbe02061`.
  - S1.7: `2707c842f6480a61a3ebd5b82e750380130226e9a6b66a7078c9320836dd5904`.
  - S1.8: `53fda00f6589832c1e7c9a943e0d64a7cb7074cbfe75a5cebc33dfef40eb6a89`.
  - S1.9: `e5536a27f078d6a0030aab03fa98f19b407c2a024dee29a74dce5563dce73584`.
  The overlay hash is
  `8c0b2b34491daeda2fb5a635050d456a3e7f8e0f487fad059ee5500bed46d024`.
  Reports, baselines, build outputs, and the publication audit are under
  `/tmp/s1_remaining_batch_20261019/`. Reproduce with the pytorch-rocm
  environment and USE_TORCH=0, USE_TF=0, PYTHONPATH=., running
  `python -m runtime.world_model.s1_6_corpus --overlay runtime/world_model/s1_v6.yaml`
  and the corresponding s1_7, s1_8, and s1_9 modules. Use an explicit
  temporary output directory when reproducing any trained legacy version.
- **Remaining review scope.** The user asked to leave passing types
  unchanged. Applying the existing tiered_v3 criteria per question type
  gives 21 failing types before the S1.4.1 increment: S1.4.1/2/5/6/8,
  S1.6.1/2/3/4/6, S1.7.2/4/8, S1.8.1-6, and S1.9.1-2. This is a
  diagnostic breakdown, not a separately published gate. Eleven other
  types in these sections pass; S1.6.7 has no held-out test rows.
  S1.3.2 and S1.3.3 each score 125/125 and need no rework. With S1.4's
  five failing types and S1.6.1-3 revised, the final batch above completes
  the 13 remaining failing types. No failing type remains unreviewed.
  Passing types stay unchanged, and S1.6.7 still has no held-out gate
  result. Final aggregate assembly, run configuration, and model evaluation
  remain separate work. These data-readiness checks do not establish that
  the v6 model passes its gate.

S1 v5 (corpus finalized 2026-10-13; trained 2026-10-14):

The corpus is now frozen. The commands below record its build before
training; reproduce it in a separate output directory. Do not rebuild or
repin the trained corpus when recording new results in this document.

- **Training result (2026-10-14).** `slurm-5671251` completed with exit
  code zero on 32 nodes. It ran all 1,728 optimizer steps and exposed all
  55,276 eligible training rows. The final checkpoint and its exposure
  receipt match the pinned v5 config and corpus.
  - Validation completion loss fell from 1.8794368 to 0.2419846 over one
    epoch on 6,875 rows. The last logged training completion loss was
    0.0006077 at step 1,720. All logged losses are finite.
  - V4 (`slurm-5658398`) finished at validation loss 0.0021871. V5's much
    higher held-out loss prevents claiming an improvement from training
    metrics. Wording, reasoning, and answer-template changes affect this
    proxy; only the exact evaluation will establish structured accuracy.
- **Evaluations submitted (2026-10-14).** Both final-checkpoint preflights
  pass. The S1 gate is job `5677219`, with 6,875 questions and the unchanged
  `eff17405…` panel. The S0 retention test is job `5677220`, with 120,766
  questions from the unchanged `c9714921…` full-registry panel.
  - Both use 512 nodes on `batch`, with a four-hour limit. Retention has
    `afterany:5677219`, so a failed S1 gate still permits the diagnostic,
    and the jobs do not resume the same W&B run concurrently.
  - The retention config is
    `config/world_model_s1_120b_s0_forgetting_test_v5_512n.json`. It pins
    the trained v5 corpus without changing the S0 questions, keys, or gate.
  - Reports will be written under
    `data/world_model_v4/eval_runs/s1_v5_120b_32n_test/oss120b-plain-base-tokenizer-lora-r32/slurm-5677219/report/`
    and
    `data/world_model_v4/eval_runs/s1_v5_120b_s0_forgetting/oss120b-plain-base-tokenizer-lora-r32/slurm-5677220/report/`.
- **Evaluation results (reviewed 2026-10-15).** Both jobs completed with
  exit code zero. S1 failed its promotion gate; S0 retention passed. Keep
  v4 as the comparison baseline and do not promote v5.
  - V5 structured accuracy is 5,533/6,875 (80.48%). For a fair comparison,
    the frozen v4 predictions were rescored with `score_s1_record`, the
    current prose-insensitive scorer, and the same v5 questions and answer
    keys. V4 gets 6,035/6,875 (87.78%): v5 loses 502 correct answers, or
    7.30 percentage points. This comparison changes no saved predictions,
    published reports, held-out files, or trained artifacts.
  - Do not compare against v4's published 82.69% headline. It used the old
    blocked-reason prose comparison and the older S1.1/S1.3 answer templates.
    Correcting prose alone gave 87.33%; using the v5 keys gives 87.78%.

  | Stage | Structured accuracy: v4 on v5 keys | V5 |
  | --- | ---: | ---: |
  | S1.1 symbol to ID | 95.44% | 87.68% |
  | S1.2 ID to symbol | 88.95% | 72.95% |
  | S1.3 ambiguous symbols | 67.91% | 71.11% |
  | S1.4 cross-context alignment | 88.80% | 82.90% |
  | S1.6 layer tags | 88.48% | 86.88% |
  | S1.7 layer claims | 96.40% | 91.60% |
  | S1.8 metric directions | 90.60% | 78.40% |
  | S1.9 module IDs | 95.20% | 73.20% |

  - The largest trace loss is S1.2.4 (valid ID without a symbol): 283 to
    74 correct out of 388. V5 emits no usable final answer on 143 rows;
    repeated or extra lookup calls and inappropriate deferral are common.
    S1.1.9 membership falls from 119 to 65 of 125, S1.1.10 from 87 to 42
    of 88, and S1.4.8 from 123 to 69 of 125. Both valid module families
    lose accuracy: RWR 121 to 89 of 125, Mentor 117 to 94 of 125. Module
    failures again invent result fields or omit the required ones.
  - Some targeted fixes work. S1.2.11 tool-error answers improve from 14
    to 40 of 50; S1.3.8 user selection improves from 78 to 94 of 125;
    S1.4.6 improves from 27 to 44 of 125. Layer-parse result accuracy
    improves from 94.72% to 96.48%, while continuation errors lower its
    complete structured accuracy. Metric-name mapping improves from
    51/80 to 66/80, but shared aliases fall from 40/40 to 28/40, and
    canonical direction contracts fall from 280/280 to 264/280.
  - The S1 gate uses 6,755 canonical/in-scope rows. V5 gets 2,976/3,375
    expected tool argument sequences right (88.18%; target 99.5%),
    continuation 5,910/6,755 (87.49%; target 98%), evidence consistency
    6,221/6,755 (92.09%; target 99.5%), and final-envelope validity
    6,214/6,755 (91.99%; target 99.5%). Tool-followup thinking falls to
    2,853/3,375 (84.53%; target 99%). Closed-book no-call behavior and
    network isolation still pass. These are completed model evaluations,
    not failed Slurm or harness runs.
  - S0 retention passes every gate check over 120,766 questions. Result
    accuracy is 120,715/120,766 (99.9578%), status accuracy is 99.9603%,
    and exact tool arguments are 99.9487%. The S0 parent had 99.9743%
    result accuracy on this panel. Core registry lookup is largely retained;
    the larger failure is S1 behavior in its richer request contexts.
  - The combined v5 change regresses overall. These results do not isolate
    which wording, reasoning, or allocation change caused it. The next
    version should use controlled ablations before more combined changes.

```bash
for stage in s1_1 s1_2 s1_3 s1_4 s1_6 s1_7 s1_8 s1_9; do
  python -m runtime.world_model.${stage}_corpus --overlay runtime/world_model/s1_v5.yaml
done
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v5.yaml
```

- **What changes.** Two S1.1 answers, in every split. The overlay
  `runtime/world_model/s1_v5.yaml` (schema v4) sets
  `family_overrides.S1.1.membership_answers`.
  - S1.1.10: `status` is `resolved`, with `present: false`. It was
    `not_found`.
  - S1.1.11: the result adds `"present": null`.
  - The S1.1.10 training reasoning now states the answer: "The graph result
    directly answers the membership question: the gene is not a node in the
    named multiplex." It is in `runtime/world_model/s1_v5_prompt_banks.json`,
    which also carries the S1.2 and S1.6 changes below.
- **Why.**
  - S1.1.10: the v4 test gave `resolved` on 88 of 88 rows, with
    `present: false` right on 87. In training, the answer status equals the
    last tool status in 92% of tool-call rows, and the membership tool
    returns `resolved`. All 4,330 other `not_found` training rows defer with
    a `blocked_reason`; S1.1.10 was the only exception. The question is
    answered, so it is `resolved`, and `not_found` always means a blocked
    answer.
  - S1.1.11: 8 of the 9 v4 test failures added an invented membership key
    (`member`, `membership`, `membership_status`, `graph_membership`). Forms
    9 and 10 carry `present` in 1,700 training rows; form 11 had no such key.
- **Held-out splits.** Only the answer keys of these two traces change: 88
  and 37 rows in each of validation and test. The questions, and so the
  test panel ID (`eff17405…`), are unchanged. Validation and test keep the
  inline reasoning.
- **S1.2.4 training reasoning.** The bank keys `s1_2_no_symbol_step` and
  `s1_2_no_symbol_direct` replace the final reasoning of the 3,100 training
  rows. Both say that the registry has the ID but no symbol, and that many
  genes are not yet named. The step sentence ends "so the analysis can
  continue with the ID"; the direct one, "so this is a valid answer and the
  task is complete."
  - Why: v4 deferred on 81 of 388 test rows and swapped continue and stop
    on 24. The tool result for a gene without a symbol
    (`resolved`, `gene_symbol: null`) differs from the S1.2.5 result for an
    ID not in release 116 (`not_found`, `gene_symbol: null`) only in
    `status`, and S1.2.5 always defers. S1.2.5 shows the reverse error: 5
    `stop` answers instead of `defer`.
- **Identity wrappers (S1.2.9).** 4 direct and 4 step wrappers are reworded
  as pairs that differ only in the later-step cue: "Identify what this
  request asks for", "Resolve the request below", "Answer the requested gene
  question", and "Look at the following request". The 8 replaced wrappers
  were near-duplicates of other entries, and none is in the S1.3 subset, so
  the bank keeps 64 + 64 wrappers and the same assignment.
  - Why: v4 swapped continue and stop on 4 of 62 S1.2.9 test rows, all under
    two held-out wrappers ("Identify the requested human gene label",
    "Resolve the following gene-label request for downstream use"). In
    training, "identify" never occurred, "resolve the" and "the requested"
    were rare, and "the following" occurred only in step wrappers.
  - `s1_prompt_banks.py` no longer counts "following" as a later-step cue:
    "the following request" points at the text. "follow", "follows",
    "followed", and "follow-up" stay cues. Every v3 to v5 bank keeps its
    status under the narrower rule.
  - Before the final cleanup, other non-cue words occurred in only one
    framing: "the" (10 direct wrappers, 42 step), "analysis" (0, 20),
    "about" (5, 0), and "will", "we", "so", "it", "in", "as", "part",
    "before" (0, 4 to 6). The wording pass now pairs all 64 wrappers,
    as described below.
- **S1.2.11 training rows (train only).** `family_overrides.S1.2.
  train_form_counts` sets S1.2.11 to 800 training rows (was 400) and S1.2.1
  to 800 (was 500), so each failure still copies its own retained S1.2.1
  row. S1.2.10 and S1.2.12 give up the rows (1,000 to 650 each); both were
  right on 125 of 125 test rows in every v2 to v4 run. The S1.2 train total
  stays 11,000.
  - Why: v4 gave the right result shape on 14 of 50 S1.2.11 test rows, with
    a bare `{gene_id}`, the S1.2.5 shape, on 29. The same rule holds on
    S1.1.8 (47 of 50) and S1.4.9 (31 of 31).
  - The splits share one evidence-ID sequence, train first. The builder
    replays the shared-allocation train rows before validation and test, so
    their evidence IDs, and so their rows, stay byte-identical, and builds
    the revised train rows last.
  - Side effects in training: every input is kept, but the family keeps a
    50/50 framing quota, so 250 rows of S1.2.3, S1.2.4, S1.2.7, S1.2.8, and
    S1.2.9 move to the step framing, and the round-robin gives most rows of
    the flexible forms a different wrapper.
- **S1.3 membership answers (every split).** `family_overrides.S1.3.
  membership_answers` changes the S1.3.5 to S1.3.7 answers.
  - The result keeps every field of both tool results: `candidate_details`
    (added to S1.3.5 and S1.3.7) and `absent_gene_ids` (added to S1.3.5 to
    S1.3.7). Why: v4 gave the right result on 56 of 259 S1.3.5 and 6 of 52
    S1.3.7 test rows. It copied `candidate_details` on 107 and 34 of them,
    and both fields on 16 and 31.
  - S1.3.7 is `ambiguous`, not `not_found`; the code stays
    `no_multiplex_membership`. The rule is now: `resolved` when exactly one
    candidate remains, otherwise `ambiguous`. Why: v4 said `ambiguous` on
    40 of 52 rows, the status of the lookup.
  - The shared contract ties `no_multiplex_membership.candidates` to
    `not_found`. The v5 overlaid contract gives it the `lookup` status, as
    `tool_error.partial` has. `s1_kit/validate.py --overlay <overlay>` checks
    rows against the overlaid contract, and the S1.3 builder passes its
    overlay.
- **S1.3 training reasoning.** `train_multiplex_reasoning` rewrites the
  thinking at the point where a row checks membership or defers. Before a
  check, it names the multiplex and its source: "The harness names
  full_brain_multiplex_v1 as the active multiplex, so its membership can
  choose one." Before a deferral, it names what the context has instead:
  "The task names no multiplex" (S1.3.1, S1.3.2), "The harness lists
  registry entries, not a multiplex" (S1.3.4), "and names no multiplex"
  (S1.3.3).
  - Why: v4 skipped the membership check on 42 of 188 test rows whose
    harness named the multiplex, and on 16 of 187 whose task named it. Its
    thinking took the S1.3.3 and S1.3.4 deferral ("The harness does not name
    one, so I will ask for one ID"). The training text said only "The
    harness names the active multiplex".
- **S1.3.10 training rows (train only).** `train_form_counts` sets S1.3.10
  to 600 rows (was 300). S1.3.3 and S1.3.9, right on 124 or 125 of 125 test
  rows in every v2 to v4 run, give up 150 rows each; the S1.3 train total
  stays 9,000. Each failure still copies a kept S1.3.5 to S1.3.7 row, and
  the evidence-ID replay keeps validation and test unchanged. Why: v4 gave
  the right result on 17 of 37 test rows and wrote invented keys
  (`membership_check`, `checked`, `check_status`) for `failed_tool` and
  `error_code`.
- **S1.4.6 training reasoning.** `family_overrides.S1.4.
  train_candidate_reasoning` rewrites the final thinking of the 1,000
  training rows. It names the other ID, says whether it is a candidate, says
  "The module only lists that ID; no multiplex is named," and gives the
  consequence: "could still name another candidate, so the two mentions are
  undecided" or "Whichever candidate X names, it is not ENSG..., so the two
  mentions name different genes."
  - Why: v4 got 16 of 63 candidate and 12 of 62 noncandidate test rows
    right. On 57 rows it called `check_multiplex_membership` with the module
    ID (`group_...`) as the multiplex, the S1.3 membership step; S1.4 never
    trains that call (19, 4, and 19 rows in the earlier runs). It also chose
    the listed candidate (`same_gene: true`) on 14 candidate rows and
    deferred instead of answering `false` on 22 noncandidate rows.
- **S1.6 training questions and reasoning (train only).** The layer bank
  keeps 32 direct and 32 step wrappers. Every direct wrapper now has a
  paired step wrapper whose only addition is a later-use cue. Several cues
  say "next action", "subsequent action", or "upcoming action". The
  `train_layer_reasoning` switch adds the consequence to each valid parse:
  "A later action uses this answer, so I continue" or "This request is
  complete after the parse, so I stop".
  - Why: v4 got 553 of 625 S1.6 structured answers right. Of its 41
    continuation errors, 38 stopped under "Answer this layer request for
    the next action". S1.6.1 to S1.6.3 got 404 of 406 results right; most
    of their failures were decisions.
  - S1.6.4 thinking names each input tag in order. It keeps every complete
    parse, selects matching IDs without removing parsed layers, and says
    that an empty match list answers the request. Why: v4 got 64 of 94
    list results right, and 13 failures omitted the list altogether.
  - S1.6.6 thinking copies the malformed input literally and keeps its
    spelling and repeated parts. Its one v4 result error changed
    `cerebellum` to `cerebellar`. S1.6.5 keeps its answer and thinking;
    v4 got all 63 wrong-type rows right.
- **S1.6 list contents (train only).** `train_diverse_lists` replaces the
  draft HumanNet-only companion swap. It balances family and tag usage,
  weights `other` twice as much as `ppi` and `coexpression`, rotates item
  positions, and avoids repeated content triples. Alternate feasible rows
  retain all-scPEN lists; mixed rows prefer a HumanNet companion when the
  filter permits it. Content selection does not depend on wrapper count.
  - Why: two scPEN tags appeared in 730 and 683 of 750 v4 training lists.
    Ten test lists assigned the wrong HumanNet family, such as `human_net`
    instead of `other`.
  - The new 750 lists cover all 261 training tags, rather than 38. They
    include 424 lists with HumanNet and 288 all-scPEN lists. Each family
    occurs in every list position. No PEN tag occurs more than 39 times.
    Match and no-match counts stay 375 each; all seven form counts stay
    fixed, so S1.6 remains 5,000 training rows.
  - Validation and test keep their native tags, wrappers, reasoning, and
    answers byte for byte. The shared contract and old corpora stay fixed.
    The source and provenance tests now check the pinned source index,
    because the live flist changed after the corpus was frozen. Historical
    manifests keep their builder hashes; fresh builds verify the current
    hash and reproduce the old split files.
  - Checks: 85 focused tests pass, and the validator selftest passes all
    53 known-bad fixtures with no false positives. Two v5 S1.6 builds
    produce identical split files. The base builder reproduces all three
    published legacy splits. The v5 training SHA-256 is
    `d20696e9a76587353bdfd28cac9279e097ff27e87881cfb72495bda7b3c13d26`.
  - All eight v5 components and the aggregate bundle were rebuilt. The
    aggregate has 55,276 training rows and passes semantic validation with
    zero violations. Both evaluator question files are byte-identical to
    v4, including test panel `eff17405…`. S1.6 answer keys stay unchanged;
    only the previously declared S1.1 and S1.3 answers change. The
    aggregate training SHA-256 is
    `14c92564b9c9d95575a029ee81a8810e3eab74dd0cf8e2de55550818c9854e4e`.
- **S1.7 allocation and reasoning (train only, 2026-10-13).**
  `train_form_counts` moves 300 rows from S1.7.1 to S1.7.8: 1,200 scPEN
  family rows and 600 `other` cards. S1.7 stays at 6,000 training rows.
  Each of the three training `other` networks has 200 cards. The
  `train_claim_reasoning` switch changes only S1.7.2, S1.7.4, and S1.7.8.
  - Why: v4 got 723 of 750 structured answers right. All 27 failures were
    in S1.7.8 (16), S1.7.4 (8), and S1.7.2 (3). All 16 failed `other`
    cards invented the `phylogenetic` family, evidence types, and claims.
    S1.7.1 got all 188 test rows right in the last three runs.
  - `other` cards name the input network and state the native mapping:
    `string_ppi` gives `ppi`, `coexpression` gives `coexpression`, and the
    remaining native HumanNet networks give `other`. Its evidence type is
    `null`, its supported list is empty, and all eight claims are
    unsupported. The same mapping appears in claim checks and list filters.
  - Claim checks identify the canonical claim and explain its meaning
    before comparing it with the family card. Why: all eight failed
    checks gave the wrong `claim` or omitted it. Three confused a recorded
    regulatory connection with regulation direction.
  - List thinking names the family of each tag and partitions all inputs
    in input order. No match puts all three IDs in the other list and
    still gives a resolved answer. The list contents stay unchanged.
  - The other forms keep their thinking and answers. Wrapper banks and
    continuation rules stay unchanged; v4's one scored continuation error
    was a malformed JSON answer.
- **S1.7 claim phrases (train only).** The frozen v5 bank adds
  `claim_paraphrases`: 32 phrases for each of the eight claims, replacing
  the eight inline training phrases. They distinguish recorded contacts
  and connections, endpoint roles, expression aggregation, network
  integration, contextual transfer, and causal effects.
  - `s1_prompt_banks.py` checks exact vocabulary keys, coverage, duplicates,
    family and verdict cues, length, and held-out overlap. Every phrase
    has at most nine words. The largest held-out token Jaccard is 0.5556,
    below 0.6, with no shared five-word run. No LLM runs at build time.
  - The 1,500 claim checks exercise all 256 phrases. The family, outcome,
    and framing quotas stay unchanged. There are 17 groups of TF checks
    with the same subject, wrapper, and framing that contrast recorded
    regulatory links (supported) with regulator/target roles (unsupported).
  - Validation and test retain their inline phrases and complete rows byte
    for byte. Both evaluator question files and both answer-key files stay
    byte-identical to the v5 bundle from the S1.6 pass. Earlier stage split
    files stay unchanged. All 87 focused tests pass, including exact legacy
    rebuilds, claim-bank rejection cases, and wrapper-count independence.
  - Two S1.7 v5 builds give identical split files. The training SHA-256 is
    `c5baa512e994d2e30f474dd69de5703973bc8ff39bb41ae3e33587bd27cd7261`.
    All eight v5 components and the bundle were rebuilt. The 55,276-row
    aggregate passes semantic validation with zero violations; its training
    SHA-256 is
    `902feaae23b06fa3a5d635d781510a3220f4b254369774c69134c6e06dd79147`.
    The evaluator code and shared contract are unchanged, and test panel
    `eff17405…` is retained. The validator selftest passes all 53 known-bad
    fixtures with no false positives. The generated document check passes.
- **S1.8 names, coverage, and reasoning (train only, 2026-10-13).**
  The `train_variant_coverage` switch doubles metric-variant rows from 640
  to 1,280, using 640 canonical repeats. S1.8 stays at 4,000 training rows:
  1,600 canonical, 1,280 variants, 320 shared aliases, 400 ambiguous, and
  400 unknown. Form, status, framing, and continuation quotas stay fixed.
  The true-or-false rows still contain 1,000 true and 1,000 false statements.
  - Why: the v4 32-node test got 453 of 500 structured answers right.
    Canonical-name results were correct on all 280 rows, but 11 had the
    wrong continuation. Held-out variants got 48 of 80 results right.
    There were 29 wrong metric fields and 20 wrong rule fields, with
    overlap; many invented identifiers despite explaining the direction
    correctly. Three results changed the input spelling or dropped a word.
  - Each metric now has 12 training names, up from eight. The 156 names
    include symbolic correlations, correlation complements, scalar and
    inner products, centiles, tail probabilities, and overabundance.
    Three earlier added names are reworded to meet the stricter boundary.
    `validate_metric_names` in `s1_prompt_banks.py` enforces token Jaccard
    below 0.6 and no shared five-word run against the 106 held-out names.
    The maximum added-name Jaccard is 0.5. The v3/v4 banks retain their
    original boundary and stay unchanged. No LLM runs at build time.
  - Names rotate within each metric and polarity, independently of wrapper
    counts. All 156 names occur six to twelve times, cover all four truth
    polarities, and have both direct and step lookups. Only seven of 104
    names covered all four polarities before this change.
  - The `metric` bank now has 32 matched direct and step pairs. Check,
    verify, and preparation wording occurs across the pairs. A direct
    wrapper makes this answer the complete task; a step wrapper says
    subsequent work uses it. The maximum held-out wrapper Jaccard is 0.5.
    Why: 13 of 16 continuation errors used held-out check/verify wording.
    Two standalone workflow questions continued because their inner
    request mentioned a next action.
  - `train_metric_reasoning` preserves the exact input, names the canonical
    metric and its fixed rule, explains semantic normalization, and
    compares the claim's direction and outcome before choosing its truth.
    It also states the continuation choice. Standalone workflow questions
    explain that the enclosing request asks only for this answer.
    Shared-alias candidates and deferred null fields keep their meanings.
  - Validation and test rows, both evaluator question files, and both
    answer-key files stay byte-identical to the S1.7 pass. Other stage
    split files stay unchanged. Legacy split reproduction, deterministic
    builds, name-bank rejection cases, and wrapper-count independence pass.
    Training prose stays within 20 instruction words and 25 description
    words per sentence; the observed description maximum is 19.
    All 94 focused tests pass. The complete 55,276-row training file passes
    semantic validation with zero violations. The selftest passes all 53
    known-bad fixtures with no false positives; the document check passes.
  - The S1.8 training SHA-256 is
    `6c0852b9fd16f7446596df452c5446c12c202d995c188e9d5cb2f9a557ed4abf`.
    The rebuilt 55,276-row aggregate training SHA-256 is
    `59b41786ec488787f537c7a5e2c4b91d3ef1eb91b82901d6c47b413b74c5a049`.
    Canonical metric gates stay unchanged. Held-out names and aliases are
    diagnostics. The evaluator code, contract, and test panel remain fixed.
- **S1.9 grammar and result binding (train only, 2026-10-13).**
  The `train_module_reasoning` switch binds each colon component to its
  exact result field. S1.9 stays at 2,000 training rows, with 1,000 per
  module family. The selected source IDs, answers, and framing quotas stay
  unchanged. Validation and test keep their complete rows byte for byte.
  - Why: the v4 32-node test got 238 of 250 structured answers right:
    121 of 125 RWR-LOE parses and 117 of 125 MENTOR-EV parses. All twelve
    failures used the held-out wrapper "Use the module grammar." They
    copied the full IDs correctly, but invented result fields or rejected
    a valid clade. Eleven omitted `existence_asserted`; all eight failed
    MENTOR-EV parses omitted `dendrogram_id` and `clade_id`.
  - Thinking now maps four RWR-LOE or three MENTOR-EV components to the
    exact keys. It strips only `seed_` for the seed gene, retains `clade_`
    and all six digits, and identifies the final RWR part as the threshold
    recipe. It lists the closed result fields and sets
    `existence_asserted: false`. A valid grammar resolves the syntax
    question; it does not establish existence. Thinking also states the
    framing decision.
  - The shared `module` bank has 32 matched direct and step pairs,
    including twelve grammar-focused pairs. Grammar wording occurs on
    768 rows and is balanced over both families and both frames. The
    maximum held-out token Jaccard is 0.4167, below 0.6, with no shared
    five-word run. Other prompt-bank entries stay unchanged.
  - `train_balanced_requests` removes a training shortcut: the eight
    inner parse requests previously predicted continuation perfectly.
    Every request now occurs in both frames for each family, with counts
    differing by at most one. Request choices and module selection stay
    independent of wrapper bank length. Training row IDs change with the
    reassigned request indices; held-out row IDs stay fixed.
  - All 92 focused tests pass, including source-ID preservation, exact
    legacy reproduction, deterministic builds, and wrapper-count
    independence. Training instructions have at most 14 words per
    sentence and descriptions at most 17. The shared limits remain 20
    and 25. The full 55,276-row training file passes semantic validation
    with zero violations. The selftest passes all 53 known-bad fixtures
    with no false positives, and the generated document check passes.
  - The S1.9 training SHA-256 is
    `4d7cc2bf63f6e68f367be782b9902ef90b6a068e2bcab636ce7a8f1af3b94126`.
    The rebuilt aggregate training SHA-256 is
    `a96046d41fd5e0d2a84851dd2d59e8c3b31c3bd902c42371b77cca5a9e72b18c`.
    Other stage row files, evaluator questions and keys, the shared
    contract, and evaluator code remain byte-identical to the S1.8 pass.
- **Gate review after the S1.9 pass.** These findings use the frozen v4
  32-node predictions (`slurm-5658399`), not an evaluated v5 checkpoint.
  The test panel stays `eff17405…`. The v3 gate excludes the 120 held-out
  metric-name and alias rows, leaving 6,755 gated rows.
  - **The original headline scorer needed correction.**
    `structured_answer_accuracy` masks `summary` but compares
    `blocked_reason` wording exactly. AGENTS.md excludes free-text wording
    and scores blocked reasons by presence. A read-only diagnostic that masks
    blocked-reason wording while retaining presence and evidence checks
    changes 5,685 of 6,875 reported successes to 6,004. The 319 rescued
    rows include 53 of S1.2.6, 40 of S1.3.4, 111 of S1.4.5, and 22 of
    S1.4.9. S1.4.9's 31 results are all correct. This does not change or
    pass the gate: its v3 blocked-reason check already scores presence.
    The final cleanup applies this rule to structured accuracy.
  - **Protocol and continuation need a new checkpoint evaluation.**
    Exact tool arguments and payloads are 3,234/3,375 (95.82%), evidence
    consistency is 6,531/6,755 (96.68%), and final-envelope validity is
    6,528/6,755 (96.64%). Their protocol target is 99.5%. Continuation is
    6,398/6,755 (94.72%), below its 98% decision target. Most argument
    errors occur in S1.4.6 and S1.3.5 to S1.3.7, which v5 targets.
    All 66 rejected S1.2.4 null-symbol trajectories emitted JSON but used
    `defer` with a null blocked reason. Another 126 final-envelope
    rejections concern changed evidence channels. These are semantic
    envelope and evidence failures, not evidence of a token-budget limit.
  - **S1.3.8 exposed a result-schema weakness.** It gets 78 of
    125 results right. Forty-one failures replace `candidate_gene_ids`
    with `candidate_details`; five add details, and one substitutes a
    candidate count. The v5 branch still returns only the chosen ID,
    symbol, and candidate-ID list. Its thinking explains the choice but
    not the exact output fields. Adjacent membership forms now retain
    more tool fields. The final cleanup adds selection-specific field binding
    to training reasoning without changing the answers or held-out rows.
  - **A passing failure-harness check does not prove correct error
    answers.** That check passes at 165/168. S1.2.11 gets only 14 of 50
    results right, and S1.3.10 gets 17 of 37. V5 doubles these training
    forms, but final error fields and retained partial evidence still need
    direct evaluation. Exact blocked-reason prose should not be trained
    merely to satisfy the flawed headline metric.
  - **The original claim-leakage diagnostic missed unknown claims.** Sixteen
    invented HumanNet supported claims passed
    `no_unsupported_claim_leakage` because the check intersects only the
    expected known unsupported vocabulary. Exact claim-set and result
    checks caught the rows. The final cleanup requires supported claims to
    belong to the expected allowed set too.
  - **Keep canonical directions and name generalization separate.** V4
    gets all 280 canonical direction results and all 175 canonical truth
    checks right. Variants get only 48 of 80 exact results. The v5 S1.8
    changes target this diagnostic gap; it is not a reason to weaken or
    expand the runtime's canonical-name gate.
  - **V5 launch wiring.** Training, evaluation, and generation now register
    the v5 overlay schema. Dedicated v5 configs pin the final corpus and
    answer keys and retain the v4 32-node recipe. Corpus validation alone
    does not establish a model gate pass.
- **Checks.** Without the v5 keys, the S1.1 to S1.4 builders reproduce the
  v4 split files byte for byte. With the v5 overlay, S1.1 changes only
  in the S1.1.10 and S1.1.11 final answers and in the 608 training prompts
  that use a reworded wrapper. S1.2 changes only in training: the S1.2.4
  reasoning, the moved rows, and the framing and wrapper reassignments
  above. The S1.2 validation and test files are byte-identical to v4.
  S1.3 validation and test change only in the S1.3.5 to S1.3.7 final
  answers (338 rows each); S1.3 training changes in the thinking above, the
  S1.3.5 to S1.3.7 answers, and the moved rows. S1.4 changes only in the
  S1.4.6 training thinking; its validation and test files are byte-identical
  to v4. The golden rows and
  `S1-question-classes.md` follow v5, and the golden test validates them
  with the v5 overlay.
- **Final cleanup.** All 64 shared identity wrappers now have matched step
  versions. A separate later-use sentence supplies the only framing change,
  with eight cue wordings. Five direct wrappers lose restrictive phrases
  such as "nothing more" and "stands on its own" so they work in both
  framings. S1.3 keeps its 32-pair subset. The bank passes the held-out
  overlap audit; wrapper counts and content allocation stay fixed.
  - The S1.3.8 training thinking binds the chosen ID and symbol, copies the
    earlier candidate-ID list in order, and names the exact three result
    fields. The existing training-reasoning switch controls this addition.
  - The headline scorer ignores both prose fields while retaining blocked
    reason presence and evidence binding. The claim diagnostic also rejects
    supported claims outside the expected allowed set. Neither change
    weakens the tiered-v3 gate or changes held-out questions or answers.
- **Submission recipe.** Activate the same environment used by the launcher,
  run the contract-only preflight, then submit the 32-node job. Run these
  commands from the repository root. The evaluation command queues a
  512-node job that starts only after successful training.

  ```bash
  source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
  source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
  mkdir -p logs

  CONTRACT_ONLY=1 JOB_MODE=train \
    RUN_CONFIG=config/world_model_s1_120b_training_v5_32n.json \
    S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
    bash world_model_v2_s0.slurm

  s1_submission=$(sbatch --parsable --nodes=32 --job-name=s1_v5_32n \
    --export=ALL,JOB_MODE=train,CONTRACT_ONLY=0,RUN_CONFIG=config/world_model_s1_120b_training_v5_32n.json,S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
    world_model_v2_s0.slurm)
  s1_train_job=${s1_submission%%;*}

  sbatch --nodes=512 --partition=batch --time=04:00:00 \
    --job-name=s1_v5_eval --dependency=afterok:${s1_train_job} \
    --export=ALL,CONTRACT_ONLY=0,RUN_CONFIG=config/world_model_s1_120b_32n_test_v5_512n.json,S1_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32,CHECKPOINT_PATH=data/world_model_v4/checkpoints/s1_v5_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-${s1_train_job} \
    world_model_s1_eval.slurm
  ```

  No v5 job has been submitted as part of corpus preparation. The training
  recipe remains global batch 32, one epoch, 1,728 steps, learning rate
  1e-4, r32 LoRA, and the S0 r32 uniform parent `slurm-5631883`.

Rebuild S1 v4 (2026-10-11):

```bash
for stage in s1_1 s1_2 s1_3 s1_4 s1_6 s1_7 s1_8 s1_9; do
  python -m runtime.world_model.${stage}_corpus --overlay runtime/world_model/s1_v4.yaml
done
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v4.yaml
```

- **What changes.** At the stage-row level, only the final reasoning of the
  700 S1.1.10 training rows. It now states the rule: "The gene is not a node
  in the named multiplex, so the answer status is not_found." The aggregate
  train publication also normalizes evidence IDs and replay metadata as
  described below; the stage rows remain reusable inputs.
- **Why.** S1.1.9 (present) and S1.1.10 (absent) shared the reasoning "The
  graph result directly answers the membership question." Only the status
  differed, and the reasoning never said why. S1.1.10 `status` was right on
  at most 16 of 88 test rows in any S1 run (16-node v2: 16; 184-node v2: 5;
  v3 at 3 and 6 epochs: 1 and 14).
- **Data.** `runtime/world_model/s1_v4_prompt_banks.json` is the v3 bank
  plus a `reasoning` entry. `s1_prompt_banks.py` accepts only known
  reasoning keys, one sentence of at most 25 words. The overlay
  `runtime/world_model/s1_v4.yaml` moves every stage to
  `data/world_model_v4/corpora/s1_v4/`, so the v3 corpus stays as trained.
- **Corpus gate repair (2026-10-12).** The v4 overlay opts into train-only
  aggregate normalization. One bundle-wide deterministic sequence replaces
  the stage-local evidence IDs consistently everywhere they occur in a row.
  The S0 replay copy drops only the five builder fields declared by its pinned
  source manifest. The canonical validator checks all 55,276 train rows and
  39,426 evidence IDs with zero violations.
  - Train SHA-256: `ff559c3de88f5c548a561d150572a0883f6025b5daedfc6748d1e84292a07d53`.
  - Corpus manifest SHA-256:
    `018094c076db74d66478d2bcaa0d1fdd53ed94b97ade2ccf9cd55d4ba66706a8`.
  - Evaluator manifest SHA-256:
    `db9fb46f0b7c7467cf89daee7ecc09df9f47689694a803c561223f05c077f508`.
- **Held-out splits.** Validation and test keep the old reasoning. Every
  stage file, aggregate held-out file, evaluator question file, and answer
  key is byte-identical to v3. They retain 251 validation and 148 test
  legacy cross-row evidence-ID collision reports. Those IDs occur in the
  frozen evaluator transcripts, so changing them would create a new panel;
  this comparable rerun keeps the panel fixed and limits normalization to
  training.
- **Training.** `config/world_model_s1_120b_training_v4_32n.json` repeats
  the 32-node v3 run (batch 32, 1,728 steps at 1e-4, 3% warmup, cosine), so
  only the corpus changes. The v3 32-node run reached validation loss
  0.0015; its held-out test was skipped to save compute. The v4 test config
  is `config/world_model_s1_120b_32n_test_v4_512n.json` (gate `tiered_v3`,
  the same test panel).
- **Job 5657218 (2026-10-12).** The run stopped before training when the
  Slurm controller timed out while creating the 256-rank preflight step. All
  32 model caches and per-node ROCm checks had passed, and no checkpoint was
  written. The launcher now retries that read-only preflight three times and
  clears partial rank receipts between attempts.

Rebuild S1 v3 (2026-10-10):

```bash
for stage in s1_1 s1_2 s1_3 s1_4 s1_6 s1_7 s1_8 s1_9; do
  python -m runtime.world_model.${stage}_corpus --overlay runtime/world_model/s1_v3.yaml
done
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v3.yaml
```

- **What changes.** Only the training lists. The v2 test showed that the
  model does not generalize from 4 direct and 4 step wrappers (8 in S1.8
  and S1.9), each used about 100 times, to held-out paraphrases. For
  example, no S1.1 to S1.3 training prompt said "workflow". S1.8 also
  trained on 3 names per metric and 2 shared aliases, and it scored 5% on
  held-out shared aliases.
- **Prompt bank.** `runtime/world_model/s1_v3_prompt_banks.json`. An LLM
  drafted the entries. They are frozen data; no LLM runs at build time. The
  overlay `runtime/world_model/s1_v3.yaml` pins the file by SHA-256 and
  moves every stage to `data/world_model_v4/corpora/s1_v3/`.
  - Wrappers: 64 direct and 64 step for S1.1 and S1.2, and 32 + 32 for
    S1.3, S1.4, S1.6, S1.7, S1.8, and S1.9. The S1.3 subset ends at the
    request, because S1.3 wraps imperative requests and multi-turn replies.
    Its size lets the 218 keeps-several rows cover every wrapper with both
    multiplex placements.
  - S1.8 names: 8 training names per metric (the 3 v2 names and 5 new
    names) and 16 shared aliases. Each new alias copies the convention card
    of the v2 alias "distance" or "correlation", so the bank adds wording,
    not conventions. The S1.8 contract section stays frozen.
  - `runtime/world_model/s1_prompt_banks.py` rejects an entry that names an
    input type, a tool, an outcome, or a case-specific operation; has the
    wrong later-step cue for its framing; contains a builder answer phrase;
    breaks the 20-word sentence limit; or comes close to any validation or
    test wrapper or name.
- **Assignment.** Training rows take wrappers round-robin, so each
  (trace, framing) uses every wrapper (median 6 to 16 uses).
  - S1.6, S1.7, and S1.9: only the wrapper changes. Content choices stay on
    the inline wrapper cycle.
  - S1.1 to S1.4: a new assignment also draws new near-miss distractors,
    failure pairs, case perturbations, and (S1.4) direct/step splits, with
    the same counts.
  - S1.8: rows that use a new name change with that name.
- **Held-out splits.** Validation and test keep the inline lists and the v2
  assignment. Their rows, and the evaluator questions and answer keys in
  `data/world_model_v4/eval/s1_v3/`, are byte-identical to v2. The test
  panel ID is unchanged (`eff17405…`).
- **Training.** `config/world_model_s1_120b_training_v3_184n.json`: the
  184-node topology with the 16-node learning rate (1e-4, 3% warmup,
  cosine) and 6 epochs, so 1,806 optimizer steps against 1,728 for the
  16-node run. At the measured 4.2 s per step, training takes about 2 hours.
  The held-out test config is
  `config/world_model_s1_120b_184n_test_v3_512n.json`.
- **Provenance tests.** The generated-corpus tests for S1.6 to S1.9 now
  check the v3 directories, because the v1 manifests record the builder
  source from before prompt-bank support. The S1.6 test also checks the
  live `full_brain_flist.tsv`, which changed on 2026-10-06 (213 lines, new
  `cPEN` prefix) after the contract pinned it. The 2026-10-13 S1.6 revision
  fixes those tests to read the pinned source index, as the builder does.
  S1.6 rebuilds remain independent of the live flist.

## Scope

S1 teaches the coordinate system and the conventions of the multiplex
lattice:

- gene IDs
- context IDs
- layer tags and layer families
- the claims that each layer family supports
- direction conventions
- module IDs

S1 does not teach graph facts. Graph facts start at S2.

**Recall rule (v3).** The model may copy an identifier from its context. It
may parse an identifier whose structure is compositional. It never recalls
an arbitrary identifier mapping from its weights. Recall goes to a tool.

**Book mode.** S1 is mixed:

- The identity families (S1.1 to S1.5) run in tool-call mode.
- The convention families (S1.6 to S1.9) run closed book.
- The book mode is a backend label. The model does not see it. Every row
  lists the same tools, and the model decides from the task whether to call.

## Families

| Family | `question_family` (proposed) | Book mode | State | Traces |
|---|---|---|---|---|
| S1.1 Symbol to Ensembl | `human_symbol_to_ensembl` | Tool call | Ready | S1.1.1–S1.1.11 |
| S1.2 Ensembl to symbol | `human_ensembl_to_symbol` | Tool call | Ready | S1.2.1–S1.2.12 |
| S1.3 Ambiguous symbol | `human_ambiguous_symbol` | Tool call | Ready | S1.3.1–S1.3.10 |
| S1.4 Cross-context alignment | `cross_context_alignment` | Tool call | Ready | S1.4.1–S1.4.9 |
| S1.5 Context ID resolution | `context_id_resolution` | Tool call | Blocked: context ontology data (Manesh and Erica) | None |
| S1.6 Layer tag parsing | `layer_tag_parsing` | Closed book | Ready. bulkPEN and TFs tag form frozen. | S1.6.1–S1.6.7 |
| S1.7 Layer family and supported claims | `layer_family_claims` | Closed book | Ready for all six families, `other` included | S1.7.1–S1.7.9 |
| S1.8 Direction conventions | `direction_conventions` | Closed book | Ready | S1.8.1–S1.8.6 |
| S1.9 Module ID parsing | `module_id_parsing` | Closed book | Ready | S1.9.1–S1.9.2 |

S0.1 to S0.7 are the worked traces for the S0 rows, in the S1 envelope. See
"S0 rebuild."

## Trace contract

Each row has this message order:

1. The system message: the one S1 system prompt, with the tool list.
2. Optional harness context blocks.
3. The user message.
4. Assistant thinking.
5. Zero or more pairs of a tool call and a tool result. Assistant thinking
   follows each tool result.
6. The assistant final answer.

Rules:

- A Thinking step follows each tool result, before the next call or the
  answer. A row with no call has one Thinking step.
- Apply loss to all assistant thinking, the tool calls, and the final answer.
  Mask the system message, the harness context, the user message, and the
  tool results.
- **Multi-turn rows** (S1.3.8, S1.3.9):
  - Render the earlier turn as the harness does at inference: the user
    message and the final answer, with no thinking.
  - Mask the earlier turn. Train the last turn only.
- The first Thinking step of an identity row does not predict the tool
  outcome. The model cannot know before the call that a symbol is ambiguous,
  absent, or unreachable.
- **Thinking content (for now: fixed wording).**
  - Use the wording of the traces. For each form, the generator fills a
    template taken from its trace. Only the slot values change: IDs,
    symbols, tags, and counts.
  - Each Thinking step gives the tool status and the key value, whether the
    result answers the task, and the reason for the decision.
  - Exact thinking text is a diagnostic metric, not a gate, as in S0.
- **Later: model-written wording.** Move to thinking that the model writes
  itself, checked by the runtime filter only. Keep the filter now, so that
  the move changes only the generator.
- **Tokenizer.** The S0 renderer drops the thinking after the tool result.
  Change it, and use the Harmony library as the reference renderer. Add unit
  tests:
  - the label mask covers the thinking after each tool result
  - the label mask excludes the earlier turn of a multi-turn row

### System prompt

Every row in one experiment uses the same prompt and the same three tool
schemas, including closed-book rows and S0 replay. No prompt or tool-list
variation reveals the book mode, family, or expected outcome.

The frozen shared prompt in `runtime/world_model/contract.yaml` remains in
use by S0 and S1 v1–v6. The opt-in v7 overlay selects the approved text in
`runtime/world_model/prompts/task_scope_v1.txt`; see
[the v7 experiment](S1-prompt-v7.md) for its exact identities and files.
It distinguishes the user's task from quoted claims, scopes continuation to
an explicitly requested workflow, retains evidence across calls and turns,
separates registry resolution from membership, and bounds retries by the
runtime's explicit allowance. A resolved ID with no symbol remains resolved
with a null symbol. Confirmed absence and unknown information stay distinct.

`runtime/world_model/system_prompts.py` pins the approved asset. Corpus
builders, training validation/exposure receipts, generation, and scoring
all use the selected prompt identity. The tokenizer renders the system row
as native developer instructions, with its existing `Reasoning: low` scaffold.
No `StagePromptContract` or per-book-mode overlay is used in this pipeline.

## Answer envelope

The final answer is one JSON object. Use the keys in this order:

| Key | When | Values |
|---|---|---|
| `status` | Always | `resolved`, `ambiguous`, `not_found`, `invalid_input`, `error` |
| `provenance` | Always | Composite `{"registry": <block or null>, "graph": <block or null>}`; overall `null` when the answer rests on no successful evidence |
| `result` | Always | The family payload |
| `continuation_decision` | Always | `continue`, `stop`, `defer` |
| `blocked_reason` | Always | A sentence when the decision is `defer`; `null` otherwise |
| `summary` | Always, last | Readable text with anchors |

### `status`

- `invalid_input` marks input that the model parsed but cannot act on.
- `error` marks a transport failure. It is a fact about the call, not about
  the gene. A failed call establishes nothing, so the model never reports it
  as `not_found`.
- **Partial failure.** If one call succeeds and a later call fails, `status`
  gives the outcome of the successful lookup. `blocked_reason` is
  `tool_error`. The result keeps the lookup result and names the failed tool
  (S1.3.10).
- `status` is `error` only when the answer rests on failed calls alone
  (S1.1.8, S1.2.11).
- An answer that is not `resolved` makes no claim about the unresolved item.

### `continuation_decision`

| Situation | Decision |
|---|---|
| The task is a step in a larger task, and the result is usable | `continue` |
| The task is fully answered, also when the answer is "no" | `stop` |
| Work is blocked until the user gives input that the model cannot supply | `defer` |

- About half of the rows are steps ("…before the next analysis step") and half
  are direct questions.
- A result that is not usable gives `defer` in both framings.
- An Ensembl ID with no symbol is usable. The answer is `resolved`, with
  `gene_symbol` set to `null`.
- `revise` is not an S1 value. The Stage 2 trajectory state has a separate
  field, `loop_control` (`continue`, `revise`, `stop`).
- The summary asks the user for input only when the decision is `defer`.

### `blocked_reason`

On every `defer` answer, `blocked_reason` is one sentence that tells the
user what blocks the task. On every other answer it is `null`.

- **Fixed templates, for now.** Each reason code has one or more sentence
  templates. The runtime fills the slots from the result.
- **The code stays in the backend.** The row metadata records
  `blocked_reason_code`, hidden from the model like `question_family`. The
  validator and the Stage 2 scorer read the code. They check that the
  sentence equals the template for that code, filled from the result.
- **Normative source.** `runtime/world_model/contract.yaml` holds each
  template, the status that it fits, the source of each slot, and the prose
  labels. The table below is explanatory. If they differ, the runtime
  contract wins.
- **Slots that the result does not hold.** `{found}` and `{species}` come from
  the hidden row metadata (`found_input_type`, `input_species`), rendered with
  the YAML labels. Every other slot comes from the result.
- Numbers in the sentence are digits ("2 genes").

| Code | `status` | Template | Traces |
|---|---|---|---|
| `ambiguous_symbol` | `ambiguous` | The symbol {symbol} matches {n} genes in Ensembl release 116, and nothing in the task chooses one. | S0.5, S1.3.1–S1.3.4, S1.4.6 |
| | | The symbol {symbol} matches {n} genes, and {k} of them are nodes in {multiplex_id}, so the multiplex does not choose one. | S1.3.6 |
| | | The reply does not choose one of the {n} candidates for {symbol}. | S1.3.9 |
| `not_found` | `not_found` | The registry has no gene with the symbol {input} in Ensembl release 116. | S0.6, S1.1.3, S1.4.5 |
| | | The registry has no gene with the ID {input} in Ensembl release 116. | S0.7, S1.2.5 |
| `invalid_input` | `invalid_input` | The input {input} is not a well-formed {expected}. | S1.2.6, S1.6.6 |
| `wrong_input_type` | `invalid_input` | The input {input} is a {found}, not a {expected}. | S1.6.5, S1.7.5; planned for the S1.2 `ENST` and `ENSP` inputs |
| `no_multiplex_supplied` | Planned | The task asks about multiplex membership, but it names no multiplex. | None |
| `no_multiplex_membership` | `not_found`; from S1 v5, the lookup status (`ambiguous`) | No candidate for {symbol} is a node in {multiplex_id}. | S1.3.7 |
| | Planned | The gene {gene_id} is not a node in {multiplex_id}. | None. For a clean resolution to a gene that is not in the multiplex. Depends on open question 11. |
| `out_of_scope_species` | `invalid_input` | The input {input} is a {species} gene {kind}, and the registry holds human genes only. | S1.1.7, S1.2.10 |
| `ambiguous_metric` | `ambiguous` | The name "{input}" can mean {c1} or {c2}, and the two have opposite directions. | S1.8.5 |
| `unrecognized_metric` | `not_found` | The metric "{input}" is not in the direction conventions. | S1.8.6 |
| `unknown_multiplex` | `not_found` | The multiplex {multiplex_id} is not registered. | S1.1.11 |
| `tool_error` | `error` | The call to {failed_tool} failed with {error_code}, so there is no registry result for {input}. | S1.1.8, S1.2.11, S1.4.9 |
| | Status of the successful lookup | The lookup found {n} candidates, but the call to {failed_tool} failed with {error_code}, so no candidate is checked in {multiplex_id}. | S1.3.10 |

- `blocked_reason` explains the blocker. The summary tells the user what to
  give to continue.
- The earlier-turn answers in S1.3.8 and S1.3.9 carry the sentence too.

### `provenance`

`provenance` is a field of every tool result and every answer. The key is
always present. A non-null value has exactly two typed channels, in order:

```json
{"registry": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_..."},
 "graph": {"multiplex_id": "full_brain_multiplex_v1", "graph_version": "<sha256>",
           "layer_scope": "all", "evidence_id": "ev_mem_..."}}
```

- A registry lookup or harness registry block populates only `registry`.
- A successful membership query populates only `graph`, including a
  `not_found` result for an unknown multiplex.
- An answer merges the successful evidence it uses. S1.3.5–S1.3.7 and the
  S1.1 membership forms therefore carry both channels.
- A partial failure keeps the successful channels. A transport error is not
  evidence and contributes no channel.
- With no successful evidence, the entire value is `null`; do not emit an
  object whose two channels are both null.
- Each call gets a new, non-input-derived `evidence_id`, unique across rows.
  The answer copies every populated block exactly.

### `result` shapes shared across families

- One gene: `{"gene_id": ..., "gene_symbol": ...}`. `gene_symbol` is `null`
  when the gene has no symbol.
- Invalid input: `{"input": ..., "expected": ...}`. `expected` is one of
  `human_gene_symbol`, `human_gene_id`, or `layer_tag`.
- **Tool failure, answer on failed calls alone:** `{"failed_tool": ...,
  "error_code": ..., <input key>: ...}`, for example `"gene_symbol": "VARS1"`.
- **Tool failure, partial:** the lookup result, plus `"multiplex_id"`,
  `"failed_tool"`, and `"error_code"`. `multiplex_id` names the multiplex
  that the failed check was to read (S1.3.10).

### Mapping to Stage 2

Keep every S1 answer field. `blocked_reason` and `defer` tell the user why
the analysis stopped, so Stage 2 must carry them. Do not add fields that
exist only in S1.

| S1 field | Stage 2 today | Action |
|---|---|---|
| `status` | No per-item field. `relationship_status` in `updated_state` describes a group, not one identifier. | Add an item status to the Stage 2 state, for identifiers that the agent resolves |
| `provenance` | The state records the evidence and provenance collected so far | Add each `evidence_id` to that record |
| `result` | `updated_state` | Use the same object shapes in the state: the one-gene object and the candidate lists |
| `continuation_decision` | The verifier's `continuation_decision` (`continue`, `revise`, `stop`) and `loop_control` | Add `defer` |
| `blocked_reason` | None | Add it next to `defer`: the sentence in the state, and the code in the backend |
| `summary` | `updated_interpretation` | Write it as interpretation text |

### `summary` and anchors

- The summary is readable text, and it is the last key.
- **For now, use the fixed wording of the traces.** Each form has one
  summary template, and only the slot values change.
- **Later,** let the model write the summary. The rule checks below stay the
  same.
- **Rule checks:**
  - Every anchor matches the result: the same genes, IDs, and layers.
  - It makes no claim that the result does not make.
  - It gives no ID or symbol that is not in the result, the prompt, or a tool
    result.
  - It asks the user for input only when the decision is `defer`, and then it
    says what input is needed.
- Anchors:
  - `<GENE:ENSG…|SYMBOL>` for a gene with a symbol, and `<GENE:ENSG…>` for a
    gene with no symbol
  - `<LAYER:tag>` for a layer
- Anchor rules:
  - Anchor every resolved gene.
  - For an ambiguous set, anchor all candidates or none. Above 5 candidates,
    anchor none and give the count.
  - An input that did not resolve, or an invalid input, appears as plain text.
  - An anchor has a symbol label only when an item or a tool result gives
    that symbol.
- Use `<MULTIPLEX:id>` for every multiplex mention in a summary. The ID must
  also appear in the result.

## Tools

S1 uses three tools. `resolve_context_ontology` joins them when S1.5 is
unblocked. `query_mygene` appears only as a harness block (S1.4.8). The model
does not call it in S1.

```
lookup_human_gene_symbol({"gene_symbol": ...})
  input: trims spaces, ignores case
  {"status": "resolved",  "provenance": P, "gene_id": ..., "gene_symbol": ...}
  {"status": "ambiguous", "provenance": P, "gene_symbol": ...,
   "candidate_gene_ids": [...],
   "candidate_details": {"<gene_id>": {"chromosome": ..., "biotype": ...}}}
  {"status": "not_found", "provenance": P, "gene_symbol": ..., "candidate_gene_ids": []}

lookup_human_ensembl_gene_id({"gene_id": ...})
  input: trims spaces, ignores case, strips a numeric ".N" suffix only
  {"status": "resolved",      "provenance": P, "gene_id": ..., "gene_symbol": ...}   (gene_symbol can be null)
  {"status": "not_found",     "provenance": P, "gene_id": <input>, "gene_symbol": null}
  {"status": "invalid_input", "provenance": P, "input": ..., "expected": "human_gene_id"}

check_multiplex_membership({"multiplex_id": ..., "gene_ids": [...]})
  {"status": "resolved", "provenance": P, "multiplex_id": ...,
   "present_gene_ids": [...], "absent_gene_ids": [...]}
  {"status": "not_found", "provenance": P, "multiplex_id": ...}

any tool, transport failure
  {"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": ...}

P = {"registry": <registry block or null>, "graph": <graph block or null>}
```

- Every tool result has `provenance` as its second key, after `status`. The
  runtime writes it into the result that the model sees.

- The tool argument copies the user's text exactly, including case and any
  suffix. The answer uses the canonical form from the tool result.
- The ID tool keeps any suffix other than `.N`. `ENSG00000182378.14_PAR_Y`
  returns `not_found`, and the tool does not strip its way to another gene.
  From GENCODE 44, the chromosome Y PAR copies have their own IDs.
- The ID tool checks the format (ENSG followed by 11 digits). The model does
  not check or repair an ID.
- If case folding joins two registry symbols, the symbol tool returns
  `ambiguous` for them.
- A gene is a member of a multiplex when it is in the gene table of the
  multiplex store, in any layer. The membership tool reports. It does not
  choose.
- An error result has `"provenance": null`.
- A membership result carries a graph-channel block. Registry lookup results
  carry a registry-channel block. Answers merge the channels they use.
- **Registry.** Include every Ensembl release 116 human gene, named or
  unnamed.
  - Set `gene_symbol` to `null` when the GTF `gene_name` is missing or equals
    the gene ID.
  - Store the chromosome and `gene_biotype` of each gene, for
    `candidate_details`.

### Canonical runtime

`runtime/world_model/tools.py` implements this contract from the verified
source index. The incompatible `runtime/world_model_s0_tools.py` remains
frozen only for reproducing legacy corpora; new S0 and S1 builders do not
import it.

| Legacy runtime | Canonical v3 runtime |
|---|---|
| Exact match; whitespace raises an error | Trim spaces; ignore case; collision check |
| A noncanonical ID raises an error | Strip `.N` only; return `invalid_input` as a payload |
| The ID tool returns `gene_symbols` (a list) | `gene_symbol` (one value or `null`) |
| Ambiguous results have `"action": "defer"` | Remove `action` |
| No `candidate_details` | Add chromosome and biotype |
| Provenance sits in a separate object (`ensembl_release: "Ensembl_116"`), and the S0 builder shows the payload only | Add the composite `provenance` object to every result the model sees, with the registry channel populated. Make a new random `evidence_id` for each call. In the log, map each ID to full internal provenance. |
| Bad input raises `ToolExecutionError`, and there is no error payload | Return the error payload with `"provenance": null`, with a switch that injects failures for training rows |
| No membership tool | Add `check_multiplex_membership` |

The canonical tool contract is
`mentor-rl-world-model-s0-identifier-tools-v3`. Registry fields, retired IDs,
historical symbols, and multiplex membership all come from the one
content-addressed source index; `scripts/grab_human_gene_ids.py` is not on the
new path.

## Harness context blocks

- **Registry block.** The block has the header `Registry entries, Ensembl
  release 116:`, then one JSON object:

  ```
  {"provenance": {"registry": {"registry": "ensembl", "release": "116",
                                 "evidence_id": ...},
                  "graph": null},
   "entries": [...]}
  ```

  - A resolved entry is `{"gene_id": ..., "gene_symbol": ...}`.
  - An ambiguous entry has the ambiguous tool-result shape, with no
    provenance of its own (S1.3.3). The block's `provenance` covers every
    entry.
  - The model trusts only this block. A mapping in the user's free text is
    the conflict case, which is not in S1.
  - The model skips a call only when the block lists the exact input for
    release 116. Each no-call form has a near-miss form, in which the block
    lists another gene (S1.1.5, S1.2.8, S1.3.4).
  - The answer copies the block's `provenance` object, including the
    `evidence_id`. It does not parse the header.
- **Active multiplex.** The block is the line `Active multiplex: {multiplex_id}`.
  The user text can also name the multiplex ("in {multiplex_id}").
- **S1.4 items:**
  - `Module {group_id}, gene_ids: [...]`
  - `Seed gene set: {"seed_gene_ids": [...], "seed_gene_symbols": [...]}`
  - `Abstract`: quoted text, marked as example text when it is not a real
    abstract
  - a `query_mygene result`

## Generation rules

- **Real identifiers only.**
  - Take every symbol, ID, and candidate set from the pinned Ensembl release
    116 registry.
  - Take previous symbols and aliases from a pinned HGNC file, and retired
    IDs from a pinned Ensembl stable-ID history.
  - Do not invent identifiers. The placeholders in S0.4 to S0.7 are generator
    slots.
- **Multiplex IDs.** Positive examples use only multiplexes that exist in the
  store. Today that is `full_brain_multiplex_v1`. The explicit unknown-ID
  forms use the frozen negative `retired_brain_multiplex_v0`.
- **Layer tags.** Use the real tags from the flist. Do not use random
  combinations.
- **Templates.**
  - Use one shared pool for each family. S1.1 and S1.3 share one pool,
    because the tool result, not the wording, decides the family.
  - Pair every template with every form, so that the input decides the form.
  - Remove words that are true for only one form.
  - Keep some templates and paraphrases out of training, for the test split.
- **Framing.** Split the rows about 50/50 between steps and direct
  questions.
- **Case.** In the identity families, about 10% of the inputs use lowercase
  or mixed case. Do not correct spelling.
- **Prompt leakage checks.** Replace the request payload with `<INPUT>`.
  Train a simple classifier on the outer wrappers.
  - Its form accuracy must stay near the largest form share.
  - Also report an ungated classifier on the full visible prompt. Mask source
    identifiers and values first.
  - Required task semantics define several forms. Thus, full-prompt accuracy
    is diagnostic.
  - Check required structures directly. Registry blocks span forms 3 and 4.
    Multiplex placement varies within forms 5 to 7 and 10.
  - Earlier turns span forms 8 and 9.
  - No prompt cue can predict a membership outcome before the tool result.
- **Thinking and summary (for now).** Fill the fixed per-form templates from
  the traces. The runtime computes every slot value.
- **Later: model-written wording.** When you move to it:
  - Generate the thinking from the base model and keep only the traces that
    the runtime confirms (STaR).
  - Identity families: put the tool contract in the generation prompt.
  - Convention families: put the grammar or the convention table in the
    generation prompt, and remove it for training (context distillation).
  - Run a pilot of about 200 prompts for each form, and record the yield by
    form.
  - For a form with a low yield, sample more attempts, then give the runtime
    answer as a hint (rationalization). Remove any trace that mentions the
    hint.
- **Runtime filter.** Remove a trace if its thinking:
  - names an ID, a symbol, or a value before the prompt or a tool result
    gives it
  - gives a status that does not agree with the tool result
  - gives a decision that does not agree with the answer
  - discusses the answer schema, for example provenance

  The filter does not check length or style.
- **Tool-failure rows.**
  - Inject `BACKEND_UNAVAILABLE` at the runtime.
  - Build each failure row as a minimal pair: the same prompt as a success
    row, as in S1.1.1 and S1.1.8. The only cue is the tool result.
  - Include failures on the first call and on the membership call.
  - Provisional share: 5% of the tool-call rows in S1.1 to S1.4.
  - In S1.1, failures replace rows in the direct-form allocation, so the
    family remains 10,000 training rows. Every failure keeps an identical
    successful direct row as its minimal pair. The 5% is computed from the
    ordinary tool-call allocation before replacement. S1.1 injects the first
    call only; the membership-call failure cases live in the families whose
    reviewed traces define a partial-failure result.
- **Splits.**
  - Identity families: the S0 identifier graph components. Assign the new
    pools (unnamed IDs, retired IDs, previous symbols) to splits before
    generation.
  - Convention families: by context. Record the contexts of each row in its
    metadata.
  - A family `train_rows` count is the 80% split anchor. Derive validation and
    test counts from `curriculum.split_fractions`, and apportion form shares
    with deterministic largest-remainder rounding.
  - S1.2 splits its anchor into `base_train_rows` and `added_form_rows`.
    Derive and apportion those two allocations separately, then add them; the
    sum must equal `train_rows` in every split.
- **Row metadata.** Each row has a metadata record. The backend reads it for
  generation, splits, validation, and scoring. The model does not see the
  hidden fields.

  | Field | Model sees it | Why |
  |---|---|---|
  | `book_mode` | No | It tells the model whether to call a tool |
  | `step` | No | It names the curriculum stage, which is not task content |
  | `question_family` | No | It tells the model the family, for example that a symbol is ambiguous (S1.3) before the lookup |
  | Form number, source trace ID, split unit | No | Backend only |
  | `blocked_reason_code`, `found_input_type`, `input_species` | No | They give the answer. The validator and the scorer read them to check the `blocked_reason` sentence. |
  | `schema_version`, `system_prompt_sha256`, `tool_contract_version`, `ensembl_release`, `identifier_registry_id` | No | Backend only, as in S0 |
  | `multiplex_id`, `layer_scope`, `layer_ids`, `entity_namespace` (the v3 fields) | Only as harness context, when the task needs it | See the rule below |

  **Rule for visible fields.** A field that the model sees must not change
  with the form, the book mode, or the answer. If it does, hide it or keep it
  constant. For example:
  - `layer_ids` that is empty only on wrong-type rows shows the answer.
  - `multiplex_id` that is present only on membership rows shows the form.

  A multiplex that the task names reaches the model as the `Active multiplex`
  harness line or in the user text, not as metadata.

## S0 rebuild

Every identity answer changed shape. The replacement corpus is built in the
S1 envelope by `python -m runtime.world_model.s0_corpus` before any S0 rows
are mixed into S1.

The current S0 corpus is v4 (`world_model_v4_s0_human_identifiers`). Its mix,
training sweep, test, and gate are in `S0-human-gene-identifiers.md`.

- Use S0.1 to S0.7 as the trace pattern. The reviewed examples are:

  | Trace | Fill with |
  |---|---|
  | S0.4 (unnamed gene) | ENSG00000299624 (applied) |
  | S0.5 (ambiguous symbol) | GOLGA2P9 (applied) |
  | S0.6 (absent symbol) | LINC02170 (applied) |
  | S0.7 (ID not in 116) | ENSG00000281376 from the pinned stable-ID history (applied) |

- The envelope has `provenance` (with `evidence_id`), `blocked_reason` (a
  sentence on every `defer` row), and support for the runtime `error` status.
- The tool result carries `provenance` as a field that the model sees.
- Use the S1 system prompt and the S1 metadata rules.
- **Replay.** Keep 0.25–1% canonical S0 rows in the S1 data. Check the S0 gate again
  on the S1 checkpoint.
  - Take the replay rows from the S0 v4 corpus.
  - S1 v2 replays 276 rows (0.5%) from the S0 v4 corpus. The S1 v1 corpus
    replays 276 rows from the v3 S0 corpus.
  - Without an overlay, `scripts/build_world_model_s1_bundle.py` reads the
    replay from the S0 entry in `runtime/world_model/contract.yaml`. With
    `--overlay`, it reads the source that the overlay names. The selection
    seed and share still come from the shared contract.
- If you replay old v6 rows, they teach the old envelope on the same prompts.

## Start from the S0 checkpoint

S1 continues training from the selected S0 v4 checkpoint. It does not start
from the base model.

- **Adapter.** Load the LoRA adapter of the selected S0 arm. Use the same base
  model, tokenizer, LoRA rank, and LoRA alpha as that arm.
- **Optimizer and schedule.** Start a new optimizer and a new schedule at
  step 0. Do not continue the S0 optimizer state.
- **Config.** `config/world_model_s1_120b_training_v2.json` starts from the
  S0 r32 uniform arm. The v2 training schema requires the `initialization`
  block. `config/world_model_s1_120b_training_v1.json` still starts from the
  base model.
- **Trainer support (added 2026-10-08).** Add an `initialization` block to the
  S1 training config:

  ```json
  "initialization": {
    "parent": "s0_checkpoint",
    "checkpoint_path": "data/world_model_v4/checkpoints/s0_120b/<arm>/slurm-<job>",
    "run_id": "<the parent run ID>",
    "adapter_sha256": "<SHA-256 of adapter_model.safetensors>",
    "lora_rank": 32,
    "loss_contract": "multiplex_sft_v3"
  }
  ```

  - The training contract accepts the block only for an S1 schema. It checks
    that the parent is a complete S0 v4 run, that the adapter hash agrees,
    and that the run ID, the base model, the tokenizer, the LoRA rank and
    alpha, and the parent loss contract agree.
  - The trainer loads the consolidated parent adapter, slices it for the TP
    layout of the new run, and checks that each gathered tensor equals the
    parent tensor. The optimizer and the schedule start new at step 0.
  - `--resume_from_checkpoint` still continues the same run. A resumed S1 run
    uses its own checkpoint, not the parent.
  - Each saved S1 adapter manifest records the parent in `initialization`.
- **Effect on the corpus.**
  - S1.1 to S1.3 continue the S0 identity training. They must not undo the
    S0 v4 mix. For example, see the no-symbol share in S1.2.
  - The S0 replay protects the S0 skills. Take it from the v4 corpus.
  - The S1 rows and the S0 v4 rows use the same contract, system prompt, and
    tools, so the answer shapes agree.
- **Forgetting check (added 2026-10-08).** Run the S0 full-registry test on
  the final S1 checkpoint with
  `config/world_model_s1_120b_s0_forgetting_test_512n.json`. The S1
  checkpoint must pass the S0 gate.
  - The test accepts only an S1 checkpoint whose manifest records an S0
    parent.
  - It uses the S0 panel, the two-tier S0 gate, and the same 512-node command
    as the S0 test (see `S0-human-gene-identifiers.md`).
  - It writes to `data/world_model_v3/eval_runs/s1_120b_s0_forgetting/`. It
    logs to the W&B run of the S1 job with the prefix `s0_forgetting/`, so
    the metrics stay apart from the S1 test metrics.
  - The config pins the S1 v1 corpus. Update the corpus pins when the S1
    corpus is rebuilt.
- **Open decision.** Decide the S1 loss weighting and learning rate for the
  continued run. The S0 sweep compares a uniform and a priority loss.

## S1.1 Symbol to Ensembl

- **Book mode:** tool call. Some rows make no call.
- **Input:** one gene symbol, optionally with a multiplex ID. Gene sets go to
  S1.4.
- **Tools:** `lookup_human_gene_symbol`, and `lookup_human_ensembl_gene_id`
  for an ID input.
- **Result:** `{"gene_id": ..., "gene_symbol": ...}`. Forms 9–11 add
  `multiplex_id` and `present`. From S1 v5, `present` is `null` in form 11,
  because an unknown multiplex gives no membership answer.

| # | Form | Share | Call | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|
| 1 | Direct | 16% | Symbol tool | `resolved` | – | S1.1.1 |
| 2 | Inside a task | 16% | Symbol tool | `resolved` | – | S1.1.2 |
| 3 | Absent symbol (previous symbol or alias) | 12% | Symbol tool | `not_found` | `not_found` | S1.1.3 |
| 4 | Harness block lists the symbol | 12% | None | `resolved` | – | S1.1.4 |
| 5 | Block without the symbol (near miss) | 8% | Symbol tool | `resolved` | – | S1.1.5 |
| 6 | Ensembl gene ID given as the input | 8% | ID tool | `resolved` | – | S1.1.6 |
| 7 | Not a human gene symbol | 8% | None | `invalid_input` | `out_of_scope_species` | S1.1.7 |
| 8 | Lookup tool unavailable | Failure share | Symbol tool | `error` | `tool_error` | S1.1.8 |
| 9 | Named multiplex, gene present | 10% | Symbol, membership | `resolved` | – | S1.1.9 |
| 10 | Named multiplex, gene absent | 7% | Symbol, membership | `resolved` (`not_found` before S1 v5) | – | S1.1.10 |
| 11 | Unknown named multiplex | 3% | Symbol, membership | `not_found` | `unknown_multiplex` | S1.1.11 |

- **Absent pool.**
  - Use previous HGNC symbols and aliases, including well-known aliases such
    as HER2.
  - Remove any string that matches a current registry symbol without regard
    to case. For example, VARS2 is a previous symbol of VARS1 and the
    approved symbol of another gene.
  - The answer never suggests the current symbol.
- **Other-species pool.** Use named mouse symbols from the hash-pinned Ensembl
  release 116 GRCm39 GTF in the runtime contract. Do not synthesize mouse
  spellings from human symbols.
- **Tool choice.** The input form selects the tool. An Ensembl gene ID goes to
  the ID tool.
- **Species.**
  - The task names another species: `invalid_input`, `out_of_scope_species`,
    and no call. The tools ignore case, so mouse `Gapdh` would match human
    GAPDH.
  - The task names no species: treat the gene as human.
- **Wording.** The tool maps a symbol to an ID. It does not verify the gene,
  so the thinking does not say "verified."
- **Gate** (the S0 floors, plus):
  - at most 1% calls on form 4, and at least 99% calls on form 5
  - the ID tool on at least 99% of form 6 rows
  - no call on form 7
  - no ID and no current symbol from memory on form 3
  - form 8: 100% `error` with the `tool_error` sentence, 0% `not_found`, and
    `"provenance": null`
  - forms 9–11: the membership call is required and summaries anchor the
    named multiplex

## S1.2 Ensembl to symbol

- **Book mode:** tool call. Some rows make no call.
- **Input:** one Ensembl gene ID, with or without a version suffix.
- **Tools:** `lookup_human_ensembl_gene_id`, and `lookup_human_gene_symbol`
  for a symbol input.
- **Result:** `{"gene_id": ..., "gene_symbol": ...}`.

The 11,000-row training split consists of a 10,000-row base allocation for
forms 1–10 plus 1,000 added transcript rows for form 12. The base shares below
sum to 100%. Form 11 replaces 5% of the base tool-call allocation in form 1;
it does not add rows. The same `direct`/`replace`/`first` failure policy as
S1.1 applies. With the 80/10/10 split fractions, validation and test each have
1,250 base rows, 125 transcript rows, and 1,375 rows in total.

| # | Form | Allocation | Example | Call | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|---|
| 1 | Direct | 20% of base, before failure replacement | ENSG00000255121 | ID tool | `resolved` | – | S1.2.1 |
| 2 | Inside a task | 15% of base | ENSG00000177830 | ID tool | `resolved` | – | S1.2.2 |
| 3 | Version suffix | 10% of base | ENSG00000099204.23 | ID tool | `resolved` | – | S1.2.3 |
| 4 | No symbol | 10% of base | ENSG00000300082 | ID tool | `resolved`, `gene_symbol` `null` | – | S1.2.4 |
| 5 | Not in release 116 | 10% of base | ENSG00000182378.14_PAR_Y, retired IDs | ID tool | `not_found` | `not_found` | S1.2.5 |
| 6 | Malformed ENSG | 5% of base | ENSG0000023134 | ID tool | `invalid_input` | `invalid_input` | S1.2.6 |
| 7 | Harness block lists the ID | 10% of base | ENSG00000230860 | None | `resolved` | – | S1.2.7 |
| 8 | Block without the ID (near miss) | 5% of base | ENSG00000177830 | ID tool | `resolved` | – | S1.2.8 |
| 9 | Symbol given as the input | 5% of base | CENATAC-DT | Symbol tool | `resolved` | – | S1.2.9 |
| 10 | Not a human ID | 10% of base | ENSMUSG00000112237 | None | `invalid_input` | `out_of_scope_species` | S1.2.10 |
| 11 | Lookup tool unavailable | 5% of base tool-call rows; replaces form 1 | ENSG00000141510 | ID tool | `error` | `tool_error` | S1.2.11 |
| 12 | Transcript ID, wrong type | 1,000 added train rows | ENST00000269305 | None | `invalid_input` | `wrong_input_type` | S1.2.12 |

- **v3 required negatives:**
  - a version suffix that must be accepted
  - a transcript accession
  - a non-human accession
  - a wrong digit count
  - an accession absent from release 116
  - a tool error

  A transcript accession is `invalid_input` with `wrong_input_type`, and
  makes no call. Take verified ENST IDs from the release 116 GTF.
- **Format check.** An input that starts with ENSG always goes to the tool.
  A non-gene or non-human Ensembl prefix (ENST, ENSP, ENSMUSG) gets no call.
- **No symbol is not a failure.** State that the gene has no known symbol, and
  show it by its ID. Do not refuse, do not invent a symbol, and do not call
  the ID invalid. Clone-based names from older annotations are not release
  116 symbols.
- **No-symbol share (S1 v2, 2026-10-08).**
  - In S1 v1, form 4 is 10% of the base allocation, so 14.1% of the resolved
    ID-to-symbol rows (1,000 of 7,100) have no symbol.
  - The v3 S0 corpus had 15.0%. The v3 S0 model invented a symbol in 5,205
    full-registry rows for genes without a symbol. The S0 v4 corpus has
    52.3%, and the S0 v4 model invented no symbol in the full-registry test.
    The full release 116 registry has 44.9%.
  - S1 continues the S0 v4 checkpoint, so a low share can undo the S0 v4
    behavior.
  - The S1 v2 overlay sets these base shares: form 1 9%, form 2 5%, and
    form 4 31%. The other forms keep their v1 shares. In every split, 43.7%
    of the resolved ID-to-symbol rows now have no symbol (3,100 of 7,100 in
    train).
  - Forms 1 and 2 give the shares because they repeat the main S0 skill.
    Form 1 keeps 500 train rows after the failure replacement, so each
    failure row keeps its minimal pair.
- **Tool over memory.** Sample renamed genes in forms 1 to 3. For example,
  ENSG00000154274 is PGCKA1, previously C4orf19.
- **Registry, not graph.** `resolved` means that the ID is in release 116. It
  does not mean that the gene is a multiplex node.
- **Retired IDs.** Take them from the pinned stable-ID history. Do not give a
  successor from memory.
- **Gate** (the S0 floors, plus):
  - the tool argument equals the user's text, suffix included
  - the answer symbol equals the tool symbol
  - form 4: no invented symbol and no refusal
  - forms 5 and 6: no ID, symbol, or successor from memory, and no repaired
    ID
  - at most 1% calls on form 7, and at least 99% calls on form 8
  - no call on form 10
  - form 11: the same floors as S1.1 form 8

## S1.3 Ambiguous symbol

- **Book mode:** tool call. Some rows make no call.
- **Input:** one gene symbol, from the S1.1 pool. A row is S1.3 when the
  lookup returns `ambiguous`.
- **Tools:** `lookup_human_gene_symbol` and `check_multiplex_membership`.
- **Result:**
  - Unresolved: `{"gene_symbol", "candidate_gene_ids", "candidate_details"}`.
    After a membership check, add `multiplex_id` and `present_gene_ids`.
  - Resolved: `{"gene_id", "gene_symbol", "candidate_gene_ids"}`. Add
    `multiplex_id` and `present_gene_ids` when the membership check decided.
  - Partial failure: the unresolved result, plus `failed_tool` and
    `error_code`.
  - From S1 v5, an answer after a membership check (forms 5 to 7) keeps
    every field of both tool results: `candidate_details` and both
    `present_gene_ids` and `absent_gene_ids`.
- **Allocation:** The training split has 9,000 rows.
  - Forms 1 to 4, 8, and 9 each have 1,000 rows.
  - The three membership outcomes have 3,000 rows before failure replacement.
  - Their source ratio gives 2,300 `keeps_one`, 242 `keeps_several`, and 458
    `keeps_none` rows.
  - Form 10 replaces 300 membership rows with paired failures.
  - Forms 5 to 7 keep 2,070, 218, and 412 successful rows, respectively.
  - Validation and test use deterministic largest-remainder apportionment from
    these training counts.

| # | Form | Context | Calls | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|
| 1 | Direct, no multiplex | None | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.1 |
| 2 | Inside a task, no multiplex | None | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.2 |
| 3 | Already marked | Block marks the symbol | None | `ambiguous` | `ambiguous_symbol` | S1.3.3 |
| 4 | Block for another symbol | Block marks another symbol | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.4 |
| 5 | Multiplex keeps one | Multiplex named | Symbol, membership | `resolved` | – | S1.3.5 |
| 6 | Multiplex keeps several | Multiplex named | Symbol, membership | `ambiguous` | `ambiguous_symbol` | S1.3.6 |
| 7 | Multiplex keeps none | Multiplex named | Symbol, membership | `ambiguous` (`not_found` before S1 v5) | `no_multiplex_membership` | S1.3.7 |
| 8 | User chooses at the next turn | Earlier `defer` answer | None | `resolved` | – | S1.3.8 |
| 9 | Reply does not choose | Earlier `defer` answer | None | `ambiguous` | `ambiguous_symbol` | S1.3.9 |
| 10 | Membership check fails | Multiplex named | Symbol, membership (error) | `ambiguous` | `tool_error` | S1.3.10 |

- Set the shares of forms 5 to 7 from the real membership counts.
- **Membership check.**
  - Call it only when the task or the harness names a multiplex. With no
    multiplex, defer.
  - Pass the exact candidate set.
  - One candidate present: `resolved`.
  - Several present: `ambiguous`. The summary offers only the present
    candidates.
  - None present: `ambiguous` (`not_found` before S1 v5), because nothing
    chooses one. The summary asks for another gene or another multiplex.
- **No choice from memory.** The model chooses a candidate in two cases only:
  - the membership check leaves exactly one
  - the user's reply selects exactly one
- **Basis.** A resolved answer names its basis: `multiplex_id`, or the user's
  choice. A membership result holds for that multiplex only.
- **Next turn, not an ask-user tool.**
  - `defer` ends the turn, and the user's reply is the next user turn.
  - The model reads the candidates from the earlier answer's `result`.
  - A reply with an ID that is not a candidate is the conflict case, which is
    not in S1.
- **Candidate details.** An `ambiguous` answer copies `candidate_details`, so
  that the user can choose.
- **Long lists.** Keep the S0 cap on candidate-list length. Above 5
  candidates, the summary anchors none and gives the count.
- **Ambiguity classes.** Stratify the pool by class:
  - PAR copies on X and Y: PPP2R3B (both protein-coding)
  - a pseudogene and a lncRNA with the same name: GOLGA2P9, FAM197Y5
  - multicopy families

  No current trace pairs a coding gene with a non-coding gene, and none has
  more than 2 candidates. See "Planned traces."
- **Gate** (the S0 floors, plus):
  - forms 1 to 4: 100% `defer`, no chosen candidate, and no membership call
  - forms 5 to 7 and 10: a membership call in at least 99% of rows, with the
    exact candidate set
  - form 5 resolves only to the one present ID
  - forms 8 and 9: the chosen ID is in the earlier candidate set, and the
    model defers when the reply selects none or more than one
  - form 10: `tool_error`, the lookup provenance kept, and no claim that a
    candidate is absent

## S1.4 Cross-context alignment

- **Book mode:** tool call. Some rows make no call.
- **Input:** two gene mentions from two items, one pair per question.
- **Question:** do the two mentions name the same gene? S1.4 teaches identity
  only:
  - Set comparisons belong to S2.6 and S5.12.
  - Parsing tool outputs belongs to S4.17 and S9.4.
- **Tools:** `lookup_human_gene_symbol` and `lookup_human_ensembl_gene_id`.
- **Result:**

```
{"first":  {"item": ..., "input": ..., "gene_id": ...},
 "second": {"item": ..., "input": ..., "gene_id": ...},
 "same_gene": true | false | null}
```

- `item` is one of `typed_seed_list`, `seed_gene_set`, `module`, `abstract`,
  or `query_mygene`.
- `gene_id` is `null` when that side does not resolve.
- An ambiguous side also carries `candidate_gene_ids` and `candidate_details`.
- `status` is `resolved` when `same_gene` is `true` or `false`. Otherwise
  `status` is the status of the side that blocks the answer, and the decision
  is `defer`.

| # | Form | Example pair | Calls | `same_gene` | Trace |
|---|---|---|---|---|---|
| 1 | Symbol and ID, match | GLP1R, ENSG00000112164 | Symbol lookup | `true` | S1.4.1 |
| 2 | Symbol and ID, no match | MDM2, ENSG00000198625 (MDM4) | Symbol lookup | `false` | S1.4.2 |
| 3 | ID and ID, notation only | ENSG00000204394.15, ensg00000204394 | None | `true` | S1.4.3 |
| 4 | ID and ID, near miss | ENSG00000135679, ENSG00000135678 | None | `false` | S1.4.4 |
| 5 | Alias or previous symbol | HER2, ENSG00000141736 | Symbol lookup (`not_found`) | `null` | S1.4.5 |
| 6 | Ambiguous symbol | PPP2R3B, ENSG00000167393 | Symbol lookup (`ambiguous`) | `null` | S1.4.6 |
| 7 | Registry block given | TP53, ENSG00000141510 | None | `true` | S1.4.7 |
| 8 | External ID from MyGene | HER2 via query_mygene, ENSG00000141736 | ID lookup | `true` | S1.4.8 |
| 9 | Registry backend unavailable | TP53, ENSG00000141510 | Symbol lookup (`error`) | No pair result | S1.4.9 |

- **Build allocation.** Forms 1 to 8 start with 1,000 training rows each.
  Form 9 replaces 250 form-1 rows and keeps each paired success row.
- **Ambiguous balance.** Form 6 uses 500 candidate targets and 500
  noncandidate targets in training. Held-out splits use largest-remainder
  rounding in that order.
- **Candidate limit.** Include ambiguous symbols with at most 49 candidate
  gene IDs.
- **MyGene source.** Form 8 uses the frozen, content-addressed cache in the
  runtime contract. A usable hit maps one alias to one current release-116
  ID and agrees with its current symbol.
- **Failure result.** Form 9 uses the shared tool-failure payload. It does not
  emit the pair result because the failed lookup cannot resolve one side.

- **Compare in the Ensembl ID space.** Look up a symbol even when the other
  item also shows a symbol.
- **ID and ID.** Two IDs name the same gene when they are equal after the
  `.N` suffix is removed and case is ignored. This rule needs no call. Use
  well-formed IDs only.
- **No mapping from memory.** A mapping from an alias must come from a tool
  result (form 8). The model then checks the ID in release 116.
- **Ambiguous symbols.**
  - If the other ID is a candidate, `same_gene` is `null`.
  - If the other ID is not a candidate, `same_gene` is `false`.
- **Calls.** At most one lookup for each side.
- **Gate** (the S0 floors, plus):
  - exact `same_gene`, an exact `gene_id` on each side, and the correct status
  - a symbol lookup in at least 99% of the rows of forms 1, 2, 5, and 6
  - an ID lookup in at least 99% of the rows of form 8
  - at most 1% calls on forms 3, 4, and 7
  - 100% `defer` when `same_gene` is `null`

## S1.5 Context ID resolution

**Blocked** until the context ontology data exists.

- **Book mode:** tool call.
- **Call:** `resolve_context_ontology({"context_id": ...})`. The tool returns:
  - anatomy (UBERON, and HBA where present)
  - cell type (CL, and PCL for transcriptomic types)
  - taxonomy cluster ID
  - developmental stage (HsapDv)
  - `pen_layers`
  - `ontology_versions` in its provenance
- **Result:** the `context_id`, and each term copied from the tool result.
- **Required negatives:**
  - An unknown context ID: `not_found` and `defer`. Invent no terms.
  - A partially mapped context: `null` for the missing field, for example no
    PCL term.
- Register the tool in the RWR++ tool inventory first.

## S1.6 Layer tag parsing

- **Book mode:** closed book. The prompt lists the S1 tools, as on every S1
  row. The model makes no call.
- **Input:** one layer tag, or a short list of tags. No genes and no graph
  content.
- **Result (`LayerRef`):** all seven keys, always.

```
{"layer_id": ..., "layer_family": ..., "source": ..., "network": ...,
 "organ": ..., "region_path": [...], "cell_type_path": [...]}
```

- A key that the grammar does not fill is `null`.
- The list form returns `{"layers": [LayerRef, ...], "filter": {"field": ...,
  "value": ...}, "matching_layer_ids": [...]}` (S1.6.4).

**Grammars.** The first colon-separated field selects the grammar. The build
reads native tags from the pinned flist. Do not rename them.

| First field | Parts | Arity | `layer_family` |
|---|---|---|---|
| `HumanNetV3` | source : network | 2 | From the network field (see S1.7) |
| `bulkPEN` | prefix : organ : region path (1–2 parts) | 3–4 | `bulk_pen` |
| `scPEN` | prefix : organ : region : cell-type path (1–2 parts) | 4–5 | `scpen` |
| `TFs` | prefix : organ | 2 | `tf_target` |

- Preserve the native layer ID byte for byte. Do not add a family field or a
  synthetic version suffix.
- HumanNet tags fill `source` and `network`; the context paths are `null`.
- bulkPEN tags fill `organ` and `region_path`. For example,
  `bulkPEN:brain:basal_ganglia:caudate` has region path
  `["basal_ganglia", "caudate"]`.
- scPEN tags fill `organ`, the one-part `region_path`, and a one- or two-part
  `cell_type_path`. For example,
  `scPEN:brain:pfc:inhibitory_interneuron:pv` keeps the two cell-type parts.
- `TFs:brain` fills `organ: "brain"`; the path fields are `null`.
- The normative grammars are in `runtime/world_model/contract.yaml`.
- Layer tags and context addresses are separate namespaces. A string such as
  `brain:cortex:L5:microglia:v1` addresses a context, not a layer.

**Frozen native tag form (2026-10-01).** The 353-layer canonical flist is the
source of layer IDs. All 353 tags pass the variable-depth grammars above. The
golden traces use `HumanNetV3:string_ppi`, `bulkPEN:brain:amygdala`,
`scPEN:brain:hippocampus:astrocyte`, and `TFs:brain`. A parse copies hyphens,
underscores, and path depth; it does not normalize spelling.

| # | Form | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|
| 1 | Direct parse | `resolved` | – | S1.6.1 (bulkPEN), S1.6.3 (HumanNet), S1.6.7 (TFs) |
| 2 | One field | `resolved` | – | S1.6.2 |
| 3 | List filter by field | `resolved` | – | S1.6.4 |
| 4 | Wrong type (a module ID) | `invalid_input` | `wrong_input_type` | S1.6.5 |
| 5 | Malformed tag | `invalid_input` | `invalid_input` | S1.6.6 (missing cell-type path) |

- **Build shares:** 40% direct parse, 25% one field, 15% list filter, 10%
  wrong type, and 10% malformed tag.
- The 5,000 training rows use all seven reviewed trace classes.
  - S1.6.1 has 1,250 bulkPEN direct parses.
  - S1.6.2 has 1,250 scPEN field questions.
  - S1.6.3 has 625 HumanNet direct parses.
  - S1.6.4 has 750 list filters.
  - S1.6.5 has 500 wrong-type inputs.
  - S1.6.6 has 500 malformed tags.
  - S1.6.7 has 125 TFs direct parses.
- Each held-out split has 625 rows. Its seven trace counts are
  167, 156, 83, 94, 63, 62, and 0.
- The TF trace stays in training because the pinned source has one TF tag.
- **Required malformed-tag cases:**
  - an unrecognized prefix
  - the right prefix with the wrong arity
  - a missing region or cell-type path
  - more path parts than the native family permits
  - a synthetic family or version field that is not in the flist ID
- The malformed bucket also covers mixed separators.
- S1.6.5 uses source-backed RWR-LOE and MENTOR-EV module IDs.
  Context-ID negatives wait for the S1.5 context source.
- S1.6.2 balances questions across family, source, network, organ, region,
  and cell-type fields. Its answer still returns the complete `LayerRef`.
- **Rules:**
  - **Read the tag only.** A parse does not say that a layer is in a
    multiplex, and it makes no claim about edges.
  - **No inference.** Do not infer a part from biology. The region path of
    `scPEN:brain:amygdaloid-complex:astrocyte` is
    `["amygdaloid-complex"]`, and a filter for `amygdala` does not match it
    (S1.6.4).
  - **Copy values exactly.** Keep case, underscores, hyphens, and path depth.
    The one derived value is `layer_family`, from the S1.7 enum.
  - **Exact filters.** List filters compare field values exactly.
    Path filters compare one component exactly. Each list contains three
    same-split native tags and has equal match and no-match coverage.
  - **Family agreement.** The `layer_family` of a parse agrees with S1.7.
  - **Thinking.** One Thinking step reads the tag in order: the prefix, the
    grammar, and each part.
- **Splits:** hold out whole contexts.
  - bulkPEN and scPEN use the exact organ and region path as the context key.
  - HumanNet uses its network as the context key.
  - TFs uses its single native tag as one context key.
  - The seeded assignment gives 27 training, 4 validation, and 4 test units.
  - Each held-out split has one HumanNet, bulk-only, scPEN-only, and shared
    PEN context.
  - A list contains tags from one split only.
  - Singleton families are not copied into held-out splits.
- **Gate:**
  - at least 99% field accuracy on held-out tags
  - dispatch accuracy by family, with a confusion matrix
  - no inferred value
  - at least 99% correct `invalid_input` and `blocked_reason`
  - at most 1% tool calls

## S1.7 Layer family and supported claims

- **Book mode:** closed book. The prompt lists the S1 tools, as on every S1
  row. The model makes no call.
- **Input:** one layer tag, a family name, or a short list of tags. No genes
  and no edge records.
- **Result (family card):**

```
{"layer_id": ..., "layer_family": ..., "evidence_type": ...,
 "supported_claims": [...], "unsupported_claims": [...]}
```

- A check of one claim adds `"claim"` and `"claim_supported"`.
- The list form returns `{"layer_family": ..., "matching_layer_ids": [...],
  "other_layer_ids": [...]}`.
- `layer_id` is `null` when the question names a family.
- **`layer_family` enum:** `ppi`, `coexpression`, `tf_target`, `bulk_pen`,
  `scpen`, `other`.
- **Where the family comes from:**
  - A PEN or TF prefix settles the family. Later fields describe its context.
  - A HumanNetV3 network settles the family. `string_ppi` gives `ppi`, and
    `coexpression` gives `coexpression` in the pinned source.
  - The 5 other native HumanNetV3 networks give `other`.
  - `biogrid_ppi` would give `ppi`, but the pinned flist has no such tag.
  - Do not create a positive tag that is absent from the flist.

**Claim vocabulary.** Always write claims in this order:

1. `context_predictive_expression`
2. `physical_interaction_recorded`
3. `regulatory_link_recorded`
4. `regulation_direction`
5. `global_coexpression`
6. `cofunctional_association`
7. `holds_in_other_contexts`
8. `causal_effect`

`supported_claims` and `unsupported_claims` partition all eight claims, with
no overlap and nothing left out. Each list keeps the vocabulary order.

| `layer_family` | `evidence_type` | `supported_claims` | Trace |
|---|---|---|---|
| `ppi` | `physical_interaction_recorded` | `physical_interaction_recorded` | S1.7.6 |
| `coexpression` | `global_coexpression` | `global_coexpression` | S1.7.3, S1.7.4 |
| `tf_target` | `regulatory_link_recorded` | `regulatory_link_recorded` | S1.7.9 |
| `bulk_pen` | `context_predictive_expression` | `context_predictive_expression` | S1.7.7 |
| `scpen` | `context_predictive_expression` | `context_predictive_expression` | S1.7.1 |
| `other` | `null` | None | S1.7.8 |

- **`tf_target` as loaded.** The one TF layer in the repo,
  `TFs:brain`, has no sign and no direction (see "Repo data
  findings"):
  - Its edges support `regulatory_link_recorded`.
  - They do not support `regulation_direction`.
  - They do not show which gene is the TF.

  No signed TF layer is planned for now. So `regulation_direction` is
  unsupported for every `tf_target` layer.
- `cofunctional_association`: an integrated functional network would support
  it. No current family card supports it. The remaining HumanNet networks
  stay in `other` and support no claims.
- `holds_in_other_contexts` and `causal_effect` are unsupported for every
  layer:
  - The first needs a comparison across multiplexes.
  - The second needs an intervention.
- A PPI edge reports what a database asserts, not what happens in the
  context.
- A coexpression edge is aggregate evidence, so it says nothing about one
  context.
- The non-PEN layers are induced on the context gene universe, but their
  evidence stays global.

| # | Form | Status | Trace |
|---|---|---|---|
| 1 | Family of a layer | `resolved` | S1.7.1 |
| 2 | List filter | `resolved` | S1.7.2 |
| 3 | Claims of a layer | `resolved` | S1.7.3, S1.7.6, S1.7.7, S1.7.9 |
| 4 | Check one claim (hard negatives) | `resolved` | S1.7.4 |
| 5 | Wrong type | `invalid_input`, `wrong_input_type` | S1.7.5 |
| 6 | Native nonmapped network | `resolved`, family `other` | S1.7.8 |

**Exact row allocation.** Validation and test each use the evaluation count.

| Allocation | Trace | Train | Evaluation |
|---|---|---:|---:|
| `scpen_family` | S1.7.1 | 1,500 | 188 |
| `list_filter` | S1.7.2 | 900 | 113 |
| `coexpression_claims` | S1.7.3 | 375 | 47 |
| `claim_check` | S1.7.4 | 1,500 | 187 |
| `wrong_input_type` | S1.7.5 | 300 | 37 |
| `ppi_claims` | S1.7.6 | 375 | 47 |
| `bulk_pen_claims` | S1.7.7 | 375 | 47 |
| `other_claims` | S1.7.8 | 300 | 37 |
| `tf_target_claims` | S1.7.9 | 375 | 47 |
| **Total** | | **6,000** | **750** |

S1 v5 training uses 1,200 `scpen_family` rows and 600 `other_claims` rows.
All other allocations and all evaluation counts stay as in the table.

The base six semantic buckets contain 1,500 family rows, 900 list rows, 1,500
claim cards, 1,500 claim checks, 300 wrong-type rows, and 300 `other` rows.

- **List filters:**
  - Every list has 3 native tags from one split.
  - Training asks about each family 150 times.
  - Training has 450 matches and 450 no-matches.
  - Each evaluation split asks about `bulk_pen` 38 times, `scpen` 38 times,
    and `other` 37 times.
  - Each evaluation split has 57 matches and 56 no-matches.
- **Claim cards:**
  - Each training card allocation uses a family name 75 times.
  - Each held-out coexpression, PPI, and TF allocation uses its family name
    for all 47 rows.
  - Held-out bulk-PEN cards use native held-out tags.
- **Claim checks:**
  - Training uses each family 250 times and each family name 50 times.
  - Each non-`other` family has 120 supported checks. `other` has none.
  - Training has 600 supported and 900 unsupported checks.
  - Each evaluation split has 75 supported and 112 unsupported checks.
  - Each evaluation split uses PPI 32 times and each other family 31 times.
  - Each non-`other` family has 15 supported evaluation checks.
  - Held-out PPI, coexpression, and TF checks use family names.
  - Held-out PEN and `other` checks use native held-out tags.
  - Allocate other unsupported checks by largest remainder in claim order.
  - Include coexpression versus physical interaction.
  - Include PPI versus regulatory link.
  - Include TF target versus regulation direction.
  - Include each PEN family versus `holds_in_other_contexts`.
  - Include every family versus `causal_effect`.
  - Do not use the scoped bulk-PEN single-cell negative.
- **Generated wrong-type rows:** use only source-backed `mentor_ev` module
  IDs. Training has 300 rows, and each evaluation split has 37.
- **`other` sources:** use `cocitation`, `database`, `gene_interaction`,
  `gene_neighbor`, and `phylogenetic`. Assign 3 to train and 1 to each
  held-out split.
- **Context splits:**
  - Use the S1.6 key: exact PEN organ and region path, or HumanNet network.
  - Train uses 27 contexts: 5 HumanNet, 9 bulk-only, 11 scPEN-only, 1 shared
    PEN, and 1 TF context.
  - Each held-out split uses 4 contexts: 1 HumanNet, 1 bulk-only, 1
    scPEN-only, and 1 shared PEN context.
  - Keep the only native TF tag in train.
  - Family names use the shared closed vocabulary across splits.
- **Coverage:** the corpus covers all six families.
- **Rules about layers, not facts about genes.** The summary names no gene.
  S2.13, S3.4, and S7.5 apply this table to real edges, paths, and
  projections.
- **Gate:**
  - at least 99% family accuracy on held-out tags, with a confusion matrix
  - exact claim sets in vocabulary order
  - 0% leak: no unsupported claim marked as supported
  - at least 99% accuracy on held-out claim paraphrases
  - at most 1% tool calls

## S1.8 Direction conventions

**Ready.** The corpus uses the fixed direction table and split-specific name
sets in `runtime/world_model/contract.yaml`.

- **Book mode:** closed book. The prompt lists the S1 tools, as on every S1
  row. The model makes no call.
- **Input:** a metric name, or a true-or-false statement about a metric. No
  genes and no values.
- **Result:**

```
{"metric": ..., "input": ..., "lower_is_stronger": true | false | null,
 "meaning": "proximity" | "similarity" | "evidence" | null,
 "rule": ..., "statement_true": true | false | null}
```

- An ambiguous name also carries `"candidate_metrics": [...]`.

**Convention table (the closed set).** The canonical names are the payload
keys of the runtime tools.

| `metric` | Name variants | `lower_is_stronger` | `meaning` | `rule` |
|---|---|---|---|---|
| `rank` | RWR rank, RWR-LOE rank, random walk with restart rank | `true` | `proximity` | `lower_rank_closer` |
| `score` | RWR score, restart score, visiting probability | `false` | `proximity` | `higher_score_closer` |
| `spearman_distance`, `pearson_distance` | Spearman distance, 1 − Spearman correlation | `true` | `proximity` | `lower_distance_closer` |
| `spearman_correlation`, `pearson_correlation` | Spearman ρ, Pearson r | `false` | `similarity` | `higher_correlation_more_similar` |
| `dot_similarity` | dot similarity, RWR dot product | `false` | `similarity` | `higher_similarity_more_similar` |
| `distance_percentile` | distance percentile | `true` | `proximity` | `lower_distance_percentile_closer` |
| `score_percentile` | score percentile | `false` | `proximity` | `higher_score_percentile_closer` |
| `hop_count` | path length, number of hops | `true` | `proximity` | `fewer_hops_closer` |
| `empirical_p_value`, `q` | p-value, FDR q-value | `true` | `evidence` | `lower_p_stronger_evidence` |
| `enrichment_score` | enrichment score | `false` | `evidence` | `higher_enrichment_stronger_evidence` |

| # | Question class | Train | Validation | Test | Status | Trace |
|---|---|---:|---:|---:|---|---|
| 1 | Direction lookup | 1,200 | 150 | 150 | `resolved` | S1.8.1 |
| 2 | Distance-style true or false | 667 | 84 | 84 | `resolved` | S1.8.2 |
| 3 | Similarity-style true or false | 667 | 83 | 83 | `resolved` | S1.8.3 |
| 4 | Workflow-style true or false | 666 | 83 | 83 | `resolved` | S1.8.4 |
| 5 | Ambiguous percentile name | 400 | 50 | 50 | `ambiguous` | S1.8.5 |
| 6 | Unrecognized metric | 400 | 50 | 50 | `not_found` | S1.8.6 |

- The train split has 4,000 rows. Each evaluation split has 500 rows.
- The resolved rows use a noncanonical name in exactly 30% of cases.
- Metric variants and shared aliases do not cross splits.
- The base corpus has three metric variants per metric in each split.
  The v5 training bank has 12; validation and test retain three.
- **A bare name with several candidates:**
  - If the candidates agree, use `resolved` and set `metric` to null.
  - Add the ordered candidates and the shared direction fields.
  - For example, "distance" maps to both correlation-distance metrics.
  - The candidates disagree: `ambiguous`, with `candidate_metrics`.
  - A percentile alias maps to `distance_percentile` and `score_percentile`.
  - These candidates have opposite directions, so all direction fields are
    null.
- **True-or-false balance:**
  - Train has 1,000 true statements and 1,000 false statements.
  - Each evaluation split has 125 true statements and 125 false statements.
  - Each single metric uses all four direction polarities.
- Canonical metric names can occur in all splits.
- Noncanonical metric names, percentile aliases, and unknown names are
  split-disjoint.
- **No assumptions.** The model answers from the table only, also for a
  metric that it may know from the literature.
- **`lca_depth` is not in S1.8.** S5 defines it.
- **Runtime rule.** Public `get_distance` accepts only Spearman or Pearson
  distance. `get_dot_similarity` handles the dot product.
- **Gate:**
  - 100% correct direction for each canonical metric
  - at least 99% true-or-false accuracy for canonical names, by metric and polarity
  - report held-out name and alias mapping separately; do not gate it
  - 100% `defer`, with no direction claim, on ambiguous and unrecognized rows
  - the masked-wording classifier near 50%
  - at most 1% tool calls

## S1.9 Module ID parsing

**Ready.** Use the all-gene MENTOR-EV tree at
`/lustre/orion/syb111/proj-shared/Projects/mentor-ev/brain/full_brain`.

- **Book mode:** closed book. The prompt lists the S1 tools. The model makes
  no call.
- **Forms:**

| # | Allocation | Train | Evaluation | Status | Trace |
|---|---|---:|---:|---|---|
| 1 | `rwr_loe_parse` | 1,000 | 125 | `resolved` | S1.9.1 |
| 2 | `mentor_ev_parse` | 1,000 | 125 | `resolved` | S1.9.2 |
| **Total** | | **2,000** | **250** | | |

Validation and test each use the evaluation count.

- **RWR-LOE result shape:**

```
{"module_id": "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000112164:geometric_elbow_v1",
 "module_family": "rwr_loe", "multiplex_id": "full_brain_multiplex_v1",
 "seed_gene_id": "ENSG00000112164",
 "threshold_recipe": "geometric_elbow_v1", "existence_asserted": false}
```

- **MENTOR-EV result shape:**

```
{"module_id": "mentor_ev:full_brain_all_genes_v1:clade_000417",
 "module_family": "mentor_ev",
 "dendrogram_id": "full_brain_all_genes_v1",
 "clade_id": "clade_000417", "existence_asserted": false}
```

- The RWR-LOE grammar is
  `rwr_loe:full_brain_multiplex_v1:seed_<human Ensembl gene ID>:geometric_elbow_v1`.
- The RWR-LOE source pool has 47,017 IDs from `multiplex_gene_ids`.
- The MENTOR-EV grammar is
  `mentor_ev:full_brain_all_genes_v1:clade_<six digits>`.
- The MENTOR-EV source pool has 15,679 eligible records from `mentor_clades`.
- Eligible clades contain 5–30 leaves.
- Reuse the deterministic S1.6 module split with assignment seed 1729.
- The split orders each namespace by seeded SHA-256 and uses its index modulo
  10.
- Residue 0 goes to validation. Residue 1 goes to test. Residues 2–9 go to
  train.
- The RWR-LOE pool has 37,613 train, 4,702 validation, and 4,702 test IDs.
- The MENTOR-EV pool has 12,543 train, 1,568 validation, and 1,568 test IDs.
- Exact module IDs never cross splits.
- Each form has equal direct and step framing in train.
- Each evaluation split has 125 direct and 125 step rows in total.
- Direct rows stop. Step rows continue.
- S1.9 has no invalid, ambiguous, unrecognized, or unobserved-valid rows.
- Tokenize each ID compositionally. Keep the source and fixed IDs atomic.
  Use digit tokens for the clade index.
- A parse never says that a module exists.
- A clade ID that is not in the context is recall, which the recall rule
  forbids.
- Every answer has `provenance: null` and `blocked_reason: null`.
- **Gate:**
  - 100% exact parse fields; compare the result as a mapping
  - 100% `existence_asserted: false`
  - 100% exact split isolation
  - the masked-wording classifier near 50%
  - at most 1% tool calls

## Stage gate

The v3 S1 gate:

- On the identity families, tool-call correctness:
  - selection accuracy
  - schema-valid arguments
  - faithful reading of the result
  - correct deferral
- Near-perfect accuracy on the closed-book families.

Measure it with these floors, on held-out phrasings and held-out split units.

**Tiered gate v3 (2026-10-11).** Test configs select it with
`"gate": {"profile": "tiered_v3", "require_network_used_false": true}`. The
targets are fixed in `scripts/world_model_s1_eval_common.py`, so a config
cannot weaken them. The summary text is never scored.

- **Why it changed.** `tiered_v2` compared observed rates with 99.5%. A
  model whose true accuracy is exactly 99.5% fails each large check about
  half the time, a subset of 40 or 80 rows allows no error, and
  direction-by-metric (13 groups of about 28 rows) fails 89% of the time.
  Held-out names also need a different bar from mechanical checks.
- **Rule.** A check fails only when its 95% Wilson interval lies wholly on
  the wrong side of its target. The gate records each check's count,
  total, interval, target, and tier under `gate.details`.
- **Protocol (99.5%):** valid trajectory, valid final JSON, tool arguments,
  tool payloads, evidence, no calls on closed-book rows, no network (100%),
  and at most 0.5% unexpected direct answers.
- **Format floors (99%):** reasoning present, and reasoning after each tool
  result.
- **Decisions (98%):** tool selection, `status`, `result`,
  `continuation_decision`, `blocked_reason_presence`, family contracts,
  ambiguous deferral, failure handling, direction by metric, and direction
  truth by metric and polarity.
- **Generalization (90%):** the layer-tag and layer-claim families.
- **`blocked_reason` is scored by presence.** A row passes when the answer
  gives a non-empty `blocked_reason` exactly when the answer key does. The
  sentence is free text, like the summary, so its wording is not scored.
- **Canonical metric names only.** The runtime gives the model a metric's
  canonical name, so the gate scores only those rows. Rows that use a
  held-out name variant or shared alias are reported, not gated, with
  `confident_wrong_direction` (a name answered with the opposite
  `lower_is_stronger`; a deferral is safe).
- **Headline metric.** `structured_answer_accuracy`: the final answer
  matches exactly except for `summary`.
- **S1.9 module results.** The answer key stores result keys sorted; the
  model writes them in training order. The module contract now compares the
  result as a mapping, so key order no longer fails the row.
- **Model errors.** If the model invents a malformed multiplex ID, the tool
  echoes it in the provenance. The row now fails evidence consistency; the
  run no longer stops.

**Tiered gate (S1 v2 tests, 2026-10-09).** The S1 v2 test schema
(`mentor-rl-world-model-s1-test-v2`) uses the gate profile `tiered_v2`. It
follows the two tiers of the S0 gate (see `S0-human-gene-identifiers.md`).
Each rate is pooled over all test rows, not computed for each form.

- **Strict tier (99.5%):** valid trajectory; exact tool selection, arguments,
  and payloads; valid final JSON; evidence copied exactly; `status`; `result`;
  the family contract invariants (for example, no invented identifier, the
  null symbol kept, and no choice among ambiguous genes); ambiguous deferral;
  failure handling; no calls on closed-book rows; the layer-family, claim,
  direction, held-out metric, and shared-alias checks; at most 0.5% direct
  answers; and no network use (100%).
- **Target tier (90%):** `continuation_decision` and the `blocked_reason`
  sentence. The `blocked_reason` choice was made for these tests; change it
  if the sentence must match exactly.
- **Format floors (99%):** thinking present, and thinking after each tool
  result.
- **Reported only:** the whole answer, the summary text, the exact behavior
  for each family, and the envelope accuracy for each form. Review a sample
  of summaries by hand.

The configs are `config/world_model_s1_120b_test_v2_512n.json` and
`config/world_model_s1_120b_184n_test_v2_512n.json`, one for each S1 v2
training config. Without `--gate-profile tiered_v2`, the evaluator still
applies the promotion floors below, so the S1 v1 test config is unchanged.

Promotion floors (S1 v1 test schema):

- **The S0 v6 floors:**
  - at least 99% valid trajectories, reasoning present, exact tool
    arguments, and exact tool payloads
  - 100% correct ambiguous deferral
  - at most 1% direct answers
  - no network use
- **Envelope:**
  - at least 99% exact `status`, `result`, and `continuation_decision` for
    each form
  - `blocked_reason`: the filled template on every `defer` row, and `null`
    on every other row
- **Provenance:**
  - the key present in every answer
  - the `evidence_id` copied exactly
  - each populated channel equal to its registry or graph evidence
  - `null` on every row with no successful evidence, so 0 invented blocks
- **Convention rows:** at most 1% tool calls, with the tools listed.
- **Tool failure:**
  - 0% `not_found` after a failed call
  - `status: "error"` with the `tool_error` sentence when the answer rests on
    failed calls alone
  - the partial-failure sentence on every partial failure
- **Thinking:**
  - thinking after each tool result in at least 99% of rows
  - the runtime filter passes
  - exact thinking text as a diagnostic only
- **Summary:** the summary rule checks pass. The evaluator compares the exact
  text. The planned gate removes this check.
- **The family gates above.**
- Report the false-defer rate: rows where the model deferred but the task was
  answerable.

## Validation

- The golden JSONL and every generated row pass `validate.py` against
  `runtime/world_model/contract.yaml` before entering the corpus. QC is
  all-or-nothing. The Markdown question file is a generated review view.
- Run `selftest.sh` after any change to the contract, and `mutate.sh` after
  any change to `validate.py`. `selftest.sh` checks that the right check
  catches each broken fixture. `mutate.sh` turns off each `flag(...)` call in
  turn and checks that a fixture notices. Keep each `flag(...)` call on one
  line, so that each mutant is a real one.
- A generated corpus is JSONL in the row form of
  `runtime/world_model/contract.yaml`.
  The validator reads the hidden metadata of each row. So for a row it also
  checks the `blocked_reason_code`, the label slots (`found_input_type`,
  `input_species`), the row form, and the prompt rules below.
- The kit validated the advisor's file, but it did not catch the problems
  listed in "Trace fixes." The kit now has a check, and a broken fixture in
  the selftest, for each of these (2026-10-01):
  - two traces with the same visible input and the same tool outcomes. The
    visible input is the harness context, the earlier turn, and the user
    text. The tool outcomes are the tool and the status of each call.
    `evidence_id` values do not count. Two rows can share user text on
    purpose: a failure row repeats the prompt of its success row (S1.1.1 and
    S1.1.8), and a near-miss row repeats a prompt under another harness block
    (S1.3.1 and S1.3.4).
  - a `blocked_reason_code` that does not fit the `status` (see the
    `blocked_reason` table)
  - a `blocked_reason` sentence that differs from its filled template
  - a tool result with no Thinking step after it
  - a provenance block whose type does not fit the tool that produced it
  - an answer or a tool result with no `provenance` key
  - an answer whose `evidence_id` differs from its evidence
  - an `evidence_id` that appears in more than one row
  - a system prompt or a tool list that differs between rows, and a hidden
    field name or value (book mode, step, question family, source trace) in
    the visible text. These need generated rows.
  - an anchor type that is not in the anchor list
  - claim lists that are not in vocabulary order
  - a summary that asks for input when the decision is not `defer`, and a
    `defer` summary that asks for nothing. The phrases that ask are
    `input_request_phrases` in the canonical contract.
  - a summary with an anchor that is not in the result
  - a failure result that does not use `failed_tool`
  - a tool result whose keys differ from the canonical return shape for its
    status
- The visible-field rule also has a statistical part that one row cannot
  show. The wrapper gate, full-prompt diagnostic, and structure gates cover
  it (see "Generation rules").

## Trace fixes

The updated question file applies these fixes to the advisor's traces
(2026-09-30):

1. **S1.8.6** is now the unknown-metric trace ("diffusion state distance",
   `not_found`, `defer`). It repeated S1.8.5 before.
2. **S1.3.10:**
   - The first Thinking step no longer plans the membership check.
   - A Thinking step now follows the lookup result.
   - It uses `full_brain_multiplex_v1`.
   - The summary anchors the named multiplex with `<MULTIPLEX:…>`.
3. **S1.1.8, S1.2.11, and S1.3.10:** the thinking is shorter and in STE
   style, and it does not mention provenance. S1.1.8 no longer says "stop."
4. **S1.7.8:**
   - The network is `cocitation`, not the placeholder "NETWORK_A."
   - The decision stays `stop`, and the summary no longer asks for input.
5. **S1.7.3 and S1.7.4** list the claims in vocabulary order.
6. **S0.4 to S0.7** use real source-backed examples: ENSG00000299624,
   GOLGA2P9, LINC02170, and the retired ID ENSG00000281376.
7. **Envelope.**
   - Every tool result and every answer has `provenance`, with an
     `evidence_id` for each call.
   - The registry blocks are JSON.
   - Every `defer` answer has a `blocked_reason` sentence, and every other
     answer has `null`.
8. **Tag form.** S1.6.1, S1.6.4, and S1.7.7 use native flist IDs such as
   `bulkPEN:brain:amygdala`. S1.6.4 tests a filter with no match.
9. **New traces:** S1.6.7 (TFs parse) and S1.7.9 (`tf_target` claims).
10. **Formatting:** the S1.3.9 earlier-turn label and answer had stray
    backticks.
11. **Guidance bullets:**
    - The S1.1 "Absent-symbol pool" bullet is filled.
    - `invalid_id` is now `invalid_input`.
    - The S1.4 bullets are filled.
    - The S1.8 question marks are removed.

    This document still replaces the guidance bullets where they differ.

Fixes of 2026-10-01:

12. **S1.3.10:** the partial-failure result now has `"multiplex_id"`, so
    every slot of its `blocked_reason` sentence comes from the result.
13. **S0.5:** the summary says "2 genes", not "two genes". Numbers are
    digits in every summary and every `blocked_reason` sentence.
14. **scPEN tag form.** S1.6.2, S1.6.4, S1.6.6, S1.7.1, and S1.7.2 use real
    native tags from the canonical flist.
    - S1.6.2, S1.7.1, and S1.7.2 use
      `scPEN:brain:hippocampus:astrocyte`. S1.6.6 omits the cell-type path.
    - S1.6.4 still has no match, and it still tests the no-inference rule: it
      filters for the region `amygdala`, and the scPEN tag has the region
      `amygdaloid-complex`.
15. **S1.3.6:** GOLGA2P9 has only 1 candidate in the
    `full_brain_multiplex_v1` store (ENSG00000269332), so the trace did not
    hold. It now uses BMS1P4, with the same pair class on chromosome 10:
    ENSG00000242338 (transcribed unprocessed pseudogene) and ENSG00000271816
    (lncRNA). Both occur in the canonical 47,017-gene multiplex universe.
16. **S1.1.10:** GLP1R is a node in the canonical multiplex, so the absent
    membership trace now uses IL17A (ENSG00000112115), which is in release
    116 and absent from the 47,017-gene universe.

Membership results now use the graph provenance channel; answers that also
use an identifier lookup merge the registry and graph channels.

## Differences between v3 and the traces

The traces win in each case. Update the v3 text to match.

| Topic | v3 text | Traces |
|---|---|---|
| Registry tool result | Payload nested under `result` | Payload flat, beside `status` and `provenance` |
| Membership result | `gene_id` when one candidate survives, `candidate_gene_ids` when several do | `present_gene_ids` and `absent_gene_ids`, as in v3 `get_nodes_by_layer` |
| S1.4 answer | `same_entity`, `gene_id`, `gene_symbol`, `reason` | `first`, `second`, `same_gene` |
| S1.8 answer | `lower_is_closer`, `rule` | `metric`, `input`, `lower_is_stronger`, `meaning`, `rule`, `statement_true` |
| `summary` | One sentence | One to three sentences |
| S1.1 and S1.2 validators | A `canonical_entity` field | No such field |
| S1.3 | The multiplex route is "still to be written" | S1.3.5 to S1.3.7 and S1.3.10 exist |
| S1.5 | A four-field envelope | Identity answers carry `provenance` |
| S1 heading | "closed-book" | Mixed book mode, as the v3 curriculum says |
| Error codes | The list includes `UNKNOWN_GENE` and `OUT_OF_SCOPE_SPECIES`, but `error` is for transport failures only | Registry tools emit only `BACKEND_UNAVAILABLE`. Species is an answer-level `blocked_reason`, with no call. |
| Release value | `"116"` | The same. The runtime uses `"Ensembl_116"`, and the registry file uses `116`. |

## Decisions that differ from the advisor's sources

Peter made these decisions after the advisor's review. They hold until the
advisor reviews them. Thinking and summaries keep the advisor's fixed
wording for now. A later move to model-written wording is planned.

| Topic | Advisor's sources | This spec | Reason |
|---|---|---|---|
| Tools in closed-book prompts | The v3 closed-book overlay says "You have no tools" | Every S1 row lists the S1 tools, and every row uses the same system prompt | A tool list that appears only on tool-call rows tells the model whether to call |
| Row metadata | v3 gives each question a metadata block "so the model learns which graph, layer scope, and namespace it is operating in" | `book_mode`, `step`, and `question_family` are backend fields only | These fields show the model the form or the answer |
| `provenance` key | The traces and v3 leave the key out when there is no evidence, and the error results have none | Every tool result and every answer has the key. Non-null values have registry and graph channels; the value is `null` when there is no successful evidence. | One fixed key set. The validator checks channel ownership and exact copying. |
| `evidence_id` | Optional, "when the tool returns one"; no trace has one; the registry block is a prose header | Required and new for each call. The registry block is JSON with a `provenance` object. | A constant block can be memorized. A changing ID must be copied. |
| `blocked_reason` | A code from 9 values on every `defer` answer | A sentence from a fixed template on every `defer` answer, and `null` otherwise. The code stays in the backend. | The user reads why the analysis stopped. The code still drives validation. |
| Layer tag form | Fixed synthetic shapes with version suffixes | Preserve the native IDs in the current full-brain flist: 2-part HumanNet/TF, 3–4-part bulkPEN, and 4–5-part scPEN | Avoids a permanent normalization registry and trains the IDs graph tools actually expose |
| S1.3.6 symbol | GOLGA2P9, both candidates nodes | BMS1P4 | In the store, only one GOLGA2P9 candidate is a node |
| TF sign | Up to two `regulation_direction` traces | No signed TF layer for now. `regulation_direction` is unsupported for every `tf_target` layer. | The loaded TF layer has no sign and no direction |

## Repo data findings (2026-09-30)

**The TF layer.**

- **Tags.** One TF network is in the repo:
  - `TFs:brain` in the canonical full-brain flist
- **Source.** The file is `TF-target-Pearl2019.to.network.spec_cleaned.tsv`
  (Pearl et al. 2019, brain). The edges come from LASSO regression on Allen
  Brain Atlas expression. The weight is r², from 0.50 to 0.96.
- **No sign in the loaded file.** It has only `source`, `target`, and
  `weight`, and all weights are positive.
- **No direction in the loaded file:**
  - In all 198,504 edges, the source ID sorts before the target ID.
  - There are 10,431 distinct sources against 740 real TFs.
  - Cleaning treated the network as undirected and kept the highest weight
    of each duplicate edge. It merged 812 pairs in which each gene regulates
    the other.
- **The raw files keep both.**
  - The sign is in `TF-target-Pearl2019.full.edges.tsv` on Lustre, under
    `.../Transcription_Factor_Target/Brain/`. Its column
    `activate1_repress-1` has 157,868 activating edges and 43,350 repressing
    edges. No pair has both signs.
  - That file matches the uncleaned `to.network.spec.tsv` row for row
    (201,218 of 201,218 on TF, target, and weight).
  - So a join can restore the sign and the TF → target direction.
- **A second TF source.** `TFs:cd4+t_cells` (ChIP-Atlas ChIP-seq, 34 TFs) is
  directed and unsigned. It is in the project-wide master list
  (`mentor-ev_multiplexes/data/h_sap_all_layer_flist.tsv`), not in this
  repo's brain multiplex.

**Current full-brain snapshot (2026-10-01).**

- The canonical flist is
  `/lustre/orion/syb111/proj-shared/Data/H_sap/mentor-ev_multiplexes/brain_multiplexes/full_brain/full_brain_flist.tsv`.
- It contains 353 layers: 7 HumanNetV3, 1 TF, 14 bulkPEN, and 331 scPEN.
  The five missing scPEN layers from the older 358-layer inventory are
  retired rather than synthesized.
- The multiplex gene universe contains 47,017 genes; 45,408 intersect the
  pinned release-116 identifier registry.
- The identity-history pools contain 58,014 absent HGNC symbol spellings
  (previous names and aliases) and 86,256 retired Ensembl gene IDs. Links to
  current components are retained only for leakage-safe split assignment.
- The contract pins the flist SHA-256. Corpus builders consume a compact
  membership index derived from this snapshot, not the multi-gigabyte graph
  store.
- `python -m runtime.world_model.source_index build` constructs the single
  S0/S1 source artifact directly from the pinned release-116 GTF and stable-ID
  history, HGNC history, flist, and all-gene dendrogram. Its logical SHA-256
  is `08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e`.
  The artifact is generated under `data/`, ignored by Git, and contains no
  graph edges, per-layer memberships, or redundant clade memberships.

## Settled decisions and remaining blocker

The runtime contract records the settled choices:

- typed composite registry/graph provenance
- membership checks for task-dependent clean symbols
- immediate deferral on transport failure
- `not_found` for an unknown multiplex
- `other` for a valid nonmapped HumanNet network, with the decision set by
  request framing
- regulatory-link support only for the currently loaded TF layer
- `<MULTIPLEX:...>` anchors for every multiplex mention in a summary
- `failed_tool` for every failure payload
- separate `ambiguous_metric` and `unrecognized_metric` reason codes

S1.5 remains disabled until the context ontology contract and data are
available. Later TF-layer expansion is outside the S1 corpus build.

## Planned traces

These are the advisor's next items, plus the gaps that v3 requires:

- **`no_multiplex_supplied`:** a membership question with no multiplex. The
  model makes no membership call and defers.
- **Batch resolution with mixed outcomes:** for example VARS1 (resolved),
  GOLGA2P9 (ambiguous), and LINC02170 (not found). First decide the result
  shape for each item and the rule for the top-level status.
- **A clean resolution to a gene that is not in the supplied multiplex:**
  - `not_found`, `no_multiplex_membership`. S1 v5 answers a single absent
    gene (S1.1.10) with `resolved` and `present: false`; revisit this
    before it is built.
  - Take the gene from the registry genes that are absent from the store's
    gene table.
- **Input noise beyond lowercase:**
  - List which noise the tools normalize: spaces, case, and `.N`.
  - All other noise, for example a Unicode hyphen, trailing punctuation, or a
    protein name, gives `not_found`.
  - The model never repairs the input.
- **Ambiguity:**
  - more than 2 candidates
  - a coding plus non-coding pair; it gives the positive version of S1.3.9
  - more than 5 candidates, for the long-list rule

  Take them from the registry.
- **S1.2:** a transcript accession (`wrong_input_type`).
- **S1.6:** an unrecognized prefix, the wrong arity, and mixed separators.
- **S1.6 and S1.7:** synthetic family/version fields or path depth outside
  the native family grammar.

## Future evaluation work

**Explicit missing-value states across S0/S1 (after v6, 2026-10-19).**
The user chose to finish v6 before testing this representation change.
Keep JSON null and the shared contract unchanged throughout v6. Treat
performance benefits as a hypothesis, not an established improvement.

- Distinguish a known gene with no assigned symbol, a symbol with no
  established ID, an ambiguous identity, an undecided comparison, and an
  inapplicable attribute. A single replacement string would still conflate
  these meanings. Candidate representations retain null values and add
  explicit states, such as symbol_state:unassigned or identity_state:not_found.
  These are experiment proposals, not approved contract additions.
- Compare the completed v6 reasoning baseline with a separately versioned
  explicit-state experiment. Match examples, split membership, starting
  checkpoint, optimizer steps, and training settings. Score domain outcomes
  and native response validity separately so changing the format does not
  manufacture an accuracy gain.
- A consistent migration must cover affected S0 and S1 tool payloads,
  harness context, training targets and reasoning, replay, validators,
  scorers, and versioned answer keys. Preserve all historical artifacts
  and frozen question/promotion files; create separate experimental artifacts.
- S1.4.5 supplies a concrete diagnostic: nine v5 errors retain a null
  abstract gene_id but assert same_gene:false. Test whether explicit states
  improve the inference from unresolved identity to unknown comparison and
  deferral beyond the approved reasoning change. Include known unnamed genes
  and ambiguous identities to check that the new states preserve their
  distinct decisions. No schema migration, new panel, or model run is part
  of the current S1.4.5 increment.

**S1.3.8 attribute selection (separate follow-up, 2026-10-19).** The
reference trace selects by biotype ("Use the lncRNA"), but current training
and frozen test replies explicitly name an ID. A separate diagnostic
should test uniquely selecting a candidate from visible attributes.
Keep frozen promotion rows and keys fixed, distinguish this coverage
from explicit-ID selection, and check identities against training exposure
before calling a new panel held out. This reasoning-only increment adds
no panel or model run.

**S1.3.6 coverage expansion (deferred, 2026-10-19).** The user deferred
this work during the question-by-question v6 review. Track it separately
from the current training-data changes; no new panel or evaluation run is
part of this item yet.

- The current 27 test questions all use POLR2J3 with two candidates, both
  present. Preserve these questions, answer keys, and the promotion panel
  ID for comparable v4/v5/v6 scores.
- Add a separately versioned coverage panel with all candidates present
  and mixed present/absent results, two and three or more survivors, and
  varied candidate-list sizes. Balance direct/workflow requests and
  harness/task multiplex placement.
- The current eligible source pool has 31 training, four validation, and
  one test symbol for this outcome. Removing the candidate-count cap
  exposes no additional test symbols. Do not move training identities into
  test and describe them as unseen.
- First evaluate the broader validation cases as a labeled validation
  diagnostic: GOLGA8M has 2 candidates/2 present, SNORA73 has 4/2,
  SNORA62 has 7/3, and SNORA72 has 8/3. This evaluation remains future work.
- Further coverage may use explicitly labeled synthetic tool-response
  fixtures. A broader independent test panel needs additional identities
  checked against training exposure across S0 and S1. Report validation,
  synthetic diagnostics, and the frozen test score separately.

## Carried to later steps

These rules came out of the S1 design. They belong to S2, S4, S5, and S9.

- **Scope sets what absence means:**

  | Item | What absence means |
  |---|---|
  | Multiplex | Not a node |
  | Layer | Not in the layer |
  | RWR results | Not in the top `top_k` |
  | Neighbors | Not a neighbor in the queried layers |
  | Subgraph | Not a node |
  | Path | Not on this path |

- **RWR results.** They include the seeds. The seeds that a run used are in
  `active_seed_gene_ids`. Compare full lists, not the compact previews.
- **Ordering by a metric** belongs to S4.2, S4.3, S4.10, and S4.11:
  - Use the S1.8 table.
  - Ties keep the tool order.
  - Missing values stay out of the ordering.
- **Tool changes:**
  - `check_multiplex_membership` gets an optional `layer` argument.
  - The rwr_hpc payload uses `gene_id` as the only ranked-gene key.
- **Stage 2:** add `defer` to `ContinuationState` and to the verifier prompt,
  and score it as an abstention. Also add the item status and
  `blocked_reason` (see "Mapping to Stage 2").
- **v3 S4 legend.** It says that "percentile … lower is closer." Name the
  percentile there (`distance_percentile` or `score_percentile`).

## References

Internal:

- The advisor's original S1 question classes file and email of 2026-09-30;
  the migrated golden set now contains 73 rows.
- `MENTOR-RL-World-Model_v3.md`
- `agents/world_model_v3_curriculum/S0-human-gene-identifiers.md`
- Runtime and scripts:
  - `runtime/world_model_s0_tools.py`
  - `runtime/world_model_prompts.py`
  - `scripts/grab_human_gene_ids.py`
  - `scripts/build_multiplex_store.py`
  - `scripts/token_sequence_estimator.py`
- Canonical layer list:
  `/lustre/orion/syb111/proj-shared/Data/H_sap/mentor-ev_multiplexes/brain_multiplexes/full_brain/full_brain_flist.tsv`
- MENTOR-EV all-gene tree:
  `/lustre/orion/syb111/proj-shared/Projects/mentor-ev/brain/full_brain/dendrograms/full_brain_all_genes_dendrogram.txt`
- TF source files (Lustre): `TF-target-Pearl2019.full.edges.tsv`,
  `TF-target-Pearl2019.to.network.spec.tsv`, and
  `TF-target-Pearl2019.to.network.spec_cleaned.tsv`
- Tests: `tests/test_world_model_s0_tools.py`,
  `tests/test_build_world_model_v2_s0_tool_sft.py`

External (the rules that depend on them):

- ReAct (Yao et al., ICLR 2023) and the OpenAI Harmony format: thinking after
  each tool result.
- STaR (Zelikman et al., NeurIPS 2022) and context distillation (Snell et
  al., 2022): generating the thinking.
- When2Call (Ross et al., NAACL 2025): near-miss rows for when not to call.
- Gururangan et al. (NAACL 2018): the masked-wording check.
- Scialom et al. (EMNLP 2022): the S0 replay share.
- The GENCODE FAQ: `_PAR_Y` IDs in GENCODE 25 to 43.
- Ensembl, retirement of clone-based gene names (2021).
- HumanNet v3 (Kim et al., NAR 2022).
