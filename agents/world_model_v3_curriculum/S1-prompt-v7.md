# S1 v7: task-scope prompt experiment

Prepared on 2026-10-20. This experiment implements the approved prompt and
an 8,192-token training sequence limit. It has not been submitted.

The executable checkout is `../mentor-rl-s1-prompt-v7/`, on branch
`codex/s1-prompt-v7`. It includes the current uncommitted v6 source state.
The original `mentor-rl/` checkout is preserved for training job `5712804`
and its five queued evaluations (`5712872`–`5712876`), whose code and
artifact hashes are pinned. The main checkout contains this documentation,
the new prompt asset, and the current question catalog; run v7 commands
from the isolated checkout.

## Scope and identities

The approved text is in
[`runtime/world_model/prompts/task_scope_v1.txt`](../../runtime/world_model/prompts/task_scope_v1.txt).
Its file SHA-256 is
`0b4c6427d6a2498b0a678ff8d0a364dd340a62a5c37c18dc9c7785d6974691ee`;
the model-facing text omits the final newline and has SHA-256
`eadeb313c5f071c1072d1556443c9fd5ef771c31a99d588d8678bfcd7b8404f9`.

`runtime/world_model/s1_v7.yaml` inherits the finalized v6 selections,
wrappers, reasoning, retry policies, answers, and replay selection. Its
versioned overlay adds the new prompt without editing the shared
`runtime/world_model/contract.yaml`. All S1 rows and the 276 S0 replay
rows receive the same prompt and tool definitions. The original S0 corpus
and checkpoint remain unchanged; the null-symbol marker remains `null`.

The dataset ID is `world_model_v4_s1_coordinate_system_v7`. Training and
evaluation load the prompt through the pinned overlay; legacy datasets
still require their original prompt. The native tokenizer renders the
row's system text in its developer instruction block, with the existing
tool schema and `Reasoning: low` scaffold.

The aggregate contains 55,276 training rows, 6,875 validation rows, and
6,875 test rows. See [all 66 current question types](S1-question-classes.md)
for counts, outcomes, source files, and full gold traces with the reviewed
reasoning. S1.5 remains disabled.

This is an explicitly approved prompt experiment on new output paths.
Validation/test user wording, facts, tool interactions, reasoning, row
order, and targets are retained from v6. Their system prefix changes, so
their file hashes and panel IDs must change. Compare matched record IDs;
do not label the v7 panel byte-identical to v6. The audit checks that
private answer keys stay byte-identical.

## Files and recipe

All paths below are relative to the isolated checkout.

| Purpose | Path |
|---|---|
| Prompt selection | `runtime/world_model/s1_v7.yaml` |
| Corpus and family components | `data/world_model_v4/corpora/s1_v7/` |
| Validation/test panels and keys | `data/world_model_v4/eval/s1_v7/` |
| Training config | `config/world_model_s1_120b_training_v7_32n.json` |
| S1 gate config | `config/world_model_s1_120b_32n_test_v7_512n.json` |
| S0 retention config | `config/world_model_s1_120b_s0_forgetting_test_v7_512n.json` |
| Build, audit, and preflight reports | `data/world_model_v4/launch_reports/s1_v7_120b_32n/` |
| V7 checkpoint output | `data/world_model_v4/checkpoints/s1_v7_120b_32n/oss120b-plain-base-tokenizer-lora-r32/` |

Training retains the selected S0 r32 uniform parent `slurm-5631883`,
32 nodes, TP/EP 8, global batch 32, one epoch, 1,728 optimizer steps,
learning rate 1e-4, and 3% warmup. Only the prompt and accommodating
sequence cap change. Dynamic padding still uses each batch's actual
length. No row is truncated to make it fit. The 8,192 limit bounds the
whole serialized sequence, including instructions, tools, history, and
assistant output; it is not an 8,192-token output allowance.

S1 generation keeps the v6 settings: 8,192 total tokens, up to 4,096 new
tokens, a 6,144-token final-prompt guard, deterministic decoding, and the
`tiered_v3` gate. S0 retention intentionally uses the original frozen S0
panel and original prompt to measure retention under the baseline protocol.
It does not test S0 performance under the new prompt.

## Rebuild and preflight

Activate the established environment:

```bash
cd /lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl-s1-prompt-v7
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
```

Use the existing pinned source index and frozen MyGene cache. The isolated
checkout has hardlinked copies of immutable input artifacts, saving disk
space; do not edit those inputs in place. Historical live source paths
are not required for this rebuild.

Before any v7 job is launched, a deterministic rebuild is:

```bash
set -e
for section in 1 2 3 4 6 7 8 9; do
  USE_TORCH=0 USE_TF=0 PYTHONPATH=. python -m "runtime.world_model.s1_${section}_corpus" \
    --overlay runtime/world_model/s1_v7.yaml
done
USE_TORCH=0 USE_TF=0 PYTHONPATH=. python scripts/build_world_model_s1_bundle.py \
  --overlay runtime/world_model/s1_v7.yaml
python scripts/prepare_world_model_s1_v7_configs.py
USE_TORCH=0 USE_TF=0 PYTHONPATH=. python scripts/audit_world_model_s1_prompt.py \
  --workers 8 \
  --output data/world_model_v4/launch_reports/s1_v7_120b_32n/prompt-tokenizer-audit.json
CONTRACT_ONLY=1 JOB_MODE=train \
  RUN_CONFIG=config/world_model_s1_120b_training_v7_32n.json \
  S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
  bash world_model_v2_s0.slurm
```

After launch, freeze the resulting v7 artifacts and configs. GPU memory,
training behavior, and model accuracy remain unmeasured until a job runs.
The longer permitted sequence can increase peak activation memory even
though dynamic padding avoids padding every row to the full cap.

## Submit when ready

These commands are a recipe; preparing this experiment did not submit them.
Run from the activated isolated checkout:

```bash
unset USE_TORCH USE_TF
mkdir -p logs
S1_V7_JOB=$(sbatch --parsable --nodes=32 --partition=extended --time=16:00:00 \
  --job-name=s1_v7_32n \
  --export=ALL,JOB_MODE=train,CONTRACT_ONLY=0,RUN_CONFIG=config/world_model_s1_120b_training_v7_32n.json,S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
  world_model_v2_s0.slurm)
S1_V7_JOB=${S1_V7_JOB%%;*}
S1_V7_CHECKPOINT="$PWD/data/world_model_v4/checkpoints/s1_v7_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-${S1_V7_JOB}"
```

To queue the configured gate and retention after that training submission:

```bash
S1_V7_GATE_JOB=$(sbatch --parsable --nodes=512 --ntasks-per-node=1 \
  --partition=batch --time=04:00:00 --job-name=s1_v7_eval \
  --dependency="afterok:${S1_V7_JOB}" \
  --export=ALL,CONTRACT_ONLY=0,RUN_CONFIG=config/world_model_s1_120b_32n_test_v7_512n.json,S1_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32,CHECKPOINT_PATH="${S1_V7_CHECKPOINT}" \
  world_model_s1_eval.slurm)
S1_V7_GATE_JOB=${S1_V7_GATE_JOB%%;*}
sbatch --parsable --nodes=512 --ntasks-per-node=1 \
  --partition=batch --time=04:00:00 --job-name=s1_v7_s0_retention \
  --dependency="afterok:${S1_V7_JOB},afterany:${S1_V7_GATE_JOB}" \
  --export=ALL,JOB_MODE=test,CONTRACT_ONLY=0,RUN_CONFIG=config/world_model_s1_120b_s0_forgetting_test_v7_512n.json,S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32,CHECKPOINT_PATH="${S1_V7_CHECKPOINT}" \
  world_model_v2_s0.slurm
```

The evaluation launchers require a completed matching checkpoint and check
it when the dependent jobs start. Retention waits for the gate to finish
even if the gate fails, so the two W&B writers run sequentially.
The queued v6 retry diagnostics remain associated with v6; this preparation
does not repoint them at v7.

## Preparation results (2026-10-20)

The full actual-tokenizer audit passed on all 69,026 rows. V7 differs from
v6 only in the system prompt and its metadata hash. All supervised token
targets are identical; both private answer-key files are byte-identical.

| Split | Rows | Maximum serialized tokens | Rows above 4,096 |
|---|---:|---:|---:|
| Train | 55,276 | 4,461 | 8 |
| Validation | 6,875 | 2,076 | 0 |
| Test | 6,875 | 1,808 | 0 |

The real training launcher passes `CONTRACT_ONLY=1`, exporting the v7 dataset,
8,192-token cap, selected S0 parent, and 1,728-step schedule. The focused v7
and parent/retention suite passes 23 checks; the S1 evaluation suite with the
initial seven v7 checks passes 55, with four absent-checkpoint skips. The
full generation/scoring panel loads with the new prompt and no answer-key
leakage. Completed-checkpoint retention is tested with a fixture; no real
v7 checkpoint exists yet.

All eight family builders validated their splits. The aggregate training
validator reports zero violations, the 53 bad-fixture selftest passes with
zero false positives, and the shared golden JSONL passes its 73-trace
contract check. The consolidated `S1-question-classes.md` now documents the
66 current question types, complete gold traces from the finalized training
rows, and the full approved prompt. Aggregate held-out validation still reports the inherited
cross-family evidence-ID collisions: 251 in validation and 148 in test.
The complete reports match v6 exactly. These IDs remain frozen; per-family
checks and the evaluation harness handle the existing held-out structure.

Legacy regression checks also passed except two retired S0 trajectory
artifact-identity checks whose inputs are unavailable in this checkout.
The no-overlay historical aggregate rebuild is likewise unavailable because
`data/world_model_v3/corpora/s0/manifest.json` is absent. These limitations do
not affect the selected S0 v4 replay source or v7 preflight.

The preservation audit verifies 700 existing non-document files in the main
checkout and 253 frozen input identities unchanged. Rebuilding the v6 aggregate
with the updated builder reproduces all seven corpus/panel/key files byte for
byte (`v6-reproduction.json`). Detailed evidence is in
the report directory listed above, particularly `prompt-tokenizer-audit.json`,
`training-preflight.log`, `final-integration-tests.log`,
`heldout-semantic-comparison.json`, and `main-checkout-preservation-audit.json`.
Model accuracy, loss, GPU memory use, and promotion remain pending a real run.
