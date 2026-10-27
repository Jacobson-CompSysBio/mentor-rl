# S0: Human Gene Identifier Traces

This document defines the current S0 contract (v4), how to build, train, and
test it, and its status. The answer envelope, the tools, and the trace rules
are shared with S1. They are in `runtime/world_model/contract.yaml`.

## Status (2026-10-12)

- The v4 corpus is built.
- The four arms of the v4 training sweep are complete.
- The full-registry tests of the four final checkpoints are complete. Each arm
  passes the strict tier and the target tier of the gate (see "Results").
- Next step: select one arm. S1 starts from the selected checkpoint.

| Arm | Training config | Training run | Test job | Gate |
|---|---|---|---|---|
| LoRA r32, uniform loss | `config/world_model_s0_120b_training_v4_r32_uniform.json` | `slurm-5631883` | 5637834 | passed |
| LoRA r32, priority loss | `config/world_model_s0_120b_training_v4_r32_priority.json` | `slurm-5631884` | 5637835 | passed |
| LoRA r128, uniform loss | `config/world_model_s0_120b_training_v4_r128_uniform.json` | `slurm-5631885` | 5637836 | passed |
| LoRA r128, priority loss | `config/world_model_s0_120b_training_v4_r128_priority.json` | `slurm-5631886` | 5637837 | passed |

The training runs are under `data/world_model_v4/checkpoints/s0_120b/<arm>/`.

## Scope

S0 teaches the model to resolve human gene identifiers with one local tool
call. The model must use Ensembl release 116.

S0 does not teach graph facts, context structure, module identifiers, or
biological operators. Do not use a direct-answer format.

## Sources of truth

Use these sources, in this order:

1. `runtime/world_model/contract.yaml`: the shared S0/S1 wire contract. It
   owns the system prompt, the tools, and the answer envelope.
2. `runtime/world_model/s0_v4.yaml`: the v4 overlay. It owns only the v4 row
   counts, the mix, and the added row metadata (`fact_kind`, `framing`,
   `prompt_template_id`, `rendering_index`, `input_variant`).
3. The golden traces S0.1 to S0.7 in `runtime/world_model/golden_traces.jsonl`.
   `python -m runtime.world_model.contract render` prints them for review.
4. This document.

If this document and the runtime contract disagree, the runtime contract wins.

## Forms

| Form | Task | Tool | Expected answer | v4 train rows | v3 train rows |
|---|---|---|---|---|---|
| S0.1 | Symbol to Ensembl ID | Symbol tool | `resolved` | 4,480 | 1,129 |
| S0.2 | Ensembl ID to symbol, named gene | ID tool | `resolved` | 4,608 | 1,741 |
| S0.3 | Symbol in lowercase or mixed case | Symbol tool | `resolved` | 640 | 128 |
| S0.4 | Ensembl ID to symbol, gene without a symbol | ID tool | `resolved`, `gene_symbol` `null` | 5,056 | 307 |
| S0.5 | Ambiguous symbol | Symbol tool | `ambiguous`, `defer` | 832 | 381 |
| S0.6 | Absent symbol (previous HGNC symbol) | Symbol tool | `not_found`, `defer` | 384 | 205 |
| S0.7 | ID not in release 116 (retired) | ID tool | `not_found`, `defer` | 384 | 205 |

Each answer also sets `continuation_decision` from the request framing:

- A direct question gets `stop`.
- A request for a later analysis step gets `continue`.
- An ambiguous, absent, or retired input gets `defer`, with a
  `blocked_reason` sentence.

## Code

| Step | Files |
|---|---|
| Contract and its checks | `runtime/world_model/contract.py`, `contract.yaml`, `golden_traces.jsonl`, `agents/world_model_v3_curriculum/s1_kit/` |
| Source index and tools | `runtime/world_model/sources.py`, `source_index.py`, `tools.py` |
| Corpus | `runtime/world_model/s0_corpus.py`, `s0_v4.yaml` |
| Validation panels | `scripts/build_world_model_s0_validation_panels.py` |
| Training | `world_model_v2_s0.slurm`, `scripts/validate_world_model_v2_s0_train_contract.py`, `scripts/train_sft_dp_tp_ep.py`, `runtime/world_model_training.py` |
| Test | `world_model_v2_s0.slurm`, `scripts/validate_world_model_v2_s0_eval_contract.py`, `scripts/build_world_model_v2_s0_generation_bundle.py`, `scripts/run_world_model_v2_s0_exact_generation.py`, `scripts/merge_world_model_v2_s0_generation_shards.py`, `scripts/evaluate_world_model_v2_s0.py` |

The `v2` in the script names is historical. The scripts implement only the v4
path. The row schema of the shared contract is `multiplex_sft_v3`, so the code
calls the rows "v3 rows".

## Corpus (v4)

- Dataset ID: `world_model_v4_s0_human_identifiers`.
- Location: `data/world_model_v4/corpora/s0/`.
- Rows: 16,384 train, 622 validation, and 600 test.
- Build the source index first (see `s1_kit/README.md`). Then build the
  corpus from the pinned source index:

  ```bash
  python -m runtime.world_model.s0_corpus --overlay runtime/world_model/s0_v4.yaml
  ```

- Build the validation panels from the corpus:

  ```bash
  python scripts/build_world_model_s0_validation_panels.py
  ```

- Do not commit the generated corpus or panels.

The v4 mix:

- **Framing.** 50% direct questions and 50% step requests (8,192 rows each).
- **Case.** 12.5% of the train inputs use lowercase or mixed case.
- **Templates.** Training uses 8 prompt template families for each input
  type. Validation and test use different template families.
- **Splits.** Each identifier graph component stays in one split.
- **Long candidate lists.** Train facts have at most 49 ambiguous candidates.
- **Maximum length.** 4,096 tokens.

### Reproducibility

The training configs pin the corpus manifest and the validation panel
manifest by SHA-256 value. The corpus manifest in turn pins the SHA-256 values
of `s0_corpus.py`, `tools.py`, `contract.yaml`, and `s0_v4.yaml`. Any change
to these four files, including a comment, gives a new corpus manifest. Make
such a change only with a new corpus version.

From the current code, both builds give the pinned files:

| Artifact | SHA-256 |
|---|---|
| `data/world_model_v4/corpora/s0/manifest.json` | `20cbea3bb7e1f1bf9a54e9686b31a29abea992938a07aac58866b24c9832d638` |
| `data/world_model_v4/eval/s0_validation/manifest.json` | `ffc2eb62de88fc851a3a32d6768a6e0fb35e8c4dc5a0ce04bbabbcb071706af3` |

The trainer reads only the `teacher_forced_validation` panel of the validation
manifest. The manifest also holds the `behavioral_validation` and
`snapshot_validation` panels of the removed snapshot pipeline. They stay in
the build so that the manifest keeps its pinned SHA-256 value.

The plain base tokenizer arm at
`data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer`
is a pinned artifact of an earlier tokenizer study. Its directory name
contains `v4` for artifact compatibility. Do not move it.

### Why the mix changed from v3

The v3 model invented a symbol for genes that have no symbol (see "v3
baseline"). The v3 corpus had few of these genes. v4 adds them:

| Data | Resolved ID-to-symbol rows for genes without a symbol |
|---|---|
| v3 S0 train corpus | 15.0% (307 of 2,048) |
| v4 S0 train corpus | 52.3% (5,056 of 9,664) |
| Full Ensembl 116 registry | 44.9% (35,483 of 78,941) |

v4 also uses four times as many rows, balanced framing, and more template
families.

## Training (v4 sweep)

The sweep has two LoRA ranks and two loss weightings. All four arms share
these settings:

- Base model: `gpt-oss-120b-bf16`, with the unchanged base tokenizer
  (`plain_base_tokenizer`).
- 16 nodes with 8 GPUs each. Tensor parallel 8, expert parallel 8, data
  parallel 16.
- 3 epochs. Global batch 32. 512 updates for each epoch and 1,536 steps in
  total.
- Learning rate 1e-4, cosine schedule, 3% warmup. Maximum length 4,096.
- LoRA alpha is two times the rank.
- A checkpoint every 512 steps, and the final checkpoint.

The train contract validator pins each of these settings.

The **uniform** arms (`multiplex_sft_v3`) give the same weight to each
supervised token. The **priority** arms (`multiplex_sft_v4_priority`) use
these token weights:

| Tokens | Weight |
|---|---|
| System, user, and tool messages | 0 |
| Assistant prose | 1 |
| Tool call | 4 |
| Final JSON | 4 |
| `status`, `result`, `continuation_decision`, `blocked_reason` | 8 |

The four sweep runs also ran the inference snapshot pipeline
(`snapshot_steps` 128). It saved an inference copy of the adapter 12 times in
each run. That pipeline is removed, and the configs no longer have a
`snapshot_steps` key. The exact config of each run is in its
`run_contract/run_config.json`.

Check a training contract before you submit it:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_s0_120b_training_v4_r32_uniform.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
```

Submit a training run with the same values. The `#SBATCH` defaults of the
launcher match one arm (16 nodes):

```bash
sbatch --export=ALL,JOB_MODE=train,RUN_CONFIG=config/world_model_s0_120b_training_v4_r32_uniform.json,S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
  world_model_v2_s0.slurm
```

**Teacher-forced validation loss does not separate the arms.** Each arm
reaches a loss below 0.005 after epoch 1, and the uniform arms reach about
1e-4. Use the generation test to compare the arms.

## Test

### Panel

The test uses the frozen v3 full-registry panel:

- Location: `data/world_model_v3/eval/s0_full_registry/`.
- 120,766 rows: 78,941 ID-to-symbol, 41,344 symbol-to-ID, and 481 ambiguous
  symbols. `Metazoa_SRP`, `U3`, and `Y_RNA` are excluded. Each has more than
  49 candidate IDs.
- The panel labels use the v4 rules. The system prompt and the tools are the
  same as in v4 training.
- The panel has no absent or retired inputs. It does not test the
  `not_found` deferral.
- The v3 builder made the panel. The current builder writes v4 rows, so it
  cannot rebuild the panel. The test contract pins the panel manifest, the
  questions, and the answer key by SHA-256 value.

### Final-checkpoint test

Test the final checkpoint of each training run. Use
`config/world_model_s0_120b_test_v4_final_512n.json`. Set `CHECKPOINT_PATH`
to the training output directory, for example
`data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883`.

Check the test contract:

```bash
CONTRACT_ONLY=1 JOB_MODE=test \
RUN_CONFIG=config/world_model_s0_120b_test_v4_final_512n.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
CHECKPOINT_PATH=$PWD/data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883 \
bash world_model_v2_s0.slurm
```

Submit the test on the `batch` partition with 512 nodes:

```bash
sbatch --job-name=wm4_s0_full_r32_u --nodes=512 --ntasks=512 \
  --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=56 \
  --partition=batch --time=04:00:00 --constraint=nvme \
  --export=ALL,JOB_MODE=test,RUN_CONFIG=config/world_model_s0_120b_test_v4_final_512n.json,S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32,CHECKPOINT_PATH=$PWD/data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883,EVAL_MODEL_STAGE_WAVE_SIZE=256,S0_GENERATION_ATTEMPTS=2 \
  world_model_v2_s0.slurm
```

Why these resources:

- The node count in the config must equal the allocation.
- Generation needs about 525 node-hours. With 512 nodes, a test takes about
  2 hours.
- The `extended` partition has 192 nodes and runs one job for each user at a
  time. On `batch`, a 64-node job is limited to 2 hours.
- `EVAL_MODEL_STAGE_WAVE_SIZE=256` copies the base model to the nodes in 2
  waves. The default of 8 nodes for each wave takes hours on 512 nodes.

The test runs in four stages:

1. The bundle builder writes the public questions. It does not copy the
   answer key.
2. Each of the 512 model processes generates one shard. For each question, it
   generates the thinking and one tool call, replays the call on the pinned
   source index, and then generates the final thinking and the JSON answer. A
   failed process resumes from its last completed batch.
3. The merge step joins the shards in the bundle order.
4. The scorer replays each call again, scores the answers against the answer
   key, and applies the gate.

The test writes its report to
`data/world_model_v4/eval_runs/s0_120b_full_registry_final/<method>/slurm-<job>/report/`.
`test_metrics.json` has the metrics and the gate. `exact_tool_report.json` has
the score of each row. The test also adds its metrics to the W&B run of the
training job, with the prefix `test/`.

## Gate

The test config must set each floor. The gate has two tiers.

**Strict tier: at least 99.5%.** The model uses the tool correctly and reports
exactly what the tool returned.

| Check | Config key | Floor |
|---|---|---|
| One well-formed tool call | `minimum_valid_single_tool_call` | 99.5% |
| Correct tool and exact identifier | `minimum_exact_tool_argument` | 99.5% |
| Exact tool result, for each family | `minimum_family_payload_accuracy` | 99.5% |
| Answer without a tool call | `maximum_direct_answer_rate` | at most 0.5% |
| Valid trajectory | `minimum_valid_tool_trajectory` | 99.5% |
| Valid answer JSON | `minimum_structured_json_valid` | 99.5% |
| `status` agrees with the tool | `minimum_status_exact` | 99.5% |
| `result` agrees with the tool, with `null` kept | `minimum_result_exact` | 99.5% |
| Ambiguous symbol gets `defer` | `minimum_ambiguous_defer_accuracy` | 99.5% |
| Absent input gets `defer` | `minimum_not_found_defer_accuracy` | 99.5% |
| No network use | `require_network_used_false` | 100% |

**Target tier: at least 90%.**

| Check | Config key | Floor |
|---|---|---|
| `continuation_decision` | `minimum_continuation_decision_exact` | 90% |

**Format floors: at least 99%.** Thinking is present before the call and
before the answer (`minimum_reasoning_present`).

**Reported only.** These metrics have a floor of 0:

- the exact match of the whole answer (`minimum_family_final_answer_accuracy`)
- `unnamed_id_resolved` (`minimum_unnamed_id_resolved_accuracy`). It also
  requires a correct `continuation_decision`. `result_exact` covers the `null`
  symbol by itself.
- the exact thinking text (`minimum_exact_reasoning`)

**The summary is not evaluated.** The summary is a prose form of the JSON
answer. Review a sample of summaries by hand.

## Results (v4)

Each arm passes every check of the gate. The table gives the number of
failed rows out of 120,766.

| Check | r32 uniform | r32 priority | r128 uniform | r128 priority |
|---|---|---|---|---|
| One well-formed tool call | 0 | 3 | 1 | 1 |
| Correct tool and exact identifier | 16 | 46 | 20 | 41 |
| Exact tool result | 15 | 46 | 20 | 41 |
| Answer without a tool call | 0 | 0 | 0 | 0 |
| Valid answer JSON | 0 | 10 | 6 | 12 |
| `status` exact | 15 | 45 | 20 | 41 |
| `result` exact | 31 | 49 | 20 | 41 |
| `continuation_decision` exact | 15 | 46 | 20 | 41 |
| Ambiguous symbol gets `defer` (481 rows) | 0 | 0 | 0 | 0 |
| Whole answer exact (reported only) | 44 | 59 | 31 | 63 |

Almost all strict-tier failures are in the symbol-to-ID family. For every
ID-to-symbol row and every ambiguous symbol, each arm calls the right tool and
gets the exact tool result, `status`, and `continuation_decision`.

## v3 baseline

The v3 S0 model (training run `slurm-5629215`, test job 5629216) is the
baseline. Its full-registry results:

| Metric | v3 |
|---|---|
| One well-formed tool call | 100% |
| Correct tool and exact identifier | 95.8% (93.5% for ID to symbol) |
| `status` exact | 95.2% |
| `result` exact | 90.9% |
| Valid answer JSON | 99.4% |
| Ambiguous symbol gets `defer` | 99.4% (478 of 481) |
| `continuation_decision` exact | 81.2% |
| Whole answer exact | 78.6% |

v3 fails the strict tier and the target tier. These systematic errors cause
most failures:

- **Wrong tool.** In 5,110 ID-to-symbol rows, the model sent the Ensembl ID to
  the symbol tool. Only 16 rows had a copy error in the argument.
- **Invented symbol.** In 5,205 rows, the tool returned no symbol, and the
  model wrote a symbol, for example `LINC00116`.
- **Wrong decision.** In 16,881 rows, the model gave `stop` when the request
  was for a later step.
- **Incomplete deferral.** In 730 rows, the model gave `defer` without a
  `blocked_reason`. The schema rejects this answer.

## Open items

- **Arm selection.** Each arm passes the gate. Select one arm for S1.
- **Absent inputs.** The full-registry panel has no absent or retired inputs.
  Test the `not_found` deferral on the v4 test split. No test config or panel
  exists for that split yet.
- **Pinned HPC source.** On 2026-10-09, the MENTOR-EV dendrogram was not at
  the path that `contract.yaml` pins. `python -m runtime.world_model.contract
  check` and a new source-index build fail until it is back. The corpus build
  reads the existing source index and is not affected.

## Removed paths

Only the v4 path is supported. These paths are removed:

- **v6 tool trajectories** (`world_model_v2_s0_human_identifier_trajectories_v6`),
  the v2 identifier registry, and the tokenizer-arm builder. They are in the
  history of `main` before this change.
- **v7 shared traces** and the v3 S0 training and test configs. They were not
  merged to `main`.
- **The snapshot and behavioral promotion pipeline**, including the
  promotion test config. The final-checkpoint test replaces it.
