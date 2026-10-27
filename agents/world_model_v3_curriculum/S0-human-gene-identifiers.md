# S0: Human Gene Identifier Traces

Current cross-stage note (2026-10-20): the selected S0 v4 corpus and parent
checkpoint remain frozen. The [S1 v7 prompt experiment](S1-prompt-v7.md)
applies the new prompt to its 276 selected S0 replay rows only. Its S0
retention evaluation keeps the original frozen S0 panel and prompt. Changing
the null-symbol marker across S0 and S1 remains a separate future improvement.

This document defines the current S0 contract (v4) and records its status.
The answer envelope, the tools, and the trace rules are shared with S1. They
are in `runtime/world_model/contract.yaml` and `S1-coordinate-system.md`.

The legacy v7 contract is at the end of this document, for reproducibility.
Do not extend the legacy path.

## Status (2026-10-08)

- The v4 corpus is built.
- The four arms of the v4 training sweep are complete.
- The full-registry tests of the four final checkpoints are running.
- Next step: select one arm with the gate in this document. S1 starts from
  the selected checkpoint (see `S1-coordinate-system.md`, "Start from the S0
  checkpoint").

| Arm | Training config | Training run | Test job |
|---|---|---|---|
| LoRA r32, uniform loss | `config/world_model_s0_120b_training_v4_r32_uniform.json` | `slurm-5631883` | 5637834 |
| LoRA r32, priority loss | `config/world_model_s0_120b_training_v4_r32_priority.json` | `slurm-5631884` | 5637835 |
| LoRA r128, uniform loss | `config/world_model_s0_120b_training_v4_r128_uniform.json` | `slurm-5631885` | 5637836 |
| LoRA r128, priority loss | `config/world_model_s0_120b_training_v4_r128_priority.json` | `slurm-5631886` | 5637837 |

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
   These JSONL fixtures remain separate from the current S1 question catalog.
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

## Corpus (v4)

- Dataset ID: `world_model_v4_s0_human_identifiers`.
- Location: `data/world_model_v4/corpora/s0/`.
- Rows: 16,384 train, 622 validation, and 600 test.
- Build it from the pinned source index:

  ```bash
  python -m runtime.world_model.s0_corpus --overlay runtime/world_model/s0_v4.yaml
  ```

- Do not commit the generated corpus.

The v4 mix:

- **Framing.** 50% direct questions and 50% step requests (8,192 rows each).
- **Case.** 12.5% of the train inputs use lowercase or mixed case.
- **Templates.** Training uses 8 prompt template families for each input
  type. Validation and test use different template families.
- **Splits.** Each identifier graph component stays in one split.
- **Long candidate lists.** Train facts have at most 49 ambiguous candidates.
- **Maximum length.** 4,096 tokens.

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
- **No inference snapshots.** `snapshot_steps` is `0` in each config. A
  value of `128` turns the snapshot pipeline on again. That pipeline submits
  12 snapshot tests and 1 finalizer job for each run.

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

Check a training contract before you submit it:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_s0_120b_training_v4_r32_uniform.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
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

### Final-checkpoint test

Test the final checkpoint of each training run directly. Use
`config/world_model_s0_120b_test_v4_final_512n.json`. Set `CHECKPOINT_PATH`
to the training output directory, for example
`data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883`.

`config/world_model_s0_120b_test_v4.json` is the promotion variant. It
accepts only a snapshot that won the full snapshot and behavioral selection.
That selection is not used now.

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

The test writes its report to
`data/world_model_v4/eval_runs/s0_120b_full_registry_final/<method>/slurm-<job>/report/`.
`test_metrics.json` has the metrics and the gate. The test also adds its
metrics to the W&B run of the training job.

## Gate

The gate has two tiers.

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

Configs without the new floor keys keep the earlier exact-match gate.

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

Compare each v4 arm with these numbers.

## Changes to the test pipeline (2026-10-08)

- **Snapshot generation.** The generator prompted the snapshot test
  (`held_out_multiplex_sft_v4`) for a JSON answer without thinking. The scorer
  requires thinking before the answer, so each snapshot test scored 0. The
  generator now uses the scorer's list of contracts with a thinking step.
- **Final-checkpoint test.** The test contract accepts the final checkpoint of
  a v4 training run.
- **Gate.** The test config sets each floor. The scorer reports
  `result_exact`, and the W&B run gets `test/result_exact_rate` and
  `test/gate_passed`.
- **Snapshots.** The training contract accepts `snapshot_steps: 0`.
- **S1 forgetting test.** The same test also runs on a final S1 checkpoint
  that started from an S0 checkpoint
  (`config/world_model_s1_120b_s0_forgetting_test_512n.json`). It logs to the
  S1 W&B run with the prefix `s0_forgetting/`. The S0 test keeps the prefix
  `test/`.

## Open items

- **Absent inputs.** The full-registry panel has no absent or retired inputs.
  Test the `not_found` deferral on the v4 test split.
- **`continuation_decision`.** If v4 still gives `stop` for step requests,
  compare the step phrasing in the test prompts with the train templates.
  Then add phrasings to the S0 and S1 corpora.

## Legacy v7 contract (frozen)

The v7 corpus, configs, and gate below stay frozen for reproducibility.

### Scope

S0 teaches the model to resolve human gene identifiers with one local tool.
The model must use Ensembl release 116.

S0 does not teach graph facts, context structure, module identifiers, or
biological operators. Do not use a direct-answer format.

### Shared trace contract

The v7 corpus uses the shared supervised trace from the S1 plan. Each trace
has one tool action and one final assistant action.

Each assistant action has a deterministic explanation and an exact receipt.
The final action also has one structured JSON answer.

The trainer applies loss to both explanations, the tool call, and the final
JSON answer. It masks the system, user, and tool messages.

The shared row builder covers these result paths:

- A symbol resolves to one Ensembl gene ID.
- A symbol maps to multiple candidate IDs.
- A symbol has no registry match.
- An Ensembl gene ID has one symbol.
- An Ensembl gene ID has no symbol.
- An Ensembl gene ID has no registry entry.

The shared path uses three reviewed explanation templates for each action
path. It selects one template from the row ID and action index.

Each explanation has no more than two sentences and 40 words. The explanation
audit checks both prompt forms and all reachable action paths.

### Pinned sources and tools

Use this Ensembl registry:

```text
data/world_model_v2/sources/ensembl_116_human_gene_ids_and_symbols_v2.json
```

Its SHA-256 value is:

```text
7a2f71fdfe9db21a96dcbc3775d099ec1691443041ac5639abe3dab7f21ffbaf
```

Registry v2 contains all 78,941 valid gene IDs in the release 116 GTF. It
uses an empty symbol list for each gene ID without a `gene_name`.

The registry has 43,458 named IDs and 35,483 unnamed IDs. It has 41,828
unique symbols, including 484 ambiguous symbols.

Use the pinned HGNC file for absent symbols:

```text
data/world_model_v2/sources/hgnc_complete_set_2026-07-28.txt
```

Its SHA-256 value is:

```text
84ddffcbdb7761b065c8c45e4fc7f626c3bf055b3ab054dbf039ac738a935170
```

Use only these tools:

- `lookup_human_gene_symbol` resolves one gene symbol.
- `lookup_human_ensembl_gene_id` resolves one Ensembl gene ID.

Tool contract v2 returns `gene_symbol` as one string or `null`. It returns
`resolved` for a known gene ID without a symbol.

Tool contract v2 returns `not_found` only for an identifier outside the
registry. It does not return the continuation decision.

`runtime/world_model_s0_tools.py` runs both tools without network access. The
runtime rejects a changed registry or a noncanonical argument.

### Canonical configuration files

Use these five v7 configuration files:

- `config/world_model_v2_s0_shared_traces_v7.json`
- `config/world_model_v2_s0_20b_shared_trace_qualification_v7.json`
- `config/world_model_v2_s0_120b_shared_trace_training_v7.json`
- `config/world_model_v2_s0_full_shared_eval_build_v7.json`
- `config/world_model_v2_s0_120b_full_shared_trace_test_v7_64n_bs4_sans_3_long_rows.json`

### Corpus contract

The corpus ID is
`world_model_v2_s0_human_identifier_supervised_traces_v7`.

The train corpus has 4,096 rows:

- 1,638 registry symbol rows
- 1,741 named gene ID rows
- 307 unnamed gene ID rows
- 410 absent HGNC symbol rows

The validation panel has 622 rows. The split test panel has 600 rows.

Keep each identifier graph component in one split. Validation and test facts
must not occur in the train corpus.

The corpus has these three families:

- `human_symbol_to_ensembl`
- `human_ensembl_to_symbol`
- `human_ambiguous_symbol`

An ambiguous symbol must return all candidate IDs and the `defer` decision.
The model must not select one candidate.

An absent symbol must return `not_found` and the `defer` decision. A known
unnamed gene ID must return `resolved`, `null`, and the `continue` decision.

Exclude these seven long ambiguous facts from the train corpus. Their traces
exceed the 1,024-token training limit:

- `Metazoa_SRP`: `fact_1544b1236c875b2ad88138d0`
- `U3`: `fact_39c1052bc4d239a961544728`
- `U8`: `fact_8cbb53717cd0060ee0c71600`
- `U2`: `fact_99053f9453bab00167f25957`
- `U6`: `fact_a8b5a2d0eb05a25c3900ac94`
- `SNORA70`: `fact_f1d551226c17ab0a816e96a6`
- `Y_RNA`: `fact_fd3fc526542e105e08c060b9`

Also exclude `Metazoa_SRP`, `U3`, and `Y_RNA` from the full panel. Each symbol
has more than 49 candidate gene IDs.

The full v7 panel has 120,830 rows after these exclusions. This count includes
all registry IDs, all registry symbols, and 64 fixed absent symbols.

### Tokenizer contract

Use only the unchanged GPT-OSS tokenizer with method
`plain_base_tokenizer`.

The passed gate pins this directory:

```text
data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer
```

The directory name contains `v4` for artifact compatibility. Do not move it.

The tokenizer arm manifest has this SHA-256 value:

```text
b1379f9a1a751ce7bdc2a99389717ba3095c64e3fbfa608ef77eee181bb8cb6e
```

The token manifest has this internal SHA-256 value:

```text
245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0
```

### Build the data

Build registry v2 from the pinned GTF:

```bash
python scripts/grab_human_gene_ids.py \
  --gtf-path data/world_model_v2/sources/Homo_sapiens.GRCh38.116.gtf.gz \
  --out-path data/world_model_v2/sources/ensembl_116_human_gene_ids_and_symbols_v2.json
```

Build the v7 corpus and split panels:

```bash
python scripts/build_world_model_v2_s0_shared_sft.py \
  --config config/world_model_v2_s0_shared_traces_v7.json
```

Build the filtered full panel:

```bash
python scripts/build_world_model_v2_s0_full_shared_eval.py \
  --config config/world_model_v2_s0_full_shared_eval_build_v7.json
```

Do not commit the generated corpus, panel, checkpoint, or report files.

### Check the train contracts

Check the 20B qualification contract:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_v2_s0_20b_shared_trace_qualification_v7.json \
S0_METHOD_ID=oss20b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
```

Check the 120B production contract:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_v2_s0_120b_shared_trace_training_v7.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
```

If the contract check fails, do not submit the job.

### Check the production test contract

Set `CHECKPOINT_PATH` to one complete 120B LoRA checkpoint.

```bash
CONTRACT_ONLY=1 JOB_MODE=test \
RUN_CONFIG=config/world_model_v2_s0_120b_full_shared_trace_test_v7_64n_bs4_sans_3_long_rows.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
CHECKPOINT_PATH=/absolute/path/to/completed/checkpoint \
bash world_model_v2_s0.slurm
```

The production gate requires these results:

- At least 99 percent of rows have valid tool traces.
- At least 99 percent of rows contain explanations.
- At least 99 percent of tool arguments are exact.
- Each family has at least 99 percent exact tool results.
- Each family has at least 99 percent exact final answers.
- All structured JSON answers are valid.
- All status values and continuation decisions are exact.
- All ambiguous and absent inputs defer.
- All known unnamed IDs resolve.
- At most 1 percent of rows give a direct answer.
- No row uses the network.

The exact explanation floor is zero. Exact explanation remains a diagnostic
metric.

### Frozen v6 contract

The v6 files stay frozen for reproducibility. The runtime, trainer, bundle
builder, generator, evaluator, and validators still accept the v6 contracts.

Use `--registry-schema-version` with the v1 schema to reproduce registry v1.
Do not change a v6 config to use a v7 artifact.
