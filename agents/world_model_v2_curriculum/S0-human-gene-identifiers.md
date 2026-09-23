# S0: Human Gene Identifier Tool Trajectories

This document defines the canonical S0 v6 contract.

## Scope

S0 teaches the model to resolve human gene identifiers with one local tool.
The model must use Ensembl release 116.

S0 does not teach graph facts, context structure, module identifiers, or
biological operators. Do not use an earlier S0 corpus or a direct-answer format.

## Pinned source and tools

Use this registry:

```text
data/world_model_v2/sources/ensembl_116_human_gene_ids_and_symbols.json
```

Its SHA-256 value is:

```text
e67a4a1d8839c7f8ea568c3040c18941a189f65afeed66dfeedb03c6e305c4ab
```

Use only these tools:

- `lookup_human_gene_symbol` resolves one gene symbol.
- `lookup_human_ensembl_gene_id` resolves one Ensembl gene ID.

`runtime/world_model_s0_tools.py` executes both tools without network access.
The runtime rejects a changed registry or a noncanonical argument.

## Canonical configuration files

Use these five configuration files:

- `config/world_model_v2_s0_tool_calls_v6.json` defines the corpus.
- `config/world_model_v2_s0_20b_tool_trajectory_qualification_v6.json` defines the qualification run.
- `config/world_model_v2_s0_120b_tool_trajectory_training_v6.json` defines the production run.
- `config/world_model_v2_s0_full_registry_tool_trajectory_eval_build_v6_sans_3_long_rows.json` defines the full panel.
- `config/world_model_v2_s0_120b_full_registry_tool_trajectory_test_v6_64n_bs4_sans_3_long_rows.json` defines the production test.

## Corpus contract

The corpus ID is
`world_model_v2_s0_human_identifier_trajectories_v6`.

The train corpus has 4,096 rows. It has 2,048 symbol rows and 2,048 gene ID rows.
The validation panel has 670 rows. The split test panel has 659 rows.

Keep each identifier graph component in one split. Validation and test facts
must not occur in the train corpus.

The corpus has these three families:

- `human_symbol_to_ensembl`
- `human_ensembl_to_symbol`
- `human_ambiguous_symbol`

An ambiguous symbol must return all candidate IDs and the `defer` action.
The model must not select one candidate.

Exclude these three long ambiguous facts from the train corpus and full panel:

- `Metazoa_SRP`: `fact_1544b1236c875b2ad88138d0`
- `U3`: `fact_39c1052bc4d239a961544728`
- `Y_RNA`: `fact_fd3fc526542e105e08c060b9`

The full panel has 85,283 rows after these exclusions.

## Tool trajectory contract

Use the `s0_tool_trajectory_v1` loss contract. Each row has this message order:

1. The system message defines the task and tools.
2. The user message gives one identifier.
3. The assistant writes the required reasoning and calls one tool.
4. The tool message returns the registry result.
5. The assistant returns the final answer.

Apply loss to the assistant reasoning, tool call, and final answer.
Mask the system, user, and tool messages.

Use thought mode with low reasoning effort. Require one tool call.
Do not put the registry result in the user message.

## Tokenizer contract

Use only the unchanged GPT-OSS tokenizer with method
`plain_base_tokenizer`.

The passed gate pins this directory:

```text
data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer
```

The directory name contains `v4` for artifact compatibility. Do not move it.

The tokenizer arm manifest file has this SHA-256 value:

```text
b1379f9a1a751ce7bdc2a99389717ba3095c64e3fbfa608ef77eee181bb8cb6e
```

The token manifest has this internal SHA-256 value:

```text
245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0
```

`scripts/build_world_model_v2_s0_tokenizers.py` reproduces this frozen artifact.
The script does not create custom tokenizer methods.

## Build the data

Build the v6 corpus:

```bash
python scripts/build_world_model_v2_s0_tool_sft.py \
  --config config/world_model_v2_s0_tool_calls_v6.json
```

Build the filtered full panel:

```bash
python scripts/build_world_model_v2_s0_full_tool_eval.py \
  --config config/world_model_v2_s0_full_registry_tool_trajectory_eval_build_v6_sans_3_long_rows.json
```

Do not commit the generated corpus, panel, tokenizer, checkpoint, or report files.

## Check the train contracts

Check the 20B qualification contract:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_v2_s0_20b_tool_trajectory_qualification_v6.json \
S0_METHOD_ID=oss20b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
```

Check the 120B production contract:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_v2_s0_120b_tool_trajectory_training_v6.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm
```

If the contract check fails, do not submit the job.

## Check the production test contract

Set `CHECKPOINT_PATH` to one complete 120B LoRA checkpoint.

```bash
CONTRACT_ONLY=1 JOB_MODE=test \
RUN_CONFIG=config/world_model_v2_s0_120b_full_registry_tool_trajectory_test_v6_64n_bs4_sans_3_long_rows.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
CHECKPOINT_PATH=/absolute/path/to/completed/checkpoint \
bash world_model_v2_s0.slurm
```

The production gate requires these results:

- The gate requires at least 99 percent valid tool trajectories.
- The gate requires reasoning in at least 99 percent of rows.
- The gate requires at least 99 percent exact tool arguments.
- The gate requires at least 99 percent exact tool payloads for each family.
- The gate requires at least 99 percent exact final answers for each family.
- The gate requires exactly 100 percent correct ambiguous deferrals.
- The gate permits at most 1 percent direct answers.
- The gate prohibits network use.

The exact reasoning floor is zero. Exact reasoning remains a diagnostic metric.
