# S0: Human Gene Identifier Traces

This document defines the current S0 v7 contract.

## Scope

S0 teaches the model to resolve human gene identifiers with one local tool.
The model must use Ensembl release 116.

S0 does not teach graph facts, context structure, module identifiers, or
biological operators. Do not use a direct-answer format.

## Shared trace contract

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

## Pinned sources and tools

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

## Canonical configuration files

Use these five v7 configuration files:

- `config/world_model_v2_s0_shared_traces_v7.json`
- `config/world_model_v2_s0_20b_shared_trace_qualification_v7.json`
- `config/world_model_v2_s0_120b_shared_trace_training_v7.json`
- `config/world_model_v2_s0_full_shared_eval_build_v7.json`
- `config/world_model_v2_s0_120b_full_shared_trace_test_v7_64n_bs4_sans_3_long_rows.json`

## Corpus contract

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

## Tokenizer contract

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

## Build the data

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

## Check the train contracts

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

## Check the production test contract

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

## Frozen v6 contract

The v6 files stay frozen for reproducibility. The runtime, trainer, bundle
builder, generator, evaluator, and validators still accept the v6 contracts.

Use `--registry-schema-version` with the v1 schema to reproduce registry v1.
Do not change a v6 config to use a v7 artifact.
