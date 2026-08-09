# S0: Human Gene Identifier Foundation

This file defines the S0 corpus, tokenizer, train, and gate contracts.

## Objective and boundary

S0 is the first supervised curriculum stage. It is not the WS0 divergence
workstream.

The first S0 model uses only human genes from one pinned Ensembl release. S0
teaches exact reciprocal mappings between Ensembl gene IDs and gene symbols.
It also teaches the correct response for ambiguous symbols.

The S0 gate selects the tokenizer, fine-tune configuration, and checkpoint for
S1. S0 does not teach graph facts, context structure, module identifiers, or
biological operators.

Use `closed_book` for all S0 train, validation, and test rows. Do not create
`open_book` rows in this stage.

## System prompt contract

Instruct the model to resolve human Ensembl gene IDs and gene symbols. Require
one exact JSON object with no extra text.

Require the complete candidate set and `defer` action for an ambiguous symbol.
Do not permit graph facts, tools, or another species.

Do not supply registry evidence in an S0 row. Set the record's top-level
`context` field to `null`.

`runtime/world_model_prompts.py` contains this prompt. Its SHA-256 hash is
`06646540ea70f94d8cb8ca5fc9764980e0a8b251953d53ff2259c55962d5005b`.

## Corpus families

The S0 corpus contains three required families:

- `S0.1` (`human_symbol_to_ensembl`): Map a symbol to its Ensembl gene ID.
- `S0.2` (`human_ensembl_to_symbol`): Return all Ensembl symbols for one gene ID.
- `S0.3` (`human_ambiguous_symbol`): Return all candidate IDs for an ambiguous
  symbol.

For `S0.3`, require an explicit ambiguous status and a deferral action. If the
symbol is ambiguous, do not permit a graph lookup.

Use Ensembl release 116 as the only identifier source. Do not use the symbol
values in `data/gene_source/human_gene_ids.json`.

Do not use MyGene or HGNC data in the v4 corpus. A later corpus version can add
these sources.

Modify `scripts/grab_human_gene_ids.py` to read the pinned Ensembl release 116
GTF. Do not query the live MyGene service.

Write the extracted source data to
`data/world_model_v2/sources/ensembl_116_human_gene_ids_and_symbols.json`.
Do not commit this generated source file.

Select every `gene` feature with a valid human Ensembl gene ID. Collect every
distinct, nonempty `gene_name` for each gene ID.

Treat the symbols for one gene ID as an unordered set. Do not use source order
to select one symbol.

If a gene ID has no `gene_name`, exclude it and record the reason. If a gene ID
has multiple symbols, retain the complete symbol set.

For a symbol that maps to one gene ID, create one `S0.1` fact. Return that symbol
and the gene ID.

For each retained gene ID, create one `S0.2` fact. Return the complete sorted
symbol set in `gene_symbols`.

If a symbol maps to multiple gene IDs, create one `S0.3` fact. Return the
complete sorted candidate ID set.

Keep all linked values in one identifier graph component. Use components to
keep related identifiers together during corpus selection and audit.

The pinned GTF has 78,941 human gene IDs. It has 43,458 IDs with `gene_name` and
35,483 IDs without `gene_name`.

The named set has 41,828 unique symbols. It has 484 symbols that map to multiple
gene IDs. The largest current candidate set contains 758 gene IDs.

Build the closed-book recall corpus at
`data/world_model_v2/sft/s0_human_identifiers_v4/`. Do not commit the generated
corpus files.

The source manifest must record the GTF hash, release, assembly, counts, and
normalization rules. The corpus manifest must record all source and output
hashes.

The corpus manifest must also record row counts, exclusion counts, relation
cardinalities, and the largest candidate set. Add final artifact values only
after the audit passes.

Every validation and test fact must have a train row for the same question
family. Use a new question form for each evaluation split. Every row must be
closed book.

The training contract is `closed_book_only_v1`. The evaluation contract is
`seen_fact_closed_book_recall_v1`.

`config/world_model_v2_s0_closed_book_recall_v4.json` pins this contract.
`scripts/build_world_model_v2_s0.py` builds the v4 corpus.
`scripts/audit_world_model_v2_s0.py` checks hashes, records, answer keys, and
split contracts. It does not inspect the GTF or tokenizer artifacts.

## Question contract

Store the task input in the record's top-level `input` field. Convert this
input to compact canonical JSON. Insert that JSON once at `{input_json}` in the
selected question template.

Use this train question for `S0.1`:

```text
Return one JSON object with exactly these keys: gene_id, gene_symbol, status. Resolve the human Ensembl gene symbol to its Ensembl gene ID. Use the pinned Ensembl release 116 registry. Input: {input_json}.
```

Use this train question for `S0.2`:

```text
Return one JSON object with exactly these keys: gene_id, gene_symbols, status. Return every Ensembl gene symbol for the human Ensembl gene ID. Use the pinned Ensembl release 116 registry. Input: {input_json}.
```

Use this train question for `S0.3`:

```text
Return one JSON object with exactly these keys: action, candidate_gene_ids, gene_symbol, status. Return every candidate Ensembl gene ID and defer. Use the pinned Ensembl release 116 registry. Input: {input_json}.
```

The train runtime sends one user message with this form:

```text
<rendered question>
```

Do not put metadata, provenance, validators, the standalone input field, or the
answer in the user message. The rendered question contains the input once. Use
separate prompt forms for train, validation, and test rows.

For `S0.2`, return `gene_symbols` as a sorted list. Use a one-item list when the
gene ID has one symbol.

## Exact answer contracts

Use the graph-free `identifier_sft_v2` metadata. Do not add `multiplex_id`,
`layer_scope`, or `layer_ids` to S0 rows.

Put `identifier_registry_id` and `system_prompt_sha256` in S0 metadata. Use the
hashes of the pinned registry and S0 system prompt.

Use `{"gene_symbol":"GLP1R"}` as the `S0.1` input. Use
`{"gene_id":"ENSG00000112164"}` as the `S0.2` input.

Use this `S0.1` answer:

```json
{"status":"resolved","gene_id":"ENSG00000112164","gene_symbol":"GLP1R"}
```

Validate `status` with `ENUM`, `gene_id` with `EXACT_ID`, and `gene_symbol` with
`SYMBOL`.

Use this `S0.2` answer:

```json
{"status":"resolved","gene_id":"ENSG00000112164","gene_symbols":["GLP1R"]}
```

Validate `status` with `ENUM`, `gene_id` with `EXACT_ID`, and `gene_symbols` with
`SYMBOL_SET`.

Use `{"gene_symbol":"<ambiguous symbol>"}` as the `S0.3` input. Use this answer
shape:

```json
{"status":"ambiguous","gene_symbol":"<ambiguous symbol>","candidate_gene_ids":["<candidate 1>","<candidate 2>"],"action":"defer"}
```

Validate `status` and `action` with `ENUM`. Validate `gene_symbol` with `SYMBOL`
and `candidate_gene_ids` with `ID_SET`.

Before serialization, sort `gene_symbols` and `candidate_gene_ids`. Use exact
whole-record accuracy for all three families.

Set the record's top-level `context` field to `null` for every S0 row.

Put `fact_id` and `rendering_index` in `provenance`. Use a nonnegative integer
for `rendering_index`.

## Split contract

Before tokenizer fit, create fixed train rows and fixed validation and test
panels. Create a bipartite graph from symbols and Ensembl gene IDs.

Use each connected component as one panel unit and one `fact_group_id`. Assign
each component to `train`, `validation`, or `test` for evaluation.

Keep both directions and each complete ambiguous set in the same component.
Keep validation and test components separate. Keep all assignments fixed across
methods.

Use one closed-book train row for every fact. Set `fact_role` to `seen` for
every train, validation, and test row.

For a validation component, create one validation row for each fact. Use the
same `fact_id` as the matching train row and a new question form.

For a test component, create one test row for each fact. Use the same `fact_id`
as the matching train row and a new question form.

Use `train_closed_book`, `validation_closed_book`, and `test_closed_book` as the
three prompt forms. Do not reuse exact question text across these forms.

Set `context` to `null` for every row. Do not create unseen, source-reserve, or
open-book rows.

Fit each tokenizer only on train data. Because every evaluation fact has a train
row, include all evaluation identifiers in tokenizer fit.

Base-model pretrain exposure is unknown. Report base-checkpoint metrics
separately from trained-checkpoint metrics.

Track validation mapping accuracy at each configured evaluation interval. Save
the checkpoint and evaluation step with each metric record.

For `S0.1`, mapping accuracy requires an exact `gene_id`. For `S0.2`, it requires
an exact `gene_symbols` set. For `S0.3`, it requires an exact
`candidate_gene_ids` set.

Use the unweighted mean of the three family accuracies as macro mapping
accuracy. Use this metric to select the trained checkpoint.

Also record whole-record accuracy, valid JSON rate, schema compliance rate,
correct `defer` rate, family accuracies, and validation loss. Treat validation
loss as a diagnostic.

After training ends, run one test inference pass with the selected checkpoint.
Do not use test metrics to select or change the checkpoint.

Audit that each validation and test `fact_id` has one matching train row. Audit
component separation, prompt-form separation, duplicate rows, and answer-key
joins. Audit tokenizer coverage in the model training PR.

## Matched experiment matrix

Use the v4 corpus, question forms, answer schema, and split contract for every
run. Do not compare a v3 artifact with a v4 artifact.

The GPT-OSS-20B qualification uses three tokenizer methods with LoRA r32. The
GPT-OSS-120B comparison uses all 12 methods in this 3 by 4 matrix.

| Axis | Required values |
| --- | --- |
| Tokenizer | Vanilla GPT-OSS (`plain_base_tokenizer`), Domain-BPE (`ordinary_domain_bpe`), atomic plus Domain-BPE (`atomic_plus_domain_bpe`) |
| Fine-tune configuration | LoRA r32, LoRA r128, LoRA r1024, full fine-tune |

Vanilla GPT-OSS means the unchanged tokenizer from the pinned base checkpoint.
Pin its tokenizer hash in each run contract.

Domain-BPE uses one 240-piece BPE for human Ensembl IDs. It uses another
240-piece BPE for human symbols.

Domain-BPE adds two namespace rows and 480 BPE rows. It uses 482 model rows.

Atomic plus Domain-BPE adds one literal `ENSG` token. It fits the Ensembl BPE
only on each numeric suffix.

Encode an Ensembl ID as the atomic `ENSG` token and one or more suffix BPE
tokens. Encode each symbol only with symbol BPE tokens.

This method adds one atomic row and 480 BPE rows. It uses 481 model rows and
leaves 588 spare model rows.

Use the same representation for every retained ID and symbol. Do not create a
value registry, type token, or random code token.

Within one model size, use the same base checkpoint for all methods. Hold the
corpus, panels, row order, prompt, prompt forms, seed, and fact exposure fixed.

Hold the optimizer family and run settings constant across tokenizers for each
fine-tune configuration. Pin the learning rate, global batch size, update
count, schedule, and regularization values.

Use r32 with alpha 64, r128 with alpha 256, and r1024 with alpha 2048. Pin these
values in each run contract.

`config/world_model_v2_s0_tokenizer_matrix_v4.json` records the search grid. It
gives one declared trial to each configuration.

Derive update counts from the audited v4 train row count and global batch size.
Record logical and physical row exposures. Do not copy v3 update counts.

For a full run, expose every eligible train row. Do not count a pinned corpus
exclusion as an eligible train row.

For a debug qualification, expose only its bounded row subset. Record that the
debug run did not expose the complete eligible train set.

### Shared loss contract

Use this loss for the 20B qualification and the 120B matrix:

`loss = 0.5 * completion_loss + 0.5 * mapping_target_loss`.

Use `s0_target_aware_v2` as the loss contract. Keep the full completion loss.
Apply the mapping loss only to mapped output values.

For `S0.1`, target `gene_id`. For `S0.2`, target each string in `gene_symbols`.
For `S0.3`, target each string in `candidate_gene_ids`.

Create the mapping mask after the text codec and tokenizer encode the answer.
Do not include JSON keys, list punctuation, or structural tokens in this mask.

Track completion loss, mapping-target loss, and combined loss separately. Do
not select a checkpoint from a train loss.

### GPT-OSS-20B qualification

Use the pinned GPT-OSS-20B BF16 checkpoint. Run each tokenizer method with LoRA
r32 and alpha 64 on the v4 corpus.

Use one audited epoch for each qualification run. Derive the update count from
the final v4 train row count and the pinned global batch size.

Require each run to complete its train pass, validation inference pass, metric
record, checkpoint receipt, and exposure receipt.

For each run, select its checkpoint with validation macro mapping accuracy.
Require at least 20 percent validation macro mapping accuracy.

If one run is below 20 percent, disqualify that qualification result. Inspect
the failed contract and repeat the qualification before a 120B submission.

Set `promotion_eligible` to `false`. Use the qualification to verify the corpus,
tokenizer, loss, trainer, Slurm path, and minimum mapping accuracy.

Do not use 20B metrics to select the 120B method. Do not start the 120B matrix
until all three runs pass the infrastructure and accuracy gates.

Pin the 20B topology, batch size, schedule, learning rate, warmup, and seed in
`config/world_model_v2_s0_20b_qualification_v4.json`.

### GPT-OSS-120B matched matrix

Use the pinned GPT-OSS-120B BF16 checkpoint. Run all 12 tokenizer and fine-tune
configurations on the same v4 corpus.

Use validation macro mapping accuracy to select one method and one checkpoint.
After selection, run one test inference pass with that checkpoint.

Pin each 120B topology, batch size, schedule, learning rate, warmup, and seed in
`config/world_model_v2_s0_120b_matrix_v4.json`.

For LoRA, use the adapter loader. For full fine-tune, use the full-checkpoint
loader. Pin the loader identity in each run contract.

### Launch contract

Use `world_model_v2_s0.slurm` for every S0 train and inference job. Pass one
job mode and one run-config path to this file.

Use `JOB_MODE=train`, `JOB_MODE=validation`, or `JOB_MODE=test`. The run
contract must declare the method ID and all mode-specific inputs.

For train mode, select the model size, tokenizer method, fine-tune method,
topology, trainer, and output root from the run contract. Build the run ID and
output path from the method ID and Slurm job ID.

For evaluation modes, select the checkpoint, panel, generation settings,
scorer, and output path from the run contract.

Do not use a shell launcher. Do not put model-specific or evaluation-specific
settings in a second Slurm file.

`runtime/world_model_s0_tokenizer.py` implements the two custom text contracts.
`scripts/build_world_model_v2_s0_tokenizers.py` builds and audits all methods.

Write the v4 tokenizer artifacts to
`data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/`. Do not commit
these generated artifacts.

Record each tokenizer content hash, manifest hash, added row count, and spare
row count after its v4 audit passes.

Use `scripts/train_sft_dp_tp_ep.py` for LoRA. Use
`scripts/train_sft_full_zero3.py` for full fine-tune.

The shared Slurm file must use these startup controls:

- Bind NCCL and Gloo sockets to IPv4.
- Clear only user-owned ROCm-SMI state.
- Run cleanup before the first `torch` import.
- Run one Python and ROCm check on each node.
- Require one launch receipt from all ranks.
- Check the world and local rank counts.
- Retry once on a new port only before `DISTRIBUTED_READY`.
- Require `config.json` before reuse of a staged model.

## Three-step promotion protocol

1. Run the three 20B qualification methods.
   Require each method to pass its infrastructure gates.
   Require at least 20 percent validation macro mapping accuracy for each
   method.
2. Run all 12 methods in the 120B matched matrix.
   Use validation mapping accuracy to select one method and one checkpoint.
   Require at least 90 percent mapping accuracy in each validation family.
3. Freeze the method, checkpoint, run contract, and test generation settings.
   After training ends, run one test inference pass.
   Promote only if each test family reaches 90 percent mapping accuracy.

If a 20B method fails, do not start the 120B matrix. Change only the declared
run contract. Let the next Slurm job assign a new run identity.

If no 120B checkpoint passes the validation floor, do not open the test panel.
Use only train and validation results to define a new run contract.

If the test gate fails, record the failure and keep all exact artifacts. Do not
use test metrics to change the selected checkpoint or its run contract.

For a new campaign after a test failure, create a new test prompt form and test
manifest. Assign a new campaign identity.

## Evaluation and gate

Compare all 12 methods with validation macro mapping accuracy. Test only the
selected 120B checkpoint.

Select the checkpoint with the highest macro mapping accuracy. If two
checkpoints tie, select the higher minimum family accuracy.

If the tie remains, select the checkpoint with fewer trainable parameters. Pin
the tie-break result in the selection receipt.

Require at least 90 percent mapping accuracy in each validation family before
the test pass. Require the same floor in each test family for promotion.

Report whether each family and the macro score reach the 95-percent objective.
Report 95-percent Wilson intervals for family mapping accuracy.

Report whole-record accuracy, valid JSON rate, schema compliance rate, and
correct `defer` rate as diagnostics. Do not use these diagnostics for selection.

Report base-checkpoint metrics separately. Base-model pretrain exposure is
unknown.

A completed job, train loss, or token accuracy cannot pass S0. Require the
mapping report, split audit, tokenizer manifest, exposure receipt, and model
identity.

Write fixed validation and test manifests. Store their answer keys in
evaluator-only storage. Record each manifest and answer-key hash.

Give only an answer-free panel to the model process. Keep the test panel closed
until method and checkpoint selection are final.

Pin all inference generation settings in the evaluation contract. This PR
implements the generation, parser, scorer, selection, and publication paths.

## S1 handoff

This PR does not implement S1. Add S1 in a separate PR after S0 passes.

An S0 pass does not pass WS0, WS1, S1, or a later curriculum stage.

Pin the S0 source hashes, identifier registry, corpus hash, tokenizer manifests,
system prompt, schema, train row hash, selected configuration, and checkpoint.

Do not change a pinned S0 artifact when S1 starts. Extend stage mappings without
changing the S0 prompt or prompt hash.

If S1 uses S0 replay, generate replay rows from the pinned S0 facts. Use a replay
prompt form that differs from the S0 validation and test prompt forms.

The future S1 contract must define a new S0 retention prompt form and panel.
Track S0 family and macro mapping accuracy during S1.

Do not use S0 test metrics to select an S1 checkpoint.

If S1 needs module tokens, add them in a separate namespace. Do not change the
pinned S0 gene tokens.

## Stacked PR implementation plan

Use three stacked PRs. The model training PR uses the data generation branch as
its base. The model evaluation PR uses the model training branch as its base.

### PR 1: Data Generation

Add only the S0 specification, prompt contract, source scripts, and corpus
scripts.

Specification and prompt files:

- `agents/world_model_v2_curriculum/S0-human-gene-identifiers.md`
- `runtime/world_model_prompts.py`

Source and corpus files:

- `scripts/grab_human_gene_ids.py`
- `config/world_model_v2_s0_closed_book_recall_v4.json`
- `scripts/build_world_model_v2_s0.py`
- `scripts/audit_world_model_v2_s0.py`

Add the data generation tests after PR 1.

### PR 2: Model Training

Add the tokenizer code, training runtime, run contracts, and train launch mode.

Runtime files:

- `runtime/world_model_schemas.py`
- `runtime/world_model_s0.py`
- `runtime/world_model_training.py`
- `runtime/world_model_validators.py`

Tokenizer files:

- `runtime/world_model_s0_tokenizer.py`
- `scripts/build_world_model_v2_s0_tokenizers.py`
- `config/world_model_v2_s0_tokenizer_matrix_v4.json`

Training and launch files:

- `config/ds_zero3_full_finetune_cpu_offload.json`
- `config/world_model_v2_s0_20b_debug_v4.json`
- `config/world_model_v2_s0_20b_qualification_v4.json`
- `config/world_model_v2_s0_120b_matrix_v4.json`
- `scripts/train_sft_dp_tp_ep.py`
- `scripts/train_sft_full_zero3.py`
- `world_model_v2_s0.slurm`

The Slurm file supports `JOB_MODE=train` in this PR. The next PR adds the two
evaluation modes to the same file.

Each train job measures target-aware validation loss before the first update.
It measures the same loss after each epoch.
The job writes these metrics to W&B and `validation_metrics.jsonl`.
Use this loss only as a training diagnostic.

The debug contract uses one GPT-OSS-20B node and one update.
It checks both validation passes, the first update, and the final receipts.
It sends metrics to W&B online.
It can use the standard W&B credential store.
It is not a promotion result.

Defer these focused test files to a later test change. They do not block PR 2.

- `tests/test_world_model_v2_s0_tokenizers.py`
- `tests/test_s0_target_aware_loss.py`
- `tests/test_world_model_training_validation.py`
- `tests/test_world_model_v2_s0_training_contract.py`

### PR 3: Model Eval

Add exact-generation validation, checkpoint selection, test inference, and
metric publication.

Evaluation files:

- `scripts/build_world_model_v2_s0_generation_bundle.py`
- `scripts/run_world_model_v2_s0_exact_generation.py`
- `scripts/score_world_model_v2_s0_validation.py`
- `scripts/build_world_model_v2_s0_validation_campaign.py`
- `scripts/select_world_model_v2_s0_method.py`
- `scripts/evaluate_world_model_v2_s0.py`
- `scripts/publish_world_model_v2_s0_validation_to_training_wandb.py`

Extend `world_model_v2_s0.slurm` with `JOB_MODE=validation` and `JOB_MODE=test`.

Focused test files:

- `tests/test_run_world_model_v2_s0_exact_generation.py`
- `tests/test_score_world_model_v2_s0_validation.py`
- `tests/test_select_world_model_v2_s0_method.py`
- `tests/test_evaluate_world_model_v2_s0.py`
- `tests/test_world_model_v2_s0_eval_wandb_append.py`
- `tests/test_world_model_v2_s0_evaluation_contract.py`

Do not commit these generated source and artifact paths:

- `data/world_model_v2/sources/ensembl_116_human_gene_ids_and_symbols.json`
- `data/world_model_v2/sft/s0_human_identifiers_v4/`
- `data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/`
- Checkpoints, receipts, logs, or W&B exports

## Non-goals

- Do not use MyGene or HGNC data in the v4 corpus. A later corpus version can
  add these sources.
- Do not include mouse genes or another species in the first S0 campaign.
- Do not use graph-dependent facts or continuous values.
- Do not select a tokenizer from train loss.
- Do not add old evaluation shell launchers or separate Slurm files.
- Do not treat a prior typed-wrapper, pure-atomic, or atomic-prefix run as an
  S0 matrix result.
