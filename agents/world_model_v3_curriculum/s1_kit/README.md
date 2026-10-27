# S0/S1 contract validation

The shared wire contract is
[`runtime/world_model/contract.yaml`](../../../runtime/world_model/contract.yaml).
The selected S1 overlay owns versioned allocations, reasoning, prompt pins,
and output paths. Published manifests and rows identify the exact corpus.

- [Question classes, gold traces, and system prompt](../S1-question-classes.md):
  66 S1 types, actual split counts, tool-call counts, outcomes, builder files,
  full worked traces from the finalized corpus, and the approved v7 prompt.
- [V7 prompt experiment](../S1-prompt-v7.md): prepared in the isolated
  `../mentor-rl-s1-prompt-v7/` checkout, with an 8,192-token training cap.
- [Stage design and review history](../S1-coordinate-system.md).
- `runtime/world_model/golden_traces.jsonl` retains the shared golden wire
  examples with v5 answer semantics.

S1 v6 is frozen for training job `5712804` and its queued evaluations.
Its files are `data/world_model_v4/corpora/s1_v6/` and
`data/world_model_v4/eval/s1_v6/`. Do not rebuild published v1–v6 outputs
in place. V7 writes separate `s1_v7/` directories in the isolated checkout.
S1.5 remains disabled until its context ontology is available.

Run read-only checks from the appropriate repository root in the established
`pytorch-rocm` environment:

```bash
python scripts/render_world_model_s1_catalog.py --check
python agents/world_model_v3_curriculum/s1_kit/validate.py \
  --overlay runtime/world_model/s1_v5.yaml runtime/world_model/golden_traces.jsonl
PYTHON=python agents/world_model_v3_curriculum/s1_kit/selftest.sh
python agents/world_model_v3_curriculum/s1_kit/validate.py \
  --overlay runtime/world_model/s1_v6.yaml \
  data/world_model_v4/corpora/s1_v6/train.jsonl
```

In the v7 checkout, validate its rows with `--overlay
runtime/world_model/s1_v7.yaml`. Each family builder validates all three
splits before publishing its output. See the experiment document for the
complete rebuild, tokenizer audit, config pinning, and training preflight.
Use the overlay on every family builder and on the aggregate builder.

The frozen source index is:

```text
data/world_model_v3/source_indexes/world_model_source_index_08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz
```

It contains the release-116 registry, HGNC historical-symbol and retired-ID
pools, multiplex membership, native layer IDs, and eligible MENTOR-EV clades.
S1.4 also reads the frozen MyGene cache pinned in the contract. Use these
verified snapshots for reproduction. `python -m runtime.world_model.contract
check` also checks live external source paths; historical dendrogram/flist
availability or changes can make that separate live-source check fail.
Do not replace a pinned snapshot to accommodate current live files.

After changing the semantic validator, run its fixture and mutation checks:

```bash
PYTHON=python agents/world_model_v3_curriculum/s1_kit/selftest.sh
PYTHON=python agents/world_model_v3_curriculum/s1_kit/mutate.sh
```

The fixtures exercise 53 known failures. Exit codes are 0 for clean, 1 for
violations, and 2 or more when validation could not run.

Regenerate the consolidated document from published rows and the prompt asset:

```bash
python scripts/render_world_model_s1_catalog.py
python scripts/render_world_model_s1_catalog.py --check
```

Use this standalone renderer/checker in both checkouts. The main checkout's
runtime CLI is frozen for the v6 jobs: its old `contract render` and
`contract check-doc` commands still target the retired golden Markdown format
and must not regenerate the current document. In the v7 checkout those CLI
commands use the current catalog with full corpus traces. Golden-fixture
updates are checked against the JSONL with the semantic validator shown above.
