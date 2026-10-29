# Curriculum Data Dashboard Spec

Oct 27, 2026 · @Peter Kruse

The dashboard is a static site that one Python script builds on Frontier from each curriculum stage's manifests and a sample of its rows: one index page, one page per stage (S0–S9), and one section per question family.

## Scope

The dashboard answers five questions per stage: what was generated, where it lives, how it was built, how it splits, and whether its gate cleared. It is a read-only report over finished corpora.

In scope:

- One page per stage (S0–S9), plus an index page that compares stages.
- One section per question family, with counts and a fixed sample of rendered traces.
- Every path, size, row count and SHA-256 that a methods section or reviewer would cite.
- Checks that the manifests, the files and the sample agree.

Out of scope:

- Training curves and checkpoints. Weights & Biases already covers these.
- Browsing the full corpus. The site shows a sample, never every row.
- Answer keys. The site reads evaluator counts and hashes only, and never copies a protected evaluator file.
- Writing to data. The build never writes into a corpus or evaluator directory.

## Architecture

Two scripts run in sequence: a collector reads each corpus and writes one summary per stage, and a renderer turns the summaries into static HTML. Keeping them apart means a layout change reruns in seconds, and summaries can be diffed between corpus versions.

1. **Read the registry.** `config/dashboard/stages.yaml` lists each stage, its corpus root, its adapter and its status (see Stage registry).
2. **Collect.** For each stage marked `built` or later, `python -m dashboard.collect` reads the manifests and streams each JSONL once. It counts rows by family × form × split × outcome and draws a fixed-seed sample.
3. **Check.** The collector runs the checks below and records each result in the summary. A failed check marks that stage red but does not stop the other stages.
4. **Render.** `python -m dashboard.render` reads only the summaries and writes the site with Jinja2 templates. Expandable traces use `<details>`; one small script handles family filters.
5. **Copy or serve.** The output folder is self-contained, with sample data inlined into each page, so it opens from disk without a server.

Per-stage adapters absorb schema differences. S0 v4 and S1 rows follow `runtime/world_model/contract.yaml`; S2 onward will add graph metadata. Each adapter maps its rows to the common fields in Summary schema, so a new stage needs one adapter and no renderer change.

S0 uses the completed v4 corpus at `data/world_model_v4/corpora/s0/`, built by `runtime/world_model/s0_corpus.py`. All four v4 training arms passed the gate; their results are recorded in [the S0 stage document](S0-human-gene-identifiers.md#results-v4). The retired v6 tool-trajectory generator is not needed. Fill in the S0 adapter and remaining registry fields in build step 6.

For S1 counting logic, use `scripts/render_world_model_s1_catalog.py`, restored from `codex/s1-v10-to-main`. It checks per-type training counts against each component manifest, counts tool calls only after the last user message, and selects examples by call count, status, continuation decision, and blocked-reason code. Its default corpus is the frozen v10 corpus. The S1 question catalog is regenerated from those rows and verified with `--check`.

Suggested layout in the repo:

```text
dashboard/
  collect.py          # registry -> adapters -> summary.json + samples
  render.py           # summaries -> site/
  checks.py
  adapters/
    s0_contract_v4.py
    s1_contract_v1.py
  templates/
    index.html.j2
    stage.html.j2
    family.html.j2
    trace.html.j2
config/dashboard/stages.yaml
site_data/<stage>/summary.json, samples.jsonl   # git-ignored
site/index.html, stages/S0.html ... S9.html, assets/   # git-ignored
```

## Stage registry

One YAML file, `config/dashboard/stages.yaml`, is the only place the dashboard learns where data lives. Families, forms and target shares come from each stage's contract, so the dashboard never keeps a second copy of the curriculum proportions.

| Field | Example (S1) | Use |
| --- | --- | --- |
| `stage` | `S1` | Page id and sort order |
| `title` | Lattice coordinate system | Page heading |
| `status` | `built` | `planned`, `designing`, `built`, `training` or `gated`; only `built` and later are collected |
| `adapter` | `s1_contract_v1` | Maps rows to the common fields |
| `corpus_root` | `data/world_model_v3/corpora/s1` | Manifest and split files |
| `evaluator_root` | evaluator directory, or `null` | Read for its manifest only |
| `contract` | `runtime/world_model/contract.yaml` | Families, forms, target shares, tools |
| `golden_traces` | `runtime/world_model/golden_traces.jsonl` | Reviewed examples shown beside the sample |
| `question_doc` | `S1 Question Classes.md` | Linked from each family heading |
| `generator` | `scripts/build_world_model_s1_bundle.py` | Shown in provenance |
| `validator` | `agents/world_model_v3_curriculum/s1_kit/validate.py` | Run by the validator check when set |
| `gate_report` | `test_metrics.json` path, or `null` | Feeds the Gate section |
| `notes` | `dashboard/notes/S1.md` | Prose for the Construction section |
| `sample_seed` | `0` | Fixed seed for the trace sample |

Stages still in design, such as S2 today, appear on the index with their status and question-doc link but get no stage page. Replay rows stay with the stage that trains on them: S1's 276 S0 rows count toward S1 and are shown as a separate line.

## Inputs

The collector reads the corpus manifest, the split files, the evaluator manifest and the stage contract. S0 v4 and S1 component manifests supply the fields below; the S1 adapter also reads the aggregate manifest for component membership and replay counts.

| File | Fields read | Feeds |
| --- | --- | --- |
| Corpus/component `manifest.json` | `dataset_id`, `schema_version`, `corpus_schema_version`, `contract`, `implementation`, `row_counts`, `form_counts`, `files`; aggregate S1 adds `components`, `family_counts`, `s0_replay` | Header, provenance, checks |
| Corpus/component `manifest.json` | `assignment` seed and split units or hashes, per-split `row_counts` and `form_counts` | Splits and leakage |
| `<corpus_root>/train.jsonl`, `validation.jsonl` | Per row: `row_id`, `messages`, `tools`, `metadata`; the adapter maps `row_id` to summary `record_id` | Counts, distributions, sample |
| `<evaluator_root>/manifest.json` | Validation and test `row_count`, paths and hashes | Counts and hashes only |
| Stage contract | Families, forms, form names, target share, tool vocabulary | Labels, target vs realized |
| Golden traces | All rows | Reference example per form |
| Gate report | Metric, floor, value, pass | Gate section |

Rules for the collector:

- Never open `validation_answer_key.jsonl`, `test_answer_key.jsonl` or `test_questions.jsonl`. The dashboard must not copy protected evaluator rows out.
- Read trace fields through the adapter. S0 v4 and S1 rows carry an ordered `messages` list; adapters preserve the turns and identify masked context before the current user message.
- Read hidden metadata (`book_mode`, `step`, `question_family`, form, framing, `blocked_reason_code`) for counting and labels only. It appears on the page in the backend strip, never mixed into a turn.

## Summary schema

Each stage's `summary.json` is the contract between collector and renderer: the renderer reads nothing else. Write it as a dataclass with a version string, and keep a fixture summary in the tests. Row counts below are the S1 v1 corpus; other values are placeholders.

```json
{
  "schema_version": "mentor-rl-dashboard-summary-v1",
  "stage": "S1",
  "title": "Lattice coordinate system",
  "status": "built",
  "built_at": "2026-10-27T15:02:11Z",
  "repo_commit": "<git rev-parse HEAD>",
  "dataset_id": "<manifest dataset_id>",
  "provenance": {
    "generator": "scripts/build_world_model_s1_bundle.py",
    "config": {"path": "<config path>", "sha256": "<sha256>"},
    "contract": {"path": "runtime/world_model/contract.yaml", "sha256": "<sha256>"},
    "registry": {"id": "sha256:<...>"},
    "graph_version": null
  },
  "rows": {"train": 55276, "validation": 6875, "test": 6875},
  "replay": {"source_stage": "S0", "train_rows": 276},
  "share": {"target": 0.04, "realized_rows": null, "realized_tokens": null},
  "book_mode": {"closed_book": 0, "open_book": 0, "tool_call": 0},
  "families": [{
    "family_id": "S1.3",
    "question_family": "<metadata value>",
    "title": "Ambiguous symbol resolution",
    "disabled": false,
    "rows": {"train": 0, "validation": 0, "test": 0},
    "forms": [{"form": 5, "name": "Multiplex keeps one", "train_rows": 0}],
    "framing": {"direct": 0, "step": 0},
    "outcome": {"resolved|continue": 0, "ambiguous|defer": 0},
    "tool_calls_per_row": {"0": 0, "1": 0, "2+": 0},
    "harness_block_rows": 0,
    "tokens": {"p50": 0, "p95": 0, "max": 0},
    "sample_record_ids": ["<record_id>"],
    "golden_ids": ["S1.3.5"]
  }],
  "tools": [{"name": "check_multiplex_membership", "introduced_at": "S1",
             "arguments": ["multiplex_id", "gene_ids"], "calls": 0,
             "rows_with_call": 0, "result_status": {"resolved": 0}}],
  "files": [{"role": "train", "path": "<absolute path>", "bytes": 0,
             "rows": 55276, "sha256": "<sha256>", "manifest_match": true}],
  "splits": {"unit": "identifier graph component", "seed": 0,
             "held_out_groups": {"validation": 0, "test": 0}, "overlap": 0},
  "gate": {"report": null, "metrics": []},
  "checks": [{"id": "hash.train", "ok": true, "detail": ""}]
}
```

`samples.jsonl` sits beside it: one object per sampled row, holding `record_id`, the backend metadata strip and the ordered turns, already normalized by the adapter.

## Pages

The site has two page types: the index compares stages, and a stage page explains one corpus from top to bottom.

**Index page.** One table, one row per stage S0–S9:

| Column | Source |
| --- | --- |
| Stage, title | Registry |
| Status | Registry |
| Book mode | Contract: closed, open, open + tool, or tool |
| Train / val / test rows | Summary `rows` |
| Target share | Contract (S1 is 4%) |
| Realized share | Train rows, and tokens when counted, over all built stages |
| Families | Count, linked to the stage page |
| Checks | Passed / total; red when any fail |
| Gate | Pass, fail or not run, with the report date |
| Built | `built_at` and short commit |

Add a chart of target vs realized share once three or more stages are built; until then the table carries it.

**Stage page**, in this order, with a side nav listing the families:

1. **Header.** Stage, title, status, dataset id, build date, commit and a check badge.
2. **At a glance.** Rows per split, replay rows, book-mode mix, family and tool counts, target vs realized share.
3. **Construction.** Short prose from the stage's `notes` file: how questions are sampled, how ground truth is computed, what is held out. This is the seed of the stage's methods subsection.
4. **Question families.** One section per family (next section).
5. **Tools.** The tool table.
6. **Splits and leakage.** Split unit, seed, held-out group counts and overlap results. Later stages add held-out contexts, modules and seed genes.
7. **Gate.** Each metric with its floor, value and pass, linked to the report path.
8. **Files and provenance.** The file table, plus config, contract, registry and generator.
9. **Checks.** Every check with its result and detail.

## Question families and traces

Each family section shows what the family teaches, how its rows divide, and real examples, so a reader can judge it without opening a JSONL file.

Per family, in order:

- **Heading.** Family id and title (`S1.3 Ambiguous symbol resolution`), the `question_family` value, and a link to the family in the question doc. A disabled family (S1.5 today) shows its heading and the reason only.
- **Counts.** Rows by split, and train rows by form, with form names from the contract.
- **Outcome mix.** A small `status` × `continuation_decision` table. It shows at a glance whether a family is balanced, such as true vs false rows in S1.8.
- **Shape.** Direct vs step framing, tool calls per row (0, 1, 2+), rows with a harness block, and token length p50 / p95 / max.
- **Examples.** One collapsed `<details>` per form, opening to one sampled train row, with a toggle to the matching golden trace when one exists.

Trace rendering follows the question-doc layout:

- A muted backend strip sits above each trace, as in the S2 question classes: `question_family · form · framing · book_mode · blocked_reason_code · split · record_id`. The model never sees these fields, and the styling makes that plain.
- Model-facing turns follow, each label on its own line: Context (from the harness), User, Thinking, Tool call, Tool result, Thinking, Answer.
- JSON is pretty-printed with a 2-space indent and wrapped near 100 characters. `<GENE:…>` and `<LAYER:…>` anchors in summaries are highlighted.
- In multi-turn rows, earlier turns are greyed and marked "in context, masked", as in S1.3.8.
- The system prompt and tool definitions appear once per stage in a collapsed block, not on every trace.

An expanded S1.1 example would read:

```text
question_family=<S1.1 value> · form=1 · framing=direct · book_mode=tool_call · split=train

User:
Find the human Ensembl gene ID for VARS1.

Thinking:
The task gives only the symbol VARS1. I will look it up in the registry.

Tool call:
lookup_human_gene_symbol({"gene_symbol": "VARS1"})

Tool result:
{"status": "resolved", "gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}

Thinking:
The registry maps VARS1 to one ID, ENSG00000204394. The task asks only for
this ID, so the task is complete.

Answer:
{
  "status": "resolved",
  "result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"},
  "continuation_decision": "stop",
  "summary": "VARS1 is <GENE:ENSG00000204394|VARS1> in Ensembl release 116."
}
```

Sampling rules:

- A fixed seed per stage (`sample_seed`), so a rebuild shows the same rows until the corpus changes.
- Stratified by form: up to 3 train rows per form, one shown and two behind "more". Every form with rows gets at least one.
- Never from test. Validation only when a form has no train rows.

## Tool calls and paths

Two tables close each stage page: which tools the rows exercise, and where every file lives.

**Tool table**, one row per tool in the stage's tool definitions:

| Column | Meaning |
| --- | --- |
| Tool | Name, such as `lookup_human_gene_symbol` |
| Introduced | First stage whose definitions list it |
| Arguments | Argument names from the definition |
| Calls (train) | Total calls across train rows |
| Rows with a call | Share of train rows that call it at least once |
| Result status | Counts of `resolved`, `ambiguous`, `not_found`, `invalid_input` and `error` in its results |
| Families | Families whose rows call it |
| Example | Link to one sampled trace that calls it |

A tool listed in the prompt but never called still gets a row with zero calls. Closed-book rows list tools too, so zero-call rows are expected, and the table should show them.

**File table**, one row per file:

| Column | Meaning |
| --- | --- |
| Role | train, validation, split manifest, corpus manifest, evaluator manifest, config, contract, golden traces or gate report |
| Path | Absolute Lustre path, with a copy button |
| Size | MB or GB |
| Rows | JSONL files only |
| SHA-256 | First 12 characters; full value on hover |
| Matches manifest | Yes, no, or not listed |
| Modified | File mtime |

Evaluator files show path, rows and hash only. From S2 on, add the cached ground-truth stores (graph store, RWR caches) with path and total size, since the proposal expects 2–4 TB of cached ground truth.

## Checks

Every build runs these checks per stage, records them in `summary.json` and lists them on the stage page. A failed check turns the stage red on the index, but the build still finishes.

| Check | Passes when |
| --- | --- |
| Hashes | Each file's SHA-256 equals its `file_hashes` entry in the manifest |
| Row counts | Lines in each JSONL equal `row_counts` for that split |
| Family counts | Counted rows per family equal the manifest's `family_counts` |
| Unique ids | `record_id` is unique within and across splits |
| Split groups | No split unit (fact group, context, module or seed gene) appears in two splits |
| Hidden fields | No user, harness or system text contains a hidden label: the step name, the book mode or the `question_family` value |
| Forms covered | Every contract form has at least one train row, or is marked disabled |
| Replay share | Replay rows are 0.25–1% of train rows, as the S1 design requires |
| Validator | The stage's validator exits 0 on each split; runs only with `--run-validator`, since it is slower |
| Sample render | Every sampled row parses and renders with no missing label |

To prove the checks work, keep a small broken fixture per check in the tests (one bad hash, one duplicate id, one leaked label) and assert that each turns red.

## Running it on Frontier

The build is plain Python. It runs on a login node at S0–S1 sizes, and as a short batch job once corpora reach millions of rows.

Use the repo's `pytorch-rocm` environment. The only extra packages are `jinja2` and `pyyaml`; check `pip list` before installing. Token counts load the tokenizer named in the stage's training config.

```bash
# One stage, then the site
python -m dashboard.collect --registry config/dashboard/stages.yaml --stage S1
python -m dashboard.render --data site_data --out site

# Every built stage, with token counts and the validator
python -m dashboard.collect --registry config/dashboard/stages.yaml --all \
  --tokens --run-validator
```

For large corpora, add a `dashboard.slurm` that runs one collect task per stage on a single node, then renders. Each JSONL is streamed once, so memory stays flat as row counts grow.

To view the site:

- **On your laptop.** `rsync -a <frontier>:<repo>/site/ ./mentor-dashboard/`, then open `index.html`. Sample data is inlined, so no server is needed.
- **Through VS Code Remote SSH.** Run `python -m http.server 8765 --directory site` and let VS Code forward the port.
- **For your advisor or committee.** Copy the folder to any static host, or send me the `site_data` summaries and I can publish them as a hosted page.

## Build order

Build S1 first and use it as the template, since its corpus, contract and golden traces all exist and it has the most families.

- [ ] Add `config/dashboard/stages.yaml` with S0–S9: S0 and S1 `built`, S2 `designing`, the rest `planned`.
- [ ] Write the summary dataclass and a fixture summary for tests.
- [ ] Write the S1 adapter against `data/world_model_v3/corpora/s1/` and the contract. Confirm 55,276 train (276 replay), 6,875 validation and 6,875 test rows.
- [ ] Write the templates: index, stage page, family section, trace. Compare one rendered trace per S1 family against the question doc.
- [ ] Add the checks, with one broken fixture per check.
- [ ] Write the S0 adapter and render the S0 page.
- [ ] Add the Gate section once the S1 gate report exists.
- [ ] Write construction notes for S0 and S1 in plain prose.
- [ ] Add the S2 adapter when the S2 corpus is built; its graph metadata adds multiplex, layer scope and graph version to the header.
- [ ] Add token counts and the share chart once three stages are built.

## Open questions

- Count realized share in rows or tokens? The proposal states targets against both \~3 million examples and \~1.2 billion tokens; tokens track compute better.
- Keep a page per corpus version (S1 v1, then the rebuild with S0 v4 replay), so changes between rebuilds are visible?
- Do construction notes live in `dashboard/notes/`, or get pulled from the existing design docs?
- Who reads the site besides you? If your advisor or committee does, hosting it off Frontier matters early.
