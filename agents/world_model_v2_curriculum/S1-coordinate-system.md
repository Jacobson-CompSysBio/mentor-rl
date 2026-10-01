# S1: Coordinate System and Conventions

This document defines the canonical S1 contract, draft 1 (2026-09-30).

## Sources of truth

Use these sources, in this order:

1. The S1 question classes file, `S1-question-classes.md`. Keep it beside
   this file. Refer to a trace by its ID.
   - It is the advisor's file, with the agreed trace fixes and the decisions
     in this document applied (2026-09-30).
   - It has 66 worked traces: S0.1 to S0.7, S1.1.1 to S1.8.6, and the new
     S1.6.7 and S1.7.9.
2. The advisor's kit:
   - four YAML contract files: the vocabularies, the three layer tag
     grammars, the tool signatures, and the provenance rule
   - `validate.py`
   - `selftest.sh`, with 21 broken traces
   - `mutate.sh`

   The kit is not in the repo yet. Add the commands to this file when you
   commit the kit. Update its YAML contracts first:
   - `provenance` on every tool result and answer, with `evidence_id`
   - the registry block as JSON
   - `blocked_reason` as a sentence
   - the five-field bulkPEN and TFs grammar
3. `MENTOR-RL-World-Model_v3.md` (2026-09-29): the Data Catalog S1 section,
   the answer envelope, the layer tag grammars, the layer claim vocabulary,
   and the RWR++ Tool API contract.
4. This document.

If this document disagrees with sources 1 to 3, sources 1 to 3 win. Record
the difference in "Open questions." The exception is the decisions in
"Decisions that differ from the advisor's sources." They hold until the
advisor reviews them.

If the v3 text disagrees with a trace, the trace wins, because the advisor
edited the traces last. "Differences between v3 and the traces" lists each
case.

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
| S1.1 Symbol to Ensembl | `human_symbol_to_ensembl` | Tool call | Ready | S1.1.1–S1.1.8 |
| S1.2 Ensembl to symbol | `human_ensembl_to_symbol` | Tool call | Ready | S1.2.1–S1.2.11 |
| S1.3 Ambiguous symbol | `human_ambiguous_symbol` | Tool call | Ready | S1.3.1–S1.3.10 |
| S1.4 Cross-context alignment | `cross_context_alignment` | Tool call | Ready | S1.4.1–S1.4.8 |
| S1.5 Context ID resolution | `context_id_resolution` | Tool call | Blocked: context ontology data (Manesh and Erica) | None |
| S1.6 Layer tag parsing | `layer_tag_parsing` | Closed book | Ready. bulkPEN and TFs tag form frozen. | S1.6.1–S1.6.7 |
| S1.7 Layer family and supported claims | `layer_family_claims` | Closed book | Ready for all six families, `other` included | S1.7.1–S1.7.9 |
| S1.8 Direction conventions | `direction_conventions` | Closed book | Ready | S1.8.1–S1.8.6 |
| S1.9 Module ID parsing | `module_id_parsing` | Closed book | Blocked: updated MENTOR-EV dendrogram files (Ken) | None |

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

Every S1 row uses the same system prompt. The prompt must not tell the model
the book mode of the row.

- The prompt is the v3 core prompt plus one S1 overlay.
- The overlay lists the S1 tools in every row, closed-book rows included. If
  the tool list changed with the book mode, the list itself would tell the
  model whether to call.
- The model decides from the task whether to call a tool. On a convention row
  it answers from the conventions it learned and makes no call.
- The overlay does not name the book mode, the step, or the question family.
  Use one overlay text that fits both tool-call and closed-book rows. Do not
  use the v3 per-book-mode overlays.
- Keep the bootstrap toggle on until the lattice exists. The current graph is
  the only context.
- Edit the identifier rule of the v3 core prompt. It says to carry "a
  blocked_reason of ambiguous_symbol." Under this spec, the answer carries the
  `ambiguous_symbol` sentence, and the code stays in the backend.
- Add an S1 `StagePromptContract` to `runtime/world_model_prompts.py`:
  - `allowed_book_modes`: `("tool_call", "closed_book")`
  - `source_document`: `agents/world_model_v2_curriculum/S1-coordinate-system.md`

  All S1 rows then have the same `system_prompt_sha256`.

The S0 prompt ("Use exactly one provided tool", "one short sentence") does
not apply to S1.

## Answer envelope

The final answer is one JSON object. Use the keys in this order:

| Key | When | Values |
|---|---|---|
| `status` | Always | `resolved`, `ambiguous`, `not_found`, `invalid_input`, `error` |
| `provenance` | Always | `{"registry": "ensembl", "release": "116", "evidence_id": ...}`; `null` when the answer rests on no registry evidence |
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

- **Fixed templates, for now.** Each reason code has one sentence template.
  The runtime fills the slots from the result.
- **The code stays in the backend.** The row metadata records
  `blocked_reason_code`, hidden from the model like `question_family`. The
  validator and the Stage 2 scorer read the code. They check that the
  sentence equals the template for that code, filled from the result.

| Code | `status` | Template | Traces |
|---|---|---|---|
| `ambiguous_symbol` | `ambiguous` | The symbol {symbol} matches {n} genes in Ensembl release 116, and nothing in the task chooses one. | S0.5, S1.3.1–S1.3.4, S1.4.6 |
| | | The symbol {symbol} matches {n} genes, and {k} of them are nodes in {multiplex_id}, so the multiplex does not choose one. | S1.3.6 |
| | | The reply does not choose one of the {n} candidates for {symbol}. | S1.3.9 |
| `not_found` | `not_found` | The registry has no gene with the symbol {input} in Ensembl release 116. | S0.6, S1.1.3, S1.4.5 |
| | | The registry has no gene with the ID {input} in Ensembl release 116. | S0.7, S1.2.5 |
| `invalid_input` | `invalid_input` | The input {input} is not a well-formed {expected}. | S1.2.6, S1.6.6 |
| `wrong_input_type` | `invalid_input` | The input {input} is a {found}, not a {expected}. | S1.6.5, S1.7.5 |
| `no_multiplex_supplied` | Planned | The task asks about multiplex membership, but it names no multiplex. | None |
| `no_multiplex_membership` | `not_found` | No candidate for {symbol} is a node in {multiplex_id}. | S1.3.7 |
| `out_of_scope_species` | `invalid_input` | The input {input} is a {species} gene {kind}, and the registry holds human genes only. | S1.1.7, S1.2.10 |
| `unrecognized_metric` | `ambiguous` | The name "{input}" can mean {c1} or {c2}, and the two have opposite directions. | S1.8.5 |
| | `not_found` | The metric "{input}" is not in the direction conventions. | S1.8.6 |
| `tool_error` | `error` | The call to {tool} failed with {error_code}, so there is no registry result for {input}. | S1.1.8, S1.2.11 |
| | Status of the successful lookup | The lookup found {n} candidates, but the call to {failed_tool} failed with {error_code}, so no candidate is checked in {multiplex_id}. | S1.3.10 |

- `blocked_reason` explains the blocker. The summary tells the user what to
  give to continue.
- The earlier-turn answers in S1.3.8 and S1.3.9 carry the sentence too.

### `provenance`

`provenance` is a field of every tool result and every answer. The key is
always present, so the model decides its value, not whether to write the key.

- **Tool results.** Every tool result carries `provenance`. The tool fills it:
  - A successful registry call returns the registry block.
  - An error result has `"provenance": null`, because a failed call reads no
    record.
- **Answers with registry evidence** carry the block. Copy every field from
  the evidence that the answer rests on:
  - this turn's tool result
  - a harness registry block: rows with no call, for example S1.1.4, S1.2.7,
    S1.3.3, and S1.4.7
  - an earlier turn in context (S1.3.8, S1.3.9)

  The validator compares the answer with that evidence. In the question
  file, 38 answers carry the block.
- **Answers with no registry evidence** have `"provenance": null`:
  - convention families (S1.6 to S1.9)
  - rejections with no call (S1.1.7, S1.2.10)
  - answers that rest on failed calls alone (S1.1.8, S1.2.11)
  - ID-to-ID comparisons with no lookup (S1.4.3, S1.4.4)

  The updated question file carries the key on all of them.
- **`evidence_id` is required.**
  - The runtime makes a new random `evidence_id` for each call, for example
    `ev_reg_7f3k2q`.
  - It does not derive the ID from the input, and it does not reuse IDs
    across rows. So the model can only copy it.
  - The answer copies the `evidence_id` of the evidence it rests on. Without
    a changing value, a copied block and a memorized block are the same
    string.
- An answer that rests on a lookup and a membership check (S1.3.5 to S1.3.7)
  copies the provenance of the lookup. The provenance of the membership result
  is open (see "Open questions").
- A partial failure keeps the provenance of the successful lookup (S1.3.10).
- Never invent a provenance block. A block with no evidence teaches the model
  to fabricate one. `null` is the only value for an answer with no registry
  evidence.

### `result` shapes shared across families

- One gene: `{"gene_id": ..., "gene_symbol": ...}`. `gene_symbol` is `null`
  when the gene has no symbol.
- Invalid input: `{"input": ..., "expected": ...}`. `expected` is one of
  `human_gene_symbol`, `human_gene_id`, or `layer_tag`.
- **Tool failure, answer on failed calls alone:** `{"tool": ...,
  "error_code": ..., <input key>: ...}`, for example `"gene_symbol": "VARS1"`.
- **Tool failure, partial:** the lookup result, plus `"failed_tool"` and
  `"error_code"`.

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
- `<MULTIPLEX:…>` appears once (S1.3.10). Do not use it until the advisor
  confirms it. See "Open questions."

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

any tool, transport failure
  {"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": ...}

P = {"registry": "ensembl", "release": "116", "evidence_id": <new for each call>}
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
- The membership result carries the registry block in S1.3.5 to S1.3.7. v3
  gives graph tools a graph provenance block. See "Open questions."
- **Registry.** Include every Ensembl release 116 human gene, named or
  unnamed.
  - Set `gene_symbol` to `null` when the GTF `gene_name` is missing or equals
    the gene ID.
  - Store the chromosome and `gene_biotype` of each gene, for
    `candidate_details`.

### Runtime changes required

`runtime/world_model_s0_tools.py` does not yet meet this contract:

| Today | Required |
|---|---|
| Exact match; whitespace raises an error | Trim spaces; ignore case; collision check |
| A noncanonical ID raises an error | Strip `.N` only; return `invalid_input` as a payload |
| The ID tool returns `gene_symbols` (a list) | `gene_symbol` (one value or `null`) |
| Ambiguous results have `"action": "defer"` | Remove `action` |
| No `candidate_details` | Add chromosome and biotype |
| Provenance sits in a separate object (`ensembl_release: "Ensembl_116"`, no `registry` key), and the S0 builder shows the payload only | Add `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": ...}` to every result that the model sees. Make a new random `evidence_id` for each call. In the log, map each `evidence_id` to the full internal provenance. |
| Bad input raises `ToolExecutionError`, and there is no error payload | Return the error payload with `"provenance": null`, with a switch that injects failures for training rows |
| No membership tool | Add `check_multiplex_membership` |

Also:

- `scripts/grab_human_gene_ids.py` drops unnamed genes. Keep them, and add
  the chromosome and the biotype.
- Change the registry schema version and the tool contract version.

## Harness context blocks

- **Registry block.** The block has the header `Registry entries, Ensembl
  release 116:`, then one JSON object:

  ```
  {"provenance": {"registry": "ensembl", "release": "116", "evidence_id": ...},
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
- **Multiplex IDs.** Use only multiplexes that exist in the store. Today that
  is `full_brain_multiplex_v1`.
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
- **Masked-wording check.** Train a simple classifier on the prompts with the
  identifiers masked. Its form accuracy must be near the share of the largest
  form.
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
- **Splits.**
  - Identity families: the S0 identifier graph components. Assign the new
    pools (unnamed IDs, retired IDs, previous symbols) to splits before
    generation.
  - Convention families: by context. Record the contexts of each row in its
    metadata.
- **Row metadata.** Each row has a metadata record. The backend reads it for
  generation, splits, validation, and scoring. The model does not see the
  hidden fields.

  | Field | Model sees it | Why |
  |---|---|---|
  | `book_mode` | No | It tells the model whether to call a tool |
  | `step` | No | It names the curriculum stage, which is not task content |
  | `question_family` | No | It tells the model the family, for example that a symbol is ambiguous (S1.3) before the lookup |
  | Form number, source trace ID, split unit | No | Backend only |
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

Every identity answer changed shape. Rebuild S0 as v7 in the S1 envelope
before you mix any S0 rows into S1.

- Use S0.1 to S0.7 as the trace pattern. Fill the placeholders from the
  registry, for example:

  | Trace | Fill with |
  |---|---|
  | S0.4 (unnamed gene) | ENSG00000299624 (applied) |
  | S0.5 (ambiguous symbol) | GOLGA2P9 (applied) |
  | S0.6 (absent symbol) | LINC02170 (applied) |
  | S0.7 (ID not in 116) | A retired ID from the pinned stable-ID history (still a placeholder) |

- Add the envelope fields: `provenance` (with `evidence_id`),
  `blocked_reason` (a sentence on every `defer` row), and the `error`
  status.
- The tool result carries `provenance` as a field that the model sees.
- Use the S1 system prompt and the S1 metadata rules.
- **Replay.** Keep 0.25–1% S0 v7 rows in the S1 data. Check the S0 gate again
  on the S1 checkpoint.
- If you replay old v6 rows, they teach the old envelope on the same prompts.

## S1.1 Symbol to Ensembl

- **Book mode:** tool call. Some rows make no call.
- **Input:** one gene symbol, optionally with a multiplex ID. Gene sets go to
  S1.4.
- **Tools:** `lookup_human_gene_symbol`, and `lookup_human_ensembl_gene_id`
  for an ID input.
- **Result:** `{"gene_id": ..., "gene_symbol": ...}`.

| # | Form | Share (provisional) | Call | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|
| 1 | Direct | 20% | Symbol tool | `resolved` | – | S1.1.1 |
| 2 | Inside a task | 20% | Symbol tool | `resolved` | – | S1.1.2 |
| 3 | Absent symbol (previous symbol or alias) | 15% | Symbol tool | `not_found` | `not_found` | S1.1.3 |
| 4 | Harness block lists the symbol | 15% | None | `resolved` | – | S1.1.4 |
| 5 | Block without the symbol (near miss) | 10% | Symbol tool | `resolved` | – | S1.1.5 |
| 6 | Ensembl gene ID given as the input | 10% | ID tool | `resolved` | – | S1.1.6 |
| 7 | Not a human gene symbol | 10% | None | `invalid_input` | `out_of_scope_species` | S1.1.7 |
| 8 | Lookup tool unavailable | Failure share | Symbol tool | `error` | `tool_error` | S1.1.8 |

- **Absent pool.**
  - Use previous HGNC symbols and aliases, including well-known aliases such
    as HER2.
  - Remove any string that matches a current registry symbol without regard
    to case. For example, VARS2 is a previous symbol of VARS1 and the
    approved symbol of another gene.
  - The answer never suggests the current symbol.
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

## S1.2 Ensembl to symbol

- **Book mode:** tool call. Some rows make no call.
- **Input:** one Ensembl gene ID, with or without a version suffix.
- **Tools:** `lookup_human_ensembl_gene_id`, and `lookup_human_gene_symbol`
  for a symbol input.
- **Result:** `{"gene_id": ..., "gene_symbol": ...}`.

| # | Form | Share (provisional) | Example | Call | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|---|
| 1 | Direct | 20% | ENSG00000255121 | ID tool | `resolved` | – | S1.2.1 |
| 2 | Inside a task | 15% | ENSG00000177830 | ID tool | `resolved` | – | S1.2.2 |
| 3 | Version suffix | 10% | ENSG00000099204.23 | ID tool | `resolved` | – | S1.2.3 |
| 4 | No symbol | 10% | ENSG00000300082 | ID tool | `resolved`, `gene_symbol` `null` | – | S1.2.4 |
| 5 | Not in release 116 | 10% | ENSG00000182378.14_PAR_Y, retired IDs | ID tool | `not_found` | `not_found` | S1.2.5 |
| 6 | Malformed ENSG | 5% | ENSG0000023134 | ID tool | `invalid_input` | `invalid_input` | S1.2.6 |
| 7 | Harness block lists the ID | 10% | ENSG00000230860 | None | `resolved` | – | S1.2.7 |
| 8 | Block without the ID (near miss) | 5% | ENSG00000177830 | ID tool | `resolved` | – | S1.2.8 |
| 9 | Symbol given as the input | 5% | CENATAC-DT | Symbol tool | `resolved` | – | S1.2.9 |
| 10 | Not a human ID | 10% | ENSMUSG00000112237 | None | `invalid_input` | `out_of_scope_species` | S1.2.10 |
| 11 | Lookup tool unavailable | Failure share | ENSG00000141510 | ID tool | `error` | `tool_error` | S1.2.11 |

- **v3 required negatives:**
  - a version suffix that must be accepted
  - a transcript accession
  - a non-human accession
  - a wrong digit count
  - an accession absent from release 116
  - a tool error

  A transcript accession has no trace yet. Treat an ENST input as
  `invalid_input` with `wrong_input_type`, and make no call. Take a verified
  ENST ID from the release 116 GTF.
- **Format check.** An input that starts with ENSG always goes to the tool.
  A non-gene or non-human Ensembl prefix (ENST, ENSP, ENSMUSG) gets no call.
- **No symbol is not a failure.** State that the gene has no known symbol, and
  show it by its ID. Do not refuse, do not invent a symbol, and do not call
  the ID invalid. Clone-based names from older annotations are not release
  116 symbols.
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

| # | Form | Context | Calls | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|---|
| 1 | Direct, no multiplex | None | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.1 |
| 2 | Inside a task, no multiplex | None | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.2 |
| 3 | Already marked | Block marks the symbol | None | `ambiguous` | `ambiguous_symbol` | S1.3.3 |
| 4 | Block for another symbol | Block marks another symbol | Symbol | `ambiguous` | `ambiguous_symbol` | S1.3.4 |
| 5 | Multiplex keeps one | Multiplex named | Symbol, membership | `resolved` | – | S1.3.5 |
| 6 | Multiplex keeps several | Multiplex named | Symbol, membership | `ambiguous` | `ambiguous_symbol` | S1.3.6 |
| 7 | Multiplex keeps none | Multiplex named | Symbol, membership | `not_found` | `no_multiplex_membership` | S1.3.7 |
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
  - None present: `not_found`. The summary asks for another gene or another
    multiplex.
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
- **Result (`LayerRef`):** all eight keys, always.

```
{"layer_id": ..., "layer_family": ..., "source": ..., "network": ...,
 "organ": ..., "region": ..., "cell_type": ..., "version": ...}
```

- A key that the grammar does not fill is `null`.
- The list form returns `{"layers": [LayerRef, ...], "filter": {"field": ...,
  "value": ...}, "matching_layer_ids": [...]}` (S1.6.4).

**Grammars.** The first colon-separated field selects the grammar. The build
mints the tags. Do not rename them.

| First field | Parts | Arity | `layer_family` |
|---|---|---|---|
| `HumanNetV3` | source : network : organ : cell type : version | 5 | From the network field (see S1.7) |
| `bulkPEN` | prefix : family : organ : region : version | 5 | `bulk_pen` |
| `scPEN` | prefix : organ : region : cell type : version | 5 | `scpen` |
| `TFs` | prefix : family : organ : region : version | 5 | `tf_target` |

- HumanNet tags have no region. For example, `HumanNetV3:string_ppi:global:all:v3`
  gives `organ: "global"`, `cell_type: "all"`, and `region: null`.
- bulkPEN and TFs tags have no source, no network, and no cell type. Field 2
  names the family, and it must match the prefix (`bulk_pen` for bulkPEN,
  `tf_target` for TFs). Field 4 fills `region`. In
  `TFs:tf_target:brain:all:v1`, `region` is `"all"`.
- scPEN tags have no source and no network.
- Layer tags and context addresses are separate namespaces. A string such as
  `brain:cortex:L5:microglia:v1` addresses a context, not a layer.

**Frozen tag form (2026-09-30).** bulkPEN and TFs use the five-field form
of the repo's lattice flist
(`data/world_model_v2/lattice/ws0_brain_v1/flists/full_brain_multiplex_v1.tsv`),
for example `bulkPEN:bulk_pen:brain:amygdala:v1` and
`TFs:tf_target:brain:all:v1`.

- The v3 three-field bulkPEN form (`bulkPEN:GTEx_v9-brain-hippocampus:v1`)
  is not used. S1.6.1, S1.6.4, and S1.7.7 now use the five-field form.
- The corrected PEN build must mint its bulkPEN tags in this form.
- HumanNetV3 and scPEN keep the v3 form. When the PEN rebuild lands, check
  their tags against the lattice flist.
- The older `data/full_brain_flist.tsv` names the TF layer `TFs:brain`. Do not
  use that name.

| # | Form | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|
| 1 | Direct parse | `resolved` | – | S1.6.1 (bulkPEN), S1.6.3 (HumanNet), S1.6.7 (TFs) |
| 2 | One field | `resolved` | – | S1.6.2 |
| 3 | List filter by field | `resolved` | – | S1.6.4 |
| 4 | Wrong type (a context ID or a module ID) | `invalid_input` | `wrong_input_type` | S1.6.5 |
| 5 | Malformed tag | `invalid_input` | `invalid_input` | S1.6.6 (missing version) |

- **Provisional shares:** 40% (form 1), 25% (form 2), 15% (form 3), 10%
  (form 4), 10% (form 5).
- **v3 required negatives for form 5:**
  - an unrecognized prefix
  - the right prefix with the wrong arity
  - a tag that mixes hyphens and colons
  - a missing version
  - a family part that does not match the prefix, for example
    `bulkPEN:tf_target:brain:amygdala:v1`
- **Rules:**
  - **Read the tag only.** A parse does not say that a layer is in a
    multiplex, and it makes no claim about edges.
  - **No inference.** Do not infer a part from biology. The region of
    `scPEN:brain:CA1:astrocyte:v1` is `CA1`, not "hippocampus."
  - **Copy values exactly.** Keep the case, the underscores, and the version.
    The one exception is `layer_family`, which comes from the S1.7 enum.
  - **Exact filters.** List filters compare field values exactly.
  - **Family agreement.** The `layer_family` of a parse agrees with S1.7.
  - **Thinking.** One Thinking step reads the tag in order: the prefix, the
    grammar, and each part.
- **Splits:** by context. Hold out whole contexts, so that the model learns
  the grammar and does not memorize the tag strings.
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
  - PEN and TF layers: the prefix settles it. In bulkPEN and TFs tags, field
    2 repeats the family.
  - HumanNetV3 layers: the network field settles it. `string_ppi` and
    `biogrid_ppi` give `ppi`. `coexpression` gives `coexpression`.
  - An unrecognized network gives `other`. Do not guess the nearest named
    family.

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
  `TFs:tf_target:brain:all:v1`, has no sign and no direction (see "Repo data
  findings"):
  - Its edges support `regulatory_link_recorded`.
  - They do not support `regulation_direction`.
  - They do not show which gene is the TF.

  No signed TF layer is planned for now. So `regulation_direction` is
  unsupported for every `tf_target` layer.
- `cofunctional_association`: an integrated functional network would support
  it. No current layer does, because the HumanNet layers in use are
  `string_ppi` and `coexpression`.
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
| 6 | Unrecognized network | `resolved`, family `other` | S1.7.8 |

- **Provisional shares:** 25% (form 1), 15% (form 2), 25% (form 3), 25%
  (form 4), 10% (forms 5 and 6 together).
- **Form 4.** About 40% of the claims asked are supported and 60% are
  unsupported. Weight the hard negatives:
  - coexpression and binding
  - `ppi` and regulation
  - `tf_target` and direction
  - a PEN layer and another context
  - bulk PEN and one cell type
  - any family and a causal effect
- **Coverage.**
  - The corpus covers all six families: `coexpression`, `scpen`, `ppi`,
    `bulk_pen`, `tf_target` (S1.7.9), and `other`.
  - S1.7.8 uses `HumanNetV3:cocitation:global:all:v3` as the unrecognized
    network. Vary the unrecognized network names in the generated rows.
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

| # | Form | Share (provisional) | Status | `blocked_reason` code | Trace |
|---|---|---|---|---|---|
| 1 | Direction lookup | 30% | `resolved` | – | S1.8.1 |
| 2 | True or false | 50% | `resolved` | – | S1.8.2–S1.8.4 |
| 3 | Ambiguous name ("percentile") | 10% | `ambiguous` | `unrecognized_metric` | S1.8.5 |
| 4 | Unknown metric (diffusion state distance) | 10% | `not_found` | `unrecognized_metric` | S1.8.6, after you replace it |

- **Name variants.** About 30% of rows use a variant name.
- **A bare name with several candidates:**
  - The candidates agree on the direction: `resolved`. For example,
    "distance" is lower-is-closer for both distances.
  - The candidates disagree: `ambiguous`, with `candidate_metrics`.
- **Form 2 balance.** Half of the statements are true and half are false, for
  each metric and each of the four polarities.
- **No assumptions.** The model answers from the table only, also for a
  metric that it may know from the literature.
- **`lca_depth` is not in S1.8.** S5 defines it.
- **Runtime fix before S1.8 data.** `get_distance` returns the dot product
  under the keys `distance` and `dissimilarity`. Fix it, so that a value named
  "distance" is always lower-is-closer.
- **Gate:**
  - 100% correct direction for each metric
  - at least 99% true-or-false accuracy, by metric and by polarity
  - at least 99% correct mapping of held-out name variants
  - 100% `defer`, with no direction claim, on forms 3 and 4
  - the masked-wording classifier near 50%
  - at most 1% tool calls

## S1.9 Module ID parsing

**Blocked** until the updated MENTOR-EV dendrogram files are ready.

- **Book mode:** closed book. The prompt lists the S1 tools. The model makes
  no call.
- **v3 shape:**

```
{"module_id": "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000112164:geometric_elbow_v1",
 "module_source": "rwr_loe", "seed_gene_id": "ENSG00000112164",
 "construction_rule": "seed_centered_rwr_loe_geometric_elbow"}
```

- For a MENTOR-EV ID, the source is `mentor_ev`, and the parse returns the
  dendrogram and clade identifiers.
- Tokenize the ID compositionally: atomic tokens for the source and the
  dendrogram, and digit tokens for the clade index.
- A parse never says that a module exists.
- A clade ID that is not in the context is recall, which the recall rule
  forbids.
- Freeze the grammar from the dendrogram files.

## Stage gate

The v3 S1 gate:

- On the identity families, tool-call correctness:
  - selection accuracy
  - schema-valid arguments
  - faithful reading of the result
  - correct deferral
- Near-perfect accuracy on the closed-book families.

Measure it with these floors, on held-out phrasings and held-out split units:

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
  - the block equal to the evidence on every row with registry evidence
  - `null` on every row with no registry evidence, so 0 invented blocks
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
- **Summary:** the summary rule checks pass.
- **The family gates above.**
- Report the false-defer rate: rows where the model deferred but the task was
  answerable.

## Validation

- Every trace in the question file, and every generated row, passes
  `validate.py` against the kit's YAML contracts before it enters the corpus.
  QC is all-or-nothing.
- Run `selftest.sh` after any change to the contracts.
- The kit validated the advisor's file, but it did not catch the problems
  listed in "Trace fixes." Add a broken trace to the selftest for each of
  these checks:
  - two traces with the same user text
  - a `blocked_reason_code` that does not fit the `status` (see the
    `blocked_reason` table)
  - a `blocked_reason` sentence that differs from its filled template
  - a tool result with no Thinking step after it
  - a provenance block whose type does not fit the tool that produced it
  - an answer or a tool result with no `provenance` key
  - an answer whose `evidence_id` differs from its evidence
  - an `evidence_id` that appears in more than one row
  - a system prompt or a visible field that differs by book mode, step, or
    question family
  - an anchor type that is not in the anchor list
  - claim lists that are not in vocabulary order
  - a summary that asks for input when the decision is not `defer`
  - a summary with an anchor that is not in the result
  - failure-result keys that do not fit the failure rule (`tool` versus
    `failed_tool`)

## Trace fixes

The updated question file applies these fixes to the advisor's traces
(2026-09-30):

1. **S1.8.6** is now the unknown-metric trace ("diffusion state distance",
   `not_found`, `defer`). It repeated S1.8.5 before.
2. **S1.3.10:**
   - The first Thinking step no longer plans the membership check.
   - A Thinking step now follows the lookup result.
   - It uses `full_brain_multiplex_v1`.
   - The summary uses no `<MULTIPLEX:…>` anchor.
3. **S1.1.8, S1.2.11, and S1.3.10:** the thinking is shorter and in STE
   style, and it does not mention provenance. S1.1.8 no longer says "stop."
4. **S1.7.8:**
   - The network is `cocitation`, not the placeholder "NETWORK_A."
   - The decision stays `stop`, and the summary no longer asks for input.
5. **S1.7.3 and S1.7.4** list the claims in vocabulary order.
6. **S0.4 to S0.6** use real IDs: ENSG00000299624, GOLGA2P9, and LINC02170.
   S0.7 still uses a placeholder. Fill it with a retired ID from the pinned
   stable-ID history.
7. **Envelope.**
   - Every tool result and every answer has `provenance`, with an
     `evidence_id` for each call.
   - The registry blocks are JSON.
   - Every `defer` answer has a `blocked_reason` sentence, and every other
     answer has `null`.
8. **Tag form.** S1.6.1, S1.6.4, and S1.7.7 use
   `bulkPEN:bulk_pen:brain:amygdala:v1`. S1.6.4 now tests a filter with no
   match.
9. **New traces:** S1.6.7 (TFs parse) and S1.7.9 (`tf_target` claims).
10. **Formatting:** the S1.3.9 earlier-turn label and answer had stray
    backticks.
11. **Guidance bullets:**
    - The S1.1 "Absent-symbol pool" bullet is filled.
    - `invalid_id` is now `invalid_input`.
    - The S1.4 bullets are filled.
    - The S1.8 question marks are removed.

    This document still replaces the guidance bullets where they differ.

Still open for the advisor: the provenance on membership results (S1.3.5 to
S1.3.7).

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
| `provenance` key | The traces and v3 leave the key out when there is no registry evidence, and the error results have none | Every tool result and every answer has the key. It is `null` when there is no registry evidence. | One fixed key set. The model decides the value, and the validator checks one shape. |
| `evidence_id` | Optional, "when the tool returns one"; no trace has one; the registry block is a prose header | Required and new for each call. The registry block is JSON with a `provenance` object. | A constant block can be memorized. A changing ID must be copied. |
| `blocked_reason` | A code from 9 values on every `defer` answer | A sentence from a fixed template on every `defer` answer, and `null` otherwise. The code stays in the backend. | The user reads why the analysis stopped. The code still drives validation. |
| Layer tag form | bulkPEN in 3 fields (`bulkPEN:GTEx_v9-brain-hippocampus:v1`), and TF tags in the same shape | bulkPEN and TFs frozen to the 5-field lattice form (`bulkPEN:bulk_pen:brain:amygdala:v1`, `TFs:tf_target:brain:all:v1`) | These are the tags that the repo's lattice flist uses |
| TF sign | Up to two `regulation_direction` traces | No signed TF layer for now. `regulation_direction` is unsupported for every `tf_target` layer. | The loaded TF layer has no sign and no direction |

## Repo data findings (2026-09-30)

**The TF layer.**

- **Tags.** One TF network is in the repo:
  - `TFs:brain` in `data/full_brain_flist.tsv` (line 8)
  - `TFs:tf_target:brain:all:v1` in the lattice flist (line 8)
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

**Stale layer paths.**

- In both repo flists, 357 of 358 layer paths no longer exist, including
  every bulkPEN and scPEN layer. Only the TF file resolves.
- A multiplex README entry of 2026-08-28 says that those networks moved to
  `_archive` because they "contain incorrectly created PEN networks."
- Effects on S1:
  - Take the bulkPEN and scPEN tags from the corrected PEN build, not from
    the current flists. bulkPEN tags keep the frozen five-field form.
  - The gene universe of a multiplex is the union of its PEN gene sets. Check
    which PEN build the `full_brain_multiplex_v1` store used. If it used the
    archived networks, rebuild the store before you take S1.3 membership
    results from it.

## Open questions

**From the advisor:**

1. **The real TF network prefix.** Answered: `TFs`. The lattice flist tag
   is `TFs:tf_target:brain:all:v1`. It matches the lattice form of bulkPEN,
   not the v3 three-field form. Both are now frozen in that form (see S1.6).
2. **Whether the TF sources record the sign of regulatory edges.** Answered:
   - The brain source (Pearl 2019) records the sign.
   - The CD4+ T-cell source records no sign.
   - The brain layer that the repo loads has lost both the sign and the
     direction.

   Decided: no signed TF layer for now (see "Repo data findings").
3. **Whether the registry tool returns a provenance block with `registry`
   and `release`.** Not yet:
   - `_provenance()` returns `ensembl_release: "Ensembl_116"`, with no
     `registry` key and no `evidence_id`.
   - That object sits outside the payload, and the S0 builder shows the
     payload only.
   - See "Runtime changes required."

**For the advisor to decide:**

4. The provenance on a membership result: the registry block, as in S1.3.5
   to S1.3.7, or the v3 graph block (`multiplex_id`, `graph_version`,
   `evidence_id`).
5. An unknown `multiplex_id` in the membership tool: `not_found`, or `error`
   with `UNKNOWN_CONTEXT`.
6. The `<MULTIPLEX:id>` anchor: use it for every multiplex mention, or drop it.
7. The failure keys: keep the two cases (`tool` when the answer rests on
   failed calls alone, `failed_tool` for a partial failure), or use one key.
8. An ambiguous metric name: keep `unrecognized_metric`, or add
   `ambiguous_metric`.
9. S1.7.8: keep `stop` with no request in the summary (applied), or change
   to `defer` with a new reason (for example `unrecognized_layer`).
10. The policy after a transport error: defer at once (the current traces),
    or retry once. Stage 2 inherits this choice.
11. A symbol that resolves cleanly while the task names a multiplex. Does the
    model also check membership? v3 says that a resolved ID does not by
    itself establish presence in a multiplex. If yes, the S1.1 rows that name
    a multiplex change.
12. Applied: a per-call `evidence_id` (see "`provenance`").
13. Applied: the registry block is JSON with a `provenance` object (see
    "Harness context blocks").
14. The decisions in "Decisions that differ from the advisor's sources."
15. Decided: bulkPEN and TFs are frozen to the five-field lattice form.
16. Decided for now: no signed TF layer. If one is added later, the
    recommended path stays:
    - Keep the RWR layer undirected, with positive weights.
    - Build a directed, signed edge table from the raw Pearl 2019 files.
    - Serve it through an S2 edge-record tool (S2.1, S2.13).
17. Whether a TF edge that comes from regression on brain expression
    (Pearl 2019) also supports `context_predictive_expression` for the brain,
    or only `regulatory_link_recorded`.
18. Later: whether to add the CD4+ T-cell TF layer (directed, unsigned) to a
    multiplex. Its tag must pass the frozen grammar, including the `+` in
    `cd4+t_cells`.

## Planned traces

These are the advisor's next items, plus the gaps that v3 requires:

- **`no_multiplex_supplied`:** a membership question with no multiplex. The
  model makes no membership call and defers.
- **Batch resolution with mixed outcomes:** for example VARS1 (resolved),
  GOLGA2P9 (ambiguous), and LINC02170 (not found). First decide the result
  shape for each item and the rule for the top-level status.
- **A clean resolution to a gene that is not in the supplied multiplex:**
  - `not_found`, `no_multiplex_membership`
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
- **S1.6 and S1.7:** a family part that does not match the prefix (for
  example `bulkPEN:tf_target:brain:amygdala:v1`).
- **S0.7:** a real retired ID in place of the placeholder.

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

- The advisor's S1 question classes file (64 traces), and the advisor's email
  of 2026-09-30.
- `MENTOR-RL-World-Model_v3.md`
- `agents/world_model_v2_curriculum/S0-human-gene-identifiers.md`
- Runtime and scripts:
  - `runtime/world_model_s0_tools.py`
  - `runtime/world_model_prompts.py`
  - `scripts/grab_human_gene_ids.py`
  - `scripts/build_multiplex_store.py`
  - `scripts/token_sequence_estimator.py`
- Layer lists: `data/full_brain_flist.tsv`,
  `data/world_model_v2/lattice/ws0_brain_v1/flists/full_brain_multiplex_v1.tsv`,
  and the master list `mentor-ev_multiplexes/data/h_sap_all_layer_flist.tsv`
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
