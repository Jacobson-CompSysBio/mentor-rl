# MENTOR-RL S1 contract and validator

Added on 2026-10-01 from snapshot 94fa4d5. The question traces exist in ../S1-question-classes.md. Local changes are listed in this README.

This is a copy, not a live checkout. It has no history and it will go stale the
moment the contract changes, so check the commit above against Dan's before
relying on it for anything that matters.

## What is here

    contract/vocabularies.yaml    status, continuation_decision,
                                  blocked_reason_code, layer_family,
                                  layer_claims
    contract/blocked_reasons.yaml the blocked_reason sentence templates, with
                                  status fit, slot sources and prose labels
    contract/layer_grammars.yaml  the four per-family tag grammars, with
                                  prefixes, arities and field order
    contract/tools.yaml           tool names, arguments, result shapes
    contract/provenance.yaml      which answers carry provenance, which fields
    contract/rows.yaml            the form of one row of a generated corpus
    validate.py                   checks a question file (.md) or a generated
                                  corpus (.jsonl) against the contract
    fixtures/, selftest.sh        checks the validator against known-bad and
                                  known-good traces and rows
    mutate.sh                     breaks the validator to prove the fixtures bite

The YAML files are the normative version of what the spec says in prose.
Generate against them rather than against the prose, because prose drifts and
these do not.

## Running it

    python3 -m pip install pyyaml      # the same python3 you will run with
    ./selftest.sh                      # run this FIRST, see below
    python3 validate.py ../S1-question-classes.md
    python3 validate.py your_new_questions.md
    python3 validate.py <corpus>/train.jsonl    # rows in the form of rows.yaml

Run `selftest.sh` before trusting a clean result. The validator can fail in a
way that looks like success: an early version parsed zero traces out of a file
and reported zero violations. It now refuses to report a pass when it cannot do
its job, and `selftest.sh` confirms that by running it against 53 deliberately
broken traces and rows and 24 clean ones, checking that the right check
catches each. A row of a .jsonl file that is not JSON counts as skipped, so
the run fails.
`mutate.sh` goes further and breaks the validator itself one site at a time to
prove the fixtures would notice. Both found real holes this week.

Exit codes: 0 clean, 1 violations found, 2 or more could not run.

## Local changes

Changes from snapshot 94fa4d5, made in this repo:

- `S1_Question_Classes.md` is not copied. The question file is
  `../S1-question-classes.md`.
- `vocabularies.yaml`: `blocked_reason` is renamed `blocked_reason_code`. The
  code stays in the backend. The answer carries a sentence.
- `blocked_reasons.yaml` is new. It holds the sentence templates of the
  S1 spec, section "`blocked_reason`".
- `provenance.yaml`, `tools.yaml`, `layer_grammars.yaml`, and
  `vocabularies.yaml` follow the S1 spec of 2026-10-01: a required
  `evidence_id`, a `provenance` key on every result and answer, one return
  shape per status, and the frozen 5-field bulkPEN and TFs and 6-field scPEN
  tag forms.
- `validate.py`:
  - checks the `blocked_reason` sentence against its templates (it must
    refill exactly and fit the status), and `null` on every other answer
  - checks that an answer copies the block of the evidence it rests on, in
    this order: this turn's first successful lookup, an earlier turn, the
    harness registry block. An answer with no registry evidence has `null`.
  - checks calls against `s1_callable`, and parses `TFs` tags, family parts,
    and versions
  - parses each multi-line tool call as one call, so that the n-th call
    pairs with the n-th result
  - writes each `flag(...)` on one line. `mutate.sh` replaces only the first
    line of a call, so a multi-line call made a syntax error, and its mutant
    died without a fixture.
  - adds the checks of the S1 spec, section "Validation":
    - no duplicate prompt. The key is the harness context, the earlier
      turn, the user text, and the outcome of each call, without
      evidence_ids, so the S1.1.1 and S1.1.8 minimal pair passes.
    - a Thinking step after each tool result
    - the tool result shape of `tools.yaml`
    - the `provenance` key on the answer, each tool result, and the harness
      block
    - the block that each producer writes, by `evidence_id` prefix
    - each `evidence_id` in one row only
    - anchor types from `anchor_type`, and each anchor in the result
    - claim lists in vocabulary order
    - a summary asks for input on a `defer` answer and only then, by the
      phrases in `input_request_phrases`
    - the failure keys: `tool` when every call failed, `failed_tool` after a
      successful lookup
- `vocabularies.yaml` adds `input_request_phrases`.
- `rows.yaml` is new. It is the form of one row of a generated corpus, and it
  lists the hidden metadata fields and the ones that must never be visible.
- `validate.py` reads a .jsonl corpus as well as a question file. For a row it
  also checks:
  - the row form of `rows.yaml`
  - the `blocked_reason_code` of the row: only its templates may match, and a
    row that does not defer has none
  - the label slots, filled exactly from `found_input_type` and
    `input_species`
  - one system prompt and one tool list for every row, and a
    `system_prompt_sha256` that hashes the prompt
  - no hidden field name or value in the system prompt, a harness block, or
    a user message
- `selftest.sh` also runs `fixtures/must_fail.jsonl` (the J fixtures, after a
  clean reference row R1) and `fixtures/must_pass.jsonl`, and it reports a
  declared check that has no fixture.
- `fixtures/`: the 21 original must-fail traces and 10 must-pass traces are
  rewritten in the new envelope. F22 to F45, D34, P11 to P17, and the JSONL
  fixtures (R1, J1 to J8, Q1 to Q7) are new. D34 and R1 are not declared
  fixtures: D34 is the trace that F34 repeats, and R1 is the reference row
  for the prompt check.

## If something in the contract looks wrong

Say so. These files are one reading of the question corpus, not gospel, and the
validator will enforce whatever they say whether or not it is right. Sixty lines
of YAML is a reviewable amount in a way that the full spec is not, so this is
the cheapest place to catch a misreading.
