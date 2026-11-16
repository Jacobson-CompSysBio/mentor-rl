# AGENTS.md — MENTOR-RL

Context for coding agents working in this repo. Read this file first. Keep it short; put task detail in the files it points to.

## What is current

- The active work is the **world-model curriculum** (stages S0–S9): systematically generated tool-use traces for supervised training of the 120B model on Frontier.
- S0 and S1 have passed their gates. **S2 v1** scored 80.58% structured accuracy (20,320 / 25,216 test rows). The proposed S2 gate is **93%**.
- The current task is **implementing the complete S2 v2 corpus revision**; v2 has not been built or queued. **Step 1 is complete:** the v1 baseline and release rules are frozen in `data/world_model_v4/diagnostics/s2_v2_baseline/`. The next step is the five-case convention audit. The accepted plan is `docs/s2_v2_changes.md`; aggregate per-type v1 results are in `docs/s2_v1_type_results.csv`.
- The revision scope is the **208 types below 95%** (`failing=true` in the CSV), following the existing v1 failure analysis. The proposed 93% S2 gate is separate; 192 types fall below it. v1 ran in report-only mode and has no pass/fail gate result.
- The catalog has 399 designs; v1 retained 394. Preserve the five exclusions in `runtime/world_model/s2_production_scope_v1.json` for the initial revision.
- The versioned release policy is `config/world_model_s2_release_v2.json`: 512–1,024 training rows per editable type, with contrastive examples inside that cap; unchanged passing rows; 80/15/5 S2/S1/S0 mix; cached local 120B rewrites; one combined training run. Preserve existing replay rows and extend only without replacement. The pinned replay pools also constrain expansion.
- The current shared memory is `shared_memory/Memo - S2 v1 Training Update.md`. Its statement that v2 is queued is forward-looking; see the plan for reconciled counts and status. `README.md` describes older CORUM and dendrogram work; `shared_memory/current_status.md` has been removed.

## Map of the S2 code

| What | Where |
|---|---|
| Question catalog (approved designs, worked traces) | `agents/world_model_v3_curriculum/S2-question-classes.md` |
| Baseline freezer and v2 release policy | `scripts/freeze_world_model_s2_baseline.py`, `config/world_model_s2_release_v2.json`; receipts: `data/world_model_v4/diagnostics/s2_v2_baseline/` |
| Catalog parser and default row quotas (512 train / 64 validation / 64 test per type) | `runtime/world_model/s2_production_cases.py` |
| Per-family generators | `runtime/world_model/s2_<N>_corpus.py`, `_prompts.py`, `_evidence.py`, `_export.py`, `_sampling.py`, `_case_repairs.py` (S2.1: `s2_1_production.py`, `s2_corpus.py`) |
| Shared graph indices | `runtime/world_model/s2_sources.py`, `s2_neighbors.py`, `s2_degrees.py`, `s2_paths.py`, `s2_subgraphs.py`, `s2_components.py`, `s2_multiplex.py` |
| Prompt banks and wording overlays | `runtime/world_model/s2_v1_prompt_banks.json`, `s2_production_wording*.py` |
| Quota allocation and audits | `runtime/world_model/s2_audit_*.py`, `scripts/audit_world_model_s2_*.py` |
| Corpus build, assembly, verification | `scripts/build_world_model_s2_production.py` (and the `_v2` overlay), `scripts/assemble_world_model_s2_production.py`, `scripts/verify_world_model_s2_production.py`. The existing `_v2` overlay helped build **v1**; it is not the planned v2 corpus revision. |
| Evaluator panel and scoring | `scripts/build_world_model_s2_eval_bundle.py`, `scripts/evaluate_world_model_s2.py`, `scripts/validate_world_model_s2_eval_contract.py` |
| Training contract and launch | `scripts/validate_world_model_s2_train_contract.py`, `config/world_model_s2_120b_*.json`, `world_model_s2*.slurm` |
| Tests | `tests/test_world_model_s2_*.py` |
| Curriculum dashboard and viewer | `dashboard/`, `config/dashboard/stages.yaml` |
| Data (git-ignored) | `data/world_model_v4/corpora/`, `data/world_model_v4/eval/`, `data/world_model_v4/checkpoints/` |

**The catalog is parsed by code.** `s2_production_cases.py` reads every `### S2.N.M: title` heading and the `Final answer:` JSON block under it. A change to a heading or a final answer changes the corpus.

**Do not read the catalog whole.** It is about 93,000 lines. Find a case with `rg -n '^### S2\.8\.16:' agents/world_model_v3_curriculum/S2-question-classes.md` and read only that section.

## Commands

```bash
# Login-node CPU checks (Python 3.12; run from this repo root)
source /lustre/orion/syb111/world-shared/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
# For a single command without activation, use this environment's bin/python.

# Tests for one family
python -m pytest -q tests/test_world_model_s2_8_prompts.py
python -m pytest -q -k "world_model_s2_8"

# All S2 tests
python -m pytest -q -k "world_model_s2"

# Freeze/recheck the v1 baseline (CPU only; immutable, deterministic receipts)
python scripts/freeze_world_model_s2_baseline.py
python -m pytest -q tests/test_world_model_s2_baseline.py

# Generator replay check and corpus verification (see --help for arguments)
python scripts/check_world_model_s2_generators.py --help
python scripts/verify_world_model_s2_production.py --help

```

`pytest.ini` puts `.` and `scripts` on the Python path, so run everything from the repo root.

**The baseline freezer does not certify revised traces.** v2 trace validators are not implemented yet. Their required checks and implementation order are in `docs/s2_v2_changes.md`. Add executable commands here when they land; existing replay/production checks alone do not certify the new trace conventions. Inspect input paths before running a verifier: the current production verifier reads retention evaluator manifests under the protected evaluator tree.

## Hard rules

1. **Never submit jobs.** No `sbatch`, `srun`, or `scripts/submit_world_model_s2.sh`. Prepare and validate; a human submits.
2. **Never open evaluator answers or the sealed outer fold.** Do not read, print, copy, or write under `data/world_model_v4/eval/`. Use existing aggregate reports, counts, and hashes only; dashboard failure bundles can embed answer keys too. **The outer fold has not been built, and no path is registered.** Its construction and path registration belong to a separate split-design task; do not assume the current inner test is an untouched outer fold.
3. **v1 artifacts are immutable.** Do not edit built corpora, completed shards, `*_v1*.json` files, or pinned run configs. New work goes in new versioned files, following the existing overlay pattern (`build_world_model_s2_production_v2.py`, `s2_production_wording_v2.py`, `*_case_repairs.py`).
4. **Touch only failing types.** Edit a question type only if `docs/s2_v1_type_results.csv` has `failing=true` (the frozen 95% revision bar), or the change list explicitly names it as an exception. Merely listing a suspected gold-convention problem is not authorization to change approved semantics. For every other type, rendered rows must be byte-identical before and after; prove it with hashes.
5. **Do not weaken checks.** Do not edit tests, scorers, gate thresholds, or contract validators to make a change pass. If a check looks wrong, stop and report it.
6. **No invented values.** Every ID, weight, count, rank, and list in a trace or answer must be computed by code from the source view. Never type one in by hand.
7. **Keep builds deterministic.** Fixed seeds, stable ordering, and identical hashes on a rerun.
8. **Do not change approved semantics.** If a catalog answer or convention looks wrong, report it with the case ID. Do not fix it silently.
9. **Stay in scope.** One family per task. Shared helpers live in shared modules and change only on the base branch, not in a family branch.

## Trace conventions for v2

These apply to the reasoning text of every revised type.

- **Restate the request first.** Name the operation, its scope (layer, direction, representation), the query kind, and the answer fields required.
- **Resolve identities next.** Map each symbol to its supplied registry record and carry the registry evidence beside the graph evidence.
- **Inventory the evidence before any call.** List each supplied observation with its ID and whether it is complete, bounded, truncated, or failed. Say what it implies and what the request still needs.
- **Show every derivation.** Before a count, union, intersection, rank, or component, list the members, overlaps, or pages. Then state the value. The builder must check the listed members against the stated value.
- **Give one reason for the decision.** "Stop" ends a standalone request. "Continue" feeds a named later step. "Defer" names the prerequisite or blocker.
- **Every sentence must be true for its row.** Render sentences from the row's actual state. No stock openers such as "The supplied observations are sufficient; no call is needed," "No compatible observation is supplied," or "The requested mode is complete."
- **Cite only evidence that exists.** Every evidence ID in the reasoning or provenance must appear in the row's context.

## Done means

- The family's tests and the v2 validators pass.
- Unchanged types are byte-identical, shown by hash.
- A short report lists: types changed, rows per type before and after, validator output, and anything skipped with the reason.

## Working style

- Plan before editing shared modules. Family changes can follow the pilot pattern directly.
- Prefer small diffs that follow an existing pattern over new abstractions.
- Ask when a requirement is ambiguous. Do not guess at approved behavior.
