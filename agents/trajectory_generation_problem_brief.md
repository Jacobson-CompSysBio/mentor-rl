# Trajectory Generation Problem Brief

Date: 2026-06-24

This note describes the current bottleneck in MENTOR-RL trajectory generation so
the problem can be handed to another engineer or researcher for solution
design. It is intentionally framed as a problem brief, not as a single proposed
patch.

## Executive Summary

MENTOR-RL needs high-quality preference pairs and trajectories for DPO over
mechanistic exploration tasks. The current infrastructure can run the
structured RWR++ tools, score candidate branches, audit artifacts, and mine DPO
pairs. The blocker is data quality: recovery and refinement tasks rarely
produce exact-positive branches.

The model often finds evidence that points in the right direction, but it does
not reliably convert that evidence into the exact gene set required by the
hidden module target. It produces partial near-misses, plausible explanations,
or over/under-inclusive gene sets. Those are useful as rejected examples, but
they are not valid chosen positives for exact recovery/refinement DPO.

The goal is to design a trajectory-generation system that can scale to
large-volume training data while preserving the central constraint: hidden
targets are for scoring and validation only. They must not be exposed in model
prompts, branch text, tool observations, or exported training records.

## Current System Shape

MENTOR-RL generates shared-prefix branch pools:

1. The actor proposes candidate reasoning steps and optional tool calls.
2. Deterministic tools execute those calls.
3. The verifier updates the interpretation, structured state, and continuation
   decision.
4. The local scorer evaluates each branch using hidden targets.
5. One branch is selected as the trajectory continuation.
6. DPO pairs are mined from chosen vs rejected branches sharing the same prefix.

The important task types are:

- `recovery`: input is a partial gene group; the agent should add missing
  members.
- `refinement`: input is a noisy gene group; the agent should remove unrelated
  members.
- `explanation`: input is already coherent; the agent should explain it.
- `none`: input lacks a supported shared mechanism; the agent should abstain or
  split groups.

The immediate quality bottleneck is recovery/refinement, because those tasks
require exact set editing rather than only mechanism explanation.

## Evidence for the Bottleneck

The recent exact-performance pilot showed that runtime plumbing is not the main
issue. Smoke-cache precompute, RWR-HPC/native-store access, annotation plumbing,
and vLLM serving were functional.

The useful diagnostic result was quality-related:

- The run completed 18 of 24 tasks before Slurm timeout.
- Overall strict-positive success appeared on some task types.
- Exact recovery was 0/4 in the completed portion.
- Exact refinement was 0/4 in the completed portion.
- Recovery/refinement mostly ended as partial near-misses.
- Preference pairs were dominated by categories such as evidence improvement,
  recovery recall, and refinement precision, not exact-membership categories.

This means more walltime alone is unlikely to solve the problem. The system
needs better candidate generation, search, curriculum, or supervision so exact
recovery/refinement branches exist often enough to mine DPO pairs.

## What Is Actually Failing

The model is not simply "bad at biology." It is failing at a specific
combinatorial control problem: deciding which exact gene additions or removals
are warranted by visible evidence under a limited trajectory budget.

### Recovery Failure Cases

- The model keeps only the seed genes and treats them as a complete group.
- It adds one plausible non-seed gene but misses other required members.
- It sees an RWR-ranked candidate frontier but does not systematically test
  enough candidates.
- It overweights mechanism plausibility and underweights exact membership.
- It stops at `partially_observed_group` while there is still budget to test
  additions.
- It gathers evidence with useful signal but fails to turn that signal into an
  updated `predicted_gene_ids` set.

### Refinement Failure Cases

- The model recognizes that the input is noisy but does not remove the right
  extras.
- It removes one unsupported gene while leaving other unsupported genes.
- It keeps distractor genes because they are biologically plausible in text.
- It produces a high-overlap group, but precision, recall, and Jaccard are not
  all exactly 1.0.
- It lets mechanism-label quality compensate for membership errors.
- It lacks a direct tool or policy that marks which genes should be removed,
  so pruning decisions are inconsistent.

### Preference Data Failure Cases

- No exact-positive branch exists in the branch pool, so pair mining cannot
  create exact-vs-partial DPO examples.
- The selected branch is partial, so it cannot be used as a chosen positive for
  exact recovery/refinement.
- Pair categories become dominated by mechanism or evidence improvements rather
  than exact membership corrections.
- Scaling such data would teach near-miss behavior instead of exact recovery.

## Why a Single Edit-Search Patch Is Not Enough

A deterministic edit generator can help by trying visible-evidence-based
add/drop variants that the model did not enumerate. That is useful, but it only
works when the correct genes are already present in the visible candidate
frontier and the scorer can identify the exact branch.

Large-scale DPO training needs more than one edit heuristic. It needs a
reliable pipeline that can:

- Surface the right candidate genes.
- Explore enough membership variants without exploding cost.
- Keep hidden targets out of the generated content.
- Produce diverse exact-positive branches across task sizes, difficulty bins,
  and corpus sources.
- Preserve hard near-misses as rejected examples.
- Audit and export only pairs that are safe for training.

The core design problem is therefore not "add a bigger search." It is
"construct a scalable visible-evidence search and selection process that
generates exact-positive chosen branches often enough for DPO."

## Hard Constraints

Any solution should preserve these constraints:

- Hidden module targets may be used only for scoring, validation, and offline
  analysis.
- Hidden targets must not appear in prompts, tool outputs, branch metadata,
  rendered summaries, preference records, or SFT/DPO exports.
- Partial recovery/refinement cannot be a chosen-positive training target.
- Exact recovery/refinement requires exact gene membership, not high overlap.
- DPO pairs should compare same-prefix continuations whenever possible.
- Runtime must remain auditable: every chosen branch should have provenance
  linking it to visible evidence and deterministic scoring.
- The solution should support held-out evaluation without leakage across
  module splits.

## Solution Directions to Brainstorm

The outsourced task should consider multiple solution families, not just one
patch.

### 1. Better Candidate Frontier Construction

Improve the set of genes that the search process can consider.

Possible approaches:

- RWR frontier aggregation over multiple seed subsets.
- Leave-one-out RWR for refinement-specific support scoring.
- Layer-aware candidate ranking, not only full-multiplex ranking.
- Combining RWR ranks with induced-subgraph degree, shortest-path support,
  annotation coherence, and perturbation/ablation signals.
- Separate frontier strategies for MENTOR dendrogram modules vs RWR-LOE modules.

Key question: when exact recovery fails, was the missing gene present in any
visible tool output? If not, search depth is not the primary bottleneck.

### 2. Structured Membership Search

Replace ad hoc branch sampling with explicit search over gene-set states.

Possible approaches:

- Beam search over add/drop actions.
- Monte Carlo tree search over membership states.
- A* or best-first search using visible evidence scores as heuristics.
- Bounded combinatorial search over candidate frontiers.
- A two-stage process: retrieve candidate frontier first, then perform
  membership optimization over that frontier.

Key risk: search can become an oracle-like shortcut if branch text stops
reflecting visible evidence. Every generated state must remain evidence-backed.

### 3. Curriculum and Task Shaping

Large-scale training may need easier exact tasks before hard exact tasks.

Possible approaches:

- Start with one-edit recovery/refinement tasks where the needed add/drop is in
  the top visible frontier.
- Increase edit distance gradually.
- Increase distractor similarity gradually.
- Separate module-size curricula from task-difficulty curricula.
- Oversample cases where visible tools can recover exact membership.
- Track performance separately for GW dendrogram vs RWR-LOE corpora.

Key question: should early DPO teach exact one-step membership edits before
multi-step exploration?

### 4. Process Supervision Instead of Only Terminal Pairs

Exact terminal membership may be too sparse for early DPO.

Possible approaches:

- Add process labels for productive intermediate moves, such as "tests a
  plausible missing member" or "removes a low-support distractor."
- Mine DPO pairs where both branches are nonterminal, but one creates a better
  search state.
- Reward evidence-gathering actions that make exact recovery possible later.
- Separate exploration quality from final membership quality in pair categories.

Key risk: process labels must not reward plausible but misleading exploration
that never improves exact recovery.

### 5. Model Policy Improvements

The model may need training or prompting changes that make it better at using
the available tools.

Possible approaches:

- SFT on schema-valid tool use and evidence-to-state updates, but only after
  exact-positive examples exist.
- Tool-use curricula focused on RWR, induced subgraph, and pruning decisions.
- Actor/verifier specialization, possibly separate adapters or mode-specific
  losses.
- Verifier training that explicitly rejects over/under-inclusive gene sets.
- Self-critique prompts that require the verifier to list untested candidate
  additions/removals before stopping.

Key risk: SFT on current near-miss trajectories would reinforce the failure
mode rather than fix it.

### 6. Offline Teacher or Oracle Use, Quarantined

An oracle can be useful for diagnostics, but it is dangerous for training data.

Acceptable uses:

- Measure whether hidden target genes appear in visible tool frontiers.
- Estimate upper bounds for recovery/refinement if search were perfect.
- Label why a trajectory failed after the fact.
- Build non-exported debug reports for algorithm design.

Unsafe uses:

- Creating chosen branch text from hidden target genes.
- Injecting hidden target genes into candidate frontiers.
- Exporting oracle-generated branches as DPO chosen examples.

If an oracle teacher is used, its artifacts should be quarantined and excluded
from DPO/SFT exports unless explicitly converted into a leakage-safe form.

### 7. Data Quality Gates and Export Policy

Large-scale generation should be blocked unless the generated data is actually
useful for DPO.

Suggested gates:

- At least one exact-positive recovery and one exact-positive refinement in
  small pilots before scaling.
- Nonzero exact-over-partial pair counts.
- Exact-membership pair rate above a minimum threshold.
- Low RWR-HPC observation error rate.
- Low selected no-tool rate for recovery/refinement.
- Low weak-evidence `validated_group` rate.
- Pair category distribution not dominated by mechanism-only improvements.

Export policy:

- Exact recovery/refinement positives can be exported for DPO.
- Partial recovery/refinement can be exported only as rejected examples.
- Exact-positive SFT should wait until there are enough evidence-grounded
  exact trajectories.
- Incomplete or reconstructed diagnostic runs should not be training sources.

## Questions for the Outsourced Solver

The outsourced task should answer these questions with evidence:

1. For failed recovery cases, how often are missing target genes present in the
   visible RWR or graph-tool frontier?
2. For failed refinement cases, how often are true distractors identifiable by
   visible low-support signals?
3. Is the bottleneck frontier recall, search over the frontier, verifier state
   updating, branch selection, or pair export?
4. What search method gives the best exact-positive yield per tool-call dollar?
5. What curriculum produces exact-positive pairs without making the task
   distribution too artificial?
6. How can the system generate many exact-over-partial pairs while preserving
   same-prefix DPO structure?
7. What metrics should decide whether to scale from pilot to production?

## Desired Deliverables

A good solution proposal should include:

- A diagnosis plan that separates frontier recall, search quality, verifier
  update quality, and scoring/export issues.
- A concrete algorithm for generating exact-positive recovery/refinement branch
  candidates without hidden-target leakage.
- A curriculum for moving from easy one-edit tasks to harder multi-edit tasks.
- An audit plan with thresholds for pilot, medium-scale, and production runs.
- A data-export policy for DPO and later SFT.
- Ablations showing whether improvements come from frontier construction,
  search, verifier behavior, selection, or curriculum.

## Current Working Hypothesis

The model is generating useful but incomplete evidence. The biggest current
failure is not infrastructure, and it is not simply lack of model knowledge. It
is insufficient, poorly controlled exploration over membership edits under a
strict exact-match objective.

The most promising large-scale solution is likely a combination of:

- stronger visible candidate frontier construction,
- explicit bounded search over add/drop membership states,
- curriculum over edit distance and distractor difficulty,
- exact-membership-focused DPO pair mining,
- and strict leakage-safe export gates.

This should be treated as a data-generation systems problem, not just a prompt
engineering problem.
