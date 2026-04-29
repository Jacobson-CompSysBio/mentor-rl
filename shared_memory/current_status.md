# Mentor-RL Current Status

## Snapshot (2026-04-24)

The repo now implements most of the environment and generation machinery described in `shared_memory/methods_proposal.tex`: CORUM task construction, structured runtime state, deterministic scoring, a Python/C++ graph runtime, shared-prefix trajectory generation, and SFT/GRPO launchers.

The immediate blocker is no longer corpus construction, runtime wiring, model staging, import paths, or request timeout plumbing. The actor contract has been relaxed: actor reasoning is free-form ReAct-style text, while only runtime tool calls and verifier/state updates remain structured. The remaining blocker is getting a served model on Frontier to produce usable actor text/tool calls and schema-valid verifier JSON during the preflight smoke step.

## Progress

- CORUM corpus is built in `data/corum_corpus/`.
  - `4,914` retained deduplicated complexes.
  - `214,368` canonical tasks total.
  - Split counts: `171,408` train, `21,480` val, `21,480` test.
  - Task types: `19,656` explanation, `28,560` recovery, `58,968` refinement, `107,184` none.
- HumanNet multiplex binary store is built in `data/humannet_multiplex_store/`.
  - Format: `mentor-rl-multiplex-store-v1`.
  - `44,491` genes, `384` layers, `24,211,378` aggregate undirected edges.
  - Store size is about `1.8G`.
- Runtime stack is in place.
  - `runtime/schemas.py`, `state.py`, `validators.py`, and `scoring.py` cover structured state, validation, local branch scoring, terminal scoring, trajectory turns, and preference pairs.
  - `runtime/environment.py` can use either the Python reference graph tools or the compiled C++ backend.
  - C++ runtime and store-builder tests exist for the core graph operations.
- Trajectory generation is implemented in `scripts/generate_trajectories.py`.
  - Supports heuristic and model-backed candidate generation.
  - Writes branch pools, trajectory turns, findings, raw/balanced preference pairs, final summaries, manifest, and progress files when generation succeeds.
  - Actor generation now accepts free-form reasoning text, parses native runtime tool calls when present, and keeps strict JSON on the verifier/state update side.
  - Includes gpt-oss-specific handling for completions/chat modes, hidden-thinking suppression, fallback paths, and request timeouts.
- Frontier launcher work is mostly done in `generate_trajectories.slurm`.
  - Resolves model directories, stages models to NVMe, bootstraps tiktoken files for gpt-oss, starts vLLM/Ray, waits on health, records `/v1/models`, exports `PYTHONPATH`, runs a preflight smoke test, and passes `--generator-timeout-seconds`.
- Training entry points exist for SFT and GRPO.
  - `scripts/train_sft.py` / `train_sft.slurm`.
  - `scripts/train_grpo.py` / `train_grpo.slurm`.
  - The proposal's DPO stage is partially represented by generated preference-pair artifacts, but a dedicated DPO training script is not currently the blocking item.

## Frontier Run State

- Old infrastructure failures have been addressed or bypassed:
  - Ray/vLLM GPU visibility mismatch (`ROCR_VISIBLE_DEVICES` vs `HIP_VISIBLE_DEVICES`).
  - `ModuleNotFoundError: runtime` in trajectory jobs.
  - Too-short model request timeout; timeout is now configurable from the Slurm launcher.
- No successful model-backed trajectory corpus exists yet.
  - `data/corum_trajectories/gpt_oss_120b_job_4440835/` has empty `branch_pools.jsonl`, `trajectory_turns.jsonl`, and `final_summaries.jsonl`.
  - Later trajectory directories mostly contain only `served_models.json` because smoke tests failed before full generation.
- Notable recent results:
  - `4440835` served `gpt-oss-120b-bf16` and launched generation, but failed at step 0 with no usable actor candidates.
  - `4442804` passed a smoke-only run with `Llama-3.2-1B-Instruct`; it did not generate trajectory artifacts.
  - Later Llama/gpt-oss smoke attempts failed with malformed or missing actor outputs.
  - `4451591` and `4452380` served `gpt-oss-20b-bf16`, but smoke failed with blank/truncated/malformed outputs rather than usable JSON/tool calls.

## Roadblocks

- Served gpt-oss models are not reliably producing usable actor/verifier outputs under the current vLLM setup.
  - Observed actor failures include empty `tool_calls`, blank visible output, malformed repeated text, and missing usable action text.
  - Verifier output still must be schema-valid JSON.
- The first smoke diagnostics are noisy, which makes malformed completions harder to diagnose.
  - Keep prompt and diagnostic payloads compact while debugging served-model behavior.
- There are still zero non-empty model-backed branch pools, turns, summaries, or preference pairs.
  - Do not start DPO extraction/training assumptions until at least one end-to-end generation run writes non-empty artifacts.
- Frontier serving is usable but fragile.
  - ROCm flash-attn import warnings/fallbacks remain noisy.
  - One run failed during model init from the flash-attn import path; later runs loaded by falling back to SDPA/naive attention.
- The methods proposal still describes future pipeline stages that are not all implemented as training jobs.
  - In particular, DPO is conceptually designed and preference-pair generation exists, but the immediate work is still trajectory production.

## Next Steps

1. Add or run a minimal served-model sanity check before the biology smoke test.
   - Prompt for one tiny free-form actor step, with and without a native runtime tool call.
   - Separately prompt for one tiny verifier JSON object with no CORUM task, no graph metadata, and no long schema text.
   - Compare `chat_completions`, `completions`, and `responses` where supported.
   - Persist the exact request payload, rendered prompt, raw response, parsed output, and errors.
2. Reduce the real smoke prompt.
   - Prefer compact task context plus tool names and argument schemas.
   - Avoid printing full runtime layer inventories unless explicitly requested.
   - Keep `MAX_TASKS=1`, `N_ACT=1`, `N_VER=1`, low temperature, and small smoke token limits while debugging.
3. Use a small known instruction-following model to validate the writer path if gpt-oss continues to fail.
   - Goal is one non-empty run with branch pools, turns, findings, summaries, and preference pairs, even if the model quality is poor.
4. After one non-empty run exists, inspect artifact counts and terminal scores.
   - Then scale to a small multi-task run.
   - Only then decide whether to prioritize a DPO training entry point or move directly into the existing GRPO path.
5. Later documentation cleanup:
   - Update `methods_proposal.tex` to distinguish implemented components from planned training stages.
   - Fix drafting artifacts in the proposal before treating it as submission-ready.

## Assumptions To Avoid

- Do not assume real model-backed trajectory artifacts exist yet.
- Do not assume vLLM health plus `/v1/models` success means generation is usable.
- Do not treat the project as DPO-only; a GRPO path is already implemented.
- Do not spend time on pair mining or DPO training until the first non-empty model-backed trajectory run lands.
