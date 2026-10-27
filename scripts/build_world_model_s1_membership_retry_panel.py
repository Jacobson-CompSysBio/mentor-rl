#!/usr/bin/env python3
"""Build a separate membership-retry diagnostic from 37 frozen S1.3.10 questions."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.s0_corpus import EvidenceSequence  # noqa: E402
from runtime.world_model.s1_lookup_retry import RETRY_CASES  # noqa: E402
from runtime.world_model.s1_membership_retry import (  # noqa: E402
    DIAGNOSTIC_PREFIX, audit_membership_retry_row, load_membership_retry_policy,
    render_membership_retry_case,
)
from runtime.world_model.s1_overlay import S1_V6_OVERLAY_PATH, apply_s1_overlay, overlay_pin  # noqa: E402
from runtime.world_model.sources import file_sha256  # noqa: E402
from runtime.world_model.tools import WorldModelToolRuntime  # noqa: E402
from scripts.build_world_model_s1_bundle import project_row  # noqa: E402
from scripts.world_model_s1_eval_common import validate_answer_row, validate_question, write_json  # noqa: E402


def _pair_key(row: dict[str, Any]) -> str:
    """Match a failure to its retained success using only visible inputs and calls."""
    return json.dumps([
        [m for m in row["messages"] if m["role"] in ("system", "harness", "user")],
        [m["tool_call"] for m in row["messages"] if "tool_call" in m],
    ], sort_keys=True)


def build_panel(output_directory: Path) -> dict[str, Any]:
    """Publish 37 matched questions per fault schedule without changing promotion."""
    contract, overlay = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    settings = contract["curriculum"]["families"]["S1.3"]
    pin = settings["train_membership_retries"]
    policy = load_membership_retry_policy(pin)
    source = REPO_ROOT / contract["source_index"]["output_directory"] / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz")
    baseline = REPO_ROOT / "data/world_model_v4/corpora/s1_v5/s1_3"
    rows_path = baseline / "test.jsonl"
    base_manifest = json.loads((baseline / "manifest.json").read_text())
    if file_sha256(rows_path) != base_manifest["files"]["test.jsonl"]["sha256"]:
        raise ValueError("frozen membership test rows do not match their pin")
    all_rows = list(map(json.loads, rows_path.read_text().splitlines()))
    rows = [row for row in all_rows if row["metadata"]["form"] == 10]
    successes = {_pair_key(row): row for row in all_rows if row["metadata"]["form"] in (5, 6, 7)}
    if len(rows) != 37:
        raise ValueError("membership diagnostic requires 37 frozen test questions")
    runtime = WorldModelToolRuntime.from_source_index(source, contract=contract, evidence_id_factory=EvidenceSequence())
    questions, answers = [], []
    outcomes = Counter()
    for original in rows:
        success = deepcopy(successes[_pair_key(original)])
        form = success["metadata"]["form"]
        final = success["messages"][-1]
        lookup = next(m["content"] for m in success["messages"] if m["role"] == "tool")
        member = success["messages"][-2]["content"]
        reasoning = settings[{5: "train_single_membership_reasoning", 6: "train_multiple_membership_reasoning", 7: "train_no_membership_reasoning"}[form]]
        phase = ("direct" if final["content"]["continuation_decision"] == "stop" else "step") if form == 5 else "final"
        final["thinking"] = reasoning[phase].format(
            symbol=lookup["gene_symbol"], count=len(lookup["candidate_gene_ids"]),
            multiplex_id=member["multiplex_id"], present_count=len(member["present_gene_ids"]),
            gene_id=member["present_gene_ids"][0] if form == 5 else None,
        )
        outcomes[str(success["metadata"]["form"])] += 1
        for case in RETRY_CASES:
            row = deepcopy(original)
            row["row_id"] = DIAGNOSTIC_PREFIX + case + "_" + original["row_id"].rsplit("_", 1)[-1]
            # Each schedule is a distinct row, so allocate its own registry evidence.
            call = next(m["tool_call"] for m in row["messages"] if "tool_call" in m)
            lookup = runtime.execute(call["name"], call["arguments"])
            next(m for m in row["messages"] if m["role"] == "tool")["content"] = lookup.output
            row["messages"][-1]["content"]["provenance"] = deepcopy(lookup.output["provenance"])
            row, _ = render_membership_retry_case(row, success=success, case=case, runtime=runtime, policy=policy)
            audit_membership_retry_row(row, case)
            question, answer = project_row(row["row_id"], row["messages"], row["tools"], row["metadata"], "test")
            questions.append(validate_question(question, contract))
            answers.append(validate_answer_row(answer, question, contract))
    if dict(outcomes) != {"5": 28, "6": 3, "7": 6}:
        raise ValueError("membership diagnostic recovery outcomes changed")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, records in (("test_questions.jsonl", questions), ("test_answer_key.jsonl", answers)):
        (output_directory / name).write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in records))
    manifest = {
        "schema_version": "mentor-rl-s1-membership-retry-diagnostic-v1",
        "purpose": "Separate diagnostic; do not combine with the frozen promotion score.",
        "row_count": len(questions), "case_counts": {case: 37 for case in RETRY_CASES},
        "recovery_outcomes": dict(outcomes), "max_retries": 1, "maximum_tool_calls": 3,
        "split_unit_count": len({row["metadata"]["split_unit"] for row in questions}),
        "base_test_rows": {"path": str(rows_path.relative_to(REPO_ROOT)), "sha256": file_sha256(rows_path)},
        "base_manifest_sha256": file_sha256(baseline / "manifest.json"),
        "s1_overlay": overlay_pin(S1_V6_OVERLAY_PATH, overlay), "policy": dict(pin),
        "source_index_sha256": runtime.source_index_sha256,
        "implementation_sha256": {path: file_sha256(REPO_ROOT / path) for path in (
            "runtime/world_model/s1_membership_retry.py", "scripts/build_world_model_s1_membership_retry_panel.py",
            "scripts/world_model_s1_eval_common.py", "scripts/run_world_model_s1_exact_generation.py",
            "scripts/evaluate_world_model_s1.py",
        )},
        "files": {name: {"sha256": file_sha256(output_directory / name), "row_count": len(questions)}
                  for name in ("test_questions.jsonl", "test_answer_key.jsonl")},
        "test_panel_id": file_sha256(output_directory / "test_questions.jsonl"),
    }
    write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    """Build the diagnostic without submitting model jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=REPO_ROOT / "data/world_model_v4/eval/s1_v6/membership_retries")
    manifest = build_panel(parser.parse_args().output_directory)
    print(json.dumps({"row_count": manifest["row_count"], "test_panel_id": manifest["test_panel_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
