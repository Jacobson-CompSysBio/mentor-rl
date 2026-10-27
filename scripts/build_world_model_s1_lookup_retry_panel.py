#!/usr/bin/env python3
"""Build separate retry diagnostics from frozen symbol or ID lookup test genes.

This panel does not replace the frozen promotion panel. Its questions and
answers use the existing native trajectory runner and structured scorer.
"""

from __future__ import annotations

import argparse
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
from runtime.world_model.s1_lookup_retry import (  # noqa: E402
    DIAGNOSTIC_PREFIX,
    ID_DIAGNOSTIC_PREFIX,
    RETRY_CASES,
    audit_retry_row,
    load_retry_policy,
    render_retry_case,
)
from runtime.world_model.s1_overlay import (  # noqa: E402
    S1_V6_OVERLAY_PATH,
    apply_s1_overlay,
    overlay_pin,
)
from runtime.world_model.sources import file_sha256  # noqa: E402
from runtime.world_model.tools import GENE_ID_TOOL, SYMBOL_TOOL, WorldModelToolRuntime  # noqa: E402
from scripts.build_world_model_s1_bundle import project_row  # noqa: E402
from scripts.world_model_s1_eval_common import (  # noqa: E402
    validate_answer_row, validate_question, write_json,
)


def build_panel(output_directory: Path, *, section: str = "S1.1") -> dict[str, Any]:
    """Publish 50 matched examples per retry case, preserving the legacy panel."""

    if section not in {"S1.1", "S1.2"}:
        raise ValueError("retry diagnostics support only S1.1 and S1.2")
    is_id = section == "S1.2"
    component, form = ("s1_2", 11) if is_id else ("s1_1", 8)
    tool_name = GENE_ID_TOOL if is_id else SYMBOL_TOOL
    prefix = ID_DIAGNOSTIC_PREFIX if is_id else DIAGNOSTIC_PREFIX
    contract, overlay = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    pin = contract["curriculum"]["families"][section]["train_lookup_retries"]
    policy = load_retry_policy(pin, tool_name=tool_name)
    source = REPO_ROOT / contract["source_index"]["output_directory"] / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    baseline = REPO_ROOT / "data/world_model_v4/corpora/s1_v5" / component
    baseline_manifest = json.loads((baseline / "manifest.json").read_text())
    rows_path = baseline / "test.jsonl"
    if file_sha256(rows_path) != baseline_manifest["files"]["test.jsonl"]["sha256"]:
        raise ValueError("frozen lookup held-out rows do not match their pin")
    rows = [
        row for row in map(json.loads, rows_path.read_text().splitlines())
        if row["metadata"]["form"] == form
    ]
    if len(rows) != 50:
        raise ValueError("the diagnostic requires 50 frozen lookup test genes")
    runtime = WorldModelToolRuntime.from_source_index(
        source, contract=contract, evidence_id_factory=EvidenceSequence(),
    )
    questions, answers = [], []
    for original in rows:
        for case in RETRY_CASES:
            row = deepcopy(original)
            row["row_id"] = (
                prefix + case + "_" + original["row_id"].rsplit("_", 1)[-1]
            )
            row, _ = render_retry_case(row, case=case, runtime=runtime, policy=policy)
            audit_retry_row(row, case)
            question, answer = project_row(
                row["row_id"], row["messages"], row["tools"], row["metadata"], "test",
            )
            question = validate_question(question, contract)
            answer = validate_answer_row(answer, question, contract)
            questions.append(question)
            answers.append(answer)
    output_directory.mkdir(parents=True, exist_ok=True)
    # Preserve the native answer-field order required by the trajectory parser.
    for name, records in (
        ("test_questions.jsonl", questions), ("test_answer_key.jsonl", answers),
    ):
        (output_directory / name).write_text("".join(
            json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in records
        ))
    manifest = {
        "schema_version": "mentor-rl-s1-lookup-retry-diagnostic-v1",
        "purpose": "Separate diagnostic; do not combine with the frozen promotion score.",
        "row_count": len(questions), "case_counts": {case: 50 for case in RETRY_CASES},
        "max_retries": 1,
        "base_test_rows": {
            "path": str(rows_path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(rows_path),
        },
        "base_manifest_sha256": file_sha256(baseline / "manifest.json"),
        "s1_overlay": overlay_pin(S1_V6_OVERLAY_PATH, overlay),
        "policy": dict(pin), "source_index_sha256": runtime.source_index_sha256,
        "implementation_sha256": {
            str(path): file_sha256(REPO_ROOT / path) for path in (
                "runtime/world_model/s1_lookup_retry.py",
                "scripts/build_world_model_s1_lookup_retry_panel.py",
                "scripts/world_model_s1_eval_common.py",
                "scripts/run_world_model_s1_exact_generation.py",
                "scripts/evaluate_world_model_s1.py",
            )
        },
        "files": {
            name: {
                "sha256": file_sha256(output_directory / name),
                "row_count": len(questions),
            }
            for name in ("test_questions.jsonl", "test_answer_key.jsonl")
        },
        "test_panel_id": file_sha256(output_directory / "test_questions.jsonl"),
    }
    write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    """Build the diagnostic without submitting training or evaluation jobs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("S1.1", "S1.2"), default="S1.1")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    directory = "id_lookup_retries" if args.section == "S1.2" else "lookup_retries"
    output = args.output_directory or REPO_ROOT / "data/world_model_v4/eval/s1_v6" / directory
    manifest = build_panel(output, section=args.section)
    print(json.dumps({"row_count": manifest["row_count"], "test_panel_id": manifest["test_panel_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
