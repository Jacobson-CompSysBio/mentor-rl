"""Read gate reports and decide whether two runs compare directly.

Gate reports are small files, so render.py reads them on every build: a
version's gate run can be re-pinned in the registry without rescanning its
corpus with diff.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ACCURACY = "structured_answer_accuracy"
# Rates shown in a stage's changelog when a report has them. The free-text
# summary is never scored, so whole-answer exactness is not listed.
HEADLINE = (ACCURACY, "status_exact", "result_exact", "continuation_decision_exact",
            "exact_tool_argument", "structured_json_valid")


def gate_info(path: str | None) -> dict | None:
    """Read a gate report's accuracy and what decides comparability.

    Two runs compare directly only with the same test panel, gate profile,
    history policy, and answer keys. The policy and the answer-key hash come
    from the run's copies of its test config and evaluator manifest; the
    answer keys themselves are never opened. A run whose config records no
    history policy used the default.
    """
    if not path:
        return None
    report = json.loads(Path(path).read_text())
    contract = Path(path).parent.parent / "run_contract"
    config = contract / "test_config.json"
    evaluator = contract / "evaluator_manifest.json"
    history = (json.loads(config.read_text()).get("generation", {}).get("history_policy")
               if config.is_file() else None)
    answer_key = (json.loads(evaluator.read_text()).get("test", {}).get("answer_key_sha256")
                  if evaluator.is_file() else None)

    def pair(metrics):
        if f"{ACCURACY}_count" not in metrics:
            return None
        return [metrics[f"{ACCURACY}_count"], metrics[f"{ACCURACY}_total"]]

    gate = report.get("gate", {})
    overall = report["overall"]
    # Rates per question type (S1 reports) or per question family (S0 reports).
    by_type = "source_traces" in report
    groups = report.get("source_traces") or report.get("families") or {}
    return {
        "report": path,
        "run": Path(path).parts[-5],  # eval_runs/<run>/<method>/<job>/report/test_metrics.json
        "panel": report["test_panel_id"],
        "profile": gate.get("profile"),
        "passed": gate.get("passed"),
        "history_policy": history or "default",
        "answer_key": answer_key,
        "overall": pair(overall),
        "metrics": rates(overall),
        "thresholds": thresholds(gate),
        "types": {trace: pair(metrics) for trace, metrics in report.get("source_traces", {}).items()},
        "groups": {"by": "question type" if by_type else "question family",
                   "rows": {name: {"rates": rates(metrics), "total": total(metrics)}
                            for name, metrics in groups.items()}},
    }


def rates(metrics: dict) -> dict[str, float]:
    """The headline rates a report section has, by metric name."""
    return {name: metrics[f"{name}_rate"] for name in HEADLINE if f"{name}_rate" in metrics}


def total(metrics: dict) -> int | None:
    """How many rows a report section scores."""
    return metrics.get(f"{ACCURACY}_total") or metrics.get("record_count") or metrics.get("row_count")


def thresholds(gate: dict) -> dict[str, float]:
    """The gate's pass mark for each headline rate it sets one for.

    structured_95_v1 states its target per metric under details; the S0 gate
    states minimum_<metric>. Tiered S1 gates set no single structured target.
    """
    found = {}
    for name in HEADLINE:
        detail = gate.get("details", {}).get(name, {})
        if "target" in detail:
            found[name] = detail["target"]
        elif f"minimum_{name}" in gate:
            found[name] = gate[f"minimum_{name}"]
    return found


def comparability(old: dict | None, new: dict | None) -> list[str]:
    """List why two gate results do not compare directly; empty when they do."""
    if not (old and new):
        return ["a gate report is missing"]
    reasons = []
    if old["panel"] != new["panel"]:
        reasons.append("the test panels differ")
    if old["profile"] != new["profile"]:
        reasons.append(f"the gate profiles differ ({old['profile']}, {new['profile']})")
    if old["history_policy"] != new["history_policy"]:
        reasons.append(f"the history policies differ ({old['history_policy']}, {new['history_policy']})")
    if old["answer_key"] != new["answer_key"]:
        reasons.append("the answer keys differ")
    if (old["overall"] is None) != (new["overall"] is None):
        reasons.append("only one report has structured accuracy")
    return reasons
