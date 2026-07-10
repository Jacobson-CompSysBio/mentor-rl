#!/usr/bin/env python3
"""Gate pre-trajectory SFT artifacts before DPO trajectory generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def nested_metric(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def first_present_metric(payload: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = nested_metric(payload, path)
        if value is not None:
            return value
    return None


def add_gate(
    gates: list[dict[str, Any]],
    *,
    name: str,
    observed: float | int | None,
    threshold: float | int,
    op: str,
    required: bool = True,
) -> None:
    if observed is None:
        passed = not required
    elif op == ">=":
        passed = observed >= threshold
    elif op == "<=":
        passed = observed <= threshold
    elif op == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"Unsupported gate op: {op}")
    gates.append(
        {
            "name": name,
            "observed": observed,
            "threshold": threshold,
            "op": op,
            "required": required,
            "passed": passed,
        }
    )


def check_readiness(
    *,
    dataset_audit: dict[str, Any],
    exact_report: dict[str, Any],
    trajectory_audit: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    add_gate(
        gates,
        name="dataset_audit_fatal_errors",
        observed=dataset_audit.get("fatal_error_count"),
        threshold=0,
        op="==",
    )
    if args.fail_on_dataset_warnings:
        add_gate(
            gates,
            name="dataset_audit_warnings",
            observed=dataset_audit.get("warning_count"),
            threshold=0,
            op="==",
        )

    add_gate(
        gates,
        name="overall_exact_pass_rate",
        observed=nested_metric(exact_report, ("summary", "exact_graph_fact_pass_rate")),
        threshold=args.min_overall_exact,
        op=">=",
    )
    add_gate(
        gates,
        name="overall_unsupported_language_rate",
        observed=nested_metric(exact_report, ("summary", "unsupported_language_rate")),
        threshold=args.max_unsupported_language_rate,
        op="<=",
    )
    add_gate(
        gates,
        name="mean_id_recall",
        observed=nested_metric(exact_report, ("summary", "mean_id_recall")),
        threshold=args.min_id_recall,
        op=">=",
    )
    add_gate(
        gates,
        name="mean_layer_recall",
        observed=nested_metric(exact_report, ("summary", "mean_layer_recall")),
        threshold=args.min_layer_recall,
        op=">=",
    )
    add_gate(
        gates,
        name="mean_number_recall",
        observed=nested_metric(exact_report, ("summary", "mean_number_recall")),
        threshold=args.min_number_recall,
        op=">=",
        required=False,
    )

    for context_mode, threshold in (
        ("open_book_context", args.min_open_book_exact),
        ("tool_observation", args.min_tool_observation_exact),
    ):
        add_gate(
            gates,
            name=f"{context_mode}_exact_pass_rate",
            observed=nested_metric(exact_report, ("by_context_mode", context_mode, "exact_graph_fact_pass_rate")),
            threshold=threshold,
            op=">=",
            required=False,
        )
    add_gate(
        gates,
        name="no_context_exact_pass_rate",
        observed=nested_metric(exact_report, ("by_context_mode", "no_context", "exact_graph_fact_pass_rate")),
        threshold=args.min_no_context_exact,
        op=">=",
        required=False,
    )

    for bucket, threshold in (
        ("entity_schema_grounding", args.min_schema_exact),
        ("multiplex_layer_metadata", args.min_schema_exact),
        ("local_topology", args.min_topology_exact),
        ("shortest_paths", args.min_topology_exact),
        ("rwr_vector_lookup", args.min_rwr_exact),
        ("module_set_algebra", args.min_module_exact),
        ("tool_observation_state_updates", args.min_tool_schema_exact),
    ):
        add_gate(
            gates,
            name=f"{bucket}_exact_pass_rate",
            observed=nested_metric(exact_report, ("by_mixture_bucket", bucket, "exact_graph_fact_pass_rate")),
            threshold=threshold,
            op=">=",
            required=False,
        )

    if trajectory_audit:
        add_gate(
            gates,
            name="trajectory_recovery_exact_success_rate",
            observed=first_present_metric(
                trajectory_audit,
                ("summary", "recovery_exact_success_rate"),
                ("recovery_exact_success_rate",),
            ),
            threshold=args.min_recovery_exact_success_rate,
            op=">=",
        )
        add_gate(
            gates,
            name="trajectory_refinement_exact_success_rate",
            observed=first_present_metric(
                trajectory_audit,
                ("summary", "refinement_exact_success_rate"),
                ("refinement_exact_success_rate",),
            ),
            threshold=args.min_refinement_exact_success_rate,
            op=">=",
        )
        add_gate(
            gates,
            name="trajectory_exact_membership_pair_rate",
            observed=first_present_metric(
                trajectory_audit,
                ("summary", "exact_membership_pair_rate"),
                ("exact_membership_pair_rate",),
            ),
            threshold=args.min_exact_membership_pair_rate,
            op=">=",
        )

    failed = [gate for gate in gates if gate["required"] and not gate["passed"]]
    advisory_failed = [gate for gate in gates if not gate["required"] and not gate["passed"]]
    return {
        "passed": not failed,
        "required_failure_count": len(failed),
        "advisory_failure_count": len(advisory_failed),
        "gates": gates,
        "failed_required_gates": failed,
        "failed_advisory_gates": advisory_failed,
        "decision": "move_to_dpo_trajectory_generation" if not failed else "continue_pretrajectory_sft_or_data_repair",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether pre-trajectory SFT is ready for DPO trajectory generation.")
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--exact-report", type=Path, required=True)
    parser.add_argument("--trajectory-audit", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fail-on-dataset-warnings", action="store_true")
    parser.add_argument("--min-overall-exact", type=float, default=0.80)
    parser.add_argument("--min-open-book-exact", type=float, default=0.90)
    parser.add_argument("--min-tool-observation-exact", type=float, default=0.90)
    parser.add_argument("--min-no-context-exact", type=float, default=0.55)
    parser.add_argument("--min-schema-exact", type=float, default=0.95)
    parser.add_argument("--min-topology-exact", type=float, default=0.80)
    parser.add_argument("--min-rwr-exact", type=float, default=0.85)
    parser.add_argument("--min-module-exact", type=float, default=0.85)
    parser.add_argument("--min-tool-schema-exact", type=float, default=0.95)
    parser.add_argument("--min-id-recall", type=float, default=0.90)
    parser.add_argument("--min-layer-recall", type=float, default=0.85)
    parser.add_argument("--min-number-recall", type=float, default=0.85)
    parser.add_argument("--max-unsupported-language-rate", type=float, default=0.005)
    parser.add_argument("--min-recovery-exact-success-rate", type=float, default=0.25)
    parser.add_argument("--min-refinement-exact-success-rate", type=float, default=0.25)
    parser.add_argument("--min-exact-membership-pair-rate", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_readiness(
        dataset_audit=read_json(args.dataset_audit),
        exact_report=read_json(args.exact_report),
        trajectory_audit=read_json(args.trajectory_audit) if args.trajectory_audit else None,
        args=args,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
