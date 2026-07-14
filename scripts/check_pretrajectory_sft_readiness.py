#!/usr/bin/env python3
"""Gate pre-trajectory SFT artifacts before DPO trajectory generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


READINESS_SCHEMA_VERSION = "pretrajectory-sft-readiness-v2"
CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v3"
CURRICULUM_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
REQUIRED_DATASET_AUDIT_SCHEMA_VERSION = CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION
REQUIRED_EVALUATOR_CONTRACT_VERSION = "pretrajectory-sft-exact-v3"
CURRICULUM_NATIVE_REPORT_NAMES = (
    "audit_report.json",
    "leakage_report.json",
    "coverage_report.json",
)


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
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def first_present_metric(payload: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        value = nested_metric(payload, path)
        if value is not None:
            return value
    return None


def _curriculum_audit_schema_is_current(dataset_audit: dict[str, Any]) -> bool:
    return (
        dataset_audit.get("schema_version") == REQUIRED_DATASET_AUDIT_SCHEMA_VERSION
        and dataset_audit.get("dataset_schema_version") == CURRICULUM_DATASET_SCHEMA_VERSION
    )


def _curriculum_native_reports_are_current(dataset_audit: dict[str, Any]) -> bool:
    native_reports = dataset_audit.get("native_reports")
    plan_hash = dataset_audit.get("plan_hash")
    return (
        isinstance(plan_hash, str)
        and bool(plan_hash)
        and isinstance(native_reports, dict)
        and all(
            isinstance(native_reports.get(name), dict)
            and native_reports[name].get("passed") is True
            and native_reports[name].get("plan_hash") == plan_hash
            for name in CURRICULUM_NATIVE_REPORT_NAMES
        )
    )


def _curriculum_exact_identity_matches(
    dataset_audit: dict[str, Any],
    exact_report: dict[str, Any],
) -> bool:
    identity = exact_report.get("dataset_identity")
    return (
        isinstance(identity, dict)
        and identity.get("dataset_schema_version") == CURRICULUM_DATASET_SCHEMA_VERSION
        and identity.get("plan_hash") == dataset_audit.get("plan_hash")
        and identity.get("content_hash") == dataset_audit.get("content_hash")
        and isinstance(identity.get("content_hash"), str)
        and bool(identity["content_hash"])
    )


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
        name="dataset_audit_schema_current",
        observed=int(_curriculum_audit_schema_is_current(dataset_audit)),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_audit_fatal_errors",
        observed=nested_metric(dataset_audit, ("fatal_error_count",)),
        threshold=0,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_audit_passed",
        observed=int(dataset_audit.get("passed") is True),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_native_reports_current",
        observed=int(_curriculum_native_reports_are_current(dataset_audit)),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_manifest_budget_contract_present",
        observed=int(
            isinstance(dataset_audit.get("answer_budget_report"), dict)
            and dataset_audit["answer_budget_report"].get("manifest_contract_present") is True
        ),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_over_budget_record_count",
        observed=nested_metric(dataset_audit, ("answer_budget_report", "over_budget_record_count")),
        threshold=0,
        op="==",
    )
    add_gate(
        gates,
        name="dataset_missing_budget_metadata_count",
        observed=nested_metric(
            dataset_audit,
            ("answer_budget_report", "missing_record_budget_metadata_count"),
        ),
        threshold=0,
        op="==",
    )
    if args.fail_on_dataset_warnings:
        add_gate(
            gates,
            name="dataset_audit_warnings",
            observed=nested_metric(dataset_audit, ("warning_count",)),
            threshold=0,
            op="==",
        )

    add_gate(
        gates,
        name="evaluator_contract_current",
        observed=int(
            exact_report.get("evaluator_contract", {}).get("version")
            == REQUIRED_EVALUATOR_CONTRACT_VERSION
            if isinstance(exact_report.get("evaluator_contract"), dict)
            else False
        ),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="gold_self_evaluation_passed",
        observed=int(
            exact_report.get("gold_self_evaluation", {}).get("passed") is True
            if isinstance(exact_report.get("gold_self_evaluation"), dict)
            else False
        ),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="exact_only_summary_present",
        observed=int(
            isinstance(exact_report.get("summary"), dict)
            and isinstance(exact_report["summary"].get("exact_only"), dict)
        ),
        threshold=1,
        op="==",
    )
    add_gate(
        gates,
        name="exact_report_dataset_identity_current",
        observed=int(_curriculum_exact_identity_matches(dataset_audit, exact_report)),
        threshold=1,
        op="==",
    )

    if exact_report.get("official_readiness_eligible") is False:
        failed_contract_gates = [
            gate for gate in gates if gate["required"] and not gate["passed"]
        ]
        if failed_contract_gates:
            return {
                "schema_version": READINESS_SCHEMA_VERSION,
                "valid": False,
                "applicable": False,
                "passed": False,
                "official_readiness_eligible": False,
                "evaluation_regime": exact_report.get("evaluation_regime"),
                "required_failure_count": len(failed_contract_gates),
                "advisory_failure_count": 0,
                "gates": gates,
                "failed_required_gates": failed_contract_gates,
                "failed_advisory_gates": [],
                "failed_contract_gates": failed_contract_gates,
                "decision": "repair_evaluation_or_dataset_contract",
            }
        return {
            "schema_version": READINESS_SCHEMA_VERSION,
            "valid": True,
            "applicable": False,
            "passed": None,
            "official_readiness_eligible": False,
            "evaluation_regime": exact_report.get("evaluation_regime"),
            "required_failure_count": 0,
            "advisory_failure_count": 0,
            "gates": gates,
            "failed_required_gates": [],
            "failed_advisory_gates": [],
            "failed_contract_gates": [],
            "decision": "diagnostic_only_no_readiness_decision",
        }

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
        observed=nested_metric(exact_report, ("summary", "exact_only", "mean_id_recall")),
        threshold=args.min_id_recall,
        op=">=",
    )
    add_gate(
        gates,
        name="mean_layer_recall",
        observed=nested_metric(exact_report, ("summary", "exact_only", "mean_layer_recall")),
        threshold=args.min_layer_recall,
        op=">=",
    )
    add_gate(
        gates,
        name="mean_number_recall",
        observed=nested_metric(exact_report, ("summary", "exact_only", "mean_number_recall")),
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

    bucket_thresholds = (
        ("entity_normalization_schema", args.min_schema_exact),
        ("layer_metadata_membership", args.min_schema_exact),
        ("edge_neighbor_topology", args.min_topology_exact),
        ("paths_layer_counts", args.min_topology_exact),
        ("subgraph_components_hubness", args.min_topology_exact),
        ("global_cohesion_calibration", args.min_topology_exact),
        ("rwr_distance_vectors", args.min_rwr_exact),
        ("module_set_algebra", args.min_module_exact),
        ("structured_context_and_tools", args.min_tool_schema_exact),
    )
    for bucket, threshold in bucket_thresholds:
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

    contract_gate_names = {
        "dataset_audit_schema_current",
        "dataset_audit_fatal_errors",
        "dataset_audit_passed",
        "dataset_native_reports_current",
        "dataset_manifest_budget_contract_present",
        "dataset_over_budget_record_count",
        "dataset_missing_budget_metadata_count",
        "evaluator_contract_current",
        "gold_self_evaluation_passed",
        "exact_only_summary_present",
        "exact_report_dataset_identity_current",
    }
    failed = [gate for gate in gates if gate["required"] and not gate["passed"]]
    advisory_failed = [gate for gate in gates if not gate["required"] and not gate["passed"]]
    failed_contract_gates = [gate for gate in failed if gate["name"] in contract_gate_names]
    contract_valid = not failed_contract_gates
    if failed_contract_gates:
        decision = "repair_evaluation_or_dataset_contract"
    elif failed:
        decision = "continue_pretrajectory_sft_or_data_repair"
    else:
        decision = "move_to_dpo_trajectory_generation"
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "valid": contract_valid,
        "passed": not failed,
        "required_failure_count": len(failed),
        "advisory_failure_count": len(advisory_failed),
        "gates": gates,
        "failed_required_gates": failed,
        "failed_advisory_gates": advisory_failed,
        "failed_contract_gates": failed_contract_gates,
        "decision": decision,
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
