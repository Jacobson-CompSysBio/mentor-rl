#!/usr/bin/env python
"""Precompute smoke-test caches before the live generator smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import RuntimeEnvironment, ToolAction, ToolObservation, ToolObservationStatus
from scripts.generate_trajectories import (
    DEFAULT_FULL_BRAIN_STORE_DIR,
    DEFAULT_REQUIRE_RWR_HPC,
    DEFAULT_RWR_HPC_EDGELIST_HAS_HEADERS,
    DEFAULT_RWR_HPC_CACHE_DIR,
    DEFAULT_TASKS_PATH,
    DEFAULT_USE_FULL_BRAIN_RWR_HPC,
    _prefetch_mechanism_evidence_cache,
)
from scripts.smoke_test_generator import _build_environment, _load_task


RWR_PRECOMPUTE_PROFILES = ("core", "extended")
EXTENDED_RWR_PRECOMPUTE_TOOLS = ("rwr_loe", "get_seed_essentiality", "get_layer_ablation")


def _safe_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _visible_seed_gene_ids(task_row: dict[str, Any]) -> list[str]:
    visible_inputs = task_row.get("visible_inputs")
    if not isinstance(visible_inputs, dict):
        return []
    return _safe_string_list(visible_inputs.get("seed_gene_ids"))


def build_rwr_precompute_actions(
    task_row: dict[str, Any],
    *,
    profile: str = "core",
    rwr_top_k: int,
    rwr_loe_top_k: int,
    shortest_paths_max_pairs: int,
    shortest_paths_max_paths: int,
) -> list[ToolAction]:
    """Build deterministic RWR++ cache-warming requests for one smoke task."""

    if profile not in RWR_PRECOMPUTE_PROFILES:
        raise ValueError(f"profile must be one of {RWR_PRECOMPUTE_PROFILES}")

    seed_genes = _visible_seed_gene_ids(task_row)
    if not seed_genes:
        return []

    actions: list[ToolAction] = []
    if profile == "extended":
        actions.append(
            ToolAction(
                tool_name="rwr_loe",
                arguments={"seed_genes": seed_genes, "top_k": rwr_loe_top_k},
                call_id="precompute.rwr_loe.visible_seeds",
            )
        )

    actions.append(
        ToolAction(
            tool_name="rwr",
            arguments={"seed_genes": seed_genes, "top_k": rwr_top_k},
            call_id="precompute.rwr.visible_seeds",
        )
    )

    if len(seed_genes) >= 2 and shortest_paths_max_pairs > 0:
        source_gene = seed_genes[0]
        for pair_index, target_gene in enumerate(seed_genes[1 : 1 + shortest_paths_max_pairs]):
            actions.append(
                ToolAction(
                    tool_name="shortest_paths",
                    arguments={
                        "source_genes": [source_gene],
                        "target_genes": [target_gene],
                        "max_paths": shortest_paths_max_paths,
                    },
                    call_id=f"precompute.shortest_paths.visible_seed_pair_{pair_index}",
                )
            )

    if len(seed_genes) >= 2:
        source_gene = seed_genes[0]
        target_gene = seed_genes[1]
        actions.extend(
            [
                ToolAction(
                    tool_name="get_rank",
                    arguments={"source_gene": source_gene, "target_gene": target_gene},
                    call_id="precompute.get_rank.visible_seed_pair_0",
                ),
                ToolAction(
                    tool_name="get_distance",
                    arguments={"gene_a": source_gene, "gene_b": target_gene},
                    call_id="precompute.get_distance.visible_seed_pair_0",
                ),
                ToolAction(
                    tool_name="get_spearman",
                    arguments={"gene_a": source_gene, "gene_b": target_gene},
                    call_id="precompute.get_spearman.visible_seed_pair_0",
                ),
                ToolAction(
                    tool_name="get_pearson",
                    arguments={"gene_a": source_gene, "gene_b": target_gene},
                    call_id="precompute.get_pearson.visible_seed_pair_0",
                ),
                ToolAction(
                    tool_name="get_dot_similarity",
                    arguments={"gene_a": source_gene, "gene_b": target_gene},
                    call_id="precompute.get_dot_similarity.visible_seed_pair_0",
                ),
                ToolAction(
                    tool_name="get_path_layer_counts",
                    arguments={
                        "source_genes": [source_gene],
                        "target_genes": [target_gene],
                        "max_paths": shortest_paths_max_paths,
                        "top_k": 10,
                    },
                    call_id="precompute.get_path_layer_counts.visible_seed_pair_0",
                ),
            ]
        )

    actions.extend(
        [
            ToolAction(
                tool_name="get_rank_vector_summary",
                arguments={"seed_genes": seed_genes, "top_k": rwr_top_k, "include_seed_genes": False},
                call_id="precompute.get_rank_vector_summary.visible_seeds",
            ),
            ToolAction(
                tool_name="get_encoding_summary",
                arguments={"seed_genes": seed_genes, "top_k": rwr_top_k, "include_seed_genes": False},
                call_id="precompute.get_encoding_summary.visible_seeds",
            ),
            ToolAction(
                tool_name="get_layer_stats",
                arguments={"top_k": 10},
                call_id="precompute.get_layer_stats",
            ),
            ToolAction(
                tool_name="get_component_summary",
                arguments={"genes": seed_genes, "max_components": 10},
                call_id="precompute.get_component_summary.visible_seeds",
            ),
        ]
    )

    if profile == "extended":
        actions.extend(
            [
                ToolAction(
                    tool_name="get_seed_essentiality",
                    arguments={"seed_genes": seed_genes, "top_k": len(seed_genes), "n_samples_null_dist": 10},
                    call_id="precompute.get_seed_essentiality.visible_seeds",
                ),
                ToolAction(
                    tool_name="get_layer_ablation",
                    arguments={"seed_genes": seed_genes, "top_k": 10},
                    call_id="precompute.get_layer_ablation.visible_seeds",
                ),
            ]
        )

    actions.extend(
        [
            ToolAction(
                tool_name="get_gene_layers",
                arguments={"gene": seed_genes[0]},
                call_id="precompute.get_gene_layers.visible_seed_0",
            ),
            ToolAction(
                tool_name="get_nodes_by_layer",
                arguments={"gene": seed_genes[0]},
                call_id="precompute.get_nodes_by_layer.visible_seed_0",
            ),
        ]
    )

    if len(seed_genes) >= 2:
        actions.append(
            ToolAction(
                tool_name="get_node_perturbation",
                arguments={
                    "seed_genes": [seed_genes[0]],
                    "perturb_genes": [seed_genes[1]],
                    "top_k": 10,
                },
                call_id="precompute.get_node_perturbation.visible_seed_pair_0",
            )
        )

    return actions


def skipped_rwr_precompute_tools(profile: str) -> list[dict[str, str]]:
    if profile == "extended":
        return []
    if profile != "core":
        raise ValueError(f"profile must be one of {RWR_PRECOMPUTE_PROFILES}")
    reason = (
        "Skipped by the default smoke precompute profile because this full-brain "
        "RWR++ app is heavyweight or currently brittle; run with "
        "--rwr-precompute-profile extended for a dedicated diagnostic warmup."
    )
    return [{"tool_name": tool_name, "reason": reason} for tool_name in EXTENDED_RWR_PRECOMPUTE_TOOLS]


def build_native_store_precompute_actions(
    task_row: dict[str, Any],
    *,
    max_genes: int,
) -> list[ToolAction]:
    """Build native graph calls that touch the compiled full-brain store."""

    seed_genes = _visible_seed_gene_ids(task_row)
    if not seed_genes or max_genes <= 0:
        return []

    selected = seed_genes[:max_genes]
    actions: list[ToolAction] = [
        ToolAction(
            tool_name="get_neighbors",
            arguments={"gene": gene_id},
            call_id=f"precompute.get_neighbors.{index}",
        )
        for index, gene_id in enumerate(selected)
    ]
    if len(selected) >= 2:
        actions.append(
            ToolAction(
                tool_name="induce_subgraph",
                arguments={"genes": selected},
                call_id="precompute.induce_subgraph.visible_seeds",
            )
        )
    return actions


def _observation_summary(observation: ToolObservation) -> dict[str, Any]:
    payload = observation.payload or {}
    summary: dict[str, Any] = {
        "tool_name": observation.provenance.get("tool_name"),
        "call_id": observation.call_id,
        "status": observation.status.value,
        "cache_hit": observation.provenance.get("cache_hit"),
        "network_used": observation.provenance.get("network_used"),
        "backend": observation.provenance.get("backend"),
        "error": observation.error,
    }
    for key in (
        "result_count",
        "ranked_gene_count",
        "path_count",
        "combined_edge_count",
        "target_rank",
        "distance",
        "spearman_correlation",
        "pearson_correlation",
        "dot_similarity",
        "layer_count",
        "total_components",
    ):
        if key in payload:
            summary[key] = payload[key]
    if "results" in payload and isinstance(payload["results"], list):
        summary["result_count"] = len(payload["results"])
    if "unique_neighbors" in payload and isinstance(payload["unique_neighbors"], list):
        summary["unique_neighbor_count"] = len(payload["unique_neighbors"])
    if "ranked_genes" in payload and isinstance(payload["ranked_genes"], list):
        summary["ranked_gene_count"] = len(payload["ranked_genes"])
    if "paths" in payload and isinstance(payload["paths"], list):
        summary["path_count"] = len(payload["paths"])
    return summary


def _execute_actions(environment: RuntimeEnvironment, actions: list[ToolAction]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    tool_status_counts: Counter[str] = Counter()
    observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for action in actions:
        observation = environment.execute(action)
        status_counts[observation.status.value] += 1
        tool_name = observation.provenance.get("tool_name", action.tool_name)
        tool_status_counts[f"{tool_name}.{observation.status.value}"] += 1
        summary = _observation_summary(observation)
        observations.append(summary)
        if observation.status in {ToolObservationStatus.INVALID, ToolObservationStatus.ERROR}:
            errors.append(summary)

    return {
        "action_count": len(actions),
        "status_counts": dict(sorted(status_counts.items())),
        "tool_status_counts": dict(sorted(tool_status_counts.items())),
        "error_count": len(errors),
        "errors_preview": errors[:10],
        "observations": observations,
    }


def _count_rwr_cache_entries(cache_dir: Path | None) -> int | None:
    if cache_dir is None or not cache_dir.exists():
        return 0 if cache_dir is not None else None
    return sum(1 for path in cache_dir.rglob("payload.json") if path.is_file())


def _runtime_cache_summary(environment: RuntimeEnvironment) -> dict[str, Any]:
    description = environment.describe()
    return {
        "runtime_version": description.get("runtime_version"),
        "graph_backend_kind": description.get("graph_backend_kind"),
        "gene_count": description.get("gene_count"),
        "layer_count": description.get("layer_count"),
        "mygene_cache_size": description.get("mygene_cache_size"),
        "enrichment_cache_size": description.get("enrichment_cache_size"),
        "enrichment_background_gene_count": description.get("enrichment_background_gene_count"),
        "rwr_hpc_apps_enabled": description.get("rwr_hpc_apps_enabled"),
        "rwr_hpc_apps_missing_required": description.get("rwr_hpc_apps_missing_required"),
        "rwr_hpc_structured_tools_enabled": description.get("rwr_hpc_structured_tools_enabled"),
        "rwr_hpc_structured_tools_required": description.get("rwr_hpc_structured_tools_required"),
        "rwr_hpc_flist": description.get("rwr_hpc_flist"),
        "rwr_hpc_cache_dir": description.get("rwr_hpc_cache_dir"),
        "rwr_hpc_scratch_root": description.get("rwr_hpc_scratch_root"),
    }


def precompute_smoke_caches(args: argparse.Namespace) -> dict[str, Any]:
    task_row = _load_task(args.tasks_path, task_id=args.task_id, task_index=args.task_index)
    environment = _build_environment(args)
    report: dict[str, Any] = {
        "task_id": task_row.get("task_id"),
        "task_index": args.task_index if args.task_id is None else None,
        "visible_seed_gene_ids": _visible_seed_gene_ids(task_row),
        "runtime": _runtime_cache_summary(environment),
        "rwr_cache_entries_before": _count_rwr_cache_entries(args.rwr_hpc_cache_dir),
        "annotation_prefetch": None,
        "native_store_precompute": None,
        "rwr_precompute": None,
    }

    if args.prefetch_mechanism_cache:
        report["annotation_prefetch"] = _prefetch_mechanism_evidence_cache(
            [task_row],
            environment,
            mygene_per_task=args.prefetch_mygene_per_task,
            enrichment_top_k=args.prefetch_enrichment_top_k,
        )

    if args.precompute_native_store:
        native_actions = build_native_store_precompute_actions(
            task_row,
            max_genes=args.native_precompute_max_genes,
        )
        report["native_store_precompute"] = _execute_actions(environment, native_actions)

    if args.precompute_rwr_cache:
        rwr_actions = build_rwr_precompute_actions(
            task_row,
            profile=args.rwr_precompute_profile,
            rwr_top_k=args.rwr_top_k,
            rwr_loe_top_k=args.rwr_loe_top_k,
            shortest_paths_max_pairs=args.shortest_paths_max_pairs,
            shortest_paths_max_paths=args.shortest_paths_max_paths,
        )
        report["rwr_precompute"] = _execute_actions(environment, rwr_actions)
        report["rwr_precompute"]["profile"] = args.rwr_precompute_profile
        report["rwr_precompute"]["skipped"] = skipped_rwr_precompute_tools(args.rwr_precompute_profile)

    report["rwr_cache_entries_after"] = _count_rwr_cache_entries(args.rwr_hpc_cache_dir)
    report["mygene_cache_size_after"] = len(environment.mygene_cache)
    report["enrichment_cache_size_after"] = len(environment.enrichment_cache)
    return report


def _report_has_errors(report: dict[str, Any]) -> bool:
    for key in ("annotation_prefetch", "native_store_precompute", "rwr_precompute"):
        section = report.get(key)
        if isinstance(section, dict) and int(section.get("error_count", 0) or 0) > 0:
            return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute annotation and RWR++ caches for the smoke-test task."
    )
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--compiled-library-path", type=Path, default=None)
    parser.add_argument("--multiplex-flist", type=Path, default=None)
    parser.add_argument(
        "--use-full-brain-rwr-hpc",
        dest="use_full_brain_rwr_hpc",
        action="store_true",
        default=DEFAULT_USE_FULL_BRAIN_RWR_HPC,
    )
    parser.add_argument(
        "--no-use-full-brain-rwr-hpc",
        dest="use_full_brain_rwr_hpc",
        action="store_false",
    )
    parser.add_argument("--rwr-hpc-flist", type=Path, default=None)
    parser.add_argument("--full-brain-store-dir", type=Path, default=DEFAULT_FULL_BRAIN_STORE_DIR)
    parser.add_argument("--rwr-hpc-build-dir", type=Path, default=None)
    parser.add_argument("--rwr-hpc-app-manifest-path", type=Path, default=None)
    parser.add_argument("--rwr-hpc-cache-dir", type=Path, default=DEFAULT_RWR_HPC_CACHE_DIR)
    parser.add_argument("--rwr-hpc-scratch-root", type=Path, default=None)
    parser.add_argument("--rwr-hpc-build-id", type=str, default=None)
    parser.add_argument(
        "--rwr-hpc-edgelist-has-headers",
        dest="rwr_hpc_edgelist_has_headers",
        action="store_true",
        default=DEFAULT_RWR_HPC_EDGELIST_HAS_HEADERS,
    )
    parser.add_argument(
        "--rwr-hpc-edgelist-no-headers",
        dest="rwr_hpc_edgelist_has_headers",
        action="store_false",
    )
    parser.add_argument(
        "--require-rwr-hpc",
        dest="require_rwr_hpc",
        action="store_true",
        default=DEFAULT_REQUIRE_RWR_HPC,
    )
    parser.add_argument(
        "--no-require-rwr-hpc",
        dest="require_rwr_hpc",
        action="store_false",
    )
    parser.add_argument("--mygene-cache-path", type=Path, default=None)
    parser.add_argument("--allow-network-mygene", action="store_true")
    parser.add_argument("--enrichment-cache-path", type=Path, default=None)
    parser.add_argument("--allow-network-enrichment", action="store_true")
    parser.add_argument("--enrichment-background-path", type=Path, default=None)
    parser.add_argument("--prefetch-mechanism-cache", action="store_true")
    parser.add_argument("--prefetch-mygene-per-task", type=int, default=3)
    parser.add_argument("--prefetch-enrichment-top-k", type=int, default=10)
    parser.add_argument("--prefetch-max-tasks", type=int, default=None)
    parser.add_argument("--precompute-rwr-cache", action="store_true")
    parser.add_argument(
        "--rwr-precompute-profile",
        choices=RWR_PRECOMPUTE_PROFILES,
        default="core",
        help=(
            "RWR++ smoke cache profile. 'core' warms stable inexpensive cache "
            "entries; 'extended' also runs heavyweight rwr_loe, GRIN "
            "essentiality, and layer ablation diagnostics."
        ),
    )
    parser.add_argument("--rwr-top-k", type=int, default=500)
    parser.add_argument("--rwr-loe-top-k", type=int, default=20)
    parser.add_argument("--shortest-paths-max-pairs", type=int, default=1)
    parser.add_argument("--shortest-paths-max-paths", type=int, default=20)
    parser.add_argument("--precompute-native-store", action="store_true")
    parser.add_argument("--native-precompute-max-genes", type=int, default=2)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = precompute_smoke_caches(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered + "\n", encoding="utf-8")
    if args.require_success and _report_has_errors(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
