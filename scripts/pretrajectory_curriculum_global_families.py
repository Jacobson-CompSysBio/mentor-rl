"""Oracle-backed generators for global curriculum families 56--63 and 81.

The production builder owns rendering, split assignment, budgets, and dataset
selection.  This module only constructs bounded ``CurriculumExample`` objects
from facts already exposed by that builder.  In particular:

* Spearman values come from materialized lower-triangular RWR shards;
* graph edge, degree, and layer-membership counts come from the full CSR oracle;
* module membership comes from the loaded mixed module oracle; and
* every aggregate states its bounded scope and supplies the values needed to
  recompute the answer.

No filesystem paths are accepted as evidence or returned as provenance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


GLOBAL_FAMILY_NAMES = (
    "within_clade_distance",
    "within_clade_vs_random",
    "clustering_ratio",
    "subgraph_density",
    "conductance_boundary_ratio",
    "cell_type_specific_cohesion",
    "layer_sensitive_cohesion",
    "nearest_modules",
    "whole_multiplex_context_profile",
)

DISTANCE_PAGE_SIZE = 5
MAX_DISTANCE_SHARDS = 8
NULL_SET_COUNT = 8
MODULE_NEIGHBOR_COUNT = 5
TOPOLOGY_PAGE_SIZE = 6
MAX_CELL_CONTEXTS_PER_PAGE = 6
LAYER_ID_PAGE_SIZE = 16


class GlobalFamilyGenerationError(RuntimeError):
    """Raised when real artifacts cannot fill a required global family."""


@dataclass(frozen=True)
class ModuleDistancePage:
    module_id: str
    module_source: str
    module_gene_ids: tuple[str, ...]
    shard_id: str
    gene_ids: tuple[str, ...]
    page_index: int
    page_count: int
    same_shard_gene_count: int

    @property
    def page_id(self) -> str:
        return f"{self.module_id}:{self.shard_id}:page_{self.page_index:04d}"

    def page_payload(self) -> dict[str, Any]:
        return {
            "scope": "same_shard_module_gene_page",
            "page_index": self.page_index,
            "page_count": self.page_count,
            "page_size": len(self.gene_ids),
            "same_shard_gene_count": self.same_shard_gene_count,
            "module_gene_count": len(self.module_gene_ids),
            "complete_for_same_shard_subset": self.page_count == 1,
            "complete_for_module": len(self.gene_ids) == len(self.module_gene_ids),
        }

    def metadata_page(self) -> dict[str, Any]:
        return {
            "index": self.page_index,
            "count": self.page_count,
            "size": len(self.gene_ids),
            "total_items": self.same_shard_gene_count,
            "scope": "same_shard_module_gene_page",
            "parent_total_items": len(self.module_gene_ids),
        }


@dataclass(frozen=True)
class WithinDistanceFact:
    page: ModuleDistancePage
    pair_distances: tuple[tuple[str, str, float], ...]
    mean_distance: float


@dataclass(frozen=True)
class LayerPageStats:
    layer_id: str
    internal_edges: tuple[tuple[str, str], ...]
    possible_edge_count: int
    density: float

    @property
    def edge_count(self) -> int:
        return len(self.internal_edges)

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "internal_edges": [
                {"gene_a": gene_a, "gene_b": gene_b}
                for gene_a, gene_b in self.internal_edges
            ],
            "internal_edge_count": self.edge_count,
            "possible_edge_count": self.possible_edge_count,
            "edge_density": self.density,
        }


@dataclass(frozen=True)
class TopologyPage:
    module_id: str
    module_source: str
    module_gene_count: int
    gene_ids: tuple[str, ...]
    higher: LayerPageStats
    lower: LayerPageStats

    @property
    def page_id(self) -> str:
        return f"{self.module_id}:{self.higher.layer_id}:{self.lower.layer_id}:{','.join(self.gene_ids)}"

    def page_payload(self) -> dict[str, Any]:
        return {
            "scope": "edge_anchored_bounded_module_page",
            "page_index": 0,
            "page_count": 1,
            "page_size": len(self.gene_ids),
            "module_gene_count": self.module_gene_count,
            "complete_for_module": len(self.gene_ids) == self.module_gene_count,
            "selection_rule": "real_scpen_edge_endpoints_plus_deterministic_module_genes",
        }

    def metadata_page(self) -> dict[str, Any]:
        return {
            "index": 0,
            "count": 1,
            "size": len(self.gene_ids),
            "total_items": len(self.gene_ids),
            "scope": "edge_anchored_bounded_module_page",
            "parent_total_items": self.module_gene_count,
        }


def _digest(*, seed: int, namespace: str, value: Any) -> str:
    encoded = json.dumps(
        {"seed": int(seed), "namespace": namespace, "value": value},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered(values: Iterable[Any], *, builder: Any, namespace: str) -> list[Any]:
    return sorted(
        values,
        key=lambda value: _digest(seed=builder.seed, namespace=namespace, value=value),
    )


def _source_label(module: Mapping[str, Any]) -> str:
    if module.get("source") == "MENTOR_GW_DENDROGRAM":
        return "mentor_ev"
    if module.get("source") == "RWR_LOE_FULL_BRAIN":
        return "rwr_loe"
    raise GlobalFamilyGenerationError(
        f"Unsupported module source for global curriculum: {module.get('source')!r}."
    )


def _module_coverage_key(source: str) -> str:
    return "mentor_ev_modules" if source == "mentor_ev" else "rwr_loe_modules"


def _layer_family(layer_id: str) -> str:
    prefix = layer_id.split(":", 1)[0]
    aliases = {
        "scPEN": "sc_pen",
        "bulkPEN": "bulk_pen",
        "HumanNetV3": "humannet_v3",
        "TFs": "tf_target",
    }
    return aliases.get(prefix, prefix.lower())


def _example_type(builder: Any) -> Any:
    explicit = getattr(builder, "curriculum_example_type", None)
    if explicit is not None:
        return explicit
    # Delayed import keeps this helper safe to import at the top of the builder
    # without creating a module-initialization cycle.
    from scripts.build_pretrajectory_sft_curriculum import CurriculumExample

    return CurriculumExample


def _rwr_provenance(builder: Any, source: str) -> dict[str, Any]:
    return {
        **builder._base_provenance(source),
        **builder.rwr.public_provenance(),
        "artifact_scope": "bounded_or_summary_only",
    }


def _pair_distances(shard: Any, genes: Sequence[str]) -> tuple[tuple[str, str, float], ...]:
    rows: list[tuple[str, str, float]] = []
    for offset, gene_a in enumerate(genes):
        for gene_b in genes[offset + 1 :]:
            rows.append((gene_a, gene_b, float(shard.distance(gene_a, gene_b))))
    return tuple(rows)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise GlobalFamilyGenerationError("A global numeric aggregate had no input values.")
    return float(statistics.fmean(materialized))


def _distance_page_pool(builder: Any, required: int) -> list[ModuleDistancePage]:
    if builder.rwr is None:
        raise GlobalFamilyGenerationError("RWR rank/distance artifacts are required.")
    shard_ids = _ordered(builder.rwr.shard_ids, builder=builder, namespace="global_distance_shards")
    selected_shards = set(shard_ids[: min(MAX_DISTANCE_SHARDS, len(shard_ids))])
    if not selected_shards:
        raise GlobalFamilyGenerationError("No materialized distance shards are available.")

    modules = sorted(
        builder.modules,
        key=lambda module: _digest(
            seed=builder.seed,
            namespace="global_distance_modules",
            value=str(module.get("module_id", "")),
        ),
    )
    pages: list[ModuleDistancePage] = []
    seen: set[tuple[str, str, int]] = set()
    for module in modules:
        module_id = str(module.get("module_id", ""))
        raw_genes = module.get("gene_ids")
        if not module_id or not isinstance(raw_genes, list):
            continue
        module_genes = tuple(sorted({str(gene) for gene in raw_genes}))
        by_shard: dict[str, list[str]] = defaultdict(list)
        for gene in module_genes:
            try:
                route = builder.rwr.route_seed(gene)
            except (KeyError, ValueError):
                continue
            if route.shard_id in selected_shards:
                by_shard[route.shard_id].append(gene)
        for shard_id in _ordered(by_shard, builder=builder, namespace=f"module_shards:{module_id}"):
            genes = _ordered(
                sorted(set(by_shard[shard_id])),
                builder=builder,
                namespace=f"module_shard_genes:{module_id}:{shard_id}",
            )
            if len(genes) < 3:
                continue
            page_count = math.ceil(len(genes) / DISTANCE_PAGE_SIZE)
            for page_index in range(page_count):
                page_genes = tuple(
                    genes[
                        page_index * DISTANCE_PAGE_SIZE : (page_index + 1) * DISTANCE_PAGE_SIZE
                    ]
                )
                if len(page_genes) < 3:
                    continue
                key = (module_id, shard_id, page_index)
                if key in seen:
                    continue
                seen.add(key)
                pages.append(
                    ModuleDistancePage(
                        module_id=module_id,
                        module_source=_source_label(module),
                        module_gene_ids=module_genes,
                        shard_id=shard_id,
                        gene_ids=page_genes,
                        page_index=page_index,
                        page_count=page_count,
                        same_shard_gene_count=len(genes),
                    )
                )
                if len(pages) >= required:
                    return pages
    if len(pages) < required:
        raise GlobalFamilyGenerationError(
            f"Only {len(pages)} real same-shard module pages were available; {required} required."
        )
    return pages


def _within_facts(builder: Any, pages: Sequence[ModuleDistancePage]) -> list[WithinDistanceFact]:
    facts: list[WithinDistanceFact] = []
    for page in pages:
        shard = builder.rwr.distance_shard(page.shard_id)
        pairs = _pair_distances(shard, page.gene_ids)
        facts.append(
            WithinDistanceFact(
                page=page,
                pair_distances=pairs,
                mean_distance=_mean(value for _, _, value in pairs),
            )
        )
    return facts


def _distance_evidence(fact: WithinDistanceFact) -> dict[str, Any]:
    return {
        "context_type": "rwr_spearman_distance_module_page",
        "distance_metric": "spearman_distance",
        "lower_is_closer": True,
        "module_id": fact.page.module_id,
        "module_source": fact.page.module_source,
        "distance_shard_id": fact.page.shard_id,
        "page": fact.page.page_payload(),
        "gene_ids_page": list(fact.page.gene_ids),
        "pair_distances": [
            {"gene_a": gene_a, "gene_b": gene_b, "distance": distance}
            for gene_a, gene_b, distance in fact.pair_distances
        ],
    }


def _distance_example_common(fact: WithinDistanceFact) -> dict[str, Any]:
    module_key = _module_coverage_key(fact.page.module_source)
    return {
        "strongest_group_id": f"module:{fact.page.module_id}",
        "module_source": fact.page.module_source,
        "context_budget_profile": "evidence_2k",
        "provenance": None,
        "coverage": {
            "canonical_genes": list(fact.page.gene_ids),
            module_key: [fact.page.module_id],
            "rwr_distance_shards": [fact.page.shard_id],
        },
        "page": fact.page.metadata_page(),
    }


def _emit_within_distance(
    builder: Any,
    example_cls: Any,
    facts: Sequence[WithinDistanceFact],
    goal: int,
) -> None:
    for fact in facts[:goal]:
        evidence = _distance_evidence(fact)
        answer = {
            "module_id": fact.page.module_id,
            "aggregation_scope": "supplied_same_shard_module_gene_page",
            "page_index": fact.page.page_index,
            "pair_count": len(fact.pair_distances),
            "mean_within_distance": fact.mean_distance,
            "distance_metric": "spearman_distance",
            "lower_is_closer": True,
        }
        common = _distance_example_common(fact)
        common["provenance"] = _rwr_provenance(builder, "rwr_distance_shard_and_module_oracle")
        builder.add(
            example_cls(
                family=builder.family("within_clade_distance"),
                book_mode="open_book",
                task=(
                    "Compute the arithmetic mean of every supplied pair distance for this bounded "
                    "same-shard module page. Do not generalize it to the complete module. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                evidence_handles=[f"module_distance_page:{fact.page.page_id}"],
                validator={"type": "recompute_arithmetic_mean", "value_field": "distance"},
                **common,
            )
        )


def _null_sets_for_fact(builder: Any, fact: WithinDistanceFact) -> tuple[list[dict[str, Any]], int]:
    shard = builder.rwr.distance_shard(fact.page.shard_id)
    module_genes = set(fact.page.module_gene_ids)
    eligible = [gene for gene in shard.genes if gene not in module_genes]
    size = len(fact.page.gene_ids)
    if len(eligible) < size:
        return [], 0
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in range(NULL_SET_COUNT * 4):
        ordered = _ordered(
            eligible,
            builder=builder,
            namespace=f"null:{fact.page.page_id}:{variant}",
        )
        genes = tuple(sorted(ordered[:size]))
        if genes in seen:
            continue
        seen.add(genes)
        pairs = _pair_distances(shard, genes)
        rows.append(
            {
                "null_set_index": len(rows),
                "gene_ids": list(genes),
                "pair_count": len(pairs),
                "mean_distance": _mean(value for _, _, value in pairs),
            }
        )
        if len(rows) == NULL_SET_COUNT:
            break
    return rows, size


def _emit_null_comparison(
    builder: Any,
    example_cls: Any,
    facts: Sequence[WithinDistanceFact],
    goal: int,
) -> None:
    emitted = 0
    for fact in facts:
        null_sets, set_size = _null_sets_for_fact(builder, fact)
        if not null_sets:
            continue
        at_least_as_close = sum(
            float(row["mean_distance"]) <= fact.mean_distance for row in null_sets
        )
        p_value = (1 + at_least_as_close) / (len(null_sets) + 1)
        evidence = {
            **_distance_evidence(fact),
            "observed_mean_within_distance": fact.mean_distance,
            "null_sampling": {
                "scope": "deterministic_same_shard_nonmodule_sets",
                "same_size": True,
                "set_size": set_size,
                "without_replacement_within_each_set": True,
                "null_set_count": len(null_sets),
            },
            "null_set_summaries": null_sets,
            "empirical_p_value_rule": "(1 + count(null_mean <= observed_mean)) / (1 + null_set_count)",
        }
        answer = {
            "module_id": fact.page.module_id,
            "aggregation_scope": "supplied_same_shard_module_page_and_bounded_null_sets",
            "observed_mean_within_distance": fact.mean_distance,
            "null_set_count": len(null_sets),
            "null_means_at_least_as_close_count": at_least_as_close,
            "empirical_p_value": p_value,
            "empirical_p_value_rule": "add_one_lower_tail",
            "interpretation": (
                "closer_than_the_supplied_same_shard_null_at_alpha_0.05"
                if p_value <= 0.05
                else "not_closer_than_the_supplied_same_shard_null_at_alpha_0.05"
            ),
        }
        common = _distance_example_common(fact)
        common["provenance"] = _rwr_provenance(builder, "rwr_distance_shard_module_and_null_oracle")
        builder.add(
            example_cls(
                family=builder.family("within_clade_vs_random"),
                book_mode="open_book",
                task=(
                    "Using the declared add-one lower-tail rule, compute the empirical p-value from "
                    "the supplied observed and same-shard null means. Restrict the claim to this null. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                evidence_handles=[f"module_distance_null_page:{fact.page.page_id}"],
                validator={"type": "recompute_empirical_p_value_add_one_lower_tail"},
                **common,
            )
        )
        emitted += 1
        if emitted >= goal:
            return
    raise GlobalFamilyGenerationError(
        f"within_clade_vs_random emitted {emitted} examples; required {goal}."
    )


def _emit_clustering_ratio(
    builder: Any,
    example_cls: Any,
    facts: Sequence[WithinDistanceFact],
    goal: int,
) -> None:
    emitted = 0
    for fact in facts:
        if fact.mean_distance <= 0.0:
            continue
        shard = builder.rwr.distance_shard(fact.page.shard_id)
        outside = [gene for gene in shard.genes if gene not in set(fact.page.module_gene_ids)]
        if len(outside) < len(fact.page.gene_ids):
            continue
        outside_page = tuple(
            _ordered(
                outside,
                builder=builder,
                namespace=f"outside_page:{fact.page.page_id}",
            )[: len(fact.page.gene_ids)]
        )
        cross_rows = [
            (inside_gene, outside_gene, float(shard.distance(inside_gene, outside_gene)))
            for inside_gene in fact.page.gene_ids
            for outside_gene in outside_page
        ]
        outside_mean = _mean(value for _, _, value in cross_rows)
        ratio = outside_mean / fact.mean_distance
        evidence = {
            **_distance_evidence(fact),
            "outside_module_gene_page": list(outside_page),
            "cross_boundary_distances": [
                {"module_gene": gene_a, "outside_gene": gene_b, "distance": value}
                for gene_a, gene_b, value in cross_rows
            ],
            "ratio_definition": "mean_cross_boundary_distance / mean_within_page_distance",
        }
        answer = {
            "module_id": fact.page.module_id,
            "aggregation_scope": "bounded_same_shard_module_and_outside_pages",
            "mean_within_distance": fact.mean_distance,
            "mean_outside_distance": outside_mean,
            "clustering_ratio": ratio,
            "ratio_definition": "mean_outside_distance / mean_within_distance",
            "interpretation": (
                "within_page_is_closer_than_cross_boundary_page"
                if ratio > 1.0
                else "within_page_is_not_closer_than_cross_boundary_page"
            ),
        }
        common = _distance_example_common(fact)
        common["provenance"] = _rwr_provenance(builder, "rwr_distance_shard_and_module_oracle")
        coverage = dict(common["coverage"])
        coverage["canonical_genes"] = list(fact.page.gene_ids + outside_page)
        common["coverage"] = coverage
        builder.add(
            example_cls(
                family=builder.family("clustering_ratio"),
                book_mode="open_book",
                task=(
                    "Compute the declared outside-to-within distance ratio from all supplied values. "
                    "Keep the interpretation scoped to these bounded pages. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                evidence_handles=[f"module_clustering_ratio_page:{fact.page.page_id}"],
                validator={"type": "recompute_mean_distance_ratio"},
                **common,
            )
        )
        emitted += 1
        if emitted >= goal:
            return
    raise GlobalFamilyGenerationError(f"clustering_ratio emitted {emitted} examples; required {goal}.")


def _canonical_edge_pairs(edges: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    result = {
        tuple(sorted((str(edge.source_gene_id), str(edge.target_gene_id))))
        for edge in edges
    }
    return tuple(sorted(result))


def _layer_stats(builder: Any, genes: Sequence[str], layer_id: str) -> LayerPageStats:
    edges = _canonical_edge_pairs(builder.oracle.induced_edges(genes, layer=layer_id))
    possible = len(set(genes)) * (len(set(genes)) - 1) // 2
    return LayerPageStats(
        layer_id=layer_id,
        internal_edges=edges,
        possible_edge_count=possible,
        density=(len(edges) / possible if possible else 0.0),
    )


def _topology_page_pool(builder: Any, required: int) -> list[TopologyPage]:
    sc_layers = [layer for layer in builder.oracle.layer_names if str(layer).startswith("scPEN:")]
    sc_layers = _ordered(sc_layers, builder=builder, namespace="global_scpen_layers")
    if len(sc_layers) < 2:
        raise GlobalFamilyGenerationError("At least two scPEN layers are required for families 61--62.")
    source_layers = sc_layers[: min(24, len(sc_layers))]
    comparison_layers = sc_layers[: min(MAX_CELL_CONTEXTS_PER_PAGE, len(sc_layers))]
    rwr_modules = builder.modules_by_source.get("RWR_LOE_FULL_BRAIN", [])
    module_by_seed = {
        str(module.get("seed_gene_id")): module
        for module in rwr_modules
        if isinstance(module.get("seed_gene_id"), str)
    }
    if not module_by_seed:
        raise GlobalFamilyGenerationError("RWR-LOE seed modules are required for topology pages.")

    per_layer = max(32, math.ceil(required * 4 / len(source_layers)))
    result: list[TopologyPage] = []
    seen: set[str] = set()
    module_gene_sets: dict[str, set[str]] = {}
    for layer_index, source_layer in enumerate(source_layers):
        sample_seed = int(
            _digest(
                seed=builder.seed,
                namespace="global_scpen_edges",
                value={"layer": source_layer, "index": layer_index},
            )[:16],
            16,
        )
        edges = builder.oracle.sample_edges(
            layer=source_layer,
            count=per_layer,
            seed=sample_seed,
        )
        for edge in edges:
            source = str(edge.source_gene_id)
            target = str(edge.target_gene_id)
            module: Mapping[str, Any] | None = None
            anchor = ""
            other = ""
            for candidate_anchor, candidate_other in ((source, target), (target, source)):
                candidate = module_by_seed.get(candidate_anchor)
                if candidate is None:
                    continue
                module_id = str(candidate["module_id"])
                gene_set = module_gene_sets.setdefault(
                    module_id, set(str(gene) for gene in candidate["gene_ids"])
                )
                if candidate_other in gene_set:
                    module = candidate
                    anchor = candidate_anchor
                    other = candidate_other
                    break
            if module is None:
                continue
            module_id = str(module["module_id"])
            remaining = [
                str(gene)
                for gene in module["gene_ids"]
                if str(gene) not in {anchor, other}
            ]
            remaining = _ordered(
                remaining,
                builder=builder,
                namespace=f"topology_page:{module_id}:{source_layer}:{anchor}:{other}",
            )
            genes = tuple(sorted({anchor, other, *remaining[: TOPOLOGY_PAGE_SIZE - 2]}))
            if len(genes) < 3:
                continue
            contexts = [source_layer] + [layer for layer in comparison_layers if layer != source_layer]
            contexts = contexts[:MAX_CELL_CONTEXTS_PER_PAGE]
            stats = [_layer_stats(builder, genes, layer) for layer in contexts]
            stats.sort(key=lambda item: (-item.density, -item.edge_count, item.layer_id))
            higher = stats[0]
            lower = min(stats, key=lambda item: (item.density, item.edge_count, item.layer_id))
            if higher.edge_count == 0 or higher.density <= lower.density:
                continue
            fixture = TopologyPage(
                module_id=module_id,
                module_source="rwr_loe",
                module_gene_count=len(module["gene_ids"]),
                gene_ids=genes,
                higher=higher,
                lower=lower,
            )
            if fixture.page_id in seen:
                continue
            seen.add(fixture.page_id)
            result.append(fixture)
            if len(result) >= required:
                return result
    raise GlobalFamilyGenerationError(
        f"Only {len(result)} real layer-contrast module pages were available; {required} required."
    )


def _topology_common(builder: Any, page: TopologyPage) -> dict[str, Any]:
    return {
        "strongest_group_id": f"module:{page.module_id}",
        "module_source": page.module_source,
        "context_budget_profile": "evidence_2k",
        "evidence_handles": [f"full_store_module_layer_page:{page.page_id}"],
        "provenance": builder._base_provenance("full_store_csr_and_mixed_module_oracle"),
        "coverage": {
            "canonical_genes": list(page.gene_ids),
            "rwr_loe_modules": [page.module_id],
            "layers": [page.higher.layer_id, page.lower.layer_id],
            "gene_layer_pairs": [
                f"{gene}|{layer}"
                for gene in page.gene_ids
                for layer in (page.higher.layer_id, page.lower.layer_id)
            ],
        },
        "page": page.metadata_page(),
    }


def _topology_evidence(page: TopologyPage) -> dict[str, Any]:
    return {
        "module_id": page.module_id,
        "module_source": page.module_source,
        "page": page.page_payload(),
        "gene_ids_page": list(page.gene_ids),
        "layer_summaries": [page.higher.evidence_payload(), page.lower.evidence_payload()],
    }


def _emit_density(builder: Any, example_cls: Any, pages: Sequence[TopologyPage], goal: int) -> None:
    for page in pages[:goal]:
        stats = page.higher
        answer = {
            "module_id": page.module_id,
            "aggregation_scope": "edge_anchored_bounded_module_page",
            "layer_id": stats.layer_id,
            "node_count": len(page.gene_ids),
            "edge_count": stats.edge_count,
            "possible_edge_count": stats.possible_edge_count,
            "edge_density": stats.density,
            "density_definition": "undirected_internal_edges / (node_count * (node_count - 1) / 2)",
        }
        common = _topology_common(builder, page)
        common["layer_scope"] = "single_layer"
        common["layer_ids"] = [stats.layer_id]
        common["layer_families"] = [_layer_family(stats.layer_id)]
        builder.add(
            example_cls(
                family=builder.family("subgraph_density"),
                book_mode="open_book",
                task=(
                    "Compute undirected edge density for the supplied bounded module page in the declared "
                    "layer. Do not report it as complete-module density. Return JSON."
                ),
                answer=answer,
                evidence={
                    "module_id": page.module_id,
                    "page": page.page_payload(),
                    "gene_ids_page": list(page.gene_ids),
                    "layer_summary": stats.evidence_payload(),
                },
                fact_payload=answer,
                validator={"type": "recompute_undirected_density"},
                **common,
            )
        )


def _emit_boundary_ratio(builder: Any, example_cls: Any, pages: Sequence[TopologyPage], goal: int) -> None:
    for page in pages[:goal]:
        stats = page.higher
        degrees = [
            {"gene_id": gene, "layer_degree": int(builder.oracle.degree(gene, layer=stats.layer_id))}
            for gene in page.gene_ids
        ]
        degree_sum = sum(row["layer_degree"] for row in degrees)
        boundary_count = degree_sum - 2 * stats.edge_count
        if boundary_count < 0:
            raise GlobalFamilyGenerationError("Full-store degree and induced-edge counts are inconsistent.")
        denominator = stats.edge_count + boundary_count
        ratio = boundary_count / denominator if denominator else 0.0
        evidence = {
            "module_id": page.module_id,
            "page": page.page_payload(),
            "gene_ids_page": list(page.gene_ids),
            "layer_id": stats.layer_id,
            "internal_edges": stats.evidence_payload()["internal_edges"],
            "per_gene_layer_degrees": degrees,
            "boundary_count_rule": "sum(page_gene_degrees) - 2 * internal_edge_count",
            "boundary_ratio_rule": "boundary_edge_count / (internal_edge_count + boundary_edge_count)",
        }
        answer = {
            "module_id": page.module_id,
            "aggregation_scope": "edge_anchored_bounded_module_page",
            "layer_id": stats.layer_id,
            "internal_edge_count": stats.edge_count,
            "boundary_edge_count": boundary_count,
            "boundary_ratio": ratio,
            "separation": "requires_same_size_null_for_calibration",
            "caveat": "This exact page-level boundary ratio is not a calibrated complete-module conductance claim.",
        }
        common = _topology_common(builder, page)
        common["layer_scope"] = "single_layer"
        common["layer_ids"] = [stats.layer_id]
        common["layer_families"] = [_layer_family(stats.layer_id)]
        builder.add(
            example_cls(
                family=builder.family("conductance_boundary_ratio"),
                book_mode="open_book",
                task=(
                    "Derive the exact boundary-edge count and declared boundary ratio from the supplied "
                    "internal edges and full-layer degrees. Do not claim calibrated conductance. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                validator={"type": "recompute_boundary_ratio_from_degrees"},
                **common,
            )
        )


def _emit_cell_context(builder: Any, example_cls: Any, pages: Sequence[TopologyPage], goal: int) -> None:
    for page in pages[:goal]:
        answer = {
            "module_id": page.module_id,
            "aggregation_scope": "edge_anchored_bounded_module_page",
            "higher_density_context": page.higher.layer_id,
            "higher_density": page.higher.density,
            "lower_density_context": page.lower.layer_id,
            "lower_density": page.lower.density,
            "relative_cohesion": "higher_in_first_scpen_context",
            "allowed_claim": (
                "Within the supplied bounded page, internal edge density is higher in the first scPEN context."
            ),
            "disallowed_claim": (
                "Do not infer universal brain-wide cohesion or cell-type-specific biological causality."
            ),
        }
        common = _topology_common(builder, page)
        common["layer_scope"] = "layer_subset"
        common["layer_ids"] = [page.higher.layer_id, page.lower.layer_id]
        common["layer_families"] = ["sc_pen"]
        builder.add(
            example_cls(
                family=builder.family("cell_type_specific_cohesion"),
                book_mode="open_book",
                task=(
                    "Compare exact internal-edge density between the two supplied scPEN contexts and state "
                    "only the supported page-level topological claim. Return JSON."
                ),
                answer=answer,
                evidence=_topology_evidence(page),
                fact_payload=answer,
                validator={"type": "compare_layer_specific_density"},
                **common,
            )
        )


def _emit_layer_ablation(builder: Any, example_cls: Any, pages: Sequence[TopologyPage], goal: int) -> None:
    for page in pages[:goal]:
        higher_edges = set(page.higher.internal_edges)
        lower_edges = set(page.lower.internal_edges)
        baseline_edges = higher_edges | lower_edges
        remaining_edges = lower_edges
        possible = page.higher.possible_edge_count
        baseline_density = len(baseline_edges) / possible if possible else 0.0
        remaining_density = len(remaining_edges) / possible if possible else 0.0
        effect = "cohesion_decreases" if remaining_density < baseline_density else "no_density_change"
        evidence = {
            "module_id": page.module_id,
            "page": page.page_payload(),
            "gene_ids_page": list(page.gene_ids),
            "layer_subset": [page.higher.evidence_payload(), page.lower.evidence_payload()],
            "ablation_rule": (
                "Take the union of undirected internal edges across both supplied layers, then remove all "
                "support unique to the ablated layer."
            ),
        }
        answer = {
            "module_id": page.module_id,
            "aggregation_scope": "supplied_two_layer_bounded_module_page",
            "ablated_layer": page.higher.layer_id,
            "remaining_layer": page.lower.layer_id,
            "baseline_unique_internal_edge_count": len(baseline_edges),
            "post_ablation_unique_internal_edge_count": len(remaining_edges),
            "baseline_density": baseline_density,
            "post_ablation_density": remaining_density,
            "effect": effect,
            "caveat": "The page-level cohesion is sensitive to this layer within the supplied two-layer scope.",
        }
        common = _topology_common(builder, page)
        common["layer_scope"] = "layer_subset"
        common["layer_ids"] = [page.higher.layer_id, page.lower.layer_id]
        common["layer_families"] = ["sc_pen"]
        builder.add(
            example_cls(
                family=builder.family("layer_sensitive_cohesion"),
                book_mode="open_book",
                task=(
                    "Apply the supplied two-layer edge-union ablation rule and report the exact density "
                    "change with its bounded-scope caveat. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                validator={"type": "recompute_two_layer_edge_ablation"},
                **common,
            )
        )


def _cross_page_mean(shard: Any, left: ModuleDistancePage, right: ModuleDistancePage) -> tuple[float, int]:
    values = [
        float(shard.distance(gene_a, gene_b))
        for gene_a in left.gene_ids
        for gene_b in right.gene_ids
    ]
    return _mean(values), len(values)


def _emit_nearest_modules(
    builder: Any,
    example_cls: Any,
    pages: Sequence[ModuleDistancePage],
    goal: int,
) -> None:
    by_shard: dict[str, list[ModuleDistancePage]] = defaultdict(list)
    for page in pages:
        by_shard[page.shard_id].append(page)
    emitted = 0
    for query in pages:
        candidate_pool = [
            page
            for page in by_shard[query.shard_id]
            if page.module_id != query.module_id
        ]
        candidate_pool = _ordered(
            candidate_pool,
            builder=builder,
            namespace=f"nearest_candidates:{query.page_id}",
        )
        candidates = candidate_pool[:MODULE_NEIGHBOR_COUNT]
        if len(candidates) < 2:
            continue
        shard = builder.rwr.distance_shard(query.shard_id)
        table: list[dict[str, Any]] = []
        for candidate in candidates:
            distance, pair_count = _cross_page_mean(shard, query, candidate)
            table.append(
                {
                    "module_id": candidate.module_id,
                    "module_source": candidate.module_source,
                    "candidate_page": candidate.page_payload(),
                    "candidate_gene_ids_page": list(candidate.gene_ids),
                    "cross_pair_count": pair_count,
                    "distance": distance,
                }
            )
        ranked = sorted(table, key=lambda row: (row["distance"], row["module_id"]))
        evidence = {
            "query_module": query.module_id,
            "query_module_source": query.module_source,
            "query_page": query.page_payload(),
            "query_gene_ids_page": list(query.gene_ids),
            "distance_shard_id": query.shard_id,
            "distance_metric": "bounded_page_mean_cross_spearman_distance",
            "lower_is_closer": True,
            "retrieval_scope": "supplied_same_shard_candidate_table",
            "candidate_table": table,
        }
        answer = {
            "query_module": query.module_id,
            "retrieval_scope": "supplied_same_shard_candidate_table",
            "ranking_metric": "bounded_page_mean_cross_spearman_distance",
            "lower_is_closer": True,
            "top_k": len(ranked),
            "nearest_modules": [
                {"module_id": row["module_id"], "distance": row["distance"]}
                for row in ranked
            ],
            "caveat": "This ranking is exact for the supplied bounded candidate pages, not all modules.",
        }
        module_ids = [query.module_id] + [str(row["module_id"]) for row in table]
        mentor_ids = [
            page.module_id
            for page in [query, *candidates]
            if page.module_source == "mentor_ev"
        ]
        rwr_ids = [
            page.module_id
            for page in [query, *candidates]
            if page.module_source == "rwr_loe"
        ]
        coverage: dict[str, list[str]] = {
            "canonical_genes": sorted(
                set(query.gene_ids).union(*(set(page.gene_ids) for page in candidates))
            ),
            "rwr_distance_shards": [query.shard_id],
            "module_relations": [f"bounded_distance_ranking|{query.module_id}|{module_id}" for module_id in module_ids[1:]],
        }
        if mentor_ids:
            coverage["mentor_ev_modules"] = mentor_ids
        if rwr_ids:
            coverage["rwr_loe_modules"] = rwr_ids
        builder.add(
            example_cls(
                family=builder.family("nearest_modules"),
                book_mode="open_book",
                task=(
                    "Sort the supplied bounded same-shard module-page distance table using lower-is-closer. "
                    "Do not claim an exhaustive all-module search. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                strongest_group_id=f"module:{query.module_id}",
                module_source="mixed" if mentor_ids and rwr_ids else query.module_source,
                context_budget_profile="evidence_2k",
                evidence_handles=[f"bounded_module_distance_ranking:{query.page_id}"],
                provenance=_rwr_provenance(builder, "rwr_distance_shard_and_module_oracle"),
                coverage=coverage,
                validator={"type": "sort_bounded_module_distance_table"},
                page=query.metadata_page(),
            )
        )
        emitted += 1
        if emitted >= goal:
            return
    raise GlobalFamilyGenerationError(f"nearest_modules emitted {emitted} examples; required {goal}.")


def _distance_rows(shard: Any, gene_id: str) -> list[Any]:
    rows = list(shard.row(gene_id))
    return sorted(rows, key=lambda row: (float(row.distance), str(row.gene_id)))


def _coverage_bucket(fraction: float) -> str:
    if fraction < 0.10:
        return "narrow_layer_coverage"
    if fraction < 0.50:
        return "moderate_layer_coverage"
    return "broad_layer_coverage"


def _emit_whole_multiplex_profiles(
    builder: Any,
    example_cls: Any,
    pages: Sequence[ModuleDistancePage],
    goal: int,
) -> None:
    shard_ids = _ordered(
        {page.shard_id for page in pages},
        builder=builder,
        namespace="profile_shards",
    )
    candidates: list[tuple[str, str]] = []
    for shard_id in shard_ids:
        shard = builder.rwr.distance_shard(shard_id)
        genes = _ordered(shard.genes, builder=builder, namespace=f"profile_genes:{shard_id}")
        candidates.extend((shard_id, str(gene)) for gene in genes)
    candidates = _ordered(candidates, builder=builder, namespace="whole_multiplex_profiles")

    emitted = 0
    for shard_id, gene_id in candidates:
        shard = builder.rwr.distance_shard(shard_id)
        rows = _distance_rows(shard, gene_id)
        if not rows:
            continue
        try:
            seed_metadata = builder.rwr.seed_metadata(gene_id)
            layer_ids = sorted(builder.oracle.gene_layers(gene_id))
            aggregate_degree = int(builder.oracle.degree(gene_id, layer=None))
        except (KeyError, ValueError):
            continue
        nearest = rows[0]
        median = rows[len(rows) // 2]
        farthest = rows[-1]
        total_layers = int(builder.oracle.layer_count)
        coverage_fraction = len(layer_ids) / total_layers if total_layers else 0.0
        bucket = _coverage_bucket(coverage_fraction)
        layer_page = layer_ids[:LAYER_ID_PAGE_SIZE]
        layer_page_count = max(1, math.ceil(len(layer_ids) / LAYER_ID_PAGE_SIZE))
        evidence = {
            "gene_id": gene_id,
            "multiplex_id": builder.multiplex_id,
            "full_store_summary": {
                "aggregate_degree": aggregate_degree,
                "layer_coverage_count": len(layer_ids),
                "multiplex_layer_count": total_layers,
                "layer_coverage_fraction": coverage_fraction,
                "layer_ids_page": layer_page,
                "layer_page": {
                    "page_index": 0,
                    "page_count": layer_page_count,
                    "page_size": len(layer_page),
                    "total_items": len(layer_ids),
                },
            },
            "rwr_rank_cache_summary": {
                "ranked_gene_count": int(seed_metadata.ranked_gene_count),
                "ranking_semantics": builder.rwr.identity.ranking_semantics,
            },
            "materialized_distance_shard_summary": {
                "distance_shard_id": shard_id,
                "distance_metric": "spearman_distance",
                "comparison_gene_count": len(rows),
                "nearest": {"gene_id": nearest.gene_id, "distance": float(nearest.distance)},
                "median": {"gene_id": median.gene_id, "distance": float(median.distance)},
                "farthest": {"gene_id": farthest.gene_id, "distance": float(farthest.distance)},
                "scope": "materialized_same_shard_seed_encodings",
            },
            "layer_coverage_bucket_rules": {
                "narrow_layer_coverage": "fraction < 0.10",
                "moderate_layer_coverage": "0.10 <= fraction < 0.50",
                "broad_layer_coverage": "fraction >= 0.50",
            },
        }
        answer = {
            "gene_id": gene_id,
            "multiplex_id": builder.multiplex_id,
            "aggregate_degree": aggregate_degree,
            "full_ranked_gene_count": int(seed_metadata.ranked_gene_count),
            "nearest_distance_bucket": "nearest_in_materialized_same_shard_context",
            "median_distance_bucket": "middle_of_materialized_same_shard_context",
            "farthest_distance_bucket": "farthest_in_materialized_same_shard_context",
            "layer_coverage_count": len(layer_ids),
            "multiplex_layer_count": total_layers,
            "layer_coverage_fraction": coverage_fraction,
            "layer_coverage_bucket": bucket,
            "global_profile": f"{bucket}_with_exact_aggregate_degree_and_bounded_rwr_distance_context",
            "scope_note": (
                "Layer coverage and aggregate degree use the full multiplex; distance extrema use the declared "
                "materialized same-shard seed context."
            ),
        }
        builder.add(
            example_cls(
                family=builder.family("whole_multiplex_context_profile"),
                book_mode="open_book",
                task=(
                    "Summarize the supplied gene profile using the declared full-multiplex graph facts and "
                    "bounded same-shard RWR distance scope. Preserve those scope distinctions. Return JSON."
                ),
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                strongest_group_id=f"global_gene_profile:{gene_id}",
                layer_scope="all_layers",
                layer_ids=layer_page,
                layer_families=sorted({_layer_family(layer) for layer in layer_ids}),
                module_source="none",
                context_budget_profile="evidence_2k",
                evidence_handles=[f"whole_multiplex_profile:{gene_id}:{shard_id}"],
                provenance=_rwr_provenance(builder, "full_store_csr_and_rwr_summary_oracle"),
                coverage={
                    "canonical_genes": [gene_id, str(nearest.gene_id), str(median.gene_id), str(farthest.gene_id)],
                    "layers": layer_page,
                    "rwr_seed_sets": [gene_id],
                    "rwr_distance_shards": [shard_id],
                },
                validator={"type": "extract_scoped_whole_multiplex_profile"},
                page={
                    "index": 0,
                    "count": layer_page_count,
                    "size": len(layer_page),
                    "total_items": len(layer_ids),
                    "scope": "full_store_gene_layer_membership_page",
                },
            )
        )
        emitted += 1
        if emitted >= goal:
            return
    raise GlobalFamilyGenerationError(
        f"whole_multiplex_context_profile emitted {emitted} examples; required {goal}."
    )


def generate_global_families(builder: Any) -> dict[str, int]:
    """Generate curriculum families 56--63 and 81 into ``builder``.

    Integration in ``build_pretrajectory_sft_curriculum.py`` is intentionally
    one import and one call::

        from scripts.pretrajectory_curriculum_global_families import generate_global_families
        ...
        generate_global_families(builder)

    Call it after ``builder.load_sources()`` and the ordinary stage-4 module
    generators, before selection/rendering.
    """

    if builder.rwr is None:
        raise GlobalFamilyGenerationError("builder.load_sources() must run before global generation.")
    missing_sources = {
        "MENTOR_GW_DENDROGRAM",
        "RWR_LOE_FULL_BRAIN",
    } - set(builder.modules_by_source)
    if missing_sources:
        raise GlobalFamilyGenerationError(
            f"Global generation is missing module sources: {sorted(missing_sources)}."
        )

    example_cls = _example_type(builder)
    goals = {name: int(builder.candidate_goal(name)) for name in GLOBAL_FAMILY_NAMES}
    if any(goal <= 0 for goal in goals.values()):
        raise GlobalFamilyGenerationError("Every required global family must have a positive candidate goal.")

    distance_required = max(
        goals["within_clade_distance"],
        goals["within_clade_vs_random"],
        goals["clustering_ratio"],
        goals["nearest_modules"] + MODULE_NEIGHBOR_COUNT + 16,
    )
    distance_pages = _distance_page_pool(builder, distance_required)
    within_facts = _within_facts(builder, distance_pages)

    _emit_within_distance(
        builder,
        example_cls,
        within_facts,
        goals["within_clade_distance"],
    )
    _emit_null_comparison(
        builder,
        example_cls,
        within_facts,
        goals["within_clade_vs_random"],
    )
    _emit_clustering_ratio(
        builder,
        example_cls,
        within_facts,
        goals["clustering_ratio"],
    )

    topology_goal = max(
        goals["subgraph_density"],
        goals["conductance_boundary_ratio"],
        goals["cell_type_specific_cohesion"],
        goals["layer_sensitive_cohesion"],
    )
    topology_pages = _topology_page_pool(builder, topology_goal)
    _emit_density(builder, example_cls, topology_pages, goals["subgraph_density"])
    _emit_boundary_ratio(
        builder,
        example_cls,
        topology_pages,
        goals["conductance_boundary_ratio"],
    )
    _emit_cell_context(
        builder,
        example_cls,
        topology_pages,
        goals["cell_type_specific_cohesion"],
    )
    _emit_layer_ablation(
        builder,
        example_cls,
        topology_pages,
        goals["layer_sensitive_cohesion"],
    )
    _emit_nearest_modules(
        builder,
        example_cls,
        distance_pages,
        goals["nearest_modules"],
    )
    _emit_whole_multiplex_profiles(
        builder,
        example_cls,
        distance_pages,
        goals["whole_multiplex_context_profile"],
    )

    return dict(goals)


__all__ = [
    "GLOBAL_FAMILY_NAMES",
    "GlobalFamilyGenerationError",
    "generate_global_families",
]
