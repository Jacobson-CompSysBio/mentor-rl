"""Generate stage-5 tool curriculum families from live runtime contracts.

This module is intentionally isolated from the main curriculum builder so the
model-facing tool surface cannot drift into a second collection of handwritten
schemas.  ``generate_tool_families(builder)`` expects a prepared builder whose
full-store oracle, mixed module corpus, and RWR curriculum reader are loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from runtime.schemas import (
    ContinuationState,
    EvidenceRecord,
    EvidenceSourceType,
    GeneGroup,
    RelationshipStatus,
    StructuredState,
    ToolAction,
    UserAnchors,
)
from runtime.tool_curriculum_contract import (
    assert_no_provenance_leakage,
    build_tool_action,
    build_tool_exchange,
    build_tool_observation,
    select_tool_for_intent,
    tool_policy_metadata,
)
from runtime.validators import validate_tool_action_schema


TOOL_FAMILY_NAMES = (
    "choose_rwr_loe_tool",
    "choose_pairwise_distance_tool",
    "choose_layer_membership_tool",
    "choose_component_summary_tool",
    "choose_induced_subgraph_tool",
    "choose_layer_ablation_tool",
    "parse_rwr_loe_result",
    "parse_distance_shard",
    "parse_module_overlap",
    "provenance_answer",
    "refuse_raw_cli_path",
    "structured_state_update",
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ordered(values: Iterable[Any], *, seed: int, namespace: str) -> list[Any]:
    return sorted(
        values,
        key=lambda value: _stable_hash(
            {"seed": int(seed), "namespace": namespace, "value": value}
        ),
    )


def _example_type(builder: Any) -> type:
    """Resolve CurriculumExample without importing a second __main__ module."""

    explicit = getattr(builder, "curriculum_example_type", None)
    if isinstance(explicit, type):
        return explicit
    owner_module = sys.modules.get(type(builder).__module__)
    candidate = getattr(owner_module, "CurriculumExample", None) if owner_module else None
    if isinstance(candidate, type):
        return candidate
    from scripts.build_pretrajectory_sft_curriculum import CurriculumExample

    return CurriculumExample


@dataclass
class _Context:
    builder: Any

    def __post_init__(self) -> None:
        if self.builder.rwr is None:
            raise RuntimeError("Tool curriculum requires the loaded RWR curriculum reader.")
        if not self.builder.modules_by_source.get("MENTOR_GW_DENDROGRAM"):
            raise RuntimeError("Tool curriculum requires MENTOR-EV module evidence.")
        if not self.builder.modules_by_source.get("RWR_LOE_FULL_BRAIN"):
            raise RuntimeError("Tool curriculum requires RWR-LOE module evidence.")
        self.Example = _example_type(self.builder)
        self.gene_ids = set(self.builder.oracle.gene_ids)
        self.layer_ids = set(self.builder.oracle.layer_names)
        cached_seeds = sorted(getattr(self.builder.rwr, "_rank_vectors", {}))
        all_seeds = _ordered(
            self.builder.rwr.seed_gene_ids,
            seed=self.builder.seed,
            namespace="stage5_tool_seed_pool",
        )
        self.all_seeds = all_seeds
        self.seeds = list(dict.fromkeys(cached_seeds + all_seeds))[:8]
        if len(self.seeds) < 2:
            raise RuntimeError("Tool curriculum requires at least two RWR seed vectors.")
        self._vectors: dict[str, Any] = {}
        self._distance_shard: Any | None = None

    def vector_case(self, index: int, *, row_count: int = 5) -> tuple[str, Any, list[Any]]:
        seed = self.seeds[index % len(self.seeds)]
        vector = self._vectors.get(seed)
        if vector is None:
            vector = self.builder.rwr.rank_vector(seed)
            self._vectors[seed] = vector
        if len(vector.rows) < row_count + 8:
            raise RuntimeError(f"RWR vector for {seed} is too small for tool curriculum.")
        span = len(vector.rows) - row_count
        offset = (index * 97 + (index // len(self.seeds)) * 193 + 3) % span
        return seed, vector, list(vector.rows[offset : offset + row_count])

    def distance_case(self, index: int, *, column_count: int = 3) -> tuple[Any, str, list[str]]:
        if self._distance_shard is None:
            cached = getattr(self.builder.rwr, "_distance_shards", {})
            if cached:
                self._distance_shard = cached[sorted(cached)[0]]
            else:
                shard_ids = _ordered(
                    self.builder.rwr.shard_ids,
                    seed=self.builder.seed,
                    namespace="stage5_distance_shard",
                )
                self._distance_shard = self.builder.rwr.distance_shard(shard_ids[0])
        shard = self._distance_shard
        if len(shard.genes) < column_count + 1:
            raise RuntimeError("Distance shard is too small for tool curriculum.")
        anchor = shard.genes[(index * 17 + index // len(shard.genes)) % len(shard.genes)]
        candidates = [gene for gene in shard.genes if gene != anchor]
        start = (index * 31 + (index // len(candidates)) * 47) % len(candidates)
        columns = [candidates[(start + offset * 13) % len(candidates)] for offset in range(column_count)]
        return shard, anchor, columns

    def public_provenance(self, source: str, **extra: Any) -> dict[str, Any]:
        provenance = {
            "source": source,
            "backend": "versioned_curriculum_oracle",
            "implementation": "artifact_replay",
            "multiplex_id": self.builder.multiplex_id,
            "graph_version": self.builder.multiplex_id,
            "store_id": self.builder.store_id,
            "flist_id": self.builder.flist_id,
            "layer_count": self.builder.oracle.layer_count,
            "network_flist_sha256": self.builder.rwr.identity.network_flist_sha256,
        }
        provenance.update(extra)
        return provenance

    def exchange(
        self,
        intent: str,
        arguments: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        source: str,
        provenance: Mapping[str, Any] | None = None,
        empty: bool = False,
    ) -> dict[str, Any]:
        merged_provenance = self.public_provenance(source)
        merged_provenance.update(provenance or {})
        if empty:
            tool_name = select_tool_for_intent(intent)
            action_object = build_tool_action(
                tool_name,
                arguments,
                available_gene_ids=self.gene_ids,
                available_layers=self.layer_ids,
            )
            observation = build_tool_observation(
                action_object,
                payload=payload,
                provenance=merged_provenance,
                status="empty",
            )
            exchange = {
                "tool_action": action_object.to_dict(),
                "tool_observation": observation.to_dict(),
                "tool_policy": tool_policy_metadata(intent, selected_tool=tool_name),
            }
        else:
            exchange = build_tool_exchange(
                intent,
                arguments,
                payload=payload,
                provenance=merged_provenance,
                available_gene_ids=self.gene_ids,
                available_layers=self.layer_ids,
            )
        action = ToolAction.from_dict(exchange["tool_action"])
        validation = validate_tool_action_schema(action)
        if not validation.valid:  # pragma: no cover - build_tool_exchange already checks this
            raise RuntimeError("Invalid tool action: " + "; ".join(validation.errors))
        assert_no_provenance_leakage(exchange)
        return exchange

    def add(
        self,
        family_name: str,
        *,
        task: str,
        answer: dict[str, Any],
        evidence: dict[str, Any],
        exchange: dict[str, Any],
        group_id: str,
        evidence_handle: str,
        source: str,
        coverage: dict[str, list[str]],
        layer_scope: str = "all_layers",
        layer_ids: list[str] | None = None,
        layer_families: list[str] | None = None,
        module_source: str = "none",
        budget: str = "evidence_2k",
        validator: dict[str, Any] | None = None,
        polarity: str = "positive",
    ) -> None:
        assert_no_provenance_leakage(evidence)
        coverage = {key: list(values) for key, values in coverage.items()}
        coverage.setdefault("live_tool_schemas", []).append(
            str(exchange["tool_action"]["tool_name"])
        )
        self.builder.add(
            self.Example(
                family=self.builder.family(family_name),
                book_mode="tool_call",
                task=task,
                answer=answer,
                evidence=evidence,
                fact_payload=answer,
                strongest_group_id=group_id,
                layer_scope=layer_scope,
                layer_ids=layer_ids or [],
                layer_families=layer_families or [],
                module_source=module_source,
                context_budget_profile=budget,
                evidence_handles=[evidence_handle],
                provenance={
                    "source": source,
                    "multiplex_id": self.builder.multiplex_id,
                    "store_id": self.builder.store_id,
                    "flist_id": self.builder.flist_id,
                },
                coverage=coverage,
                validator=validator or {"type": "exact_json"},
                polarity=polarity,
                tool_exchange=exchange,
            )
        )


def _action_answer(exchange: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tool_action": exchange["tool_action"],
        "tool_policy": exchange["tool_policy"],
    }


def _ranked_payload(tool_name: str, seed: str, rows: list[Any], *, top_k: int) -> dict[str, Any]:
    ranked = [
        {
            "gene": row.gene_id,
            "gene_id": row.gene_id,
            "rank": int(row.rank),
            "score": float(row.score),
            "is_seed": False,
        }
        for row in sorted(rows, key=lambda item: (item.rank, item.gene_id))[:top_k]
    ]
    return {
        "tool_name": tool_name,
        "seed_genes": [seed],
        "top_k": top_k,
        "ranked_genes": ranked,
        "results": ranked,
    }


def _generate_rank_summary_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_rwr_loe_tool")
    preexisting = set(getattr(ctx.builder.rwr, "_rank_vectors", {}))
    for index in range(goal):
        variant = index % 8
        seed = ctx.all_seeds[(index // 8) % len(ctx.all_seeds)]
        vector = ctx.builder.rwr.rank_vector(seed)
        top_k = 3 + variant
        rows = list(vector.top_k(top_k, exclude_genes=[seed]))
        arguments = {"seed_genes": [seed], "top_k": top_k, "include_seed_genes": False}
        payload = {
            "tool_name": "get_rank_vector_summary",
            "seed_genes": [seed],
            "top_k": top_k,
            "include_seed_genes": False,
            "rank_summary": [],
            "results": [],
            "ranked_gene_count": int(vector.metadata.ranked_gene_count),
            "result_status": "schema_validated_request_not_materialized",
        }
        exchange = ctx.exchange(
            "rank_vector_summary",
            arguments,
            payload,
            source="runtime_tool_schema",
            empty=True,
        )
        compared = [rows[0].gene_id, rows[1].gene_id]
        answer = {**_action_answer(exchange), "comparison_gene_ids": compared}
        rank_reference = {
            "observation_type": "rwr_loe_rank_reference",
            "seed_gene_id": seed,
            "comparison_rows": [
                {"gene_id": row.gene_id, "rank": row.rank, "score": row.score}
                for row in rows[:2]
            ],
        }
        ctx.add(
            "choose_rwr_loe_tool",
            task=(
                "Choose the cheapest live structured action for comparing the supplied genes in one "
                "seed's cached rank-vector summary. Return the action and policy from the observation."
            ),
            answer=answer,
            evidence={
                "tool_exchange": exchange,
                "comparison_gene_ids": compared,
                "rank_reference_observation": rank_reference,
            },
            exchange=exchange,
            group_id=f"tool_rank_summary:{seed}:{':'.join(compared)}",
            evidence_handle=f"tool_rank_summary:{seed}:{rows[0].rank}:{rows[1].rank}",
            source="rwr_rank_cache",
            coverage={
                "canonical_genes": [seed] + compared,
                "rwr_seed_sets": [seed],
                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in rows[:2]],
            },
            budget="matrix_state_4k",
            validator={"type": "schema_valid_tool_action"},
        )
        if variant == 7 and seed not in preexisting:
            getattr(ctx.builder.rwr, "_rank_vectors", {}).pop(seed, None)


def _generate_distance_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_pairwise_distance_tool")
    for index in range(goal):
        shard, gene_a, columns = ctx.distance_case(index, column_count=1)
        gene_b = columns[0]
        distance = float(shard.distance(gene_a, gene_b))
        arguments = {"gene_a": gene_a, "gene_b": gene_b, "distance_metric": "spearman"}
        payload = {
            "tool_name": "get_distance",
            "gene_a": gene_a,
            "gene_b": gene_b,
            "layers": [],
            "distance_metric": "spearman",
            "distance": distance,
            "dissimilarity": distance,
            "results": [
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "distance_metric": "spearman",
                    "distance": distance,
                }
            ],
        }
        exchange = ctx.exchange(
            "pair_distance",
            arguments,
            payload,
            source="rwr_distance_shard",
            provenance={"distance_type": "spearman_distance"},
        )
        answer = {**_action_answer(exchange), "observed_distance": distance}
        ctx.add(
            "choose_pairwise_distance_tool",
            task="Return the schema-valid action and observed RWR distance for this pair.",
            answer=answer,
            evidence={"tool_exchange": exchange},
            exchange=exchange,
            group_id=f"distance_pair:{min(gene_a, gene_b)}:{max(gene_a, gene_b)}",
            evidence_handle=f"distance_shard:{shard.shard_id}:{gene_a}:{gene_b}",
            source="rwr_distance_shard",
            coverage={"canonical_genes": [gene_a, gene_b]},
            validator={"type": "schema_valid_tool_action_and_float", "float_fields": ["observed_distance"]},
        )


def _generate_layer_membership_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_layer_membership_tool")
    node_indices = ctx.builder.oracle.sample_node_indices(
        layer=None,
        count=goal,
        seed=ctx.builder.seed + 7100,
        minimum_degree=1,
    )
    if len(node_indices) < goal:
        raise RuntimeError("Full-store oracle could not supply enough layer-membership genes.")
    for index, node_index in enumerate(node_indices):
        gene = ctx.builder.oracle.gene_ids[node_index]
        layers = ctx.builder.oracle.gene_layers(gene)
        arguments = {"gene": gene}
        payload = {
            "tool_name": "get_gene_layers",
            "gene": gene,
            "layers": [],
            "results": [],
            "result_status": "schema_validated_request_not_materialized",
        }
        exchange = ctx.exchange(
            "gene_layer_membership",
            arguments,
            payload,
            source="runtime_tool_schema",
            empty=True,
        )
        answer = _action_answer(exchange)
        reference = {
            "observation_type": "bounded_full_store_layer_membership",
            "gene": gene,
            "layer_count": len(layers),
            "layer_page": layers[:12],
            "page_size": 12,
            "has_more": len(layers) > 12,
        }
        ctx.add(
            "choose_layer_membership_tool",
            task="Return the live point-lookup action and exact multiplex layer membership from its observation.",
            answer=answer,
            evidence={"tool_exchange": exchange, "layer_membership_reference": reference},
            exchange=exchange,
            group_id=f"gene_layers:{gene}",
            evidence_handle=f"gene_layers:{gene}",
            source="full_binary_csr_store",
            coverage={"canonical_genes": [gene], "layers": layers},
            validator={"type": "schema_valid_tool_action_and_exact_set"},
        )


def _component_cases(ctx: _Context, goal: int) -> list[tuple[list[str], list[list[str]]]]:
    edges = ctx.builder.oracle.sample_edges(
        layer=None,
        count=goal,
        seed=ctx.builder.seed + 7200,
    )
    cases: list[tuple[list[str], list[list[str]]]] = []
    for edge_index, edge in enumerate(edges):
        neighbors = ctx.builder.oracle.neighbors(edge.source_gene_id, layer=None, limit=12)
        extra = next(
            (
                row["gene_id"]
                for row in neighbors[(edge_index % max(1, len(neighbors))) :] + neighbors[: (edge_index % max(1, len(neighbors)))]
                if row["gene_id"] != edge.target_gene_id
            ),
            None,
        )
        genes = sorted({edge.source_gene_id, edge.target_gene_id, *( [extra] if extra else [] )})
        components = ctx.builder.oracle.induced_components(genes, layer=None)
        cases.append((genes, components))
    if len(cases) < goal:
        raise RuntimeError("Full-store oracle could not supply enough component cases.")
    return cases


def _generate_component_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_component_summary_tool")
    for genes, components in _component_cases(ctx, goal):
        arguments = {"genes": genes, "max_components": 10}
        payload = {
            "tool_name": "get_component_summary",
            "genes": genes,
            "max_components": 10,
            "components": [],
            "results": [],
            "result_status": "schema_validated_request_not_materialized",
        }
        exchange = ctx.exchange(
            "component_summary",
            arguments,
            payload,
            source="runtime_tool_schema",
            empty=True,
        )
        answer = {
            **_action_answer(exchange),
            "component_count": len(components),
            "one_component": len(components) == 1,
        }
        group_hash = _stable_hash(genes)[:16]
        ctx.add(
            "choose_component_summary_tool",
            task="Choose the bounded component-summary action and report whether its query genes are connected.",
            answer=answer,
            evidence={
                "tool_exchange": exchange,
                "component_connectivity_observation": {
                    "scope": "query_induced_connectivity",
                    "genes": genes,
                    "components": components,
                    "component_count": len(components),
                },
            },
            exchange=exchange,
            group_id=f"component_query:{group_hash}",
            evidence_handle=f"component_query:{group_hash}",
            source="full_binary_csr_store",
            coverage={"canonical_genes": genes},
            budget="matrix_state_4k",
            validator={"type": "schema_valid_tool_action_and_component_count"},
        )


def _subgraph_cases(ctx: _Context, goal: int) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    ordered_layers = _ordered(
        ctx.builder.oracle.layer_names,
        seed=ctx.builder.seed,
        namespace="stage5_subgraph_layers",
    )
    layer_rows = getattr(ctx.builder.oracle, "layer_to_row", {})
    layer_pool = sorted(
        ordered_layers,
        key=lambda layer: -int(layer_rows.get(layer, {}).get("undirected_edge_count", 0)),
    )[: min(64, ctx.builder.oracle.layer_count)]
    per_layer = max(1, math.ceil(goal / len(layer_pool)))
    cases: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for layer_index, layer in enumerate(layer_pool):
        edges = ctx.builder.oracle.sample_edges(
            layer=layer,
            count=per_layer,
            seed=ctx.builder.seed + 7300 + layer_index,
        )
        for edge_index, edge in enumerate(edges):
            neighbors = ctx.builder.oracle.neighbors(edge.source_gene_id, layer=layer, limit=10)
            extras = [
                row["gene_id"]
                for row in neighbors
                if row["gene_id"] != edge.target_gene_id
            ]
            extra = extras[edge_index % len(extras)] if extras else None
            genes = sorted({edge.source_gene_id, edge.target_gene_id, *( [extra] if extra else [] )})
            induced = [fact.as_dict() for fact in ctx.builder.oracle.induced_edges(genes, layer=layer)]
            cases.append((layer, genes, induced))
            if len(cases) >= goal:
                return cases
    if len(cases) < goal:
        raise RuntimeError("Full-store oracle could not supply enough induced-subgraph cases.")
    return cases


def _generate_subgraph_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_induced_subgraph_tool")
    for layer, genes, edges in _subgraph_cases(ctx, goal):
        arguments = {"genes": genes, "layers": [layer]}
        payload = {
            "tool_name": "induce_subgraph",
            "query_gene_ids": genes,
            "present_gene_ids": genes,
            "missing_gene_ids": [],
            "layers": [
                {
                    "layer_name": layer,
                    "present_gene_ids": genes,
                    "edges": edges,
                    "edge_count": len(edges),
                }
            ],
            "combined_edge_count": len(edges),
        }
        exchange = ctx.exchange(
            "induced_subgraph",
            arguments,
            payload,
            source="full_binary_csr_store",
            provenance={"layer_name": layer, "queried_layers": [layer]},
        )
        answer = {
            **_action_answer(exchange),
            "combined_edge_count": len(edges),
            "edges": edges,
        }
        group_hash = _stable_hash([layer, genes])[:16]
        ctx.add(
            "choose_induced_subgraph_tool",
            task="Return the exact live action and all observed intra-set edges in the requested layer.",
            answer=answer,
            evidence={"tool_exchange": exchange},
            exchange=exchange,
            group_id=f"induced_subgraph:{group_hash}",
            evidence_handle=f"induced_subgraph:{group_hash}",
            source="full_binary_csr_store",
            coverage={"canonical_genes": genes, "layers": [layer]},
            layer_scope="single_layer",
            layer_ids=[layer],
            layer_families=[layer.split(":", 1)[0]],
            budget="matrix_state_4k",
            validator={"type": "schema_valid_tool_action_and_exact_edges"},
        )


def _generate_ablation_selection(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("choose_layer_ablation_tool")
    for index in range(goal):
        seed_a = ctx.all_seeds[index % len(ctx.all_seeds)]
        seed_b = ctx.all_seeds[(index * 17 + 1 + index // len(ctx.all_seeds)) % len(ctx.all_seeds)]
        if seed_b == seed_a:
            seed_b = ctx.all_seeds[(index + 1) % len(ctx.all_seeds)]
        seed_genes = sorted({seed_a, seed_b})
        top_k = (20, 50, 100)[(index // len(ctx.all_seeds)) % 3]
        arguments = {
            "seed_genes": seed_genes,
            "distance_metric": "spearman",
            "top_k": top_k,
        }
        # No precomputed layer-ablation artifact is available in the curriculum
        # sources.  This observation therefore exposes only the validated request
        # and the exact layer universe; it never fabricates ablation effects.
        payload = {
            "tool_name": "get_layer_ablation",
            "seed_genes": seed_genes,
            "distance_metric": "spearman",
            "top_k": top_k,
            "layer_count": ctx.builder.oracle.layer_count,
            "result_status": "schema_validated_request_not_materialized",
            "layer_effects": [],
            "results": [],
        }
        exchange = ctx.exchange(
            "layer_ablation",
            arguments,
            payload,
            source="full_binary_csr_store_and_rwr_schema",
            empty=True,
        )
        answer = {
            **_action_answer(exchange),
            "result_status": payload["result_status"],
            "layer_count": payload["layer_count"],
        }
        ctx.add(
            "choose_layer_ablation_tool",
            task=(
                "Choose the live layer-ablation action. Do not invent a layer selector or ablation values "
                "that are absent from the observation."
            ),
            answer=answer,
            evidence={"tool_exchange": exchange},
            exchange=exchange,
            group_id=f"layer_ablation_request:{':'.join(seed_genes)}:{top_k}",
            evidence_handle=f"layer_ablation_request:{':'.join(seed_genes)}:{top_k}",
            source="full_binary_csr_store_and_rwr_schema",
            coverage={"canonical_genes": seed_genes, "rwr_seed_sets": seed_genes},
            budget="matrix_state_4k",
            validator={"type": "schema_valid_tool_action_no_fabricated_result"},
        )


def _generate_rwr_parse(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("parse_rwr_loe_result")
    for index in range(goal):
        seed, _vector, rows = ctx.vector_case(index + 1000, row_count=7)
        query_rows = [rows[position] for position in (0, 2, 4, 6)]
        query_genes = [row.gene_id for row in query_rows]
        ranked = sorted(query_rows, key=lambda row: (row.rank, row.gene_id))
        arguments = {
            "seed_genes": [seed],
            "query_genes": query_genes,
            "top_k": len(query_genes),
            "exclude_seed_genes": True,
        }
        payload = _ranked_payload("rwr_loe", seed, ranked, top_k=len(query_genes))
        payload.update({"query_genes": query_genes, "exclude_seed_genes": True})
        exchange = ctx.exchange(
            "query_filtered_ranking",
            arguments,
            payload,
            source="rwr_rank_cache",
        )
        closest = payload["ranked_genes"][:3]
        answer = {"seed_gene_ids": [seed], "closest_non_seed_genes": closest}
        query_hash = _stable_hash([seed, query_genes])[:16]
        ctx.add(
            "parse_rwr_loe_result",
            task="Parse the tool observation and return its three lowest-rank non-seed results in order.",
            answer=answer,
            evidence={"tool_exchange": exchange},
            exchange=exchange,
            group_id=f"rwr_query:{query_hash}",
            evidence_handle=f"rwr_query:{query_hash}",
            source="rwr_rank_cache",
            coverage={
                "canonical_genes": [seed] + query_genes,
                "rwr_seed_sets": [seed],
                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in query_rows],
            },
            validator={"type": "top_k", "k": 3, "float_fields": ["score"]},
        )


def _generate_distance_parse(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("parse_distance_shard")
    for index in range(goal):
        shard, anchor, columns = ctx.distance_case(index + 2000, column_count=3)
        distance_rows = [
            {"column_gene_id": gene, "distance": float(shard.distance(anchor, gene))}
            for gene in columns
        ]
        first = distance_rows[0]
        arguments = {
            "gene_a": anchor,
            "gene_b": first["column_gene_id"],
            "distance_metric": "spearman",
        }
        scalar_payload = {
            "tool_name": "get_distance",
            "gene_a": anchor,
            "gene_b": first["column_gene_id"],
            "layers": [],
            "distance_metric": "spearman",
            "distance": first["distance"],
            "dissimilarity": first["distance"],
            "results": [{**arguments, "distance": first["distance"]}],
        }
        exchange = ctx.exchange(
            "pair_distance",
            arguments,
            scalar_payload,
            source="rwr_distance_shard",
            provenance={"distance_type": "spearman_distance"},
        )
        bounded_observation = {
            "observation_type": "bounded_distance_matrix_row",
            "shard_id": shard.shard_id,
            "row_gene_id": anchor,
            "distance_metric": "spearman_distance",
            "cells": distance_rows,
        }
        answer = {
            "row_gene_id": anchor,
            "distance_metric": "spearman_distance",
            "distances": distance_rows,
        }
        group_hash = _stable_hash([anchor, columns])[:16]
        ctx.add(
            "parse_distance_shard",
            task="Extract every listed distance from the bounded matrix-row observation. Return them in listed order.",
            answer=answer,
            evidence={
                "tool_exchange": exchange,
                "bounded_distance_observation": bounded_observation,
            },
            exchange=exchange,
            group_id=f"distance_row:{group_hash}",
            evidence_handle=f"distance_row:{shard.shard_id}:{group_hash}",
            source="rwr_distance_shard",
            coverage={"canonical_genes": [anchor] + columns},
            budget="matrix_state_4k",
            validator={"type": "ordered_matrix_row_extraction", "float_fields": ["distance"]},
        )


def _generate_module_parse(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("parse_module_overlap")
    mentors = ctx.builder.modules_by_source["MENTOR_GW_DENDROGRAM"]
    rwr_modules = ctx.builder.modules_by_source["RWR_LOE_FULL_BRAIN"]
    ordered_mentors = _ordered(
        [module for module in mentors if any(gene in ctx.gene_ids for gene in module["gene_ids"])],
        seed=ctx.builder.seed,
        namespace="stage5_overlap_mentor",
    )
    ordered_rwr = _ordered(
        rwr_modules,
        seed=ctx.builder.seed,
        namespace="stage5_overlap_rwr",
    )
    for index in range(goal):
        mentor = ordered_mentors[index % len(ordered_mentors)]
        mentor_genes = set(mentor["gene_ids"])
        candidates = [
            ordered_rwr[(index * 7 + offset * 17) % len(ordered_rwr)]
            for offset in range(4)
        ]
        table: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_genes = set(candidate["gene_ids"])
            contains = mentor_genes.issubset(candidate_genes)
            table.append(
                {
                    "module_id": candidate["module_id"],
                    "contains_query_module": contains,
                    "intersection_size": len(mentor_genes & candidate_genes),
                    "query_gene_count": len(mentor_genes),
                    "extra_gene_count": len(candidate_genes - mentor_genes) if contains else None,
                }
            )
        query_genes = [gene for gene in mentor["gene_ids"] if gene in ctx.gene_ids][:12]
        components = ctx.builder.oracle.induced_components(query_genes, layer=None)
        arguments = {"genes": query_genes, "max_components": 10}
        exchange = ctx.exchange(
            "component_summary",
            arguments,
            {
                "tool_name": "get_component_summary",
                "genes": query_genes,
                "max_components": 10,
                "components": [],
                "results": [],
                "result_status": "schema_validated_request_not_materialized",
            },
            source="runtime_tool_schema",
            empty=True,
        )
        supersets = [
            {
                "module_id": row["module_id"],
                "contains_query_module": True,
                "extra_gene_count": row["extra_gene_count"],
            }
            for row in table
            if row["contains_query_module"]
        ]
        answer = {"query_module": mentor["module_id"], "superset_modules": supersets}
        group_hash = _stable_hash([mentor["module_id"], [row["module_id"] for row in table]])[:16]
        ctx.add(
            "parse_module_overlap",
            task="Identify every candidate marked as a complete superset in the exact module-overlap observation.",
            answer=answer,
            evidence={
                "tool_exchange": exchange,
                "module_overlap_observation": {
                    "query_module": mentor["module_id"],
                    "query_gene_count": len(mentor_genes),
                    "candidate_rows": table,
                },
            },
            exchange=exchange,
            group_id=f"module_overlap_table:{group_hash}",
            evidence_handle=f"module_overlap_table:{group_hash}",
            source="mixed_module_oracle",
            coverage={
                "canonical_genes": query_genes,
                "mentor_ev_modules": [str(mentor["module_id"])],
                "rwr_loe_modules": [str(row["module_id"]) for row in table],
            },
            module_source="mixed",
            budget="matrix_state_4k",
            validator={"type": "module_superset_filter"},
        )


def _generate_provenance_answers(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("provenance_answer")
    for index in range(goal):
        seed, _vector, rows = ctx.vector_case(index + 3000, row_count=1)
        row = rows[0]
        evidence_id = f"ev_{_stable_hash([seed, row.gene_id, row.rank])[:16]}"
        arguments = {
            "seed_genes": [seed],
            "query_genes": [row.gene_id],
            "top_k": 1,
            "exclude_seed_genes": True,
        }
        payload = _ranked_payload("rwr_loe", seed, [row], top_k=1)
        payload.update({
            "query_genes": [row.gene_id],
            "exclude_seed_genes": True,
            "evidence_id": evidence_id,
            "layer_scope": "all_layers",
            "multiplex_id": ctx.builder.multiplex_id,
            "provenance_complete": True,
        })
        exchange = ctx.exchange(
            "query_filtered_ranking",
            arguments,
            payload,
            source="rwr_rank_cache",
        )
        observation = exchange["tool_observation"]
        answer = {
            "tool_name": exchange["tool_action"]["tool_name"],
            "multiplex_id": observation["payload"]["multiplex_id"],
            "layer_scope": observation["payload"]["layer_scope"],
            "evidence_id": observation["payload"]["evidence_id"],
            "provenance_complete": observation["payload"]["provenance_complete"],
        }
        ctx.add(
            "provenance_answer",
            task="Return the tool, multiplex, layer scope, and evidence identifier carried by this observation.",
            answer=answer,
            evidence={"tool_exchange": exchange},
            exchange=exchange,
            group_id=f"provenance:{evidence_id}",
            evidence_handle=f"provenance:{evidence_id}",
            source="rwr_rank_cache",
            coverage={
                "canonical_genes": [seed, row.gene_id],
                "rwr_seed_sets": [seed],
                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}"],
            },
            validator={"type": "provenance_projection"},
        )


def _generate_cli_refusals(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("refuse_raw_cli_path")
    rejected_names = ["seed_file", "query_file", "output_dir"]
    for index in range(goal):
        seed, _vector, rows = ctx.vector_case(index + 4000, row_count=3)
        query_genes = [row.gene_id for row in rows]
        arguments = {
            "seed_genes": [seed],
            "query_genes": query_genes,
            "top_k": len(query_genes),
            "exclude_seed_genes": True,
        }
        payload = _ranked_payload("rwr_loe", seed, rows, top_k=len(rows))
        payload.update({"query_genes": query_genes, "exclude_seed_genes": True})
        exchange = ctx.exchange(
            "query_filtered_ranking",
            arguments,
            payload,
            source="rwr_rank_cache",
        )
        contract_observation = {
            "accepted": False,
            "reason_code": "structured_biological_arguments_required",
            "rejected_argument_names": rejected_names,
            "corrected_tool_action": exchange["tool_action"],
        }
        answer = dict(contract_observation)
        group_hash = _stable_hash([seed, query_genes])[:16]
        ctx.add(
            "refuse_raw_cli_path",
            task=(
                "Reject the named low-level execution fields and return the corrected structured biological "
                "action from the contract observation."
            ),
            answer=answer,
            evidence={"tool_exchange": exchange, "request_contract_observation": contract_observation},
            exchange=exchange,
            group_id=f"structured_argument_refusal:{group_hash}",
            evidence_handle=f"structured_argument_refusal:{group_hash}",
            source="runtime_tool_schema",
            coverage={"canonical_genes": [seed] + query_genes, "rwr_seed_sets": [seed]},
            validator={"type": "schema_refusal_and_corrected_action"},
            polarity="negative",
        )


def _generate_state_updates(ctx: _Context) -> None:
    goal = ctx.builder.candidate_goal("structured_state_update")
    for index in range(goal):
        seed, _vector, rows = ctx.vector_case(index + 5000, row_count=4)
        supported_rows = sorted(rows, key=lambda row: (row.rank, row.gene_id))[:2]
        supported_genes = [row.gene_id for row in supported_rows]
        arguments = {
            "seed_genes": [seed],
            "query_genes": [row.gene_id for row in rows],
            "top_k": len(rows),
            "exclude_seed_genes": True,
        }
        payload = _ranked_payload("rwr_loe", seed, rows, top_k=len(rows))
        payload.update({"query_genes": arguments["query_genes"], "exclude_seed_genes": True})
        exchange = ctx.exchange(
            "query_filtered_ranking",
            arguments,
            payload,
            source="rwr_rank_cache",
        )
        evidence_id = f"ev_{_stable_hash([seed, supported_genes])[:16]}"
        full_state = StructuredState(
            user_anchors=UserAnchors(
                query_text="Refine the candidate group from visible RWR-LOE evidence.",
                evidence={"seed_gene_ids": [seed]},
                evidence_mode="tool_observation",
            ),
            relationship_status=RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
            predicted_groups=[
                GeneGroup(
                    group_id="candidate_group_1",
                    gene_ids=[seed] + supported_genes,
                    rationale="Membership is limited to the seed and the two lowest-rank visible results.",
                )
            ],
            evidence_log=[
                EvidenceRecord(
                    evidence_id=evidence_id,
                    source_type=EvidenceSourceType.TOOL_OBSERVATION,
                    summary="Visible RWR-LOE ranks support two additions to the candidate group.",
                    provenance={
                        "tool_name": "rwr_loe",
                        "multiplex_id": ctx.builder.multiplex_id,
                    },
                    supporting_gene_ids=supported_genes,
                    tool_call_id=exchange["tool_action"]["call_id"],
                )
            ],
            mechanistic_labels=[],
            remaining_budget=1,
            continuation_state=ContinuationState.CONTINUE,
            invalid_tool_call_count=0,
            total_tool_call_count=1,
        )
        # Validate against the live state schema, then emit the compact state-edit
        # projection required by this SFT family so the answer stays in budget.
        StructuredState.from_dict(full_state.to_dict())
        answer = {
            "relationship_status": full_state.relationship_status.value,
            "predicted_groups": [
                {
                    "group_id": full_state.predicted_groups[0].group_id,
                    "gene_ids": full_state.predicted_groups[0].gene_ids,
                    "evidence_ids": [evidence_id],
                }
            ],
            "continuation_state": full_state.continuation_state.value,
            "reason_code": "visible_rank_evidence_only",
        }
        state_rule = {
            "relationship_status": "partially_observed_group",
            "select_lowest_rank_non_seed_count": 2,
            "continuation_state": "continue",
            "evidence_id": evidence_id,
            "reason_code": "visible_rank_evidence_only",
        }
        group_hash = _stable_hash([seed, supported_genes])[:16]
        ctx.add(
            "structured_state_update",
            task=(
                "Apply the supplied state-update rule to the visible tool observation. Add only the two "
                "lowest-rank non-seed genes; do not add a mechanistic label."
            ),
            answer=answer,
            evidence={"tool_exchange": exchange, "state_update_rule": state_rule},
            exchange=exchange,
            group_id=f"state_update:{group_hash}",
            evidence_handle=f"state_update:{group_hash}",
            source="rwr_rank_cache_and_runtime_state_schema",
            coverage={
                "canonical_genes": [seed] + supported_genes,
                "rwr_seed_sets": [seed],
                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in supported_rows],
            },
            budget="matrix_state_4k",
            validator={"type": "runtime_structured_state_projection"},
        )


def generate_tool_families(builder: Any) -> None:
    """Generate required curriculum families 69-80 into ``builder``.

    Integration in the main builder is deliberately one import and one call::

        from scripts.pretrajectory_curriculum_tool_families import generate_tool_families
        generate_tool_families(self)
    """

    ctx = _Context(builder)
    _generate_rank_summary_selection(ctx)
    _generate_distance_selection(ctx)
    _generate_layer_membership_selection(ctx)
    _generate_component_selection(ctx)
    _generate_subgraph_selection(ctx)
    _generate_ablation_selection(ctx)
    _generate_rwr_parse(ctx)
    _generate_distance_parse(ctx)
    _generate_module_parse(ctx)
    _generate_provenance_answers(ctx)
    _generate_cli_refusals(ctx)
    _generate_state_updates(ctx)


__all__ = ["TOOL_FAMILY_NAMES", "generate_tool_families"]
