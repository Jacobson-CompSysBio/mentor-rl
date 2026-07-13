#!/usr/bin/env python3
"""Build the plan-driven, oracle-backed pre-trajectory multiplex curriculum.

This is the production replacement for the legacy prefix-sampled curriculum.
The source-controlled plan is executable: it defines the 81 required question
families, legal stage/book-mode combinations, budgets, mixture, split unit, and
required reports.  All graph facts come from the full binary CSR store.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.schemas import StructuredState, ToolAction, ToolObservation
from runtime.rwr_curriculum_reader import RwrCurriculumReader
from runtime.tool_curriculum_contract import (
    CURRICULUM_TOOL_NAMES,
    ToolCurriculumContractError,
    assert_no_provenance_leakage,
    build_tool_action,
    build_tool_exchange,
    sanitize_tool_payload,
    tool_policy_metadata,
)
from runtime.validators import validate_tool_action_schema
from scripts.multiplex_store_oracle import EdgeFact, MultiplexStoreOracle
from scripts.validate_pretrajectory_sft_curriculum_plan import (
    curriculum_plan_hash,
    load_curriculum_plan,
)


DEFAULT_PLAN_PATH = REPO_ROOT / "config" / "pretrajectory_sft_curriculum_v1.json"
DEFAULT_ALIAS_CACHE_PATH = REPO_ROOT / "data" / "corum_corpus" / "cache" / "mygene_query_cache.json"
DEFAULT_RANK_CACHE_ROOT = REPO_ROOT / "data" / "runtime" / "rwr_loe_full_brain_rank_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "pretrajectory_sft" / "v5_curriculum_patchcheck"
SPLITS = ("train", "val", "test")
RAW_PATH_RE = re.compile(
    r"(?:file://|(?:^|[\s\"'`])/(?:autofs|lustre|home|tmp|gpfs|scratch)/|"
    r"(?:^|[\s\"'`])(?:\.\.?/|data/|runtime/|scripts/|checkpoints/))",
    re.IGNORECASE,
)

SYSTEM_BY_STAGE = {
    "stage1_entity_schema": (
        "You are Mentor-RL learning the immutable coordinate system of a versioned biological multiplex. "
        "Use canonical Ensembl gene IDs, preserve exact identifiers, and never guess through ambiguity."
    ),
    "stage2_topology_priors": (
        "You are Mentor-RL learning calibrated atlas priors. Return the requested JSON exactly and distinguish "
        "recorded graph support from biological or causal claims."
    ),
    "stage3_open_book_vectors": (
        "You are Mentor-RL reading bounded oracle evidence from the declared multiplex. Extract, filter, sort, "
        "or recompute only what the evidence supports and return exact JSON."
    ),
    "stage4_module_world_model": (
        "You are Mentor-RL learning module and global multiplex structure. Apply exact set algebra and declared "
        "numeric rules; do not convert topology into unsupported causality."
    ),
    "stage5_structured_tools": (
        "You are Mentor-RL using the live structured runtime. Choose the cheapest sufficient tool, emit only "
        "schema-valid biological arguments, preserve public provenance, and never expose execution paths."
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, length: int = 20) -> str:
    return f"{prefix}_{stable_hash(value)[:length]}"


def estimate_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate used by the generation contract."""

    byte_estimate = math.ceil(len(text.encode("utf-8")) / 4) if text else 0
    lexical = 0
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        token = match.group(0)
        lexical += (
            max(1, math.ceil(len(token.encode("utf-8")) / 4))
            if re.fullmatch(r"\w+", token, flags=re.UNICODE)
            else 1
        )
    return max(byte_estimate, lexical)


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def recursively_round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round_float(value, digits)
    if isinstance(value, list):
        return [recursively_round(item, digits) for item in value]
    if isinstance(value, tuple):
        return [recursively_round(item, digits) for item in value]
    if isinstance(value, dict):
        return {str(key): recursively_round(item, digits) for key, item in value.items()}
    return value


def contains_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, list):
        return any(contains_float(item) for item in value)
    if isinstance(value, dict):
        return any(contains_float(item) for item in value.values())
    return False


def assign_split(plan: Mapping[str, Any], strongest_group_id: str) -> str:
    fractions = plan["split_contract"]["assignment_fractions"]
    value = int(
        stable_hash(
            {
                "plan_id": plan["plan_id"],
                "multiplex_id": plan["graph_contract"]["multiplex_id"],
                "strongest_group_id": strongest_group_id,
            }
        )[:16],
        16,
    ) / float(16**16)
    train_boundary = float(fractions["train"])
    val_boundary = train_boundary + float(fractions["val"])
    if value < train_boundary:
        return "train"
    if value < val_boundary:
        return "val"
    return "test"


def largest_remainder_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    raw = {key: total * float(weight) for key, weight in weights.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def deterministic_order(values: Iterable[Any], *, seed: int, namespace: str) -> list[Any]:
    return sorted(values, key=lambda value: stable_hash({"seed": seed, "namespace": namespace, "value": value}))


def layer_family(layer_name: str) -> str:
    prefix = layer_name.split(":", 1)[0]
    aliases = {
        "HumanNetV3": "humannet_v3",
        "bulkPEN": "bulk_pen",
        "scPEN": "sc_pen",
        "TFs": "tf_target",
        "tf": "tf_target",
    }
    return aliases.get(prefix, re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_") or "other")


@dataclass(frozen=True)
class Family:
    id: int
    name: str
    primary_stage: str
    mixture_bucket: str
    allowed_book_modes: tuple[str, ...]
    difficulty_source: str


@dataclass
class CurriculumExample:
    family: Family
    book_mode: str
    task: str
    answer: dict[str, Any]
    evidence: dict[str, Any] | None
    fact_payload: dict[str, Any]
    strongest_group_id: str
    layer_scope: str = "all_layers"
    layer_ids: list[str] = field(default_factory=list)
    layer_families: list[str] = field(default_factory=list)
    module_source: str = "none"
    context_budget_profile: str = "atomic_1k"
    evidence_handles: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, list[str]] = field(default_factory=dict)
    validator: dict[str, Any] = field(default_factory=lambda: {"type": "exact_json"})
    polarity: str = "positive"
    page: dict[str, Any] | None = None
    tool_exchange: dict[str, Any] | None = None

    @property
    def oracle_fact_id(self) -> str:
        return stable_id(
            "fact",
            {
                "family": self.family.name,
                "payload": self.fact_payload,
                "page": self.page,
            },
        )


class CurriculumBuilder:
    def __init__(
        self,
        *,
        plan_path: Path,
        profile: str,
        out_dir: Path,
        seed: int,
        alias_cache_path: Path | None,
        overwrite: bool,
    ) -> None:
        self.plan_path = plan_path
        self.plan = load_curriculum_plan(plan_path)
        self.plan_hash = curriculum_plan_hash(self.plan)
        if profile not in self.plan["build_profiles"]:
            raise ValueError(f"Unknown curriculum build profile: {profile}")
        self.profile_name = profile
        self.profile = self.plan["build_profiles"][profile]
        self.out_dir = out_dir
        self.seed = int(seed)
        self.alias_cache_path = alias_cache_path
        self.overwrite = overwrite
        self.families = {
            row["name"]: Family(
                id=int(row["id"]),
                name=str(row["name"]),
                primary_stage=str(row["primary_stage"]),
                mixture_bucket=str(row["mixture_bucket"]),
                allowed_book_modes=tuple(str(mode) for mode in row["allowed_book_modes"]),
                difficulty_source=str(row["difficulty_source"]),
            )
            for row in self.plan["question_families"]
        }
        graph = self.plan["graph_contract"]
        self.store_path = REPO_ROOT / graph["store_path"]
        self.flist_path = REPO_ROOT / graph["flist_path"]
        self.module_path = REPO_ROOT / graph["mixed_module_source"] / "modules.jsonl"
        self.module_manifest_path = REPO_ROOT / graph["mixed_module_source"] / "manifest.json"
        self.mentor_manifest_path = REPO_ROOT / graph["mentor_ev_source"] / "manifest.json"
        self.rwr_module_manifest_path = REPO_ROOT / graph["rwr_loe_source"] / "manifest.json"
        self.oracle = MultiplexStoreOracle(self.store_path)
        self.flist_id = f"sha256:{self._sha256(self.flist_path)}"
        self.store_id = self.oracle.store_id
        self.alias_cache_id = (
            f"sha256:{self._sha256(alias_cache_path)}"
            if alias_cache_path is not None and alias_cache_path.is_file()
            else None
        )
        self.module_corpus_id = f"sha256:{self._sha256(self.module_path)}"
        self.module_manifest_id = f"sha256:{self._sha256(self.module_manifest_path)}"
        self.mentor_manifest_id = f"sha256:{self._sha256(self.mentor_manifest_path)}"
        self.rwr_module_manifest_id = f"sha256:{self._sha256(self.rwr_module_manifest_path)}"
        self.multiplex_id = str(graph["multiplex_id"])
        self.examples: list[CurriculumExample] = []
        self.generated_counts: Counter[str] = Counter()
        self.filtered_counts: Counter[str] = Counter()
        self.alias_to_genes: dict[str, set[str]] = {}
        self.gene_to_symbol: dict[str, str] = {}
        self.modules: list[dict[str, Any]] = []
        self.modules_by_source: dict[str, list[dict[str, Any]]] = {}
        self.modules_by_id: dict[str, dict[str, Any]] = {}
        self.rank_context_dir: Path | None = None
        self.rwr: RwrCurriculumReader | None = None
        self.rank_metadata_by_seed: dict[str, Path] = {}
        self._rank_rows_cache: dict[str, list[dict[str, Any]]] = {}
        self.distance_shards: list[Path] = []
        self._distance_cache: dict[Path, tuple[list[str], dict[str, list[float | None]]]] = {}
        self._effective_fact_groups: dict[str, str] = {}
        self.coalesced_fact_group_count = 0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def prepare_output(self) -> None:
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            if not self.overwrite:
                raise FileExistsError(f"Output directory is not empty: {self.out_dir}")
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.plan_path, self.out_dir / "curriculum_plan.json")

    def load_sources(self) -> None:
        self._load_aliases()
        self._load_modules()
        self._discover_rank_cache()

    def _load_aliases(self) -> None:
        if self.alias_cache_path is None or not self.alias_cache_path.exists():
            raise FileNotFoundError(
                "A versioned alias cache is required for entity families 1-4; "
                f"missing {self.alias_cache_path}."
            )
        raw = json.loads(self.alias_cache_path.read_text(encoding="utf-8"))
        aliases: dict[str, set[str]] = defaultdict(set)
        symbols_by_gene: dict[str, Counter[str]] = defaultdict(Counter)
        graph_genes = set(self.oracle.gene_ids)
        for cache_key, hits in raw.items():
            if not isinstance(hits, list):
                continue
            key_alias = str(cache_key).split("::", 1)[-1]
            for hit in hits:
                if not isinstance(hit, dict) or hit.get("notfound") is True:
                    continue
                if hit.get("taxid") not in (None, 9606):
                    continue
                ensembl = hit.get("ensembl")
                ids: list[str] = []
                if isinstance(ensembl, dict):
                    ids = [ensembl.get("gene")]
                elif isinstance(ensembl, list):
                    ids = [item.get("gene") for item in ensembl if isinstance(item, dict)]
                gene_ids = sorted(
                    {
                        gene_id
                        for gene_id in ids
                        if isinstance(gene_id, str) and gene_id in graph_genes
                    }
                )
                if not gene_ids:
                    continue
                symbol = hit.get("symbol")
                candidate_aliases = [key_alias, hit.get("query"), symbol]
                if isinstance(hit.get("alias"), list):
                    candidate_aliases.extend(hit["alias"])
                for alias in candidate_aliases:
                    if isinstance(alias, str) and alias.strip():
                        aliases[alias.strip().upper()].update(gene_ids)
                if isinstance(symbol, str) and symbol:
                    for gene_id in gene_ids:
                        symbols_by_gene[gene_id][symbol] += 1
        self.alias_to_genes = dict(aliases)
        self.gene_to_symbol = {
            gene_id: counts.most_common(1)[0][0]
            for gene_id, counts in symbols_by_gene.items()
            if counts
        }
        resolved = sum(1 for ids in self.alias_to_genes.values() if len(ids) == 1)
        ambiguous = sum(1 for ids in self.alias_to_genes.values() if len(ids) > 1)
        if resolved < 1_000 or ambiguous < 100:
            raise ValueError(
                f"Alias registry is too small for the curriculum: resolved={resolved}, ambiguous={ambiguous}."
            )

    def _load_modules(self) -> None:
        if not self.module_path.exists():
            raise FileNotFoundError(f"Missing mixed module oracle: {self.module_path}")
        modules: list[dict[str, Any]] = []
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with self.module_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("module_id"), str):
                    raise ValueError(f"Malformed module at {self.module_path}:{line_number}")
                genes = row.get("gene_ids")
                if not isinstance(genes, list) or not genes:
                    continue
                clean = dict(row)
                clean["gene_ids"] = sorted({str(gene) for gene in genes})
                modules.append(clean)
                by_source[str(clean.get("source", "unknown"))].append(clean)
        for source in by_source:
            by_source[source] = deterministic_order(
                by_source[source], seed=self.seed, namespace=f"module:{source}"
            )
        self.modules = modules
        self.modules_by_source = dict(by_source)
        self.modules_by_id = {str(row["module_id"]): row for row in modules}
        if not self.modules_by_source.get("MENTOR_GW_DENDROGRAM"):
            raise ValueError("Mixed module oracle has no MENTOR-EV modules.")
        if not self.modules_by_source.get("RWR_LOE_FULL_BRAIN"):
            raise ValueError("Mixed module oracle has no RWR-LOE modules.")

    def _discover_rank_cache(self) -> None:
        contexts = sorted(DEFAULT_RANK_CACHE_ROOT.glob("context_*/cache_context.json"))
        compatible: list[Path] = []
        for path in contexts:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("network_flist_sha256") == self.flist_id.split(":", 1)[1]:
                compatible.append(path.parent)
        if len(compatible) != 1:
            raise ValueError(
                "Expected exactly one RWR-LOE cache matching the curriculum flist; "
                f"found {len(compatible)}."
            )
        self.rank_context_dir = compatible[0]
        self.rwr = RwrCurriculumReader(self.rank_context_dir)
        if self.rwr.identity.network_flist_sha256 != self.flist_id.split(":", 1)[1]:
            raise ValueError("RWR reader identity does not match the curriculum flist.")
        rank_dir = self.rank_context_dir / "ranks"
        for metadata_path in rank_dir.glob("*.metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            seed = metadata.get("seed_gene_id")
            if isinstance(seed, str):
                self.rank_metadata_by_seed[seed] = metadata_path
        self.distance_shards = sorted(
            self.rank_context_dir.glob("shards/shard_*/rwr_output/*_spearman_dist_matrix.tsv")
        )
        if len(self.rank_metadata_by_seed) < 1_000 or not self.distance_shards:
            raise ValueError("RWR-LOE cache is incomplete for rank/distance curriculum generation.")

    def family(self, name: str) -> Family:
        return self.families[name]

    def add(self, example: CurriculumExample) -> None:
        family = example.family
        if example.book_mode not in family.allowed_book_modes:
            raise ValueError(f"{family.name} does not allow book mode {example.book_mode}.")
        stage = next(row for row in self.plan["stages"] if row["name"] == family.primary_stage)
        if example.book_mode not in stage["allowed_book_modes"]:
            raise ValueError(
                f"Stage {family.primary_stage} does not allow {example.book_mode} for {family.name}."
            )
        if example.context_budget_profile not in stage["allowed_budget_profiles"]:
            raise ValueError(
                f"Stage {family.primary_stage} does not allow budget {example.context_budget_profile}."
            )
        if example.book_mode == "closed_book" and (
            contains_float(example.answer) or (example.evidence is not None and contains_float(example.evidence))
        ):
            raise ValueError(f"Closed-book arbitrary float leaked into {family.name}.")
        self.generated_counts[family.name] += 1
        self.examples.append(example)

    def coalesce_oracle_fact_groups(self) -> int:
        """Assign one split group to every pre-render oracle fact.

        Most facts inherit a stronger entity/module/seed group.  If multiple
        generator routes produce the identical fact but disagree about that
        stronger group, the only safe common split unit is the fact itself.
        This normalization happens before rendering and split assignment.
        """

        groups: dict[str, set[str]] = defaultdict(set)
        for example in self.examples:
            groups[example.oracle_fact_id].add(example.strongest_group_id)
        effective: dict[str, str] = {}
        coalesced = 0
        for fact_id, values in groups.items():
            if len(values) == 1:
                effective[fact_id] = next(iter(values))
            else:
                effective[fact_id] = f"oracle_fact_group:{fact_id}"
                coalesced += 1
        self._effective_fact_groups = effective
        self.coalesced_fact_group_count = coalesced
        return coalesced

    def render(self, example: CurriculumExample) -> dict[str, Any] | None:
        answer = recursively_round(example.answer)
        evidence = recursively_round(sanitize_tool_payload(example.evidence)) if example.evidence else None
        if example.book_mode == "closed_book":
            question = example.task
        else:
            if evidence is None:
                raise ValueError(f"{example.family.name} requires evidence in {example.book_mode} mode.")
            question = f"{example.task}\n\nEVIDENCE_JSON:\n{json_text(evidence)}"
        answer_text = json_text(answer)
        system = SYSTEM_BY_STAGE[example.family.primary_stage]
        fact_id = example.oracle_fact_id
        fact_group_id = self._effective_fact_groups.get(fact_id, example.strongest_group_id)
        split = assign_split(self.plan, fact_group_id)
        record_id = stable_id(
            "sft",
            {
                "plan_hash": self.plan_hash,
                "fact_id": fact_id,
                "fact_group_id": fact_group_id,
                "book_mode": example.book_mode,
                "question": question,
                "answer": answer_text,
                "context_budget_profile": example.context_budget_profile,
                "layer_scope": example.layer_scope,
                "layer_ids": sorted(set(example.layer_ids)),
                "evidence_handles": list(example.evidence_handles),
                "provenance": sanitize_tool_payload(example.provenance),
                "page": example.page,
                "tool_exchange": sanitize_tool_payload(example.tool_exchange),
            },
        )
        metadata: dict[str, Any] = {
            "schema_version": self.plan["dataset_schema_version"],
            "record_id": record_id,
            "oracle_fact_id": fact_id,
            "oracle_fact_group_id": fact_group_id,
            "book_mode": example.book_mode,
            "question_family": example.family.name,
            "question_family_id": example.family.id,
            "curriculum_stage": example.family.primary_stage,
            "mixture_bucket": example.family.mixture_bucket,
            "multiplex_id": self.multiplex_id,
            "store_id": self.store_id,
            "flist_id": self.flist_id,
            "layer_scope": example.layer_scope,
            "layer_ids": sorted(set(example.layer_ids)),
            "layer_families": sorted(set(example.layer_families)),
            "entity_namespace": self.plan["graph_contract"]["entity_namespace"],
            "module_source": example.module_source,
            "answer_format": "json",
            "difficulty_source": example.family.difficulty_source,
            "context_budget_profile": example.context_budget_profile,
            "evidence_handles": list(example.evidence_handles),
            "provenance": {
                "plan_hash": self.plan_hash,
                "oracle": "full_binary_csr_store",
                "oracle_validation": True,
                **sanitize_tool_payload(example.provenance),
            },
            "validator": dict(example.validator),
            "polarity": example.polarity,
            "coverage_objects": {
                key: sorted(set(values)) for key, values in example.coverage.items()
            },
            "split": split,
        }
        if example.page is not None:
            metadata["page"] = example.page
        if contains_float(answer) or contains_float(evidence):
            metadata["numeric_tolerance"] = {
                "absolute": 0.0001,
                "relative": 0.0001,
                "rounding_digits": 4,
            }
        if example.tool_exchange is not None:
            assert_no_provenance_leakage(example.tool_exchange)
            metadata["tool_schema_validated"] = True
            metadata["tool_name"] = example.tool_exchange["tool_action"]["tool_name"]
            metadata["tool_policy"] = example.tool_exchange["tool_policy"]
            metadata["tool_exchange"] = sanitize_tool_payload(example.tool_exchange)

        if RAW_PATH_RE.search(system) or RAW_PATH_RE.search(question) or RAW_PATH_RE.search(answer_text):
            self.filtered_counts[example.family.name] += 1
            return None
        profile = self.plan["context_budget_profiles"][example.context_budget_profile]
        prompt_tokens = estimate_tokens(system) + estimate_tokens(question) + 16
        answer_tokens = estimate_tokens(answer_text)
        violations: list[str] = []
        if prompt_tokens > int(profile["max_prompt_tokens"]):
            violations.append("prompt_tokens")
        if answer_tokens > int(profile["max_answer_tokens"]):
            violations.append("answer_tokens")
        if prompt_tokens + answer_tokens > int(profile["max_total_tokens"]):
            violations.append("total_tokens")
        if len(answer_text) > int(profile["max_answer_characters"]):
            violations.append("answer_characters")
        metadata["answer_budget"] = {
            "profile": example.context_budget_profile,
            "prompt_token_estimate": prompt_tokens,
            "answer_token_estimate": answer_tokens,
            "total_token_estimate": prompt_tokens + answer_tokens,
            "answer_character_count": len(answer_text),
            "violations": violations,
        }
        if violations:
            self.filtered_counts[example.family.name] += 1
            return None
        return {"system": system, "question": question, "answer": answer_text, "metadata": metadata}

    def canonical_object(self, example: CurriculumExample) -> dict[str, Any]:
        """Return the pre-render oracle object referenced by one or more rows."""

        return {
            "object_id": example.oracle_fact_id,
            "object_type": example.family.name,
            "multiplex_id": self.multiplex_id,
            "store_id": self.store_id,
            "flist_id": self.flist_id,
            "payload": recursively_round(sanitize_tool_payload(example.fact_payload)),
        }

    def candidate_goal(self, family_name: str) -> int:
        """Return a buffered patchcheck pool size for one required family."""

        family = self.family(family_name)
        family_count = sum(
            item.mixture_bucket == family.mixture_bucket for item in self.families.values()
        )
        bucket_weight = float(self.plan["mixture"]["content_buckets"][family.mixture_bucket])
        fractions = self.plan["split_contract"]["assignment_fractions"]
        needs: list[float] = []
        for split in SPLITS:
            selected = int(self.profile["split_counts"][split]) * bucket_weight / family_count
            needs.append(selected / max(float(fractions[split]), 1e-9))
        # Stage 5 is intentionally only 7% of the primary mixture but supplies
        # 20% of the stage-6 consolidation blend.  Size its candidate frontier
        # for the latter as well, otherwise val/test tool rows underfill even
        # when the primary split quotas pass.
        if family.primary_stage == "stage5_structured_tools":
            stage6 = next(row for row in self.plan["stages"] if row["name"] == "stage6_blend")
            stage_weight = float(stage6["source_stage_weights"][family.primary_stage])
            stage_family_count = sum(
                item.primary_stage == family.primary_stage for item in self.families.values()
            )
            for split in SPLITS:
                per_family = (
                    int(self.profile["split_counts"][split])
                    * stage_weight
                    / stage_family_count
                )
                needs.append(per_family / max(float(fractions[split]), 1e-9))
        minimum = int(self.profile["minimum_selected_per_required_family"])
        return max(minimum * 4, int(math.ceil(max(needs) * 1.75)))

    def _base_provenance(self, source: str) -> dict[str, Any]:
        if source == "versioned_alias_registry":
            oracle = "versioned_mygene_alias_cache"
        elif source in {"calibration_rule", "rwr_calibration_rule"}:
            oracle = "rule_derived_calibration"
        elif "distance_shard" in source:
            oracle = "rwr_loe_distance_shard"
        elif "rwr_rank_cache" in source:
            oracle = "rwr_loe_rank_cache"
        elif "module" in source or source == "mentor_ev_dendrogram_oracle":
            oracle = "versioned_module_corpus"
        elif source == "runtime_tool_schema":
            oracle = "live_runtime_tool_schema"
        elif source == "multiplex_manifest":
            oracle = "full_binary_store_manifest"
        else:
            oracle = "full_binary_csr_store"
        provenance = {
            "source": source,
            "oracle": oracle,
            "store_id": self.store_id,
            "flist_id": self.flist_id,
            "multiplex_id": self.multiplex_id,
        }
        if source == "versioned_alias_registry":
            provenance["alias_cache_id"] = self.alias_cache_id
        if "module" in source or source == "mentor_ev_dendrogram_oracle":
            provenance["module_corpus_id"] = self.module_corpus_id
            provenance["module_manifest_id"] = self.module_manifest_id
        return provenance

    def _balanced_rwr_seed_pool(self, per_split: int = 16) -> list[str]:
        """Choose a bounded, deterministic seed pool with split coverage.

        Full rank vectors contain roughly 33k rows and the reader intentionally
        caches validated vectors.  Generating one vector per requested example
        would consume many gigabytes while adding no new seed-group coverage.
        Reuse a bounded set of independently split seed groups and vary the
        queried rows within each full vector instead.
        """

        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        ordered = deterministic_order(
            self.rwr.seed_gene_ids,
            seed=self.seed,
            namespace="rwr_curriculum_seeds",
        )
        by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
        for seed_gene_id in ordered:
            split = assign_split(self.plan, f"rwr_seed:{seed_gene_id}")
            if len(by_split[split]) < per_split:
                by_split[split].append(seed_gene_id)
            if all(len(values) >= per_split for values in by_split.values()):
                break
        underfilled = {
            split: len(values)
            for split, values in by_split.items()
            if len(values) < per_split
        }
        if underfilled:
            raise ValueError(f"RWR seed split pool underfilled: {underfilled}.")
        # Interleave splits so short probes and early generation remain balanced.
        return [
            by_split[split][index]
            for index in range(per_split)
            for split in SPLITS
        ]

    def generate_stage1_and_stage2(self) -> None:
        self._generate_entity_families()
        self._generate_schema_families()
        self._generate_topology_prior_families()

    def _generate_entity_families(self) -> None:
        resolved = [
            (alias, next(iter(gene_ids)))
            for alias, gene_ids in self.alias_to_genes.items()
            if len(gene_ids) == 1
            and next(iter(gene_ids)) in self.gene_to_symbol
            and alias != next(iter(gene_ids))
        ]
        resolved = deterministic_order(resolved, seed=self.seed, namespace="resolved_aliases")
        ambiguous = deterministic_order(
            [(alias, sorted(gene_ids)) for alias, gene_ids in self.alias_to_genes.items() if len(gene_ids) > 1],
            seed=self.seed,
            namespace="ambiguous_aliases",
        )
        goals = {
            name: self.candidate_goal(name)
            for name in (
                "entity_symbol_to_ensembl",
                "entity_ensembl_to_symbol",
                "ambiguous_alias_resolution",
                "cross_context_entity_alignment",
            )
        }
        if len(resolved) < max(goals["entity_symbol_to_ensembl"], goals["cross_context_entity_alignment"]):
            raise ValueError("Alias registry cannot fill the resolved entity curriculum.")
        if len(ambiguous) < goals["ambiguous_alias_resolution"]:
            raise ValueError("Alias registry cannot fill ambiguous-alias examples.")

        for alias, gene_id in resolved[: goals["entity_symbol_to_ensembl"]]:
            symbol = self.gene_to_symbol[gene_id]
            self.add(
                CurriculumExample(
                    family=self.family("entity_symbol_to_ensembl"),
                    book_mode="closed_book",
                    task=(
                        f"Normalize `{alias}` for multiplex `{self.multiplex_id}`. "
                        "Return the canonical graph identity as JSON."
                    ),
                    answer={
                        "status": "resolved",
                        "gene_id": gene_id,
                        "gene_symbol": symbol,
                        "canonical_entity": f"<GENE:{gene_id}|{symbol}>",
                        "multiplex_id": self.multiplex_id,
                    },
                    evidence=None,
                    fact_payload={"alias": alias, "gene_id": gene_id, "symbol": symbol},
                    strongest_group_id=f"entity:{gene_id}",
                    evidence_handles=["alias_registry:mygene_human_cache"],
                    provenance=self._base_provenance("versioned_alias_registry"),
                    coverage={"canonical_genes": [gene_id]},
                )
            )

        gene_symbol_rows = deterministic_order(
            list(self.gene_to_symbol.items()), seed=self.seed, namespace="gene_to_symbol"
        )
        for gene_id, symbol in gene_symbol_rows[: goals["entity_ensembl_to_symbol"]]:
            self.add(
                CurriculumExample(
                    family=self.family("entity_ensembl_to_symbol"),
                    book_mode="closed_book",
                    task=(
                        f"What display symbol is associated with canonical graph ID `{gene_id}` "
                        f"in multiplex `{self.multiplex_id}`? Return JSON."
                    ),
                    answer={
                        "gene_id": gene_id,
                        "gene_symbol": symbol,
                        "canonical_entity": f"<GENE:{gene_id}|{symbol}>",
                    },
                    evidence=None,
                    fact_payload={"gene_id": gene_id, "symbol": symbol},
                    strongest_group_id=f"entity:{gene_id}",
                    evidence_handles=["alias_registry:mygene_human_cache"],
                    provenance=self._base_provenance("versioned_alias_registry"),
                    coverage={"canonical_genes": [gene_id]},
                )
            )

        for alias, gene_ids in ambiguous[: goals["ambiguous_alias_resolution"]]:
            self.add(
                CurriculumExample(
                    family=self.family("ambiguous_alias_resolution"),
                    book_mode="closed_book",
                    task=(
                        f"Alias `{alias}` maps to multiple canonical graph IDs. "
                        "What must happen before any graph lookup? Return JSON."
                    ),
                    answer={
                        "status": "ambiguous",
                        "candidate_gene_ids": gene_ids,
                        "action": "ask_for_disambiguation_or_use_context",
                        "allowed_claim": (
                            "Do not perform graph lookup until the canonical Ensembl gene ID is resolved."
                        ),
                    },
                    evidence=None,
                    fact_payload={"alias": alias, "candidate_gene_ids": gene_ids},
                    strongest_group_id=f"alias:{alias}",
                    evidence_handles=["alias_registry:mygene_human_cache"],
                    provenance=self._base_provenance("versioned_alias_registry"),
                    coverage={"canonical_genes": gene_ids},
                    polarity="insufficient_context",
                )
            )

        for alias, gene_id in resolved[: goals["cross_context_entity_alignment"]]:
            symbol = self.gene_to_symbol[gene_id]
            self.add(
                CurriculumExample(
                    family=self.family("cross_context_entity_alignment"),
                    book_mode="closed_book",
                    task=(
                        f"An RWR vector uses `{gene_id}` while a module table displays `{symbol}`. "
                        "Do they identify the same graph entity under the alias registry? Return JSON."
                    ),
                    answer={
                        "same_entity": True,
                        "gene_id": gene_id,
                        "gene_symbol": symbol,
                        "reason": (
                            "The display symbol maps to the same canonical Ensembl gene identifier used by the vector."
                        ),
                    },
                    evidence=None,
                    fact_payload={"gene_id": gene_id, "symbol": symbol, "alias": alias},
                    strongest_group_id=f"entity:{gene_id}",
                    evidence_handles=["alias_registry:mygene_human_cache"],
                    provenance=self._base_provenance("versioned_alias_registry"),
                    coverage={"canonical_genes": [gene_id]},
                )
            )

    def _generate_schema_families(self) -> None:
        multiplex_fields = [
            ("species", "Homo sapiens"),
            ("context", "full_brain"),
            ("version", "v1"),
            ("graph_type", "biological_multiplex"),
            ("entity_namespace", "ensembl_gene_id_primary"),
            ("layer_count", self.oracle.layer_count),
            ("gene_count", self.oracle.gene_count),
            ("store_format", str(self.oracle.manifest["format_version"])),
        ]
        goal = self.candidate_goal("multiplex_identifier_parsing")
        templates = (
            "Parse the multiplex identifier",
            "Identify the graph context encoded by",
            "Return the immutable schema fields for",
            "Describe the versioned coordinate system named",
        )
        for index in range(goal):
            field_name, field_value = multiplex_fields[index % len(multiplex_fields)]
            self.add(
                CurriculumExample(
                    family=self.family("multiplex_identifier_parsing"),
                    book_mode="closed_book",
                    task=(
                        f"{templates[index % len(templates)]} `{self.multiplex_id}` and return "
                        f"its `{field_name}` field as JSON."
                    ),
                    answer={
                        "multiplex_id": self.multiplex_id,
                        field_name: field_value,
                    },
                    evidence=None,
                    fact_payload={"multiplex_id": self.multiplex_id, "field": field_name, "value": field_value},
                    strongest_group_id=f"multiplex_schema:{field_name}",
                    evidence_handles=["multiplex_manifest:full_store"],
                    provenance=self._base_provenance("multiplex_manifest"),
                )
            )

        layers = deterministic_order(self.oracle.layer_names, seed=self.seed, namespace="schema_layers")
        for family_name in ("layer_tag_parsing", "layer_family_classification"):
            goal = self.candidate_goal(family_name)
            for layer in layers[:goal]:
                metadata = self.oracle.layer_metadata(layer)
                parts = layer.split(":")
                parsed = {
                    "layer_id": layer,
                    "source": parts[0],
                    "tags": parts[1:],
                    "layer_family": layer_family(layer),
                }
                if family_name == "layer_family_classification":
                    answer = {
                        "layer_id": layer,
                        "layer_family": layer_family(layer),
                        "source": parts[0],
                    }
                    task = f"Classify layer `{layer}` into its normalized layer family. Return JSON."
                else:
                    answer = parsed
                    task = f"Parse versioned layer tag `{layer}` into source, ordered tags, and family. Return JSON."
                self.add(
                    CurriculumExample(
                        family=self.family(family_name),
                        book_mode="closed_book",
                        task=task,
                        answer=answer,
                        evidence=None,
                        fact_payload=parsed,
                        strongest_group_id=f"layer:{layer}",
                        layer_scope="single_layer",
                        layer_ids=[layer],
                        layer_families=[layer_family(layer)],
                        evidence_handles=[f"layer_manifest:{metadata['layer_id']}"],
                        provenance=self._base_provenance("multiplex_manifest"),
                        coverage={"layers": [layer]},
                    )
                )

    def _generate_topology_prior_families(self) -> None:
        aggregate = self.oracle.view(None)
        degrees = np.diff(aggregate.indptr)
        nonzero = degrees[degrees > 0]
        q50, q90, q99 = (int(np.quantile(nonzero, q)) for q in (0.50, 0.90, 0.99))
        node_order = self.oracle.sample_node_indices(
            layer=None,
            count=self.candidate_goal("degree_hub_bias"),
            seed=self.seed + 2100,
            minimum_degree=1,
        )
        for node_index in node_order:
            degree = int(degrees[node_index])
            if degree >= q99:
                degree_bin, hub_like = "top_1_percent", True
            elif degree >= q90:
                degree_bin, hub_like = "top_10_percent", True
            elif degree >= q50:
                degree_bin, hub_like = "above_median", False
            else:
                degree_bin, hub_like = "below_median", False
            gene_id = self.oracle.gene_ids[node_index]
            self.add(
                CurriculumExample(
                    family=self.family("degree_hub_bias"),
                    book_mode="closed_book",
                    task=(
                        f"Gene `{gene_id}` is in aggregate degree bin `{degree_bin}`. "
                        "Return the calibrated hub-bias interpretation as JSON."
                    ),
                    answer={
                        "gene_id": gene_id,
                        "scope": "all_layers",
                        "degree_bin": degree_bin,
                        "hub_like": hub_like,
                        "caveat": (
                            "High-degree genes can appear proximal to many genes; network proximity alone is not causal evidence."
                            if hub_like
                            else "Degree alone does not establish a biological mechanism."
                        ),
                    },
                    evidence=None,
                    fact_payload={"gene_id": gene_id, "degree": degree, "degree_bin": degree_bin},
                    strongest_group_id=f"entity:{gene_id}",
                    evidence_handles=["aggregate_degree_distribution:full_store"],
                    provenance=self._base_provenance("full_store_degree_distribution"),
                    coverage={"canonical_genes": [gene_id]},
                )
            )

        layer_pairs = []
        by_family: dict[str, list[str]] = defaultdict(list)
        for layer in self.oracle.layer_names:
            by_family[layer_family(layer)].append(layer)
        families = sorted(by_family)
        for left in families:
            for right in families:
                if left != right:
                    layer_pairs.append((left, right))
        goal = self.candidate_goal("layer_specificity")
        for index in range(goal):
            support_family, absent_family = layer_pairs[index % len(layer_pairs)]
            self.add(
                CurriculumExample(
                    family=self.family("layer_specificity"),
                    book_mode="closed_book",
                    task=(
                        f"A gene pair is supported only by `{support_family}` layers and not by `{absent_family}` layers. "
                        "Return the strongest allowed and disallowed claims as JSON."
                    ),
                    answer={
                        "supporting_layer_family": support_family,
                        "unsupported_layer_family": absent_family,
                        "allowed_claim": f"The pair has recorded {support_family} support in this graph version.",
                        "disallowed_claim": (
                            "Do not generalize the relationship to an unsupported modality or claim a direct physical interaction."
                        ),
                    },
                    evidence=None,
                    fact_payload={"support_family": support_family, "absent_family": absent_family},
                    strongest_group_id=f"layer_family_rule:{support_family}:{absent_family}",
                    layer_families=[support_family, absent_family],
                    provenance=self._base_provenance("calibration_rule"),
                    polarity="insufficient_context",
                )
            )

        sparse_layers = deterministic_order(
            self.oracle.layer_names, seed=self.seed, namespace="calibration_sparse_layers"
        )
        no_path_goal = self.candidate_goal("no_edge_no_path_calibration")
        emitted = 0
        for layer in sparse_layers:
            view = self.oracle.view(layer)
            absent_indices = np.flatnonzero(np.diff(view.indptr) == 0)
            present_indices = view.nonisolated_node_indices()
            if not len(absent_indices) or not len(present_indices):
                continue
            rng = np.random.default_rng(self.seed + emitted + 6400)
            for _ in range(min(4, no_path_goal - emitted)):
                source = self.oracle.gene_ids[int(rng.choice(absent_indices))]
                target = self.oracle.gene_ids[int(rng.choice(present_indices))]
                self.add(
                    CurriculumExample(
                        family=self.family("no_edge_no_path_calibration"),
                        book_mode="closed_book",
                        task=(
                            f"`{source}` is absent from layer `{layer}`, so no edge or path to `{target}` is recorded "
                            "in that layer. What can and cannot be concluded? Return JSON."
                        ),
                        answer={
                            "source_gene_id": source,
                            "target_gene_id": target,
                            "layer_id": layer,
                            "edge_exists": False,
                            "path_exists": False,
                            "allowed_claim": "No edge or path is recorded in this graph version and layer scope.",
                            "disallowed_claims": [
                                "The genes are biologically unrelated in every context.",
                                "No interaction exists in biology.",
                            ],
                        },
                        evidence=None,
                        fact_payload={"layer": layer, "source": source, "target": target, "source_absent": True},
                        strongest_group_id=f"gene_layer:{source}:{layer}",
                        layer_scope="single_layer",
                        layer_ids=[layer],
                        layer_families=[layer_family(layer)],
                        provenance=self._base_provenance("full_store_csr"),
                        coverage={
                            "canonical_genes": [source, target],
                            "layers": [layer],
                            "negative_edges": [f"{layer}|{source}|{target}"],
                            "gene_layer_pairs": [f"{source}|{layer}"],
                        },
                        polarity="negative",
                    )
                )
                emitted += 1
                if emitted >= no_path_goal:
                    break
            if emitted >= no_path_goal:
                break
        if emitted < no_path_goal:
            raise ValueError("Could not fill no-edge/no-path calibration examples.")

        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        seeds = self._balanced_rwr_seed_pool()
        proximity_goal = self.candidate_goal("no_edge_with_network_proximity")
        emitted = 0
        for seed_gene in seeds:
            # Several independently validated nonedges can come from one full
            # seed vector.  The strongest split group remains the seed, so no
            # render variant can cross splits.
            for row in self.rwr.top_k(seed_gene, 96):
                if not self.oracle.has_edge(seed_gene, row.gene_id, layer=None):
                    self.add(
                        CurriculumExample(
                            family=self.family("no_edge_with_network_proximity"),
                            book_mode="closed_book",
                            task=(
                                f"`{seed_gene}` and `{row.gene_id}` have no direct aggregate edge but the target is in "
                                "the seed's high-support RWR neighborhood. Return the calibrated interpretation as JSON."
                            ),
                            answer={
                                "source_gene_id": seed_gene,
                                "target_gene_id": row.gene_id,
                                "direct_edge_exists": False,
                                "network_proximity": "high_rwr_support",
                                "allowed_claim": "The pair is network-proximal under RWR in this multiplex.",
                                "disallowed_claim": "Do not claim a direct interaction or causal relationship from RWR proximity.",
                            },
                            evidence=None,
                            fact_payload={"seed": seed_gene, "target": row.gene_id, "rank": row.rank, "edge": False},
                            strongest_group_id=f"rwr_seed:{seed_gene}",
                            evidence_handles=[f"rank_vector:{seed_gene}"],
                            provenance={
                                **self._base_provenance("rwr_rank_cache_and_full_store"),
                                **self.rwr.public_provenance(),
                            },
                            coverage={
                                "canonical_genes": [seed_gene, row.gene_id],
                                "negative_edges": [f"aggregate|{seed_gene}|{row.gene_id}"],
                                "rwr_seed_sets": [seed_gene],
                                "rwr_rank_facts": [f"{seed_gene}|{row.gene_id}|{row.rank}"],
                            },
                        )
                    )
                    emitted += 1
                    if emitted >= proximity_goal:
                        break
            if emitted >= proximity_goal:
                break
        if emitted < proximity_goal:
            raise ValueError("Could not fill calibrated no-edge/high-RWR examples.")

        outside_goal = self.candidate_goal("rwr_outside_top_k_calibration")
        for index in range(outside_goal):
            seed_gene = seeds[index % len(seeds)]
            variant = index // len(seeds)
            vector = self.rwr.rank_vector(seed_gene)
            row = vector.rows[min(99 + variant * 17 + index % 11, len(vector.rows) - 1)]
            self.add(
                CurriculumExample(
                    family=self.family("rwr_outside_top_k_calibration"),
                    book_mode="closed_book",
                    task=(
                        f"`{row.gene_id}` is not shown in the top-20 preview for RWR seed `{seed_gene}`. "
                        "Return the calibrated conclusion when the full vector has not been inspected."
                    ),
                    answer={
                        "seed_gene_id": seed_gene,
                        "target_gene_id": row.gene_id,
                        "top_k": 20,
                        "visible_in_top_k": False,
                        "full_vector_checked": False,
                        "allowed_claim": "The target is absent from the displayed top-k only.",
                        "disallowed_claim": "Do not claim the target is absent from the complete rank vector.",
                    },
                    evidence=None,
                    fact_payload={"seed": seed_gene, "target": row.gene_id, "top_k": 20},
                    strongest_group_id=f"rwr_seed:{seed_gene}",
                    provenance={**self._base_provenance("rwr_calibration_rule"), **self.rwr.public_provenance()},
                    coverage={"canonical_genes": [seed_gene, row.gene_id], "rwr_seed_sets": [seed_gene]},
                    polarity="insufficient_context",
                )
            )

        phrase_goal = self.candidate_goal("phrase_only_evidence_calibration")
        phrases = (
            "shares a pathway",
            "is related to",
            "appears mechanistically plausible",
            "has a similar description",
            "is mentioned with",
            "may influence",
        )
        phrase_genes = deterministic_order(
            self.oracle.gene_ids,
            seed=self.seed,
            namespace="phrase_only_calibration_genes",
        )
        for index in range(phrase_goal):
            phrase = phrases[index % len(phrases)]
            gene_a = phrase_genes[(index * 2) % len(phrase_genes)]
            gene_b = phrase_genes[(index * 2 + 1) % len(phrase_genes)]
            self.add(
                CurriculumExample(
                    family=self.family("phrase_only_evidence_calibration"),
                    book_mode="closed_book",
                    task=(
                        f"The only evidence offered for `{gene_a}` and `{gene_b}` is the phrase `{phrase}`, "
                        "with no exact graph, set, vector, matrix, or tool support. Return the permitted "
                        "structured relationship status as JSON."
                    ),
                    answer={
                        "evidence_type": "phrase_only",
                        "query_gene_ids": [gene_a, gene_b],
                        "relationship_status": "unknown",
                        "validated_group_allowed": False,
                        "reason": "String plausibility is not exact graph or module evidence.",
                    },
                    evidence=None,
                    fact_payload={"phrase": phrase, "query_gene_ids": [gene_a, gene_b]},
                    strongest_group_id=f"phrase_claim:{gene_a}:{gene_b}",
                    provenance=self._base_provenance("calibration_rule"),
                    coverage={"canonical_genes": [gene_a, gene_b]},
                    polarity="insufficient_context",
                )
            )

    def generate_stage3_topology(self) -> None:
        self._generate_layer_membership_examples()
        self._generate_edge_neighbor_examples()
        layer_paths, aggregate_paths = self._build_path_pools()
        self._generate_path_examples(layer_paths, aggregate_paths)
        self._generate_subgraph_examples(layer_paths)

    def _layer_iteration(self, namespace: str) -> list[str]:
        return deterministic_order(self.oracle.layer_names, seed=self.seed, namespace=namespace)

    def _generate_layer_membership_examples(self) -> None:
        gene_goal = self.candidate_goal("gene_layer_membership")
        node_indices = self.oracle.sample_node_indices(
            layer=None,
            count=gene_goal,
            seed=self.seed + 800,
            minimum_degree=1,
        )
        for node_index in node_indices:
            gene_id = self.oracle.gene_ids[node_index]
            layers = self.oracle.gene_layers(gene_id)
            if not layers:
                continue
            page_size = 12
            page_count = math.ceil(len(layers) / page_size)
            page_index = int(stable_hash({"gene": gene_id, "seed": self.seed})[:8], 16) % page_count
            start = page_index * page_size
            page_layers = layers[start : start + page_size]
            evidence = {
                "gene_id": gene_id,
                "layer_count": len(layers),
                "page_index": page_index,
                "page_count": page_count,
                "page_layers": page_layers,
                "page_rule": "ordered_manifest_page",
            }
            self.add(
                CurriculumExample(
                    family=self.family("gene_layer_membership"),
                    book_mode="open_book",
                    task="Return the exact layer-membership page and total layer count from the evidence as JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload={"gene_id": gene_id, "layers": layers, "page_index": page_index},
                    strongest_group_id=f"entity:{gene_id}",
                    layer_scope="all_layers",
                    layer_ids=page_layers,
                    layer_families=[layer_family(layer) for layer in page_layers],
                    context_budget_profile="evidence_2k",
                    evidence_handles=[f"csr_gene_layers:{gene_id}:page:{page_index}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": [gene_id],
                        "layers": page_layers,
                        "gene_layer_pairs": [f"{gene_id}|{layer}" for layer in page_layers],
                    },
                    page={"index": page_index, "count": page_count, "size": page_size},
                )
            )

        layer_goal = self.candidate_goal("nodes_by_layer")
        absent_goal = self.candidate_goal("gene_absent_from_layer")
        emitted_nodes = 0
        emitted_absent = 0
        for layer_index, layer in enumerate(self._layer_iteration("membership_layers")):
            view = self.oracle.view(layer)
            present = view.nonisolated_node_indices()
            absent = np.flatnonzero(np.diff(view.indptr) == 0)
            if not len(present) or not len(absent):
                continue
            rng = np.random.default_rng(self.seed + 900 + layer_index)
            repetitions = max(
                math.ceil(layer_goal / self.oracle.layer_count),
                math.ceil(absent_goal / self.oracle.layer_count),
                1,
            )
            for _ in range(repetitions + 1):
                if emitted_nodes < layer_goal:
                    present_ids = [
                        self.oracle.gene_ids[int(index)]
                        for index in rng.choice(present, size=min(4, len(present)), replace=False)
                    ]
                    absent_ids = [
                        self.oracle.gene_ids[int(index)]
                        for index in rng.choice(absent, size=min(4, len(absent)), replace=False)
                    ]
                    query = deterministic_order(
                        present_ids + absent_ids,
                        seed=self.seed,
                        namespace=f"nodes:{layer}:{emitted_nodes}",
                    )
                    evidence = {
                        "layer_id": layer,
                        "query_gene_ids": query,
                        "node_presence": {gene: gene in set(present_ids) for gene in query},
                    }
                    answer = {
                        "layer_id": layer,
                        "present_gene_ids": sorted(present_ids),
                        "absent_gene_ids": sorted(absent_ids),
                        "present_count": len(present_ids),
                        "absent_count": len(absent_ids),
                    }
                    self.add(
                        CurriculumExample(
                            family=self.family("nodes_by_layer"),
                            book_mode="open_book",
                            task="Partition the query genes into present and absent sets for the supplied layer. Return JSON.",
                            answer=answer,
                            evidence=evidence,
                            fact_payload=answer,
                            strongest_group_id=f"layer_gene_set:{layer}:{stable_hash(query)[:12]}",
                            layer_scope="single_layer",
                            layer_ids=[layer],
                            layer_families=[layer_family(layer)],
                            context_budget_profile="atomic_1k",
                            evidence_handles=[f"csr_node_presence:{layer}:{stable_hash(query)[:12]}"],
                            provenance=self._base_provenance("full_store_csr"),
                            coverage={
                                "canonical_genes": query,
                                "layers": [layer],
                                "gene_layer_pairs": [f"{gene}|{layer}" for gene in query],
                            },
                        )
                    )
                    emitted_nodes += 1
                if emitted_absent < absent_goal:
                    gene_id = self.oracle.gene_ids[int(rng.choice(absent))]
                    evidence = {
                        "gene_id": gene_id,
                        "layer_id": layer,
                        "node_present": False,
                        "store_id": self.store_id,
                    }
                    self.add(
                        CurriculumExample(
                            family=self.family("gene_absent_from_layer"),
                            book_mode="open_book",
                            task="State the exact layer-presence result and its calibrated scope. Return JSON.",
                            answer={
                                **evidence,
                                "allowed_claim": "The gene is not recorded as a node in this layer and graph version.",
                                "disallowed_claim": "Do not infer that the gene is biologically absent from the tissue or cell type.",
                            },
                            evidence=evidence,
                            fact_payload=evidence,
                            strongest_group_id=f"gene_layer:{gene_id}:{layer}",
                            layer_scope="single_layer",
                            layer_ids=[layer],
                            layer_families=[layer_family(layer)],
                            context_budget_profile="atomic_1k",
                            evidence_handles=[f"csr_node_presence:{layer}:{gene_id}"],
                            provenance=self._base_provenance("full_store_csr"),
                            coverage={
                                "canonical_genes": [gene_id],
                                "layers": [layer],
                                "gene_layer_pairs": [f"{gene_id}|{layer}"],
                            },
                            polarity="negative",
                        )
                    )
                    emitted_absent += 1
                if emitted_nodes >= layer_goal and emitted_absent >= absent_goal:
                    break
            if emitted_nodes >= layer_goal and emitted_absent >= absent_goal:
                break
        if emitted_nodes < layer_goal or emitted_absent < absent_goal:
            raise ValueError("Layer membership generators underfilled required families.")

    def _build_layer_edge_pool(self, goal: int) -> list[tuple[str, EdgeFact]]:
        pool: list[tuple[str, EdgeFact]] = []
        per_layer = max(2, math.ceil(goal / self.oracle.layer_count))
        for layer_index, layer in enumerate(self._layer_iteration("edge_layers")):
            edges = self.oracle.sample_edges(
                layer=layer,
                count=per_layer,
                seed=self.seed + 10_000 + layer_index,
            )
            pool.extend((layer, edge) for edge in edges)
            if len(pool) >= goal:
                break
        return pool[:goal]

    def _generate_edge_neighbor_examples(self) -> None:
        max_goal = max(
            self.candidate_goal("layer_edge_existence"),
            self.candidate_goal("multiplex_edge_existence"),
            self.candidate_goal("layer_direct_neighbors"),
            self.candidate_goal("multiplex_direct_neighbors"),
        )
        edge_pool = self._build_layer_edge_pool(max_goal)
        if len(edge_pool) < max_goal:
            raise ValueError("Full-store sampler could not fill the layer-edge pool.")

        layer_edge_goal = self.candidate_goal("layer_edge_existence")
        for index in range(layer_edge_goal):
            layer, edge = edge_pool[index]
            if index % 2 == 0:
                exists = True
                source, target = edge.source_gene_id, edge.target_gene_id
                weight = edge.weight
            else:
                nonedges: list[tuple[str, str]] = []
                # A few tiny layers are complete on their non-isolated nodes.
                # Deterministically route a negative query to the next layer
                # with a real present-node nonedge instead of dropping a row.
                for offset in range(len(edge_pool)):
                    candidate_layer, _ = edge_pool[(index + offset) % len(edge_pool)]
                    nonedges = self.oracle.sample_nonedges(
                        layer=candidate_layer,
                        count=1,
                        seed=self.seed + 20_000 + index + offset,
                        require_present=True,
                    )
                    if nonedges:
                        layer = candidate_layer
                        break
                if not nonedges:
                    raise ValueError("No layer with a present-node nonedge was available.")
                source, target = nonedges[0]
                exists = False
                weight = None
            evidence = {
                "source_gene_id": source,
                "target_gene_id": target,
                "layer_id": layer,
                "edge_exists": exists,
                "weight": weight,
            }
            answer = {key: value for key, value in evidence.items() if value is not None}
            if not exists:
                answer["allowed_claim"] = "No edge is recorded for this pair in this layer and graph version."
            self.add(
                CurriculumExample(
                    family=self.family("layer_edge_existence"),
                    book_mode="open_book",
                    task="Return the exact layer-specific edge-existence result from the CSR evidence as JSON.",
                    answer=answer,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"edge_pair:{source}:{target}",
                    layer_scope="single_layer",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="atomic_1k",
                    evidence_handles=[f"csr_edge:{layer}:{source}:{target}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": [source, target],
                        "layers": [layer],
                        "positive_edges" if exists else "negative_edges": [f"{layer}|{source}|{target}"],
                    },
                    polarity="positive" if exists else "negative",
                )
            )

        multiplex_goal = self.candidate_goal("multiplex_edge_existence")
        aggregate_edges = self.oracle.sample_edges(
            layer=None,
            count=math.ceil(multiplex_goal / 2),
            seed=self.seed + 30_000,
        )
        aggregate_nonedges = self.oracle.sample_nonedges(
            layer=None,
            count=math.ceil(multiplex_goal / 2),
            seed=self.seed + 31_000,
            require_present=True,
        )
        positives = iter(aggregate_edges)
        negatives = iter(aggregate_nonedges)
        for index in range(multiplex_goal):
            if index % 2 == 0:
                edge = next(positives)
                source, target, exists = edge.source_gene_id, edge.target_gene_id, True
                supporting = self.oracle.edge_layers(source, target)
            else:
                source, target = next(negatives)
                exists, supporting = False, []
            page_size = 12
            page = supporting[:page_size]
            evidence = {
                "source_gene_id": source,
                "target_gene_id": target,
                "edge_exists": exists,
                "supporting_layer_count": len(supporting),
                "supporting_layers_page": page,
                "omitted_supporting_layer_count": max(0, len(supporting) - len(page)),
            }
            self.add(
                CurriculumExample(
                    family=self.family("multiplex_edge_existence"),
                    book_mode="open_book",
                    task="Return aggregate edge existence and the exact bounded support-layer page. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload={"source": source, "target": target, "supporting_layers": supporting},
                    strongest_group_id=f"edge_pair:{source}:{target}",
                    layer_scope="all_layers",
                    layer_ids=[row["layer_id"] for row in page],
                    layer_families=[layer_family(row["layer_id"]) for row in page],
                    context_budget_profile="matrix_state_4k",
                    evidence_handles=[f"csr_aggregate_edge:{source}:{target}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": [source, target],
                        "layers": [row["layer_id"] for row in page],
                        "positive_edges" if exists else "negative_edges": [f"aggregate|{source}|{target}"],
                    },
                    polarity="positive" if exists else "negative",
                    page={"index": 0, "size": page_size, "total_items": len(supporting)},
                )
            )

        direct_goal = self.candidate_goal("layer_direct_neighbors")
        for index in range(direct_goal):
            layer, edge = edge_pool[index]
            gene_id = edge.source_gene_id if index % 2 == 0 else edge.target_gene_id
            degree = self.oracle.degree(gene_id, layer=layer)
            page = self.oracle.neighbors(gene_id, layer=layer, limit=10)
            evidence = {
                "gene_id": gene_id,
                "layer_id": layer,
                "neighbor_count": degree,
                "neighbor_page": page,
                "omitted_neighbor_count": max(0, degree - len(page)),
                "page_rule": "csr_order_prefix",
            }
            self.add(
                CurriculumExample(
                    family=self.family("layer_direct_neighbors"),
                    book_mode="open_book",
                    task="Return the exact neighbor total and bounded neighbor page from the layer evidence. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"gene_layer:{gene_id}:{layer}",
                    layer_scope="single_layer",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="matrix_state_4k",
                    evidence_handles=[f"csr_neighbors:{layer}:{gene_id}:page:0"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": [gene_id] + [row["gene_id"] for row in page],
                        "layers": [layer],
                        "gene_layer_pairs": [f"{gene_id}|{layer}"],
                    },
                    page={"index": 0, "size": 10, "total_items": degree},
                )
            )

        multiplex_neighbor_goal = self.candidate_goal("multiplex_direct_neighbors")
        aggregate_nodes = self.oracle.sample_node_indices(
            layer=None,
            count=multiplex_neighbor_goal,
            seed=self.seed + 40_000,
            minimum_degree=1,
        )
        for node_index in aggregate_nodes:
            gene_id = self.oracle.gene_ids[node_index]
            total = self.oracle.degree(gene_id, layer=None)
            neighbors = self.oracle.neighbors(gene_id, layer=None, limit=4)
            neighbor_map = {
                row["gene_id"]: self.oracle.edge_layers(gene_id, row["gene_id"])[:8]
                for row in neighbors
            }
            evidence = {
                "gene_id": gene_id,
                "unique_neighbor_count": total,
                "neighbor_layer_map_page": neighbor_map,
                "omitted_neighbor_count": max(0, total - len(neighbor_map)),
                "page_rule": "csr_order_prefix",
            }
            layer_ids = [
                row["layer_id"]
                for rows in neighbor_map.values()
                for row in rows
            ]
            self.add(
                CurriculumExample(
                    family=self.family("multiplex_direct_neighbors"),
                    book_mode="open_book",
                    task="Deduplicate the bounded aggregate neighbor evidence and preserve layer provenance. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"entity:{gene_id}",
                    layer_scope="all_layers",
                    layer_ids=layer_ids,
                    layer_families=[layer_family(layer) for layer in layer_ids],
                    context_budget_profile="matrix_state_4k",
                    evidence_handles=[f"csr_multiplex_neighbors:{gene_id}:page:0"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": [gene_id] + list(neighbor_map),
                        "layers": layer_ids,
                    },
                    page={"index": 0, "size": 4, "total_items": total},
                )
            )

    def _build_path_pools(self) -> tuple[list[tuple[str, list[str]]], list[list[str]]]:
        layer_goal = max(
            self.candidate_goal("monoplex_shortest_path"),
            self.candidate_goal("monoplex_multiplex_path_comparison"),
            self.candidate_goal("shared_neighbors"),
            self.candidate_goal("induced_subgraph"),
        )
        layer_paths: list[tuple[str, list[str]]] = []
        # Sparse layers do not always yield every requested nontrivial path.
        # Ask for a buffered number per layer so the global pool cannot hinge
        # on near-perfect yield from all 358 layers.
        per_layer = max(4, math.ceil(layer_goal / self.oracle.layer_count) * 2)
        for layer_index, layer in enumerate(self._layer_iteration("path_layers")):
            paths = self.oracle.sample_nontrivial_paths(
                layer=layer,
                count=per_layer,
                seed=self.seed + 50_000 + layer_index,
                minimum_hops=2,
                maximum_hops=4,
            )
            layer_paths.extend((layer, path) for path in paths)
            if len(layer_paths) >= layer_goal:
                break
        if len(layer_paths) < layer_goal:
            raise ValueError(
                f"Nontrivial monoplex path pool underfilled: {len(layer_paths)}/{layer_goal}."
            )

        aggregate_goal = max(
            self.candidate_goal("multiplex_shortest_path"),
            self.candidate_goal("path_layer_decomposition"),
        )
        aggregate_paths = self.oracle.sample_nontrivial_paths(
            layer=None,
            count=aggregate_goal,
            seed=self.seed + 55_000,
            minimum_hops=2,
            maximum_hops=3,
        )
        if len(aggregate_paths) < aggregate_goal:
            raise ValueError(
                f"Nontrivial aggregate path pool underfilled: {len(aggregate_paths)}/{aggregate_goal}."
            )
        return layer_paths[:layer_goal], aggregate_paths[:aggregate_goal]

    def _generate_path_examples(
        self,
        layer_paths: list[tuple[str, list[str]]],
        aggregate_paths: list[list[str]],
    ) -> None:
        mono_goal = self.candidate_goal("monoplex_shortest_path")
        for layer, path in layer_paths[:mono_goal]:
            evidence = {
                "layer_id": layer,
                "source_gene_id": path[0],
                "target_gene_id": path[-1],
                "path_gene_ids": path,
                "hop_count": len(path) - 1,
                "path_is_shortest": True,
            }
            self.add(
                CurriculumExample(
                    family=self.family("monoplex_shortest_path"),
                    book_mode="open_book",
                    task="Return the exact shortest path and hop count from the supplied layer-path evidence. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"path:{path[0]}:{path[-1]}",
                    layer_scope="single_layer",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="evidence_2k",
                    evidence_handles=[f"csr_shortest_path:{layer}:{path[0]}:{path[-1]}"],
                    provenance=self._base_provenance("full_store_csr_bfs"),
                    coverage={
                        "canonical_genes": path,
                        "layers": [layer],
                        "path_endpoint_pairs": [f"{path[0]}|{path[-1]}"],
                    },
                )
            )

        multiplex_goal = self.candidate_goal("multiplex_shortest_path")
        for path in aggregate_paths[:multiplex_goal]:
            evidence = {
                "scope": "aggregate_multiplex",
                "source_gene_id": path[0],
                "target_gene_id": path[-1],
                "path_gene_ids": path,
                "hop_count": len(path) - 1,
                "path_is_shortest": True,
            }
            self.add(
                CurriculumExample(
                    family=self.family("multiplex_shortest_path"),
                    book_mode="open_book",
                    task="Return the exact aggregate-multiplex shortest path from the supplied evidence. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"path:{path[0]}:{path[-1]}",
                    context_budget_profile="evidence_2k",
                    evidence_handles=[f"csr_shortest_path:aggregate:{path[0]}:{path[-1]}"],
                    provenance=self._base_provenance("full_store_csr_bfs"),
                    coverage={
                        "canonical_genes": path,
                        "path_endpoint_pairs": [f"{path[0]}|{path[-1]}"],
                    },
                )
            )

        decomposition_goal = self.candidate_goal("path_layer_decomposition")
        for path in aggregate_paths[:decomposition_goal]:
            edges = []
            layer_counts: Counter[str] = Counter()
            all_layers: list[str] = []
            for source, target in zip(path[:-1], path[1:]):
                supporting = self.oracle.edge_layers(source, target)
                layer_ids = [row["layer_id"] for row in supporting]
                all_layers.extend(layer_ids)
                layer_counts.update(layer_ids)
                edges.append(
                    {
                        "source_gene_id": source,
                        "target_gene_id": target,
                        "supporting_layer_count": len(layer_ids),
                        "supporting_layers_page": supporting[:8],
                        "omitted_supporting_layer_count": max(0, len(supporting) - 8),
                    }
                )
            layer_count_rows = [
                {"layer_id": layer, "path_edge_count": count}
                for layer, count in sorted(layer_counts.items())
            ]
            layer_count_page = layer_count_rows[:16]
            evidence = {
                "path_gene_ids": path,
                "hop_count": len(path) - 1,
                "path_edges": edges,
                "distinct_supporting_layer_count": len(layer_count_rows),
                "layer_count_page": layer_count_page,
                "omitted_layer_count": max(0, len(layer_count_rows) - len(layer_count_page)),
                "page_rule": "canonical_layer_order_prefix",
            }
            self.add(
                CurriculumExample(
                    family=self.family("path_layer_decomposition"),
                    book_mode="open_book",
                    task=(
                        "Return the exact bounded layer-decomposition page and declared totals for the "
                        "aggregate path. Return JSON."
                    ),
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"path:{path[0]}:{path[-1]}",
                    layer_scope="all_layers",
                    layer_ids=[row["layer_id"] for row in layer_count_page],
                    layer_families=[layer_family(row["layer_id"]) for row in layer_count_page],
                    context_budget_profile="matrix_state_4k",
                    evidence_handles=[f"csr_path_layers:{path[0]}:{path[-1]}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": path,
                        "layers": [row["layer_id"] for row in layer_count_page],
                        "path_endpoint_pairs": [f"{path[0]}|{path[-1]}"],
                    },
                    page={
                        "index": 0,
                        "size": 16,
                        "total_items": len(layer_count_rows),
                    },
                )
            )

        compare_goal = self.candidate_goal("monoplex_multiplex_path_comparison")
        for layer, monoplex_path in layer_paths[:compare_goal]:
            aggregate_path = self.oracle.shortest_path(monoplex_path[0], monoplex_path[-1], layer=None, max_hops=4)
            if aggregate_path is None:
                continue
            evidence = {
                "source_gene_id": monoplex_path[0],
                "target_gene_id": monoplex_path[-1],
                "layer_id": layer,
                "monoplex_path": monoplex_path,
                "monoplex_hops": len(monoplex_path) - 1,
                "multiplex_path": aggregate_path,
                "multiplex_hops": len(aggregate_path) - 1,
                "multiplex_is_shorter": len(aggregate_path) < len(monoplex_path),
            }
            self.add(
                CurriculumExample(
                    family=self.family("monoplex_multiplex_path_comparison"),
                    book_mode="open_book",
                    task="Compare the supplied monoplex and aggregate shortest paths without inferring causality. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"path:{monoplex_path[0]}:{monoplex_path[-1]}",
                    layer_scope="layer_subset",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="evidence_2k",
                    evidence_handles=[f"csr_path_compare:{layer}:{monoplex_path[0]}:{monoplex_path[-1]}"],
                    provenance=self._base_provenance("full_store_csr_bfs"),
                    coverage={
                        "canonical_genes": monoplex_path + aggregate_path,
                        "layers": [layer],
                        "path_endpoint_pairs": [f"{monoplex_path[0]}|{monoplex_path[-1]}"],
                    },
                )
            )

    def _generate_subgraph_examples(self, layer_paths: list[tuple[str, list[str]]]) -> None:
        induced_goal = self.candidate_goal("induced_subgraph")
        component_goal = self.candidate_goal("connected_component")
        shared_goal = self.candidate_goal("shared_neighbors")
        edge_pool = self._build_layer_edge_pool(component_goal * 3)

        for index, (layer, path) in enumerate(layer_paths[:induced_goal]):
            view = self.oracle.view(layer)
            extra_candidates = view.neighbor_indices(self.oracle.gene_to_index[path[1]])
            extra = next(
                (
                    self.oracle.gene_ids[int(raw)]
                    for raw in extra_candidates
                    if self.oracle.gene_ids[int(raw)] not in path
                ),
                None,
            )
            genes = sorted(set(path + ([extra] if extra else [])))
            edges = [edge.as_dict() for edge in self.oracle.induced_edges(genes, layer=layer)]
            evidence = {
                "query_gene_ids": genes,
                "layer_id": layer,
                "present_gene_ids": genes,
                "missing_gene_ids": [],
                "edge_count": len(edges),
                "edges": edges,
            }
            self.add(
                CurriculumExample(
                    family=self.family("induced_subgraph"),
                    book_mode="open_book",
                    task="Return every recorded edge induced by the supplied bounded gene set. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"subgraph:{layer}:{stable_hash(genes)[:12]}",
                    layer_scope="single_layer",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="matrix_state_4k",
                    evidence_handles=[f"csr_induced:{layer}:{stable_hash(genes)[:12]}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={
                        "canonical_genes": genes,
                        "layers": [layer],
                        "positive_edges": [
                            f"{layer}|{edge['source_gene_id']}|{edge['target_gene_id']}" for edge in edges
                        ],
                    },
                )
            )

        emitted = 0
        edges_by_layer: dict[str, list[EdgeFact]] = defaultdict(list)
        for layer, edge in edge_pool:
            edges_by_layer[layer].append(edge)
        for layer, edges in edges_by_layer.items():
            for left_index, left in enumerate(edges):
                for right in edges[left_index + 1 :]:
                    genes = sorted(
                        {
                            left.source_gene_id,
                            left.target_gene_id,
                            right.source_gene_id,
                            right.target_gene_id,
                        }
                    )
                    if len(genes) != 4:
                        continue
                    components = self.oracle.induced_components(genes, layer=layer)
                    if len(components) < 2:
                        continue
                    evidence = {
                        "query_gene_ids": genes,
                        "layer_id": layer,
                        "single_component": len(components) == 1,
                        "component_count": len(components),
                        "components": components,
                    }
                    self.add(
                        CurriculumExample(
                            family=self.family("connected_component"),
                            book_mode="open_book",
                            task="Return the exact induced connected components for the supplied gene set. Return JSON.",
                            answer=evidence,
                            evidence=evidence,
                            fact_payload=evidence,
                            strongest_group_id=f"subgraph:{layer}:{stable_hash(genes)[:12]}",
                            layer_scope="single_layer",
                            layer_ids=[layer],
                            layer_families=[layer_family(layer)],
                            context_budget_profile="evidence_2k",
                            evidence_handles=[f"csr_components:{layer}:{stable_hash(genes)[:12]}"],
                            provenance=self._base_provenance("full_store_csr"),
                            coverage={"canonical_genes": genes, "layers": [layer]},
                            polarity="negative",
                        )
                    )
                    emitted += 1
                    break
                if emitted >= component_goal:
                    break
            if emitted >= component_goal:
                break
        if emitted < component_goal:
            raise ValueError(f"Connected-component pool underfilled: {emitted}/{component_goal}.")

        for layer, path in layer_paths[:shared_goal]:
            source, target = path[0], path[-1]
            shared = self.oracle.common_neighbors(source, target, layer=layer)
            page = shared[:16]
            evidence = {
                "gene_a": source,
                "gene_b": target,
                "layer_id": layer,
                "shared_neighbor_count": len(shared),
                "shared_neighbors_page": page,
                "omitted_shared_neighbor_count": max(0, len(shared) - len(page)),
            }
            self.add(
                CurriculumExample(
                    family=self.family("shared_neighbors"),
                    book_mode="open_book",
                    task="Intersect the two layer-neighbor sets and return the exact bounded shared-neighbor result. Return JSON.",
                    answer=evidence,
                    evidence=evidence,
                    fact_payload=evidence,
                    strongest_group_id=f"edge_pair:{source}:{target}",
                    layer_scope="single_layer",
                    layer_ids=[layer],
                    layer_families=[layer_family(layer)],
                    context_budget_profile="evidence_2k",
                    evidence_handles=[f"csr_common_neighbors:{layer}:{source}:{target}"],
                    provenance=self._base_provenance("full_store_csr"),
                    coverage={"canonical_genes": [source, target] + page, "layers": [layer]},
                    page={"index": 0, "size": 16, "total_items": len(shared)},
                )
            )

    def generate_stage3_rwr_distance(self) -> None:
        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        seed_pool = self._balanced_rwr_seed_pool()
        if len(seed_pool) < 16:
            raise ValueError("RWR cache has too few seeds for the curriculum.")
        self._generate_rank_vector_examples(seed_pool)
        self._generate_distance_examples()
        self._generate_perturbation_semantics(seed_pool)

    def _rank_evidence_provenance(self) -> dict[str, Any]:
        if self.rwr is None:
            return self._base_provenance("rwr_rank_cache")
        return {**self._base_provenance("rwr_rank_cache"), **self.rwr.public_provenance()}

    def _generate_rank_vector_examples(self, seeds: list[str]) -> None:
        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        names = (
            "rwr_loe_rank_lookup",
            "rwr_loe_pair_comparison",
            "rwr_loe_closest_entities",
            "rwr_loe_query_filtering",
            "rwr_loe_elbow_membership",
            "rwr_loe_rank_gap",
            "rwr_loe_vector_intersection",
            "rwr_loe_leave_one_out_support",
        )
        goals = {name: self.candidate_goal(name) for name in names}
        max_goal = max(goals.values())
        vectors = {seed: self.rwr.rank_vector(seed) for seed in seeds}
        elbows = {seed: vectors[seed].elbow() for seed in seeds}

        for index in range(max_goal):
            seed = seeds[index % len(seeds)]
            vector = vectors[seed]
            variant = index // len(seeds)
            offset = 3 + (variant * 17 + index) % min(2000, len(vector.rows) - 10)

            if index < goals["rwr_loe_rank_lookup"]:
                row = vector.rows[offset]
                evidence = {
                    "seed_gene_ids": [seed],
                    "target_gene_id": row.gene_id,
                    "rank_vector_row": row.as_dict(),
                    "ranking_semantics": self.rwr.identity.ranking_semantics,
                }
                answer = {
                    "seed_gene_ids": [seed],
                    "target_gene_id": row.gene_id,
                    "rank": row.rank,
                    "score": row.score,
                    "rule": "Lower rank and higher score indicate stronger support within this RWR-LOE vector.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_rank_lookup"),
                        book_mode="open_book",
                        task="Extract the exact target rank and score from the supplied full-vector row. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"rwr_seed:{seed}",
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"rank_vector:{seed}:row:{row.rank}"],
                        provenance=self._rank_evidence_provenance(),
                        coverage={
                            "canonical_genes": [seed, row.gene_id],
                            "rwr_seed_sets": [seed],
                            "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}"],
                        },
                        validator={"type": "exact_json_with_float_tolerance", "float_fields": ["score"]},
                    )
                )

            if index < goals["rwr_loe_pair_comparison"]:
                left = vector.rows[offset]
                right = vector.rows[offset + 7]
                comparison = vector.compare(left.gene_id, right.gene_id)
                evidence = {
                    "seed_gene_ids": [seed],
                    "candidates": [left.as_dict(), right.as_dict()],
                }
                answer = {
                    "seed_gene_ids": [seed],
                    "closer_gene_id": comparison.closer_gene_id,
                    "is_tie": comparison.is_tie,
                    "comparison": [left.as_dict(), right.as_dict()],
                    "rule": "Lower rank indicates stronger RWR-LOE proximity.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_pair_comparison"),
                        book_mode="open_book",
                        task="Compare the two supplied rank-vector rows and identify the lower-rank candidate. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"rwr_seed:{seed}",
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"rank_vector:{seed}:rows:{left.rank}:{right.rank}"],
                        provenance=self._rank_evidence_provenance(),
                        coverage={
                            "canonical_genes": [seed, left.gene_id, right.gene_id],
                            "rwr_seed_sets": [seed],
                            "rwr_rank_facts": [
                                f"{seed}|{left.gene_id}|{left.rank}",
                                f"{seed}|{right.gene_id}|{right.rank}",
                            ],
                        },
                        validator={"type": "rank_order_and_float_tolerance"},
                    )
                )

            if index < goals["rwr_loe_closest_entities"]:
                top_k = 3 + variant % 10
                rows = vector.top_k(top_k, exclude_genes=[seed])
                evidence = {
                    "seed_gene_ids": [seed],
                    "rank_vector_page": [row.as_dict() for row in vector.rows[: max(16, top_k + 3)]],
                    "exclude_seed_genes": True,
                    "top_k": top_k,
                }
                answer = {
                    "seed_gene_ids": [seed],
                    "exclude_seed_genes": True,
                    "top_k": top_k,
                    "closest_genes": [row.as_dict() for row in rows],
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_closest_entities"),
                        book_mode="open_book",
                        task="Sort the supplied rank-vector page and return the requested closest non-seed genes. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"rwr_seed:{seed}",
                        context_budget_profile="matrix_state_4k",
                        evidence_handles=[f"rank_vector:{seed}:top:{top_k}"],
                        provenance=self._rank_evidence_provenance(),
                        coverage={
                            "canonical_genes": [seed] + [row.gene_id for row in rows],
                            "rwr_seed_sets": [seed],
                            "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in rows],
                        },
                        validator={"type": "top_k", "metrics": self.plan["numeric_policy"]["top_k_metrics"]},
                    )
                )

            if index < goals["rwr_loe_query_filtering"]:
                query_rows = [
                    vector.rows[(offset + step) % len(vector.rows)]
                    for step in (0, 11, 43, 101, 211)
                ]
                query_ids = deterministic_order(
                    [row.gene_id for row in query_rows],
                    seed=self.seed,
                    namespace=f"query:{seed}:{variant}",
                )
                result = vector.filter_queries(query_ids)
                evidence = {
                    "seed_gene_ids": [seed],
                    "query_gene_ids": query_ids,
                    "matching_vector_rows": [row.as_dict() for row in query_rows],
                }
                answer = {
                    "seed_gene_ids": [seed],
                    "query_gene_ids": query_ids,
                    "ranked_query_genes": [row.as_dict() for row in result.ranked_query_genes],
                    "missing_gene_ids": list(result.missing_gene_ids),
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_query_filtering"),
                        book_mode="open_book",
                        task="Filter the rank vector to the supplied query IDs and preserve rank ordering. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"rwr_seed:{seed}",
                        context_budget_profile="evidence_2k",
                        evidence_handles=[f"rank_vector:{seed}:query:{stable_hash(query_ids)[:12]}"],
                        provenance=self._rank_evidence_provenance(),
                        coverage={
                            "canonical_genes": [seed] + query_ids,
                            "rwr_seed_sets": [seed],
                            "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in query_rows],
                        },
                        validator={"type": "rank_order_and_float_tolerance"},
                    )
                )

            elbow = elbows[seed]
            if elbow.elbow_rank_cutoff is not None:
                cutoff = int(elbow.elbow_rank_cutoff)
                window_start = max(0, cutoff - 4 - (variant % 3))
                window = vector.rows[window_start : min(len(vector.rows), cutoff + 3)]
                if index < goals["rwr_loe_elbow_membership"]:
                    retained = [row.gene_id for row in window if row.rank < cutoff]
                    excluded = [row.gene_id for row in window if row.rank >= cutoff]
                    evidence = {
                        "seed_gene_id": seed,
                        "elbow_rank_cutoff": cutoff,
                        "membership_rule": "rank < elbow_rank_cutoff",
                        "candidate_rows": [row.as_dict() for row in window],
                    }
                    answer = {
                        "seed_gene_id": seed,
                        "elbow_rank_cutoff": cutoff,
                        "membership_rule": "rank < elbow_rank_cutoff",
                        "retained_gene_ids": retained,
                        "excluded_gene_ids": excluded,
                    }
                    self.add(
                        CurriculumExample(
                            family=self.family("rwr_loe_elbow_membership"),
                            book_mode="open_book",
                            task="Apply the strict elbow cutoff to the supplied candidate rows. Return exact sets as JSON.",
                            answer=answer,
                            evidence=evidence,
                            fact_payload=answer,
                            strongest_group_id=f"rwr_seed:{seed}",
                            context_budget_profile="evidence_2k",
                            evidence_handles=[f"rank_vector:{seed}:elbow:{cutoff}"],
                            provenance=self._rank_evidence_provenance(),
                            coverage={
                                "canonical_genes": [seed] + retained + excluded,
                                "rwr_seed_sets": [seed],
                                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in window],
                            },
                            validator={"type": "recompute_strict_rank_cutoff"},
                        )
                    )
                if index < goals["rwr_loe_rank_gap"]:
                    evidence = {
                        "seed_gene_id": seed,
                        "ordered_score_curve_window": [row.as_dict() for row in window],
                        "geometric_elbow_rank_cutoff": cutoff,
                    }
                    answer = {
                        "seed_gene_id": seed,
                        "elbow_rank_cutoff": cutoff,
                        "high_score_side_gene_ids": [row.gene_id for row in window if row.rank < cutoff],
                        "low_score_side_gene_ids": [row.gene_id for row in window if row.rank >= cutoff],
                        "membership_rule": "Retain genes with rank lower than the elbow cutoff.",
                    }
                    self.add(
                        CurriculumExample(
                            family=self.family("rwr_loe_rank_gap"),
                            book_mode="open_book",
                            task="Use the declared geometric elbow and partition the supplied score-curve window. Return JSON.",
                            answer=answer,
                            evidence=evidence,
                            fact_payload=answer,
                            strongest_group_id=f"rwr_seed:{seed}",
                            context_budget_profile="evidence_2k",
                            evidence_handles=[f"rank_vector:{seed}:elbow_window:{cutoff}:{window_start}"],
                            provenance=self._rank_evidence_provenance(),
                            coverage={
                                "canonical_genes": [seed] + [row.gene_id for row in window],
                                "rwr_seed_sets": [seed],
                                "rwr_rank_facts": [f"{seed}|{row.gene_id}|{row.rank}" for row in window],
                            },
                            validator={"type": "recompute_strict_rank_cutoff"},
                        )
                    )

            if index < goals["rwr_loe_vector_intersection"]:
                other = seeds[(index * 7 + 1) % len(seeds)]
                if other == seed:
                    other = seeds[(index + 1) % len(seeds)]
                top_k = (10, 20, 50)[variant % 3]
                overlap = self.rwr.top_k_intersection(seed, other, top_k)
                evidence = {
                    "seed_gene_id_a": seed,
                    "seed_gene_id_b": other,
                    "top_k": top_k,
                    "neighborhood_a": list(overlap.neighborhood_a),
                    "neighborhood_b": list(overlap.neighborhood_b),
                }
                answer = {
                    "seed_set_a": [seed],
                    "seed_set_b": [other],
                    "top_k": top_k,
                    "intersection_gene_ids": list(overlap.intersection_gene_ids),
                    "intersection_size": overlap.intersection_size,
                    "union_size": overlap.union_size,
                    "jaccard": overlap.jaccard,
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_vector_intersection"),
                        book_mode="open_book",
                        task="Compute the exact top-k intersection, union size, and Jaccard from the two neighborhoods. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"rwr_seed_pair:{min(seed, other)}:{max(seed, other)}",
                        context_budget_profile="matrix_state_4k",
                        evidence_handles=[f"rank_intersection:{seed}:{other}:top:{top_k}"],
                        provenance=self._rank_evidence_provenance(),
                        coverage={
                            "canonical_genes": [seed, other] + list(overlap.intersection_gene_ids),
                            "rwr_seed_sets": [seed, other],
                        },
                        validator={"type": "recompute_set_metrics"},
                    )
                )

            if index < goals["rwr_loe_leave_one_out_support"]:
                # The current cache contains single-seed vectors, not executed
                # multi-seed leave-one-out runs.  This family therefore teaches
                # exact interpretation of an explicitly supplied LOO table and
                # is tagged as a rule-derived fixture rather than a graph fact.
                members = [row.gene_id for row in vector.rows[offset : offset + 4]]
                loo_rows = [
                    {
                        "held_out_gene_id": gene,
                        "loo_rank": 2 + ((variant + member_index * 37) % 5000),
                    }
                    for member_index, gene in enumerate(members)
                ]
                for row in loo_rows:
                    row["recommendation"] = "drop_candidate" if row["loo_rank"] == max(
                        item["loo_rank"] for item in loo_rows
                    ) else "keep"
                weakest = max(loo_rows, key=lambda row: (row["loo_rank"], row["held_out_gene_id"]))
                evidence = {
                    "evidence_type": "supplied_leave_one_out_rank_table",
                    "gene_set": members,
                    "support_table": loo_rows,
                    "rule": "Higher leave-one-out rank indicates weaker support from the remaining seed set.",
                }
                answer = {
                    "gene_set": members,
                    "least_supported_gene_id": weakest["held_out_gene_id"],
                    "support_table": loo_rows,
                    "interpretation": "The highest-rank held-out gene is the weakest topological fit in this table.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rwr_loe_leave_one_out_support"),
                        book_mode="open_book",
                        task="Identify the least-supported gene from the supplied leave-one-out rank table. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"loo_fixture:{stable_hash(members + [str(variant)])[:16]}",
                        context_budget_profile="evidence_2k",
                        evidence_handles=[f"loo_table_fixture:{stable_hash(loo_rows)[:16]}"],
                        provenance={
                            **self._base_provenance("rule_derived_numeric_fixture"),
                            "oracle": "supplied_structured_evidence",
                            "observed_multiplex_fact": False,
                        },
                        coverage={"canonical_genes": members},
                        validator={"type": "argmax_rank"},
                    )
                )

    def _distance_shard_pool(self, count: int = 8) -> list[Any]:
        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        shard_ids = deterministic_order(self.rwr.shard_ids, seed=self.seed, namespace="distance_shards")
        return [self.rwr.distance_shard(shard_id) for shard_id in shard_ids[:count]]

    def _generate_distance_examples(self) -> None:
        if self.rwr is None:
            raise RuntimeError("RWR source was not loaded.")
        shards = self._distance_shard_pool(12)
        names = (
            "distance_shard_pair_lookup",
            "distance_shard_comparison",
            "distance_row_closest_entities",
            "distance_cross_shard_lookup",
            "distance_percentile_calibration",
            "rank_distance_consistency",
        )
        goals = {name: self.candidate_goal(name) for name in names}
        max_goal = max(goals.values())
        for index in range(max_goal):
            shard = shards[index % len(shards)]
            variant = index // len(shards)
            anchor_index = (index * 17 + variant) % len(shard.genes)
            anchor = shard.genes[anchor_index]
            row = sorted(shard.row(anchor))
            if len(row) < 8:
                continue
            left = row[(variant * 5 + 1) % len(row)]
            right = row[(variant * 11 + len(row) // 2) % len(row)]
            if right.gene_id == left.gene_id:
                right = row[(row.index(right) + 1) % len(row)]

            if index < goals["distance_shard_pair_lookup"]:
                evidence = {
                    "context_type": "rwr_distance_matrix_shard",
                    "shard_id": shard.shard_id,
                    "distance_metric": "spearman_distance",
                    "gene_a": anchor,
                    "gene_b": left.gene_id,
                    "matrix_cell": left.distance,
                }
                answer = {
                    "context_type": "rwr_distance_matrix_shard",
                    "multiplex_id": self.multiplex_id,
                    "distance_metric": "spearman_distance",
                    "lower_is_closer": True,
                    "gene_a": anchor,
                    "gene_b": left.gene_id,
                    "distance": left.distance,
                }
                self.add(
                    CurriculumExample(
                        family=self.family("distance_shard_pair_lookup"),
                        book_mode="open_book",
                        task="Extract the exact pairwise distance from the supplied lower-triangle shard cell. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"distance_pair:{min(anchor, left.gene_id)}:{max(anchor, left.gene_id)}",
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"distance_shard:{shard.shard_id}:{anchor}:{left.gene_id}"],
                        provenance={**self._rank_evidence_provenance(), **dict(shard.provenance)},
                        coverage={"canonical_genes": [anchor, left.gene_id]},
                        validator={"type": "exact_json_with_float_tolerance", "float_fields": ["distance"]},
                    )
                )

            if index < goals["distance_shard_comparison"]:
                comparison = shard.compare(anchor, left.gene_id, right.gene_id)
                evidence = {
                    "anchor_gene_id": anchor,
                    "distance_row_cells": [left.as_dict(), right.as_dict()],
                    "distance_metric": "spearman_distance",
                }
                answer = {
                    "anchor_gene_id": anchor,
                    "candidate_a": left.as_dict(),
                    "candidate_b": right.as_dict(),
                    "closer_gene_id": comparison.closer_gene_id,
                    "is_tie": comparison.is_tie,
                    "rule": "Lower distance is closer.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("distance_shard_comparison"),
                        book_mode="open_book",
                        task="Compare the two supplied distance cells and identify the closer candidate. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"distance_anchor:{anchor}",
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"distance_shard:{shard.shard_id}:row:{anchor}"],
                        provenance={**self._rank_evidence_provenance(), **dict(shard.provenance)},
                        coverage={"canonical_genes": [anchor, left.gene_id, right.gene_id]},
                        validator={"type": "argmin_distance"},
                    )
                )

            if index < goals["distance_row_closest_entities"]:
                full_row = list(shard.row(anchor))
                page_size = 64
                page_count = math.ceil(len(full_row) / page_size)
                page_index = variant % page_count
                page_rows = full_row[page_index * page_size : (page_index + 1) * page_size]
                top_k = min(3 + variant % 8, len(page_rows))
                closest = tuple(sorted(page_rows))[:top_k]
                evidence = {
                    "anchor_gene_id": anchor,
                    "distance_metric": "spearman_distance",
                    "distance_row_page": [item.as_dict() for item in page_rows],
                    "page_index": page_index,
                    "page_count": page_count,
                    "top_k": top_k,
                    "exclude_self": True,
                }
                answer = {
                    "anchor_gene_id": anchor,
                    "distance_metric": "spearman_distance",
                    "exclude_self": True,
                    "top_k": top_k,
                    "selection_scope": "supplied_distance_row_page",
                    "page_index": page_index,
                    "closest_genes": [item.as_dict() for item in closest],
                }
                self.add(
                    CurriculumExample(
                        family=self.family("distance_row_closest_entities"),
                        book_mode="open_book",
                        task="Sort the supplied distance-row page and return its nearest non-self entities. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"distance_anchor:{anchor}",
                        context_budget_profile="matrix_state_4k",
                        evidence_handles=[f"distance_shard:{shard.shard_id}:row:{anchor}"],
                        provenance={**self._rank_evidence_provenance(), **dict(shard.provenance)},
                        coverage={"canonical_genes": [anchor] + [item.gene_id for item in closest]},
                        validator={"type": "top_k_distance_order"},
                        page={"index": page_index, "count": page_count, "size": page_size},
                    )
                )

            if index < goals["distance_percentile_calibration"]:
                values = sorted(item.distance for item in row)
                target = row[(variant * 13 + len(row) // 5) % len(row)]
                percentile = 100.0 * sum(value <= target.distance for value in values) / len(values)
                if percentile <= 5:
                    classification = "unusually_close"
                elif percentile >= 95:
                    classification = "far"
                else:
                    classification = "typical"
                evidence = {
                    "gene_a": anchor,
                    "gene_b": target.gene_id,
                    "distance": target.distance,
                    "reference_scope": f"complete_{shard.shard_id}_row",
                    "reference_count": len(values),
                    "distance_percentile": percentile,
                    "classification_thresholds": {"unusually_close_max": 5, "far_min": 95},
                }
                answer = {
                    "gene_a": anchor,
                    "gene_b": target.gene_id,
                    "distance": target.distance,
                    "distance_percentile": percentile,
                    "reference_scope": evidence["reference_scope"],
                    "classification": classification,
                    "rule": "Lower percentile indicates closer-than-typical proximity within the declared reference row.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("distance_percentile_calibration"),
                        book_mode="open_book",
                        task="Apply the supplied row-percentile thresholds and return the calibrated distance class. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"distance_anchor:{anchor}",
                        context_budget_profile="evidence_2k",
                        evidence_handles=[f"distance_shard:{shard.shard_id}:row:{anchor}:percentile"],
                        provenance={**self._rank_evidence_provenance(), **dict(shard.provenance)},
                        coverage={"canonical_genes": [anchor, target.gene_id]},
                        validator={"type": "recompute_percentile_and_bin"},
                    )
                )

            if index < goals["rank_distance_consistency"]:
                rank_row = self.rwr.rank(anchor, left.gene_id)
                evidence = {
                    "rank_vector_context": {
                        "seed_gene_id": anchor,
                        "target_gene_id": left.gene_id,
                        "rank": rank_row.rank,
                        "score": rank_row.score,
                    },
                    "distance_context": {
                        "gene_a": anchor,
                        "gene_b": left.gene_id,
                        "distance": left.distance,
                        "metric": "spearman_distance",
                        "shard_id": shard.shard_id,
                    },
                    "identity": {
                        "multiplex_id": self.multiplex_id,
                        "flist_id": self.flist_id,
                        "rank_cache_id": self.rwr.identity.cache_id,
                    },
                }
                answer = {
                    "identity_checks_pass": True,
                    "checks": [
                        "same_multiplex_id",
                        "same_flist_hash",
                        "same_rank_cache_context",
                        "declared_distance_metric",
                        "same_seed_entity",
                    ],
                    "allowed_claim": (
                        "The two values share artifact identity; any apparent discrepancy must still be interpreted under their different rank and correlation-distance semantics."
                    ),
                }
                self.add(
                    CurriculumExample(
                        family=self.family("rank_distance_consistency"),
                        book_mode="open_book",
                        task="Check whether the rank and distance evidence are context-compatible before interpreting them. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload={"anchor": anchor, "target": left.gene_id, "identity": evidence["identity"]},
                        strongest_group_id=f"distance_pair:{min(anchor, left.gene_id)}:{max(anchor, left.gene_id)}",
                        context_budget_profile="evidence_2k",
                        evidence_handles=[
                            f"rank_vector:{anchor}:row:{rank_row.rank}",
                            f"distance_shard:{shard.shard_id}:{anchor}:{left.gene_id}",
                        ],
                        provenance={**self._rank_evidence_provenance(), **dict(shard.provenance)},
                        coverage={
                            "canonical_genes": [anchor, left.gene_id],
                            "rwr_seed_sets": [anchor],
                            "rwr_rank_facts": [f"{anchor}|{left.gene_id}|{rank_row.rank}"],
                        },
                        validator={"type": "artifact_identity"},
                    )
                )

            if index < goals["distance_cross_shard_lookup"]:
                other_shard = shards[(index + 1) % len(shards)]
                other_gene = other_shard.genes[(variant * 19 + index) % len(other_shard.genes)]
                route = self.rwr.route_distance_pair(anchor, other_gene)
                if route.distance_available:
                    continue
                evidence = {
                    "requested_pair": [anchor, other_gene],
                    "current_shard_id": shard.shard_id,
                    "gene_a_shard_id": route.gene_a_shard_id,
                    "gene_b_shard_id": route.gene_b_shard_id,
                    "distance_available": route.distance_available,
                    "status": route.status,
                }
                answer = {
                    "requested_pair": [anchor, other_gene],
                    "current_shard_contains_pair": False,
                    "gene_a_shard_id": route.gene_a_shard_id,
                    "gene_b_shard_id": route.gene_b_shard_id,
                    "next_action": "request_cross_shard_distance_computation",
                    "reason": route.reason,
                }
                self.add(
                    CurriculumExample(
                        family=self.family("distance_cross_shard_lookup"),
                        book_mode="open_book",
                        task="Route the cross-shard pair honestly; do not invent a materialized distance. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"distance_pair:{min(anchor, other_gene)}:{max(anchor, other_gene)}",
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"distance_route:{anchor}:{other_gene}"],
                        provenance={**self._rank_evidence_provenance(), **dict(route.provenance)},
                        coverage={"canonical_genes": [anchor, other_gene]},
                        validator={"type": "shard_route"},
                        polarity="insufficient_context",
                    )
                )

    def _generate_perturbation_semantics(self, seeds: list[str]) -> None:
        layers = self._layer_iteration("perturbation_layers")
        layer_goal = self.candidate_goal("layer_ablation")
        node_goal = self.candidate_goal("node_perturbation_seed_essentiality")
        rank_buckets = ("top_10", "top_100", "top_1000", "outside_top_1000")
        for index in range(max(layer_goal, node_goal)):
            seed_a = seeds[index % len(seeds)]
            seed_b = seeds[(index * 7 + 1) % len(seeds)]
            layer = layers[index % len(layers)]
            before = rank_buckets[index % 2]
            after = rank_buckets[2 + (index % 2)]
            if index < layer_goal:
                evidence = {
                    "evidence_type": "supplied_layer_ablation_summary",
                    "seed_gene_ids": [seed_a, seed_b],
                    "ablated_layer": layer,
                    "before_proximity_bucket": before,
                    "after_proximity_bucket": after,
                    "effect": "proximity_worsened",
                }
                answer = {
                    "pair": [seed_a, seed_b],
                    "ablated_layer": layer,
                    "effect": "proximity_worsened",
                    "caveat": "The topological relationship is layer-sensitive and depends partly on the ablated layer.",
                }
                self.add(
                    CurriculumExample(
                        family=self.family("layer_ablation"),
                        book_mode="open_book",
                        task="Interpret the supplied ablation summary without making a causal claim. Return JSON.",
                        answer=answer,
                        evidence=evidence,
                        fact_payload=answer,
                        strongest_group_id=f"ablation_fixture:{seed_a}:{seed_b}:{layer}",
                        layer_scope="layer_subset",
                        layer_ids=[layer],
                        layer_families=[layer_family(layer)],
                        context_budget_profile="atomic_1k",
                        evidence_handles=[f"ablation_summary_fixture:{seed_a}:{seed_b}:{layer}"],
                        provenance={
                            **self._base_provenance("rule_derived_perturbation_fixture"),
                            "oracle": "supplied_structured_evidence",
                            "observed_multiplex_fact": False,
                        },
                        coverage={"canonical_genes": [seed_a, seed_b], "layers": [layer]},
                        validator={"type": "calibration_rule"},
                    )
                )

        # The available cache does not contain an exhaustive node-removal
        # experiment.  Keep this semantic exercise explicitly fixture-backed;
        # it teaches how to interpret a supplied perturbation result without
        # pretending the result was observed in the full multiplex.
        for index in range(node_goal):
            seed_a = seeds[index % len(seeds)]
            shift = ("small", "moderate", "large")[index % 3]
            evidence = {
                "evidence_type": "supplied_seed_perturbation_summary",
                "seed_gene_id": seed_a,
                "removal_rank_vector_shift": shift,
            }
            description = (
                "seed_essential"
                if shift == "large"
                else "seed_influential"
                if shift == "moderate"
                else "seed_robust"
            )
            answer = {
                "gene_id": seed_a,
                "effect": f"{shift}_rank_vector_shift",
                "description": description,
                "allowed_claim": (
                    "The description applies to this supplied RWR perturbation context, "
                    "not general biological causality."
                ),
            }
            self.add(
                CurriculumExample(
                    family=self.family("node_perturbation_seed_essentiality"),
                    book_mode="open_book",
                    task=(
                        "Classify seed influence from the supplied perturbation shift and "
                        "preserve the causal caveat. Return JSON."
                    ),
                    answer=answer,
                    evidence=evidence,
                    fact_payload=answer,
                    strongest_group_id=f"seed_perturbation_fixture:{seed_a}:{index // len(seeds)}",
                    context_budget_profile="atomic_1k",
                    evidence_handles=[
                        f"seed_perturbation_fixture:{seed_a}:{index // len(seeds)}"
                    ],
                    provenance={
                        **self._base_provenance("rule_derived_perturbation_fixture"),
                        "oracle": "supplied_structured_evidence",
                        "observed_multiplex_fact": False,
                    },
                    coverage={"canonical_genes": [seed_a], "rwr_seed_sets": [seed_a]},
                    validator={"type": "calibration_rule"},
                )
            )

    @staticmethod
    def _module_source_label(module: Mapping[str, Any]) -> str:
        return "mentor_ev" if module.get("source") == "MENTOR_GW_DENDROGRAM" else "rwr_loe"

    def _module_pair_pool(self, goal: int) -> list[dict[str, Any]]:
        mentors = self.modules_by_source["MENTOR_GW_DENDROGRAM"]
        rwr_modules = self.modules_by_source["RWR_LOE_FULL_BRAIN"]
        rwr_by_seed = {
            str(module.get("seed_gene_id")): module
            for module in rwr_modules
            if isinstance(module.get("seed_gene_id"), str)
        }
        pairs: list[dict[str, Any]] = []
        for mentor in mentors:
            mentor_genes = set(mentor["gene_ids"])
            candidates: list[dict[str, Any]] = []
            seen: set[str] = set()
            for gene_id in mentor["gene_ids"]:
                rwr_module = rwr_by_seed.get(gene_id)
                if rwr_module is None or rwr_module["module_id"] in seen:
                    continue
                seen.add(str(rwr_module["module_id"]))
                rwr_genes = set(rwr_module["gene_ids"])
                intersection = mentor_genes & rwr_genes
                union = mentor_genes | rwr_genes
                if not intersection:
                    continue
                candidates.append(
                    {
                        "mentor": mentor,
                        "rwr": rwr_module,
                        "intersection": sorted(intersection),
                        "mentor_only": sorted(mentor_genes - rwr_genes),
                        "rwr_only": sorted(rwr_genes - mentor_genes),
                        "jaccard": len(intersection) / len(union),
                        "mentor_containment": len(intersection) / len(mentor_genes),
                        "rwr_containment": len(intersection) / len(rwr_genes),
                    }
                )
            candidates.sort(
                key=lambda row: (
                    -len(row["intersection"]),
                    -row["jaccard"],
                    str(row["rwr"]["module_id"]),
                )
            )
            pairs.extend(candidates[:3])
            if len(pairs) >= goal * 3:
                break
        pairs = deterministic_order(pairs, seed=self.seed, namespace="module_pairs")
        if len(pairs) < goal:
            raise ValueError(f"Module relation pool underfilled: {len(pairs)}/{goal}.")
        return pairs

    @staticmethod
    def _module_page(module: Mapping[str, Any], *, page_size: int, variant: int) -> dict[str, Any]:
        genes = list(module["gene_ids"])
        page_count = max(1, math.ceil(len(genes) / page_size))
        page_index = variant % page_count
        page = genes[page_index * page_size : (page_index + 1) * page_size]
        return {
            "module_id": module["module_id"],
            "module_source": CurriculumBuilder._module_source_label(module),
            "gene_count": len(genes),
            "page_index": page_index,
            "page_count": page_count,
            "gene_ids_page": page,
            "omitted_gene_count": len(genes) - len(page),
        }

    def generate_stage4_modules(self) -> None:
        self._generate_module_membership_and_provenance()
        self._generate_cross_source_module_algebra()
        self._generate_dendrogram_relations()

    def _generate_module_membership_and_provenance(self) -> None:
        source_by_family = {
            "mentor_ev_module_membership": "MENTOR_GW_DENDROGRAM",
            "rwr_loe_module_membership": "RWR_LOE_FULL_BRAIN",
        }
        for family_name, source in source_by_family.items():
            goal = self.candidate_goal(family_name)
            modules = self.modules_by_source[source]
            for index in range(goal):
                module = modules[index % len(modules)]
                variant = index // len(modules)
                page = self._module_page(module, page_size=16, variant=variant)
                if source == "RWR_LOE_FULL_BRAIN":
                    selection = module.get("module_selection", {})
                    page["seed_gene_id"] = module.get("seed_gene_id")
                    page["membership_rule"] = selection.get("retention_rule", "rank < elbow_rank_cutoff")
                    cutoff = selection.get("elbow_rank_cutoff")
                    page["elbow_rank_cutoff"] = (
                        int(cutoff)
                        if isinstance(cutoff, (int, float))
                        and not isinstance(cutoff, bool)
                        and float(cutoff).is_integer()
                        else cutoff
                    )
                mode = "closed_book" if index % 4 == 0 else "open_book"
                evidence = None if mode == "closed_book" else page
                self.add(
                    CurriculumExample(
                        family=self.family(family_name),
                        book_mode=mode,
                        task=(
                            "Return the exact bounded module-membership page and declared total as JSON."
                            if mode == "open_book"
                            else f"Return the memorized bounded membership page for module `{module['module_id']}` as JSON."
                        ),
                        answer=page,
                        evidence=evidence,
                        fact_payload=page,
                        strongest_group_id=f"module:{module['module_id']}",
                        module_source=self._module_source_label(module),
                        context_budget_profile="atomic_1k" if mode == "closed_book" else "evidence_2k",
                        evidence_handles=[f"module_oracle:{module['module_id']}:page:{page['page_index']}"],
                        provenance=self._base_provenance("mixed_module_oracle"),
                        coverage={
                            "canonical_genes": list(page["gene_ids_page"]),
                            "mentor_ev_modules" if source == "MENTOR_GW_DENDROGRAM" else "rwr_loe_modules": [
                                str(module["module_id"])
                            ],
                        },
                        page={
                            "index": page["page_index"],
                            "count": page["page_count"],
                            "size": 16,
                            "total_items": page["gene_count"],
                        },
                    )
                )

        provenance_goal = self.candidate_goal("module_provenance")
        all_modules = deterministic_order(self.modules, seed=self.seed, namespace="module_provenance")
        for index, module in enumerate(all_modules[:provenance_goal]):
            source = self._module_source_label(module)
            if source == "mentor_ev":
                construction_rule = "genome-wide dendrogram clade extracted from the multiplex"
            else:
                construction_rule = "seed-centered RWR-LOE vector retained below a geometric elbow cutoff"
            answer = {
                "module_id": module["module_id"],
                "module_source": source,
                "construction_rule": construction_rule,
            }
            mode = "closed_book" if index % 2 == 0 else "open_book"
            self.add(
                CurriculumExample(
                    family=self.family("module_provenance"),
                    book_mode=mode,
                    task=(
                        f"Which construction source produced module `{module['module_id']}`? Return JSON."
                        if mode == "closed_book"
                        else "Identify the module source and construction rule from the supplied metadata. Return JSON."
                    ),
                    answer=answer,
                    evidence=None if mode == "closed_book" else answer,
                    fact_payload=answer,
                    strongest_group_id=f"module:{module['module_id']}",
                    module_source=source,
                    context_budget_profile="atomic_1k",
                    evidence_handles=[f"module_oracle:{module['module_id']}:provenance"],
                    provenance=self._base_provenance("mixed_module_manifest"),
                    coverage={
                        "mentor_ev_modules" if source == "mentor_ev" else "rwr_loe_modules": [
                            str(module["module_id"])
                        ]
                    },
                )
            )

    def _generate_cross_source_module_algebra(self) -> None:
        family_names = (
            "cross_source_module_intersection",
            "module_set_difference",
            "rwr_subset_of_mentor",
            "mentor_superset_of_rwr",
            "near_subset_violations",
            "module_jaccard",
            "module_containment",
            "best_matching_module",
            "module_overlap_ranking",
            "multi_module_intersection",
            "source_unique_genes",
            "overlap_vs_topological_distance",
        )
        goals = {name: self.candidate_goal(name) for name in family_names}
        pairs = self._module_pair_pool(max(goals.values()) + 64)

        for index in range(max(goals.values())):
            pair = pairs[index % len(pairs)]
            mentor = pair["mentor"]
            rwr = pair["rwr"]
            mentor_id = str(mentor["module_id"])
            rwr_id = str(rwr["module_id"])
            mentor_genes = set(mentor["gene_ids"])
            rwr_genes = set(rwr["gene_ids"])
            intersection = list(pair["intersection"])
            module_pair_group = f"module_pair:{mentor_id}:{rwr_id}"
            common_evidence = {
                "mentor_ev_module": {"module_id": mentor_id, "gene_ids": sorted(mentor_genes)},
                "rwr_loe_module": {
                    "module_id": rwr_id,
                    "seed_gene_id": rwr.get("seed_gene_id"),
                    "gene_count": len(rwr_genes),
                    "membership_for_mentor_gene_ids": {
                        gene_id: gene_id in rwr_genes for gene_id in sorted(mentor_genes)
                    },
                    "oracle_join": {
                        "intersection_gene_ids": intersection,
                        "intersection_size": len(intersection),
                        "rwr_only_gene_count": len(pair["rwr_only"]),
                        "rwr_only_gene_ids_page": list(pair["rwr_only"][:16]),
                        "rwr_only_page_is_complete": len(pair["rwr_only"]) <= 16,
                    },
                },
            }

            if index < goals["cross_source_module_intersection"]:
                answer = {
                    "module_a": mentor_id,
                    "module_b": rwr_id,
                    "intersection_gene_ids": intersection,
                    "intersection_size": len(intersection),
                }
                self._add_module_relation_example(
                    "cross_source_module_intersection",
                    index,
                    "Verify the exact cross-source intersection from the complete MENTOR set and the supplied membership join. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                )

            if index < goals["module_set_difference"]:
                answer = {
                    "module_a": mentor_id,
                    "module_b": rwr_id,
                    "set_difference": "module_a_minus_module_b",
                    "gene_ids": list(pair["mentor_only"]),
                    "count": len(pair["mentor_only"]),
                }
                self._add_module_relation_example(
                    "module_set_difference",
                    index,
                    "Subtract RWR membership from the complete MENTOR-EV set using the supplied membership join. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                )

            if index < goals["rwr_subset_of_mentor"]:
                violations = sorted(rwr_genes - mentor_genes)
                answer = {
                    "subset": not violations,
                    "candidate_subset": rwr_id,
                    "candidate_superset": mentor_id,
                    "violating_gene_count": len(violations),
                    "violating_gene_ids_page": violations[:16],
                    "containment_fraction": len(intersection) / len(rwr_genes),
                }
                self._add_module_relation_example(
                    "rwr_subset_of_mentor",
                    index,
                    "Test whether the full RWR-LOE module is a subset of the MENTOR-EV module. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    allow_closed=True,
                    polarity="positive" if not violations else "negative",
                )

            if index < goals["mentor_superset_of_rwr"]:
                extras = sorted(mentor_genes - rwr_genes)
                violations = sorted(rwr_genes - mentor_genes)
                answer = {
                    "superset": not violations,
                    "candidate_superset": mentor_id,
                    "candidate_subset": rwr_id,
                    "extra_genes_in_superset": extras,
                    "violating_gene_count": len(violations),
                    "violating_gene_ids_page": violations[:16],
                }
                self._add_module_relation_example(
                    "mentor_superset_of_rwr",
                    index,
                    "Test the direction-aware MENTOR-EV superset claim. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    allow_closed=True,
                    polarity="positive" if not violations else "negative",
                )

            if index < goals["near_subset_violations"]:
                # The full elbow modules are generally much larger than the
                # MENTOR clades. Teach exact near-subset error detection over a
                # clearly bounded candidate page, never mislabel the page as
                # the complete RWR module.
                in_genes = intersection[: min(7, len(intersection))]
                outside = list(pair["rwr_only"][:1])
                candidate_page = in_genes + outside
                if not candidate_page or not outside:
                    continue
                answer = {
                    "candidate_scope": "bounded_rwr_module_page",
                    "exact_subset": False,
                    "containment_fraction": len(in_genes) / len(candidate_page),
                    "violating_gene_ids": outside,
                    "allowed_claim": "This bounded candidate page is a near-subset with explicit violating genes.",
                }
                evidence = {
                    "mentor_ev_module_id": mentor_id,
                    "mentor_ev_gene_ids": sorted(mentor_genes),
                    "rwr_loe_module_id": rwr_id,
                    "bounded_candidate_page": candidate_page,
                    "page_is_complete_module": False,
                }
                self._add_module_relation_example(
                    "near_subset_violations",
                    index,
                    "Find every containment violation in the supplied bounded candidate page. Return JSON.",
                    answer,
                    evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    polarity="negative",
                )

            if index < goals["module_jaccard"]:
                answer = {
                    "module_a": mentor_id,
                    "module_b": rwr_id,
                    "intersection_size": len(intersection),
                    "union_size": len(mentor_genes | rwr_genes),
                    "jaccard": pair["jaccard"],
                }
                self._add_module_relation_example(
                    "module_jaccard",
                    index,
                    "Recompute Jaccard from the supplied module sets. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    validator={"type": "recompute_set_metrics"},
                )

            if index < goals["module_containment"]:
                answer = {
                    "module_a": mentor_id,
                    "module_b": rwr_id,
                    "intersection_size": len(intersection),
                    "module_a_size": len(mentor_genes),
                    "module_b_size": len(rwr_genes),
                    "fraction_of_a_in_b": pair["mentor_containment"],
                    "fraction_of_b_in_a": pair["rwr_containment"],
                }
                self._add_module_relation_example(
                    "module_containment",
                    index,
                    "Compute both direction-aware containment coefficients. Return JSON.",
                    answer,
                    common_evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    validator={"type": "recompute_set_metrics"},
                )

            if index < goals["best_matching_module"] or index < goals["module_overlap_ranking"]:
                candidates = [row for row in pairs if row["mentor"]["module_id"] == mentor_id][:5]
                if len(candidates) < 2:
                    candidates = pairs[index : index + 5]
                ranking = sorted(
                    [
                        {
                            "module_id": row["rwr"]["module_id"],
                            "intersection_size": len(row["intersection"]),
                            "jaccard": row["jaccard"],
                            "containment_of_query": row["mentor_containment"],
                        }
                        for row in candidates
                    ],
                    key=lambda row: (-row["jaccard"], -row["intersection_size"], row["module_id"]),
                )
                rank_evidence = {"query_module": mentor_id, "ranking_metric": "jaccard", "candidate_table": ranking}
                if index < goals["best_matching_module"]:
                    answer = {
                        "query_module": mentor_id,
                        "ranking_metric": "jaccard",
                        "best_match_module": ranking[0]["module_id"],
                        "best_match_score": ranking[0]["jaccard"],
                        "ranked_matches": ranking,
                    }
                    self._add_module_relation_example(
                        "best_matching_module",
                        index,
                        "Select the maximum-Jaccard module from the supplied candidate table. Return JSON.",
                        answer,
                        rank_evidence,
                        f"module_ranking:{mentor_id}:{stable_hash(ranking)[:12]}",
                        mentor,
                        rwr,
                        validator={"type": "module_ranking"},
                    )
                if index < goals["module_overlap_ranking"]:
                    answer = {"query_module": mentor_id, "ranking_metric": "jaccard", "ranked_modules": ranking}
                    self._add_module_relation_example(
                        "module_overlap_ranking",
                        index,
                        "Sort every supplied candidate by Jaccard with deterministic tie-breaking. Return JSON.",
                        answer,
                        rank_evidence,
                        f"module_ranking:{mentor_id}:{stable_hash(ranking)[:12]}",
                        mentor,
                        rwr,
                        validator={"type": "module_ranking"},
                    )

            if index < goals["multi_module_intersection"]:
                other_pair = pairs[(index + 11) % len(pairs)]
                other_genes = set(other_pair["rwr"]["gene_ids"])
                multi = sorted(mentor_genes & rwr_genes & other_genes)
                evidence = {
                    "modules": [mentor_id, rwr_id, other_pair["rwr"]["module_id"]],
                    "gene_sets": [sorted(mentor_genes), sorted(rwr_genes)[:64], sorted(other_genes)[:64]],
                    "set_sizes": [len(mentor_genes), len(rwr_genes), len(other_genes)],
                    "pages_are_prefixes": [False, len(rwr_genes) > 64, len(other_genes) > 64],
                    "oracle_intersection_candidates": multi,
                }
                answer = {
                    "modules": evidence["modules"],
                    "intersection_gene_ids": multi,
                    "intersection_size": len(multi),
                }
                self._add_module_relation_example(
                    "multi_module_intersection",
                    index,
                    "Return the exact three-module intersection supplied by the oracle join. Return JSON.",
                    answer,
                    evidence,
                    f"module_triple:{stable_hash(evidence['modules'])[:16]}",
                    mentor,
                    rwr,
                )

            if index < goals["source_unique_genes"]:
                other_pair = pairs[(index + 17) % len(pairs)]
                other_rwr_genes = set(other_pair["rwr"]["gene_ids"])
                unique = sorted(mentor_genes - rwr_genes - other_rwr_genes)
                answer = {
                    "mentor_ev_module": mentor_id,
                    "rwr_loe_modules": [rwr_id, other_pair["rwr"]["module_id"]],
                    "mentor_ev_unique_gene_ids": unique,
                    "count": len(unique),
                }
                evidence = {
                    "mentor_ev_gene_ids": sorted(mentor_genes),
                    "oracle_union_of_rwr_gene_ids_intersecting_mentor": sorted(
                        mentor_genes & (rwr_genes | other_rwr_genes)
                    ),
                    "module_ids": answer["rwr_loe_modules"],
                }
                self._add_module_relation_example(
                    "source_unique_genes",
                    index,
                    "Return genes unique to the MENTOR-EV module relative to both RWR-LOE modules. Return JSON.",
                    answer,
                    evidence,
                    f"module_triple:{stable_hash([mentor_id] + answer['rwr_loe_modules'])[:16]}",
                    mentor,
                    rwr,
                )

            if index < goals["overlap_vs_topological_distance"]:
                sampled_pairs = []
                left_genes = sorted(mentor_genes)[:4]
                right_genes = sorted(rwr_genes - mentor_genes)[:4] or sorted(rwr_genes)[:4]
                for left_gene, right_gene in zip(left_genes, right_genes):
                    path = self.oracle.shortest_path(left_gene, right_gene, layer=None, max_hops=3)
                    if path is not None:
                        sampled_pairs.append(
                            {
                                "gene_a": left_gene,
                                "gene_b": right_gene,
                                "hop_distance": len(path) - 1,
                            }
                        )
                if not sampled_pairs:
                    continue
                mean_distance = statistics.fmean(row["hop_distance"] for row in sampled_pairs)
                overlap_class = "strong_overlap" if pair["jaccard"] >= 0.5 else "weak_overlap"
                topology_class = "close_in_sample" if mean_distance <= 2 else "far_in_sample"
                evidence = {
                    "module_a": mentor_id,
                    "module_b": rwr_id,
                    "intersection_size": len(intersection),
                    "union_size": len(mentor_genes | rwr_genes),
                    "jaccard": pair["jaccard"],
                    "sampled_cross_module_paths": sampled_pairs,
                    "mean_sampled_hop_distance": mean_distance,
                }
                answer = {
                    "set_relationship": overlap_class,
                    "topological_relationship": topology_class,
                    "basis": {
                        "jaccard": pair["jaccard"],
                        "mean_sampled_hop_distance": mean_distance,
                        "distance_scope": "sampled_aggregate_shortest_paths",
                    },
                }
                self._add_module_relation_example(
                    "overlap_vs_topological_distance",
                    index,
                    "Classify set overlap separately from the supplied sampled path-distance evidence. Return JSON.",
                    answer,
                    evidence,
                    module_pair_group,
                    mentor,
                    rwr,
                    validator={"type": "recompute_set_and_mean_metrics"},
                )

    def _add_module_relation_example(
        self,
        family_name: str,
        index: int,
        task: str,
        answer: dict[str, Any],
        evidence: dict[str, Any],
        group_id: str,
        mentor: Mapping[str, Any],
        rwr: Mapping[str, Any],
        *,
        allow_closed: bool = False,
        polarity: str = "positive",
        validator: dict[str, Any] | None = None,
    ) -> None:
        family = self.family(family_name)
        mode = "closed_book" if allow_closed and "closed_book" in family.allowed_book_modes and index % 4 == 0 else "open_book"
        if mode == "closed_book" and contains_float(answer):
            mode = "open_book"
        mentor_id = str(mentor["module_id"])
        rwr_id = str(rwr["module_id"])
        budget = (
            "matrix_state_4k"
            if estimate_tokens(json_text(evidence)) > 1500
            or estimate_tokens(json_text(answer)) > 240
            else "evidence_2k"
        )
        self.add(
            CurriculumExample(
                family=family,
                book_mode=mode,
                task=task if mode == "open_book" else f"Recall the selected stable relation between `{mentor_id}` and `{rwr_id}`. Return JSON.",
                answer=answer,
                evidence=evidence if mode == "open_book" else None,
                fact_payload=answer,
                strongest_group_id=group_id,
                module_source="mixed",
                context_budget_profile=budget,
                evidence_handles=[f"module_relation:{mentor_id}:{rwr_id}:{family_name}"],
                provenance=self._base_provenance("mixed_module_oracle"),
                coverage={
                    "canonical_genes": list(answer.get("intersection_gene_ids", [])),
                    "mentor_ev_modules": [mentor_id],
                    "rwr_loe_modules": [rwr_id],
                    "module_relations": [f"{family_name}|{mentor_id}|{rwr_id}"],
                },
                validator=validator or {"type": "exact_json"},
                polarity=polarity,
            )
        )

    def _generate_dendrogram_relations(self) -> None:
        mentors = self.modules_by_source["MENTOR_GW_DENDROGRAM"]
        by_node = {
            int(module["source_node_id"]): module
            for module in mentors
            if isinstance(module.get("source_node_id"), int)
        }
        relations: list[tuple[dict[str, Any], dict[str, Any]]] = []
        children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for parent in mentors:
            for field_name in ("left_id", "right_id"):
                child_node = parent.get(field_name)
                if isinstance(child_node, int) and child_node in by_node:
                    child = by_node[child_node]
                    relations.append((parent, child))
                    children_by_parent[str(parent["module_id"])].append(child)
        relations = deterministic_order(relations, seed=self.seed, namespace="dendrogram_relations")

        parent_goal = self.candidate_goal("module_parent_child")
        for index, (parent, child) in enumerate(relations[:parent_goal]):
            answer = {
                "child_module": child["module_id"],
                "parent_module": parent["module_id"],
                "is_nested": set(child["gene_ids"]).issubset(parent["gene_ids"]),
                "child_gene_count": len(child["gene_ids"]),
                "parent_gene_count": len(parent["gene_ids"]),
            }
            mode = "closed_book" if index % 2 == 0 else "open_book"
            self.add(
                CurriculumExample(
                    family=self.family("module_parent_child"),
                    book_mode=mode,
                    task=(
                        f"Is MENTOR-EV module `{child['module_id']}` nested immediately inside `{parent['module_id']}`? Return JSON."
                        if mode == "closed_book"
                        else "Verify the immediate parent-child clade relation from the supplied module sets. Return JSON."
                    ),
                    answer=answer,
                    evidence=None if mode == "closed_book" else {
                        "parent_module": parent["module_id"],
                        "parent_gene_ids": parent["gene_ids"],
                        "child_module": child["module_id"],
                        "child_gene_ids": child["gene_ids"],
                        "tree_relation": "immediate_child",
                    },
                    fact_payload=answer,
                    strongest_group_id=f"dendrogram_parent:{parent['module_id']}",
                    module_source="mentor_ev",
                    context_budget_profile="atomic_1k" if mode == "closed_book" else "evidence_2k",
                    evidence_handles=[f"dendrogram_relation:{parent['module_id']}:{child['module_id']}"],
                    provenance=self._base_provenance("mentor_ev_dendrogram_oracle"),
                    coverage={
                        "mentor_ev_modules": [str(parent["module_id"]), str(child["module_id"])],
                        "module_relations": [f"parent_child|{parent['module_id']}|{child['module_id']}"],
                    },
                )
            )

        sibling_sets = [
            (parent_id, children)
            for parent_id, children in children_by_parent.items()
            if len(children) >= 2
        ]
        sibling_sets = deterministic_order(sibling_sets, seed=self.seed, namespace="dendrogram_siblings")
        sibling_goal = self.candidate_goal("sibling_modules")
        emitted = 0
        while emitted < sibling_goal:
            parent_id, children = sibling_sets[emitted % len(sibling_sets)]
            query = children[(emitted // len(sibling_sets)) % len(children)]
            siblings = sorted(
                str(child["module_id"])
                for child in children
                if child["module_id"] != query["module_id"]
            )
            answer = {
                "query_module": query["module_id"],
                "parent_module": parent_id,
                "sibling_modules": siblings,
            }
            mode = "closed_book" if emitted % 2 == 0 else "open_book"
            self.add(
                CurriculumExample(
                    family=self.family("sibling_modules"),
                    book_mode=mode,
                    task=(
                        f"Which MENTOR-EV modules share the immediate parent of `{query['module_id']}`? Return JSON."
                        if mode == "closed_book"
                        else "Return the exact siblings from the supplied immediate-parent table. Return JSON."
                    ),
                    answer=answer,
                    evidence=None if mode == "closed_book" else answer,
                    fact_payload=answer,
                    strongest_group_id=f"dendrogram_parent:{parent_id}",
                    module_source="mentor_ev",
                    context_budget_profile="atomic_1k",
                    evidence_handles=[f"dendrogram_siblings:{parent_id}"],
                    provenance=self._base_provenance("mentor_ev_dendrogram_oracle"),
                    coverage={
                        "mentor_ev_modules": [str(query["module_id"]), parent_id] + siblings,
                        "module_relations": [f"siblings|{parent_id}|{query['module_id']}"],
                    },
                )
            )
            emitted += 1


def validate_rendered_tool_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate a rendered tool row against the live runtime vocabulary."""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return {"valid": False, "reason": "missing metadata"}
    exchange = metadata.get("tool_exchange")
    if not isinstance(exchange, Mapping):
        return {"valid": False, "reason": "missing tool exchange"}
    try:
        assert_no_provenance_leakage(exchange)
        action = ToolAction.from_dict(dict(exchange["tool_action"]))
        observation = ToolObservation.from_dict(dict(exchange["tool_observation"]))
        validation = validate_tool_action_schema(action)
        if not validation.valid:
            return {"valid": False, "reason": "; ".join(validation.errors)}
        if action.tool_name not in CURRICULUM_TOOL_NAMES:
            return {"valid": False, "reason": "tool is not model-facing"}
        if observation.call_id != action.call_id:
            return {"valid": False, "reason": "action/observation call_id mismatch"}
    except (KeyError, TypeError, ValueError, ToolCurriculumContractError) as exc:
        return {"valid": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"valid": True}


def build_curriculum(builder: CurriculumBuilder) -> dict[str, Any]:
    """Generate every planned family, compile it, and publish audited artifacts."""

    from scripts.pretrajectory_curriculum_artifacts import (
        compile_pretrajectory_curriculum_artifacts,
    )
    from scripts.pretrajectory_curriculum_global_families import generate_global_families
    from scripts.pretrajectory_curriculum_tool_families import generate_tool_families

    builder.load_sources()
    builder.generate_stage1_and_stage2()
    builder.generate_stage3_topology()
    builder.generate_stage3_rwr_distance()
    builder.generate_stage4_modules()
    generate_global_families(builder)
    generate_tool_families(builder)
    builder.coalesce_oracle_fact_groups()

    def render_candidate(example: CurriculumExample) -> dict[str, Any]:
        record = builder.render(example)
        if record is None:
            raise ValueError(
                f"candidate failed generator-side path or budget checks: {example.family.name}"
            )
        return {
            "record": record,
            "canonical_object": builder.canonical_object(example),
        }

    result = compile_pretrajectory_curriculum_artifacts(
        candidates=builder.examples,
        out_dir=builder.out_dir,
        plan=builder.plan,
        build_profile=builder.profile_name,
        seed=builder.seed,
        render_candidate=render_candidate,
        tool_validator=validate_rendered_tool_record,
        source_identities={
            "multiplex_id": builder.multiplex_id,
            "store_id": builder.store_id,
            "flist_id": builder.flist_id,
            "alias_registry": {
                "cache_id": builder.alias_cache_id,
            },
            "module_corpus": {
                "modules_id": builder.module_corpus_id,
                "manifest_id": builder.module_manifest_id,
                "mentor_ev_manifest_id": builder.mentor_manifest_id,
                "rwr_loe_manifest_id": builder.rwr_module_manifest_id,
            },
            "rwr_cache": builder.rwr.public_provenance() if builder.rwr is not None else {},
        },
        overwrite=builder.overwrite,
    )
    result["generator"] = {
        "generated_candidate_count": len(builder.examples),
        "generated_by_family": dict(sorted(builder.generated_counts.items())),
        "generator_filtered_by_family": dict(sorted(builder.filtered_counts.items())),
        "coalesced_oracle_fact_group_count": builder.coalesced_fact_group_count,
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--profile", default="patchcheck")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--alias-cache", type=Path, default=DEFAULT_ALIAS_CACHE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    builder = CurriculumBuilder(
        plan_path=args.plan,
        profile=args.profile,
        out_dir=args.out_dir,
        seed=args.seed,
        alias_cache_path=args.alias_cache,
        overwrite=args.overwrite,
    )
    result = build_curriculum(builder)
    summary = {
        "output_dir": str(args.out_dir),
        "manifest": result["manifest"],
        "audit_passed": result["audit_report"]["passed"],
        "leakage_passed": result["leakage_report"]["passed"],
        "generator": result["generator"],
    }
    if args.json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {result['manifest']['selected_record_count']} audited records "
            f"to {args.out_dir}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
