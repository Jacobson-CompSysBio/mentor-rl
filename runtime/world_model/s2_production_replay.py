"""Select pinned training-only replay without rewriting historical graph meaning.

The explicit legacy adapter keeps S0/S1 evidence, status, and result conventions
local to their original contract. Its visible system instruction separates them
from S2's list-valued evidence. The existing S2 tokenizer can render its complete
traces and current-turn supervision masks without a second serialization path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from ..world_model_training import flatten_v3_trace_for_arrow, renderable_v3_tools
from .sources import file_sha256


ADAPTER_VERSION = "s2_production_legacy_replay_v1"
S1_COMPONENTS = ("s1_1", "s1_2", "s1_3", "s1_4", "s1_6", "s1_7", "s1_8", "s1_9")
LEGACY_INSTRUCTION = (
    "Contract: legacy S0/S1 identifier and coordinate-system replay, "
    "multiplex_sft_v3. Use this legacy contract throughout this trace: "
    "provenance channels are objects or null, and root provenance may be null. "
    "Retain the original membership scope, graph version, required-operation "
    "status, and result conventions. Supplied legacy records are not S2 graph "
    "observations. Do not convert them into S2's list-valued provenance or "
    "reinterpret their graph hashes as current S2 graph content. Legacy "
    "acquisition evidence IDs are local to this source trace; the same ID in "
    "another source trace does not identify the same acquisition.\n\n"
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def capped_equal_quotas(capacities: Mapping[str, int], count: int) -> dict[str, int]:
    """Allocate an exact count evenly, exhausting small pools before larger ones."""
    if type(count) is not int or count < 0 or any(type(n) is not int or n < 0 for n in capacities.values()):
        raise ValueError("replay counts and capacities must be nonnegative integers")
    if count > sum(capacities.values()):
        raise ValueError("the replay quota exceeds the distinct available training rows")
    quotas = dict.fromkeys(sorted(capacities), 0)
    pending = sorted(name for name, size in capacities.items() if size)
    remaining = count
    while remaining:
        share, remainder = divmod(remaining, len(pending))
        small = [name for name in pending if capacities[name] - quotas[name] <= share]
        if small:
            for name in small:
                available = capacities[name] - quotas[name]
                quotas[name] += available
                remaining -= available
            pending = [name for name in pending if name not in small]
        else:
            for index, name in enumerate(pending):
                quotas[name] += share + (index < remainder)
            remaining = 0
    return quotas


def replay_type(row: Mapping[str, Any], stage: str) -> str:
    """Return the published case ID, retaining one-sided type-specific framing."""
    if stage not in ("S0", "S1"):
        raise ValueError("replay stage must be S0 or S1")
    meta = row["metadata"]
    source_trace = meta.get("source_trace")
    if meta.get("step") != stage or not isinstance(source_trace, str) or not source_trace.startswith(stage + "."):
        raise ValueError("replay row does not belong to the requested source stage")
    return source_trace


def adapt_replay_row(row: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Make the legacy contract visible while preserving every observation/target.

    The only model-visible changes are the explicit system instruction and the
    tool descriptions already used by the canonical legacy training serializer.
    Tool arguments, responses, harness strings, reasoning, partials, and every
    answer remain unchanged, including historical object/null provenance.
    """
    if row.get("metadata", {}).get("contract_adapter") == ADAPTER_VERSION:
        validate_replay_adapter(row, stage)
        return copy.deepcopy(dict(row))
    flatten_v3_trace_for_arrow(row, expected_split="train")
    trace = replay_type(row, stage)
    result = copy.deepcopy(dict(row))
    source_id = row["row_id"]
    source_system_hash = row["metadata"]["system_prompt_sha256"]
    result["row_id"] = f"s2_prod_replay_{stage.lower()}_{_digest(source_id)[:24]}"
    result["messages"][0]["content"] = LEGACY_INSTRUCTION + result["messages"][0]["content"]
    result["tools"] = renderable_v3_tools(result["tools"])
    result["metadata"].update(
        system_prompt_sha256=hashlib.sha256(result["messages"][0]["content"].encode()).hexdigest(),
        trace_family=row["metadata"]["question_family"],
        source_stage=stage,
        source_record_id=source_id,
        contract_adapter=ADAPTER_VERSION,
        evidence_namespace={"stage": stage, "source_record_id": source_id},
        production_replay={
            "adapter_version": ADAPTER_VERSION,
            "source_trace": trace,
            "source_system_prompt_sha256": source_system_hash,
            "source_tool_contract_version": row["metadata"]["tool_contract_version"],
            "source_row_sha256": _digest(row),
        },
    )
    return result


def validate_replay_adapter(row: Mapping[str, Any], stage: str) -> None:
    """Reject an adapter declaration without its matching visible legacy contract."""
    from ..world_model_training import _validate_v3_message_sequence

    if list(row) != ["row_id", "messages", "tools", "metadata"]:
        raise ValueError("the replay row field order changed")
    _validate_v3_message_sequence(row["messages"])
    meta = row["metadata"]
    replay_type(row, stage)
    if (meta.get("contract_adapter") != ADAPTER_VERSION or meta.get("source_stage") != stage
            or not isinstance(meta.get("source_record_id"), str)
            or meta.get("trace_family") != meta.get("question_family")
            or meta.get("evidence_namespace") != {
                "stage": stage, "source_record_id": meta.get("source_record_id")}):
        raise ValueError("the legacy replay adapter metadata is invalid")
    prompt = row["messages"][0]["content"]
    if (not prompt.startswith(LEGACY_INSTRUCTION)
            or hashlib.sha256(prompt.encode()).hexdigest() != meta.get("system_prompt_sha256")):
        raise ValueError("the legacy replay adapter lacks its visible contract instruction")
    if meta.get("production_replay", {}).get("adapter_version") != ADAPTER_VERSION:
        raise ValueError("the legacy replay adapter version differs from its receipt")
    if row["tools"] != renderable_v3_tools(row["tools"]):
        raise ValueError("the legacy replay tool descriptions differ from their definitions")


def _selection_stratum(row: Mapping[str, Any]) -> str:
    meta, answer = row["metadata"], row["messages"][-1]["content"]
    return json.dumps([meta.get("framing"), meta.get("book_mode"), answer["status"],
                       answer["continuation_decision"], meta.get("blocked_reason_code"),
                       meta.get("fact_kind"), meta.get("input_variant")], separators=(",", ":"))


def select_replay_rows(rows: Iterable[Mapping[str, Any]], count: int, *, stage: str,
                       seed: int = 20261030) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select distinct rows by capped type/condition quotas and stable SHA ranking.

    Callers may first remove rows whose adapted complete traces exceed their
    context budget; quotas are then allocated only over the eligible pool.
    """
    groups, ids = defaultdict(list), set()
    for row in rows:
        source_id = row["metadata"].get("source_record_id", row["row_id"])
        if source_id in ids:
            raise ValueError("duplicate source row in the replay pool")
        ids.add(source_id)
        groups[replay_type(row, stage)].append(row)
    capacities = {name: len(group) for name, group in groups.items()}
    quotas = capped_equal_quotas(capacities, count)
    selected, stratum_counts = [], {}
    for name in sorted(groups):
        strata = defaultdict(list)
        for row in groups[name]:
            strata[_selection_stratum(row)].append(row)
        allocation = capped_equal_quotas({key: len(value) for key, value in strata.items()}, quotas[name])
        stratum_counts[name] = allocation
        for key in sorted(strata):
            ranked = sorted(strata[key], key=lambda row: _digest([seed, stage,
                row["metadata"].get("source_record_id", row["row_id"])]))
            selected.extend(adapt_replay_row(row, stage) for row in ranked[:allocation[key]])
    selected.sort(key=lambda row: _digest([seed, stage, row["metadata"]["source_record_id"]]))
    source_ids = [row["metadata"]["source_record_id"] for row in selected]
    return selected, {
        "stage": stage, "adapter_version": ADAPTER_VERSION, "seed": seed,
        "selection": "capped_equal_types_then_joint_condition_strata_then_sha256_v1",
        "training_only": True, "replacement": False, "available_rows": len(ids),
        "selected_rows": len(selected), "available_by_type": dict(sorted(capacities.items())),
        "selected_by_type": quotas, "selected_by_condition": stratum_counts,
        "selected_source_row_ids_sha256": _digest(source_ids),
        "continuation_counts": dict(Counter(row["messages"][-1]["content"]["continuation_decision"]
                                             for row in selected)),
    }


def _verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"replay manifest differs from its published pin: {path}")
    return json.loads(path.read_text())


def load_replay_pool(repo_root: Path, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load pinned published training rows, excluding S1's embedded S0 replay.

    The passing S1 v10 training config anchors its aggregate manifest. Its
    component pins anchor the eight S1 pools, and its S0 replay receipt anchors
    the original S0 source manifest. Every source train hash/count is checked;
    validation and test files are never read for selection.
    """
    if stage not in ("S0", "S1"):
        raise ValueError("replay stage must be S0 or S1")
    root = Path(repo_root)
    config_path = root / "config/world_model_s1_120b_training_v10_32n.json"
    config = json.loads(config_path.read_text())
    corpus = config["corpus"]
    aggregate_path = root / corpus["root"] / "manifest.json"
    aggregate = _verified_json(aggregate_path, corpus["manifest_sha256"])
    if stage == "S1":
        manifests = [(root / aggregate["components"][name]["manifest_path"],
                      aggregate["components"][name]["manifest_sha256"]) for name in S1_COMPONENTS]
    else:
        replay = aggregate["s0_replay"]
        manifests = [(root / replay["manifest_path"], replay["manifest_sha256"])]
    rows, sources, seen = [], [], set()
    for manifest_path, expected in manifests:
        manifest = _verified_json(manifest_path, expected)
        contract = manifest.get("contract") or aggregate["contract"]
        pins = [contract, *(value for key, value in contract.items() if key.endswith("_overlay"))]
        pins.extend(value for key, value in manifest.items() if key.endswith("_overlay"))
        for pin in pins:
            pinned_path = root / pin["path"]
            if file_sha256(pinned_path) != pin["sha256"]:
                raise ValueError(f"replay contract/overlay differs from its pin: {pinned_path}")
        source = manifest_path.with_name("train.jsonl")
        file_pin = manifest["files"]["train.jsonl"]
        if file_sha256(source) != file_pin["sha256"]:
            raise ValueError(f"replay train split differs from its pin: {source}")
        loaded = 0
        with source.open() as handle:
            for line in handle:
                row = json.loads(line)
                flatten_v3_trace_for_arrow(row, expected_split="train")
                replay_type(row, stage)
                if row["row_id"] in seen:
                    raise ValueError("source component row IDs overlap")
                seen.add(row["row_id"])
                rows.append(row)
                loaded += 1
        if loaded != file_pin["row_count"]:
            raise ValueError(f"replay training row count differs from its pin: {source}")
        sources.append({"manifest_path": str(manifest_path.relative_to(root)),
                        "manifest_sha256": expected, "path": str(source.relative_to(root)),
                        "sha256": file_pin["sha256"], "rows": loaded})
    return rows, {"stage": stage, "source_config": str(config_path.relative_to(root)),
                  "source_config_sha256": file_sha256(config_path),
                  "aggregate_manifest_sha256": corpus["manifest_sha256"],
                  "training_only": True, "embedded_s0_replay_excluded": stage == "S1",
                  "sources": sources, "rows": len(rows)}
