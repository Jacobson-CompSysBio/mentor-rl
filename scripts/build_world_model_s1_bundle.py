#!/usr/bin/env python3
"""Build the deterministic aggregate S1 corpus and held-out eval panels.

The stage-specific corpus builders remain the owners of row construction.  This
script verifies and concatenates their artifacts in curriculum order, then
projects validation and test rows into answer-separated evaluator panels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import (  # noqa: E402
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from runtime.world_model.s0_corpus import (  # noqa: E402
    EvidenceSequence,
    S0_V4_METADATA_ADDITIONS,
)
from runtime.world_model.s1_overlay import (  # noqa: E402
    apply_s1_overlay,
    overlay_pin,
)


DATASET_ID = "world_model_v3_s1_coordinate_system"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-manifest-v1"
CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-corpus-v1"
EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s1-evaluator-v1"
EVALUATION_CONTRACT = "held_out_s1_multiplex_sft_v3"
LOSS_CONTRACT = "multiplex_sft_v3"
SPLITS = ("train", "validation", "test")
EVALUATION_SPLITS = ("validation", "test")
STAGE_ORDER = (
    "s1_1",
    "s1_2",
    "s1_3",
    "s1_4",
    "s1_6",
    "s1_7",
    "s1_8",
    "s1_9",
)
STAGE_SECTION = {
    stage: stage.upper().replace("_", ".") for stage in STAGE_ORDER
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EVIDENCE_ID_PATTERN = re.compile(
    r"(?<![a-z0-9_])ev_(?P<channel>reg|mem)_[a-z0-9]{6}(?![a-z0-9_])"
)
EVIDENCE_CHANNELS = {"reg": "registry", "mem": "graph"}


class BundleBuildError(ValueError):
    """Report an invalid component artifact or aggregate row."""


@dataclass(frozen=True, slots=True)
class Component:
    """One verified stage-specific corpus component."""

    directory: str
    section: str
    question_family: str
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    files: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """The verified, deterministic S0 train replay selection."""

    rows: tuple[tuple[bytes, dict[str, Any]], ...]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    source_path: Path
    source_sha256: str
    source_row_count: int
    metadata_additions: tuple[str, ...]
    seed: int
    requested_share: float
    s1_train_row_count: int
    selected_row_ids_sha256: str
    selection_strata: dict[str, Any] | None = None


class AtomicBatch:
    """Stage a related set of files before replacing their destinations."""

    def __init__(self) -> None:
        self._staged: list[tuple[Path, Path]] = []

    def path_for(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.fchmod(descriptor, 0o644)
        os.close(descriptor)
        temporary = Path(temporary_name)
        self._staged.append((destination, temporary))
        return temporary

    def commit(self) -> None:
        for destination, temporary in self._staged:
            os.replace(temporary, destination)
        self._staged.clear()

    def cleanup(self) -> None:
        for _, temporary in self._staged:
            temporary.unlink(missing_ok=True)
        self._staged.clear()


def canonical_json(value: Any) -> str:
    """Serialize one stable, compact JSON value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def ordered_json(value: Any) -> str:
    """Serialize one corpus row without changing its contract-owned key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def rekey_training_evidence(
    row: Mapping[str, Any], evidence_ids: EvidenceSequence
) -> dict[str, Any]:
    """Give one aggregate train row fresh IDs from a bundle-wide sequence.

    Stage builders own independent sequences, so their IDs collide after the
    stages are concatenated. One mapping per row keeps every copy of an ID
    equal, including copies embedded in harness strings, while the shared
    sequence makes IDs unique across the aggregate training corpus.
    """

    replacements: dict[str, str] = {}

    def replace(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if not isinstance(value, str):
            return value

        def replacement(match: re.Match[str]) -> str:
            old_id = match.group(0)
            if old_id not in replacements:
                channel = EVIDENCE_CHANNELS[match.group("channel")]
                replacements[old_id] = evidence_ids(channel)
            return replacements[old_id]

        return EVIDENCE_ID_PATTERN.sub(replacement, value)

    return replace(copy.deepcopy(dict(row)))


def normalize_s0_replay_row(
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
    metadata_additions: Sequence[str],
) -> dict[str, Any]:
    """Drop only source-manifest-declared S0 builder metadata additions."""

    normalized = copy.deepcopy(dict(row))
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        raise BundleBuildError("S0 replay row has no metadata object")
    metadata_contract = contract["rows"]["metadata"]
    required = list(metadata_contract["required"])
    optional = list(metadata_contract["optional"])
    allowed = required + optional
    missing = [key for key in required if key not in metadata]
    extras = [key for key in metadata if key not in allowed]
    unsupported = [key for key in extras if key not in metadata_additions]
    if missing or unsupported:
        raise BundleBuildError(
            "S0 replay metadata cannot be normalized: "
            f"missing {missing}, unsupported {unsupported}"
        )
    normalized["metadata"] = {
        key: metadata[key] for key in allowed if key in metadata
    }
    return normalized


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 of one canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleBuildError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleBuildError(f"expected one JSON object in {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic, human-readable JSON object."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise BundleBuildError(
            f"output path escapes repository root: {path}"
        ) from error


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleBuildError(f"{label} must be a nonempty string")
    return value


def _required_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BundleBuildError(f"{label} must be a nonnegative integer")
    return value


def _required_sha256(value: Any, label: str) -> str:
    digest = _required_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise BundleBuildError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _component_schema(section: str, kind: str) -> str:
    suffix = section.lower().replace(".", "-")
    return f"mentor-rl-world-model-{suffix}-{kind}-v1"


def _load_components(
    *,
    component_root: Path,
    repo_root: Path,
    contract: Mapping[str, Any],
    use_contract_directories: bool = False,
) -> list[Component]:
    """Load component manifests and verify every pinned input artifact.

    An overlay build reads each stage from the directory its contract names, so
    a revised stage can sit beside the unchanged stages of the earlier corpus.
    """

    families = contract["curriculum"]["families"]
    components: list[Component] = []
    for directory in STAGE_ORDER:
        section = STAGE_SECTION[directory]
        settings = families.get(section)
        if not isinstance(settings, Mapping):
            raise BundleBuildError(f"current contract has no {section} settings")
        if settings.get("enabled") is False:
            raise BundleBuildError(f"enabled aggregate stage {section} is disabled")
        question_family = _required_string(
            settings.get("question_family"),
            f"contract {section} question_family",
        )
        component_dir = (
            repo_root
            / _required_string(
                settings.get("output_directory"),
                f"contract {section} output_directory",
            )
            if use_contract_directories
            else component_root / directory
        )
        manifest_path = component_dir / "manifest.json"
        manifest = read_json(manifest_path)
        expected_manifest_schema = _component_schema(section, "manifest")
        expected_corpus_schema = _component_schema(section, "corpus")
        if manifest.get("schema_version") != expected_manifest_schema:
            raise BundleBuildError(f"{section} component manifest schema changed")
        if manifest.get("corpus_schema_version") != expected_corpus_schema:
            raise BundleBuildError(f"{section} component corpus schema changed")
        if manifest.get("dataset_id") != settings.get("dataset_id"):
            raise BundleBuildError(f"{section} component dataset ID changed")
        if manifest.get("section") != section:
            raise BundleBuildError(f"{section} component section changed")
        manifest_files = manifest.get("files")
        row_counts = manifest.get("row_counts")
        if not isinstance(manifest_files, Mapping) or not isinstance(
            row_counts, Mapping
        ):
            raise BundleBuildError(f"{section} manifest lacks files or row_counts")

        pinned_files: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            file_name = f"{split}.jsonl"
            source_path = component_dir / file_name
            file_spec = manifest_files.get(file_name)
            if not isinstance(file_spec, Mapping):
                raise BundleBuildError(f"{section} manifest lacks {file_name}")
            expected_hash = _required_sha256(
                file_spec.get("sha256"), f"{section} {file_name} SHA-256"
            )
            row_count = _required_count(
                file_spec.get("row_count"), f"{section} {file_name} row_count"
            )
            if row_counts.get(split) != row_count:
                raise BundleBuildError(
                    f"{section} {split} row counts disagree in its manifest"
                )
            actual_hash = sha256_file(source_path)
            if actual_hash != expected_hash:
                raise BundleBuildError(
                    f"{section} {file_name} hash changed: {actual_hash}"
                )
            pinned_files[file_name] = {
                "path": _repo_relative(source_path, repo_root),
                "row_count": row_count,
                "sha256": actual_hash,
            }
        components.append(
            Component(
                directory=directory,
                section=section,
                question_family=question_family,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                files=pinned_files,
            )
        )
    return components


def _validate_disabled_stage(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Pin the intentional S1.5 context-ontology gap."""

    settings = contract["curriculum"]["families"].get("S1.5")
    expected = {
        "enabled": False,
        "question_family": "context_id_resolution",
        "blocked_on": "context_ontology",
    }
    if settings != expected:
        raise BundleBuildError(
            "S1.5 must remain disabled on the context_ontology dependency"
        )
    return dict(expected)


def _validate_replay_row(
    row: Any,
    *,
    contract: Mapping[str, Any],
    expected_tools: Sequence[Mapping[str, Any]],
    prompt_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Validate one canonical S0 row admitted to replay selection."""

    if not isinstance(row, Mapping):
        raise BundleBuildError("S0 replay source row is not an object")
    if list(row) != ["row_id", "messages", "tools", "metadata"]:
        raise BundleBuildError("S0 replay source row field order changed")
    row_id = _required_string(row.get("row_id"), "S0 replay row_id")
    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or not messages:
        raise BundleBuildError(f"{row_id} has no messages")
    if any(not isinstance(message, Mapping) for message in messages):
        raise BundleBuildError(f"{row_id} has a non-object message")
    if (
        messages[0].get("role") != "system"
        or messages[0].get("content") != contract["runtime"]["system_prompt"]
    ):
        raise BundleBuildError(f"{row_id} does not use the current system prompt")
    if not isinstance(tools, list) or tools != list(expected_tools):
        raise BundleBuildError(f"{row_id} does not use the current tool contract")
    if not isinstance(metadata, dict):
        raise BundleBuildError(f"{row_id} has no metadata object")
    expected_metadata = {
        "step": "S0",
        "schema_version": contract["runtime"]["row_schema_version"],
        "system_prompt_sha256": prompt_sha256,
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise BundleBuildError(
                f"{row_id} replay metadata {key} is {metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    family = metadata.get("question_family")
    if family not in {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }:
        raise BundleBuildError(f"{row_id} has an unsupported S0 replay family")
    source_trace = metadata.get("source_trace")
    if not isinstance(source_trace, str) or not source_trace.startswith("S0."):
        raise BundleBuildError(f"{row_id} is not an S0 replay row")
    split_unit = metadata.get("split_unit")
    if not isinstance(split_unit, str) or not split_unit:
        raise BundleBuildError(f"{row_id} has no split_unit")
    return row_id, metadata


def replay_stratum(row: Mapping[str, Any]) -> str:
    """Classify existing S0 rows by behavior, including resolved unnamed IDs."""

    kind = row["metadata"]["fact_kind"]
    result = row["messages"][-1]["content"]["result"]
    if kind == "named_gene_id" and result.get("gene_symbol") is None:
        return "unnamed_gene_id"
    return kind


def select_stratified_replay(
    ranked: Sequence[tuple[str, bytes, dict[str, Any]]],
    counts: Mapping[str, int],
) -> tuple[tuple[bytes, dict[str, Any]], ...]:
    """Select unchanged source rows evenly across available framing/input cells.

    Balance framing first, then candidate count and presentation for ambiguous
    symbols. Source text stays intact even when both framings must defer.
    """

    selected = []
    for kind, count in counts.items():
        framings = defaultdict(list)
        for candidate in ranked:
            row = candidate[2]
            if replay_stratum(row) == kind:
                framings[row["metadata"]["framing"]].append(candidate)
        if sum(map(len, framings.values())) < count:
            raise BundleBuildError(f"S0 replay stratum {kind} has fewer than {count} rows")
        quotient, remainder = divmod(count, len(framings))
        for framing_index, framing in enumerate(sorted(framings)):
            target = quotient + (framing_index < remainder)
            if len(framings[framing]) < target:
                raise BundleBuildError(f"S0 replay {kind}/{framing} cannot balance framing")
            cells = defaultdict(list)
            for candidate in framings[framing]:
                row = candidate[2]
                key = ()
                if kind == "ambiguous_symbol":
                    result = row["messages"][-1]["content"]["result"]
                    key = (len(result["candidate_gene_ids"]), row["metadata"]["input_variant"])
                cells[key].append(candidate)
            positions = Counter()
            picked = 0
            while picked < target:
                for key in sorted(cells):
                    index = positions[key]
                    if index < len(cells[key]):
                        selected.append(cells[key][index])
                        positions[key] += 1
                        picked += 1
                        if picked == target:
                            break
    selected.sort(key=lambda item: (item[0], item[2]["row_id"]))
    return tuple((raw_line, row) for _, raw_line, row in selected)


def _load_replay_selection(
    *,
    repo_root: Path,
    contract: Mapping[str, Any],
    expected_tools: Sequence[Mapping[str, Any]],
    prompt_sha256: str,
    s1_train_row_count: int,
    replay_source: Mapping[str, Any] | None = None,
) -> ReplaySelection:
    """Verify S0 train and select the exact replay rows from the contract.

    An overlay may name a newer S0 corpus as the replay source. The selection
    seed and share still come from the shared contract.
    """

    curriculum = contract["curriculum"]
    share_value = curriculum.get("s0_replay_share")
    if not isinstance(share_value, (int, float)) or isinstance(share_value, bool):
        raise BundleBuildError("curriculum.s0_replay_share must be numeric")
    requested_share = float(share_value)
    if not 0.0 < requested_share < 1.0:
        raise BundleBuildError(
            "curriculum.s0_replay_share must be between zero and one"
        )
    s0_settings = curriculum["families"].get("S0")
    if not isinstance(s0_settings, Mapping):
        raise BundleBuildError("current contract has no S0 replay settings")
    seed = _required_count(s0_settings.get("assignment_seed"), "S0 assignment_seed")
    source = replay_source or {
        "dataset_id": s0_settings.get("dataset_id"),
        "output_directory": s0_settings.get("output_directory"),
        "manifest_schema_version": "mentor-rl-world-model-s0-manifest-v1",
        "corpus_schema_version": "mentor-rl-world-model-s0-corpus-v1",
    }
    output_directory = _required_string(
        source.get("output_directory"), "S0 output_directory"
    )
    s0_root = (repo_root / output_directory).resolve()
    manifest_path = s0_root / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != source.get("manifest_schema_version")
        or manifest.get("corpus_schema_version")
        != source.get("corpus_schema_version")
        or manifest.get("dataset_id") != source.get("dataset_id")
    ):
        raise BundleBuildError("S0 replay manifest contract changed")
    manifest_contract = manifest.get("contract")
    overlay_contract = (
        manifest_contract.get("s0_overlay")
        if isinstance(manifest_contract, Mapping)
        else None
    )
    metadata_additions_value = (
        overlay_contract.get("metadata_optional_additions")
        if isinstance(overlay_contract, Mapping)
        else []
    )
    if not isinstance(metadata_additions_value, list) or any(
        not isinstance(key, str) or not key
        for key in metadata_additions_value
    ):
        raise BundleBuildError(
            "S0 replay manifest metadata additions are invalid"
        )
    metadata_additions = tuple(metadata_additions_value)
    if len(set(metadata_additions)) != len(metadata_additions):
        raise BundleBuildError(
            "S0 replay manifest metadata additions contain duplicates"
        )
    if metadata_additions and metadata_additions != S0_V4_METADATA_ADDITIONS:
        raise BundleBuildError(
            "S0 replay manifest metadata additions changed"
        )
    files = manifest.get("files")
    row_counts = manifest.get("row_counts")
    if not isinstance(files, Mapping) or not isinstance(row_counts, Mapping):
        raise BundleBuildError("S0 replay manifest lacks files or row_counts")
    train_spec = files.get("train.jsonl")
    if not isinstance(train_spec, Mapping):
        raise BundleBuildError("S0 replay manifest lacks train.jsonl")
    source_sha256 = _required_sha256(
        train_spec.get("sha256"), "S0 replay train SHA-256"
    )
    source_row_count = _required_count(
        train_spec.get("row_count"), "S0 replay train row_count"
    )
    if row_counts.get("train") != source_row_count:
        raise BundleBuildError("S0 replay manifest train row counts disagree")
    source_path = s0_root / "train.jsonl"
    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != source_sha256:
        raise BundleBuildError(
            f"S0 replay train hash changed: {actual_source_sha256}"
        )

    candidates: list[tuple[str, bytes, dict[str, Any]]] = []
    source_ids: set[str] = set()
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BundleBuildError(
                    f"invalid S0 replay JSON at {source_path}:{line_number}: {error}"
                ) from error
            row_id, _ = _validate_replay_row(
                row,
                contract=contract,
                expected_tools=expected_tools,
                prompt_sha256=prompt_sha256,
            )
            if row_id in source_ids:
                raise BundleBuildError(f"duplicate S0 replay source row ID {row_id}")
            source_ids.add(row_id)
            rank = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
            candidates.append((rank, raw_line.rstrip(b"\r\n") + b"\n", row))
    if len(candidates) != source_row_count:
        raise BundleBuildError(
            f"S0 replay train has {len(candidates)} rows, expected {source_row_count}"
        )
    replay_count = round(
        s1_train_row_count * requested_share / (1.0 - requested_share)
    )
    if replay_count < 1 or replay_count > len(candidates):
        raise BundleBuildError(f"invalid derived S0 replay count {replay_count}")
    ranked = sorted(candidates, key=lambda item: (item[0], item[2]["row_id"]))
    selection = replay_source.get("selection") if replay_source else None
    strata = None
    if selection is None:
        selected = tuple((raw_line, row) for _, raw_line, row in ranked[:replay_count])
    else:
        counts = selection["counts"]
        if sum(counts.values()) != replay_count:
            raise BundleBuildError("S0 replay strata do not sum to the derived row count")
        selected = select_stratified_replay(ranked, counts)
        strata = {
            "counts": dict(counts),
            "framing_counts": {
                kind: dict(sorted(Counter(
                    row["metadata"]["framing"] for _, row in selected
                    if replay_stratum(row) == kind
                ).items()))
                for kind in counts
            },
            "ambiguous_cells": dict(sorted(Counter(
                f"{row['metadata']['framing']}:{len(row['messages'][-1]['content']['result']['candidate_gene_ids'])}:{row['metadata']['input_variant']}"
                for _, row in selected if replay_stratum(row) == "ambiguous_symbol"
            ).items())),
        }
    selected_ids = [row["row_id"] for _, row in selected]
    return ReplaySelection(
        rows=selected,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        source_path=source_path,
        source_sha256=source_sha256,
        source_row_count=source_row_count,
        metadata_additions=metadata_additions,
        seed=seed,
        requested_share=requested_share,
        s1_train_row_count=s1_train_row_count,
        selected_row_ids_sha256=stable_sha256(selected_ids),
        selection_strata=strata,
    )


def _validate_row(
    row: Any,
    *,
    component: Component,
    contract: Mapping[str, Any],
    expected_tools: Sequence[Mapping[str, Any]],
    prompt_sha256: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate one aggregate input row against its stage and live contract."""

    if not isinstance(row, Mapping):
        raise BundleBuildError(f"{component.section} corpus row is not an object")
    if list(row) != ["row_id", "messages", "tools", "metadata"]:
        raise BundleBuildError(
            f"{component.section} corpus row field order changed"
        )
    row_id = _required_string(row.get("row_id"), "row_id")
    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(messages, list) or not messages:
        raise BundleBuildError(f"{row_id} has no messages")
    if not isinstance(tools, list) or tools != list(expected_tools):
        raise BundleBuildError(f"{row_id} does not use the current tool contract")
    if not isinstance(metadata, dict):
        raise BundleBuildError(f"{row_id} has no metadata object")
    if any(not isinstance(message, dict) for message in messages):
        raise BundleBuildError(f"{row_id} has a non-object message")
    if (
        messages[0].get("role") != "system"
        or messages[0].get("content") != contract["runtime"]["system_prompt"]
    ):
        raise BundleBuildError(f"{row_id} does not use the current system prompt")
    expected_metadata = {
        "step": "S1",
        "question_family": component.question_family,
        "schema_version": contract["runtime"]["row_schema_version"],
        "system_prompt_sha256": prompt_sha256,
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise BundleBuildError(
                f"{row_id} metadata {key} is {metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    source_trace = metadata.get("source_trace")
    if not isinstance(source_trace, str) or not source_trace.startswith(
        component.section + "."
    ):
        raise BundleBuildError(
            f"{row_id} does not belong to stage {component.section}"
        )
    split_unit = metadata.get("split_unit")
    if not isinstance(split_unit, str) or not split_unit:
        raise BundleBuildError(f"{row_id} has no split_unit")
    return row_id, messages, tools, metadata


def project_row(
    row_id: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate one generic multi-turn row at its final/current user turn."""

    user_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if not user_indexes:
        raise BundleBuildError(f"{row_id} has no user message")
    last_user_index = user_indexes[-1]
    expected_messages = [dict(message) for message in messages[last_user_index + 1 :]]
    if not expected_messages:
        raise BundleBuildError(f"{row_id} has no expected suffix after its last user")
    question = {
        "record_id": row_id,
        "split": split,
        "messages": [dict(message) for message in messages[: last_user_index + 1]],
        "tools": [dict(tool) for tool in tools],
        "metadata": dict(metadata),
    }
    answer = {
        "record_id": row_id,
        "split": split,
        "expected_messages": expected_messages,
    }
    return question, answer


def _write_jsonl_record(destination: BinaryIO, value: Mapping[str, Any]) -> None:
    destination.write((canonical_json(value) + "\n").encode("utf-8"))


def _component_manifest_entry(
    component: Component, repo_root: Path
) -> dict[str, Any]:
    return {
        "stage": component.section,
        "dataset_id": component.manifest["dataset_id"],
        "manifest_path": _repo_relative(component.manifest_path, repo_root),
        "manifest_sha256": component.manifest_sha256,
        "manifest_schema_version": component.manifest["schema_version"],
        "corpus_schema_version": component.manifest["corpus_schema_version"],
        "files": component.files,
    }


def build(
    *,
    repo_root: Path = REPO_ROOT,
    component_root: Path | None = None,
    corpus_root: Path | None = None,
    evaluator_root: Path | None = None,
    contract_path: Path | None = None,
    overlay_path: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the aggregate S1 corpus and evaluator."""

    repo_root = repo_root.resolve()
    contract_path = (
        contract_path or repo_root / "runtime/world_model/contract.yaml"
    ).resolve()
    contract = load_contract(contract_path)
    overlay = None
    replay_source = None
    if overlay_path is not None:
        contract, overlay = apply_s1_overlay(
            contract, overlay_path, base_contract_path=contract_path
        )
        replay_source = overlay["s0_replay"]
    train_normalization = (
        overlay.get("train_normalization") if overlay is not None else None
    )
    component_root = (
        component_root or repo_root / "data/world_model_v3/corpora/s1"
    ).resolve()
    default_corpus_root = (
        repo_root / overlay["corpus_root"] if overlay else component_root
    )
    default_evaluator_root = repo_root / (
        overlay["evaluator_root"] if overlay else "data/world_model_v3/eval/s1"
    )
    corpus_root = (corpus_root or default_corpus_root).resolve()
    evaluator_root = (evaluator_root or default_evaluator_root).resolve()
    prompt_sha256 = system_prompt_sha256(contract)
    expected_tools = tool_definitions(contract)
    contract_sha256 = sha256_file(contract_path)
    disabled_s1_5 = _validate_disabled_stage(contract)
    components = _load_components(
        component_root=component_root,
        repo_root=repo_root,
        contract=contract,
        use_contract_directories=overlay is not None,
    )
    s1_train_row_count = sum(
        component.files["train.jsonl"]["row_count"]
        for component in components
    )
    replay = _load_replay_selection(
        repo_root=repo_root,
        contract=contract,
        expected_tools=expected_tools,
        prompt_sha256=prompt_sha256,
        s1_train_row_count=s1_train_row_count,
        replay_source=replay_source,
    )

    batch = AtomicBatch()
    try:
        corpus_temporary = {
            split: batch.path_for(corpus_root / f"{split}.jsonl")
            for split in SPLITS
        }
        question_temporary = {
            split: batch.path_for(evaluator_root / f"{split}_questions.jsonl")
            for split in EVALUATION_SPLITS
        }
        answer_temporary = {
            split: batch.path_for(evaluator_root / f"{split}_answer_key.jsonl")
            for split in EVALUATION_SPLITS
        }
        row_ids: set[str] = set()
        row_counts: dict[str, int] = {}
        family_counts: dict[str, dict[str, int]] = {}
        stage_counts: dict[str, dict[str, int]] = {}
        training_evidence_ids = (
            EvidenceSequence() if train_normalization is not None else None
        )

        for split in SPLITS:
            split_count = 0
            split_families: Counter[str] = Counter()
            split_stages: Counter[str] = Counter()
            question_output: BinaryIO | None = None
            answer_output: BinaryIO | None = None
            with corpus_temporary[split].open("wb") as corpus_output:
                try:
                    if split in EVALUATION_SPLITS:
                        question_output = question_temporary[split].open("wb")
                        answer_output = answer_temporary[split].open("wb")
                    for component in components:
                        source_path = (
                            component.manifest_path.parent / f"{split}.jsonl"
                        )
                        component_count = 0
                        with source_path.open("rb") as source:
                            for line_number, raw_line in enumerate(source, start=1):
                                if not raw_line.strip():
                                    continue
                                try:
                                    row = json.loads(raw_line)
                                except (
                                    UnicodeDecodeError,
                                    json.JSONDecodeError,
                                ) as error:
                                    raise BundleBuildError(
                                        "invalid JSON at "
                                        f"{source_path}:{line_number}: {error}"
                                    ) from error
                                row_id, messages, tools, metadata = _validate_row(
                                    row,
                                    component=component,
                                    contract=contract,
                                    expected_tools=expected_tools,
                                    prompt_sha256=prompt_sha256,
                                )
                                if row_id in row_ids:
                                    raise BundleBuildError(
                                        f"duplicate aggregate row ID {row_id}"
                                    )
                                row_ids.add(row_id)
                                if (
                                    split == "train"
                                    and training_evidence_ids is not None
                                ):
                                    train_row = rekey_training_evidence(
                                        row, training_evidence_ids
                                    )
                                    corpus_output.write(
                                        (ordered_json(train_row) + "\n").encode(
                                            "utf-8"
                                        )
                                    )
                                else:
                                    corpus_output.write(
                                        raw_line.rstrip(b"\r\n") + b"\n"
                                    )
                                component_count += 1
                                split_count += 1
                                split_families[component.question_family] += 1
                                split_stages[component.directory] += 1
                                if split in EVALUATION_SPLITS:
                                    assert question_output is not None
                                    assert answer_output is not None
                                    question, answer = project_row(
                                        row_id,
                                        messages,
                                        tools,
                                        metadata,
                                        split,
                                    )
                                    _write_jsonl_record(question_output, question)
                                    _write_jsonl_record(answer_output, answer)
                        expected_count = component.files[f"{split}.jsonl"]["row_count"]
                        if component_count != expected_count:
                            raise BundleBuildError(
                                f"{component.section} {split} has "
                                f"{component_count} rows, "
                                f"expected {expected_count}"
                            )
                    if split == "train":
                        for raw_line, row in replay.rows:
                            row_id, metadata = _validate_replay_row(
                                row,
                                contract=contract,
                                expected_tools=expected_tools,
                                prompt_sha256=prompt_sha256,
                            )
                            if row_id in row_ids:
                                raise BundleBuildError(
                                    f"duplicate aggregate row ID {row_id}"
                                )
                            row_ids.add(row_id)
                            if training_evidence_ids is not None:
                                replay_row = normalize_s0_replay_row(
                                    row,
                                    contract,
                                    replay.metadata_additions,
                                )
                                train_row = rekey_training_evidence(
                                    replay_row, training_evidence_ids
                                )
                                corpus_output.write(
                                    (ordered_json(train_row) + "\n").encode(
                                        "utf-8"
                                    )
                                )
                            else:
                                corpus_output.write(raw_line)
                            split_count += 1
                            split_families[str(metadata["question_family"])] += 1
                            split_stages["s0_replay"] += 1
                finally:
                    if question_output is not None:
                        question_output.close()
                    if answer_output is not None:
                        answer_output.close()
            row_counts[split] = split_count
            family_counts[split] = dict(split_families)
            stage_counts[split] = dict(split_stages)

        aggregate_files = {
            f"{split}.jsonl": {
                "row_count": row_counts[split],
                "sha256": sha256_file(corpus_temporary[split]),
            }
            for split in SPLITS
        }
        contract_pin = {
            "path": _repo_relative(contract_path, repo_root),
            "sha256": contract_sha256,
            "row_schema_version": contract["runtime"]["row_schema_version"],
            "system_prompt_sha256": prompt_sha256,
            "tool_contract_version": contract["runtime"]["tool_contract_version"],
            "tool_definitions_sha256": stable_sha256(expected_tools),
        }
        if overlay is not None:
            contract_pin["s1_overlay"] = overlay_pin(overlay_path, overlay)
        replay_count = len(replay.rows)
        replay_manifest = {
            "stage": "S0",
            "dataset_id": replay.manifest["dataset_id"],
            "manifest_path": _repo_relative(replay.manifest_path, repo_root),
            "manifest_sha256": replay.manifest_sha256,
            "manifest_schema_version": replay.manifest["schema_version"],
            "corpus_schema_version": replay.manifest["corpus_schema_version"],
            "source_file": {
                "path": _repo_relative(replay.source_path, repo_root),
                "row_count": replay.source_row_count,
                "sha256": replay.source_sha256,
            },
            "selection": {
                "algorithm": "sha256_seeded_row_id_sort_v1",
                "rank_input": "{seed}:{row_id}",
                "seed": replay.seed,
                "requested_share": replay.requested_share,
                "s1_train_row_count": replay.s1_train_row_count,
                "selected_row_count": replay_count,
                "actual_share": replay_count / row_counts["train"],
                "selected_row_ids_sha256": replay.selected_row_ids_sha256,
                "selected_row_ids_sha256_encoding": "canonical_json_array",
            },
        }
        if replay.selection_strata is not None:
            replay_manifest["selection"]["algorithm"] = "stratified_sha256_v1"
            replay_manifest["selection"]["strata"] = replay.selection_strata
        corpus_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "training_contract": LOSS_CONTRACT,
            "loss_contract": LOSS_CONTRACT,
            "evaluation_contract": EVALUATION_CONTRACT,
            "contract": contract_pin,
            "stage_order": list(STAGE_ORDER),
            "train_block_order": [*STAGE_ORDER, "s0_replay"],
            "disabled_stages": {"S1.5": disabled_s1_5},
            "components": {
                component.directory: _component_manifest_entry(component, repo_root)
                for component in components
            },
            "s0_replay": replay_manifest,
            "row_counts": row_counts,
            "total_row_count": sum(row_counts.values()),
            "family_counts": family_counts,
            "stage_counts": stage_counts,
            "files": aggregate_files,
        }
        if train_normalization is not None:
            corpus_manifest["train_normalization"] = {
                "evidence_id_policy": train_normalization[
                    "evidence_id_policy"
                ],
                "s0_metadata_additions_removed": list(
                    replay.metadata_additions
                ),
            }
        corpus_manifest_path = corpus_root / "manifest.json"
        corpus_manifest_temporary = batch.path_for(corpus_manifest_path)
        write_json(corpus_manifest_temporary, corpus_manifest)
        corpus_manifest_sha256 = sha256_file(corpus_manifest_temporary)

        panels: dict[str, Any] = {}
        for split in EVALUATION_SPLITS:
            questions_path = evaluator_root / f"{split}_questions.jsonl"
            answer_key_path = evaluator_root / f"{split}_answer_key.jsonl"
            questions_sha256 = sha256_file(question_temporary[split])
            panels[split] = {
                "questions_path": _repo_relative(questions_path, repo_root),
                "questions_sha256": questions_sha256,
                "answer_key_path": _repo_relative(answer_key_path, repo_root),
                "answer_key_sha256": sha256_file(answer_temporary[split]),
                "row_count": row_counts[split],
                "family_counts": family_counts[split],
                f"{split}_panel_id": questions_sha256,
            }
        evaluator_manifest = {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "evaluation_contract": EVALUATION_CONTRACT,
            "loss_contract": LOSS_CONTRACT,
            "corpus_manifest": _repo_relative(corpus_manifest_path, repo_root),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "contract": contract_pin,
            "stage_order": list(STAGE_ORDER),
            "validation": panels["validation"],
            "test": panels["test"],
        }
        evaluator_manifest_path = evaluator_root / "manifest.json"
        evaluator_manifest_temporary = batch.path_for(evaluator_manifest_path)
        write_json(evaluator_manifest_temporary, evaluator_manifest)
        evaluator_manifest_sha256 = sha256_file(evaluator_manifest_temporary)

        batch.commit()
    except BaseException:
        batch.cleanup()
        raise

    return {
        "corpus_manifest": _repo_relative(corpus_manifest_path, repo_root),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "evaluator_manifest": _repo_relative(evaluator_manifest_path, repo_root),
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "row_counts": row_counts,
        "corpus_files": aggregate_files,
        "validation": panels["validation"],
        "test": panels["test"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--component-root", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--contract-path", type=Path)
    parser.add_argument(
        "--overlay",
        type=Path,
        help="apply an S1 overlay such as runtime/world_model/s1_v2.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        repo_root=args.repo_root,
        component_root=args.component_root,
        corpus_root=args.corpus_root,
        evaluator_root=args.evaluator_root,
        contract_path=args.contract_path,
        overlay_path=args.overlay,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
