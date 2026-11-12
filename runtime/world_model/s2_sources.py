"""Load the pinned native S2 source view and answer layer edge queries.

The original four-file review view is the first implemented source domain.
Its node inventories come from complete endpoint unions, never the historical
S1 membership index. Native direction, loops, and source weights are retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .sources import GENE_ID_PATTERN, file_sha256


SOURCE_MANIFEST_PATH = Path(__file__).with_name("s2_v1_sources.json")
SOURCE_SCHEMA_VERSION = "mentor-rl-world-model-s2-sources-v1"
_HEADERS = {
    ("from", "to", "weight"),
    ("node1", "node2", "edge_weight"),
    ("TF", "target", "score"),
}
_INTERPRETATION = {
    "representation": "complete-source-files",
    "membership": "layer_nodes=union_of_all_source_row_endpoints; multiplex_nodes=union_of_layer_nodes; no registry filtering",
    "undirected_pair_policy": "unordered; one source record per pair; no added reverse source records",
    "directed_pair_policy": "ordered source-to-target; no symmetrization; retain self-loops",
    "duplicate_policy": "reject duplicate unordered pairs for undirected layers and duplicate ordered pairs for directed layers",
    "weights": "parse source decimal text as JSON number; no rescaling, rounding, or float32 conversion",
}


class S2SourceError(ValueError):
    """Report an invalid source pin, interpretation, or source record."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON keys before they can change a source manifest."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise S2SourceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _digest(value: Any) -> str:
    """Hash descriptor JSON using the approved canonical encoding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_pin(path: Path, expected: str) -> None:
    """Recheck complete file bytes even when a parsed layer is cached."""
    try:
        actual = file_sha256(path)
    except OSError as error:
        raise S2SourceError(f"cannot read pinned source {path}: {error}") from error
    if actual != expected:
        raise S2SourceError(f"source SHA-256 mismatch for {path}: {actual}")


def _valid_text(value: Any) -> bool:
    """Recognize an explicit nonempty manifest string."""
    return isinstance(value, str) and bool(value.strip())


def _valid_digest(value: Any) -> bool:
    """Recognize the lowercase SHA-256 representation used by source pins."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class LayerGraph:
    """One immutable native edge map and its authoritative endpoint inventory."""

    layer_id: str
    directed: bool
    edges: Mapping[tuple[str, str], int | float]
    nodes: frozenset[str]
    source_path: str = ""
    source_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", MappingProxyType(dict(self.edges)))
        object.__setattr__(self, "nodes", frozenset(self.nodes))


@dataclass(frozen=True, slots=True)
class GraphView:
    """A versioned set of native layers with the approved edge-tool behavior."""

    multiplex_id: str
    graph_version: str
    layers: Mapping[str, LayerGraph]
    manifest_sha256: str = ""
    source_pins: tuple[tuple[str, str], ...] = ()
    multiplex_nodes: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))
        object.__setattr__(
            self, "multiplex_nodes",
            frozenset().union(*(layer.nodes for layer in self.layers.values())),
        )

    @property
    def layer_order(self) -> tuple[str, ...]:
        """Return the manifest order used for deterministic layer selection."""
        return tuple(self.layers)

    def get_edge(
        self,
        multiplex_id: str,
        layer_id: str,
        gene_ids: Any,
        *,
        evidence_id: str,
    ) -> dict[str, Any]:
        """Check both endpoint memberships and one native edge predicate.

        The caller owns dataset-wide acquisition-ID uniqueness and reuse.
        """
        arguments = {
            "multiplex_id": multiplex_id,
            "layer_id": layer_id,
            "gene_ids": gene_ids,
        }

        def reject(status: str, code: str) -> dict[str, Any]:
            return {
                "status": status,
                "provenance": {"registry": [], "graph": []},
                "result": {"issue_code": code, "arguments": arguments},
            }

        if not multiplex_id or not layer_id:
            return reject("invalid_input", "MISSING_SCOPE")
        if multiplex_id != self.multiplex_id:
            return reject("not_found", "UNKNOWN_MULTIPLEX")
        if not isinstance(layer_id, str) or layer_id not in self.layers:
            return reject("not_found", "UNKNOWN_LAYER")
        if not isinstance(gene_ids, list) or len(gene_ids) != 2:
            return reject("invalid_input", "INVALID_PAIR")
        if any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)
               for gene in gene_ids):
            return reject("invalid_input", "INVALID_GENE_INPUT")
        if gene_ids[0] == gene_ids[1]:
            return reject("invalid_input", "INVALID_PAIR")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise S2SourceError("a successful acquisition requires an evidence ID")

        layer = self.layers[layer_id]
        pair = tuple(gene_ids) if layer.directed else tuple(sorted(gene_ids))
        missing = sorted(set(gene_ids) - self.multiplex_nodes)
        outside = sorted((set(gene_ids) & self.multiplex_nodes) - layer.nodes)
        exists = None if missing else pair in layer.edges
        return {
            "status": "not_found" if missing else "resolved",
            "provenance": {
                "registry": [],
                "graph": [{
                    "multiplex_id": self.multiplex_id,
                    "graph_version": self.graph_version,
                    "layer_scope": {
                        "kind": "layers", "layer_ids": [layer_id], "layer_family": None,
                    },
                    "evidence_id": evidence_id,
                }],
            },
            "result": {
                "gene_ids": list(pair),
                "multiplex_id": self.multiplex_id,
                "layer_id": layer_id,
                "directed": layer.directed,
                "edge_exists": exists,
                "weight": layer.edges[pair] if exists else None,
                "missing_gene_ids": missing,
                "not_in_layer_gene_ids": outside,
            },
        }


@lru_cache(maxsize=16)
def _read_layer(path_text: str, sha256: str, directed: bool, layer_id: str) -> LayerGraph:
    """Parse one exact pinned file, retaining native keys and JSON weights."""
    path = Path(path_text)
    digest = hashlib.sha256()
    edges: dict[tuple[str, str], int | float] = {}
    nodes: set[str] = set()
    try:
        with path.open("rb") as handle:
            header = next(handle, b"")
            digest.update(header)
            if tuple(header.decode("utf-8").rstrip("\r\n").split("\t")) not in _HEADERS:
                raise S2SourceError(f"unrecognized edge header in {path}")
            for line_number, raw in enumerate(handle, 2):
                digest.update(raw)
                if not raw.strip():
                    continue
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
                if len(fields) != 3 or any(not GENE_ID_PATTERN.fullmatch(gene) for gene in fields[:2]):
                    raise S2SourceError(f"{path}:{line_number}: invalid native edge row")
                left, right, weight_text = fields
                weight = json.loads(weight_text)
                if type(weight) not in (int, float) or (
                    isinstance(weight, float) and not math.isfinite(weight)
                ):
                    raise S2SourceError(f"{path}:{line_number}: weight is not a finite JSON number")
                pair = (left, right) if directed else tuple(sorted((left, right)))
                if pair in edges:
                    raise S2SourceError(f"{path}:{line_number}: duplicate native edge key {pair}")
                edges[pair] = weight
                nodes.update((left, right))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S2SourceError(f"cannot parse pinned source {path}: {error}") from error
    if digest.hexdigest() != sha256:
        raise S2SourceError(f"source changed while reading {path}")
    return LayerGraph(layer_id, directed, edges, frozenset(nodes), path_text, sha256)


def load_source_view(manifest_path: Path = SOURCE_MANIFEST_PATH) -> GraphView:
    """Validate source/interpretation pins and load the original S2 native view."""
    manifest_path = Path(manifest_path).resolve()
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S2SourceError(f"cannot read source manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "graph_version", "descriptor"}:
        raise S2SourceError("source manifest has an unsupported shape")
    if manifest["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise S2SourceError("unsupported S2 source schema")
    descriptor = manifest["descriptor"]
    if (
        not isinstance(descriptor, dict)
        or not _valid_digest(manifest["graph_version"])
        or _digest(descriptor) != manifest["graph_version"]
    ):
        raise S2SourceError("graph descriptor SHA-256 mismatch")
    if set(descriptor) != set(_INTERPRETATION) | {
        "multiplex_id", "layer_order", "layers", "direction_metadata",
    } or not _valid_text(descriptor["multiplex_id"]):
        raise S2SourceError("invalid graph descriptor fields")
    if any(descriptor.get(key) != value for key, value in _INTERPRETATION.items()):
        raise S2SourceError("unsupported native graph interpretation")
    layer_specs = descriptor.get("layers")
    if not isinstance(layer_specs, list) or not layer_specs:
        raise S2SourceError("the source view needs registered layers")
    if any(
        not isinstance(spec, dict)
        or set(spec) != {"layer_id", "source_path", "source_sha256", "directed"}
        or type(spec["directed"]) is not bool
        or not _valid_text(spec["layer_id"])
        or not _valid_text(spec["source_path"])
        or not _valid_digest(spec["source_sha256"])
        for spec in layer_specs
    ):
        raise S2SourceError("invalid native layer descriptor")
    layer_ids = [spec["layer_id"] for spec in layer_specs]
    if descriptor.get("layer_order") != layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise S2SourceError("layer order or uniqueness does not match the descriptor")

    source_pins: list[tuple[str, str]] = []
    metadata = descriptor.get("direction_metadata")
    if not isinstance(metadata, list):
        raise S2SourceError("direction metadata must be an explicit pinned list")
    if any(spec["directed"] for spec in layer_specs) and not metadata:
        raise S2SourceError("directed sources require pinned direction metadata")
    for reference in metadata:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or not _valid_text(reference["path"])
            or not _valid_digest(reference["sha256"])
        ):
            raise S2SourceError("invalid direction metadata pin")
        path = (manifest_path.parent / reference["path"]).resolve()
        _require_pin(path, reference["sha256"])
        source_pins.append((str(path), reference["sha256"]))

    layers: dict[str, LayerGraph] = {}
    for spec in layer_specs:
        path = (manifest_path.parent / spec["source_path"]).resolve()
        _require_pin(path, spec["source_sha256"])
        layers[spec["layer_id"]] = _read_layer(
            str(path), spec["source_sha256"], spec["directed"], spec["layer_id"],
        )
        source_pins.append((str(path), spec["source_sha256"]))
    return GraphView(
        descriptor["multiplex_id"], manifest["graph_version"], layers,
        hashlib.sha256(raw).hexdigest(), tuple(source_pins),
    )
