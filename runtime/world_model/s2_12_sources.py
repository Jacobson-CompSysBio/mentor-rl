"""Load the separate six-file S2.12 view without changing the four-file view.

The archived bulk metadata pins a version-suffixed file. Its endpoint-only
normalization must reproduce every byte of the current bulk source.
"""

from __future__ import annotations

import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
from typing import Any

from .s2_sources import (
    GraphView, S2SourceError, _INTERPRETATION, _digest, _read_layer,
    _require_pin, _unique_object, _valid_digest, _valid_text,
)


SOURCE_MANIFEST_PATH = Path(__file__).with_name("s2_12_v1_sources.json")
SOURCE_SCHEMA_VERSION = "mentor-rl-world-model-s2-12-sources-v1"
MULTIPLEX_ID = "s2_dataset_review_20261028_pen_v1"
GRAPH_VERSION = "75538f73d7cc84197b7d8d67093fa3133569076f0f2151f89024e8abbf90db29"
_TRANSFORMATION = (
    "Remove only terminal numeric Ensembl version suffixes from each endpoint ID; "
    "preserve header, row order, weights, delimiters, and line endings."
)
_RELATION = "Transformed archived bytes equal current source bytes exactly."
_VERSIONED_ID = re.compile(rb"(ENSG\d{11})(?:\.\d+)?")


def verify_source_transformation(spec: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Verify the archived bulk relation, including every unchanged non-ID byte."""
    fields = {
        "layer_id", "archived_source_path", "archived_source_sha256",
        "interpretation_metadata_path", "interpretation_metadata_sha256",
        "transformation", "source_path", "source_sha256", "verified_relation",
    }
    if (not isinstance(spec, dict) or set(spec) != fields
            or spec["transformation"] != _TRANSFORMATION or spec["verified_relation"] != _RELATION
            or any(not _valid_text(spec[key]) for key in fields)
            or any(not _valid_digest(spec[key]) for key in fields if key.endswith("sha256"))):
        raise S2SourceError("invalid source transformation descriptor")
    paths = {}
    for name in ("archived_source", "interpretation_metadata", "source"):
        path = (directory / spec[name + "_path"]).resolve()
        _require_pin(path, spec[name + "_sha256"])
        paths[name] = path
    metadata = json.loads(paths["interpretation_metadata"].read_bytes(), object_pairs_hook=_unique_object)
    if (metadata.get("configuration", {}).get("network_mode") != "undirected"
            or metadata.get("output", {}).get("table_sha256") != spec["archived_source_sha256"]):
        raise S2SourceError("archived bulk metadata does not pin the undirected source")
    rows = endpoints = stripped = 0
    transformed_digest = hashlib.sha256()
    with paths["archived_source"].open("rb") as archived, paths["source"].open("rb") as current:
        header = next(archived, None)
        if header is None or header != next(current, None):
            raise S2SourceError("bulk transformation changed the header")
        transformed_digest.update(header)
        for old, new in zip_longest(archived, current):
            if old is None or new is None:
                raise S2SourceError("bulk transformation changed the row count")
            fields = old.split(b"\t")
            if len(fields) != 3:
                raise S2SourceError("invalid archived bulk row")
            for position in (0, 1):
                match = _VERSIONED_ID.fullmatch(fields[position])
                if match is None:
                    raise S2SourceError("invalid archived bulk endpoint")
                endpoints += 1
                stripped += fields[position] != match[1]
                fields[position] = match[1]
            transformed = b"\t".join(fields)
            if transformed != new:
                raise S2SourceError("bulk endpoint normalization does not reproduce current bytes")
            transformed_digest.update(transformed)
            rows += 1
    if transformed_digest.hexdigest() != spec["source_sha256"]:
        raise S2SourceError("bulk transformation SHA-256 mismatch")
    return {"layer_id": spec["layer_id"], "source_rows": rows,
            "endpoint_occurrences": endpoints, "removed_version_suffixes": stripped,
            "transformed_sha256": transformed_digest.hexdigest(), "byte_identical": True}


def load_source_view(manifest_path: Path = SOURCE_MANIFEST_PATH) -> GraphView:
    """Validate native pins and load the versioned S2.12 endpoint inventories."""
    manifest_path = Path(manifest_path).resolve()
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S2SourceError(f"cannot read source manifest {manifest_path}: {error}") from error
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "graph_version", "descriptor"}
            or manifest["schema_version"] != SOURCE_SCHEMA_VERSION):
        raise S2SourceError("unsupported S2.12 source manifest")
    descriptor = manifest["descriptor"]
    if (not isinstance(descriptor, dict) or not _valid_digest(manifest["graph_version"])
            or _digest(descriptor) != manifest["graph_version"]):
        raise S2SourceError("graph descriptor SHA-256 mismatch")
    if (set(descriptor) != set(_INTERPRETATION) | {
            "multiplex_id", "layer_order", "layers", "direction_metadata", "source_transformations"}
            or not _valid_text(descriptor["multiplex_id"])
            or any(descriptor[key] != value for key, value in _INTERPRETATION.items())):
        raise S2SourceError("unsupported native graph interpretation")
    specs = descriptor["layers"]
    if (not isinstance(specs, list) or not specs or any(
            not isinstance(spec, dict)
            or set(spec) != {"layer_id", "source_path", "source_sha256", "directed"}
            or type(spec["directed"]) is not bool
            or not _valid_text(spec["layer_id"]) or not _valid_text(spec["source_path"])
            or not _valid_digest(spec["source_sha256"]) for spec in specs)):
        raise S2SourceError("invalid native layer descriptor")
    layer_ids = [spec["layer_id"] for spec in specs]
    if descriptor["layer_order"] != layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise S2SourceError("layer order or uniqueness does not match the descriptor")
    pins = []
    metadata = descriptor["direction_metadata"]
    if not isinstance(metadata, list) or not metadata:
        raise S2SourceError("direction metadata requires pinned references")
    for reference in metadata:
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not _valid_text(reference["path"]) or not _valid_digest(reference["sha256"])):
            raise S2SourceError("invalid direction metadata pin")
        path = (manifest_path.parent / reference["path"]).resolve()
        _require_pin(path, reference["sha256"])
        pins.append((str(path), reference["sha256"]))
    transformations = descriptor["source_transformations"]
    if not isinstance(transformations, list) or len(transformations) != 1:
        raise S2SourceError("the source view requires its archived bulk relation")
    for relation in transformations:
        verify_source_transformation(relation, manifest_path.parent)
        layer = next((spec for spec in specs if spec["layer_id"] == relation["layer_id"]), None)
        if (layer is None or layer["directed"]
                or any(layer[key] != relation[key] for key in ("source_path", "source_sha256"))):
            raise S2SourceError("source transformation does not bind its native layer")
        pin = (str((manifest_path.parent / relation["archived_source_path"]).resolve()),
               relation["archived_source_sha256"])
        pins.append(pin)
    layers = {}
    for spec in specs:
        path = (manifest_path.parent / spec["source_path"]).resolve()
        _require_pin(path, spec["source_sha256"])
        layers[spec["layer_id"]] = _read_layer(
            str(path), spec["source_sha256"], spec["directed"], spec["layer_id"])
        pins.append((str(path), spec["source_sha256"]))
    return GraphView(descriptor["multiplex_id"], manifest["graph_version"], layers,
                     hashlib.sha256(raw).hexdigest(), tuple(pins))
