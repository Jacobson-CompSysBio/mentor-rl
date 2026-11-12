#!/usr/bin/env python3
"""Audit complete S2 verification traces against a production token budget.

Cached lengths are reused only after checking the corpus, token records,
serializer implementations, and tokenizer artifacts against their manifests.
This reports observed coverage, not a proof that other source queries cannot
fit. Historical corpora and their held-out panels are never modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
BEHAVIOR_FIELDS = ("variant", "question_kind", "request_kind", "category", "mode", "direction", "framing",
                   "claim", "graph_claim", "truth_scope", "graph_truth_scope", "connectivity_kind",
                   "edge_query", "layer_family", "truth_family")
COUNT_FIELDS = ("neighbor_count", "unique_neighbor_count", "shared_count", "layer_count", "support_count",
                "scope_size", "input_size", "edge_count", "component_count", "hop_count", "degree",
                "neighbor_counts", "category_counts")


def sha256(path: Path) -> str:
    """Hash a file in bounded memory so large complete traces remain practical."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_paths(root: Path = ROOT) -> dict[str, Path]:
    """Name the current thirteen families without selecting historical snapshots."""
    base = root / "data/world_model_v4"
    return {"S2.1": base / "corpora/s2_v1_single_edge", **{
        f"S2.{number}": base / f"diagnostics/s2_{number}_builder_v1" for number in range(2, 14)
    }}


def load_verified_lengths(directory: Path, root: Path = ROOT,
                          token_cache: Path | None = None) -> tuple[dict, dict[str, dict]]:
    """Verify pinned inputs and return cached whole-trace metrics by row ID.

    A missing token audit raises LookupError, while changed or contradictory
    bytes raise ValueError. These cases must never be treated as fitting rows.
    """
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("audit.json", *(f"{split}.jsonl" for split in SPLITS)):
        pin = manifest["files"][name]
        path = directory / name
        if path.stat().st_size != pin["bytes"] or sha256(path) != pin["sha256"]:
            raise ValueError(f"changed corpus file: {path}")
    audit = json.loads((directory / "audit.json").read_text())
    token_audit = audit.get("tokenizer", {})
    if token_cache is not None:
        cache = json.loads(token_cache.read_text())
        if cache["source_manifest_sha256"] != sha256(manifest_path):
            raise ValueError(f"token cache belongs to a different source manifest: {token_cache}")
        for name, expected in cache["serializer_implementation_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError(f"changed token cache serializer: {name}")
        token_audit = cache["tokenizer_audit"]
    if not token_audit.get("current_turn_masks_checked") or token_audit.get("truncation_applied") is not False:
        raise LookupError("no verified untruncated current-turn token audit")
    records = token_audit["row_counts"]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != token_audit["row_token_counts_sha256"]:
        raise ValueError(f"changed cached token records: {directory}")
    by_id = {record["row_id"]: record for record in records}
    if len(by_id) != len(records) or len(records) != token_audit["rows"]:
        raise ValueError(f"duplicate or missing cached token record: {directory}")
    for name, expected in manifest["implementation_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"changed pinned implementation: {name}")
    tokenizer = cache["tokenizer_pin"] if token_cache is not None else manifest["tokenizer"]
    for name, expected in tokenizer["artifacts"].items():
        if sha256(root / tokenizer["path"] / name) != expected:
            raise ValueError(f"changed pinned tokenizer artifact: {name}")
    return manifest, by_id


def tokenize_missing_corpus(directory: Path, cache_path: Path, root: Path = ROOT) -> Path:
    """Create a separately pinned token cache without changing older artifacts."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from runtime.world_model.s2_3_tokens import audit_rows, load_tokenizer

    manifest = json.loads((directory / "manifest.json").read_text())
    tokenizer_pin = json.loads((root / "runtime/world_model/s2_3_v1.json").read_text())["tokenizer"]
    implementations = {name: sha256(root / name) for name in (
        "runtime/world_model/s2_3_tokens.py", "runtime/world_model_training.py")}
    source_manifest_sha256 = sha256(directory / "manifest.json")
    if cache_path.exists():
        load_verified_lengths(directory, root, cache_path)
        return cache_path
    for name in ("audit.json", *(f"{split}.jsonl" for split in SPLITS)):
        pin = manifest["files"][name]
        if sha256(directory / name) != pin["sha256"]:
            raise ValueError(f"changed source before fresh token audit: {directory / name}")
    tokenizer = load_tokenizer(tokenizer_pin)

    def rows():
        count, start = 0, time.monotonic()
        for split in SPLITS:
            with (directory / f"{split}.jsonl").open() as stream:
                for line in stream:
                    count += 1
                    if count % 2000 == 0:
                        print(f"{directory.name}: token audit row {count} "
                              f"({time.monotonic() - start:.1f}s)", file=sys.stderr, flush=True)
                    yield json.loads(line)

    token_audit = audit_rows(rows(), tokenizer, 8192)
    cache = {"schema_version": "mentor-rl-s2-separate-token-cache-v1",
             "source_manifest_sha256": source_manifest_sha256,
             "source_files": manifest["files"], "tokenizer_pin": tokenizer_pin,
             "serializer_implementation_sha256": implementations, "tokenizer_audit": token_audit}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")
    load_verified_lengths(directory, root, cache_path)
    return cache_path


def _counter() -> dict:
    return {"rows": 0, "retained": 0, "excluded": 0, "min_tokens": None, "max_tokens": 0}


def _add(counter: dict, tokens: int, max_length: int) -> None:
    counter["rows"] += 1
    counter["retained" if tokens <= max_length else "excluded"] += 1
    counter["min_tokens"] = min(counter["min_tokens"] or tokens, tokens)
    counter["max_tokens"] = max(counter["max_tokens"], tokens)


def audit_corpus(family: str, directory: Path, max_length: int, root: Path = ROOT,
                 token_cache: Path | None = None) -> dict[str, Any]:
    """Summarize exclusions by behavior, framing, split, and observed source size."""
    if not (directory / "manifest.json").exists():
        return {"family": family, "path": str(directory.relative_to(root)), "status": "missing_artifacts"}
    try:
        manifest, lengths = load_verified_lengths(directory, root, token_cache)
    except LookupError as error:
        return {"family": family, "path": str(directory.relative_to(root)), "status": "not_token_audited",
                "reason": str(error)}
    totals = _counter()
    split_counts = {split: _counter() for split in SPLITS}
    dimensions = {field: defaultdict(_counter) for field in ("variant", "question_kind", "framing")}
    behaviors, source_sizes = {}, {}
    seen = set()
    excluded_ids = []
    for split in SPLITS:
        count = 0
        with (directory / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                row_id = row["row_id"]
                if row_id in seen or row_id not in lengths:
                    raise ValueError(f"unexpected or repeated row ID: {row_id}")
                seen.add(row_id)
                count += 1
                tokens = lengths[row_id]["tokens"]
                if type(tokens) is not int or tokens < 1:
                    raise ValueError(f"invalid token length: {row_id}")
                metadata = row["metadata"]
                declared_family = metadata["question_family"]
                if (declared_family != family and not declared_family.startswith(family + ".")) or metadata["split"] != split:
                    raise ValueError(f"contradictory family or split: {row_id}")
                _add(totals, tokens, max_length)
                _add(split_counts[split], tokens, max_length)
                for field, groups in dimensions.items():
                    _add(groups[str(metadata.get(field))], tokens, max_length)
                key = json.dumps({field: metadata.get(field) for field in BEHAVIOR_FIELDS}, sort_keys=True)
                record = behaviors.setdefault(key, {"behavior": json.loads(key), "splits": {
                    name: _counter() for name in SPLITS}, **_counter()})
                _add(record, tokens, max_length)
                _add(record["splits"][split], tokens, max_length)
                for field in COUNT_FIELDS:
                    value = metadata.get(field)
                    entries = (list(value.items()) if isinstance(value, dict) else
                               list(enumerate(value)) if isinstance(value, list) else [(None, value)])
                    for component, count_value in entries:
                        if type(count_value) in (int, float):
                            count_field = field if component is None else f"{field}.{component}"
                            size = source_sizes.setdefault((count_field, count_value), {
                                "field": count_field, "value": count_value, **_counter()})
                            _add(size, tokens, max_length)
                if tokens > max_length:
                    excluded_ids.append(row_id)
        if count != manifest["row_counts"][split]:
            raise ValueError(f"split row count mismatch: {family} {split}")
    if seen != set(lengths):
        raise ValueError(f"cached token rows do not match corpus rows: {directory}")
    for record in behaviors.values():
        record["observed_coverage"] = ("no_fitting_verification_witness" if not record["retained"] else
            "partial_source_range" if record["excluded"] else "all_observed_rows_fit")
    token_audit = (json.loads(token_cache.read_text())["tokenizer_audit"] if token_cache is not None else
                   json.loads((directory / "audit.json").read_text())["tokenizer"])
    return {"family": family, "path": str(directory.relative_to(root)), "status": "verified_cached_lengths",
            "manifest_sha256": sha256(directory / "manifest.json"), "totals": totals, "splits": split_counts,
            "source_files": {name: manifest["files"][name] for name in ("audit.json", *(f"{split}.jsonl" for split in SPLITS))},
            "separate_token_cache": ({"path": str(token_cache.relative_to(root)), "sha256": sha256(token_cache)}
                                      if token_cache is not None else None),
            "dimensions": {field: dict(sorted(groups.items())) for field, groups in dimensions.items()},
            "behaviors": [behaviors[key] for key in sorted(behaviors)],
            "source_sizes": [source_sizes[key] for key in sorted(source_sizes)],
            "excluded_row_ids": excluded_ids,
            "token_count_cache_sha256": token_audit["row_token_counts_sha256"]}


def build_report(root: Path = ROOT, max_length: int = 8192, sources: dict[str, Path] | None = None,
                 tokenize_missing: bool = False, cache_dir: Path | None = None) -> dict[str, Any]:
    """Inspect each current family and preserve explicit gaps in measured coverage."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    paths = corpus_paths(root)
    paths.update(sources or {})
    families = []
    for family, path in paths.items():
        path = path if path.is_absolute() else root / path
        cache = None
        audit_path = path / "audit.json"
        if tokenize_missing and audit_path.exists() and not json.loads(audit_path.read_text()).get("tokenizer"):
            cache = tokenize_missing_corpus(path, (cache_dir or root / "diagnostics/s2_production_8k/token_caches") /
                                           f"{family.replace('.', '_')}.json", root)
        families.append(audit_corpus(family, path, max_length, root, cache))
    measured = [family for family in families if family["status"] == "verified_cached_lengths"]
    return {"schema_version": "mentor-rl-s2-context-audit-v1", "max_length": max_length,
            "length_scope": "complete tokenizer wire including tools, supplied evidence, all current-turn reasoning and final answer",
            "truncation_applied": False, "historical_artifacts_modified": False,
            "coverage_limit": "No-fitting verification witnesses are candidate-pool gaps, not proof that a behavior is universally too long. Fresh production rows need their own tokenizer audit.",
            "totals": {key: sum(family["totals"][key] for family in measured) for key in ("rows", "retained", "excluded")},
            "unmeasured_families": [family["family"] for family in families if family not in measured],
            "families": families}


def main() -> None:
    """Write a new reviewable report while leaving verification corpora frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="FAMILY=PATH")
    parser.add_argument("--tokenize-missing", action="store_true")
    args = parser.parse_args()
    sources = dict(item.split("=", 1) for item in args.source)
    if set(sources) - set(corpus_paths()):
        parser.error("--source requires a known S2 family")
    report = build_report(max_length=args.max_length, sources={key: Path(path) for key, path in sources.items()},
                          tokenize_missing=args.tokenize_missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "max_length": args.max_length,
                      "totals": report["totals"], "unmeasured_families": report["unmeasured_families"]}, indent=2))


if __name__ == "__main__":
    main()
