"""Facts a stage page shows beside the traces: splits, tools, data sources, and text.

They come from small files (corpus manifests, the stage contract, and the
descriptions file), so render.py reads them on every build and editing the
text or a contract needs no new corpus scan.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import yaml

# Keys that name a file or folder in a contract or manifest, in the order tried.
PATH_KEYS = ("path", "source_path", "manifest_path", "source_flist_path", "output_directory", "file_name")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Contract argument types, in the words the JSON schemas use.
TYPE_WORDS = {"str": "string", "int": "integer", "bool": "boolean", "dict": "object", "list": "array"}


def load(path: Path):
    """Read a JSON or YAML file."""
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def descriptions(stage: dict) -> dict:
    """The stage's editable text: families, versions, question types, and tools."""
    path = Path(stage.get("descriptions") or "")
    text = load(path) if path.is_file() else {}
    return {key: text.get(key) or {} for key in ("families", "versions", "types", "tools")}


def splits(entry: dict) -> dict | None:
    """Rows per split, replay rows, and rows per question type and split for one version.

    S0 lists its type counts in the corpus manifest; S1 lists them in each
    component manifest. None for a version whose corpus is not on disk.
    """
    if not entry.get("corpus_root"):
        return None
    root = Path(entry.get("repo_root", "."))
    corpus = root / entry["corpus_root"]
    manifest = load(corpus / "manifest.json")
    types: dict[str, dict[str, int]] = {}

    def add(form_counts: dict) -> None:
        for split, counts in form_counts.items():
            for trace, rows in counts.items():
                types.setdefault(trace, {})[split] = rows

    add(manifest.get("form_counts", {}))
    for component in manifest.get("stage_order", []):
        pinned = manifest.get("components", {}).get(component, {}).get("manifest_path")
        path = root / pinned if pinned else corpus / component / "manifest.json"
        if path.is_file():
            add(load(path).get("form_counts", {}))
    replay = manifest.get("stage_counts", {}).get("train", {}).get("s0_replay", 0)
    return {"rows": manifest["row_counts"], "replay": replay, "types": types}


def parameters(schema: dict, arg_types: dict) -> list[dict]:
    """Describe each argument of a tool's JSON schema in words."""
    required = set(schema.get("required", []))
    found = []
    for name, spec in schema.get("properties", {}).items():
        kind = spec.get("type") or arg_types.get(name, "")
        if kind == "array":
            kind = f"list of {spec.get('items', {}).get('type', 'value')}s"
        notes = []
        if "description" in spec:
            notes.append(spec["description"])
        if "enum" in spec:
            notes.append("one of " + ", ".join(map(str, spec["enum"])))
        if spec.get("minLength"):
            notes.append(f"at least {spec['minLength']} character{'s' if spec['minLength'] > 1 else ''}")
        if spec.get("minItems"):
            notes.append(f"at least {spec['minItems']} item{'s' if spec['minItems'] > 1 else ''}")
        if spec.get("maxItems"):
            notes.append(f"at most {spec['maxItems']} items")
        if spec.get("uniqueItems"):
            notes.append("no repeats")
        if "pattern" in spec:
            notes.append(f"matches {spec['pattern']}")
        found.append({"name": name, "type": kind, "required": name in required, "notes": notes})
    return found


def tool_docs(contract_path: str | None, tools: list | None) -> dict:
    """Documentation for a version's tools: their row schemas joined with the contract.

    contract.yaml (S0, S1) adds input normalization, the result fields for each
    status, and the provenance channel. An S2 config lists its tools with
    descriptions instead. Contract tools that rows do not offer are listed
    separately, with why they are absent.
    """
    contract = load(Path(contract_path)) if contract_path and Path(contract_path).is_file() else {}
    section = contract.get("tools", {})
    definitions = section.get("definitions", {}) if isinstance(section, dict) else {}
    offered = tools or (section if isinstance(section, list) else [])
    docs = []
    for tool in offered:
        function = tool.get("function", tool)
        definition = definitions.get(function["name"], {})
        docs.append({
            "name": function["name"],
            "description": function.get("description"),
            "parameters": parameters(function.get("parameters", {}), definition.get("args", {})),
            "normalizes": definition.get("normalizes", []),
            "canonical_format": definition.get("canonical_format"),
            "returns": definition.get("returns", {}),
            "candidate_details": definition.get("candidate_details"),
            "provenance_channel": definition.get("provenance_channel"),
        })
    names = {doc["name"] for doc in docs}
    other = []
    for name, definition in definitions.items():
        if name in names:
            continue
        if definition.get("harness_only"):
            note = "Appears only as a harness block; the model does not call it."
        elif definition.get("available_from"):
            note = f"Available from {definition['available_from']}."
        else:
            note = "Defined in the contract but not offered to the model."
        # A contract lists only the arguments a tool takes, so each is required.
        arg_types = definition.get("args", {})
        args = {"properties": {arg: {"type": TYPE_WORDS.get(kind, kind)} for arg, kind in arg_types.items()},
                "required": list(arg_types)}
        other.append({"name": name, "note": note, "parameters": parameters(args, {}),
                      "returns": definition.get("returns", {})})
    error = section.get("error_result") if isinstance(section, dict) else None
    return {"tools": docs, "other": other, "error_result": error}


def pins(value, trail: tuple = ()) -> list[dict]:
    """Find every pinned file in a contract or manifest, with its SHA-256 and the facts beside it.

    A pin is a mapping with a path key (path, source_path, ...), or a mapping
    from absolute paths to SHA-256 values, as in an S2 manifest's source_pins.
    An output_directory without a hash is where a build writes, not a pinned
    input, so it is skipped.
    """
    found = []
    if isinstance(value, dict):
        key = next((k for k in PATH_KEYS if isinstance(value.get(k), str)), None)
        if key:
            prefix = key[:-len("path")] if key.endswith("path") else ""
            sha = value.get(f"{prefix}sha256")
            if not isinstance(sha, str):
                sha = next((v for k, v in value.items() if "sha256" in k and isinstance(v, str)), None)
            facts = {k: v for k, v in value.items() if k != key and "sha256" not in k
                     and isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if sha or key != "output_directory":
                found.append({"name": ".".join(trail) or key, "path": value[key], "sha256": sha, "facts": facts})
        for k, v in value.items():
            if isinstance(k, str) and k.startswith("/") and isinstance(v, str) and SHA256.match(v):
                found.append({"name": ".".join(trail), "path": k, "sha256": v, "facts": {}})
            elif isinstance(v, (dict, list)):
                found.extend(pins(v, trail + (str(k),)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(pins(item, trail + (str(index),)))
    return found


def sources(stage: dict) -> dict | None:
    """The raw data a stage is built from: every file its contract pins.

    A contract may pin a source manifest instead of the files themselves, as
    the S2 config does for its multiplex layers; that manifest's pins are
    listed too. Contracts live under <checkout>/runtime/, so a pinned path is
    resolved against that checkout.
    """
    contract = Path(stage.get("contract") or "")
    if not contract.is_file():
        return None
    checkout = next((parent.parent for parent in contract.parents if parent.name == "runtime"), Path("."))
    found = pins(load(contract))
    for pin in list(found):
        if pin["name"].split(".")[-1] == "source_manifest" and (checkout / pin["path"]).is_file():
            found += [{**inner, "name": f"{pin['name']}.{inner['name']}"}
                      for inner in pins(load(checkout / pin["path"]))]
    return {"file": str(contract), "pins": found}
