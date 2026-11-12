"""Collect what a stage page needs to list and compare corpus versions.

The page compares gold traces: the first training row for each outcome of a
question type, the rule the S1 catalog uses. For a pair of versions, the
newer one picks its gold traces, and each is paired with the row that has
the same id in the older one, so an edit shows as a before/after of one
question. Rows that a version regenerated have new ids and no pair. Every
training row is also counted, so an edit that no gold trace shows still
appears as an edit count.

A registry version with corpus_root: null has no corpus on disk; the page
lists it with its gate results only.

Each version is read once and every pair is compared in memory. Writes:

  site_data/<stage>/versions/<version>.json  a version's gold traces (plus any
      older rows a newer version's traces pair with), system prompts, and tools
  site_data/<stage>/diffs/<old>_<new>.json   how each gold trace pairs up,
      edit counts per question type, and the held-out check

render.py embeds both in the stage page, so the page holds each trace once.
It reads gate reports itself (gates.py), so re-pinning a gate run needs no
new scan.

    python -m dashboard.diff --stage S1
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import re
import subprocess

import yaml

from scripts.render_world_model_s1_catalog import case_names, gold_key

LABELS = {
    ("system", "content"): "system prompt",
    ("harness", "content"): "context",
    ("user", "content"): "user message",
    ("assistant", "thinking"): "thinking",
    ("assistant", "tool_call"): "tool call",
    ("assistant", "content"): "answer",
    ("tool", "content"): "tool result",
}
KEYS: dict[tuple, tuple] = {}  # one shared copy of each part name, to keep scans small
S2_CATALOG = Path("agents/world_model_v3_curriculum/S2-question-classes.md")


def resolve(entry: dict, path: str) -> Path:
    """Resolve a registry path against the checkout the version was built in."""
    return Path(entry.get("repo_root", ".")) / path


def read_rows(entry: dict):
    """Yield every training row of one version that belongs to its stage.

    A component corpus (S1) is read from its component files, in published
    order; a version that rebuilt only some components points at the older
    component files in its manifest. Its aggregate train.jsonl also holds
    replay rows from the previous stage, which the component files do not.
    A corpus without components (S0, S2) is read from its train.jsonl. S2's
    mixes in replay rows from S1 and S0; they name their source_stage and are
    skipped.
    """
    corpus = resolve(entry, entry["corpus_root"])
    manifest = json.loads((corpus / "manifest.json").read_text())
    files = [corpus / "train.jsonl"]
    if "stage_order" in manifest:
        files = []
        for component in manifest["stage_order"]:
            pinned = manifest.get("components", {}).get(component, {}).get("manifest_path")
            folder = resolve(entry, pinned).parent if pinned else corpus / component
            files.append(folder / "train.jsonl")
    for path in files:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if "source_stage" not in row["metadata"]:
                    yield row


def row_parts(row: dict) -> dict[tuple[int, str], int]:
    """Hash each part of a row under the name a reader would use for it."""
    messages = row["messages"]
    parts = {(-1, "metadata"): row["metadata"], (-1, "tool definitions"): row.get("tools")}
    for index, message in enumerate(messages):
        for field, value in message.items():
            if field == "role":
                continue
            label = LABELS.get((message["role"], field), f"{message['role']} {field}")
            if label == "thinking" and index == len(messages) - 1:
                label = "final thinking"
            parts[(index, label)] = value
    return {KEYS.setdefault(key, key): hash(json.dumps(value, sort_keys=True))
            for key, value in parts.items()}


def changed_labels(old: dict, new: dict) -> set[str]:
    """Name the parts that differ between two hashed versions of one row."""
    return {key[1] for key in old.keys() | new.keys() if old.get(key) != new.get(key)}


def question_type(row: dict) -> str:
    """A row's question type: its source trace (S0, S1), its catalog case (S2),
    else its question family (the S2 pilot)."""
    meta = row["metadata"]
    return meta.get("source_trace") or meta.get("catalog_case_id") or meta["question_family"]


def family_of(trace: str) -> str:
    """S1.3.8 and S2.9.3 belong to S1.3 and S2.9; S0.4 belongs to S0 (as familyOf in compare.js)."""
    match = re.match(r"(S\d+\.\d+)\.", trace) or re.match(r"(S\d+)\.", trace)
    return match.group(1) if match else trace


def type_order(trace: str) -> tuple[int, ...]:
    """Sort S1.1.10 after S1.1.9."""
    return tuple(int(part) for part in re.findall(r"\d+", trace))


def gold_order(key: tuple) -> tuple:
    """Sort gold traces by question type, then by outcome, as the catalog does."""
    return type_order(key[0]), key[1]


def describe(key: tuple) -> str:
    """Write a gold-trace key as 'calls: 1 · resolved / stop · blocked reason'."""
    calls, status, decision, reason = key
    return f"calls: {calls} · {status} / {decision}" + (f" · {reason}" if reason else "")


@functools.cache
def s2_case_names() -> dict[str, str]:
    """Map each S2 catalog case (S2.9.3) to the title of its section in the S2 catalog."""
    text = S2_CATALOG.read_text() if S2_CATALOG.is_file() else ""
    return dict(re.findall(r"^### (S2\.\d+\.\d+): (.+)$", text, re.MULTILINE))


def type_names(rows: list[dict]) -> dict[str, str]:
    """Name each question type: the S1 builder's or S2 catalog's case name, else its fact kind.

    S0 rows carry a fact kind (resolved_symbol) and an input variant; a
    non-canonical variant is part of the name (resolved symbol, casefold input).
    """
    names, sections = {}, {}
    for row in rows:
        meta = row["metadata"]
        name = (meta.get("fact_kind") or meta.get("question_family", "")).replace("_", " ")
        if meta.get("input_variant") not in (None, "canonical"):
            name += f", {meta['input_variant']} input"
        names.setdefault(question_type(row), name)
    for trace in names:
        parts = type_order(trace)  # (1, section, form) for an S1 type
        if parts[0] == 1 and len(parts) == 3:
            if parts[1] not in sections:
                sections[parts[1]] = case_names(parts[1])
            names[trace] = sections[parts[1]].get(parts[2], names[trace])
        elif parts[0] == 2:
            names[trace] = s2_case_names().get(trace, names[trace])
    return names


def scan(entry: dict) -> dict:
    """Read one version once: each row's hashed parts, type, and prompt, and its gold traces.

    Rows share a system prompt and tool list: one per version in S0 and S1, one
    per family in S2. Each distinct pair is kept once, with the families that
    use it, and prompt_of says which one each row uses.
    """
    parts, types, gold, prompt_of, prompts, index = {}, {}, {}, {}, [], {}
    for row in read_rows(entry):
        trace, hashed = question_type(row), row_parts(row)
        parts[row["row_id"]], types[row["row_id"]] = hashed, trace
        gold.setdefault((trace, gold_key(row)), row)
        shared = (hashed[(0, "system prompt")], hashed[(-1, "tool definitions")])
        if shared not in index:
            index[shared] = len(prompts)
            prompts.append({"system_prompt": row["messages"][0]["content"], "tools": row.get("tools"),
                            "families": set()})
        prompts[index[shared]]["families"].add(family_of(trace))
        prompt_of[row["row_id"]] = index[shared]
    # S2 rows are shuffled, so list the prompts in catalog order of their families.
    for prompt in prompts:
        prompt["families"] = sorted(prompt["families"], key=type_order)
    order = sorted(range(len(prompts)), key=lambda i: type_order(prompts[i]["families"][0]))
    renumber = {old: new for new, old in enumerate(order)}
    return {"parts": parts, "types": types, "gold": gold, "prompts": [prompts[i] for i in order],
            "prompt_of": {row_id: renumber[i] for row_id, i in prompt_of.items()}}


def intended_change(entry: dict) -> str | None:
    """Return the registry summary, else the overlay's first comment line."""
    if entry.get("summary"):
        return entry["summary"]
    overlay = resolve(entry, entry["overlay"]) if entry.get("overlay") else None
    if not overlay or not overlay.is_file():
        return None
    first = overlay.read_text().splitlines()[0]
    return first.lstrip("#").strip() if first.startswith("#") else None


def compare(stage: dict, old: dict, new: dict, scans: dict) -> tuple[dict, set[str]]:
    """Diff two registry versions of one stage.

    Returns the JSON record for render.py and the ids of older rows that the
    page needs but the older version's gold traces do not include.
    """
    a, b = scans[old["version"]], scans[new["version"]]

    # Count edits over every training row, by question type.
    rows, fields = defaultdict(Counter), defaultdict(Counter)
    for row_id, trace in b["types"].items():
        rows[trace]["total"] += 1
        if row_id not in a["parts"]:
            rows[trace]["regenerated"] += 1
            continue
        changed = changed_labels(a["parts"][row_id], b["parts"][row_id])
        if changed:
            rows[trace]["edited"] += 1
            fields[trace].update(changed)
    for row_id, trace in a["types"].items():
        if row_id not in b["parts"]:
            rows[trace]["removed"] += 1

    # Pair each gold trace of the newer version with the same row in the older one.
    old_gold_ids = {row["row_id"] for row in a["gold"].values()}
    needed, tally, traces = set(), defaultdict(Counter), []
    for key in sorted(a["gold"].keys() | b["gold"].keys(), key=gold_order):
        trace, after = key[0], b["gold"].get(key)
        before_id, labels = None, []
        if after is None:
            state, before_id = "removed", a["gold"][key]["row_id"]
        elif after["row_id"] in a["parts"]:
            before_id = after["row_id"]
            labels = sorted(changed_labels(a["parts"][before_id], b["parts"][before_id]))
            state = "edited" if labels else "identical"
            if labels and before_id not in old_gold_ids:
                needed.add(before_id)
        elif key in a["gold"]:
            state, before_id = "regenerated", a["gold"][key]["row_id"]
        else:
            state = "added"
        tally[trace][state] += 1
        traces.append({"type": trace, "outcome": describe(key[1]), "state": state,
                       "old": before_id, "new": after["row_id"] if after else None, "fields": labels})

    names = type_names(list(a["gold"].values()) + list(b["gold"].values()))
    types = [{
        "type": trace,
        "name": names.get(trace, ""),
        "rows": {key: rows[trace][key] for key in ("total", "edited", "regenerated", "removed")},
        "fields": dict(fields[trace].most_common()),
        "gold": dict(tally[trace]),
    } for trace in sorted(rows.keys() | tally.keys(), key=type_order)]

    order = [entry["version"] for entry in stage["versions"]]
    between = stage["versions"][order.index(old["version"]) + 1:order.index(new["version"]) + 1]
    files = [json.loads((resolve(entry, entry["corpus_root"]) / "manifest.json").read_text())["files"]
             for entry in (old, new)]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    record = {
        "stage": stage["stage"],
        "old": old["version"],
        "new": new["version"],
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_commit": commit,
        "changes": [{"version": entry["version"], "summary": intended_change(entry)} for entry in between],
        "held_out_frozen": {name: files[0][name]["sha256"] == files[1][name]["sha256"]
                            for name in ("validation.jsonl", "test.jsonl")},
        "gold": dict(sum(tally.values(), Counter())),
        "rows": {key: sum(item["rows"][key] for item in types)
                 for key in ("total", "edited", "regenerated", "removed")},
        "types": types,
        "traces": traces,
    }
    return record, needed


def version_record(entry: dict, found: dict | None, extra: dict[str, dict]) -> dict:
    """One version's gold traces and the rows the page shows, without their shared system prompt.

    Each row names the entry of prompts (system prompt and tools) it uses.

    found is None for a version whose corpus is no longer on disk.
    """
    gold = found["gold"] if found else {}
    shown = {row["row_id"]: row for row in gold.values()} | extra
    return {
        "version": entry["version"],
        "corpus_root": entry["corpus_root"],
        "repo_root": entry.get("repo_root", "."),
        "summary": intended_change(entry),
        "prompts": found["prompts"] if found else [],
        "names": type_names(list(gold.values())),
        "gold": [{"type": key[0], "outcome": describe(key[1]), "row_id": gold[key]["row_id"]}
                 for key in sorted(gold, key=gold_order)],
        "rows": {row_id: {"messages": row["messages"][1:], "metadata": row["metadata"],
                          "prompt": found["prompt_of"][row_id]}
                 for row_id, row in shown.items()},
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    """Write a record for every version and every pair of versions of one stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("config/dashboard/stages.yaml"))
    parser.add_argument("--stage", default="S1")
    parser.add_argument("--out", type=Path, default=Path("site_data"))
    args = parser.parse_args()
    stages = yaml.safe_load(args.registry.read_text())["stages"]
    stage = next(entry for entry in stages if entry["stage"] == args.stage)
    versions = stage["versions"]
    root = args.out / stage["stage"]

    scans = {}
    for entry in versions:
        if entry.get("corpus_root"):
            scans[entry["version"]] = scan(entry)
            print(f"read {entry['version']}: {len(scans[entry['version']]['parts']):,} rows")

    available = [entry for entry in versions if entry["version"] in scans]
    needed = defaultdict(set)
    for index, old in enumerate(available):
        for new in available[index + 1:]:
            record, extra = compare(stage, old, new, scans)
            needed[old["version"]] |= extra
            write_json(root / "diffs" / f"{old['version']}_{new['version']}.json", record)
            print(f"{old['version']} -> {new['version']}: gold {record['gold']}")

    for entry in versions:
        ids = needed[entry["version"]]
        extra = {row["row_id"]: row for row in read_rows(entry) if row["row_id"] in ids} if ids else {}
        write_json(root / "versions" / f"{entry['version']}.json",
                   version_record(entry, scans.get(entry["version"]), extra))


if __name__ == "__main__":
    main()
