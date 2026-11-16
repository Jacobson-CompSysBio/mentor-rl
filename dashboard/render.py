"""Render the dashboard site from the collected JSON files.

Writes an index of stage labels (with the hand-written executive summary
above them, once it exists) and one page per stage with versions. The stage
page embeds each version's gold traces once, with a small record per pair
of versions, and its tabs show:

- Compare: the chosen pair side by side, older on the left, then questions
  and splits (or the one version on disk);
- Versions: a gate-rate chart and every version's change and gate results;
- Question types: accuracy by question type, a histogram of how those
  accuracies spread, and what each type asks;
- Failures, when failures.py has collected them: where the held-out answers
  of the current version go wrong, by field, category, and case;
- Tools and Raw data sources: the tool reference and every file the stage
  contract pins.

The browser code is templates/compare.js, charts.js, failures.js, and page.js,
joined in that order. This reads the registry, site_data/, gate reports, manifests,
contracts, and dashboard/descriptions and dashboard/notes, never a corpus,
so a layout or text change reruns in seconds. Pages are self-contained HTML
and open from disk.

It also writes curriculum-dashboard.html: the index and every stage page in
one file, to send to someone. A browser on macOS can hang opening a second
file from a protected folder such as Downloads, so in that file a stage link
reopens the file itself (?stage=S1) and it shows the stage's page instead.

    python -m dashboard.render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt
from markupsafe import Markup
import yaml

from dashboard import facts
from dashboard.gates import comparability, gate_info

TEMPLATES = Path(__file__).parent / "templates"
NOTES = Path(__file__).parent / "notes"
SCRIPTS = ("compare.js", "charts.js", "failures.js", "page.js")
BUNDLE = "curriculum-dashboard.html"


def markdown(path: Path) -> Markup | None:
    """Render a hand-written note, or return None when it does not exist yet."""
    return Markup(MarkdownIt("commonmark").render(path.read_text())) if path.is_file() else None


def offered_tools(version: dict | None) -> tuple[list, dict[str, list[str]]]:
    """Every tool a version's rows offer, once each, and the families that offer each one."""
    tools, families = [], {}
    for prompt in version["prompts"] if version else []:
        for tool in prompt["tools"] or []:
            name = tool.get("function", tool)["name"]
            if name not in families:
                tools.append(tool)
                families[name] = []
            families[name] += [family for family in prompt["families"] if family not in families[name]]
    return tools, families


def stage_data(stage: dict, data: Path) -> dict | None:
    """Gather everything one stage page embeds.

    available lists the versions whose corpus is on disk. With two or more,
    the page compares pairs; with one, it shows that version's gold traces.
    Gate results, splits, tool documentation, data sources, and text are read
    here from small files; only the traces come from diff.py's scan. Each
    version's gate is checked against the nearest earlier version with a gate run.
    """
    label = stage["stage"]
    text = facts.descriptions(stage)
    entries = {entry["version"]: entry for entry in stage.get("versions", [])}
    versions, previous = [], None
    for name, entry in entries.items():
        path = data / label / "versions" / f"{name}.json"
        if not path.is_file():
            continue
        version = json.loads(path.read_text())
        version["gate"] = gate_info(entry.get("gate_report"))
        version["vs_previous"] = None
        if version["gate"]:
            if previous:
                version["vs_previous"] = {"version": previous["version"],
                                          "reasons": comparability(previous["gate"], version["gate"])}
            previous = version
        version["description"] = text["versions"].get(name)
        version["splits"] = facts.splits(entry)
        versions.append(version)
    if not versions:
        return None
    gates = {version["version"]: version["gate"] for version in versions}
    available = [version["version"] for version in versions if version["gold"]]
    pairs, notes = {}, {}
    for path in (data / label / "diffs").glob("*.json"):
        pair = json.loads(path.read_text())
        if pair["old"] in available and pair["new"] in available:
            key = f"{pair['old']}_{pair['new']}"
            pair["not_comparable"] = comparability(gates[pair["old"]], gates[pair["new"]])
            pairs[key] = pair
            note = markdown(NOTES / label / f"{key}.md")
            if note:
                notes[key] = str(note)
    # The page opens on the current version against the one before it.
    current, default = stage.get("current"), None
    if pairs:
        default = available[-2:]
        if current in available and available.index(current) > 0:
            default = [available[available.index(current) - 1], current]
    # Tools describe the current version, or the newest one on disk. When its
    # families offer different tools (S2), each tool names the families offering it.
    shown = current if current in available else (available[-1] if available else None)
    shown_record = next((v for v in versions if v["version"] == shown), None)
    offered, families = offered_tools(shown_record)
    contract = facts.contracts(stage)
    tools = facts.tool_docs(str(contract[0]) if contract else None, offered or None)
    if shown_record and len(shown_record["prompts"]) > 1:
        for doc in tools["tools"]:
            doc["families"] = families[doc["name"]]
    shown_lists = [prompt["tools"] for prompt in shown_record["prompts"]] if shown_record else None
    tools.update(version=shown, text=text["tools"],
                 differs_in=[v["version"] for v in versions if v["prompts"] and shown_record
                             and [prompt["tools"] for prompt in v["prompts"]] != shown_lists])
    # The Failures tab shows the current version's gate run, once failures.py has read it.
    failures = None
    failures_path = data / label / "failures" / f"{current}.json"
    if failures_path.is_file():
        failures = {**json.loads(failures_path.read_text()), "version": current}
    return {"stage": label, "current": current, "versions": versions, "available": available,
            "pairs": pairs, "notes": notes, "default": default,
            "families": text["families"], "type_text": text["types"], "tools": tools,
            "sources": facts.sources(stage), "failures": failures}


def embed(value) -> Markup:
    """JSON for a <script> element: "<" is escaped so it cannot close the element."""
    return Markup(json.dumps(value, ensure_ascii=False).replace("<", "\\u003c"))


def render(registry: Path, data: Path, out: Path) -> None:
    """Write site/index.html, one page per stage that has versions to compare, and
    site/curriculum-dashboard.html, which holds them all in one file."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    script = Markup("\n".join((TEMPLATES / name).read_text() for name in SCRIPTS))
    stages = yaml.safe_load(registry.read_text())["stages"]
    links, bundled = {}, {}
    for stage in stages:
        page_data = stage_data(stage, data)
        if page_data is None:
            continue
        name = stage["stage"]
        context = dict(stage=stage, versions=page_data["available"], compare=bool(page_data["pairs"]),
                       failures=bool(page_data["failures"]), data=embed(page_data), script=script)
        target = out / "stages" / f"{name}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(env.get_template("stage.html.j2").render(home="../index.html", **context),
                          encoding="utf-8")
        links[name] = f"stages/{name}.html"
        bundled[name] = env.get_template("stage.html.j2").render(home=BUNDLE, **context)
        print(f"{target}: {target.stat().st_size / 1e6:.1f} MB")

    out.mkdir(parents=True, exist_ok=True)
    summary = markdown(NOTES / "executive-summary.md")
    index = out / "index.html"
    index.write_text(env.get_template("index.html.j2").render(stages=stages, links=links, summary=summary),
                     encoding="utf-8")
    print(index)
    bundle = out / BUNDLE
    bundle.write_text(env.get_template("index.html.j2").render(
        stages=stages, links={name: f"?stage={name}" for name in bundled}, summary=summary,
        pages=embed(bundled)), encoding="utf-8")
    print(f"{bundle}: {bundle.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("config/dashboard/stages.yaml"))
    parser.add_argument("--data", type=Path, default=Path("site_data"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    render(args.registry, args.data, args.out)


if __name__ == "__main__":
    main()
