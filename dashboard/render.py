"""Render the dashboard site from the collected JSON files.

Writes an index of stage labels (with the hand-written executive summary
above them, once it exists) and one page per stage with versions. The stage
page lists every version with its gate results, and embeds each version's
gold traces once, with a small record per pair of versions;
templates/compare.js renders whichever pair the reader picks, older version
on the left and newer on the right, or the one version on disk. It reads the registry,
site_data/ and dashboard/notes/, never a corpus, so a layout change reruns
in seconds. Pages are self-contained HTML and open from disk.

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

from dashboard.gates import comparability, gate_info

TEMPLATES = Path(__file__).parent / "templates"
NOTES = Path(__file__).parent / "notes"


def markdown(path: Path) -> Markup | None:
    """Render a hand-written note, or return None when it does not exist yet."""
    return Markup(MarkdownIt("commonmark").render(path.read_text())) if path.is_file() else None


def stage_data(stage: dict, data: Path) -> dict | None:
    """Gather one stage's versions, pairs, and notes into the object its page embeds.

    available lists the versions whose corpus is on disk. With two or more,
    the page compares pairs; with one, it shows that version's gold traces.
    Gate results are read here, from the registry's gate_report paths. Each
    version is checked against the nearest earlier version with a gate run.
    """
    label = stage["stage"]
    versions, previous = [], None
    for entry in stage.get("versions", []):
        path = data / label / "versions" / f"{entry['version']}.json"
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
    return {"stage": label, "current": current, "versions": versions, "available": available,
            "pairs": pairs, "notes": notes, "default": default}


def render(registry: Path, data: Path, out: Path) -> None:
    """Write site/index.html and one page per stage that has versions to compare."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True,
                      trim_blocks=True, lstrip_blocks=True)
    script = Markup((TEMPLATES / "compare.js").read_text())
    stages = yaml.safe_load(registry.read_text())["stages"]
    links = {}
    for stage in stages:
        page_data = stage_data(stage, data)
        if page_data is None:
            continue
        # "<" is escaped so the embedded JSON cannot close its <script> element.
        embedded = Markup(json.dumps(page_data, ensure_ascii=False).replace("<", "\\u003c"))
        target = out / "stages" / f"{stage['stage']}.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(env.get_template("stage.html.j2").render(
            stage=stage, versions=page_data["available"], compare=bool(page_data["pairs"]),
            data=embedded, script=script), encoding="utf-8")
        links[stage["stage"]] = f"stages/{stage['stage']}.html"
        print(f"{target}: {target.stat().st_size / 1e6:.1f} MB")

    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.html"
    index.write_text(env.get_template("index.html.j2").render(
        stages=stages, links=links, summary=markdown(NOTES / "executive-summary.md")),
        encoding="utf-8")
    print(index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=Path("config/dashboard/stages.yaml"))
    parser.add_argument("--data", type=Path, default=Path("site_data"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    render(args.registry, args.data, args.out)


if __name__ == "__main__":
    main()
