"""Collect one S2 gate run's held-out failures for the stage page's Failures tab.

The S2 scorer marks every test row on six structured fields (status, result,
decision, provenance, blocker presence, and the exact ordered tool calls). This
reads its per-row report together with the run's predictions and the
evaluator's questions and answer keys, and sorts each failed row into one
category: an invalid trajectory (and why), an answer given before the expected
tool calls, or the final-answer fields that differ. For a final answer it also
lists where the predicted result and provenance differ from the key, by JSON
path, so the page can show which part of an answer goes wrong across a case.

The free-text summary and blocked-reason wording are never scored; the page
shows them only as the model wrote them.

Writes:

  site_data/<stage>/failures/<version>.json        counts for every row, and up
      to EXAMPLES failed rows per case in full, which the page embeds
  site_data/<stage>/failures/<version>.rows.jsonl  every failed row in full, for
      analysis outside the page

The page embeds test-panel answers. Do not paste them into a tool that drafts
training wording: held-out rows must not leak into training data.

    python -m dashboard.failures --stage S2
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import yaml

# Report field -> the short name the page uses.
FIELDS = {"status_accuracy": "status", "result_accuracy": "result",
          "continuation_decision_accuracy": "decision", "provenance_accuracy": "provenance",
          "blocked_reason_presence_accuracy": "blocker", "exact_tool_calls": "tool calls"}
EXAMPLES = 3  # full failed rows per case on the page
TEXT_LIMIT = 1500  # characters of one shown text before it is cut
DIFF_LIMIT = 12  # differing JSON paths kept per row
PATHS_SHOWN = 6  # most common differing paths listed per case


def sha256(path: Path) -> str:
    """The SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, keep: set[str] | None = None) -> dict[str, dict]:
    """Rows of a JSONL file by row_id, optionally only the given ids."""
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if keep is None or row["row_id"] in keep:
                rows[row["row_id"]] = row
    return rows


def cut(value, limit: int = TEXT_LIMIT):
    """A long text cut to limit characters; a large object becomes cut JSON text."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return value
    return f"{text[:limit]}… [{len(text) - limit:,} more characters]"


def error_label(text: str) -> str:
    """A parse or trajectory error without its character position, so equal errors group."""
    text = re.sub(r": line \d+ column \d+ \(char \d+\)", "", text)
    return text.replace("final_answer:", "").replace("_", " ").strip()


def first_mismatch(expected: list[dict], generated: list[dict]) -> str:
    """Name the first call where a stopped trajectory left the expected calls."""
    for index, call in enumerate(generated):
        if index >= len(expected):
            return "called a tool when none was expected" if not expected else "extra call after the expected calls"
        if call["name"] != expected[index]["name"]:
            return f"wrong tool at call {index + 1}"
        if call["arguments"] != expected[index]["arguments"]:
            return f"wrong arguments at call {index + 1}"
    return "no call matched"


def category(record: dict, expected_calls: list[dict], generated_calls: list[dict]) -> str:
    """One label per failed row: why the trajectory is invalid, or which fields differ.

    A wrong call stops the harness, because it serves a tool result only for the
    expected call. So a valid trajectory with inexact calls answered early.
    """
    if not record["valid_trajectory"]:
        if record["trajectory_error"] == "unexpected_tool_call_or_arguments":
            return f"invalid: {first_mismatch(expected_calls, generated_calls)}"
        return f"invalid: {error_label(record['trajectory_error'] or record['parse_error'] or record['stop_reason'])}"
    if not record["exact_tool_calls"]:
        return "answered before the expected tool calls"
    return "final answer: " + " + ".join(name for field, name in FIELDS.items()
                                        if field != "exact_tool_calls" and not record[field])


def json_diff(expected, got, path: str = "") -> list[dict]:
    """Where got differs from expected, as [{path, kind, expected, got}].

    kind is missing, extra, differs, order (same items in another order), or
    length (lists of different lengths, with the items only one side has).
    """
    if isinstance(expected, dict) and isinstance(got, dict):
        found = []
        for key in expected:
            if key not in got:
                found.append({"path": f"{path}.{key}", "kind": "missing", "expected": cut(expected[key], 200), "got": None})
            else:
                found += json_diff(expected[key], got[key], f"{path}.{key}")
        found += [{"path": f"{path}.{key}", "kind": "extra", "expected": None, "got": cut(got[key], 200)}
                  for key in got if key not in expected]
        return found
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) == len(got):
            if sorted(map(json.dumps, expected)) == sorted(map(json.dumps, got)) and expected != got:
                return [{"path": f"{path}[]", "kind": "order", "expected": cut(expected, 200), "got": cut(got, 200)}]
            return [item for index, (a, b) in enumerate(zip(expected, got)) for item in json_diff(a, b, f"{path}[{index}]")]
        a, b = Counter(map(json.dumps, expected)), Counter(map(json.dumps, got))
        return [{"path": f"{path}[]", "kind": "length", "expected": len(expected), "got": len(got),
                 "only_expected": [cut(json.loads(x), 200) for x in list((a - b).elements())[:3]],
                 "only_got": [cut(json.loads(x), 200) for x in list((b - a).elements())[:3]]}]
    if expected != got:
        return [{"path": path, "kind": "differs", "expected": cut(expected, 200), "got": cut(got, 200)}]
    return []


def general_path(path: str) -> str:
    """A path with list positions removed, so the same field groups across rows: .result.neighbors[].gene_id."""
    return re.sub(r"\[\d+\]", "[]", path)


def failed_row(record: dict, question: dict, answer: dict, prediction: dict) -> dict:
    """One failed row as the page shows it: the trace the model produced, the key, and the differences."""
    target = answer["target"]
    generated = prediction["generated_messages"]
    calls = [m["tool_call"] for m in generated if m.get("role") == "assistant" and "tool_call" in m]
    final = next((m["content"] for m in reversed(generated)
                  if m.get("role") == "assistant" and isinstance(m.get("content"), dict)), None)
    diffs = []
    if final is not None and record["valid_trajectory"]:
        for field in ("result", "provenance"):
            diffs += json_diff(target[field], final.get(field), f".{field}")
    harness, user = question["messages"][1], question["messages"][2]
    shown = [{"role": "harness", "content": cut(harness["content"])}, {"role": "user", "content": user["content"]}]
    for message in generated:
        message = dict(message)
        if message.get("thinking"):
            message["thinking"] = cut(message["thinking"])
        if message.get("role") == "tool":
            message["content"] = cut(message["content"])
        shown.append(message)
    return {
        "row_id": record["row_id"], "case": record["catalog_case_id"], "family": record["question_family"],
        "category": category(record, target["expected_tool_calls"], calls),
        "failed": [name for field, name in FIELDS.items() if not record[field]],
        "error": record["trajectory_error"] or record["parse_error"],
        "messages": shown,
        "expected": {"tool_calls": target["expected_tool_calls"], "status": target["status"],
                     "result": target["result"], "continuation_decision": target["continuation_decision"],
                     "provenance": target["provenance"], "blocked_reason_required": target["blocked_reason_required"]},
        "diffs": diffs[:DIFF_LIMIT], "diff_count": len(diffs),
    }


def examples(rows: list[dict], limit: int = EXAMPLES) -> list[dict]:
    """Up to limit rows, one per category in turn, so each failure mode of a case has an example."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    picked, queues = [], sorted(by_category.values(), key=len, reverse=True)
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def collect(run: Path, root: Path = Path(".")) -> tuple[dict, list[dict]]:
    """Read one S2 run and return the page's failure summary and every failed row.

    run is the gate run directory (holding report/ and generation/). Its
    copied test config names the evaluator files, which must still match the
    hashes the run pinned.
    """
    report = json.loads((run / "report/report.json").read_text())
    metrics = json.loads((run / "report/metrics.json").read_text())
    panel = json.loads((run / "run_contract/test_config.json").read_text())["test_panel"]
    evaluator = root / panel["evaluator_root"]
    files = {"questions": evaluator / panel["questions_file"], "answers": evaluator / panel["answer_key_file"]}
    for name, key in (("questions", "questions_sha256"), ("answers", "answer_key_sha256")):
        if sha256(files[name]) != panel[key]:
            raise SystemExit(f"{files[name]} no longer matches the hash the run pinned")
    failed_ids = {r["row_id"] for r in report["records"] if not r["structured_accuracy"]}
    questions = read_jsonl(files["questions"], failed_ids)
    answers = read_jsonl(files["answers"], failed_ids)
    predictions = read_jsonl(run / "generation/predictions.jsonl", failed_ids)

    rows, cases = [], {}
    for name, counts in metrics["cases"].items():
        cases[name] = {"rows": counts["row_count"], "correct": counts["structured_accuracy_count"],
                       "fields": {short: counts[f"{field}_count"] for field, short in FIELDS.items()},
                       "categories": Counter(), "paths": Counter()}
    for record in report["records"]:
        if record["row_id"] not in failed_ids:
            continue
        row = failed_row(record, questions[record["row_id"]], answers[record["row_id"]],
                         predictions[record["row_id"]])
        rows.append(row)
        case = cases[row["case"]]
        case["categories"][row["category"]] += 1
        # Count each differing field once per row, however many list items differ.
        case["paths"].update({f"{general_path(d['path'])} ({d['kind']})" for d in row["diffs"]})
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    for name, case in cases.items():
        case["categories"] = dict(case["categories"].most_common())
        case["paths"] = [[path, n] for path, n in case["paths"].most_common(PATHS_SHOWN)]
        case["examples"] = examples(by_case[name])
    families = defaultdict(Counter)
    for row in rows:
        families[row["family"]][row["category"]] += 1
    summary = {
        "run": str(run), "panel": metrics["test_panel_id"], "profile": metrics["gate"]["profile"],
        "rows": metrics["overall"]["row_count"], "correct": metrics["overall"]["structured_accuracy_count"],
        "fields": {short: metrics["overall"][f"{field}_count"] for field, short in FIELDS.items()},
        "family_fields": {family: {"rows": counts["row_count"], "correct": counts["structured_accuracy_count"],
                                   **{short: counts[f"{field}_count"] for field, short in FIELDS.items()}}
                          for family, counts in metrics["families"].items()},
        "categories": dict(Counter(row["category"] for row in rows).most_common()),
        "family_categories": {family: dict(counts.most_common()) for family, counts in families.items()},
        "cases": cases, "examples_per_case": EXAMPLES,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="S2")
    parser.add_argument("--registry", type=Path, default=Path("config/dashboard/stages.yaml"))
    parser.add_argument("--data", type=Path, default=Path("site_data"))
    args = parser.parse_args()
    stage = next(s for s in yaml.safe_load(args.registry.read_text())["stages"] if s["stage"] == args.stage)
    for entry in stage["versions"]:
        if not entry.get("gate_report"):
            continue
        summary, rows = collect(Path(entry["gate_report"]).parent.parent)
        out = args.data / args.stage / "failures"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{entry['version']}.json").write_text(json.dumps(summary, ensure_ascii=False))
        with (out / f"{entry['version']}.rows.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        size = (out / f"{entry['version']}.json").stat().st_size / 1e6
        print(f"{args.stage} {entry['version']}: {len(rows):,} failed rows; page data {size:.1f} MB")


if __name__ == "__main__":
    main()
