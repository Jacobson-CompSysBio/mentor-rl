"""records shared by the dashboard collector and renderer.

* the collector writes one summary.json and one samples.jsonl per stage
* the renderer reads the summary.json and samples.jsonl
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import pathlib
from typing import Any

SCHEMA_VERSION = "mentor-rl-dashboard-summary-v1"

@dataclass
class Family:
    """Counts for one question family, for example S1.3"""

    family_id: str
    question_family: str
    title: str
    disabled_reason: str | None # set only for a disabled/postponed family, such as S1.5

    rows: dict[str, int] # split -> rows
    forms: list[dict[str, Any]] # {"form", "name", "train_rows"}
    framing: dict[str, int] # train rows; rows carry no framining, so this comes from the component manifest
    outcome: dict[str, int] # status|continuation_decision -> train rows
    tool_calls_per_row: dict[str, int] # 0, 1, 2+ -> train rows
    harness_block_rows: int

    tokens: dict[str, int] | None # p50, p95, max; None until --tokens
    sample_record_ids: list[str]
    golden_ids: list[str]

@dataclass
class Tool:
    """Use of one tool from the stage's tool definitions."""

    name: str
    introduced_at: str
    arguments: list[str]
    calls: int
    rows_with_call: int
    result_status: dict[str, int]
    families: list[str]
    example_record_id: str | None

@dataclass
class FileRecord:
    """One file that a methods section or reviewer would cite."""

    role: str
    path: str
    bytes: int
    rows: int | None # jsonl files only
    sha256: str
    manifest_match: bool | None # None when the manifest does not list the file
    modified: str

@dataclass
class Check:
    """Result of one check; a failed check turns the stage red."""
    id: str
    ok: bool
    detail: str = ""


@dataclass
class Summary:
    """Everything the rendered shows for one stage."""

    stage: str
    title: str
    status: str
    built_at: str
    repo_commit: str
    dataset_id: str
    corpus_root: str # names the version, since S1 versions share one dataset_id
    provenance: dict[str, Any]
    rows: dict[str, int] 
    replay: dict[str, Any] | None
    share: dict[str, Any]
    book_mode: dict[str, int]
    system_prompt: str
    tool_definitions: list[dict[str, Any]]
    families: list[Family]
    tools: list[Tool]
    files: list[FileRecord]
    splits: dict[str, Any]
    gate: dict[str, Any] # {"report", "profile", "passed", "metrics": [{"name", "target", "value", "passed"}]}
    checks: list[Check]
    schema_version: str = SCHEMA_VERSION

@dataclass
class Sample:
    """One sampled or golden row, already shaped for display by the adapter."""

    record_id: str
    backend: dict[str, Any] # hidden metadata for the muted strip, never shown as a turn
    turns: list[dict[str, Any]] # chat messages without the system message, each with "masked"

def write_summary(
    summary: Summary,
    samples: list[Sample],
    directory: Path
) -> None:
    """Write summary.json and samples.jsonl for one stage."""
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n"
    (directory / "summary.json").write_text(text, encoding="utf-8")
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n") 
def read_summary(
    directory: Path
) -> tuple[Summary, list[Sample]]:
    """Read one stage's summary and samples, refusing an unkown schema version."""
    data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{directory}: summary schema {data['schema_version']}, expected {SCHEMA_VERSION}"
        )
    data["families"] = [Family(**item) for item in data["families"]]
    data["tools"] = [Tool(**item) for item in data["tools"]]
    data["files"] = [FileRecord(**item) for item in data["files"]]
    data["checks"] = [Check(**item) for item in data["checks"]]
    with (directory / "samples.jsonl").open(encoding="utf-8") as stream:
        samples = [Sample(**json.loads(line)) for line in stream]
    return Summary(**data), samples

    

