"""Check deterministic S2 evaluator construction and answer separation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat

import pytest

from runtime.world_model.s2_production_adapter import evaluation_target, tools_sha256
from scripts import build_world_model_s2_eval_bundle as builder


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data/world_model_v4/corpora/s2_production_v1_8k"


@pytest.fixture(scope="module")
def production_manifest():
    """Load the immutable production identities without scanning its large panel."""

    path = CORPUS_ROOT / "manifest.json"
    if not path.is_file():
        pytest.skip("the frozen S2 production corpus is unavailable")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def production_row():
    """Use one actual held-out row to exercise the production wire."""

    path = CORPUS_ROOT / "test.jsonl"
    if not path.is_file():
        pytest.skip("the frozen S2 test panel is unavailable")
    with path.open() as source:
        return json.loads(next(source))


@pytest.fixture
def synthetic_row():
    """Return one complete S2 wire independent of large local data artifacts."""

    system = "Use the visible evidence and return the structured S2 result."
    return {
        "row_id": "synthetic-source",
        "messages": [
            {"role": "system", "content": system},
            {"role": "harness", "content": '{"observations":[]}'},
            {"role": "user", "content": "Return the supported result."},
            {
                "role": "assistant",
                "thinking": "The visible evidence is sufficient.",
                "content": {
                    "status": "resolved",
                    "provenance": {"registry": [], "graph": []},
                    "result": {"supported": True},
                    "continuation_decision": "stop",
                    "blocked_reason": None,
                    "summary": "The requested result is supported.",
                },
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect_visible_evidence",
                    "description": "Return the visible evidence without guessing.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "metadata": {
            "schema_version": "mentor-rl-world-model-s2-corpus-v1",
            "tool_contract_version": "synthetic-contract",
            "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "question_family": "S2.13",
            "split": "test",
            "split_unit": "synthetic-split-unit",
            "supervision": "current_turn_assistant_only",
            "catalog_case_id": "S2.13.1",
        },
    }


def test_project_row_exposes_only_the_prompt_and_safe_metadata(
    production_manifest,
    production_row,
):
    """Keep gold turns, targets, and outcome metadata out of public questions."""

    original = deepcopy(production_row)
    question, answer = builder.project_row(
        production_row,
        production_manifest["contract_routes"],
    )

    assert production_row == original
    assert [message["role"] for message in question["messages"]] == [
        "system",
        "harness",
        "user",
    ]
    assert set(question["metadata"]) == set(builder.PUBLIC_METADATA_KEYS)
    assert "catalog_case_id" not in question["metadata"]
    assert not any(
        message["role"] in {"assistant", "tool"}
        for message in question["messages"]
    )

    assert answer["target"] == evaluation_target(production_row)
    assert answer["scoring_metadata"] == {
        "question_family": production_row["metadata"]["question_family"],
        "catalog_case_id": production_row["metadata"]["catalog_case_id"],
    }
    assert tuple(answer["target"]) == builder.TARGET_KEYS
    for free_text in ("thinking", "summary", "blocked_reason"):
        assert free_text not in answer["target"]


def _miniature_corpus(root: Path, base_row: dict) -> tuple[Path, str]:
    """Write a 13-route corpus that obeys the production manifest shape."""

    corpus = root / "corpus"
    corpus.mkdir()
    rows = []
    routes = {}
    case_entries = []
    prompt_sha256 = hashlib.sha256(
        base_row["messages"][0]["content"].encode()
    ).hexdigest()
    tools_identity = tools_sha256(base_row["tools"])
    for index, family in enumerate(builder.EXPECTED_FAMILIES, start=1):
        row = deepcopy(base_row)
        contract_version = f"miniature-contract-{index}"
        row["row_id"] = f"miniature-{index}"
        row["metadata"]["question_family"] = family
        row["metadata"]["catalog_case_id"] = f"{family}.1"
        row["metadata"]["tool_contract_version"] = contract_version
        row["metadata"]["split"] = "test"
        row["metadata"]["system_prompt_sha256"] = prompt_sha256
        rows.append(row)
        routes[family] = {
            "contract_version": contract_version,
            "system_prompt_sha256": prompt_sha256,
            "tools_sha256": tools_identity,
        }
        case_entries.append(
            {"case_id": f"{family}.1", "counts": {"test": 1}}
        )

    test_bytes = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for row in rows
    )
    (corpus / "test.jsonl").write_bytes(test_bytes)
    test_sha256 = hashlib.sha256(test_bytes).hexdigest()
    validation_sha256 = "1" * 64
    manifest = {
        "schema_version": builder.CORPUS_SCHEMA_VERSION,
        "production_id": builder.PRODUCTION_ID,
        "data_ready": True,
        "case_coverage_ready": True,
        "production_export_allowed": True,
        "contract_routes": routes,
        "files": {
            "train.jsonl": {},
            "validation.jsonl": {
                "sha256": validation_sha256,
                "row_count": len(rows),
                "bytes": 1,
            },
            "test.jsonl": {
                "sha256": test_sha256,
                "row_count": len(rows),
                "bytes": len(test_bytes),
            },
        },
        "panel_id": builder.stable_sha256(
            {"validation": validation_sha256, "test": test_sha256}
        ),
        "budget": {"test": len(rows), "s2": {"test": len(rows)}},
        "retained_cases": len(rows),
        "per_case_quotas": {"test": 1},
        "case_entries": case_entries,
    }
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, builder.sha256_file(manifest_path)


def test_build_is_deterministic_private_and_immutable(tmp_path, synthetic_row):
    """Publish stable files, accept an exact rerun, and protect the answer key."""

    manifest_path, manifest_sha256 = _miniature_corpus(tmp_path, synthetic_row)
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = builder.build_eval_bundle(
        corpus_manifest_path=manifest_path,
        corpus_manifest_sha256=manifest_sha256,
        output_dir=first,
        repo_root=tmp_path,
    )
    rerun = builder.build_eval_bundle(
        corpus_manifest_path=manifest_path,
        corpus_manifest_sha256=manifest_sha256,
        output_dir=first,
        repo_root=tmp_path,
    )
    builder.build_eval_bundle(
        corpus_manifest_path=manifest_path,
        corpus_manifest_sha256=manifest_sha256,
        output_dir=second,
        repo_root=tmp_path,
    )

    assert rerun == result
    assert result["test"]["row_count"] == 13
    assert set(result["test"]["family_counts"]) == set(builder.EXPECTED_FAMILIES)
    assert set(result["test"]["case_counts"]) == {
        f"{family}.1" for family in builder.EXPECTED_FAMILIES
    }
    assert result["scoring"]["target_fields"] == list(builder.SCORED_TARGET_KEYS)
    assert "expected_tool_responses" not in result["scoring"]["target_fields"]
    assert result["answer_separation"]["generation_reads_private_answer_keys"] is False
    assert stat.S_IMODE((first / "test_answer_key.jsonl").stat().st_mode) == 0o600
    for filename in ("manifest.json", "test_questions.jsonl", "test_answer_key.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    with (first / "test_questions.jsonl").open() as questions:
        public = json.loads(next(questions))
    assert [message["role"] for message in public["messages"]] == [
        "system",
        "harness",
        "user",
    ]
    assert set(public["metadata"]) == set(builder.PUBLIC_METADATA_KEYS)
    route = result["contract_routes"][public["metadata"]["question_family"]]
    assert tools_sha256(public["tools"]) == route["tools_sha256"]

    (first / "test_questions.jsonl").write_text("changed\n")
    with pytest.raises(builder.S2EvalBuildError, match="immutable"):
        builder.build_eval_bundle(
            corpus_manifest_path=manifest_path,
            corpus_manifest_sha256=manifest_sha256,
            output_dir=first,
            repo_root=tmp_path,
        )


def test_source_drift_fails_atomically(tmp_path, synthetic_row):
    """Reject changed source bytes without leaving a partial evaluator directory."""

    manifest_path, manifest_sha256 = _miniature_corpus(tmp_path, synthetic_row)
    with (manifest_path.parent / "test.jsonl").open("ab") as destination:
        destination.write(b"\n")
    output = tmp_path / "failed-output"
    with pytest.raises(builder.S2EvalBuildError, match="blank record"):
        builder.build_eval_bundle(
            corpus_manifest_path=manifest_path,
            corpus_manifest_sha256=manifest_sha256,
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not output.exists()


def test_manifest_and_route_identity_fail_closed(
    tmp_path,
    synthetic_row,
):
    """Reject an unpinned manifest and a changed prompt route before publication."""

    manifest_path, manifest_sha256 = _miniature_corpus(tmp_path, synthetic_row)
    with pytest.raises(builder.S2EvalBuildError, match="manifest identity"):
        builder.build_eval_bundle(
            corpus_manifest_path=manifest_path,
            corpus_manifest_sha256="0" * 64,
            output_dir=tmp_path / "wrong-manifest",
            repo_root=tmp_path,
        )

    manifest = json.loads(manifest_path.read_text())
    manifest["contract_routes"]["S2.13"]["system_prompt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    changed_sha256 = builder.sha256_file(manifest_path)
    with pytest.raises(builder.S2EvalBuildError, match="invalid S2 test record"):
        builder.build_eval_bundle(
            corpus_manifest_path=manifest_path,
            corpus_manifest_sha256=changed_sha256,
            output_dir=tmp_path / "wrong-route",
            repo_root=tmp_path,
        )


def test_production_manifest_pins_the_complete_panel(production_manifest):
    """Lock the published 25,216-row, 394-case, 13-family panel identity."""

    routes, cases = builder._validate_manifest(production_manifest)
    assert set(routes) == set(builder.EXPECTED_FAMILIES)
    assert len(cases) == 394
    assert set(cases.values()) == {64}
    assert sum(cases.values()) == 25216
    assert builder.DEFAULT_QUESTIONS_SHA256 == (
        "0b60dc7ea96031baa5b87a000a70e9de5eb20a10b5febb39ac078ac00c9471c9"
    )
    assert builder.DEFAULT_ANSWER_KEY_SHA256 == (
        "e27144117ec11e2e65710ccc37dda3fefb5dc92f2c11a2cc13f9b6b05f7ed811"
    )
    assert production_manifest["panel_id"] == (
        "335950a9b202d0ef9b76b2b23019e4e43638c3e8419dd1fb3feb270dd1ca8d0a"
    )
