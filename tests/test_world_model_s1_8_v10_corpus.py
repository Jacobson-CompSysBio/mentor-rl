"""Keep v9 metric examples fixed while grounding form-4 reasoning in visible scope."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from runtime.world_model import s1_8_corpus as directions
from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_V9_OVERLAY_PATH, S1_V10_OVERLAY_PATH, apply_s1_overlay,
)


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build both versions and retain the exact scenarios and wrapper bank for checks."""

    results = {}
    for version, overlay in ((9, S1_V9_OVERLAY_PATH), (10, S1_V10_OVERLAY_PATH)):
        contract, payload = apply_s1_overlay(load_contract(), overlay)
        root = tmp_path_factory.mktemp(f"s1_8_v{version}_scope")
        captured = {}
        original = directions.select_scenarios

        def capture(*args, **kwargs):
            captured["prompts"] = kwargs["prompts"]
            captured["scenarios"] = original(*args, **kwargs)
            return captured["scenarios"]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(directions, "select_scenarios", capture)
            manifest = directions.build_corpus(
                root, contract=contract, prompt_banks=payload["prompt_banks"],
            )
        results[version] = dict(root=root, contract=contract, manifest=manifest, **captured)
    return results


def test_only_666_final_training_thoughts_change(revision):
    """Preserve historical bytes, every prompt and target, and v9's matched allocation."""

    old, new = revision[9], revision[10]
    published = Path("data/world_model_v4/corpora/s1_v9/s1_8")
    for split in directions.SPLITS:
        assert (old["root"] / f"{split}.jsonl").read_bytes() == (
            published / f"{split}.jsonl"
        ).read_bytes()
        assert new["scenarios"][split] == old["scenarios"][split]
        if split != "train":
            assert (new["root"] / f"{split}.jsonl").read_bytes() == (
                old["root"] / f"{split}.jsonl"
            ).read_bytes()
    changed = Counter()
    for old_line, new_line in zip(
        (old["root"] / "train.jsonl").read_bytes().splitlines(),
        (new["root"] / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        before, after = json.loads(old_line), json.loads(new_line)
        if before["metadata"]["form"] != 4:
            assert new_line == old_line
            continue
        assert after["messages"][-1]["thinking"] != before["messages"][-1]["thinking"]
        before["messages"][-1]["thinking"] = after["messages"][-1]["thinking"]
        assert after == before
        changed[after["messages"][-1]["content"]["continuation_decision"]] += 1
    assert changed == {"stop": 333, "continue": 333}
    for field in ("row_counts", "form_counts", "scenario_assignment_sha256",
                  "name_sources_sha256", "source_coverage", "evidence"):
        assert new["manifest"][field] == old["manifest"][field]


def test_scope_reasoning_quotes_the_visible_outer_request(revision):
    """Use the actual wrapper to choose scope independently of the enclosed question's wording."""

    result = revision[10]
    rows = [json.loads(line) for line in (result["root"] / "train.jsonl").read_text().splitlines()]
    seen = Counter()
    for row, case in zip(rows, result["scenarios"]["train"], strict=True):
        if case.form != 4:
            continue
        wrapper = result["prompts"]["train"][case.framing][case.template_index]
        scope = wrapper.replace("{request}", "").strip()
        user = row["messages"][1]["content"]
        final = row["messages"][-1]
        thought, answer = final["thinking"], final["content"]
        assert scope in user
        assert directions._request(case) in user
        assert f'The enclosing request states: {json.dumps(scope, ensure_ascii=False)}' in thought
        assert "Workflow language in the enclosed metric question describes its use" in thought
        assert "it does not override the enclosing request" in thought
        assert "Whether statement_true is true or false does not change this task scope" in thought
        decision = "stop" if case.framing == "direct" else "continue"
        assert thought.endswith(f"so continuation_decision is {decision}.")
        assert answer["continuation_decision"] == decision
        assert answer["status"] == "resolved" and answer["blocked_reason"] is None
        assert f'makes statement_true {json.dumps(case.statement_true)}.' in thought
        seen[case.framing, case.statement_true] += 1
    assert set(seen) == {(frame, truth) for frame in ("direct", "step") for truth in (False, True)}


@pytest.mark.parametrize("mutation", ["scope", "thought", "decision", "status", "result", "call"])
def test_scope_binding_keeps_exact_thinking_and_answer_checks(revision, mutation):
    """A scope quotation cannot weaken evidence, schema, or forbidden-action checks."""

    result = revision[10]
    case = next(case for case in result["scenarios"]["train"]
                if case.form == 4 and case.framing == "direct")
    templates = result["contract"]["curriculum"]["families"]["S1.8"]["train_final_reasoning"]
    row, _ = directions.build_row(
        case, contract=result["contract"], final_reasoning=templates,
        prompts=result["prompts"],
    )
    assert directions.filter_row_thinking(
        row, case, final_reasoning=templates, prompts=result["prompts"],
    ) == 1
    final = row["messages"][-1]
    prompts = result["prompts"]
    if mutation == "scope":
        prompts = deepcopy(prompts)
        prompts["train"]["direct"] = prompts["train"]["step"]
    elif mutation == "thought":
        final["thinking"] = final["thinking"].replace("continuation_decision is stop", "continuation_decision is continue")
    elif mutation == "decision":
        final["content"]["continuation_decision"] = "continue"
    elif mutation == "status":
        final["content"]["status"] = "found"
    elif mutation == "result":
        del final["content"]["result"]["statement_true"]
    else:
        final["tool_call"] = {"name": "query_mygene", "arguments": {}}
    with pytest.raises(directions.CorpusBuildError):
        directions.filter_row_thinking(
            row, case, final_reasoning=templates, prompts=prompts,
        )
