"""Verify v9 layer parsing and claim coverage without changing published corpora."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from runtime.world_model import s1_6_corpus as parsing, s1_7_corpus as claims
from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import S1_V8_OVERLAY_PATH, apply_s1_overlay


@pytest.fixture(scope="module", params=("s1_6", "s1_7"))
def revision(request, tmp_path_factory):
    """Build each new section through its actual semantic and leakage audits."""

    family = request.param
    module = parsing if family == "s1_6" else claims
    contract, payload = apply_s1_overlay(load_contract(), S1_V8_OVERLAY_PATH)
    settings = contract["curriculum"]["families"][family.upper().replace("_", ".")]
    settings["train_targeted_v9"] = True
    root = tmp_path_factory.mktemp(f"{family}_v9")
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    captured = {}
    original = module.select_scenarios

    def capture(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        captured["scenarios"] = original(*args, **kwargs)
        return captured["scenarios"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "select_scenarios", capture)
        manifest = module.build_corpus(source, root, contract=contract, prompt_banks=payload["prompt_banks"])
    rows = {split: [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation", "test")}
    return family, module, contract, root, manifest, rows, captured


def test_heldout_and_untargeted_training_rows_are_frozen(revision):
    """V9 preserves every held-out byte and every protected training row."""

    family, _, _, root, manifest, rows, _ = revision
    baseline = Path("data/world_model_v4/corpora/s1_v8") / family
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()
    protected = {5, 7} if family == "s1_6" else {1, 3, 5, 6, 7, 9}
    old = {row["row_id"]: row for line in (baseline / "train.jsonl").read_text().splitlines()
           if (row := json.loads(line))["metadata"]["form"] in protected}
    new = {row["row_id"]: row for row in rows["train"] if row["metadata"]["form"] in protected}
    assert new == old
    assert manifest["train_targeted_v9"]["pass"]


def test_historical_v8_rendering_and_bank_size_independence(revision):
    """Disabling the flag reproduces v8; changing bank size changes only wrappers."""

    family, module, contract, _, _, _, captured = revision
    options = deepcopy(captured["kwargs"])
    options["settings"].pop("train_targeted_v9")
    historical = module.select_scenarios(*captured["args"], **options)
    baseline = Path("data/world_model_v4/corpora/s1_v8") / family
    settings = options["settings"]
    render = dict(contract=contract, prompts=options["prompts"], final_reasoning=settings["train_final_reasoning"])
    if family == "s1_6":
        render.update(bulk_pen_reasoning=settings["train_bulk_pen_reasoning"],
                      scpen_reasoning=settings["train_scpen_reasoning"],
                      humannet_reasoning=settings["train_humannet_reasoning"])
    for split, cases in historical.items():
        expected = [json.loads(line) for line in (baseline / f"{split}.jsonl").read_text().splitlines()]
        assert [module.build_row(case, **render)[0] for case in cases] == expected
    expanded_options = deepcopy(captured["kwargs"])
    for frame in ("direct", "step"):
        expanded_options["prompts"]["train"][frame] *= 2
    expanded = module.select_scenarios(*captured["args"], **expanded_options)

    def content(case):
        data = asdict(case)
        data.pop("template_index")
        return json.dumps(data, sort_keys=True)

    for split in ("train", "validation", "test"):
        assert Counter(map(content, expanded[split])) == Counter(map(content, captured["scenarios"][split]))


def test_concrete_parses_ordered_partitions_and_scope(revision):
    """Every targeted valid parse names all seven concrete values and an independent scope."""

    family, _, _, _, _, rows, captured = revision
    if family != "s1_6":
        pytest.skip("layer-parse component")
    coverage = set()
    for row, case in zip(rows["train"], captured["scenarios"]["train"], strict=True):
        if case.form not in {1, 2, 3, 4, 6}:
            continue
        thought = row["messages"][-1]["thinking"]
        answer = row["messages"][-1]["content"]
        if case.form == 6:
            assert case.malformed_input in thought
            assert "standalone or part of a workflow" in thought
            assert answer["continuation_decision"] == "defer"
            continue
        for fact in case.companions if case.form == 4 else (case.fact,):
            assert all(f"{key} = {json.dumps(value, ensure_ascii=False)}" in thought
                       for key, value in fact.layer_ref.items())
        assert "later workflow step, so continue" in thought if case.framing == "step" else "standalone task, so stop" in thought
        if case.form == 4:
            matching = list(case.expected_matching_layer_ids)
            other = [fact.layer_id for fact in case.companions if fact.layer_id not in matching]
            assert f"matching_layer_ids is {json.dumps(matching, ensure_ascii=False)}" in thought
            assert f"remaining input IDs are {json.dumps(other, ensure_ascii=False)}" in thought
            coverage.add((len(matching), case.framing))
    assert coverage == {(count, frame) for count in range(4) for frame in ("direct", "step")}


def test_claim_balance_paraphrases_and_negative_resolution(revision):
    """Equal canonical coverage includes every applicable verdict/scope/paraphrase combination."""

    family, _, _, _, manifest, rows, captured = revision
    if family != "s1_7":
        pytest.skip("claim component")
    cases = [case for case in captured["scenarios"]["train"] if case.form == 4]
    assert Counter(case.claim for case in cases) == {
        claim: 188 if index < 4 else 187 for index, claim in enumerate(claims.CLAIM_VOCABULARY)
    }
    assert Counter(case.claim_supported for case in cases) == {True: 376, False: 1124}
    assert set(manifest["train_targeted_v9"]["paraphrase_verdict_scope_coverage"].values()) == {32}
    for row, case in zip(rows["train"], captured["scenarios"]["train"], strict=True):
        if case.form not in {2, 4, 8}:
            continue
        answer = row["messages"][-1]["content"]
        thought = row["messages"][-1]["thinking"]
        assert answer["status"] == "resolved"
        assert answer["continuation_decision"] == ("continue" if case.framing == "step" else "stop")
        if case.form == 4:
            assert answer["result"]["claim_supported"] == (case.claim in answer["result"]["supported_claims"])
            assert thought.index(f"canonical claim {case.claim}") < thought.index("supported_claims is")
            if not case.claim_supported:
                assert "false support verdict still answers" in thought


def test_native_other_names_cover_both_scopes_and_wrappers(revision):
    """All three existing unmapped training networks receive independent neutral wording."""

    family, _, _, _, _, rows, captured = revision
    if family != "s1_7":
        pytest.skip("claim component")
    cells = defaultdict(set)
    for row, case in zip(rows["train"], captured["scenarios"]["train"], strict=True):
        if case.form != 8:
            continue
        assert case.fact in captured["args"][0]["train"]["other"]
        cells[(case.fact.network, case.framing)].add(case.template_index)
        assert "string_ppi and biogrid_ppi map to ppi" in row["messages"][-1]["thinking"]
    assert len(cells) == 6
    assert all(indexes == set(range(32)) for indexes in cells.values())
