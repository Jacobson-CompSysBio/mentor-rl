"""Verify S1.8 v9's matched views and preserve historical corpus bytes."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from runtime.world_model import s1_8_corpus as directions
from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_V6_OVERLAY_PATH, S1_V8_OVERLAY_PATH, S1_V9_OVERLAY_PATH, apply_s1_overlay,
)


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build historical versions and the training-only revision in temporary paths."""

    results = {}
    for version, overlay in ((6, S1_V6_OVERLAY_PATH), (8, S1_V8_OVERLAY_PATH),
                             (9, S1_V9_OVERLAY_PATH)):
        contract, payload = apply_s1_overlay(load_contract(), overlay)
        root = tmp_path_factory.mktemp(f"s1_8_v{version}_matched")
        captured = {}
        original = directions.select_scenarios

        def capture(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            captured["scenarios"] = original(*args, **kwargs)
            return captured["scenarios"]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(directions, "select_scenarios", capture)
            manifest = directions.build_corpus(
                root, contract=contract, prompt_banks=payload["prompt_banks"],
            )
        results[version] = dict(root=root, manifest=manifest, **captured)
    return results


def test_historical_bytes_and_v9_heldout_bytes_are_frozen(revision):
    """An absent v9 flag reproduces both trained versions and all held-out rows."""

    for version in (6, 8):
        published = Path(f"data/world_model_v4/corpora/s1_v{version}/s1_8")
        for split in directions.SPLITS:
            assert (revision[version]["root"] / f"{split}.jsonl").read_bytes() == (
                published / f"{split}.jsonl"
            ).read_bytes()
    for split in ("validation", "test"):
        assert (revision[9]["root"] / f"{split}.jsonl").read_bytes() == (
            revision[8]["root"] / f"{split}.jsonl"
        ).read_bytes()
        assert revision[9]["scenarios"][split] == revision[8]["scenarios"][split]
    assert (revision[9]["root"] / "train.jsonl").read_bytes() != (
        revision[8]["root"] / "train.jsonl"
    ).read_bytes()


def _scope_cells(cases, kind):
    """Group true/false views by the name and proposition polarity."""

    cells = defaultdict(set)
    for case in cases:
        if case.name_kind == kind and case.form in (2, 3, 4):
            cells[case.input_name, case.statement_polarity].add(case.framing)
    return cells


def test_matched_coverage_improves_within_the_existing_budget(revision):
    """Use every extra variant slot for a scope partner and complete alias cells."""

    old, new = (revision[version]["scenarios"]["train"] for version in (8, 9))
    for kind, expected in (("metric_variant", (94, 176)), ("shared_alias", (49, 64))):
        paired = tuple(sum(len(frames) == 2 for frames in _scope_cells(cases, kind).values())
                       for cases in (old, new))
        assert paired == expected
    variant_cells = _scope_cells(new, "metric_variant")
    by_name = defaultdict(set)
    by_name_frames = defaultdict(set)
    for (name, polarity), frames in variant_cells.items():
        by_name[name].add(polarity)
        by_name_frames[name].update(frames)
    assert len(by_name) == 156
    assert all(polarities == set(directions.POLARITIES) for polarities in by_name.values())
    assert all(frames == {"direct", "step"} for frames in by_name_frames.values())
    assert sum(case.name_kind == "metric_variant" and case.form > 1 for case in new) == 800
    assert revision[9]["manifest"]["prompt_structure_audit"]["splits"]["train"]["matched_views"] == {
        "canonical_form_polarity_scope_cells": 312,
        "shared_alias_polarity_scope_cells": 128,
        "variant_opposite_scope_pairs": 176,
        "variant_maximum_opposite_scope_pairs": 176,
        "ambiguous_polarity_scope_cells": 64,
        "unknown_polarity_scope_cells": 160,
        "pass": True,
    }
    for form, names in ((5, 8), (6, 20)):
        cells = {(case.input_name, case.statement_polarity, case.framing)
                 for case in new if case.form == form}
        assert len(cells) == names * 4 * 2


def test_semantic_allocations_are_preserved(revision):
    """Matching changes views, while preserving all convention and outcome quotas."""

    def signature(case):
        return (case.form, case.name_kind, case.metric, case.candidate_metrics,
                case.lower_is_stronger, case.meaning, case.rule,
                case.statement_polarity, case.statement_true)

    old, new = (revision[version]["scenarios"]["train"] for version in (8, 9))
    assert Counter(map(signature, old)) == Counter(map(signature, new))
    assert Counter((case.form, case.name_kind, case.statement_true, case.framing)
                   for case in old) == Counter(
        (case.form, case.name_kind, case.statement_true, case.framing) for case in new
    )
    for key in ("row_counts", "form_counts", "status_counts", "framing_counts",
                "name_kind_counts", "continuation_counts", "statement_truth_counts"):
        assert revision[9]["manifest"][key] == revision[8]["manifest"][key]


def test_reasoning_keeps_truth_schema_and_scope_distinct(revision):
    """False claims resolve; opposing or absent conventions retain nulls and defer."""

    rows = [json.loads(line) for line in (revision[9]["root"] / "train.jsonl").read_text().splitlines()]
    cases = revision[9]["scenarios"]["train"]
    result_fields = {"metric", "input", "lower_is_stronger", "meaning", "rule", "statement_true"}
    for row, case in zip(rows, cases, strict=True):
        final = row["messages"][-1]
        answer, thought = final["content"], final["thinking"]
        result = answer["result"]
        assert json.dumps(case.input_name, ensure_ascii=False) in thought
        assert set(result) == result_fields | ({"candidate_metrics"} if case.candidate_metrics else set())
        assert answer["provenance"] is None
        assert max(directions._sentence_word_counts(row["messages"][1]["content"])) <= 20
        if case.form in (5, 6):
            assert answer["status"] == ("ambiguous" if case.form == 5 else "not_found")
            assert answer["continuation_decision"] == "defer" and answer["blocked_reason"]
            assert all(result[field] is None for field in result_fields - {"input"})
            if case.form == 6:
                assert thought.index("truth remain undetermined") < thought.index("Copy the supplied name")
            continue
        assert answer["status"] == "resolved" and answer["blocked_reason"] is None
        assert answer["continuation_decision"] == ("stop" if case.framing == "direct" else "continue")
        if case.form == 1:
            assert result["statement_true"] is None
            continue
        lower_claim = case.statement_polarity.startswith("lower_")
        stronger_claim = case.statement_polarity.endswith("_stronger")
        assert result["statement_true"] == ((lower_claim == result["lower_is_stronger"]) == stronger_claim)
        assert "A false claim is still fully answered" in thought
        assert thought.index("The supplied name") < thought.index("The fixed convention sets")
        assert thought.index("The fixed convention sets") < thought.index("The claim pairs")
        assert thought.index("The claim pairs") < thought.index("The result contains")
        assert thought.index("The result contains") < thought.index("enclosing request")


def test_matched_requests_and_bank_size_independence(revision):
    """Matched views share neutral requests; changing wrapper count keeps content."""

    result = revision[9]
    options = deepcopy(result["kwargs"])
    baseline = result["scenarios"]
    assert directions.select_scenarios(*result["args"], **options) == baseline
    for factor in (0.5, 2):
        changed = deepcopy(options)
        for framing in ("direct", "step"):
            bank = tuple(changed["prompts"]["train"][framing])
            changed["prompts"]["train"][framing] = bank[:len(bank) // 2] if factor < 1 else bank * 2
        expanded = directions.select_scenarios(*result["args"], **changed)

        def signature(case):
            content = asdict(case)
            for key in ("scenario_id", "template_index"):
                content.pop(key)
            return json.dumps(content, sort_keys=True)

        for split in directions.SPLITS:
            assert Counter(map(signature, expanded[split])) == Counter(map(signature, baseline[split]))
    request_cells = defaultdict(lambda: defaultdict(set))
    for case in baseline["train"]:
        if case.name_kind == "canonical" and case.form in (2, 3, 4):
            request_cells[case.form, case.input_name, case.statement_polarity][case.framing].add(case.request_index)
    assert len(request_cells) == 156
    assert all(cell["direct"] & cell["step"] for cell in request_cells.values())


def test_audit_rejects_a_missing_alias_scope_partner(revision):
    """A repeated same-scope view cannot silently replace a required partner."""

    cases = list(revision[9]["scenarios"]["train"])
    counts = Counter((case.input_name, case.statement_polarity, case.framing)
                     for case in cases if case.name_kind == "shared_alias" and case.form > 1)
    index = next(index for index, case in enumerate(cases)
                 if case.name_kind == "shared_alias" and case.form > 1
                 and counts[case.input_name, case.statement_polarity, case.framing] == 1)
    case = cases[index]
    cases[index] = replace(case, framing="step" if case.framing == "direct" else "direct")
    with pytest.raises(directions.CorpusBuildError, match="opposite-scope"):
        directions._audit_training_matches(cases, revision[9]["kwargs"]["name_variants"]["train"])
