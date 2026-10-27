"""Check v8 outcome/wording independence and frozen S1.7/S1.8 boundaries."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from runtime.world_model import s1_7_corpus as layers, s1_8_corpus as directions
from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_V6_OVERLAY_PATH, S1_V8_OVERLAY_PATH, apply_s1_overlay, overlay_pin,
)


@pytest.fixture(scope="module", params=("s1_7", "s1_8"))
def revision(request, tmp_path_factory):
    """Build both versions through their complete semantic and leakage audits."""

    family = request.param
    module = layers if family == "s1_7" else directions
    results = {}
    for version, overlay in ((6, S1_V6_OVERLAY_PATH), (8, S1_V8_OVERLAY_PATH)):
        contract, payload = apply_s1_overlay(load_contract(), overlay)
        root = tmp_path_factory.mktemp(f"{family}_v{version}_decorrelation")
        captured = {}
        original = module.select_scenarios

        def capture(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            captured["scenarios"] = original(*args, **kwargs)
            return captured["scenarios"]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "select_scenarios", capture)
            options = dict(contract=contract, prompt_banks=payload["prompt_banks"],
                           overlay=overlay_pin(overlay, payload))
            if family == "s1_7":
                source = Path(contract["source_index"]["output_directory"]) / (
                    f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
                )
                manifest = module.build_corpus(source, root, **options)
            else:
                manifest = module.build_corpus(root, **options)
        results[version] = dict(root=root, manifest=manifest, **captured)
    return family, module, results


def test_legacy_and_heldout_bytes_are_frozen(revision):
    """Absent flags reproduce v6, and the enabled flag never changes heldout rows."""

    family, _, results = revision
    published = Path("data/world_model_v4/corpora/s1_v6") / family
    for split in ("train", "validation", "test"):
        assert (results[6]["root"] / f"{split}.jsonl").read_bytes() == (
            published / f"{split}.jsonl"
        ).read_bytes()
        if split != "train":
            assert (results[8]["root"] / f"{split}.jsonl").read_bytes() == (
                published / f"{split}.jsonl"
            ).read_bytes()
    assert results[8]["manifest"]["row_counts"] == results[6]["manifest"]["row_counts"]


def test_determinism_and_bank_size_do_not_change_content(revision):
    """A larger wrapper bank changes neither evidence nor request/polarity choices."""

    _, module, results = revision
    result = results[8]
    assert module.select_scenarios(*result["args"], **result["kwargs"]) == result["scenarios"]
    options = deepcopy(result["kwargs"])
    for framing in ("direct", "step"):
        options["prompts"]["train"][framing] = tuple(options["prompts"]["train"][framing]) * 2
    expanded = module.select_scenarios(*result["args"], **options)

    def content(case):
        record = asdict(case)
        for key in ("template_index", "scenario_id"):
            record.pop(key, None)
        return json.dumps(record, sort_keys=True)

    for split in ("train", "validation", "test"):
        assert Counter(map(content, expanded[split])) == Counter(map(content, result["scenarios"][split]))


def test_fixed_semantics_and_untargeted_forms(revision):
    """Changes retain evidence/answer multisets and every untargeted row."""

    family, module, results = revision
    old, new = (results[version]["scenarios"]["train"] for version in (6, 8))
    if family == "s1_7":
        def semantic(case):
            record = asdict(case)
            record.pop("template_index")
            record.pop("request_variant")
            return json.dumps(record, sort_keys=True)
        assert Counter(map(semantic, old)) == Counter(map(semantic, new))
        assert {case.scenario_id for case in old if case.form not in (2, 4)} == {
            case.scenario_id for case in new if case.form not in (2, 4)
        }
    else:
        def answer(case):
            result = module._answer(case)
            # Canonical frame repairs move stop/continue while retaining facts.
            result.pop("continuation_decision")
            return json.dumps(result, sort_keys=True)
        assert Counter(map(answer, old)) == Counter(map(answer, new))
        assert [case for case in old if case.form == 1] == [case for case in new if case.form == 1]


def test_list_wording_covers_both_outcomes_in_every_family_and_frame(revision):
    """Neutral wrappers and requests cannot predict match/no-match within a family."""

    family, _, results = revision
    if family != "s1_7":
        pytest.skip("list-filter component")
    cases = [case for case in results[8]["scenarios"]["train"] if case.form == 2]
    for attribute, width in (("template_index", 32), ("request_variant", 8)):
        cells = defaultdict(Counter)
        for case in cases:
            cells[case.family, case.framing, getattr(case, attribute)][case.list_outcome] += 1
        assert len(cells) == 6 * 2 * width
        for counts in cells.values():
            assert set(counts) == {"match", "no_match"}
            assert abs(counts["match"] - counts["no_match"]) <= 1
    assert Counter(case.list_outcome for case in cases) == {"match": 450, "no_match": 450}


def test_claim_request_covers_both_outcomes_for_every_applicable_claim(revision):
    """Every wording sees support and rejection; never-supported claims stay false."""

    family, _, results = revision
    if family != "s1_7":
        pytest.skip("claim-check component")
    cases = [case for case in results[8]["scenarios"]["train"] if case.form == 4]
    requests = defaultdict(Counter)
    strata = defaultdict(Counter)
    for case in cases:
        requests[case.request_variant][case.claim_supported] += 1
        strata[case.claim, case.claim_supported, case.framing][case.request_variant] += 1
        answer = layers._resolved_answer(case)["result"]
        assert answer["claim_supported"] == (answer["claim"] in answer["supported_claims"])
    assert set(requests) == set(range(24))
    assert all(set(counts) == {False, True} for counts in requests.values())
    assert sum(max(counts.values()) for counts in requests.values()) == 900
    for counts in strata.values():
        assert set(counts) == set(range(24))
        assert max(counts.values()) - min(counts.values()) <= 1


def test_deferred_names_and_requests_cover_polarities(revision):
    """Each deferred name and request covers all polarities in both framings."""

    family, _, results = revision
    if family != "s1_8":
        pytest.skip("direction component")
    for form, names in ((5, 8), (6, 20)):
        cases = [case for case in results[8]["scenarios"]["train"] if case.form == form]
        assert Counter(case.statement_polarity for case in cases) == {
            polarity: 100 for polarity in directions.POLARITIES
        }
        for attribute, width in (("input_name", names), ("request_index", 8)):
            cells = defaultdict(Counter)
            for case in cases:
                cells[getattr(case, attribute), case.framing][case.statement_polarity] += 1
                assert directions._answer(case)["continuation_decision"] == "defer"
                assert case.statement_true is None
            assert len(cells) == width * 2
            for counts in cells.values():
                assert set(counts) == set(directions.POLARITIES)
                assert max(counts.values()) - min(counts.values()) <= 1


def test_canonical_metric_form_polarity_frame_lattice_is_complete(revision):
    """Repair the six missing canonical cells without changing truth/polarity margins."""

    family, _, results = revision
    if family != "s1_8":
        pytest.skip("direction component")
    old, new = ([case for case in results[version]["scenarios"]["train"]
                 if case.form in (2, 3, 4) and case.name_kind == "canonical"]
                for version in (6, 8))
    def cells(cases):
        return Counter((case.metric, case.form, case.statement_polarity, case.framing) for case in cases)
    assert len(cells(old)) == 306
    assert len(cells(new)) == 312
    assert len({case.scenario_id for case in old} - {case.scenario_id for case in new}) <= 12
    for attributes in (("metric", "form", "statement_polarity"),
                       ("form", "statement_true", "framing")):
        def margins(cases):
            return Counter(tuple(getattr(case, key) for key in attributes) for case in cases)
        assert margins(old) == margins(new)
