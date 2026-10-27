"""Verify training-only identity decorrelation against full frozen corpora."""

from collections import Counter, defaultdict
from copy import deepcopy
import importlib
import json
from pathlib import Path
import re

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_V6_OVERLAY_PATH, S1_V8_OVERLAY_PATH, apply_s1_overlay, overlay_pin,
)


@pytest.fixture(scope="module", params=("s1_1", "s1_2"))
def revision(request, tmp_path_factory):
    """Build both full versions and retain selection inputs for independence checks."""

    family = request.param
    builder = importlib.import_module(f"runtime.world_model.{family}_corpus")
    base = Path("data/world_model_v4/corpora/s1_v6") / family
    roots = {}
    captured = {}
    original = builder.select_scenarios

    def capture(*args, **kwargs):
        """Capture the revised selection, excluding the legacy evidence replay."""

        selected = original(*args, **kwargs)
        if kwargs.get("train_decorrelate"):
            captured.update(args=args, kwargs=kwargs, selected=selected)
        return selected

    for version, overlay in (("v6", S1_V6_OVERLAY_PATH), ("v8", S1_V8_OVERLAY_PATH)):
        contract, payload = apply_s1_overlay(load_contract(), overlay)
        source = Path(contract["source_index"]["output_directory"]) / (
            f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
        )
        output = tmp_path_factory.mktemp(f"{family}_{version}")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(builder, "select_scenarios", capture)
            manifest = builder.build_corpus(
                source, output, contract=contract, validate=version == "v8",
                overlay=overlay_pin(overlay, payload), prompt_banks=payload["prompt_banks"],
            )
        roots[version] = output
        if version == "v8":
            revised = (contract, payload, source, manifest)
    return family, builder, base, roots, captured, revised


def _rows(path):
    """Read rendered rows for checks independent of scenario metadata."""

    return [json.loads(line) for line in path.read_text().splitlines()]


def _user(row):
    """Return the single user request, excluding registry evidence."""

    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def _wrapper(row, banks):
    """Recover the rendered wrapper while preventing nested prefix matches."""

    user = _user(row)
    request_pattern = (
        r"(?:the human Ensembl gene ID for (?:the mouse gene )?\S+"
        r"|the human gene symbol corresponding to \S+"
        r"|whether the human gene \S+ is a node in \S+)"
    )
    matches = []
    for framing, templates in banks.items():
        for template in templates:
            before, after = template.split("{request}")
            if user.startswith(before) and user.endswith(after):
                request = user[len(before):len(user) - len(after) if after else None]
                if re.fullmatch(request_pattern, request):
                    matches.append((framing, template))
    assert len(matches) == 1, user
    return matches[0]


def test_full_build_preserves_frozen_versions_and_heldout(revision):
    """The flag changes train only; an absent flag reproduces every v6 row byte."""

    family, _, base, roots, _, revised = revision
    expected = 10000 if family == "s1_1" else 11000
    assert revised[3]["row_counts"]["train"] == expected
    for split in ("train", "validation", "test"):
        frozen = (base / f"{split}.jsonl").read_bytes()
        assert (roots["v6"] / f"{split}.jsonl").read_bytes() == frozen
        if split != "train":
            assert (roots["v8"] / f"{split}.jsonl").read_bytes() == frozen
    assert "train_decorrelation" in revised[3]


def test_actual_wrappers_cross_outcomes_and_evidence_match(revision):
    """Every wrapper supports both outcomes instead of predicting the answer or call."""

    family, _, _, roots, captured, _ = revision
    banks = captured["kwargs"]["prompts"]["train"]
    coverage = defaultdict(Counter)
    rows = _rows(roots["v8"] / "train.jsonl")
    for row in rows:
        framing, template = _wrapper(row, banks)
        coverage[(row["metadata"]["form"], framing)][template] += 1
    pairs = ((1, 3, "direct"), (2, 3, "step"), (4, 5, "direct"), (4, 5, "step"))
    if family == "s1_2":
        pairs = ((4, 5, "direct"), (4, 5, "step"), (7, 8, "direct"), (7, 8, "step"))
    for first, second, framing in pairs:
        assert set(coverage[(first, framing)]) == set(banks[framing])
        assert set(coverage[(second, framing)]) == set(banks[framing])


def test_all_failure_partners_have_identical_user_and_first_call(revision):
    """Runtime outcomes alone distinguish success, terminal failure, and retries."""

    family, _, _, roots, _, _ = revision
    rows = _rows(roots["v8"] / "train.jsonl")
    first_call = lambda row: next(m["tool_call"] for m in row["messages"] if "tool_call" in m)
    successes = {
        json.dumps(first_call(row), sort_keys=True): row
        for row in rows if row["metadata"]["form"] == 1
    }
    failure_form = 8 if family == "s1_1" else 11
    failures = [row for row in rows if row["metadata"]["form"] == failure_form]
    assert len(failures) == (400 if family == "s1_1" else 800)
    for row in failures:
        success = successes[json.dumps(first_call(row), sort_keys=True)]
        assert _user(row) == _user(success)
        assert row["messages"][-1]["content"]["continuation_decision"] in {"stop", "defer"}


def test_wrapper_count_does_not_select_content_or_failure_targets(revision):
    """Changing only train wrapper count leaves facts, distractors, and frames intact."""

    _, builder, _, _, captured, _ = revision
    kwargs = deepcopy(captured["kwargs"])
    kwargs["prompts"]["train"] = {
        framing: templates[:32]
        for framing, templates in kwargs["prompts"]["train"].items()
    }
    changed = builder.select_scenarios(*captured["args"], **kwargs)

    def content(scenarios):
        """Compare semantic selection independently of wrapper-derived row IDs."""

        return sorted(
            (s.form, s.fact.fact_id, s.occurrence, s.input_value, s.framing,
             s.distractor.fact_id if s.distractor else "")
            for s in scenarios
        )

    assert content(changed["train"]) == content(captured["selected"]["train"])
    for split in ("validation", "test"):
        assert changed[split] == captured["selected"][split]


def test_decorrelation_rejects_per_form_bank_overrides(revision, tmp_path):
    """Prevent a future overlay from silently restoring outcome-specific wording."""

    family, builder, _, _, _, revised = revision
    contract, payload, source, _ = revised
    contract = deepcopy(contract)
    contract["curriculum"]["families"][family.replace("s", "S", 1).replace("_", ".")][
        "train_prompt_banks"
    ] = {1: payload["prompt_banks"]}
    with pytest.raises(builder.CorpusBuildError, match="form-specific wrapper banks"):
        builder.build_corpus(source, tmp_path, contract=contract, prompt_banks=payload["prompt_banks"])


def test_joint_audit_rejects_a_wrapper_outside_shared_bank(revision):
    """Check rendered rows as well as scheduled wrapper-index coverage."""

    _, builder, _, roots, captured, _ = revision
    rows = _rows(roots["v8"] / "train.jsonl")
    next(m for m in rows[0]["messages"] if m["role"] == "user")["content"] += " Changed wording."
    with pytest.raises(builder.CorpusBuildError, match="shared wrapper bank"):
        builder._audit_train_decorrelation(
            rows, captured["selected"]["train"], captured["kwargs"]["prompts"],
        )
