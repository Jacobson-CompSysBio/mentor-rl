"""Test the S1 v3/v4 training prompt banks, their overlays, and corpora."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_1_corpus import IDENTITY_PROMPTS as S1_1_PROMPTS
from runtime.world_model.s1_2_corpus import IDENTITY_PROMPTS as S1_2_PROMPTS
from runtime.world_model.s1_overlay import (
    S1_V2_OVERLAY_PATH,
    S1_V3_OVERLAY_PATH,
    S1_V4_OVERLAY_PATH,
    S1OverlayError,
    apply_s1_overlay,
)
from runtime.world_model.s1_prompt_banks import (
    LATER_STEP_CUES,
    WRAPPER_BANKS,
    PromptBankError,
    _validated_banks,
    _validated_names,
    bank_reasoning,
    claim_phrases,
    held_out_identity_wordings,
    held_out_wordings,
    identity_prompts,
    load_prompt_banks,
    metric_names,
    validate_identity_bank,
    validate_wrapper_bank,
)
from runtime.world_model.s1_8_corpus import NAME_VARIANTS, SHARED_ALIASES


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_EVALUATOR = REPO_ROOT / "data/world_model_v4/eval/s1"
V3_CORPUS = REPO_ROOT / "data/world_model_v4/corpora/s1_v3"
V3_EVALUATOR = REPO_ROOT / "data/world_model_v4/eval/s1_v3"
V4_CORPUS = REPO_ROOT / "data/world_model_v4/corpora/s1_v4"
V4_EVALUATOR = REPO_ROOT / "data/world_model_v4/eval/s1_v4"
ABSENT_REASONING = (
    "The gene is not a node in the named multiplex, so the answer status is "
    "not_found."
)
V2_COMPONENTS = {
    component: REPO_ROOT / "data/world_model_v3/corpora/s1" / component
    for component in ("s1_1", "s1_3", "s1_4", "s1_6", "s1_7", "s1_8", "s1_9")
} | {"s1_2": REPO_ROOT / "data/world_model_v4/corpora/s1/s1_2"}
BANK_BY_COMPONENT = {
    "s1_1": "identity",
    "s1_2": "identity",
    "s1_3": "identity_s1_3",
    "s1_4": "alignment",
    "s1_6": "layer",
    "s1_7": "layer_family",
    "s1_8": "metric",
    "s1_9": "module",
}


def _v3_payload() -> dict:
    return yaml.safe_load(S1_V3_OVERLAY_PATH.read_text(encoding="utf-8"))


def _bank_payload() -> dict:
    pin = _v3_payload()["prompt_banks"]
    return json.loads((REPO_ROOT / pin["path"]).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_bank_replaces_only_the_training_wrappers() -> None:
    pin = _v3_payload()["prompt_banks"]
    for legacy, bank_name in (
        (S1_1_PROMPTS, "identity"),
        (S1_2_PROMPTS, "identity"),
        (S1_1_PROMPTS, "identity_s1_3"),
    ):
        prompts = identity_prompts(pin, legacy, bank_name=bank_name)
        assert prompts["validation"] == legacy["validation"]
        assert prompts["test"] == legacy["test"]
        assert prompts["train"] == {
            framing: tuple(_bank_payload()[bank_name][framing])
            for framing in ("direct", "step")
        }


def test_v3_banks_are_large_and_far_from_held_out_wrappers() -> None:
    audits = _validated_banks(_bank_payload())
    assert audits["identity"]["wrapper_counts"] == {"direct": 64, "step": 64}
    for name in WRAPPER_BANKS:
        assert audits[name]["held_out_wrappers_checked"] == len(held_out_wordings())
        assert audits[name]["maximum_held_out_jaccard"] < 0.6
        if name != "identity":
            assert audits[name]["wrapper_counts"] == {"direct": 32, "step": 32}
    names = audits["s1_8_names"]
    assert set(names["metric_variant_counts"].values()) == {8}
    assert names["shared_alias_count"] == 16
    assert names["maximum_new_name_jaccard"] < 0.67


def _bank_with(framing: str, wrapper: str) -> dict[str, tuple[str, ...]]:
    bank = {
        name: list(wrappers)
        for name, wrappers in _bank_payload()["identity"].items()
    }
    bank[framing][-1] = wrapper
    return {name: tuple(wrappers) for name, wrappers in bank.items()}


@pytest.mark.parametrize(
    ("framing", "wrapper", "message"),
    [
        ("direct", "Answer this gene request.", "exactly one {request} slot"),
        ("direct", "Answer {request} and {request}.", "exactly one {request}"),
        ("direct", "Look up this gene: {request}.", "names 'Look up'"),
        ("direct", "Map this symbol: {request}.", "names 'symbol'"),
        ("direct", "Answer this before the next stage: {request}.", "implies a later"),
        ("step", "Answer this one gene question: {request}.", "no later-step cue"),
        ("step", "Answer this gene query {request} then.", "known verb"),
        (
            "step",
            "The workflow first needs this gene request resolved: {request}.",
            "is a held-out wrapper",
        ),
        (
            "step",
            "Before the workflow advances, determine this gene fact now: {request}.",
            "5-word run",
        ),
    ],
)
def test_bank_validator_rejects_an_unsafe_wrapper(
    framing: str, wrapper: str, message: str
) -> None:
    with pytest.raises(PromptBankError, match=message):
        validate_identity_bank(
            _bank_with(framing, wrapper), held_out=held_out_identity_wordings()
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["identity_s1_3"]["direct"].__setitem__(
                0, "Tell me {request}."
            ),
            "must end at the request",
        ),
        (
            lambda payload: payload["identity_s1_3"]["step"].__setitem__(
                0, "A later stage relies on this gene answer: {request}."
            ),
            "not in the identity bank",
        ),
    ],
)
def test_s1_3_subset_must_end_at_the_request_and_share_wording(
    change, message
) -> None:
    payload = copy.deepcopy(_bank_payload())
    change(payload)
    with pytest.raises(PromptBankError, match=message):
        _validated_banks(payload)


def test_v3_overlay_moves_every_enabled_stage_and_keeps_v2_settings() -> None:
    contract = load_contract()
    v2, _ = apply_s1_overlay(contract, S1_V2_OVERLAY_PATH)
    v3, payload = apply_s1_overlay(contract, S1_V3_OVERLAY_PATH)

    families, v2_families = (
        v3["curriculum"]["families"],
        v2["curriculum"]["families"],
    )
    moved = set(payload["family_overrides"])
    assert moved == {"S1.1", "S1.2", "S1.3", "S1.4", "S1.6", "S1.7", "S1.8", "S1.9"}
    for section, settings in families.items():
        if section in moved:
            assert settings["output_directory"] == (
                f"data/world_model_v4/corpora/s1_v3/{section.lower().replace('.', '_')}"
            )
            settings = {**settings, "output_directory": None}
            expected = {**v2_families[section], "output_directory": None}
            assert settings == expected
        else:
            assert settings == v2_families[section]
    assert payload["prompt_banks"]["sha256"] == _sha256(
        REPO_ROOT / payload["prompt_banks"]["path"]
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["prompt_banks"].update(sha256="0" * 64),
            "does not match its pin",
        ),
        (
            lambda payload: payload["family_overrides"]["S1.1"].update(
                shares={}
            ),
            "S1.1 override keys changed",
        ),
        (lambda payload: payload.pop("prompt_banks"), "keys changed"),
    ],
)
def test_v3_overlay_rejects_an_invalid_change(tmp_path: Path, change, message) -> None:
    payload = _v3_payload()
    change(payload)
    path = tmp_path / "s1_overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(S1OverlayError, match=message):
        apply_s1_overlay(load_contract(), path)


def test_v1_overlay_schema_still_rejects_s1_1(tmp_path: Path) -> None:
    payload = yaml.safe_load(S1_V2_OVERLAY_PATH.read_text(encoding="utf-8"))
    payload["family_overrides"]["S1.1"] = {"output_directory": "x"}
    path = tmp_path / "s1_overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(S1OverlayError, match="unsupported family"):
        apply_s1_overlay(load_contract(), path)


def _wrapper_bank_with(name: str, framing: str, wrapper: str) -> dict[str, tuple[str, ...]]:
    bank = {key: list(value) for key, value in _bank_payload()[name].items()}
    bank[framing][-1] = wrapper
    return {key: tuple(value) for key, value in bank.items()}


@pytest.mark.parametrize(
    ("name", "framing", "wrapper", "message"),
    [
        ("layer", "step", "The analysis will later use the answer to this one layer request: {request}.", "above 10"),
        ("layer", "direct", "Filter this layer request: {request}.", "names 'Filter'"),
        ("layer_family", "direct", "Answer this layer question: {request}.", "newline"),
        ("metric", "direct", "Is a higher value closer?\n{request}", "names 'higher'"),
        ("module", "direct", "Parse this seed module.\n{request}", "names 'seed'"),
        ("alignment", "direct", "Say whether these name the same gene: {request}.", "names"),
        ("identity", "direct", "Note that the task is complete: {request}.", "answer phrase"),
    ],
)
def test_family_rules_reject_an_unsafe_wrapper(name, framing, wrapper, message) -> None:
    with pytest.raises(PromptBankError, match=message):
        validate_wrapper_bank(
            _wrapper_bank_with(name, framing, wrapper),
            spec=WRAPPER_BANKS[name],
            held_out=held_out_wordings(),
            phrases=claim_phrases(),
            name=name,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["metric_name_variants"]["rank"].append("random-walk rank"), "held-out name"),
        (lambda p: p["metric_name_variants"]["rank"].append("rank of a node in the walk"), "more than 4 words"),
        (lambda p: p["metric_name_variants"]["rank"].pop(0), "start with its v2 training names"),
        (lambda p: p["shared_aliases"].update({"Spearman-style distance": "distance"}), "names a method"),
        (lambda p: p["shared_aliases"].update({"closeness": "proximity"}), "unknown family"),
        (lambda p: p["shared_aliases"].update({"cosine distance": "distance"}), "repeats another"),
    ],
)
def test_s1_8_name_rules_reject_an_unsafe_name(change, message) -> None:
    payload = copy.deepcopy(_bank_payload())
    change(payload)
    with pytest.raises(PromptBankError, match=message):
        _validated_names(payload)


def test_new_s1_8_aliases_copy_a_v2_convention_card() -> None:
    variants, aliases = metric_names(
        _v3_payload()["prompt_banks"], NAME_VARIANTS, SHARED_ALIASES
    )
    assert variants["validation"] == {
        metric: tuple(names) for metric, names in NAME_VARIANTS["validation"].items()
    }
    assert aliases["test"] == SHARED_ALIASES["test"]
    for alias, family in _bank_payload()["shared_aliases"].items():
        assert aliases["train"][alias] is SHARED_ALIASES["train"][family]


@pytest.mark.skipif(not V3_EVALUATOR.exists(), reason="the v3 corpus is not built")
def test_published_v3_corpus_keeps_the_v2_held_out_panels() -> None:
    for name in (
        "validation_questions.jsonl",
        "validation_answer_key.jsonl",
        "test_questions.jsonl",
        "test_answer_key.jsonl",
    ):
        assert _sha256(V3_EVALUATOR / name) == _sha256(V2_EVALUATOR / name)
    for component, v2_directory in V2_COMPONENTS.items():
        for split in ("validation", "test"):
            assert _sha256(V3_CORPUS / component / f"{split}.jsonl") == _sha256(
                v2_directory / f"{split}.jsonl"
            )


def _training_template_use(manifest: dict) -> dict[tuple[str, str], set[int]]:
    """Read either manifest layout of training template counts."""

    used: dict[tuple[str, str], set[int]] = defaultdict(set)
    for key, value in manifest["prompt_template_counts"]["train"].items():
        if isinstance(value, dict):
            for framing, counts in value.items():
                used[(key, framing)].update(int(index) for index in counts)
        else:
            trace, framing, index = key.rsplit(":", 2)
            used[(trace, framing)].add(int(index))
    return used


@pytest.mark.skipif(not V3_CORPUS.exists(), reason="the v3 corpus is not built")
def test_published_v3_training_rows_cover_every_bank_wrapper() -> None:
    bank_payload = _bank_payload()
    corpus = json.loads((V3_CORPUS / "manifest.json").read_text(encoding="utf-8"))
    for component, bank_name in BANK_BY_COMPONENT.items():
        manifest = json.loads(
            (V3_CORPUS / component / "manifest.json").read_text(encoding="utf-8")
        )
        assert corpus["components"][component]["manifest_path"] == (
            f"data/world_model_v4/corpora/s1_v3/{component}/manifest.json"
        )
        assert manifest["prompt_bank"]["sha256"] == _v3_payload()["prompt_banks"][
            "sha256"
        ]
        used = _training_template_use(manifest)
        assert used
        for (_, framing), indexes in used.items():
            assert indexes == set(range(len(bank_payload[bank_name][framing])))


def _v4_pin() -> dict:
    return yaml.safe_load(S1_V4_OVERLAY_PATH.read_text(encoding="utf-8"))[
        "prompt_banks"
    ]


def test_v4_bank_adds_only_the_s1_1_10_reasoning() -> None:
    v3 = _bank_payload()
    v4 = json.loads((REPO_ROOT / _v4_pin()["path"]).read_text(encoding="utf-8"))
    assert v4.pop("reasoning") == {"s1_1_absent_membership": ABSENT_REASONING}
    v3.pop("description")
    v4.pop("description")
    assert v4 == v3
    assert bank_reasoning(_v4_pin(), "s1_1_absent_membership") == ABSENT_REASONING
    assert bank_reasoning(_v3_payload()["prompt_banks"], "s1_1_absent_membership") is None


@pytest.mark.parametrize(
    ("reasoning", "message"),
    [
        ({"s1_2_other": "A short sentence."}, "reasoning keys changed"),
        ({"s1_1_absent_membership": "Use {request} here."}, "must be nonempty, trimmed"),
        ({"s1_1_absent_membership": ""}, "must be nonempty, trimmed"),
    ],
)
def test_bank_reasoning_must_be_known_and_well_formed(
    tmp_path: Path, reasoning: dict, message: str
) -> None:
    payload = _bank_payload() | {"reasoning": reasoning}
    path = tmp_path / "banks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptBankError, match=message):
        load_prompt_banks({"path": str(path), "sha256": _sha256(path)})


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.skipif(not V4_EVALUATOR.exists(), reason="the v4 corpus is not built")
def test_published_v4_corpus_changes_only_s1_1_10_training_reasoning() -> None:
    for name in (
        "validation_questions.jsonl",
        "validation_answer_key.jsonl",
        "test_questions.jsonl",
        "test_answer_key.jsonl",
    ):
        assert _sha256(V4_EVALUATOR / name) == _sha256(V3_EVALUATOR / name)
    for split in ("validation", "test"):
        assert _sha256(V4_CORPUS / f"{split}.jsonl") == _sha256(
            V3_CORPUS / f"{split}.jsonl"
        )
    manifest = json.loads(
        (V4_CORPUS / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["train_normalization"] == {
        "evidence_id_policy": "aggregate_sequence_v1",
        "s0_metadata_additions_removed": [
            "fact_kind",
            "framing",
            "prompt_template_id",
            "rendering_index",
            "input_variant",
        ],
    }
    for component in BANK_BY_COMPONENT:
        for split in ("train", "validation", "test"):
            if (component, split) == ("s1_1", "train"):
                continue
            assert _sha256(V4_CORPUS / component / f"{split}.jsonl") == _sha256(
                V3_CORPUS / component / f"{split}.jsonl"
            )
    v3_rows = _rows(V3_CORPUS / "s1_1/train.jsonl")
    v4_rows = _rows(V4_CORPUS / "s1_1/train.jsonl")
    assert len(v4_rows) == len(v3_rows)
    changed = 0
    for v3_row, v4_row in zip(v3_rows, v4_rows):
        if v4_row["metadata"]["source_trace"] != "S1.1.10":
            assert v4_row == v3_row
            continue
        final = v4_row["messages"][-1]
        assert final["thinking"] == ABSENT_REASONING
        assert final["content"]["status"] == "not_found"
        v3_row["messages"][-1]["thinking"] = ABSENT_REASONING
        assert v4_row == v3_row
        changed += 1
    assert changed == 700


def test_the_following_is_not_a_later_step_cue() -> None:
    assert not LATER_STEP_CUES.search("Answer the following request: {request}.")
    for cue in ("A follow-up step", "More steps follow", "Then it is followed up"):
        assert LATER_STEP_CUES.search(cue)


def test_v5_identity_wrappers_differ_only_by_the_later_use_sentence() -> None:
    """Keep ordinary request wording balanced across direct and step frames."""

    bank = json.loads(
        (REPO_ROOT / "runtime/world_model/s1_v5_prompt_banks.json").read_text()
    )
    for name, count in (("identity", 64), ("identity_s1_3", 32)):
        assert len(bank[name]["direct"]) == len(bank[name]["step"]) == count
        for direct, step in zip(
            bank[name]["direct"], bank[name]["step"], strict=True,
        ):
            cue, request = step.split(". ", 1)
            assert request == direct
            assert LATER_STEP_CUES.search(cue)


def test_v5_bank_states_that_a_gene_without_a_symbol_is_usable() -> None:
    pin = yaml.safe_load(
        (REPO_ROOT / "runtime/world_model/s1_v5.yaml").read_text(encoding="utf-8")
    )["prompt_banks"]
    for framing, decision in (("step", "continue"), ("direct", "complete")):
        sentence = bank_reasoning(pin, f"s1_2_no_symbol_{framing}")
        assert "not yet named" in sentence and decision in sentence


def test_bank_reasoning_allows_longer_traces(tmp_path):
    """Reasoning length is a review choice, not a sentence or word gate."""

    reasoning = "This explanation needs several steps. " + "The evidence remains available " * 30 + ". The result follows."
    payload = _bank_payload() | {"reasoning": {"s1_1_absent_membership": reasoning}}
    path = tmp_path / "banks.json"
    path.write_text(json.dumps(payload))
    pin = {"path": str(path), "sha256": _sha256(path)}
    assert bank_reasoning(pin, "s1_1_absent_membership") == reasoning
