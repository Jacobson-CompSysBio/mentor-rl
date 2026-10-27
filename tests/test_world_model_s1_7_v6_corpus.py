"""Check the reviewed S1.7 reasoning and its frozen training and held-out boundaries."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_7_corpus import (
    CLAIM_VOCABULARY, FAMILY_PROMPTS, CorpusBuildError, LayerFact, Scenario,
    _prose_length_audit, build_corpus, build_row, filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)
from runtime.world_model.s1_prompt_banks import wrapper_prompts


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_7")
REASONING_SHA256 = "2e04db7a489d3d3549bca2fbbef97829fa2643c6f47f7bf108e90a4c57c025c2"
FIELDS = {
    2: ("requested_family", "layer_assignments", "matching_layer_ids", "other_layer_ids"),
    4: ("claim_phrase", "claim", "claim_definition", "family_assignment", "layer_id",
        "layer_family", "evidence_type", "supported_claims", "unsupported_claims", "claim_supported"),
    8: ("layer_id", "network"),
}


def _reasoning():
    """Read the reviewed templates without depending on a built corpus."""

    return yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())["family_overrides"]["S1.7"][
        "train_final_reasoning"
    ]


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build and semantically validate the complete component in a temporary directory."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_7_v6")
    manifest = build_corpus(
        source, root, contract=contract, overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_training_thoughts_change(revision):
    """Preserve every prompt, target, row identity, passing type, and held-out byte."""

    contract, payload, _, root, manifest = revision
    templates = contract["curriculum"]["families"]["S1.7"]["train_final_reasoning"]
    serialized = json.dumps(templates, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode()).hexdigest() == REASONING_SHA256
    assert manifest["train_final_reasoning"] == templates
    prompts = wrapper_prompts(payload["prompt_banks"], FAMILY_PROMPTS, bank_name="layer_family")
    changed, unchanged = Counter(), 0
    families, list_outcomes, claim_outcomes, other_networks = Counter(), Counter(), Counter(), Counter()
    phrases = defaultdict(set)
    for before, after in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(before), json.loads(after)
        form = old["metadata"]["form"]
        if form not in FIELDS:
            assert before == after
            unchanged += 1
            continue
        thought = new["messages"][-1]["thinking"]
        assert thought != old["messages"][-1]["thinking"]
        restored = deepcopy(new)
        restored["messages"][-1]["thinking"] = old["messages"][-1]["thinking"]
        assert restored == old
        user = new["messages"][1]["content"]
        frames = {
            frame for frame, wrappers in prompts["train"].items()
            if any(user.startswith(wrapper.split("{request}")[0]) for wrapper in wrappers)
        }
        assert len(frames) == 1
        frame = frames.pop()
        answer = new["messages"][-1]["content"]
        result = answer["result"]
        assert answer["status"] == "resolved"
        assert answer["blocked_reason"] is None and answer["provenance"] is None
        assert answer["continuation_decision"] == ("stop" if frame == "direct" else "continue")
        assert "The user asked only" in thought if frame == "direct" else "later workflow step" in thought
        assert len(new["messages"]) == 3
        assert "biogrid_ppi" in thought
        changed[form, frame] += 1
        if form == 2:
            tags = re.findall(r"(?:bulkPEN|scPEN|HumanNetV3|TFs):[A-Za-z0-9_:-]+", user)
            assert len(tags) == 3
            families[result["layer_family"]] += 1
            for index, tag in enumerate(tags, start=1):
                prefix, remainder = tag.split(":", 1)
                family = {
                    "bulkPEN": "bulk_pen", "scPEN": "scpen", "TFs": "tf_target",
                    "HumanNetV3": {"string_ppi": "ppi", "biogrid_ppi": "ppi",
                                   "coexpression": "coexpression"}.get(remainder, "other"),
                }[prefix]
                assert f"Input {index} is {tag}, which maps to {family}." in thought
            assert f"requested family {result['layer_family']}." in thought
            for key in ("matching_layer_ids", "other_layer_ids"):
                assert json.dumps(result[key], ensure_ascii=False) in thought
                assert result[key] == [tag for tag in tags if tag in result[key]]
            assert set(result["matching_layer_ids"]).isdisjoint(result["other_layer_ids"])
            assert set(result["matching_layer_ids"] + result["other_layer_ids"]) == set(tags)
            list_outcomes[bool(result["matching_layer_ids"])] += 1
        elif form == 4:
            phrase = re.search(r'phrase "([^"]+)" is supported', thought).group(1)
            assert phrase in user
            phrases[result["claim"]].add(phrase)
            assert f"Its canonical claim is {result['claim']}:" in thought
            for key in ("layer_id", "layer_family", "evidence_type", "supported_claims",
                        "unsupported_claims", "claim_supported"):
                assert f"{key} is {json.dumps(result[key])}" in thought or (
                    key in {"supported_claims", "unsupported_claims"}
                    and f"{key} {json.dumps(result[key])}" in thought
                )
            assert result["claim_supported"] == (result["claim"] in result["supported_claims"])
            assert result["unsupported_claims"] == [
                claim for claim in CLAIM_VOCABULARY if claim not in result["supported_claims"]
            ]
            if result["layer_family"] == "tf_target":
                assert result["supported_claims"] == ["regulatory_link_recorded"]
                assert "regulation_direction" in result["unsupported_claims"]
            claim_outcomes[result["claim_supported"]] += 1
        else:
            network = result["layer_id"].split(":")[1]
            other_networks[network] += 1
            assert f"complete network component {network}." in thought
            assert result["evidence_type"] is None and result["supported_claims"] == []
            assert result["unsupported_claims"] == list(CLAIM_VOCABULARY)
            assert ", ".join(CLAIM_VOCABULARY) in thought
        assert "HumanNetV3:phylogenetic" not in thought and "HumanNetV3:database" not in thought
    assert changed == {(2, "direct"): 450, (2, "step"): 450,
                       (4, "direct"): 750, (4, "step"): 750,
                       (8, "direct"): 300, (8, "step"): 300}
    assert unchanged == 3000
    assert families == {family: 150 for family in ("ppi", "coexpression", "tf_target", "bulk_pen", "scpen", "other")}
    assert list_outcomes == {True: 450, False: 450}
    assert claim_outcomes == {True: 600, False: 900}
    assert {claim: len(values) for claim, values in phrases.items()} == {
        claim: 32 for claim in CLAIM_VOCABULARY
    }
    assert other_networks == {"cocitation": 200, "gene_interaction": 200, "gene_neighbor": 200}
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    assert manifest["evidence"] == json.loads((BASELINE / "manifest.json").read_text())["evidence"]


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """Reproduce every row and manifest from the same pinned inputs."""

    contract, payload, source, root, _ = revision
    build_corpus(source, tmp_path, contract=contract, prompt_banks=payload["prompt_banks"],
                 overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload))
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_7")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_7")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_splits(overlay_path, baseline, tmp_path):
    """Keep no-overlay, v4, and v5 corpus bytes reproducible."""

    contract, payload = load_contract(), {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    build_corpus(source, tmp_path, contract=contract, prompt_banks=payload.get("prompt_banks"))
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize(("form", "field"), [(form, field) for form, fields in FIELDS.items() for field in fields])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "attribute"])
def test_overlay_requires_exact_bindings(tmp_path, form, field, mutation):
    """Reject missing, repeated, or attribute-access source slots."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    templates = payload["family_overrides"]["S1.7"]["train_final_reasoning"]
    slot = "{" + field + "}"
    replacement = {"missing": "a value", "duplicate": slot + " " + slot,
                   "attribute": "{" + field + ".real}"}[mutation]
    templates[form]["direct"] = templates[form]["direct"].replace(slot, replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="reasoning"):
        apply_s1_overlay(load_contract(), path)


def _scenario(form, *, split="train", framing="direct"):
    """Provide distinct native tags and an unsupported regulatory-direction check."""

    def fact(tag, family, source=None, network=None):
        return LayerFact(tag, tag, family, source, network, "brain" if source is None else None,
                         ("amygdala",) if source is None else None,
                         ("astrocyte",) if family == "scpen" else None, tag)
    other = fact("HumanNetV3:gene_neighbor", "other", "HumanNetV3", "gene_neighbor")
    if form == 2:
        companions = (other, fact("scPEN:brain:amygdala:astrocyte", "scpen"),
                      fact("bulkPEN:brain:amygdala", "bulk_pen"))
        return Scenario(split, form, "scpen", None, companions=companions,
                        list_outcome="match", input_kind="layer_list", framing=framing,
                        train_reasoning=split == "train")
    if form == 4:
        return Scenario(split, form, "tf_target", None, claim="regulation_direction",
                        claim_supported=False, input_kind="family_name", framing=framing,
                        train_reasoning=split == "train")
    return Scenario(split, form, "other", other, framing=framing, train_reasoning=split == "train")


@pytest.mark.parametrize("form", [2, 4, 8])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_templates_cannot_change_heldout_thoughts(form, split):
    """Ignore reviewed training templates on every held-out form."""

    scenario = _scenario(form, split=split)
    old, _ = build_row(scenario, contract=load_contract())
    new, _ = build_row(scenario, contract=load_contract(), final_reasoning=_reasoning())
    assert new == old
    assert filter_row_thinking(new, scenario, final_reasoning=_reasoning()) == 1


@pytest.mark.parametrize("form", [2, 4, 8])
@pytest.mark.parametrize("framing", ["direct", "step"])
@pytest.mark.parametrize("mutation", ["thought", "decision", "claims"])
def test_filter_keeps_binding_and_answer_checks(form, framing, mutation):
    """Reject changed reasoning, decision, and structured result semantics."""

    scenario = _scenario(form, framing=framing)
    row, _ = build_row(scenario, contract=load_contract(), final_reasoning=_reasoning())
    assert filter_row_thinking(row, scenario, final_reasoning=_reasoning()) == 1
    if mutation == "thought":
        row["messages"][-1]["thinking"] += " The classification differs."
    elif mutation == "decision":
        row["messages"][-1]["content"]["continuation_decision"] = "defer"
    else:
        row["messages"][-1]["content"]["result"]["supported_claims"] = ["invented_claim"]
    with pytest.raises(CorpusBuildError):
        filter_row_thinking(row, scenario, final_reasoning=_reasoning())


@pytest.mark.parametrize(("suffix", "error"), [
    ("The tag HumanNetV3:unseen_network is available.", "invents the tag"),
    ("The tool result confirms this.", "runtime fields"),
    ("Preserve provenance.", "runtime fields"),
])
@pytest.mark.parametrize("form", [2, 4, 8])
def test_template_match_does_not_bypass_visibility(form, suffix, error):
    """Reject fabricated identifiers and unavailable runtime evidence even in a template."""

    scenario = _scenario(form)
    templates = _reasoning()
    templates[form] = {frame: text + " " + suffix for frame, text in templates[form].items()}
    row, _ = build_row(scenario, contract=load_contract(), final_reasoning=templates)
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, final_reasoning=templates)


def test_reasoning_has_no_sentence_word_cap():
    """Allow an exact long thought while retaining the wrapper and answer prose checks."""

    scenario = _scenario(2)
    templates = _reasoning()
    templates[2]["direct"] += " " + " ".join(["clear"] * 40) + "."
    row, _ = build_row(scenario, contract=load_contract(), final_reasoning=templates)
    assert filter_row_thinking(row, scenario, final_reasoning=templates) == 1
    assert _prose_length_audit({split: [row] for split in ("train", "validation", "test")})["pass"]
