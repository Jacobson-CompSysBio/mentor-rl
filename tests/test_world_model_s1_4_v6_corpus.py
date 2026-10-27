"""Verify the approved S1.4.1-2/5-6/8 reasoning against frozen corpus rows."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_4_corpus import (
    ALIGNMENT_PROMPTS, AmbiguousFact, CorpusBuildError, ExternalFact, GeneFact, Scenario,
    build_corpus, filter_row_thinking,
)
from runtime.world_model.s1_overlay import (
    S1_V4_OVERLAY_PATH, S1_V5_OVERLAY_PATH, S1_V6_OVERLAY_PATH,
    S1OverlayError, apply_s1_overlay, overlay_pin,
)
from runtime.world_model.s1_prompt_banks import wrapper_prompts


BASELINE = Path("data/world_model_v4/corpora/s1_v5/s1_4")
REASONING_BODY = (
    "The registry maps {symbol} to {gene_id}, matching the ID supplied for the module entry, "
    "so both mentions name the same gene. "
    "Build first and second in the user’s order, with exactly item, input, and gene_id in each. "
    "Use typed_seed_list for the seed’s item and module for the module entry’s item. "
    "Preserve both literal inputs, use {gene_id} for both gene_id values, and set same_gene to true. "
    "Preserve the lookup’s registry evidence, with no graph evidence, and return resolved with no blocked reason. "
    "The supplied module context is sufficient for this comparison, so no further tool call is needed. "
)
REASONING = {
    "direct": REASONING_BODY + "The user asked only for this identity comparison. It is complete, so stop.",
    "step": REASONING_BODY + "The user’s later workflow step needs this identity comparison, so continue with the resolved result.",
}
MISMATCH_BODY = (
    "The registry maps {symbol} to {gene_id}, which differs from the supplied seed ID {target_id}. "
    "The two mentions name different genes, so the identity comparison is resolved with same_gene set to false. "
    "Both IDs are available; no further lookup or multiplex membership check is needed. "
    "Build first and second in the user’s order, with exactly item, input, and gene_id in each. "
    "Use abstract for the abstract mention’s item and seed_gene_set for the seed’s item. "
    "Preserve both literal inputs, use {gene_id} for the abstract mention’s gene_id, "
    "and use {target_id} for the seed’s gene_id. "
    "Preserve the lookup’s registry evidence, with no graph evidence, and return resolved with no blocked reason. "
)
MISMATCH_REASONING = {
    "direct": MISMATCH_BODY + "The user asked only for this identity comparison. It is complete, so stop.",
    "step": MISMATCH_BODY + "The user’s later workflow step needs this identity comparison, so continue with the resolved result.",
}
UNRESOLVED_BODY = (
    "The registry returned not_found for {input}, with no candidate IDs. "
    "The abstract mention’s identity remains unknown; an empty candidate list does not justify setting same_gene to false. "
    "I cannot infer a mapping from memory, the module context, or the literal input strings. "
    "Build result with exactly first, second, and same_gene. "
    "Put first and second in the user’s order, with exactly item, input, and gene_id in each. "
    "Use abstract for the abstract mention’s item and module for the module entry’s item. "
    "Preserve both literal inputs, set the abstract mention’s gene_id to null, "
    "and retain {target_id} as the module entry’s gene_id. "
    "Set same_gene to null and return not_found with a nonempty blocked reason explaining the unresolved symbol. "
    "Preserve the lookup’s registry evidence, with no graph evidence. "
    "Do not retry the lookup, reinterpret the symbol as an ID, or call membership. "
)
UNRESOLVED_REASONING = {
    "direct": UNRESOLVED_BODY + (
        "The user’s identity comparison remains unanswered, so defer until they supply "
        "a current gene symbol or Ensembl gene ID for the abstract mention."
    ),
    "step": UNRESOLVED_BODY + (
        "The user’s later workflow step needs this comparison, and the unresolved abstract mention blocks it. "
        "Defer until the user supplies a current gene symbol or Ensembl gene ID for that mention."
    ),
}


AMBIGUOUS_BODY = (
    "Build result with exactly first, second, and same_gene, keeping first and second in the user’s mention order. "
    "For the typed seed, use exactly item, input, gene_id, candidate_gene_ids, and candidate_details, "
    "with item set to typed_seed_list and gene_id set to null. "
    "Preserve the full returned candidate_gene_ids list in its original order and candidate_details unchanged; "
    "the comparison does not filter them. "
    "For the module entry, use exactly item, input, and gene_id, with item set to module and gene_id set to {target_id}. "
    "Preserve both literal inputs and the lookup’s registry evidence, with no graph evidence. "
    "No multiplex is named, and no further lookup or membership check is needed. "
)
CANDIDATE_BODY = (
    "The registry maps {symbol} to {count} candidate IDs, and {target_id} is one of them. "
    "The module identifies the comparison target but does not select which candidate the typed seed means. "
    "The seed could refer to another candidate, so the comparison remains unknown. "
    + AMBIGUOUS_BODY +
    "Set same_gene to null and return ambiguous with a nonempty blocked reason explaining that the typed seed has multiple candidates. "
)
CANDIDATE_REASONING = {
    "direct": CANDIDATE_BODY + (
        "The user’s identity comparison remains unanswered, so defer until they choose one Ensembl gene ID for the typed seed."
    ),
    "step": CANDIDATE_BODY + (
        "The user’s later workflow step needs this comparison, and the unselected seed identity blocks it. "
        "Defer until the user chooses one Ensembl gene ID for the typed seed."
    ),
}
NONCANDIDATE_BODY = (
    "The registry maps {symbol} to {count} candidate IDs, and {target_id} is not one of them. "
    "Every candidate in this nonempty list differs from the comparison target, so the two mentions name different genes. "
    "The comparison is resolved without selecting a single candidate for the typed seed. "
    + AMBIGUOUS_BODY +
    "Set same_gene to false and return resolved with no blocked reason. "
)
NONCANDIDATE_REASONING = {
    "direct": NONCANDIDATE_BODY + "The user asked only for this identity comparison. It is complete, so stop.",
    "step": NONCANDIDATE_BODY + "The user’s later workflow step needs this identity comparison, so continue with the resolved result.",
}


MYGENE_BODY = (
    "The release 116 registry confirms {gene_id} with symbol {symbol}, "
    "matching the ID supplied by query_mygene for {query} and the module entry. "
    "Both mentions name the same gene. "
    "Build result with exactly first, second, and same_gene. "
    "Put first and second in the user’s mention order, with exactly item, input, and gene_id in each. "
    "Use query_mygene for the external entry’s item and module for the module entry’s item. "
    "Keep {query} as the query_mygene input and preserve the module input exactly as supplied. "
    "Use {gene_id} for both gene_id values and set same_gene to true. "
    "Do not add gene_symbol to either item. "
    "Preserve the lookup’s registry evidence, with no graph evidence, and return resolved with no blocked reason. "
    "The supplied module context is sufficient for this identity comparison, so no further lookup or membership check is needed. "
)
MYGENE_REASONING = {
    "direct": MYGENE_BODY + "The user asked only for this identity comparison. It is complete, so stop.",
    "step": MYGENE_BODY + "The user’s later workflow step needs this identity comparison, so continue with the resolved result.",
}


def _training_framing(row, prompts):
    """Read framing from the user wrapper, including when both outcomes defer."""

    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    matches = {
        framing for framing, templates in prompts["train"].items()
        if any(user.startswith(text.split("{request}")[0]) for text in templates)
    }
    assert len(matches) == 1
    return matches.pop()


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build all S1.4 splits with semantic and wording audits enabled."""

    contract, payload = apply_s1_overlay(load_contract(), S1_V6_OVERLAY_PATH)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_4_v6")
    manifest = build_corpus(
        source, root, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    return contract, payload, source, root, manifest


def test_only_reviewed_final_training_thoughts_change(revision):
    """Keep every prompt, call, answer, other training row, and held-out byte."""

    root, manifest = revision[3:]
    coverage, unchanged, failure_partners = {form: Counter() for form in (1, 2, 5, 6, 8)}, 0, 0
    ambiguous_coverage = Counter()
    prompts = wrapper_prompts(revision[1]["prompt_banks"], ALIGNMENT_PROMPTS, bank_name="alignment")
    for old_line, new_line in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (root / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_line), json.loads(new_line)
        form = old["metadata"]["form"]
        if form not in {1, 2, 5, 6, 8}:
            assert old_line == new_line
            unchanged += 1
            failure_partners += old["metadata"]["form"] == 9
            continue
        output = next(m["content"] for m in old["messages"] if m["role"] == "tool")
        final = old["messages"][-1]["content"]
        framing = _training_framing(old, prompts)
        expected = deepcopy(old)
        bindings = {"symbol": output["gene_symbol"], "gene_id": output.get("gene_id")}
        reasoning = {1: REASONING, 2: MISMATCH_REASONING, 5: UNRESOLVED_REASONING, 6: CANDIDATE_REASONING, 8: MYGENE_REASONING}[form]
        if form in {2, 5, 6}:
            bindings["target_id"] = next(
                final["result"][side]["gene_id"] for side in ("first", "second")
                if final["result"][side]["item"] == ("seed_gene_set" if form == 2 else "module")
            )
        if form == 5:
            bindings = {"input": output["gene_symbol"], "target_id": bindings["target_id"]}
        if form == 6:
            candidate = bindings["target_id"] in output["candidate_gene_ids"]
            reasoning = CANDIDATE_REASONING if candidate else NONCANDIDATE_REASONING
            bindings["count"] = len(output["candidate_gene_ids"])
            ambiguous_coverage[candidate, framing] += 1
            assert final["result"]["same_gene"] is (None if candidate else False)
            assert final["status"] == ("ambiguous" if candidate else "resolved")
            seed = next(final["result"][side] for side in ("first", "second") if final["result"][side]["item"] == "typed_seed_list")
            assert seed["gene_id"] is None
            assert seed["candidate_gene_ids"] == output["candidate_gene_ids"]
            assert seed["candidate_details"] == output["candidate_details"]
        if form == 8:
            result = final["result"]
            external = next(result[side] for side in ("first", "second") if result[side]["item"] == "query_mygene")
            bindings["query"] = external["input"]
            harness = next(m["content"] for m in old["messages"] if m["role"] == "harness")
            mygene = json.loads(harness.split("query_mygene result:\n", 1)[1].split("\nModule ", 1)[0])
            assert external["input"] == mygene["query"]
            assert set(result) == {"first", "second", "same_gene"}
            assert result["same_gene"] is True
            for side in ("first", "second"):
                assert set(result[side]) == {"item", "input", "gene_id"}
                assert result[side]["gene_id"] == output["gene_id"]
        expected["messages"][-1]["thinking"] = reasoning[framing].format(**bindings)
        assert old != expected
        assert new == expected
        coverage[form][framing] += 1
        coverage[form][final["result"]["first"]["item"]] += 1
    assert coverage[1] == {"direct": 375, "step": 375, "typed_seed_list": 375, "module": 375}
    assert coverage[2] == {"direct": 500, "step": 500, "abstract": 500, "seed_gene_set": 500}
    assert coverage[5] == {"direct": 500, "step": 500, "abstract": 500, "module": 500}
    assert coverage[6] == {"direct": 500, "step": 500, "typed_seed_list": 500, "module": 500}
    assert coverage[8] == {"direct": 500, "step": 500, "query_mygene": 500, "module": 500}
    assert ambiguous_coverage == {(candidate, framing): 250 for candidate in (False, True) for framing in ("direct", "step")}
    assert unchanged == 3250
    assert failure_partners == 250
    for split in ("validation", "test"):
        assert (root / f"{split}.jsonl").read_bytes() == (BASELINE / f"{split}.jsonl").read_bytes()
    assert manifest["train_match_reasoning"] == REASONING
    assert manifest["train_mismatch_reasoning"] == MISMATCH_REASONING
    assert manifest["train_unresolved_symbol_reasoning"] == UNRESOLVED_REASONING
    assert manifest["train_ambiguous_candidate_reasoning"] == CANDIDATE_REASONING
    assert manifest["train_ambiguous_noncandidate_reasoning"] == NONCANDIDATE_REASONING
    assert manifest["train_mygene_match_reasoning"] == MYGENE_REASONING
    baseline_manifest = json.loads((BASELINE / "manifest.json").read_text())
    assert manifest["evidence"] == baseline_manifest["evidence"]


def test_repeated_build_is_byte_identical(revision, tmp_path):
    """Reproduce the same splits and manifest from the pinned inputs."""

    contract, payload, source, root, _ = revision
    build_corpus(
        source, tmp_path, contract=contract,
        overlay=overlay_pin(S1_V6_OVERLAY_PATH, payload),
        prompt_banks=payload["prompt_banks"],
    )
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize(("overlay_path", "baseline"), [
    (None, Path("data/world_model_v3/corpora/s1/s1_4")),
    (S1_V4_OVERLAY_PATH, Path("data/world_model_v4/corpora/s1_v4/s1_4")),
    (S1_V5_OVERLAY_PATH, BASELINE),
])
def test_legacy_builds_reproduce_frozen_splits(overlay_path, baseline, tmp_path):
    """Preserve the published no-overlay, v4, and v5 corpus bytes."""

    contract, payload = load_contract(), {}
    if overlay_path is not None:
        contract, payload = apply_s1_overlay(contract, overlay_path)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    build_corpus(
        source, tmp_path, contract=contract, prompt_banks=payload.get("prompt_banks"),
        overlay=overlay_pin(overlay_path, payload) if overlay_path is not None else None,
    )
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "extra", "symbol", "slot", "newline", "framing"])
@pytest.mark.parametrize("kind", ["match", "mismatch", "unresolved_symbol", "ambiguous_candidate", "ambiguous_noncandidate", "mygene_match"])
def test_overlay_rejects_invalid_templates(tmp_path, mutation, kind):
    """Require the declared binding counts and both reviewed framings."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.4"][f"train_{kind}_reasoning"]
    if mutation == "framing":
        reasoning["workflow"] = reasoning.pop("step")
    else:
        text = reasoning["direct"]
        id_slot = "{gene_id}" if kind in {"match", "mismatch", "mygene_match"} else "{target_id}"
        symbol_slot = "{input}" if kind == "unresolved_symbol" else "{symbol}"
        reasoning["direct"] = {
            "missing": text.replace(id_slot, "an ID", 1),
            "extra": text + " " + id_slot + ".",
            "symbol": text.replace(symbol_slot, "a symbol"),
            "slot": text.replace(symbol_slot, "{symbol.real}"),
            "newline": text + "\nAnother sentence.",
        }[mutation]
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=f"alignment-{kind.replace('_', '-')} reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("replacement", ["an ID", "{gene_id}", "{target_id} {target_id}"])
def test_mismatch_requires_two_seed_id_bindings(tmp_path, replacement):
    """Reject missing, swapped, or extra seed bindings before building."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.4"]["train_mismatch_reasoning"]
    reasoning["direct"] = reasoning["direct"].replace("{target_id}", replacement, 1)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="alignment-mismatch reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("replacement", ["the query", "{symbol}", "{query} {query}"])
def test_mygene_requires_two_original_query_bindings(tmp_path, replacement):
    """Reject missing, substituted, or extra bindings for the literal MyGene query."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.4"]["train_mygene_match_reasoning"]
    reasoning["direct"] = reasoning["direct"].replace("{query}", replacement, 1)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="alignment-mygene-match reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("kind", ["candidate", "noncandidate"])
@pytest.mark.parametrize("replacement", ["two", "{target_id}", "{count} {count}"])
def test_ambiguous_reasoning_requires_returned_count(tmp_path, kind, replacement):
    """Require one candidate-count slot rather than a fixed or unrelated value."""

    payload = yaml.safe_load(S1_V6_OVERLAY_PATH.read_text())
    reasoning = payload["family_overrides"]["S1.4"][f"train_ambiguous_{kind}_reasoning"]
    reasoning["direct"] = reasoning["direct"].replace("{count}", replacement)
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match=f"alignment-ambiguous-{kind} reasoning"):
        apply_s1_overlay(load_contract(), path)


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("mutation", ["thinking", "status", "decision", "alignment", "premature", "early_id", "count_output"])
def test_ambiguous_reasoning_keeps_binding_and_semantic_checks(revision, candidate, mutation):
    """Reject unbound evidence and contradictory conclusions in both S1.4.6 branches."""

    row = next(
        r for r in map(json.loads, (revision[3] / "train.jsonl").open())
        if r["metadata"]["form"] == 6
        and (r["messages"][-1]["content"]["result"]["same_gene"] is None) == candidate
    )
    output = next(m["content"] for m in row["messages"] if m["role"] == "tool")
    final = row["messages"][-1]["content"]
    module = next(final["result"][side] for side in ("first", "second") if final["result"][side]["item"] == "module")
    target = GeneFact("module", ("test",), None, module["gene_id"])
    fact = AmbiguousFact("test", ("test",), output["gene_symbol"], tuple(output["candidate_gene_ids"]), output["candidate_details"])
    prompts = wrapper_prompts(revision[1]["prompt_banks"], ALIGNMENT_PROMPTS, bank_name="alignment")
    scenario = Scenario("train", 6, fact, target, _training_framing(row, prompts), 0, 0, ambiguous_target_is_candidate=candidate)
    options = {
        "ambiguous_candidate_reasoning": CANDIDATE_REASONING,
        "ambiguous_noncandidate_reasoning": NONCANDIDATE_REASONING,
    }
    assert filter_row_thinking(row, scenario, **options) == 2
    if mutation == "thinking":
        row["messages"][-1]["thinking"] += " Extra text."
        error = "unbound ambiguous-comparison reasoning"
    elif mutation == "count_output":
        output["candidate_gene_ids"] = output["candidate_gene_ids"][:-1]
        error = "unbound ambiguous-comparison reasoning"
    elif mutation == "status":
        final["status"] = "resolved" if candidate else "ambiguous"
        error = "wrong status"
    elif mutation == "decision":
        final["continuation_decision"] = "stop" if candidate else "defer"
        error = "wrong decision"
    elif mutation == "alignment":
        final["result"]["same_gene"] = False if candidate else None
        error = "wrong alignment claim"
    else:
        call = next(m for m in row["messages"] if "tool_call" in m)
        if mutation == "early_id":
            visible = " ".join(m["content"] for m in row["messages"] if m["role"] in {"user", "harness"})
            hidden_id = next(gene_id for gene_id in fact.candidate_gene_ids if gene_id not in visible)
            call["thinking"] += f" The ID is {hidden_id}."
            error = "reveals .* too early"
        else:
            call["thinking"] += " The two mentions name the same gene."
            error = "predicts the lookup outcome"
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, **options)


@pytest.mark.parametrize(("form", "mutation"), [
    (form, mutation)
    for form in (1, 2, 5, 8)
    for mutation in ("thinking", "status", "decision", "premature", "schema", "alignment", "early_id")
    if mutation != "early_id" or form == 2
])
def test_thinking_keeps_binding_and_semantic_checks(revision, mutation, form):
    """Reject altered templates, conflicting answers, and invalid pre-call thoughts."""

    row = next(
        r for r in map(json.loads, (revision[3] / "train.jsonl").open())
        if r["metadata"]["form"] == form
    )
    output = next(m["content"] for m in row["messages"] if m["role"] == "tool")
    prompts = wrapper_prompts(revision[1]["prompt_banks"], ALIGNMENT_PROMPTS, bank_name="alignment")
    framing = _training_framing(row, prompts)
    if form == 5:
        result = row["messages"][-1]["content"]["result"]
        module = next(result[side] for side in ("first", "second") if result[side]["item"] == "module")
        target = GeneFact("module", ("test",), None, module["gene_id"])
        fact = ExternalFact("test", ("test",), output["gene_symbol"], None, target.gene_id)
    else:
        fact = GeneFact("test", ("test",), output["gene_symbol"], output["gene_id"])
        target = fact
    if form == 2:
        result = row["messages"][-1]["content"]["result"]
        seed = next(result[side] for side in ("first", "second") if result[side]["item"] == "seed_gene_set")
        target = GeneFact("seed", ("seed",), None, seed["gene_id"])
    if form == 8:
        result = row["messages"][-1]["content"]["result"]
        external = next(result[side] for side in ("first", "second") if result[side]["item"] == "query_mygene")
        fact = ExternalFact("test", ("test",), external["input"], output["gene_symbol"], output["gene_id"])
    scenario = Scenario("train", form, fact, target, framing, 0, 0)
    options = {
        "match_reasoning": REASONING, "mismatch_reasoning": MISMATCH_REASONING,
        "unresolved_symbol_reasoning": UNRESOLVED_REASONING,
        "mygene_match_reasoning": MYGENE_REASONING,
    }
    assert filter_row_thinking(row, scenario, **options) == 2
    if mutation == "thinking":
        row["messages"][-1]["thinking"] += " Extra text."
        error = "unbound .* reasoning"
    elif mutation in {"status", "decision"}:
        key, value = ("status", "ambiguous") if mutation == "status" else (
            "continuation_decision", "stop" if form == 5 else "defer",
        )
        row["messages"][-1]["content"][key] = value
        error = "wrong " + mutation
    elif mutation == "alignment":
        row["messages"][-1]["content"]["result"]["same_gene"] = form == 2
        error = "wrong alignment claim"
    else:
        call = next(m for m in row["messages"] if "tool_call" in m)
        if mutation == "early_id":
            call["thinking"] += f" The ID is {fact.gene_id}."
            error = "reveals .* too early"
        else:
            call["thinking"] += " The two mentions name the same gene." if mutation == "premature" else " Use the answer schema."
            error = "predicts the lookup outcome" if mutation == "premature" else "discusses the answer schema"
    with pytest.raises(CorpusBuildError, match=error):
        filter_row_thinking(row, scenario, **options)
