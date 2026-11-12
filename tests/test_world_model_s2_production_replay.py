"""Check distinct replay selection, pinned sources, and the actual training wire."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s2_production_replay import (
    ADAPTER_VERSION, LEGACY_INSTRUCTION, S1_COMPONENTS, adapt_replay_row,
    capped_equal_quotas, load_replay_pool, select_replay_rows, validate_replay_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


def legacy_row(number=0, stage="S1", trace="S1.1.1", framing="direct"):
    """Make a canonical legacy lookup retaining registry and null graph evidence."""
    contract = load_contract()
    prompt = contract["runtime"]["system_prompt"]
    evidence = {"registry": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_000000"},
                "graph": None}
    answer = {"status": "resolved", "provenance": copy.deepcopy(evidence),
              "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"},
              "continuation_decision": "stop" if framing == "direct" else "continue",
              "blocked_reason": None, "summary": "TP53 is <GENE:ENSG00000141510|TP53>."}
    return {"row_id": f"{stage.lower()}_source_{trace}_{number}", "messages": [
        {"role": "system", "content": prompt},
        {"role": "harness", "content": json.dumps({"unrelated": "leave this context unchanged"})},
        {"role": "user", "content": "Resolve TP53."},
        {"role": "assistant", "thinking": "I need the authoritative human symbol mapping.",
         "tool_call": {"name": "lookup_human_gene_symbol", "arguments": {"gene_symbol": "TP53"}}},
        {"role": "tool", "name": "lookup_human_gene_symbol", "content": {
            "status": "resolved", "provenance": copy.deepcopy(evidence),
            "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}},
        {"role": "assistant", "thinking": "The returned canonical ID settles this lookup.", "content": answer}],
        "tools": tool_definitions(contract), "metadata": {
            "step": stage, "question_family": "human_symbol_to_ensembl", "source_trace": trace,
            "form": int(trace.split(".")[-1]), "framing": framing, "book_mode": "tool_call",
            "split_unit": f"unit_{number}", "blocked_reason_code": None,
            "schema_version": contract["runtime"]["row_schema_version"],
            "tool_contract_version": contract["runtime"]["tool_contract_version"],
            "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}}


@pytest.mark.parametrize("capacities,count,expected", [
    ({"a": 2, "b": 20, "c": 20}, 14, {"a": 2, "b": 6, "c": 6}),
    ({"a": 2, "b": 20, "c": 20}, 15, {"a": 2, "b": 7, "c": 6}),
    ({"a": 0, "b": 2}, 2, {"a": 0, "b": 2}),
    ({}, 0, {}),
])
def test_capped_equal_allocation(capacities, count, expected):
    assert capped_equal_quotas(capacities, count) == expected


@pytest.mark.parametrize("capacities,count", [({"a": 2}, 3), ({"a": -1}, 0),
                                                       ({"a": 2}, -1), ({"a": 2}, True)])
def test_unsupported_quota_fails(capacities, count):
    with pytest.raises(ValueError):
        capped_equal_quotas(capacities, count)


def test_legacy_adapter_preserves_every_tool_context_reasoning_and_answer():
    row = legacy_row()
    original = copy.deepcopy(row)
    adapted = adapt_replay_row(row, "S1")
    assert row == original
    assert list(adapted) == ["row_id", "messages", "tools", "metadata"]
    assert adapted["messages"][0]["content"] == LEGACY_INSTRUCTION + original["messages"][0]["content"]
    assert adapted["messages"][1:] == original["messages"][1:]
    assert adapted["messages"][-1]["content"]["provenance"]["graph"] is None
    assert adapted["metadata"]["source_record_id"] == original["row_id"]
    assert adapted["metadata"]["trace_family"] == original["metadata"]["question_family"]
    assert adapted["metadata"]["contract_adapter"] == ADAPTER_VERSION
    for old, new in zip(original["tools"], adapted["tools"]):
        assert old["function"]["name"] == new["function"]["name"]
        assert old["function"]["parameters"] == new["function"]["parameters"]
        assert isinstance(new["function"]["description"], str)
    validate_replay_adapter(adapted, "S1")
    assert adapt_replay_row(adapted, "S1") == adapted


def test_adapter_does_not_reinterpret_unknown_multiplex_or_partial_results():
    row = legacy_row()
    graph = {"multiplex_id": "unknown_legacy_view", "graph_version": "opaque_old_hash",
             "layer_scope": "all", "evidence_id": "ev_mem_000000"}
    answer = row["messages"][-1]["content"]
    answer.update(status="not_found", continuation_decision="defer", blocked_reason="The view is unknown.")
    answer["provenance"] = {"registry": None, "graph": graph}
    answer["result"] = {"partial_results": [{"status": "resolved", "provenance": copy.deepcopy(answer["provenance"])}]}
    adapted = adapt_replay_row(row, "S1")
    assert adapted["messages"][-1]["content"] == answer


def test_overlapping_legacy_evidence_ids_have_distinct_source_trace_namespaces():
    first, second = [adapt_replay_row(legacy_row(number), "S1") for number in (1, 2)]
    assert first["messages"][-1]["content"]["provenance"] == second["messages"][-1]["content"]["provenance"]
    assert first["metadata"]["evidence_namespace"] == {
        "stage": "S1", "source_record_id": first["metadata"]["source_record_id"]}
    assert first["metadata"]["evidence_namespace"] != second["metadata"]["evidence_namespace"]
    assert "Legacy acquisition evidence IDs are local to this source trace" in first["messages"][0]["content"]
    first["metadata"]["evidence_namespace"] = second["metadata"]["evidence_namespace"]
    with pytest.raises(ValueError, match="adapter metadata"):
        validate_replay_adapter(first, "S1")


def test_wrong_stage_and_forged_legacy_declaration_fail():
    with pytest.raises(ValueError, match="source stage"):
        adapt_replay_row(legacy_row(), "S0")
    row = adapt_replay_row(legacy_row(), "S1")
    row["messages"][0]["content"] = "Use S2 provenance."
    with pytest.raises(ValueError, match="visible contract"):
        adapt_replay_row(row, "S1")


def test_selection_is_order_independent_balanced_and_distinct():
    rows = [legacy_row(i, trace=trace, framing="direct" if i % 2 else "step")
            for trace, capacity in (("S1.1.1", 2), ("S1.1.2", 20), ("S1.1.3", 20))
            for i in range(capacity)]
    selected, receipt = select_replay_rows(rows, 14, stage="S1", seed=99)
    repeated, repeat_receipt = select_replay_rows(reversed(rows), 14, stage="S1", seed=99)
    assert selected == repeated and receipt == repeat_receipt
    assert receipt["selected_by_type"] == {"S1.1.1": 2, "S1.1.2": 6, "S1.1.3": 6}
    assert len({row["metadata"]["source_record_id"] for row in selected}) == 14
    assert receipt["continuation_counts"] == {"stop": 7, "continue": 7}
    assert select_replay_rows(rows, 14, stage="S1", seed=100)[0] != selected
    assert select_replay_rows([adapt_replay_row(row, "S1") for row in rows], 14, stage="S1", seed=99)[0] == selected


def test_duplicate_replay_row_fails():
    row = legacy_row()
    with pytest.raises(ValueError, match="duplicate source"):
        select_replay_rows([row, row], 1, stage="S1")


def pinned_pool(root, stage="S0"):
    """Write small source manifests with the same published pin relationships."""
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    contract_path = root / "runtime/world_model/contract.yaml"
    contract_hash = write(contract_path, "fixture pinned legacy contract")
    contract = {"path": str(contract_path.relative_to(root)), "sha256": contract_hash}
    components = {}
    for name in (("s0",) if stage == "S0" else S1_COMPONENTS):
        folder = root / "data/world_model_v4/corpora" / ("s0" if name == "s0" else f"s1_v10/{name}")
        trace = "S0.1" if name == "s0" else name.upper().replace("_", ".") + ".1"
        row = legacy_row(stage=stage, trace=trace)
        train_hash = write(folder / "train.jsonl", json.dumps(row) + "\n")
        manifest = {"contract": contract, "files": {"train.jsonl": {"row_count": 1, "sha256": train_hash}}}
        manifest_hash = write(folder / "manifest.json", manifest)
        components[name] = {"manifest_path": str((folder / "manifest.json").relative_to(root)),
                            "manifest_sha256": manifest_hash}
    aggregate = {"components": components, "s0_replay": components.get("s0", {})}
    aggregate_root = root / "data/world_model_v4/corpora/s1_v10"
    aggregate_hash = write(aggregate_root / "manifest.json", aggregate)
    write(root / "config/world_model_s1_120b_training_v10_32n.json", {"corpus": {
        "root": str(aggregate_root.relative_to(root)), "manifest_sha256": aggregate_hash}})
    return contract_path


@pytest.mark.parametrize("stage,expected", [("S0", 1), ("S1", 8)])
def test_source_pool_follows_pins_and_reads_only_component_train_files(tmp_path, stage, expected):
    pinned_pool(tmp_path, stage)
    rows, receipt = load_replay_pool(tmp_path, stage)
    assert len(rows) == receipt["rows"] == expected
    assert all(row["metadata"]["step"] == stage for row in rows)
    assert receipt["training_only"]
    assert receipt["embedded_s0_replay_excluded"] == (stage == "S1")
    assert len(receipt["sources"]) == expected


@pytest.mark.parametrize("tamper", ["manifest", "train", "contract"])
def test_changed_source_pins_fail(tmp_path, tamper):
    contract = pinned_pool(tmp_path)
    target = {"manifest": tmp_path / "data/world_model_v4/corpora/s0/manifest.json",
              "train": tmp_path / "data/world_model_v4/corpora/s0/train.jsonl", "contract": contract}[tamper]
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="pin"):
        load_replay_pool(tmp_path, "S0")


def test_component_can_inherit_the_aggregate_contract_pin(tmp_path):
    contract_path = pinned_pool(tmp_path, "S1")
    aggregate_path = tmp_path / "data/world_model_v4/corpora/s1_v10/manifest.json"
    aggregate = json.loads(aggregate_path.read_text())
    component_path = tmp_path / aggregate["components"]["s1_6"]["manifest_path"]
    component = json.loads(component_path.read_text())
    aggregate["contract"] = component.pop("contract")
    component_path.write_text(json.dumps(component))
    aggregate["components"]["s1_6"]["manifest_sha256"] = hashlib.sha256(component_path.read_bytes()).hexdigest()
    aggregate_path.write_text(json.dumps(aggregate))
    config_path = tmp_path / "config/world_model_s1_120b_training_v10_32n.json"
    config = json.loads(config_path.read_text())
    config["corpus"]["manifest_sha256"] = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    assert contract_path.exists()
    assert len(load_replay_pool(tmp_path, "S1")[0]) == 8


def test_actual_s2_tokenizer_retains_complete_legacy_trace_and_supervision():
    from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace

    pin = json.loads((ROOT / "runtime/world_model/s2_13_v1.json").read_text())["tokenizer"]
    if not (ROOT / pin["path"]).exists():
        pytest.skip("the pinned production tokenizer is unavailable")
    tokenizer = load_tokenizer(pin)
    row = adapt_replay_row(legacy_row(), "S1")
    encoded = tokenize_trace(row, tokenizer)
    assert 0 < sum(encoded["completion_mask"]) < len(encoded["input_ids"]) < 8192
    assert len(encoded["input_ids"]) == len(encoded["completion_mask"])
    decoded = tokenizer.decode(encoded["input_ids"])
    assert "legacy S0/S1" in decoded and "ev_reg_000000" in decoded
    supervised = tokenizer.decode([token for token, flag in zip(encoded["input_ids"], encoded["completion_mask"]) if flag])
    assert "authoritative human symbol mapping" in supervised
    assert "returned canonical ID" in supervised
    assert '"graph":null' in supervised
