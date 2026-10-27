"""Test the minimal adapter from canonical v3 rows to S0 training/eval."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_world_model_v2_s0_exact_generation as generation_module

from runtime.world_model.contract import (
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from runtime.world_model_training import (
    S0_V4_DATASET_ID,
    flatten_v3_trace_for_arrow,
    renderable_v3_tools,
    tokenize_v3_trace_for_sft,
)
from scripts.build_world_model_s0_validation_panels import project_row
from scripts.build_world_model_v2_s0_generation_bundle import (
    FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION,
    FULL_REGISTRY_V3_EVALUATION_CONTRACT,
)
from scripts.run_world_model_v2_s0_exact_generation import (
    ALLOWED_SHARED_FINAL_SPECIAL_TOKENS,
    clean_decoded_generation,
    final_prompt_groups,
)
from scripts.evaluate_world_model_v2_s0 import (
    WorldModelContractError,
    parse_v3_answer,
    score_tool_trajectory_record,
)


class _Tokenizer:
    def __init__(self):
        self.rendered_messages = []

    def apply_chat_template(self, messages, **options):
        assert all("description" in tool["function"] for tool in options["tools"])
        self.rendered_messages.append(
            json.loads(json.dumps(messages, ensure_ascii=False))
        )
        result = [1]
        for index, message in enumerate(messages):
            if message["role"] == "system":
                result.append(10)
            elif message["role"] == "user":
                result.append(20)
            elif message["role"] == "tool":
                result.append(50)
            elif "tool_calls" in message:
                has_final = any(
                    future["role"] == "assistant" and "tool_calls" not in future
                    for future in messages[index + 1 :]
                )
                if message.get("thinking") and not has_final:
                    result.append(30)
                result.append(40)
            else:
                if message.get("thinking"):
                    result.append(60)
                result.append(61)
        return result


class _PriorityTokenizer(_Tokenizer):
    """Expose stable character offsets for priority-loss mask tests."""

    @staticmethod
    def _text_ids(text):
        return [1000 + ord(character) for character in text]

    def __call__(self, text, **options):
        assert options == {
            "add_special_tokens": False,
            "return_offsets_mapping": True,
        }
        return {
            "input_ids": self._text_ids(text),
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    def apply_chat_template(self, messages, **options):
        assert all("description" in tool["function"] for tool in options["tools"])
        result = [1]
        for index, message in enumerate(messages):
            if message["role"] == "system":
                result.append(10)
            elif message["role"] == "user":
                result.append(20)
            elif message["role"] == "tool":
                result.append(50)
            elif "tool_calls" in message:
                has_final = any(
                    future["role"] == "assistant"
                    and "tool_calls" not in future
                    for future in messages[index + 1 :]
                )
                if message.get("thinking") and not has_final:
                    result.append(30)
                result.extend([40, 41])
            else:
                if message.get("thinking"):
                    result.append(60)
                result.append(61)
                result.extend(self._text_ids(message["content"]))
                result.append(62)
        return result


def _row():
    contract = load_contract()
    provenance = {
        "registry": {
            "registry": "ensembl",
            "release": "116",
            "evidence_id": "ev_reg_abc123",
        },
        "graph": None,
    }
    return {
        "row_id": "wm3_test_row",
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {"role": "user", "content": "Resolve TP53."},
            {
                "role": "assistant",
                "thinking": "I need the registry mapping.",
                "tool_call": {
                    "name": "lookup_human_gene_symbol",
                    "arguments": {"gene_symbol": "TP53"},
                },
            },
            {
                "role": "tool",
                "name": "lookup_human_gene_symbol",
                "content": {
                    "status": "resolved",
                    "provenance": provenance,
                    "gene_id": "ENSG00000141510",
                    "gene_symbol": "TP53",
                },
            },
            {
                "role": "assistant",
                "thinking": "The mapping is unique.",
                "content": {
                    "status": "resolved",
                    "provenance": provenance,
                    "result": {
                        "gene_id": "ENSG00000141510",
                        "gene_symbol": "TP53",
                    },
                    "continuation_decision": "stop",
                    "blocked_reason": None,
                    "summary": "TP53 resolves to ENSG00000141510.",
                },
            },
        ],
        "tools": tool_definitions(contract),
        "metadata": {
            "book_mode": "tool_call",
            "step": "S0",
            "question_family": "human_symbol_to_ensembl",
            "form": 1,
            "source_trace": "S0.1",
            "split_unit": "group_test",
            "blocked_reason_code": None,
            "schema_version": "multiplex_sft_v3",
            "system_prompt_sha256": system_prompt_sha256(contract),
            "tool_contract_version": contract["runtime"]["tool_contract_version"],
            "ensembl_release": "116",
            "identifier_registry_id": "ensembl_homo_sapiens_116_v3",
        },
    }


def _s1_row(messages):
    row = _row()
    row["messages"] = messages
    row["metadata"].update(
        {
            "step": "S1",
            "question_family": "human_ambiguous_symbol",
            "source_trace": "S1.3.test",
        }
    )
    return row


def _no_call_harness_row():
    source = _row()
    return _s1_row(
        [
            source["messages"][0],
            {
                "role": "harness",
                "content": "Registry entries, Ensembl release 116:\n{}",
            },
            {"role": "user", "content": "Resolve PPP2R3B."},
            source["messages"][-1],
        ]
    )


def _two_call_row():
    source = _row()
    return _s1_row(
        [
            source["messages"][0],
            {"role": "user", "content": "Resolve PPP2R3B in mux."},
            source["messages"][2],
            source["messages"][3],
            {
                "role": "assistant",
                "thinking": "The symbol is ambiguous. I will check the multiplex.",
                "tool_call": {
                    "name": "check_multiplex_membership",
                    "arguments": {
                        "multiplex_id": "mux",
                        "gene_ids": ["ENSG00000167393", "ENSG00000292327"],
                    },
                },
            },
            {
                "role": "tool",
                "name": "check_multiplex_membership",
                "content": {
                    "status": "resolved",
                    "multiplex_id": "mux",
                    "present_gene_ids": ["ENSG00000167393"],
                },
            },
            source["messages"][-1],
        ]
    )


def _multi_turn_row():
    source = _row()
    earlier_answer = {
        "role": "assistant",
        "content": source["messages"][-1]["content"],
    }
    return _s1_row(
        [
            source["messages"][0],
            {"role": "user", "content": "Resolve PPP2R3B."},
            earlier_answer,
            {"role": "user", "content": "Use ENSG00000167393."},
            source["messages"][-1],
        ]
    )


def _s1_4_row():
    row = _row()
    row["metadata"].update(
        {
            "step": "S1",
            "question_family": "cross_context_alignment",
            "form": 8,
            "source_trace": "S1.4.8",
        }
    )
    row["messages"].insert(
        1,
        {
            "role": "harness",
            "content": (
                "query_mygene result:\n"
                '{"query":"HER2","results":[{"ensembl":'
                '{"gene":"ENSG00000141736"},"symbol":"ERBB2"}]}'
            ),
        },
    )
    return row


def test_v3_training_adapter_preserves_identity_and_masks_nonassistant_turns():
    row = _row()
    flattened = flatten_v3_trace_for_arrow(row)
    assert flattened["record_id"] == row["row_id"]
    assert flattened["fact_id"] == "group_test"
    flattened_row = json.loads(flattened["trace_json"])
    assert flattened_row == row
    expected_answer_keys = load_contract()["answer"]["keys"]
    assert list(flattened_row["messages"][-1]["content"]) == (
        expected_answer_keys
    )

    tokenizer = _Tokenizer()
    tokenized = tokenize_v3_trace_for_sft(flattened_row, tokenizer, 32)
    assert tokenized["input_ids"] == [1, 10, 20, 30, 40, 50, 60, 61]
    assert tokenized["completion_mask"] == [0, 0, 0, 1, 1, 0, 1, 1]
    assert tokenizer.rendered_messages[-1][-1]["content"] == json.dumps(
        row["messages"][-1]["content"],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_v3_priority_loss_weights_structural_tokens_and_normalizes_each_row():
    row = _row()
    tokenized = tokenize_v3_trace_for_sft(
        row,
        _PriorityTokenizer(),
        512,
        priority_weights=True,
    )

    mask = tokenized["completion_mask"]
    weights = tokenized["loss_weights"]
    assert len(weights) == len(tokenized["input_ids"])
    assert all(weight == 0.0 for weight, active in zip(weights, mask) if not active)
    positive = [weight for weight, active in zip(weights, mask) if active]
    assert abs(sum(positive) / len(positive) - 1.0) < 1e-9

    unit = min(positive)
    raw_levels = {
        round(weight / unit) for weight in positive
    }
    assert raw_levels == {1, 4, 8}
    call_index = tokenized["input_ids"].index(40)
    thinking_index = tokenized["input_ids"].index(30)
    assert abs(weights[call_index] / weights[thinking_index] - 4.0) < 1e-9
    second_call_index = tokenized["input_ids"].index(41)
    assert abs(weights[second_call_index] / weights[thinking_index] - 4.0) < 1e-9

    final_json = json.dumps(
        row["messages"][-1]["content"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    json_ids = _PriorityTokenizer._text_ids(final_json)
    input_ids = tokenized["input_ids"]
    matches = [
        index
        for index in range(len(input_ids) - len(json_ids) + 1)
        if input_ids[index : index + len(json_ids)] == json_ids
    ]
    assert len(matches) == 1
    json_weights = weights[matches[0] : matches[0] + len(json_ids)]
    assert min(json_weights) >= 4.0 * unit - 1e-9
    assert max(json_weights) == 8.0 * unit
    final_payload = row["messages"][-1]["content"]
    critical_fields = {
        "status",
        "result",
        "continuation_decision",
        "blocked_reason",
    }
    for key, value in final_payload.items():
        rendered_field = (
            json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            + ":"
            + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        field_start = final_json.index(rendered_field)
        field_weights = json_weights[
            field_start : field_start + len(rendered_field)
        ]
        expected_multiplier = 8.0 if key in critical_fields else 4.0
        assert all(
            abs(weight / unit - expected_multiplier) < 1e-9
            for weight in field_weights
        )

    final_json_start = matches[0]
    assert abs(weights[final_json_start - 1] / unit - 1.0) < 1e-9
    assert abs(weights[final_json_start + len(json_ids)] / unit - 1.0) < 1e-9


def test_v3_training_adapter_rejects_misordered_final_answer() -> None:
    row = _row()
    answer = row["messages"][-1]["content"]
    row["messages"][-1]["content"] = {
        key: answer[key] for key in sorted(answer)
    }

    for operation in (
        lambda: flatten_v3_trace_for_arrow(row),
        lambda: tokenize_v3_trace_for_sft(row, _Tokenizer(), 32),
    ):
        try:
            operation()
        except ValueError as error:
            assert str(error) == "The v3 final answer field order changed"
        else:
            raise AssertionError("The v3 adapter accepted misordered fields")


def test_v3_validator_reads_order_from_the_contract() -> None:
    answer = _row()["messages"][-1]["content"]
    encoded = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))

    assert parse_v3_answer(encoded).payload == answer

    alphabetized = json.dumps(
        answer,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        parse_v3_answer(alphabetized)
    except WorldModelContractError as error:
        assert error.code == "answer_field_order_invalid"
    else:
        raise AssertionError("The v3 validator accepted alphabetized fields")


def test_v3_validator_rejects_duplicate_keys_and_nonstandard_constants() -> None:
    answer = _row()["messages"][-1]["content"]
    encoded = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    duplicate = encoded.replace(
        '"status":"resolved"',
        '"status":"resolved","status":"resolved"',
        1,
    )
    nonstandard = encoded.replace('"status":"resolved"', '"status":NaN', 1)

    for invalid, expected_code in (
        (duplicate, "answer_duplicate_key"),
        (nonstandard, "answer_json_constant"),
    ):
        try:
            parse_v3_answer(invalid)
        except WorldModelContractError as error:
            assert error.code == expected_code
        else:
            raise AssertionError(f"The v3 validator accepted {expected_code}")


def test_v3_validator_enforces_contract_values_and_types() -> None:
    answer = _row()["messages"][-1]["content"]
    cases = (
        ({**answer, "status": "unknown"}, "answer_status_invalid"),
        ({**answer, "provenance": []}, "answer_provenance_invalid"),
        ({**answer, "result": []}, "answer_result_invalid"),
        (
            {**answer, "continuation_decision": "wait"},
            "answer_continuation_decision_invalid",
        ),
        ({**answer, "blocked_reason": "blocked"}, "answer_blocked_reason_invalid"),
        ({**answer, "summary": ""}, "answer_summary_invalid"),
    )

    for invalid, expected_code in cases:
        encoded = json.dumps(invalid, ensure_ascii=False, separators=(",", ":"))
        try:
            parse_v3_answer(encoded)
        except WorldModelContractError as error:
            assert error.code == expected_code
        else:
            raise AssertionError(f"The v3 validator accepted {expected_code}")


def test_v3_final_generation_allows_required_channel_framing() -> None:
    raw = (
        "<|channel|>analysis<|message|>Use the tool result.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '{"status":"resolved"}<|return|>'
    )

    cleaned, disallowed = clean_decoded_generation(
        raw,
        eos_token="<|return|>",
        pad_token="<|end|>",
        allowed_special_tokens=ALLOWED_SHARED_FINAL_SPECIAL_TOKENS,
    )

    assert disallowed == []
    assert cleaned.endswith('{"status":"resolved"}')

    _, disallowed_call = clean_decoded_generation(
        raw.replace("<|return|>", "<|call|><|return|>"),
        eos_token="<|return|>",
        pad_token="<|end|>",
        allowed_special_tokens=ALLOWED_SHARED_FINAL_SPECIAL_TOKENS,
    )
    assert disallowed_call == ["<|call|>"]


def test_v3_score_path_accepts_one_canonical_complete_trajectory() -> None:
    row = _row()
    call_message = row["messages"][2]
    tool_message = row["messages"][3]
    final_message = row["messages"][4]
    call = call_message["tool_call"]
    tool_payload = tool_message["content"]
    final_answer = json.dumps(
        final_message["content"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded_call = (
        "<|channel|>analysis<|message|>"
        f"{call_message['thinking']}<|end|><|start|>assistant "
        f"to=functions.{call['name']}"
        "<|channel|>commentary json<|message|>"
        f"{json.dumps(call['arguments'], separators=(',', ':'))}<|call|>"
    )
    encoded_final = (
        "<|channel|>analysis<|message|>"
        f"{final_message['thinking']}<|end|><|start|>assistant"
        "<|channel|>final<|message|>"
        f"{final_answer}"
    )

    class _Result:
        output = tool_payload

    class _Runtime:
        def execute(self, name, arguments):
            assert name == call["name"]
            assert arguments == call["arguments"]
            return _Result()

    scored = score_tool_trajectory_record(
        {
            "record_id": row["row_id"],
            "metadata": {"schema_version": "multiplex_sft_v3"},
        },
        {
            "record_id": row["row_id"],
            "fact_id": row["metadata"]["split_unit"],
            "family": row["metadata"]["question_family"],
            "expected_status": final_message["content"]["status"],
            "expected_continuation_decision": final_message["content"][
                "continuation_decision"
            ],
            "expected_tool_call": call,
            "expected_tool_payload": tool_payload,
            "expected_assistant_thinking": call_message["thinking"],
            "expected_final_explanation": final_message["thinking"],
            "expected_final_answer": final_answer,
        },
        {
            "record_id": row["row_id"],
            "encoded_prediction": encoded_call,
            "assistant_thinking": call_message["thinking"],
            "tool_payload": tool_payload,
            "tool_error": None,
            "network_used_false": True,
            "disallowed_special_tokens": [],
            "encoded_final_answer": encoded_final,
            "assistant_final_explanation": final_message["thinking"],
            "assistant_final": final_answer,
            "final_disallowed_special_tokens": [],
        },
        runtime=_Runtime(),
    )

    assert scored["structured_json_valid"] is True
    assert scored["valid_tool_trajectory"] is True
    assert scored["final_answer_exact"] is True


def test_v3_training_adapter_merges_harness_and_supports_no_call():
    row = _no_call_harness_row()
    flattened = flatten_v3_trace_for_arrow(row)
    assert flattened["question_family"] == "human_ambiguous_symbol"

    tokenizer = _Tokenizer()
    tokenized = tokenize_v3_trace_for_sft(row, tokenizer, 32)
    assert tokenized["input_ids"] == [1, 10, 20, 60, 61]
    assert tokenized["completion_mask"] == [0, 0, 0, 1, 1]
    assert all(
        message["role"] != "harness"
        for rendered in tokenizer.rendered_messages
        for message in rendered
    )
    assert tokenizer.rendered_messages[0][1]["content"] == (
        "Context (from the harness):\n"
        "Registry entries, Ensembl release 116:\n{}\n\n"
        "User request:\nResolve PPP2R3B."
    )


def test_v3_training_adapter_trains_both_calls_and_post_tool_thinking():
    row = _two_call_row()
    assert flatten_v3_trace_for_arrow(row)["record_id"] == row["row_id"]

    tokenized = tokenize_v3_trace_for_sft(row, _Tokenizer(), 32)
    assert tokenized["input_ids"] == [
        1,
        10,
        20,
        30,
        40,
        50,
        30,
        40,
        50,
        60,
        61,
    ]
    assert tokenized["completion_mask"] == [
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
    ]


def test_v3_training_adapter_masks_the_earlier_turn():
    row = _multi_turn_row()
    assert flatten_v3_trace_for_arrow(row)["record_id"] == row["row_id"]

    tokenized = tokenize_v3_trace_for_sft(row, _Tokenizer(), 32)
    assert tokenized["input_ids"] == [1, 10, 20, 61, 20, 60, 61]
    assert tokenized["completion_mask"] == [0, 0, 0, 0, 0, 1, 1]


def test_s0_training_adapter_rejects_s1_families():
    """The S0 trainer accepts only the three S0 question families."""

    row = _s1_4_row()

    for dataset_id in (S0_V4_DATASET_ID,):
        try:
            flatten_v3_trace_for_arrow(row, dataset_id=dataset_id)
        except ValueError as error:
            assert str(error) == "The v3 row has an unsupported question family"
        else:
            raise AssertionError("The S0 adapter accepted an S1-only family")
    assert flatten_v3_trace_for_arrow(
        _s1_4_row(), dataset_id="world_model_v3_s1_coordinate_system"
    )["question_family"] == "cross_context_alignment"


def test_v3_eval_projection_separates_the_answer():
    question, answer = project_row(_row(), "test")
    assert "messages" not in question
    assert question["input"] == {"gene_symbol": "TP53"}
    assert answer["expected_tool_call"]["name"] == "lookup_human_gene_symbol"
    assert json.loads(answer["expected_final_answer"])["status"] == "resolved"


def test_large_final_prompts_are_generated_as_singletons():
    prompts = ["short-a", "x" * 9000, "short-b", "short-c"]

    assert final_prompt_groups(prompts, batch_size=2) == [
        [0],
        [1],
        [2, 3],
    ]


def test_v3_render_tools_adds_only_tokenizer_descriptions():
    original = _row()["tools"]
    rendered = renderable_v3_tools(original)
    assert all("description" not in tool["function"] for tool in original)
    assert all("description" in tool["function"] for tool in rendered)


def test_v3_generation_bundle_uses_the_source_index_content_hash(
    tmp_path: Path, monkeypatch
):
    """Do not compare a compressed source index to its logical digest."""

    rows = [
        {
            "input": {"gene_symbol": "TP53"},
            "metadata": {},
            "provenance": {},
            "question": "Resolve TP53.",
            "record_id": "record-01",
            "split": "test",
            "system": "Use the tool.",
            "tools": [],
        }
    ]
    questions_path = tmp_path / "questions.jsonl"
    generation_module.write_jsonl(questions_path, rows)
    registry_sha256 = "a" * 64
    manifest = {
        "schema_version": FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION,
        "evaluation_contract": FULL_REGISTRY_V3_EVALUATION_CONTRACT,
        "record_count": 1,
        "record_ids_sha256": generation_module.stable_sha256(
            ["record-01"]
        ),
        "questions_sha256": generation_module.sha256_file(questions_path),
        "reads_private_answer_keys": False,
        "identifier_registry": {
            "path": "runtime/world_model/contract.yaml",
            "id": f"sha256:{registry_sha256}",
            "sha256": registry_sha256,
        },
    }
    manifest["bundle_sha256"] = generation_module.stable_sha256(manifest)
    generation_module.write_json(tmp_path / "manifest.json", manifest)

    source_index = {"logical": "source index"}
    monkeypatch.setattr(
        generation_module,
        "check_source_index",
        lambda path, contract: source_index,
    )
    monkeypatch.setattr(
        generation_module,
        "content_sha256",
        lambda payload: registry_sha256,
    )

    loaded_manifest, loaded_rows = generation_module.load_bundle(tmp_path)

    assert loaded_manifest["bundle_sha256"] == manifest["bundle_sha256"]
    assert loaded_rows == rows
