"""Verify loss telemetry against actual tokens without changing SFT supervision."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from runtime.world_model_training import (
    LOSS_SEGMENT_NAMES,
    LOSS_SEGMENT_SCHEMA_VERSION,
    tokenize_v3_trace_for_sft,
    v3_assistant_loss_segments,
)
from test_world_model_s0_v3_training import _row


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(os.environ.get("S1_AUDIT_SOURCE_ROOT", ROOT.parent / "mentor-rl"))


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer pinned by the completed S1 runs, without model weights."""

    path = (
        SOURCE / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4"
        "/plain_base_tokenizer/tokenizer"
    )
    if not (path / "tokenizer.json").is_file():
        pytest.skip("Set S1_AUDIT_SOURCE_ROOT to the frozen tokenizer checkout")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _blocks(encoded, tokenizer):
    """Expose real rendered block boundaries independently of field segmentation."""

    ids = encoded["input_ids"]
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    message_id = tokenizer.convert_tokens_to_ids("<|message|>")
    starts = [index for index, token in enumerate(ids) if token == start_id]
    for start, end in zip(starts, starts[1:] + [len(ids)], strict=True):
        separator = ids.index(message_id, start, end)
        yield start, separator, end, tokenizer.decode(ids[start + 1 : separator])


@pytest.mark.parametrize("priority", [False, True])
def test_telemetry_leaves_tokens_masks_and_weights_identical(tokenizer, priority):
    """Annotations must be additive under both existing training objectives."""

    row = _row()
    before = deepcopy(row)
    original = tokenize_v3_trace_for_sft(row, tokenizer, 4096, priority_weights=priority)
    annotated = tokenize_v3_trace_for_sft(
        row, tokenizer, 4096, priority_weights=priority, segment_telemetry=True
    )
    segments = annotated.pop("loss_segments")
    fields = annotated.pop("loss_field_masks")
    assert len(fields) == len(segments)
    assert all(not value or segment == 3 for value, segment in zip(fields, segments, strict=True))
    assert annotated == original and row == before
    assert len(segments) == len(original["input_ids"])
    assert all(
        bool(segment) == bool(active)
        for segment, active in zip(segments, original["completion_mask"], strict=True)
    )
    assert set(segments) == {0, *LOSS_SEGMENT_NAMES}
    assert LOSS_SEGMENT_SCHEMA_VERSION == "mentor-rl-v3-loss-segments-v1"


def test_repeated_escaped_prose_never_relabels_structured_result(tokenizer):
    """Equal strings in reasoning, result, and prose must retain distinct labels."""

    row = _row()
    repeated = 'TP53 "null" \\ café α\nsame content'
    final = row["messages"][-1]
    final["thinking"] = repeated
    final["content"]["status"] = "not_found"
    final["content"]["result"] = {"summary": repeated, "blocked_reason": repeated}
    final["content"]["continuation_decision"] = "defer"
    final["content"]["blocked_reason"] = repeated
    final["content"]["summary"] = repeated
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096, segment_telemetry=True)
    ids, segments = encoded["input_ids"], encoded["loss_segments"]
    for _, separator, end, header in _blocks(encoded, tokenizer):
        if header == "assistant<|channel|>analysis":
            assert set(segments[separator + 1 : end - 1]) == {1}
        if header != "assistant<|channel|>final":
            continue
        body = tokenizer.decode(ids[separator + 1 : end - 1])
        parsed = json.loads(body)
        assert parsed["result"]["summary"] == repeated
        # Use independently tokenized literal spans to identify the nested result.
        nested = json.dumps(parsed["result"], ensure_ascii=False, separators=(",", ":"))
        result_start = body.index(nested)
        result_end = result_start + len(nested)
        mapping = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
        for index, (left, right) in enumerate(mapping["offset_mapping"]):
            if left >= result_start and right <= result_end:
                assert segments[separator + 1 + index] == 3
        prose_ids = [
            token for token, segment in zip(
                ids[separator + 1 : end - 1], segments[separator + 1 : end - 1]
            ) if segment == 4
        ]
        assert tokenizer.decode(prose_ids).count("same content") == 2


def test_null_and_empty_prose_values_have_no_wording_tokens(tokenizer):
    """Null and empty strings retain structural presence without fake prose loss."""

    row = _row()
    row["messages"][-1]["content"]["summary"] = ""
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096, segment_telemetry=True)
    assert 4 not in encoded["loss_segments"]
    row["messages"][-1]["content"]["blocked_reason"] = ""
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096, segment_telemetry=True)
    assert 4 not in encoded["loss_segments"]


def test_multi_call_history_and_control_tokens_are_partitioned(tokenizer):
    """Historical actions/results stay masked while current calls retain names."""

    row = _row()
    old_turn = deepcopy(row["messages"][1:])
    current = deepcopy(row["messages"][1:])
    current[1]["thinking"] = "Check the current request, not the old answer."
    current[-1:-1] = deepcopy(current[1:3])
    row["messages"] = [row["messages"][0], *old_turn, *current]
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096, segment_telemetry=True)
    segments = encoded["loss_segments"]
    current_call_blocks = 0
    for start, separator, end, header in _blocks(encoded, tokenizer):
        if not encoded["completion_mask"][start]:
            assert set(segments[start:end]) == {0}
            continue
        assert segments[start] == segments[separator] == segments[end - 1] == 5
        if header.startswith("assistant to=functions."):
            current_call_blocks += 1
            assert set(segments[separator + 1 : end - 1]) == {2}
            name_ids = [
                token for token, segment in zip(
                    encoded["input_ids"][start + 1 : separator],
                    segments[start + 1 : separator],
                ) if segment == 2
            ]
            assert "lookup_human_gene_symbol" in tokenizer.decode(name_ids)
    assert current_call_blocks == 2


def test_segmentation_rejects_misaligned_actual_tokens(tokenizer):
    """A changed training body must fail instead of silently measuring wrong spans."""

    row = _row()
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096)
    broken = list(encoded["input_ids"])
    for _, separator, end, header in _blocks(encoded, tokenizer):
        if header == "assistant<|channel|>final":
            broken[separator + 1] = tokenizer.convert_tokens_to_ids("<|end|>")
            break
    with pytest.raises(RuntimeError, match="offsets do not match"):
        v3_assistant_loss_segments(row, tokenizer, broken, encoded["completion_mask"])


def test_structured_value_annotations_use_exact_top_level_spans(tokenizer):
    """Repeated strings in prose and nested objects cannot move field diagnostics."""
    from runtime.world_model_training import LOSS_FIELD_NAMES

    row = _row()
    row["messages"][-1]["content"]["summary"] = 'result status provenance continuation_decision "stop"'
    encoded = tokenize_v3_trace_for_sft(row, tokenizer, 4096, segment_telemetry=True)
    fields = encoded["loss_field_masks"]
    for _, separator, end, header in _blocks(encoded, tokenizer):
        if header != "assistant<|channel|>final":
            assert not any(fields[separator + 1:end - 1])
            continue
        body = tokenizer.decode(encoded["input_ids"][separator + 1:end - 1])
        values = json.loads(body)
        spans = []
        cursor = 1
        for key, value in values.items():
            key_text = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            value_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            begin = cursor + len(key_text) + 1
            spans.append((key, begin, begin + len(value_text)))
            cursor = begin + len(value_text) + 1
        offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for index, (left, right) in enumerate(offsets):
            expected = sum(bit for bit, name in LOSS_FIELD_NAMES.items()
                           if any(key == name and right > begin and left < finish for key, begin, finish in spans))
            if encoded["loss_segments"][separator + 1 + index] != 3:
                expected = 0
            assert fields[separator + 1 + index] == expected
        assert {bit for bit in LOSS_FIELD_NAMES if any(value & bit for value in fields)} == set(LOSS_FIELD_NAMES)
