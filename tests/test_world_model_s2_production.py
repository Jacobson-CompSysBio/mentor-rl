"""Protect the production context boundary, original splits, and replay budget."""

from copy import deepcopy

import pytest

from runtime.world_model.s2_production import load_recipe, partition_by_length, production_budget


def row(row_id, split, tokens):
    """Supply a test wire whose total includes prompt and assistant tokens."""
    return {"row_id": row_id, "metadata": {"split": split}, "wire_tokens": tokens,
            "messages": [{"role": "tool", "content": {"evidence": "preserve me"}}]}


def encode(item, _tokenizer):
    """Model complete-wire length with a nonempty assistant-only final mask."""
    size = item["wire_tokens"]
    return {"input_ids": [1] * size, "completion_mask": [0] * (size - 1) + [1]}


def test_exact_limit_kept_and_overlong_trace_excluded_whole():
    source = [row("edge", "train", 8192), row("long", "validation", 8193)]
    original = deepcopy(source)
    kept, report = partition_by_length(source, None, encode=encode)
    assert kept == [source[0]]
    assert kept[0] is source[0]
    assert source == original
    assert report["excluded"][0]["row_id"] == "long"
    assert report["excluded"][0]["tokens"] == 8193
    assert report["truncation_applied"] is False
    assert report["by_split"]["validation"] == {"eligible": 0, "excluded": 1}


def test_smaller_faithful_instance_can_retain_the_same_behavior():
    source = [row("large", "train", 20000), row("small", "train", 7000)]
    for item in source:
        item["metadata"]["variant"] = "complete_context"
    kept, report = partition_by_length(source, None, encode=encode)
    assert kept == [source[1]]
    assert kept[0]["metadata"]["variant"] == "complete_context"
    assert report["max_retained_tokens"] == 7000


def test_limit_applies_to_prompt_and_evidence_even_when_answer_is_short():
    kept, report = partition_by_length([row("long_prompt", "test", 9000)], None, encode=encode)
    assert not kept
    assert report["excluded"][0]["assistant_tokens"] == 1


@pytest.mark.parametrize("split", [None, "development"])
def test_missing_or_invalid_source_split_rejected(split):
    with pytest.raises(ValueError, match="source split"):
        partition_by_length([row("bad", split, 100)], None, encode=encode)


def test_duplicate_ids_rejected_without_silent_deduplication():
    with pytest.raises(ValueError, match="unique"):
        partition_by_length([row("same", "train", 100), row("same", "test", 100)], None, encode=encode)


def test_recomputed_budget_preserves_approved_mixture_after_case_exclusions():
    original = production_budget(399)
    assert original == {"retained_cases": 399,
                        "s2": {"train": 204288, "validation": 25536, "test": 25536},
                        "replay": {"S1": 38304, "S0": 12768},
                        "train": 255360, "validation": 25536, "test": 25536}
    smaller = production_budget(398)
    assert original["train"] - smaller["train"] == 640
    assert original["validation"] - smaller["validation"] == 64
    assert smaller["s2"]["train"] / smaller["train"] == .8
    assert smaller["replay"]["S1"] / smaller["train"] == .15
    assert smaller["replay"]["S0"] / smaller["train"] == .05
    assert load_recipe()["max_length"] == 8192
