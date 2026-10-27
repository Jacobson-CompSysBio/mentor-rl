"""Measure unweighted teacher-forced token loss without changing SFT gradients."""

from __future__ import annotations

from typing import Any

import torch

from runtime.world_model_training import LOSS_FIELD_NAMES, LOSS_SEGMENT_NAMES, V3_TRACE_FAMILIES


@torch.no_grad()
def segment_token_totals(
    *,
    token_loss: torch.Tensor,
    shift_labels: torch.Tensor,
    shift_segments: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Return detached [segment, NLL sum/count/correct] totals for shifted targets.

    The caller shifts labels and segment IDs together. Summing these totals,
    before division, gives token-normalized metrics across unequal row lengths.
    These label-conditioned accuracies are separate from rollout answer accuracy.
    """
    if (
        token_loss.shape != shift_labels.shape
        or shift_segments.shape != shift_labels.shape
        or logits.shape[:-1] != shift_labels.shape
        or shift_segments.dtype != torch.long
    ):
        raise ValueError("Loss segments, labels, losses, and logits must align")
    labels = shift_labels.to(logits.device)
    segments = shift_segments.to(logits.device)
    supervised = labels != -100
    if (
        ((segments < 0) | (segments > len(LOSS_SEGMENT_NAMES))).any()
        or ((segments > 0) != supervised).any()
    ):
        raise ValueError("Every supervised target needs one segment; masked targets need zero")
    correct = logits.detach().argmax(dim=-1) == labels
    result = torch.zeros(
        (len(LOSS_SEGMENT_NAMES), 3), dtype=torch.float64, device=logits.device
    )
    for segment_id in LOSS_SEGMENT_NAMES:
        selected = segments == segment_id
        result[segment_id - 1, 0] = token_loss.detach()[selected].double().sum()
        result[segment_id - 1, 1] = selected.sum()
        result[segment_id - 1, 2] = (correct & selected).sum()
    return result


def summarize_segment_totals(totals: torch.Tensor) -> dict[str, dict[str, Any]]:
    """Convert globally summed token totals to metrics, leaving empty means null."""
    if totals.shape != (len(LOSS_SEGMENT_NAMES), 3):
        raise ValueError("Loss segment totals have an invalid shape")
    if not torch.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Loss segment totals must be finite and nonnegative")
    summaries = {}
    for segment_id, name in LOSS_SEGMENT_NAMES.items():
        nll_sum, count, correct = totals[segment_id - 1].detach().cpu().tolist()
        if count != int(count) or correct != int(correct) or correct > count:
            raise ValueError("Loss segment token counts are invalid")
        summaries[name] = {
            "nll_sum": nll_sum,
            "token_count": int(count),
            "nll": nll_sum / count if count else None,
            "correct_tokens": int(correct),
            "teacher_forced_token_accuracy": correct / count if count else None,
        }
    return summaries


# Fixed IDs keep all replicas' reduction tensors identical even when a batch
# lacks a stage, family, or trace. Metadata never reaches the model forward.
LOSS_TELEMETRY_SCHEMA_VERSION = "mentor-rl-stratified-loss-v1"

LOSS_CATEGORY_NAMES = tuple(LOSS_SEGMENT_NAMES.values()) + tuple(LOSS_FIELD_NAMES.values())
LOSS_GROUP_NAMES = (
    "overall",
    "stage:S0", "stage:S1",
    *(f"family:{name}" for name in sorted(V3_TRACE_FAMILIES)),
    *(f"source_trace:S0.{form}" for form in range(1, 8)),
    *(f"source_trace:S1.{section}.{form}"
      for section, count in ((1, 11), (2, 12), (3, 10), (4, 9), (6, 7), (7, 9), (8, 6), (9, 2))
      for form in range(1, count + 1)),
)


def loss_group_ids(metadata: dict[str, Any]) -> list[int]:
    """Encode reviewed stage/family/trace metadata using the fixed run catalog."""
    keys = (
        f"stage:{metadata.get('step')}",
        f"family:{metadata.get('question_family')}",
        f"source_trace:{metadata.get('source_trace')}",
    )
    try:
        return [LOSS_GROUP_NAMES.index(key) for key in keys]
    except ValueError as exc:
        raise ValueError(f"Unknown stratified loss metadata: {keys}") from exc


@torch.no_grad()
def stratified_token_totals(
    *, token_loss: torch.Tensor, shift_labels: torch.Tensor,
    shift_segments: torch.Tensor, shift_field_masks: torch.Tensor,
    group_ids: torch.Tensor, logits: torch.Tensor,
) -> torch.Tensor:
    """Pool detached segment and field diagnostics by row metadata.

    Value fields can overlap at tokenizer boundaries and do not form a token
    partition. Only the five original segments partition completion loss.
    """
    if (
        token_loss.shape != shift_labels.shape
        or shift_segments.shape != shift_labels.shape
        or shift_field_masks.shape != shift_labels.shape
        or logits.shape[:-1] != shift_labels.shape
        or group_ids.shape != (shift_labels.shape[0], 3)
    ):
        raise ValueError("Stratified loss annotations must align with shifted labels")
    labels = shift_labels.to(logits.device)
    segments = shift_segments.to(logits.device)
    fields = shift_field_masks.to(logits.device)
    groups = group_ids.to(logits.device)
    if (
        ((segments < 0) | (segments > len(LOSS_SEGMENT_NAMES))).any()
        or ((segments > 0) != (labels != -100)).any()
        or ((fields < 0) | (fields > sum(LOSS_FIELD_NAMES))).any()
        or ((fields > 0) & (segments != 3)).any()
        or ((groups <= 0) | (groups >= len(LOSS_GROUP_NAMES))).any()
    ):
        raise ValueError("Invalid stratified loss annotation")
    correct = logits.detach().argmax(dim=-1) == labels
    result = torch.zeros(
        (len(LOSS_GROUP_NAMES), len(LOSS_CATEGORY_NAMES), 3),
        dtype=torch.float64, device=logits.device,
    )
    for category_index in range(len(LOSS_CATEGORY_NAMES)):
        selected = (
            segments == category_index + 1 if category_index < len(LOSS_SEGMENT_NAMES)
            else (fields & tuple(LOSS_FIELD_NAMES)[category_index - len(LOSS_SEGMENT_NAMES)]) != 0
        )
        row_totals = torch.stack((
            (token_loss.detach().double() * selected).sum(dim=-1),
            selected.sum(dim=-1), (correct & selected).sum(dim=-1),
        ), dim=-1)
        result[0, category_index] = row_totals.sum(dim=0)
        for column in range(3):
            result[:, category_index].index_add_(0, groups[:, column], row_totals)
    return result


def summarize_stratified_totals(totals: torch.Tensor) -> dict[str, Any]:
    """Expose global fields and nonempty stage/family/source-trace summaries."""
    if totals.shape != (len(LOSS_GROUP_NAMES), len(LOSS_CATEGORY_NAMES), 3):
        raise ValueError("Stratified loss totals have an invalid shape")
    if not torch.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Stratified loss totals must be finite and nonnegative")
    groups = {}
    for group_name, group in zip(LOSS_GROUP_NAMES, totals.detach().cpu().tolist(), strict=True):
        summaries = {}
        for category_name, (nll_sum, count, correct) in zip(LOSS_CATEGORY_NAMES, group, strict=True):
            if count != int(count) or correct != int(correct) or correct > count:
                raise ValueError("Stratified loss token counts are invalid")
            summaries[category_name] = {
                "nll_sum": nll_sum, "token_count": int(count),
                "nll": nll_sum / count if count else None,
                "correct_tokens": int(correct),
                "teacher_forced_token_accuracy": correct / count if count else None,
            }
        if group_name == "overall" or any(item["token_count"] for item in summaries.values()):
            groups[group_name] = summaries
    return {
        "loss_telemetry_schema_version": LOSS_TELEMETRY_SCHEMA_VERSION,
        "loss_fields": {name: groups["overall"][name] for name in LOSS_FIELD_NAMES.values()},
        "loss_groups": {name: summary for name, summary in groups.items() if name != "overall"},
    }
