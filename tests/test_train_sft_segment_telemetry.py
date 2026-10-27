"""Check segment reporting preserves gradients and pools token counts correctly."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from runtime.world_model_loss_metrics import segment_token_totals, summarize_segment_totals
from scripts import train_sft_dp_tp_ep as trainer


class PositionModel(nn.Module):
    """Expose trainable CPU logits and reject accidental annotation forwarding."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7)
        self.values = nn.Parameter(torch.arange(98).reshape(2, 7, 7).float().sin())

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.values)


@pytest.mark.parametrize("priority", [False, True])
def test_reporting_preserves_objective_and_gradients_with_shifted_masked_targets(priority):
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 0, 0], [0, 5, 4, 3, 2, 1, 6]]),
        "labels": torch.tensor([[-100, 1, 2, 3, 4, -100, -100], [-100, 5, 4, 3, -100, 1, 6]]),
    }
    segments = torch.tensor([[0, 1, 2, 3, 4, 0, 0], [0, 5, 3, 1, 0, 2, 3]])
    contract = trainer.S0_V3_TRACE_LOSS_CONTRACT
    if priority:
        batch["loss_weights"] = torch.tensor([
            [0, 1, 2, 4, 1, 0, 0], [0, 1, 4, 1, 0, 2, 4],
        ]).float()
        contract = trainer.S0_V4_PRIORITY_LOSS_CONTRACT
    model = PositionModel()
    baseline = trainer.compute_loss_objective(model, batch, contract)
    baseline["loss"].backward()
    original_gradient = model.values.grad.clone()
    model.zero_grad()
    reported = trainer.compute_loss_objective(model, {**batch, "loss_segments": segments}, contract)
    reported["loss"].backward()
    assert torch.equal(reported["loss"], baseline["loss"])
    assert torch.equal(reported["completion_loss"], baseline["completion_loss"])
    assert torch.equal(model.values.grad, original_gradient)
    assert not reported["segment_totals"].requires_grad
    assert "segment_totals" not in baseline

    # Independent next-token calculation excludes the terminal logit and padding.
    expected_loss = nn.functional.cross_entropy(
        model.values[:, :-1].reshape(-1, 7), batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(2, 6)
    expected_correct = model.values[:, :-1].argmax(-1) == batch["labels"][:, 1:]
    totals = reported["segment_totals"]
    for segment_id in trainer.LOSS_SEGMENT_NAMES:
        selected = segments[:, 1:] == segment_id
        assert totals[segment_id - 1, 0].item() == pytest.approx(expected_loss[selected].double().sum().item())
        assert totals[segment_id - 1, 1].item() == selected.sum().item()
        assert totals[segment_id - 1, 2].item() == (expected_correct & selected).sum().item()
    assert totals[:, 1].sum().item() == reported["completion_tokens"] == 9
    assert (totals[:, 0].sum() / totals[:, 1].sum()).item() == pytest.approx(baseline["completion_loss"].item())


@pytest.mark.parametrize("bad_segments", [
    [[0, 0]],  # A supervised target has no segment.
    [[1, 1]],  # A masked target receives a segment.
    [[0, 6]],  # Unknown code.
])
def test_segment_coverage_errors_fail_instead_of_silently_dropping_tokens(bad_segments):
    with pytest.raises(ValueError, match="Every supervised"):
        segment_token_totals(
            token_loss=torch.zeros(1, 2),
            shift_labels=torch.tensor([[-100, 1]]),
            shift_segments=torch.tensor(bad_segments),
            logits=torch.zeros(1, 2, 3),
        )


def test_collator_pads_annotations_without_changing_historical_batch():
    collator = trainer.S0DataCollator(
        pad_token_id=0, completion_only_loss=True, pad_to_multiple_of=8,
    )
    examples = [
        {"input_ids": [1, 2], "completion_mask": [0, 1]},
        {"input_ids": [1, 3, 4, 5], "completion_mask": [0, 1, 0, 1]},
    ]
    baseline = collator(examples)
    reported = collator([
        {**examples[0], "loss_segments": [0, 3]},
        {**examples[1], "loss_segments": [0, 1, 0, 5]},
    ])
    assert all(torch.equal(reported[key], value) for key, value in baseline.items())
    assert reported["loss_segments"].tolist() == [
        [0, 3, 0, 0, 0, 0, 0, 0], [0, 1, 0, 5, 0, 0, 0, 0],
    ]
    with pytest.raises(RuntimeError, match="cannot mix"):
        collator([{**examples[0], "loss_segments": [0, 3]}, examples[1]])
    with pytest.raises(RuntimeError, match="completion-only"):
        collator([{**examples[0], "loss_segments": [1, 3]}])
    with pytest.raises(RuntimeError, match="completion-only"):
        collator([{**examples[0], "loss_segments": [0, 6]}])


def test_reduce_pools_sums_and_counts_only_over_supplied_data_parallel_group(monkeypatch):
    local = torch.zeros(5, 3, dtype=torch.float64)
    local[0] = torch.tensor([2, 1, 1])
    remote = torch.zeros_like(local)
    remote[0] = torch.tensor([9, 3, 1])
    marker = object()
    calls = []

    def all_reduce(tensor, *, op, group):
        assert op == trainer.dist.ReduceOp.SUM and group is marker
        calls.append(group)
        tensor.add_(remote)

    monkeypatch.setattr(trainer.dist, "all_reduce", all_reduce)
    result = trainer.reduce_loss_segment_window(local, dp_group=marker)
    reasoning = result["loss_segments"]["reasoning"]
    assert reasoning["nll"] == 11 / 4  # Neither rank-mean 2.5 nor a TP-multiplied count.
    assert reasoning["token_count"] == 4
    assert reasoning["teacher_forced_token_accuracy"] == 0.5
    assert result["loss_segments"]["structured_final"]["nll"] is None
    assert local[0].tolist() == [2, 1, 1]
    assert trainer.reduce_loss_segment_window(None, dp_group=marker) == {}
    assert len(calls) == 1


def test_cache_default_identity_unchanged_but_telemetry_and_adapter_changes_invalidate(monkeypatch):
    identity = {
        "source_fingerprint": "source", "max_length": 4096,
        "tokenizer_identity_sha256": "tokenizer", "loss_contract": trainer.S0_V3_TRACE_LOSS_CONTRACT,
    }
    historical = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert trainer._tokenization_cache_fingerprint(**identity) == historical
    monkeypatch.setattr(trainer, "sha256_file", lambda path: "first-implementation")
    first = trainer._tokenization_cache_fingerprint(**identity, segment_loss_telemetry=True)
    monkeypatch.setattr(trainer, "sha256_file", lambda path: "second-implementation")
    second = trainer._tokenization_cache_fingerprint(**identity, segment_loss_telemetry=True)
    assert len({historical, first, second}) == 3


def test_summary_rejects_corrupt_counts():
    totals = torch.zeros(5, 3)
    totals[0] = torch.tensor([1, 1, 2])
    with pytest.raises(ValueError, match="counts"):
        summarize_segment_totals(totals)


def test_cli_telemetry_is_opt_in_and_restricted_to_v3(tmp_path, monkeypatch):
    source = tmp_path / "fixture.jsonl"
    source.write_text("{}\n")
    argv = ["train_sft_dp_tp_ep.py", "--bf16", "--loss_contract",
            trainer.S0_V3_TRACE_LOSS_CONTRACT]
    for name in (
        "model_path", "tokenizer_path", "dataset_path", "validation_dataset_path",
        "validation_answer_key_path", "output_dir",
    ):
        argv.extend([f"--{name}", str(source)])
    monkeypatch.setattr(sys, "argv", argv)
    assert trainer.parse_args().segment_loss_telemetry is False
    monkeypatch.setattr(sys, "argv", argv + [
        "--segment_loss_telemetry", "--loss_contract", "s0_tool_trajectory_v1",
    ])
    with pytest.raises(SystemExit):
        trainer.parse_args()
    monkeypatch.setattr(sys, "argv", argv + [
        "--segment_loss_telemetry", "--loss_contract", trainer.S0_V3_TRACE_LOSS_CONTRACT,
    ])
    args = trainer.parse_args()
    trainer.validate_cli(args)
    assert args.segment_loss_telemetry is True


def test_stratified_fields_and_groups_preserve_gradients_and_pool_exact_tokens():
    """Rows contribute once to overall, stage, family, and source-trace totals."""
    from runtime.world_model_loss_metrics import LOSS_GROUP_NAMES, loss_group_ids

    model = PositionModel()
    batch = {
        "input_ids": torch.zeros(2, 7, dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2, 3, 4, -100, -100], [-100, 5, 4, 3, -100, 1, 6]]),
    }
    annotations = {
        "loss_segments": torch.tensor([[0, 1, 2, 3, 3, 0, 0], [0, 5, 3, 1, 0, 2, 3]]),
        "loss_field_masks": torch.tensor([[0, 0, 0, 1, 2, 0, 0], [0, 0, 4, 0, 0, 0, 8]]),
        "loss_group_ids": torch.tensor([
            loss_group_ids({"step": "S0", "question_family": "human_symbol_to_ensembl", "source_trace": "S0.1"}),
            loss_group_ids({"step": "S1", "question_family": "direction_conventions", "source_trace": "S1.8.1"}),
        ]),
    }
    baseline = trainer.compute_loss_objective(model, batch, trainer.S0_V3_TRACE_LOSS_CONTRACT)
    baseline["loss"].backward()
    gradients = model.values.grad.clone()
    model.zero_grad()
    measured = trainer.compute_loss_objective(model, {**batch, **annotations}, trainer.S0_V3_TRACE_LOSS_CONTRACT)
    measured["loss"].backward()
    assert torch.equal(measured["loss"], baseline["loss"])
    assert torch.equal(model.values.grad, gradients)
    totals = measured["stratified_totals"]
    assert not totals.requires_grad
    assert totals[0, :5, 1].sum() == 9
    assert totals[0, 5:, 1].tolist() == [1, 1, 1, 1]
    assert totals[LOSS_GROUP_NAMES.index("stage:S0"), :5, 1].sum() == 4
    assert totals[LOSS_GROUP_NAMES.index("stage:S1"), :5, 1].sum() == 5
    for dimension in ("stage:", "family:", "source_trace:"):
        selected = [index for index, name in enumerate(LOSS_GROUP_NAMES) if name.startswith(dimension)]
        assert torch.equal(totals[selected].sum(dim=0), totals[0])


def test_collator_keeps_field_and_group_annotations_out_of_training_inputs():
    """Metadata padding leaves historical IDs and labels unchanged."""
    collator = trainer.S0DataCollator(pad_token_id=0, completion_only_loss=True, pad_to_multiple_of=8)
    example = {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]}
    baseline = collator([example])
    measured = collator([{**example, "loss_segments": [0, 1, 3],
                          "loss_field_masks": [0, 0, 4], "loss_group_ids": [1, 3, 11]}])
    assert all(torch.equal(measured[key], value) for key, value in baseline.items())
    assert measured["loss_field_masks"].tolist() == [[0, 0, 4, 0, 0, 0, 0, 0]]
    assert measured["loss_group_ids"].tolist() == [[1, 3, 11]]


def test_grouped_reduction_keeps_empty_groups_absent_and_empty_fields_null(monkeypatch):
    """Distributed reduction sums sufficient statistics before computing means."""
    from runtime.world_model_loss_metrics import LOSS_CATEGORY_NAMES, LOSS_GROUP_NAMES

    totals = torch.zeros(len(LOSS_GROUP_NAMES), len(LOSS_CATEGORY_NAMES), 3, dtype=torch.float64)
    totals[0, 0] = torch.tensor([2, 1, 1])
    totals[LOSS_GROUP_NAMES.index("stage:S1"), 0] = totals[0, 0]
    monkeypatch.setattr(trainer.dist, "all_reduce", lambda tensor, **kwargs: tensor.mul_(2))
    result = trainer.reduce_loss_segment_window(totals, dp_group=object())
    assert result["loss_segments"]["reasoning"]["nll"] == 2
    assert result["loss_segments"]["reasoning"]["token_count"] == 2
    assert result["loss_fields"]["status"]["nll"] is None
    assert set(result["loss_groups"]) == {"stage:S1"}
    assert result["loss_groups"]["stage:S1"]["reasoning"]["nll"] == 2


@pytest.mark.parametrize("steps", ["400,200", "200,200", "201", "bad", "0"])
def test_snapshot_cli_rejects_steps_that_cannot_be_saved(steps, tmp_path, monkeypatch):
    """Selected diagnostic adapters must correspond to completed step saves."""
    source = tmp_path / "fixture.jsonl"
    source.write_text("{}\n")
    argv = ["trainer", "--bf16", "--loss_contract", trainer.S0_V3_TRACE_LOSS_CONTRACT,
            "--diagnostic_snapshot_steps", steps]
    for name in ("model_path", "tokenizer_path", "dataset_path", "validation_dataset_path", "validation_answer_key_path", "output_dir"):
        argv.extend([f"--{name}", str(source)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="Diagnostic snapshot"):
        trainer.validate_cli(trainer.parse_args())


def test_completed_snapshot_is_captured_before_rolling_checkpoint_pruning(tmp_path, monkeypatch):
    """A selected adapter is retained while its source and older save still exist."""
    from runtime import world_model_checkpoint_diagnostics as snapshots

    old = tmp_path / "checkpoint-200"
    old.mkdir()
    (old / "CHECKPOINT_COMPLETE").write_text("global_step=200\n")
    events = []
    monkeypatch.setattr(trainer, "save_adapter", lambda *args: None)
    monkeypatch.setattr(trainer, "save_optimizer_shard", lambda *args: None)
    monkeypatch.setattr(trainer, "save_rng", lambda *args: None)
    monkeypatch.setattr(trainer.dist, "barrier", lambda: None)

    def capture(checkpoint, output, step, exposure):
        assert (checkpoint / "CHECKPOINT_COMPLETE").is_file()
        assert json.loads((checkpoint / "training_state.json").read_text())["global_step"] == step == 400
        assert old.exists() and exposure == {"fixture": "receipt"}
        events.append(step)

    monkeypatch.setattr(snapshots, "capture_validation_snapshot", capture)
    args = SimpleNamespace(save_total_limit=1, diagnostic_snapshot_steps=[400],
                           diagnostic_exposure_manifest_args={"fixture": "receipt"})
    trainer.save_checkpoint(None, None, SimpleNamespace(state_dict=lambda: {}),
                            trainer.LoopState(global_step=400), tmp_path, args, None, 0)
    assert events == [400]
    assert not old.exists()
    assert (tmp_path / "checkpoint-400" / "CHECKPOINT_COMPLETE").is_file()
