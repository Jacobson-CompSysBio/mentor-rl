"""Bound training curves and evaluation summaries without losing local detail."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_world_model_s1 as s1
from scripts import evaluate_world_model_v2_s0 as s0
from scripts.train_sft_dp_tp_ep import training_wandb_metrics


@pytest.mark.parametrize("s1_passed,s0_passed", [(True, False), (False, True)])
def test_s1_lifecycle_publishes_eight_training_and_four_test_metrics(
    monkeypatch, s1_passed, s0_passed,
):
    """Gate failures remain visible even when pooled accuracies look good."""
    logged = []
    resumed = []
    finished = []
    run = SimpleNamespace(
        id="training-run", name="fixture", entity="jail-ai", project="mentor-sft",
        url="https://wandb.ai/fixture", log=logged.append,
        finish=lambda: finished.append(True),
    )

    def initialize(**kwargs):
        resumed.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    for key, value in {"WANDB_MODE": "online", "WANDB_ENTITY": "jail-ai",
                       "WANDB_PROJECT": "mentor-sft"}.items():
        monkeypatch.setenv(key, value)
    training = {
        "global_step": 1728, "loss": 0.4, "completion_loss": 0.3,
        "loss_segments": {name: {"nll": nll} for name, nll in (
            ("reasoning", 1.7), ("tool_call", 0.0), ("structured_final", 0.13),
            ("free_text", 0.2), ("formatting", 0.3),
        )},
        "loss_groups": {str(i): {"nll": 0.5} for i in range(3000)},
    }
    original_training = deepcopy(training)
    for prefix in ("train", "eval"):
        logged.append(training_wandb_metrics(training, prefix))
    assert len({key for row in logged for key in row} - {"global_step"}) == 8
    s1_metrics = {
        "train_run_id": run.id,
        "overall": {"structured_answer_accuracy_rate": 0.95,
                    "final_answer_accuracy_rate": 0.2},
        "gate": {"passed": s1_passed, "checks": {"continuation": s1_passed}},
        "families": {"example": {"result_accuracy_rate": 0.7}},
    }
    s0_metrics = {
        "train_run_id": run.id, "overall": {"result_exact_rate": 0.999},
        "gate": {"passed": s0_passed}, "families": {"example": {"accuracy": 0.99}},
    }
    original_s1, original_s0 = deepcopy(s1_metrics), deepcopy(s0_metrics)
    assert s1.publish_wandb(s1_metrics)["id"] == run.id
    assert s0.publish_wandb(s0_metrics, "s0_forgetting")["id"] == run.id
    merged = {key: value for row in logged for key, value in row.items()}
    assert set(merged) - {"global_step"} == {
        "train/loss", "eval/loss", "test/s1/structured_answer_accuracy",
        "test/s1/gate_passed", "s0_forgetting/result_exact_rate",
        "s0_forgetting/gate_passed",
        *{f"{prefix}/nll/{name}" for prefix in ("train", "eval")
          for name in ("reasoning", "tool_call", "structured_final")},
    }
    assert merged["train/nll/tool_call"] == 0.0
    assert merged["eval/nll/reasoning"] == 1.7
    assert merged["test/s1/structured_answer_accuracy"] == 0.95
    assert merged["test/s1/gate_passed"] == int(s1_passed)
    assert merged["s0_forgetting/gate_passed"] == int(s0_passed)
    assert training == original_training
    assert s1_metrics == original_s1 and s0_metrics == original_s0
    assert len(resumed) == len(finished) == 2
    assert all(args["id"] == run.id and args["resume"] == "must" for args in resumed)


@pytest.mark.parametrize("segments", [{}, {"tool_call": {"nll": None}}])
def test_missing_segment_loss_is_omitted_instead_of_reported_as_zero(segments):
    """Legacy runs and windows without tool calls cannot invent segment losses."""
    payload = {"global_step": 20, "loss": 0.4, "loss_segments": segments}
    assert training_wandb_metrics(payload, "train") == {
        "global_step": 20, "train/loss": 0.4,
    }


def test_standalone_s0_keeps_default_test_namespace(monkeypatch):
    """S0 training still publishes its own result and gate without S1 prefixes."""
    logged = []
    run = SimpleNamespace(id="s0", name="s0", entity="jail-ai", project="mentor-sft",
                          url="fixture", log=logged.append, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run))
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_ENTITY", "jail-ai")
    monkeypatch.setenv("WANDB_PROJECT", "mentor-sft")
    s0.publish_wandb({"train_run_id": "s0", "overall": {"result_exact_rate": 1.0},
                      "gate": {"passed": True}})
    assert logged == [{"test/result_exact_rate": 1.0, "test/gate_passed": 1}]
