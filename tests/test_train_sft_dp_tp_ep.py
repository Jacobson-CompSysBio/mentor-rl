"""Test the S0 trainer loss contracts and batch collation."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from runtime.world_model_training import S0_FAMILIES, S0_V4_DATASET_ID
from scripts.train_sft_dp_tp_ep import (
    S0DataCollator,
    S0_V4_PRIORITY_LOSS_CONTRACT,
    compute_loss_objective,
    loss_contract_config,
    pad_loss_weights,
    parse_args,
    validate_dataset_question_families,
)


S0_V3_TRACE_LOSS_CONTRACT = "multiplex_sft_v3"
TRAINER_ARGV = [
    "train_sft_dp_tp_ep.py",
    "--model_path",
    "model",
    "--tokenizer_path",
    "tokenizer",
    "--dataset_path",
    "train.jsonl",
    "--validation_dataset_path",
    "val.jsonl",
    "--validation_answer_key_path",
    "val_answers.jsonl",
    "--output_dir",
    "output",
]


class _TinyModel:
    """Return fixed CPU logits for one small batch."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(vocab_size=4)

    def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
        input_ids = batch["input_ids"]
        logits = torch.zeros(
            (*input_ids.shape, self.config.vocab_size),
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)


class _PositionModel(_TinyModel):
    """Make each target position have a different cross-entropy value."""

    def __call__(self, **batch: torch.Tensor) -> SimpleNamespace:
        input_ids = batch["input_ids"]
        logits = torch.zeros(
            (*input_ids.shape, self.config.vocab_size),
            dtype=torch.float32,
        )
        logits[:, 0, 1] = 4.0
        logits[:, 1, 2] = 2.0
        logits[:, 2, 3] = -2.0
        return SimpleNamespace(logits=logits)


class LossContractTests(unittest.TestCase):
    """Test the two v4 loss contracts at the trainer gates."""

    def test_parser_accepts_only_the_v4_loss_contracts(self) -> None:
        for loss_contract in (
            S0_V3_TRACE_LOSS_CONTRACT,
            S0_V4_PRIORITY_LOSS_CONTRACT,
        ):
            argv = [*TRAINER_ARGV, "--loss_contract", loss_contract]
            with patch.object(sys, "argv", argv):
                self.assertEqual(parse_args().loss_contract, loss_contract)
        for argv in (
            TRAINER_ARGV,
            [*TRAINER_ARGV, "--loss_contract", "s0_tool_trajectory_v1"],
        ):
            with patch.object(sys, "argv", argv):
                with patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_uniform_loss_is_the_mean_completion_loss(self) -> None:
        batch = {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
        }

        objective = compute_loss_objective(
            _TinyModel(),
            batch,
            S0_V3_TRACE_LOSS_CONTRACT,
        )

        self.assertEqual(objective["completion_tokens"], 2)
        self.assertTrue(torch.isfinite(objective["loss"]))
        self.assertEqual(
            objective["loss"].item(), objective["completion_loss"].item()
        )

    def test_loss_rejects_a_removed_loss_contract(self) -> None:
        batch = {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "Unknown loss contract"):
            compute_loss_objective(
                _TinyModel(), batch, "supervised_explanation_trace_v1"
            )

    def test_v3_loss_contract_reports_all_masked_messages(self) -> None:
        """Report each message class that the v3 adapter masks."""

        config = loss_contract_config(S0_V3_TRACE_LOSS_CONTRACT)

        self.assertEqual(
            config["masked_messages"],
            [
                "system",
                "harness",
                "user",
                "tool",
                "earlier_assistant_final",
            ],
        )

    def test_priority_loss_contract_reports_exact_weight_policy(self) -> None:
        config = loss_contract_config(S0_V4_PRIORITY_LOSS_CONTRACT)

        self.assertEqual(config["target_format"], "multiplex_sft_v3")
        self.assertEqual(
            config["token_loss_weights"],
            {
                "nonassistant": 0.0,
                "assistant_prose": 1.0,
                "tool_call": 4.0,
                "final_json": 4.0,
                "critical_final_fields": 8.0,
                "critical_field_names": [
                    "status",
                    "result",
                    "continuation_decision",
                    "blocked_reason",
                ],
                "normalization": "positive_token_mean_per_row",
            },
        )

    def test_priority_objective_uses_shifted_token_weights(self) -> None:
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 3]], dtype=torch.long),
            "loss_weights": torch.tensor(
                [[0.0, 0.5, 0.5, 2.0]], dtype=torch.float32
            ),
        }

        objective = compute_loss_objective(
            _PositionModel(), batch, S0_V4_PRIORITY_LOSS_CONTRACT
        )
        uniform = compute_loss_objective(
            _PositionModel(),
            {key: value for key, value in batch.items() if key != "loss_weights"},
            S0_V3_TRACE_LOSS_CONTRACT,
        )

        self.assertEqual(objective["completion_tokens"], 3)
        self.assertAlmostEqual(
            objective["completion_loss"].item(),
            uniform["loss"].item(),
        )
        self.assertNotAlmostEqual(
            objective["loss"].item(),
            objective["completion_loss"].item(),
        )

    def test_priority_weights_follow_left_and_right_padding(self) -> None:
        examples = [
            {"input_ids": [1, 2], "loss_weights": [0.5, 1.5]},
            {"input_ids": [3, 4, 5], "loss_weights": [0.0, 0.5, 1.5]},
        ]

        right = pad_loss_weights(
            examples, padded_length=3, padding_side="right"
        )
        left = pad_loss_weights(
            examples, padded_length=3, padding_side="left"
        )

        self.assertEqual(
            right.tolist(), [[0.5, 1.5, 0.0], [0.0, 0.5, 1.5]]
        )
        self.assertEqual(
            left.tolist(), [[0.0, 0.5, 1.5], [0.0, 0.5, 1.5]]
        )

    def test_priority_collator_matches_trl_right_padding(self) -> None:
        collator = S0DataCollator(
            pad_token_id=0,
            completion_only_loss=True,
            pad_to_multiple_of=4,
        )
        examples = [
            {
                "input_ids": [1, 2],
                "completion_mask": [0, 1],
                "loss_weights": [0.0, 1.0],
            },
            {
                "input_ids": [3, 4, 5],
                "completion_mask": [0, 1, 1],
                "loss_weights": [0.0, 0.5, 1.5],
            },
        ]

        batch = collator(examples)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(
            batch["loss_weights"].tolist(),
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.5, 0.0]],
        )
        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, 2, -100, -100], [-100, 4, 5, -100]],
        )

    def test_family_gate_requires_the_three_s0_families(self) -> None:
        validate_dataset_question_families(
            S0_V4_DATASET_ID,
            sorted(S0_FAMILIES),
            label="validation",
        )
        with self.assertRaisesRegex(RuntimeError, "wrong question families"):
            validate_dataset_question_families(
                S0_V4_DATASET_ID,
                ["human_symbol_to_ensembl", "layer_tag_parsing"],
                label="validation",
            )


if __name__ == "__main__":
    unittest.main()
