"""Test the S0 trainer loss contract selection."""

from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from runtime.world_model_contracts import (
    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
)
from scripts.train_sft_dp_tp_ep import (
    compute_loss_objective,
    parse_args,
)


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


class SharedTraceLossContractTests(unittest.TestCase):
    """Test the shared trace loss contract at both trainer gates."""

    def test_parser_accepts_shared_trace_loss_contract(self) -> None:
        """Accept the v7 loss contract as a trainer argument."""

        argv = [
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
            "--loss_contract",
            SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(
            args.loss_contract,
            SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        )

    def test_loss_accepts_shared_trace_loss_contract(self) -> None:
        """Compute a CPU loss with the v7 loss contract."""

        batch = {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
        }

        objective = compute_loss_objective(
            _TinyModel(),
            batch,
            SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        )

        self.assertEqual(objective["completion_tokens"], 2)
        self.assertTrue(torch.isfinite(objective["loss"]))


if __name__ == "__main__":
    unittest.main()
