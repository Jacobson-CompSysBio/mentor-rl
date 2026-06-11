import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_expert_alignment import evaluate_alignment


class ExpertAlignmentTests(unittest.TestCase):
    def test_evaluate_alignment_reports_selected_model_and_pairwise_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packets_path = tmp / "packets.jsonl"
            labels_path = tmp / "labels.jsonl"
            packet = {
                "review_item_id": "traj-1.step0",
                "selected_branch_id": "branch-selected",
                "expert_packet": {
                    "candidate_branches": [
                        {"branch_id": "branch-selected"},
                        {"branch_id": "branch-expert"},
                    ]
                },
                "hidden_scoring": {
                    "branch-selected": {"normalized_score": 0.4},
                    "branch-expert": {"normalized_score": 0.9},
                },
            }
            label = {
                "review_item_id": "traj-1.step0",
                "expert_id": "expert-a",
                "chosen_branch_id": "branch-expert",
                "confidence": 0.9,
                "acceptable_branch_ids": ["branch-expert"],
                "notes": "Better supported trajectory.",
            }
            packets_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")
            labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")

            report = evaluate_alignment(packets_path, labels_path)

            self.assertEqual(report["usable_label_count"], 1)
            self.assertEqual(report["selected_vs_expert_top1_agreement"], 0.0)
            self.assertEqual(report["model_score_vs_expert_top1_agreement"], 1.0)
            self.assertEqual(report["model_score_top2_agreement"], 1.0)
            self.assertEqual(report["pairwise_preference_accuracy"], 1.0)
            self.assertEqual(report["high_confidence_disagreement_count"], 1)


if __name__ == "__main__":
    unittest.main()
