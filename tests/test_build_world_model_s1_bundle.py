"""Focused tests for the aggregate S1 bundle projection and publication."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest

from runtime.world_model.contract import load_contract
from runtime.world_model.s0_corpus import (
    EvidenceSequence,
    S0_V4_METADATA_ADDITIONS,
)
from scripts.build_world_model_s1_bundle import (
    AtomicBatch,
    BundleBuildError,
    normalize_s0_replay_row,
    project_row,
    rekey_training_evidence,
)


class WorldModelS1BundleTests(unittest.TestCase):
    def test_projection_splits_at_the_last_user_turn(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "harness", "content": "context"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": {"status": "resolved"}},
            {"role": "user", "content": "current"},
            {"role": "assistant", "thinking": "rule", "tool_call": {}},
            {"role": "tool", "name": "lookup", "content": {}},
            {"role": "assistant", "content": {"status": "resolved"}},
        ]
        question, answer = project_row(
            "row_1",
            messages,
            [{"type": "function", "function": {"name": "lookup"}}],
            {"question_family": "family"},
            "validation",
        )
        self.assertEqual(
            [message["role"] for message in question["messages"]],
            ["system", "harness", "user", "assistant", "user"],
        )
        self.assertEqual(
            [message["role"] for message in answer["expected_messages"]],
            ["assistant", "tool", "assistant"],
        )
        self.assertEqual(question["record_id"], answer["record_id"])
        self.assertEqual(question["split"], answer["split"])

    def test_projection_requires_a_user_and_expected_suffix(self) -> None:
        with self.assertRaises(BundleBuildError):
            project_row("row_1", [{"role": "system"}], [], {}, "test")
        with self.assertRaises(BundleBuildError):
            project_row("row_2", [{"role": "user"}], [], {}, "test")

    def test_atomic_publication_uses_shared_read_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "artifact.json"
            batch = AtomicBatch()
            staged = batch.path_for(destination)
            staged.write_text("{}\n", encoding="utf-8")
            batch.commit()
            self.assertEqual(
                stat.S_IMODE(destination.stat().st_mode),
                0o644,
            )

    def test_training_evidence_is_rekeyed_consistently_and_globally(self) -> None:
        old_registry_id = "ev_reg_zzzzzz"
        old_graph_id = "ev_mem_yyyyyy"
        first = {
            "messages": [
                {
                    "role": "harness",
                    "content": f'{{"evidence_id":"{old_registry_id}"}}',
                },
                {
                    "role": "tool",
                    "content": {
                        "provenance": {
                            "registry": {
                                "evidence_id": old_registry_id,
                            },
                            "graph": {
                                "evidence_id": old_graph_id,
                            },
                        }
                    },
                },
                {
                    "role": "assistant",
                    "content": {
                        "provenance": {
                            "registry": {
                                "evidence_id": old_registry_id,
                            },
                            "graph": {
                                "evidence_id": old_graph_id,
                            },
                        }
                    },
                },
            ]
        }
        second = {
            "messages": [
                {
                    "role": "tool",
                    "content": {"evidence_id": old_registry_id},
                }
            ]
        }
        original = copy.deepcopy(first)
        evidence_ids = EvidenceSequence()

        normalized_first = rekey_training_evidence(first, evidence_ids)
        normalized_second = rekey_training_evidence(second, evidence_ids)

        self.assertEqual(first, original)
        messages = normalized_first["messages"]
        harness_id = json.loads(messages[0]["content"])["evidence_id"]
        tool_provenance = messages[1]["content"]["provenance"]
        answer_provenance = messages[2]["content"]["provenance"]
        registry_ids = {
            harness_id,
            tool_provenance["registry"]["evidence_id"],
            answer_provenance["registry"]["evidence_id"],
        }
        graph_ids = {
            tool_provenance["graph"]["evidence_id"],
            answer_provenance["graph"]["evidence_id"],
        }
        self.assertEqual(len(registry_ids), 1)
        self.assertEqual(len(graph_ids), 1)
        new_registry_id = registry_ids.pop()
        new_graph_id = graph_ids.pop()
        self.assertRegex(new_registry_id, r"^ev_reg_[a-z0-9]{6}$")
        self.assertRegex(new_graph_id, r"^ev_mem_[a-z0-9]{6}$")
        self.assertNotEqual(new_registry_id, old_registry_id)
        self.assertNotEqual(new_graph_id, old_graph_id)
        second_id = normalized_second["messages"][0]["content"]["evidence_id"]
        self.assertRegex(second_id, r"^ev_reg_[a-z0-9]{6}$")
        self.assertNotEqual(second_id, new_registry_id)

    def test_replay_normalization_strips_only_manifest_additions(self) -> None:
        contract = load_contract()
        metadata_contract = contract["rows"]["metadata"]
        metadata = {
            key: f"value-{key}"
            for key in (
                list(metadata_contract["required"])
                + list(metadata_contract["optional"])
            )
        }
        metadata.update(
            {
                key: f"private-{key}"
                for key in S0_V4_METADATA_ADDITIONS
            }
        )
        row = {"metadata": metadata}
        original = copy.deepcopy(row)

        normalized = normalize_s0_replay_row(
            row,
            contract,
            S0_V4_METADATA_ADDITIONS,
        )

        self.assertEqual(row, original)
        self.assertEqual(
            list(normalized["metadata"]),
            list(metadata_contract["required"])
            + list(metadata_contract["optional"]),
        )
        self.assertTrue(
            set(S0_V4_METADATA_ADDITIONS).isdisjoint(normalized["metadata"])
        )

    def test_replay_normalization_rejects_an_undeclared_extra(self) -> None:
        contract = load_contract()
        metadata = {
            key: f"value-{key}"
            for key in contract["rows"]["metadata"]["required"]
        }
        metadata["undeclared"] = "not allowed"

        with self.assertRaisesRegex(BundleBuildError, "unsupported"):
            normalize_s0_replay_row(
                {"metadata": metadata},
                contract,
                S0_V4_METADATA_ADDITIONS,
            )


if __name__ == "__main__":
    unittest.main()
