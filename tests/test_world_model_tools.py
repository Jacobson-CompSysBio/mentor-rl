"""Focused tests for the canonical S0/S1 identifier-tool runtime."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from runtime.world_model.contract import GOLDEN_TRACES_PATH, load_contract
from runtime.world_model.source_index import (
    GENE_FIELDS,
    HISTORICAL_SYMBOL_FIELDS,
    MENTOR_CLADE_FIELDS,
    RETIRED_GENE_FIELDS,
    SOURCE_INDEX_SCHEMA_VERSION,
    content_sha256,
    write_source_index,
)
from runtime.world_model.tools import ToolCallError, WorldModelToolRuntime


class EvidenceIds:
    """Return predictable six-character evidence suffixes."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, channel: str) -> str:
        self.count += 1
        prefix = "ev_reg_" if channel == "registry" else "ev_mem_"
        return prefix + f"{self.count:06d}"


class WorldModelToolRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.ids = tuple(f"ENSG{number:011d}" for number in range(1, 7))
        hashes = {
            "ensembl_gtf": "a" * 64,
            "ensembl_stable_id_history": "b" * 64,
            "hgnc_history": "c" * 64,
            "multiplex_flist": "d" * 64,
            "mentor_ev_dendrogram": "e" * 64,
        }
        self.payload = {
            "schema_version": SOURCE_INDEX_SCHEMA_VERSION,
            "sources": {
                name: {"file_name": f"{name}.fixture", "sha256": digest}
                for name, digest in hashes.items()
            },
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "tiny_registry",
                "multiplex_id": "tiny_multiplex",
                "dendrogram_id": "tiny_tree",
                "eligible_leaf_count": {"min": 2, "max": 3},
                "multiplex_gene_rule": "mentor_ev_all_gene_dendrogram_leaves",
            },
            "counts": {
                "registry_gene_count": 4,
                "registry_named_gene_count": 3,
                "registry_unnamed_gene_count": 1,
                "registry_symbol_count": 2,
                "registry_ambiguous_symbol_count": 1,
                "registry_largest_candidate_count": 2,
                "registry_chromosome_count": 4,
                "registry_biotype_count": 3,
                "registry_casefold_collision_count": 1,
                "historical_symbol_count": 1,
                "historical_symbol_key_count": 1,
                "historical_symbol_linked_count": 1,
                "retired_gene_count": 1,
                "retired_gene_linked_count": 1,
                "layer_count": 1,
                "multiplex_gene_count": 3,
                "registry_multiplex_intersection_count": 2,
                "mentor_node_count": 5,
                "mentor_leaf_count": 3,
                "mentor_eligible_clade_count": 1,
            },
            "gene_fields": GENE_FIELDS,
            "genes_by_id": {
                self.ids[0]: ["Alpha", "1", "protein_coding"],
                self.ids[1]: ["Dup", "2", "lncRNA"],
                self.ids[2]: ["dup", "X", "pseudogene"],
                self.ids[3]: [None, "MT", "protein_coding"],
            },
            "historical_symbol_fields": HISTORICAL_SYMBOL_FIELDS,
            "historical_symbols": [
                ["OldAlpha", ["previous"], [self.ids[0]]]
            ],
            "retired_gene_fields": RETIRED_GENE_FIELDS,
            "retired_genes": [[self.ids[5], [self.ids[0]]]],
            "multiplex_gene_ids": [self.ids[0], self.ids[2], self.ids[4]],
            "layer_ids": ["HumanNetV3:string_ppi"],
            "mentor_clade_fields": MENTOR_CLADE_FIELDS,
            "mentor_clades": [[10, 3]],
        }
        self.index_path = write_source_index(self.payload, self.root)
        self.contract = {
            "source_index": {
                "content_sha256": content_sha256(self.payload),
            },
            "coordinate_system": {
                "ensembl_release": "116",
                "identifier_registry_id": "tiny_registry",
                "multiplex_id": "tiny_multiplex",
                "source_flist_sha256": hashes["multiplex_flist"],
                "identifier_registry": {
                    "source_sha256": hashes["ensembl_gtf"],
                    "stable_id_history": {
                        "source_sha256": hashes["ensembl_stable_id_history"]
                    },
                    "hgnc_history": {"source_sha256": hashes["hgnc_history"]},
                },
                "mentor_ev": {
                    "sha256": hashes["mentor_ev_dendrogram"],
                    "dendrogram_id": "tiny_tree",
                    "eligible_leaf_count": {"min": 2, "max": 3},
                },
            },
            "provenance": {
                "channels": ["registry", "graph"],
                "registry_block": {
                    "registry": "ensembl",
                    "evidence_id_pattern": r"^ev_reg_[a-z0-9]{6}$",
                },
                "graph_block": {
                    "layer_scope": "all",
                    "evidence_id_pattern": r"^ev_mem_[a-z0-9]{6}$",
                },
            },
        }
        self.runtime = WorldModelToolRuntime.from_source_index(
            self.index_path,
            contract=self.contract,
            evidence_id_factory=EvidenceIds(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_verified_index_sets_runtime_identity(self) -> None:
        self.assertEqual(self.runtime.ensembl_release, "116")
        self.assertEqual(self.runtime.identifier_registry_id, "tiny_registry")
        self.assertEqual(self.runtime.multiplex_id, "tiny_multiplex")
        self.assertEqual(self.runtime.graph_version, "d" * 64)
        self.assertEqual(
            self.runtime.source_index_sha256, content_sha256(self.payload)
        )

        wrong = deepcopy(self.contract)
        wrong["source_index"]["content_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "content SHA-256"):
            WorldModelToolRuntime.from_source_index(
                self.index_path, contract=wrong
            )

    def test_symbol_lookup_normalizes_and_returns_fresh_exact_shapes(self) -> None:
        resolved = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": "  aLpHa  "}
        )
        self.assertEqual(
            list(resolved.output),
            ["status", "provenance", "gene_id", "gene_symbol"],
        )
        self.assertEqual(resolved.output["gene_symbol"], "Alpha")
        self.assertEqual(resolved.output["gene_id"], self.ids[0])
        self.assertEqual(list(resolved.output["provenance"]), ["registry", "graph"])
        self.assertIsNone(resolved.output["provenance"]["graph"])

        ambiguous = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": " DUP "}
        ).output
        self.assertEqual(
            list(ambiguous),
            [
                "status",
                "provenance",
                "gene_symbol",
                "candidate_gene_ids",
                "candidate_details",
            ],
        )
        self.assertEqual(ambiguous["gene_symbol"], "DUP")
        self.assertEqual(ambiguous["candidate_gene_ids"], [self.ids[1], self.ids[2]])
        self.assertEqual(
            list(ambiguous["candidate_details"]), [self.ids[1], self.ids[2]]
        )
        ambiguous["candidate_gene_ids"].clear()
        again = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": "dup"}
        ).output
        self.assertEqual(again["candidate_gene_ids"], [self.ids[1], self.ids[2]])

        missing = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": " Missing "}
        ).output
        self.assertEqual(
            list(missing),
            ["status", "provenance", "gene_symbol", "candidate_gene_ids"],
        )
        self.assertEqual(missing["gene_symbol"], "Missing")
        self.assertEqual(missing["candidate_gene_ids"], [])

    def test_gene_id_lookup_covers_every_semantic_outcome(self) -> None:
        cases = (
            (self.ids[0], "resolved", self.ids[0], "Alpha"),
            (f"  {self.ids[0].lower()}.23  ", "resolved", self.ids[0], "Alpha"),
            (self.ids[3], "resolved", self.ids[3], None),
            (self.ids[5], "not_found", self.ids[5], None),
            (f"{self.ids[5]}.14_PAR_Y", "not_found", f"{self.ids[5]}.14_PAR_Y", None),
        )
        for query, status, gene_id, symbol in cases:
            with self.subTest(query=query):
                output = self.runtime.execute(
                    "lookup_human_ensembl_gene_id", {"gene_id": query}
                ).output
                self.assertEqual(
                    list(output),
                    ["status", "provenance", "gene_id", "gene_symbol"],
                )
                self.assertEqual(
                    (output["status"], output["gene_id"], output["gene_symbol"]),
                    (status, gene_id, symbol),
                )

        invalid = self.runtime.execute(
            "lookup_human_ensembl_gene_id", {"gene_id": "ENSG123"}
        ).output
        self.assertEqual(
            list(invalid), ["status", "provenance", "input", "expected"]
        )
        self.assertEqual(invalid["status"], "invalid_input")
        self.assertEqual(invalid["expected"], "human_gene_id")

    def test_membership_partitions_in_request_order_and_handles_unknown(self) -> None:
        result = self.runtime.execute(
            "check_multiplex_membership",
            {
                "multiplex_id": "tiny_multiplex",
                "gene_ids": [
                    self.ids[4],
                    self.ids[1],
                    self.ids[0],
                    self.ids[4],
                ],
            },
        )
        self.assertEqual(
            list(result.output),
            [
                "status",
                "provenance",
                "multiplex_id",
                "present_gene_ids",
                "absent_gene_ids",
            ],
        )
        self.assertEqual(result.output["present_gene_ids"], [self.ids[4], self.ids[0]])
        self.assertEqual(result.output["absent_gene_ids"], [self.ids[1]])
        graph = result.output["provenance"]["graph"]
        self.assertEqual(graph["multiplex_id"], "tiny_multiplex")
        self.assertEqual(graph["graph_version"], "d" * 64)
        self.assertIsNone(result.output["provenance"]["registry"])

        unknown = self.runtime.execute(
            "check_multiplex_membership",
            {"multiplex_id": "unknown", "gene_ids": [self.ids[0]]},
        ).output
        self.assertEqual(list(unknown), ["status", "provenance", "multiplex_id"])
        self.assertEqual(unknown["status"], "not_found")
        self.assertEqual(unknown["provenance"]["graph"]["multiplex_id"], "unknown")

    def test_evidence_is_unique_private_and_failures_create_none(self) -> None:
        first = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": "Alpha"}
        )
        second = self.runtime.execute(
            "lookup_human_gene_symbol", {"gene_symbol": "Alpha"}
        )
        first_id = first.output["provenance"]["registry"]["evidence_id"]
        second_id = second.output["provenance"]["registry"]["evidence_id"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first.evidence["evidence_id"], first_id)
        self.assertEqual(
            first.evidence["source_index_sha256"], content_sha256(self.payload)
        )

        failure = self.runtime.execute(
            "check_multiplex_membership",
            {"multiplex_id": "tiny_multiplex", "gene_ids": [self.ids[0]]},
            inject_backend_failure=True,
        )
        self.assertEqual(
            failure.output,
            {
                "status": "error",
                "provenance": None,
                "error_code": "BACKEND_UNAVAILABLE",
                "message": "graph service unreachable",
            },
        )
        self.assertIsNone(failure.evidence)

    def test_call_contract_errors_are_not_wire_results(self) -> None:
        bad_calls = (
            ("unknown", {}),
            ("lookup_human_gene_symbol", {}),
            ("lookup_human_gene_symbol", {"gene_symbol": "Alpha", "extra": 1}),
            ("lookup_human_gene_symbol", {"gene_symbol": " "}),
            ("lookup_human_ensembl_gene_id", {"gene_id": 3}),
            (
                "check_multiplex_membership",
                {"multiplex_id": "tiny_multiplex", "gene_ids": []},
            ),
            (
                "check_multiplex_membership",
                {"multiplex_id": "tiny_multiplex", "gene_ids": ["ENSG123"]},
            ),
        )
        for name, arguments in bad_calls:
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(ToolCallError):
                    self.runtime.execute(name, arguments)

    def test_injected_factory_must_be_valid_and_unique(self) -> None:
        invalid = WorldModelToolRuntime.from_source_index(
            self.index_path,
            contract=self.contract,
            evidence_id_factory=lambda channel: "bad",
        )
        with self.assertRaisesRegex(ToolCallError, "invalid registry ID"):
            invalid.execute(
                "lookup_human_gene_symbol", {"gene_symbol": "Alpha"}
            )

        duplicate = WorldModelToolRuntime.from_source_index(
            self.index_path,
            contract=self.contract,
            evidence_id_factory=lambda channel: "ev_reg_repeat",
        )
        duplicate.execute("lookup_human_gene_symbol", {"gene_symbol": "Alpha"})
        with self.assertRaisesRegex(ToolCallError, "was reused"):
            duplicate.execute(
                "lookup_human_gene_symbol", {"gene_symbol": "Alpha"}
            )


class WorldModelToolRealIndexTests(unittest.TestCase):
    def test_every_golden_tool_result_replays_from_the_pinned_index(self) -> None:
        contract = load_contract()
        digest = contract["source_index"]["content_sha256"]
        path = (
            Path("data/world_model_v3/source_indexes")
            / f"world_model_source_index_{digest}.json.gz"
        )
        if not path.is_file():
            self.skipTest(f"generated source index is absent: {path}")

        rows = [
            json.loads(line)
            for line in GOLDEN_TRACES_PATH.read_text().splitlines()
            if line
        ]
        evidence_ids: list[tuple[str, str]] = []
        for row in rows:
            for message in row["messages"]:
                if message.get("role") != "tool":
                    continue
                output = message["content"]
                if output["status"] == "error":
                    continue
                provenance = output["provenance"]
                channel = "registry" if provenance["registry"] else "graph"
                evidence_ids.append(
                    (channel, provenance[channel]["evidence_id"])
                )

        evidence_iterator = iter(evidence_ids)

        def next_evidence_id(channel: str) -> str:
            expected_channel, evidence_id = next(evidence_iterator)
            self.assertEqual(expected_channel, channel)
            return evidence_id

        runtime = WorldModelToolRuntime.from_source_index(
            path,
            contract=contract,
            evidence_id_factory=next_evidence_id,
        )
        replayed = 0
        for row in rows:
            pending_call = None
            for message in row["messages"]:
                if message.get("role") == "assistant" and "tool_call" in message:
                    pending_call = message["tool_call"]
                elif message.get("role") == "tool":
                    self.assertIsNotNone(pending_call)
                    expected = message["content"]
                    actual = runtime.execute(
                        pending_call["name"],
                        pending_call["arguments"],
                        inject_backend_failure=expected["status"] == "error",
                    ).output
                    self.assertEqual(
                        actual,
                        expected,
                        row["metadata"]["source_trace"],
                    )
                    replayed += 1
                    pending_call = None
        self.assertEqual(replayed, 45)


if __name__ == "__main__":
    unittest.main()
