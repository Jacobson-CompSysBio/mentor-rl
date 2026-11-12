"""Check inclusive context filtering and rejection of stale length caches."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_world_model_s2_context import audit_corpus, load_verified_lengths


class S2ContextAuditTests(unittest.TestCase):
    """Use pinned small corpora to test the cache trust boundary."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "candidate"
        self.directory.mkdir()
        (self.root / "serializer.py").write_text("# pinned serializer\n")
        (self.root / "tokenizer").mkdir()
        (self.root / "tokenizer/tokenizer.json").write_text("{}\n")
        records = []
        for split, tokens in (("train", 8192), ("validation", 8193), ("test", 4000)):
            row = {"row_id": split, "metadata": {"question_family": "S2.4", "split": split,
                "variant": "lookup", "framing": "direct", "unique_neighbor_count": 10}}
            (self.directory / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            records.append({"row_id": split, "tokens": tokens})
        audit = {"tokenizer": {"current_turn_masks_checked": True, "truncation_applied": False,
            "rows": 3, "row_counts": records, "row_token_counts_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}}
        (self.directory / "audit.json").write_text(json.dumps(audit))
        self.manifest = {"files": {name: self.pin(self.directory / name) for name in (
            "audit.json", "train.jsonl", "validation.jsonl", "test.jsonl")},
            "row_counts": {split: 1 for split in ("train", "validation", "test")},
            "implementation_sha256": {"serializer.py": self.pin(self.root / "serializer.py")["sha256"]},
            "tokenizer": {"path": "tokenizer", "artifacts": {
                "tokenizer.json": self.pin(self.root / "tokenizer/tokenizer.json")["sha256"]}}}
        self.write_manifest()

    @staticmethod
    def pin(path):
        """Make a source pin from exact fixture bytes."""
        data = path.read_bytes()
        return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def write_manifest(self):
        """Publish updated fixture pins when testing a deeper validation layer."""
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def test_limit_is_inclusive_and_only_long_row_is_removed(self):
        result = audit_corpus("S2.4", self.directory, 8192, self.root)
        self.assertEqual(result["totals"]["retained"], 2)
        self.assertEqual(result["excluded_row_ids"], ["validation"])
        self.assertEqual(result["behaviors"][0]["observed_coverage"], "partial_source_range")
        self.assertEqual(result["splits"]["train"]["retained"], 1)

    def test_source_byte_change_rejects_cached_lengths(self):
        with (self.directory / "train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed corpus file"):
            load_verified_lengths(self.directory, self.root)

    def test_serializer_change_rejects_cached_lengths(self):
        (self.root / "serializer.py").write_text("# revised serializer\n")
        with self.assertRaisesRegex(ValueError, "changed pinned implementation"):
            load_verified_lengths(self.directory, self.root)

    def test_tampered_token_records_rejected_after_source_repin(self):
        path = self.directory / "audit.json"
        audit = json.loads(path.read_text())
        audit["tokenizer"]["row_counts"][0]["tokens"] = 1
        path.write_text(json.dumps(audit))
        self.manifest["files"]["audit.json"] = self.pin(path)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "changed cached token records"):
            load_verified_lengths(self.directory, self.root)

    def test_no_fitting_observed_behavior_is_not_universal_impossibility(self):
        path = self.directory / "train.jsonl"
        row = json.loads(path.read_text())
        row["metadata"]["variant"] = "context"
        path.write_text(json.dumps(row) + "\n")
        self.manifest["files"]["train.jsonl"] = self.pin(path)
        path = self.directory / "test.jsonl"
        row = json.loads(path.read_text())
        row["metadata"]["variant"] = "context"
        path.write_text(json.dumps(row) + "\n")
        self.manifest["files"]["test.jsonl"] = self.pin(path)
        self.write_manifest()
        result = audit_corpus("S2.4", self.directory, 8192, self.root)
        missing = [behavior for behavior in result["behaviors"] if not behavior["retained"]]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["observed_coverage"], "no_fitting_verification_witness")
        self.assertEqual(result["totals"]["retained"], 2)

    def test_distinct_claims_keep_separate_coverage(self):
        for split, claim in (("train", "causal_effect"), ("validation", "physical_interaction_recorded"),
                             ("test", "causal_effect")):
            path = self.directory / f"{split}.jsonl"
            row = json.loads(path.read_text())
            row["metadata"]["claim"] = claim
            path.write_text(json.dumps(row) + "\n")
            self.manifest["files"][path.name] = self.pin(path)
        self.write_manifest()
        result = audit_corpus("S2.4", self.directory, 8192, self.root)
        coverage = {record["behavior"]["claim"]: record for record in result["behaviors"]}
        self.assertEqual(coverage["causal_effect"]["retained"], 2)
        self.assertEqual(coverage["physical_interaction_recorded"]["observed_coverage"],
                         "no_fitting_verification_witness")


if __name__ == "__main__":
    unittest.main()
