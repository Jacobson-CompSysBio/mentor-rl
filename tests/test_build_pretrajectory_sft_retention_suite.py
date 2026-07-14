import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import build_pretrajectory_sft_retention_suite as builder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(root: Path, *, count_per_family: int = 6) -> list[dict]:
    rows: list[dict] = []
    canonical: list[dict] = []
    for family in builder.REQUIRED_FAMILIES:
        for index in range(count_per_family):
            object_id = f"fact-{family}-{index}"
            gene_id = f"ENSG{index:011d}"
            symbol = f"SYM{index}_{family[:3]}"
            alias = f"ALIAS{index}_{family[:3]}"
            if family == "entity_symbol_to_ensembl":
                payload = {"alias": alias, "gene_id": gene_id, "symbol": symbol}
                question = f"Normalize `{alias}`."
            elif family == "entity_ensembl_to_symbol":
                payload = {"alias": alias, "gene_id": gene_id, "symbol": symbol}
                question = f"Decode `{gene_id}`."
            else:
                payload = {
                    "alias": alias,
                    "candidate_gene_ids": [gene_id, f"ENSG9{index:010d}"],
                }
                question = f"Resolve ambiguous `{alias}`."
            answer = json.dumps({"target": payload}, separators=(",", ":"), sort_keys=True)
            rows.append(
                {
                    "system": "Return exact JSON.",
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "record_id": f"row-{family}-{index}",
                        "schema_version": "pretrajectory-sft-v3",
                        "split": "train",
                        "question_family": family,
                        "canonical_object_id": object_id,
                        "oracle_fact_id": object_id,
                        "multiplex_id": "full_brain_multiplex_v1",
                    },
                }
            )
            canonical.append(
                {
                    "object_id": object_id,
                    "object_type": family,
                    "payload": payload,
                }
            )
    _write_jsonl(
        root / "curriculum" / "stage1_entity_schema" / "train.jsonl",
        rows,
    )
    _write_jsonl(root / "train.jsonl", rows)
    _write_jsonl(root / "canonical_objects.jsonl", canonical)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": "pretrajectory-sft-v3",
                "plan_hash": "plan-hash",
                "content_hash": "content-hash",
            }
        ),
        encoding="utf-8",
    )
    return rows


class RetentionSuiteBuilderTests(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            rows = _fixture(root)
            first = builder.build_retention_suite(
                dataset_root=root,
                out_dir=Path(temporary_directory) / "first",
                seed=17,
                samples_per_family=2,
            )
            first_rows = builder.read_jsonl(Path(first["suite"]["path"]))

            _write_jsonl(
                root / "curriculum" / "stage1_entity_schema" / "train.jsonl",
                list(reversed(rows)),
            )
            _write_jsonl(root / "train.jsonl", list(reversed(rows)))
            second = builder.build_retention_suite(
                dataset_root=root,
                out_dir=Path(temporary_directory) / "second",
                seed=17,
                samples_per_family=2,
            )
            second_rows = builder.read_jsonl(Path(second["suite"]["path"]))
            third = builder.build_retention_suite(
                dataset_root=root,
                out_dir=Path(temporary_directory) / "third",
                seed=19,
                samples_per_family=2,
            )
            third_rows = builder.read_jsonl(Path(third["suite"]["path"]))

        selected = lambda values: {row["metadata"]["source_record_id"] for row in values}
        self.assertEqual(selected(first_rows), selected(second_rows))
        self.assertNotEqual(selected(first_rows), selected(third_rows))
        self.assertEqual(
            Counter(row["metadata"]["question_family"] for row in first_rows),
            Counter({family: 2 for family in builder.REQUIRED_FAMILIES}),
        )

    def test_underfilled_family_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            _fixture(root, count_per_family=2)
            with self.assertRaisesRegex(builder.RetentionSuiteError, "underfilled|cannot satisfy"):
                builder.build_retention_suite(
                    dataset_root=root,
                    out_dir=Path(temporary_directory) / "out",
                    seed=3,
                    samples_per_family=3,
                )

    def test_rows_preserve_seen_facts_without_prompt_or_target_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            source_rows = _fixture(root, count_per_family=3)
            out_dir = Path(temporary_directory) / "out"
            manifest = builder.build_retention_suite(
                dataset_root=root,
                out_dir=out_dir,
                seed=5,
                samples_per_family=2,
            )
            suite_rows = builder.read_jsonl(Path(manifest["suite"]["path"]))

        source_by_id = {row["metadata"]["record_id"]: row for row in source_rows}
        training_prompts = {builder.normalized_prompt(row) for row in source_rows}
        self.assertTrue(manifest["passed"])
        self.assertFalse(manifest["official_readiness_eligible"])
        self.assertEqual(manifest["suite"]["row_count"], 6)
        self.assertEqual(manifest["audit"]["source_prompt_overlap_count"], 0)
        self.assertEqual(manifest["audit"]["hidden_target_leak_count"], 0)
        for row in suite_rows:
            metadata = row["metadata"]
            source = source_by_id[metadata["source_record_id"]]
            self.assertEqual(row["answer"], source["answer"])
            self.assertEqual(metadata["oracle_fact_id"], source["metadata"]["oracle_fact_id"])
            self.assertEqual(metadata["source_split"], "train")
            self.assertEqual(metadata["evaluation_regime"], builder.EVALUATION_REGIME)
            self.assertFalse(metadata["official_readiness_eligible"])
            self.assertNotIn(builder.normalized_prompt(row), training_prompts)

            payload = json.loads(row["answer"])["target"]
            family = metadata["question_family"]
            hidden = builder._hidden_targets(family, payload)
            self.assertTrue(all(not builder._contains_identifier(row["question"], value) for value in hidden))

    def test_canonical_family_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            _fixture(root, count_per_family=2)
            canonical_path = root / "canonical_objects.jsonl"
            canonical = builder.read_jsonl(canonical_path)
            canonical[0]["object_type"] = "wrong_family"
            _write_jsonl(canonical_path, canonical)
            with self.assertRaisesRegex(builder.RetentionSuiteError, "type mismatch"):
                builder.build_retention_suite(
                    dataset_root=root,
                    out_dir=Path(temporary_directory) / "out",
                    seed=7,
                    samples_per_family=2,
                )


if __name__ == "__main__":
    unittest.main()
