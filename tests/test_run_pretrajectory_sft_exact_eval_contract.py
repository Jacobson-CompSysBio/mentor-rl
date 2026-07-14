import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_pretrajectory_sft_retention_suite as retention_builder
from tests.test_build_pretrajectory_sft_retention_suite import _fixture as _retention_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_pretrajectory_sft_exact_eval.py"


def _load_runner():
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.bfloat16 = object()
    torch.distributed = types.ModuleType("torch.distributed")

    peft = types.ModuleType("peft")
    peft.PeftModel = object

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.HfArgumentParser = object

    stubs = {
        "torch": torch,
        "torch.distributed": torch.distributed,
        "peft": peft,
        "transformers": transformers,
    }
    spec = importlib.util.spec_from_file_location("pretrajectory_exact_eval_runner_for_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _write_v3_dataset(root: Path) -> tuple[Path, Path]:
    rows = {
        split: {
            "system": "system",
            "question": f"question {split}",
            "answer": '{"ok":true}',
            "metadata": {
                "record_id": f"row-{split}",
                "schema_version": "pretrajectory-sft-v3",
                "split": split,
            },
        }
        for split in ("train", "val", "test")
    }
    for split, row in rows.items():
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    content_hash = runner._stable_json_hash(
        sorted(rows.values(), key=lambda row: row["metadata"]["record_id"])
    )
    plan = {"dataset_schema_version": "pretrajectory-sft-v3", "plan_id": "test-plan"}
    (root / "curriculum_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = runner._stable_json_hash(plan)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "pretrajectory-sft-curriculum-artifacts-v1",
                "dataset_schema_version": "pretrajectory-sft-v3",
                "plan_hash": plan_hash,
                "content_hash": content_hash,
            }
        ),
        encoding="utf-8",
    )
    audit_path = root / "audit_report_contract_v3.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "pretrajectory-sft-audit-v3",
                "dataset_schema_version": "pretrajectory-sft-v3",
                "passed": True,
                "fatal_error_count": 0,
                "warning_count": 0,
                "plan_hash": plan_hash,
                "content_hash": content_hash,
                "native_reports": {
                    name: {"passed": True, "plan_hash": plan_hash}
                    for name in runner.CURRICULUM_NATIVE_REPORT_NAMES
                },
                "answer_budget_report": {
                    "manifest_contract_present": True,
                    "over_budget_record_count": 0,
                    "missing_record_budget_metadata_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return root / "val.jsonl", audit_path


class ExactEvalPipelineContractTests(unittest.TestCase):
    def test_stratified_selection_is_balanced_deterministic_and_strict(self) -> None:
        rows = [
            {
                "metadata": {
                    "record_id": f"{family}-{index}",
                    "question_family": family,
                }
            }
            for family in runner.DEFAULT_RETENTION_FAMILIES
            for index in range(5)
        ]
        selected = runner._select_rows(
            rows,
            sample_size=0,
            seed=11,
            question_families=runner.DEFAULT_RETENTION_FAMILIES,
            samples_per_family=2,
        )
        selected_again = runner._select_rows(
            list(reversed(rows)),
            sample_size=0,
            seed=11,
            question_families=runner.DEFAULT_RETENTION_FAMILIES,
            samples_per_family=2,
        )
        ids = {row["metadata"]["record_id"] for _, row in selected}
        ids_again = {row["metadata"]["record_id"] for _, row in selected_again}
        counts = {}
        for _, row in selected:
            family = row["metadata"]["question_family"]
            counts[family] = counts.get(family, 0) + 1

        self.assertEqual(ids, ids_again)
        self.assertEqual(counts, {family: 2 for family in runner.DEFAULT_RETENTION_FAMILIES})
        with self.assertRaisesRegex(runner.ExactEvalContractError, "underfilled"):
            runner._select_rows(
                rows,
                sample_size=0,
                seed=11,
                question_families=runner.DEFAULT_RETENTION_FAMILIES,
                samples_per_family=6,
            )

    def test_retention_suite_contract_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            _retention_fixture(root, count_per_family=3)
            out_dir = Path(temporary_directory) / "suite"
            manifest = retention_builder.build_retention_suite(
                dataset_root=root,
                out_dir=out_dir,
                seed=13,
                samples_per_family=2,
            )
            suite_path = Path(manifest["suite"]["path"])
            manifest_path = out_dir / "suite_manifest.json"
            rows = retention_builder.read_jsonl(suite_path)

            contract = runner._retention_suite_contract(
                manifest_path=manifest_path,
                dataset_path=suite_path,
                rows=rows,
                source_dataset_root=root,
            )
            self.assertTrue(contract["valid"])

            suite_path.write_text(
                suite_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            tampered = runner._retention_suite_contract(
                manifest_path=manifest_path,
                dataset_path=suite_path,
                rows=rows,
                source_dataset_root=root,
            )

        self.assertFalse(tampered["valid"])
        self.assertIn("retention_suite_sha256_mismatch", tampered["failures"])

    def test_retention_report_is_family_gated_and_template_stratified(self) -> None:
        prediction_rows = []
        evaluated = []
        for family in runner.DEFAULT_RETENTION_FAMILIES:
            for index in range(2):
                prediction_rows.append(
                    {
                        "metadata": {
                            "question_family": family,
                            "oracle_fact_id": f"{family}-{index}",
                            "source_oracle_fact_id": f"{family}-{index}",
                            "retention_template_id": f"template-{index}",
                        }
                    }
                )
                evaluated.append(
                    {
                        "question_family": family,
                        "exact_graph_fact_pass": True,
                        "id_recall": 1.0,
                    }
                )

        with mock.patch.object(runner, "evaluate_row", side_effect=evaluated):
            report = runner._build_retention_report(
                prediction_rows=prediction_rows,
                canonical_objects={},
                regime=runner.SEEN_FACT_RETENTION_REGIME,
                question_families=runner.DEFAULT_RETENTION_FAMILIES,
                samples_per_family=2,
                minimum_family_exact=0.80,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["decision"], "strong_memorization_signal")
        self.assertFalse(report["official_readiness_eligible"])
        self.assertEqual(report["unique_source_fact_count"], 6)
        self.assertEqual(set(report["by_prompt_template_id"]), {"template-0", "template-1"})

    def test_diagnostic_readiness_is_explicitly_not_applicable(self) -> None:
        report = runner._diagnostic_readiness(
            regime=runner.SEEN_FACT_RETENTION_REGIME,
            retention_report_path=Path("retention_report.json"),
        )
        self.assertTrue(report["valid"])
        self.assertIsNone(report["passed"])
        self.assertFalse(report["official_readiness_eligible"])
        self.assertEqual(report["decision"], "diagnostic_only_no_readiness_decision")

    def test_diagnostic_run_does_not_call_or_coerce_official_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            out_dir = root / "eval"
            model_path = root / "model"
            model_path.mkdir()
            dataset_path, audit_path = _write_v3_dataset(root)
            canonical_path = root / "canonical_objects.jsonl"
            canonical_path.write_text("", encoding="utf-8")
            args = types.SimpleNamespace(
                out_dir=out_dir,
                dataset_path=dataset_path,
                sample_size=0,
                seed=7,
                canonical_objects=canonical_path,
                dataset_audit=audit_path,
                source_dataset_root=root,
                evaluation_regime="unseen_fact_generalization",
                question_families="",
                samples_per_family=0,
                retention_suite_manifest=None,
                minimum_diagnostic_family_exact=0.80,
                adapter_path=None,
                model_source_path=None,
                model_path=model_path,
                max_new_tokens=32,
                max_total_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                enable_thinking=False,
                reasoning_effort="",
                local_files_only=True,
                trust_remote_code=False,
                max_report_examples=10,
                progress_every=0,
            )
            passing_gold_contract = {
                "contract_version": "pretrajectory-sft-exact-v3",
                "status": "passed",
                "passed": True,
                "failure_count": 0,
            }
            exact_report = {
                "sample_count": 0,
                "evaluator_contract": {"version": "pretrajectory-sft-exact-v3"},
                "gold_self_evaluation": {"passed": True},
                "summary": {
                    "exact_graph_fact_pass_rate": 0.0,
                    "unsupported_language_rate": 0.0,
                    "exact_only": {
                        "mean_id_recall": 0.0,
                        "mean_layer_recall": 0.0,
                        "mean_number_recall": 0.0,
                    },
                },
                "by_context_mode": {},
                "by_mixture_bucket": {},
                "examples": [],
            }
            fake_model = types.SimpleNamespace(
                parameters=lambda: iter([types.SimpleNamespace(device="cpu")])
            )
            with (
                mock.patch.object(runner, "_select_rows", return_value=[]),
                mock.patch.object(runner, "load_canonical_objects", return_value={}),
                mock.patch.object(runner, "evaluate_gold_self_contract", return_value=passing_gold_contract),
                mock.patch.object(runner, "_tokenizer", return_value=object()),
                mock.patch.object(runner, "_load_model", return_value=fake_model),
                mock.patch.object(runner, "build_formatting_func", return_value=lambda row: ""),
                mock.patch.object(runner.torch, "no_grad", return_value=contextlib.nullcontext(), create=True),
                mock.patch.object(runner, "evaluate_prediction_rows", return_value=exact_report),
                mock.patch.object(runner, "render_html_report", return_value="<html></html>"),
                mock.patch.object(runner, "_run_readiness_checker") as official_readiness,
            ):
                summary = runner.run_exact_eval(args)

            readiness = json.loads((out_dir / "readiness_report.json").read_text(encoding="utf-8"))
            saved_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))

        official_readiness.assert_not_called()
        self.assertTrue(summary["valid"])
        self.assertIsNone(saved_summary["readiness"]["passed"])
        self.assertIsNone(readiness["passed"])
        self.assertEqual(readiness["decision"], "diagnostic_only_no_readiness_decision")

    def test_v3_dataset_audit_contract_accepts_fresh_passed_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))

            contract = runner._dataset_audit_contract(
                audit_path,
                dataset_path=dataset_path,
            )
            discovered_audit = runner._find_dataset_audit(dataset_path)

        self.assertTrue(contract["valid"])
        self.assertEqual(contract["required_schema_version"], "pretrajectory-sft-audit-v3")
        self.assertTrue(contract["native_reports_valid"])
        self.assertTrue(contract["dataset_identity"]["valid"])
        self.assertEqual(discovered_audit, audit_path)
        self.assertEqual(contract["supported_schema_versions"], ["pretrajectory-sft-audit-v3"])

    def test_dataset_audit_discovery_does_not_fallback_to_legacy_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path, audit_path = _write_v3_dataset(root)
            audit_path.unlink()
            (root / "audit_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "fatal_error_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            discovered_audit = runner._find_dataset_audit(dataset_path)

        self.assertIsNone(discovered_audit)

    def test_v3_dataset_audit_contract_rejects_stale_rows_and_native_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path, audit_path = _write_v3_dataset(root)
            changed = json.loads(dataset_path.read_text(encoding="utf-8"))
            changed["answer"] = '{"ok":false}'
            dataset_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["native_reports"]["leakage_report.json"]["passed"] = False
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            contract = runner._dataset_audit_contract(
                audit_path,
                dataset_path=dataset_path,
            )

        self.assertFalse(contract["valid"])
        self.assertIn("dataset_content_hash_mismatch", contract["failures"])
        self.assertIn(
            "native_report_not_passed:leakage_report.json",
            contract["failures"],
        )

    def test_invalid_dataset_audit_writes_contract_artifacts_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            out_dir = root / "eval"
            model_path = root / "model"
            model_path.mkdir()
            dataset_path, audit_path = _write_v3_dataset(root)
            example = json.loads(dataset_path.read_text(encoding="utf-8"))
            canonical_path = root / "canonical_objects.jsonl"
            canonical_path.write_text("{}\n", encoding="utf-8")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["passed"] = False
            audit["fatal_error_count"] = 2
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            args = types.SimpleNamespace(
                out_dir=out_dir,
                dataset_path=dataset_path,
                sample_size=1,
                seed=7,
                canonical_objects=canonical_path,
                dataset_audit=audit_path,
                adapter_path=None,
                model_source_path=None,
                model_path=model_path,
                max_new_tokens=32,
                max_total_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                enable_thinking=False,
                reasoning_effort="",
                local_files_only=True,
                trust_remote_code=False,
                max_report_examples=10,
            )
            passing_gold_contract = {
                "contract_version": "pretrajectory-sft-exact-v3",
                "status": "passed",
                "passed": True,
                "failure_count": 0,
            }

            with (
                mock.patch.object(runner, "_select_rows", return_value=[(0, example)]),
                mock.patch.object(runner, "load_canonical_objects", return_value={}),
                mock.patch.object(runner, "evaluate_gold_self_contract", return_value=passing_gold_contract),
                mock.patch.object(runner, "_tokenizer") as load_tokenizer,
                mock.patch.object(runner, "_load_model") as load_model,
            ):
                with self.assertRaises(runner.ExactEvalContractError):
                    runner.run_exact_eval(args)

            load_tokenizer.assert_not_called()
            load_model.assert_not_called()
            run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
            readiness = json.loads((out_dir / "readiness_report.json").read_text(encoding="utf-8"))
            self.assertEqual(run_summary["status"], "invalid_dataset_contract")
            self.assertFalse(run_summary["valid"])
            self.assertEqual(readiness["decision"], "invalid_dataset_contract")
            self.assertTrue((out_dir / "gold_reference_report.json").is_file())

    def test_dataset_audit_contract_requires_zero_fatal_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["warning_count"] = 2
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            contract = runner._dataset_audit_contract(audit_path, dataset_path=dataset_path)
            self.assertTrue(contract["valid"])
            self.assertEqual(contract["fatal_error_count"], 0)

            audit["passed"] = False
            audit["fatal_error_count"] = 3
            audit["warning_count"] = 0
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            contract = runner._dataset_audit_contract(audit_path, dataset_path=dataset_path)
            self.assertFalse(contract["valid"])
            self.assertIn("nonzero_fatal_error_count", contract["failures"])

            readiness = runner._invalid_readiness(
                decision="invalid_dataset_contract",
                reason=contract["message"],
                gold_report_path=Path(temporary_directory) / "gold_reference_report.json",
                gate_name="dataset_audit_contract_valid",
                dataset_audit_path=audit_path,
            )
            self.assertEqual(readiness["decision"], "invalid_dataset_contract")
            self.assertEqual(readiness["failed_required_gates"][0]["name"], "dataset_audit_contract_valid")
            self.assertEqual(readiness["dataset_audit"], str(audit_path))

    def test_dataset_audit_contract_rejects_missing_fatal_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.pop("fatal_error_count")
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            contract = runner._dataset_audit_contract(audit_path, dataset_path=dataset_path)

            self.assertFalse(contract["valid"])
            self.assertIn("missing_or_invalid_fatal_error_count", contract["failures"])

    def test_dataset_audit_contract_rejects_legacy_audit_schemas(self) -> None:
        for stale_schema in (
            "pretrajectory-sft-audit-v1",
            "pretrajectory-sft-audit-v2",
        ):
            with self.subTest(stale_schema=stale_schema):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    audit["schema_version"] = stale_schema
                    audit_path.write_text(json.dumps(audit), encoding="utf-8")

                    contract = runner._dataset_audit_contract(
                        audit_path,
                        dataset_path=dataset_path,
                    )

                self.assertFalse(contract["valid"])
                self.assertEqual(contract["schema_version"], stale_schema)
                self.assertIn("unsupported_dataset_audit_schema_version", contract["failures"])

    def test_dataset_audit_contract_rejects_stale_dataset_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["dataset_schema_version"] = "pretrajectory-sft-v2"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            contract = runner._dataset_audit_contract(
                audit_path,
                dataset_path=dataset_path,
            )

        self.assertFalse(contract["valid"])
        self.assertIn("unsupported_dataset_schema_version", contract["failures"])

    def test_dataset_audit_contract_requires_manifest_and_zero_over_budget_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path, audit_path = _write_v3_dataset(Path(temporary_directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["answer_budget_report"]["manifest_contract_present"] = False
            audit["answer_budget_report"]["over_budget_record_count"] = 4
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            contract = runner._dataset_audit_contract(audit_path, dataset_path=dataset_path)

            self.assertFalse(contract["valid"])
            self.assertIn("answer_budget_manifest_contract_not_present", contract["failures"])
            self.assertIn("nonzero_over_budget_record_count", contract["failures"])

    def test_model_report_contract_requires_embedded_gold_self_pass(self) -> None:
        report = {
            "sample_count": 2,
            "evaluator_contract": {"version": "pretrajectory-sft-exact-v3"},
            "gold_self_evaluation": {"passed": True},
            "summary": {
                "exact_graph_fact_pass_rate": 0.5,
                "mean_id_recall": 0.8,
                "mean_layer_recall": 0.7,
                "unsupported_language_rate": 0.0,
                "exact_only": {
                    "mean_id_recall": 0.8,
                    "mean_layer_recall": 0.7,
                    "mean_number_recall": 0.8,
                },
            },
        }
        contract = runner._model_report_contract(report, expected_count=2)
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["failures"], [])

        report["evaluator_contract"]["version"] = "pretrajectory-sft-exact-v2"
        contract = runner._model_report_contract(report, expected_count=2)
        self.assertFalse(contract["valid"])
        self.assertIn("unsupported_evaluator_contract_version", contract["failures"])

        report["evaluator_contract"]["version"] = "pretrajectory-sft-exact-v3"
        report["summary"]["exact_only"].pop("mean_number_recall")
        contract = runner._model_report_contract(report, expected_count=2)
        self.assertFalse(contract["valid"])
        self.assertIn("missing_exact_only_metric:mean_number_recall", contract["failures"])

        report["summary"]["exact_only"]["mean_number_recall"] = 0.8
        report["gold_self_evaluation"]["passed"] = False
        contract = runner._model_report_contract(report, expected_count=2)
        self.assertFalse(contract["valid"])
        self.assertIn("gold_self_evaluation_not_passed", contract["failures"])

    def test_invalid_readiness_contract_is_not_completed_or_rewritten_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            out_dir = root / "eval"
            model_path = root / "model"
            model_path.mkdir()
            dataset_path, audit_path = _write_v3_dataset(root)
            canonical_path = root / "canonical_objects.jsonl"
            canonical_path.write_text("", encoding="utf-8")
            args = types.SimpleNamespace(
                out_dir=out_dir,
                dataset_path=dataset_path,
                sample_size=0,
                seed=7,
                canonical_objects=canonical_path,
                dataset_audit=audit_path,
                adapter_path=None,
                model_source_path=None,
                model_path=model_path,
                max_new_tokens=32,
                max_total_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                enable_thinking=False,
                reasoning_effort="",
                local_files_only=True,
                trust_remote_code=False,
                max_report_examples=10,
                progress_every=0,
            )
            passing_gold_contract = {
                "contract_version": "pretrajectory-sft-exact-v3",
                "status": "passed",
                "passed": True,
                "failure_count": 0,
            }
            exact_report = {
                "sample_count": 0,
                "evaluator_contract": {"version": "pretrajectory-sft-exact-v3"},
                "gold_self_evaluation": {"passed": True},
                "summary": {
                    "exact_graph_fact_pass_rate": 0.5,
                    "unsupported_language_rate": 0.0,
                    "exact_only": {
                        "mean_id_recall": 0.8,
                        "mean_layer_recall": 0.7,
                        "mean_number_recall": 0.8,
                    },
                },
                "by_context_mode": {},
                "by_mixture_bucket": {},
                "examples": [],
            }
            invalid_readiness = {
                "schema_version": "pretrajectory-sft-readiness-v2",
                "valid": False,
                "passed": False,
                "decision": "repair_evaluation_or_dataset_contract",
                "required_failure_count": 1,
                "advisory_failure_count": 0,
                "failed_contract_gates": [{"name": "exact_only_summary_present"}],
            }

            def write_invalid_readiness(**kwargs):
                runner.write_json(kwargs["output_path"], invalid_readiness)
                return dict(invalid_readiness)

            fake_model = types.SimpleNamespace(
                parameters=lambda: iter([types.SimpleNamespace(device="cpu")])
            )
            with (
                mock.patch.object(runner, "_select_rows", return_value=[]),
                mock.patch.object(runner, "load_canonical_objects", return_value={}),
                mock.patch.object(runner, "evaluate_gold_self_contract", return_value=passing_gold_contract),
                mock.patch.object(runner, "_tokenizer", return_value=object()),
                mock.patch.object(runner, "_load_model", return_value=fake_model),
                mock.patch.object(runner, "build_formatting_func", return_value=lambda row: ""),
                mock.patch.object(runner.torch, "no_grad", return_value=contextlib.nullcontext(), create=True),
                mock.patch.object(runner, "evaluate_prediction_rows", return_value=exact_report),
                mock.patch.object(runner, "render_html_report", return_value="<html></html>"),
                mock.patch.object(runner, "_run_readiness_checker", side_effect=write_invalid_readiness),
            ):
                with self.assertRaises(runner.ExactEvalContractError):
                    runner.run_exact_eval(args)

            run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
            readiness = json.loads((out_dir / "readiness_report.json").read_text(encoding="utf-8"))
            self.assertEqual(run_summary["status"], "invalid_readiness_contract")
            self.assertFalse(run_summary["valid"])
            self.assertFalse(run_summary["readiness"]["valid"])
            self.assertFalse(readiness["valid"])
            self.assertEqual(readiness["decision"], "repair_evaluation_or_dataset_contract")

    def test_readiness_checker_exit_one_is_a_valid_not_ready_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, audit_path = _write_v3_dataset(root)
            exact_path = root / "exact_report.json"
            readiness_path = root / "readiness_report.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            exact_path.write_text(
                json.dumps(
                    {
                        "evaluator_contract": {"version": "pretrajectory-sft-exact-v3"},
                        "gold_self_evaluation": {"passed": True},
                        "summary": {
                            "exact_graph_fact_pass_rate": 0.5,
                            "unsupported_language_rate": 0.0,
                            "mean_id_recall": 0.8,
                            "mean_layer_recall": 0.7,
                            "mean_number_recall": 0.8,
                            "exact_only": {
                                "mean_id_recall": 0.8,
                                "mean_layer_recall": 0.7,
                                "mean_number_recall": 0.8,
                            },
                        },
                        "by_context_mode": {},
                        "by_mixture_bucket": {},
                        "dataset_identity": {
                            "dataset_schema_version": "pretrajectory-sft-v3",
                            "plan_hash": audit["plan_hash"],
                            "content_hash": audit["content_hash"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            readiness = runner._run_readiness_checker(
                dataset_audit=audit_path,
                exact_report=exact_path,
                output_path=readiness_path,
            )

            self.assertTrue(readiness_path.is_file())
            self.assertFalse(readiness["passed"])
            self.assertEqual(readiness["decision"], "continue_pretrajectory_sft_or_data_repair")


if __name__ == "__main__":
    unittest.main()
