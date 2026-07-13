import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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
            dataset_path = root / "holdout.jsonl"
            example = {
                "system": "system",
                "question": "question",
                "answer": "answer",
                "metadata": {"record_id": "row-1"},
            }
            dataset_path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            canonical_path = root / "canonical_objects.jsonl"
            canonical_path.write_text("{}\n", encoding="utf-8")
            audit_path = root / "audit_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": False,
                        "fatal_error_count": 2,
                        "warning_count": 0,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
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
                "contract_version": "pretrajectory-sft-exact-v2",
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
            audit_path = Path(temporary_directory) / "audit_report.json"

            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": True,
                        "fatal_error_count": 0,
                        "warning_count": 2,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            contract = runner._dataset_audit_contract(audit_path)
            self.assertTrue(contract["valid"])
            self.assertEqual(contract["fatal_error_count"], 0)

            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": False,
                        "fatal_error_count": 3,
                        "warning_count": 0,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            contract = runner._dataset_audit_contract(audit_path)
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
            audit_path = Path(temporary_directory) / "audit_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": True,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            contract = runner._dataset_audit_contract(audit_path)

            self.assertFalse(contract["valid"])
            self.assertIn("missing_or_invalid_fatal_error_count", contract["failures"])

    def test_dataset_audit_contract_rejects_stale_v1_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_path = Path(temporary_directory) / "audit_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v1",
                        "passed": True,
                        "fatal_error_count": 0,
                        "warning_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            contract = runner._dataset_audit_contract(audit_path)

            self.assertFalse(contract["valid"])
            self.assertEqual(contract["schema_version"], "pretrajectory-sft-audit-v1")
            self.assertIn("unsupported_dataset_audit_schema_version", contract["failures"])
            self.assertIn("missing_answer_budget_report", contract["failures"])

    def test_dataset_audit_contract_requires_manifest_and_zero_over_budget_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_path = Path(temporary_directory) / "audit_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": False,
                        "fatal_error_count": 0,
                        "warning_count": 0,
                        "answer_budget_report": {
                            "manifest_contract_present": False,
                            "over_budget_record_count": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )

            contract = runner._dataset_audit_contract(audit_path)

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
            dataset_path = root / "holdout.jsonl"
            dataset_path.write_text("", encoding="utf-8")
            canonical_path = root / "canonical_objects.jsonl"
            canonical_path.write_text("", encoding="utf-8")
            audit_path = root / "audit_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "passed": True,
                        "fatal_error_count": 0,
                        "warning_count": 0,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
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
            audit_path = root / "audit_report.json"
            exact_path = root / "exact_report.json"
            readiness_path = root / "readiness_report.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pretrajectory-sft-audit-v2",
                        "fatal_error_count": 0,
                        "warning_count": 0,
                        "answer_budget_report": {
                            "manifest_contract_present": True,
                            "over_budget_record_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
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
