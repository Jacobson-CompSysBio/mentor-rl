import importlib.util
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "utils" / "utils.py"
TRAIN_PATH = REPO_ROOT / "scripts" / "train_sft.py"


def _load_utils():
    torch = types.ModuleType("torch")
    torch.__path__ = []
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    torch.distributed = distributed

    transformers = types.ModuleType("transformers")
    transformers.HfArgumentParser = object

    spec = importlib.util.spec_from_file_location("sft_utils_for_completion_test", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch,
            "torch.distributed": distributed,
            "transformers": transformers,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


utils = _load_utils()


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = sorted({key for row in rows for key in row})

    def map(self, function):
        mapped = []
        for row in self.rows:
            updated = dict(row)
            updated.update(function(dict(row)))
            mapped.append(updated)
        return _FakeDataset(mapped)

    def __getitem__(self, index):
        return self.rows[index]


def _load_train_module():
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.bfloat16 = object()
    torch.distributed = types.SimpleNamespace()

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    peft = types.ModuleType("peft")
    peft.LoraConfig = object
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.TrainingArguments = object
    transformers.HfArgumentParser = object
    trainer_utils = types.ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda *_args, **_kwargs: None
    accelerate = types.ModuleType("accelerate")
    accelerate.__path__ = []
    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils.gather_object = lambda value: value
    trl = types.ModuleType("trl")
    trl.SFTTrainer = object
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    inference = types.ModuleType("inference")
    inference.infer = lambda *_args, **_kwargs: None
    evaluator = types.ModuleType("evaluate_pretrajectory_sft_predictions")
    evaluator.evaluate_prediction_rows = lambda *_args, **_kwargs: None
    evaluator.load_canonical_objects = lambda *_args, **_kwargs: None
    evaluator.render_html_report = lambda *_args, **_kwargs: ""
    evaluator.write_json = lambda *_args, **_kwargs: None

    spec = importlib.util.spec_from_file_location("train_sft_for_completion_test", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "torch": torch,
        "datasets": datasets,
        "peft": peft,
        "transformers": transformers,
        "transformers.trainer_utils": trainer_utils,
        "accelerate": accelerate,
        "accelerate.utils": accelerate_utils,
        "trl": trl,
        "dotenv": dotenv,
        "inference": inference,
        "evaluate_pretrajectory_sft_predictions": evaluator,
        "utils.utils": utils,
    }
    with (
        mock.patch.dict(sys.modules, stubs),
        mock.patch.dict(
            "os.environ",
            {"WANDB_ENTITY": "test", "WANDB_API_KEY": "test"},
        ),
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


train_sft = _load_train_module()


class _Trainer:
    completion_only_loss = True

    def __init__(self, examples, collator):
        self.train_dataset = examples
        self.data_collator = collator


def _correct_collator(examples):
    example = examples[0]
    input_ids = example["input_ids"]
    completion_mask = example["completion_mask"]
    labels = [token if flag else -100 for token, flag in zip(input_ids, completion_mask)]
    return {"input_ids": [input_ids], "labels": [labels]}


class PromptCompletionDatasetTests(unittest.TestCase):
    def test_record_is_converted_to_conversational_prompt_and_completion(self):
        converted = utils.build_prompt_completion_example(
            {"system": "Follow the graph contract.", "question": "Which layer?", "answer": "layer-7"}
        )

        self.assertEqual(
            converted["prompt"],
            [
                {"role": "system", "content": "Follow the graph contract."},
                {"role": "user", "content": "Which layer?"},
            ],
        )
        self.assertEqual(
            converted["completion"],
            [{"role": "assistant", "content": "layer-7"}],
        )

    def test_missing_system_uses_default_but_empty_question_or_answer_is_rejected(self):
        converted = utils.build_prompt_completion_example({"question": "Q", "answer": "A"})
        self.assertEqual(converted["prompt"][0]["content"], utils.SYSTEM_PROMPT)

        for record in ({"question": "", "answer": "A"}, {"question": "Q", "answer": " "}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                utils.build_prompt_completion_example(record)

    def test_train_loader_adds_prompt_completion_columns_and_requires_completion_loss(self):
        raw = _FakeDataset([{"question": "Which layer?", "answer": "layer-7"}])
        with mock.patch.object(train_sft, "load_dataset", return_value=raw):
            loaded = train_sft._load_sft_dataset("unused.jsonl")

        self.assertEqual(loaded[0]["system"], utils.SYSTEM_PROMPT)
        self.assertEqual(loaded[0]["prompt"][-1], {"role": "user", "content": "Which layer?"})
        self.assertEqual(
            loaded[0]["completion"],
            [{"role": "assistant", "content": "layer-7"}],
        )
        self.assertTrue(train_sft.SftTrainingArguments().completion_only_loss)

    def test_trainer_wiring_does_not_collapse_records_with_a_formatting_function(self):
        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
        trainer_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SFTTrainer"
        ]
        self.assertEqual(len(trainer_calls), 1)
        self.assertNotIn("formatting_func", {keyword.arg for keyword in trainer_calls[0].keywords})
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "assert_completion_only_supervision"
                for node in ast.walk(tree)
            )
        )

    def test_stage_holdout_finds_canonical_objects_at_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            stage_dir = dataset_root / "curriculum" / "stage1_entity_schema"
            stage_dir.mkdir(parents=True)
            holdout = stage_dir / "test.jsonl"
            holdout.touch()
            canonical = dataset_root / "canonical_objects.jsonl"
            canonical.touch()

            self.assertEqual(
                train_sft._find_canonical_objects_path(str(holdout)),
                canonical.resolve(),
            )


class CompletionOnlyPreflightTests(unittest.TestCase):
    def test_preflight_proves_prompt_masking_and_completion_learning(self):
        examples = [
            {"input_ids": [11, 12, 13, 14], "completion_mask": [0, 0, 1, 1]},
            {"input_ids": [21, 22, 23], "completion_mask": [0, 1, 1]},
        ]
        report = utils.assert_completion_only_supervision(
            _Trainer(examples, _correct_collator),
            max_examples=8,
        )

        self.assertEqual(
            report,
            {
                "examples_checked": 2,
                "prompt_tokens_masked": 3,
                "completion_tokens_trainable": 4,
            },
        )

    def test_preflight_rejects_a_trainable_prompt_label(self):
        example = {"input_ids": [11, 12, 13], "completion_mask": [0, 0, 1]}

        def leaking_collator(_examples):
            return {"input_ids": [[11, 12, 13]], "labels": [[-100, 12, 13]]}

        with self.assertRaisesRegex(RuntimeError, "prompt/system/user tokens have trainable labels"):
            utils.assert_completion_only_supervision(_Trainer([example], leaking_collator))

    def test_preflight_rejects_a_masked_assistant_label(self):
        example = {"input_ids": [11, 12, 13], "completion_mask": [0, 1, 1]}

        def overmasked_collator(_examples):
            return {"input_ids": [[11, 12, 13]], "labels": [[-100, -100, 13]]}

        with self.assertRaisesRegex(RuntimeError, "assistant completion tokens are masked"):
            utils.assert_completion_only_supervision(_Trainer([example], overmasked_collator))

    def test_preflight_requires_prompt_completion_processing_and_enabled_trainer_flag(self):
        missing_mask = {"input_ids": [11, 12, 13]}
        with self.assertRaisesRegex(RuntimeError, "has no completion_mask"):
            utils.assert_completion_only_supervision(_Trainer([missing_mask], _correct_collator))

        trainer = _Trainer(
            [{"input_ids": [11, 12], "completion_mask": [0, 1]}],
            _correct_collator,
        )
        trainer.completion_only_loss = False
        with self.assertRaisesRegex(RuntimeError, "completion_only_loss is not True"):
            utils.assert_completion_only_supervision(trainer)


if __name__ == "__main__":
    unittest.main()
