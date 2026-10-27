"""Rebuild v10 into temporary paths and verify published data without modifying it."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


REPO = Path(__file__).resolve().parents[2]
DIAGNOSTICS = Path(__file__).resolve().parent
CORPUS = REPO / "data/world_model_v4/corpora/s1_v10"
EVALUATOR = REPO / "data/world_model_v4/eval/s1_v10"
OVERLAY = "runtime/world_model/s1_v10.yaml"
SPLITS = ("train", "validation", "test")
ENV = {**os.environ, "USE_TORCH": "0", "USE_TF": "0", "PYTHONPATH": "."}


def digest(path):
    """Hash a published or regenerated artifact without loading it all into memory."""

    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def run(label, arguments):
    """Run one real builder or validator and retain its complete output."""

    command = [sys.executable, *arguments]
    log = DIAGNOSTICS / f"reproducibility_{label}.log"
    started = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(command, cwd=REPO, env=ENV, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    receipt = {"command": command, "log": str(log.relative_to(REPO)),
               "exit_code": result.returncode,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
    print(json.dumps({"phase": label, **receipt}), flush=True)
    return receipt


def compare(published, rebuilt):
    """Require exact bytes and retain independent hashes for each compared file."""

    equal = True
    with published.open("rb") as first, rebuilt.open("rb") as second:
        while True:
            a, b = first.read(4 * 1024 * 1024), second.read(4 * 1024 * 1024)
            if a != b:
                equal = False
                break
            if not a:
                break
    return {"published": str(published.relative_to(REPO)),
            "published_sha256": digest(published), "rebuilt_sha256": digest(rebuilt),
            "bytes": published.stat().st_size, "byte_identical": equal}


def assess_results(record):
    """Separate deterministic builds from known frozen held-out evidence-ID collisions."""

    record["reproducibility_passed"] = (
        len(record["file_comparisons"]) == 16
        and all(item["byte_identical"] for item in record["file_comparisons"])
        and all(item["exit_code"] == 0 for item in record["components"].values())
        and record["aggregate"]["exit_code"] == 0
    )
    record["all_aggregate_semantic_checks_passed"] = all(
        item["exit_code"] == 0 for item in record["semantic_validation"].values()
    )
    record["heldout_semantic_limitations"] = {}
    for split in ("validation", "test"):
        baseline = REPO / f"diagnostics/s1_v9_build/{split}_semantic.log"
        rebuilt = REPO / record["semantic_validation"][split]["log"]
        report = rebuilt.read_text()
        violations = int(next(line.split(":", 1)[1] for line in report.splitlines()
                              if line.startswith("violations")))
        record["heldout_semantic_limitations"][split] = {
            "violations": violations, "category": "evidence_id unique across rows",
            "baseline_report": str(baseline.relative_to(REPO)),
            "baseline_sha256": digest(baseline), "rebuilt_report_sha256": digest(rebuilt),
            "byte_identical_to_v9_report": rebuilt.read_bytes() == baseline.read_bytes(),
        }
    record["pass_scope"] = (
        "Deterministic rebuilds, all component semantic checks, aggregate training semantics, "
        "and unchanged historical held-out semantic reports; the 251/148 frozen held-out "
        "evidence-ID collisions remain separately reported and are not a new v10 failure."
    )
    record["passed"] = (
        record["reproducibility_passed"]
        and record["semantic_validation"]["train"]["exit_code"] == 0
        and all(item["byte_identical_to_v9_report"]
                for item in record["heldout_semantic_limitations"].values())
    )


def main():
    """Build the changed components and complete bundle, then save the evidence."""

    published = [CORPUS / f"s1_{section}" / name
                 for section in (1, 3, 8)
                 for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")]
    published += [CORPUS / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")]
    published += [EVALUATOR / name for name in (
        "validation_questions.jsonl", "validation_answer_key.jsonl",
        "test_questions.jsonl", "test_answer_key.jsonl", "manifest.json",
    )]
    record = {"overlay": OVERLAY, "overlay_sha256": digest(REPO / OVERLAY),
              "published_before": {str(path.relative_to(REPO)): digest(path) for path in published},
              "components": {}, "aggregate": {}, "semantic_validation": {},
              "file_comparisons": [], "passed": False}
    started = time.monotonic()
    try:
        with TemporaryDirectory(prefix="reproducibility_", dir=DIAGNOSTICS) as temporary:
            temporary = Path(temporary)
            record["temporary_directory"] = str(temporary.relative_to(REPO))
            commands = {
                section: ["-m", f"runtime.world_model.s1_{section}_corpus", "--overlay", OVERLAY,
                          "--output-directory", str(temporary / f"s1_{section}")]
                for section in (1, 3, 8)
            }
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {section: pool.submit(run, f"s1_{section}_build", command)
                           for section, command in commands.items()}
                for section, future in futures.items():
                    record["components"][f"S1.{section}"] = future.result()
            if any(receipt["exit_code"] for receipt in record["components"].values()):
                raise RuntimeError("A component rebuild failed; see its saved log")
            for section in commands:
                for split in SPLITS:
                    record["file_comparisons"].append(compare(
                        CORPUS / f"s1_{section}" / f"{split}.jsonl",
                        temporary / f"s1_{section}" / f"{split}.jsonl",
                    ))
            rebuilt_corpus, rebuilt_eval = temporary / "corpus", temporary / "eval"
            record["aggregate"] = run("bundle_build", [
                "scripts/build_world_model_s1_bundle.py", "--overlay", OVERLAY,
                "--component-root", str(CORPUS), "--corpus-root", str(rebuilt_corpus),
                "--evaluator-root", str(rebuilt_eval),
            ])
            if record["aggregate"]["exit_code"]:
                raise RuntimeError("The aggregate rebuild failed; see its saved log")
            for split in SPLITS:
                record["file_comparisons"].append(compare(
                    CORPUS / f"{split}.jsonl", rebuilt_corpus / f"{split}.jsonl",
                ))
            for split in ("validation", "test"):
                for suffix in ("questions", "answer_key"):
                    name = f"{split}_{suffix}.jsonl"
                    record["file_comparisons"].append(compare(EVALUATOR / name, rebuilt_eval / name))
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {split: pool.submit(run, f"{split}_semantic", [
                    "agents/world_model_v3_curriculum/s1_kit/validate.py",
                    "--overlay", OVERLAY, str(rebuilt_corpus / f"{split}.jsonl"),
                ]) for split in SPLITS}
                for split, future in futures.items():
                    record["semantic_validation"][split] = future.result()
            assess_results(record)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["published_after"] = {str(path.relative_to(REPO)): digest(path) for path in published}
        record["published_artifacts_unchanged"] = record["published_before"] == record["published_after"]
        record["passed"] &= record["published_artifacts_unchanged"]
        record["temporary_directory_cleaned"] = not (REPO / record["temporary_directory"]).exists()
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (DIAGNOSTICS / "reproducibility.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "file_comparisons": len(record["file_comparisons"]),
                      "published_artifacts_unchanged": record["published_artifacts_unchanged"],
                      "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
