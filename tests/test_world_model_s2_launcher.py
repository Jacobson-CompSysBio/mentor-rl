"""Check S2 launch guards without contacting Slurm or loading GPU libraries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher_repo(tmp_path: Path) -> Path:
    """Provide a trusted validator fixture and poison every scheduler command."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config").mkdir()
    for name in ("world_model_s2.slurm", "world_model_s2_check.slurm",
                 "world_model_s2_prepare.slurm", "scripts/submit_world_model_s2.sh"):
        shutil.copyfile(REPO / name, tmp_path / name)
    (tmp_path / "config/world_model_s2_120b_training_v1_184n.json").write_text("{}\n")
    (tmp_path / "scripts/validate_world_model_s2_train_contract.py").write_text(
        "import hashlib, pathlib, shlex, sys\n"
        "root=pathlib.Path(sys.argv[1]); config=root/sys.argv[2]\n"
        "values={'S0_RUN_CONFIG_SHA256':hashlib.sha256(config.read_bytes()).hexdigest(),"
        "'REQUESTED_PARTITION':'batch','REQUESTED_WALLTIME':'10:00:00',"
        "'NUM_NODES':'184','RANKS_PER_NODE':'8','WORLD_SIZE_EXPECTED':'1472',"
        "'S0_METHOD_ID':'fixture-lora','OUTPUT_ROOT':str(root/'checkpoints')}\n"
        "for k,v in values.items(): print(k+'='+shlex.quote(v))\n"
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    for name in ("sbatch", "srun", "scontrol"):
        target = binary_dir / name
        target.write_text('#!/bin/bash\necho "Unexpected scheduler access" >&2\nexit 99\n')
        target.chmod(0o755)
    return tmp_path


def run_launcher(root: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    """Run the real shell guards against the fixture's shell-safe contract."""
    env = {
        **os.environ,
        "S2_PYTHON": sys.executable,
        "PATH": f"{root / 'bin'}:{os.environ['PATH']}",
        "CONTRACT_ONLY": "1",
        "PREPARE_DATA_ONLY": "0",
    }
    for key in ("EXPECTED_RUN_CONFIG_SHA256", "SLURM_SUBMIT_DIR", "S2_RESUME_CHECKPOINT"):
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(
        ["bash", "world_model_s2.slurm"], cwd=root, env=env, text=True,
        capture_output=True, check=False,
    )


def test_contract_only_does_not_touch_scheduler(launcher_repo: Path) -> None:
    result = run_launcher(launcher_repo)
    assert result.returncode == 0, result.stderr
    assert "No GPU work or Slurm job ran" in result.stdout
    assert not (launcher_repo / "checkpoints").exists()


def test_model_staging_defaults_to_one_allocation_wide_wave() -> None:
    """Keep the 184-node production launch out of sequential staging waves."""
    source = (REPO / "world_model_s2.slurm").read_text()
    assert 'MODEL_STAGE_WAVE_SIZE="${MODEL_STAGE_WAVE_SIZE:-${NUM_NODES}}"' in source
    assert "MODEL_STAGE_WAVE_SIZE=4" not in source
    assert "first += MODEL_STAGE_WAVE_SIZE" in source


def test_changed_queued_config_is_rejected_before_validation(launcher_repo: Path) -> None:
    result = run_launcher(launcher_repo, EXPECTED_RUN_CONFIG_SHA256="0" * 64)
    assert result.returncode == 2
    assert "queued run config identity changed" in result.stderr


def test_production_requires_sealed_config_identity(launcher_repo: Path) -> None:
    result = run_launcher(launcher_repo, CONTRACT_ONLY="0")
    assert result.returncode == 2
    assert "EXPECTED_RUN_CONFIG_SHA256 is required" in result.stderr


def test_wrong_allocation_is_rejected_before_gpu_setup(launcher_repo: Path) -> None:
    config = launcher_repo / "config/world_model_s2_120b_training_v1_184n.json"
    result = run_launcher(
        launcher_repo, CONTRACT_ONLY="0", SLURM_JOB_ID="123",
        SLURM_NNODES="32", SLURM_NTASKS="256",
        EXPECTED_RUN_CONFIG_SHA256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert result.returncode == 2
    assert "allocation node/rank count differs" in result.stderr


def test_resume_check_requires_complete_same_topology(launcher_repo: Path) -> None:
    checkpoint = launcher_repo / "checkpoints/slurm-1/checkpoint-200"
    checkpoint.mkdir(parents=True)
    for name in ("CHECKPOINT_COMPLETE", "training_state.json", "scheduler.pt"):
        (checkpoint / name).write_text("fixture\n")
    manifest = checkpoint / "tp_adapter_manifest.json"
    manifest.write_text(json.dumps({"topology": {"world_size": 256}}))
    result = run_launcher(launcher_repo, S2_RESUME_CHECKPOINT=str(checkpoint))
    assert result.returncode != 0
    assert "original 184-node topology" in result.stderr
    config = launcher_repo / "config/world_model_s2_120b_training_v1_184n.json"
    manifest.write_text(json.dumps({
        "topology": {"world_size": 1472},
        "identity": {"run_id": "original-s2-run", "run_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
    }))
    result = run_launcher(launcher_repo, S2_RESUME_CHECKPOINT=str(checkpoint))
    assert result.returncode == 0, result.stderr


def test_submission_helper_defaults_to_check(launcher_repo: Path) -> None:
    env = {**os.environ, "S2_PYTHON": sys.executable, "PATH": f"{launcher_repo / 'bin'}:{os.environ['PATH']}"}
    env.pop("SLURM_SUBMIT_DIR", None)
    result = subprocess.run(
        ["bash", "scripts/submit_world_model_s2.sh"], cwd=launcher_repo,
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "No job submitted" in result.stdout
    assert not (launcher_repo / "data").exists()


def test_explicit_submit_seals_inputs_and_requests_ten_hours(launcher_repo: Path) -> None:
    """Mock sbatch to check the actual submission path and source receipt."""
    config = launcher_repo / "config/world_model_s2_120b_training_v1_184n.json"
    source = launcher_repo / "world_model_s2.slurm"
    config.write_text(json.dumps({"implementation_files": [{
        "path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }]}))
    preparation = launcher_repo / "data/world_model_v4/cache/s2_training_v1_184n/preparation"
    preparation.mkdir(parents=True)
    (preparation / "data_preparation.json").write_text(json.dumps({
        "status": "complete", "run_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "cache_files": [str(source)],
    }))
    scheduler = launcher_repo / "bin/sbatch"
    scheduler.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > scheduler_args.txt\nprintf "%s\\n" "${CONTRACT_ONLY}" "${PREPARE_DATA_ONLY}" > scheduler_mode_env.txt\necho 12345\n')
    scheduler.chmod(0o755)
    env = {**os.environ, "S2_PYTHON": sys.executable, "PATH": f"{launcher_repo / 'bin'}:{os.environ['PATH']}"}
    env.pop("SLURM_SUBMIT_DIR", None)
    env.update(CONTRACT_ONLY="1", PREPARE_DATA_ONLY="1")
    result = subprocess.run(
        ["bash", "scripts/submit_world_model_s2.sh", "--submit"], cwd=launcher_repo,
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    args = (launcher_repo / "scheduler_args.txt").read_text().splitlines()
    assert {"--nodes=184", "--ntasks=1472", "--partition=batch", "--time=10:00:00"} <= set(args)
    assert (launcher_repo / "scheduler_mode_env.txt").read_text().splitlines() == ["0", "0"]
    reports = list((launcher_repo / "data/world_model_v4/launch_reports/s2_v1_184n").glob("prepared-*/preparation.json"))
    assert len(reports) == 1
    receipt = json.loads(reports[0].read_text())
    assert receipt["submitted"] is True and receipt["job_id"] == "12345"
    assert receipt["run_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert (reports[0].parent / "sources/world_model_s2.slurm").read_bytes() == source.read_bytes()


def test_stale_cache_is_rejected_before_submission(launcher_repo: Path) -> None:
    preparation = launcher_repo / "data/world_model_v4/cache/s2_training_v1_184n/preparation"
    preparation.mkdir(parents=True)
    (preparation / "data_preparation.json").write_text(json.dumps({
        "status": "complete", "run_config_sha256": "0" * 64,
        "cache_files": [str(launcher_repo / "world_model_s2.slurm")],
    }))
    env = {**os.environ, "S2_PYTHON": sys.executable, "PATH": f"{launcher_repo / 'bin'}:{os.environ['PATH']}"}
    env.pop("SLURM_SUBMIT_DIR", None)
    result = subprocess.run(
        ["bash", "scripts/submit_world_model_s2.sh", "--submit"], cwd=launcher_repo,
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Prepared S2 cache is stale or incomplete" in result.stderr
    assert not (launcher_repo / "data/world_model_v4/launch_reports").exists()


def test_pipeline_queues_three_dependent_jobs_without_local_preparation(launcher_repo: Path) -> None:
    """Queue CPU prerequisites and training with pinned inputs and afterok gates."""
    validator = launcher_repo / "scripts/validate_world_model_s2_train_contract.py"
    validator.write_text("raise SystemExit('Run the contract check in its Slurm job')\n")
    names = ["world_model_s2.slurm", "world_model_s2_check.slurm",
             "world_model_s2_prepare.slurm", "scripts/submit_world_model_s2.sh",
             "scripts/validate_world_model_s2_train_contract.py"]
    config = launcher_repo / "config/world_model_s2_120b_training_v1_184n.json"
    config.write_text(json.dumps({"implementation_files": [
        {"path": name, "sha256": hashlib.sha256((launcher_repo/name).read_bytes()).hexdigest()}
        for name in names]}))
    scheduler = launcher_repo / "bin/sbatch"
    scheduler.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "path=pathlib.Path('scheduler_calls.json')\n"
        "calls=json.loads(path.read_text()) if path.exists() else []\n"
        "calls.append({'args':sys.argv[1:], 'config_sha':os.environ['EXPECTED_RUN_CONFIG_SHA256'],"
        "'modes':[os.environ['CONTRACT_ONLY'],os.environ['PREPARE_DATA_ONLY']]})\n"
        "path.write_text(json.dumps(calls))\n"
        "print(str(42000+len(calls))+';frontier')\n"
    )
    scheduler.chmod(0o755)
    env = {**os.environ, "S2_PYTHON": sys.executable,
           "PATH": f"{launcher_repo/'bin'}:{os.environ['PATH']}",
           "CONTRACT_ONLY": "1", "PREPARE_DATA_ONLY": "1"}
    env.pop("SLURM_SUBMIT_DIR", None)
    result = subprocess.run(["bash", "scripts/submit_world_model_s2.sh", "--submit-pipeline"],
        cwd=launcher_repo, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = json.loads((launcher_repo/"scheduler_calls.json").read_text())
    assert len(calls) == 3
    assert calls[0]["args"][-1] == "world_model_s2_check.slurm"
    assert calls[1]["args"][-1] == "world_model_s2_prepare.slurm"
    assert "--dependency=afterok:42001" in calls[1]["args"]
    assert calls[2]["args"][-1] == "world_model_s2.slurm"
    assert {"--dependency=afterok:42002", "--nodes=184", "--ntasks=1472",
            "--partition=batch", "--time=10:00:00"} <= set(calls[2]["args"])
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    assert all(call["config_sha"] == digest and call["modes"] == ["0", "0"] for call in calls)
    reports = list((launcher_repo/"data/world_model_v4/launch_reports/s2_v1_184n").glob("prepared-*/preparation.json"))
    assert len(reports) == 1
    receipt = json.loads(reports[0].read_text())
    assert receipt["job_id"] == "42003"
    assert receipt["jobs"] == [
        {"role": "check", "job_id": "42001", "dependency": None},
        {"role": "prepare", "job_id": "42002", "dependency": "afterok:42001"},
        {"role": "train", "job_id": "42003", "dependency": "afterok:42002"},
    ]
    assert not (launcher_repo/"data/world_model_v4/cache").exists()
