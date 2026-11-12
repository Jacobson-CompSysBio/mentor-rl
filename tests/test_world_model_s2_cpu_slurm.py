"""Exercise the CPU Slurm wrappers without scheduling or importing GPU code."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (("world_model_s2_check.slurm", "1"), ("world_model_s2_prepare.slurm", "8"))


@pytest.fixture
def wrapper_repo(tmp_path):
    """Record the real wrappers' arguments at the shared launcher boundary."""
    for name, _ in WRAPPERS:
        shutil.copyfile(ROOT / name, tmp_path / name)
    (tmp_path / "world_model_s2.slurm").write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"${CONTRACT_ONLY}\" \"${PREPARE_DATA_ONLY}\" "
        "\"${OMP_NUM_THREADS}\" \"${S2_PREPARATION_WORKERS:-}\" "
        "\"${S2_RESUME_CHECKPOINT:-}\" \"${EXPECTED_RUN_CONFIG_SHA256}\" > mode.env\n"
        "exit \"${FIXTURE_STATUS:-0}\"\n"
    )
    return tmp_path


def run_wrapper(root, name, cpus, **updates):
    """Supply one simulated CPU allocation; the wrapper executes its real body."""
    env = {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(root), "SLURM_JOB_PARTITION": "extended",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": cpus,
        "EXPECTED_RUN_CONFIG_SHA256": "a" * 64,
        "CONTRACT_ONLY": "1", "PREPARE_DATA_ONLY": "1",
    }
    env.pop("S2_PREPARATION_WORKERS", None)
    for key, value in updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(["bash", name], cwd=root, env=env, text=True,
        capture_output=True, check=False)


@pytest.mark.parametrize("name,cpus", WRAPPERS)
def test_cpu_wrapper_overrides_inherited_modes_and_keeps_queued_identity(wrapper_repo, name, cpus):
    resume = "/original/run/checkpoint-200"
    result = run_wrapper(wrapper_repo, name, cpus, S2_RESUME_CHECKPOINT=resume)
    assert result.returncode == 0, result.stderr
    actual = (wrapper_repo / "mode.env").read_text().splitlines()
    expected = ["1", "0", "1", "", resume, "a" * 64] if cpus == "1" else ["0", "1", "1", "8", "", "a" * 64]
    assert actual == expected


@pytest.mark.parametrize("name,cpus", WRAPPERS)
def test_cpu_wrapper_rejects_wrong_allocation_before_launcher(wrapper_repo, name, cpus):
    result = run_wrapper(wrapper_repo, name, cpus, SLURM_NNODES="184")
    assert result.returncode == 2
    assert not (wrapper_repo / "mode.env").exists()


@pytest.mark.parametrize("name,cpus", WRAPPERS)
def test_cpu_wrapper_requires_queued_config_pin(wrapper_repo, name, cpus):
    result = run_wrapper(wrapper_repo, name, cpus, EXPECTED_RUN_CONFIG_SHA256=None)
    assert result.returncode != 0
    assert "Pass the pinned config SHA" in result.stderr
    assert not (wrapper_repo / "mode.env").exists()


@pytest.mark.parametrize("name,cpus", WRAPPERS)
def test_cpu_wrapper_propagates_launcher_failure(wrapper_repo, name, cpus):
    result = run_wrapper(wrapper_repo, name, cpus, FIXTURE_STATUS="7")
    assert result.returncode == 7
