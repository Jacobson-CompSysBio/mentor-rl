"""Test that a parent adapter loads into a new tensor-parallel run."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


WORKER = Path(__file__).resolve().parent / "adapter_initialization_worker.py"
try:
    import torch

    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
except (ImportError, KeyError, RuntimeError):
    # Login nodes lack the ROCm management library that GPU imports load.
    GPU_COUNT = 0


@pytest.mark.skipif(GPU_COUNT < 2, reason="the TP load needs at least two GPUs")
def test_parent_adapter_loads_into_a_new_tp_layout(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={min(GPU_COUNT, 8)}",
            str(WORKER),
            str(tmp_path / "parent"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "ADAPTER_INIT_PARITY_OK" in result.stdout
