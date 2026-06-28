import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tasks(path: Path, count: int) -> None:
    rows = []
    for index in range(count):
        rows.append(
            json.dumps(
                {
                    "task_id": f"module_{index}.recovery.easy.graph",
                    "task_type": "recovery",
                    "difficulty": "easy",
                    "evidence_mode": "graph",
                    "visible_inputs": {"seed_gene_ids": ["ENSG1"], "seed_gene_symbols": ["GENE1"]},
                    "hidden_target": {"relationship_status": "validated_group"},
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_gemma4_fat_launcher_prepares_multi_node_run(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    model_path = tmp_path / "gemma-4-26B-A4B-it"
    sif_path = tmp_path / "vllm.sif"
    run_root = tmp_path / "run"
    model_path.mkdir()
    (model_path / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    sif_path.write_text("placeholder", encoding="utf-8")
    _write_tasks(tasks_path, 5)

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "REPO_ROOT": str(REPO_ROOT),
            "DATA_ROOT": str(tmp_path / "data_root"),
            "SCRATCH": str(tmp_path / "scratch"),
            "RUN_ROOT": str(run_root),
            "TASKS_PATH": str(tasks_path),
            "TASK_COUNT": "5",
            "TASK_OFFSET": "0",
            "NODES": "3",
            "TASK_CONCURRENCY": "7",
            "MODEL_PATH": str(model_path),
            "SIF": str(sif_path),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "submit_gemma4_frontier_fat.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "DRY_RUN=1; prepared fat-allocation run" in result.stdout

    selected_tasks = run_root / "inputs" / "tasks_selected.jsonl"
    assert selected_tasks.exists()
    assert len([line for line in selected_tasks.read_text(encoding="utf-8").splitlines() if line]) == 5
    lane_counts = []
    for lane_index in range(3):
        lane_tasks = run_root / "inputs" / "lane_tasks" / f"lane_{lane_index}.jsonl"
        assert lane_tasks.exists()
        lane_counts.append(
            len([line for line in lane_tasks.read_text(encoding="utf-8").splitlines() if line])
        )
    assert lane_counts == [2, 2, 1]

    config = json.loads((run_root / "submit_config.json").read_text(encoding="utf-8"))
    assert config["nodes"] == 3
    assert config["task_concurrency_per_node"] == 7
    assert config["max_model_len"] == "16384"
    assert config["max_num_batched_tokens"] == "16384"
    assert config["lane_task_strategy"] == "round_robin_selected_tasks"

    entry = (run_root / "entry.sh").read_text(encoding="utf-8")
    assert "run_lane" in entry
    assert 'TASKS_PATH="${lane_tasks_path}"' in entry
    assert "export MAX_TASKS=all" in entry
    assert "FRONTIER_PYTHON" in entry
    assert "merge_trajectory_lanes.py" in entry


def _write_lane(run_root: Path, lane_index: int, *, status: str = "completed") -> None:
    lane = run_root / "trajectories" / f"node_{lane_index}"
    (lane / "logs").mkdir(parents=True)
    (lane / "progress.json").write_text(
        json.dumps({"status": status, "started_at": f"2026-06-26T00:0{lane_index}:00Z"}) + "\n",
        encoding="utf-8",
    )
    (lane / "manifest.json").write_text(
        json.dumps({"task_selection": {}, "artifacts": {}, "outputs": {}}) + "\n",
        encoding="utf-8",
    )
    (lane / "branch_pools.jsonl").write_text(
        json.dumps({"task_id": f"task-{lane_index}", "branches": [{"id": "a"}, {"id": "b"}]}) + "\n",
        encoding="utf-8",
    )
    for name in [
        "trajectory_turns.jsonl",
        "finding_records.jsonl",
        "preference_pairs_raw.jsonl",
        "preference_pairs.jsonl",
        "final_summaries.jsonl",
    ]:
        (lane / name).write_text(json.dumps({"task_id": f"task-{lane_index}"}) + "\n", encoding="utf-8")
    (lane / "vllm_server.log").write_text(f"server log lane {lane_index}\n", encoding="utf-8")
    (lane / "logs" / "lane.out").write_text(f"lane {lane_index} stdout\n", encoding="utf-8")
    (lane / "logs" / "lane.err").write_text(f"lane {lane_index} stderr\n", encoding="utf-8")


def test_lane_merge_preserves_audit_visible_server_log_and_lane_logs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    inputs.joinpath("tasks_selected.jsonl").write_text(
        json.dumps({"task_id": "task-0"}) + "\n" + json.dumps({"task_id": "task-1"}) + "\n",
        encoding="utf-8",
    )
    _write_lane(run_root, 0)
    _write_lane(run_root, 1)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "merge_trajectory_lanes.py"),
            str(run_root),
            "--expected-lanes",
            "2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "completed_lanes\t2" in result.stdout
    merged = run_root / "trajectories_merged"
    assert (merged / "vllm_server.log").read_text(encoding="utf-8") == "server log lane 0\n"
    assert (merged / "lane_logs" / "node_0" / "vllm_server.log").read_text(encoding="utf-8") == "server log lane 0\n"
    assert (merged / "lane_logs" / "node_1" / "vllm_server.log").read_text(encoding="utf-8") == "server log lane 1\n"
    assert (merged / "lane_logs" / "node_0" / "lane.out").read_text(encoding="utf-8") == "lane 0 stdout\n"
    assert (merged / "lane_logs" / "node_1" / "lane.err").read_text(encoding="utf-8") == "lane 1 stderr\n"
    progress = json.loads((merged / "progress.json").read_text(encoding="utf-8"))
    assert progress["metrics"]["completed_lanes"] == 2
    assert progress["metrics"]["total_branches"] == 4
