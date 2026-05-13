import json
from collections import Counter, defaultdict
from pathlib import Path

from runtime.scoring import score_terminal_trajectory
from runtime.schemas import ContinuationState, GeneGroup, RelationshipStatus, TerminationReason
from runtime.state import initialize_state_from_corum_task, replace_predicted_groups
from scripts import build_gw_dendrogram_corpus as bgd


class DendrogramBuilder:
    def __init__(self) -> None:
        self.rows = []
        self.next_id = 0

    def leaf(self, label: str) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.rows.append(
            {
                "node_id": node_id,
                "left_id": -1,
                "right_id": -1,
                "height": 0.0,
                "label": label,
            }
        )
        return node_id

    def internal(self, left_id: int, right_id: int, height: float) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.rows.append(
            {
                "node_id": node_id,
                "left_id": left_id,
                "right_id": right_id,
                "height": height,
                "label": "NA",
            }
        )
        return node_id

    def subtree(self, labels: list[str], height: float = 0.1) -> int:
        node_ids = [self.leaf(label) for label in labels]
        current_height = height
        while len(node_ids) > 1:
            next_level = []
            for index in range(0, len(node_ids), 2):
                if index + 1 >= len(node_ids):
                    next_level.append(node_ids[index])
                else:
                    next_level.append(self.internal(node_ids[index], node_ids[index + 1], current_height))
                    current_height += 0.01
            node_ids = next_level
        return node_ids[0]

    def write(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("node_id\tleft_id\tright_id\theight\tlabel\n")
            for row in self.rows:
                handle.write(
                    f"{row['node_id']}\t{row['left_id']}\t{row['right_id']}\t"
                    f"{row['height']}\t{row['label']}\n"
                )


def _ensg(index: int) -> str:
    return f"ENSG{index:011d}"


def _chain_roots(builder: DendrogramBuilder, roots: list[int], height: float = 1.0) -> int:
    root = roots[-1]
    current_height = height
    for next_root in reversed(roots[:-1]):
        root = builder.internal(next_root, root, current_height)
        current_height += 0.1
    return root


def _write_generation_dendrogram(path: Path) -> set[str]:
    builder = DendrogramBuilder()
    target_root = builder.subtree([_ensg(i) for i in range(1, 6)])
    external_roots = [
        builder.subtree([_ensg(i) for i in range(start, start + 16)])
        for start in range(101, 613, 16)
    ]
    _chain_roots(builder, [target_root] + external_roots)
    builder.write(path)
    return {_ensg(i) for i in range(1, 6)} | {_ensg(i) for i in range(101, 613)}


def _distance_sampling_fixture() -> tuple[DendrogramBuilder, int, set[str]]:
    builder = DendrogramBuilder()
    target_root = builder.subtree([f"T{i}" for i in range(5)], height=0.1)
    easy_root = builder.subtree([f"E{i}" for i in range(4)], height=0.2)
    easy_parent = builder.internal(target_root, easy_root, 0.3)
    medium_root = builder.subtree([f"M{i}" for i in range(4)], height=0.2)
    medium_parent = builder.internal(easy_parent, medium_root, 0.7)
    hard_root = builder.subtree([f"H{i}" for i in range(4)], height=0.2)
    hard_parent = builder.internal(medium_parent, hard_root, 1.1)
    unused_far_root = builder.subtree([f"F{i}" for i in range(4)], height=0.2)
    builder.internal(hard_parent, unused_far_root, 1.5)
    allowed_gene_ids = {f"T{i}" for i in range(5)}
    allowed_gene_ids |= {f"E{i}" for i in range(4)}
    allowed_gene_ids |= {f"M{i}" for i in range(4)}
    allowed_gene_ids |= {f"H{i}" for i in range(4)}
    allowed_gene_ids |= {f"F{i}" for i in range(4)}
    return builder, target_root, allowed_gene_ids


def test_parse_extract_modules_filters_store_genes_and_deduplicates(tmp_path: Path) -> None:
    builder = DendrogramBuilder()
    compact_root = builder.subtree([_ensg(i) for i in range(1, 6)], height=0.1)
    missing_leaf = builder.leaf("ENSG_MISSING")
    duplicate_root = builder.internal(compact_root, missing_leaf, 0.5)
    extra_leaf = builder.leaf(_ensg(99))
    builder.internal(duplicate_root, extra_leaf, 1.0)
    dendrogram_path = tmp_path / "toy_dendrogram.txt"
    builder.write(dendrogram_path)

    nodes = bgd.parse_dendrogram(dendrogram_path)
    modules, summary = bgd.extract_modules(nodes, {_ensg(i) for i in range(1, 6)} | {_ensg(99)})

    matching = [module for module in modules if module["gene_ids"] == [_ensg(i) for i in range(1, 6)]]
    assert len(matching) == 1
    assert matching[0]["source_node_id"] == compact_root
    assert duplicate_root in matching[0]["duplicate_source_node_ids"]
    assert summary["duplicate_module_count"] == 1
    assert matching[0]["size_bin"] == "small"


def test_dendrogram_negative_sampling_uses_requested_distance_bands(tmp_path: Path) -> None:
    builder, target_root, allowed_gene_ids = _distance_sampling_fixture()
    dendrogram_path = tmp_path / "distance_dendrogram.txt"
    builder.write(dendrogram_path)
    nodes = bgd.parse_dendrogram(dendrogram_path)
    distance_index = bgd.build_dendrogram_distance_index(nodes, allowed_gene_ids)
    module = {
        "source_node_id": target_root,
        "gene_ids": sorted(f"T{i}" for i in range(5)),
    }

    hard, hard_meta = bgd.select_dendrogram_negative_genes(
        module=module,
        distance_index=distance_index,
        candidate_gene_ids=allowed_gene_ids,
        sample_size=2,
        difficulty="hard",
        seed=1,
        salt="hard",
    )
    medium, medium_meta = bgd.select_dendrogram_negative_genes(
        module=module,
        distance_index=distance_index,
        candidate_gene_ids=allowed_gene_ids,
        sample_size=2,
        difficulty="medium",
        seed=1,
        salt="medium",
    )
    easy, easy_meta = bgd.select_dendrogram_negative_genes(
        module=module,
        distance_index=distance_index,
        candidate_gene_ids=allowed_gene_ids,
        sample_size=2,
        difficulty="easy",
        seed=1,
        salt="easy",
    )

    assert all(gene_id.startswith("E") for gene_id in easy)
    assert easy_meta["selection_mode"] == "preferred_band"
    assert all(gene_id.startswith("M") for gene_id in medium)
    assert medium_meta["selection_mode"] == "preferred_band"
    assert all(gene_id.startswith("H") for gene_id in hard)
    assert hard_meta["selection_mode"] == "preferred_band"
    assert max(easy_meta["selected_distances"].values()) < min(medium_meta["selected_distances"].values())
    assert max(medium_meta["selected_distances"].values()) < min(hard_meta["selected_distances"].values())


def test_build_task_prototypes_and_materialization_are_balanced_and_compatible(tmp_path: Path) -> None:
    dendrogram_path = tmp_path / "gw_dendrogram.txt"
    allowed_gene_ids = _write_generation_dendrogram(dendrogram_path)
    store_dir = tmp_path / "fake_store"
    out_dir = tmp_path / "out"

    result = bgd.build_gw_dendrogram_corpus(
        dendrogram_path=dendrogram_path,
        store_dir=store_dir,
        out_dir=out_dir,
        seed=7,
        allowed_gene_ids=allowed_gene_ids,
    )

    for file_name in (
        "manifest.json",
        "split_report.json",
        "modules.jsonl",
        "prototypes.jsonl",
        "tasks.train.jsonl",
        "tasks.val.jsonl",
        "tasks.test.jsonl",
        "progress.json",
    ):
        assert (out_dir / file_name).exists()

    tasks = result["tasks"]
    assert tasks
    counts_by_group = defaultdict(Counter)
    module_size_lookup = {module["module_id"]: module["size_bin"] for module in result["modules"]}
    for task in tasks:
        module_id = task["provenance"].get("source_module_id") or task["provenance"].get("anchor_module_id")
        counts_by_group[(task["split"], module_size_lookup[module_id])][task["task_type"]] += 1
    for counts in counts_by_group.values():
        assert set(counts) == set(bgd.TASK_TYPES)
        assert len(set(counts.values())) == 1

    for task in tasks:
        seed_gene_ids = set(task["visible_inputs"]["seed_gene_ids"])
        if task["task_type"] == "explanation":
            assert seed_gene_ids == set(task["hidden_target"]["target_gene_ids"])
            assert task["difficulty"] == "complete"
        elif task["task_type"] == "recovery":
            assert seed_gene_ids < set(task["hidden_target"]["target_gene_ids"])
            assert len(seed_gene_ids) >= 2
        elif task["task_type"] == "refinement":
            target_gene_ids = set(task["hidden_target"]["target_gene_ids"])
            assert target_gene_ids < seed_gene_ids
            assert len(seed_gene_ids - target_gene_ids) == bgd.noise_gene_count(
                len(target_gene_ids),
                task["difficulty"],
            )
        else:
            assert task["hidden_target"]["target_gene_ids"] is None
            assert task["hidden_target"]["relationship_status"] == "insufficient_support"

    recovery_task = next(task for task in tasks if task["task_type"] == "recovery")
    _, initial_state = initialize_state_from_corum_task(recovery_task, max_budget=4)
    final_state = replace_predicted_groups(
        initial_state,
        predicted_groups=[
            GeneGroup(
                group_id="group_0",
                gene_ids=recovery_task["hidden_target"]["target_gene_ids"],
                gene_symbols=recovery_task["hidden_target"]["target_gene_symbols"],
                rationale="Recovered the dendrogram module.",
            )
        ],
        relationship_status=RelationshipStatus.VALIDATED_GROUP,
    )
    final_state.continuation_state = ContinuationState.STOP
    final_state.termination_reason = TerminationReason.MODEL_STOP
    score = score_terminal_trajectory(recovery_task, initial_state, final_state, step_count=1, max_steps=4)
    assert score["terminal_reward"] > 0


def test_build_is_deterministic_with_same_seed(tmp_path: Path) -> None:
    dendrogram_path = tmp_path / "gw_dendrogram.txt"
    allowed_gene_ids = _write_generation_dendrogram(dendrogram_path)

    result_1 = bgd.build_gw_dendrogram_corpus(
        dendrogram_path=dendrogram_path,
        store_dir=tmp_path / "fake_store",
        out_dir=tmp_path / "out1",
        seed=17,
        allowed_gene_ids=allowed_gene_ids,
    )
    result_2 = bgd.build_gw_dendrogram_corpus(
        dendrogram_path=dendrogram_path,
        store_dir=tmp_path / "fake_store",
        out_dir=tmp_path / "out2",
        seed=17,
        allowed_gene_ids=allowed_gene_ids,
    )

    assert result_1["modules"] == result_2["modules"]
    assert result_1["prototypes"] == result_2["prototypes"]
    assert result_1["tasks"] == result_2["tasks"]
    assert json.loads((tmp_path / "out1" / "manifest.json").read_text())["task_count"] == len(result_1["tasks"])
