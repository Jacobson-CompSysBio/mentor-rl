import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scripts import build_rwr_loe_corpus as bloe
from scripts import mix_module_corpora as mixer


def _write_store_genes(store_dir: Path, genes: list[str]) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")


def _write_mentor_distribution(corpus_dir: Path, size_bins: list[str]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"module_id": f"mentor_{index}", "size_bin": size_bin}
        for index, size_bin in enumerate(size_bins)
    ]
    bloe.write_jsonl(corpus_dir / "modules.jsonl", rows)


def _rank_cache_context(tmp_path: Path) -> tuple[Path, Path, dict]:
    flist = tmp_path / "full_brain_flist.tsv"
    flist.write_text("layer\tpath\n", encoding="utf-8")
    rank_cache_dir = tmp_path / "rank_cache"
    context = bloe.build_rank_cache_context(
        rwr_hpc_flist=flist,
        rwr_hpc_build_id="test-build",
        restart=0.7,
        delta=0.5,
        reduction_method="geometric",
        threshold=1e-10,
        edgelist_has_headers=True,
    )
    return flist, rank_cache_dir, context


def _grouped_ranked_genes(genes: list[str], seed_gene_id: str, *, group_size: int = 3) -> list[dict]:
    seed_index = genes.index(seed_gene_id)
    seed_group = seed_index // group_size
    groups = [genes[index : index + group_size] for index in range(0, len(genes), group_size)]
    same_group = [gene for gene in groups[seed_group] if gene != seed_gene_id]
    other_groups = [group for index, group in enumerate(groups) if index != seed_group]
    ordered = same_group[:]
    for offset in range(group_size):
        for group in other_groups:
            if offset < len(group):
                ordered.append(group[offset])
    ranked = []
    for rank, gene_id in enumerate(ordered, start=1):
        score = 1.0 - (rank * 0.05) if rank <= 3 else 0.01 / rank
        ranked.append({"gene": gene_id, "score": score, "rank": rank})
    return ranked


def _write_fake_rank_caches(
    *,
    genes: list[str],
    rank_cache_dir: Path,
    context: dict,
) -> Path:
    context_dir = bloe.rank_cache_context_dir(rank_cache_dir, context)
    bloe.write_json(context_dir / "cache_context.json", context)
    for gene_id in genes:
        bloe.write_seed_rank_cache(
            context_dir=context_dir,
            seed_gene_id=gene_id,
            ranked_genes=_grouped_ranked_genes(genes, gene_id),
            metadata={"seed_gene_id": gene_id, "cache_context": context},
        )
    return context_dir


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rank_cache_postprocessor_uses_loe_min_rank_ties(tmp_path: Path) -> None:
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    context_dir = bloe.rank_cache_context_dir(rank_cache_dir, context)
    encoding_matrix = tmp_path / "encodings.tsv"
    encoding_matrix.write_text(
        "\n".join(
            [
                "INDEX\tA\tB\tC\tD",
                "A\t1.0\t0.9\t0.9\t0.1",
                "B\t0.8\t1.0\t0.2\t0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = bloe.write_rank_cache_from_encoding_matrix(
        encoding_matrix_path=encoding_matrix,
        context_dir=context_dir,
        cache_context=context,
        shard_index=0,
        shard_count=1,
        requested_seed_gene_ids=["A", "B"],
    )

    assert flist.exists()
    assert manifest["completed_seed_count"] == 2
    ranks = bloe.load_seed_rank_cache(context_dir, "A")
    assert [(row["gene"], row["rank"]) for row in ranks] == [("B", 1), ("C", 1), ("D", 3)]


def test_elbow_module_selection_matches_rwr_geometric_cutoff() -> None:
    ranked = [
        {"gene": f"G{rank}", "score": score, "rank": rank}
        for rank, score in enumerate([1.0, 0.9, 0.8, 0.1, 0.09, 0.08], start=1)
    ]

    retained, metadata = bloe.select_elbow_module_ranked_genes(ranked)

    assert metadata["method"] == "elbow"
    assert metadata["elbow_rank_cutoff"] == 4
    assert metadata["retention_rule"] == "rank < elbow_rank_cutoff"
    assert [row["gene"] for row in retained] == ["G1", "G2", "G3"]


def test_shard_partitioning_is_deterministic_and_disjoint() -> None:
    genes = [f"G{index:02d}" for index in range(10)]
    shards = [bloe._shard_genes(genes, shard_index=index, shard_count=3) for index in range(3)]

    assert sorted(gene for shard in shards for gene in shard) == genes
    assert not (set(shards[0]) & set(shards[1]))
    assert bloe._shard_genes(genes, shard_index=1, shard_count=3) == shards[1]


def test_rank_band_negative_sampling_marks_nearest_as_hard_and_farthest_as_easy() -> None:
    ranked = [
        {"gene": f"G{index}", "score": 1.0 / index, "rank": index}
        for index in range(1, 17)
    ]
    candidate_gene_ids = {row["gene"] for row in ranked}

    hard, hard_meta = bloe.select_rank_band_negative_genes(
        ranked_genes=ranked,
        target_gene_ids=[],
        candidate_gene_ids=candidate_gene_ids,
        sample_size=2,
        difficulty="hard",
        seed=7,
        salt="hard",
    )
    medium, medium_meta = bloe.select_rank_band_negative_genes(
        ranked_genes=ranked,
        target_gene_ids=[],
        candidate_gene_ids=candidate_gene_ids,
        sample_size=2,
        difficulty="medium",
        seed=7,
        salt="medium",
    )
    easy, easy_meta = bloe.select_rank_band_negative_genes(
        ranked_genes=ranked,
        target_gene_ids=[],
        candidate_gene_ids=candidate_gene_ids,
        sample_size=2,
        difficulty="easy",
        seed=7,
        salt="easy",
    )

    assert max(hard_meta["selected_percentiles"].values()) < min(medium_meta["selected_percentiles"].values())
    assert max(medium_meta["selected_percentiles"].values()) < min(easy_meta["selected_percentiles"].values())
    assert max(hard_meta["selected_ranks"].values()) < min(medium_meta["selected_ranks"].values())
    assert max(medium_meta["selected_ranks"].values()) < min(easy_meta["selected_ranks"].values())
    assert set(hard).isdisjoint(easy)


def test_prewarm_returns_cache_hit_without_relaunching_existing_shard(tmp_path: Path) -> None:
    genes = [f"G{index:02d}" for index in range(6)]
    store_dir = tmp_path / "store"
    _write_store_genes(store_dir, genes)
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    context_dir = _write_fake_rank_caches(genes=genes, rank_cache_dir=rank_cache_dir, context=context)

    result = bloe.prewarm_rwr_loe_rank_cache(
        store_dir=store_dir,
        rwr_hpc_flist=flist,
        rwr_hpc_build_dir=tmp_path / "missing_build",
        rank_cache_dir=rank_cache_dir,
        scratch_dir=tmp_path / "scratch",
        shard_index=0,
        shard_count=1,
        seed=7,
        restart=0.7,
        delta=0.5,
        reduction_method="geometric",
        threshold=1e-10,
        edgelist_has_headers=True,
        rwr_hpc_build_id="test-build",
    )

    assert result["status"] == "cached"
    assert result["context_dir"] == str(context_dir)
    assert result["seed_count"] == len(genes)


def test_build_rwr_loe_corpus_materializes_balanced_tasks_from_rank_cache(tmp_path: Path) -> None:
    genes = [f"G{index:02d}" for index in range(24)]
    store_dir = tmp_path / "store"
    mentor_dir = tmp_path / "mentor"
    out_dir = tmp_path / "loe_corpus"
    _write_store_genes(store_dir, genes)
    _write_mentor_distribution(mentor_dir, ["small"])
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    _write_fake_rank_caches(genes=genes, rank_cache_dir=rank_cache_dir, context=context)

    result = bloe.build_rwr_loe_corpus(
        store_dir=store_dir,
        rwr_hpc_flist=flist,
        rank_cache_dir=rank_cache_dir,
        out_dir=out_dir,
        mentor_corpus_dir=mentor_dir,
        seed=17,
        module_sizes={"small": 3, "medium": 3, "large": 3},
        rank_cache_context=context,
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
    assert result["manifest"]["schema_version"] == bloe.SCHEMA_VERSION
    assert result["manifest"]["source"] == bloe.SOURCE_NAME
    assert result["manifest"]["module_selection"]["method"] == "elbow"
    assert {module["module_selection"]["method"] for module in result["modules"]} == {"elbow"}
    assert all(module["target_module_size"] >= 3 for module in result["modules"])

    tasks = result["tasks"]
    assert tasks
    counts_by_split = defaultdict(Counter)
    for task in tasks:
        counts_by_split[task["split"]][task["task_type"]] += 1
        assert task["provenance"]["source"] == bloe.SOURCE_NAME
        assert "hidden_target" not in task["visible_inputs"]
        if task["task_type"] == "none":
            assert task["hidden_target"]["target_gene_ids"] is None
        elif task["task_type"] == "recovery":
            assert set(task["visible_inputs"]["seed_gene_ids"]) < set(task["hidden_target"]["target_gene_ids"])
        elif task["task_type"] == "refinement":
            assert set(task["hidden_target"]["target_gene_ids"]) < set(task["visible_inputs"]["seed_gene_ids"])
    for counts in counts_by_split.values():
        if counts:
            assert set(counts) == set(bloe.TASK_TYPES)
            assert len(set(counts.values())) == 1


def test_checkpointed_module_assignment_resumes_partial_work(tmp_path: Path) -> None:
    genes = [f"G{index:02d}" for index in range(12)]
    store_dir = tmp_path / "store"
    _write_store_genes(store_dir, genes)
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    context_dir = _write_fake_rank_caches(genes=genes, rank_cache_dir=rank_cache_dir, context=context)
    checkpoint_path = tmp_path / "modules.assigned.by_gene.jsonl"
    first_record = bloe._build_loe_module_checkpoint_record(
        {
            "index": 1,
            "seed_gene_id": genes[0],
            "context_dir": str(context_dir),
            "module_selection_method": "elbow",
            "size_bin": None,
            "module_sizes": bloe.DEFAULT_MODULE_SIZES,
            "min_elbow_module_size": bloe.MIN_ELBOW_MODULE_SIZE,
        }
    )
    bloe._append_checkpoint_record(checkpoint_path, first_record)
    tracker = bloe.ProgressTracker(tmp_path / "progress.json")
    tracker.start("assign_modules", total=len(genes), unit="genes")
    stats = bloe.BuildStats()

    modules = bloe.build_loe_modules_checkpointed(
        gene_ids=genes,
        context_dir=context_dir,
        stats=stats,
        checkpoint_path=checkpoint_path,
        tracker=tracker,
        workers=2,
    )

    records = bloe._read_checkpoint_records(checkpoint_path, key_field="seed_gene_id")
    assert len(records) == len(genes)
    assert [record["seed_gene_id"] for record in records].count(genes[0]) == 1
    assert modules == sorted(modules, key=lambda row: row["module_id"])
    assert len({module["module_id"] for module in modules}) == len(modules)
    assert len(modules) + sum(stats.skipped_modules.values()) == len(genes)


def test_build_rwr_loe_corpus_resumes_from_materialization_checkpoints(tmp_path: Path, monkeypatch) -> None:
    genes = [f"G{index:02d}" for index in range(24)]
    store_dir = tmp_path / "store"
    mentor_dir = tmp_path / "mentor"
    out_dir = tmp_path / "loe_corpus"
    _write_store_genes(store_dir, genes)
    _write_mentor_distribution(mentor_dir, ["small"])
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    _write_fake_rank_caches(genes=genes, rank_cache_dir=rank_cache_dir, context=context)

    first = bloe.build_rwr_loe_corpus(
        store_dir=store_dir,
        rwr_hpc_flist=flist,
        rank_cache_dir=rank_cache_dir,
        out_dir=out_dir,
        mentor_corpus_dir=mentor_dir,
        seed=17,
        module_sizes={"small": 3, "medium": 3, "large": 3},
        rank_cache_context=context,
    )
    checkpoint_dirs = list((out_dir / "_materialize_checkpoints").glob("context_*"))
    assert len(checkpoint_dirs) == 1
    checkpoint_dir = checkpoint_dirs[0]
    assert (checkpoint_dir / "modules.assigned.jsonl").exists()
    assert (checkpoint_dir / "prototypes.raw.by_module.jsonl").exists()
    assert (checkpoint_dir / "prototypes.balanced.jsonl").exists()
    assert (checkpoint_dir / "tasks.by_prototype.jsonl").exists()

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("checkpointed stage was rebuilt instead of reused")

    monkeypatch.setattr(bloe, "build_loe_modules", fail_if_rebuilt)
    monkeypatch.setattr(bloe, "build_task_prototypes_for_module", fail_if_rebuilt)
    monkeypatch.setattr(bloe, "materialize_tasks_for_prototype", fail_if_rebuilt)

    second = bloe.build_rwr_loe_corpus(
        store_dir=store_dir,
        rwr_hpc_flist=flist,
        rank_cache_dir=rank_cache_dir,
        out_dir=out_dir,
        mentor_corpus_dir=mentor_dir,
        seed=17,
        module_sizes={"small": 3, "medium": 3, "large": 3},
        rank_cache_context=context,
    )

    assert second["manifest"]["module_count"] == first["manifest"]["module_count"]
    assert second["manifest"]["task_count"] == first["manifest"]["task_count"]
    assert second["tasks"] == first["tasks"]


def test_build_rwr_loe_corpus_fails_fast_for_empty_rank_context(tmp_path: Path) -> None:
    genes = [f"G{index:02d}" for index in range(6)]
    store_dir = tmp_path / "store"
    mentor_dir = tmp_path / "mentor"
    out_dir = tmp_path / "loe_corpus"
    _write_store_genes(store_dir, genes)
    _write_mentor_distribution(mentor_dir, ["small"])
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)

    with pytest.raises(FileNotFoundError, match="No RWR-LOE rank cache files found"):
        bloe.build_rwr_loe_corpus(
            store_dir=store_dir,
            rwr_hpc_flist=flist,
            rank_cache_dir=rank_cache_dir,
            out_dir=out_dir,
            mentor_corpus_dir=mentor_dir,
            seed=17,
            rank_cache_context=context,
        )

    assert not (out_dir / "manifest.json").exists()


def test_build_rwr_loe_corpus_can_use_explicit_prewarmed_context_dir(tmp_path: Path) -> None:
    genes = [f"G{index:02d}" for index in range(24)]
    store_dir = tmp_path / "store"
    mentor_dir = tmp_path / "mentor"
    out_dir = tmp_path / "loe_corpus"
    _write_store_genes(store_dir, genes)
    _write_mentor_distribution(mentor_dir, ["small"])
    flist, rank_cache_dir, context = _rank_cache_context(tmp_path)
    context_dir = _write_fake_rank_caches(genes=genes, rank_cache_dir=rank_cache_dir, context=context)
    flist.unlink()

    result = bloe.build_rwr_loe_corpus(
        store_dir=store_dir,
        rwr_hpc_flist=flist,
        rank_cache_dir=rank_cache_dir,
        out_dir=out_dir,
        mentor_corpus_dir=mentor_dir,
        seed=17,
        module_sizes={"small": 3, "medium": 3, "large": 3},
        rank_cache_context_dir_path=context_dir,
    )

    assert result["manifest"]["module_count"] > 0
    assert result["manifest"]["rank_cache_context_dir"] == str(context_dir)


def _write_tiny_corpus(corpus_dir: Path, *, source: str, task_prefix: str) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "manifest.json").write_text(
        json.dumps({"source": source}) + "\n",
        encoding="utf-8",
    )
    bloe.write_jsonl(corpus_dir / "modules.jsonl", [{"module_id": f"{task_prefix}_module", "source": source}])
    bloe.write_jsonl(corpus_dir / "prototypes.jsonl", [{"prototype_id": f"{task_prefix}_prototype", "source": source}])
    for split in ("train", "val", "test"):
        rows = []
        for index in range(2):
            rows.append(
                {
                    "task_id": f"{task_prefix}_{split}_{index}.recovery.easy.graph",
                    "split": split,
                    "task_type": "recovery",
                    "difficulty": "easy",
                    "evidence_mode": "graph",
                    "size_bin": "small",
                    "visible_inputs": {"seed_gene_ids": ["G1", "G2"]},
                    "hidden_target": {"target_gene_ids": ["G1", "G2", "G3", "G4", "G5"]},
                    "provenance": {"source": source},
                }
            )
        bloe.write_jsonl(corpus_dir / f"tasks.{split}.jsonl", rows)


def test_mix_module_corpora_balances_sources_and_preserves_task_metadata(tmp_path: Path) -> None:
    mentor_dir = tmp_path / "mentor"
    loe_dir = tmp_path / "loe"
    out_dir = tmp_path / "mixed"
    _write_tiny_corpus(mentor_dir, source="MENTOR_GW_DENDROGRAM", task_prefix="gw_dendrogram_module_000001")
    _write_tiny_corpus(loe_dir, source="RWR_LOE_FULL_BRAIN", task_prefix="rwr_loe_module_000001")

    result = mixer.build_mixed_corpus(
        corpus_dirs=[mentor_dir, loe_dir],
        out_dir=out_dir,
        seed=5,
        balance_by_source=True,
    )

    assert (out_dir / "manifest.json").exists()
    assert result["manifest"]["task_count_by_source"] == {
        "MENTOR_GW_DENDROGRAM": 6,
        "RWR_LOE_FULL_BRAIN": 6,
    }
    task = result["tasks"][0]
    assert {"task_type", "evidence_mode", "difficulty", "size_bin"} <= set(task)
    assert task["provenance"]["source_corpus_dir"] in {str(mentor_dir), str(loe_dir)}
