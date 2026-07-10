import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_pretrajectory_sft_dataset as sft
from scripts import build_rwr_loe_corpus as bloe


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_toy_graph(tmp_path: Path) -> Path:
    ppi = tmp_path / "ppi.tsv"
    ppi.write_text("A\tB\t1.0\nB\tC\t0.5\nC\tD\t0.4\n", encoding="utf-8")
    coexp = tmp_path / "coexp.tsv"
    coexp.write_text("A\tC\t0.8\nD\tE\t0.6\n", encoding="utf-8")
    flist = tmp_path / "graph.flist"
    flist.write_text(f"{ppi}\tHumanNetV3:string_ppi\n{coexp}\tHumanNetV3:coexpression\n", encoding="utf-8")
    return flist


def _task(
    *,
    task_id: str,
    split: str,
    source: str,
    task_type: str,
    module_id: str | None,
    visible: list[str],
    target: list[str] | None,
) -> dict:
    return {
        "task_id": task_id,
        "split": split,
        "task_type": task_type,
        "difficulty": "complete" if task_type == "explanation" else "easy",
        "evidence_mode": "graph",
        "query_text": f"Toy query for {task_id}",
        "visible_inputs": {
            "seed_gene_ids": visible,
            "seed_gene_symbols": visible,
            "context_text": None,
            "graph_query_spec": {"operator": "induce_subgraph"},
            "structured_annotations": None,
        },
        "hidden_target": {
            "relationship_status": "validated_group" if target else "insufficient_support",
            "target_gene_ids": target,
            "target_gene_symbols": target,
        },
        "provenance": {
            "source": source,
            "source_module_id": module_id,
            "anchor_module_id": module_id,
            "source_corpus_dir": "toy",
        },
    }


def _write_toy_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "mixed"
    _write_json(
        corpus / "manifest.json",
        {
            "schema_version": "module-corpus-mix-v1",
            "source": "MIXED_MODULE_CORPUS_FULL_BRAIN",
            "task_count": 5,
        },
    )
    _write_jsonl(
        corpus / "modules.jsonl",
        [
            {
                "module_id": "mentor_module_1",
                "source": "MENTOR_GW_DENDROGRAM",
                "split": "train",
                "gene_ids": ["A", "B", "C"],
                "gene_symbols": ["A", "B", "C"],
                "size": 3,
                "size_bin": "small",
            },
            {
                "module_id": "rwr_module_1",
                "source": "RWR_LOE_FULL_BRAIN",
                "split": "val",
                "gene_ids": ["A", "D", "E"],
                "gene_symbols": ["A", "D", "E"],
                "size": 3,
                "size_bin": "small",
            },
        ],
    )
    _write_jsonl(
        corpus / "tasks.train.jsonl",
        [
            _task(
                task_id="mentor_module_1.explanation.complete.graph",
                split="train",
                source="MENTOR_GW_DENDROGRAM",
                task_type="explanation",
                module_id="mentor_module_1",
                visible=["A", "B", "C"],
                target=["A", "B", "C"],
            ),
            _task(
                task_id="mentor_module_1.recovery.easy.graph",
                split="train",
                source="MENTOR_GW_DENDROGRAM",
                task_type="recovery",
                module_id="mentor_module_1",
                visible=["A", "B"],
                target=["A", "B", "C"],
            ),
            _task(
                task_id="mentor_module_1.none.easy.graph",
                split="train",
                source="MENTOR_GW_DENDROGRAM",
                task_type="none",
                module_id=None,
                visible=["D", "E"],
                target=None,
            ),
        ],
    )
    _write_jsonl(
        corpus / "tasks.val.jsonl",
        [
            _task(
                task_id="rwr_module_1.explanation.complete.graph",
                split="val",
                source="RWR_LOE_FULL_BRAIN",
                task_type="explanation",
                module_id="rwr_module_1",
                visible=["A", "D", "E"],
                target=["A", "D", "E"],
            )
        ],
    )
    _write_jsonl(
        corpus / "tasks.test.jsonl",
        [
            _task(
                task_id="rwr_module_1.none.easy.graph",
                split="test",
                source="RWR_LOE_FULL_BRAIN",
                task_type="none",
                module_id=None,
                visible=["B", "E"],
                target=None,
            )
        ],
    )
    return corpus


def _write_rank_cache(tmp_path: Path) -> Path:
    context_dir = tmp_path / "rank_cache" / "context_toy"
    context = {
        "schema_version": "rwr-loe-rank-cache-v1",
        "ranking_semantics": "toy_desc_seed_excluded",
        "network_flist_sha256": "toy",
    }
    bloe.write_json(context_dir / "cache_context.json", context)
    for seed in ["A", "B", "C", "D", "E"]:
        ranked = [
            {"gene": f"{seed}_N1", "score": 0.9, "rank": 1},
            {"gene": f"{seed}_N2", "score": 0.7, "rank": 2},
            {"gene": f"{seed}_N3", "score": 0.5, "rank": 3},
        ]
        bloe.write_seed_rank_cache(
            context_dir=context_dir,
            seed_gene_id=seed,
            ranked_genes=ranked,
            metadata={"seed_gene_id": seed, "cache_context": context},
        )
    return context_dir


class PretrajectorySftDatasetTests(unittest.TestCase):
    def test_build_pretrajectory_sft_dataset_writes_expected_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_flist = _write_toy_graph(tmp_path)
            corpus_dir = _write_toy_corpus(tmp_path)
            rank_context = _write_rank_cache(tmp_path)
            store_manifest = tmp_path / "store_manifest.json"
            _write_json(store_manifest, {"format_version": "toy-graph-v1"})
            out_dir = tmp_path / "sft_out"

            result = sft.build_pretrajectory_sft_dataset(
                out_dir=out_dir,
                mixed_corpus_dir=corpus_dir,
                store_manifest_path=store_manifest,
                graph_flist=graph_flist,
                graph_layer_limit=None,
                graph_max_edges_per_layer=None,
                rank_cache_context_dir=rank_context,
                seed=11,
                max_rwr_seeds=5,
                context_modes=sft.CONTEXT_MODES,
                target_counts=None,
            )

            self.assertEqual(result["validation_report"]["fatal_error_count"], 0)
            for file_name in (
                "manifest.json",
                "validation_report.json",
                "canonical_objects.jsonl",
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
            ):
                self.assertTrue((out_dir / file_name).exists())

            records = (
                _read_jsonl(out_dir / "train.jsonl")
                + _read_jsonl(out_dir / "val.jsonl")
                + _read_jsonl(out_dir / "test.jsonl")
            )
            view_types = {record["metadata"]["view_type"] for record in records}
            context_modes = {record["metadata"]["context_mode"] for record in records}
            expected = {
                "entity_id_normalization",
                "graph_schema_provenance",
                "layer_tag_metadata",
                "layer_family_membership",
                "monoplex_edge_existence",
                "multiplex_edge_existence",
                "direct_neighbors_by_layer",
                "unique_multiplex_neighbors",
                "monoplex_shortest_path",
                "aggregate_multiplex_shortest_path",
                "path_layer_decomposition",
                "induced_subgraph",
                "connected_components",
                "degree_hub_bias",
                "layer_specific_claim_calibration",
                "no_edge_no_path_calibration",
                "closed_book_module_qa",
                "open_book_module_interpretation",
                "shadow_tool_recovery",
                "counterfactual_ablation",
                "critique_preference_sft",
                "rwr_loe_rank_lookup",
                "rwr_loe_rank_comparison",
                "rwr_loe_topk_membership",
                "rwr_neighborhood_interpretation",
                "mentor_ev_module_membership",
                "module_overlap_set_algebra",
                "module_containment_set_algebra",
                "module_source_distinction",
                "module_cohesion_summary",
                "tool_call_choice",
                "structured_state_update",
                "provenance_refusal_raw_cli",
            }
            self.assertTrue(expected <= view_types)
            self.assertEqual(set(sft.CONTEXT_MODES), context_modes)
            self.assertEqual(result["manifest"]["schema_version"], "pretrajectory-sft-v2")
            self.assertIn("record_count_by_curriculum_stage", result["manifest"])
            self.assertTrue((out_dir / "curriculum" / "stage1_entity_schema" / "train.jsonl").exists())
            self.assertTrue((out_dir / "curriculum" / "stage6_blend" / "train.jsonl").exists())

            edge_record = next(
                record
                for record in records
                if record["metadata"]["view_type"] == "monoplex_edge_existence"
                and record["metadata"].get("answer_label") == "yes"
            )
            self.assertIn("toy-graph-v1", edge_record["question"])
            self.assertIn("weight", edge_record["answer"])
            edge_labels = {
                record["metadata"].get("answer_label")
                for record in records
                if record["metadata"]["view_type"] == "monoplex_edge_existence"
            }
            self.assertTrue({"yes", "no"} <= edge_labels)
            multiplex_edge_labels = {
                record["metadata"].get("answer_label")
                for record in records
                if record["metadata"]["view_type"] == "multiplex_edge_existence"
            }
            self.assertTrue({"yes", "no"} <= multiplex_edge_labels)
            negative_edge = next(
                record
                for record in records
                if record["metadata"]["view_type"] == "monoplex_edge_existence"
                and record["metadata"].get("answer_label") == "no"
            )
            self.assertIn("If so, return the edge.", negative_edge["question"])
            self.assertTrue(negative_edge["answer"].startswith("No."))
            tool_record = next(record for record in records if record["metadata"]["context_mode"] == "tool_observation")
            self.assertIn("Tool observation:", tool_record["question"])

            rwr_record = next(record for record in records if record["metadata"]["view_type"] == "rwr_loe_rank_lookup")
            self.assertIn("rank 1", rwr_record["answer"])
            self.assertEqual(rwr_record["metadata"]["source"], "RWR_LOE_FULL_BRAIN")
            rwr_comparison = next(record for record in records if record["metadata"]["view_type"] == "rwr_loe_rank_comparison")
            self.assertIn("rank 1", rwr_comparison["answer"])
            self.assertEqual(rwr_comparison["metadata"]["mixture_bucket"], "rwr_vector_lookup")

            mentor_record = next(
                record
                for record in records
                if record["metadata"]["source"] == "MENTOR_GW_DENDROGRAM"
                and record["metadata"]["view_type"] == "closed_book_module_qa"
            )
            self.assertIn("MENTOR", mentor_record["answer"])
            module_overlap = next(record for record in records if record["metadata"]["view_type"] == "module_overlap_set_algebra")
            self.assertIn("Jaccard overlap", module_overlap["answer"])
            tool_choice = next(record for record in records if record["metadata"]["view_type"] == "tool_call_choice")
            self.assertIn("get_neighbors", tool_choice["answer"])

    def test_sampled_pretrajectory_dataset_respects_requested_split_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph_flist = _write_toy_graph(tmp_path)
            corpus_dir = _write_toy_corpus(tmp_path)
            rank_context = _write_rank_cache(tmp_path)
            store_manifest = tmp_path / "store_manifest.json"
            _write_json(store_manifest, {"format_version": "toy-graph-v1"})
            out_dir = tmp_path / "sft_sampled"

            result = sft.build_pretrajectory_sft_dataset(
                out_dir=out_dir,
                mixed_corpus_dir=corpus_dir,
                store_manifest_path=store_manifest,
                graph_flist=graph_flist,
                graph_layer_limit=None,
                graph_max_edges_per_layer=None,
                rank_cache_context_dir=rank_context,
                seed=3,
                max_rwr_seeds=5,
                target_counts={"train": 5, "val": 4, "test": 3},
            )

            counts = result["manifest"]["record_count_by_split"]
            self.assertLessEqual(counts["train"], 5)
            self.assertLessEqual(counts["val"], 4)
            self.assertLessEqual(counts["test"], 3)
            self.assertEqual(result["manifest"]["selected_record_count"], sum(counts.values()))
            self.assertEqual(result["validation_report"]["fatal_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
