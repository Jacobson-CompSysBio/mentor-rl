# tests/test_rwr_hpc_structured_backend.py

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.rwr_hpc_app_backend import RwrHpcAppResult
from runtime.rwr_hpc_cache import RwrHpcCache
from runtime.rwr_hpc_requests import (
    ComponentSummaryRequest,
    GeneLayersRequest,
    LayerAblationRequest,
    LayerStatsRequest,
    NodePerturbationRequest,
    PathLayerCountsRequest,
    RwrDistanceRequest,
    RwrDotSimilarityRequest,
    RwrEncodingSummaryRequest,
    RwrLoeRequest,
    RwrPearsonRequest,
    RwrRankRequest,
    RwrRankVectorSummaryRequest,
    RwrRequest,
    SeedEssentialityRequest,
    RwrSpearmanRequest,
    ShortestPathsRequest,
)
from runtime.rwr_hpc_structured_backend import (
    RwrHpcStructuredBackend,
    parse_component_summary,
    parse_gene_layers,
    parse_layer_stats,
    parse_path_layer_counts,
    parse_rwr_distance_matrix,
    parse_rwr_matrix_summary,
    parse_rwr_ranks,
    parse_rwr_loe_ranks,
    parse_seed_essentiality,
    parse_shortest_paths,
)


def _arg_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


class FakeRwrHpcAppBackend:
    """Fake app backend that writes a tiny RWR_LOE ranks file."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.seed_file_text: str | None = None
        self.query_file_text: str | None = None
        self.source_file_text: str | None = None
        self.target_file_text: str | None = None
        self.flist_text: str | None = None

    def run_app(
        self,
        tool_name: str,
        args: list[str],
        *,
        timeout_seconds: int = 300,
        cwd: str | Path | None = None,
    ) -> RwrHpcAppResult:
        self.calls.append((tool_name, list(args)))
        if "--flist" in args:
            self.flist_text = Path(_arg_value(args, "--flist")).read_text(encoding="utf-8")
        if "--seed_file" in args:
            self.seed_file_text = Path(_arg_value(args, "--seed_file")).read_text(encoding="utf-8")
        if "--query_file" in args:
            self.query_file_text = Path(_arg_value(args, "--query_file")).read_text(encoding="utf-8")
        if "--sources_file" in args:
            self.source_file_text = Path(_arg_value(args, "--sources_file")).read_text(encoding="utf-8")
        if "--targets_file" in args:
            self.target_file_text = Path(_arg_value(args, "--targets_file")).read_text(encoding="utf-8")

        output_dir = Path(_arg_value(args, "--output_dir"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if tool_name == "rwr":
            ranks_path = output_dir / "mentor_rwr_ranks.tsv"
            ranks_path.write_text(
                "\n".join(
                    [
                        "INDEX\tBRCA1\tTP53\tATM\tCHEK2",
                        "BRCA1_TP53\t1.000\t1.000\t2.000\t3.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            if "--record_encodings" in args:
                encodings_path = output_dir / "mentor_rwr_encoding_encodings.tsv"
                encodings_path.write_text(
                    "\n".join(
                        [
                            "INDEX\tBRCA1\tTP53\tATM\tCHEK2",
                            "BRCA1_TP53\t0.100\t0.200\t0.900\t0.700",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            seed_rows = [
                line.split("\t")[0].strip()
                for line in (self.seed_file_text or "").splitlines()
                if line.strip()
            ]
            if len(seed_rows) >= 2:
                metric = _arg_value(args, "--distance_metric") if "--distance_metric" in args else "spearman"
                distance = {
                    "spearman": "0.2500000000000000",
                    "pearson": "0.4000000000000000",
                    "dot": "0.8000000000000000",
                }.get(metric, "0.2500000000000000")
                distance_path = output_dir / f"mentor_rwr_pair_{metric}_dist_matrix.tsv"
                distance_path.write_text(
                    "\n".join(
                        [
                            f"INDEX\t{seed_rows[0]}\t{seed_rows[1]}",
                            f"{seed_rows[0]}\tNA\tNA",
                            f"{seed_rows[1]}\t{distance}\tNA",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
        elif tool_name == "shortest_paths":
            paths_path = output_dir / "mentor_shortest_paths_shortest_paths.tsv"
            paths_path.write_text(
                "\n".join(
                    [
                        "from\tto\tweight\ttype\tpathname\tpathlength\tpathelements",
                        "TP53\tCHEK2\t0.4\tppi\tpath_1\t2\tTP53->CHEK2->ATM",
                        "CHEK2\tATM\t0.3\tppi\tpath_1\t2\tTP53->CHEK2->ATM",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            layer_counts_path = output_dir / "mentor_shortest_paths_layer_counts.tsv"
            layer_counts_path.write_text("ppi\t2\ntf\t1\n", encoding="utf-8")
        elif tool_name == "gene_layer_map":
            (output_dir / "mentor_layer_mapnodes_by_layer.tsv").write_text(
                "\n".join(
                    [
                        "INDEX\tppi\ttf",
                        "TP53\t1\t0",
                        "ATM\t1\t1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "mentor_layer_mapnetwork_stats.tsv").write_text(
                "ppi\t3\t2\ntf\t2\t1\n",
                encoding="utf-8",
            )
        elif tool_name == "disconnected_components":
            (output_dir / "mentor_components_comp1_seeds.tsv").write_text(
                "mentor_components_comp1\tTP53\nmentor_components_comp1\tATM\n",
                encoding="utf-8",
            )
            (output_dir / "mentor_components_comp2_seeds.tsv").write_text(
                "mentor_components_comp2\tCHEK2\n",
                encoding="utf-8",
            )
        elif tool_name == "grin":
            (output_dir / "mentor_grin_gene_ranks.tsv").write_text(
                "INDEX\trank\nTP53\t2.000\nATM\t4.000\n",
                encoding="utf-8",
            )
            (output_dir / "mentor_grin_null_ranks.tsv").write_text(
                "INDEX\trank\nnull_rank_pos0\t5.000\nnull_rank_pos1\t6.000\n",
                encoding="utf-8",
            )
        elif tool_name == "rwr_ablation":
            metric = _arg_value(args, "--distance_metric") if "--distance_metric" in args else "spearman"
            (output_dir / f"{metric}_ablation_distance_matrix.tsv").write_text(
                "\n".join(
                    [
                        "INDEX\tppi\ttf",
                        "TP53\t0.200\t0.600",
                        "ATM\t0.400\t0.800",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        elif tool_name == "rwr_perturbation":
            metric = _arg_value(args, "--distance_metric") if "--distance_metric" in args else "spearman"
            (output_dir / f"{metric}_perturbation_distance_matrix.tsv").write_text(
                "\n".join(
                    [
                        "INDEX\tATM\tCHEK2",
                        "TP53\t0.300\t0.100",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            ranks_path = output_dir / "fake.ranks.tsv"
            ranks_path.write_text(
                "\n".join(
                    [
                        "NodeNames\tScores\trank",
                        "TP53\t0.900\t1",
                        "ATM\t0.500\t2",
                        "CHEK2\t0.400\t3",
                        "BRCA1\t0.300\t4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        return RwrHpcAppResult(
            tool_name=tool_name,
            executable=f"/fake/{tool_name}",
            command=[f"/fake/{tool_name}", *args],
            returncode=0,
            stdout="fake stdout",
            stderr="",
            payload={"stdout": "fake stdout", "stderr": "", "returncode": 0},
            provenance={"implementation": "fake"},
        )


def test_parse_rwr_loe_ranks_excludes_seeds_and_marks_queries(tmp_path: Path):
    ranks_path = tmp_path / "example.ranks.tsv"
    ranks_path.write_text(
        "\n".join(
            [
                "NodeNames\tScores\trank",
                "TP53\t0.900\t1",
                "ATM\t0.500\t2",
                "CHEK2\t0.400\t3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ranked = parse_rwr_loe_ranks(
        ranks_path,
        seed_genes=("TP53",),
        query_genes=("ATM",),
        top_k=10,
        exclude_seed_genes=True,
    )

    assert [item["gene"] for item in ranked] == ["ATM", "CHEK2"]
    assert ranked[0]["rank"] == 2
    assert ranked[0]["score"] == 0.5
    assert ranked[0]["is_seed"] is False
    assert ranked[0]["is_query"] is True
    assert ranked[1]["is_query"] is False


def test_parse_rwr_loe_ranks_applies_top_k_after_seed_exclusion(tmp_path: Path):
    ranks_path = tmp_path / "example.ranks.tsv"
    ranks_path.write_text(
        "\n".join(
            [
                "NodeNames\tScores\trank",
                "TP53\t0.900\t1",
                "ATM\t0.500\t2",
                "CHEK2\t0.400\t3",
                "BRCA1\t0.300\t4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ranked = parse_rwr_loe_ranks(
        ranks_path,
        seed_genes=("TP53",),
        query_genes=(),
        top_k=2,
        exclude_seed_genes=True,
    )

    assert [item["gene"] for item in ranked] == ["ATM", "CHEK2"]


def test_structured_backend_cache_miss_then_cache_hit(tmp_path: Path):
    flist = tmp_path / "network.flist"
    flist.write_text("layer1\t/tmp/layer1.tsv\n", encoding="utf-8")

    cache = RwrHpcCache(tmp_path / "cache")
    fake_app_backend = FakeRwrHpcAppBackend()

    backend = RwrHpcStructuredBackend(
        flist=flist,
        app_backend=fake_app_backend,
        cache=cache,
        scratch_root=tmp_path / "scratch",
        rwr_hpc_build_id="fake-build",
        timeout_seconds=60,
    )

    request = RwrLoeRequest.from_tool_arguments(
        {
            "seed_genes": ["TP53"],
            "query_genes": ["ATM"],
            "top_k": 2,
            "restart": 0.7,
            "delta": 0.5,
            "reduction_method": "geometric",
            "threshold": 1e-10,
            "exclude_seed_genes": True,
        }
    )

    first = backend.run_rwr_loe(request)

    assert first.provenance["cache_hit"] is False
    assert first.provenance["backend"] == "rwr_hpc_app"
    assert first.payload["tool_name"] == "rwr_loe"
    assert first.payload["seed_genes"] == ["TP53"]
    assert first.payload["query_genes"] == ["ATM"]
    assert [item["gene"] for item in first.payload["ranked_genes"]] == ["ATM", "CHEK2"]
    assert len(fake_app_backend.calls) == 1

    second = backend.run_rwr_loe(request)

    assert second.provenance["cache_hit"] is True
    assert second.payload == first.payload
    assert len(fake_app_backend.calls) == 1


def test_structured_backend_builds_hidden_file_args(tmp_path: Path):
    flist = tmp_path / "network.flist"
    flist.write_text("layer1\t/tmp/layer1.tsv\n", encoding="utf-8")

    fake_app_backend = FakeRwrHpcAppBackend()

    backend = RwrHpcStructuredBackend(
        flist=flist,
        app_backend=fake_app_backend,
        cache=None,
        scratch_root=tmp_path / "scratch",
        rwr_hpc_build_id="fake-build",
    )

    request = RwrLoeRequest.from_tool_arguments(
        {
            "seed_genes": ["TP53", "BRCA1"],
            "query_genes": ["ATM", "CHEK2"],
            "top_k": 2,
        }
    )

    backend.run_rwr_loe(request)

    assert len(fake_app_backend.calls) == 1

    tool_name, args = fake_app_backend.calls[0]
    assert tool_name == "rwr_loe"

    assert "--flist" in args
    assert "--seed_file" in args
    assert "--no_set_ids" in args
    assert "--no_edgelist_headers" in args
    assert "--query_file" in args
    assert "--output_dir" in args

    seed_file = Path(_arg_value(args, "--seed_file"))
    query_file = Path(_arg_value(args, "--query_file"))
    output_dir = Path(_arg_value(args, "--output_dir"))

    assert seed_file.name == "seed_genes.txt"
    assert query_file.name == "query_genes.txt"
    assert output_dir.name == "output"
    assert fake_app_backend.seed_file_text == "BRCA1\tTP53\n"
    assert fake_app_backend.query_file_text == "ATM\tCHEK2\n"


class RwrHpcStructuredBackendUnittestTests(unittest.TestCase):
    def test_parse_rwr_ranks_returns_ranked_genes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ranks_path = Path(tmp) / "mentor_rwr_ranks.tsv"
            ranks_path.write_text(
                "\n".join(
                    [
                        "INDEX\tTP53\tATM\tCHEK2",
                        "TP53\t1.000\t2.000\t3.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ranked = parse_rwr_ranks(ranks_path, seed_genes=("TP53",), top_k=2)

        self.assertEqual([item["gene"] for item in ranked], ["TP53", "ATM"])
        self.assertTrue(ranked[0]["is_seed"])
        self.assertEqual(ranked[1]["rank"], 2.0)

    def test_parse_rwr_distance_matrix_reads_lower_triangle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            distance_path = Path(tmp) / "mentor_rwr_pair_spearman_dist_matrix.tsv"
            distance_path.write_text(
                "\n".join(
                    [
                        "INDEX\tTP53\tATM",
                        "TP53\tNA\tNA",
                        "ATM\t0.2500000000000000\tNA",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            distance = parse_rwr_distance_matrix(
                distance_path,
                gene_a="TP53",
                gene_b="ATM",
            )

        self.assertEqual(distance, 0.25)

    def test_parse_shortest_paths_groups_edges_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths_path = Path(tmp) / "mentor_shortest_paths_shortest_paths.tsv"
            paths_path.write_text(
                "\n".join(
                    [
                        "from\tto\tweight\ttype\tpathname\tpathlength\tpathelements",
                        "TP53\tCHEK2\t0.4\tppi\tpath_1\t2\tTP53->CHEK2->ATM",
                        "CHEK2\tATM\t0.3\tppi\tpath_1\t2\tTP53->CHEK2->ATM",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            paths = parse_shortest_paths(paths_path)

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["source_gene"], "TP53")
        self.assertEqual(paths[0]["target_gene"], "ATM")
        self.assertEqual(paths[0]["path_genes"], ["TP53", "CHEK2", "ATM"])
        self.assertEqual(len(paths[0]["edges"]), 2)

    def test_parse_layer_and_component_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nodes_by_layer = root / "nodes_by_layer.tsv"
            nodes_by_layer.write_text(
                "INDEX\tppi\ttf\nTP53\t1\t0\nATM\t1\t1\n",
                encoding="utf-8",
            )
            network_stats = root / "network_stats.tsv"
            network_stats.write_text("ppi\t3\t2\ntf\t2\t1\n", encoding="utf-8")
            layer_counts = root / "layer_counts.tsv"
            layer_counts.write_text("ppi\t2\ntf\t1\n", encoding="utf-8")
            (root / "run_comp1_seeds.tsv").write_text("run_comp1\tTP53\nrun_comp1\tATM\n", encoding="utf-8")
            (root / "run_comp2_seeds.tsv").write_text("run_comp2\tCHEK2\n", encoding="utf-8")

            gene_layers = parse_gene_layers(nodes_by_layer, gene="atm")
            stats = parse_layer_stats(network_stats, top_k=1)
            counts = parse_path_layer_counts(layer_counts)
            components = parse_component_summary(root, genes=("TP53",), max_components=1)

        self.assertEqual(gene_layers["layers"], ["ppi", "tf"])
        self.assertEqual(stats[0]["layer"], "ppi")
        self.assertEqual(counts[0], {"layer": "ppi", "edge_count": 2})
        self.assertEqual(components["total_components"], 2)
        self.assertEqual(components["gene_membership"]["TP53"], "run_comp1")

    def test_parse_seed_essentiality_and_matrix_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gene_ranks = root / "gene_ranks.tsv"
            gene_ranks.write_text("INDEX\trank\nTP53\t2.000\nATM\t4.000\n", encoding="utf-8")
            null_ranks = root / "null_ranks.tsv"
            null_ranks.write_text(
                "INDEX\trank\nnull_rank_pos0\t5.000\nnull_rank_pos1\t6.000\n",
                encoding="utf-8",
            )
            matrix = root / "encodings.tsv"
            matrix.write_text(
                "INDEX\tTP53\tATM\tCHEK2\nTP53\t0.100\t0.900\t0.300\n",
                encoding="utf-8",
            )

            essentiality = parse_seed_essentiality(gene_ranks, null_ranks)
            summary = parse_rwr_matrix_summary(
                matrix,
                seed_genes=("TP53",),
                include_seed_genes=False,
                top_k=2,
            )

        self.assertEqual(essentiality[0]["gene"], "TP53")
        self.assertEqual(essentiality[0]["essentiality_delta"], 3.0)
        self.assertEqual([item["gene"] for item in summary], ["ATM", "CHEK2"])

    def test_run_rwr_uses_filtered_flist_for_selected_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n/tmp/tf.tsv\ttf\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=None,
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            result = backend.run_rwr(
                RwrRequest.from_tool_arguments(
                    {"seed_genes": ["TP53", "BRCA1"], "layers": ["ppi"], "top_k": 3}
                )
            )

        self.assertEqual(fake_app_backend.calls[0][0], "rwr")
        self.assertIn("--no_edgelist_headers", fake_app_backend.calls[0][1])
        self.assertEqual(fake_app_backend.seed_file_text, "BRCA1\tTP53\n")
        self.assertEqual(fake_app_backend.flist_text, "/tmp/ppi.tsv\tppi\n")
        self.assertEqual(result.payload["tool_name"], "rwr")
        self.assertEqual([item["gene"] for item in result.payload["ranked_genes"]], ["BRCA1", "TP53", "ATM"])

    def test_run_rwr_accepts_single_layer_alias_for_monoplex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n/tmp/tf.tsv\ttf\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=None,
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            result = backend.run_rwr(
                RwrRequest.from_tool_arguments(
                    {"seed_genes": ["TP53"], "layer": "tf", "top_k": 2}
                )
            )

        self.assertEqual(fake_app_backend.flist_text, "/tmp/tf.tsv\ttf\n")
        self.assertEqual(result.payload["layers"], ["tf"])

    def test_run_rwr_rejects_missing_selected_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=FakeRwrHpcAppBackend(),
                cache=None,
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            with self.assertRaisesRegex(Exception, "missing"):
                backend.run_rwr(
                    RwrRequest.from_tool_arguments(
                        {"seed_genes": ["TP53"], "layers": ["missing"], "top_k": 2}
                    )
                )

    def test_run_shortest_paths_hides_source_and_target_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=None,
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            result = backend.run_shortest_paths(
                ShortestPathsRequest.from_tool_arguments(
                    {
                        "source_genes": ["TP53"],
                        "target_genes": ["ATM"],
                        "merge_method": "max",
                        "ignore_weights": True,
                        "max_paths": 1,
                    }
                )
            )

        tool_name, args = fake_app_backend.calls[0]
        self.assertEqual(tool_name, "shortest_paths")
        self.assertIn("--targets_file", args)
        self.assertIn("--no_edgelist_headers", args)
        self.assertIn("--ignore_weights", args)
        self.assertEqual(fake_app_backend.source_file_text, "TP53\n")
        self.assertEqual(fake_app_backend.target_file_text, "ATM\n")
        self.assertEqual(result.payload["tool_name"], "shortest_paths")
        self.assertEqual(result.payload["paths"][0]["path_genes"], ["TP53", "CHEK2", "ATM"])

    def test_no_edgelist_header_flag_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=None,
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
                no_edgelist_headers=False,
            )

            backend.run_rwr_loe(
                RwrLoeRequest.from_tool_arguments(
                    {"seed_genes": ["TP53"], "query_genes": ["ATM"], "top_k": 1}
                )
            )

        self.assertNotIn("--no_edgelist_headers", fake_app_backend.calls[0][1])

    def test_run_get_rank_uses_rwr_rank_matrix_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=RwrHpcCache(root / "cache"),
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            request = RwrRankRequest.from_tool_arguments(
                {"source_gene": "TP53", "target_gene": "ATM"}
            )
            first = backend.run_get_rank(request)
            second = backend.run_get_rank(request)

        self.assertEqual(first.payload["target_rank"], 2.0)
        self.assertEqual(first.payload["rank_semantics"].endswith("not LoE"), True)
        self.assertFalse(first.provenance["cache_hit"])
        self.assertTrue(second.provenance["cache_hit"])
        self.assertEqual([call[0] for call in fake_app_backend.calls], ["rwr"])

    def test_run_get_distance_uses_pair_distance_matrix_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=RwrHpcCache(root / "cache"),
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            request = RwrDistanceRequest.from_tool_arguments(
                {"gene_a": "TP53", "gene_b": "ATM", "distance_metric": "spearman"}
            )
            first = backend.run_get_distance(request)
            second = backend.run_get_distance(request)

        self.assertEqual(first.payload["distance"], 0.25)
        self.assertEqual(first.payload["dissimilarity"], 0.25)
        self.assertFalse(first.provenance["cache_hit"])
        self.assertTrue(second.provenance["cache_hit"])
        self.assertEqual(fake_app_backend.seed_file_text, "TP53\nATM\n")
        self.assertEqual([call[0] for call in fake_app_backend.calls], ["rwr"])

    def test_run_get_spearman_derives_correlation_from_cached_distance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=RwrHpcCache(root / "cache"),
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            request = RwrSpearmanRequest.from_tool_arguments(
                {"gene_a": "TP53", "gene_b": "ATM"}
            )
            first = backend.run_get_spearman(request)
            second = backend.run_get_spearman(request)

        self.assertEqual(first.payload["spearman_correlation"], 0.75)
        self.assertEqual(first.payload["spearman_distance"], 0.25)
        self.assertFalse(first.provenance["cache_hit"])
        self.assertTrue(second.provenance["cache_hit"])
        self.assertEqual([call[0] for call in fake_app_backend.calls], ["rwr"])

    def test_run_new_pairwise_and_summary_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n/tmp/tf.tsv\ttf\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=RwrHpcCache(root / "cache"),
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            pearson = backend.run_get_pearson(
                RwrPearsonRequest.from_tool_arguments({"gene_a": "TP53", "gene_b": "ATM"})
            )
            dot = backend.run_get_dot_similarity(
                RwrDotSimilarityRequest.from_tool_arguments({"gene_a": "TP53", "gene_b": "ATM"})
            )
            rank_summary = backend.run_get_rank_vector_summary(
                RwrRankVectorSummaryRequest.from_tool_arguments(
                    {"seed_genes": ["TP53"], "top_k": 2, "include_seed_genes": False}
                )
            )
            encoding_summary = backend.run_get_encoding_summary(
                RwrEncodingSummaryRequest.from_tool_arguments(
                    {"seed_genes": ["TP53"], "top_k": 2, "include_seed_genes": False}
                )
            )

        self.assertEqual(pearson.payload["pearson_correlation"], 0.6)
        self.assertEqual(dot.payload["dot_similarity"], 0.8)
        self.assertEqual([item["gene"] for item in rank_summary.payload["rank_summary"]], ["BRCA1", "ATM"])
        self.assertEqual([item["gene"] for item in encoding_summary.payload["encoding_summary"]], ["ATM", "CHEK2"])

    def test_run_layer_component_and_effect_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flist = root / "network.flist"
            flist.write_text("/tmp/ppi.tsv\tppi\n/tmp/tf.tsv\ttf\n", encoding="utf-8")
            fake_app_backend = FakeRwrHpcAppBackend()
            backend = RwrHpcStructuredBackend(
                flist=flist,
                app_backend=fake_app_backend,
                cache=RwrHpcCache(root / "cache"),
                scratch_root=root / "scratch",
                rwr_hpc_build_id="fake-build",
            )

            gene_layers = backend.run_get_gene_layers(
                GeneLayersRequest.from_tool_arguments({"gene": "ATM"})
            )
            nodes_by_layer = backend.run_get_nodes_by_layer(
                GeneLayersRequest.from_tool_arguments({"gene": "TP53"})
            )
            layer_stats = backend.run_get_layer_stats(
                LayerStatsRequest.from_tool_arguments({"top_k": 2})
            )
            path_layer_counts = backend.run_get_path_layer_counts(
                PathLayerCountsRequest.from_tool_arguments(
                    {"source_genes": ["TP53"], "target_genes": ["ATM"], "top_k": 1}
                )
            )
            components = backend.run_get_component_summary(
                ComponentSummaryRequest.from_tool_arguments({"genes": ["TP53"], "max_components": 1})
            )
            essentiality = backend.run_get_seed_essentiality(
                SeedEssentialityRequest.from_tool_arguments(
                    {"seed_genes": ["TP53", "ATM"], "n_samples_null_dist": 10}
                )
            )
            ablation = backend.run_get_layer_ablation(
                LayerAblationRequest.from_tool_arguments({"seed_genes": ["TP53", "ATM"], "top_k": 1})
            )
            perturbation = backend.run_get_node_perturbation(
                NodePerturbationRequest.from_tool_arguments(
                    {"seed_genes": ["TP53"], "perturb_genes": ["ATM", "CHEK2"], "top_k": 1}
                )
            )

        self.assertEqual(gene_layers.payload["layers"], ["ppi", "tf"])
        self.assertEqual(nodes_by_layer.payload["tool_name"], "get_nodes_by_layer")
        self.assertEqual(layer_stats.payload["layer_stats"][0]["layer"], "ppi")
        self.assertEqual(path_layer_counts.payload["layer_counts"], [{"layer": "ppi", "edge_count": 2}])
        self.assertEqual(components.payload["total_components"], 2)
        self.assertEqual(essentiality.payload["essentiality"][0]["gene"], "TP53")
        self.assertEqual(ablation.payload["layer_effects"][0]["layer"], "tf")
        self.assertEqual(perturbation.payload["perturbation_effects"][0]["perturb_gene"], "ATM")
