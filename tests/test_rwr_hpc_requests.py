"""unit test for rwr_hpc_request.py"""
import unittest

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

class RwrLoeRequestTests(unittest.TestCase):
    # test that normalization of gene lists work as expected
    def test_from_tool_arguments_normalizes_gene_lists(self) -> None:
        request = RwrLoeRequest.from_tool_arguments({
            "seed_genes": [" tp53 ", "BRCA1", "tp53"],
            "query_genes": ["atm"],
            "top_k": 25,
        }
    )

        self.assertEqual(request.seed_genes, ("BRCA1", "TP53"))
        self.assertEqual(request.query_genes, ("ATM",))
        self.assertEqual(request.top_k, 25)

    # test that the call rejects file-like arguments
    def test_rejects_file_or_cli_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "file/path/CLI"):
            RwrLoeRequest.from_tool_arguments({
                "seed_genes": ["tp53"],
                "seed_file": "/tmp/seeds.txt",
                }
            )
    
    # test that invalid ranges for params are rejected
    def test_rejects_invalid_ranges(self) -> None:
        bad_cases = [
            {"seed_genes": ["TP53"], "top_k": 0},
            {"seed_genes": ["TP53"], "restart": 1.5},
            {"seed_genes": ["TP53"], "delta": -0.1},
            {"seed_genes": ["TP53"], "threshold": 0},
            {"seed_genes": ["TP53"], "reduction_method": "bad"},            
        ]

        for args in bad_cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    RwrLoeRequest.from_tool_arguments(args)


class RwrRequestTests(unittest.TestCase):
    def test_from_tool_arguments_accepts_seed_alias_and_layers(self) -> None:
        request = RwrRequest.from_tool_arguments(
            {
                "seeds": ["tp53", "BRCA1", "tp53"],
                "layers": [" ppi ", "tf"],
                "top_k": 10,
            }
        )

        self.assertEqual(request.seed_genes, ("BRCA1", "TP53"))
        self.assertEqual(request.layers, ("ppi", "tf"))
        self.assertEqual(request.top_k, 10)

    def test_from_tool_arguments_accepts_single_layer_alias(self) -> None:
        request = RwrRequest.from_tool_arguments(
            {
                "seed_genes": ["tp53"],
                "layer": " ppi ",
            }
        )

        self.assertEqual(request.seed_genes, ("TP53",))
        self.assertEqual(request.layers, ("ppi",))

    def test_rejects_file_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "file/path/CLI"):
            RwrRequest.from_tool_arguments(
                {"seed_genes": ["TP53"], "output_dir": "/tmp/out"}
            )


class RwrDerivedRequestTests(unittest.TestCase):
    def test_rank_request_normalizes_source_target_and_layers(self) -> None:
        request = RwrRankRequest.from_tool_arguments(
            {"source_gene": " tp53 ", "target_gene": "atm", "layer": " ppi "}
        )

        self.assertEqual(request.source_gene, "TP53")
        self.assertEqual(request.target_gene, "ATM")
        self.assertEqual(request.layers, ("ppi",))

    def test_distance_request_accepts_metric_and_aliases(self) -> None:
        request = RwrDistanceRequest.from_tool_arguments(
            {"source": "tp53", "target": "atm", "distance_metric": "pearson"}
        )

        self.assertEqual(request.gene_a, "TP53")
        self.assertEqual(request.gene_b, "ATM")
        self.assertEqual(request.distance_metric, "pearson")

    def test_spearman_request_rejects_file_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "file/path/CLI"):
            RwrSpearmanRequest.from_tool_arguments(
                {"gene_a": "TP53", "gene_b": "ATM", "seed_file": "/tmp/seeds.txt"}
            )

    def test_distance_request_accepts_dot_metric(self) -> None:
        request = RwrDistanceRequest.from_tool_arguments(
            {"gene_a": "TP53", "gene_b": "ATM", "distance_metric": "dot"}
        )

        self.assertEqual(request.distance_metric, "dot")

    def test_distance_request_rejects_unknown_metric(self) -> None:
        with self.assertRaises(ValueError):
            RwrDistanceRequest.from_tool_arguments(
                {"gene_a": "TP53", "gene_b": "ATM", "distance_metric": "kendall"}
            )

    def test_new_pairwise_requests_normalize_genes(self) -> None:
        pearson = RwrPearsonRequest.from_tool_arguments({"gene_a": "tp53", "gene_b": "atm"})
        dot = RwrDotSimilarityRequest.from_tool_arguments({"source": "tp53", "target": "atm"})

        self.assertEqual(pearson.gene_a, "TP53")
        self.assertEqual(dot.gene_b, "ATM")

    def test_vector_summary_requests_normalize_seed_genes(self) -> None:
        rank_request = RwrRankVectorSummaryRequest.from_tool_arguments(
            {"seed_genes": ["tp53", "atm"], "include_seed_genes": False}
        )
        encoding_request = RwrEncodingSummaryRequest.from_tool_arguments(
            {"seeds": ["tp53"], "top_k": 3}
        )

        self.assertEqual(rank_request.seed_genes, ("ATM", "TP53"))
        self.assertFalse(rank_request.include_seed_genes)
        self.assertEqual(encoding_request.top_k, 3)

    def test_layer_and_component_requests_normalize_arguments(self) -> None:
        gene_layers = GeneLayersRequest.from_tool_arguments({"gene": "tp53"})
        layer_stats = LayerStatsRequest.from_tool_arguments({"top_k": 5, "sort_by": "node_count"})
        component = ComponentSummaryRequest.from_tool_arguments({"genes": ["tp53"], "max_components": 3})

        self.assertEqual(gene_layers.gene, "TP53")
        self.assertEqual(layer_stats.sort_by, "node_count")
        self.assertEqual(component.genes, ("TP53",))

    def test_heavy_rwr_requests_normalize_arguments(self) -> None:
        path_counts = PathLayerCountsRequest.from_tool_arguments(
            {"source": "tp53", "target": "atm", "top_k": 2}
        )
        essentiality = SeedEssentialityRequest.from_tool_arguments(
            {"seed_genes": ["tp53", "atm"], "n_samples_null_dist": 10}
        )
        ablation = LayerAblationRequest.from_tool_arguments(
            {"seed_genes": ["tp53"], "distance_metric": "cos"}
        )
        perturbation = NodePerturbationRequest.from_tool_arguments(
            {"seed_genes": ["tp53"], "perturb_genes": ["atm"], "distance_metric": "dot"}
        )

        self.assertEqual(path_counts.source_genes, ("TP53",))
        self.assertEqual(essentiality.n_samples_null_dist, 10)
        self.assertEqual(ablation.distance_metric, "cos")
        self.assertEqual(perturbation.perturb_genes, ("ATM",))


class ShortestPathsRequestTests(unittest.TestCase):
    def test_from_tool_arguments_accepts_legacy_source_target_aliases(self) -> None:
        request = ShortestPathsRequest.from_tool_arguments(
            {
                "source": "tp53",
                "target": "atm",
                "merge_method": "mean",
                "ignore_weights": True,
                "max_paths": 3,
            }
        )

        self.assertEqual(request.source_genes, ("TP53",))
        self.assertEqual(request.target_genes, ("ATM",))
        self.assertEqual(request.merge_method, "mean")
        self.assertTrue(request.ignore_weights)
        self.assertEqual(request.max_paths, 3)

    def test_rejects_invalid_merge_method(self) -> None:
        with self.assertRaises(ValueError):
            ShortestPathsRequest.from_tool_arguments(
                {"source_genes": ["TP53"], "merge_method": "bad"}
            )

if __name__ == "__main__":
    unittest.main()
