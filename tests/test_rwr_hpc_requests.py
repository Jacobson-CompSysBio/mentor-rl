"""unit test for rwr_hpc_request.py"""
import unittest

from runtime.rwr_hpc_requests import RwrLoeRequest

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

if __name__ == "__main__":
    unittest.main()