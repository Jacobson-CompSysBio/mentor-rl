import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from runtime.backend import CompiledRuntimeBackend
from runtime.environment import RuntimeEnvironment
from runtime.tools import (
    build_multiplex_index,
    get_neighbors,
    induce_subgraph,
    shortest_path,
    rwr_multiplex,
)
from scripts.build_multiplex_store import build_store
from utils.multiplex import Multiplex


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_test_inputs(base_dir: Path) -> Path:
    layer_one = base_dir / "layer_one.tsv"
    layer_one.write_text(
        "\n".join(
            [
                "ENSG1 ENSG2 1.0",
                "ENSG2 ENSG3 0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    layer_two = base_dir / "layer_two.tsv"
    layer_two.write_text(
        "\n".join(
            [
                "ENSG1 ENSG3 0.8",
                "ENSG3 ENSG4 0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    flist_path = base_dir / "test.flist"
    flist_path.write_text(
        "\n".join(
            [
                f"{layer_one}\tppi\t1",
                f"{layer_two}\tcoexp\t1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return flist_path


def _build_reference_multiplex() -> Multiplex:
    multiplex = Multiplex()

    ppi = nx.Graph()
    ppi.add_edge("ENSG1", "ENSG2", weight=1.0)
    ppi.add_edge("ENSG2", "ENSG3", weight=0.9)
    multiplex.add_layer(ppi, "ppi")

    coexp = nx.Graph()
    coexp.add_edge("ENSG1", "ENSG3", weight=0.8)
    coexp.add_edge("ENSG3", "ENSG4", weight=0.7)
    multiplex.add_layer(coexp, "coexp")

    return multiplex


def _build_runtime_library(build_dir: Path) -> Path:
    cmake_binary = shutil.which("cmake")
    if cmake_binary is None:
        raise unittest.SkipTest("cmake is not available in this environment.")

    subprocess.run(
        [cmake_binary, "-S", str(REPO_ROOT / "cpp_runtime"), "-B", str(build_dir)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    subprocess.run(
        [cmake_binary, "--build", str(build_dir)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    library_path = build_dir / "libmentor_runtime.so"
    if not library_path.exists():
        raise FileNotFoundError(f"Compiled runtime library not found at {library_path}.")
    return library_path


class CompiledRuntimeSmokeTests(unittest.TestCase):
    def test_compiled_backend_matches_reference_for_core_graph_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            flist_path = _write_test_inputs(base_dir)
            store_dir = base_dir / "store"
            build_store(multiplex_flist=str(flist_path), out_dir=str(store_dir))
            library_path = _build_runtime_library(base_dir / "cpp_build")

            backend = CompiledRuntimeBackend(
                store_dir=str(store_dir),
                library_path=str(library_path),
            )
            reference_index = build_multiplex_index(_build_reference_multiplex())

            cpp_neighbors = backend.get_neighbors("ENSG1", layers=["ppi", "coexp"]).payload
            ref_neighbors = get_neighbors(reference_index, "ENSG1", layers=["ppi", "coexp"]).payload
            self.assertEqual(cpp_neighbors, ref_neighbors)

            cpp_subgraph = backend.induce_subgraph(
                ["ENSG1", "ENSG2", "ENSG4"],
                layers=["ppi", "coexp"],
            ).payload
            ref_subgraph = induce_subgraph(
                reference_index,
                ["ENSG1", "ENSG2", "ENSG4"],
                layers=["ppi", "coexp"],
            ).payload
            self.assertEqual(cpp_subgraph, ref_subgraph)

            cpp_path = backend.shortest_path("ENSG2", "ENSG4").payload
            ref_path = shortest_path(reference_index, "ENSG2", "ENSG4").payload
            self.assertEqual(cpp_path, ref_path)

            cpp_rwr = backend.rwr_monoplex(
                ["ENSG1"],
                layer_name="ppi",
                top_k=3,
                restart_probability=0.35,
            ).payload
            self.assertEqual(cpp_rwr["results"][0]["gene_id"], "ENSG1")
            self.assertEqual(cpp_rwr["layer_name"], "ppi")

            cpp_multi_rwr = backend.rwr_multiplex(
                ["ENSG1"],
                top_k=4,
                restart_probability=0.35,
            ).payload
            self.assertEqual(cpp_multi_rwr["results"][0]["gene_id"], "ENSG1")
            self.assertEqual(cpp_multi_rwr["active_layers"], ["ppi", "coexp"])
            ref_multi_rwr = rwr_multiplex(reference_index, ["ENSG1"], top_k=4).payload
            self.assertEqual(
                [item["gene_id"] for item in cpp_multi_rwr["results"]],
                [item["gene_id"] for item in ref_multi_rwr["results"]],
            )

            backend.close()

    def test_runtime_environment_can_use_compiled_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            flist_path = _write_test_inputs(base_dir)
            store_dir = base_dir / "store"
            build_store(multiplex_flist=str(flist_path), out_dir=str(store_dir))
            library_path = _build_runtime_library(base_dir / "cpp_build")

            environment = RuntimeEnvironment(
                store_dir=str(store_dir),
                compiled_library_path=str(library_path),
                mygene_cache={},
            )

            summary = environment.describe()

            self.assertEqual(summary["graph_backend_kind"], "compiled_cpp")
            self.assertEqual(summary["layer_count"], 2)
            self.assertEqual(summary["gene_count"], 4)


if __name__ == "__main__":
    unittest.main()
