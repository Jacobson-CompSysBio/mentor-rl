# tests/test_rwr_hpc_structured_backend.py

from __future__ import annotations

from pathlib import Path

from runtime.rwr_hpc_app_backend import RwrHpcAppResult
from runtime.rwr_hpc_cache import RwrHpcCache
from runtime.rwr_hpc_requests import RwrLoeRequest
from runtime.rwr_hpc_structured_backend import (
    RwrHpcStructuredBackend,
    parse_rwr_loe_ranks,
)


def _arg_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]


class FakeRwrHpcAppBackend:
    """Fake app backend that writes a tiny RWR_LOE ranks file."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run_app(
        self,
        tool_name: str,
        args: list[str],
        *,
        timeout_seconds: int = 300,
        cwd: str | Path | None = None,
    ) -> RwrHpcAppResult:
        self.calls.append((tool_name, list(args)))

        output_dir = Path(_arg_value(args, "--output_dir"))
        output_dir.mkdir(parents=True, exist_ok=True)

        ranks_path = output_dir / "fake.ranks.tsv"
        ranks_path.write_text(
            "\n".join(
                [
                    "gene\tscore\trank",
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
            executable="/fake/rwr_loe",
            command=["/fake/rwr_loe", *args],
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
                "gene\tscore\trank",
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
                "gene\tscore\trank",
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
            "seed_genes": ["TP53"],
            "query_genes": ["ATM"],
            "top_k": 2,
        }
    )

    backend.run_rwr_loe(request)

    assert len(fake_app_backend.calls) == 1

    tool_name, args = fake_app_backend.calls[0]
    assert tool_name == "rwr_loe"

    assert "--flist" in args
    assert "--seed_file" in args
    assert "--query_file" in args
    assert "--output_dir" in args

    seed_file = Path(_arg_value(args, "--seed_file"))
    query_file = Path(_arg_value(args, "--query_file"))
    output_dir = Path(_arg_value(args, "--output_dir"))

    assert seed_file.name == "seed_genes.txt"
    assert query_file.name == "query_genes.txt"
    assert output_dir.name == "output"