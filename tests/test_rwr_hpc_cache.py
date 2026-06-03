# tests/test_rwr_hpc_cache.py

from pathlib import Path

from runtime.rwr_hpc_cache import (
    CACHE_SCHEMA_VERSION,
    RwrHpcCache,
    file_sha256,
    make_rwr_loe_cache_key,
    stable_json_hash,
)
from runtime.rwr_hpc_requests import RwrLoeRequest


def make_request_payload(**overrides):
    args = {
        "seed_genes": ["TP53", "BRCA1"],
        "query_genes": ["ATM", "CHEK2"],
        "top_k": 25,
        "restart": 0.7,
        "delta": 0.5,
        "reduction_method": "geometric",
        "threshold": 1e-10,
        "exclude_seed_genes": True,
    }
    args.update(overrides)
    return RwrLoeRequest.from_tool_arguments(args).cache_key_payload()


def test_stable_json_hash_ignores_dict_key_order():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}

    assert stable_json_hash(a) == stable_json_hash(b)


def test_file_sha256_changes_when_file_changes(tmp_path: Path):
    path = tmp_path / "network.flist"

    path.write_text("layer1\t/path/a.tsv\n", encoding="utf-8")
    first = file_sha256(path)

    path.write_text("layer1\t/path/b.tsv\n", encoding="utf-8")
    second = file_sha256(path)

    assert first != second


def test_make_rwr_loe_cache_key_is_stable_for_same_logical_request():
    payload_a = make_request_payload(
        seed_genes=["TP53", "BRCA1", "TP53"],
        query_genes=["CHEK2", "ATM"],
    )
    payload_b = make_request_payload(
        seed_genes=["BRCA1", "TP53"],
        query_genes=["ATM", "CHEK2"],
    )

    key_a = make_rwr_loe_cache_key(
        request_payload=payload_a,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id",
    )
    key_b = make_rwr_loe_cache_key(
        request_payload=payload_b,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id",
    )

    assert key_a == key_b


def test_make_rwr_loe_cache_key_changes_when_request_changes():
    payload_a = make_request_payload(restart=0.7)
    payload_b = make_request_payload(restart=0.6)

    key_a = make_rwr_loe_cache_key(
        request_payload=payload_a,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id",
    )
    key_b = make_rwr_loe_cache_key(
        request_payload=payload_b,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id",
    )

    assert key_a != key_b


def test_make_rwr_loe_cache_key_changes_when_network_changes():
    payload = make_request_payload()

    key_a = make_rwr_loe_cache_key(
        request_payload=payload,
        network_flist_sha256="network-hash-a",
        rwr_hpc_build_id="build-id",
    )
    key_b = make_rwr_loe_cache_key(
        request_payload=payload,
        network_flist_sha256="network-hash-b",
        rwr_hpc_build_id="build-id",
    )

    assert key_a != key_b


def test_make_rwr_loe_cache_key_changes_when_build_changes():
    payload = make_request_payload()

    key_a = make_rwr_loe_cache_key(
        request_payload=payload,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id-a",
    )
    key_b = make_rwr_loe_cache_key(
        request_payload=payload,
        network_flist_sha256="network-hash",
        rwr_hpc_build_id="build-id-b",
    )

    assert key_a != key_b


def test_cache_put_then_get_returns_payload_provenance_and_request(tmp_path: Path):
    cache = RwrHpcCache(tmp_path)
    request = make_request_payload()
    payload = {
        "tool_name": "rwr_loe",
        "ranked_genes": [
            {"gene": "ATM", "rank": 1, "score": 0.123},
        ],
    }
    provenance = {
        "backend": "rwr_hpc_app",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }

    cache.put(
        "rwr_loe",
        "abc123",
        request=request,
        payload=payload,
        provenance=provenance,
        raw_stdout="stdout text",
        raw_stderr="stderr text",
    )

    cached = cache.get("rwr_loe", "abc123")

    assert cached is not None
    assert cached["request"] == request
    assert cached["payload"] == payload
    assert cached["provenance"] == provenance

    entry_dir = tmp_path / "rwr_loe" / "abc123"
    assert (entry_dir / "request.json").exists()
    assert (entry_dir / "payload.json").exists()
    assert (entry_dir / "provenance.json").exists()
    assert (entry_dir / "raw_stdout.txt").read_text(encoding="utf-8") == "stdout text"
    assert (entry_dir / "raw_stderr.txt").read_text(encoding="utf-8") == "stderr text"
    assert (entry_dir / "raw_outputs").is_dir()


def test_cache_get_returns_none_for_missing_entry(tmp_path: Path):
    cache = RwrHpcCache(tmp_path)

    assert cache.get("rwr_loe", "missing") is None


def test_cache_get_returns_none_for_incomplete_entry(tmp_path: Path):
    cache = RwrHpcCache(tmp_path)
    entry_dir = tmp_path / "rwr_loe" / "abc123"
    entry_dir.mkdir(parents=True)
    (entry_dir / "payload.json").write_text("{}", encoding="utf-8")

    assert cache.get("rwr_loe", "abc123") is None