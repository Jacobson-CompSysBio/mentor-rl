from __future__ import annotations

import pytest

from scripts.build_world_model_v2_s0_tokenizers import (
    ATOMIC_REGISTRY_SCHEMA,
    FULLY_ATOMIC_METHOD,
    build_fully_atomic_registry,
    stable_sha256,
)


def _registry(fit_values: dict[str, set[str]]) -> dict:
    """Build one small full atomic registry for a unit test."""

    return build_fully_atomic_registry(
        fit_values,
        parent_manifest_sha256="a" * 64,
        parent_train_sha256="b" * 64,
    )


def test_full_atomic_registry_has_stable_namespace_tokens() -> None:
    registry = _registry(
        {
            "ensembl_human_gene": {
                "ENSG00000000002",
                "ENSG00000000001",
            },
            "human_gene_symbol": {"TP53", "A1BG"},
        }
    )

    assert registry["schema_version"] == ATOMIC_REGISTRY_SCHEMA
    assert registry["method"] == FULLY_ATOMIC_METHOD
    assert registry["value_count"] == 4
    assert registry["namespaces"]["ensembl_human_gene"]["entries"] == [
        {
            "marker": "<|s0atom_s0ens_00000|>",
            "value": "ENSG00000000001",
        },
        {
            "marker": "<|s0atom_s0ens_00001|>",
            "value": "ENSG00000000002",
        },
    ]
    assert registry["namespaces"]["human_gene_symbol"]["entries"] == [
        {"marker": "<|s0atom_s0sym_00000|>", "value": "A1BG"},
        {"marker": "<|s0atom_s0sym_00001|>", "value": "TP53"},
    ]
    identity = {
        key: value
        for key, value in registry.items()
        if key != "manifest_sha256"
    }
    assert registry["manifest_sha256"] == stable_sha256(identity)


@pytest.mark.parametrize(
    "fit_values",
    [
        {
            "ensembl_human_gene": {"ENSG1"},
            "human_gene_symbol": {"TP53"},
        },
        {
            "ensembl_human_gene": {"ENSG00000000001"},
            "human_gene_symbol": {" TP53"},
        },
    ],
)
def test_full_atomic_registry_rejects_noncanonical_values(
    fit_values: dict[str, set[str]],
) -> None:
    with pytest.raises(ValueError):
        _registry(fit_values)
