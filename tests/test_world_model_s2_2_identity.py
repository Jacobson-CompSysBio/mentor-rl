"""Test source-backed supplied identities without borrowing S1 graph membership."""

from copy import deepcopy
import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s2_2_identity import (
    RegistryContext,
    get_symbols,
    identity_observations,
    load_registry_context,
    validate_identity_observations,
)


@pytest.fixture
def registry():
    """Use explicit unit-test identities, including null and casefold ambiguity."""

    return RegistryContext(
        release="116", registry_id="ensembl_homo_sapiens_116_test",
        symbols_by_id={
            "ENSG00000000001": "ALPHA",
            "ENSG00000000002": "BETA",
            "ENSG00000000003": None,
            "ENSG00000000004": "DUAL",
            "ENSG00000000005": "dual",
        },
    )


@pytest.fixture(scope="module")
def actual_registry():
    """Verify the same pinned source index that the S2 builder uses."""

    contract = json.loads((REPOSITORY_ROOT / "runtime/world_model/s2_v1.json").read_text())
    pin = contract["registry_index"]
    if not (REPOSITORY_ROOT / pin["path"]).is_file():
        pytest.skip("the pinned registry source index is not mounted")
    return load_registry_context(pin)


def test_actual_supplied_context_matches_the_approved_s2_2_4_identities(actual_registry):
    genes = ("ENSG00000005339", "ENSG00000004897")
    assert get_symbols(genes, actual_registry) == ("CREBBP", "CDC27")
    observations = identity_observations(genes, "s22_reference", actual_registry)
    validate_identity_observations(observations, actual_registry)
    assert actual_registry.release == "116"
    assert len(actual_registry.symbols_by_id) == 78941
    assert set(actual_registry.symbols_by_id) != set(genes)
    assert not hasattr(actual_registry, "multiplex_gene_ids")
    assert observations[0] == {
        "operation": "lookup_human_gene_symbol",
        "arguments": {"gene_symbol": "CREBBP"},
        "status": "resolved",
        "result": {"gene_id": genes[0], "gene_symbol": "CREBBP"},
        "provenance": {
            "registry": [{
                "registry": "ensembl", "release": "116",
                "evidence_id": "ev_s22_reference_registry_1",
            }],
            "graph": [],
        },
    }


def test_null_symbol_is_not_replaced_with_an_invented_name(actual_registry):
    assert actual_registry.symbols_by_id["ENSG00000093100"] is None
    with pytest.raises(ValueError, match="no assigned registry symbol"):
        get_symbols(("ENSG00000093100",), actual_registry)


def test_observations_are_deterministic_and_keep_first_mention_order(registry):
    genes = ("ENSG00000000002", "ENSG00000000001", "ENSG00000000002")
    first = identity_observations(genes, "row_one", registry)
    second = identity_observations(genes, "row_one", registry)
    assert first == second
    assert [row["result"]["gene_id"] for row in first] == list(genes[:2])
    assert len({row["provenance"]["registry"][0]["evidence_id"] for row in first}) == 2
    validate_identity_observations(first, registry)


@pytest.mark.parametrize("gene_id,reason", [
    ("ENSG00000000003", "no assigned registry symbol"),
    ("ENSG00000099999", "no assigned registry symbol"),
    ("ENSG00000000004", "ambiguous"),
    ("ENSG00000000005", "ambiguous"),
    ("ALPHA", "canonical human gene IDs"),
])
def test_only_unique_available_symbol_mappings_can_be_supplied(registry, gene_id, reason):
    with pytest.raises(ValueError, match=reason):
        get_symbols((gene_id,), registry)


def test_casefold_candidates_include_every_identity(registry):
    assert registry.gene_ids_by_symbol["dual"] == ("ENSG00000000004", "ENSG00000000005")
    with pytest.raises(TypeError):
        registry.symbols_by_id["ENSG00000000004"] = "UNIQUE_NOW"


def test_raw_symbol_case_and_whitespace_follow_s1_normalization(registry):
    observations = identity_observations(("ENSG00000000001",), "row", registry)
    observations[0]["arguments"]["gene_symbol"] = "  aLpHa  "
    validate_identity_observations(observations, registry)
    assert observations[0]["arguments"]["gene_symbol"] == "  aLpHa  "
    assert observations[0]["result"]["gene_symbol"] == "ALPHA"


@pytest.mark.parametrize("change", [
    "wrong_gene", "wrong_symbol", "ambiguous_input", "unknown_input",
    "wrong_release", "graph_evidence", "extra_result", "wrong_operation",
])
def test_invented_or_incompatible_identity_evidence_is_rejected(registry, change):
    observations = identity_observations(("ENSG00000000001",), "row", registry)
    row = observations[0]
    if change == "wrong_gene":
        row["result"]["gene_id"] = "ENSG00000000002"
    elif change == "wrong_symbol":
        row["result"]["gene_symbol"] = "alpha"
    elif change == "ambiguous_input":
        row["arguments"]["gene_symbol"] = "DUAL"
    elif change == "unknown_input":
        row["arguments"]["gene_symbol"] = "NOT_A_REGISTERED_SYMBOL"
    elif change == "wrong_release":
        row["provenance"]["registry"][0]["release"] = "115"
    elif change == "graph_evidence":
        row["provenance"]["graph"] = [{"present": True}]
    elif change == "extra_result":
        row["result"]["present_in_multiplex"] = True
    else:
        row["operation"] = "registry_entries"
    with pytest.raises(ValueError):
        validate_identity_observations(observations, registry)


def test_evidence_id_can_be_reused_only_for_the_same_acquisition(registry):
    records = identity_observations(("ENSG00000000001", "ENSG00000000002"), "row", registry)
    validate_identity_observations([records[0], deepcopy(records[0])], registry)
    records[1]["provenance"]["registry"][0]["evidence_id"] = records[0]["provenance"]["registry"][0]["evidence_id"]
    with pytest.raises(ValueError, match="different acquisitions"):
        validate_identity_observations(records, registry)


def test_changed_raw_arguments_cannot_share_one_acquisition_id(registry):
    original = identity_observations(("ENSG00000000001",), "row", registry)[0]
    changed = deepcopy(original)
    changed["arguments"]["gene_symbol"] = "alpha"
    with pytest.raises(ValueError, match="different acquisitions"):
        validate_identity_observations([original, changed], registry)


def test_source_pin_tampering_is_rejected(tmp_path):
    path = tmp_path / "source.json.gz"
    path.write_bytes(b"not the pinned index")
    with pytest.raises(ValueError, match="compressed-file pin"):
        load_registry_context({"path": str(path), "sha256": "0" * 64, "logical_sha256": "0" * 64})


def test_logical_pin_is_checked_even_if_compressed_file_matches(actual_registry):
    contract = json.loads((REPOSITORY_ROOT / "runtime/world_model/s2_v1.json").read_text())
    pin = {**contract["registry_index"], "logical_sha256": "0" * 64}
    with pytest.raises(ValueError, match="logical pin"):
        load_registry_context(pin)
