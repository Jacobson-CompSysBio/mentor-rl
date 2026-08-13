from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model_s0_tokenizer import (
    ATOMIC_REGISTRY_SCHEMA,
    FULLY_ATOMIC_METHOD,
    S0_CODEC_SCHEMA_VERSION,
    FullyAtomicRegistry,
    S0TokenizerCodec,
)


ENSEMBL_VALUE = "ENSG00000000001"
ENSEMBL_MARKER = "<|s0atom_s0ens_00000|>"
SYMBOL_VALUE = "TP53"
SYMBOL_MARKER = "<|s0atom_s0sym_00000|>"


def _registry_payload() -> dict:
    """Return one small valid registry."""

    return {
        "schema_version": ATOMIC_REGISTRY_SCHEMA,
        "method": FULLY_ATOMIC_METHOD,
        "namespaces": {
            "ensembl_human_gene": {
                "entries": [
                    {
                        "value": ENSEMBL_VALUE,
                        "marker": ENSEMBL_MARKER,
                    }
                ]
            },
            "human_gene_symbol": {
                "entries": [
                    {
                        "value": SYMBOL_VALUE,
                        "marker": SYMBOL_MARKER,
                    }
                ]
            },
        },
    }


def _load_registry(tmp_path: Path, payload: object) -> FullyAtomicRegistry:
    """Write and load one registry payload."""

    path = tmp_path / "fully_atomic_registry.json"
    identified = (
        _manifest_identity(dict(payload))
        if isinstance(payload, dict)
        else payload
    )
    path.write_text(json.dumps(identified), encoding="utf-8")
    return FullyAtomicRegistry(path)


def _manifest_identity(payload: dict) -> dict:
    """Add the identity that the codec manifest requires."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _atomic_codec_payload() -> dict:
    """Return one valid fully atomic codec manifest."""

    return {
        "schema_version": S0_CODEC_SCHEMA_VERSION,
        "method": FULLY_ATOMIC_METHOD,
        "fully_atomic_registry": {
            "manifest_file": "fully_atomic_registry.json"
        },
    }


def _load_atomic_codec(tmp_path: Path, payload: dict) -> S0TokenizerCodec:
    """Write the registry and load one atomic codec."""

    registry_path = tmp_path / "fully_atomic_registry.json"
    registry = _manifest_identity(_registry_payload())
    registry_text = json.dumps(registry)
    registry_path.write_text(registry_text, encoding="utf-8")
    registry_reference = payload.get("fully_atomic_registry")
    if isinstance(registry_reference, dict):
        registry_reference.update(
            {
                "manifest_sha256": hashlib.sha256(
                    registry_text.encode("utf-8")
                ).hexdigest(),
                "internal_manifest_sha256": registry[
                    "manifest_sha256"
                ],
            }
        )
    codec_path = tmp_path / "s0_tokenizer_codec_manifest.json"
    codec_path.write_text(
        json.dumps(_manifest_identity(payload)),
        encoding="utf-8",
    )
    return S0TokenizerCodec(codec_path)


def test_registry_builds_forward_and_reverse_maps(tmp_path: Path) -> None:
    registry = _load_registry(tmp_path, _registry_payload())

    assert len(registry.manifest_sha256) == 64
    assert registry.value_to_marker == {
        "ensembl_human_gene": {ENSEMBL_VALUE: ENSEMBL_MARKER},
        "human_gene_symbol": {SYMBOL_VALUE: SYMBOL_MARKER},
    }
    assert registry.marker_to_value == {
        ENSEMBL_MARKER: ENSEMBL_VALUE,
        SYMBOL_MARKER: SYMBOL_VALUE,
    }
    assert registry.marker_to_namespace == {
        ENSEMBL_MARKER: "ensembl_human_gene",
        SYMBOL_MARKER: "human_gene_symbol",
    }


def test_registry_encodes_and_decodes_values(tmp_path: Path) -> None:
    registry = _load_registry(tmp_path, _registry_payload())

    assert (
        registry.encode_value("ensembl_human_gene", ENSEMBL_VALUE)
        == ENSEMBL_MARKER
    )
    assert (
        registry.decode_value("human_gene_symbol", SYMBOL_MARKER)
        == SYMBOL_VALUE
    )
    assert registry.round_trip("human_gene_symbol", SYMBOL_VALUE)


def test_registry_decodes_known_markers_in_text(tmp_path: Path) -> None:
    registry = _load_registry(tmp_path, _registry_payload())
    encoded = (
        f'{{"gene_id":"{ENSEMBL_MARKER}",'
        f'"gene_symbol":"{SYMBOL_MARKER}"}}'
    )

    assert registry.decode_text(encoded) == (
        f'{{"gene_id":"{ENSEMBL_VALUE}",'
        f'"gene_symbol":"{SYMBOL_VALUE}"}}'
    )


def test_registry_keeps_unknown_markers_in_text(tmp_path: Path) -> None:
    registry = _load_registry(tmp_path, _registry_payload())
    unknown_marker = "<|s0atom_s0sym_99999|>"

    assert registry.decode_text(unknown_marker) == unknown_marker


def test_registry_rejects_unknown_values_and_namespaces(
    tmp_path: Path,
) -> None:
    registry = _load_registry(tmp_path, _registry_payload())

    with pytest.raises(ValueError, match="has no human_gene_symbol value"):
        registry.encode_value("human_gene_symbol", "BRCA1")
    with pytest.raises(KeyError, match="Unknown atomic namespace"):
        registry.encode_value("protein", "TP53")


def test_registry_rejects_a_marker_from_the_wrong_namespace(
    tmp_path: Path,
) -> None:
    registry = _load_registry(tmp_path, _registry_payload())

    with pytest.raises(ValueError, match="does not match the namespace"):
        registry.decode_value("ensembl_human_gene", SYMBOL_MARKER)


def test_registry_rejects_duplicate_values(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["namespaces"]["human_gene_symbol"]["entries"].append(
        {
            "value": SYMBOL_VALUE,
            "marker": "<|s0atom_s0sym_00001|>",
        }
    )

    with pytest.raises(ValueError, match="repeats this value"):
        _load_registry(tmp_path, payload)


def test_registry_rejects_duplicate_markers(tmp_path: Path) -> None:
    payload = _registry_payload()
    payload["namespaces"]["human_gene_symbol"]["entries"].append(
        {
            "value": "BRCA1",
            "marker": SYMBOL_MARKER,
        }
    )

    with pytest.raises(ValueError, match="repeats this marker"):
        _load_registry(tmp_path, payload)


def test_registry_rejects_a_marker_with_the_wrong_namespace(
    tmp_path: Path,
) -> None:
    payload = _registry_payload()
    payload["namespaces"]["human_gene_symbol"]["entries"][0][
        "marker"
    ] = ENSEMBL_MARKER

    with pytest.raises(ValueError, match="wrong namespace"):
        _load_registry(tmp_path, payload)


@pytest.mark.parametrize(
    "namespace,value,error",
    [
        ("ensembl_human_gene", "ENSG1", "Ensembl value is not canonical"),
        ("human_gene_symbol", " TP53", "gene symbol is not canonical"),
    ],
)
def test_registry_rejects_noncanonical_values(
    tmp_path: Path,
    namespace: str,
    value: str,
    error: str,
) -> None:
    payload = _registry_payload()
    payload["namespaces"][namespace]["entries"][0]["value"] = value

    with pytest.raises(ValueError, match=error):
        _load_registry(tmp_path, payload)


def test_codec_selects_the_fully_atomic_registry(tmp_path: Path) -> None:
    codec = _load_atomic_codec(
        tmp_path,
        _atomic_codec_payload(),
    )

    assert codec.domain_bpe is None
    assert isinstance(codec.atomic_registry, FullyAtomicRegistry)


def test_fully_atomic_codec_rejects_domain_bpe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot use Domain-BPE"):
        _load_atomic_codec(
            tmp_path,
            {
                "schema_version": S0_CODEC_SCHEMA_VERSION,
                "method": FULLY_ATOMIC_METHOD,
                "domain_bpe": {},
                "fully_atomic_registry": {
                    "manifest_file": "fully_atomic_registry.json"
                },
            },
        )


def test_codec_encodes_fully_atomic_messages(tmp_path: Path) -> None:
    codec = _load_atomic_codec(tmp_path, _atomic_codec_payload())
    messages = [
        {"role": "system", "content": "Use the identifier registry."},
        {
            "role": "user",
            "content": f'Find {{"gene_symbol":"{SYMBOL_VALUE}"}}',
        },
        {
            "role": "assistant",
            "content": json.dumps({"gene_id": ENSEMBL_VALUE}),
        },
    ]

    encoded = codec.encode_messages(
        messages,
        question_family="human_symbol_to_ensembl",
    )

    assert encoded == [
        {"role": "system", "content": "Use the identifier registry."},
        {
            "role": "user",
            "content": f'Find {{"gene_symbol":"{SYMBOL_MARKER}"}}',
        },
        {
            "role": "assistant",
            "content": f'{{"gene_id":"{ENSEMBL_MARKER}"}}',
        },
    ]


def test_codec_decodes_a_fully_atomic_answer(tmp_path: Path) -> None:
    codec = _load_atomic_codec(tmp_path, _atomic_codec_payload())
    encoded_answer = json.dumps(
        {
            "gene_id": ENSEMBL_MARKER,
            "gene_symbols": [SYMBOL_MARKER],
        },
        separators=(",", ":"),
        sort_keys=True,
    )

    decoded_answer, report = codec.decode_generated_answer(encoded_answer)

    assert decoded_answer == json.dumps(
        {
            "gene_id": ENSEMBL_VALUE,
            "gene_symbols": [SYMBOL_VALUE],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    assert report["valid"] is True
    assert report["representation_backend"] == "fully_atomic_registry"
    assert report["checked_biological_values"] == 2
    assert report["violations"] == []
    assert report["residual_markers"] == []


def test_codec_rejects_a_wrong_namespace_atomic_answer(
    tmp_path: Path,
) -> None:
    codec = _load_atomic_codec(tmp_path, _atomic_codec_payload())
    encoded_answer = f'{{"gene_id":"{SYMBOL_MARKER}"}}'

    decoded_answer, report = codec.decode_generated_answer(encoded_answer)

    assert decoded_answer == encoded_answer
    assert report["valid"] is False
    assert "$.gene_id:expected_exact_s0_code" in report["violations"]
    assert "$:unmapped_s0_markers" in report["violations"]
    assert report["residual_markers"] == [SYMBOL_MARKER]
