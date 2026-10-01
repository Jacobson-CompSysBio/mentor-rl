"""Provide deterministic S0 tools for the pinned Ensembl Registry."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .tools import ToolExecutionError, ToolExecutionResult

# These variables do not change.
REGISTRY_SCHEMA_VERSION_V1 = (
    "mentor-rl-world-model-s0-ensembl-registry-v1"
)
REGISTRY_SCHEMA_VERSION_V2 = (
    "mentor-rl-world-model-s0-ensembl-registry-v2"
)
REGISTRY_SCHEMA_VERSION = REGISTRY_SCHEMA_VERSION_V1
SUPPORTED_REGISTRY_SCHEMA_VERSIONS = frozenset(
    {REGISTRY_SCHEMA_VERSION_V1, REGISTRY_SCHEMA_VERSION_V2}
)
TOOL_CONTRACT_VERSION_V1 = (
    "mentor-rl-world-model-s0-identifier-tools-v1"
)
TOOL_CONTRACT_VERSION_V2 = (
    "mentor-rl-world-model-s0-identifier-tools-v2"
)
TOOL_CONTRACT_VERSION = TOOL_CONTRACT_VERSION_V1

ENSEMBL_RELEASE = "Ensembl_116"
ENSEMBL_PATTERN = re.compile(r"^ENSG[0-9]{11}$")

SYMBOL_TOOL_NAME = "lookup_human_gene_symbol"
ENSEMBL_TOOL_NAME = "lookup_human_ensembl_gene_id"


def identifier_tool_definitions(
    *,
    contract_version: str = TOOL_CONTRACT_VERSION,
) -> list[dict[str, Any]]:
    """Return the S0 function definitions."""

    if contract_version not in {
        TOOL_CONTRACT_VERSION_V1,
        TOOL_CONTRACT_VERSION_V2,
    }:
        raise ValueError("The S0 identifier tool contract version is invalid.")
    ensembl_description = (
        "Look up all gene symbols for one human Ensembl gene ID "
        "in release 116."
        if contract_version == TOOL_CONTRACT_VERSION_V1
        else (
            "Look up the gene symbol for one human Ensembl gene ID "
            "in release 116. The symbol can be null."
        )
    )

    return [
        {
            "type": "function",
            "function": {
                "name": SYMBOL_TOOL_NAME,
                "description": (
                    "Look up one human gene symbol in the pinned Ensembl "
                    "release 116 registry."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_symbol": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "The exact human gene symbol to look up."
                            ),
                        }
                    },
                    "required": ["gene_symbol"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": ENSEMBL_TOOL_NAME,
                "description": ensembl_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_id": {
                            "type": "string",
                            "pattern": ENSEMBL_PATTERN.pattern,
                            "description": (
                                "The exact human Ensembl gene ID to look up."
                            ),
                        }
                    },
                    "required": ["gene_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _require_canonical_symbol(value: Any) -> str:
    """Return one canonical gene symbol."""

    if not isinstance(value, str) or not value:
        raise ToolExecutionError(
            "gene_symbol must be a non-empty string."
        )

    if value != value.strip():
        raise ToolExecutionError(
            "gene_symbol must not contain outer whitespace."
        )

    if unicodedata.normalize("NFKC", value) != value:
        raise ToolExecutionError(
            "gene_symbol must use canonical Unicode text."
        )
    return value


def _require_canonical_gene_id(value: Any) -> str:
    """Return one canonical human Ensembl gene ID."""

    if (
        not isinstance(value, str)
        or ENSEMBL_PATTERN.fullmatch(value) is None
    ):
        raise ToolExecutionError(
            "gene_id must match ENSG plus 11 decimal digits."
        )
    return value

def _require_exact_argument(
    arguments: Any,
    *,
    field_name: str,
) -> Any:
    """Return the only tool argument."""

    if not isinstance(arguments, Mapping):
        raise ToolExecutionError(
            "Tool arguments must be one JSON object."
        )
    if set(arguments) != {field_name}:
        raise ToolExecutionError(
            f"Tool arguments must contain only {field_name!r}."
        )
    return arguments[field_name]


@dataclass(frozen=True)
class S0IdentifierToolRuntime:
    """Resolve identifiers from one verified registry."""

    registry_path: Path
    registry_sha256: str
    registry_schema_version: str
    tool_contract_version: str
    gene_symbols_by_id: Mapping[str, tuple[str, ...]]
    gene_ids_by_symbol: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_registry(
        cls,
        registry_path: str | Path,
        *,
        expected_sha256: str,
    ) -> "S0IdentifierToolRuntime":
        """Load and check one pinned Ensembl registry."""
        path = Path(registry_path).expanduser().resolve()

        try:
            actual_sha256 = _sha256_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ToolExecutionError(
                f"The identifier registry cannot be read: {path}"
            ) from error

        if actual_sha256 != expected_sha256:
            raise ToolExecutionError(
                "The identifier registry SHA-256 value changed: "
                f"{actual_sha256}"
            )

        if not isinstance(payload, Mapping):
            raise ToolExecutionError(
                "The identifier registry must be one JSON object."
            )

        registry_schema_version = payload.get("schema_version")
        if registry_schema_version not in SUPPORTED_REGISTRY_SCHEMA_VERSIONS:
            raise ToolExecutionError(
                "The identifier registry schema version changed."
            )
        tool_contract_version = (
            TOOL_CONTRACT_VERSION_V2
            if registry_schema_version == REGISTRY_SCHEMA_VERSION_V2
            else TOOL_CONTRACT_VERSION_V1
        )

        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise ToolExecutionError(
                "The identifier registry source must be one object."
            )

        if source.get("release") != 116:
            raise ToolExecutionError(
                "The identifier registry must use Ensembl release 116."
            )

        raw_symbols_by_id = payload.get("gene_symbols_by_id")
        if (
            not isinstance(raw_symbols_by_id, Mapping)
            or not raw_symbols_by_id
        ):
            raise ToolExecutionError(
                "The identifier registry contains no gene mappings."
            )

        symbols_by_id: dict[str, tuple[str, ...]] = {}
        ids_by_symbol_sets: dict[str, set[str]] = {}

        for raw_gene_id, raw_symbols in raw_symbols_by_id.items():
            gene_id = _require_canonical_gene_id(raw_gene_id)

            if not isinstance(raw_symbols, list):
                raise ToolExecutionError(
                    f"The registry symbols for {gene_id} must be a list."
                )
            if (
                registry_schema_version == REGISTRY_SCHEMA_VERSION_V1
                and not raw_symbols
            ):
                raise ToolExecutionError(
                    f"The legacy registry has no symbols for {gene_id}."
                )

            symbols = tuple(
                _require_canonical_symbol(symbol)
                for symbol in raw_symbols
            )

            if list(symbols) != sorted(set(symbols)):
                raise ToolExecutionError(
                    f"The symbols for {gene_id} are not canonical."
                )
            if (
                registry_schema_version == REGISTRY_SCHEMA_VERSION_V2
                and len(symbols) > 1
            ):
                raise ToolExecutionError(
                    f"The shared registry has multiple symbols for {gene_id}."
                )

            symbols_by_id[gene_id] = symbols

            for symbol in symbols:
                ids_by_symbol_sets.setdefault(symbol, set()).add(gene_id)

        ids_by_symbol = {
            symbol: tuple(sorted(gene_ids))
            for symbol, gene_ids in ids_by_symbol_sets.items()
        }

        return cls(
            registry_path=path,
            registry_sha256=actual_sha256,
            registry_schema_version=str(registry_schema_version),
            tool_contract_version=tool_contract_version,
            gene_symbols_by_id=MappingProxyType(symbols_by_id),
            gene_ids_by_symbol=MappingProxyType(ids_by_symbol),
        )

    def _provenance(self, tool_name: str) -> dict[str, Any]:
        """Return the common provenance for one tool result."""

        return {
            "tool_name": tool_name,
            "tool_contract_version": self.tool_contract_version,
            "source": "pinned_registry",
            "registry_schema_version": self.registry_schema_version,
            "registry_sha256": self.registry_sha256,
            "ensembl_release": ENSEMBL_RELEASE,
            "network_used": False,
        }

    def lookup_human_gene_symbol(
        self,
        arguments: Any,
    ) -> ToolExecutionResult:
        """Look up one gene symbol in the pinned registry."""

        raw_symbol = _require_exact_argument(
            arguments,
            field_name="gene_symbol",
        )
        gene_symbol = _require_canonical_symbol(raw_symbol)

        gene_ids = self.gene_ids_by_symbol.get(gene_symbol, ())

        if not gene_ids:
            payload = {
                "status": "not_found",
                "gene_symbol": gene_symbol,
                "candidate_gene_ids": [],
            }
        elif len(gene_ids) == 1:
            payload = {
                "status": "resolved",
                "gene_id": gene_ids[0],
                "gene_symbol": gene_symbol,
            }
        else:
            payload = {
                "status": "ambiguous",
                "gene_symbol": gene_symbol,
                "candidate_gene_ids": list(gene_ids),
            }
            if self.tool_contract_version == TOOL_CONTRACT_VERSION_V1:
                payload["action"] = "defer"

        return ToolExecutionResult(
            payload=payload,
            provenance=self._provenance(SYMBOL_TOOL_NAME),
            is_empty=not gene_ids,
        )

    def lookup_human_ensembl_gene_id(
        self,
        arguments: Any,
    ) -> ToolExecutionResult:
        """Look up one Ensembl gene ID in the pinned registry."""

        raw_gene_id = _require_exact_argument(
            arguments,
            field_name="gene_id",
        )
        gene_id = _require_canonical_gene_id(raw_gene_id)
        gene_is_known = gene_id in self.gene_symbols_by_id
        gene_symbols = self.gene_symbols_by_id.get(gene_id, ())

        if self.tool_contract_version == TOOL_CONTRACT_VERSION_V2:
            payload = {
                "status": "resolved" if gene_is_known else "not_found",
                "gene_id": gene_id,
                "gene_symbol": gene_symbols[0] if gene_symbols else None,
            }
        elif not gene_symbols:
            payload = {
                "status": "not_found",
                "gene_id": gene_id,
                "gene_symbols": [],
            }
        else:
            payload = {
                "status": "resolved",
                "gene_id": gene_id,
                "gene_symbols": list(gene_symbols),
            }

        return ToolExecutionResult(
            payload=payload,
            provenance=self._provenance(ENSEMBL_TOOL_NAME),
            is_empty=not gene_is_known,
        )

    def execute(
        self,
        tool_name: str,
        arguments: Any,
    ) -> ToolExecutionResult:
        """Run one S0 identifier tool."""

        if tool_name == SYMBOL_TOOL_NAME:
            return self.lookup_human_gene_symbol(arguments)

        if tool_name == ENSEMBL_TOOL_NAME:
            return self.lookup_human_ensembl_gene_id(arguments)

        raise ToolExecutionError(
            f"Unknown S0 identifier tool: {tool_name!r}."
        )
