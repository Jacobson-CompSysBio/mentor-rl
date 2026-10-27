"""Canonical source-index-backed identifier tools for S0 and S1.

This module owns the v3 model-visible tool wire.  The older S0 runtime remains
frozen for reproducing legacy corpora; new builders should depend only on this
runtime and the content-addressed source index.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import string
from threading import Lock
from typing import Any

from .contract import composite_provenance
from .source_index import check_source_index, content_sha256
from .sources import GENE_ID_PATTERN


SYMBOL_TOOL = "lookup_human_gene_symbol"
GENE_ID_TOOL = "lookup_human_ensembl_gene_id"
MEMBERSHIP_TOOL = "check_multiplex_membership"
TOOL_NAMES = (SYMBOL_TOOL, GENE_ID_TOOL, MEMBERSHIP_TOOL)
_GENE_ID_WITH_SUFFIX = re.compile(r"^(ENSG[0-9]{11})(.*)$", re.IGNORECASE)
_NUMERIC_VERSION = re.compile(r"^\.[0-9]+$")
_EVIDENCE_ALPHABET = string.ascii_lowercase + string.digits


class ToolCallError(ValueError):
    """Report an invalid tool invocation rather than a biological outcome."""


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """One model-visible result and its private audit record."""

    output: dict[str, Any]
    evidence: dict[str, Any] | None


EvidenceIdFactory = Callable[[str], str]


def _random_evidence_id(channel: str) -> str:
    prefix = "ev_reg_" if channel == "registry" else "ev_mem_"
    suffix = "".join(secrets.choice(_EVIDENCE_ALPHABET) for _ in range(6))
    return prefix + suffix


def _require_arguments(
    arguments: Mapping[str, Any], expected_keys: tuple[str, ...]
) -> None:
    if not isinstance(arguments, Mapping):
        raise ToolCallError("tool arguments must be one object")
    if set(arguments) != set(expected_keys):
        raise ToolCallError(
            f"tool arguments must have exactly the keys {list(expected_keys)!r}"
        )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ToolCallError(f"{label} must be a string")
    trimmed = value.strip()
    if not trimmed:
        raise ToolCallError(f"{label} must not be empty")
    return trimmed


class WorldModelToolRuntime:
    """Serve the three canonical S0/S1 tools from one verified source index."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        contract: Mapping[str, Any],
        evidence_id_factory: EvidenceIdFactory,
    ) -> None:
        coordinate = payload["coordinate_system"]
        self.ensembl_release = str(coordinate["ensembl_release"])
        self.identifier_registry_id = str(coordinate["identifier_registry_id"])
        self.multiplex_id = str(coordinate["multiplex_id"])
        self.graph_version = str(
            contract["coordinate_system"]["source_flist_sha256"]
        )
        self.source_index_sha256 = content_sha256(payload)
        self._contract = contract
        self._evidence_id_factory = evidence_id_factory
        self._evidence_lock = Lock()
        self._used_evidence_ids: set[str] = set()

        self._genes_by_id = {
            str(gene_id): (values[0], str(values[1]), str(values[2]))
            for gene_id, values in payload["genes_by_id"].items()
        }
        symbols: dict[str, list[str]] = defaultdict(list)
        for gene_id, (symbol, _, _) in self._genes_by_id.items():
            if symbol is not None:
                symbols[str(symbol).casefold()].append(gene_id)
        self._gene_ids_by_symbol = {
            key: tuple(sorted(gene_ids)) for key, gene_ids in symbols.items()
        }
        self._multiplex_gene_ids = frozenset(payload["multiplex_gene_ids"])
        self._sources = {
            str(name): dict(source)
            for name, source in payload["sources"].items()
        }
        self._membership_rule = str(coordinate["multiplex_gene_rule"])

    @classmethod
    def from_source_index(
        cls,
        path: Path,
        *,
        contract: Mapping[str, Any],
        evidence_id_factory: EvidenceIdFactory | None = None,
    ) -> "WorldModelToolRuntime":
        """Load one verified index once for repeated calls."""

        payload = check_source_index(path, contract)
        return cls(
            payload=payload,
            contract=contract,
            evidence_id_factory=evidence_id_factory or _random_evidence_id,
        )

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        inject_backend_failure: bool = False,
    ) -> ToolCallResult:
        """Validate and execute one canonical tool call."""

        if tool_name == SYMBOL_TOOL:
            _require_arguments(arguments, ("gene_symbol",))
            query = _require_text(arguments["gene_symbol"], "gene_symbol")
            channel = "registry"
        elif tool_name == GENE_ID_TOOL:
            _require_arguments(arguments, ("gene_id",))
            query = _require_text(arguments["gene_id"], "gene_id")
            channel = "registry"
        elif tool_name == MEMBERSHIP_TOOL:
            _require_arguments(arguments, ("multiplex_id", "gene_ids"))
            multiplex_id = _require_text(arguments["multiplex_id"], "multiplex_id")
            raw_gene_ids = arguments["gene_ids"]
            if not isinstance(raw_gene_ids, list) or not raw_gene_ids:
                raise ToolCallError("gene_ids must be a nonempty list")
            gene_ids: list[str] = []
            seen: set[str] = set()
            for value in raw_gene_ids:
                gene_id = _require_text(value, "gene_ids item")
                if GENE_ID_PATTERN.fullmatch(gene_id) is None:
                    raise ToolCallError(
                        "gene_ids items must be canonical human Ensembl gene IDs"
                    )
                if gene_id not in seen:
                    seen.add(gene_id)
                    gene_ids.append(gene_id)
            channel = "graph"
        else:
            raise ToolCallError(f"unknown world-model tool {tool_name!r}")

        if inject_backend_failure:
            message = (
                "registry unreachable"
                if channel == "registry"
                else "graph service unreachable"
            )
            return ToolCallResult(
                output={
                    "status": "error",
                    "provenance": None,
                    "error_code": "BACKEND_UNAVAILABLE",
                    "message": message,
                },
                evidence=None,
            )

        provenance, evidence = self._new_evidence(
            channel,
            tool_name,
            multiplex_id=multiplex_id if tool_name == MEMBERSHIP_TOOL else None,
        )
        if tool_name == SYMBOL_TOOL:
            output = self._lookup_symbol(query, provenance)
        elif tool_name == GENE_ID_TOOL:
            output = self._lookup_gene_id(query, provenance)
        else:
            output = self._check_membership(
                multiplex_id, gene_ids, provenance
            )
        return ToolCallResult(output=output, evidence=evidence)

    def _new_evidence(
        self,
        channel: str,
        tool_name: str,
        *,
        multiplex_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        block_spec = self._contract["provenance"][f"{channel}_block"]
        pattern = re.compile(str(block_spec["evidence_id_pattern"]))
        while True:
            evidence_id = self._evidence_id_factory(channel)
            if (
                not isinstance(evidence_id, str)
                or pattern.fullmatch(evidence_id) is None
            ):
                raise ToolCallError(
                    f"evidence factory returned invalid {channel} ID "
                    f"{evidence_id!r}"
                )
            with self._evidence_lock:
                if evidence_id not in self._used_evidence_ids:
                    self._used_evidence_ids.add(evidence_id)
                    break
            if self._evidence_id_factory is not _random_evidence_id:
                raise ToolCallError(f"evidence ID was reused: {evidence_id}")

        if channel == "registry":
            public_block = {
                "registry": str(block_spec["registry"]),
                "release": self.ensembl_release,
                "evidence_id": evidence_id,
            }
            provenance = composite_provenance(
                self._contract, registry=public_block
            )
            evidence = {
                "evidence_id": evidence_id,
                "tool_name": tool_name,
                "source_index_sha256": self.source_index_sha256,
                "registry_id": self.identifier_registry_id,
                "ensembl_release": self.ensembl_release,
                "source": dict(self._sources["ensembl_gtf"]),
            }
        else:
            if multiplex_id is None:
                raise AssertionError("graph evidence requires a multiplex ID")
            public_block = {
                "multiplex_id": multiplex_id,
                "graph_version": self.graph_version,
                "layer_scope": str(block_spec["layer_scope"]),
                "evidence_id": evidence_id,
            }
            provenance = composite_provenance(self._contract, graph=public_block)
            evidence = {
                "evidence_id": evidence_id,
                "tool_name": tool_name,
                "source_index_sha256": self.source_index_sha256,
                "multiplex_id": multiplex_id,
                "graph_version": self.graph_version,
                "membership_rule": self._membership_rule,
                "source": dict(self._sources["mentor_ev_dendrogram"]),
            }
        return provenance, evidence

    def _lookup_symbol(
        self, query: str, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        candidate_ids = self._gene_ids_by_symbol.get(query.casefold(), ())
        if not candidate_ids:
            return {
                "status": "not_found",
                "provenance": provenance,
                "gene_symbol": query,
                "candidate_gene_ids": [],
            }
        spellings = {
            str(self._genes_by_id[gene_id][0]) for gene_id in candidate_ids
        }
        symbol = next(iter(spellings)) if len(spellings) == 1 else query
        if len(candidate_ids) == 1:
            return {
                "status": "resolved",
                "provenance": provenance,
                "gene_id": candidate_ids[0],
                "gene_symbol": symbol,
            }
        return {
            "status": "ambiguous",
            "provenance": provenance,
            "gene_symbol": symbol,
            "candidate_gene_ids": list(candidate_ids),
            "candidate_details": {
                gene_id: {
                    "chromosome": self._genes_by_id[gene_id][1],
                    "biotype": self._genes_by_id[gene_id][2],
                }
                for gene_id in candidate_ids
            },
        }

    def _lookup_gene_id(
        self, query: str, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        match = _GENE_ID_WITH_SUFFIX.fullmatch(query)
        if match is None or (match[2] and match[2][0].isdigit()):
            return {
                "status": "invalid_input",
                "provenance": provenance,
                "input": query,
                "expected": "human_gene_id",
            }

        base, suffix = match.groups()
        canonical_id = base.upper()
        if suffix and _NUMERIC_VERSION.fullmatch(suffix) is None:
            return {
                "status": "not_found",
                "provenance": provenance,
                "gene_id": query,
                "gene_symbol": None,
            }
        gene = self._genes_by_id.get(canonical_id)
        if gene is None:
            return {
                "status": "not_found",
                "provenance": provenance,
                "gene_id": canonical_id if suffix else query,
                "gene_symbol": None,
            }
        return {
            "status": "resolved",
            "provenance": provenance,
            "gene_id": canonical_id,
            "gene_symbol": gene[0],
        }

    def _check_membership(
        self,
        multiplex_id: str,
        gene_ids: list[str],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        if multiplex_id != self.multiplex_id:
            return {
                "status": "not_found",
                "provenance": provenance,
                "multiplex_id": multiplex_id,
            }
        return {
            "status": "resolved",
            "provenance": provenance,
            "multiplex_id": multiplex_id,
            "present_gene_ids": [
                gene_id
                for gene_id in gene_ids
                if gene_id in self._multiplex_gene_ids
            ],
            "absent_gene_ids": [
                gene_id
                for gene_id in gene_ids
                if gene_id not in self._multiplex_gene_ids
            ],
        }
