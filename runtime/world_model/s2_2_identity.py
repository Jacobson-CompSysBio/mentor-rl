"""Supply verified unique symbol mappings as S2.2 context, without new lookups.

Only registry identities are taken from the historical S1 source index. Its
multiplex membership belongs to a different graph view and is never reused.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .contract import REPOSITORY_ROOT
from .source_index import check_source_index, content_sha256
from .sources import GENE_ID_PATTERN, file_sha256


@dataclass(frozen=True, slots=True)
class RegistryContext:
    """An immutable identity-only view with complete symbol candidate sets."""

    release: str
    registry_id: str
    symbols_by_id: Mapping[str, str | None]
    source_index_sha256: str = ""
    content_sha256: str = ""
    gene_ids_by_symbol: Mapping[str, tuple[str, ...]] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.release, str) or not self.release.isdigit():
            raise ValueError("the Ensembl release must be an explicit numeric string")
        if not isinstance(self.registry_id, str) or not self.registry_id.startswith(
            f"ensembl_homo_sapiens_{self.release}_"
        ):
            raise ValueError("the source must identify the human Ensembl registry")
        symbols = dict(self.symbols_by_id)
        candidates: dict[str, list[str]] = defaultdict(list)
        for gene_id, symbol in symbols.items():
            if not isinstance(gene_id, str) or not GENE_ID_PATTERN.fullmatch(gene_id):
                raise ValueError("registry context requires canonical human gene IDs")
            if symbol is not None:
                if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
                    raise ValueError("registry symbols must be canonical nonempty strings or null")
                candidates[symbol.casefold()].append(gene_id)
        object.__setattr__(self, "symbols_by_id", MappingProxyType(symbols))
        object.__setattr__(self, "gene_ids_by_symbol", MappingProxyType({
            symbol: tuple(sorted(gene_ids)) for symbol, gene_ids in candidates.items()
        }))


def load_registry_context(pin: Mapping[str, Any]) -> RegistryContext:
    """Verify compressed and logical index pins, then retain only identity facts."""

    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "logical_sha256"}:
        raise ValueError("registry context requires path, sha256, and logical_sha256")
    if not isinstance(pin["path"], str) or not pin["path"]:
        raise ValueError("the registry index path must be nonempty")
    path = Path(pin["path"])
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise ValueError("registry source index does not match its compressed-file pin")
    payload = check_source_index(path)
    logical_digest = content_sha256(payload)
    if logical_digest != pin["logical_sha256"]:
        raise ValueError("registry source index does not match its logical pin")
    coordinate = payload["coordinate_system"]
    return RegistryContext(
        release=coordinate["ensembl_release"],
        registry_id=coordinate["identifier_registry_id"],
        symbols_by_id={gene_id: values[0] for gene_id, values in payload["genes_by_id"].items()},
        source_index_sha256=pin["sha256"], content_sha256=logical_digest,
    )


def get_symbols(gene_ids: Sequence[str], registry: RegistryContext) -> tuple[str, ...]:
    """Return canonical symbols only when their entire candidate set is unique."""

    if isinstance(gene_ids, (str, bytes)) or not isinstance(gene_ids, Sequence):
        raise ValueError("gene_ids must be a sequence of canonical IDs")
    result = []
    for gene_id in gene_ids:
        if not isinstance(gene_id, str) or not GENE_ID_PATTERN.fullmatch(gene_id):
            raise ValueError("symbol context requires canonical human gene IDs")
        symbol = registry.symbols_by_id.get(gene_id)
        if symbol is None:
            raise ValueError(f"no assigned registry symbol is available for {gene_id}")
        if registry.gene_ids_by_symbol[symbol.casefold()] != (gene_id,):
            raise ValueError(f"the registry symbol {symbol!r} is ambiguous")
        result.append(symbol)
    return tuple(result)


def identity_observations(
    gene_ids: Sequence[str], row_id: str, registry: RegistryContext,
) -> list[dict[str, Any]]:
    """Create attributed supplied-context mappings in first-mention order."""

    if not isinstance(row_id, str) or not row_id or row_id != row_id.strip():
        raise ValueError("identity context requires a nonempty row ID")
    get_symbols(gene_ids, registry)
    unique_ids = tuple(dict.fromkeys(gene_ids))
    symbols = get_symbols(unique_ids, registry)
    return [{
        "operation": "lookup_human_gene_symbol",
        "arguments": {"gene_symbol": symbol},
        "status": "resolved",
        "result": {"gene_id": gene_id, "gene_symbol": symbol},
        "provenance": {
            "registry": [{
                "registry": "ensembl", "release": registry.release,
                "evidence_id": f"ev_{row_id}_registry_{index}",
            }],
            "graph": [],
        },
    } for index, (gene_id, symbol) in enumerate(zip(unique_ids, symbols), 1)]


def validate_identity_observations(
    observations: Sequence[Mapping[str, Any]], registry: RegistryContext,
) -> None:
    """Check exact unique-symbol truth, evidence shape, and acquisition identity."""

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ValueError("identity observations must be a sequence")
    acquisitions = {}
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or set(observation) != {"operation", "arguments", "status", "result", "provenance"}
            or observation["operation"] != "lookup_human_gene_symbol"
            or observation["status"] != "resolved"
        ):
            raise ValueError("identity context must be a resolved symbol observation")
        arguments = observation["arguments"]
        if not isinstance(arguments, Mapping) or set(arguments) != {"gene_symbol"}:
            raise ValueError("symbol observation arguments changed")
        raw_symbol = arguments["gene_symbol"]
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError("the raw registry symbol must be a nonempty string")
        candidates = registry.gene_ids_by_symbol.get(raw_symbol.strip().casefold(), ())
        if len(candidates) != 1:
            raise ValueError("the supplied symbol does not have one registry candidate")
        gene_id = candidates[0]
        expected = {"gene_id": gene_id, "gene_symbol": registry.symbols_by_id[gene_id]}
        if observation["result"] != expected:
            raise ValueError("the supplied identity result differs from registry truth")
        provenance = observation["provenance"]
        if (
            not isinstance(provenance, Mapping) or set(provenance) != {"registry", "graph"}
            or provenance["graph"] != []
            or not isinstance(provenance["registry"], list) or len(provenance["registry"]) != 1
        ):
            raise ValueError("identity evidence requires one registry record and an empty graph list")
        evidence = provenance["registry"][0]
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != {"registry", "release", "evidence_id"}
            or evidence["registry"] != "ensembl" or evidence["release"] != registry.release
            or not isinstance(evidence["evidence_id"], str)
            or not evidence["evidence_id"] or evidence["evidence_id"] != evidence["evidence_id"].strip()
        ):
            raise ValueError("identity evidence must match the pinned registry release")
        evidence_id = evidence["evidence_id"]
        if evidence_id in acquisitions and acquisitions[evidence_id] != observation:
            raise ValueError("one identity evidence ID cannot name different acquisitions")
        acquisitions[evidence_id] = observation
