# accepts clean biological arguments, refuses dangerous/low-level ones

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_ALLOWED_REDUCTION_METHODS = {"geometric", "arithmetic", "sum", "none"}

def _normalize_gene(gene: str) -> str:
    if not isinstance(gene, str):
        raise ValueError(f"Gene IDs must be strings, got {type(gene).__name__}")
    gene = gene.strip()

    if not gene:
        raise ValueError("Gene IDs cannot be empty")
    
    # reject obvious path-like input
    if "/" in gene or "\\" in gene:
        raise ValueError(f"Gene ID looks like a path and is not allowed: {gene}")
    
    return gene.upper()

def _normalize_gene_list(genes: list[str] | tuple[str, ...], *, required: bool) -> tuple[str, ...]:
    if genes is None:
        genes = []
    
    if not isinstance(genes, (list, tuple)):
        raise TypeError("Gene list must be a list or tuple of strings")
    
    normalized = sorted({_normalize_gene(g) for g in genes})

    if required and not normalized:
        raise ValueError("seed_genes must contain at least one gene")
    
    return tuple(normalized)

@dataclass(frozen=True)
class RwrLoeRequest:
    seed_genes: tuple[str, ...]
    query_genes: tuple[str, ...] = ()
    top_k: int | None = 20
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10
    exclude_seed_genes: bool = True

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrLoeRequest":
        if not isinstance(args, dict):
            raise TypeError("rwr_loe arguments must be a JSON object")
        
        forbidden = {
            "seed_file",
            "query_file",
            "output_file",
            "tmp",
            "scratch",
            "args",
            "cli_args",
            "app",
            "flist",
            "output_dir"
        }
        bad = sorted(forbidden.intersection(args))
        if bad:
            raise ValueError(f"rwr_loe does not accept file/path/CLI arguments: {bad}")
        
        seed_genes = _normalize_gene_list(args.get("seed_genes", []), required=True)
        query_genes = _normalize_gene_list(args.get("query_genes", []), required=False)

        top_k = args.get("top_k", 20)
        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer or null")
        
        restart = float(args.get("restart", 0.7))
        if not 0.0 <= restart <= 1.0:
            raise ValueError("restart must be between [0, 1]")
        
        delta = float(args.get("delta", 0.5))
        if not 0.0 <= delta <= 1.0:
            raise ValueError("delta must be in [0, 1]")
        
        reduction_method = str(args.get("reduction_method", "geometric")).lower()
        if reduction_method not in _ALLOWED_REDUCTION_METHODS:
            raise ValueError(
                f"reduction_method must be one of {sorted(_ALLOWED_REDUCTION_METHODS)}"
            )
        
        threshold = float(args.get("threshold", 1e-10))
        if threshold <= 0:
            raise ValueError("threshold must be postive")
        
        exclude_seed_genes = args.get("exclude_seed_genes", True)
        if not isinstance(exclude_seed_genes, bool):
            raise ValueError("exclude_seed_genes must be a boolean")

        return cls(
            seed_genes=seed_genes,
            query_genes=query_genes,
            top_k=top_k,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
            exclude_seed_genes=exclude_seed_genes
        )
    
    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "query_genes": list(self.query_genes),
            "top_k": self.top_k,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
            "exclude_seed_genes": self.exclude_seed_genes,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()
    