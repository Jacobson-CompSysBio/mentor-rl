# accepts clean biological arguments, refuses dangerous/low-level ones

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_ALLOWED_REDUCTION_METHODS = {"geometric", "arithmetic", "sum", "none"}
_ALLOWED_DISTANCE_METRICS = {"spearman", "pearson", "dot"}
_ALLOWED_ABLATION_DISTANCE_METRICS = {"spearman", "pearson", "cos"}
_ALLOWED_PERTURBATION_DISTANCE_METRICS = {"spearman", "pearson", "dot", "cos"}
_ALLOWED_SHORTEST_PATH_MERGE_METHODS = {"max", "min", "all", "sum", "mean"}
_ALLOWED_LAYER_STATS_SORT_KEYS = {"layer", "node_count", "edge_count"}
_FORBIDDEN_LOW_LEVEL_ARGUMENTS = {
    "seed_file",
    "query_file",
    "sources_file",
    "targets_file",
    "output_file",
    "tmp",
    "scratch",
    "args",
    "cli_args",
    "app",
    "flist",
    "output_dir",
}

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


def _normalize_layer(layer: str) -> str:
    if not isinstance(layer, str):
        raise ValueError(f"Layer names must be strings, got {type(layer).__name__}")
    layer = layer.strip()
    if not layer:
        raise ValueError("Layer names cannot be empty")
    if "/" in layer or "\\" in layer:
        raise ValueError(f"Layer name looks like a path and is not allowed: {layer}")
    return layer


def _normalize_layer_list(layers: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if layers is None:
        return ()
    if not isinstance(layers, (list, tuple)):
        raise TypeError("layers must be a list or tuple of strings")
    return tuple(sorted({_normalize_layer(layer) for layer in layers}))


def _normalize_layer_selection(args: dict[str, Any]) -> tuple[str, ...]:
    """Normalize rwr layer selection from `layers` or single-layer `layer`."""

    layers: list[str] = []
    raw_layers = args.get("layers")
    if raw_layers is not None:
        if not isinstance(raw_layers, (list, tuple)):
            raise TypeError("layers must be a list or tuple of strings")
        layers.extend(raw_layers)
    if "layer" in args:
        layers.append(args["layer"])
    return _normalize_layer_list(layers)


def _normalize_gene_arg(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in args:
            return _normalize_gene(args[key])
    raise ValueError(f"Missing required gene argument; expected one of {list(keys)}")


def _reject_low_level_arguments(args: dict[str, Any], *, tool_name: str) -> None:
    bad = sorted(_FORBIDDEN_LOW_LEVEL_ARGUMENTS.intersection(args))
    if bad:
        raise ValueError(f"{tool_name} does not accept file/path/CLI arguments: {bad}")


def _positive_int_or_none(args: dict[str, Any], key: str, default: int | None) -> int | None:
    value = args.get(key, default)
    if value is not None and (not isinstance(value, int) or value <= 0):
        raise ValueError(f"{key} must be a positive integer or null")
    return value


def _float_in_range(args: dict[str, Any], key: str, default: float) -> float:
    value = float(args.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _positive_float(args: dict[str, Any], key: str, default: float) -> float:
    value = float(args.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _boolean_arg(args: dict[str, Any], key: str, default: bool) -> bool:
    value = args.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _uint_arg(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key, default)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _metric_arg(
    args: dict[str, Any],
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    value = str(args.get(key, default)).lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}")
    return value


def _rwr_parameters_from_args(args: dict[str, Any]) -> tuple[float, float, str, float]:
    restart = _float_in_range(args, "restart", 0.7)
    delta = _float_in_range(args, "delta", 0.5)

    reduction_method = str(args.get("reduction_method", "geometric")).lower()
    if reduction_method not in _ALLOWED_REDUCTION_METHODS:
        raise ValueError(
            f"reduction_method must be one of {sorted(_ALLOWED_REDUCTION_METHODS)}"
        )

    threshold = _positive_float(args, "threshold", 1e-10)
    return restart, delta, reduction_method, threshold

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
        _reject_low_level_arguments(args, tool_name="rwr_loe")
        
        seed_genes = _normalize_gene_list(args.get("seed_genes", []), required=True)
        query_genes = _normalize_gene_list(args.get("query_genes", []), required=False)

        top_k = _positive_int_or_none(args, "top_k", 20)
        restart = _float_in_range(args, "restart", 0.7)
        delta = _float_in_range(args, "delta", 0.5)
        
        reduction_method = str(args.get("reduction_method", "geometric")).lower()
        if reduction_method not in _ALLOWED_REDUCTION_METHODS:
            raise ValueError(
                f"reduction_method must be one of {sorted(_ALLOWED_REDUCTION_METHODS)}"
            )
        
        threshold = _positive_float(args, "threshold", 1e-10)
        exclude_seed_genes = _boolean_arg(args, "exclude_seed_genes", True)

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


@dataclass(frozen=True)
class RwrRequest:
    seed_genes: tuple[str, ...]
    layers: tuple[str, ...] = ()
    top_k: int | None = 20
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrRequest":
        if not isinstance(args, dict):
            raise TypeError("rwr arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="rwr")

        seed_genes = _normalize_gene_list(
            args.get("seed_genes", args.get("seeds", [])),
            required=True,
        )
        layers = _normalize_layer_selection(args)
        top_k = _positive_int_or_none(args, "top_k", 20)
        restart = _float_in_range(args, "restart", 0.7)
        delta = _float_in_range(args, "delta", 0.5)

        reduction_method = str(args.get("reduction_method", "geometric")).lower()
        if reduction_method not in _ALLOWED_REDUCTION_METHODS:
            raise ValueError(
                f"reduction_method must be one of {sorted(_ALLOWED_REDUCTION_METHODS)}"
            )

        threshold = _positive_float(args, "threshold", 1e-10)

        return cls(
            seed_genes=seed_genes,
            layers=layers,
            top_k=top_k,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "layers": list(self.layers),
            "top_k": self.top_k,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class RwrRankRequest:
    source_gene: str
    target_gene: str
    layers: tuple[str, ...] = ()
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrRankRequest":
        if not isinstance(args, dict):
            raise TypeError("get_rank arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_rank")

        source_gene = _normalize_gene_arg(args, "source_gene", "source", "gene_a")
        target_gene = _normalize_gene_arg(args, "target_gene", "target", "gene_b")
        layers = _normalize_layer_selection(args)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            source_gene=source_gene,
            target_gene=target_gene,
            layers=layers,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_gene": self.source_gene,
            "target_gene": self.target_gene,
            "layers": list(self.layers),
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class RwrDistanceRequest:
    gene_a: str
    gene_b: str
    layers: tuple[str, ...] = ()
    distance_metric: str = "spearman"
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrDistanceRequest":
        if not isinstance(args, dict):
            raise TypeError("get_distance arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_distance")

        gene_a = _normalize_gene_arg(args, "gene_a", "source_gene", "source")
        gene_b = _normalize_gene_arg(args, "gene_b", "target_gene", "target")
        layers = _normalize_layer_selection(args)
        distance_metric = str(args.get("distance_metric", "spearman")).lower()
        if distance_metric not in _ALLOWED_DISTANCE_METRICS:
            raise ValueError(
                f"distance_metric must be one of {sorted(_ALLOWED_DISTANCE_METRICS)}"
            )
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            gene_a=gene_a,
            gene_b=gene_b,
            layers=layers,
            distance_metric=distance_metric,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "layers": list(self.layers),
            "distance_metric": self.distance_metric,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class RwrSpearmanRequest:
    gene_a: str
    gene_b: str
    layers: tuple[str, ...] = ()
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrSpearmanRequest":
        if not isinstance(args, dict):
            raise TypeError("get_spearman arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_spearman")

        gene_a = _normalize_gene_arg(args, "gene_a", "source_gene", "source")
        gene_b = _normalize_gene_arg(args, "gene_b", "target_gene", "target")
        layers = _normalize_layer_selection(args)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            gene_a=gene_a,
            gene_b=gene_b,
            layers=layers,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "layers": list(self.layers),
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()

    def to_distance_request(self) -> RwrDistanceRequest:
        return RwrDistanceRequest(
            gene_a=self.gene_a,
            gene_b=self.gene_b,
            layers=self.layers,
            distance_metric="spearman",
            restart=self.restart,
            delta=self.delta,
            reduction_method=self.reduction_method,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class RwrPearsonRequest:
    gene_a: str
    gene_b: str
    layers: tuple[str, ...] = ()
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrPearsonRequest":
        if not isinstance(args, dict):
            raise TypeError("get_pearson arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_pearson")

        gene_a = _normalize_gene_arg(args, "gene_a", "source_gene", "source")
        gene_b = _normalize_gene_arg(args, "gene_b", "target_gene", "target")
        layers = _normalize_layer_selection(args)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            gene_a=gene_a,
            gene_b=gene_b,
            layers=layers,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "layers": list(self.layers),
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()

    def to_distance_request(self) -> RwrDistanceRequest:
        return RwrDistanceRequest(
            gene_a=self.gene_a,
            gene_b=self.gene_b,
            layers=self.layers,
            distance_metric="pearson",
            restart=self.restart,
            delta=self.delta,
            reduction_method=self.reduction_method,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class RwrDotSimilarityRequest:
    gene_a: str
    gene_b: str
    layers: tuple[str, ...] = ()
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrDotSimilarityRequest":
        if not isinstance(args, dict):
            raise TypeError("get_dot_similarity arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_dot_similarity")

        gene_a = _normalize_gene_arg(args, "gene_a", "source_gene", "source")
        gene_b = _normalize_gene_arg(args, "gene_b", "target_gene", "target")
        layers = _normalize_layer_selection(args)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            gene_a=gene_a,
            gene_b=gene_b,
            layers=layers,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "layers": list(self.layers),
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()

    def to_distance_request(self) -> RwrDistanceRequest:
        return RwrDistanceRequest(
            gene_a=self.gene_a,
            gene_b=self.gene_b,
            layers=self.layers,
            distance_metric="dot",
            restart=self.restart,
            delta=self.delta,
            reduction_method=self.reduction_method,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class RwrRankVectorSummaryRequest:
    seed_genes: tuple[str, ...]
    layers: tuple[str, ...] = ()
    top_k: int | None = 20
    include_seed_genes: bool = True
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrRankVectorSummaryRequest":
        if not isinstance(args, dict):
            raise TypeError("get_rank_vector_summary arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_rank_vector_summary")

        seed_genes = _normalize_gene_list(args.get("seed_genes", args.get("seeds", [])), required=True)
        layers = _normalize_layer_selection(args)
        top_k = _positive_int_or_none(args, "top_k", 20)
        include_seed_genes = _boolean_arg(args, "include_seed_genes", True)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            seed_genes=seed_genes,
            layers=layers,
            top_k=top_k,
            include_seed_genes=include_seed_genes,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "layers": list(self.layers),
            "top_k": self.top_k,
            "include_seed_genes": self.include_seed_genes,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()

    def to_rwr_request(self) -> RwrRequest:
        return RwrRequest(
            seed_genes=self.seed_genes,
            layers=self.layers,
            top_k=self.top_k,
            restart=self.restart,
            delta=self.delta,
            reduction_method=self.reduction_method,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class RwrEncodingSummaryRequest:
    seed_genes: tuple[str, ...]
    layers: tuple[str, ...] = ()
    top_k: int | None = 20
    include_seed_genes: bool = True
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "RwrEncodingSummaryRequest":
        if not isinstance(args, dict):
            raise TypeError("get_encoding_summary arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_encoding_summary")

        seed_genes = _normalize_gene_list(args.get("seed_genes", args.get("seeds", [])), required=True)
        layers = _normalize_layer_selection(args)
        top_k = _positive_int_or_none(args, "top_k", 20)
        include_seed_genes = _boolean_arg(args, "include_seed_genes", True)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)

        return cls(
            seed_genes=seed_genes,
            layers=layers,
            top_k=top_k,
            include_seed_genes=include_seed_genes,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "layers": list(self.layers),
            "top_k": self.top_k,
            "include_seed_genes": self.include_seed_genes,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class GeneLayersRequest:
    gene: str

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "GeneLayersRequest":
        if not isinstance(args, dict):
            raise TypeError("get_gene_layers arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_gene_layers")
        return cls(gene=_normalize_gene_arg(args, "gene", "gene_id"))

    def to_payload(self) -> dict[str, Any]:
        return {"gene": self.gene}

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class LayerStatsRequest:
    top_k: int | None = 50
    sort_by: str = "edge_count"
    descending: bool = True

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "LayerStatsRequest":
        if not isinstance(args, dict):
            raise TypeError("get_layer_stats arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_layer_stats")

        top_k = _positive_int_or_none(args, "top_k", 50)
        sort_by = str(args.get("sort_by", "edge_count")).lower()
        if sort_by not in _ALLOWED_LAYER_STATS_SORT_KEYS:
            raise ValueError(f"sort_by must be one of {sorted(_ALLOWED_LAYER_STATS_SORT_KEYS)}")
        descending = _boolean_arg(args, "descending", True)
        return cls(top_k=top_k, sort_by=sort_by, descending=descending)

    def to_payload(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "sort_by": self.sort_by, "descending": self.descending}

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class PathLayerCountsRequest:
    source_genes: tuple[str, ...]
    target_genes: tuple[str, ...] = ()
    merge_method: str = "max"
    ignore_weights: bool = False
    max_paths: int | None = 20
    top_k: int | None = 20

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "PathLayerCountsRequest":
        if not isinstance(args, dict):
            raise TypeError("get_path_layer_counts arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_path_layer_counts")

        shortest = ShortestPathsRequest.from_tool_arguments(args)
        top_k = _positive_int_or_none(args, "top_k", 20)
        return cls(
            source_genes=shortest.source_genes,
            target_genes=shortest.target_genes,
            merge_method=shortest.merge_method,
            ignore_weights=shortest.ignore_weights,
            max_paths=shortest.max_paths,
            top_k=top_k,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_genes": list(self.source_genes),
            "target_genes": list(self.target_genes),
            "merge_method": self.merge_method,
            "ignore_weights": self.ignore_weights,
            "max_paths": self.max_paths,
            "top_k": self.top_k,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()

    def to_shortest_paths_request(self) -> ShortestPathsRequest:
        return ShortestPathsRequest(
            source_genes=self.source_genes,
            target_genes=self.target_genes,
            merge_method=self.merge_method,
            ignore_weights=self.ignore_weights,
            max_paths=self.max_paths,
        )


@dataclass(frozen=True)
class ComponentSummaryRequest:
    genes: tuple[str, ...] = ()
    max_components: int | None = 20

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "ComponentSummaryRequest":
        if not isinstance(args, dict):
            raise TypeError("get_component_summary arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_component_summary")

        genes = _normalize_gene_list(args.get("genes", []), required=False)
        max_components = _positive_int_or_none(args, "max_components", 20)
        return cls(genes=genes, max_components=max_components)

    def to_payload(self) -> dict[str, Any]:
        return {"genes": list(self.genes), "max_components": self.max_components}

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class SeedEssentialityRequest:
    seed_genes: tuple[str, ...]
    n_samples_null_dist: int = 100
    seed: int = 42
    top_k: int | None = None
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "SeedEssentialityRequest":
        if not isinstance(args, dict):
            raise TypeError("get_seed_essentiality arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_seed_essentiality")

        seed_genes = _normalize_gene_list(args.get("seed_genes", args.get("seeds", [])), required=True)
        n_samples_null_dist = _uint_arg(args, "n_samples_null_dist", 100)
        seed = _uint_arg(args, "seed", 42)
        top_k = _positive_int_or_none(args, "top_k", None)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)
        return cls(
            seed_genes=seed_genes,
            n_samples_null_dist=n_samples_null_dist,
            seed=seed,
            top_k=top_k,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "n_samples_null_dist": self.n_samples_null_dist,
            "seed": self.seed,
            "top_k": self.top_k,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class LayerAblationRequest:
    seed_genes: tuple[str, ...]
    distance_metric: str = "spearman"
    top_k: int | None = 20
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "LayerAblationRequest":
        if not isinstance(args, dict):
            raise TypeError("get_layer_ablation arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_layer_ablation")

        seed_genes = _normalize_gene_list(args.get("seed_genes", args.get("seeds", [])), required=True)
        distance_metric = _metric_arg(
            args,
            "distance_metric",
            "spearman",
            _ALLOWED_ABLATION_DISTANCE_METRICS,
        )
        top_k = _positive_int_or_none(args, "top_k", 20)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)
        return cls(
            seed_genes=seed_genes,
            distance_metric=distance_metric,
            top_k=top_k,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "distance_metric": self.distance_metric,
            "top_k": self.top_k,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class NodePerturbationRequest:
    seed_genes: tuple[str, ...]
    perturb_genes: tuple[str, ...]
    distance_metric: str = "spearman"
    top_k: int | None = 20
    restart: float = 0.7
    delta: float = 0.5
    reduction_method: str = "geometric"
    threshold: float = 1e-10

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "NodePerturbationRequest":
        if not isinstance(args, dict):
            raise TypeError("get_node_perturbation arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="get_node_perturbation")

        seed_genes = _normalize_gene_list(args.get("seed_genes", args.get("seeds", [])), required=True)
        perturb_genes = _normalize_gene_list(
            args.get("perturb_genes", args.get("genes", [])),
            required=True,
        )
        distance_metric = _metric_arg(
            args,
            "distance_metric",
            "spearman",
            _ALLOWED_PERTURBATION_DISTANCE_METRICS,
        )
        top_k = _positive_int_or_none(args, "top_k", 20)
        restart, delta, reduction_method, threshold = _rwr_parameters_from_args(args)
        return cls(
            seed_genes=seed_genes,
            perturb_genes=perturb_genes,
            distance_metric=distance_metric,
            top_k=top_k,
            restart=restart,
            delta=delta,
            reduction_method=reduction_method,
            threshold=threshold,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed_genes": list(self.seed_genes),
            "perturb_genes": list(self.perturb_genes),
            "distance_metric": self.distance_metric,
            "top_k": self.top_k,
            "restart": self.restart,
            "delta": self.delta,
            "reduction_method": self.reduction_method,
            "threshold": self.threshold,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class ShortestPathsRequest:
    source_genes: tuple[str, ...]
    target_genes: tuple[str, ...] = ()
    merge_method: str = "max"
    ignore_weights: bool = False
    max_paths: int | None = 20

    @classmethod
    def from_tool_arguments(cls, args: dict[str, Any]) -> "ShortestPathsRequest":
        if not isinstance(args, dict):
            raise TypeError("shortest_paths arguments must be a JSON object")
        _reject_low_level_arguments(args, tool_name="shortest_paths")

        source_genes = _normalize_gene_list(
            args.get("source_genes", [args["source"]] if "source" in args else []),
            required=True,
        )
        target_genes = _normalize_gene_list(
            args.get("target_genes", [args["target"]] if "target" in args else []),
            required=False,
        )

        merge_method = str(args.get("merge_method", "max")).lower()
        if merge_method not in _ALLOWED_SHORTEST_PATH_MERGE_METHODS:
            raise ValueError(
                f"merge_method must be one of {sorted(_ALLOWED_SHORTEST_PATH_MERGE_METHODS)}"
            )

        ignore_weights = _boolean_arg(args, "ignore_weights", False)
        max_paths = _positive_int_or_none(args, "max_paths", 20)

        return cls(
            source_genes=source_genes,
            target_genes=target_genes,
            merge_method=merge_method,
            ignore_weights=ignore_weights,
            max_paths=max_paths,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_genes": list(self.source_genes),
            "target_genes": list(self.target_genes),
            "merge_method": self.merge_method,
            "ignore_weights": self.ignore_weights,
            "max_paths": self.max_paths,
        }

    def cache_key_payload(self) -> dict[str, Any]:
        return self.to_payload()
