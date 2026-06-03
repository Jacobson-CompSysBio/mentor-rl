"""Structured, model-facing backend for RWR-HPC tools.

This module keeps RWR-HPC logically in-memory from the model's perspective:
the model provides biological arguments, while this backend handles temporary
files, app invocation, parsing, caching, and provenance.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

from .rwr_hpc_cache import (
    RwrHpcCache,
    file_sha256,
    make_rwr_loe_cache_key,
)
from .rwr_hpc_requests import RwrLoeRequest
from .tools import ToolExecutionError, ToolExecutionResult


_GENE_COLUMNS = (
    "gene",
    "Gene",
    "GENE",
    "node",
    "Node",
    "NODE",
    "NodeNames",
    "node_name",
    "gene_id",
    "GeneID",
)

_SCORE_COLUMNS = (
    "score",
    "Score",
    "SCORE",
    "rwr_score",
    "RWR_score",
    "RWRScore",
    "RWR",
    "probability",
    "Probability",
)

_RANK_COLUMNS = (
    "rank",
    "Rank",
    "RANK",
    "rerank",
    "meanrank",
    "mean_rank",
)


def _first_existing_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_rwr_loe_ranks(
    ranks_path: str | Path,
    *,
    seed_genes: tuple[str, ...] = (),
    query_genes: tuple[str, ...] = (),
    top_k: int | None = 20,
    exclude_seed_genes: bool = True,
) -> list[dict[str, Any]]:
    """Parse an RWR_LOE ranks TSV into a stable schema.

    The exact RWR-HPC output column names may vary across app versions, so this
    parser accepts several common gene/score/rank column spellings.
    """

    path = Path(ranks_path)
    if not path.exists():
        raise ToolExecutionError(f"RWR_LOE ranks file does not exist: {path}")

    seed_set = {gene.upper() for gene in seed_genes}
    query_set = {gene.upper() for gene in query_genes}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])

        if not fieldnames:
            raise ToolExecutionError(f"RWR_LOE ranks file has no header: {path}")

        gene_col = _first_existing_column(fieldnames, _GENE_COLUMNS)
        score_col = _first_existing_column(fieldnames, _SCORE_COLUMNS)
        rank_col = _first_existing_column(fieldnames, _RANK_COLUMNS)

        if gene_col is None:
            raise ToolExecutionError(
                "Could not find a gene column in RWR_LOE ranks file. "
                f"Available columns: {fieldnames}"
            )

        parsed: list[dict[str, Any]] = []

        for row_index, row in enumerate(reader, start=1):
            raw_gene = row.get(gene_col)
            if raw_gene is None:
                continue

            gene = str(raw_gene).strip()
            if not gene:
                continue

            gene_key = gene.upper()
            is_seed = gene_key in seed_set
            is_query = gene_key in query_set

            if exclude_seed_genes and is_seed:
                continue

            rank = _parse_int(row.get(rank_col)) if rank_col is not None else None
            if rank is None:
                rank = row_index

            score = _parse_float(row.get(score_col)) if score_col is not None else None

            parsed.append(
                {
                    "gene": gene_key,
                    "rank": rank,
                    "score": score,
                    "is_seed": is_seed,
                    "is_query": is_query,
                    "raw": dict(row),
                }
            )

    parsed.sort(key=lambda item: item["rank"])

    if top_k is not None:
        parsed = parsed[:top_k]

    return parsed


def _write_gene_file(path: Path, genes: tuple[str, ...]) -> None:
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")


def _find_rwr_loe_ranks_file(output_dir: Path) -> Path:
    """Find the rank file emitted by RWR_LOE.

    The fake test backend will write one .ranks.tsv file. The real app may use
    a longer prefix, so we search flexibly.
    """

    candidates: list[Path] = []
    candidates.extend(sorted(output_dir.rglob("*.ranks.tsv")))
    candidates.extend(sorted(output_dir.rglob("*ranks*.tsv")))
    candidates.extend(sorted(output_dir.rglob("*.tsv")))

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)

    if not unique_candidates:
        raise ToolExecutionError(f"No RWR_LOE ranks TSV file found under {output_dir}")

    return unique_candidates[0]


class RwrHpcStructuredBackend:
    """Structured backend for model-facing RWR-HPC tools."""

    def __init__(
        self,
        *,
        flist: str | Path,
        app_backend: Any | None,
        cache: RwrHpcCache | None = None,
        scratch_root: str | Path | None = None,
        rwr_hpc_build_id: str = "unknown",
        timeout_seconds: int = 300,
    ) -> None:
        self.flist = Path(flist).resolve()
        if not self.flist.exists():
            raise FileNotFoundError(f"RWR-HPC flist does not exist: {self.flist}")

        self.app_backend = app_backend
        self.cache = cache
        self.scratch_root = Path(scratch_root).resolve() if scratch_root else None
        self.rwr_hpc_build_id = rwr_hpc_build_id
        self.timeout_seconds = timeout_seconds

        if self.scratch_root is not None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)

    def run_rwr_loe(self, request: RwrLoeRequest) -> ToolExecutionResult:
        """Run structured RWR_LOE using cache first, then app fallback."""

        request_payload = request.cache_key_payload()
        network_flist_sha256 = file_sha256(self.flist)

        cache_key = make_rwr_loe_cache_key(
            request_payload=request_payload,
            network_flist_sha256=network_flist_sha256,
            rwr_hpc_build_id=self.rwr_hpc_build_id,
        )

        if self.cache is not None:
            cached = self.cache.get("rwr_loe", cache_key)
            if cached is not None:
                provenance = dict(cached["provenance"])
                provenance["cache_hit"] = True
                provenance["cache_key"] = cache_key

                payload = cached["payload"]
                return ToolExecutionResult(
                    payload=payload,
                    provenance=provenance,
                    is_empty=not bool(payload.get("ranked_genes")),
                )

        if self.app_backend is None:
            raise ToolExecutionError(
                "RWR-HPC structured backend has no app backend and no cached result."
            )

        result = self._run_rwr_loe_app_fallback(
            request=request,
            request_payload=request_payload,
            cache_key=cache_key,
            network_flist_sha256=network_flist_sha256,
        )

        return result

    def _run_rwr_loe_app_fallback(
        self,
        *,
        request: RwrLoeRequest,
        request_payload: dict[str, Any],
        cache_key: str,
        network_flist_sha256: str,
    ) -> ToolExecutionResult:
        """Run RWR_LOE via the standalone app using hidden scratch files."""

        temp_dir_kwargs: dict[str, Any] = {
            "prefix": "rwr_loe_",
        }
        if self.scratch_root is not None:
            temp_dir_kwargs["dir"] = str(self.scratch_root)

        with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp:
            scratch_dir = Path(tmp)
            seed_file = scratch_dir / "seed_genes.txt"
            query_file = scratch_dir / "query_genes.txt"
            output_dir = scratch_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_gene_file(seed_file, request.seed_genes)

            query_file_arg: str | None = None
            if request.query_genes:
                _write_gene_file(query_file, request.query_genes)
                query_file_arg = str(query_file)

            app_args = self._build_rwr_loe_app_args(
                seed_file=str(seed_file),
                query_file=query_file_arg,
                output_dir=str(output_dir),
                request=request,
            )

            app_result = self.app_backend.run_app(
                "rwr_loe",
                app_args,
                timeout_seconds=self.timeout_seconds,
            )

            if app_result.returncode != 0:
                text = app_result.stderr or app_result.stdout or ""
                preview = text[:1000]
                raise ToolExecutionError(
                    f"RWR_LOE app failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )

            ranks_file = _find_rwr_loe_ranks_file(output_dir)

            ranked_genes = parse_rwr_loe_ranks(
                ranks_file,
                seed_genes=request.seed_genes,
                query_genes=request.query_genes,
                top_k=request.top_k,
                exclude_seed_genes=request.exclude_seed_genes,
            )

            payload = {
                "tool_name": "rwr_loe",
                "seed_genes": list(request.seed_genes),
                "query_genes": list(request.query_genes),
                "top_k": request.top_k,
                "restart": request.restart,
                "delta": request.delta,
                "reduction_method": request.reduction_method,
                "threshold": request.threshold,
                "exclude_seed_genes": request.exclude_seed_genes,
                "ranked_genes": ranked_genes,
            }

            provenance = {
                "backend": "rwr_hpc_app",
                "implementation": "structured_app_fallback",
                "cache_hit": False,
                "cache_key": cache_key,
                "network_flist_sha256": network_flist_sha256,
                "rwr_hpc_build_id": self.rwr_hpc_build_id,
                "app_returncode": app_result.returncode,
            }

            if self.cache is not None:
                self.cache.put(
                    "rwr_loe",
                    cache_key,
                    request=request_payload,
                    payload=payload,
                    provenance=provenance,
                    raw_stdout=app_result.stdout,
                    raw_stderr=app_result.stderr,
                )

            return ToolExecutionResult(
                payload=payload,
                provenance=provenance,
                is_empty=not bool(ranked_genes),
            )

    def _build_rwr_loe_app_args(
        self,
        *,
        seed_file: str,
        query_file: str | None,
        output_dir: str,
        request: RwrLoeRequest,
    ) -> list[str]:
        """Build CLI args for the app fallback.

        Before the real Frontier smoke test, compare these flags against
        docs/rwr_hpc_help/rwr_loe.txt and adjust names if needed.
        """

        args = [
            "--flist",
            str(self.flist),
            "--seed_file",
            seed_file,
            "--output_dir",
            output_dir,
            "--restart",
            str(request.restart),
            "--delta",
            str(request.delta),
            "--reduction_method",
            request.reduction_method,
            "--threshold",
            str(request.threshold),
        ]

        if query_file is not None:
            args.extend(["--query_file", query_file])

        return args