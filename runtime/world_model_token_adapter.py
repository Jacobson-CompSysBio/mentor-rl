"""Train token row deltas for the two S0 tokenizer methods."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from torch.nn import functional as F

from runtime.world_model_s0_tokenizer import (
    load_s0_tokenizer_codec_for_token_manifest,
)
from runtime.world_model_training import (
    S0_TOKENIZER_MANIFEST_SCHEMA,
    sha256_file,
    validated_tokenizer_manifest,
)


TOKEN_ADAPTER_WEIGHTS = "biological_token_adapter.safetensors"
TOKEN_ADAPTER_MANIFEST = "tokenizer_manifest.json"
S0_TOKEN_ADAPTER_METHODS = frozenset(
    {
        "ordinary_domain_bpe",
        "atomic_plus_domain_bpe",
        "fully_atomic_identifiers",
    }
)


# Block 1: Add trainable deltas to the reserved input and output rows.
def _local_replicated_weight(
    weight: torch.Tensor,
    label: str,
) -> torch.Tensor:
    """Return the local tensor for one replicated DTensor."""

    if not isinstance(weight, DTensor):
        return weight
    expected_placements = (Replicate(),)
    if weight.placements != expected_placements:
        raise RuntimeError(
            f"{label} must be replicated across the native TP mesh; "
            f"got placements={weight.placements}"
        )
    return weight.to_local(grad_placements=expected_placements)


def _require_local_activation(value: torch.Tensor, label: str) -> None:
    """Reject one activation that remains a DTensor."""

    if isinstance(value, DTensor):
        raise RuntimeError(
            f"{label} remained a DTensor; the native TP module must return "
            "a local activation before the token adapter"
        )


class TokenRowDeltaEmbedding(nn.Module):
    """Add trainable rows to the frozen input embedding table."""

    def __init__(
        self,
        base_layer: nn.Module,
        token_start: int,
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        if token_start < 0:
            raise ValueError("token_start must be nonnegative")
        if delta.ndim != 2 or delta.shape[0] < 1:
            raise ValueError("The input token delta must be a nonempty matrix")
        self.base_layer = base_layer
        self.token_start = int(token_start)
        self.base_vocab_size = int(base_layer.weight.shape[0])
        self.token_input_delta = delta
        self.final_vocab_size = max(
            self.base_vocab_size,
            self.token_start + int(delta.shape[0]),
        )

    @property
    def token_count(self) -> int:
        return int(self.token_input_delta.shape[0])

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Return one local embedding activation with trainable token rows.
        _require_local_activation(input_ids, "Token adapter input IDs")

        invalid = (input_ids < 0) | (input_ids >= self.final_vocab_size)
        if bool(invalid.any()):
            raise RuntimeError(
                "A token ID is outside the extended vocabulary"
            )

        # Get the frozen embeddings for base token IDs.
        safe_ids = input_ids.clamp(max=self.base_vocab_size - 1)
        base_result = self.base_layer(safe_ids)
        _require_local_activation(base_result, "Base embedding output")

        # Map each trainable token ID to its adapter row.
        offsets = input_ids - self.token_start
        token_mask = (offsets >= 0) & (offsets < self.token_count)
        safe_offsets = offsets.clamp(0, self.token_count - 1)

        # Get one trainable row for each input token position.
        delta = F.embedding(
            safe_offsets,
            _local_replicated_weight(
                self.token_input_delta,
                "Trainable token input rows",
            ),
        )

        # Add deltas only where a frozen model row exists.
        reserved_result = (
            base_result
            + delta * token_mask.unsqueeze(-1).to(delta.dtype)
        )
        appended_mask = token_mask & (
            input_ids >= self.base_vocab_size
        )

        # Use complete trainable rows for appended token IDs.
        return torch.where(
            appended_mask.unsqueeze(-1),
            delta,
            reserved_result,
        )


class TokenRowDeltaOutput(nn.Module):
    """Add trainable rows to the frozen output head."""

    def __init__(
        self,
        base_layer: nn.Module,
        token_start: int,
        vocab_size: int,
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        if token_start < 0:
            raise ValueError("token_start must be nonnegative")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if token_start > vocab_size:
            raise ValueError("token_start exceeds the base vocabulary size")
        if delta.ndim != 2 or delta.shape[0] < 1:
            raise ValueError("The output token delta must be a nonempty matrix")

        self.base_layer = base_layer
        self.token_start = int(token_start)
        self.base_vocab_size = int(vocab_size)
        self.token_output_delta = delta
        self.final_vocab_size = max(
            self.base_vocab_size,
            self.token_start + int(delta.shape[0]),
        )

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_local_activation(
            hidden_states,
            "Token adapter hidden states",
        )
        # Calculate the frozen base logits.
        base_logits = self.base_layer(hidden_states)
        _require_local_activation(base_logits, "Base output logits")
        if base_logits.shape[-1] != self.base_vocab_size:
            raise RuntimeError(
                "The output head does not match the base vocabulary"
            )

        delta_weight = _local_replicated_weight(
            self.token_output_delta,
            "Trainable token output rows",
        )
        if hidden_states.shape[-1] != delta_weight.shape[-1]:
            raise RuntimeError(
                "The output token rows have the wrong hidden width"
            )

        # Calculate logits for all trainable token rows.
        delta_logits = F.linear(hidden_states, delta_weight)

        # Count trainable rows that overlap the base vocabulary.
        token_count = int(delta_logits.shape[-1])
        reserved_count = min(
            token_count,
            self.base_vocab_size - self.token_start,
        )

        # Add deltas to reserved rows and append complete new rows.
        parts = [base_logits[..., : self.token_start]]

        if reserved_count:
            parts.append(
                base_logits[
                    ...,
                    self.token_start : self.token_start + reserved_count,
                ]
                + delta_logits[..., :reserved_count]
            )

        base_tail = base_logits[
            ...,
            self.token_start + reserved_count :,
        ]
        if base_tail.shape[-1]:
            parts.append(base_tail)

        if reserved_count < token_count:
            parts.append(delta_logits[..., reserved_count:])

        result = torch.cat(parts, dim=-1)
        if result.shape[-1] != self.final_vocab_size:
            raise RuntimeError(
                "The extended output vocabulary is incorrect"
            )
        return result


# Block 2: Validate the complete S0 token manifest contract.
def _required_int(value: Any, label: str, *, minimum: int = 0) -> int:
    """Return one integer at or above the required minimum."""

    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer at or above {minimum}")
    return value


def load_token_manifest(path: Path) -> dict[str, Any]:
    """Load one strict S0 manifest for trainable token rows."""

    payload = validated_tokenizer_manifest(path)
    if payload.get("schema_version") != S0_TOKENIZER_MANIFEST_SCHEMA:
        raise ValueError(
            f"The token manifest must use {S0_TOKENIZER_MANIFEST_SCHEMA}"
        )
    method = payload.get("method")
    if method not in S0_TOKEN_ADAPTER_METHODS:
        raise ValueError(
            "The token adapter requires a supported custom tokenizer"
        )

    base_length = _required_int(
        payload.get("base_tokenizer_length"),
        "base_tokenizer_length",
        minimum=1,
    )
    model_vocab_size = _required_int(
        payload.get("model_vocab_size"),
        "model_vocab_size",
        minimum=1,
    )
    final_length = _required_int(
        payload.get("final_tokenizer_length"),
        "final_tokenizer_length",
        minimum=1,
    )
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("The token adapter manifest has no token rows")
    if final_length != base_length + len(tokens):
        raise ValueError(
            "The final tokenizer length does not match the added token rows"
        )

    reserved_capacity = max(
        model_vocab_size - base_length,
        0,
    )
    reserved_rows_used = min(
        len(tokens),
        reserved_capacity,
    )
    appended_rows = max(
        final_length - model_vocab_size,
        0,
    )

    if payload.get(
        "unused_model_rows_consumed"
    ) != reserved_rows_used:
        raise ValueError(
            "The consumed model row count is incorrect"
        )
    if payload.get(
        "unused_model_rows_remaining"
    ) != max(model_vocab_size - final_length, 0):
        raise ValueError(
            "The remaining model row count is incorrect"
        )
    if payload.get("appended_model_rows", 0) != appended_rows:
        raise ValueError(
            "The appended model row count is incorrect"
        )

    ordered_tokens = sorted(tokens, key=lambda item: int(item["token_id"]))
    expected_ids = list(range(base_length, final_length))
    token_ids = [
        _required_int(item.get("token_id"), "tokens.token_id")
        for item in ordered_tokens
        if isinstance(item, Mapping)
    ]
    if len(token_ids) != len(tokens) or token_ids != expected_ids:
        raise ValueError(
            "The trainable token IDs must start after the base tokenizer rows"
        )

    surfaces: set[str] = set()
    for index, item in enumerate(ordered_tokens):
        if not isinstance(item, Mapping):
            raise ValueError(f"tokens[{index}] must be one object")
        surface = item.get("content")
        if not isinstance(surface, str) or not surface:
            raise ValueError(f"tokens[{index}].content must be nonempty")
        if surface in surfaces:
            raise ValueError(f"The token surface occurs more than once: {surface}")
        surfaces.add(surface)
        source_ids = item.get("base_token_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"tokens[{index}].base_token_ids must be nonempty")
        for source_id in source_ids:
            source_id = _required_int(
                source_id,
                f"tokens[{index}].base_token_ids",
            )
            if source_id >= base_length:
                raise ValueError(
                    f"tokens[{index}] references a non-base source token ID"
                )

    trainable_rows = payload.get("trainable_rows")
    if trainable_rows != {
        "input_embeddings": True,
        "output_head": True,
        "initialization": "mean_of_base_subtoken_rows",
    }:
        raise ValueError("The S0 trainable row contract changed")

    codec = load_s0_tokenizer_codec_for_token_manifest(path)
    if codec is None:
        raise ValueError("The S0 token adapter requires an S0 tokenizer codec")

    expected_surfaces: set[str] = set()

    if codec.atomic_registry is not None:
        expected_surfaces.update(
            codec.atomic_registry.marker_to_value
        )
    else:
        if codec.domain_bpe is None:
            raise RuntimeError(
                "The S0 codec has no representation backend"
            )
        for spec in codec.domain_bpe.manifest[
            "namespaces"
        ].values():
            namespace_marker = spec.get("namespace_marker")
            if namespace_marker is not None:
                expected_surfaces.add(str(namespace_marker))

            expected_surfaces.update(
                str(item["marker"])
                for item in spec["pieces"]
            )
        atomic = codec.manifest.get("atomic")
        if atomic is not None:
            if (
                atomic.get("strategy") != "literal_ensembl_prefix_v1"
            ):
                raise ValueError(
                    "The S0 atomic token strategy changed"
                )
            expected_surfaces.add(str(atomic["token"]))
    if surfaces != expected_surfaces:
        raise ValueError(
            "The token rows do not exactly cover the S0 codec vocabulary"
        )

    return payload


# Block 3: Install and check replicated trainable rows.
def _full_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return one complete local or distributed weight."""

    return weight.full_tensor() if isinstance(weight, DTensor) else weight


def _mean_source_rows(
    weight: torch.Tensor,
    source_ids: list[int],
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Average source rows without one large index allocation."""

    if not source_ids:
        raise ValueError("Token row setup has no source token IDs")
    total = torch.zeros(
        weight.shape[1],
        device=weight.device,
        dtype=torch.float32,
    )
    for start in range(0, len(source_ids), chunk_size):
        indices = torch.tensor(
            source_ids[start : start + chunk_size],
            device=weight.device,
            dtype=torch.long,
        )
        total.add_(weight.index_select(0, indices).float().sum(dim=0))
    return (total / len(source_ids)).to(dtype=weight.dtype)


def _mean_source_row_batch(
    weight: torch.Tensor,
    tokens: list[Mapping[str, Any]],
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Create mean source rows in bounded batches."""

    # Allocate the final row matrix once.
    result = torch.empty(
        (len(tokens), int(weight.shape[1])),
        device=weight.device,
        dtype=weight.dtype,
    )

    # Fill the final matrix in bounded batches.
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        source_groups = [
            [int(value) for value in item["base_token_ids"]]
            for item in chunk
        ]

        flat_ids: list[int] = []
        offsets: list[int] = []

        for source_ids in source_groups:
            if not source_ids:
                raise ValueError("A token row has no source IDs")
            offsets.append(len(flat_ids))
            flat_ids.extend(source_ids)

        means = F.embedding_bag(
            torch.tensor(
                flat_ids,
                device=weight.device,
                dtype=torch.long,
            ),
            weight,
            torch.tensor(
                offsets,
                device=weight.device,
                dtype=torch.long,
            ),
            mode="mean",
            include_last_offset=False,
        )
        result[start : start + len(chunk)].copy_(means)

    return result


def _require_base_weight_shape(
    weight: torch.Tensor,
    *,
    vocab_size: int,
    label: str,
) -> None:
    """Require one full model vocabulary matrix."""

    if weight.ndim != 2 or weight.shape[0] != vocab_size:
        raise RuntimeError(
            f"{label} must have shape ({vocab_size}, hidden_size); "
            f"got {tuple(weight.shape)}"
        )


def install_trainable_token_rows(
    model,
    manifest_path: Path,
) -> dict[str, Any]:
    """Install replicated input and output deltas for one S0 tokenizer."""

    manifest = load_token_manifest(manifest_path)
    tokens = sorted(manifest["tokens"], key=lambda item: int(item["token_id"]))
    token_start = int(tokens[0]["token_id"])
    vocab_size = int(manifest["model_vocab_size"])
    mesh = getattr(model, "_device_mesh", None)
    if mesh is None:
        raise RuntimeError("Token row setup requires the native TP device mesh")

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer.weight.requires_grad or output_layer.weight.requires_grad:
        raise RuntimeError(
            "Token row deltas require frozen base input and output weights"
        )
    input_weight = _full_weight(input_layer.weight).detach()
    output_weight = _full_weight(output_layer.weight).detach()
    try:
        _require_base_weight_shape(
            input_weight,
            vocab_size=vocab_size,
            label="The input embedding weight",
        )
        _require_base_weight_shape(
            output_weight,
            vocab_size=vocab_size,
            label="The output weight",
        )
        if input_weight.shape[1] != output_weight.shape[1]:
            raise RuntimeError("The input and output hidden widths differ")

        # Calculate one mean source row for each token.
        input_delta = _mean_source_row_batch(
            input_weight, tokens
        )
        output_delta = _mean_source_row_batch(
            output_weight, tokens
        )

        # Count trainable rows that overlap the base vocabulary.
        reserved_count = min(
            len(tokens), max(vocab_size - token_start, 0),
        )

        # Convert reserved initial rows into trainable deltas.
        if reserved_count:
            token_end = token_start + reserved_count
            input_delta[:reserved_count].sub_(
                input_weight[token_start:token_end]
            )
            output_delta[:reserved_count].sub_(
                output_weight[token_start:token_end]
            )
        if not torch.isfinite(input_delta).all():
            raise RuntimeError("Token input setup produced a nonfinite value")
        if not torch.isfinite(output_delta).all():
            raise RuntimeError("Token output setup produced a nonfinite value")

    finally:
        del input_weight, output_weight

    input_parameter = nn.Parameter(
        distribute_tensor(
            input_delta,
            mesh,
            [Replicate()],
            src_data_rank=0,
        ),
        requires_grad=True,
    )
    output_parameter = nn.Parameter(
        distribute_tensor(
            output_delta,
            mesh,
            [Replicate()],
            src_data_rank=0,
        ),
        requires_grad=True,
    )
    model.set_input_embeddings(
        TokenRowDeltaEmbedding(input_layer, token_start, input_parameter)
    )
    model.set_output_embeddings(
        TokenRowDeltaOutput(
            output_layer,
            token_start,
            vocab_size,
            output_parameter,
        )
    )
    effective_vocab_size = max(
        vocab_size,
        token_start + len(tokens),
    )
    if effective_vocab_size != int(
        manifest["final_tokenizer_length"]
    ):
        raise RuntimeError(
            "The model vocabulary differs from the tokenizer manifest"
        )
    model.config.vocab_size = effective_vocab_size

    codec = load_s0_tokenizer_codec_for_token_manifest(manifest_path)
    return {
        "token_start": token_start,
        "token_count": len(tokens),
        "base_vocab_size": vocab_size,
        "effective_vocab_size": effective_vocab_size,
        "appended_token_count": max(
            effective_vocab_size - vocab_size,
            0,
        ),
        "trainable_parameters": input_delta.numel() + output_delta.numel(),
        "manifest_sha256": manifest["manifest_sha256"],
        "s0_tokenizer_codec_manifest_sha256": codec.manifest_sha256,
    }


def token_adapter_parameters(model) -> list[tuple[str, nn.Parameter]]:
    """Return all trainable token delta parameters."""

    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.endswith("token_input_delta")
            or name.endswith("token_output_delta")
        )
    ]


def assert_token_adapter_forward_contract(model) -> dict[str, Any]:
    """Require finite gradients through both token delta branches."""

    input_adapter = model.get_input_embeddings()
    output_adapter = model.get_output_embeddings()
    if not isinstance(input_adapter, TokenRowDeltaEmbedding) or not isinstance(
        output_adapter,
        TokenRowDeltaOutput,
    ):
        raise RuntimeError("The forward check requires both token adapters")

    input_weight = _local_replicated_weight(
        input_adapter.token_input_delta,
        "Trainable token input rows",
    )
    output_weight = _local_replicated_weight(
        output_adapter.token_output_delta,
        "Trainable token output rows",
    )
    if input_weight.device != output_weight.device:
        raise RuntimeError("The input and output deltas use different devices")
    if input_weight.shape != output_weight.shape:
        raise RuntimeError("The input and output delta shapes differ")

    input_ids = torch.tensor(
        [
            [
                0,
                input_adapter.token_start,
                input_adapter.token_start + input_adapter.token_count - 1,
            ]
        ],
        device=input_weight.device,
        dtype=torch.long,
    )
    hidden_states = torch.ones(
        (1, 1, int(output_weight.shape[1])),
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    embeddings = input_adapter(input_ids)
    logits = output_adapter(hidden_states)
    loss = (
        embeddings[:, 1:, :].float().square().mean()
        + logits[
            ...,
            (
                output_adapter.token_start,
                output_adapter.token_start + int(output_weight.shape[0]) - 1,
            ),
        ]
        .float()
        .square()
        .mean()
    )
    loss.backward()

    gradient_layouts: dict[str, str] = {}
    try:
        for label, parameter in (
            ("input", input_adapter.token_input_delta),
            ("output", output_adapter.token_output_delta),
        ):
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(
                    f"The {label} token delta has no gradient in the forward check"
                )
            local_gradient = (
                gradient.to_local()
                if isinstance(gradient, DTensor)
                else gradient
            )
            if not torch.isfinite(local_gradient).all():
                raise RuntimeError(
                    f"The {label} token delta has a nonfinite gradient"
                )
            gradient_layouts[label] = (
                "dtensor:"
                + ",".join(
                    type(item).__name__ for item in gradient.placements
                )
                if isinstance(gradient, DTensor)
                else "local"
            )
    finally:
        input_adapter.token_input_delta.grad = None
        output_adapter.token_output_delta.grad = None

    return {
        "status": "passed",
        "input_shape": list(input_weight.shape),
        "output_shape": list(output_weight.shape),
        "gradient_layouts": gradient_layouts,
    }


# Block 4: Copy the S0 codec contract and save exact adapter tensors.
def _local_codec_artifact(root: Path, value: Any, label: str) -> Path:
    """Resolve one codec file below its tokenizer directory."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay below the tokenizer directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{label} must stay below the tokenizer directory"
        ) from error
    if not path.is_file():
        raise ValueError(f"{label} does not name a file: {path}")
    return path


def _copy_exact_artifact(
    source: Path,
    source_root: Path,
    target_root: Path,
) -> str:
    """Copy one artifact and require the same file hash."""

    relative = source.relative_to(source_root.resolve())
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256_file(target) != sha256_file(source):
        raise RuntimeError(f"The copied tokenizer artifact changed: {relative}")
    return str(relative)


def copy_token_adapter_codec_artifacts(
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Copy one S0 token manifest and all referenced codec files."""

    source_root = manifest_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_token_manifest(manifest_path)
    target_manifest = output_dir / TOKEN_ADAPTER_MANIFEST
    shutil.copy2(manifest_path, target_manifest)
    if sha256_file(target_manifest) != sha256_file(manifest_path):
        raise RuntimeError("The copied token manifest changed")
    copied = [TOKEN_ADAPTER_MANIFEST]

    codec = load_s0_tokenizer_codec_for_token_manifest(manifest_path)
    reference = manifest["s0_tokenizer_codec"]
    source_codec = _local_codec_artifact(
        source_root,
        reference.get("manifest_file"),
        "s0_tokenizer_codec.manifest_file",
    )
    copied.append(_copy_exact_artifact(source_codec, source_root, output_dir))

    if codec.atomic_registry is not None:
        registry_reference = codec.manifest.get("fully_atomic_registry")
        if not isinstance(registry_reference, Mapping):
            raise ValueError("The fully atomic codec has no registry reference")

        source_registry = _local_codec_artifact(
            source_codec.parent,
            registry_reference.get("manifest_file"),
            "fully_atomic_registry.manifest_file",
        )
        copied.append(
            _copy_exact_artifact(
                source_registry,
                source_root,
                output_dir,
            )
        )
    else:
        domain_reference = codec.manifest.get("domain_bpe")
        if not isinstance(domain_reference, Mapping):
            raise ValueError("The S0 codec has no Domain-BPE reference")
        if codec.domain_bpe is None:
            raise RuntimeError("The S0 codec has no Domain-BPE backend")

        source_domain = _local_codec_artifact(
            source_codec.parent,
            domain_reference.get("manifest_file"),
            "domain_bpe.manifest_file",
        )
        copied.append(
            _copy_exact_artifact(
                source_domain,
                source_root,
                output_dir,
            )
        )

        namespaces = codec.domain_bpe.manifest.get("namespaces")
        if not isinstance(namespaces, Mapping) or not namespaces:
            raise ValueError(
                "The Domain-BPE manifest has no namespaces"
            )

        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "The S0 codec artifact check requires "
                "the tokenizers package"
            ) from error

        for namespace, spec in sorted(namespaces.items()):
            if not isinstance(spec, Mapping):
                raise ValueError(
                    "The Domain-BPE namespace is not an object: "
                    f"{namespace}"
                )

            source_tokenizer = _local_codec_artifact(
                source_domain.parent,
                spec.get("tokenizer_file"),
                f"namespaces.{namespace}.tokenizer_file",
            )
            if source_tokenizer.suffix != ".json":
                raise ValueError(
                    "The Domain-BPE tokenizer is not JSON: "
                    f"{source_tokenizer}"
                )

            Tokenizer.from_file(str(source_tokenizer))
            copied.append(
                _copy_exact_artifact(
                    source_tokenizer,
                    source_root,
                    output_dir,
                )
            )

    copied_codec = load_s0_tokenizer_codec_for_token_manifest(target_manifest)
    if (
        copied_codec is None
        or copied_codec.manifest_sha256 != codec.manifest_sha256
    ):
        raise RuntimeError("The copied S0 codec identity changed")
    return {
        "copied_files": sorted(set(copied)),
        "s0_tokenizer_codec_manifest_sha256": codec.manifest_sha256,
    }


def save_token_adapter(
    model,
    output_dir: Path,
    manifest_path: Path,
    rank: int,
) -> None:
    """Save exact token deltas and their S0 codec contract."""

    parameters = token_adapter_parameters(model)
    if not parameters:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, parameter in parameters:
        full = (
            parameter.full_tensor()
            if isinstance(parameter, DTensor)
            else parameter
        )
        if rank == 0:
            tensor = full.detach().cpu().contiguous()
            if not torch.isfinite(tensor).all():
                raise RuntimeError("The token adapter has a nonfinite saved value")
            state[name.rsplit(".", 1)[-1]] = tensor
    if rank == 0:
        expected_keys = {"token_input_delta", "token_output_delta"}
        if set(state) != expected_keys:
            raise RuntimeError("The token adapter has unexpected state keys")
        save_file(state, output_dir / TOKEN_ADAPTER_WEIGHTS)
        copy_token_adapter_codec_artifacts(manifest_path, output_dir)
        reloaded = load_file(
            output_dir / TOKEN_ADAPTER_WEIGHTS,
            device="cpu",
        )
        if set(reloaded) != set(state) or any(
            not torch.equal(reloaded[name], tensor)
            for name, tensor in state.items()
        ):
            raise RuntimeError("The token adapter failed save and reload parity")


# Block 5: Load one saved adapter only after exact shape checks.
def _base_weight_shape(weight: torch.Tensor) -> tuple[int, int]:
    """Return the global row and hidden dimensions for one weight."""

    if weight.ndim != 2:
        raise RuntimeError("A base token weight is not a matrix")
    return int(weight.shape[0]), int(weight.shape[1])


def _validate_saved_state(
    state: Mapping[str, torch.Tensor],
    *,
    token_count: int,
    input_width: int,
    output_width: int,
) -> None:
    """Require exact finite input and output delta matrices."""

    expected = {
        "token_input_delta": (token_count, input_width),
        "token_output_delta": (token_count, output_width),
    }
    if set(state) != set(expected):
        raise RuntimeError("The token adapter weights have unexpected keys")
    for name, shape in expected.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"{name} has shape {tuple(tensor.shape)}, expected {shape}"
            )
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains a nonfinite value")


def load_token_adapter_for_inference(
    model,
    adapter_path: Path,
) -> dict[str, Any] | None:
    """Load one complete S0 token adapter for inference."""

    manifest_path = adapter_path / TOKEN_ADAPTER_MANIFEST
    weights_path = adapter_path / TOKEN_ADAPTER_WEIGHTS
    if not manifest_path.is_file() and not weights_path.is_file():
        return None
    if not manifest_path.is_file() or not weights_path.is_file():
        raise RuntimeError("The token adapter manifest and weights must both exist")

    manifest = load_token_manifest(manifest_path)
    state = load_file(weights_path, device="cpu")
    tokens = sorted(manifest["tokens"], key=lambda item: int(item["token_id"]))
    token_start = int(tokens[0]["token_id"])
    token_count = len(tokens)
    vocab_size = int(manifest["model_vocab_size"])
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    input_layer = base_model.get_input_embeddings()
    output_layer = base_model.get_output_embeddings()
    input_rows, input_width = _base_weight_shape(input_layer.weight)
    output_rows, output_width = _base_weight_shape(output_layer.weight)
    if input_rows != vocab_size or output_rows != vocab_size:
        raise RuntimeError("The token adapter does not match the model vocabulary")
    _validate_saved_state(
        state,
        token_count=token_count,
        input_width=input_width,
        output_width=output_width,
    )

    input_delta = nn.Parameter(
        state["token_input_delta"].to(
            device=input_layer.weight.device,
            dtype=input_layer.weight.dtype,
        ),
        requires_grad=False,
    )
    output_device = (
        output_layer.weight.to_local().device
        if isinstance(output_layer.weight, DTensor)
        else output_layer.weight.device
    )
    output_delta = nn.Parameter(
        state["token_output_delta"].to(
            device=output_device,
            dtype=output_layer.weight.dtype,
        ),
        requires_grad=False,
    )
    base_model.set_input_embeddings(
        TokenRowDeltaEmbedding(input_layer, token_start, input_delta)
    )
    base_model.set_output_embeddings(
        TokenRowDeltaOutput(
            output_layer,
            token_start,
            vocab_size,
            output_delta,
        )
    )
    codec = load_s0_tokenizer_codec_for_token_manifest(manifest_path)
    return {
        "token_start": token_start,
        "token_count": token_count,
        "manifest_sha256": manifest["manifest_sha256"],
        "s0_tokenizer_codec_manifest_sha256": codec.manifest_sha256,
    }
