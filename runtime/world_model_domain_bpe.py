"""Load the two reversible Domain-BPE models for S0.

The Ensembl and symbol namespaces use separate 240-piece BPE models. This
module validates each content-addressed model and maps its pieces to added
token markers. It does not fit or change a tokenizer.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DOMAIN_BPE_MANIFEST = "domain_bpe_manifest.json"
DOMAIN_BPE_SCHEMA_VERSION = "mentor-rl-world-model-s0-domain-bpe-v3"
DOMAIN_BPE_VOCAB_SIZE = 240
DOMAIN_BPE_METHODS = (
    "ordinary_domain_bpe",
    "atomic_plus_domain_bpe",
)
NAMESPACE_SLUGS = {
    "ensembl_human_gene": "s0ens",
    "human_gene_symbol": "s0sym",
}
MARKER_PATTERN = re.compile(r"<\|dbpe_(?:ns|p)_[a-z0-9_]+\|>")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _stable_sha256(value: Any) -> str:
    """Return the SHA-256 digest for one canonical JSON value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: Any, label: str) -> str:
    """Return one required nonempty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_sha256(value: Any, label: str) -> str:
    """Return one required lowercase SHA-256 digest."""

    text = _required_string(value, label)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _local_file(
    root: Path,
    value: Any,
    expected_sha256: Any,
    label: str,
) -> Path:
    """Resolve and validate one content-addressed local file.

    The resolved path must stay below the manifest directory. This rule blocks
    parent traversal and symlink escape from a tokenizer artifact.
    """

    relative = Path(_required_string(value, f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must stay below the manifest directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{label}.path must stay below the manifest directory"
        ) from error
    if not path.is_file():
        raise ValueError(f"{label}.path does not name a file: {path}")
    required_hash = _required_sha256(
        expected_sha256,
        f"{label}.sha256",
    )
    if _sha256_file(path) != required_hash:
        raise ValueError(f"{label} file identity changed")
    return path


class DomainBPE:
    """Load and validate both namespace BPE models."""

    def __init__(self, manifest_path: Path):
        # Check the internal manifest identity before any referenced file.
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("The Domain-BPE manifest must be one object")
        claimed = _required_sha256(
            payload.get("manifest_sha256"),
            "Domain-BPE manifest_sha256",
        )
        identity = {
            str(key): value
            for key, value in payload.items()
            if key != "manifest_sha256"
        }
        if _stable_sha256(identity) != claimed:
            raise ValueError(
                "The Domain-BPE manifest failed its internal identity"
            )

        # Require one v4 S0 method and exactly two 240-piece namespaces.
        if payload.get("schema_version") != DOMAIN_BPE_SCHEMA_VERSION:
            raise ValueError("The Domain-BPE schema is not supported")
        method = payload.get("method")
        if method not in DOMAIN_BPE_METHODS:
            raise ValueError(f"The Domain-BPE method is invalid: {method!r}")
        if payload.get("vocab_per_namespace") != DOMAIN_BPE_VOCAB_SIZE:
            raise ValueError("Each Domain-BPE namespace must have 240 pieces")
        namespaces = payload.get("namespaces")
        if not isinstance(namespaces, Mapping) or set(namespaces) != set(
            NAMESPACE_SLUGS
        ):
            raise ValueError("The Domain-BPE namespace set is not exact")

        self.manifest_path = manifest_path.resolve()
        self.manifest = dict(payload)
        self.manifest_sha256 = claimed
        self.method = str(method)
        self._tokenizer_paths: dict[str, Path] = {}
        self._tokenizers: dict[str, Any] = {}
        self._piece_markers: dict[str, dict[str, str]] = {}
        self._marker_pieces: dict[str, tuple[str, str]] = {}
        self._run_patterns: dict[str, re.Pattern[str]] = {}
        self._value_prefixes: dict[str, str] = {}

        # Validate each model file, piece table, and added-token marker.
        for namespace, slug in NAMESPACE_SLUGS.items():
            spec = namespaces[namespace]
            if not isinstance(spec, Mapping):
                raise ValueError(
                    f"The Domain-BPE namespace is not an object: {namespace}"
                )
            self._tokenizer_paths[namespace] = _local_file(
                self.manifest_path.parent,
                spec.get("tokenizer_file"),
                spec.get("tokenizer_sha256"),
                f"Domain-BPE {namespace} tokenizer",
            )
            pieces = spec.get("pieces")
            if not isinstance(pieces, list) or len(pieces) != (
                DOMAIN_BPE_VOCAB_SIZE
            ):
                raise ValueError(
                    f"The Domain-BPE {namespace} piece count is not 240"
                )

            piece_markers: dict[str, str] = {}
            for piece_id, row in enumerate(pieces):
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"The Domain-BPE {namespace} piece {piece_id} "
                        "is not an object"
                    )
                piece = _required_string(
                    row.get("piece"),
                    f"Domain-BPE {namespace} piece {piece_id}",
                )
                expected_marker = f"<|dbpe_p_{slug}_{piece_id:03d}|>"
                if row.get("piece_id") != piece_id:
                    raise ValueError(
                        f"The Domain-BPE {namespace} piece IDs are not ordered"
                    )
                if row.get("marker") != expected_marker:
                    raise ValueError(
                        f"The Domain-BPE {namespace} marker is not exact"
                    )
                if piece in piece_markers:
                    raise ValueError(
                        f"The Domain-BPE {namespace} pieces are not unique"
                    )
                if expected_marker in self._marker_pieces:
                    raise ValueError("The Domain-BPE markers are not unique")
                piece_markers[piece] = expected_marker
                self._marker_pieces[expected_marker] = (namespace, piece)
            self._piece_markers[namespace] = piece_markers

            # Ordinary Domain-BPE uses two namespace rows. The atomic method
            # uses literal ENSG and no symbol namespace row.
            if self.method == "ordinary_domain_bpe":
                expected_namespace_marker: str | None = (
                    f"<|dbpe_ns_{slug}|>"
                )
                expected_prefix = ""
            elif namespace == "ensembl_human_gene":
                expected_namespace_marker = "ENSG"
                expected_prefix = "ENSG"
            else:
                expected_namespace_marker = None
                expected_prefix = ""
            if spec.get("namespace_marker") != expected_namespace_marker:
                raise ValueError(
                    f"The Domain-BPE {namespace} namespace marker changed"
                )
            if spec.get("value_prefix", "") != expected_prefix:
                raise ValueError(
                    f"The Domain-BPE {namespace} value prefix changed"
                )
            self._value_prefixes[namespace] = expected_prefix

            alternatives = "|".join(
                re.escape(marker)
                for marker in sorted(
                    piece_markers.values(),
                    key=len,
                    reverse=True,
                )
            )
            marker_prefix = (
                ""
                if expected_namespace_marker is None
                else re.escape(expected_namespace_marker)
            )
            self._run_patterns[namespace] = re.compile(
                f"{marker_prefix}((?:{alternatives})+)"
            )

    @property
    def namespaces(self) -> tuple[str, ...]:
        """Return the two namespaces in canonical order."""

        return tuple(NAMESPACE_SLUGS)

    # Block 2: Encode one typed value and restore marker runs.
    def _tokenizer(self, namespace: str):
        """Load one validated namespace tokenizer on first use.

        The delayed import keeps manifest checks available without the
        `tokenizers` package. A train or tokenizer job requires that package.
        """

        if namespace not in NAMESPACE_SLUGS:
            raise KeyError(f"Unknown Domain-BPE namespace: {namespace}")
        if namespace not in self._tokenizers:
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Domain-BPE requires the tokenizers package"
                ) from error
            tokenizer = Tokenizer.from_file(
                str(self._tokenizer_paths[namespace])
            )

            # The file hash is authoritative. This check gives a clear error
            # if its vocabulary does not match the declared piece table.
            expected_vocab = {
                piece: piece_id
                for piece_id, piece in enumerate(
                    self._piece_markers[namespace]
                )
            }
            if tokenizer.get_vocab() != expected_vocab:
                raise ValueError(
                    f"The Domain-BPE {namespace} vocabulary changed"
                )
            self._tokenizers[namespace] = tokenizer
        return self._tokenizers[namespace]

    def encode_value(self, namespace: str, value: str) -> str:
        """Encode one complete typed value as added-token markers.

        The function removes only a declared prefix before BPE. It then checks
        that the pieces join to the exact source text with no fallback token.
        """

        if namespace not in NAMESPACE_SLUGS:
            raise KeyError(f"Unknown Domain-BPE namespace: {namespace}")
        if not isinstance(value, str) or not value:
            raise ValueError("Domain-BPE values must be non-empty strings")

        value_prefix = self._value_prefixes[namespace]
        if value_prefix:
            if not value.startswith(value_prefix) or len(value) == len(
                value_prefix
            ):
                raise ValueError(
                    f"Domain-BPE value lacks prefix {value_prefix!r}: {value!r}"
                )
            bpe_value = value[len(value_prefix) :]
        else:
            bpe_value = value

        pieces = self._tokenizer(namespace).encode(
            bpe_value,
            add_special_tokens=False,
        ).tokens
        if not pieces or "".join(pieces) != bpe_value:
            raise ValueError(
                f"Domain-BPE failed exact piece round trip for {value!r}"
            )
        try:
            markers = [
                self._piece_markers[namespace][piece]
                for piece in pieces
            ]
        except KeyError as error:
            raise ValueError(
                f"Domain-BPE emitted an undeclared piece: {error}"
            ) from error

        namespace_marker = self.manifest["namespaces"][namespace].get(
            "namespace_marker"
        )
        return (
            "" if namespace_marker is None else str(namespace_marker)
        ) + "".join(markers)

    def decode_text(self, text: str) -> str:
        """Restore all valid Domain-BPE marker runs in arbitrary text.

        A malformed or mixed-namespace run stays unchanged. This rule lets the
        evaluation path report residual markers instead of hiding corruption.
        """

        if not isinstance(text, str):
            raise TypeError("Domain-BPE text must be a string")
        for namespace in NAMESPACE_SLUGS:
            run_pattern = self._run_patterns[namespace]

            def restore(match: re.Match[str]) -> str:
                restored_pieces: list[str] = []
                for marker in MARKER_PATTERN.findall(match.group(1)):
                    marker_namespace, piece = self._marker_pieces[marker]
                    if marker_namespace != namespace:
                        return match.group(0)
                    restored_pieces.append(piece)
                return self._value_prefixes[namespace] + "".join(
                    restored_pieces
                )

            text = run_pattern.sub(restore, text)
        return text

    def round_trip(self, namespace: str, value: str) -> bool:
        """Return true when one value has an exact encode-decode cycle."""

        return self.decode_text(self.encode_value(namespace, value)) == value


def load_domain_bpe(path: Path) -> DomainBPE:
    """Load one Domain-BPE artifact from a directory or manifest path."""

    if path.is_dir():
        path = path / DOMAIN_BPE_MANIFEST
    return DomainBPE(path)
