"""Apply the two custom S0 text contracts to typed JSON values.

This module selects biological fields in user and assistant messages. It uses
Domain-BPE to replace their complete values with reversible added-token marker
text. It also checks and decodes model output for exact evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from runtime.world_model_domain_bpe import DomainBPE, load_domain_bpe


S0_CODEC_MANIFEST = "s0_tokenizer_codec_manifest.json"
S0_CODEC_SCHEMA_VERSION = "mentor-rl-world-model-s0-tokenizer-codec-v3"
S0_CODEC_REFERENCE_KEY = "s0_tokenizer_codec"
ORDINARY_DOMAIN_BPE_METHOD = "ordinary_domain_bpe"
ATOMIC_PLUS_DOMAIN_BPE_METHOD = "atomic_plus_domain_bpe"
FULLY_ATOMIC_METHOD = "fully_atomic_identifiers"

S0_CODEC_METHODS = (
    ORDINARY_DOMAIN_BPE_METHOD,
    ATOMIC_PLUS_DOMAIN_BPE_METHOD,
    FULLY_ATOMIC_METHOD,
)
S0_QUESTION_FAMILIES = frozenset(
    {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }
)
ATOMIC_PREFIX_STRATEGY = "literal_ensembl_prefix_v1"
ENSEMBL_PREFIX = "ENSG"

# These fields contain complete Ensembl IDs or gene symbols.
ENSEMBL_FIELDS = frozenset({"gene_id", "candidate_gene_ids"})
SYMBOL_FIELDS = frozenset({"gene_symbol", "gene_symbols"})

# User text contains one compact JSON input object at the end of the question.
ENSEMBL_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])ENSG[0-9]{11}(?![A-Za-z0-9])"
)
SYMBOL_JSON_PATTERN = re.compile(
    r'("gene_symbol"\s*:\s*)("(?:[^"\\]|\\.)*")'
)

# A residual marker after decode proves an incomplete or corrupt result.
S0_MARKER_PATTERN = re.compile(
    r"<\|(?:dbpe_(?:ns|p)_[a-z0-9_]+|s0atom_[a-z0-9_]+)\|>"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


# The atomic method uses no object registry and no symbol namespace token.
ATOMIC_PREFIX_CONTRACT = {
    "strategy": ATOMIC_PREFIX_STRATEGY,
    "namespace": "ensembl_human_gene",
    "prefix": ENSEMBL_PREFIX,
    "token": ENSEMBL_PREFIX,
    "suffix_representation": "domain_bpe",
    "symbol_representation": "domain_bpe",
    "applies_to_fact_roles": ["seen"],
    "object_registry": False,
}
ATOMIC_REGISTRY_SCHEMA = "mentor-rl-world-model-s0-atomic-registry-v1"
ATOMIC_REGISTRY_MANIFEST = "fully_atomic_registry.json"
ATOMIC_MARKER_INDEX_WIDTH = 5
ATOMIC_NAMESPACES = (
    ("ensembl_human_gene", "s0ens"),
    ("human_gene_symbol", "s0sym"),
)
ATOMIC_MARKER_PATTERN = re.compile(
    r"<\|s0atom_(?:s0ens|s0sym)_[0-9]{5}\|>"
)

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


def _manifest_identity(payload: Mapping[str, Any], label: str) -> str:
    """Check and return one internal manifest identity."""

    claimed = _required_sha256(payload.get("manifest_sha256"), label)
    identity = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if _stable_sha256(identity) != claimed:
        raise ValueError(f"{label} failed its internal identity")
    return claimed


def _local_manifest_path(root: Path, value: Any, label: str) -> Path:
    """Resolve one manifest path below its content-addressed directory."""

    relative = Path(_required_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay below the codec directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{label} must stay below the codec directory"
        ) from error
    if not path.is_file():
        raise ValueError(f"{label} does not name a file: {path}")
    return path

class FullyAtomicRegistry:
    """Map each biological entity to one unique atomic marker."""

    def __init__(self, manifest_path: Path):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("The full atomic registry must be one object")

        self.manifest_sha256 = _manifest_identity(
            payload,
            "fully atomic registry manifest_sha256",
        )

        # Validate the schema and method.
        if payload.get("schema_version") != ATOMIC_REGISTRY_SCHEMA:
            raise ValueError("The full atomic registry schema changed")
        if payload.get("method") != FULLY_ATOMIC_METHOD:
            raise ValueError("The full atomic registry method changed")

        # Get the namespaces for encode and decode operations.
        namespaces = payload.get("namespaces")
        if not isinstance(namespaces, Mapping):
            raise ValueError("The full atomic registry has no namespaces")
        expected_namespaces = {
            namespace for namespace, _ in ATOMIC_NAMESPACES
        }
        if set(namespaces) != expected_namespaces:
            raise ValueError("The full atomic registry namespaces changed")

        # build mappings for encoding and decoding
        self.value_to_marker: dict[str, dict[str, str]] = {}
        self.marker_to_value: dict[str, str] = {}
        self.marker_to_namespace: dict[str, str] = {}

        # populate mappings for each namespace
        for namespace, slug in ATOMIC_NAMESPACES:
            # get the namespace specification from the manifest
            spec = namespaces[namespace]
            if not isinstance(spec, Mapping):
                raise ValueError(f"The full atomic namespace {namespace} must be one object")

            # get the entries for the namespace
            entries = spec.get("entries")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"the full atomic namespace {namespace} has no entries")

            namespace_map: dict[str, str] = {}
            # loop through each entry and validate
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"the full atomic namespace {namespace} has an invalid entry")

                # get the value and marker for the entry
                value = _required_string(
                    entry.get("value"),
                    f"{namespace} value",
                )
                marker = _required_string(
                    entry.get("marker"),
                    f"{namespace} marker",
                )

                # validate the marker format
                if namespace == "ensembl_human_gene":
                    if ENSEMBL_TEXT_PATTERN.fullmatch(value) is None:
                        raise ValueError(f"The Ensembl value is not canonical: {value!r}")
                # check symbol format
                elif value != value.strip():
                    raise ValueError(
                        f"The gene symbol is not canonical: {value!r}"
                    )

                if (
                    ATOMIC_MARKER_PATTERN.fullmatch(marker) is None
                    or not marker.startswith(f"<|s0atom_{slug}_")
                ):
                    raise ValueError(
                        f"The marker has the wrong namespace: {marker!r}"
                    )

                # check for duplicate values and markers
                if value in namespace_map:
                    raise ValueError(
                        f"The registry repeats this value: {value!r}"
                    )
                if marker in self.marker_to_value:
                    raise ValueError(
                        f"The registry repeats this marker: {marker!r}"
                    )

                # add the value to the namespace map if all checks pass
                namespace_map[value] = marker
                self.marker_to_value[marker] = value
                self.marker_to_namespace[marker] = namespace

            # add the reverse mapping
            self.value_to_marker[namespace] = namespace_map

        self.manifest = dict(payload)
        self.manifest_path = manifest_path.resolve()

    # Block 2: encode values according to the registry
    def encode_value(self, namespace: str, value: str) -> str:
        """given a namespace and a value, return the atomic marker for that value"""

        # retrieve the namespace map
        namespace_map = self.value_to_marker.get(namespace)
        if namespace_map is None:
            raise KeyError(f"Unknown atomic namespace: {namespace}")

        # retrieve the marker for the value
        marker = namespace_map.get(value)
        if marker is None:
            raise ValueError(f"The full atomic registry has no {namespace} value: {value!r}")

        # return the marker given a value
        return marker

    # Block 3: decode markers according to registry
    def decode_value(self, namespace: str, marker: str) -> str:
        """decode a marker to its original value given a namespace"""

        # check that namespace, value exist and are valid
        if namespace not in self.value_to_marker:
            raise KeyError(f"Unknown full atomic namespace: {namespace}")
        value = self.marker_to_value.get(marker)
        if value is None:
            raise ValueError(
                f"The full atomic registry has no marker: {marker!r}"
            )

        marker_namespace = self.marker_to_namespace[marker]
        if marker_namespace != namespace:
            raise ValueError(
                f"The marker {marker!r} does not match the namespace {namespace!r}"
            )
        return value

    # text decoding: convert all markers in a string to their original values
    def decode_text(self, text: str) -> str:
        """decode all atomic markers in text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # define a function to restore markers to values
        def restore(match: re.Match[str]) -> str:
            marker = match.group(0)
            return self.marker_to_value.get(marker, marker)

        # use regex to find all markers and replace them with their corresponding values
        return ATOMIC_MARKER_PATTERN.sub(restore, text)


    # round-trip test: encode a value and then decode it to check for consistency
    def round_trip(self, namespace: str, value: str) -> bool:
        """Return true when one value has an exact encode-decode cycle."""
        marker = self.encode_value(namespace, value)
        return self.decode_value(namespace, marker) == value


class S0TokenizerCodec:
    """Apply one S0 tokenizer representation to typed biological values."""

    def __init__(self, manifest_path: Path):
        # Validate the codec manifest before it can select another artifact.
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("The S0 tokenizer codec manifest must be an object")
        self.manifest_sha256 = _manifest_identity(
            payload,
            "S0 tokenizer codec manifest_sha256",
        )
        if payload.get("schema_version") != S0_CODEC_SCHEMA_VERSION:
            raise ValueError("The S0 tokenizer codec schema is not supported")
        method = payload.get("method")
        if method not in S0_CODEC_METHODS:
            raise ValueError(f"The S0 tokenizer method is invalid: {method!r}")

        self.schema_version = S0_CODEC_SCHEMA_VERSION
        self.method = str(method)
        self.manifest = dict(payload)
        self.manifest_path = manifest_path.resolve()

        # Load the representation backend for this tokenizer method.
        self.domain_bpe: DomainBPE | None = None
        self.atomic_registry: FullyAtomicRegistry | None = None

        if self.method == FULLY_ATOMIC_METHOD:
            if payload.get("domain_bpe") is not None:
                raise ValueError(
                    "The fully atomic codec cannot use Domain-BPE"
                )
            if payload.get("atomic") is not None:
                raise ValueError(
                    "The fully atomic codec cannot use the atomic-prefix contract"
                )

            # Resolve the registry below the codec directory.
            registry_reference = payload.get("fully_atomic_registry")
            if not isinstance(registry_reference, Mapping):
                raise ValueError(
                    "The fully atomic codec requires a registry reference"
                )

            registry_path = _local_manifest_path(
                self.manifest_path.parent,
                registry_reference.get("manifest_file"),
                "fully_atomic_registry.manifest_file",
            )
            if _sha256_file(registry_path) != _required_sha256(
                registry_reference.get("manifest_sha256"),
                "fully_atomic_registry.manifest_sha256",
            ):
                raise ValueError(
                    "The fully atomic registry file identity changed"
                )
            self.atomic_registry = FullyAtomicRegistry(registry_path)
            if self.atomic_registry.manifest_sha256 != _required_sha256(
                registry_reference.get("internal_manifest_sha256"),
                "fully_atomic_registry.internal_manifest_sha256",
            ):
                raise ValueError(
                    "The fully atomic registry internal identity changed"
                )
            return

        # Domain-BPE methods cannot declare a full atomic registry.
        if payload.get("fully_atomic_registry") is not None:
            raise ValueError(
                "A Domain-BPE codec cannot use a full atomic registry"
            )

        domain_reference = payload.get("domain_bpe")
        if not isinstance(domain_reference, Mapping):
            raise ValueError("The S0 codec requires a Domain-BPE reference")

        domain_path = _local_manifest_path(
            self.manifest_path.parent,
            domain_reference.get("manifest_file"),
            "domain_bpe.manifest_file",
        )
        if _sha256_file(domain_path) != _required_sha256(
            domain_reference.get("manifest_sha256"),
            "domain_bpe.manifest_sha256",
        ):
            raise ValueError("The Domain-BPE manifest file identity changed")

        self.domain_bpe = load_domain_bpe(domain_path)
        if self.domain_bpe.manifest_sha256 != _required_sha256(
            domain_reference.get("internal_manifest_sha256"),
            "domain_bpe.internal_manifest_sha256",
        ):
            raise ValueError("The Domain-BPE internal identity changed")
        if self.domain_bpe.method != self.method:
            raise ValueError("The codec and Domain-BPE methods differ")
        if self.domain_bpe.namespaces != (
            "ensembl_human_gene",
            "human_gene_symbol",
        ):
            raise ValueError("The S0 Domain-BPE namespace order changed")

        atomic = payload.get("atomic")
        if self.method == ORDINARY_DOMAIN_BPE_METHOD:
            if atomic is not None:
                raise ValueError(
                    "Ordinary Domain-BPE cannot contain an atomic contract"
                )
        elif atomic != ATOMIC_PREFIX_CONTRACT:
            raise ValueError("The S0 atomic prefix contract is not exact")

    # Block 2: Encode only typed biological values in chat messages.
    @staticmethod
    def _namespace(field_name: str) -> str | None:
        """Return the Domain-BPE namespace for one JSON field."""

        if field_name in ENSEMBL_FIELDS:
            return "ensembl_human_gene"
        if field_name in SYMBOL_FIELDS:
            return "human_gene_symbol"
        return None

    def encode_value(self, namespace: str, value: str) -> str:
        """Validate and encode one complete typed S0 value."""

        if not isinstance(value, str):
            raise TypeError("The S0 biological value must be a string")

        if namespace == "ensembl_human_gene":
            if ENSEMBL_TEXT_PATTERN.fullmatch(value) is None:
                raise ValueError(f"The Ensembl value is not canonical: {value!r}")
        elif namespace == "human_gene_symbol":
            if not value.strip():
                raise ValueError("The gene symbol must be non-empty")
        else:
            raise KeyError(f"Unknown S0 tokenizer namespace: {namespace!r}")

        if self.atomic_registry is not None:
            return self.atomic_registry.encode_value(namespace, value)

        if self.domain_bpe is None:
            raise RuntimeError("The S0 codec has no representation backend")

        return self.domain_bpe.encode_value(namespace, value)

    def _encode_typed_value(
        self,
        value: Any,
        field_name: str | None = None,
    ) -> Any:
        """Encode biological strings inside one typed JSON value.

        A list inherits its parent field name. This rule applies one namespace
        to every item in `gene_symbols` or `candidate_gene_ids`.
        """

        if isinstance(value, Mapping):
            return {
                str(key): self._encode_typed_value(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._encode_typed_value(item, field_name)
                for item in value
            ]
        if not isinstance(value, str) or field_name is None:
            return value
        namespace = self._namespace(field_name)
        if namespace is None:
            return value
        return self.encode_value(namespace, value)

    def encode_answer_text(self, text: str) -> str:
        """Encode biological values in one assistant JSON object.

        The result uses compact canonical JSON. JSON keys, punctuation, status,
        and the defer action remain ordinary base-tokenizer text.
        """

        try:
            answer = json.loads(text)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("The S0 assistant target must be JSON") from error
        if not isinstance(answer, Mapping):
            raise ValueError("The S0 assistant target must be one JSON object")
        return json.dumps(
            self._encode_typed_value(answer),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _encode_user_text(self, text: str) -> str:
        """Encode the one biological value in an S0 user message.

        S0.1 and S0.3 contain a `gene_symbol` JSON value. S0.2 contains one
        canonical Ensembl ID. Other prompt text remains unchanged.
        """

        def encode_symbol(match: re.Match[str]) -> str:
            raw_symbol = json.loads(match.group(2))
            encoded_symbol = self.encode_value(
                "human_gene_symbol",
                raw_symbol,
            )
            return match.group(1) + json.dumps(
                encoded_symbol,
                ensure_ascii=False,
            )

        with_encoded_symbols = SYMBOL_JSON_PATTERN.sub(encode_symbol, text)
        return ENSEMBL_TEXT_PATTERN.sub(
            lambda match: self.encode_value(
                "ensembl_human_gene",
                match.group(0),
            ),
            with_encoded_symbols,
        )

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        question_family: str | None = None,
    ) -> list[dict[str, str]]:
        """Encode S0 user and assistant messages for the model.

        System text remains unchanged. The optional family check catches a
        record-route error before tokenization.
        """

        if (
            question_family is not None
            and question_family not in S0_QUESTION_FAMILIES
        ):
            raise ValueError(
                f"The S0 question family is invalid: {question_family!r}"
            )
        encoded: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"The chat message {index} has an invalid role: {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"The chat message {index} must contain text"
                )
            if role == "user":
                content = self._encode_user_text(content)
            elif role == "assistant":
                content = self.encode_answer_text(content)
            encoded.append({"role": role, "content": content})
        return encoded

    # Block 3: Check and decode generated S0 answers.
    def _decode_exact_value(
        self,
        namespace: str,
        value: str,
    ) -> str | None:
        """Decode one exact value from the required namespace."""

        if self.atomic_registry is not None:
            try:
                return self.atomic_registry.decode_value(namespace, value)
            except (KeyError, TypeError, ValueError):
                return None

        if self.domain_bpe is None:
            raise RuntimeError("The S0 codec has no representation backend")

        decoded = self.domain_bpe.decode_text(value)
        if decoded == value:
            return None

        try:
            encoded = self.encode_value(namespace, decoded)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None

        return decoded if encoded == value else None

    def _inspect_code(
        self,
        value: Any,
        namespace: str,
        path: str,
        violations: list[str],
    ) -> None:
        """Check one encoded biological value in a generated answer."""

        if not isinstance(value, str):
            violations.append(f"{path}:expected_string")
            return
        if self._decode_exact_value(namespace, value) is None:
            violations.append(f"{path}:expected_exact_s0_code")

    def _inspect_answer(
        self,
        value: Any,
        violations: list[str],
        *,
        field_name: str | None = None,
        path: str = "$",
    ) -> int:
        """Check each biological field in one encoded answer object.

        The return value counts checked biological strings. The evaluation
        report uses this count to distinguish representation checks from schema
        checks.
        """

        checked = 0
        if isinstance(value, Mapping):
            for key, item in value.items():
                checked += self._inspect_answer(
                    item,
                    violations,
                    field_name=str(key),
                    path=f"{path}.{key}",
                )
            return checked
        if isinstance(value, list):
            for index, item in enumerate(value):
                checked += self._inspect_answer(
                    item,
                    violations,
                    field_name=field_name,
                    path=f"{path}[{index}]",
                )
            return checked
        if field_name is None:
            return checked
        namespace = self._namespace(field_name)
        if namespace is not None:
            checked += 1
            self._inspect_code(value, namespace, path, violations)
        return checked

    def _decode_typed_value(
        self,
        value: Any,
        field_name: str | None = None,
    ) -> Any:
        """Restore biological values inside one parsed JSON object.

        This method restores values before JSON serialization. It preserves
        valid escapes if a symbol contains a quote or a backslash.
        """

        if isinstance(value, Mapping):
            return {
                str(key): self._decode_typed_value(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._decode_typed_value(item, field_name)
                for item in value
            ]
        if not isinstance(value, str) or field_name is None:
            return value

        namespace = self._namespace(field_name)
        if namespace is None:
            return value

        decoded = self._decode_exact_value(namespace, value)
        return value if decoded is None else decoded

    def decode_text(self, text: str) -> str:
        """Restore valid Domain-BPE marker runs in arbitrary text."""

        if self.atomic_registry is not None:
            return self.atomic_registry.decode_text(text)

        if self.domain_bpe is None:
            raise RuntimeError("The S0 codec has no representation backend")

        return self.domain_bpe.decode_text(text)

    def decode_generated_answer(
        self,
        text: str,
    ) -> tuple[str, dict[str, Any]]:
        """Decode one model answer and return its representation report.

        This report covers tokenizer representation only. The S0 evaluator
        separately checks JSON fields, answer values, and family metrics.
        """

        violations: list[str] = []
        checked = 0
        residual_markers: list[str]
        try:
            encoded_answer = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            violations.append("$:generation_is_not_json")
            decoded_text = self.decode_text(text)
        else:
            if not isinstance(encoded_answer, Mapping):
                violations.append("$:generation_is_not_object")
                decoded_text = self.decode_text(text)
            else:
                checked = self._inspect_answer(
                    encoded_answer,
                    violations,
                )
                decoded_answer = self._decode_typed_value(encoded_answer)
                decoded_text = json.dumps(
                    decoded_answer,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )

        residual_markers = sorted(
            set(S0_MARKER_PATTERN.findall(decoded_text))
        )
        if residual_markers:
            violations.append("$:unmapped_s0_markers")
        report = {
            "valid": not violations,
            "method": self.method,
            "codec_manifest_sha256": self.manifest_sha256,
            "representation_backend": (
                "fully_atomic_registry"
                if self.atomic_registry is not None
                else "domain_bpe"
            ),
            "domain_bpe_manifest_sha256": (
                None
                if self.domain_bpe is None
                else self.domain_bpe.manifest_sha256
            ),
            "checked_biological_values": checked,
            "violations": violations,
            "residual_markers": residual_markers,
        }
        return decoded_text, report


def load_s0_tokenizer_codec(path: Path) -> S0TokenizerCodec:
    """Load one S0 codec from a directory or manifest path."""

    if path.is_dir():
        path = path / S0_CODEC_MANIFEST
    return S0TokenizerCodec(path)


def load_s0_tokenizer_codec_for_token_manifest(
    manifest_path: Path,
) -> S0TokenizerCodec | None:
    """Load the optional codec that one token manifest references.

    A missing reference identifies the plain base tokenizer. A present
    reference must pin both the codec file and its internal manifest identity.
    """

    token_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(token_manifest, Mapping):
        raise ValueError("The tokenizer manifest must be one object")
    _manifest_identity(
        token_manifest,
        "tokenizer manifest_sha256",
    )
    reference = token_manifest.get(S0_CODEC_REFERENCE_KEY)
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise ValueError("The S0 tokenizer codec reference must be an object")

    codec_path = _local_manifest_path(
        manifest_path.parent,
        reference.get("manifest_file"),
        "s0_tokenizer_codec.manifest_file",
    )
    if _sha256_file(codec_path) != _required_sha256(
        reference.get("manifest_sha256"),
        "s0_tokenizer_codec.manifest_sha256",
    ):
        raise ValueError("The S0 tokenizer codec file identity changed")
    codec = load_s0_tokenizer_codec(codec_path)
    if codec.manifest_sha256 != _required_sha256(
        reference.get("internal_manifest_sha256"),
        "s0_tokenizer_codec.internal_manifest_sha256",
    ):
        raise ValueError("The S0 tokenizer codec internal identity changed")
    if codec.method != token_manifest.get("method"):
        raise ValueError("The token manifest and codec methods differ")
    return codec
