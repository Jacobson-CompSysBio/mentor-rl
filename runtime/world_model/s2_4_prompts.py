"""Bind frozen multiplex-map questions and audit split wording across S2.

The caller selects the graph content. Each returned row keeps every scoped
layer support, including when a canonical-ID bound limits the neighbor rows.
"""

from collections.abc import Mapping
import json
import re
from string import Formatter
from typing import Any

from .contract import REPOSITORY_ROOT
from .s1_prompt_banks import (
    FRAMINGS, HELD_OUT_JACCARD_LIMIT, HELD_OUT_SHARED_NGRAM, PromptBankError,
    S2_PROMPT_SPLITS, _ngrams, _tokens, validate_s2_edge_requests,
    validate_s2_prompt_banks,
)
from .s2_2_prompts import _rendered_variants, validate_s2_2_requests
from .s2_3_prompts import _neighbor_variants, validate_s2_3_requests
from .sources import file_sha256


SCHEMA_VERSION = "mentor-rl-world-model-s2-4-requests-v1"
MODES = ("complete", "top_k")
DIRECTIONS = ("incident", "in", "out", "undirected")
FIELDS = ("multiplex_id", "scope_text", "gene")


def _validate_template(template: Any, mode: str, direction: str) -> None:
    """Reject changed slots, native roles, or a weight-based neighbor selection."""
    if not isinstance(template, str) or not template or template != template.strip() or "\n" in template:
        raise PromptBankError("S2.4 requests must be nonempty trimmed lines")
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as error:
        raise PromptBankError("invalid S2.4 request formatting") from error
    expected = (*FIELDS, "top_k") if mode == "top_k" else FIELDS
    fields = [field for _, field, _, _ in parsed if field is not None]
    if sorted(fields) != sorted(expected) or any(spec or conversion for _, _, spec, conversion in parsed):
        raise PromptBankError("S2.4 requests must bind exactly the declared fields once")
    if not re.search(r"weight", template, re.I) or not re.search(r"layer", template, re.I):
        raise PromptBankError("S2.4 requests must retain weighted layer support")
    if mode == "strongest_unspecified":
        if not re.search(r"strongest|strength", template, re.I):
            raise PromptBankError("S2.4 ranking diagnostics must preserve unresolved strength")
        return
    if re.search(r"strongest|heaviest|highest.weight|descending weight|strength", template, re.I):
        raise PromptBankError("S2.4 ordinary requests cannot rank across source weights")
    if not re.search(r"every|all|whole|complete", template, re.I):
        raise PromptBankError("S2.4 requires all scoped supports for each returned row")
    if mode == "complete":
        if re.search(r"prefix|bounded|first \{top_k\}|earliest \{top_k\}", template, re.I):
            raise PromptBankError("S2.4 complete requests cannot impose a neighbor bound")
    elif not (
        re.search(r"first|initial|prefix|earliest", template, re.I)
        and re.search(r"canonical gene ID", template, re.I)
        and re.search(r"ascend", template, re.I)
        and re.search(r"fewer|smaller|shorter|exhaust|insufficient", template, re.I)
    ):
        raise PromptBankError("S2.4 top_k must select an ascending canonical-ID prefix and permit exhaustion")
    if direction == "incident":
        if not (re.search(r"incoming|both arrow roles", template, re.I)
                and re.search(r"outgoing|both arrow roles", template, re.I)
                and re.search(r"undirected|unordered|unoriented", template, re.I)):
            raise PromptBankError("S2.4 incident requests must retain all native orientations")
    elif direction == "in":
        if not (re.search(r"incoming", template, re.I) and re.search(
            r"into target \{gene\}|\{gene\} as the destination|point to \{gene\}", template, re.I
        )) or re.search(r"outgoing|undirected", template, re.I):
            raise PromptBankError("S2.4 incoming requests must bind the target role")
    elif direction == "out":
        if not (re.search(r"outgoing", template, re.I) and re.search(
            r"from source \{gene\}|\{gene\} supplies the origin|leave \{gene\}", template, re.I
        )) or re.search(r"incoming|undirected", template, re.I):
            raise PromptBankError("S2.4 outgoing requests must bind the source role")
    elif (not re.search(r"undirected|unordered", template, re.I)
          or re.search(r"incoming|outgoing", template, re.I)):
        raise PromptBankError("S2.4 undirected requests cannot invent arrow roles")


def validate_s2_4_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen map request schema, exact slots, and native roles."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ["schema_version", "description", *S2_PROMPT_SPLITS, "ranking_requests"]
            or payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(payload["description"], str) or not payload["description"].strip()):
        raise PromptBankError("the S2.4 request schema changed")
    counts = {}
    if not isinstance(payload["ranking_requests"], Mapping) or list(payload["ranking_requests"]) != list(S2_PROMPT_SPLITS):
        raise PromptBankError("S2.4 ranking diagnostics must declare every split")
    for split in S2_PROMPT_SPLITS:
        bank = payload[split]
        if not isinstance(bank, Mapping) or list(bank) != list(MODES):
            raise PromptBankError("S2.4 must declare complete and top_k in order")
        counts[split] = {}
        for mode in MODES:
            if not isinstance(bank[mode], Mapping) or list(bank[mode]) != list(DIRECTIONS):
                raise PromptBankError("S2.4 must declare all four map directions in order")
            counts[split][mode] = {}
            for direction in DIRECTIONS:
                templates = bank[mode][direction]
                _validate_templates(templates, mode, direction)
                counts[split][mode][direction] = len(templates)
        _validate_templates(payload["ranking_requests"][split], "strongest_unspecified", "incident")
    return {"template_counts": counts,
            "ranking_template_counts": {split: len(payload["ranking_requests"][split]) for split in S2_PROMPT_SPLITS}}


def _validate_templates(templates: Any, mode: str, direction: str) -> None:
    """Check each frozen kind once, including duplicate template detection."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError("S2.4 requires at least four requests per kind")
    for template in templates:
        _validate_template(template, mode, direction)
    if len(set(templates)) != len(templates):
        raise PromptBankError("S2.4 requests must be distinct within each kind")


def load_s2_4_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read a separately pinned map request bank without changes to older banks."""
    if not isinstance(pin, Mapping) or list(pin) != ["path", "sha256"]:
        raise PromptBankError("an S2.4 request pin must list path, then sha256")
    path = (REPOSITORY_ROOT / str(pin["path"])).resolve()
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise PromptBankError("the S2.4 request bank does not match its pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_s2_4_requests(payload)
    return payload


def render_s2_4_request(
    payload: Mapping[str, Any], *, split: str, direction: str, mode: str,
    variant_index: int, multiplex_id: str, layer_scope: Mapping[str, Any], gene: str,
    top_k: int | None = None, ranking: str | None = None,
) -> str:
    """Bind caller-selected scope and gene values to deterministic frozen prose."""
    if split not in S2_PROMPT_SPLITS or mode not in MODES or direction not in DIRECTIONS:
        raise PromptBankError("unknown S2.4 split, selection, or direction")
    if type(variant_index) is not int or variant_index < 0:
        raise PromptBankError("the S2.4 wording index must be nonnegative")
    if ranking not in (None, "strongest_unspecified") or (ranking and direction != "incident"):
        raise PromptBankError("unknown or incompatible S2.4 ranking request")
    if not isinstance(layer_scope, Mapping) or set(layer_scope) != {"kind", "layer_ids", "layer_family"}:
        raise PromptBankError("S2.4 requires a typed layer scope")
    if layer_scope["kind"] == "layers":
        if not isinstance(layer_scope["layer_ids"], list) or layer_scope["layer_family"] is not None:
            raise PromptBankError("S2.4 explicit scope requires layer IDs and no family")
        if any(not isinstance(value, str) or not value or value != value.strip() for value in layer_scope["layer_ids"]):
            raise PromptBankError("S2.4 layer IDs must be nonempty trimmed strings")
        scope_text = "layers [" + ", ".join(layer_scope["layer_ids"]) + "]"
    elif layer_scope["kind"] == "all" and layer_scope["layer_ids"] == []:
        family = layer_scope["layer_family"]
        if family is not None and (not isinstance(family, str) or not family or family != family.strip()):
            raise PromptBankError("S2.4 family must be a nonempty trimmed string")
        scope_text = "all layers" if family is None else "all layers in family " + family
    else:
        raise PromptBankError("invalid S2.4 layer scope")
    values = dict(multiplex_id=multiplex_id, scope_text=scope_text, gene=gene)
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values.values()):
        raise PromptBankError("S2.4 text bindings must be nonempty trimmed strings")
    if ranking:
        if top_k is not None:
            raise PromptBankError("S2.4 unresolved ranking requests require top_k=None")
        templates = payload["ranking_requests"][split]
    else:
        if mode == "top_k":
            if type(top_k) is not int or top_k <= 0:
                raise PromptBankError("S2.4 top_k requires a positive integer")
            values["top_k"] = top_k
        elif top_k is not None:
            raise PromptBankError("S2.4 complete requests require top_k=None")
        templates = payload[split][mode][direction]
    return templates[variant_index % len(templates)].format(**values)


def _map_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask bound graph values in all wrapper and map request combinations."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        variants[split] = []
        cores = [(f"{mode}/{direction}/{index}", template)
                 for mode in MODES for direction in DIRECTIONS
                 for index, template in enumerate(requests[split][mode][direction])]
        cores.extend((f"ranking/{index}", template)
                     for index, template in enumerate(requests["ranking_requests"][split]))
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template in cores:
                    core = template.format(**{field: "" for field in (*FIELDS, "top_k")})
                    tokens = _tokens(wrapper.format(request=core))
                    variants[split].append((f"S2.4/{split}/{framing}/{wrapper_index}/{label}",
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_4_rendered_prompts(
    wrappers: Mapping[str, Any], requests: Mapping[str, Any],
    s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
    s23_requests: Mapping[str, Any],
) -> dict[str, Any]:
    """Check full masked questions across the frozen S2.1 through S2.4 splits."""
    validate_s2_prompt_banks(wrappers)
    validate_s2_4_requests(requests)
    validate_s2_edge_requests(s21_requests)
    validate_s2_2_requests(s22_requests)
    validate_s2_3_requests(s23_requests)
    families = {"s2_4": _map_variants(wrappers, requests),
                "s2_3": _neighbor_variants(wrappers, s23_requests),
                "s2_2": _rendered_variants(wrappers, s22_requests, "S2.2"),
                "s2_1": _rendered_variants(wrappers, s21_requests, "S2.1")}
    combined = {split: sum((family[split] for family in families.values()), []) for split in S2_PROMPT_SPLITS}
    vocabulary = sorted({word for rows in combined.values() for _, words, _ in rows for word in words})
    bits = {word: 1 << index for index, word in enumerate(vocabulary)}
    bitsets = {split: [(label, sum(bits[word] for word in words), len(words)) for label, words, _ in rows]
               for split, rows in combined.items()}
    closest, compared = 0.0, 0
    for index, split in enumerate(S2_PROMPT_SPLITS):
        for other in S2_PROMPT_SPLITS[index + 1:]:
            left_ngrams = {gram: label for label, _, grams in combined[split] for gram in grams}
            for label, _, grams in combined[other]:
                shared = grams & left_ngrams.keys()
                if shared:
                    gram = min(shared)
                    raise PromptBankError(f"S2 full questions share a 5-word run: {left_ngrams[gram]} and {label}: {' '.join(gram)}")
            for label, tokens, size in bitsets[split]:
                for other_label, other_tokens, other_size in bitsets[other]:
                    compared += 1
                    common = (tokens & other_tokens).bit_count()
                    overlap = common / (size + other_size - common)
                    closest = max(closest, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(f"S2 full questions overlap: {label} and {other_label}; Jaccard={overlap:.4f}")
    return {**{f"{name}_rendered_questions": {split: len(rows) for split, rows in family.items()}
               for name, family in families.items()},
            "cross_split_pairs_checked": compared, "maximum_cross_split_jaccard": round(closest, 4),
            "shared_cross_split_5grams": 0,
            "masking": "remove_bound_multiplex_scope_gene_and_top_k_values"}
