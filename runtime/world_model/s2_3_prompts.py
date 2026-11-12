"""Load frozen neighbor requests and check wording isolation across S2 families.

The shared framing bank and earlier request banks remain immutable. Selection
of graph facts, directions, and bounds belongs to the caller, not this module.
"""

from collections.abc import Mapping
import json
import re
from string import Formatter
from typing import Any

from .contract import REPOSITORY_ROOT
from .s1_prompt_banks import (
    FRAMINGS,
    HELD_OUT_JACCARD_LIMIT,
    HELD_OUT_SHARED_NGRAM,
    PromptBankError,
    S2_PROMPT_SPLITS,
    _ngrams,
    _tokens,
    validate_s2_edge_requests,
    validate_s2_prompt_banks,
)
from .s2_2_prompts import _rendered_variants, validate_s2_2_requests
from .sources import file_sha256


SCHEMA_VERSION = "mentor-rl-world-model-s2-3-requests-v1"
MODES = ("complete", "top_k")
DIRECTIONS = ("undirected", "in", "out", "both")
FIELDS = ("multiplex_id", "layer_id", "gene")
INCOMING_BINDING = re.compile(
    r"target \{gene\}|\{gene\} (?:as|is) (?:the )?target|"
    r"(?:toward|into|to) \{gene\}|destination \{gene\}|enter \{gene\}|"
    r"(?:terminate|terminating|aimed|arriving) at \{gene\}", re.I,
)
OUTGOING_BINDING = re.compile(
    r"source \{gene\}|\{gene\} (?:as|is) (?:the )?source|"
    r"(?:from|leaving) \{gene\}|originating at \{gene\}|originate at \{gene\}", re.I,
)


def validate_s2_3_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact slots and the requested neighbor selection and roles."""

    if (
        not isinstance(payload, Mapping)
        or list(payload) != ["schema_version", "description", *S2_PROMPT_SPLITS]
        or payload["schema_version"] != SCHEMA_VERSION
        or not isinstance(payload["description"], str)
        or not payload["description"].strip()
    ):
        raise PromptBankError("the S2.3 request schema changed")
    counts = {}
    for split in S2_PROMPT_SPLITS:
        bank = payload[split]
        if not isinstance(bank, Mapping) or list(bank) != list(MODES):
            raise PromptBankError("S2.3 must declare complete and top_k in order")
        counts[split] = {}
        for mode in MODES:
            directions = bank[mode]
            if not isinstance(directions, Mapping) or list(directions) != list(DIRECTIONS):
                raise PromptBankError("S2.3 must declare all four request directions in order")
            counts[split][mode] = {}
            for direction in DIRECTIONS:
                templates = directions[direction]
                if not isinstance(templates, list) or len(templates) < 4:
                    raise PromptBankError("S2.3 requires at least four requests per kind")
                seen = set()
                for template in templates:
                    _validate_template(template, mode, direction)
                    if template in seen:
                        raise PromptBankError("S2.3 requests must be distinct within each kind")
                    seen.add(template)
                counts[split][mode][direction] = len(templates)
    return {"template_counts": counts}


def _validate_template(template: Any, mode: str, direction: str) -> None:
    """Reject changed format slots or a request for the wrong adjacency view."""

    if (
        not isinstance(template, str) or not template
        or template != template.strip() or "\n" in template
    ):
        raise PromptBankError("S2.3 requests must be nonempty trimmed lines")
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise PromptBankError("invalid S2.3 request formatting") from exc
    fields = [field for _, field, _, _ in parsed if field is not None]
    expected = (*FIELDS, "top_k") if mode == "top_k" else FIELDS
    if sorted(fields) != sorted(expected) or any(
        format_spec or conversion for _, _, format_spec, conversion in parsed
    ):
        raise PromptBankError("S2.3 requests must bind exactly the declared fields once")
    if not re.search(r"\bweights?\b", template, re.I):
        raise PromptBankError("S2.3 requests must retain neighbor weights")
    if direction == "undirected":
        if not re.search(r"undirected|unordered|without orientation|unoriented|nonoriented", template, re.I):
            raise PromptBankError("S2.3 undirected requests must preserve unordered adjacency")
        if re.search(r"incoming|outgoing|regulat|target|source \{gene\}", template, re.I):
            raise PromptBankError("S2.3 undirected requests cannot invent directional roles")
    elif direction in ("in", "out"):
        name, opposite, binding = (
            ("incoming", "outgoing", INCOMING_BINDING) if direction == "in"
            else ("outgoing", "incoming", OUTGOING_BINDING)
        )
        if not re.search(rf"\b{name}\b", template, re.I) or not binding.search(template):
            raise PromptBankError("S2.3 directional requests must bind the query's endpoint role")
        if re.search(rf"\b{opposite}\b|undirected|unordered", template, re.I):
            raise PromptBankError("S2.3 one-direction requests cannot include another direction")
    else:
        if not (
            re.search(r"\bincoming\b", template, re.I)
            and re.search(r"\boutgoing\b", template, re.I)
            and re.search(r"separat|split|partition|distinct|independent|distinguish", template, re.I)
        ):
            raise PromptBankError("S2.3 both-direction requests must keep incoming and outgoing separate")
        incoming = re.search(r"incoming\b(.*?)\boutgoing\b", template, re.I)
        outgoing = re.search(r"outgoing\b(.*)", template, re.I)
        if not incoming or not outgoing or not (
            re.search(r"(?:N|neighbor|source) → (?:Q|query)|toward|point to|enter|terminate|target|into|linking to", incoming[1], re.I)
            and re.search(r"(?:Q|query) → (?:N|neighbor|target)|start|from|reached|leav|originate|sourced|away|depart|receiv", outgoing[1], re.I)
        ):
            raise PromptBankError("S2.3 compound requests must preserve both endpoint roles")
    if mode == "complete":
        if not re.search(r"every|\ball\b|complete|full|exhaustive|entire", template, re.I):
            raise PromptBankError("S2.3 complete requests must ask for every neighbor")
        if re.search(r"strongest|highest.weight|\btop\b|\bbounded\b", template, re.I):
            raise PromptBankError("S2.3 complete requests cannot impose a ranked bound")
    else:
        if not re.search(r"strongest|highest.weight|highest.weighted|top .*descending weight|heaviest", template, re.I):
            raise PromptBankError("S2.3 bounded requests must select the strongest neighbors")
        if not re.search(r"fewer|smaller|exhaust|available|shorter|insufficient", template, re.I):
            raise PromptBankError("S2.3 bounded requests must allow fewer neighbors upon exhaustion")
        if direction == "both" and not re.search(
            r"bound independently|limit separately|per direction|each directional|each side|independently to (?:both|the incoming)",
            template, re.I,
        ):
            raise PromptBankError("S2.3 both-direction bounds must apply K separately per direction")
        if re.search(r"overall rank|combined rank|global rank|summed weights|maximum of", template, re.I):
            raise PromptBankError("S2.3 cannot invent a combined directional ranking")


def load_s2_3_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Load a separately pinned neighbor wording bank, leaving prior banks frozen."""

    if not isinstance(pin, Mapping) or list(pin) != ["path", "sha256"]:
        raise PromptBankError("an S2.3 request pin must list path, then sha256")
    path = (REPOSITORY_ROOT / str(pin["path"])).resolve()
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise PromptBankError("the S2.3 request bank does not match its pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_s2_3_requests(payload)
    return payload


def render_s2_3_request(
    payload: Mapping[str, Any], *, split: str, mode: str, direction: str,
    variant_index: int, multiplex_id: str, layer_id: str, gene: str,
    top_k: int | None = None,
) -> str:
    """Bind caller-selected adjacency content to one deterministic wording."""

    if split not in S2_PROMPT_SPLITS or mode not in MODES or direction not in DIRECTIONS:
        raise PromptBankError("unknown S2.3 split, mode, or direction")
    if type(variant_index) is not int or variant_index < 0:
        raise PromptBankError("the S2.3 wording index must be nonnegative")
    values = dict(multiplex_id=multiplex_id, layer_id=layer_id, gene=gene)
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values.values()):
        raise PromptBankError("S2.3 text bindings must be nonempty trimmed strings")
    if mode == "top_k":
        if type(top_k) is not int or top_k <= 0:
            raise PromptBankError("S2.3 top_k requires a positive integer")
        values["top_k"] = str(top_k)
    elif top_k is not None:
        raise PromptBankError("S2.3 complete requests require top_k=None")
    templates = payload[split][mode][direction]
    return templates[variant_index % len(templates)].format(**values)


def _neighbor_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Tokenize all neutral-wrapper/request products with bound values removed."""

    variants = {}
    for split in S2_PROMPT_SPLITS:
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for mode in MODES:
                    for direction in DIRECTIONS:
                        for index, template in enumerate(requests[split][mode][direction]):
                            core = template.format(**{field: "" for field in (*FIELDS, "top_k")})
                            tokens = _tokens(wrapper.format(request=core))
                            variants[split].append((
                                f"S2.3/{split}/{framing}/{wrapper_index}/{mode}/{direction}/{index}",
                                frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM),
                            ))
    return variants


def validate_s2_3_rendered_prompts(
    wrappers: Mapping[str, Any], requests: Mapping[str, Any],
    s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit complete masked questions across all S2.1–S2.3 split pairs."""

    validate_s2_prompt_banks(wrappers)
    validate_s2_3_requests(requests)
    validate_s2_edge_requests(s21_requests)
    validate_s2_2_requests(s22_requests)
    families = {
        "s2_3": _neighbor_variants(wrappers, requests),
        "s2_2": _rendered_variants(wrappers, s22_requests, "S2.2"),
        "s2_1": _rendered_variants(wrappers, s21_requests, "S2.1"),
    }
    combined = {split: sum((family[split] for family in families.values()), []) for split in S2_PROMPT_SPLITS}
    vocabulary = sorted({word for rows in combined.values() for _, words, _ in rows for word in words})
    bits = {word: 1 << index for index, word in enumerate(vocabulary)}
    # Integer bitsets preserve exact token-set Jaccard while avoiding millions
    # of temporary Python sets in the growing cross-family audit.
    bitsets = {
        split: [(label, sum(bits[word] for word in words), len(words)) for label, words, _ in rows]
        for split, rows in combined.items()
    }
    closest = 0.0
    compared = 0
    for index, split in enumerate(S2_PROMPT_SPLITS):
        for other in S2_PROMPT_SPLITS[index + 1:]:
            left_ngrams = {gram: label for label, _, grams in combined[split] for gram in grams}
            for label, _, grams in combined[other]:
                shared = grams & left_ngrams.keys()
                if shared:
                    gram = min(shared)
                    raise PromptBankError(
                        f"S2 full questions share a 5-word run: {left_ngrams[gram]} and {label}: {' '.join(gram)}"
                    )
            for label, tokens, size in bitsets[split]:
                for other_label, other_tokens, other_size in bitsets[other]:
                    compared += 1
                    common = (tokens & other_tokens).bit_count()
                    overlap = common / (size + other_size - common)
                    closest = max(closest, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(
                            f"S2 full questions overlap: {label} and {other_label}; Jaccard={overlap:.4f}"
                        )
    return {
        **{f"{name}_rendered_questions": {split: len(rows) for split, rows in family.items()}
           for name, family in families.items()},
        "cross_split_pairs_checked": compared,
        "maximum_cross_split_jaccard": round(closest, 4),
        "shared_cross_split_5grams": 0,
        "masking": "remove_bound_scope_endpoint_multiplex_layer_gene_and_top_k_values",
    }
