"""Load frozen S2.2 requests and audit complete questions across S2 splits.

The S2.1 wording and implementation pins stay frozen. This module reuses their
lexical checks and neutral framing bank while adding explicit multiplex scans.
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
    S2_EDGE_REQUEST_FIELDS,
    S2_ORDERED_EDGE_BINDING,
    S2_PROMPT_SPLITS,
    _ngrams,
    _tokens,
    validate_s2_edge_requests,
    validate_s2_prompt_banks,
)
from .sources import file_sha256


SCHEMA_VERSION = "mentor-rl-world-model-s2-2-requests-v1"
MODES = ("complete", "top_k", "exists")
EDGE_QUERIES = ("any_direction", "source_to_target", "both_directions")
FIELDS = ("multiplex_id", "first", "second", "scope_text")


def validate_s2_2_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact slots, retrieval meaning, and directional role bindings."""

    if (
        not isinstance(payload, Mapping)
        or list(payload) != ["schema_version", "description", *S2_PROMPT_SPLITS]
        or payload["schema_version"] != SCHEMA_VERSION
        or not isinstance(payload["description"], str)
        or not payload["description"].strip()
    ):
        raise PromptBankError("the S2.2 request schema changed")
    counts = {}
    for split in S2_PROMPT_SPLITS:
        bank = payload[split]
        if not isinstance(bank, Mapping) or list(bank) != list(MODES):
            raise PromptBankError(f"S2.2 {split} must declare complete, top_k, and exists")
        counts[split] = {}
        for mode in MODES:
            queries = bank[mode]
            if not isinstance(queries, Mapping) or list(queries) != list(EDGE_QUERIES):
                raise PromptBankError("S2.2 must declare every edge predicate in order")
            counts[split][mode] = {}
            for edge_query in EDGE_QUERIES:
                templates = queries[edge_query]
                if not isinstance(templates, list) or len(templates) < 4:
                    raise PromptBankError("S2.2 needs at least four requests per kind")
                seen = set()
                for template in templates:
                    _validate_template(template, mode, edge_query)
                    if template in seen:
                        raise PromptBankError("S2.2 requests must be distinct within each kind")
                    seen.add(template)
                counts[split][mode][edge_query] = len(templates)
    return {"template_counts": counts}


def _validate_template(template: Any, mode: str, edge_query: str) -> None:
    """Reject format or semantic cues that would change the requested operation."""

    if (
        not isinstance(template, str) or not template
        or template != template.strip() or "\n" in template
    ):
        raise PromptBankError("S2.2 requests must be nonempty trimmed lines")
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise PromptBankError("invalid S2.2 request formatting") from exc
    fields = [field for _, field, _, _ in parsed if field is not None]
    expected = (*FIELDS, "top_k") if mode == "top_k" else FIELDS
    if sorted(fields) != sorted(expected) or any(
        format_spec or conversion for _, _, format_spec, conversion in parsed
    ):
        raise PromptBankError("S2.2 requests must bind exactly the declared fields once")
    if template.index("{first}") > template.index("{second}"):
        raise PromptBankError("S2.2 endpoint role order must remain first then second")
    if edge_query == "source_to_target" and not S2_ORDERED_EDGE_BINDING.search(template):
        raise PromptBankError("S2.2 source-to-target requests must bind ordered roles")
    if edge_query == "both_directions" and not (
        re.search(r"\bboth\b|reciprocal|opposite", template, re.I)
        and re.search(r"\b(?:same|one|single) layer\b", template, re.I)
    ):
        raise PromptBankError("S2.2 reciprocity must require both orientations in one layer")
    if edge_query == "any_direction" and not (
        re.search(r"undirected|unordered|unoriented|nonoriented", template, re.I)
        and re.search(r"\b(?:directed|ordered|arrows?|arcs?)\b", template, re.I)
        and re.search(r"either|at least one|one or both", template, re.I)
    ):
        raise PromptBankError("S2.2 any-direction requests must allow native undirected or directed support")
    if mode == "exists":
        # The {first} slot is an endpoint, not a first-matching-layer request.
        outside = template.replace("{first}", "")
        if re.search(r"weights?|exact (?:count|total)|manifest|first|earliest", outside, re.I):
            raise PromptBankError("S2.2 existence requests must stay Boolean and unordered")
        if not re.search(r"whether|\bdoes\b|\bis\b|\bcan\b", template, re.I):
            raise PromptBankError("S2.2 existence requests must ask a Boolean question")
    else:
        if not re.search(r"observed|observations|witness", template, re.I) or "weight" not in template.casefold():
            raise PromptBankError("S2.2 collection requests must retain observed source weights")
        if mode == "complete" and not (
            re.search(r"every|\ball\b|full|entire|exhaustive", template, re.I)
            and re.search(r"exact|precise", template, re.I)
            and re.search(r"count|total|number|tally", template, re.I)
        ):
            raise PromptBankError("S2.2 complete requests must ask for all matches and an exact count")
        if mode == "top_k" and not (
            re.search(r"manifest", template, re.I)
            and re.search(r"first|earliest|prefix|initial", template, re.I)
        ):
            raise PromptBankError("S2.2 bounded requests must select the manifest prefix")
        if mode == "top_k" and re.search(
            r"every (?:supporting|matching|qualifying) layer|"
            r"all (?:supporting|matching|qualifying) layers|"
            r"(?:exhaustive|complete) (?:list|listing|scan)|exact (?:count|total)",
            template, re.I,
        ):
            raise PromptBankError("S2.2 bounded requests cannot promise exhaustive retrieval")


def load_s2_2_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Load independently pinned S2.2 wording without altering S2.1 artifacts."""

    if not isinstance(pin, Mapping) or list(pin) != ["path", "sha256"]:
        raise PromptBankError("an S2.2 request pin must list path, then sha256")
    path = (REPOSITORY_ROOT / str(pin["path"])).resolve()
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise PromptBankError("the S2.2 request bank does not match its pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_s2_2_requests(payload)
    return payload


def render_s2_2_request(
    payload: Mapping[str, Any], *, split: str, mode: str, edge_query: str,
    variant_index: int, multiplex_id: str, first: str, second: str,
    scope_text: str, top_k: int | None = None,
) -> str:
    """Render a preselected multiplex query without choosing content or outcomes."""

    if split not in S2_PROMPT_SPLITS or mode not in MODES or edge_query not in EDGE_QUERIES:
        raise PromptBankError("unknown S2.2 split, mode, or edge predicate")
    if type(variant_index) is not int or variant_index < 0:
        raise PromptBankError("the S2.2 wording index must be nonnegative")
    values = dict(multiplex_id=multiplex_id, first=first, second=second, scope_text=scope_text)
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values.values()):
        raise PromptBankError("S2.2 text bindings must be nonempty trimmed strings")
    if mode == "top_k":
        if type(top_k) is not int or top_k <= 0:
            raise PromptBankError("S2.2 top_k requires a positive integer")
        values["top_k"] = str(top_k)
    elif top_k is not None:
        raise PromptBankError("S2.2 complete and exists requests require top_k=None")
    templates = payload[split][mode][edge_query]
    return templates[variant_index % len(templates)].format(**values)


def _rendered_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any], family: str):
    """Tokenize every wrapper/core combination after removing bound values."""

    variants = {}
    for split in S2_PROMPT_SPLITS:
        if family == "S2.2":
            cores = [
                (f"{mode}/{query}/{index}", template)
                for mode in MODES for query in EDGE_QUERIES
                for index, template in enumerate(requests[split][mode][query])
            ]
            fields = (*FIELDS, "top_k")
        else:
            cores = [
                (f"{representation}/{index}", template)
                for representation, templates in requests[split].items()
                for index, template in enumerate(templates)
            ]
            fields = S2_EDGE_REQUEST_FIELDS
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template in cores:
                    text = wrapper.format(request=template.format(**{field: "" for field in fields}))
                    tokens = _tokens(text)
                    variants[split].append((
                        f"{family}/{split}/{framing}/{wrapper_index}/{label}",
                        frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM),
                    ))
    return variants


def validate_s2_2_rendered_prompts(
    wrappers: Mapping[str, Any], requests: Mapping[str, Any],
    s21_requests: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit full questions within S2.2 and against frozen S2.1 across splits."""

    validate_s2_prompt_banks(wrappers)
    validate_s2_2_requests(requests)
    validate_s2_edge_requests(s21_requests)
    current = _rendered_variants(wrappers, requests, "S2.2")
    previous = _rendered_variants(wrappers, s21_requests, "S2.1")
    combined = {split: current[split] + previous[split] for split in S2_PROMPT_SPLITS}
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
            for label, tokens, _ in combined[split]:
                for other_label, other_tokens, _ in combined[other]:
                    compared += 1
                    common = len(tokens & other_tokens)
                    overlap = common / (len(tokens) + len(other_tokens) - common)
                    closest = max(closest, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(
                            f"S2 full questions overlap: {label} and {other_label}; Jaccard={overlap:.4f}"
                        )
    return {
        "s2_2_rendered_questions": {split: len(rows) for split, rows in current.items()},
        "s2_1_rendered_questions": {split: len(rows) for split, rows in previous.items()},
        "cross_split_pairs_checked": compared,
        "maximum_cross_split_jaccard": round(closest, 4),
        "shared_cross_split_5grams": 0,
        "masking": "remove_bound_scope_endpoint_multiplex_and_top_k_values",
    }
