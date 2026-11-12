"""Bind degree requests and audit their wording against earlier frozen S2 banks.

The renderer keeps the literal identity, scope, direction, and policy inputs.
It does not choose a statistic, resolve an identity, or derive a graph fact.
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
from .s2_4_prompts import _map_variants, validate_s2_4_requests
from .s2_5_prompts import _membership_variants, validate_s2_5_requests
from .s2_6_prompts import _classification_variants, _json_value, validate_s2_6_requests
from .s2_7_prompts import _shared_variants, validate_s2_7_requests
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-8-requests-v1'
KINDS = ('degree_only', 'degree_and_hub')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_gene', 'missing_scope',
               'invalid_scope', 'invalid_direction', 'invalid_policy')
FIELDS = ('multiplex_id', 'gene', 'layer_scope', 'direction', 'request_kind', 'hub_policy')
DIRECTIONS = ('incident', 'in', 'out', 'undirected')


def _validate_templates(templates: Any) -> None:
    """Require each literal input exactly once and a degree question."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.8 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.8 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.8 request formatting') from error
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(FIELDS) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.8 requests must bind exactly the declared fields once')
        if not re.search(r'\bdegree\b|neighbor count|adjacency count', template, re.I):
            raise PromptBankError('S2.8 requests must ask for a distinct-neighbor degree')
        if re.search(r'weight|caus|activat|repress|top.?k|cursor|strongest', template, re.I):
            raise PromptBankError('S2.8 degree requests cannot add unsupported graph claims or selectors')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.8 requests must be distinct within each kind')


def validate_s2_8_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check the two requested statistics and seven prerequisite diagnostic kinds."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.8 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.8 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        if not isinstance(payload[split], Mapping) or list(payload[split]) != list(KINDS):
            raise PromptBankError('S2.8 must declare both statistic kinds in order')
        counts[split] = {}
        for kind in KINDS:
            _validate_templates(payload[split][kind])
            counts[split][kind] = len(payload[split][kind])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.8 must declare all seven diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind])
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_8_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned degree bank without a change to earlier frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.8 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.8 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_8_requests(payload)
    return payload


def _json_literal(value: Any) -> str:
    """Keep JSON values exact and reject Python-only or nonfinite values."""
    if not _json_value(value):
        raise PromptBankError('S2.8 inputs require a JSON literal')
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.8 inputs require a JSON literal') from error


def _bindings(split: str, variant_index: int, values: Mapping[str, Any]) -> dict[str, str]:
    """Check the wording selector and preserve every supplied JSON value."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.8 split or invalid wording index')
    return {field: _json_literal(values[field]) for field in FIELDS}


def render_s2_8_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                       multiplex_id: str, gene: str, layer_scope: Mapping[str, Any],
                       direction: str, request_kind: str, hub_policy: dict | None) -> str:
    """Bind a count or hub question without changes to its raw identity or scope."""
    values = _bindings(split, variant_index, locals())
    if (not isinstance(multiplex_id, str) or not multiplex_id or multiplex_id != multiplex_id.strip()
            or not isinstance(gene, str) or not gene):
        raise PromptBankError('ordinary S2.8 requests require a gene string and explicit multiplex')
    if (not isinstance(layer_scope, Mapping) or set(layer_scope) != {'kind', 'layer_ids', 'layer_family'}
            or layer_scope['kind'] not in ('layers', 'all') or not isinstance(layer_scope['layer_ids'], list)
            or any(not isinstance(layer, str) or not layer for layer in layer_scope['layer_ids'])
            or len(set(layer_scope['layer_ids'])) != len(layer_scope['layer_ids'])
            or layer_scope['kind'] == 'layers' and (not layer_scope['layer_ids'] or layer_scope['layer_family'] is not None)
            or layer_scope['kind'] == 'all' and (layer_scope['layer_ids']
                or layer_scope['layer_family'] is not None and not isinstance(layer_scope['layer_family'], str))):
        raise PromptBankError('ordinary S2.8 requests require a typed layer scope')
    if direction not in DIRECTIONS or request_kind not in KINDS:
        raise PromptBankError('ordinary S2.8 requests require a defined direction and statistic')
    if ((request_kind == 'degree_only' and hub_policy is not None)
            or request_kind == 'degree_and_hub' and (not isinstance(hub_policy, dict)
                or set(hub_policy) != {'policy_id', 'threshold_percentile'}
                or hub_policy['policy_id'] != 'degree_midrank_v1'
                or type(hub_policy['threshold_percentile']) is not int
                or not 0 <= hub_policy['threshold_percentile'] <= 100)):
        raise PromptBankError('ordinary S2.8 requests require the declared hub policy or null')
    templates = payload[split][request_kind]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_8_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                  variant_index: int, multiplex_id: Any, gene: Any,
                                  layer_scope: Any, direction: Any, request_kind: Any,
                                  hub_policy: Any) -> str:
    """Preserve each literal prerequisite input, including malformed JSON shapes."""
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.8 diagnostic kind')
    values = _bindings(split, variant_index, locals())
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _degree_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask bound identities, selectors, directions, and policies in full questions."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        cores = [(f'{kind}/{index}', template) for kind in KINDS
                 for index, template in enumerate(requests[split][kind])]
        cores.extend((f'diagnostic/{kind}/{index}', template) for kind in DIAGNOSTICS
                     for index, template in enumerate(requests['diagnostic_requests'][split][kind]))
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template in cores:
                    tokens = _tokens(wrapper.format(request=template.format(**dict.fromkeys(FIELDS, ''))))
                    variants[split].append((f'S2.8/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_8_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any],
                                  s25_requests: Mapping[str, Any], s26_requests: Mapping[str, Any],
                                  s27_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every masked full question across the frozen S2.1 through S2.8 splits."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_8_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests),
                            (validate_s2_5_requests, s25_requests), (validate_s2_6_requests, s26_requests),
                            (validate_s2_7_requests, s27_requests)):
        validator(bank)
    families = {'s2_8': _degree_variants(wrappers, requests), 's2_7': _shared_variants(wrappers, s27_requests),
                's2_6': _classification_variants(wrappers, s26_requests),
                's2_5': _membership_variants(wrappers, s25_requests),
                's2_4': _map_variants(wrappers, s24_requests), 's2_3': _neighbor_variants(wrappers, s23_requests),
                's2_2': _rendered_variants(wrappers, s22_requests, 'S2.2'),
                's2_1': _rendered_variants(wrappers, s21_requests, 'S2.1')}
    combined = {split: sum((family[split] for family in families.values()), []) for split in S2_PROMPT_SPLITS}
    vocabulary = sorted({word for rows in combined.values() for _, words, _ in rows for word in words})
    bits = {word: 1 << index for index, word in enumerate(vocabulary)}
    bitsets = {}
    for split, rows in combined.items():
        unique = {}
        for label, words, _ in rows:
            unique.setdefault(words, (label, sum(bits[word] for word in words), len(words)))
        bitsets[split] = list(unique.values())
    closest, compared = 0.0, 0
    for index, split in enumerate(S2_PROMPT_SPLITS):
        for other in S2_PROMPT_SPLITS[index + 1:]:
            compared += len(combined[split]) * len(combined[other])
            left_ngrams = {gram: label for label, _, grams in combined[split] for gram in grams}
            for label, _, grams in combined[other]:
                shared = grams & left_ngrams.keys()
                if shared:
                    gram = min(shared)
                    raise PromptBankError(f'S2 full questions share a 5-word run: {left_ngrams[gram]} and {label}: {" ".join(gram)}')
            for label, tokens, size in bitsets[split]:
                for other_label, other_tokens, other_size in bitsets[other]:
                    common = (tokens & other_tokens).bit_count()
                    overlap = common / (size + other_size - common)
                    closest = max(closest, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(f'S2 full questions overlap: {label} and {other_label}; Jaccard={overlap:.4f}')
    return {**{f'{name}_rendered_questions': {split: len(rows) for split, rows in family.items()}
               for name, family in families.items()}, 'cross_split_pairs_checked': compared,
            'maximum_cross_split_jaccard': round(closest, 4), 'shared_cross_split_5grams': 0,
            'masking': 'remove_bound_multiplex_gene_scope_direction_statistic_and_policy_values'}
