"""Bind frozen membership requests and audit wording across all S2 families.

The caller selects graph facts and scopes before choosing request wording.
The diagnostic renderer retains raw invalid selections for local rejection.
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
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-5-requests-v1'
MODES = ('complete', 'top_k')
DIAGNOSTICS = ('unknown_layer', 'unknown_family', 'missing_scope', 'empty_scope', 'missing_limit', 'invalid_limit')
FIELDS = ('multiplex_id', 'scope_text', 'gene')


def _validate_templates(templates: Any, kind: str) -> None:
    """Check exact bindings and preserve membership rather than adjacency requests."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.5 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.5 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.5 request formatting') from error
        expected = (*FIELDS, 'top_k') if kind in ('top_k', 'invalid_limit') else FIELDS
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(expected) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.5 requests must bind exactly the declared fields once')
        if not re.search(r'layer|membership', template, re.I):
            raise PromptBankError('S2.5 requests must ask for layer membership')
        if re.search(r'incoming|outgoing|strongest|weight|degree|caus|activat|repress', template, re.I):
            raise PromptBankError('S2.5 membership requests cannot select direction, degree, or weight')
        if kind == 'complete' and (not re.search(r'every|\ball\b|complete|full|exhaustive|entire', template, re.I)
                or re.search(r'prefix|bounded|\{top_k\}', template, re.I)):
            raise PromptBankError('S2.5 complete requests must retain all containing layers')
        if kind in ('top_k', 'invalid_limit') and not (
            re.search(r'first|initial|prefix|earliest', template, re.I)
            and re.search(r'manifest', template, re.I)
            and re.search(r'fewer|smaller|shorter|exhaust|none remain|no further', template, re.I)
        ):
            raise PromptBankError('S2.5 bounded requests require a manifest prefix and allow exhaustion')
        if kind == 'missing_limit' and not re.search(r'bounded|prefix|limited|initial', template, re.I):
            raise PromptBankError('S2.5 missing-limit requests must remain bounded')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.5 requests must be distinct within each kind')


def validate_s2_5_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen membership request and diagnostic schema."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.5 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.5 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        bank = payload[split]
        if not isinstance(bank, Mapping) or list(bank) != list(MODES):
            raise PromptBankError('S2.5 must declare complete and top_k in order')
        counts[split] = {}
        for mode in MODES:
            _validate_templates(bank[mode], mode)
            counts[split][mode] = len(bank[mode])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.5 must declare all six diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind], kind)
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_5_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read a separately pinned request bank without changes to earlier banks."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.5 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.5 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_5_requests(payload)
    return payload


def _bindings(split: str, variant_index: int, multiplex_id: Any,
              layer_scope: Any, gene: str, *, diagnostic: bool = False) -> dict:
    """Preserve scope mention order while keeping normalization in the runtime."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.5 split or invalid wording index')
    if diagnostic and layer_scope is None:
        scope_text = 'not supplied'
    else:
        if not isinstance(layer_scope, Mapping) or set(layer_scope) != {'kind', 'layer_ids', 'layer_family'}:
            raise PromptBankError('S2.5 requires a typed layer scope')
        layers, family = layer_scope['layer_ids'], layer_scope['layer_family']
        if layer_scope['kind'] == 'layers':
            if (not isinstance(layers, list) or family is not None or not layers and not diagnostic
                    or any(not isinstance(value, str) or not value or value != value.strip() for value in layers)):
                raise PromptBankError('S2.5 explicit scope requires nonempty layer IDs and no family')
            scope_text = 'layers [' + ', '.join(layers) + ']'
        elif layer_scope['kind'] == 'all' and layers == []:
            if family is not None and (not isinstance(family, str) or not family or family != family.strip()):
                raise PromptBankError('S2.5 family must be a nonempty trimmed string')
            scope_text = 'all layers' if family is None else 'all layers in family ' + family
        else:
            raise PromptBankError('invalid S2.5 layer scope')
    if diagnostic and multiplex_id is None:
        multiplex_id = 'not supplied'
    values = {'multiplex_id': multiplex_id, 'scope_text': scope_text, 'gene': gene}
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values.values()):
        raise PromptBankError('S2.5 text bindings must be nonempty trimmed strings')
    return values


def render_s2_5_request(payload: Mapping[str, Any], *, split: str, mode: str,
                       variant_index: int, multiplex_id: str, layer_scope: Mapping[str, Any],
                       gene: str, top_k: int | None = None) -> str:
    """Bind caller-selected membership content to one deterministic frozen request."""
    values = _bindings(split, variant_index, multiplex_id, layer_scope, gene)
    if mode not in MODES:
        raise PromptBankError('unknown S2.5 selection')
    if mode == 'top_k':
        if type(top_k) is not int or top_k <= 0:
            raise PromptBankError('S2.5 top_k requires a positive integer')
        values['top_k'] = top_k
    elif top_k is not None:
        raise PromptBankError('S2.5 complete requests require top_k=None')
    templates = payload[split][mode]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_5_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                  variant_index: int, multiplex_id: str | None,
                                  layer_scope: Mapping[str, Any] | None, gene: str,
                                  top_k: Any = None) -> str:
    """Retain missing or invalid request values without converting them into calls."""
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.5 diagnostic kind')
    values = _bindings(split, variant_index, multiplex_id, layer_scope, gene, diagnostic=True)
    if kind == 'invalid_limit':
        try:
            values['top_k'] = json.dumps(top_k, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise PromptBankError('S2.5 diagnostic limits require a JSON literal') from error
    elif top_k is not None:
        raise PromptBankError('only invalid_limit binds a diagnostic top_k value')
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _membership_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask caller-selected graph values in all wrapper and request combinations."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        cores = [(f'{mode}/{index}', template) for mode in MODES
                 for index, template in enumerate(requests[split][mode])]
        cores.extend((f'diagnostic/{kind}/{index}', template) for kind in DIAGNOSTICS
                     for index, template in enumerate(requests['diagnostic_requests'][split][kind]))
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template in cores:
                    core = template.format(**{field: '' for field in (*FIELDS, 'top_k')})
                    tokens = _tokens(wrapper.format(request=core))
                    variants[split].append((f'S2.5/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_5_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Check full masked questions across the frozen S2.1 through S2.5 splits."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_5_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests)):
        validator(bank)
    families = {'s2_5': _membership_variants(wrappers, requests), 's2_4': _map_variants(wrappers, s24_requests),
                's2_3': _neighbor_variants(wrappers, s23_requests),
                's2_2': _rendered_variants(wrappers, s22_requests, 'S2.2'),
                's2_1': _rendered_variants(wrappers, s21_requests, 'S2.1')}
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
                    raise PromptBankError(f'S2 full questions share a 5-word run: {left_ngrams[gram]} and {label}: {" ".join(gram)}')
            for label, tokens, size in bitsets[split]:
                for other_label, other_tokens, other_size in bitsets[other]:
                    compared += 1
                    common = (tokens & other_tokens).bit_count()
                    overlap = common / (size + other_size - common)
                    closest = max(closest, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(f'S2 full questions overlap: {label} and {other_label}; Jaccard={overlap:.4f}')
    return {**{f'{name}_rendered_questions': {split: len(rows) for split, rows in family.items()}
               for name, family in families.items()}, 'cross_split_pairs_checked': compared,
            'maximum_cross_split_jaccard': round(closest, 4), 'shared_cross_split_5grams': 0,
            'masking': 'remove_bound_multiplex_scope_gene_and_top_k_values'}
