"""Bind finite gene-list requests and audit wording across six S2 families.

The renderer preserves the raw list. The runtime resolves identities and removes
duplicates only after the user request is visible.
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
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-6-requests-v1'
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_scope', 'missing_genes', 'invalid_scalar', 'invalid_list')
FIELDS = ('multiplex_id', 'layer_id', 'genes_json')


def _validate_templates(templates: Any) -> None:
    """Require one explicit finite list and one selected layer per request."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.6 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.6 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.6 request formatting') from error
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(FIELDS) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.6 requests must bind exactly the declared fields once')
        if not re.search(r'layer|membership', template, re.I) or not re.search(
                r'each|every|\ball\b|complete|entire|exhaustive|full|whole|throughout', template, re.I):
            raise PromptBankError('S2.6 requests must classify the complete supplied gene list')
        if re.search(r'incoming|outgoing|strongest|weight|degree|caus|activat|repress|top.?k|cursor', template, re.I):
            raise PromptBankError('S2.6 classification has no direction, ranking, or paging selector')
        if re.search(r'(?:all|every)\s+(?:the\s+)?nodes?\b|inventory', template, re.I):
            raise PromptBankError('S2.6 classifies supplied genes rather than the entire layer inventory')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.6 requests must be distinct within each kind')


def validate_s2_6_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check the frozen ordinary request bank and six diagnostic kinds."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.6 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.6 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        _validate_templates(payload[split])
        counts[split] = len(payload[split])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.6 must declare all six diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind])
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_6_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned S2.6 bank without changes to earlier frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.6 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.6 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_6_requests(payload)
    return payload


def _json_value(value: Any) -> bool:
    """Reject Python containers that JSON would silently change into new shapes."""
    if value is None or type(value) in (str, int, float, bool):
        return True
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    return isinstance(value, dict) and all(type(key) is str and _json_value(item) for key, item in value.items())


def _bindings(split: str, variant_index: int, multiplex_id: Any,
              layer_id: Any, genes: Any, *, diagnostic: bool = False) -> dict:
    """Preserve JSON list order, duplicate entries, and invalid diagnostic values."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.6 split or invalid wording index')
    if not diagnostic and (not isinstance(genes, list) or any(not isinstance(gene, str) for gene in genes)):
        raise PromptBankError('ordinary S2.6 requests require a gene list')
    values = {'multiplex_id': multiplex_id, 'layer_id': layer_id}
    for field, value in values.items():
        if diagnostic and value is None:
            values[field] = 'null'
        elif not isinstance(value, str) or not value or not diagnostic and value != value.strip():
            raise PromptBankError('S2.6 scope bindings must be nonempty strings')
    if not _json_value(genes):
        raise PromptBankError('S2.6 genes require a JSON literal')
    try:
        values['genes_json'] = json.dumps(genes, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.6 genes require a JSON literal') from error
    return values


def render_s2_6_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                       multiplex_id: str, layer_id: str, genes: list[str]) -> str:
    """Bind a caller-selected finite list without changing order or duplicates."""
    values = _bindings(split, variant_index, multiplex_id, layer_id, genes)
    templates = payload[split]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_6_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                  variant_index: int, multiplex_id: str | None,
                                  layer_id: str | None, genes: Any) -> str:
    """Keep invalid or missing inputs visible without constructing graph calls."""
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.6 diagnostic kind')
    values = _bindings(split, variant_index, multiplex_id, layer_id, genes, diagnostic=True)
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _classification_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask bound gene lists and scopes in every full question combination."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        cores = [(f'complete/{index}', template) for index, template in enumerate(requests[split])]
        cores.extend((f'diagnostic/{kind}/{index}', template) for kind in DIAGNOSTICS
                     for index, template in enumerate(requests['diagnostic_requests'][split][kind]))
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template in cores:
                    core = template.format(**{field: '' for field in FIELDS})
                    tokens = _tokens(wrapper.format(request=core))
                    variants[split].append((f'S2.6/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_6_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any],
                                  s25_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit all masked questions across the frozen S2.1 through S2.6 splits."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_6_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests),
                            (validate_s2_5_requests, s25_requests)):
        validator(bank)
    families = {'s2_6': _classification_variants(wrappers, requests),
                's2_5': _membership_variants(wrappers, s25_requests), 's2_4': _map_variants(wrappers, s24_requests),
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
            'masking': 'remove_bound_multiplex_layer_scope_gene_list_mode_and_limit_values'}
