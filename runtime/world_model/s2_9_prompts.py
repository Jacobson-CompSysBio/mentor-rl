"""Bind induced-subgraph requests and audit their wording against earlier frozen S2 banks.

The renderer keeps the literal inputs, scope, selection, and membership requirement.
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
from .s2_8_prompts import _degree_variants, validate_s2_8_requests
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-9-requests-v1'
KINDS = ('complete', 'top_k')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope',
               'invalid_list', 'invalid_selection', 'invalid_cursor', 'ranking_required', 'unresolved_identity')
FIELDS = ('multiplex_id', 'layer_id', 'genes', 'mode', 'top_k', 'cursor', 'require_all_in_layer')


def _validate_templates(templates: Any) -> None:
    """Require each literal input exactly once and an induced-subgraph question."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.9 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.9 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.9 request formatting') from error
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(FIELDS) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.9 requests must bind exactly the declared fields once')
        if not re.search(r'\bsubgraph\b|projected graph|graph projection|induced graph|graph (?:is )?induced', template, re.I):
            raise PromptBankError('S2.9 requests must ask for an induced native graph')
        if re.search(r'caus|activat|repress|strongest|average|rescale', template, re.I):
            raise PromptBankError('S2.9 subgraph requests cannot add unsupported graph claims or weight rules')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.9 requests must be distinct within each kind')


def validate_s2_9_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check the two edge selections and nine prerequisite diagnostic kinds."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.9 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.9 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        if not isinstance(payload[split], Mapping) or list(payload[split]) != list(KINDS):
            raise PromptBankError('S2.9 must declare both edge selections in order')
        counts[split] = {}
        for kind in KINDS:
            _validate_templates(payload[split][kind])
            counts[split][kind] = len(payload[split][kind])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.9 must declare all nine diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind])
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_9_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned subgraph bank without a change to earlier frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.9 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.9 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_9_requests(payload)
    return payload


def _json_literal(value: Any) -> str:
    """Keep JSON values exact and reject Python-only or nonfinite values."""
    if not _json_value(value):
        raise PromptBankError('S2.9 inputs require a JSON literal')
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.9 inputs require a JSON literal') from error


def _bindings(split: str, variant_index: int, values: Mapping[str, Any]) -> dict[str, str]:
    """Check the wording selector and preserve every supplied JSON value."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.9 split or invalid wording index')
    return {field: _json_literal(values[field]) for field in FIELDS}


def render_s2_9_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                       multiplex_id: str, layer_id: str, genes: list[str], mode: str,
                       top_k: int | None, cursor: str | None, require_all_in_layer: bool) -> str:
    """Bind an edge projection or preview without changes to literal input order."""
    values = _bindings(split, variant_index, locals())
    if any(not isinstance(value, str) or not value or value != value.strip() for value in (multiplex_id, layer_id)):
        raise PromptBankError('ordinary S2.9 requests require explicit multiplex and layer strings')
    if not isinstance(genes, list) or any(not isinstance(gene, str) or not gene for gene in genes):
        raise PromptBankError('ordinary S2.9 requests require a literal gene list')
    if (mode not in KINDS or mode == 'complete' and top_k is not None
            or mode == 'top_k' and (type(top_k) is not int or top_k < 1)):
        raise PromptBankError('ordinary S2.9 requests require complete or a positive canonical prefix')
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise PromptBankError('ordinary S2.9 requests require a nonempty cursor or null')
    if type(require_all_in_layer) is not bool:
        raise PromptBankError('ordinary S2.9 membership requirements must be explicit booleans')
    templates = payload[split][mode]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_9_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                  variant_index: int, multiplex_id: Any, layer_id: Any,
                                  genes: Any, mode: Any, top_k: Any, cursor: Any,
                                  require_all_in_layer: Any) -> str:
    """Preserve literal missing or malformed values before prerequisite resolution."""
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.9 diagnostic kind')
    values = _bindings(split, variant_index, locals())
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _subgraph_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask bound inputs, scope, edge selection, cursor, and membership requirement."""
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
                    variants[split].append((f'S2.9/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_9_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any],
                                  s25_requests: Mapping[str, Any], s26_requests: Mapping[str, Any],
                                  s27_requests: Mapping[str, Any], s28_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every masked full question across the frozen S2.1 through S2.9 splits."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_9_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests),
                            (validate_s2_5_requests, s25_requests), (validate_s2_6_requests, s26_requests),
                            (validate_s2_7_requests, s27_requests), (validate_s2_8_requests, s28_requests)):
        validator(bank)
    families = {'s2_9': _subgraph_variants(wrappers, requests), 's2_8': _degree_variants(wrappers, s28_requests), 's2_7': _shared_variants(wrappers, s27_requests),
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
            'masking': 'remove_bound_multiplex_layer_genes_edge_selection_cursor_and_membership_requirement'}
