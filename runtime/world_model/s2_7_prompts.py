"""Bind two neighbor roles and audit their wording against earlier S2 banks.

The request keeps both roles, including repeated genes. Direction selects the
wording; the renderer never resolves identities or changes the role order.
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
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-7-requests-v1'
DIRECTION_PAIRS = {
    'undirected': ('undirected', 'undirected'),
    'in_in': ('in', 'in'), 'out_out': ('out', 'out'),
    'out_in': ('out', 'in'), 'in_out': ('in', 'out'),
    'both_both': ('both', 'both'), 'both_in': ('both', 'in'),
    'in_both': ('in', 'both'), 'both_out': ('both', 'out'),
    'out_both': ('out', 'both'),
}
KINDS = tuple(DIRECTION_PAIRS)
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_scope', 'missing_genes',
               'invalid_pair', 'invalid_directions', 'ranking_required')
FIELDS = ('multiplex_id', 'layer_id', 'gene_a', 'gene_b')
DIAGNOSTIC_FIELDS = ('multiplex_id', 'layer_id', 'genes_json', 'directions_json')


def _validate_templates(templates: Any, *, diagnostic: bool = False,
                        ranking: bool = False) -> None:
    """Require exact bindings and complete, unranked ordinary intersections."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.7 requires at least four requests per kind')
    expected = DIAGNOSTIC_FIELDS if diagnostic else FIELDS
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.7 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.7 request formatting') from error
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(expected) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.7 requests must bind exactly the declared fields once')
        if not re.search(r'intersect|shared|common|overlap|both full collections', template, re.I):
            raise PromptBankError('S2.7 requests must ask for a neighbor intersection')
        if not diagnostic and not re.search(r'\ball\b|every|complete|entire|exhaustive|full|whole', template, re.I):
            raise PromptBankError('ordinary S2.7 requests must ask for the complete shared set')
        if not ranking and re.search(r'top.?k|rank|strongest|highest|heaviest|weight|cursor', template, re.I):
            raise PromptBankError('ordinary S2.7 intersections have no ranking, weight, or paging selector')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.7 requests must be distinct within each kind')


def validate_s2_7_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check ten ordered direction pairs and seven raw-input diagnostic kinds."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.7 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.7 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        if not isinstance(payload[split], Mapping) or list(payload[split]) != list(KINDS):
            raise PromptBankError('S2.7 must declare all ten direction pairs in order')
        counts[split] = {}
        for kind in KINDS:
            _validate_templates(payload[split][kind])
            counts[split][kind] = len(payload[split][kind])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.7 must declare all seven diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind], diagnostic=True, ranking=kind == 'ranking_required')
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_7_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned bank without changes to earlier frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.7 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.7 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_7_requests(payload)
    return payload


def _bindings(split: str, variant_index: int, multiplex_id: Any,
              layer_id: Any, *, diagnostic: bool = False) -> dict:
    """Check the split, wording index, and explicit scope bindings."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.7 split or invalid wording index')
    values = {'multiplex_id': multiplex_id, 'layer_id': layer_id}
    for field, value in values.items():
        if diagnostic and value is None:
            values[field] = 'null'
        elif not isinstance(value, str) or not value or not diagnostic and value != value.strip():
            raise PromptBankError('S2.7 scope bindings must be nonempty strings')
    return values


def _json_literal(value: Any) -> str:
    """Keep JSON values exact and reject Python-only or nonfinite values."""
    if not _json_value(value):
        raise PromptBankError('S2.7 inputs require a JSON literal')
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.7 inputs require a JSON literal') from error


def render_s2_7_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                       multiplex_id: str, layer_id: str, genes: list[str],
                       directions: list[str]) -> str:
    """Bind exactly two raw gene roles without identity or role deduplication."""
    values = _bindings(split, variant_index, multiplex_id, layer_id)
    if not isinstance(genes, list) or len(genes) != 2 or any(not isinstance(gene, str) or not gene for gene in genes):
        raise PromptBankError('ordinary S2.7 requests require exactly two gene strings')
    if not isinstance(directions, list) or tuple(directions) not in DIRECTION_PAIRS.values():
        raise PromptBankError('ordinary S2.7 requests require two compatible directions')
    kind = next(kind for kind, pair in DIRECTION_PAIRS.items() if pair == tuple(directions))
    values.update(gene_a=_json_literal(genes[0]), gene_b=_json_literal(genes[1]))
    templates = payload[split][kind]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_7_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                  variant_index: int, multiplex_id: str | None,
                                  layer_id: str | None, genes: Any, directions: Any) -> str:
    """Preserve each raw JSON input, including invalid shapes and duplicate roles."""
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.7 diagnostic kind')
    values = _bindings(split, variant_index, multiplex_id, layer_id, diagnostic=True)
    values.update(genes_json=_json_literal(genes), directions_json=_json_literal(directions))
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _shared_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask role values and scopes in every full shared-neighbor question."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        cores = [(f'{kind}/{index}', template, FIELDS) for kind in KINDS
                 for index, template in enumerate(requests[split][kind])]
        cores.extend((f'diagnostic/{kind}/{index}', template, DIAGNOSTIC_FIELDS) for kind in DIAGNOSTICS
                     for index, template in enumerate(requests['diagnostic_requests'][split][kind]))
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template, fields in cores:
                    core = template.format(**{field: '' for field in fields})
                    tokens = _tokens(wrapper.format(request=core))
                    variants[split].append((f'S2.7/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_7_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any],
                                  s25_requests: Mapping[str, Any], s26_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit all masked questions across the frozen S2.1 through S2.7 splits."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_7_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests),
                            (validate_s2_5_requests, s25_requests), (validate_s2_6_requests, s26_requests)):
        validator(bank)
    families = {'s2_7': _shared_variants(wrappers, requests),
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
            'masking': 'remove_bound_multiplex_layer_gene_roles_and_raw_diagnostic_values'}
