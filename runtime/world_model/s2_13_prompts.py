"""Bind family capability, pair support, acquisition, and explicit truth requests.

Every request preserves its literal inputs. Neutral framing stays in the
unchanged wrapper bank, separate from biological and graph evidence tasks.
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
from .s2_9_prompts import _subgraph_variants, validate_s2_9_requests
from .s2_10_prompts import _component_variants, validate_s2_10_requests
from .s2_11_prompts import _path_variants, validate_s2_11_requests
from .s2_12_prompts import _singleton_variants, validate_s2_12_requests
from .s2_13_evidence import CLAIM_VOCABULARY as BIOLOGICAL_CLAIMS, FAMILIES
from .s2_graph_claims import CLAIM_VOCABULARY as GRAPH_CLAIMS
from .sources import GENE_ID_PATTERN, file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-13-requests-v1'
KINDS = ('family_card', 'family_support', 'edge_card', 'edge_support', 'claim_card',
         'claim_support', 'biological_truth', 'graph_truth')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope',
               'invalid_pair', 'incompatible_direction', 'unresolved_identity')
FIELDS = ('genes', 'multiplex_id', 'layer_id', 'direction')
PAIR_FIELDS = (*FIELDS, 'subjects')
TASK_FIELDS = {'family_card': ('layer_family',), 'family_support': ('layer_family', 'claim'),
    'edge_card': PAIR_FIELDS, 'edge_support': (*PAIR_FIELDS, 'claim'),
    'claim_card': PAIR_FIELDS, 'claim_support': (*PAIR_FIELDS, 'claim'),
    'biological_truth': (*PAIR_FIELDS, 'claim', 'statement'),
    'graph_truth': (*PAIR_FIELDS, 'claim', 'graph_claim', 'graph_truth_scope', 'statement')}


def _validate_templates(templates: Any, fields: tuple[str, ...]) -> None:
    """Require each explicit request value exactly once in every frozen template."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.13 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.13 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.13 request formatting') from error
        slots = [field for _, field, _, _ in parsed if field is not None]
        if sorted(slots) != sorted(fields) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.13 requests must bind exactly the declared fields once')
        if not re.search(r'family|claim|concept|evidence|support|proposition|statement|assertion|scope|pair|prerequisite|source|relation', template, re.I):
            raise PromptBankError('S2.13 requests require an evidence task cue')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.13 requests must be distinct within each kind')


def validate_s2_13_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check eight question kinds and all seven raw prerequisite diagnostic kinds."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.13 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.13 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        if not isinstance(payload[split], Mapping) or list(payload[split]) != list(KINDS):
            raise PromptBankError('S2.13 must declare all eight question kinds in order')
        counts[split] = {}
        for kind in KINDS:
            _validate_templates(payload[split][kind], TASK_FIELDS[kind])
            counts[split][kind] = len(payload[split][kind])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.13 must declare all seven diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind], FIELDS)
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_13_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned request bank without changing older frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.13 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.13 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_13_requests(payload)
    return payload


def _json_literal(value: Any) -> str:
    """Preserve JSON values while rejecting Python-only and nonfinite values."""
    if not _json_value(value):
        raise PromptBankError('S2.13 inputs require a JSON literal')
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.13 inputs require a JSON literal') from error


def _bindings(split: str, variant_index: int, values: Mapping[str, Any], fields) -> dict[str, str]:
    """Check the wording selector and preserve each declared literal field."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.13 split or invalid wording index')
    return {field: _json_literal(values[field]) for field in fields}


def render_s2_13_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                        question_kind='edge_card', genes=None, multiplex_id=None, layer_id=None,
                        direction=None, subjects=None, layer_family=None, claim=None,
                        graph_claim='no_edge_recorded_in_scope', graph_truth_scope='scope',
                        statement=None) -> str:
    """Bind only the requested family, pair assessment, acquisition, or exact truth."""
    if question_kind not in KINDS:
        raise PromptBankError('unknown S2.13 question kind')
    subjects = [] if subjects is None else subjects
    values = _bindings(split, variant_index, locals(), TASK_FIELDS[question_kind])
    if question_kind.startswith('family_'):
        if layer_family not in FAMILIES:
            raise PromptBankError('S2.13 family requests require a closed S1 family')
        if subjects or any(value is not None for value in (genes, multiplex_id, layer_id, direction)):
            raise PromptBankError('a family capability request has no pair or graph scope')
    else:
        if any(not isinstance(value, str) or not value or value != value.strip()
               for value in (multiplex_id, layer_id)):
            raise PromptBankError('ordinary S2.13 requests require explicit multiplex and layer strings')
        if (not isinstance(genes, list) or len(genes) != 2 or any(
                not isinstance(gene, str) or not gene for gene in genes) or genes[0] == genes[1]):
            raise PromptBankError('ordinary S2.13 requests require a literal distinct pair')
        if direction not in (None, 'undirected', 'source_to_target'):
            raise PromptBankError('S2.13 requires a native interpretation direction')
        if not isinstance(subjects, list) or any(not isinstance(subject, Mapping)
                or set(subject) != {'multiplex_id', 'layer_id', 'gene_ids'}
                or any(not isinstance(subject[key], str) or not subject[key]
                       for key in ('multiplex_id', 'layer_id'))
                or not isinstance(subject['gene_ids'], list) or len(subject['gene_ids']) != 2
                or subject['gene_ids'][0] == subject['gene_ids'][1]
                or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)
                       for gene in subject['gene_ids']) for subject in subjects):
            raise PromptBankError('ordered S2.13 subjects require exact edge selectors')
    if question_kind in ('family_support', 'edge_support', 'claim_support', 'biological_truth') and claim not in BIOLOGICAL_CLAIMS:
        raise PromptBankError('an assessed S2.13 biological card requires a canonical concept')
    if question_kind == 'graph_truth' and (claim is not None and claim not in BIOLOGICAL_CLAIMS
            or graph_claim not in GRAPH_CLAIMS or graph_truth_scope != 'scope'):
        raise PromptBankError('S2.13 graph truth requires separate canonical vocabularies and scope')
    if question_kind.endswith('truth') and (not isinstance(statement, str) or not statement.strip()):
        raise PromptBankError('an S2.13 truth task requires its exact literal statement')
    templates = payload[split][question_kind]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_13_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                   variant_index: int, genes: Any, multiplex_id: Any,
                                   layer_id: Any, direction: Any) -> str:
    """Preserve all four raw values before identity or graph prerequisite resolution."""
    kind = kind.removeprefix('tool_')
    if kind in ('ambiguous_identity', 'invalid_identity'):
        kind = 'unresolved_identity'
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.13 diagnostic kind')
    values = _bindings(split, variant_index, locals(), FIELDS)
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _specificity_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask every bound family, pair, subject order, concept, and exact proposition."""
    variants = {}
    for split in S2_PROMPT_SPLITS:
        cores = [(f'{kind}/{index}', template, TASK_FIELDS[kind]) for kind in KINDS
                 for index, template in enumerate(requests[split][kind])]
        cores.extend((f'diagnostic/{kind}/{index}', template, FIELDS) for kind in DIAGNOSTICS
                     for index, template in enumerate(requests['diagnostic_requests'][split][kind]))
        variants[split] = []
        for framing in FRAMINGS:
            for wrapper_index, wrapper in enumerate(wrappers[split][framing]):
                for label, template, fields in cores:
                    tokens = _tokens(wrapper.format(request=template.format(**dict.fromkeys(fields, ''))))
                    variants[split].append((f'S2.13/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_13_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  *earlier_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit masked full questions across all thirteen frozen S2 families."""
    if len(earlier_requests) != 12:
        raise PromptBankError('S2.13 requires all twelve earlier request banks')
    validators = (validate_s2_edge_requests, validate_s2_2_requests, validate_s2_3_requests,
        validate_s2_4_requests, validate_s2_5_requests, validate_s2_6_requests,
        validate_s2_7_requests, validate_s2_8_requests, validate_s2_9_requests,
        validate_s2_10_requests, validate_s2_11_requests, validate_s2_12_requests)
    validate_s2_prompt_banks(wrappers)
    validate_s2_13_requests(requests)
    for validator, bank in zip(validators, earlier_requests):
        validator(bank)
    renderers = (_rendered_variants, _rendered_variants, _neighbor_variants,
        _map_variants, _membership_variants, _classification_variants, _shared_variants,
        _degree_variants, _subgraph_variants, _component_variants, _path_variants, _singleton_variants)
    families = {'s2_13': _specificity_variants(wrappers, requests)}
    for number, (renderer, bank) in enumerate(zip(renderers, earlier_requests), 1):
        families[f's2_{number}'] = (renderer(wrappers, bank, f'S2.{number}')
            if number in (1, 2) else renderer(wrappers, bank))
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
            'masking': 'remove_bound_family_pair_scope_subject_order_biological_and_graph_claim_and_statement'}
