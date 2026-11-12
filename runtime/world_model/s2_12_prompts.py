"""Bind singleton membership and claim tasks without an implicit graph verdict.

The frozen bank separates observed support, literal truth, and explicit
outer membership prerequisites. It preserves every raw JSON input.
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
from .sources import file_sha256


SCHEMA_VERSION = 'mentor-rl-world-model-s2-12-requests-v1'
KINDS = ('membership_card', 'membership_support', 'membership_truth', 'claim_card', 'claim_support', 'truth')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope',
               'invalid_gene_list', 'unresolved_identity')
FIELDS = ('multiplex_id', 'layer_id', 'genes')
TRUTH_FIELDS = ('claim', 'truth_scope', 'truth_layer_ids', 'truth_family', 'statement')
TASK_FIELDS = {
    'membership_card': (*FIELDS, 'require_membership'),
    'membership_support': (*FIELDS, 'require_membership', 'claim'),
    'membership_truth': (*FIELDS, 'require_membership', *TRUTH_FIELDS),
    'claim_card': FIELDS,
    'claim_support': (*FIELDS, 'claim'),
    'truth': (*FIELDS, *TRUTH_FIELDS),
}


def _validate_templates(templates: Any, kind: str) -> None:
    """Require exact literal bindings and a task cue in every frozen request."""
    if not isinstance(templates, list) or len(templates) < 4:
        raise PromptBankError('S2.12 requires at least four requests per kind')
    for template in templates:
        if not isinstance(template, str) or not template or template != template.strip() or '\n' in template:
            raise PromptBankError('S2.12 requests must be nonempty trimmed lines')
        try:
            parsed = list(Formatter().parse(template))
        except ValueError as error:
            raise PromptBankError('invalid S2.12 request formatting') from error
        fields = [field for _, field, _, _ in parsed if field is not None]
        if sorted(fields) != sorted(TASK_FIELDS[kind]) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise PromptBankError('S2.12 requests must bind exactly the declared fields once')
        if not re.search(r'membership|classif|claim|proposition|statement|evidence|observation|assertion', template, re.I):
            raise PromptBankError('S2.12 requests require a membership or evidence task cue')
    if len(set(templates)) != len(templates):
        raise PromptBankError('S2.12 requests must be distinct within each kind')


def validate_s2_12_requests(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check all six task forms and all prerequisite diagnostic requests."""
    if (not isinstance(payload, Mapping)
            or list(payload) != ['schema_version', 'description', *S2_PROMPT_SPLITS, 'diagnostic_requests']
            or payload['schema_version'] != SCHEMA_VERSION
            or not isinstance(payload['description'], str) or not payload['description'].strip()):
        raise PromptBankError('the S2.12 request schema changed')
    diagnostics = payload['diagnostic_requests']
    if not isinstance(diagnostics, Mapping) or list(diagnostics) != list(S2_PROMPT_SPLITS):
        raise PromptBankError('S2.12 diagnostics must declare every split')
    counts, diagnostic_counts = {}, {}
    for split in S2_PROMPT_SPLITS:
        if not isinstance(payload[split], Mapping) or list(payload[split]) != list(KINDS):
            raise PromptBankError('S2.12 must declare all six question kinds in order')
        counts[split] = {}
        for kind in KINDS:
            _validate_templates(payload[split][kind], kind)
            counts[split][kind] = len(payload[split][kind])
        if not isinstance(diagnostics[split], Mapping) or list(diagnostics[split]) != list(DIAGNOSTICS):
            raise PromptBankError('S2.12 must declare all six diagnostic kinds in order')
        diagnostic_counts[split] = {}
        for kind in DIAGNOSTICS:
            _validate_templates(diagnostics[split][kind], 'claim_card')
            diagnostic_counts[split][kind] = len(diagnostics[split][kind])
    return {'template_counts': counts, 'diagnostic_template_counts': diagnostic_counts}


def load_s2_12_requests(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pinned request bank without changing earlier frozen wording."""
    if not isinstance(pin, Mapping) or list(pin) != ['path', 'sha256']:
        raise PromptBankError('an S2.12 request pin must list path, then sha256')
    path = (REPOSITORY_ROOT / str(pin['path'])).resolve()
    if not path.is_file() or file_sha256(path) != pin['sha256']:
        raise PromptBankError('the S2.12 request bank does not match its pin')
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_s2_12_requests(payload)
    return payload


def _json_literal(value: Any) -> str:
    """Retain JSON values exactly and reject Python-only or nonfinite values."""
    if not _json_value(value):
        raise PromptBankError('S2.12 inputs require a JSON literal')
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PromptBankError('S2.12 inputs require a JSON literal') from error


def _bindings(split: str, variant_index: int, values: Mapping[str, Any], kind: str) -> dict[str, str]:
    """Check the selector and preserve every literal field for this task."""
    if split not in S2_PROMPT_SPLITS or type(variant_index) is not int or variant_index < 0:
        raise PromptBankError('unknown S2.12 split or invalid wording index')
    return {field: _json_literal(values[field]) for field in TASK_FIELDS[kind]}


def render_s2_12_request(payload: Mapping[str, Any], *, split: str, variant_index: int,
                        multiplex_id: str, layer_id: str, genes: list[str],
                        question_kind: str = 'membership_card', claim: str | None = None,
                        truth_scope: str | None = None, truth_layer_ids: list[str] | None = None,
                        truth_family: str | None = None, statement: str | None = None,
                        require_membership: str | None = None) -> str:
    """Bind one membership subject and preserve the exact assessed proposition."""
    if question_kind not in KINDS:
        raise PromptBankError('unknown S2.12 question kind')
    values = _bindings(split, variant_index, locals(), question_kind)
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in (multiplex_id, layer_id)):
        raise PromptBankError('ordinary S2.12 requests require explicit multiplex and layer strings')
    if not isinstance(genes, list) or len(genes) != 1 or not isinstance(genes[0], str) or not genes[0]:
        raise PromptBankError('ordinary S2.12 requests require a literal singleton list')
    if require_membership not in (None, 'multiplex', 'layer'):
        raise PromptBankError('S2.12 requires a recognized explicit membership prerequisite')
    if not question_kind.startswith('membership_') and require_membership is not None:
        raise PromptBankError('claim-only S2.12 tasks cannot impose an acquisition prerequisite')
    if question_kind in ('membership_support', 'membership_truth', 'claim_support', 'truth') and (
            not isinstance(claim, str) or not claim.strip()):
        raise PromptBankError('an assessed S2.12 card requires an explicit claim')
    if question_kind in ('membership_truth', 'truth'):
        if (truth_scope not in ('category', 'multiplex', 'all_layers', 'layers', 'family',
                               'future_versions', 'expression')
                or not isinstance(statement, str) or not statement.strip()):
            raise PromptBankError('an S2.12 truth task requires its exact statement and scope')
        if truth_scope == 'layers' and (not isinstance(truth_layer_ids, list) or not truth_layer_ids
                or any(not isinstance(layer, str) or not layer for layer in truth_layer_ids)
                or len(set(truth_layer_ids)) != len(truth_layer_ids)):
            raise PromptBankError('an S2.12 layers truth task requires distinct explicit layers')
        if truth_scope == 'family' and (not isinstance(truth_family, str) or not truth_family.strip()):
            raise PromptBankError('an S2.12 family truth task requires its explicit family')
        if truth_scope != 'layers' and truth_layer_ids is not None:
            raise PromptBankError('S2.12 truth layers belong only to an explicit layers scope')
        if truth_scope != 'family' and truth_family is not None:
            raise PromptBankError('S2.12 truth family belongs only to an explicit family scope')
    templates = payload[split][question_kind]
    return templates[variant_index % len(templates)].format(**values)


def render_s2_12_diagnostic_request(payload: Mapping[str, Any], *, split: str, kind: str,
                                   variant_index: int, multiplex_id: Any, layer_id: Any,
                                   genes: Any) -> str:
    """Retain all three raw JSON fields before any prerequisite resolution."""
    if kind == 'tool_unknown_layer':
        kind = 'unknown_layer'
    if kind == 'tool_unknown_multiplex':
        kind = 'unknown_multiplex'
    if kind not in DIAGNOSTICS:
        raise PromptBankError('unknown S2.12 diagnostic kind')
    values = _bindings(split, variant_index, locals(), 'claim_card')
    templates = payload['diagnostic_requests'][split][kind]
    return templates[variant_index % len(templates)].format(**values)


def _singleton_variants(wrappers: Mapping[str, Any], requests: Mapping[str, Any]):
    """Mask literal singleton inputs, truth selectors, and explicit prerequisites."""
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
                    variants[split].append((f'S2.12/{split}/{framing}/{wrapper_index}/{label}',
                                            frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    return variants


def validate_s2_12_rendered_prompts(wrappers: Mapping[str, Any], requests: Mapping[str, Any],
                                  s21_requests: Mapping[str, Any], s22_requests: Mapping[str, Any],
                                  s23_requests: Mapping[str, Any], s24_requests: Mapping[str, Any],
                                  s25_requests: Mapping[str, Any], s26_requests: Mapping[str, Any],
                                  s27_requests: Mapping[str, Any], s28_requests: Mapping[str, Any],
                                  s29_requests: Mapping[str, Any], s210_requests: Mapping[str, Any],
                                  s211_requests: Mapping[str, Any]) -> dict[str, Any]:
    """Audit all masked full questions across the frozen twelve-family corpus."""
    for validator, bank in ((validate_s2_prompt_banks, wrappers), (validate_s2_12_requests, requests),
                            (validate_s2_edge_requests, s21_requests), (validate_s2_2_requests, s22_requests),
                            (validate_s2_3_requests, s23_requests), (validate_s2_4_requests, s24_requests),
                            (validate_s2_5_requests, s25_requests), (validate_s2_6_requests, s26_requests),
                            (validate_s2_7_requests, s27_requests), (validate_s2_8_requests, s28_requests),
                            (validate_s2_9_requests, s29_requests), (validate_s2_10_requests, s210_requests),
                            (validate_s2_11_requests, s211_requests)):
        validator(bank)
    families = {'s2_12': _singleton_variants(wrappers, requests),
                's2_11': _path_variants(wrappers, s211_requests),
                's2_10': _component_variants(wrappers, s210_requests),
                's2_9': _subgraph_variants(wrappers, s29_requests),
                's2_8': _degree_variants(wrappers, s28_requests), 's2_7': _shared_variants(wrappers, s27_requests),
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
            'masking': 'remove_bound_multiplex_layer_singleton_claim_scope_statement_and_prerequisite'}
