"""Build scoped layer-evidence cards without promoting family tags to pair facts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_2_identity import get_symbols
from .s2_3_corpus import _provenance
from .s2_11_evidence import _registry_key
from .s2_12_corpus import _identity_output, _identity_record
from .s2_12_sources import GRAPH_VERSION
from .s2_13_evidence import CLAIM_VOCABULARY, FAMILIES, FAMILY_CARDS, LayerProof, collect, family_result
from .s2_13_sampling import PairFact, fact_from_pair
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure, _pair_group
from .s2_multiplex import EvidenceError, MultiplexRuntime
from .s2_subgraphs import SubgraphIndex
from .sources import GENE_ID_PATTERN, file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_13_v1.json')
VARIANTS = ('lookup', 'context', 'duplicate_context', 'reverse_context', 'reversed_request',
    'family_context', 'wrong_layer_context', 'same_family_context', 'wrong_pair_context',
    'stale', 'old_view_context', 'pen_contrast_context', 'two_context_context', 'reciprocal_context',
    'route_context', 'symbol_context', 'symbol_lookup', 'symbol_case_lookup', 'id_context',
    'id_lookup', 'null_symbol_context', 'terminal', 'recovered', 'exhausted',
    'symbol_terminal', 'symbol_recovered', 'symbol_exhausted',
    'partial_terminal', 'partial_recovered', 'partial_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'tool_unknown_layer', 'tool_unknown_multiplex',
    'missing_scope', 'missing_genes', 'invalid_pair', 'incompatible_direction',
    'unresolved_identity', 'ambiguous_identity', 'invalid_identity')
LABELS = ('directed', 'weight', 'reverse_exists', 'registry_states', 'positive_layers')
TASK_FIELDS = ('question_kind', 'claim', 'subjects', 'statement', 'graph_claim', 'graph_truth_scope')


def load_build_contract(path=CONTRACT_PATH):
    """Load the isolated overlay without changing any earlier source or contract."""
    from .s2_13_prompts import load_s2_13_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.13']
            or contract['training_ready'] is not False or contract['production_export_allowed'] is not False):
        raise ValueError('unsupported S2.13 build contract')
    contract['_requests'] = load_s2_13_requests(contract['request_bank'])
    for key in ('source_manifest', 'previous_source_manifest'):
        pin = contract[key]
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'the {key} differs from its pin')
    return contract


def _arguments(view, fact, layer_id=None, gene_ids=None):
    return {'multiplex_id': view.multiplex_id, 'layer_id': layer_id or fact.layer_id,
            'gene_ids': list(fact.gene_ids if gene_ids is None else gene_ids)}


def _active_context(view):
    """Expose qualified interpretation metadata, with no unobserved pair facts."""
    runtime, layers = MultiplexRuntime(view), []
    columns = {'HumanNetV3:string_ppi': ['from', 'to', 'weight'],
        'HumanNetV3:coexpression': ['from', 'to', 'weight'],
        'TFs:brain': ['node1', 'node2', 'edge_weight'], 'TFs:cd4+t_cells': ['TF', 'target', 'score']}
    for layer_id in view.layer_order:
        family = runtime.family_of(layer_id)
        layer = {'layer_id': layer_id, 'layer_family': family, 'directed': view.layers[layer_id].directed,
            'sign_returned': False, 'expression_measurements_returned': False, 'weight_is_probability': False}
        if view.graph_version == GRAPH_VERSION:
            layer['selected_source_columns'] = columns.get(layer_id, ['from', 'to', 'weight'])
            if family in ('bulk_pen', 'scpen'):
                parts = layer_id.split(':')
                layer.update(edge_weighting='r2' if family == 'bulk_pen' else 'importance',
                    network_mode='undirected', representative_expansion=True, source_weight_cutoff=.05,
                    predictor_response_roles_returned=False,
                    context={'organ': parts[1], 'region': parts[2],
                             'cell_type': parts[3] if family == 'scpen' else None})
        layers.append(layer)
    return {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version, 'layers': layers,
        'family_conventions': {family: {'evidence_type': FAMILY_CARDS[family]['evidence_type'],
            'supported_claims': list(FAMILY_CARDS[family]['supported_claims'])} for family in FAMILIES},
        'interpretation_policy': 'A family describes capability, not an observation for an arbitrary pair. '
            'Match the pair, view, layer, and native orientation. Weights preserve their source meaning. '
            'Bulk and single-cell PEN weights do not share a demonstrated comparison scale. '
            'No supplied edge output establishes causality, regulatory sign, expression abundance, or unobserved contexts.'}


def truth_statement(view, fact, claim=None, truth_kind='graph',
                    graph_claim='no_edge_recorded_in_scope', graph_truth_scope='scope'):
    """Bind the exact proposition to the pair and its requested source scope."""
    pair = list(fact.gene_ids if view.layers[fact.layer_id].directed else sorted(fact.gene_ids))
    scope = f'{fact.layer_id} in {view.multiplex_id} at graph version {view.graph_version}'
    if truth_kind == 'graph' and graph_claim == 'no_edge_recorded_in_scope' and graph_truth_scope == 'scope':
        return f'{scope} records no native edge for the pair {canonical_json(pair)}.'
    if truth_kind == 'biological' and claim == 'causal_effect':
        return f'The two genes {canonical_json(pair)} have a causal effect on one another in the setting of {fact.layer_id}.'
    if truth_kind == 'biological' and claim == 'holds_in_other_contexts':
        return f'The association of {canonical_json(pair)} in {fact.layer_id} also holds in additional unexamined contexts.'
    raise ValueError('the generated truth request requires a reviewed exact proposition')


def validate_task(view, fact, question_kind='edge_card', claim=None, subjects=None,
                  statement=None, graph_claim=None, graph_truth_scope=None):
    """Keep support, required acquisition, and exact truth in separate branches."""
    from .s2_13_prompts import KINDS
    if question_kind not in KINDS or question_kind.startswith('family_'):
        raise ValueError('a pair task requires a supported pair question kind')
    assessed = question_kind in ('edge_support', 'claim_support', 'biological_truth', 'graph_truth')
    if assessed:
        claim = claim or ('causal_effect' if question_kind == 'biological_truth'
            else next(iter(FAMILY_CARDS[fact.layer_family]['supported_claims']), 'causal_effect'))
        if claim not in CLAIM_VOCABULARY:
            raise ValueError('the biological claim is outside the unchanged S1 vocabulary')
    elif claim is not None:
        raise ValueError('a list-only task cannot assess a single claim')
    subjects = [] if subjects is None else deepcopy(subjects)
    if not isinstance(subjects, list):
        raise ValueError('subject selectors require an ordered list')
    from .s2_13_evidence import pair_subject
    for selector in subjects:
        pair_subject(view, selector)
    if question_kind == 'graph_truth':
        graph_claim, graph_truth_scope = graph_claim or 'no_edge_recorded_in_scope', graph_truth_scope or 'scope'
    elif graph_claim is not None or graph_truth_scope is not None:
        raise ValueError('only a graph truth request can select the graph vocabulary')
    if question_kind.endswith('truth'):
        expected = truth_statement(view, fact, claim, 'graph' if question_kind == 'graph_truth' else 'biological',
                                   graph_claim, graph_truth_scope)
        statement = expected if statement is None else statement
        if statement != expected:
            raise ValueError('the literal statement differs from its exact proposition')
    elif statement is not None:
        raise ValueError('a support request cannot silently assess statement truth')
    return {'question_kind': question_kind, 'claim': claim, 'subjects': subjects,
            'statement': statement, 'graph_claim': graph_claim, 'graph_truth_scope': graph_truth_scope}


def _other_layer(view, fact, same_family=False, pen=False):
    runtime = MultiplexRuntime(view)
    choices = [layer for layer in view.layer_order if layer != fact.layer_id
        and (not same_family or runtime.family_of(layer) == fact.layer_family)
        and (not pen or {runtime.family_of(layer), fact.layer_family} == {'bulk_pen', 'scpen'})]
    if not choices:
        raise VariantUnavailable('no qualified second layer fits this context')
    return choices[0]


def _variant_subjects(view, fact, variant):
    if variant in ('pen_contrast_context', 'two_context_context'):
        other = _other_layer(view, fact, pen=variant == 'pen_contrast_context')
        if variant == 'pen_contrast_context' and not set(fact.gene_ids) <= (view.layers[other].nodes & view.layers[fact.layer_id].nodes):
            raise VariantUnavailable('the controlled PEN comparison requires both endpoints in both layers')
        return [_arguments(view, fact, other), _arguments(view, fact)]
    if variant == 'reciprocal_context':
        if not fact.directed or fact.category != 'recorded' or not fact.reverse_exists:
            raise VariantUnavailable('the selected query has no pair of native reciprocal records')
        return [_arguments(view, fact), _arguments(view, fact, gene_ids=fact.gene_ids[::-1])]
    return []


def _context(view, fact, variant, row_id, index, registry, contract, old_view, seed):
    """Supply actual original observations for one declared semantic contrast."""
    observations, serial = [], 0

    def edge(selected_view, source):
        nonlocal serial
        serial += 1
        observations.append(_observation('get_edge', source,
            selected_view.get_edge(**source, evidence_id=f'ev_{row_id}_{serial}')))

    source = _arguments(view, fact)
    if variant in ('context', 'duplicate_context', 'reverse_context'):
        if variant == 'reverse_context':
            source['gene_ids'].reverse()
        edge(view, source)
        if variant == 'duplicate_context':
            observations.append(deepcopy(observations[0]))
    elif variant in ('wrong_layer_context', 'same_family_context') or variant.startswith('partial_'):
        source['layer_id'] = _other_layer(view, fact, same_family=variant == 'same_family_context')
        edge(view, source)
    elif variant == 'wrong_pair_context':
        query = frozenset(fact.gene_ids)
        pair = next((pair for pair in view.layers[fact.layer_id].edges if pair[0] != pair[1]
            and frozenset(pair) != query and _pair_group(pair, seed)[1] == fact.split), None)
        if pair is None:
            raise VariantUnavailable('no different same-split native pair is available')
        source['gene_ids'] = list(pair)
        edge(view, source)
    elif variant in ('stale', 'old_view_context'):
        if old_view is None or variant == 'stale' and fact.layer_id not in old_view.layers:
            raise VariantUnavailable('no qualified earlier representation of this layer is available')
        source = {'multiplex_id': old_view.multiplex_id,
            'layer_id': fact.layer_id if fact.layer_id in old_view.layers else old_view.layer_order[0],
            'gene_ids': list(fact.gene_ids)}
        edge(old_view, source)
    elif variant == 'pen_contrast_context':
        edge(view, _variant_subjects(view, fact, variant)[0])
    elif variant in ('two_context_context', 'reciprocal_context'):
        for selected in _variant_subjects(view, fact, variant):
            edge(view, selected)
    elif variant == 'route_context':
        if fact.layer_family != 'ppi' or fact.category != 'no_edge':
            raise VariantUnavailable('this route contrast needs distinct PPI members without a direct edge')
        first, last = fact.gene_ids
        left = {gene for gene, _ in index.native_index.neighbors(fact.layer_id, first, 'undirected')}
        right = {gene for gene, _ in index.native_index.neighbors(fact.layer_id, last, 'undirected')}
        choices = sorted((left & right) - set(fact.gene_ids))
        if not choices:
            raise VariantUnavailable('the selected pair has no observed two-edge route')
        for pair in ((first, choices[0]), (choices[0], last)):
            edge(view, _arguments(view, fact, gene_ids=pair))
    if variant in ('symbol_context', 'symbol_terminal', 'symbol_recovered', 'symbol_exhausted',
                   'id_context', 'null_symbol_context'):
        if variant.startswith('symbol'):
            try:
                values = get_symbols(fact.gene_ids, registry)
            except ValueError as error:
                raise VariantUnavailable(str(error)) from error
            operation, field = 'lookup_human_gene_symbol', 'gene_symbol'
        else:
            if ('graph_only' in fact.registry_states
                    or variant == 'null_symbol_context' and 'null_symbol' not in fact.registry_states):
                raise VariantUnavailable('the ID context requires its authoritative registry identity state')
            values = [gene + '.4' for gene in fact.gene_ids]
            operation, field = 'lookup_human_ensembl_gene_id', 'gene_id'
        for raw in values:
            serial += 1
            response = _identity_output(registry, contract['registry_index'], operation, raw, f'ev_{row_id}_{serial}')
            observations.append(_identity_record(operation, {field: raw}, response))
    return observations, serial


def _identities(observations):
    return [item for position, item in enumerate(observations)
            if item['operation'].startswith('lookup_human_') and item not in observations[:position]]


def _plan(view, request, observations, task):
    """Resolve literal inputs first, then plan from visible evidence alone."""
    identities, genes = _identities(observations), []
    for raw in request['genes']:
        if GENE_ID_PATTERN.fullmatch(raw):
            genes.append(raw)
            continue
        operation = 'lookup_human_ensembl_gene_id' if raw.strip().lower().startswith('ensg') else 'lookup_human_gene_symbol'
        field = 'gene_id' if operation.endswith('gene_id') else 'gene_symbol'
        matching = []
        for item in identities:
            if item['operation'] != operation:
                continue
            if item['status'] != 'resolved':
                if item['arguments'] == {field: raw}:
                    matching.append(item)
                continue
            key = _registry_key(operation, item['arguments'], item['result'])
            try:
                requested = _registry_key(operation, {field: raw}, item['result'])
            except EvidenceError:
                continue
            if key == requested:
                matching.append(item)
        if not matching:
            return LayerProof(None, {field: raw}, operation, tuple(identities), _provenance(identities)), identities
        chosen = matching[0]
        if chosen['status'] != 'resolved':
            return LayerProof(None, None, None, tuple(identities), _provenance(identities),
                              chosen['status'], 'UNRESOLVED_IDENTITY'), identities
        genes.append(chosen['result']['gene_id'])
    if len(set(genes)) != 2:
        return LayerProof(None, None, None, tuple(identities), _provenance(identities),
                          'invalid_input', 'INVALID_PAIR'), identities
    source = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'], 'gene_ids': genes}
    proof = collect(view, source, observations, acquisition_requested=task['question_kind'].startswith('edge_'),
        claim=task['claim'], subjects=task['subjects'], graph_direction=request['direction'],
        truth_kind='graph' if task['question_kind'] == 'graph_truth' else
                   'biological' if task['question_kind'] == 'biological_truth' else None,
        graph_claim=task['graph_claim'] or 'no_edge_recorded_in_scope', graph_truth_scope=task['graph_truth_scope'] or 'scope')
    used = []
    for item in observations:
        if (item in identities or item in proof.observations) and item not in used:
            used.append(item)
    return proof, used


def _blocked(request, issue, used, rejected=None):
    return {'issue_code': issue, 'request': deepcopy(request), 'rejected_arguments': deepcopy(rejected),
            'partial_results': list(used)}


def _answer(request, framing, proof, used, failure=None, observations=()):
    if failure:
        operation, source, response = failure
        retained = []
        for item in observations:
            if item not in retained:
                retained.append(item)
        return {'status': 'error', 'provenance': _provenance(retained),
            'result': {'failed_tool': operation, 'error_code': response['error_code'], 'arguments': deepcopy(source),
                       'partial_results': retained, 'coverage': None}, 'continuation_decision': 'defer',
            'blocked_reason': 'The required pair acquisition failed after its visible allowance ended.',
            'summary': 'Retain the original successful partials. The failed call supplies no biological evidence.'}
    if proof.issue_code:
        return {'status': proof.status, 'provenance': _provenance(used),
            'result': _blocked(request, proof.issue_code, used), 'continuation_decision': 'defer',
            'blocked_reason': 'Resolve the identity or provide adequate evidence for the exact biological truth request.',
            'summary': 'An unsupported claim is not a false biological statement. Preserve the assessed evidence and defer.'}
    if proof.result is None or proof.next_arguments is not None:
        raise ValueError('a required pair result remains unresolved')
    missing = proof.status == 'not_found'
    return {'status': proof.status, 'provenance': _provenance(used), 'result': proof.result,
        'continuation_decision': 'defer' if missing else 'continue' if framing == 'step' else 'stop',
        'blocked_reason': 'A required endpoint is outside the multiplex, so the edge predicate is unavailable.' if missing else None,
        'summary': 'The native pair evidence retains its exact scope and source meaning.'}


def _reason(answer, framing):
    if answer['continuation_decision'] == 'defer':
        return ('Retain the original successful observations. A missing identity or multiplex prerequisite is not a checked edge negative. '
                'A backend failure adds no evidence. Biological truth needs adequate evidence; unsupported does not mean false. Defer for the actual blocker.')
    parts = ['Use the qualified family and native representation. A family tag alone is not evidence for an arbitrary pair.']
    if 'edge_exists' in answer['result']:
        result = answer['result']
        parts.append(f"The ordinary result has edge_exists {result['edge_exists']} and weight {result['weight']}; preserve its memberships and endpoint order.")
    for card in answer['result']['claim_cards']:
        parts.append(f"For {card['subject']}, {card['claim_vocabulary']} supports {card['supported_claims']}.")
        if 'claim' in card:
            parts.append(f"The requested concept {card['claim']} has claim_supported {card['claim_supported']}.")
        if 'statement_true' in card:
            parts.append(f"The exact scoped graph statement has statement_true {card['statement_true']}; this is not a biological truth verdict.")
    parts.append('Do not compare different weight scales or infer causality, regulatory sign, abundance, feedback, or unobserved contexts.')
    parts.append('This workflow step is complete, so continue.' if framing == 'step' else 'The standalone task is complete, so stop.')
    return ' '.join(parts)


def _raw_genes(fact, variant, registry):
    if variant.startswith('symbol'):
        try:
            result = list(get_symbols(fact.gene_ids, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
        return [' ' + value.lower() + ' ' for value in result] if variant == 'symbol_case_lookup' else result
    if variant in ('id_context', 'id_lookup', 'null_symbol_context'):
        if 'graph_only' in fact.registry_states:
            raise VariantUnavailable('a versioned input requires its authoritative registry mapping')
        return [gene + '.4' for gene in fact.gene_ids]
    if variant == 'reversed_request':
        if fact.directed:
            raise VariantUnavailable('a directed reversed pair is a different source fact')
        return list(reversed(fact.gene_ids))
    return list(fact.gene_ids)


def _metadata(view, fact, split, framing, variant, request, task, wrapper, template, contract, seed):
    family = fact.layer_family if fact is not None else task.pop('layer_family')
    return {'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.13', 'split': split, 'selection_seed': seed,
        'split_unit': fact.split_unit if fact is not None else digest(['S2.13.family', family]),
        'graph_version': view.graph_version, 'layer_id': fact.layer_id if fact is not None else None,
        'layer_family': family, 'gene_ids': list(fact.gene_ids) if fact is not None else [],
        'category': fact.category if fact is not None else 'family_capability',
        **{key: getattr(fact, key) if fact is not None else [] if key in ('registry_states', 'positive_layers') else None for key in LABELS},
        'framing': framing, 'variant': variant, **task, 'request': request,
        'wrapper_index': wrapper, 'request_template_index': template,
        'supervision': 'current_turn_assistant_only', 'training_ready': False}


def build_row(view, fact, framing, variant, variant_index, bank, contract, *, index, registry,
              old_view=None, seed=20261028, question_kind='edge_card', claim=None, subjects=None,
              statement=None, graph_claim=None, graph_truth_scope=None):
    """Construct one pair trace with visible identity, acquisition, and retry steps."""
    from .s2_13_prompts import render_s2_13_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported layer-specificity contrast or framing')
    subjects = _variant_subjects(view, fact, variant) if subjects is None else subjects
    task = validate_task(view, fact, question_kind, claim, subjects, statement, graph_claim, graph_truth_scope)
    request = {'genes': _raw_genes(fact, variant, registry), 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'direction': None}
    core = render_s2_13_request(contract['_requests'], split=fact.split, variant_index=variant_index, **request, **task)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's213_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, task, user])[:32]
    observations, serial = _context(view, fact, variant, row_id, index, registry, contract, old_view, seed)
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view), 'observations': observations})},
        {'role': 'user', 'content': user}]
    fault = next((kind for kind in ('terminal', 'recovered', 'exhausted') if variant.endswith(kind)), None)
    failure, injected = None, False
    while True:
        proof, used = _plan(view, request, observations, task)
        if proof.next_arguments is None:
            break
        operation, source = proof.next_tool, deepcopy(proof.next_arguments)

        def execute():
            nonlocal serial
            serial += 1
            identity = f'ev_{row_id}_{serial}'
            if operation.startswith('lookup_human_'):
                return _identity_output(registry, contract['registry_index'], operation, next(iter(source.values())), identity)
            if operation != 'get_edge':
                raise ValueError('a missing required pair observation must use the unchanged edge tool')
            return view.get_edge(**source, evidence_id=identity)

        def append(response, thought):
            messages.extend([{'role': 'assistant', 'thinking': thought,
                'tool_call': {'name': operation, 'arguments': deepcopy(source)}},
                {'role': 'tool', 'name': operation, 'content': response}])

        if fault and operation == 'get_edge' and not injected:
            injected = True
            response = _failure(None if fault == 'terminal' else 1)
            append(response, 'The required native pair observation is missing. Request only its exact scope; supplied contrasts do not replace it.')
            if fault != 'terminal':
                response = _failure(0) if fault == 'exhausted' else execute()
                append(response, 'The visible failure permits one retry. Repeat the identical tool and arguments, then stop after recovery or exhaustion.')
            if response['status'] == 'error':
                failure = operation, source, response
                break
        else:
            response = execute()
            append(response, 'Resolve the literal identity when needed. Acquire only the required missing pair observation, with its native endpoint roles.')
        observations.append(_identity_record(operation, source, response) if operation.startswith('lookup_human_')
                            else _observation(operation, source, response))
    if fault and not injected:
        raise VariantUnavailable('this task needs no required missing edge acquisition to fail')
    answer = _answer(request, framing, proof, used, failure, observations)
    messages.append({'role': 'assistant', 'thinking': _reason(answer, framing), 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'],
        'metadata': _metadata(view, fact, fact.split, framing, variant, request, task, wrapper,
            variant_index % len(contract['_requests'][fact.split][question_kind]), contract, seed)}


def build_family_row(view, layer_family, split, framing, variant_index, bank, contract, *,
                     question_kind='family_card', claim=None, seed=20261028):
    """Teach all six S1 family meanings without graph calls or pair observations."""
    from .s2_13_prompts import render_s2_13_request
    if layer_family not in FAMILIES or split not in SPLITS or framing not in FRAMINGS:
        raise ValueError('unsupported family capability row')
    if question_kind not in ('family_card', 'family_support') or (question_kind == 'family_card' and claim is not None):
        raise ValueError('family capability requires its list or single-concept form')
    if question_kind == 'family_support' and claim not in CLAIM_VOCABULARY:
        raise ValueError('a family support check requires a closed concept')
    core = render_s2_13_request(contract['_requests'], split=split, variant_index=variant_index,
                               question_kind=question_kind, layer_family=layer_family, claim=claim)
    user, wrapper = render_s2_prompt(bank, split=split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's213_' + digest([contract['contract_version'], layer_family, split, framing, question_kind, claim, user])[:32]
    answer = {'status': 'resolved', 'provenance': {'registry': [], 'graph': []},
        'result': family_result(layer_family, claim=claim),
        'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
        'summary': 'This conceptual family lookup supplies no pair observation and requires no graph acquisition.'}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view), 'observations': []})},
        {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking': _reason(answer, framing), 'content': answer}]
    task = {'question_kind': question_kind, 'claim': claim, 'subjects': [], 'statement': None,
            'graph_claim': None, 'graph_truth_scope': None, 'layer_family': layer_family}
    request = dict.fromkeys(('genes', 'multiplex_id', 'layer_id', 'direction'))
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'],
        'metadata': _metadata(view, None, split, framing, 'taxonomy', request, task, wrapper,
            variant_index % len(contract['_requests'][split][question_kind]), contract, seed)}


def _diagnostic_request(view, fact, kind):
    request = {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'direction': None}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'tool_unknown_multiplex': 'UNKNOWN_MULTIPLEX',
        'missing_scope': 'MISSING_SCOPE', 'missing_genes': 'MISSING_GENES', 'invalid_pair': 'INVALID_PAIR',
        'incompatible_direction': 'DIRECTION_UNAVAILABLE', 'unresolved_identity': 'UNRESOLVED_IDENTITY',
        'ambiguous_identity': 'UNRESOLVED_IDENTITY', 'invalid_identity': 'UNRESOLVED_IDENTITY'}
    if kind not in codes:
        raise ValueError('unsupported layer-specificity diagnostic')
    if kind.endswith('unknown_layer'):
        request['layer_id'] = 'bulkPEN:brain:unregistered-context'
    elif kind.endswith('unknown_multiplex'):
        request['multiplex_id'] = 'unknown_specificity_multiplex'
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_pair':
        request['genes'] = [fact.gene_ids[0], fact.gene_ids[0]]
    elif kind == 'incompatible_direction':
        request['direction'] = 'undirected' if fact.directed else 'source_to_target'
    elif kind.endswith('identity'):
        request['genes'][0] = {'unresolved_identity': 'HER2', 'ambiguous_identity': 'SHOX',
                               'invalid_identity': 'ENSGinvalid'}[kind]
    status = ('ambiguous' if kind == 'ambiguous_identity' else 'not_found' if kind in
        ('unknown_layer', 'unknown_multiplex', 'tool_unknown_layer', 'tool_unknown_multiplex', 'unresolved_identity')
        else 'invalid_input')
    return request, status, codes[kind]


def _diagnostic_wording(kind):
    return 'unresolved_identity' if kind in ('ambiguous_identity', 'invalid_identity') else kind.removeprefix('tool_')


def _diagnostic_steps(view, fact, kind, row_id, registry, contract):
    request, status, code = _diagnostic_request(view, fact, kind)
    active, used, rejected, steps = _active_context(view), [], None, []
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    if kind.startswith('tool_'):
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
                    'gene_ids': request['genes']}
        response = view.get_edge(**rejected, evidence_id='unused_rejected_specificity_scope')
        steps = [{'role': 'assistant', 'thinking': 'Resolve the explicit graph scope without substituting a registered layer.',
                  'tool_call': {'name': 'get_edge', 'arguments': deepcopy(rejected)}},
                 {'role': 'tool', 'name': 'get_edge', 'content': response}]
    elif kind.endswith('identity'):
        operation = 'lookup_human_ensembl_gene_id' if kind == 'invalid_identity' else 'lookup_human_gene_symbol'
        source = {'gene_id' if kind == 'invalid_identity' else 'gene_symbol': request['genes'][0]}
        response = _identity_output(registry, contract['registry_index'], operation, request['genes'][0], f'ev_{row_id}_identity')
        if response['status'] != status:
            raise ValueError('the declared identity blocker disagrees with the current complete registry')
        used.append(_identity_record(operation, source, response))
        steps = [{'role': 'assistant', 'thinking': 'Resolve the literal identity under S1 rules. Retain every candidate without selecting by graph outcome.',
                  'tool_call': {'name': operation, 'arguments': source}},
                 {'role': 'tool', 'name': operation, 'content': response}]
    answer = {'status': status, 'provenance': _provenance(used), 'result': _blocked(request, code, used, rejected),
        'continuation_decision': 'defer', 'blocked_reason': 'Resolve the identity, pair, native direction, or requested graph scope.',
        'summary': 'Preserve the raw request and original observations without an invented pair or biological claim.'}
    return request, active, steps, answer


def build_diagnostic(view, fact, framing, variant_index, bank, contract, *,
                     kind='missing_genes', registry, seed=20261028):
    """Retain exact raw prerequisites, candidate sets, and native tool rejections."""
    from .s2_13_prompts import render_s2_13_diagnostic_request
    request, _, _ = _diagnostic_request(view, fact, kind)
    wording = _diagnostic_wording(kind)
    core = render_s2_13_diagnostic_request(contract['_requests'], split=fact.split, kind=wording,
                                         variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's213_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32]
    request, active, steps, answer = _diagnostic_steps(view, fact, kind, row_id, registry, contract)
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}, *steps,
        {'role': 'assistant', 'thinking': _reason(answer, framing), 'content': answer}]
    task = {'question_kind': 'edge_card', 'claim': None, 'subjects': [], 'statement': None,
            'graph_claim': None, 'graph_truth_scope': None}
    metadata = _metadata(view, fact, fact.split, framing, kind, request, task, wrapper,
        variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording]), contract, seed)
    metadata.update({key: None for key in LABELS})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': metadata}


def validate_row(row, view, *, bank, contract, registry, index=None, old_view=None):
    """Replay literal calls and exact structured answers against the visible proof."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_13_prompts import render_s2_13_request, render_s2_13_diagnostic_request
    if old_view is None:
        from .s2_sources import load_source_view
        old_view = load_source_view(ROOT / contract['previous_source_manifest']['path'])
    index = index or SubgraphIndex(view)
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if (sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness'
            or messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']):
        raise ValueError('the row requires one literal user turn and its exact system, harness, and tools')
    for message in messages:
        fields = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != fields or message['role'] == 'assistant' and (not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    final = messages[-1]['content']
    if (list(final) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or not isinstance(final['summary'], str) or not final['summary'].strip()
            or (final['continuation_decision'] == 'defer') != bool(final['blocked_reason'])
            or final['blocked_reason'] is not None and (not isinstance(final['blocked_reason'], str) or not final['blocked_reason'].strip())):
        raise ValueError('invalid envelope, free summary, or blocker presence')
    framing, kind, variant = meta['framing'], meta['question_kind'], meta['variant']
    if framing not in FRAMINGS or meta['split'] not in SPLITS:
        raise ValueError('unknown framing or split')
    family, diagnostic = kind.startswith('family_'), variant in DIAGNOSTICS
    request = meta['request']
    if set(request) != {'genes', 'multiplex_id', 'layer_id', 'direction'}:
        raise ValueError('the literal request must preserve exactly four raw fields')
    task = {key: meta[key] for key in TASK_FIELDS}
    if family:
        fact = None
        if (meta['layer_family'] not in FAMILIES or variant != 'taxonomy' or request != dict.fromkeys(request)
                or task != {'question_kind': kind, 'claim': meta['claim'], 'subjects': [], 'statement': None,
                            'graph_claim': None, 'graph_truth_scope': None}
                or kind == 'family_card' and meta['claim'] is not None
                or kind == 'family_support' and meta['claim'] not in CLAIM_VOCABULARY):
            raise ValueError('the conceptual family lookup has an unrequested graph scope or claim')
    else:
        fact = fact_from_pair(view, index, registry, old_view, meta['layer_id'], meta['gene_ids'], meta['selection_seed'])
        if fact.split != meta['split']:
            raise ValueError('the unordered query pair crossed its fixed split')
        if diagnostic:
            expected_request, _, _ = _diagnostic_request(view, fact, variant)
            if request != expected_request or task != {'question_kind': 'edge_card', 'claim': None,
                    'subjects': [], 'statement': None, 'graph_claim': None, 'graph_truth_scope': None}:
                raise ValueError('the prerequisite diagnostic changed its original task or raw values')
        else:
            if variant not in VARIANTS or task != validate_task(view, fact, **task):
                raise ValueError('the pair task differs from its exact source scope or proposition')
            expected_request = {'genes': _raw_genes(fact, variant, registry), 'multiplex_id': view.multiplex_id,
                                'layer_id': fact.layer_id, 'direction': None}
            if request != expected_request:
                raise ValueError('the pair request changes its literal identity, native interpretation, or graph scope')
    wording = _diagnostic_wording(variant)
    templates = contract['_requests']['diagnostic_requests'][meta['split']][wording] if diagnostic else contract['_requests'][meta['split']][kind]
    for key, size in (('wrapper_index', len(bank[meta['split']][framing])), ('request_template_index', len(templates))):
        if type(meta[key]) is not int or not 0 <= meta[key] < size:
            raise ValueError('the request must select an audited frozen wording and wrapper')
    expected_meta = _metadata(view, fact, meta['split'], framing, variant, request,
        {**task, 'layer_family': meta['layer_family']} if family else deepcopy(task), meta['wrapper_index'],
        meta['request_template_index'], contract, meta['selection_seed'])
    if diagnostic:
        expected_meta.update({key: None for key in LABELS})
    if canonical_json(meta) != canonical_json(expected_meta):
        raise ValueError('the source features, build identity, or private metadata changed')
    if diagnostic:
        core = render_s2_13_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording,
            variant_index=meta['request_template_index'], **request)
    else:
        core = render_s2_13_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'],
            **request, **task, layer_family=meta['layer_family'])
    user = bank[meta['split']][framing][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': user}:
        raise ValueError('the user request differs from its frozen task wording')
    identity = ([contract['contract_version'], meta['layer_family'], meta['split'], framing, kind, task['claim'], user] if family else
        [contract['contract_version'], view.graph_version, user, variant] if diagnostic else
        [contract['contract_version'], view.graph_version, asdict(fact), framing, variant, task, user])
    if row['row_id'] != 's213_' + digest(identity)[:32]:
        raise ValueError('the row ID does not bind its literal task, source fact, and wording')
    context = json.loads(messages[1]['content'])
    if family:
        if context != {'active_graph': _active_context(view), 'observations': []} or len(messages) != 4:
            raise ValueError('a conceptual family lookup cannot invent observations or calls')
        expected = {'status': 'resolved', 'provenance': {'registry': [], 'graph': []},
            'result': family_result(meta['layer_family'], claim=task['claim']),
            'continuation_decision': 'continue' if framing == 'step' else 'stop'}
    elif diagnostic:
        _, active, steps, answer = _diagnostic_steps(view, fact, variant, row['row_id'], registry, contract)
        if context != {'active_graph': active, 'observations': []} or len(messages) != 4 + len(steps):
            raise ValueError('the raw prerequisite changes its visible scope or observations')
        for actual, prescribed in zip(messages[3:-1], steps):
            if any(canonical_json(actual[key]) != canonical_json(value) for key, value in prescribed.items() if key != 'thinking'):
                raise ValueError('the prerequisite changes its exact lookup, rejection, or candidate set')
        expected = answer
    else:
        supplied, serial = _context(view, fact, variant, row['row_id'], index, registry, contract, old_view, meta['selection_seed'])
        if canonical_json(context) != canonical_json({'active_graph': _active_context(view), 'observations': supplied}):
            raise ValueError('the declared contrast changes its original source observations')
        available, identities = list(supplied), {}

        def verify(record):
            if set(record) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
                raise ValueError('a saved acquisition changes its original fields')
            for records in record['provenance'].values():
                for evidence in records:
                    key = evidence['evidence_id']
                    if key in identities and canonical_json(identities[key]) != canonical_json(record):
                        raise ValueError('one evidence ID names contradictory acquisitions')
                    identities[key] = record
            operation, source = record['operation'], record['arguments']
            if operation.startswith('lookup_human_'):
                records = record['provenance']['registry']
                if len(records) != 1:
                    raise ValueError('a registry query needs one original registry record')
                response = _identity_output(registry, contract['registry_index'], operation,
                    next(iter(source.values())), records[0]['evidence_id'])
                checked = _identity_record(operation, source, response)
            elif operation == 'get_edge':
                records = record['provenance']['graph']
                if len(records) != 1:
                    raise ValueError('an edge acquisition needs one original graph record')
                selected = (view if (source['multiplex_id'], records[0]['graph_version']) == (view.multiplex_id, view.graph_version)
                    else old_view if (source['multiplex_id'], records[0]['graph_version']) == (old_view.multiplex_id, old_view.graph_version) else None)
                if selected is None:
                    raise ValueError('the acquisition has no qualified source view')
                checked = _observation(operation, source, selected.get_edge(**source, evidence_id=records[0]['evidence_id']))
            else:
                raise ValueError('this builder has no declared source acquisition for that operation')
            if canonical_json(record) != canonical_json(checked):
                raise ValueError('the saved acquisition differs from its complete pinned source')

        for record in available:
            verify(record)
        fault = next((mode for mode in ('terminal', 'recovered', 'exhausted') if variant.endswith(mode)), None)
        previous_call = previous_response = failure = None
        errors = []
        for position, message in enumerate(messages):
            if 'tool_call' not in message:
                continue
            proof, used = _plan(view, request, available, task)
            call = message['tool_call']
            if proof.next_arguments is None or call != {'name': proof.next_tool, 'arguments': proof.next_arguments}:
                raise ValueError('the call changes the required identity or missing native pair acquisition')
            response_message = messages[position + 1]
            if response_message['role'] != 'tool' or response_message['name'] != call['name']:
                raise ValueError('the response does not match its immediate call')
            response = response_message['content']
            if previous_response and previous_response['status'] == 'error' and (previous_response != _failure(1) or call != previous_call):
                raise ValueError('the retry lacks visible allowance or changes its exact arguments')
            if response['status'] == 'error':
                if call['name'] != 'get_edge' or fault is None or response not in (_failure(None), _failure(1), _failure(0)):
                    raise ValueError('an undeclared failure supplies false evidence or retry permission')
                errors.append(response)
                failure = call['name'], call['arguments'], response
            else:
                record = (_identity_record(call['name'], call['arguments'], response) if call['name'].startswith('lookup_human_')
                          else _observation(call['name'], call['arguments'], response))
                verify(record)
                serial += 1
                channel = 'registry' if call['name'].startswith('lookup_human_') else 'graph'
                if record['provenance'][channel][0]['evidence_id'] != f"ev_{row['row_id']}_{serial}":
                    raise ValueError('the current acquisition changes its original evidence identity')
                available.append(record)
                failure = None
            previous_call, previous_response = call, response
        expected_errors = ([_failure(None)] if fault == 'terminal' else [_failure(1), _failure(0)]
                           if fault == 'exhausted' else [_failure(1)] if fault == 'recovered' else [])
        if errors != expected_errors or previous_response == _failure(1):
            raise ValueError('the declared fault schedule changed or an allowed retry remains unused')
        proof, used = _plan(view, request, available, task)
        expected = _answer(request, framing, proof, used, failure, available)
    if any(canonical_json(final[key]) != canonical_json(expected[key]) for key in
            ('status', 'result', 'provenance', 'continuation_decision')):
        raise ValueError('the final scope, cards, prerequisite, evidence, or completion decision changed')


def build_corpus(output_dir, *, counts=None, smoke=False, seed=20261028,
                 contract_path=CONTRACT_PATH, source_manifest=None):
    """Export isolated S2.13 rows after source, evidence, wording, and mask checks."""
    from .s2_13_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
