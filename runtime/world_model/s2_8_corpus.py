"""Build degree and hub traces from attributed counts and exact rank support.

The sampler supplies private labels for verification. The planner can use only
visible observations and manifest metadata to answer the current request.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _active, _observation
from .s2_2_identity import RegistryContext, get_symbols, identity_observations, validate_identity_observations
from .s2_3_corpus import gene_group, _provenance
from .s2_6_corpus import _global_observation
from .s2_8_sampling import DegreeFact, fact_from_arguments
from .s2_corpus import SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_degrees import DegreeIndex, DegreeRuntime, DEGREE_POLICY_ID, POPULATION_POLICY_ID, HUB_POLICY_ID
from .s2_multiplex import SelectionError
from .s2_neighbor_layers import NeighborLayersRuntime
from .s2_neighbors import NeighborRuntime
from .s2_sources import GraphView


CONTRACT_PATH = Path(__file__).with_name('s2_8_v1.json')
VARIANTS = ('lookup', 'context', 'degree_context', 'histogram_context', 'native_context',
    'bounded_total', 'unknown_total', 'prefix_total', 'complete_pages', 'final_page_only',
    'map_context', 'map_total', 'union_counts', 'equivalent_scope', 'equivalent_direction',
    'new_threshold', 'threshold_zero', 'threshold_fifty', 'threshold_hundred', 'duplicate_context',
    'global_context', 'symbol_context', 'stale', 'wrong_scope', 'wrong_direction', 'distractor',
    'terminal', 'recovered', 'exhausted', 'rank_terminal', 'rank_recovered', 'rank_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_gene', 'missing_scope',
               'invalid_scope', 'invalid_direction', 'invalid_policy', 'tool_unknown_layer')


def load_build_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load the isolated S2.8 overlay and its frozen request bank."""
    from .s2_8_prompts import load_s2_8_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.8']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.8 build contract')
    contract['_requests'] = load_s2_8_requests(contract['request_bank'])
    return contract


def _arguments(view: GraphView, fact: DegreeFact, request_kind: str, threshold: int) -> dict:
    """Bind all six fields while keeping the requested count separate from rank."""
    return {'multiplex_id': view.multiplex_id, 'layer_scope': deepcopy(fact.layer_scope), 'gene_id': fact.gene_id,
        'direction': fact.direction, 'request_kind': request_kind,
        'hub_policy': {'policy_id': HUB_POLICY_ID, 'threshold_percentile': threshold} if request_kind == 'degree_and_hub' else None}


def _active_context(view: GraphView) -> dict:
    """Declare the active graph and all policies that affect the answer."""
    runtime = DegreeRuntime(view)
    return {**_active(view), 'degree_policy_id': DEGREE_POLICY_ID, 'population_policy_id': POPULATION_POLICY_ID,
        'default_hub_policy': {'policy_id': HUB_POLICY_ID, 'threshold_percentile': 95},
        'percentile_display': {'decimal_places': 6, 'rounding': 'ROUND_HALF_UP'},
        'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed,
                    'family': runtime.multiplex.family_of(layer)} for layer in view.layer_order]}


def _request(arguments: Mapping, raw_gene: Any) -> dict:
    """Retain the literal gene and all requested policy and scope values."""
    return {'gene': raw_gene, **{key: deepcopy(arguments[key]) for key in (
        'multiplex_id', 'layer_scope', 'direction', 'request_kind', 'hub_policy')}}


def _blocked_result(request: dict, issue_code: str, used: Sequence[dict], rejected: dict | None = None) -> dict:
    """Keep supported partial data without an invented rank or hub label."""
    return {'issue_code': issue_code, 'request': deepcopy(request), 'rejected_arguments': rejected,
            'partial_results': list(used), 'coverage': None}


def _plan(view: GraphView, arguments: dict, observations: Sequence[dict]):
    from .s2_8_evidence import collect
    proof = collect(view, arguments, observations)
    identities = [observation for observation in observations if observation['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _histogram_observation(view: GraphView, index: DegreeIndex, arguments: dict, evidence_id: str) -> dict:
    """Supply the approved complete degree_distribution record from full sources."""
    runtime = DegreeRuntime(view, index=index)
    selected = runtime.validate_arguments(arguments)
    scope = runtime.multiplex._scope(arguments['layer_scope'], selected)
    source = {key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_scope', 'direction')}
    source.update(degree_policy_id=DEGREE_POLICY_ID, population_policy_id=POPULATION_POLICY_ID)
    return {'operation': 'degree_distribution', 'arguments': source, 'status': 'resolved',
        'result': {**source, 'layer_scope': scope, 'resolved_layer_ids': list(selected), 'population_count': len(index.population(selected)),
                   'degree_histogram': index.histogram(selected, arguments['direction']), 'complete': True},
        'provenance': {'registry': [], 'graph': [{'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
            'layer_scope': scope, 'evidence_id': evidence_id}]}}


def _transport(variant: str) -> dict:
    """Declare context page schedules; the count acquisition itself has no cap."""
    return {'page_size': 2 if variant in ('unknown_total', 'prefix_total', 'complete_pages', 'final_page_only') else None,
            'disclose_total': variant not in ('unknown_total', 'final_page_only')}


def _reason(result: dict, framing: str) -> str:
    """Explain the counted union, exact policy comparison, and task completion."""
    text = (f"The requested {result['direction']} union covers {result['resolved_layer_ids']}. "
            f"Its exact distinct-neighbor degree is {result['degree']}. Membership is {result['membership']}. "
            'Count each ID once across layers, reciprocal directions, and self-loops. Do not add overlapping counts or use weights. ')
    if result['request_kind'] == 'degree_and_hub':
        rank = result['rank_support']
        text += (f"The exact support is L={rank['lower_degree_count']}, E={rank['equal_degree_count']}, N={rank['population_count']}. "
                 f"Compare the exact normalized average rank with {result['hub_policy']['threshold_percentile']}, including equality. "
                 f"The six-decimal display is {result['degree_percentile']}; the exact comparison gives hub_like={result['hub_like']}. "
                 'The reference population includes every selected-layer member, directional zeros, and graph-only IDs. '
                 'A true label supplies the policy warning about possible degree-related proximity bias; it does not measure that effect. ')
    else:
        text += 'Only the count is requested. No percentile or hub evaluation is needed. '
    return text + ('The requested workflow step is complete, so continue.' if framing == 'step'
                   else 'The standalone question is complete, so stop.')


def build_row(view: GraphView, fact: DegreeFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, index: DegreeIndex,
              registry: RegistryContext | None = None, request_kind: str = 'degree_and_hub',
              threshold: int = 95, seed: int = 20261028) -> dict:
    """Build one source-backed trace with explicit visible evidence and policies."""
    from .s2_8_prompts import render_s2_8_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported degree contrast or framing')
    if variant in ('new_threshold', 'threshold_zero', 'threshold_fifty', 'threshold_hundred'):
        if request_kind != 'degree_and_hub':
            raise VariantUnavailable('a threshold override requires a requested hub evaluation')
        threshold = {'new_threshold': 100, 'threshold_zero': 0, 'threshold_fifty': 50, 'threshold_hundred': 100}[variant]
    arguments = _arguments(view, fact, request_kind, threshold)
    runtime = DegreeRuntime(view, index=index)
    selected = runtime.validate_arguments(arguments)
    raw_gene = fact.gene_id
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('symbol context requires the pinned registry')
        try:
            raw_gene = get_symbols([fact.gene_id], registry)[0]
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = _request(arguments, raw_gene)
    core = render_s2_8_request(contract['_requests'], split=fact.split, variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's28_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, request, user])[:32]
    transport = _transport(variant)
    native = NeighborRuntime(view, index=index.native_index, **transport)
    maps = NeighborLayersRuntime(view, index=index.native_index, **transport)
    observations, serial = [], 0

    def evidence_id():
        nonlocal serial
        serial += 1
        return f'ev_{row_id}_{serial}'

    def save_degree(source):
        observations.append(_observation('get_degree', source, runtime.get_degree(**source, evidence_id=evidence_id())))

    native_queries = [(layer, side) for layer in selected for side in (
        index.native_index.directions(layer) if fact.direction == 'incident' else (fact.direction,))]

    def save_native(layer, side, *, bounded=False, first_only=False, last_only=False):
        source = {'multiplex_id': view.multiplex_id, 'layer_id': layer, 'gene_id': fact.gene_id, 'direction': side,
                  'mode': 'top_k' if bounded else 'complete', 'top_k': 1 if bounded else None, 'cursor': None}
        pages = []
        while True:
            response = native.get_neighbors(**source, evidence_id=evidence_id())
            pages.append(_observation('get_neighbors', source, response))
            cursor = (response['result'].get('collection') or {}).get('next_cursor')
            if cursor is None or first_only:
                break
            source = {**source, 'cursor': cursor}
        observations.extend(pages[-1:] if last_only else pages)

    if variant in ('context', 'duplicate_context', 'stale'):
        save_degree(arguments)
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
        elif variant == 'stale':
            observations[0]['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('degree_context', 'histogram_context', 'rank_terminal', 'rank_recovered', 'rank_exhausted'):
        save_degree({**arguments, 'request_kind': 'degree_only', 'hub_policy': None})
        if variant == 'histogram_context':
            if request_kind != 'degree_and_hub':
                raise VariantUnavailable('a histogram contrast requires a hub request')
            observations.append(_histogram_observation(view, index, arguments, evidence_id()))
    elif variant in ('native_context', 'bounded_total', 'unknown_total', 'prefix_total', 'complete_pages', 'final_page_only'):
        if not native_queries:
            raise VariantUnavailable('native neighbor context requires a nonempty selected scope')
        if variant in ('bounded_total', 'unknown_total', 'prefix_total', 'final_page_only') and len(native_queries) != 1:
            raise VariantUnavailable('one native total can establish only that primitive count')
        if variant in ('unknown_total', 'prefix_total', 'complete_pages', 'final_page_only') and (fact.degree or 0) < 3:
            raise VariantUnavailable('the selected list needs a genuine partial page')
        for layer, side in native_queries:
            save_native(layer, side, bounded=variant == 'bounded_total',
                        first_only=variant in ('unknown_total', 'prefix_total'), last_only=variant == 'final_page_only')
    elif variant in ('map_context', 'map_total'):
        source = maps.arguments(fact.gene_id, fact.layer_scope, fact.direction,
                                mode='top_k' if variant == 'map_total' else 'complete', top_k=1 if variant == 'map_total' else None)
        observations.append(_observation('get_neighbor_layers', source, maps.get_neighbor_layers(**source, evidence_id=evidence_id())))
    elif variant == 'union_counts':
        if len(native_queries) < 2:
            raise VariantUnavailable('separate counts require at least two primitive sets')
        for layer, side in native_queries:
            save_degree({**arguments, 'layer_scope': {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None},
                         'direction': side, 'request_kind': 'degree_only', 'hub_policy': None})
    elif variant == 'equivalent_scope':
        scopes = [{'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in (None, *runtime.multiplex.families)]
        if selected:
            scopes.append({'kind': 'layers', 'layer_ids': list(reversed(selected)), 'layer_family': None})
        equivalent = None
        for scope in scopes:
            try:
                matches = runtime.resolve_scope(scope, fact.direction) == selected
            except SelectionError:
                continue
            if matches and scope != arguments['layer_scope']:
                equivalent = scope
                break
        if equivalent is None:
            raise VariantUnavailable('no different equivalent selector exists for this scope')
        save_degree({**arguments, 'layer_scope': equivalent})
    elif variant == 'equivalent_direction':
        if not selected or any(view.layers[layer].directed for layer in selected):
            raise VariantUnavailable('undirected/incident equivalence requires undirected layers')
        save_degree({**arguments, 'direction': 'incident' if fact.direction == 'undirected' else 'undirected'})
    elif variant == 'new_threshold':
        save_degree({**arguments, 'hub_policy': {'policy_id': HUB_POLICY_ID, 'threshold_percentile': 95}})
    elif variant == 'global_context':
        observations.append(_global_observation(view, [fact.gene_id], evidence_id()))
    elif variant == 'symbol_context':
        observations.extend(identity_observations([fact.gene_id], row_id, registry))
    elif variant == 'wrong_scope':
        layer = next((layer for layer in view.layer_order if (layer,) != selected), None)
        if layer is None:
            raise VariantUnavailable('no other selected layer exists')
        scope = {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None}
        direction = 'incident'
        save_degree({**arguments, 'layer_scope': scope, 'direction': direction, 'request_kind': 'degree_only', 'hub_policy': None})
    elif variant == 'wrong_direction':
        direction = next((side for side in ('in', 'out') if side != fact.direction
                          and selected and all(view.layers[layer].directed for layer in selected)), None)
        if direction is None:
            raise VariantUnavailable('a wrong native direction requires directed layers')
        save_degree({**arguments, 'direction': direction, 'request_kind': 'degree_only', 'hub_policy': None})
    elif variant == 'distractor':
        pool = (gene for gene in view.multiplex_nodes if gene != fact.gene_id and gene_group(gene, seed)[1] == fact.split)
        gene = min(pool, key=lambda value: digest([seed, fact.split_unit, fact.layer_scope, fact.direction, 'distractor', value]), default=None)
        if gene is None:
            raise VariantUnavailable('no same-split source distractor exists')
        save_degree({**arguments, 'gene_id': gene, 'request_kind': 'degree_only', 'hub_policy': None})
    proof, _ = _plan(view, arguments, observations)
    if variant.startswith('rank_') and (request_kind != 'degree_and_hub' or proof.next_arguments is None):
        raise VariantUnavailable('the rank acquisition must remain required after the supplied degree')
    context = {'active_graph': _active_context(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    failure = None
    kind = variant.removeprefix('rank_')
    while True:
        proof, used = _plan(view, arguments, observations)
        if proof.next_arguments is None:
            break
        pending = proof.next_arguments
        thought = (f"The {request_kind} request uses {fact.layer_scope} and direction {fact.direction}. "
            'Visible evidence does not yet supply every required count or exact rank. '
            'Acquire the exact statistic. A complete count does not supply its population rank, and separate totals do not supply a union.')

        def append(response, reasoning):
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                              'tool_call': {'name': 'get_degree', 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': 'get_degree', 'content': response}])

        if kind in ('terminal', 'recovered', 'exhausted'):
            response = _failure(None if kind == 'terminal' else 1)
            append(response, thought)
            if kind != 'terminal':
                response = _failure(0) if kind == 'exhausted' else runtime.get_degree(**pending, evidence_id=evidence_id())
                append(response, 'The visible error supplies retries_remaining=1. Repeat the identical tool and all six fields. '
                                 'Stop after recovery or exhaustion. Failed calls add no evidence.')
            if response['status'] == 'error':
                failure = (pending, response)
                break
        else:
            response = runtime.get_degree(**pending, evidence_id=evidence_id())
            append(response, thought)
        observations.append(_observation('get_degree', pending, response))
    if kind in ('terminal', 'recovered', 'exhausted') and len(messages) == 3:
        raise VariantUnavailable('supplied evidence settles the question before any required failure')
    if failure:
        pending, response = failure
        result = {'failed_tool': 'get_degree', 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': None}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result, 'continuation_decision': 'defer',
            'blocked_reason': 'The required statistic backend must recover before this request can finish.',
            'summary': 'A required statistic failed. Retain every supported earlier count and membership record.'}
        thought = 'The required acquisition failed without a remaining retry allowance. Keep attributed partial data and defer. '
        thought += 'An unavailable rank does not erase a known degree, and no failed call supplies a zero or a hub label.'
    elif proof.issue_code:
        answer = {'status': 'not_found', 'provenance': _provenance(used),
            'result': _blocked_result(request, proof.issue_code, used), 'continuation_decision': 'defer',
            'blocked_reason': 'The required query membership or reference population is unavailable.',
            'summary': 'The requested statistic is unavailable. Retain supported measurements without a fabricated rank or hub label.'}
        thought = f"The visible prerequisite fails with {proof.issue_code}. Retain every supported count and membership as partial data. "
        thought += 'Global absence is not degree zero. A known scoped zero can survive an unavailable hub evaluation. Defer.'
    else:
        result = proof.result
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': result,
            'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
            'summary': 'The requested distinct-neighbor statistic follows from the visible compatible evidence.'}
        thought = ('The supplied observations are sufficient; no call is needed. ' if len(messages) == 3 else '') + _reason(result, framing)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.8', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'layer_scope': fact.layer_scope,
        'direction': fact.direction, 'request_kind': request_kind, 'hub_policy': arguments['hub_policy'],
        'input_gene': raw_gene, 'request': request, 'degree': fact.degree, 'membership': fact.membership, 'rank_support': fact.rank_support,
        'category': fact.category, 'framing': framing, 'variant': variant, 'transport': transport,
        'wrapper_index': wrapper, 'request_template_index': variant_index % len(contract['_requests'][fact.split][request_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view: GraphView, fact: DegreeFact, kind: str) -> tuple[dict, str, str]:
    """Keep malformed JSON inputs separate from a valid sampling coordinate."""
    request = _request(_arguments(view, fact, 'degree_and_hub', 95), fact.gene_id)
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'missing_gene': 'MISSING_GENE',
        'missing_scope': 'MISSING_SCOPE', 'invalid_scope': 'INVALID_SCOPE',
        'invalid_direction': 'DIRECTION_UNAVAILABLE', 'invalid_policy': 'INVALID_HUB_POLICY'}
    if kind not in codes:
        raise ValueError('unsupported degree prerequisite diagnostic')
    if kind in ('unknown_layer', 'tool_unknown_layer'):
        request['layer_scope'] = {'kind': 'layers', 'layer_ids': ['TFs:missing_context'], 'layer_family': None}
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_degree_multiplex'
    elif kind == 'missing_gene':
        request['gene'] = None
    elif kind == 'missing_scope':
        request['layer_scope'] = None
    elif kind == 'invalid_scope':
        request['layer_scope'] = {'kind': 'layers', 'layer_ids': [], 'layer_family': None}
    elif kind == 'invalid_direction':
        request['direction'] = 'both'
    elif kind == 'invalid_policy':
        request['hub_policy'] = {'policy_id': HUB_POLICY_ID, 'threshold_percentile': True}
    status = 'not_found' if kind in ('unknown_layer', 'tool_unknown_layer', 'unknown_multiplex') else 'invalid_input'
    return request, status, codes[kind]


def build_diagnostic(view: GraphView, fact: DegreeFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'missing_gene', seed: int = 20261028) -> dict:
    """Reject a literal prerequisite locally or through the uncatalogued scope tool."""
    from .s2_8_prompts import render_s2_8_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    wording_kind = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    core = render_s2_8_diagnostic_request(contract['_requests'], split=fact.split, kind=wording_kind,
                                          variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    active = _active_context(view)
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    rejected = None
    if kind == 'tool_unknown_layer':
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_scope': request['layer_scope'],
            'gene_id': request['gene'], 'direction': request['direction'],
            'request_kind': request['request_kind'], 'hub_policy': request['hub_policy']}
        response = DegreeRuntime(view).get_degree(**rejected, evidence_id='unused_rejected_degree_request')
        messages.extend([{'role': 'assistant', 'thinking':
            'The visible context withholds the layer catalogue. Ask the resolver for the requested scope without an invented substitute.',
            'tool_call': {'name': 'get_degree', 'arguments': deepcopy(rejected)}},
            {'role': 'tool', 'name': 'get_degree', 'content': response}])
    answer = {'status': status, 'provenance': {'registry': [], 'graph': []},
        'result': _blocked_result(request, code, [], rejected), 'continuation_decision': 'defer',
        'blocked_reason': 'The requested identity, graph scope, direction, or hub policy is invalid or unavailable.',
        'summary': 'A request prerequisite is unresolved; no graph statistic or hub label was computed.'}
    messages.append({'role': 'assistant', 'thinking': f"The literal request is {request}. Its prerequisite fails with {code}. "
        'Retain the original JSON values and any rejected arguments. An invalid request is not a measured zero. Defer.', 'content': answer})
    return {'row_id': 's28_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32],
        'messages': messages, 'tools': contract['tools'], 'metadata': {
            'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
            'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
            'question_family': 'S2.8', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
            'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'layer_scope': fact.layer_scope,
            'direction': fact.direction, 'request_kind': 'degree_and_hub',
            'hub_policy': {'policy_id': HUB_POLICY_ID, 'threshold_percentile': 95},
            'input_gene': request['gene'], 'request': request, 'degree': None, 'membership': None, 'rank_support': None,
            'category': fact.category, 'framing': framing, 'variant': kind, 'transport': None,
            'wrapper_index': wrapper,
            'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording_kind]),
            'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: DegreeIndex | None = None) -> None:
    """Replay source acquisitions, then grade only the visible statistic proof."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_8_prompts import render_s2_8_request, render_s2_8_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.8 requires one user turn and explicit harness context')
    for message in messages:
        expected = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected or message['role'] == 'assistant' and not message['thinking'].strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the degree overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.8'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('degree row identity differs from the build contract')
    index = index or DegreeIndex(view)
    fact = fact_from_arguments(view, index, meta['gene_id'], meta['layer_scope'], meta['direction'],
                                meta['selection_seed'], category=meta['category'])
    if fact.split != meta['split'] or fact.split_unit != meta['split_unit']:
        raise ValueError('the query differs from its fixed subject split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if ((answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason'])
            or answer['blocked_reason'] is not None and (not isinstance(answer['blocked_reason'], str)
                                                        or not answer['blocked_reason'].strip())):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if meta['framing'] not in FRAMINGS or not diagnostic and meta['variant'] not in VARIANTS:
        raise ValueError('unsupported framing or contrast')
    wording_kind = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
    if meta['request_kind'] not in ('degree_only', 'degree_and_hub'):
        raise ValueError('unsupported requested statistic')
    templates = (contract['_requests']['diagnostic_requests'][meta['split']][wording_kind] if diagnostic
                 else contract['_requests'][meta['split']][meta['request_kind']])
    if any(type(meta[key]) is not int or not 0 <= meta[key] < size for key, size in (
            ('wrapper_index', len(bank[meta['split']][meta['framing']])), ('request_template_index', len(templates)))):
        raise ValueError('wrapper and request indices must select their audited banks')
    if diagnostic:
        request, _, _ = _diagnostic_request(view, fact, meta['variant'])
        if canonical_json(meta['request']) != canonical_json(request) or meta['input_gene'] != request['gene']:
            raise ValueError('the diagnostic changes the literal request')
        core = render_s2_8_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording_kind,
            variant_index=meta['request_template_index'], **request)
    else:
        if any(canonical_json(getattr(fact, key)) != canonical_json(meta[key]) for key in ('degree', 'membership', 'rank_support')):
            raise ValueError('private sampling labels differ from full source statistics')
        arguments = {'multiplex_id': view.multiplex_id, 'layer_scope': deepcopy(fact.layer_scope),
            'gene_id': fact.gene_id, 'direction': fact.direction, 'request_kind': meta['request_kind'],
            'hub_policy': deepcopy(meta['hub_policy'])}
        DegreeRuntime(view).validate_arguments(arguments)
        if meta['variant'] == 'symbol_context':
            if registry is None or get_symbols([fact.gene_id], registry)[0] != meta['input_gene']:
                raise ValueError('the supplied symbol does not uniquely identify the query')
        elif meta['input_gene'] != fact.gene_id:
            raise ValueError('the raw ID changes the canonical query')
        request = _request(arguments, meta['input_gene'])
        if canonical_json(meta['request']) != canonical_json(request) or meta['transport'] != _transport(meta['variant']):
            raise ValueError('the literal request or declared page contrast changed')
        core = render_s2_8_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'], **request)
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    active = _active_context(view)
    if meta['variant'] == 'tool_unknown_layer':
        active.pop('layers')
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != active:
        raise ValueError('the active graph or declared policies changed')
    if diagnostic:
        expected = build_diagnostic(view, fact, meta['framing'], meta['request_template_index'], bank, contract,
                                    kind=meta['variant'], seed=meta['selection_seed'])
        if context['observations'] or meta['transport'] is not None or any(meta[key] is not None for key in ('degree', 'membership', 'rank_support')):
            raise ValueError('malformed prerequisite diagnostic')
        expected_answer = expected['messages'][-1]['content']
        if any(canonical_json(meta[key]) != canonical_json(expected['metadata'][key])
               for key in ('request_kind', 'hub_policy')):
            raise ValueError('the diagnostic changes its declared request metadata')
        if len(messages) != len(expected['messages']):
            raise ValueError('a local rejection adds a call or a tool rejection drops its call')
        if any(canonical_json(answer[key]) != canonical_json(expected_answer[key])
               for key in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('the diagnostic changes the rejected request')
        for actual_message, expected_message in zip(messages[3:-1], expected['messages'][3:-1]):
            fields = ('tool_call',) if actual_message['role'] == 'assistant' else ('name', 'content')
            if any(canonical_json(actual_message[key]) != canonical_json(expected_message[key]) for key in fields):
                raise ValueError('the scope rejection changes the call or response')
        return
    runtime = DegreeRuntime(view, index=index)
    native = NeighborRuntime(view, index=index.native_index, **meta['transport'])
    maps = NeighborLayersRuntime(view, index=index.native_index, **meta['transport'])
    available, evidence_ids = list(context['observations']), {}
    identities = [observation for observation in available if observation['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        validate_identity_observations(identities, registry)
        if ([observation['result']['gene_id'] for observation in identities] != [fact.gene_id]
                or [observation['arguments']['gene_symbol'] for observation in identities] != [meta['input_gene']]):
            raise ValueError('required identity evidence is missing or changed')
    elif identities:
        raise ValueError('unexpected identity context')

    def verify(observation):
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected saved-observation fields')
        for records in observation['provenance'].values():
            for record in records:
                evidence_id = record['evidence_id']
                if evidence_id in evidence_ids and canonical_json(evidence_ids[evidence_id]) != canonical_json(observation):
                    raise ValueError('one evidence ID names conflicting acquisitions')
                evidence_ids[evidence_id] = observation
        tool, source = observation['operation'], observation['arguments']
        if tool == 'lookup_human_gene_symbol':
            return
        if tool not in ('get_degree', 'get_neighbors', 'get_neighbor_layers', 'degree_distribution', 'check_multiplex_membership'):
            raise ValueError('unsupported supplied graph operation')
        if len(observation['provenance']['graph']) != 1:
            raise ValueError('a graph observation needs its original acquisition')
        graph = observation['provenance']['graph'][0]
        subjects = source.get('gene_ids', [source['gene_id']] if 'gene_id' in source else [])
        if any(gene_group(gene, meta['selection_seed'])[1] != meta['split'] for gene in subjects):
            raise ValueError('a supplied query subject crossed split boundaries')
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        if tool == 'check_multiplex_membership':
            expected = _global_observation(view, subjects, graph['evidence_id'])
        elif tool == 'degree_distribution':
            histogram_args = {**arguments, **{key: source[key] for key in ('multiplex_id', 'layer_scope', 'direction')}}
            expected = _histogram_observation(view, index, histogram_args, graph['evidence_id'])
        else:
            executor = {'get_degree': runtime.get_degree, 'get_neighbors': native.get_neighbors,
                        'get_neighbor_layers': maps.get_neighbor_layers}[tool]
            expected = _observation(tool, source, executor(**source, evidence_id=graph['evidence_id']))
        if canonical_json(observation) != canonical_json(expected):
            raise ValueError('an observation differs from its pinned source or page schedule')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available)
        call = message['tool_call']
        if proof.next_arguments is None or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments}):
            raise ValueError('a call changes the required missing statistic or declared policy')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes the failed arguments')
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('invalid fault schedule or invented failed-statistic evidence')
            consecutive_failures += 1
        else:
            if set(response) != {'status', 'provenance', 'result'}:
                raise ValueError('unexpected successful response fields')
            consecutive_failures = 0
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
        previous_call, previous_response = call, response
    proof, used = _plan(view, arguments, available)
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from the supporting visible observations')
    if previous_response and previous_response['status'] == 'error':
        if proof.next_arguments is None or previous_response == _failure(1):
            raise ValueError('an error requires an unresolved terminal acquisition failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': None}
        status, decision = 'error', 'defer'
    elif proof.issue_code:
        expected = _blocked_result(request, proof.issue_code, used)
        status, decision = 'not_found', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes sufficient count and rank evidence')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer changes the visible statistic, exact policy, or task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028, contract_path: Path = CONTRACT_PATH,
                 source_manifest: Path | None = None) -> dict:
    """Export the separate S2.8 verification corpus with exact source and token audits."""
    from .s2_8_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
