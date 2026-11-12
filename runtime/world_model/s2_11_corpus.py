"""Build full-layer path and claim traces from visible native evidence.

Private source labels check construction. Only supplied records and acquired
responses determine the next call and the final structured result.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_2_identity import (RegistryContext, get_symbols, identity_observations,
                            validate_identity_observations)
from .s2_3_corpus import gene_group, _provenance
from .s2_9_corpus import _active_context, _identity_partial, _identity_rejection
from .s2_11_sampling import PathFact, fact_from_genes
from .s2_corpus import ROOT, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_graph_claims import CLAIM_VOCABULARY
from .s2_neighbors import NeighborRuntime
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_paths import PathIndex, PathRuntime
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphIndex, SubgraphRuntime


CONTRACT_PATH = Path(__file__).with_name('s2_11_v1.json')
VARIANTS = ('lookup', 'reversed', 'unqualified', 'context', 'duplicate_context',
    'edge_context', 'route_context', 'longer_route_context', 'tie_route_context',
    'membership_context', 'subgraph_context', 'native_context', 'closed_component_context',
    'prefix_context', 'stale', 'wrong_layer', 'wrong_direction', 'reversed_context',
    'distractor', 'symbol_context', 'symbol_terminal', 'terminal', 'recovered', 'exhausted',
    'execution_limit', 'edge_terminal', 'edge_recovered', 'edge_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope',
    'invalid_pair', 'missing_direction', 'invalid_direction', 'unavailable_direction',
    'unresolved_identity', 'tool_unknown_layer')
LABELS = ('category_counts', 'path_found', 'path_gene_ids', 'hop_count', 'tied_shortest_paths')
ROUTE_CONSTRUCTION_BUDGET = {'max_states': 4096, 'neighbor_window': 16, 'extra_hops': 3}


def load_build_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load the isolated path overlay and its frozen request bank."""
    from .s2_11_prompts import load_s2_11_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.11']
            or contract['training_ready'] is not False or contract['production_export_allowed'] is not False):
        raise ValueError('unsupported S2.11 build contract')
    contract['_requests'] = load_s2_11_requests(contract['request_bank'])
    return contract


def _arguments(view, fact):
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
            'gene_ids': list(fact.gene_ids), 'direction': fact.direction}


def truth_statement(view, fact, claim, truth_scope):
    """Bind a truth proposition to its exact endpoints, scope, and graph version."""
    if (truth_scope == 'scope' and claim not in ('no_edge_recorded_in_scope', 'no_path_recorded_in_scope')
            or truth_scope == 'all_layers' and claim != 'absence_holds_in_other_layers'
            or truth_scope not in ('scope', 'all_layers')):
        raise ValueError('the truth proposition is outside the reviewed path scopes')
    source, target = fact.gene_ids
    subject = 'native edge' if claim == 'no_edge_recorded_in_scope' else 'path'
    scope = f'any layer of {view.multiplex_id}' if truth_scope == 'all_layers' else f'{fact.layer_id} of {view.multiplex_id}'
    return f'There is no {subject} from {source} to {target} in {scope} at graph version {view.graph_version}.'


def _task(view, fact, question_kind, claim, truth_scope, statement, membership_cards):
    """Keep optional truth and support outputs tied to an explicit user task."""
    from .s2_11_prompts import KINDS
    if question_kind not in KINDS or type(membership_cards) is not bool:
        raise ValueError('unsupported path question or membership-card choice')
    assessed = question_kind in ('path_support', 'path_truth', 'claim_support', 'truth')
    truth = question_kind in ('path_truth', 'truth')
    if assessed:
        claim = claim or ('absence_holds_in_other_layers' if truth_scope == 'all_layers' else 'no_path_recorded_in_scope')
        if claim not in CLAIM_VOCABULARY:
            raise ValueError('the assessed graph claim is outside the closed vocabulary')
    elif claim is not None:
        raise ValueError('the requested task does not assess a single claim')
    if truth:
        truth_scope = truth_scope or 'scope'
        if truth_scope not in ('scope', 'all_layers'):
            raise ValueError('the truth scope must be explicit and supported')
        expected_statement = truth_statement(view, fact, claim, truth_scope)
        statement = expected_statement if statement is None else statement
        if statement != expected_statement:
            raise ValueError('the literal truth statement differs from its exact graph proposition')
    elif truth_scope is not None or statement is not None:
        raise ValueError('a support task cannot silently become a truth task')
    if question_kind == 'path' and membership_cards:
        raise ValueError('ordinary path tasks do not request claim cards')
    return {'question_kind': question_kind, 'claim': claim, 'truth_scope': truth_scope,
            'statement': statement, 'membership_cards': membership_cards}


def _plan(view, arguments, observations, task):
    from .s2_11_evidence import collect, calibrate
    if task['question_kind'] == 'path':
        proof = collect(view, arguments, observations)
    else:
        proof = calibrate(view, arguments, observations,
            path_requested=task['question_kind'].startswith('path'), claim=task['claim'],
            truth_scope=task['truth_scope'], membership_cards=task['membership_cards'])
    identities = [item for item in observations if item['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _blocked(request, code, used, rejected=None):
    return {'issue_code': code, 'request': deepcopy(request), 'rejected_arguments': deepcopy(rejected),
            'partial_results': list(used)}


def _limit_failure():
    return {'status': 'error', 'provenance': {'registry': [], 'graph': []},
            'error_code': 'SEARCH_INCOMPLETE', 'message': 'The search ended before exhaustive completion.'}


def _fault_kind(variant):
    if variant == 'execution_limit':
        return 'execution_limit'
    return next((kind for kind in ('terminal', 'recovered', 'exhausted') if variant.endswith(kind)), None)


def _alternate_route(index, fact, *, longer=False):
    """Find a genuine alternate route within a declared construction budget."""
    if fact.path_found is not True or fact.hop_count == 0:
        raise VariantUnavailable('an alternate route needs distinct reachable endpoints')
    source, target = fact.gene_ids
    direction = 'out' if index.view.layers[fact.layer_id].directed else 'undirected'
    if not longer:
        if fact.tied_shortest_paths < 2:
            raise VariantUnavailable('this pair has no tied shortest route')
        distances = index.distances(fact.layer_id, source)
        ancestors, pending = {target}, [target]
        reverse = 'in' if direction == 'out' else direction
        while pending:
            node = pending.pop()
            for parent, _ in index.native_index.neighbors(fact.layer_id, node, reverse):
                if distances.get(parent) == distances[node] - 1 and parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        route = [source]
        while route[-1] != target:
            options = [gene for gene, _ in index.native_index.neighbors(fact.layer_id, route[-1], direction)
                       if gene in ancestors and distances.get(gene) == distances[route[-1]] + 1]
            route.append(max(options))
        return route
    # This budget selects an evidence contrast. It never bounds a path search.
    pending, examined = deque([(source,)]), 0
    while pending and examined < ROUTE_CONSTRUCTION_BUDGET['max_states']:
        route = pending.popleft()
        examined += 1
        if len(route) > fact.hop_count + ROUTE_CONSTRUCTION_BUDGET['extra_hops']:
            continue
        for gene, _ in sorted(index.native_index.neighbors(fact.layer_id, route[-1], direction))[:ROUTE_CONSTRUCTION_BUDGET['neighbor_window']]:
            if gene in route:
                continue
            candidate = (*route, gene)
            if gene == target and len(candidate) - 1 > fact.hop_count:
                return list(candidate)
            if gene != target:
                pending.append(candidate)
    raise VariantUnavailable('the finite alternate-route search found no longer source route')


def _context(view, fact, variant, row_id, index, registry, seed):
    """Create exact source observations without a hidden answer in the harness."""
    observations, serial = [], 0
    paths = PathRuntime(view, index=index)
    nodes = NodesByLayerRuntime(view)
    native = NeighborRuntime(view, index=index.native_index)
    arguments = _arguments(view, fact)

    def add(operation, source, execute):
        nonlocal serial
        serial += 1
        observations.append(_observation(operation, source, execute(**source, evidence_id=f'ev_{row_id}_{serial}')))

    if variant in ('symbol_context', 'symbol_terminal'):
        if registry is None:
            raise VariantUnavailable('symbol context requires the pinned registry')
        try:
            observations.extend(identity_observations(fact.gene_ids, row_id, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    elif variant in ('context', 'duplicate_context', 'stale', 'wrong_layer', 'wrong_direction', 'reversed_context', 'distractor'):
        source = deepcopy(arguments)
        if variant == 'wrong_layer':
            source['layer_id'] = next((layer for layer in view.layer_order if layer != fact.layer_id
                and view.layers[layer].directed == view.layers[fact.layer_id].directed), None)
            if source['layer_id'] is None:
                raise VariantUnavailable('no second layer has the same native direction')
        elif variant in ('wrong_direction', 'reversed_context'):
            if fact.gene_ids[0] == fact.gene_ids[1] or (variant == 'wrong_direction') != view.layers[fact.layer_id].directed:
                raise VariantUnavailable('the reverse-evidence contrast needs a distinct compatible pair')
            source['gene_ids'].reverse()
        elif variant == 'distractor':
            pool = sorted((gene for gene in view.layers[fact.layer_id].nodes if gene not in fact.gene_ids
                and gene_group(gene, seed)[1] == fact.split), key=lambda gene: gene_group(gene, seed)[0])
            if len(pool) < 2:
                raise VariantUnavailable('no disjoint same-split source pair is available')
            source['gene_ids'] = pool[:2]
        add('shortest_path', source, paths.shortest_path)
        if variant == 'duplicate_context':
            observations.append(deepcopy(observations[0]))
        elif variant == 'stale':
            observations[0]['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('edge_context', 'execution_limit', 'edge_terminal', 'edge_recovered', 'edge_exhausted'):
        if fact.gene_ids[0] == fact.gene_ids[1]:
            raise VariantUnavailable('the direct edge operation takes distinct endpoint IDs')
        source = {key: deepcopy(value) for key, value in arguments.items() if key != 'direction'}
        add('get_edge', source, view.get_edge)
    elif variant in ('route_context', 'longer_route_context', 'tie_route_context'):
        if variant == 'tie_route_context' and fact.hop_count != 2:
            raise VariantUnavailable('this minimum-length tie contrast uses a two-hop source route')
        route = list(fact.path_gene_ids or ())
        if variant != 'route_context':
            route = _alternate_route(index, fact, longer=variant == 'longer_route_context')
        if len(route) < 3:
            raise VariantUnavailable('a multihop evidence contrast needs at least two native arcs')
        if variant == 'tie_route_context':
            runtime = SubgraphRuntime(view, index=SubgraphIndex(view, index.native_index))
            add('induce_subgraph', runtime.arguments(route, fact.layer_id), runtime.induce_subgraph)
        else:
            for left, right in zip(route, route[1:]):
                source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': [left, right]}
                add('get_edge', source, view.get_edge)
    elif variant == 'membership_context':
        source = {key: deepcopy(value) for key, value in arguments.items() if key != 'direction'}
        add('get_nodes_by_layer', source, nodes.get_nodes_by_layer)
    elif variant == 'subgraph_context':
        runtime = SubgraphRuntime(view, index=SubgraphIndex(view, index.native_index))
        source = runtime.arguments(list(fact.gene_ids), fact.layer_id)
        add('induce_subgraph', source, runtime.induce_subgraph)
    elif variant in ('native_context', 'prefix_context', 'closed_component_context'):
        source = {key: deepcopy(value) for key, value in arguments.items() if key != 'direction'}
        add('get_nodes_by_layer', source, nodes.get_nodes_by_layer)
        subjects = [fact.gene_ids[0]]
        if variant == 'closed_component_context':
            if view.layers[fact.layer_id].directed or fact.path_found is not False or fact.category_counts != (2, 0, 0):
                raise VariantUnavailable('a closed-component contrast needs disconnected undirected layer members')
            subjects = sorted(index.distances(fact.layer_id, fact.gene_ids[0]))
            if len(subjects) > 8:
                raise VariantUnavailable('this semantic construction uses components of at most eight members')
        direction = 'out' if view.layers[fact.layer_id].directed else 'undirected'
        if variant == 'prefix_context' and len(index.native_index.neighbors(fact.layer_id, subjects[0], direction)) < 2:
            raise VariantUnavailable('a genuine native prefix must omit at least one neighbor')
        for gene in subjects:
            source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': gene,
                'direction': direction, 'mode': 'top_k' if variant == 'prefix_context' else 'complete',
                'top_k': 1 if variant == 'prefix_context' else None, 'cursor': None}
            add('get_neighbors', source, native.get_neighbors)
    return observations, serial


def _answer(request, framing, proof, used, failure=None):
    """Keep path, calibration, failure, and truth-scope results distinct."""
    if failure:
        tool, source, response = failure
        return {'status': 'error', 'provenance': _provenance(used), 'result': {
            'failed_tool': tool, 'error_code': response['error_code'], 'arguments': deepcopy(source),
            'partial_results': list(used), 'coverage': None}, 'continuation_decision': 'defer',
            'blocked_reason': 'The required full-layer search must recover before this exact path task can complete.',
            'summary': 'The required search failed. Retain successful observations and leave unknown path outputs unassessed.'}
    if proof.issue_code:
        return {'status': proof.status, 'provenance': _provenance(used),
            'result': _blocked(request, proof.issue_code, used), 'continuation_decision': 'defer',
            'blocked_reason': 'The exact proposition needs adequate evidence for its requested scope.',
            'summary': 'The supplied observations do not decide this truth request. Preserve the evidence and defer.'}
    if proof.result is None or proof.next_arguments is not None:
        raise ValueError('an exact path or assessed claim remains unresolved')
    missing = proof.status == 'not_found'
    return {'status': proof.status, 'provenance': _provenance(used), 'result': proof.result,
        'continuation_decision': 'defer' if missing else 'continue' if framing == 'step' else 'stop',
        'blocked_reason': 'Every requested endpoint must belong to the multiplex before traversal can be evaluated.' if missing else None,
        'summary': 'The requested scoped path and evidence assessment follow from the visible compatible observations.'}


def _reason(answer, framing):
    result = answer['result']
    if answer['status'] == 'error':
        return ('The required search did not complete. A search limit cannot establish path absence. '
                'Retain every successful partial observation with null path coverage. Defer after the visible retry allowance ends.')
    if 'issue_code' in result:
        return ('The supplied evidence cannot decide the exact proposition in its requested scope. Unsupported does not mean false. '
                'Preserve the supplied observation and defer for adequate scope evidence or a narrower request.')
    parts = ['Use the requested layer, graph version, endpoint order, and native direction.']
    if 'path_found' in result:
        parts.append(f"The path outputs are {result['path_found']}, {result['path_gene_ids']}, and {result['hop_count']} hops.")
        parts.append(f"Global exclusions are {result['missing_gene_ids']}; layer exclusions are {result['not_in_layer_gene_ids']}.")
        parts.append('A missing multiplex endpoint leaves traversal null. Checked layer absence completes a negative result.')
        parts.append('A repeated layer member has zero hops. For distinct endpoints, preserve the minimum length and canonical tied witness.')
    if 'claim_cards' in result:
        for card in result['claim_cards']:
            parts.append(f"For {card['subject']['gene_ids']}, the supported claims are {card['supported_claims']}.")
            if 'claim' in card:
                parts.append(f"The requested claim is {card['claim']} and its support verdict is {card['claim_supported']}.")
            if 'statement_true' in card:
                parts.append(f"The exact requested proposition has statement_true {card['statement_true']}.")
        parts.append('Every other vocabulary item remains unsupported by these observations.')
        parts.append('Keep the closed vocabulary in order and bind each card to its own subject and scope.')
        parts.append('A negative direct edge leaves longer paths possible. A route alone does not prove minimum length or a canonical tie.')
        parts.append('Scoped graph absence does not establish biological noninteraction or absence in other layers or versions.')
        parts.append('Only an explicit truth task receives a truth verdict; support and truth remain separate.')
    parts.append('Defer for the missing traversal prerequisite.' if answer['continuation_decision'] == 'defer' else
                 'The workflow step is complete, so continue.' if framing == 'step' else 'The standalone task is complete, so stop.')
    return ' '.join(parts)


def build_row(view: GraphView, fact: PathFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, index: PathIndex,
              registry: RegistryContext | None = None, question_kind: str = 'path',
              claim: str | None = None, truth_scope: str | None = None, statement: str | None = None,
              membership_cards: bool = False, seed: int = 20261028) -> dict:
    """Construct a source-backed path answer or a declared prerequisite/failure branch."""
    from .s2_11_prompts import render_s2_11_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported path contrast or framing')
    if variant == 'reversed':
        if fact.gene_ids[0] == fact.gene_ids[1]:
            raise VariantUnavailable('a reverse contrast requires distinct endpoints')
        fact = fact_from_genes(view, index, fact.layer_id, tuple(reversed(fact.gene_ids)), seed,
                               split=fact.split, category=fact.category)
    task = _task(view, fact, question_kind, claim, truth_scope, statement, membership_cards)
    raw_direction = fact.direction
    if variant == 'unqualified':
        if view.layers[fact.layer_id].directed:
            raise VariantUnavailable('a directed path needs its explicit source-to-target reference')
        raw_direction = None
    raw_genes = list(fact.gene_ids)
    if variant in ('symbol_context', 'symbol_terminal'):
        if registry is None:
            raise VariantUnavailable('symbol context requires the pinned registry')
        try:
            raw_genes = list(get_symbols(fact.gene_ids, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = {'genes': raw_genes, 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'direction': raw_direction}
    core = render_s2_11_request(contract['_requests'], split=fact.split, variant_index=variant_index, **request, **task)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's211_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, task, user])[:32]
    observations, serial = _context(view, fact, variant, row_id, index, registry, seed)
    if question_kind in ('claim_card', 'claim_support', 'truth') and not observations:
        raise VariantUnavailable('claim-only assessment requires a supplied observation')
    arguments = _arguments(view, fact)
    proof, used = _plan(view, arguments, observations, task)
    if variant in ('route_context', 'longer_route_context', 'tie_route_context', 'prefix_context') and question_kind.startswith('path') and proof.next_arguments is None:
        raise VariantUnavailable('this evidence already proves the exact requested path')
    fault = _fault_kind(variant)
    if fault and proof.next_arguments is None:
        raise VariantUnavailable('the supplied facts settle this task before a failed call')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view), 'observations': observations})},
        {'role': 'user', 'content': user}]
    failure = None
    if proof.next_arguments is not None:
        source, tool = deepcopy(proof.next_arguments), proof.next_tool
        if tool != 'shortest_path':
            raise ValueError('a new full-layer acquisition must use the reviewed path operation')
        serial += 1
        response = (_limit_failure() if fault == 'execution_limit' else _failure(None if fault == 'terminal' else 1)
                    if fault else PathRuntime(view, index=index).shortest_path(**source, evidence_id=f'ev_{row_id}_{serial}'))

        def append(value, thought):
            messages.extend([{'role': 'assistant', 'thinking': thought,
                              'tool_call': {'name': tool, 'arguments': deepcopy(source)}},
                             {'role': 'tool', 'name': tool, 'content': value}])

        append(response, 'The visible evidence does not establish every required path field and canonical tie. '
               'Request the unchanged full-layer operation with the exact ordered endpoints. '
               'Do not add a hop limit, weight threshold, reverse direction, or weak projection.')
        if fault in ('recovered', 'exhausted'):
            serial += 1
            response = (_failure(0) if fault == 'exhausted' else
                PathRuntime(view, index=index).shortest_path(**source, evidence_id=f'ev_{row_id}_{serial}'))
            append(response, 'The visible error authorizes one retry. Repeat the identical tool and arguments. '
                   'Retain earlier successful evidence and stop after recovery or exhaustion.')
        if response['status'] == 'error':
            failure = tool, source, response
        else:
            observations.append(_observation(tool, source, response))
            proof, used = _plan(view, arguments, observations, task)
    answer = _answer(request, framing, proof, used, failure)
    messages.append({'role': 'assistant', 'thinking': _reason(answer, framing), 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.11', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'direction': fact.direction, **{key: getattr(fact, key) for key in LABELS}, 'category': fact.category,
        'framing': framing, 'variant': variant, **task, 'request': request, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][question_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view, fact, kind):
    request = {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'direction': fact.direction}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'missing_genes': 'INVALID_PAIR', 'invalid_pair': 'INVALID_PAIR',
        'missing_scope': 'MISSING_SCOPE', 'missing_direction': 'AMBIGUOUS_PATH_DIRECTION',
        'invalid_direction': 'INVALID_DIRECTION', 'unavailable_direction': 'DIRECTION_UNAVAILABLE',
        'unresolved_identity': 'UNRESOLVED_IDENTITY'}
    if kind not in codes:
        raise ValueError('unsupported path prerequisite diagnostic')
    if kind in ('unknown_layer', 'tool_unknown_layer'):
        request['layer_id'] = 'TFs:missing_context'
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_path_multiplex'
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_pair':
        request['genes'] = [*fact.gene_ids, fact.gene_ids[0]]
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'missing_direction':
        request['layer_id'] = next((layer for layer in view.layer_order if view.layers[layer].directed), None)
        if request['layer_id'] is None:
            raise VariantUnavailable('no directed layer requires an explicit orientation')
        request['direction'] = None
    elif kind == 'invalid_direction':
        request['direction'] = {'direction': 'out'}
    elif kind == 'unavailable_direction':
        request['direction'] = 'undirected' if view.layers[fact.layer_id].directed else 'source_to_target'
    elif kind == 'unresolved_identity':
        request['genes'] = ['HER2', fact.gene_ids[-1]]
    status = ('ambiguous' if kind == 'missing_direction' else 'not_found' if kind in
        ('unknown_layer', 'tool_unknown_layer', 'unknown_multiplex', 'unresolved_identity') else 'invalid_input')
    return request, status, codes[kind]


def build_diagnostic(view: GraphView, fact: PathFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *, kind: str = 'missing_genes',
                     registry: RegistryContext | None = None, seed: int = 20261028) -> dict:
    """Retain raw path prerequisites and current-symbol failures without guesses."""
    from .s2_11_prompts import render_s2_11_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    wording = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    core = render_s2_11_diagnostic_request(contract['_requests'], split=fact.split, kind=wording,
        variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's211_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32]
    active, used, rejected = _active_context(view), [], None
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    if kind == 'tool_unknown_layer':
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
                    'gene_ids': request['genes'], 'direction': request['direction']}
        response = PathRuntime(view).shortest_path(**rejected, evidence_id='unused_rejected_path_scope')
        messages.extend([{'role': 'assistant', 'thinking': 'The layer catalogue is withheld. Ask the native resolver for this explicit scope.',
                          'tool_call': {'name': 'shortest_path', 'arguments': deepcopy(rejected)}},
                         {'role': 'tool', 'name': 'shortest_path', 'content': response}])
    elif kind == 'unresolved_identity':
        if registry is None:
            raise ValueError('the current-symbol negative requires the pinned registry')
        response = _identity_rejection(registry, request['genes'][0], f'ev_{row_id}_identity')
        used.append(_identity_partial(request['genes'][0], response))
        messages.extend([{'role': 'assistant', 'thinking': 'Resolve the exact supplied symbol before a graph query. Do not infer an alias.',
                          'tool_call': {'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': request['genes'][0]}}},
                         {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}])
    answer = {'status': status, 'provenance': _provenance(used), 'result': _blocked(request, code, used, rejected),
        'continuation_decision': 'defer', 'blocked_reason': 'Resolve the endpoint identities, graph scope, pair, or native orientation.',
        'summary': 'A path prerequisite is unresolved. Preserve the original request without an invented path or absence claim.'}
    messages.append({'role': 'assistant', 'thinking': f'The original request is {request}. Its prerequisite fails with {code}. '
        'Preserve literal endpoint order, repeated positions, direction values, and successful registry evidence. Defer.', 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.11', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'direction': fact.direction, **{key: None for key in LABELS}, 'category': fact.category,
        'framing': framing, 'variant': kind, 'question_kind': 'path', 'claim': None, 'truth_scope': None,
        'statement': None, 'membership_cards': False, 'request': request, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: PathIndex | None = None) -> None:
    """Check every native source record, then replay the source-blind path proof."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_11_prompts import render_s2_11_request, render_s2_11_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.11 requires one user turn and an explicit harness')
    for message in messages:
        fields = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != fields or message['role'] == 'assistant' and (not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('unexpected message fields or absent assistant reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('the system or native tools differ from the path overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.11'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'
            or meta['framing'] not in FRAMINGS):
        raise ValueError('the path row differs from its build identity')
    index = index or PathIndex(view)
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['selection_seed'],
                           split=meta['split'], category=meta['category'])
    if fact.split_unit != meta['split_unit'] or fact.direction != meta['direction']:
        raise ValueError('the ordered query differs from its fixed subject split or direction')
    answer = messages[-1]['content']
    if (list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or not isinstance(answer['summary'], str) or not answer['summary'].strip()
            or ((answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']))
            or answer['blocked_reason'] is not None and (not isinstance(answer['blocked_reason'], str)
                or not answer['blocked_reason'].strip())):
        raise ValueError('the final envelope, free summary, or blocker is invalid')
    diagnostic = meta['variant'] in DIAGNOSTICS
    request = meta['request']
    if set(request) != {'genes', 'multiplex_id', 'layer_id', 'direction'}:
        raise ValueError('the raw path request must retain exactly four fields')
    task = {key: meta[key] for key in ('question_kind', 'claim', 'truth_scope', 'statement', 'membership_cards')}
    if task != _task(view, fact, **task):
        raise ValueError('the task metadata changes an explicit support or truth request')
    wording = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
    templates = (contract['_requests']['diagnostic_requests'][meta['split']][wording] if diagnostic
                 else contract['_requests'][meta['split']][meta['question_kind']])
    for key, size in (('wrapper_index', len(bank[meta['split']][meta['framing']])), ('request_template_index', len(templates))):
        if type(meta[key]) is not int or not 0 <= meta[key] < size:
            raise ValueError('the request must select its frozen wording and wrapper')
    if diagnostic:
        expected_request, status, code = _diagnostic_request(view, fact, meta['variant'])
        if canonical_json(request) != canonical_json(expected_request) or any(meta[key] is not None for key in LABELS):
            raise ValueError('the prerequisite changes its raw request or private labels')
        if task != {'question_kind': 'path', 'claim': None, 'truth_scope': None, 'statement': None, 'membership_cards': False}:
            raise ValueError('a prerequisite diagnostic cannot assess unrequested claims')
        core = render_s2_11_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording,
            variant_index=meta['request_template_index'], **request)
    else:
        if meta['variant'] not in VARIANTS or any(canonical_json(meta[key]) != canonical_json(getattr(fact, key)) for key in LABELS):
            raise ValueError('the source labels or declared contrast changed')
        expected_genes = list(fact.gene_ids)
        if meta['variant'] in ('symbol_context', 'symbol_terminal'):
            if registry is None:
                raise ValueError('the supplied symbols require the pinned registry')
            expected_genes = list(get_symbols(fact.gene_ids, registry))
        expected_direction = None if meta['variant'] == 'unqualified' else fact.direction
        if (request != {'genes': expected_genes, 'multiplex_id': view.multiplex_id,
                'layer_id': fact.layer_id, 'direction': expected_direction}
                or meta['variant'] == 'unqualified' and view.layers[fact.layer_id].directed):
            raise ValueError('the raw endpoint identities, order, or native orientation changed')
        core = render_s2_11_request(contract['_requests'], split=meta['split'],
            variant_index=meta['request_template_index'], **request, **task)
    if messages[2] != {'role': 'user', 'content': bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)}:
        raise ValueError('the visible request differs from its frozen task wording')
    identity = ([contract['contract_version'], view.graph_version, messages[2]['content'], meta['variant']]
        if diagnostic else [contract['contract_version'], view.graph_version, asdict(fact),
            meta['framing'], meta['variant'], task, messages[2]['content']])
    if row['row_id'] != 's211_' + digest(identity)[:32]:
        raise ValueError('the row ID does not bind its source fact, task, wording, and contrast')
    context = json.loads(messages[1]['content'])
    active = _active_context(view)
    if meta['variant'] == 'tool_unknown_layer':
        active.pop('layers')
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != active:
        raise ValueError('the visible graph context changed')
    if diagnostic:
        if context['observations']:
            raise ValueError('a local prerequisite diagnostic cannot invent supplied evidence')
        used, rejected, middle = [], None, []
        if meta['variant'] == 'tool_unknown_layer':
            rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
                        'gene_ids': request['genes'], 'direction': request['direction']}
            response = PathRuntime(view).shortest_path(**rejected, evidence_id='unused_rejected_path_scope')
            middle = [{'role': 'assistant', 'tool_call': {'name': 'shortest_path', 'arguments': rejected}},
                      {'role': 'tool', 'name': 'shortest_path', 'content': response}]
        elif meta['variant'] == 'unresolved_identity':
            if registry is None:
                raise ValueError('the unresolved identity requires the pinned registry')
            response = _identity_rejection(registry, request['genes'][0], f"ev_{row['row_id']}_identity")
            used.append(_identity_partial(request['genes'][0], response))
            middle = [{'role': 'assistant', 'tool_call': {'name': 'lookup_human_gene_symbol',
                      'arguments': {'gene_symbol': request['genes'][0]}}},
                      {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}]
        if len(messages) != 4 + len(middle) or any(any(canonical_json(actual[key]) != canonical_json(value)
                for key, value in expected.items()) for actual, expected in zip(messages[3:-1], middle)):
            raise ValueError('the prerequisite changes its exact calls or responses')
        expected = _blocked(request, code, used, rejected)
        if (answer['status'] != status or answer['result'] != expected or answer['provenance'] != _provenance(used)
                or answer['continuation_decision'] != 'defer'):
            raise ValueError('the prerequisite diagnostic changes its structured result')
        return
    expected_context, _ = _context(view, fact, meta['variant'], row['row_id'], index, registry, meta['selection_seed'])
    if canonical_json(context['observations']) != canonical_json(expected_context):
        raise ValueError('the declared contrast differs from its complete source observations')
    available, evidence_ids = list(context['observations']), {}
    identities = [item for item in available if item['operation'] == 'lookup_human_gene_symbol']
    if identities:
        validate_identity_observations(identities, registry)
        if [item['result']['gene_id'] for item in identities] != list(dict.fromkeys(fact.gene_ids)):
            raise ValueError('the identity mapping changed the ordered endpoint domain')

    def verify(observation):
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected saved observation fields')
        for records in observation['provenance'].values():
            for record in records:
                identity = record['evidence_id']
                if identity in evidence_ids and canonical_json(evidence_ids[identity]) != canonical_json(observation):
                    raise ValueError('one evidence ID names different acquisitions')
                evidence_ids[identity] = observation
        operation, source = observation['operation'], observation['arguments']
        if operation == 'lookup_human_gene_symbol':
            return
        if len(observation['provenance']['graph']) != 1:
            raise ValueError('each native acquisition must retain one original graph record')
        record = observation['provenance']['graph'][0]
        checked = deepcopy(observation)
        if record['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or record['graph_version'] != '0' * 64:
                raise ValueError('an undeclared stale observation cannot supply evidence')
            checked['provenance']['graph'][0]['graph_version'] = view.graph_version
        runtimes = {'shortest_path': PathRuntime(view, index=index).shortest_path,
            'get_nodes_by_layer': NodesByLayerRuntime(view).get_nodes_by_layer,
            'get_neighbors': NeighborRuntime(view, index=index.native_index).get_neighbors,
            'get_edge': view.get_edge,
            'induce_subgraph': SubgraphRuntime(view, index=SubgraphIndex(view, index.native_index)).induce_subgraph}
        if operation not in runtimes:
            raise ValueError('the path corpus contains an unsupported native observation')
        expected = _observation(operation, source, runtimes[operation](**source, evidence_id=record['evidence_id']))
        if canonical_json(checked) != canonical_json(expected):
            raise ValueError('an attributed observation differs from the exact native source')

    for observation in available:
        verify(observation)
    arguments = _arguments(view, fact)
    previous_call = previous_response = None
    responses, failure = [], None
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, used = _plan(view, arguments, available, task)
        call = message['tool_call']
        if proof.next_arguments is None or call != {'name': proof.next_tool, 'arguments': proof.next_arguments}:
            raise ValueError('the call changes the required native path, scope, or endpoint order')
        tool_message = messages[position + 1]
        if tool_message['role'] != 'tool' or tool_message['name'] != call['name']:
            raise ValueError('the tool response does not match its required call')
        response = tool_message['content']
        if previous_response and previous_response['status'] == 'error' and (previous_response != _failure(1) or call != previous_call):
            raise ValueError('the retry changes arguments or lacks a positive visible allowance')
        responses.append(response)
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0), _limit_failure()):
                raise ValueError('the failed search contains an undeclared fault or invented evidence')
            failure = call['name'], call['arguments'], response
        else:
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
            failure = None
        previous_call, previous_response = call, response
    fault = _fault_kind(meta['variant'])
    error_responses = [response for response in responses if response['status'] == 'error']
    expected_errors = ([_limit_failure()] if fault == 'execution_limit' else [_failure(None)] if fault == 'terminal' else
        [_failure(1), _failure(0)] if fault == 'exhausted' else [_failure(1)] if fault == 'recovered' else [])
    if error_responses != expected_errors or fault in ('recovered', 'exhausted') and len(responses) != 2:
        raise ValueError('the visible failure schedule differs from the declared contrast')
    if previous_response == _failure(1):
        raise ValueError('the authorized retry remains unused')
    proof, used = _plan(view, arguments, available, task)
    expected = _answer(request, meta['framing'], proof, used, failure)
    if any(canonical_json(answer[key]) != canonical_json(expected[key]) for key in
            ('status', 'provenance', 'result', 'continuation_decision')):
        raise ValueError('the final answer changes the exact path, claim assessment, provenance, or task decision')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, source_window: int | None = None, seed: int = 20261028,
                 contract_path: Path = CONTRACT_PATH, source_manifest: Path | None = None) -> dict:
    """Export S2.11 after exact source, evidence, wording, and token checks."""
    from .s2_11_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, source_window=source_window,
                  seed=seed, contract_path=contract_path, source_manifest=source_manifest)
