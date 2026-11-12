"""Generate the 41 reviewed S2.1 behaviors from new native source queries.

This is a separate verification generator. It leaves the published initial
edge corpus unchanged. Case bindings are explicit; source selection and graph
responses are independent of wording. Production quotas and training export
remain the aggregate builder's responsibility.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from .s1_prompt_banks import load_s2_prompt_banks, render_s2_prompt
from .s2_2_identity import (get_symbols, identity_observations, load_registry_context,
                            validate_identity_observations)
from .s2_corpus import ROOT, SCHEMA_VERSION, canonical_json, digest, _pair_group, _failure, _outcome_reason
from .s2_neighbors import NeighborIndex, NeighborRuntime
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_11_evidence import _inspect, _provenance
from .s2_13_evidence import _records, _ordinary
from .sources import file_sha256
from ..world_model_training import _validate_v3_message_sequence


CONTRACT_PATH = Path(__file__).with_name('s2_1_production_v1.json')

# source predicate, evidence/fault behavior, and reviewed framing. These IDs
# denote catalog cases, never positions in a generated variant list.
_SPECS = (
    ('upos', 'lookup', 'direct'), ('upos', 'lookup', 'step'),
    ('uneg', 'lookup', 'direct'), ('uneg', 'lookup', 'step'),
    ('identity', 'symbol_context', 'direct'), ('outside_layer', 'lookup', 'direct'),
    ('outside_multiplex', 'lookup', 'step'), ('upos', 'complete_positive', 'step'),
    ('upos_omit', 'bounded_omission', 'direct'), ('upos_top', 'bounded_positive', 'step'),
    ('uneg', 'complete_negative', 'direct'), ('uneg_bounded', 'bounded_omission', 'step'),
    ('upos', 'wrong_layer', 'direct'), ('upos', 'distractor_then_edge', 'direct'),
    ('identity', 'symbol_terminal', 'direct'), ('identity', 'symbol_lookup', 'step'),
    ('upos', 'recovered', 'direct'), ('upos', 'exhausted', 'step'),
    ('upos_omit', 'bounded_terminal', 'direct'), ('uneg', 'complete_omission', 'direct'),
    ('dpos', 'lookup', 'direct'), ('dreverse', 'lookup', 'step'),
    ('dreverse', 'reverse_context', 'direct'), ('dpos_omit', 'complete_positive', 'step'),
    ('dpos', 'incoming_positive', 'direct'), ('dsink', 'complete_negative', 'direct'),
    ('dpos_omit', 'bounded_omission', 'direct'), ('upos', 'direction_conflict', 'direct'),
    ('reciprocal', 'lookup', 'direct'), ('reciprocal_reverse', 'lookup', 'step'),
    ('dpos', 'directionality', 'direct'), ('dreverse', 'directionality', 'step'),
    ('reciprocal', 'directionality', 'direct'), ('dneither', 'directionality', 'step'),
    ('dpos', 'direction_context', 'direct'), ('dpos', 'direction_terminal', 'direct'),
    ('upos', 'directionality', 'direct'), ('dpos', 'any_direction', 'direct'),
    ('dreverse', 'both_directions', 'step'), ('directed_missing', 'directionality', 'step'),
    ('dpos', 'direction_recovered', 'step'),
)
CASE_BINDINGS = MappingProxyType({f'S2.1.{n}': {
    'source_predicate': source, 'behavior': behavior, 'framing': framing,
} for n, (source, behavior, framing) in enumerate(_SPECS, 1)})


@dataclass(frozen=True)
class CaseFact:
    """A selected native query, with its fixed unordered-pair split identity."""

    layer_id: str
    gene_ids: tuple[str, str]
    split: str
    split_unit: str


def load_build_contract(path=CONTRACT_PATH):
    """Load separate pinned S2.1 generation inputs and identity-only registry."""
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION
            or contract['implemented_families'] != ['S2.1']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.1 production-verification contract')
    contract['_bank'] = load_s2_prompt_banks(contract['prompt_bank'])
    request_path = ROOT / contract['request_bank']['path']
    if file_sha256(request_path) != contract['request_bank']['sha256']:
        raise ValueError('S2.1 request wording pin changed')
    contract['_requests'] = json.loads(request_path.read_text())
    contract['_registry'] = load_registry_context(contract['registry_index'])
    return contract


def _arguments(view, fact, genes=None):
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
            'gene_ids': list(fact.gene_ids if genes is None else genes)}


def _observation(operation, arguments, response):
    return {'operation': operation, 'arguments': deepcopy(arguments),
            **deepcopy(response)}


def _proof(view, arguments, observations):
    """Derive an ordinary edge using visible evidence and representation metadata."""
    # The evidence planner cannot read private source inventories or adjacency.
    view = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layers={layer: SimpleNamespace(directed=view.layers[layer].directed) for layer in view.layer_order},
        layer_order=view.layer_order)
    records = _records(view, observations)
    path = {**arguments, 'direction': 'source_to_target'
            if view.layers[arguments['layer_id']].directed else 'undirected'}
    facts = _inspect(view, path, records)
    result = _ordinary(view, arguments, records, facts)
    matching = [item for item in records if item['operation'] == 'get_edge'
                and item['arguments'] == arguments]
    if matching:
        used = matching
    else:
        used = list(facts.used)
    identities = [item for item in records if item['operation'].startswith('lookup_human_')]
    return result, tuple(item for item in records if item in [*identities, *used])


def _pair_coverage(view, arguments, results):
    pair = arguments['gene_ids']
    requested = [pair, pair[::-1]] if view.layers[arguments['layer_id']].directed else [pair]
    checked = [requested[index] for index, item in enumerate(results)
               if item['edge_exists'] is not None]
    return {'operation': 'edge_checks', 'requested_pairs': deepcopy(requested),
            'checked_pairs': deepcopy(checked),
            'unchecked_pairs': [p for p in requested if p not in checked]}


def _direction_result(view, arguments, results, query_kind):
    """Keep unknown orientations null; early witnesses settle only their request."""
    directed = view.layers[arguments['layer_id']].directed
    values = [item['edge_exists'] for item in results]
    complete = len(results) == (2 if directed else 1) and all(v is not None for v in values)
    any_edge = True if True in values else False if complete else None
    both = False if False in values else True if complete else None
    directionality = None
    if complete:
        directionality = ('undirected' if not directed and any_edge else
            'neither' if not any_edge else 'both' if both else
            'forward_only' if values[0] else 'reverse_only')
    if not directed:
        both = None
    return {'query_kind': query_kind, 'gene_ids': list(arguments['gene_ids']),
            'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
            'directed': directed, 'directionality': directionality,
            'any_edge_exists': any_edge, 'both_directions_exist': both,
            'edge_results': deepcopy(results),
            'coverage': _pair_coverage(view, arguments, results)}


def _request(view, fact, case_id, contract, index):
    binding = CASE_BINDINGS[case_id]
    behavior = binding['behavior']
    kind = ('any_direction' if behavior == 'any_direction' else
            'both_directions' if behavior == 'both_directions' else
            'directionality' if behavior.startswith('direction_') or behavior == 'directionality' else
            'conflict' if behavior == 'direction_conflict' else
            'directed' if view.layers[fact.layer_id].directed else 'undirected')
    # The representation conflict is deliberately prior to the generic prefix.
    if behavior == 'direction_conflict':
        kind = 'conflict'
    genes = (get_symbols(fact.gene_ids, contract['_registry'])
             if behavior.startswith('symbol_') else fact.gene_ids)
    templates = contract['_requests'][fact.split][kind]
    core = templates[index % len(templates)].format(
        first=genes[0], second=genes[1], layer=fact.layer_id, multiplex=view.multiplex_id)
    return render_s2_prompt(contract['_bank'], split=fact.split,
        framing=binding['framing'], variant_index=index, request=core)


def _context(view, fact, case_id, row_id, index, registry, seed=20261028):
    """Acquire honest saved context; partial lists retain original collection state."""
    behavior = CASE_BINDINGS[case_id]['behavior']
    arguments = _arguments(view, fact)
    observations = []
    runtime = NeighborRuntime(view, index=index)

    def edge(genes=None, layer=None):
        selected = _arguments(view, fact, genes)
        selected['layer_id'] = layer or fact.layer_id
        observations.append(_observation('get_edge', selected, view.get_edge(
            **selected, evidence_id=f'ev_{row_id}_saved_{len(observations)}')))

    if behavior in ('symbol_context', 'symbol_terminal'):
        observations.extend(identity_observations(fact.gene_ids, row_id, registry))
    if behavior in ('complete_positive', 'complete_negative', 'complete_omission',
                    'bounded_omission', 'bounded_positive', 'bounded_terminal', 'incoming_positive'):
        incoming = behavior == 'incoming_positive'
        direction = ('in' if incoming else 'out') if view.layers[fact.layer_id].directed else 'undirected'
        source = fact.gene_ids[1] if incoming else fact.gene_ids[0]
        bounded = behavior.startswith('bounded_')
        selected = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
                    'gene_id': source, 'direction': direction,
                    'mode': 'top_k' if bounded else 'complete', 'top_k': 1 if bounded else None,
                    'cursor': None}
        observations.append(_observation('get_neighbors', selected,
            runtime.get_neighbors(**selected, evidence_id=f'ev_{row_id}_neighbors')))
        if behavior == 'complete_negative':
            observations.append(_observation('get_nodes_by_layer', arguments,
                NodesByLayerRuntime(view).get_nodes_by_layer(**arguments,
                    evidence_id=f'ev_{row_id}_memberships')))
    if behavior == 'wrong_layer':
        other = next((layer for layer in view.layer_order if layer != fact.layer_id), None)
        if other is None:
            raise ValueError('wrong-layer case needs a second qualified layer')
        edge(layer=other)
    if behavior == 'distractor_then_edge':
        other = next(gene for gene in sorted(view.layers[fact.layer_id].nodes)
                     if gene not in fact.gene_ids
                     and _pair_group((fact.gene_ids[0], gene), seed)[1] == fact.split)
        edge([fact.gene_ids[0], other])
        edge()
    if behavior == 'reverse_context':
        edge(fact.gene_ids[::-1])
    if behavior in ('direction_context', 'direction_terminal', 'direction_recovered'):
        edge()
        if behavior == 'direction_context':
            edge(fact.gene_ids[::-1])
    return observations


def _answer(status, provenance, result, framing, reason=None):
    decision = ('continue' if framing == 'step' else 'stop') if status == 'resolved' else 'defer'
    return {'status': status, 'provenance': provenance, 'result': result,
            'continuation_decision': decision,
            'blocked_reason': reason if decision == 'defer' else None,
            'summary': 'The visible evidence supplies the scoped result.' if decision != 'defer'
                       else 'The requested operation needs the stated prerequisite or backend recovery.'}


def _run(view, fact, case_id, observations, row_id, registry):
    """Plan a fresh trace from visible context, acquiring only missing evidence."""
    behavior = CASE_BINDINGS[case_id]['behavior']
    framing = CASE_BINDINGS[case_id]['framing']
    arguments = _arguments(view, fact)
    available = deepcopy(observations)
    messages, results, used = [], [], []
    pairwide = case_id.split('.')[-1] in tuple(str(n) for n in range(31, 42))

    def call(name, selected, response, reason):
        messages.extend([{'role': 'assistant', 'thinking': reason,
                          'tool_call': {'name': name, 'arguments': deepcopy(selected)}},
                         {'role': 'tool', 'name': name, 'content': deepcopy(response)}])
        if response['status'] != 'error':
            available.append(_observation(name, selected, response))

    if behavior == 'direction_conflict':
        result = {'issue_code': 'DIRECTION_UNAVAILABLE', 'request': {
            'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
            'layer_id': fact.layer_id, 'direction': 'source_to_target'}, 'partial_results': []}
        return messages, _answer('invalid_input', {'registry': [], 'graph': []}, result,
            framing, 'The registered layer is undirected; the requested ordered representation is unavailable.')
    if behavior == 'symbol_lookup':
        for observation in identity_observations(fact.gene_ids, row_id, registry):
            response = {key: observation[key] for key in ('status', 'provenance', 'result')}
            call(observation['operation'], observation['arguments'], response,
                 f"Resolve {observation['arguments']['gene_symbol']} through the pinned current-symbol registry before constructing the graph query.")

    pairs = [fact.gene_ids, fact.gene_ids[::-1]] if pairwide and view.layers[fact.layer_id].directed else [fact.gene_ids]
    for ordinal, pair in enumerate(pairs):
        selected = _arguments(view, fact, pair)
        ordinary, proof_used = _proof(view, selected, available)
        if ordinary is None:
            terminal = behavior in ('symbol_terminal', 'bounded_terminal', 'exhausted', 'direction_terminal')
            recover = behavior in ('recovered', 'direction_recovered')
            inject = terminal or recover
            sufficiency = ('No compatible observation is supplied.' if not available else
                'The saved bounded omission leaves unseen neighbors possible.' if behavior in ('bounded_omission', 'bounded_terminal') else
                'The complete omission does not establish the other endpoint membership.' if behavior == 'complete_omission' else
                'The saved arrow has the opposite orientation.' if behavior == 'reverse_context' else
                'The available records concern a different layer or omit the required orientation or memberships.')
            reason = (f"The required query is {pair[0]} to {pair[1]} in {fact.layer_id}. "
                      f'{sufficiency} Acquire both memberships and the exact native edge result.')
            if inject:
                allowance = 1 if behavior in ('recovered', 'direction_recovered', 'exhausted') else None
                failure = _failure(allowance)
                call('get_edge', selected, failure, reason)
                if allowance:
                    response = (_failure(0) if behavior == 'exhausted' else view.get_edge(
                        **selected, evidence_id=f'ev_{row_id}_edge_{ordinal}'))
                    call('get_edge', selected, response,
                         'The error explicitly grants one retry. Repeat the identical tool and every argument, then stop retrying.')
                    failure = response if response['status'] == 'error' else None
                if failure is not None:
                    partials = []
                    # Retain original acquisition objects, including useful incomplete context.
                    for observation in available:
                        if observation not in partials:
                            partials.append(observation)
                    result = {'failed_tool': 'get_edge', 'error_code': failure['error_code'],
                              'arguments': selected, 'partial_results': partials,
                              'coverage': _pair_coverage(view, arguments, results) if pairwide else None}
                    return messages, _answer('error', _provenance(partials), result, framing,
                        'The required edge backend did not recover within its visible retry allowance.')
            else:
                response = view.get_edge(**selected, evidence_id=f'ev_{row_id}_edge_{ordinal}')
                call('get_edge', selected, response, reason)
            ordinary, proof_used = _proof(view, selected, available)
            if ordinary is None:
                raise ValueError('a successful edge acquisition did not establish the requested result')
        for observation in proof_used:
            if observation not in used:
                used.append(observation)
        results.append(ordinary)
        if ordinary['edge_exists'] is None:
            break
        if behavior == 'any_direction' and ordinary['edge_exists']:
            break
        if behavior == 'both_directions' and not ordinary['edge_exists']:
            break
    result = _direction_result(view, arguments, results,
        behavior if behavior in ('any_direction', 'both_directions') else 'directionality') if pairwide else results[0]
    status = 'not_found' if any(item['missing_gene_ids'] for item in results) else 'resolved'
    return messages, _answer(status, _provenance(used), result, framing,
        'A required canonical endpoint is outside the active multiplex.')


def _reason(view, fact, case_id, observations, steps, answer):
    """Explain actual observed values, proof limits, and the requested completion."""
    behavior = CASE_BINDINGS[case_id]['behavior']
    framing = CASE_BINDINGS[case_id]['framing']
    result = answer['result']
    subject = f'{fact.gene_ids[0]} and {fact.gene_ids[1]} in {fact.layer_id} of {view.multiplex_id}'
    if answer['status'] == 'error':
        attempts = [item['content'] for item in steps if item['role'] == 'tool' and item['name'] == 'get_edge']
        allowance = attempts[-1].get('message', '')
        partials = result['partial_results']
        retained = ', '.join(item['operation'] for item in partials) or 'none'
        return (f"The required {result['failed_tool']} lookup for {subject} failed. "
                f'The final runtime message is {allowance!r}; no further retry is permitted. '
                f'Retain {len(partials)} successful attributed observations ({retained}) in their original form. '
                'They do not supply the failed edge or missing orientation. Preserve checked and unchecked pairs when applicable; '
                'leave uncomputed graph values absent and defer for backend recovery.')
    if behavior == 'direction_conflict':
        return (f'The manifest declares {fact.layer_id} undirected. The requested ordered arrow for {subject} '
                'cannot be represented by that source. Preserve the literal source/target request under '
                'DIRECTION_UNAVAILABLE, acquire no unrelated unordered edge, and defer until a directed representation is supplied.')
    if 'edge_results' in result:
        values = '; '.join(f"{item['gene_ids'][0]} to {item['gene_ids'][1]}: edge_exists={item['edge_exists']}, weight={item['weight']}"
                           for item in result['edge_results'])
        coverage = result['coverage']
        if behavior == 'any_direction':
            conclusion = 'One checked positive proves the requested any-direction existence; the reverse predicate stays unknown.'
        elif behavior == 'both_directions':
            conclusion = 'One checked negative refutes the requested conjunction; the other predicate stays unknown.'
        elif answer['status'] == 'not_found':
            conclusion = 'Missing multiplex membership leaves every requested edge predicate unevaluated, with null classification and existence fields.'
        else:
            conclusion = f"The required native checks establish directionality={result['directionality']}; preserve each independent weight."
        ending = ('The unavailable endpoint blocks this operation, so defer.' if answer['status'] == 'not_found' else
                  'This completed step feeds the stated later workflow, so continue.' if framing == 'step' else
                  'The standalone direction question is complete, so stop.')
        return (f'The observations for {subject} give {values}. {conclusion} '
                f"Coverage retains {len(coverage['checked_pairs'])} checked and {len(coverage['unchecked_pairs'])} unchecked requested pairs. {ending}")
    context = []
    for observation in observations:
        if observation['operation'] == 'get_neighbors':
            collection = observation['result']['collection']
            context.append(f"The saved {observation['arguments']['direction']} list returns {collection['returned_count']} of {collection['total_count']} neighbors, complete={collection['complete']}.")
    if behavior in ('complete_positive', 'incoming_positive', 'bounded_positive'):
        context.append('The visible matching native neighbor entry supplies the positive pair and its weight, including both endpoint memberships, without another call.')
    elif behavior == 'complete_negative':
        context.append('Complete native omission and separately visible endpoint membership establish the negative; a directional zero does not imply node absence.')
    elif behavior in ('bounded_omission', 'complete_omission', 'wrong_layer', 'reverse_context'):
        context.append('The saved evidence was insufficient for this exact pair, scope, orientation, or membership; the successful exact lookup supplies the final result.')
    elif behavior.startswith('symbol_'):
        context.append('Both symbols have unique authoritative mappings; keep each original registry acquisition alongside the graph proof.')
    if behavior == 'recovered':
        context.append('The explicitly authorized identical retry recovered; no further retry is needed.')
    return ' '.join([*context, _outcome_reason(result, framing)])


def _candidates(view, index, registry, predicate):
    """Enumerate real source features in increasing complete-context cost."""
    directed = predicate.startswith('d') or predicate.startswith('reciprocal')
    for layer_id, layer in view.layers.items():
        if layer.directed != directed:
            continue
        direction = 'out' if directed else 'undirected'
        genes = sorted(layer.nodes, key=lambda gene: (len(index.neighbors(layer_id, gene, direction)), gene))
        members = sorted(layer.nodes)
        if predicate in ('outside_layer', 'outside_multiplex', 'directed_missing'):
            pool = (sorted(view.multiplex_nodes - layer.nodes) if predicate == 'outside_layer'
                    else sorted(set(registry.symbols_by_id) - view.multiplex_nodes))
            for source in genes:
                for target in pool[:128]:
                    yield layer_id, (source, target)
            continue
        for source in genes:
            rows = [(gene, weight) for gene, weight in index.neighbors(layer_id, source, direction) if gene != source]
            if (source, source) in layer.edges:
                # Reviewed list contrasts concern distinct pair predicates. A
                # self-loop could occupy top-one and change that contrast.
                continue
            neighbors = {gene for gene, _ in rows}
            if predicate in ('uneg', 'uneg_bounded', 'dneither', 'dsink'):
                if predicate == 'dsink' and rows:
                    continue
                if predicate == 'uneg_bounded' and len(rows) < 2:
                    continue
                for target in members:
                    if target == source or target in neighbors:
                        continue
                    if predicate == 'dneither' and (target, source) in layer.edges:
                        continue
                    if predicate == 'dsink' and (target, source) not in layer.edges:
                        continue
                    yield layer_id, (source, target)
                continue
            if (predicate.endswith('_omit') or predicate == 'upos_top') and len(rows) < 2:
                continue
            for ordinal, (target, _) in enumerate(rows):
                if predicate.endswith('_omit') and ordinal == 0:
                    continue
                if predicate == 'upos_top' and ordinal != 0:
                    continue
                if predicate == 'identity':
                    try:
                        get_symbols((source, target), registry)
                    except ValueError:
                        continue
                if predicate in ('dpos', 'dreverse'):
                    if (target, source) in layer.edges or source < target:
                        continue
                if predicate.startswith('reciprocal') and (target, source) not in layer.edges:
                    continue
                pair = (target, source) if predicate in ('dreverse', 'reciprocal_reverse') else (source, target)
                yield layer_id, pair


def _assert_source_predicate(view, fact, predicate, index, registry):
    """Check the numbered case's factual precondition independently of its trace."""
    layer, (first, second) = view.layers[fact.layer_id], fact.gene_ids
    pair = (first, second) if layer.directed else tuple(sorted(fact.gene_ids))
    exists = pair in layer.edges
    reverse = (second, first) in layer.edges
    missing = set(fact.gene_ids) - view.multiplex_nodes
    outside = (set(fact.gene_ids) & view.multiplex_nodes) - layer.nodes
    directed = predicate.startswith('d') or predicate.startswith('reciprocal')
    if layer.directed != directed:
        raise ValueError('catalog case changed its native representation')
    if predicate in ('outside_multiplex', 'directed_missing'):
        valid = len(missing) == 1 and not outside
    elif predicate == 'outside_layer':
        valid = not missing and len(outside) == 1
    elif missing or outside:
        valid = False
    elif predicate in ('uneg', 'uneg_bounded', 'dneither', 'dsink'):
        valid = not exists and (predicate != 'dneither' or not reverse)
        if predicate == 'uneg_bounded':
            valid = valid and len(index.neighbors(fact.layer_id, first, 'undirected')) > 1
        if predicate == 'dsink':
            valid = valid and not index.neighbors(fact.layer_id, first, 'out') and reverse
    elif predicate == 'dreverse':
        valid = not exists and reverse and second > first
    else:
        valid = exists
        if predicate == 'dpos':
            valid = valid and not reverse and first > second
        if predicate.startswith('reciprocal'):
            valid = valid and reverse
        if predicate.endswith('_omit'):
            native = index.neighbors(fact.layer_id, first, 'out' if layer.directed else 'undirected')
            valid = valid and len(native) > 1 and native[0][0] != second
        if predicate == 'upos_top':
            native = index.neighbors(fact.layer_id, first, 'undirected')
            valid = valid and len(native) > 1 and native[0][0] == second
        if predicate == 'identity':
            get_symbols(fact.gene_ids, registry)
    if not valid:
        raise ValueError('query does not meet the catalog case source predicate')


def build_case_rows(view, contract, *, split, case_ids=None, instances_per_case=1):
    """Generate distinct native witnesses for each requested numbered case.

    Unattainable cells raise instead of silently disappearing. The fixed pair
    hash is never changed to make a case fit a held-out split.
    """
    if split not in ('train', 'validation', 'test') or type(instances_per_case) is not int or instances_per_case < 1:
        raise ValueError('a fixed split and positive witness count are required')
    case_ids = tuple(CASE_BINDINGS if case_ids is None else case_ids)
    if any(case_id not in CASE_BINDINGS for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError('case IDs must be unique reviewed S2.1 IDs')
    registry = contract.get('_registry') or load_registry_context(contract['registry_index'])
    index = NeighborIndex(view)
    selected, rows = {}, []
    seed = contract.get('seed', 20261028)
    for case_id in case_ids:
        binding = CASE_BINDINGS[case_id]
        predicate = binding['source_predicate']
        if predicate not in selected:
            facts, seen = [], set()
            for layer, pair in _candidates(view, index, registry, predicate):
                unit, actual_split = _pair_group(pair, seed)
                key = (layer, pair)
                if actual_split == split and key not in seen:
                    seen.add(key)
                    facts.append(CaseFact(layer, pair, split, unit))
                    if len(facts) == instances_per_case:
                        break
            if len(facts) < instances_per_case:
                raise ValueError(f'no sufficient native source population: {predicate}/{split}')
            selected[predicate] = facts
        for ordinal, fact in enumerate(selected[predicate]):
            user, wrapper = _request(view, fact, case_id, contract, ordinal)
            row_id = 's21p_' + digest([case_id, fact.layer_id, fact.gene_ids, split, user, view.graph_version])[:32]
            observations = _context(view, fact, case_id, row_id, index, registry, seed)
            harness = {'active_graph': {'multiplex_id': view.multiplex_id,
                'graph_version': view.graph_version,
                'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed} for layer in view.layer_order]},
                'observations': observations}
            steps, answer = _run(view, fact, case_id, observations, row_id, registry)
            reason = _reason(view, fact, case_id, observations, steps, answer)
            row = {'row_id': row_id, 'messages': [
                {'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(harness)},
                {'role': 'user', 'content': user}, *steps,
                {'role': 'assistant', 'thinking': reason, 'content': answer}],
                'tools': deepcopy(contract['tools']), 'metadata': {
                    'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                    'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                    'question_family': 'S2.1', 'catalog_case_id': case_id,
                    'split': split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                    'graph_version': view.graph_version, 'layer_id': fact.layer_id,
                    'gene_ids': list(fact.gene_ids), 'framing': binding['framing'],
                    'variant': binding['behavior'], 'source_predicate': predicate,
                    'wrapper_index': wrapper, 'wording_index': ordinal,
                    'supervision': 'current_turn_assistant_only', 'training_ready': False}}
            validate_case_row(row, view, contract=contract, registry=registry, index=index)
            rows.append(row)
    return rows


def validate_case_row(row, view, *, contract=None, registry=None, index=None):
    """Replay source observations, then independently check visible decisions/calls.

    Source replay detects fabricated context. The planner replay detects wrong
    direction, insufficient membership, invented retries, altered coverage,
    unnecessary calls, and fabricated final fields. Free prose is not exact.
    """
    contract = load_build_contract() if contract is None else contract
    registry = registry or contract.get('_registry') or load_registry_context(contract['registry_index'])
    meta, messages = row['metadata'], row['messages']
    _validate_v3_message_sequence(messages)
    if messages[1].get('role') != 'harness' or not isinstance(messages[1].get('content'), str):
        raise ValueError('supplied observations require a harness message')
    case_id = meta['catalog_case_id']
    if case_id not in CASE_BINDINGS:
        raise ValueError('unknown S2.1 catalog binding')
    binding = CASE_BINDINGS[case_id]
    fact = CaseFact(meta['layer_id'], tuple(meta['gene_ids']), meta['split'], meta['split_unit'])
    unit, split = _pair_group(fact.gene_ids, meta['selection_seed'])
    if split != fact.split or unit != fact.split_unit or meta['graph_version'] != view.graph_version:
        raise ValueError('source query changed its fixed split or graph version')
    if (meta['framing'], meta['variant'], meta['source_predicate']) != (
            binding['framing'], binding['behavior'], binding['source_predicate']):
        raise ValueError('catalog behavior metadata changed')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['tool_contract_version'] != contract['contract_version']
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['question_family'] != 'S2.1' or meta['supervision'] != 'current_turn_assistant_only'
            or meta['training_ready'] is not False):
        raise ValueError('row schema, prompt, tool identity, or supervision contract changed')
    user, wrapper = _request(view, fact, case_id, contract, meta['wording_index'])
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or messages[2] != {'role': 'user', 'content': user}:
        raise ValueError('request changed its frozen wording or bound subjects')
    if row['tools'] != contract['tools'] or wrapper != meta['wrapper_index']:
        raise ValueError('tool or wording identity changed')
    harness = json.loads(messages[1]['content'])
    expected_active = {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
        'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed} for layer in view.layer_order]}
    if harness['active_graph'] != expected_active:
        raise ValueError('active representation metadata changed')
    runtime = NeighborRuntime(view, index=index)
    _assert_source_predicate(view, fact, binding['source_predicate'], runtime.index, registry)
    for observation in harness['observations']:
        operation, arguments = observation['operation'], observation['arguments']
        if operation == 'lookup_human_gene_symbol':
            validate_identity_observations([observation], registry)
            continue
        evidence_id = observation['provenance']['graph'][0]['evidence_id']
        if operation == 'get_edge':
            expected = view.get_edge(**arguments, evidence_id=evidence_id)
        elif operation == 'get_neighbors':
            expected = runtime.get_neighbors(**arguments, evidence_id=evidence_id)
        elif operation == 'get_nodes_by_layer':
            expected = NodesByLayerRuntime(view).get_nodes_by_layer(**arguments, evidence_id=evidence_id)
        else:
            raise ValueError('unsupported source context operation')
        if observation != _observation(operation, arguments, expected):
            raise ValueError('saved observation does not match its pinned native source')
    expected_id = 's21p_' + digest([case_id, fact.layer_id, fact.gene_ids, split, user, view.graph_version])[:32]
    if row['row_id'] != expected_id:
        raise ValueError('row identity no longer binds the exact case, source, split, and wording')
    expected_context = _context(view, fact, case_id, row['row_id'], runtime.index, registry, meta['selection_seed'])
    if harness['observations'] != expected_context:
        raise ValueError('case context does not exercise its declared catalog behavior')
    steps, expected = _run(view, fact, case_id, harness['observations'], row['row_id'], registry)
    actual_steps = messages[3:-1]
    if len(actual_steps) != len(steps):
        raise ValueError('wrong number of required calls or responses')
    for actual, target in zip(actual_steps, steps):
        if actual['role'] != target['role']:
            raise ValueError('call/response sequence changed')
        if actual['role'] == 'assistant':
            if actual.get('tool_call') != target['tool_call'] or not actual.get('thinking'):
                raise ValueError('required call, retry, or reasoning changed')
        elif actual != target:
            raise ValueError('tool response changed its source truth or fault schedule')
    final = messages[-1]['content']
    for field in ('status', 'provenance', 'result', 'continuation_decision'):
        if final.get(field) != expected[field]:
            raise ValueError(f'final {field} is not derived from visible evidence')
    blocked = final.get('blocked_reason')
    bad_blocker = ((not isinstance(blocked, str) or not blocked.strip())
                   if expected['continuation_decision'] == 'defer' else blocked is not None)
    if set(final) != set(expected) or bad_blocker:
        raise ValueError('final envelope or blocker presence changed')
    if not isinstance(final['summary'], str) or not final['summary'] or not messages[-1].get('thinking'):
        raise ValueError('final reasoning and reader summary are required')


def audit_request_wording(contract):
    """Check new masked full questions against every frozen S2 family and split.

    Historical-to-historical comparisons have their published audit. This
    check includes both directions of each new/historical comparison and all
    new/new cross-split pairs, including validation versus test.
    """
    from .s1_prompt_banks import _tokens, _ngrams, PromptBankError
    from . import s2_13_prompts as old

    wrappers = contract['_bank']
    renderers = (old._rendered_variants, old._rendered_variants, old._neighbor_variants,
        old._map_variants, old._membership_variants, old._classification_variants,
        old._shared_variants, old._degree_variants, old._subgraph_variants,
        old._component_variants, old._path_variants, old._singleton_variants,
        old._specificity_variants)
    splits = ('train', 'validation', 'test')
    historical = {split: [] for split in splits}
    for number, renderer in enumerate(renderers, 1):
        path = Path(__file__).with_name('s2_v1_edge_requests.json' if number == 1 else f's2_{number}_v1_requests.json')
        bank = json.loads(path.read_text())
        variants = renderer(wrappers, bank, f'S2.{number}') if number < 3 else renderer(wrappers, bank)
        for split in splits:
            historical[split].extend(variants[split])
    fresh = {split: [] for split in splits}
    for split in splits:
        for framing in ('direct', 'step'):
            for index, wrapper in enumerate(wrappers[split][framing]):
                for kind, templates in contract['_requests'][split].items():
                    for ordinal, template in enumerate(templates):
                        core = template.format(first='', second='', layer='', multiplex='')
                        tokens = _tokens(wrapper.format(request=core))
                        fresh[split].append((f'S2.1complete/{split}/{framing}/{index}/{kind}/{ordinal}',
                                             frozenset(tokens), frozenset(_ngrams(tokens, 5))))
    vocabulary = sorted({word for bank in (historical, fresh) for rows in bank.values()
                         for _, words, _ in rows for word in words})
    bits = {word: 1 << index for index, word in enumerate(vocabulary)}

    def encode(rows):
        unique = {}
        for label, words, _ in rows:
            unique.setdefault(frozenset(words), (label, sum(bits[word] for word in words), len(words)))
        return list(unique.values())

    closest, compared = 0., 0
    for index, split in enumerate(splits):
        for other in splits[index + 1:]:
            for left, right in ((fresh[split], fresh[other]),
                                (fresh[split], historical[other]),
                                (historical[split], fresh[other])):
                compared += len(left) * len(right)
                grams = {gram: label for label, _, values in left for gram in values}
                for label, _, values in right:
                    shared = values & grams.keys()
                    if shared:
                        gram = min(shared)
                        raise PromptBankError(f'new S2.1 wording shares five words: {grams[gram]}, {label}: {" ".join(gram)}')
                left_bits, right_bits = encode(left), encode(right)
                for label, words, size in left_bits:
                    for other_label, other_words, other_size in right_bits:
                        overlap = (words & other_words).bit_count()
                        score = overlap / (size + other_size - overlap)
                        closest = max(closest, score)
                        if score >= .6:
                            raise PromptBankError(f'new S2.1 wording overlaps: {label}, {other_label}: {score:.4f}')
    return {'cross_split_pairs_checked': compared,
            'maximum_cross_split_jaccard': round(closest, 4), 'shared_cross_split_5grams': 0,
            'new_rendered_questions': {split: len(rows) for split, rows in fresh.items()}}

