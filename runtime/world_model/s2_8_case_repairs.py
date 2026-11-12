"""Build the two reviewed degree evidence combinations missing from S2.8.

The historical builder and request banks stay frozen. These isolated repairs
combine source-replayed saved observations with the existing acquisition and
retry contract; only visible evidence can settle the answer.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

from . import s2_8_corpus as legacy
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_3_corpus import gene_group, _provenance
from .s2_8_sampling import fact_from_arguments
from .s2_corpus import canonical_json, digest, _failure
from .s2_degrees import DegreeIndex, DegreeRuntime
from .s2_neighbors import NeighborRuntime


CASE_IDS = ('S2.8.22', 'S2.8.26')
REPAIR = 's2_8_reviewed_case_repairs_v1'
_CASES = {'S2.8.22': ('direct', 'stale'), 'S2.8.26': ('step', 'rank_exhausted')}


def _index(view, index):
    """Share native adjacency and cache exact full-population degree support."""
    if isinstance(index, DegreeIndex):
        return index
    if index.view is not view:
        raise ValueError('the shared index belongs to another source view')
    if not hasattr(index, '_s2_8_case_repair_degree_index'):
        index._s2_8_case_repair_degree_index = DegreeIndex(view, index.native_index)
    return index._s2_8_case_repair_degree_index


def _visible_plan(view, arguments, observations):
    """Deny the planner access to source node inventories and adjacency."""
    visible = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})
    return legacy._plan(visible, arguments, observations)


def _scope(layer):
    return {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None}


def _row_id(row):
    meta = row['metadata']
    return 's28repair_' + digest([REPAIR, meta['catalog_case_id'], meta['graph_version'],
        meta['selection_seed'], meta['gene_id'], meta['layer_scope'], meta['direction'],
        meta['saved_layer_id'], meta['request'], row['messages'][2]['content']])[:32]


def _saved(view, index, row, arguments):
    """Acquire the original saved statistic or complete native list."""
    meta = row['metadata']
    evidence_id = f"ev_{row['row_id']}_saved"
    if meta['catalog_case_id'] == 'S2.8.22':
        source = {**arguments, 'layer_scope': _scope(meta['saved_layer_id']), 'direction': 'in'}
        response = DegreeRuntime(view, index=index).get_degree(**source, evidence_id=evidence_id)
        observation = _observation('get_degree', source, response)
        observation['provenance']['graph'][0]['graph_version'] = '0' * 64
        return observation
    source = {'multiplex_id': view.multiplex_id, 'layer_id': meta['saved_layer_id'],
        'gene_id': meta['gene_id'], 'direction': 'undirected', 'mode': 'complete', 'top_k': None, 'cursor': None}
    response = NeighborRuntime(view, index=index.native_index).get_neighbors(**source, evidence_id=evidence_id)
    return _observation('get_neighbors', source, response)


def _build_case(view, index, bank, contract, registry, fact, case_id, saved_layer, seed):
    """Retain frozen request rendering while assembling the repaired evidence."""
    framing, variant = _CASES[case_id]
    row = legacy.build_row(view, fact, framing, variant, 0, bank, contract,
                           index=index, registry=registry, seed=seed)
    meta = row['metadata']
    meta.update(catalog_case_id=case_id, repair=REPAIR, saved_layer_id=saved_layer)
    row['row_id'] = _row_id(row)
    arguments = legacy._arguments(view, fact, 'degree_and_hub', 95)
    observation = _saved(view, index, row, arguments)
    row['messages'] = row['messages'][:3]
    row['messages'][1]['content'] = canonical_json({'active_graph': legacy._active_context(view),
                                                   'observations': [observation]})
    proof, used = _visible_plan(view, arguments, [observation])
    if proof.next_arguments != arguments or proof.issue_code:
        raise VariantUnavailable('the saved evidence must leave exactly the requested statistic unavailable')
    thought = ('The saved statistic has a different scope, incoming direction, and graph version. '
               'Acquire the active outgoing degree and hub statistic without relabeling that record.'
               if case_id == 'S2.8.22' else
               'The complete native list proves the exact degree. It supplies no reference-population rank. '
               'Acquire the requested degree and hub statistic while retaining this original observation.')
    if case_id == 'S2.8.22':
        response = DegreeRuntime(view, index=index).get_degree(**arguments,
            evidence_id=f"ev_{row['row_id']}_active")
        row['messages'].extend([
            {'role': 'assistant', 'thinking': thought, 'tool_call': {'name': 'get_degree', 'arguments': deepcopy(arguments)}},
            {'role': 'tool', 'name': 'get_degree', 'content': response}])
        proof, used = _visible_plan(view, arguments,
            [observation, _observation('get_degree', arguments, response)])
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'stop', 'blocked_reason': None,
            'summary': 'The active requested statistic follows from its compatible acquisition.'}
        thought = legacy._reason(proof.result, framing)
    else:
        for allowance, reasoning in ((1, thought), (0,
                'The error permits one retry. Repeat all six arguments exactly; a failed call supplies no evidence.')):
            row['messages'].extend([
                {'role': 'assistant', 'thinking': reasoning,
                 'tool_call': {'name': 'get_degree', 'arguments': deepcopy(arguments)}},
                {'role': 'tool', 'name': 'get_degree', 'content': _failure(allowance)}])
        answer = {'status': 'error', 'provenance': _provenance(used),
            'result': {'failed_tool': 'get_degree', 'error_code': 'BACKEND_UNAVAILABLE',
                       'arguments': deepcopy(arguments), 'partial_results': list(used), 'coverage': None},
            'continuation_decision': 'defer',
            'blocked_reason': 'The reference-population statistic remains unavailable after the authorized retry.',
            'summary': 'Retain the complete native-neighbor evidence and its exact count while rank remains unavailable.'}
        thought = (f"The saved complete native list proves degree {fact.degree}. "
            'Both statistic calls failed and the retry allowance is exhausted. Preserve the original native record '
            'and provenance. No percentile or hub label is supported; defer the unfinished step.')
    row['messages'].append({'role': 'assistant', 'thinking': thought, 'content': answer})
    validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    return row


def build_case_rows(view, index, bank, contract, registry, *, split, seed=20261028):
    """Select fresh fixed-split source queries for the two approved combinations."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('a fixed source split is required')
    index = _index(view, index)
    directed = [layer for layer in view.layer_order if view.layers[layer].directed]
    candidates = []
    for active in directed:
        for saved in directed:
            if active == saved:
                continue
            for gene in view.layers[active].nodes & view.layers[saved].nodes:
                if (gene_group(gene, seed)[1] == split
                        and index.native_index.neighbors(active, gene, 'out')
                        and index.native_index.neighbors(saved, gene, 'in')):
                    candidates.append((active, saved, gene))
    if not candidates:
        raise VariantUnavailable(f'no distinct directed-scope statistic pair in fixed {split} split')
    active, saved, gene = min(candidates, key=lambda item: digest([seed, 'S2.8.22', *item]))
    fact = fact_from_arguments(view, index, gene, _scope(active), 'out', seed, category='case_repair')
    rows = [_build_case(view, index, bank, contract, registry, fact, 'S2.8.22', saved, seed)]
    selected = {'S2.8.22': {'candidate_count': len(candidates), 'gene_id': gene,
                           'active_layer_id': active, 'saved_layer_id': saved}}
    candidates = [(len(index.native_index.neighbors(layer, gene, 'undirected')), layer, gene)
        for layer in view.layer_order if not view.layers[layer].directed
        for gene in view.layers[layer].nodes if gene_group(gene, seed)[1] == split]
    candidates = [item for item in candidates if item[0] > 0]
    if not candidates:
        raise VariantUnavailable(f'no positive complete undirected neighbor list in fixed {split} split')
    degree, layer, gene = min(candidates, key=lambda item: (item[0], digest([seed, 'S2.8.26', *item])))
    fact = fact_from_arguments(view, index, gene, _scope(layer), 'undirected', seed, category='case_repair')
    rows.append(_build_case(view, index, bank, contract, registry, fact, 'S2.8.26', layer, seed))
    selected['S2.8.26'] = {'candidate_count': len(candidates), 'gene_id': gene,
                          'layer_id': layer, 'complete_native_neighbor_count': degree}
    return rows, {'repair': REPAIR, 'split': split, 'selection': selected,
                  'split_policy': 'unchanged gene_group(seed,gene) for every query subject',
                  'selection_policy': 'distinct directed scope pairs; smallest positive complete undirected list'}


def validate_row(row, view, *, index, bank, contract, registry=None):
    """Replay all evidence and faults, then enforce the exact reviewed case."""
    index = _index(view, index)
    meta, messages = row['metadata'], row['messages']
    case_id = meta.get('catalog_case_id')
    if (case_id not in CASE_IDS or meta.get('repair') != REPAIR
            or (meta['framing'], meta['variant']) != _CASES[case_id]
            or list(row) != ['row_id', 'messages', 'tools', 'metadata']
            or type(meta.get('selection_seed')) is not int or meta['category'] != 'case_repair'
            or meta['request_template_index'] != 0 or row['row_id'] != _row_id(row)):
        raise ValueError('the repaired degree case identity, framing, or wire changed')
    legacy.validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    fact = fact_from_arguments(view, index, meta['gene_id'], meta['layer_scope'], meta['direction'],
                               meta['selection_seed'], category='case_repair')
    arguments = legacy._arguments(view, fact, 'degree_and_hub', 95)
    if meta['request'] != legacy._request(arguments, fact.gene_id):
        raise ValueError('the repaired case changes its approved degree and hub request')
    context = json.loads(messages[1]['content'])
    saved = _saved(view, index, row, arguments)
    if context['observations'] != [saved] or saved['status'] != 'resolved':
        raise ValueError('the exact original saved observation changed')
    proof, used = _visible_plan(view, arguments, [saved])
    calls = [message['tool_call'] for message in messages if 'tool_call' in message]
    responses = [message['content'] for message in messages if message['role'] == 'tool']
    if proof.next_arguments != arguments or proof.next_tool != 'get_degree' or proof.issue_code:
        raise ValueError('the saved evidence must leave the active statistic required')
    if case_id == 'S2.8.22':
        active = meta['layer_scope']['layer_ids']
        if (meta['layer_scope'] != _scope(active[0]) or len(active) != 1
                or meta['direction'] != 'out' or not view.layers[active[0]].directed
                or meta['saved_layer_id'] == active[0] or not view.layers[meta['saved_layer_id']].directed
                or saved['arguments']['direction'] != 'in'
                or saved['provenance']['graph'][0]['graph_version'] != '0' * 64
                or len(messages) != 6 or calls != [{'name': 'get_degree', 'arguments': arguments}]):
            raise ValueError('one saved record must combine wrong scope, native direction, and version')
        expected = DegreeRuntime(view, index=index).get_degree(**arguments,
            evidence_id=f"ev_{row['row_id']}_active")
        if responses != [expected]:
            raise ValueError('the active source statistic acquisition changed')
    else:
        native = saved['result']
        if (meta['layer_scope'] != _scope(meta['saved_layer_id']) or meta['direction'] != 'undirected'
                or view.layers[meta['saved_layer_id']].directed or len(messages) != 8
                or not native['collection']['complete'] or not native['neighbors']
                or len(native['neighbors']) != fact.degree
                or calls != [{'name': 'get_degree', 'arguments': arguments}] * 2
                or responses != [_failure(1), _failure(0)]
                or messages[-1]['content']['result']['partial_results'] != [saved]
                or list(used) != [saved]):
            raise ValueError('exhausted rank acquisition must retain the exact complete native count evidence')
