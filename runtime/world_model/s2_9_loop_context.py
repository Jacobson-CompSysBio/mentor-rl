"""Generate compact positive-loop proofs from genuine native neighbor pages.

The original S2.9 generator supplies a top-K prefix reaching the loop. A later
source page can supply the same positive singleton proof without displaying
unrelated earlier neighbors. This extension preserves the exact acquisition,
fixed subject split, source graph, and original frozen request wording.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_3_corpus import gene_group, _provenance
from .s2_9_corpus import (
    CONTRACT_PATH, _active_context, _arguments, _plan, _reason, _request,
    load_build_contract,
)
from .s2_9_prompts import render_s2_9_request
from .s2_9_sampling import fact_from_genes
from .s2_corpus import SCHEMA_VERSION, canonical_json, digest
from .s2_neighbors import NeighborRuntime
from ..world_model_training import _validate_v3_message_sequence


EXTENSION = 's2_9_native_loop_page_v1'
VARIANT = 'positive_loop_context'
NATIVE_TRANSPORT = {'page_size': 1, 'disclose_total': True, 'overlap': 0,
                    'empty_first_page': False, 'empty_page_at': None}


def loop_inventory(view, index, seed=20261028):
    """Enumerate every native loop and its rank in every allowed direction."""
    records = []
    for layer_id in view.layer_order:
        layer = view.layers[layer_id]
        loops = sorted((pair, weight) for pair, weight in layer.edges.items() if pair[0] == pair[1])
        for (first, second), weight in loops:
            for direction in index.native_index.directions(layer_id):
                neighbors = index.native_index.neighbors(layer_id, first, direction)
                rank = next(position for position, (gene, _) in enumerate(neighbors, 1) if gene == first)
                records.append({'layer_id': layer_id, 'gene_id': first,
                    'split': gene_group(first, seed)[1], 'direction': direction,
                    'weight': weight, 'native_neighbor_count': len(neighbors),
                    'loop_rank': rank, 'has_later_page': rank < len(neighbors)})
    return records


def select_fact(view, index, *, split='test', seed=20261028):
    """Select a real singleton loop with a genuinely incomplete native page."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('a fixed source split is required')
    inventory = loop_inventory(view, index, seed)
    candidates = [record for record in inventory if record['split'] == split and record['has_later_page']]
    if not candidates:
        raise VariantUnavailable(f'no native loop with a trailing neighbor in fixed {split} split')
    selected = min(candidates, key=lambda record: (
        record['loop_rank'], view.layer_order.index(record['layer_id']), record['gene_id'], record['direction']))
    fact = fact_from_genes(view, index, selected['layer_id'], [selected['gene_id']], seed,
                           split=split, category='native_loop_page')
    return fact, {'selection': selected, 'native_loop_direction_inventory': inventory,
                  'inventory_exhaustive_for_pinned_view': True,
                  'split_policy': 'unchanged gene_group(seed,gene) for the singleton subject',
                  'native_transport': deepcopy(NATIVE_TRANSPORT)}


def _loop_page(view, index, selection, row_id):
    """Follow backend-issued cursors and retain the original page containing its loop."""
    runtime = NeighborRuntime(view, index=index.native_index, **NATIVE_TRANSPORT)
    arguments = {'multiplex_id': view.multiplex_id, 'layer_id': selection['layer_id'],
        'gene_id': selection['gene_id'], 'direction': selection['direction'],
        'mode': 'complete', 'top_k': None, 'cursor': None}
    visited = 0
    while True:
        visited += 1
        response = runtime.get_neighbors(**arguments, evidence_id=f'ev_{row_id}_native_page_{visited}')
        if response['status'] != 'resolved':
            raise ValueError('a pinned source traversal did not resolve')
        if any(item['gene_id'] == selection['gene_id'] for item in response['result']['neighbors']):
            if response['result']['collection']['complete'] or response['result']['collection']['next_cursor'] is None:
                raise ValueError('the selected loop page must leave later native neighbors unobserved')
            return _observation('get_neighbors', arguments, response), visited
        cursor = response['result']['collection']['next_cursor']
        if cursor is None:
            raise ValueError('the exact native traversal did not contain its declared source loop')
        arguments = {**arguments, 'cursor': cursor}


def _visible_plan(view, arguments, observations):
    """Derive the singleton projection using representation metadata and visible pages."""
    metadata = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})
    return _plan(metadata, arguments, observations, False)


def build_rows(view, index, bank, contract, registry=None, *, split='test', seed=20261028):
    """Build both task framings with one faithful original partial page each."""
    fact, report = select_fact(view, index, split=split, seed=seed)
    arguments = _arguments(view, fact)
    request = _request(arguments, list(fact.gene_ids), False)
    rows, acquisitions = [], []
    for framing in ('direct', 'step'):
        core = render_s2_9_request(contract['_requests'], split=split, variant_index=0, **request)
        user, wrapper = render_s2_prompt(bank, split=split, framing=framing, variant_index=0, request=core)
        row_id = 's29loop_' + digest([EXTENSION, view.graph_version, fact.layer_id,
            fact.gene_ids, framing, request, user, report['selection'], NATIVE_TRANSPORT])[:32]
        observation, pages = _loop_page(view, index, report['selection'], row_id)
        proof, used = _visible_plan(view, arguments, [observation])
        if proof.next_arguments is not None or proof.issue_code or proof.result is None:
            raise ValueError('the positive singleton page did not settle its sole candidate edge')
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'continue' if framing == 'step' else 'stop',
            'blocked_reason': None, 'summary': 'The observed native self-loop settles the singleton projection.'}
        thought = (f"The supplied {observation['arguments']['direction']} page contains the positive self-loop "
            f"for {fact.gene_ids[0]} at weight {report['selection']['weight']}. Its collection remains incomplete. "
            'The positive endpoint establishes layer membership, and this singleton has only its self-pair to assess. '
            'The original page proves that pair without a new call; it does not complete the source neighbor list. ')
        row = {'row_id': row_id, 'messages': [
            {'role': 'system', 'content': contract['system_prompt']},
            {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view),
                                                        'observations': [observation]})},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'thinking': thought + _reason(proof.result, framing), 'content': answer}],
            'tools': deepcopy(contract['tools']), 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.9', 'split': split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
                'category_counts': list(fact.category_counts), 'edge_count': fact.edge_count,
                'isolated_count': fact.isolated_count, 'loop_count': fact.loop_count,
                'reciprocal_pairs': fact.reciprocal_pairs, 'descending_arc_count': fact.descending_arc_count,
                'category': fact.category, 'framing': framing, 'variant': VARIANT, 'request': request,
                'transport': {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None},
                'native_transport': deepcopy(NATIVE_TRANSPORT), 'native_page_ordinal': pages,
                'native_direction': report['selection']['direction'], 'extension': EXTENSION,
                'wrapper_index': wrapper, 'request_template_index': 0,
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}
        validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
        rows.append(row)
        acquisitions.append({'framing': framing, 'page_ordinal': pages,
            'arguments': deepcopy(observation['arguments']),
            'collection': deepcopy(observation['result']['collection']),
            'visible_neighbors': deepcopy(observation['result']['neighbors'])})
    return rows, {**report, 'actual_acquisitions': acquisitions,
        'context_policy': 'one original issued native page; earlier background pages are not supplied or claimed complete',
        'validator_module': 'runtime.world_model.s2_9_loop_context'}


def validate_row(row, view, *, bank, contract, registry=None, index=None):
    """Check exact native page replay, unchanged split, and visible-only singleton proof."""
    from .s2_subgraphs import SubgraphIndex
    index = SubgraphIndex(view) if index is None else index
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if len(messages) != 4 or [message['role'] for message in messages] != ['system', 'harness', 'user', 'assistant']:
        raise ValueError('the positive singleton proof requires one context page and no calls')
    if (list(row) != ['row_id', 'messages', 'tools', 'metadata']
            or any(set(message) != ({'role', 'thinking', 'content'} if message['role'] == 'assistant'
                                    else {'role', 'content'}) for message in messages)
            or type(meta.get('selection_seed')) is not int or type(meta.get('native_page_ordinal')) is not int):
        raise ValueError('the canonical row wire or integer source selection changed')
    if (messages[0] != {'role': 'system', 'content': contract['system_prompt']}
            or row['tools'] != contract['tools'] or meta.get('extension') != EXTENSION
            or meta.get('schema_version') != SCHEMA_VERSION or meta.get('question_family') != 'S2.9'
            or meta.get('tool_contract_version') != contract['contract_version']
            or meta.get('system_prompt_sha256') != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta.get('graph_version') != view.graph_version or meta.get('variant') != VARIANT
            or meta.get('native_transport') != NATIVE_TRANSPORT
            or meta.get('transport') != {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}
            or meta.get('supervision') != 'current_turn_assistant_only' or meta.get('training_ready') is not False):
        raise ValueError('the positive-loop extension identity or native transport changed')
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['selection_seed'],
                           split=meta['split'], category='native_loop_page')
    if len(fact.gene_ids) != 1 or fact.loop_count != 1 or fact.split_unit != meta['split_unit']:
        raise ValueError('the proof must concern one source-backed loop in its fixed subject split')
    labels = ('category_counts', 'edge_count', 'isolated_count', 'loop_count', 'reciprocal_pairs', 'descending_arc_count')
    if any(canonical_json(meta[key]) != canonical_json(getattr(fact, key)) for key in labels):
        raise ValueError('source sampling labels changed')
    selection = next((record for record in loop_inventory(view, index, meta['selection_seed'])
        if record['layer_id'] == fact.layer_id and record['gene_id'] == fact.gene_ids[0]
        and record['direction'] == meta['native_direction'] and record['has_later_page']), None)
    if selection is None:
        raise ValueError('the native loop page does not leave a genuine incomplete collection')
    expected_observation, ordinal = _loop_page(view, index, selection, row['row_id'])
    context = json.loads(messages[1]['content'])
    if (set(context) != {'active_graph', 'observations'} or context['active_graph'] != _active_context(view)
            or context['observations'] != [expected_observation] or meta['native_page_ordinal'] != ordinal):
        raise ValueError('the original issued native page, cursor, weight, or coverage changed')
    arguments = _arguments(view, fact)
    request = _request(arguments, list(fact.gene_ids), False)
    if meta['request'] != request or meta['framing'] not in ('direct', 'step'):
        raise ValueError('the singleton request changed its scope, selection, or task framing')
    if meta['request_template_index'] != 0:
        raise ValueError('the frozen request template selector changed')
    core = render_s2_9_request(contract['_requests'], split=fact.split, variant_index=0, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=meta['framing'], variant_index=0, request=core)
    expected_id = 's29loop_' + digest([EXTENSION, view.graph_version, fact.layer_id,
        fact.gene_ids, meta['framing'], request, user, selection, NATIVE_TRANSPORT])[:32]
    if messages[2] != {'role': 'user', 'content': user} or wrapper != meta['wrapper_index'] or row['row_id'] != expected_id:
        raise ValueError('the bound source subject or frozen wording identity changed')
    proof, used = _visible_plan(view, arguments, context['observations'])
    final = messages[-1]['content']
    expected = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
                'continuation_decision': 'continue' if meta['framing'] == 'step' else 'stop'}
    if (proof.next_arguments is not None or proof.issue_code or proof.result is None
            or any(final.get(key) != value for key, value in expected.items())
            or list(final) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or final['blocked_reason'] is not None or not isinstance(final['summary'], str) or not final['summary'].strip()
            or not isinstance(messages[-1].get('thinking'), str) or not messages[-1]['thinking'].strip()):
        raise ValueError('the answer is not the exact visible positive singleton proof')
