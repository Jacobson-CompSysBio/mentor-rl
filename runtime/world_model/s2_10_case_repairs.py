"""Supply the two reviewed S2.10 combinations absent from the frozen sampler.

The source graph, wording, contracts, and historical rows remain unchanged.
Component deferral is derived from visible records using a source-blind view.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_3_corpus import gene_group, _provenance
from .s2_10_corpus import (
    LABELS, _active_context, _arguments, _plan, _request, build_diagnostic,
    validate_row as validate_original_row,
)
from .s2_10_prompts import render_s2_10_request
from .s2_10_sampling import fact_from_genes
from .s2_corpus import SCHEMA_VERSION, SPLITS, canonical_json, digest, _failure
from .s2_neighbors import NeighborRuntime
from ..world_model_training import _validate_v3_message_sequence


CASE_IDS = ('S2.10.17', 'S2.10.28')
REPAIR = 's2_10_case_repairs_v1'
TRANSPORT = {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}


def _visible_plan(view, arguments, observations):
    """Use only representation metadata and the supplied native records."""
    visible = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})
    return _plan(visible, arguments, observations, False, False)


def _mark(row, case_id):
    """Identify fresh repair rows without changing their existing variant labels."""
    row['metadata'].update(repair=REPAIR, catalog_case_id=case_id)
    row['row_id'] = 's210repair_' + digest([REPAIR, case_id, row['row_id']])[:32]
    return row


def _diagnostic_row(view, index, fact, bank, contract, registry, seed):
    """Reject the explicit undirected request against a known directed layer."""
    if not view.layers[fact.layer_id].directed or len(fact.gene_ids) != 3:
        raise ValueError('case 17 requires three selected genes and a native directed layer')
    row = build_diagnostic(view, fact, 'direct', 0, bank, contract,
                           kind='unavailable_kind', registry=registry, seed=seed)
    return _mark(row, CASE_IDS[0])


def _membership_row(view, index, fact, bank, contract, seed):
    """Preserve two complete lists when only the third membership check fails."""
    if (view.layers[fact.layer_id].directed or len(fact.gene_ids) != 3
            or fact.connectivity_kind != 'undirected'):
        raise ValueError('case 28 requires three inputs in a native undirected layer')
    arguments = _arguments(view, fact, fact.connectivity_kind)
    request = _request(arguments, fact.gene_ids, False)
    core = render_s2_10_request(contract['_requests'], split=fact.split, variant_index=0,
                                question_kind='partition', **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing='step', variant_index=0, request=core)
    row_id = 's210native_' + digest([REPAIR, view.graph_version, fact.split_unit, request, user])[:32]
    native, saved = NeighborRuntime(view, index=index.native_index), []
    for number, gene in enumerate(fact.gene_ids[:2], 1):
        source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': gene,
                  'direction': 'undirected', 'mode': 'complete', 'top_k': None, 'cursor': None}
        response = native.get_neighbors(**source, evidence_id=f'ev_{row_id}_native_{number}')
        saved.append(_observation('get_neighbors', source, response))
    first, second, third = fact.gene_ids
    if (not any(row['gene_id'] == second for row in saved[0]['result']['neighbors'])
            or any(row['gene_id'] == third for item in saved for row in item['result']['neighbors'])):
        raise ValueError('the saved pair must be connected while leaving the third input unclassified')
    proof, used = _visible_plan(view, arguments, saved)
    pending = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': [third]}
    coverage = {'operation': 'layer_membership', 'requested_gene_ids': list(fact.gene_ids),
                'checked_gene_ids': [first, second], 'unchecked_gene_ids': [third]}
    if (proof.result is not None or proof.next_tool != 'get_nodes_by_layer'
            or proof.next_arguments != pending or proof.coverage != coverage or used != saved):
        raise ValueError('the visible complete lists must leave exactly one membership prerequisite')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view), 'observations': saved})},
        {'role': 'user', 'content': user}]
    for remaining, thought in ((1,
            f'The complete native lists place {first} and {second} together and settle their edges to {third}. '
            'Only the third membership remains unknown. Its self-loop cannot change components; check its membership alone.'),
            (0, 'The visible error authorizes one retry. Repeat the identical membership tool and every argument; failed checks add no evidence.')):
        messages.extend([{'role': 'assistant', 'thinking': thought,
            'tool_call': {'name': 'get_nodes_by_layer', 'arguments': deepcopy(pending)}},
            {'role': 'tool', 'name': 'get_nodes_by_layer', 'content': _failure(remaining)}])
    answer = {'status': 'error', 'provenance': _provenance(used),
        'result': {'failed_tool': 'get_nodes_by_layer', 'error_code': 'BACKEND_UNAVAILABLE',
                   'arguments': pending, 'partial_results': used, 'coverage': proof.coverage},
        'continuation_decision': 'defer',
        'blocked_reason': f'Membership of {third} remains unknown after the authorized retry; the complete partition needs backend recovery.',
        'summary': f'The saved native lists establish the pair {first}, {second}; the third membership remains unresolved.'}
    messages.append({'role': 'assistant', 'thinking':
        'The allowed retry is exhausted. Preserve both original complete neighbor records and their provenance. '
        'Only their two input memberships count as checked. The third could be a layer member, a multiplex-only member, '
        'or globally absent; do not guess components or counts. Defer for backend recovery.', 'content': answer})
    row = {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.10', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'connectivity_kind': fact.connectivity_kind, **{key: getattr(fact, key) for key in LABELS},
        'category': fact.category, 'framing': 'step', 'variant': 'membership_exhausted', 'question_kind': 'partition',
        'request': request, 'membership_first': False, 'transport': deepcopy(TRANSPORT), 'wrapper_index': wrapper,
        'request_template_index': 0, 'supervision': 'current_turn_assistant_only', 'training_ready': False}}
    return _mark(row, CASE_IDS[1])


def build_case_rows(view, index, bank, contract, registry, *, split, seed=20261028):
    """Build one fresh source-backed witness per repaired case in a fixed split."""
    if split not in SPLITS or type(seed) is not int:
        raise ValueError('a supported fixed split and integer selection seed are required')
    groups = {gene: gene_group(gene, seed) for gene in view.multiplex_nodes}
    directed = next(((layer, sorted((gene for gene in view.layers[layer].nodes if groups[gene][1] == split),
                                    key=lambda gene: groups[gene][0])) for layer in view.layer_order
        if view.layers[layer].directed and sum(groups[gene][1] == split for gene in view.layers[layer].nodes) >= 3), None)
    if directed is None:
        raise VariantUnavailable(f'no directed three-member request exists in fixed {split} split')
    layer, pool = directed
    diagnostic = fact_from_genes(view, index, layer, pool[:3], 'weak', seed,
                                  split=split, category='directed_unavailable_kind')
    best, examined = None, 0
    for ordinal, layer in enumerate(view.layer_order):
        native = view.layers[layer]
        if native.directed:
            continue
        degrees = {gene: len(index.native_index.neighbors(layer, gene, 'undirected')) for gene in native.nodes}
        for first, second in native.edges:
            if first == second or groups[first][1] != split or groups[second][1] != split:
                continue
            examined += 1
            pair = sorted((first, second))
            score = degrees[first] + degrees[second], ordinal, pair
            if best is None or score < best[0]:
                blocked = set(pair) | {gene for query in pair
                    for gene, _ in index.native_index.neighbors(layer, query, 'undirected')}
                third = min((gene for gene in view.multiplex_nodes
                             if groups[gene][1] == split and gene not in blocked),
                            key=lambda gene: groups[gene][0], default=None)
                if third is not None:
                    best = score, layer, [*pair, third]
    if best is None:
        raise VariantUnavailable(f'no connected native pair with a separate unresolved input in fixed {split} split')
    score, layer, genes = best
    remainder = fact_from_genes(view, index, layer, genes, 'undirected', seed,
                                 split=split, category='membership_remainder_after_native')
    rows = [_diagnostic_row(view, index, diagnostic, bank, contract, registry, seed),
            _membership_row(view, index, remainder, bank, contract, seed)]
    for row in rows:
        validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    return rows, {'repair': REPAIR, 'split': split, 'selection_seed': seed,
        'split_policy': 'unchanged gene_group(seed,gene) for every requested and saved query subject',
        'candidate_native_pairs_examined': examined, 'selected_native_neighbor_rows': score[0],
        'selections': [{'case_id': row['metadata']['catalog_case_id'], 'layer_id': row['metadata']['layer_id'],
                        'gene_ids': row['metadata']['gene_ids'], 'framing': row['metadata']['framing']} for row in rows]}


def _structured(row):
    """Remove only the prose that the contract deliberately does not exact-score."""
    result = deepcopy(row)
    for message in result['messages']:
        if message['role'] == 'assistant':
            message.pop('thinking')
    final = result['messages'][-1]['content']
    final['blocked_reason'] = bool(final['blocked_reason'])
    final.pop('summary')
    return result


def validate_row(row, view, *, index, bank, contract, registry=None):
    """Recheck source records, source-blind proof, exact calls, wording, and splits."""
    if set(row) != {'row_id', 'messages', 'tools', 'metadata'}:
        raise ValueError('the canonical repair row fields changed')
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    for message in messages:
        fields = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                  {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                  {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != fields or message['role'] == 'assistant' and (
                not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('the repair wire requires exact message fields and nonempty reasoning')
    final = messages[-1]['content']
    if (list(final) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or any(not isinstance(final[field], str) or not final[field].strip()
                   for field in ('summary', 'blocked_reason'))):
        raise ValueError('a deferred repair requires a canonical envelope and readable summary and blocker')
    case_id = meta.get('catalog_case_id')
    if (meta.get('repair') != REPAIR or case_id not in CASE_IDS or meta.get('split') not in SPLITS
            or type(meta.get('selection_seed')) is not int or not isinstance(meta.get('gene_ids'), list)
            or len(meta['gene_ids']) != 3 or len(set(meta['gene_ids'])) != 3):
        raise ValueError('the repair identity or three-input split domain changed')
    if meta.get('category') != ('directed_unavailable_kind' if case_id == CASE_IDS[0]
                                else 'membership_remainder_after_native'):
        raise ValueError('the repaired source selection category changed')
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['connectivity_kind'],
                           meta['selection_seed'], split=meta['split'], category=meta['category'])
    if any(gene_group(gene, meta['selection_seed'])[1] != meta['split'] for gene in fact.gene_ids):
        raise ValueError('a selected query subject crossed its fixed source split')
    if case_id == CASE_IDS[0]:
        expected = _diagnostic_row(view, index, fact, bank, contract, registry, meta['selection_seed'])
        validate_original_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    else:
        expected = _membership_row(view, index, fact, bank, contract, meta['selection_seed'])
    if canonical_json(_structured(row)) != canonical_json(_structured(expected)):
        raise ValueError('the repair changes source evidence, visible proof, request, fault schedule, or row identity')
