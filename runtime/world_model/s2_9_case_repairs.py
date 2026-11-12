"""Build the reviewed saved-preview and exhausted native-context S2.9 cases.

New rows use the pinned source graph and frozen request banks. The historical
builder stays unchanged, and complete original neighbor lists are never cropped
to fit a context budget or expanded into unobserved membership or edge claims.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

from . import s2_9_corpus as legacy
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_3_corpus import gene_group, _provenance
from .s2_9_sampling import fact_from_genes
from .s2_corpus import canonical_json, digest, _failure
from .s2_neighbors import NeighborRuntime
from .s2_subgraphs import SubgraphRuntime


CASE_IDS = ('S2.9.16', 'S2.9.21')
REPAIR = 's2_9_reviewed_case_repairs_v1'
_CASES = {'S2.9.16': ('direct', 'unknown_total'), 'S2.9.21': ('step', 'exhausted')}


def _visible_plan(view, arguments, observations):
    """Compute only from supplied records and manifest representation metadata."""
    visible = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})
    return legacy._plan(visible, arguments, observations, False)


def _row_id(row):
    meta = row['metadata']
    return 's29repair_' + digest([REPAIR, meta['catalog_case_id'], meta['graph_version'],
        meta['selection_seed'], meta['layer_id'], meta['gene_ids'], meta['saved_gene_ids'],
        meta['request'], row['messages'][2]['content']])[:32]


def _saved(view, index, row, arguments):
    """Acquire faithful original saved graph or native-neighbor observations."""
    meta = row['metadata']
    if meta['catalog_case_id'] == 'S2.9.16':
        response = SubgraphRuntime(view, index=index, **meta['transport']).induce_subgraph(**arguments,
            evidence_id=f"ev_{row['row_id']}_preview")
        return [_observation('induce_subgraph', arguments, response)]
    native = NeighborRuntime(view, index=index.native_index)
    observations = []
    for position, gene in enumerate(meta['saved_gene_ids'], 1):
        source = {'multiplex_id': view.multiplex_id, 'layer_id': meta['layer_id'],
            'gene_id': gene, 'direction': 'undirected', 'mode': 'complete', 'top_k': None, 'cursor': None}
        response = native.get_neighbors(**source, evidence_id=f"ev_{row['row_id']}_native_{position}")
        observations.append(_observation('get_neighbors', source, response))
    return observations


def _build_case(view, index, bank, contract, registry, fact, case_id, saved_genes, seed):
    """Reuse original wording and metadata with the missing evidence combination."""
    framing, variant = _CASES[case_id]
    mode, top_k = ('top_k', 1) if case_id == 'S2.9.16' else ('complete', None)
    row = legacy.build_row(view, fact, framing, variant, 0, bank, contract,
        index=index, registry=registry, mode=mode, top_k=top_k, seed=seed)
    meta = row['metadata']
    meta.update(catalog_case_id=case_id, repair=REPAIR, saved_gene_ids=list(saved_genes))
    row['row_id'] = _row_id(row)
    arguments = legacy._arguments(view, fact, mode, top_k)
    observations = _saved(view, index, row, arguments)
    row['messages'] = row['messages'][:3]
    row['messages'][1]['content'] = canonical_json({'active_graph': legacy._active_context(view),
                                                   'observations': observations})
    proof, used = _visible_plan(view, arguments, observations)
    if case_id == 'S2.9.16':
        if proof.next_arguments is not None or proof.result is None:
            raise VariantUnavailable('the saved unknown-total preview must settle the same first-K request')
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'stop', 'blocked_reason': None,
            'summary': 'The saved canonical edge prefix and complete membership partition settle the requested preview.'}
        thought = ('The saved acquisition has the same input set, layer, canonical order, and first-one bound. '
            'It supplies that exact prefix and all input membership categories. The full edge total is unknown '
            'and remains null; collection.complete stays false. Reuse the original evidence without a call. '
            'Omitted edges and selected-set isolation remain unproved. The standalone preview is complete, so stop.')
    else:
        if proof.next_tool != 'induce_subgraph' or proof.next_arguments != arguments:
            raise VariantUnavailable('the saved native records must leave the requested projection unavailable')
        for allowance, reasoning in ((1,
                'The complete saved native lists establish their own memberships and edge facts. '
                'At least one requested input remains unchecked, and some candidate edges remain unavailable. '
                'Acquire the complete projection while retaining both original observations.'), (0,
                'The visible error permits one retry. Repeat the identical tool and every argument. '
                'Failed calls add no evidence; retain all earlier native records.')):
            row['messages'].extend([
                {'role': 'assistant', 'thinking': reasoning,
                 'tool_call': {'name': 'induce_subgraph', 'arguments': deepcopy(arguments)}},
                {'role': 'tool', 'name': 'induce_subgraph', 'content': _failure(allowance)}])
        answer = {'status': 'error', 'provenance': _provenance(used),
            'result': {'failed_tool': 'induce_subgraph', 'error_code': 'BACKEND_UNAVAILABLE',
                'arguments': deepcopy(arguments), 'partial_results': list(used), 'coverage': proof.coverage},
            'continuation_decision': 'defer',
            'blocked_reason': 'The remaining membership and edge proof is unavailable after the authorized retry.',
            'summary': 'Preserve both complete native observations and truthful partial membership coverage.'}
        thought = ('The retry allowance is exhausted. Keep both original complete native-neighbor observations '
            'and their exact provenance. Their supported memberships and edges survive both failures. '
            'The unchecked input and unproved edges remain unavailable; do not substitute an empty graph. Defer this step.')
    row['messages'].append({'role': 'assistant', 'thinking': thought, 'content': answer})
    validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    return row


def _preview_candidates(view, index, split, seed):
    """Construct small same-split projections from two actual native edges."""
    candidates = set()
    splits = {gene: gene_group(gene, seed)[1] for gene in view.multiplex_nodes}
    for layer in view.layer_order:
        side = 'out' if view.layers[layer].directed else 'undirected'
        for gene in view.layers[layer].nodes:
            if splits[gene] != split:
                continue
            neighbors = []
            for neighbor, _ in index.native_index.neighbors(layer, gene, side):
                if splits[neighbor] == split and neighbor != gene:
                    neighbors.append(neighbor)
                    if len(neighbors) == 2:
                        break
            genes = tuple(sorted({gene, *neighbors}))
            if len(genes) < 2:
                continue
            native = view.layers[layer]
            pairs = [(first, second) for first in genes for second in genes
                     if native.directed or first <= second]
            if sum(pair in native.edges for pair in pairs) >= 2:
                candidates.add((layer, genes))
    return candidates


def _native_candidates(view, index, split, seed):
    """Keep two complete small native lists and a genuinely unclassified input."""
    candidates, best_size = [], None
    splits = {gene: gene_group(gene, seed)[1] for gene in view.multiplex_nodes}
    for layer in view.layer_order:
        if view.layers[layer].directed:
            continue
        pool = sorted(gene for gene in view.layers[layer].nodes if splits[gene] == split)
        for first in pool:
            first_rows = index.native_index.neighbors(layer, first, 'undirected')
            if best_size is not None and len(first_rows) >= best_size:
                continue
            for second, _ in first_rows:
                if second <= first or splits[second] != split:
                    continue
                second_rows = index.native_index.neighbors(layer, second, 'undirected')
                size = len(first_rows) + len(second_rows)
                if best_size is not None and size > best_size:
                    continue
                visible = {first, second, *(gene for gene, _ in first_rows), *(gene for gene, _ in second_rows)}
                third = next((gene for gene in pool if gene not in visible), None)
                if third is not None:
                    if best_size is None or size < best_size:
                        candidates, best_size = [], size
                    candidates.append((size, layer, first, second, third))
    return candidates


def build_case_rows(view, index, bank, contract, registry, *, split, seed=20261028):
    """Select fresh source projections in the unchanged three subject splits."""
    if split not in ('train', 'validation', 'test') or index.view is not view:
        raise ValueError('a fixed source split and its own index are required')
    candidates = _preview_candidates(view, index, split, seed)
    if not candidates:
        raise VariantUnavailable(f'no same-split native projection with two edges in fixed {split} split')
    layer, genes = min(candidates, key=lambda item: digest([seed, 'S2.9.16', item]))
    fact = fact_from_genes(view, index, layer, genes, seed, split=split, category='case_repair')
    rows = [_build_case(view, index, bank, contract, registry, fact, 'S2.9.16', [], seed)]
    selected = {'S2.9.16': {'candidate_count': len(candidates), 'layer_id': layer,
        'gene_ids': list(genes), 'source_edge_count': fact.edge_count, 'requested_top_k': 1}}
    candidates = _native_candidates(view, index, split, seed)
    if not candidates:
        raise VariantUnavailable(f'no incomplete membership proof after two complete native lists in fixed {split} split')
    size, layer, first, second, third = min(candidates,
        key=lambda item: (item[0], digest([seed, 'S2.9.21', *item])))
    fact = fact_from_genes(view, index, layer, [first, second, third], seed, split=split, category='case_repair')
    rows.append(_build_case(view, index, bank, contract, registry, fact, 'S2.9.21', [first, second], seed))
    selected['S2.9.21'] = {'candidate_count': len(candidates), 'layer_id': layer,
        'gene_ids': list(fact.gene_ids), 'saved_gene_ids': [first, second],
        'unchecked_gene_id': third, 'complete_native_rows': size}
    return rows, {'repair': REPAIR, 'split': split, 'selection': selected,
        'split_policy': 'every selected gene retains its unchanged gene_group(seed,gene) split',
        'selection_policy': 'small native edge projections; two smallest full native lists leaving an input unchecked'}


def validate_row(row, view, *, index, bank, contract, registry=None):
    """Replay source records, exact requests and retries, and visible-only results."""
    meta, messages = row['metadata'], row['messages']
    case_id = meta.get('catalog_case_id')
    if (case_id not in CASE_IDS or meta.get('repair') != REPAIR or index.view is not view
            or (meta['framing'], meta['variant']) != _CASES[case_id]
            or list(row) != ['row_id', 'messages', 'tools', 'metadata']
            or type(meta.get('selection_seed')) is not int or meta['category'] != 'case_repair'
            or meta['request_template_index'] != 0 or row['row_id'] != _row_id(row)):
        raise ValueError('the repaired projection case identity, framing, or wire changed')
    legacy.validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['selection_seed'],
        split=meta['split'], category='case_repair')
    mode, top_k = ('top_k', 1) if case_id == 'S2.9.16' else ('complete', None)
    arguments = legacy._arguments(view, fact, mode, top_k)
    if meta['request'] != legacy._request(arguments, list(fact.gene_ids), False):
        raise ValueError('the repaired case changes its approved native projection request')
    context = json.loads(messages[1]['content'])
    observations = _saved(view, index, row, arguments)
    if context['observations'] != observations or any(item['status'] != 'resolved' for item in observations):
        raise ValueError('the original native observations or their acquisition identity changed')
    proof, used = _visible_plan(view, arguments, observations)
    answer = messages[-1]['content']
    if case_id == 'S2.9.16':
        saved = observations[0]
        if (meta['saved_gene_ids'] or len(messages) != 4 or fact.edge_count <= 1
                or saved['result']['collection']['complete'] is not False
                or saved['result']['collection']['total_count'] is not None
                or saved['result']['edge_count'] != 1 or proof.next_arguments is not None
                or proof.issue_code or proof.result is None
                or answer['result'] != proof.result or answer['provenance'] != _provenance(used)
                or answer['result']['collection']['total_count'] is not None
                or answer['result']['collection']['complete'] is not False):
            raise ValueError('the saved exact first-K proof must preserve an unknown whole-graph total without calls')
    else:
        calls = [message['tool_call'] for message in messages if 'tool_call' in message]
        responses = [message['content'] for message in messages if message['role'] == 'tool']
        if (view.layers[meta['layer_id']].directed or len(fact.gene_ids) != 3
                or len(meta['saved_gene_ids']) != 2 or len(set(meta['saved_gene_ids'])) != 2
                or not set(meta['saved_gene_ids']) < set(fact.gene_ids)
                or len(messages) != 8 or any(not item['result']['collection']['complete'] for item in observations)
                or any(not item['result']['neighbors'] for item in observations)
                or proof.next_tool != 'induce_subgraph' or proof.next_arguments != arguments
                or proof.issue_code or not proof.coverage['unchecked_gene_ids']
                or list(used) != observations or answer['result']['partial_results'] != observations
                or answer['result']['coverage'] != proof.coverage
                or calls != [{'name': 'induce_subgraph', 'arguments': arguments}] * 2
                or responses != [_failure(1), _failure(0)]):
            raise ValueError('exhausted graph retries must retain both full native lists and an unchecked input')
