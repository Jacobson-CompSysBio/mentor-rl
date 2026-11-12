"""Generate the reviewed S2.4–S2.6 production combinations from native facts.

Historical builders and validators remain frozen. This module selects the
small faithful source instances missing from their verification schedules and
independently combines existing frozen requests and framing wrappers.
"""

from copy import deepcopy
from itertools import combinations
import heapq

from . import s2_4_corpus as maps, s2_5_corpus as memberships, s2_6_corpus as nodes
from . import s2_4_case_repairs as map_repairs, s2_5_case_repairs as membership_repairs
from .s2_2_corpus import VariantUnavailable
from .s2_2_identity import get_symbols
from .s2_3_corpus import gene_group
from .s2_audit_cases_3_6 import CASE_BINDINGS, _RULES, matches
from .s2_corpus import digest
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import MultiplexRuntime
from .s2_neighbor_layers import NeighborLayersRuntime


CASE_IDS = tuple(f'S2.{family}.{number}' for family in (4, 5, 6) for number in range(1, 26))
INPUT_PATHS = ()
ALL = {'kind': 'all', 'layer_ids': [], 'layer_family': None}


def _scope(layers):
    """Return an explicit scope in the caller's manifest order."""
    return {'kind': 'layers', 'layer_ids': list(layers), 'layer_family': None}


def prepare_context(context, family=None):
    """Cache subject splits and identity eligibility once per source context."""
    if '_production46' not in context:
        view, registry, seed = context['view'], context['registry'], context.get('seed', 20261028)
        universe = view.multiplex_nodes | set(registry.symbols_by_id)
        groups = {gene: gene_group(gene, seed) for gene in universe}
        symbols = set()
        for gene in universe:
            try:
                get_symbols([gene], registry)
            except ValueError:
                continue
            symbols.add(gene)
        context['_production46'] = {'groups': groups, 'symbols': symbols, 'facts': {}}
    return context


def _subjects(context, split, *, outside=False):
    """Keep every source subject in its original hash split."""
    cache = context['_production46']
    view = context['view']
    universe = set(context['registry'].symbols_by_id) - view.multiplex_nodes if outside else view.multiplex_nodes
    return sorted((gene for gene in universe if cache['groups'][gene][1] == split),
                  key=lambda gene: cache['groups'][gene][0])


def _map_facts(number, context, split):
    """Select compact real scoped maps with the reviewed membership pattern."""
    view, native = context['view'], context['index'].native_index
    seed = context.get('seed', 20261028)
    runtime = NeighborLayersRuntime(view, index=native)
    undirected = [layer for layer in view.layer_order if not view.layers[layer].directed]
    directed = [layer for layer in view.layer_order if view.layers[layer].directed]
    scopes = [_scope(pair) for pair in combinations(view.layer_order, 2)]
    direction = 'incident'
    if number in (1, 2, 4, 7, 8, 12, 13, 14, 15, 16, 17, 18, 21, 22, 24):
        scopes = [_scope(pair) for pair in combinations(undirected, 2)]
    if number in (11, 19, 25):
        scopes = [_scope([layer]) for layer in directed]
    if number == 10:
        scopes, direction = [_scope(directed)], 'out'
    if number == 20:
        scopes = [_scope([u, d]) for u in undirected for d in directed]
    if number == 23:
        scopes = [ALL]
    subjects = _subjects(context, split, outside=number == 6)
    symbols = context['_production46']['symbols']
    candidates = []
    for scope in scopes:
        selected = runtime.resolve_scope(scope, direction)
        for gene in subjects:
            present = sum(gene in view.layers[layer].nodes for layer in selected)
            if number in (1, 2, 9, 10) and present != len(selected):
                continue
            if number == 3 and not 0 < present < len(selected):
                continue
            if number == 5 and present:
                continue
            if number == 4 and gene not in symbols:
                continue
            if number in (7, 14, 15, 16, 20):
                first = next((layer for layer in selected if not view.layers[layer].directed), None)
                if first is None or not native.neighbors(first, gene, 'undirected'):
                    continue
            if number in (19, 20):
                first = next((layer for layer in selected if view.layers[layer].directed), None)
                if first is None or not native.neighbors(first, gene, 'in'):
                    continue
            if number == 25 and not any((gene, gene) in view.layers[layer].edges for layer in selected):
                continue
            fact = maps._fact(runtime, gene, scope, direction, seed)
            count = fact.unique_neighbor_count or 0
            if number in (1, 2, 4, 7, 8, 9, 12, 13, 15, 18, 24) and not count:
                continue
            if number in (12, 13, 17, 18) and count < 2:
                continue
            if number == 10 and count:
                continue
            if number == 9:
                support_kinds = {view.layers[layer].directed for layer in selected
                    if any(native.neighbors(layer, gene, side) for side in native.directions(layer))}
                if support_kinds != {True, False}:
                    continue
            bound = 1
            if number == 11:
                layer = selected[0]
                incoming = {other for other, _ in native.neighbors(layer, gene, 'in')}
                outgoing = {other for other, _ in native.neighbors(layer, gene, 'out')}
                reciprocal = (incoming & outgoing) - {gene}
                ordered = sorted(incoming | outgoing)
                positions = [ordered.index(other) + 1 for other in reciprocal]
                if not positions or min(positions) >= count:
                    continue
                bound = min(positions)
            candidates.append((count, context['_production46']['groups'][gene][0], tuple(selected), fact, bound))
    selected = heapq.nsmallest(64, candidates, key=lambda item: item[:3])
    return [(fact, 'top_k' if number in (11, 13, 17) else 'complete',
             bound if number in (11, 13, 17) else None) for _, _, _, fact, bound in selected]


def _membership_facts(number, context, split):
    """Choose source inventories rather than neighbor counts for membership."""
    view, native, seed = context['view'], context['index'].native_index, context.get('seed', 20261028)
    runtime = GeneLayersRuntime(view)
    scopes = [ALL]
    if number in (3, 4, 9):
        scopes = [{'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in runtime.multiplex.families if runtime.resolve_scope(
                      {'kind': 'all', 'layer_ids': [], 'layer_family': family})]
    if number == 19:
        scopes = [_scope([layer]) for layer in view.layer_order if view.layers[layer].directed]
    if number == 21:
        scopes = [{'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in runtime.multiplex.families if not runtime.resolve_scope(
                      {'kind': 'all', 'layer_ids': [], 'layer_family': family})]
    subjects = _subjects(context, split, outside=number in (6, 20, 24))
    candidates = []
    for scope in scopes:
        selected = runtime.resolve_scope(scope)
        for gene in subjects:
            fact = memberships._fact(runtime, gene, scope, seed)
            count = fact.layer_count
            if number not in (4, 6, 20, 21, 22, 24) and not count:
                continue
            if number == 4 and count:
                continue
            if number == 5 and gene not in context['_production46']['symbols']:
                continue
            if number == 10 and count != 1:
                continue
            if number in (11, 12, 13, 17, 18) and count < 2:
                continue
            if number in (7, 14, 15, 16) and not any(
                    gene in view.layers[layer].nodes for layer in view.layer_order
                    if runtime.multiplex.family_of(layer) == 'tf_target'):
                continue
            if number == 19 and (not selected or native.neighbors(selected[0], gene, 'out')):
                continue
            candidates.append((count, context['_production46']['groups'][gene][0], str(scope), fact))
    chosen = heapq.nsmallest(64, candidates, key=lambda item: item[:3])
    return [(fact, 'top_k' if number in (11, 13, 17, 25) else 'complete',
             fact.layer_count + 1 if number == 25 else 1 if number in (11, 13, 17) else None)
            for _, _, _, fact in chosen]


def _nodes_facts(number, context, split):
    """Select finite same-split sets with exactly the required three-way labels."""
    view, native, seed = context['view'], context['index'].native_index, context.get('seed', 20261028)
    inside = set(_subjects(context, split))
    outside = _subjects(context, split, outside=True)
    symbols = context['_production46']['symbols']
    patterns = {1: (1, 1, 0), 2: (2, 0, 0), 3: (0, 2, 0), 4: (1, 1, 1), 5: (1, 1, 0),
        6: (1, 0, 0), 7: (0, 1, 0), 8: (1, 1, 0), 9: (1, 1, 1), 10: (0, 0, 1),
        11: (1, 1, 1), 12: (1, 1, 1), 13: (1, 1, 1), 14: (1, 1, 1), 15: (1, 1, 0),
        16: (1, 0, 0), 17: (1, 1, 1), 18: (1, 0, 0), 19: (0, 0, 0), 20: (0, 1, 0),
        21: (1, 0, 0), 22: (1, 1, 1), 23: (1, 1, 0), 24: (0, 0, 0), 25: (1, 1, 0)}
    pattern = patterns[number]
    candidates = []
    for layer in view.layer_order:
        if number == 16 and not view.layers[layer].directed:
            continue
        pools = [inside & view.layers[layer].nodes, inside - view.layers[layer].nodes, set(outside)]
        if number == 5:
            pools = [pool & symbols for pool in pools]
        if number == 16:
            pools[0] = {gene for gene in pools[0] if not native.neighbors(layer, gene, 'out')}
        if number == 21:
            earlier = view.layer_order[:view.layer_order.index(layer)]
            pools[0] = {gene for gene in pools[0] if any(gene in view.layers[other].nodes for other in earlier)}
        ordered = [sorted(pool, key=lambda gene: context['_production46']['groups'][gene][0]) for pool in pools]
        if any(len(pool) < size for pool, size in zip(ordered, pattern)):
            continue
        for ordinal in range(1 if not any(pattern) else 64):
            genes = [pool[(ordinal + offset) % len(pool)]
                     for pool, size in zip(ordered, pattern) for offset in range(size)]
            candidates.append(nodes._fact(view, layer, genes, seed, split=split))
    return [(fact, None, None) for fact in candidates]


def _reword(row, context, wrapper, request_index):
    """Independently bind existing request and wrapper choices without new prose."""
    from .s2_4_prompts import render_s2_4_request
    from .s2_5_prompts import render_s2_5_request, render_s2_5_diagnostic_request
    from .s2_6_prompts import render_s2_6_request, render_s2_6_diagnostic_request
    meta, contract, view = row['metadata'], context['contract'], context['view']
    family, split = meta['question_family'], meta['split']
    fields = {'split': split, 'variant_index': request_index, 'multiplex_id': view.multiplex_id}
    if family == 'S2.4':
        core = render_s2_4_request(contract['_requests'], **fields, direction=meta['direction'], mode=meta['mode'],
            layer_scope=meta['layer_scope'], gene=meta['input_gene'], top_k=meta['top_k'], ranking=meta['ranking'])
    elif family == 'S2.5':
        fields.update(layer_scope=meta['layer_scope'], gene=meta['input_gene'], top_k=meta['top_k'])
        core = (render_s2_5_diagnostic_request(contract['_requests'], **fields, kind=meta['variant'])
                if meta['variant'] in memberships.DIAGNOSTICS else
                render_s2_5_request(contract['_requests'], **fields, mode=meta['mode']))
    else:
        fields.update(layer_id=meta['layer_id'], genes=meta['input_genes'])
        if meta['variant'] in nodes.DIAGNOSTICS:
            literal = row['messages'][-1]['content']['result']['request']
            fields.update(layer_id=literal['layer_id'], genes=literal['genes'], multiplex_id=literal['multiplex_id'])
        core = (render_s2_6_diagnostic_request(contract['_requests'], **fields, kind=meta['variant'])
                if meta['variant'] in nodes.DIAGNOSTICS else render_s2_6_request(contract['_requests'], **fields))
    row['messages'][2]['content'] = context['bank'][split][meta['framing']][wrapper].format(request=core)
    meta.update(wrapper_index=wrapper, request_template_index=request_index)
    if meta.get('repair') in (map_repairs.REPAIR, membership_repairs.REPAIR):
        prefix = 's24repair_' if family == 'S2.4' else 's25repair_'
        row['row_id'] = prefix + digest([meta['repair'], meta['catalog_case_id'],
                                        row['messages'][2]['content'], meta['split_unit']])[:32]
    else:
        row['row_id'] = 's246prod_' + digest([meta['catalog_case_id'], row['messages']])[:32]
    return row


def production_candidates(case_id, context, *, split):
    """Yield finite, case-qualified native candidates using held-out split wording."""
    if case_id not in CASE_IDS or split not in ('train', 'validation', 'test'):
        raise ValueError('unsupported production case or split')
    prepare_context(context)
    family, number = map(int, case_id.split('.')[1:])
    if family == 4 and '_repair_requests' not in context['contract']:
        context['contract'] = map_repairs.load_build_contract()
    view, bank, contract = context['view'], context['bank'], context['contract']
    registry, seed = context['registry'], context.get('seed', 20261028)
    native = context['index'].native_index
    key = case_id, split
    cache = context['_production46']['facts']
    if key not in cache:
        cache[key] = {4: _map_facts, 5: _membership_facts, 6: _nodes_facts}[family](number, context, split)
    variant = CASE_BINDINGS[case_id]['variants'][0]
    framing = _RULES[case_id][2]
    for fact, mode, bound in cache[key]:
        if family == 4 and number == 22:
            for wrapper in range(len(bank[split][framing])):
                yield map_repairs.build_row(view, fact, case_id, bank, contract, registry,
                    index=native, seed=seed, variant_index=wrapper)
            continue
        for request_index in range(4):
            try:
                if family == 4 and case_id in map_repairs.CASE_IDS:
                    row = map_repairs.build_row(view, fact, case_id, bank, contract, registry,
                        index=native, seed=seed, variant_index=request_index)
                elif family == 4 and number == 21:
                    row = maps.build_diagnostic(view, fact, framing, request_index, bank, contract, seed=seed)
                elif family == 4:
                    row = maps.build_row(view, fact, mode, bound, framing, variant, request_index,
                        bank, contract, index=native, registry=registry, seed=seed)
                elif family == 5 and case_id in membership_repairs.CASE_IDS:
                    row = membership_repairs.build_row(view, fact, case_id, bank, contract, registry,
                        index=native, seed=seed, variant_index=request_index)
                elif family == 5 and number == 22:
                    row = memberships.build_diagnostic(view, fact, framing, request_index, bank, contract,
                        kind='unknown_layer', seed=seed)
                elif family == 5:
                    row = memberships.build_row(view, fact, mode, bound, framing, variant, request_index,
                        bank, contract, index=native, registry=registry, seed=seed)
                elif number in (18, 24):
                    row = nodes.build_diagnostic(view, fact, framing, request_index, bank, contract,
                        kind=variant, seed=seed)
                else:
                    row = nodes.build_row(view, fact, framing, variant, request_index, bank, contract,
                        index=native, registry=registry, seed=seed)
            except VariantUnavailable:
                continue
            row['metadata']['catalog_case_id'] = case_id
            if family == 6 and len(fact.gene_ids) == 1:
                # The shared native-context predicate identifies a singleton by this field.
                row['metadata']['gene_id'] = fact.gene_ids[0]
            for wrapper in range(len(bank[split][framing])):
                candidate = _reword(deepcopy(row), context, wrapper, request_index)
                if matches(case_id, candidate):
                    yield candidate


def validate_row(row, context):
    """Replay every native observation and enforce the unchanged case predicate."""
    case_id = row['metadata'].get('catalog_case_id')
    if case_id not in CASE_IDS:
        raise ValueError('unknown S2.4–S2.6 production case')
    family = int(case_id.split('.')[1])
    module = (map_repairs if case_id in map_repairs.CASE_IDS else maps if family == 4 else
              membership_repairs if case_id in membership_repairs.CASE_IDS else memberships if family == 5 else nodes)
    module.validate_row(row, context['view'], index=context['index'].native_index,
        bank=context['bank'], contract=context['contract'], registry=context['registry'])
    if not matches(case_id, row):
        raise ValueError('the production row no longer satisfies the frozen case binding')
