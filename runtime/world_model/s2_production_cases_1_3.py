"""Generate quota candidates for the first three frozen S2 case families.

Source subjects are selected before wording. Every candidate retains the
published pair/gene split and is replayed by its existing semantic validator.
"""

from copy import deepcopy
import hashlib

from . import s2_1_production as edges, s2_2_production as multiplex, s2_3_corpus as neighbors
from .s2_2_corpus import VariantUnavailable
from .s2_2_identity import get_symbols
from .s2_audit_cases_3_6 import CASE_BINDINGS
from .s2_corpus import SCHEMA_VERSION, _pair_group, canonical_json, digest


def _edge_facts(case_id, context, split):
    """Cache small native witnesses under each exact reviewed source predicate."""
    predicate = edges.CASE_BINDINGS[case_id]['source_predicate']
    cache = context.setdefault('_edge_facts', {})
    key = predicate, split
    if key not in cache:
        facts, seen = [], set()
        for layer, pair in edges._candidates(context['view'], context['index'].native_index,
                                               context['registry'], predicate):
            unit, actual = _pair_group(pair, context['seed'])
            if actual != split or (layer, pair) in seen:
                continue
            seen.add((layer, pair))
            facts.append(edges.CaseFact(layer, pair, split, unit))
            if len(facts) == 64:
                break
        cache[key] = facts
    return cache[key]


def _edge_row(case_id, fact, context, ordinal):
    """Use the original source context, planner, and frozen request renderer."""
    view, contract, index = context['view'], context['contract'], context['index'].native_index
    registry, seed = context['registry'], context['seed']
    binding = edges.CASE_BINDINGS[case_id]
    user, wrapper = edges._request(view, fact, case_id, contract, ordinal)
    row_id = 's21p_' + digest([case_id, fact.layer_id, fact.gene_ids, fact.split, user, view.graph_version])[:32]
    saved = edges._context(view, fact, case_id, row_id, index, registry, seed)
    harness = {'active_graph': {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
        'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed} for layer in view.layer_order]},
        'observations': saved}
    steps, answer = edges._run(view, fact, case_id, saved, row_id, registry)
    return {'row_id': row_id, 'messages': [
        {'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json(harness)}, {'role': 'user', 'content': user}, *steps,
        {'role': 'assistant', 'thinking': edges._reason(view, fact, case_id, saved, steps, answer), 'content': answer}],
        'tools': deepcopy(contract['tools']), 'metadata': {
            'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
            'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
            'question_family': 'S2.1', 'catalog_case_id': case_id, 'split': fact.split,
            'split_unit': fact.split_unit, 'selection_seed': seed, 'graph_version': view.graph_version,
            'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids), 'framing': binding['framing'],
            'variant': binding['behavior'], 'source_predicate': binding['source_predicate'],
            'wrapper_index': wrapper, 'wording_index': ordinal,
            'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _multiplex_facts(case_id, context, split):
    """Select matching source pairs once, independently of wrapper-bank size."""
    view, registry, seed = context['view'], context['registry'], context['seed']
    if '_multiplex_pairs' not in context:
        facts, _ = multiplex.select_facts(view, tuple(sorted(registry.symbols_by_id)),
                                          dict.fromkeys(('train', 'validation', 'test'), 64), seed)
        pairs = {fact.gene_ids for fact in facts}
        # Native reciprocal pairs are a sparse stratum and need direct sampling.
        for layer in view.layers.values():
            if layer.directed:
                pairs.update(pair for pair in layer.edges if pair[0] != pair[1] and pair[::-1] in layer.edges)
        context['_multiplex_pairs'] = sorted({pair for p in pairs for pair in (p, p[::-1])},
                                             key=lambda pair: digest([seed, pair]))
    cache = context.setdefault('_multiplex_facts', {})
    key = case_id, split
    if key not in cache:
        chosen = []
        binding = multiplex.CASE_BINDINGS[case_id]
        for pair in context['_multiplex_pairs']:
            if _pair_group(pair, seed)[1] != split:
                continue
            try:
                fact = multiplex._fact(view, pair, binding, seed)
                if multiplex._condition(view, fact, binding, registry):
                    chosen.append(fact)
            except VariantUnavailable:
                continue
            if len(chosen) == 64:
                break
        cache[key] = chosen
    return cache[key]


def _multiplex_row(case_id, fact, context, ordinal):
    """Bind a source pair to the isolated reviewed multiplex extension."""
    binding = multiplex.CASE_BINDINGS[case_id]
    view, bank, contract = context['view'], context['bank'], context['contract']
    variant = 'full_context' if binding.variant == 'single_edge_context' else binding.variant
    row = (multiplex.build_direction_diagnostic(view, fact, binding.framing, ordinal, bank, contract,
                                                seed=context['seed'])
           if binding.condition == 'direction_invalid' else
           multiplex.build_row(view, fact, binding.mode, 1 if binding.mode == 'top_k' else None,
               binding.framing, variant, ordinal, bank, contract, registry=context['registry'], seed=context['seed']))
    if binding.variant == 'single_edge_context':
        multiplex._single_edge_context(row, view)
    return multiplex._tag(row, case_id, binding.variant)


def _neighbor_facts(case_id, context, split):
    """Enumerate small native instances for every exact neighbor contrast."""
    number = int(case_id.split('.')[-1])
    cache = context.setdefault('_neighbor_facts', {})
    key = number, split
    if key in cache:
        return cache[key]
    view, index, seed = context['view'], context['index'].native_index, context['seed']
    direction = 'undirected' if number in (1, 2, 3, 5, 6, 7, 13, 14, 15, 16, 17, 18, 21, 23, 24, 25) else 'in' if number in (9, 19) else 'both' if number in (12, 22) else 'out'
    candidates = []
    for layer_id in view.layer_order:
        layer = view.layers[layer_id]
        if layer.directed != (direction != 'undirected'):
            continue
        genes = (set(context['registry'].symbols_by_id) - view.multiplex_nodes if number == 6 else
                 view.multiplex_nodes - layer.nodes if number == 5 else layer.nodes)
        for gene in sorted(genes):
            if neighbors.gene_group(gene, seed)[1] != split:
                continue
            fact = neighbors._fact(view, index, gene, layer_id, direction, seed)
            counts = list(fact.neighbor_counts.values())
            size = max((n or 0 for n in counts), default=0)
            if number not in (5, 6, 10, 23) and not size:
                continue
            if number == 10 and size:
                continue
            if number in (3, 7, 13, 14, 15, 16, 17, 18, 21, 25) and size < 2:
                continue
            if number == 19 and (gene, gene) not in layer.edges:
                continue
            if number == 20 and size < 731:
                continue
            if number == 22 and not all(counts):
                continue
            if number == 4:
                try:
                    get_symbols([gene], context['registry'])
                except ValueError:
                    continue
            if number == 18:
                native = index.neighbors(layer_id, gene, direction)
                if not any(native[i - 1][1] == native[i][1] for i in range(1, len(native))):
                    continue
            candidates.append((sum(n or 0 for n in counts), layer_id, gene, fact))
    cache[key] = [entry[-1] for entry in sorted(candidates)[:64]]
    return cache[key]


def _neighbor_row(case_id, fact, context, ordinal):
    """Exercise the exact bound, native tie, pages, or saved-context behavior."""
    number = int(case_id.split('.')[-1])
    framing = 'step' if number in (2, 5, 7, 11, 13, 14, 15, 16, 22) else 'direct'
    variant = CASE_BINDINGS[case_id]['variants'][0]
    mode, bound = 'complete', None
    if number in (3, 17, 18, 20, 21, 24):
        mode, bound = 'top_k', 1
    if number == 24:
        bound = max(n or 0 for n in fact.neighbor_counts.values()) + 1
    if number == 18:
        native = context['index'].native_index.neighbors(fact.layer_id, fact.gene_id, fact.direction)
        bound = next(i for i in range(1, len(native)) if native[i - 1][1] == native[i][1])
    if number == 23:
        row = neighbors.build_direction_diagnostic(context['view'], fact, framing, ordinal,
                                                    context['bank'], context['contract'], seed=context['seed'])
    else:
        row = neighbors.build_row(context['view'], fact, mode, bound, framing, variant, ordinal,
            context['bank'], context['contract'], index=context['index'].native_index,
            registry=context['registry'], seed=context['seed'])
    row['metadata']['catalog_case_id'] = case_id
    return row


def production_candidates(case_id, context, *, split):
    """Yield distinct fact/wording candidates without changing source partitions."""
    family = int(case_id.split('.')[1])
    selectors = {1: _edge_facts, 2: _multiplex_facts, 3: _neighbor_facts}
    builders = {1: _edge_row, 2: _multiplex_row, 3: _neighbor_row}
    facts = selectors[family](case_id, context, split)
    for fact in facts:
        for ordinal in range(32 if split == 'train' else 8):
            try:
                yield builders[family](case_id, fact, context, ordinal)
            except VariantUnavailable:
                continue


def validate_row(row, context):
    """Replay every emitted native observation using its frozen family verifier."""
    family = int(row['metadata']['question_family'].split('.')[1])
    kwargs = {'contract': context['contract'], 'registry': context['registry']}
    if family == 1:
        edges.validate_case_row(row, context['view'], index=context['index'].native_index, **kwargs)
    elif family == 2:
        multiplex.validate_case_row(row, context['view'], bank=context['bank'], **kwargs)
    else:
        neighbors.validate_row(row, context['view'], index=context['index'].native_index, bank=context['bank'], **kwargs)
