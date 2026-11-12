"""Prove exact full-layer paths and calibrate claims from visible observations.

The planner reads manifest metadata only. A route proves reachability, while
an exact path also needs minimum length and the canonical tied witness.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from .s2_9_evidence import _edge_key, _scope
from .s2_9_evidence import collect as _subgraph_collect
from .s2_10_evidence import _facts as _component_facts
from .s2_graph_claims import CLAIM_VOCABULARY, claim_card
from .s2_multiplex import EvidenceError, SelectionError
from .s2_paths import PathRuntime, path_result
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN
from .tools import _GENE_ID_WITH_SUFFIX, _NUMERIC_VERSION


@dataclass(frozen=True)
class PathProof:
    """Return a proved result, an exact next call, or an explicit truth issue."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]
    status: str = 'resolved'
    coverage: None = None
    issue_code: str | None = None


@dataclass
class _Evidence:
    """Keep checked visible facts without a reference to source inventories."""

    categories: dict
    arcs: set
    positives: dict
    negatives: set
    full_domains: list
    full_native: dict
    paths: list
    used: tuple


def _provenance(observations: Sequence[Mapping]) -> dict[str, list]:
    result = {'registry': [], 'graph': []}
    for observation in observations:
        for channel in result:
            for record in observation['provenance'][channel]:
                if record not in result[channel]:
                    result[channel].append(deepcopy(record))
    return result


def _search(arcs, source, target=None):
    """Find a canonical route in visible positive arcs and their reached domain."""
    adjacency = {}
    for left, right in arcs:
        adjacency.setdefault(left, set()).add(right)
    parents, distances, pending = {source: None}, {source: 0}, deque([source])
    while pending:
        node = pending.popleft()
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in parents:
                parents[neighbor] = node
                distances[neighbor] = distances[node] + 1
                pending.append(neighbor)
    route = []
    if target in parents:
        node = target
        while node is not None:
            route.append(node)
            node = parents[node]
        route.reverse()
    return route, distances


def validate_path_observation(view: GraphView, observation: Mapping[str, Any]) -> dict:
    """Check an attributed path response without a native graph lookup."""
    source, result, provenance = observation['arguments'], observation['result'], observation['provenance']
    try:
        genes = PathRuntime(view).validate_arguments(source)
    except (SelectionError, TypeError) as error:
        raise EvidenceError('a saved path has invalid arguments') from error
    fields = {'multiplex_id', 'layer_id', 'directed', 'direction', 'gene_ids',
        'path_found', 'path_gene_ids', 'hop_count', 'missing_gene_ids', 'not_in_layer_gene_ids'}
    if (not isinstance(result, Mapping) or set(result) != fields
            or any(result[key] != source[key] for key in ('multiplex_id', 'layer_id', 'gene_ids', 'direction'))
            or type(result['directed']) is not bool or result['directed'] != view.layers[source['layer_id']].directed):
        raise EvidenceError('a saved path has inconsistent scope, direction, or endpoint order')
    for field in ('missing_gene_ids', 'not_in_layer_gene_ids'):
        values = result[field]
        if (not isinstance(values, list) or any(gene not in genes for gene in values)
                or values != sorted(set(values))):
            raise EvidenceError('path membership categories must be sorted requested sets')
    missing, outside = result['missing_gene_ids'], result['not_in_layer_gene_ids']
    found, route, hops = result['path_found'], result['path_gene_ids'], result['hop_count']
    if set(missing) & set(outside):
        raise EvidenceError('a missing multiplex endpoint cannot also be outside the layer')
    if missing:
        valid = observation['status'] == 'not_found' and found is None and route is None and hops is None
    elif outside:
        valid = observation['status'] == 'resolved' and found is False and route == [] and hops is None
    elif found is False:
        valid = observation['status'] == 'resolved' and route == [] and hops is None and genes[0] != genes[1]
    else:
        valid = (observation['status'] == 'resolved' and found is True
            and isinstance(route, list) and bool(route) and type(hops) is int and hops == len(route) - 1
            and route[0] == genes[0] and route[-1] == genes[1]
            and all(isinstance(gene, str) and GENE_ID_PATTERN.fullmatch(gene) for gene in route)
            and len(route) == len(set(route)))
    records = provenance['graph']
    if (not valid or len(records) != 1 or provenance['registry']
            or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}
            or records[0]['layer_scope'] != _scope(source['layer_id'])):
        raise EvidenceError('a saved path has inconsistent traversal, membership, or provenance')
    return {gene: 'missing_gene_ids' if gene in missing else
            'not_in_layer_gene_ids' if gene in outside else 'layer_gene_ids' for gene in genes}


def _registry_key(operation, source, result):
    """Match the preserved raw query to S1's current canonical identity rules."""
    if not isinstance(source, Mapping) or not isinstance(result, Mapping):
        raise EvidenceError('a resolved registry observation requires argument and result objects')
    gene, symbol = result.get('gene_id'), result.get('gene_symbol')
    if (set(result) != {'gene_id', 'gene_symbol'} or not isinstance(gene, str)
            or not GENE_ID_PATTERN.fullmatch(gene)
            or symbol is not None and (not isinstance(symbol, str) or not symbol)):
        raise EvidenceError('a resolved registry observation has invalid canonical identity fields')
    if operation == 'lookup_human_gene_symbol':
        raw = source.get('gene_symbol')
        if (set(source) != {'gene_symbol'} or not isinstance(raw, str) or not raw.strip()
                or symbol is None or raw.strip().casefold() != symbol.casefold()):
            raise EvidenceError('a resolved symbol query does not match its canonical current symbol')
        return operation, raw.strip().casefold()
    raw = source.get('gene_id')
    match = _GENE_ID_WITH_SUFFIX.fullmatch(raw.strip()) if isinstance(raw, str) else None
    if (set(source) != {'gene_id'} or match is None or match[1].upper() != gene
            or match[2] and _NUMERIC_VERSION.fullmatch(match[2]) is None):
        raise EvidenceError('a resolved ID query does not match its canonical current identifier')
    return operation, gene


def _inspect(view: GraphView, arguments: Mapping, observations: Sequence[Mapping]) -> _Evidence:
    """Validate records, then retain relevant current-scope facts and acquisitions."""
    genes = PathRuntime(view).validate_arguments(arguments)
    native = {key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_id', 'gene_ids')}
    native.update(gene_ids=list(dict.fromkeys(genes)), mode='complete', top_k=None, cursor=None)
    # Reuse the earlier page, scope, identity-ID, and membership checks.
    validated = _subgraph_collect(view, native, observations)
    categories, positives, negatives, domains, full_native, _ = _component_facts(view, arguments['layer_id'], observations)
    layer = arguments['layer_id']
    paths, active, registry, global_values = [], [], {}, {}
    registry_genes, resolved_symbols, observed_symbols = {}, {}, {}
    all_pairs = {key for key in positives}

    def membership(gene, category, selected):
        present = category != 'missing_gene_ids'
        if gene in global_values and global_values[gene] != present:
            raise EvidenceError('a saved path contradicts known multiplex membership')
        global_values[gene] = present
        if not present or selected == layer:
            if gene in categories and categories[gene] != category:
                raise EvidenceError('a saved path contradicts known layer membership')
            categories[gene] = category

    for gene, category in categories.items():
        global_values[gene] = category != 'missing_gene_ids'
    for (selected, pair), _ in positives.items():
        for gene in pair:
            membership(gene, 'layer_gene_ids', selected)
    for observation in observations:
        operation, source, result = observation['operation'], observation['arguments'], observation['result']
        records = observation['provenance']['graph']
        if not records or any(record.get('graph_version') != view.graph_version
                              or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if operation in ('get_nodes_by_layer', 'induce_subgraph', 'get_edge'):
            checked = source['gene_ids']
            absent = set(result['missing_gene_ids'])
        elif operation == 'check_multiplex_membership':
            checked, absent = source['gene_ids'], set(result['absent_gene_ids'])
        elif operation in ('get_neighbors', 'get_neighbor_layers', 'get_gene_layers'):
            checked = [source['gene_id']]
            absent = {source['gene_id']} if result.get('present') is False else set(result.get('missing_gene_ids', ()))
        else:
            continue
        for gene in checked:
            present = gene not in absent
            if gene in global_values and global_values[gene] != present:
                raise EvidenceError('visible acquisitions contradict multiplex membership')
            global_values[gene] = present
    for observation in observations:
        operation, source, result = observation['operation'], observation['arguments'], observation['result']
        if operation in ('lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id'):
            if observation['status'] == 'resolved':
                key = _registry_key(operation, source, result)
                gene, symbol = result['gene_id'], result['gene_symbol']
                if observation['provenance']['graph'] or not observation['provenance']['registry']:
                    raise EvidenceError('a resolved registry observation has inconsistent identity fields')
                if key in registry and registry[key] != result:
                    raise EvidenceError('saved registry identities disagree about one lookup')
                registry[key] = result
                if gene in registry_genes and registry_genes[gene] != symbol:
                    raise EvidenceError('saved registry identities give one ID different current symbols')
                registry_genes[gene] = symbol
                if symbol is not None:
                    observed_symbols.setdefault(symbol.casefold(), set()).add(gene)
                    if operation == 'lookup_human_gene_symbol':
                        resolved_symbols[symbol.casefold()] = gene
            continue
        records = observation['provenance']['graph']
        if not records or any(record.get('graph_version') != view.graph_version
                              or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if operation not in ('shortest_path', 'get_nodes_by_layer', 'induce_subgraph', 'get_edge',
                'get_neighbors', 'get_neighbor_layers', 'get_gene_layers', 'check_multiplex_membership'):
            continue
        if observation not in active:
            active.append(observation)
        if operation != 'shortest_path':
            continue
        partition = validate_path_observation(view, observation)
        selected = source['layer_id']
        for gene, category in partition.items():
            membership(gene, category, selected)
        if result['path_found']:
            for gene in result['path_gene_ids']:
                membership(gene, 'layer_gene_ids', selected)
            for pair in zip(result['path_gene_ids'], result['path_gene_ids'][1:]):
                all_pairs.add((selected, _edge_key(pair, view.layers[selected].directed)))
        paths.append(observation)
    if any(observed_symbols[symbol] != {gene} for symbol, gene in resolved_symbols.items()):
        raise EvidenceError('a resolved symbol conflicts with other observed registry IDs')
    for selected, pair in all_pairs:
        if (selected, pair) in negatives:
            raise EvidenceError('a visible route contradicts a checked negative edge')
        for domain_layer, domain, edges in domains:
            if selected == domain_layer and set(pair) <= domain and pair not in edges:
                raise EvidenceError('a visible route contradicts a complete induced projection')
        for gene, direction, neighbor in ((pair[0], 'out', pair[1]), (pair[1], 'in', pair[0])) if view.layers[selected].directed else (
                (pair[0], 'undirected', pair[1]), (pair[1], 'undirected', pair[0])):
            values = full_native.get((selected, gene, direction))
            if values is not None and neighbor not in values:
                raise EvidenceError('a visible route contradicts a complete native list')
    arcs_by_layer = {}
    for selected, pair in all_pairs:
        arcs = arcs_by_layer.setdefault(selected, set())
        arcs.add(pair)
        if not view.layers[selected].directed:
            arcs.add(tuple(reversed(pair)))
    for observation in paths:
        source, result = observation['arguments'], observation['result']
        if result['path_found'] is None or result['not_in_layer_gene_ids']:
            continue
        route, _ = _search(arcs_by_layer.get(source['layer_id'], ()), *source['gene_ids'])
        if route and (result['path_found'] is False or len(route) < len(result['path_gene_ids'])
                or len(route) == len(result['path_gene_ids']) and route < result['path_gene_ids']):
            raise EvidenceError('visible routes contradict a saved exact shortest result')
        for other in paths:
            if other is not observation and other['arguments'] == source and other['result'] != result:
                raise EvidenceError('matching exact path acquisitions disagree')
    arcs = arcs_by_layer.get(layer, set())
    closure = set(genes)
    while True:
        expanded = closure | {right for left, right in arcs if left in closure} | {left for left, right in arcs if right in closure}
        if expanded == closure:
            break
        closure = expanded
    used = []
    for observation in active:
        source, result = observation['arguments'], observation['result']
        same_layer = source.get('layer_id') == layer
        subjects = set(source.get('gene_ids', ())) | ({source['gene_id']} if 'gene_id' in source else set())
        global_absence = set(result.get('missing_gene_ids', ())) & set(genes)
        if (observation in validated.observations or same_layer and subjects & closure or global_absence) and observation not in used:
            used.append(observation)
    return _Evidence(categories, arcs, positives, negatives, domains, full_native, paths, tuple(used))


def _known_absent(view: GraphView, arguments: Mapping, facts: _Evidence, pair) -> bool:
    layer, directed = arguments['layer_id'], view.layers[arguments['layer_id']].directed
    key = _edge_key(pair, directed)
    if tuple(pair) in facts.arcs:
        return False
    return ((layer, key) in facts.negatives
        or any(selected == layer and set(key) <= domain and key not in edges for selected, domain, edges in facts.full_domains)
        or (layer, pair[0], 'out' if directed else 'undirected') in facts.full_native
            and pair[1] not in facts.full_native[layer, pair[0], 'out' if directed else 'undirected']
        or (layer, pair[1], 'in' if directed else 'undirected') in facts.full_native
            and pair[0] not in facts.full_native[layer, pair[1], 'in' if directed else 'undirected'])


def _derive(view: GraphView, arguments: Mapping, facts: _Evidence) -> list[str] | None:
    """Prove a canonical route or a closed reachable domain from full incidence."""
    source, target = arguments['gene_ids']
    route, distances = _search(facts.arcs, source, target)
    if route and len(route) <= 2:
        return route
    layer, directed = arguments['layer_id'], view.layers[arguments['layer_id']].directed
    direction = 'out' if directed else 'undirected'
    if route and all((layer, gene, direction) in facts.full_native
                     for gene, distance in distances.items() if distance < len(route) - 1):
        return route
    if not route and all((layer, gene, direction) in facts.full_native for gene in distances):
        return []
    if not route:
        _, backwards = _search({(right, left) for left, right in facts.arcs}, target)
        reverse = 'in' if directed else 'undirected'
        if all((layer, gene, reverse) in facts.full_native for gene in backwards):
            return []
    return None


def _proof(view, arguments, facts, *, result=None, status='resolved', observations=None, issue_code=None):
    used = facts.used if observations is None else tuple(observations)
    pending = deepcopy(dict(arguments)) if result is None and issue_code is None else None
    return PathProof(result, pending, 'shortest_path' if pending else None, tuple(used),
        _provenance(used), status, None, issue_code)


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> PathProof:
    """Return exact path evidence or request one full-layer native path search."""
    facts = _inspect(view, arguments, observations)
    for observation in facts.paths:
        if observation['arguments'] == arguments:
            return _proof(view, arguments, facts, result=deepcopy(observation['result']),
                status=observation['status'], observations=(observation,))
    genes = arguments['gene_ids']
    if all(gene in facts.categories for gene in genes):
        missing = sorted({gene for gene in genes if facts.categories[gene] == 'missing_gene_ids'})
        outside = sorted({gene for gene in genes if facts.categories[gene] == 'not_in_layer_gene_ids'})
        if missing or outside or genes[0] == genes[1]:
            route = None if missing else [] if outside else [genes[0]]
            result = path_result(view, arguments, missing, outside, route)
            return _proof(view, arguments, facts, result=result, status='not_found' if missing else 'resolved')
        for observation in facts.paths:
            source, result = observation['arguments'], observation['result']
            if (not result['directed'] and source['layer_id'] == arguments['layer_id']
                    and source['gene_ids'] == list(reversed(genes)) and result['path_found'] is False):
                return _proof(view, arguments, facts, result=path_result(view, arguments, [], [], []),
                              observations=(observation,))
        route = _derive(view, arguments, facts)
        if route is not None:
            return _proof(view, arguments, facts, result=path_result(view, arguments, [], [], route))
    return _proof(view, arguments, facts)


def _universal_counterexample(view: GraphView, arguments: Mapping, observations: Sequence[Mapping], facts: _Evidence):
    """Find a visible path in any registered layer without a claim of transfer."""
    source, target = arguments['gene_ids']
    for layer in view.layer_order:
        scoped = {**arguments, 'layer_id': layer,
            'direction': 'source_to_target' if view.layers[layer].directed else 'undirected'}
        counter = facts if layer == arguments['layer_id'] else _inspect(view, scoped, observations)
        route, _ = _search(counter.arcs, source, target)
        if (source == target and counter.categories.get(source) == 'layer_gene_ids'
                or source != target and route):
            return counter.used
    return None


def _universal_pair_observations(view: GraphView, arguments: Mapping, observations: Sequence[Mapping]):
    """Retain checked local pair results that bear on an all-layer proposition."""
    genes = arguments['gene_ids']
    retained = []
    for observation in observations:
        if observation['operation'] not in ('shortest_path', 'get_edge'):
            continue
        source, records = observation['arguments'], observation['provenance']['graph']
        if not isinstance(source, Mapping):
            continue
        layer = source.get('layer_id')
        if (layer not in view.layers or source.get('multiplex_id') != view.multiplex_id or not records
                or any(record.get('multiplex_id') != view.multiplex_id
                       or record.get('graph_version') != view.graph_version for record in records)):
            continue
        pair = source.get('gene_ids')
        if pair == genes or not view.layers[layer].directed and pair == list(reversed(genes)):
            retained.append(observation)
    return retained


def _retain(observations: Sequence[Mapping], selected: Sequence[Mapping]) -> tuple:
    """Preserve acquisition order and remove only identical repeated records."""
    used = []
    for observation in observations:
        if observation in selected and observation not in used:
            used.append(observation)
    return tuple(used)


def calibrate(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
              *, path_requested: bool = True, claim: str | None = None,
              truth_scope: str | None = None, membership_cards: bool = False) -> PathProof:
    """Assess scoped evidence without treating an unsupported claim as false."""
    if type(path_requested) is not bool or type(membership_cards) is not bool:
        raise ValueError('path and membership-card requests must be explicit booleans')
    if claim is not None and claim not in CLAIM_VOCABULARY:
        raise ValueError('a support request requires an approved canonical claim')
    if truth_scope not in (None, 'scope', 'all_layers') or truth_scope is not None and claim is None:
        raise ValueError('a truth request requires an explicit scope and canonical claim')
    if truth_scope == 'all_layers' and claim != 'absence_holds_in_other_layers':
        raise ValueError('an all-layer absence verdict requires its canonical support category')
    proof = collect(view, arguments, observations)
    facts, universal_used, counterexample = None, (), None
    if truth_scope == 'all_layers':
        facts = _inspect(view, arguments, observations)
        counterexample = _universal_counterexample(view, arguments, observations, facts)
        universal_used = _retain(observations, [*_universal_pair_observations(view, arguments, observations),
                                               *(counterexample or ())])
        used = _retain(observations, [*proof.observations, *universal_used])
        proof = replace(proof, observations=used, provenance=_provenance(used))
    if path_requested and (proof.result is None or proof.status != 'resolved'):
        return proof
    if facts is None:
        facts = _inspect(view, arguments, observations)
    genes, layer = arguments['gene_ids'], arguments['layer_id']
    path_supported = set()
    complete_membership = all(gene in facts.categories for gene in genes)
    missing = [gene for gene in dict.fromkeys(genes) if facts.categories.get(gene) == 'missing_gene_ids']
    outside = [gene for gene in dict.fromkeys(genes) if facts.categories.get(gene) == 'not_in_layer_gene_ids']
    if proof.result is not None and proof.result['path_found'] is not None:
        if proof.result['path_found'] is False:
            path_supported.update(CLAIM_VOCABULARY[:2])
        elif proof.result['hop_count'] >= 2:
            path_supported.add(CLAIM_VOCABULARY[0])
    supported = set(path_supported)
    if not view.layers[layer].directed and any(
            observation['arguments']['layer_id'] == layer
            and observation['arguments']['gene_ids'] == list(reversed(genes))
            and observation['result']['path_found'] is True
            and observation['result']['hop_count'] >= 2 for observation in facts.paths):
        # Reversal preserves the minimum distance, even when its canonical
        # witness still needs a separate tie proof for an ordinary path task.
        supported.add(CLAIM_VOCABULARY[0])
    if complete_membership and not missing and (outside or _known_absent(view, arguments, facts, genes)):
        supported.add(CLAIM_VOCABULARY[0])
    for observation in facts.used:
        source, result = observation['arguments'], observation['result']
        if (observation['operation'] == 'get_neighbors' and source['layer_id'] == layer
                and source['mode'] == 'top_k' and source['cursor'] is None
                and source['gene_id'] == genes[0] and source['direction'] in ('out', 'undirected')
                and complete_membership and not missing and not outside
                and genes[1] not in {row['gene_id'] for row in result['neighbors']}):
            supported.add('not_in_shown_top_k')
    statement_true = None
    if truth_scope is not None:
        route, _ = _search(facts.arcs, *genes)
        if not complete_membership or missing or outside:
            route = []
        if truth_scope == 'all_layers':
            if counterexample is not None:
                statement_true = False
        elif truth_scope == 'scope' and claim == 'no_path_recorded_in_scope':
            if route:
                statement_true = False
            elif 'no_path_recorded_in_scope' in supported:
                statement_true = True
        elif truth_scope == 'scope' and claim == 'no_edge_recorded_in_scope':
            if tuple(genes) in facts.arcs:
                statement_true = False
            elif 'no_edge_recorded_in_scope' in supported:
                statement_true = True
        if statement_true is None:
            used = _retain(observations, [*facts.used, *universal_used])
            return _proof(view, arguments, facts, status='invalid_input', observations=used,
                          issue_code='UNSUPPORTED_TRUTH_SCOPE')
    cards = [claim_card(arguments, genes, supported, claim=claim, statement_true=statement_true)]
    if membership_cards:
        for gene in dict.fromkeys(genes):
            if gene in missing:
                cards.append(claim_card(arguments, [gene], ['gene_not_node_in_multiplex']))
            elif gene in outside:
                cards.append(claim_card(arguments, [gene], [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer']))
    result = deepcopy(proof.result) if path_requested else {
        **{key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_id', 'direction', 'gene_ids')},
        'directed': view.layers[layer].directed}
    result['claim_cards'] = cards
    used = proof.observations if path_requested and supported == path_supported else facts.used
    if truth_scope == 'all_layers':
        used = _retain(observations, [*used, *universal_used])
    return _proof(view, arguments, facts, result=result, observations=used)
