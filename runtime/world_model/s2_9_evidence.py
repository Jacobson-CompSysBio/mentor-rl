"""Prove induced projections from visible membership and native edge records.

The planner reads manifest metadata only. It never reads source node inventories
or adjacency. Complete projections can use sufficient combinations of records.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
import json
import math
from typing import Any

from .s2_5_evidence import _global_membership, _map_membership, _neighbor_membership
from .s2_5_evidence import _traversal as _membership_traversal
from .s2_6_evidence import _coverage, _membership_page, _nodes_partition
from .s2_8_evidence import _native_page
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError, SelectionError
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class SubgraphProof:
    """Return a completed projection, an explicit gate issue, or the next call."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]
    coverage: dict
    issue_code: str | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _edge_key(pair: Sequence[str], directed: bool) -> tuple[str, str]:
    return tuple(pair) if directed else tuple(sorted(pair))


def _scope(layer: str) -> dict:
    return {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None}


def validate_subgraph_observation(view: GraphView, observation: Mapping[str, Any]) -> tuple[tuple[str, ...], dict]:
    """Validate one native page without consulting source memberships or edges."""
    from .s2_subgraphs import SubgraphRuntime
    source, result = observation['arguments'], observation['result']
    try:
        genes = SubgraphRuntime(view).validate_arguments(source)
    except SelectionError as error:
        raise EvidenceError('saved induced acquisition has invalid arguments') from error
    fields = {'multiplex_id', 'layer_id', 'directed', 'layer_gene_ids', 'layer_gene_count',
        'not_in_layer_gene_ids', 'missing_gene_ids', 'edges', 'edge_count', 'coverage', 'collection'}
    if (observation['status'] != 'resolved' or not isinstance(result, Mapping) or set(result) != fields
            or any(result[key] != source[key] for key in ('multiplex_id', 'layer_id'))
            or type(result['directed']) is not bool or result['directed'] != view.layers[source['layer_id']].directed
            or result['coverage'] != _coverage(genes, genes)):
        raise EvidenceError('saved projection identity, representation, or coverage is inconsistent')
    categories = {}
    for category in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids'):
        rows = result[category]
        if (not isinstance(rows, list) or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in rows)
                or rows != sorted(set(rows)) or any(gene in categories for gene in rows)):
            raise EvidenceError('projection categories must be sorted unique disjoint ID lists')
        categories.update(dict.fromkeys(rows, category))
    if set(categories) != set(genes):
        raise EvidenceError('projection categories must partition exactly the acquisition inputs')
    if type(result['layer_gene_count']) is not int or result['layer_gene_count'] != len(result['layer_gene_ids']):
        raise EvidenceError('projection member count differs from its complete classification')
    edges = result['edges']
    if not isinstance(edges, list):
        raise EvidenceError('projection edge rows must be a list')
    keys = []
    for edge in edges:
        if (not isinstance(edge, Mapping) or set(edge) != {'gene_ids', 'weight'}
                or not isinstance(edge['gene_ids'], list) or len(edge['gene_ids']) != 2
                or any(gene not in result['layer_gene_ids'] for gene in edge['gene_ids'])
                or type(edge['weight']) not in (int, float) or not math.isfinite(edge['weight'])):
            raise EvidenceError('projection edge endpoints or native weights are invalid')
        key = _edge_key(edge['gene_ids'], result['directed'])
        if list(key) != edge['gene_ids']:
            raise EvidenceError('an undirected pair must use ascending endpoints')
        keys.append(key)
    if keys != sorted(set(keys)):
        raise EvidenceError('projection edges must have unique lexicographically ordered native keys')
    collection = result['collection']
    if (type(result['edge_count']) is not int or result['edge_count'] != len(edges)
            or not isinstance(collection, Mapping)
            or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
            or collection['mode'] != source['mode'] or type(collection['returned_count']) is not int
            or collection['returned_count'] != len(edges) or type(collection['complete']) is not bool
            or source['mode'] == 'top_k' and len(edges) > source['top_k']):
        raise EvidenceError('projection page counts or collection mode are inconsistent')
    total, cursor = collection['total_count'], collection['next_cursor']
    if (total is not None and (type(total) is not int or total < len(edges))
            or cursor is not None and (not isinstance(cursor, str) or not cursor)
            or total is not None and collection['complete'] != (len(edges) == total)
            or collection['complete'] and cursor is not None):
        raise EvidenceError('projection total, cursor, or complete flag is inconsistent')
    if not result['layer_gene_ids'] and (edges or total not in (None, 0) or not collection['complete']):
        raise EvidenceError('an empty member projection must have a complete empty edge domain')
    return genes, categories


def _chain(pages: Sequence[Mapping], field: str, key) -> dict:
    """Read the connected initial prefix while retaining other pages as positives."""
    by_cursor, total, all_rows = {}, None, {}
    for page in pages:
        source, result = page['arguments'], page['result']
        cursor = source['cursor']
        if cursor in by_cursor and _canonical(by_cursor[cursor]['result']) != _canonical(result):
            raise EvidenceError('one cursor names contradictory pages')
        by_cursor[cursor] = page
        count = result['collection']['total_count']
        if count is not None:
            if total is not None and total != count:
                raise EvidenceError('saved pages disagree about their full total')
            total = count
        for row in result[field]:
            identity = tuple(row['gene_ids']) if field == 'edges' else row['gene_id']
            if identity in all_rows and all_rows[identity]['weight'] != row['weight']:
                raise EvidenceError('saved pages disagree about one native weight')
            all_rows[identity] = row
    if total is not None and len(all_rows) > total:
        raise EvidenceError('visible distinct rows exceed the authoritative full total')
    if None not in by_cursor:
        return {'rows': [], 'total': total, 'full': False, 'cursor': None, 'initial': False, 'exhausted': False}
    cursor, rows, values, visited, declared_full = None, [], {}, set(), False
    while cursor in by_cursor:
        if cursor in visited:
            raise EvidenceError('saved cursor chain loops')
        visited.add(cursor)
        page = by_cursor[cursor]
        for row in page['result'][field]:
            identity = tuple(row['gene_ids']) if field == 'edges' else row['gene_id']
            if identity in values:
                if values[identity] != row['weight']:
                    raise EvidenceError('saved pages disagree about one native weight')
                continue
            if rows and key(row) < key(rows[-1]):
                raise EvidenceError('saved pages violate their canonical collection order')
            rows.append(dict(row))
            values[identity] = row['weight']
        if total is not None and len(rows) > total:
            raise EvidenceError('accumulated rows exceed the authoritative full total')
        collection = page['result']['collection']
        declared_full |= collection['complete']
        if collection['complete'] and len(rows) != len(page['result'][field]):
            raise EvidenceError('a complete single page conflicts with earlier distinct rows')
        cursor = collection['next_cursor']
        if cursor is None:
            break
    source = pages[0]['arguments']
    if rows and any(identity not in values and key(row) < key(rows[-1]) for identity, row in all_rows.items()):
        raise EvidenceError('a detached page contradicts the connected canonical prefix')
    if source['mode'] == 'top_k' and len(rows) > source['top_k']:
        raise EvidenceError('accumulated prefix exceeds the requested preview K')
    exhausted = cursor is None
    full = exhausted and (declared_full or source['mode'] == 'complete' or len(rows) < source['top_k']
                           or total is not None and len(rows) == total)
    if full:
        if total is not None and total != len(rows):
            raise EvidenceError('an exhausted full traversal disagrees with its authoritative total')
        total = len(rows)
    return {'rows': rows, 'total': total, 'full': full, 'cursor': cursor,
            'initial': True, 'exhausted': exhausted}


def _ordinary(view: GraphView, arguments: Mapping, genes: Sequence[str], categories: Mapping[str, str],
              edges: Sequence[dict], total: int | None, full: bool) -> dict:
    members = sorted(gene for gene in genes if categories[gene] == 'layer_gene_ids')
    return {'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
        'directed': view.layers[arguments['layer_id']].directed, 'layer_gene_ids': members,
        'layer_gene_count': len(members),
        'not_in_layer_gene_ids': sorted(gene for gene in genes if categories[gene] == 'not_in_layer_gene_ids'),
        'missing_gene_ids': sorted(gene for gene in genes if categories[gene] == 'missing_gene_ids'),
        'edges': deepcopy(list(edges)), 'edge_count': len(edges), 'coverage': _coverage(genes, genes),
        'collection': {'mode': arguments['mode'], 'returned_count': len(edges), 'total_count': total,
                       'complete': full and total == len(edges), 'next_cursor': None}}


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
            *, require_all_in_layer: bool = False) -> SubgraphProof:
    """Prove every projected pair or request the missing full-input acquisition.

    The optional all-members gate is a separate visible prerequisite. It never
    changes ordinary projection semantics or hides known excluded categories.
    """
    from .s2_subgraphs import SubgraphRuntime
    from .s2_neighbors import _rank
    runtime = SubgraphRuntime(view)
    genes = runtime.validate_arguments(arguments)
    if type(require_all_in_layer) is not bool:
        raise ValueError('the all-members gate must be an explicit boolean')
    if not genes:
        result = _ordinary(view, arguments, genes, {}, [], 0, True)
        return SubgraphProof(result, None, None, (), {'registry': [], 'graph': []}, _coverage(genes))
    layer, requested = arguments['layer_id'], frozenset(genes)
    directed = view.layers[layer].directed
    memberships, nodes = GeneLayersRuntime(view), NodesByLayerRuntime(view)
    global_values, layer_values, categories = {}, {}, {}
    identities, compatible, positives = {}, [], {}
    subgraphs, natives, member_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    full_domains, full_native, summaries, native_summaries = [], {}, [], defaultdict(list)

    def membership(gene, present, facts):
        if gene in global_values and global_values[gene] != present:
            raise EvidenceError('saved observations contradict global membership')
        global_values[gene] = present
        for selected, value in facts.items():
            key = gene, selected
            if key in layer_values and layer_values[key] != value:
                raise EvidenceError('saved observations contradict layer membership')
            layer_values[key] = value
        if present is False:
            categories[gene] = 'missing_gene_ids'
        elif layer in facts:
            categories[gene] = 'layer_gene_ids' if facts[layer] else 'not_in_layer_gene_ids'

    def positive(pair, weight):
        pair = _edge_key(pair, directed)
        if pair in positives and positives[pair] != weight:
            raise EvidenceError('saved observations disagree about one native edge weight')
        positives[pair] = weight
        for gene in pair:
            membership(gene, True, {layer: True})

    for position, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise EvidenceError('saved observations require exact attributed partial-result fields')
        provenance = observation['provenance']
        if not isinstance(provenance, Mapping) or set(provenance) != {'registry', 'graph'}:
            raise EvidenceError('saved observations require both provenance channels')
        for records in provenance.values():
            if not isinstance(records, list):
                raise EvidenceError('saved provenance channels must be lists')
            for record in records:
                identity = record.get('evidence_id') if isinstance(record, Mapping) else None
                if not isinstance(identity, str) or not identity.strip():
                    raise EvidenceError('every saved acquisition requires an evidence ID')
                if identity in identities and _canonical(identities[identity]) != _canonical(observation):
                    raise EvidenceError('one evidence ID names conflicting acquisitions')
                identities[identity] = observation
        operation, source = observation['operation'], observation['arguments']
        if operation == 'lookup_human_gene_symbol':
            continue
        if operation not in ('induce_subgraph', 'get_nodes_by_layer', 'get_neighbors', 'get_gene_layers',
                             'check_multiplex_membership', 'get_neighbor_layers'):
            continue
        records = provenance['graph']
        if not records or any(record.get('graph_version') != view.graph_version
                              or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if (len(records) != 1 or provenance['registry']
                or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}):
            raise EvidenceError('a graph partial requires one original acquisition record')
        if not isinstance(source, Mapping) or source.get('multiplex_id') != arguments['multiplex_id']:
            raise EvidenceError('saved acquisition arguments conflict with their graph provenance')
        relevant = False
        if operation == 'induce_subgraph':
            supplied, partition = validate_subgraph_observation(view, observation)
            if records[0]['layer_scope'] != _scope(source['layer_id']):
                raise EvidenceError('induced acquisition provenance differs from its layer')
            for gene, category in partition.items():
                membership(gene, category != 'missing_gene_ids',
                           {} if category == 'missing_gene_ids' else {source['layer_id']: category == 'layer_gene_ids'})
            relevant = bool(requested & set(supplied))
            if source['layer_id'] == layer:
                key = tuple(supplied), source['mode'], source['top_k']
                subgraphs[key].append((position, observation))
                for edge in observation['result']['edges']:
                    positive(edge['gene_ids'], edge['weight'])
        elif operation == 'get_nodes_by_layer':
            partition, scope = _nodes_partition(nodes, observation)
            if records[0]['layer_scope'] != scope:
                raise EvidenceError('saved membership provenance differs from its layer')
            for gene, category in partition.items():
                membership(gene, category != 'missing_gene_ids',
                           {} if category == 'missing_gene_ids' else {source['layer_id']: category == 'layer_gene_ids'})
            relevant = bool(requested & partition.keys())
        elif operation == 'check_multiplex_membership':
            supplied = source.get('gene_ids', [])
            for gene in supplied:
                present, facts, scope = _global_membership({'multiplex_id': arguments['multiplex_id'], 'gene_id': gene}, observation)
                membership(gene, present, facts)
            if records[0]['layer_scope'] != {'kind': 'all', 'layer_ids': [], 'layer_family': None}:
                raise EvidenceError('global membership provenance must preserve its acquisition scope')
            relevant = bool(requested & set(supplied))
        elif operation == 'get_gene_layers':
            selected = memberships._validate(source)
            page = _membership_page(memberships, observation, selected)
            facts = {item: item in page['layer_ids'] for item in selected} if page['collection']['complete'] else dict.fromkeys(page['layer_ids'], True)
            membership(source['gene_id'], page['present'], facts)
            if records[0]['layer_scope'] != memberships.multiplex._scope(source['layer_scope'], selected):
                raise EvidenceError('saved containing-layer provenance differs from its scope')
            key = source['gene_id'], tuple(selected), source['mode'], source['top_k'], _canonical(source['layer_scope'])
            member_groups[key].append((position, observation))
            relevant = source['gene_id'] in requested
        elif operation == 'get_neighbors':
            present, facts, scope = _neighbor_membership(memberships, observation)
            membership(source['gene_id'], present, facts)
            if records[0]['layer_scope'] != scope:
                raise EvidenceError('saved native-list provenance differs from its layer')
            relevant = source['gene_id'] in requested
            if present:
                _native_page(observation)
            if present and source['layer_id'] == layer:
                key = source['gene_id'], source['direction'], source['mode'], source['top_k']
                natives[key].append((position, observation))
                for row in observation['result']['neighbors']:
                    pair = (row['gene_id'], source['gene_id']) if source['direction'] == 'in' else (source['gene_id'], row['gene_id'])
                    positive(pair, row['weight'])
                    relevant |= row['gene_id'] in requested
        else:
            # Maps can establish membership. Native edge proofs remain in their
            # explicit edge/list records in this version of the corpus builder.
            present, facts, scope = _map_membership(memberships, observation)
            membership(source['gene_id'], present, facts)
            if records[0]['layer_scope'] != scope:
                raise EvidenceError('saved map provenance differs from its selected scope')
            relevant = source['gene_id'] in requested
        if relevant:
            compatible.append(position)

    for (gene, selected, _, _, _), pages in member_groups.items():
        if not any(page['arguments']['cursor'] is None for _, page in pages):
            continue
        summary = _membership_traversal(memberships, [page for _, page in pages], selected)
        facts = {item: item in summary['layers'] for item in selected} if summary['full'] else dict.fromkeys(summary['layers'], True)
        membership(gene, summary['present'], facts)

    for (source_genes, mode, top_k), pages in subgraphs.items():
        partition = {key: pages[0][1]['result'][key] for key in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids')}
        if any(any(page['result'][key] != partition[key] for key in partition) for _, page in pages):
            raise EvidenceError('induced pages disagree about their complete membership partition')
        summary = _chain([page for _, page in pages], 'edges', lambda row: tuple(row['gene_ids']))
        summary.update(genes=frozenset(source_genes), ordered_genes=source_genes, mode=mode, top_k=top_k)
        summaries.append(summary)
        if summary['full']:
            full_domains.append((frozenset(partition['layer_gene_ids']),
                                 {tuple(edge['gene_ids']): edge['weight'] for edge in summary['rows']}))
    for summary in summaries:
        for other in summaries:
            if summary is other or summary['genes'] != other['genes']:
                continue
            if summary['total'] is not None and other['total'] is not None and summary['total'] != other['total']:
                raise EvidenceError('saved selections disagree about the same full induced-edge total')
            common = min(len(summary['rows']), len(other['rows']))
            if summary['initial'] and other['initial'] and summary['rows'][:common] != other['rows'][:common]:
                raise EvidenceError('saved induced selections disagree about their canonical prefix')
    for (gene, direction, _, _), pages in natives.items():
        summary = _chain([page for _, page in pages], 'neighbors', _rank)
        for prior in native_summaries[(gene, direction)]:
            if summary['total'] is not None and prior['total'] is not None and summary['total'] != prior['total']:
                raise EvidenceError('saved native selections disagree about their full total')
            common = min(len(summary['rows']), len(prior['rows']))
            if summary['initial'] and prior['initial'] and summary['rows'][:common] != prior['rows'][:common]:
                raise EvidenceError('saved native selections disagree about their ranked prefix')
        native_summaries[(gene, direction)].append(summary)
        if summary['full']:
            values = {row['gene_id']: row['weight'] for row in summary['rows']}
            key = gene, direction
            if key in full_native and full_native[key] != values:
                raise EvidenceError('complete native lists disagree about their full neighbor set')
            full_native[key] = values
    for pair, weight in positives.items():
        for domain, edges in full_domains:
            if set(pair) <= domain and (pair not in edges or edges[pair] != weight):
                raise EvidenceError('a positive edge conflicts with a complete induced projection')
        for gene, direction in ((pair[0], 'out'), (pair[1], 'in')) if directed else ((pair[0], 'undirected'), (pair[1], 'undirected')):
            if (gene, direction) in full_native:
                neighbor = pair[1] if gene == pair[0] else pair[0]
                if neighbor not in full_native[(gene, direction)] or full_native[(gene, direction)][neighbor] != weight:
                    raise EvidenceError('a positive edge conflicts with a complete native list')

    used = []
    for position in sorted(set(compatible)):
        if observations[position] not in used:
            used.append(observations[position])
    provenance = {'registry': [], 'graph': []}
    for observation in used:
        for channel in provenance:
            for record in observation['provenance'][channel]:
                if record not in provenance[channel]:
                    provenance[channel].append(deepcopy(record))
    coverage = _coverage(genes, categories.keys())
    if require_all_in_layer:
        if len(coverage['checked_gene_ids']) < len(genes):
            pending = {'multiplex_id': arguments['multiplex_id'], 'layer_id': layer, 'gene_ids': list(genes)}
            return SubgraphProof(None, pending, 'get_nodes_by_layer', tuple(used), provenance, coverage)
        if any(categories[gene] != 'layer_gene_ids' for gene in genes):
            return SubgraphProof(None, None, None, tuple(used), provenance, coverage, 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
    known_membership = len(coverage['checked_gene_ids']) == len(genes)
    members = sorted(gene for gene in genes if categories.get(gene) == 'layer_gene_ids')
    projected = frozenset(members)
    complete = known_membership and (not members or any(projected <= domain for domain, _ in full_domains))
    if known_membership and not complete:
        if directed:
            complete = all((gene, 'out') in full_native for gene in members) or all((gene, 'in') in full_native for gene in members)
        else:
            complete = all((gene, 'undirected') in full_native for gene in members)
        if not complete:
            def known(pair):
                return (pair in positives or any(set(pair) <= domain for domain, _ in full_domains)
                        or (pair[0], 'out' if directed else 'undirected') in full_native
                        or (pair[1], 'in' if directed else 'undirected') in full_native)

            # An unsupported loop can stop a large partial proof before any
            # cross-pair work. Complete native rows remove whole pair blocks.
            complete = all(known((gene, gene)) for gene in members)
            if complete:
                left = [gene for gene in members if (gene, 'out' if directed else 'undirected') not in full_native]
                right = [gene for gene in members if (gene, 'in') not in full_native] if directed else left
                pairs = product(left, right) if directed else combinations_with_replacement(left, 2)
                complete = all(known(pair) for pair in pairs)
    if complete:
        edges = [{'gene_ids': list(pair), 'weight': weight} for pair, weight in sorted(positives.items()) if set(pair) <= projected]
        total = len(edges)
        selected = edges if arguments['mode'] == 'complete' else edges[:arguments['top_k']]
        result = _ordinary(view, arguments, genes, categories, selected, total, True)
        return SubgraphProof(result, None, None, tuple(used), provenance, coverage)
    if known_membership and arguments['mode'] == 'top_k':
        for summary in summaries:
            if (summary['genes'] == requested and summary['initial'] and len(summary['rows']) >= arguments['top_k']
                    and summary['total'] != arguments['top_k']):
                rows = summary['rows'][:arguments['top_k']]
                result = _ordinary(view, arguments, genes, categories, rows, summary['total'], False)
                return SubgraphProof(result, None, None, tuple(used), provenance, coverage)
    pending = deepcopy(dict(arguments))
    pending['gene_ids'], pending['cursor'] = list(genes), None
    for summary in summaries:
        if (summary['ordered_genes'] == tuple(genes) and summary['mode'] == arguments['mode']
                and summary['top_k'] == arguments['top_k'] and summary['initial'] and summary['cursor'] is not None):
            pending['cursor'] = summary['cursor']
            break
    return SubgraphProof(None, pending, 'induce_subgraph', tuple(used), provenance, coverage)
