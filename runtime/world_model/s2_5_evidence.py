"""Prove scoped gene memberships from visible observations and manifest metadata.

The planner never reads source node inventories or adjacency. It preserves
original acquisitions and obtains only the unresolved membership selection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError
from .s2_neighbors import _selection
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class MembershipProof:
    """Return a proved membership result or its exact next acquisition."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    coverage: dict
    provenance: dict[str, list]


def _coverage(selected: Sequence[str], checked: Sequence[str] = ()) -> dict:
    """Keep selection evaluation separate from exhaustive positive retrieval."""
    return {'operation': 'layer_membership', 'requested_layer_ids': list(selected),
            'checked_layer_ids': [layer for layer in selected if layer in checked],
            'unchecked_layer_ids': [layer for layer in selected if layer not in checked]}


def _ordinary(runtime: GeneLayersRuntime, arguments: Mapping[str, Any], selected: Sequence[str],
              present: bool, layers: Sequence[str], total: int | None, full: bool) -> dict:
    """Serialize a derived classification without changing the requested scope."""
    return {'gene_id': arguments['gene_id'], 'multiplex_id': arguments['multiplex_id'],
            'layer_scope': runtime.multiplex._scope(arguments['layer_scope'], selected),
            'present': present, 'layer_ids': list(layers), 'layer_count': total,
            'coverage': _coverage(selected, selected),
            'collection': {'mode': arguments['mode'], 'returned_count': len(layers), 'total_count': total,
                           'complete': full and len(layers) == total, 'next_cursor': None}}


def _traversal(runtime: GeneLayersRuntime, pages: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> dict:
    """Check one exact cursor chain and accumulate its visible positive prefix."""
    by_cursor = {}
    for page in pages:
        cursor = page['arguments']['cursor']
        if cursor in by_cursor and by_cursor[cursor]['result'] != page['result']:
            raise EvidenceError('one cursor names contradictory membership pages')
        by_cursor[cursor] = page
    if None not in by_cursor:
        raise EvidenceError('saved membership traversal lacks its initial null-cursor page')
    cursor, visited, layers, total, full, present = None, set(), [], None, False, None
    order = {layer: index for index, layer in enumerate(runtime.view.layer_order)}
    while cursor in by_cursor:
        if cursor in visited:
            raise EvidenceError('saved membership cursor chain loops')
        visited.add(cursor)
        page = by_cursor[cursor]
        arguments, result = page['arguments'], page['result']
        if (set(result) != {'gene_id', 'multiplex_id', 'layer_scope', 'present', 'layer_ids', 'layer_count', 'coverage', 'collection'}
                or result['gene_id'] != arguments['gene_id'] or result['multiplex_id'] != arguments['multiplex_id']
                or result['layer_scope'] != runtime.multiplex._scope(arguments['layer_scope'], selected)
                or type(result['present']) is not bool or page['status'] != 'resolved'
                or result['coverage'] != _coverage(selected, selected)):
            raise EvidenceError('saved membership page identity or coverage is inconsistent')
        if present is not None and present != result['present']:
            raise EvidenceError('membership pages contradict global presence')
        present = result['present']
        rows, collection = result['layer_ids'], result['collection']
        if (not isinstance(rows, list) or any(layer not in selected for layer in rows)
                or rows != [layer for layer in selected if layer in rows]):
            raise EvidenceError('saved membership rows violate scope or manifest order')
        if (not isinstance(collection, dict)
                or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
                or collection['mode'] != arguments['mode'] or type(collection['returned_count']) is not int
                or collection['returned_count'] != len(rows) or type(collection['complete']) is not bool):
            raise EvidenceError('saved membership collection metadata is inconsistent')
        count = collection['total_count']
        if count is not None and (type(count) is not int or not 0 <= count <= len(selected)):
            raise EvidenceError('saved membership total is outside its selected scope')
        if type(result['layer_count']) is not type(count) or result['layer_count'] != count:
            raise EvidenceError('saved membership total locations disagree')
        if count is not None:
            if total is not None and count != total:
                raise EvidenceError('inconsistent non-null membership totals')
            total = count
            if collection['complete'] != (len(rows) == total):
                raise EvidenceError('membership page completeness disagrees with its full total')
        if not present and (rows or count != 0 or not collection['complete']):
            raise EvidenceError('global absence requires a complete empty membership list')
        for layer in rows:
            if layer in layers:
                continue
            if layers and order[layer] < order[layers[-1]]:
                raise EvidenceError('membership pages violate global manifest order')
            layers.append(layer)
        if total is not None and len(layers) > total:
            raise EvidenceError('membership rows exceed the known total')
        cursor = collection['next_cursor']
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise EvidenceError('saved membership cursor must be nonempty or null')
        if collection['complete']:
            if cursor is not None or len(layers) != len(rows):
                raise EvidenceError('complete membership page conflicts with its cursor chain')
            total, full = len(layers), True
        if cursor is None:
            break
    if len(visited) != len(by_cursor):
        raise EvidenceError('saved membership pages are disconnected from the initial cursor')
    arguments = pages[0]['arguments']
    if arguments['mode'] == 'top_k' and len(layers) > arguments['top_k']:
        raise EvidenceError('bounded membership traversal exceeds its requested K')
    if cursor is None:
        if arguments['mode'] == 'complete' or len(layers) < arguments['top_k']:
            if total is not None and total != len(layers):
                raise EvidenceError('exhausted membership traversal contradicts its full total')
            total, full = len(layers), True
        elif total is not None and len(layers) == total:
            full = True
    return {'selected': tuple(selected), 'layers': layers, 'total': total, 'full': full,
            'cursor': cursor, 'present': present, 'pages': pages,
            'mode': arguments['mode'], 'top_k': arguments['top_k']}


def _neighbor_membership(runtime: GeneLayersRuntime, observation: Mapping[str, Any]) -> tuple[bool, dict, dict]:
    """Read explicit native membership fields without using the neighbor count."""
    arguments, result = observation['arguments'], observation['result']
    if (set(arguments) != {'multiplex_id', 'layer_id', 'gene_id', 'direction', 'mode', 'top_k', 'cursor'}
            or set(result) != {'gene_id', 'multiplex_id', 'layer_id', 'direction', 'neighbors', 'missing_gene_ids',
                               'not_in_layer_gene_ids', 'collection'}):
        raise EvidenceError('unexpected native membership fields')
    gene, layer = arguments['gene_id'], arguments.get('layer_id')
    if layer not in runtime.view.layers:
        raise EvidenceError('saved native membership names an unregistered layer')
    allowed = ('in', 'out') if runtime.view.layers[layer].directed else ('undirected',)
    if arguments.get('direction') not in allowed:
        raise EvidenceError('saved neighbor acquisition uses an incompatible representation')
    _selection(arguments.get('mode'), arguments.get('top_k'))
    if any(result.get(key) != arguments[key] for key in ('multiplex_id', 'gene_id', 'layer_id', 'direction')):
        raise EvidenceError('saved neighbor subjects differ from its acquisition')
    missing, outside = result.get('missing_gene_ids'), result.get('not_in_layer_gene_ids')
    if missing not in ([], [gene]) or outside not in ([], [gene]) or missing and outside:
        raise EvidenceError('invalid explicit neighbor membership fields')
    if missing:
        if observation['status'] != 'not_found' or result.get('neighbors') is not None or result.get('collection') is not None:
            raise EvidenceError('missing neighbor membership must leave adjacency unavailable')
    elif observation['status'] != 'resolved':
        raise EvidenceError('saved neighbor membership requires a resolved acquisition')
    elif not isinstance(result['neighbors'], list) or not isinstance(result['collection'], dict):
        raise EvidenceError('resolved native membership requires an ordinary collection')
    elif outside and (result['neighbors'] or result['collection'].get('total_count') != 0
                      or result['collection'].get('complete') is not True or result['collection'].get('next_cursor') is not None):
        raise EvidenceError('confirmed layer absence requires an empty native result')
    scope = {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None}
    return not bool(missing), {} if missing else {layer: not bool(outside)}, scope


def _map_membership(runtime: GeneLayersRuntime, observation: Mapping[str, Any]) -> tuple[bool, dict, dict]:
    """Use explicit map memberships even when the neighbor rows are bounded."""
    arguments, result = observation['arguments'], observation['result']
    if (set(arguments) != {'multiplex_id', 'gene_id', 'layer_scope', 'direction', 'mode', 'top_k', 'cursor'}
            or set(result) != {'query_gene_id', 'multiplex_id', 'layer_scope', 'direction', 'layer_results', 'neighbors',
                               'unique_neighbor_count', 'missing_gene_ids', 'layers_without_gene', 'coverage', 'collection'}):
        raise EvidenceError('unexpected map membership fields')
    selected = runtime.resolve_scope(arguments['layer_scope'])
    direction = arguments['direction']
    if direction not in ('incident', 'in', 'out', 'undirected') or any(
            direction in ('in', 'out') and not runtime.view.layers[layer].directed
            or direction == 'undirected' and runtime.view.layers[layer].directed for layer in selected):
        raise EvidenceError('saved map membership uses an incompatible representation')
    _selection(arguments['mode'], arguments['top_k'])
    scope = runtime.multiplex._scope(arguments['layer_scope'], selected)
    if (result.get('query_gene_id') != arguments['gene_id'] or result.get('multiplex_id') != arguments['multiplex_id']
            or result.get('layer_scope') != scope or result.get('direction') != arguments.get('direction')):
        raise EvidenceError('saved map subjects differ from its acquisition')
    missing = result.get('missing_gene_ids')
    if missing not in ([], [arguments['gene_id']]):
        raise EvidenceError('invalid saved map global membership')
    if missing:
        if (observation['status'] != 'not_found' or any(result[field] is not None for field in
                ('layer_results', 'neighbors', 'unique_neighbor_count', 'layers_without_gene', 'collection'))
                or result['coverage'] != {'operation': 'layer_neighbor_maps', 'requested_layer_ids': list(selected),
                    'checked_layer_ids': [], 'unchecked_layer_ids': list(selected)}):
            raise EvidenceError('missing map membership must leave layer results unavailable')
        return False, {}, scope
    if observation['status'] != 'resolved' or not isinstance(result.get('layer_results'), list):
        raise EvidenceError('saved map membership requires explicit layer rows')
    rows = result['layer_results']
    if [row.get('layer_id') for row in rows] != list(selected):
        raise EvidenceError('saved map membership does not cover its declared scope')
    facts = {}
    for row in rows:
        layer, absent = row['layer_id'], row.get('not_in_layer_gene_ids')
        if type(row.get('directed')) is not bool or row['directed'] != runtime.view.layers[layer].directed or absent not in ([], [arguments['gene_id']]):
            raise EvidenceError('invalid saved map layer membership')
        facts[layer] = not bool(absent)
    if (result['layers_without_gene'] != [layer for layer in selected if not facts[layer]]
            or result['coverage'] != {'operation': 'layer_neighbor_maps', 'requested_layer_ids': list(selected),
                'checked_layer_ids': list(selected), 'unchecked_layer_ids': []}):
        raise EvidenceError('saved map membership coverage or absence fields disagree')
    return True, facts, scope


def _global_membership(arguments: Mapping[str, Any], observation: Mapping[str, Any]) -> tuple[bool, dict, dict]:
    """Read a checked multiplex partition without inferring any positive layer."""
    source, result = observation['arguments'], observation['result']
    genes = source.get('gene_ids')
    positive, negative = result.get('present_gene_ids'), result.get('absent_gene_ids')
    if (set(source) != {'multiplex_id', 'gene_ids'}
            or set(result) != {'multiplex_id', 'present_gene_ids', 'absent_gene_ids'}
            or observation['status'] != 'resolved' or result.get('multiplex_id') != arguments['multiplex_id']
            or not isinstance(genes, list) or not genes
            or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in genes)
            or len(set(genes)) != len(genes)
            or not isinstance(positive, list) or not isinstance(negative, list)
            or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in (*positive, *negative))
            or len(set(positive)) != len(positive) or len(set(negative)) != len(negative)
            or set(positive) & set(negative) or set(positive) | set(negative) != set(genes)):
        raise EvidenceError('saved multiplex membership does not partition its inputs')
    return arguments['gene_id'] in positive, {}, {'kind': 'all', 'layer_ids': [], 'layer_family': None}


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> MembershipProof:
    """Prove the requested membership list or acquire exactly the unresolved scope.

    Complete scoped observations supply both positive and negative memberships.
    Bounded lists supply only their visible positive prefix. Checked coverage
    never substitutes for omitted identities or an exhaustive negative.
    """
    runtime = GeneLayersRuntime(view)
    selected = runtime._validate(arguments)
    if arguments['cursor'] is not None:
        raise EvidenceError('a membership collection request starts at its initial null cursor')
    groups, used, identities, global_values, explicit = defaultdict(list), [], {}, set(), {}

    def fact(layer: str, present: bool) -> None:
        if layer in explicit and explicit[layer] != present:
            raise EvidenceError('saved records contradict layer membership')
        explicit[layer] = present

    for observation in observations:
        operation, source = observation.get('operation'), observation.get('arguments', {})
        if operation not in ('get_gene_layers', 'get_neighbors', 'get_neighbor_layers', 'check_multiplex_membership'):
            continue
        if source.get('multiplex_id') != arguments['multiplex_id']:
            continue
        if operation == 'check_multiplex_membership':
            if arguments['gene_id'] not in source.get('gene_ids', []):
                continue
        elif source.get('gene_id') != arguments['gene_id']:
            continue
        records = observation.get('provenance', {}).get('graph', [])
        if not records or any(record.get('graph_version') != view.graph_version or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if (len(records) != 1 or observation['provenance'].get('registry') != []
                or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}):
            raise EvidenceError('a graph observation requires one attributed acquisition')
        record = records[0]
        identity = record.get('evidence_id')
        if not isinstance(identity, str) or not identity.strip():
            raise EvidenceError('saved graph acquisition requires an evidence ID')
        if identity in identities and identities[identity] != observation:
            raise EvidenceError('one evidence ID names contradictory acquisitions')
        identities[identity] = observation
        if observation not in used:
            used.append(observation)
        if operation == 'get_gene_layers':
            source_selected = runtime._validate(source)
            expected_scope = runtime.multiplex._scope(source['layer_scope'], source_selected)
            key = (source_selected, source['mode'], source['top_k'], json.dumps(source['layer_scope'], sort_keys=True, separators=(',', ':')))
            groups[key].append(observation)
        else:
            present, facts, expected_scope = (_neighbor_membership(runtime, observation) if operation == 'get_neighbors' else
                                             _map_membership(runtime, observation) if operation == 'get_neighbor_layers' else
                                             _global_membership(arguments, observation))
            global_values.add(present)
            for layer, value in facts.items():
                fact(layer, value)
        if record.get('layer_scope') != expected_scope:
            raise EvidenceError('saved provenance scope differs from its acquisition')

    summaries = []
    for (source_selected, _, _, _), pages in groups.items():
        summary = _traversal(runtime, pages, source_selected)
        global_values.add(summary['present'])
        summaries.append(summary)
        if summary['full']:
            for layer in source_selected:
                fact(layer, layer in summary['layers'])
    if len(global_values) > 1:
        raise EvidenceError('saved records contradict global membership')
    present = next(iter(global_values)) if global_values else None
    for summary in summaries:
        for layer in summary['layers']:
            if layer in explicit and not explicit[layer]:
                raise EvidenceError('saved prefix contradicts checked layer membership')
        positives = [layer for layer in summary['selected'] if explicit.get(layer) is True]
        negatives = [layer for layer in summary['selected'] if explicit.get(layer) is False]
        if (summary['total'] is not None and (summary['total'] < len(positives)
                or summary['total'] > len(summary['selected']) - len(negatives))):
            raise EvidenceError('saved membership total contradicts checked memberships')
        if summary['layers']:
            last = summary['selected'].index(summary['layers'][-1])
            if any(layer not in summary['layers'] for layer in positives if summary['selected'].index(layer) <= last):
                raise EvidenceError('saved membership prefix omits an earlier known positive')
    for index, summary in enumerate(summaries):
        for prior in summaries[:index]:
            if summary['selected'] != prior['selected']:
                continue
            common = min(len(summary['layers']), len(prior['layers']))
            if summary['layers'][:common] != prior['layers'][:common]:
                raise EvidenceError('saved selections disagree on the manifest-ordered prefix')
            if summary['total'] is not None and prior['total'] is not None and summary['total'] != prior['total']:
                raise EvidenceError('inconsistent non-null membership totals')
    if present is False:
        if any(explicit.values()) or any(summary['layers'] for summary in summaries):
            raise EvidenceError('global absence contradicts positive layer membership')
        for layer in selected:
            fact(layer, False)

    def finish(result: dict | None, pending: dict | None, checked: Sequence[str]) -> MembershipProof:
        provenance = {'registry': [], 'graph': []}
        for observation in used:
            for channel in provenance:
                for record in observation['provenance'][channel]:
                    if record not in provenance[channel]:
                        provenance[channel].append(deepcopy(record))
        return MembershipProof(result, pending, 'get_gene_layers' if pending is not None else None,
                               tuple(used), _coverage(selected, checked), provenance)

    if present is not None and all(layer in explicit for layer in selected):
        layers = [layer for layer in selected if explicit[layer]]
        returned = layers if arguments['mode'] == 'complete' else layers[:arguments['top_k']]
        result = _ordinary(runtime, arguments, selected, present, returned, len(layers), True)
        return finish(result, None, selected)
    candidates = [summary for summary in summaries if summary['selected'] == tuple(selected)
                  and arguments['mode'] == 'top_k' and len(summary['layers']) >= arguments['top_k']]
    if candidates:
        candidate = max(candidates, key=lambda summary: (summary['full'], len(summary['layers'])))
        known = {summary['total'] for summary in summaries if summary['selected'] == tuple(selected) and summary['total'] is not None}
        total = next(iter(known)) if known else None
        result = _ordinary(runtime, arguments, selected, present, candidate['layers'][:arguments['top_k']], total, candidate['full'])
        return finish(result, None, selected)

    checked = {layer for layer in selected if layer in explicit}
    for summary in summaries:
        checked.update(layer for layer in summary['selected'] if layer in selected)
    # A matching traversal keeps its exact cursor even when its count is known.
    for summary in summaries:
        source = summary['pages'][0]['arguments']
        if (source['layer_scope'] == arguments['layer_scope'] and (source['mode'], source['top_k']) == (arguments['mode'], arguments['top_k'])
                and summary['cursor'] is not None):
            return finish(None, {**deepcopy(source), 'cursor': summary['cursor']}, checked)
    # A saved complete sub-scope must exhaust its own selection before regrouping.
    for summary in summaries:
        if (summary['mode'] == 'complete' and not summary['full'] and summary['cursor'] is not None
                and set(summary['selected']) < set(selected)):
            source = summary['pages'][0]['arguments']
            return finish(None, {**deepcopy(source), 'cursor': summary['cursor']}, checked)
    completed = [layer for layer in selected if layer in explicit]
    pending = deepcopy(dict(arguments))
    if completed:
        pending.update(layer_scope={'kind': 'layers', 'layer_ids': [layer for layer in selected if layer not in explicit],
                                    'layer_family': None}, mode='complete', top_k=None, cursor=None)
    return finish(None, pending, checked)
