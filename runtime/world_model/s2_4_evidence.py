"""Prove multiplex neighbor maps from visible maps and native collections.

The planner reads manifest metadata and supplied observations. It never reads
source adjacency, source membership inventories, or hidden collection totals.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any

from .s2_multiplex import EvidenceError, MultiplexRuntime, SelectionError
from .s2_neighbors import CollectionProof, NeighborRuntime, _selection, _traversal as _native_traversal
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ('multiplex_id', 'gene_id', 'layer_scope', 'direction', 'mode', 'top_k', 'cursor')


@dataclass(frozen=True)
class MapProof:
    """Supply a proved map or the exact next call and its supported partials."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    coverage: dict
    provenance: dict[str, list]


def _scope(runtime: MultiplexRuntime, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate map arguments using only the registered scope and representation."""
    if set(arguments) != set(ARGUMENT_KEYS):
        raise SelectionError('invalid_input', 'INVALID_SELECTION')
    if not arguments['multiplex_id']:
        raise SelectionError('invalid_input', 'MISSING_SCOPE')
    if arguments['multiplex_id'] != runtime.view.multiplex_id:
        raise SelectionError('not_found', 'UNKNOWN_MULTIPLEX')
    gene = arguments['gene_id']
    if not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene):
        raise SelectionError('invalid_input', 'INVALID_GENE_INPUT')
    selected = runtime.resolve_scope(arguments['layer_scope'])
    direction = arguments['direction']
    if direction not in ('incident', 'in', 'out', 'undirected') or any(
            direction in ('in', 'out') and not runtime.view.layers[layer].directed
            or direction == 'undirected' and runtime.view.layers[layer].directed for layer in selected):
        raise SelectionError('invalid_input', 'DIRECTION_UNAVAILABLE')
    _selection(arguments['mode'], arguments['top_k'])
    cursor = arguments['cursor']
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise SelectionError('invalid_input', 'INVALID_CURSOR')
    return selected


def _coverage(selected: Sequence[str], checked: Sequence[str] = ()) -> dict:
    """Distinguish scoped support checks from the number of retrieved rows."""
    return {'operation': 'layer_neighbor_maps', 'requested_layer_ids': list(selected),
            'checked_layer_ids': [layer for layer in selected if layer in checked],
            'unchecked_layer_ids': [layer for layer in selected if layer not in checked]}


def _provenance(observations: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    """Deduplicate acquisition records without changing their original scopes."""
    result: dict[str, list] = {'registry': [], 'graph': []}
    for observation in observations:
        for channel in result:
            for record in observation.get('provenance', {}).get(channel, []):
                if record not in result[channel]:
                    result[channel].append(deepcopy(record))
    return result


def _proof(result: dict | None, pending: dict | None, tool: str | None,
           used: Sequence[Mapping[str, Any]], coverage: dict) -> MapProof:
    unique = []
    for observation in used:
        if observation not in unique:
            unique.append(observation)
    return MapProof(result, pending, tool, tuple(unique), coverage, _provenance(unique))


def _ordinary(runtime: MultiplexRuntime, arguments: Mapping[str, Any], selected: Sequence[str],
              layer_results: list | None, neighbors: list | None, total: int | None,
              full: bool, *, missing: bool = False) -> dict:
    """Serialize derived observations using the approved fixed result shape."""
    return {'query_gene_id': arguments['gene_id'], 'multiplex_id': arguments['multiplex_id'],
            'layer_scope': runtime._scope(arguments['layer_scope'], selected), 'direction': arguments['direction'],
            'layer_results': deepcopy(layer_results), 'neighbors': deepcopy(neighbors), 'unique_neighbor_count': total,
            'missing_gene_ids': [arguments['gene_id']] if missing else [],
            'layers_without_gene': None if missing else [row['layer_id'] for row in layer_results if row['not_in_layer_gene_ids']],
            'coverage': _coverage(selected, () if missing else selected),
            'collection': None if missing else {'mode': arguments['mode'], 'returned_count': len(neighbors),
                'total_count': total, 'complete': full and len(neighbors) == total, 'next_cursor': None}}


def _validate_rows(runtime: MultiplexRuntime, arguments: Mapping[str, Any], selected: Sequence[str], result: dict) -> None:
    """Check row ordering, native endpoints, and support membership from visible data."""
    query = arguments['gene_id']
    expected_layers = []
    for layer_id in selected:
        expected_layers.append((layer_id, runtime.view.layers[layer_id].directed))
    layer_results = result.get('layer_results')
    if not isinstance(layer_results, list) or len(layer_results) != len(selected):
        raise EvidenceError('saved map must record every selected layer')
    absent = []
    for row, (layer_id, directed) in zip(layer_results, expected_layers):
        if (set(row) != {'layer_id', 'directed', 'not_in_layer_gene_ids'} or row['layer_id'] != layer_id
                or type(row['directed']) is not bool or row['directed'] != directed
                or row['not_in_layer_gene_ids'] not in ([], [query])):
            raise EvidenceError('saved map has inconsistent layer membership metadata')
        if row['not_in_layer_gene_ids']:
            absent.append(layer_id)
    if result.get('layers_without_gene') != absent or result.get('coverage') != _coverage(selected, selected):
        raise EvidenceError('saved map layer coverage or absences disagree')
    rows = result.get('neighbors')
    if not isinstance(rows, list):
        raise EvidenceError('resolved map needs neighbor rows')
    ids = []
    for row in rows:
        gene = row.get('gene_id')
        if (set(row) != {'gene_id', 'supporting_layers', 'supporting_layer_count'}
                or not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)):
            raise EvidenceError('invalid saved map neighbor row')
        ids.append(gene)
        supports = row['supporting_layers']
        if (not isinstance(supports, list) or not supports
                or type(row['supporting_layer_count']) is not int or row['supporting_layer_count'] != len(supports)):
            raise EvidenceError('saved supporting-layer count disagrees with its supports')
        support_ids = []
        for support in supports:
            layer_id = support.get('layer_id')
            if (set(support) != {'layer_id', 'directed', 'edges'} or layer_id not in selected
                    or layer_id in absent or type(support['directed']) is not bool
                    or support['directed'] != runtime.view.layers[layer_id].directed):
                raise EvidenceError('invalid or absent saved supporting layer')
            support_ids.append(layer_id)
            directed = support['directed']
            allowed = ([[gene, query], [query, gene]] if arguments['direction'] == 'incident' and directed else
                       [[gene, query]] if arguments['direction'] == 'in' else
                       [[query, gene]] if arguments['direction'] == 'out' else [sorted([query, gene])])
            allowed = [pair for i, pair in enumerate(allowed) if pair not in allowed[:i]]
            edges = support['edges']
            if not isinstance(edges, list) or not edges:
                raise EvidenceError('saved supporting layer has no native edge')
            pairs = []
            for edge in edges:
                pair, weight = edge.get('gene_ids'), edge.get('weight')
                if (set(edge) != {'gene_ids', 'weight'} or pair not in allowed
                        or type(weight) not in (int, float) or not math.isfinite(weight)):
                    raise EvidenceError('invalid native edge in saved map')
                pairs.append(pair)
            if len({tuple(pair) for pair in pairs}) != len(pairs) or pairs != [pair for pair in allowed if pair in pairs]:
                raise EvidenceError('saved native edges violate direction order or duplicate a self-loop')
        if support_ids != [layer for layer in selected if layer in support_ids] or len(set(support_ids)) != len(support_ids):
            raise EvidenceError('saved supporting layers violate manifest order')
    if ids != sorted(set(ids)):
        raise EvidenceError('saved map rows violate canonical-ID ordering')


def _traversal(runtime: MultiplexRuntime, observations: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> tuple[list, list, int | None, bool, str | None]:
    """Validate one initial-to-prefix chain without interpreting opaque cursors."""
    by_cursor = {}
    for observation in observations:
        cursor = observation['arguments']['cursor']
        if cursor in by_cursor and by_cursor[cursor]['result'] != observation['result']:
            raise EvidenceError('one cursor names contradictory map pages')
        by_cursor[cursor] = observation
    if None not in by_cursor:
        raise EvidenceError('saved map traversal lacks its initial null-cursor page')
    cursor, seen, rows, by_gene, total, full, layers = None, set(), [], {}, None, False, None
    while cursor in by_cursor:
        if cursor in seen:
            raise EvidenceError('saved map cursor chain loops')
        seen.add(cursor)
        observation = by_cursor[cursor]
        arguments, result = observation['arguments'], observation['result']
        _validate_rows(runtime, arguments, selected, result)
        current_layers = result['layer_results']
        if layers is not None and current_layers != layers:
            raise EvidenceError('saved map pages disagree on layer membership')
        layers = current_layers
        collection, page_rows = result.get('collection'), result['neighbors']
        if (not isinstance(collection, dict) or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
                or collection['mode'] != arguments['mode'] or type(collection['returned_count']) is not int
                or collection['returned_count'] != len(page_rows) or type(collection['complete']) is not bool):
            raise EvidenceError('saved map page collection metadata is inconsistent')
        current_total = collection['total_count']
        if current_total is not None and (type(current_total) is not int or current_total < 0):
            raise EvidenceError('saved map total must be a nonnegative integer or null')
        if result['unique_neighbor_count'] != current_total:
            raise EvidenceError('saved map total locations disagree')
        if current_total is not None:
            if total is not None and total != current_total:
                raise EvidenceError('inconsistent non-null map totals')
            total = current_total
            if collection['complete'] != (len(page_rows) == total):
                raise EvidenceError('saved map page completeness disagrees with its full total')
        for row in page_rows:
            gene = row['gene_id']
            if gene in by_gene:
                if by_gene[gene] != row:
                    raise EvidenceError('one neighbor has conflicting support or weights')
                continue
            if rows and gene < rows[-1]['gene_id']:
                raise EvidenceError('saved map pages violate global canonical-ID ordering')
            by_gene[gene] = row
            rows.append(deepcopy(row))
        if total is not None and len(rows) > total:
            raise EvidenceError('saved map rows exceed the known total')
        cursor = collection['next_cursor']
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise EvidenceError('saved map cursor must be nonempty or null')
        if collection['complete']:
            if cursor is not None or len(rows) != len(page_rows):
                raise EvidenceError('whole-set map page conflicts with the cursor chain')
            total, full = len(rows), True
        if cursor is None:
            break
    if len(seen) != len(by_cursor):
        raise EvidenceError('saved map pages are disconnected from the initial cursor')
    arguments = observations[0]['arguments']
    if arguments['mode'] == 'top_k' and len(rows) > arguments['top_k']:
        raise EvidenceError('bounded map traversal exceeds its requested K')
    if cursor is None and (arguments['mode'] == 'complete' or len(rows) < arguments['top_k']):
        if total is not None and total != len(rows):
            raise EvidenceError('exhausted map traversal contradicts its full total')
        total, full = len(rows), True
    if cursor is None and total is not None and len(rows) == total:
        full = True
    return rows, layers, total, full, cursor


def _native_inputs(view: GraphView, selected: Sequence[str], direction: str) -> list[dict]:
    """Expand an incident map into every required native collection."""
    return [{'layer_id': layer, 'direction': native} for layer in selected for native in
            (('in', 'out') if direction == 'incident' and view.layers[layer].directed else
             ('undirected',) if direction == 'incident' else (direction,))]


def _native_runtime(view: GraphView) -> NeighborRuntime:
    """Use the S2.3 proof helper without constructing or consulting a source index."""
    runtime = NeighborRuntime.__new__(NeighborRuntime)
    runtime.view = view
    return runtime


def _native_proof(runtime: NeighborRuntime, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> CollectionProof:
    """Require native traversal exhaustion even when its disclosed count is reached."""
    proof = runtime.collect(arguments, observations)
    if proof.result is None or proof.result['missing_gene_ids'] or proof.result['not_in_layer_gene_ids']:
        return proof
    groups: dict[tuple, list] = defaultdict(list)
    for observation in proof.observations:
        source = observation['arguments']
        groups[(source['mode'], source['top_k'])].append(observation)
    traversals = {selection: _native_traversal(pages) for selection, pages in groups.items()}
    if any(traversal[2] or traversal[3] is None and traversal[1] is not None
           and len(traversal[0]) == traversal[1] for traversal in traversals.values()):
        return proof
    pending = deepcopy(dict(arguments))
    if ('complete', None) in traversals:
        pending['cursor'] = traversals[('complete', None)][3]
    return CollectionProof(None, pending, proof.observations, proof.provenance)


def _assemble_native(view: GraphView, arguments: Mapping[str, Any], selected: Sequence[str], native: Mapping[tuple[str, str], dict]) -> tuple[list, list]:
    """Regroup complete visible collections into canonical rows with all supports."""
    query = arguments['gene_id']
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    layer_results = []
    for layer in selected:
        entries = [(direction, result) for (layer_id, direction), result in native.items() if layer_id == layer]
        absent_values = {bool(result['not_in_layer_gene_ids']) for _, result in entries}
        if len(absent_values) != 1:
            raise EvidenceError('native directions disagree on layer membership')
        absent = bool(next(iter(absent_values)))
        layer_results.append({'layer_id': layer, 'directed': view.layers[layer].directed,
                              'not_in_layer_gene_ids': [query] if absent else []})
        for direction, result in entries:
            for neighbor in result['neighbors']:
                gene = neighbor['gene_id']
                pair = [gene, query] if direction == 'in' else [query, gene] if direction == 'out' else sorted([query, gene])
                edge = {'gene_ids': pair, 'weight': neighbor['weight']}
                existing = grouped[gene][layer]
                same = next((row for row in existing if row['gene_ids'] == pair), None)
                if same is not None and same != edge:
                    raise EvidenceError('native collections give one edge conflicting weights')
                if edge not in existing:
                    existing.append(edge)
    rows = []
    for gene in sorted(grouped):
        supports = [{'layer_id': layer, 'directed': view.layers[layer].directed, 'edges': grouped[gene][layer]}
                    for layer in selected if layer in grouped[gene]]
        rows.append({'gene_id': gene, 'supporting_layers': supports, 'supporting_layer_count': len(supports)})
    return layer_results, rows


def _project_rows(rows: Sequence[dict], selected: Sequence[str]) -> list[dict]:
    """Project complete map supports without changing native edge values."""
    projected = []
    for row in rows:
        supports = [support for support in row['supporting_layers'] if support['layer_id'] in selected]
        if supports:
            projected.append({'gene_id': row['gene_id'], 'supporting_layers': deepcopy(supports),
                              'supporting_layer_count': len(supports)})
    return projected


def _map_native(view: GraphView, arguments: Mapping[str, Any], selected: Sequence[str],
                rows: Sequence[dict], layer_results: Sequence[dict]) -> dict[tuple[str, str], dict]:
    """Derive native collections from a complete map without inventing acquisitions."""
    query = arguments['gene_id']
    results = {}
    for entry in _native_inputs(view, selected, arguments['direction']):
        layer, direction = entry['layer_id'], entry['direction']
        neighbors = {}
        for row in rows:
            support = next((support for support in row['supporting_layers'] if support['layer_id'] == layer), None)
            if support is None:
                continue
            for edge in support['edges']:
                source, target = edge['gene_ids']
                if direction == 'in' and target != query or direction == 'out' and source != query:
                    continue
                gene, weight = row['gene_id'], edge['weight']
                if gene in neighbors and neighbors[gene] != weight:
                    raise EvidenceError('one native edge has conflicting map weights')
                neighbors[gene] = weight
        ranked = [{'gene_id': gene, 'weight': weight} for gene, weight in sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))]
        layer_row = next(row for row in layer_results if row['layer_id'] == layer)
        results[(layer, direction)] = {'gene_id': query, 'multiplex_id': arguments['multiplex_id'],
            'layer_id': layer, 'direction': direction, 'neighbors': ranked, 'missing_gene_ids': [],
            'not_in_layer_gene_ids': deepcopy(layer_row['not_in_layer_gene_ids']),
            'collection': {'mode': 'complete', 'returned_count': len(ranked), 'total_count': len(ranked),
                           'complete': True, 'next_cursor': None}}
    return results


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> MapProof:
    """Prove a map or acquire only the missing layers or native directions.

    Complete maps and native lists supply their layers without repeated calls.
    Entirely missing layers use one complete map. A partially retrieved native
    layer instead requires its missing collections and direction-level coverage.
    """
    runtime = MultiplexRuntime(view)
    selected = _scope(runtime, arguments)
    if arguments['cursor'] is not None:
        raise EvidenceError('a map collection request starts at the initial null cursor')

    def finish(result, pending, tool, used, coverage):
        ordered = [observation for observation in observations if observation in used]
        return _proof(result, pending, tool, ordered, coverage)

    groups: dict[tuple, list] = defaultdict(list)
    native_observations, missing_maps, identities, membership_records = [], [], {}, []
    for observation in observations:
        operation = observation.get('operation')
        if operation not in ('get_neighbor_layers', 'get_neighbors'):
            continue
        source_args = observation.get('arguments', {})
        if any(source_args.get(key) != arguments[key] for key in ('multiplex_id', 'gene_id')):
            continue
        records = observation.get('provenance', {}).get('graph', [])
        if not records or any(record.get('graph_version') != view.graph_version or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        for record in records:
            identity = record.get('evidence_id')
            if not isinstance(identity, str) or not identity:
                raise EvidenceError('saved graph acquisition requires an evidence ID')
            if identity in identities and identities[identity] != observation:
                raise EvidenceError('one evidence ID names contradictory acquisitions')
            identities[identity] = observation
        if operation == 'get_neighbors':
            if source_args.get('layer_id') in selected:
                native_observations.append(observation)
            continue
        source_selected = _scope(runtime, source_args)
        result = observation['result']
        if any(result.get(key) != source_args[key] for key in ('multiplex_id', 'direction')) or result.get('query_gene_id') != arguments['gene_id']:
            raise EvidenceError('saved map subjects differ from its acquisition arguments')
        if result.get('layer_scope') != runtime._scope(source_args['layer_scope'], source_selected):
            raise EvidenceError('saved map scope differs from its acquisition arguments')
        missing = result.get('missing_gene_ids')
        if missing not in ([], [arguments['gene_id']]):
            raise EvidenceError('invalid saved map multiplex membership')
        membership_records.append(bool(missing))
        if missing:
            if (observation['status'] != 'not_found' or any(result.get(key) is not None for key in
                    ('layer_results', 'neighbors', 'unique_neighbor_count', 'layers_without_gene', 'collection'))
                    or result.get('coverage') != _coverage(source_selected)):
                raise EvidenceError('missing multiplex membership must leave the map unavailable')
            missing_maps.append(observation)
            continue
        if observation['status'] != 'resolved':
            raise EvidenceError('a saved map page must resolve its membership prerequisites')
        if source_args['direction'] == arguments['direction']:
            scope_key = json.dumps(source_args['layer_scope'], sort_keys=True, separators=(',', ':'))
            groups[(source_selected, source_args['mode'], source_args['top_k'], scope_key)].append(observation)
    if len(set(membership_records)) > 1:
        raise EvidenceError('saved maps contradict global membership')
    if membership_records and membership_records[0]:
        result = _ordinary(runtime, arguments, selected, None, None, None, False, missing=True)
        return finish(result, None, None, missing_maps, result['coverage'])

    summaries, target_summaries, candidates, used = [], [], [], []
    matching = None
    for (source_selected, mode, top_k, _), pages in groups.items():
        rows, layers, total, full, cursor = _traversal(runtime, pages, source_selected)
        same_layers = tuple(selected) == source_selected
        subset = set(source_selected) <= set(selected)
        intersection = tuple(layer for layer in selected if layer in source_selected)
        if not full and not same_layers and (mode != 'complete' or not subset):
            continue
        if not intersection and selected:
            continue
        summary = {'selected': source_selected, 'rows': rows, 'layers': layers, 'total': total,
                   'full': full, 'cursor': cursor, 'pages': pages, 'mode': mode, 'top_k': top_k}
        summaries.append(summary)
        used.extend(pages)
        if set(selected) <= set(source_selected) and (full or same_layers):
            target_rows = rows if same_layers else _project_rows(rows, selected)
            target_layers = [row for row in layers if row['layer_id'] in selected]
            target_total = total if same_layers else len(target_rows)
            target_summaries.append((target_rows, target_layers, target_total, full))
            if full or arguments['mode'] == 'top_k' and len(target_rows) >= arguments['top_k']:
                candidates.append((target_rows, target_layers, target_total, full))
        if (same_layers and (mode, top_k) == (arguments['mode'], arguments['top_k'])
                and pages[0]['arguments']['layer_scope'] == arguments['layer_scope']):
            matching = cursor
    for i, (rows, layers, total, full) in enumerate(target_summaries):
        for other_rows, other_layers, other_total, other_full in target_summaries[:i]:
            common = min(len(rows), len(other_rows))
            if rows[:common] != other_rows[:common] or layers != other_layers:
                raise EvidenceError('saved maps disagree on canonical rows, support, or membership')
            if total is not None and other_total is not None and total != other_total:
                raise EvidenceError('inconsistent non-null map totals')
            if full and len(other_rows) > len(rows) or other_full and len(rows) > len(other_rows):
                raise EvidenceError('saved map prefix exceeds a complete map')

    inputs = _native_inputs(view, selected, arguments['direction'])
    completed = {}
    for summary in summaries:
        if not summary['full']:
            continue
        layers = tuple(layer for layer in selected if layer in summary['selected'])
        derived = _map_native(view, arguments, layers, summary['rows'], summary['layers'])
        for key, result in derived.items():
            if key in completed and result != completed[key]:
                raise EvidenceError('complete maps disagree on native support or layer membership')
            completed[key] = result
    native_runtime = _native_runtime(view)
    proofs, missing = {}, False
    for entry in inputs:
        target = {'multiplex_id': arguments['multiplex_id'], 'layer_id': entry['layer_id'], 'gene_id': arguments['gene_id'],
                  'direction': entry['direction'], 'mode': 'complete', 'top_k': None, 'cursor': None}
        proof = _native_proof(native_runtime, target, native_observations)
        key = (entry['layer_id'], entry['direction'])
        proofs[key] = proof
        used.extend(proof.observations)
        if key in completed:
            expected = completed[key]
            expected_weights = {row['gene_id']: row['weight'] for row in expected['neighbors']}
            for observation in proof.observations:
                source = observation['result']
                collection = source['collection']
                if (source['missing_gene_ids'] or source['not_in_layer_gene_ids'] != expected['not_in_layer_gene_ids']
                        or collection['total_count'] is not None and collection['total_count'] != len(expected_weights)
                        or any(row['gene_id'] not in expected_weights or expected_weights[row['gene_id']] != row['weight']
                               for row in source['neighbors'])):
                    raise EvidenceError('native collections contradict complete map support or membership')
        if proof.result is not None:
            missing = missing or bool(proof.result['missing_gene_ids'])
            if key in completed and proof.result != completed[key]:
                raise EvidenceError('native collections contradict complete map support or membership')
            completed[key] = proof.result
    if missing:
        if any(not result['missing_gene_ids'] for result in completed.values()) or summaries:
            raise EvidenceError('native collections contradict global membership')
        result = _ordinary(runtime, arguments, selected, None, None, None, False, missing=True)
        return finish(result, None, None, used, result['coverage'])
    for entry in inputs:
        key = (entry['layer_id'], entry['direction'])
        absent = next((value for (layer, _), value in completed.items()
                       if layer == entry['layer_id'] and value['not_in_layer_gene_ids']), None)
        if absent is not None:
            if key in completed and not completed[key]['not_in_layer_gene_ids']:
                raise EvidenceError('native directions disagree on layer membership')
            completed[key] = {**deepcopy(absent), 'direction': entry['direction']}
    if candidates:
        rows, layers, total, full = max(candidates, key=lambda item: (item[3], len(item[0])))
        known = {item[2] for item in target_summaries if item[2] is not None}
        if known:
            total = next(iter(known))
        selected_rows = rows if arguments['mode'] == 'complete' else rows[:arguments['top_k']]
        result = _ordinary(runtime, arguments, selected, layers, selected_rows, total, full)
        return finish(result, None, None, used, result['coverage'])

    if inputs and len(completed) == len(inputs):
        layers, rows = _assemble_native(view, arguments, selected, completed)
        for summary in summaries:
            overlap = tuple(layer for layer in selected if layer in summary['selected'])
            projected = _project_rows(rows, overlap)
            prior = _project_rows(summary['rows'], overlap)
            if set(summary['selected']) <= set(selected):
                if (projected[:len(prior)] != prior
                        or summary['total'] is not None and summary['total'] != len(projected)):
                    raise EvidenceError('native collections contradict saved map support or totals')
        total = len(rows)
        selected_rows = rows if arguments['mode'] == 'complete' else rows[:arguments['top_k']]
        result = _ordinary(runtime, arguments, selected, layers, selected_rows, total, True)
        return finish(result, None, None, used, result['coverage'])

    complete_layers = [layer for layer in selected if all((entry['layer_id'], entry['direction']) in completed
                        for entry in inputs if entry['layer_id'] == layer)]
    partial_native = any(layer not in complete_layers and any(
        (entry['layer_id'], entry['direction']) in completed or proofs[(entry['layer_id'], entry['direction'])].observations
        for entry in inputs if entry['layer_id'] == layer) for layer in selected)
    if partial_native:
        coverage = {'operation': 'neighbor_collections', 'requested_inputs': inputs,
                    'checked_inputs': [entry for entry in inputs if (entry['layer_id'], entry['direction']) in completed],
                    'unchecked_inputs': [entry for entry in inputs if (entry['layer_id'], entry['direction']) not in completed]}
        entry = coverage['unchecked_inputs'][0]
        pending = proofs[(entry['layer_id'], entry['direction'])].next_arguments
        return finish(None, pending, 'get_neighbors', used, coverage)

    missing_layers = tuple(layer for layer in selected if layer not in complete_layers)
    checked = set(complete_layers)
    for summary in summaries:
        checked.update(layer for layer in summary['selected'] if layer in selected)
    coverage = _coverage(selected, checked)
    composing = bool(complete_layers) or any(set(summary['selected']) < set(selected) for summary in summaries)
    if composing:
        pending = {**deepcopy(dict(arguments)),
                   'layer_scope': {'kind': 'layers', 'layer_ids': list(missing_layers), 'layer_family': None},
                   'mode': 'complete', 'top_k': None, 'cursor': None}
        for summary in summaries:
            if (summary['mode'] == 'complete' and not summary['full'] and summary['cursor'] is not None
                    and set(summary['selected']) <= set(missing_layers)):
                pending = {**deepcopy(summary['pages'][0]['arguments']), 'cursor': summary['cursor']}
                break
    else:
        pending = {**deepcopy(dict(arguments)), 'cursor': matching}
    return finish(None, pending, 'get_neighbor_layers', used, coverage)
