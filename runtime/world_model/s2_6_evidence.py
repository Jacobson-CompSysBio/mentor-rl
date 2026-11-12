"""Classify supplied genes from visible evidence without source inventory reads.

The planner validates compatible acquisitions before it reuses their facts.
Only an explicit harness limit can divide the remaining input into batches.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .s2_5_evidence import _global_membership, _map_membership, _neighbor_membership, _traversal
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError, SelectionError
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class NodesProof:
    """Return a complete partition or the exact next required batch."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    coverage: dict
    provenance: dict[str, list]


def _coverage(genes: Sequence[str], checked: Sequence[str] = ()) -> dict:
    """Keep the outer input order even when evidence uses another order."""
    checked = set(checked)
    return {'operation': 'layer_membership', 'requested_gene_ids': list(genes),
            'checked_gene_ids': [gene for gene in genes if gene in checked],
            'unchecked_gene_ids': [gene for gene in genes if gene not in checked]}


def _ordinary(arguments: Mapping[str, Any], genes: Sequence[str], categories: Mapping[str, str]) -> dict:
    """Count positive matches within the requested domain only."""
    positive = sorted(gene for gene in genes if categories[gene] == 'layer_gene_ids')
    return {'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
            'layer_gene_ids': positive, 'layer_gene_count': len(positive),
            'not_in_layer_gene_ids': sorted(gene for gene in genes if categories[gene] == 'not_in_layer_gene_ids'),
            'missing_gene_ids': sorted(gene for gene in genes if categories[gene] == 'missing_gene_ids'),
            'coverage': _coverage(genes, genes),
            'collection': {'mode': 'complete', 'returned_count': len(positive), 'total_count': len(positive),
                           'complete': True, 'next_cursor': None}}


def _nodes_partition(runtime: NodesByLayerRuntime, observation: Mapping[str, Any]) -> tuple[dict, dict]:
    """Check an atomic complete batch without consulting its source inventories."""
    arguments, result = observation['arguments'], observation.get('result')
    try:
        genes = runtime._validate(arguments)
    except SelectionError as error:
        raise EvidenceError('saved node classification has invalid arguments') from error
    fields = {'multiplex_id', 'layer_id', 'layer_gene_ids', 'layer_gene_count', 'not_in_layer_gene_ids',
              'missing_gene_ids', 'coverage', 'collection'}
    if (observation.get('status') != 'resolved' or not isinstance(result, dict) or set(result) != fields
            or any(result[key] != arguments[key] for key in ('multiplex_id', 'layer_id'))
            or result['coverage'] != _coverage(genes, genes)):
        raise EvidenceError('saved node classification identity or coverage is inconsistent')
    categories = {}
    for category in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids'):
        rows = result[category]
        if (not isinstance(rows, list) or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in rows)
                or rows != sorted(set(rows)) or any(gene in categories for gene in rows)):
            raise EvidenceError('saved node categories must be sorted, unique, and disjoint')
        categories.update(dict.fromkeys(rows, category))
    if set(categories) != set(genes):
        raise EvidenceError('saved node categories do not partition their requested genes')
    count, collection = result['layer_gene_count'], result['collection']
    if (type(count) is not int or count != len(result['layer_gene_ids']) or not isinstance(collection, dict)
            or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
            or collection['mode'] != 'complete' or collection['complete'] is not True
            or collection['next_cursor'] is not None
            or any(type(collection[key]) is not int or collection[key] != count for key in ('returned_count', 'total_count'))):
        raise EvidenceError('saved node collection counts or completeness are inconsistent')
    scope = {'kind': 'layers', 'layer_ids': [arguments['layer_id']], 'layer_family': None}
    return categories, scope


def _membership_page(runtime: GeneLayersRuntime, observation: Mapping[str, Any], selected: Sequence[str]) -> dict:
    """Read explicit positives even when an earlier transport page is absent."""
    source, result = observation['arguments'], observation.get('result')
    fields = {'gene_id', 'multiplex_id', 'layer_scope', 'present', 'layer_ids', 'layer_count', 'coverage', 'collection'}
    coverage = {'operation': 'layer_membership', 'requested_layer_ids': list(selected),
                'checked_layer_ids': list(selected), 'unchecked_layer_ids': []}
    if (observation.get('status') != 'resolved' or not isinstance(result, dict) or set(result) != fields
            or any(result[key] != source[key] for key in ('gene_id', 'multiplex_id'))
            or result['layer_scope'] != runtime.multiplex._scope(source['layer_scope'], selected)
            or type(result['present']) is not bool or result['coverage'] != coverage):
        raise EvidenceError('saved membership page identity or coverage is inconsistent')
    rows, collection = result['layer_ids'], result['collection']
    if (not isinstance(rows, list) or any(layer not in selected for layer in rows)
            or rows != [layer for layer in selected if layer in rows]):
        raise EvidenceError('saved membership rows violate scope or manifest order')
    if (not isinstance(collection, dict)
            or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
            or collection['mode'] != source['mode'] or type(collection['returned_count']) is not int
            or collection['returned_count'] != len(rows) or type(collection['complete']) is not bool
            or source['mode'] == 'top_k' and len(rows) > source['top_k']):
        raise EvidenceError('saved membership collection metadata is inconsistent')
    total, cursor = collection['total_count'], collection['next_cursor']
    if (total is not None and (type(total) is not int or not len(rows) <= total <= len(selected))
            or type(result['layer_count']) is not type(total) or result['layer_count'] != total
            or cursor is not None and (not isinstance(cursor, str) or not cursor)
            or total is not None and collection['complete'] != (len(rows) == total)
            or collection['complete'] and cursor is not None):
        raise EvidenceError('saved membership total or completeness is inconsistent')
    if not result['present'] and (rows or total != 0 or not collection['complete']):
        raise EvidenceError('global absence requires a complete empty membership list')
    return result


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
            *, batch_limit: int | None = None) -> NodesProof:
    """Prove every requested classification or request the unresolved ordered prefix.

    A bounded containing-layer list supplies its returned positive identities.
    Its omitted identities and checked coverage do not prove layer absence.
    Failed calls add no evidence and leave their attempted genes unchecked.
    Pending proofs retain compatible observations as attributed partial work.
    """
    if batch_limit is not None and (type(batch_limit) is not int or batch_limit < 1):
        raise ValueError('batch_limit must be a positive integer or null')
    runtime, memberships = NodesByLayerRuntime(view), GeneLayersRuntime(view)
    genes = runtime._validate(arguments)
    if not genes:
        return NodesProof(_ordinary(arguments, genes, {}), None, None, (), _coverage(genes), {'registry': [], 'graph': []})
    requested, layer = set(genes), arguments['layer_id']
    identities, global_values, layer_values, categories, providers = {}, {}, {}, {}, {}
    compatible_indices = []
    groups = defaultdict(list)

    def global_fact(gene: str, present: bool) -> None:
        if gene in global_values and global_values[gene] != present:
            raise EvidenceError('saved records contradict global membership')
        global_values[gene] = present

    def layer_fact(gene: str, selected: str, present: bool) -> None:
        key = (gene, selected)
        if key in layer_values and layer_values[key] != present:
            raise EvidenceError('saved records contradict layer membership')
        layer_values[key] = present

    def classify(gene: str, category: str, indices: Sequence[int]) -> None:
        if gene in categories and categories[gene] != category:
            raise EvidenceError('saved records contradict node classification')
        categories[gene] = category
        if gene not in providers or min(indices) < min(providers[gene]):
            providers[gene] = tuple(indices)

    def membership(gene: str, present: bool, facts: Mapping[str, bool], indices: Sequence[int]) -> None:
        global_fact(gene, present)
        for selected, value in facts.items():
            layer_fact(gene, selected, value)
        if not present:
            classify(gene, 'missing_gene_ids', indices)
        elif layer in facts:
            classify(gene, 'layer_gene_ids' if facts[layer] else 'not_in_layer_gene_ids', indices)

    for index, observation in enumerate(observations):
        operation, source = observation.get('operation'), observation.get('arguments', {})
        if operation not in ('get_nodes_by_layer', 'check_multiplex_membership', 'get_gene_layers', 'get_neighbors', 'get_neighbor_layers'):
            continue
        if not isinstance(source, Mapping) or source.get('multiplex_id') != arguments['multiplex_id']:
            continue
        if operation in ('get_nodes_by_layer', 'check_multiplex_membership'):
            supplied = source.get('gene_ids')
            if not isinstance(supplied, list) or not any(isinstance(gene, str) and gene in requested for gene in supplied):
                continue
        elif source.get('gene_id') not in requested:
            continue
        provenance = observation.get('provenance', {})
        if not isinstance(provenance, Mapping):
            raise EvidenceError('saved graph provenance must be an object')
        records = provenance.get('graph', [])
        if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
            raise EvidenceError('saved graph provenance must contain acquisition records')
        if records and any(record.get('graph_version') != view.graph_version or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if not records and observation.get('status') != 'resolved':
            continue
        if (len(records) != 1 or provenance.get('registry') != []
                or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}):
            raise EvidenceError('a graph observation requires one attributed acquisition')
        record, identity = records[0], records[0].get('evidence_id')
        if not isinstance(identity, str) or not identity.strip():
            raise EvidenceError('saved graph acquisition requires an evidence ID')
        if identity in identities and identities[identity] != observation:
            raise EvidenceError('one evidence ID names contradictory acquisitions')
        identities[identity] = observation
        if operation == 'get_nodes_by_layer':
            partition, expected_scope = _nodes_partition(runtime, observation)
            for gene in requested.intersection(partition):
                category = partition[gene]
                present = category != 'missing_gene_ids'
                membership(gene, present, {} if not present else {source['layer_id']: category == 'layer_gene_ids'}, [index])
        elif operation == 'check_multiplex_membership':
            first = next(gene for gene in supplied if isinstance(gene, str) and gene in requested)
            _, _, expected_scope = _global_membership({'multiplex_id': arguments['multiplex_id'], 'gene_id': first}, observation)
            positive = set(observation['result']['present_gene_ids'])
            for gene in requested.intersection(supplied):
                membership(gene, gene in positive, {}, [index])
        elif operation == 'get_gene_layers':
            try:
                selected = memberships._validate(source)
            except SelectionError as error:
                raise EvidenceError('saved containing-layer arguments are invalid') from error
            expected_scope = memberships.multiplex._scope(source['layer_scope'], selected)
            page = _membership_page(memberships, observation, selected)
            facts = {selected_layer: selected_layer in page['layer_ids'] for selected_layer in selected} if page['collection']['complete'] else dict.fromkeys(page['layer_ids'], True)
            membership(source['gene_id'], page['present'], facts, [index])
            key = (source['gene_id'], selected, source['mode'], source['top_k'],
                   json.dumps(source['layer_scope'], sort_keys=True, separators=(',', ':')))
            groups[key].append((index, observation))
        else:
            present, facts, expected_scope = (_neighbor_membership(memberships, observation) if operation == 'get_neighbors'
                                              else _map_membership(memberships, observation))
            membership(source['gene_id'], present, facts, [index])
        if record['layer_scope'] != expected_scope:
            raise EvidenceError('saved provenance scope differs from its acquisition')
        compatible_indices.append(index)

    summaries = []
    for (gene, selected, _, _, _), pages in groups.items():
        if any(page['arguments']['cursor'] is None for _, page in pages):
            summary = _traversal(memberships, [page for _, page in pages], selected)
        else:
            totals = {page['result']['layer_count'] for _, page in pages if page['result']['layer_count'] is not None}
            if len(totals) > 1:
                raise EvidenceError('inconsistent non-null membership totals')
            for _, page in pages:
                summary = {'gene_id': gene, 'selected': selected, 'layers': page['result']['layer_ids'],
                           'total': page['result']['layer_count'], 'detached': True}
                summaries.append(summary)
            continue
        summary['gene_id'] = gene
        summaries.append(summary)
        facts = {selected_layer: selected_layer in summary['layers'] for selected_layer in selected} if summary['full'] else dict.fromkeys(summary['layers'], True)
        membership(gene, summary['present'], facts, [index for index, _ in pages])

    for (gene, _), present in layer_values.items():
        if present and global_values.get(gene) is False:
            raise EvidenceError('global absence contradicts positive layer membership')
    for summary in summaries:
        gene, selected = summary['gene_id'], summary['selected']
        positives = [selected_layer for selected_layer in selected if layer_values.get((gene, selected_layer)) is True]
        negatives = [selected_layer for selected_layer in selected if layer_values.get((gene, selected_layer)) is False]
        total = summary['total']
        if total is not None and (total < len(positives) or total > len(selected) - len(negatives)):
            raise EvidenceError('saved membership total contradicts checked memberships')
        if summary['layers'] and not summary.get('detached'):
            last = selected.index(summary['layers'][-1])
            if any(selected_layer not in summary['layers'] for selected_layer in positives if selected.index(selected_layer) <= last):
                raise EvidenceError('saved membership prefix omits an earlier known positive')
    for index, summary in enumerate(summaries):
        for prior in summaries[:index]:
            if (summary['gene_id'], summary['selected']) != (prior['gene_id'], prior['selected']):
                continue
            if not summary.get('detached') and not prior.get('detached'):
                common = min(len(summary['layers']), len(prior['layers']))
                if summary['layers'][:common] != prior['layers'][:common]:
                    raise EvidenceError('saved selections disagree on the manifest-ordered prefix')
            if summary['total'] is not None and prior['total'] is not None and summary['total'] != prior['total']:
                raise EvidenceError('inconsistent non-null membership totals')

    complete = len(categories) == len(genes)
    indices = sorted({index for records in providers.values() for index in records}) if complete else compatible_indices
    used, provenance = [], {'registry': [], 'graph': []}
    for index in indices:
        observation = observations[index]
        if observation not in used:
            used.append(observation)
        for channel in provenance:
            for record in observation['provenance'][channel]:
                if record not in provenance[channel]:
                    provenance[channel].append(deepcopy(record))
    coverage = _coverage(genes, categories)
    if complete:
        return NodesProof(_ordinary(arguments, genes, categories), None, None, tuple(used), coverage, provenance)
    pending = coverage['unchecked_gene_ids']
    if batch_limit is not None:
        pending = pending[:batch_limit]
    next_arguments = runtime.arguments(layer, pending)
    return NodesProof(None, next_arguments, 'get_nodes_by_layer', tuple(used), coverage, provenance)
