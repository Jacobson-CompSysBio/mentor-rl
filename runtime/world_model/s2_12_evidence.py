"""Classify one gene and assess graph claims from visible acquisitions only.

The planner reads registered layer metadata. Source inventories and native
adjacency remain available only to the tools that acquire new observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .s2_6_evidence import _coverage, _ordinary
from .s2_6_evidence import collect as _nodes_collect
from .s2_9_evidence import collect as _subgraph_collect
from .s2_10_evidence import _facts
from .s2_11_evidence import _registry_key
from .s2_graph_claims import CLAIM_VOCABULARY
from .s2_multiplex import EvidenceError, MultiplexRuntime, SelectionError
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView


@dataclass(frozen=True)
class MembershipProof:
    """Return a classification, an exact next call, or an explicit blocked issue."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]
    coverage: dict
    status: str = 'resolved'
    issue_code: str | None = None


def _retain(observations: Sequence[Mapping], selected: Sequence[Mapping]) -> tuple:
    """Keep original acquisitions in first-use order and remove identical repeats."""
    used = []
    for observation in observations:
        if observation in selected and observation not in used:
            used.append(observation)
    return tuple(used)


def _provenance(observations: Sequence[Mapping]) -> dict[str, list]:
    result = {'registry': [], 'graph': []}
    for observation in observations:
        for channel in result:
            for record in observation['provenance'][channel]:
                if record not in result[channel]:
                    result[channel].append(deepcopy(record))
    return result


def _active(view: GraphView, observation: Mapping, gene: str) -> bool:
    """Select current graph records that identify the requested gene explicitly."""
    if observation['operation'] not in ('get_nodes_by_layer', 'get_gene_layers', 'check_multiplex_membership',
                                        'get_edge', 'get_neighbors', 'get_neighbor_layers', 'induce_subgraph'):
        return False
    records = observation['provenance']['graph']
    if not records or any(record.get('graph_version') != view.graph_version
                          or record.get('multiplex_id') != view.multiplex_id for record in records):
        return False
    source, result = observation['arguments'], observation['result']
    if source.get('gene_id') == gene or gene in source.get('gene_ids', ()):
        return True
    return observation['operation'] == 'get_neighbors' and any(
        row['gene_id'] == gene for row in result.get('neighbors', ()))


def _inspect(view: GraphView, arguments: Mapping, observations: Sequence[Mapping]) -> tuple:
    """Reuse the earlier validators and select the first sufficient acquisition."""
    gene, layer = arguments['gene_ids'][0], arguments['layer_id']
    native = {**deepcopy(dict(arguments)), 'mode': 'complete', 'top_k': None, 'cursor': None}
    # This validates identities, provenance, collection pages, and contradictions.
    # It can request an induced graph, but this planner never executes that request.
    _subgraph_collect(view, native, observations)
    registry, symbols, symbol_genes, unique_symbols = {}, {}, {}, {}
    for observation in observations:
        operation = observation['operation']
        if operation not in ('lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id') or observation['status'] != 'resolved':
            continue
        result, provenance = observation['result'], observation['provenance']
        key = _registry_key(operation, observation['arguments'], result)
        records = provenance['registry']
        if (provenance['graph'] or len(records) != 1 or set(records[0]) != {'registry', 'release', 'evidence_id'}
                or records[0]['registry'] != 'ensembl' or not isinstance(records[0]['release'], str)
                or not records[0]['release'].isdigit()):
            raise EvidenceError('a resolved identity requires its original Ensembl registry provenance')
        gene_id, symbol = result['gene_id'], result['gene_symbol']
        if key in registry and registry[key] != result or gene_id in symbols and symbols[gene_id] != symbol:
            raise EvidenceError('saved registry identities disagree about a current mapping')
        registry[key], symbols[gene_id] = result, symbol
        if symbol is not None:
            symbol_genes.setdefault(symbol.casefold(), set()).add(gene_id)
            if operation == 'lookup_human_gene_symbol':
                unique_symbols[symbol.casefold()] = gene_id
    if any(symbol_genes[symbol] != {gene_id} for symbol, gene_id in unique_symbols.items()):
        raise EvidenceError('a resolved symbol conflicts with another current registry identity')
    categories = _facts(view, layer, observations)[0]
    basic = _nodes_collect(view, arguments, observations)
    active = tuple(item for item in observations if _active(view, item, gene))
    if _global_presence(gene, active) is True:
        covered = {item['arguments']['layer_id'] for item in active if 'layer_id' in item['arguments']}
        for item in active:
            if item['operation'] == 'get_gene_layers' and item['result']['collection']['complete']:
                covered.update(MultiplexRuntime(view).resolve_scope(item['arguments']['layer_scope']))
        if set(view.layer_order) <= covered and all(
                _facts(view, selected, observations)[0].get(gene) == 'not_in_layer_gene_ids'
                for selected in view.layer_order):
            raise EvidenceError('positive multiplex membership contradicts absence from its complete layer union')
    candidates = []
    if basic.result is not None:
        candidates.append(tuple(basic.observations))
    for observation in active:
        if observation['operation'] not in ('get_edge', 'induce_subgraph', 'get_neighbors'):
            continue
        if gene in _facts(view, layer, [observation])[0]:
            candidates.append((observation,))
    if candidates:
        used = min(candidates, key=lambda group: tuple(observations.index(item) for item in group))
    else:
        used = active
    return categories.get(gene), _retain(observations, used), active


def _global_presence(gene: str, observations: Sequence[Mapping]) -> bool | None:
    """Read explicit global membership after the acquisition validators pass."""
    values = set()
    for observation in observations:
        source, result = observation['arguments'], observation['result']
        if observation['operation'] == 'check_multiplex_membership' and gene in source['gene_ids']:
            values.add(gene in result['present_gene_ids'])
        elif source.get('gene_id') == gene and 'present' in result:
            values.add(result['present'])
        elif gene in source.get('gene_ids', ()) or source.get('gene_id') == gene:
            values.add(gene not in result.get('missing_gene_ids', ()))
        elif observation['operation'] == 'get_neighbors' and any(row['gene_id'] == gene for row in result['neighbors']):
            values.add(True)
    if len(values) > 1:
        raise EvidenceError('saved observations contradict global membership')
    return next(iter(values)) if values else None


def _card(arguments: Mapping, supported: Sequence[str], claim: str | None, statement_true: bool | None) -> dict:
    """Bind the unchanged vocabulary to the singleton subject without direction."""
    card = {'subject': {'gene_id': arguments['gene_ids'][0],
        **{key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_id')}},
        'claim_vocabulary': 's2_graph_observations_v1',
        'supported_claims': [name for name in CLAIM_VOCABULARY if name in supported],
        'unsupported_claims': [name for name in CLAIM_VOCABULARY if name not in supported]}
    if claim is not None:
        card.update(claim=claim, claim_supported=claim in supported)
    if statement_true is not None:
        card['statement_true'] = statement_true
    return card


def _truth(view: GraphView, arguments: Mapping, observations: Sequence[Mapping], category: str | None,
           claim: str, truth_scope: str, truth_layer_ids: Sequence[str] | None,
           truth_family: str | None) -> bool | None:
    """Evaluate the exact graph proposition, including manifest union entailments."""
    gene = arguments['gene_ids'][0]
    present = _global_presence(gene, observations)
    if truth_scope == 'category':
        if claim == 'gene_not_node_in_layer' and category is not None:
            return category == 'not_in_layer_gene_ids'
        if claim == 'gene_not_node_in_multiplex' and present is not None:
            return not present
        if claim in CLAIM_VOCABULARY[:2] and category == 'not_in_layer_gene_ids':
            return True
        return None
    if truth_scope == 'multiplex':
        return None if present is None else not present
    if truth_scope in ('future_versions', 'expression'):
        return None
    if truth_scope == 'all_layers':
        selected = view.layer_order
    elif truth_scope == 'layers':
        selected = MultiplexRuntime(view).resolve_scope({
            'kind': 'layers', 'layer_ids': list(truth_layer_ids or ()), 'layer_family': None})
    else:
        selected = MultiplexRuntime(view).resolve_scope({
            'kind': 'all', 'layer_ids': [], 'layer_family': truth_family})
    if present is False:
        return True
    facts = {layer: _facts(view, layer, observations)[0].get(gene) for layer in selected}
    if any(value == 'layer_gene_ids' for value in facts.values()):
        return False
    if all(value == 'not_in_layer_gene_ids' for value in facts.values()):
        return True
    # The active representation defines the multiplex as the layer-node union.
    if set(selected) == set(view.layer_order) and present is True:
        return False
    return None


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
            *, classification_requested: bool = True, claim: str | None = None,
            truth_scope: str | None = None, truth_layer_ids: Sequence[str] | None = None,
            truth_family: str | None = None, require_membership: str | None = None) -> MembershipProof:
    """Classify one canonical ID and assess only the explicitly requested claims.

    A support-only task never acquires more membership evidence. A completed
    global negative resolves the classification unless an explicit gate fails.
    Truth requests remain separate from evidence-support requests.
    """
    genes = NodesByLayerRuntime(view)._validate(arguments)
    if len(arguments['gene_ids']) != 1 or len(genes) != 1:
        raise SelectionError('invalid_input', 'INVALID_GENE_LIST')
    if type(classification_requested) is not bool or require_membership not in (None, 'multiplex', 'layer'):
        raise ValueError('classification and outer membership gates must be explicit')
    if claim is not None and claim not in CLAIM_VOCABULARY:
        raise ValueError('a support request requires an approved canonical claim')
    scopes = (None, 'category', 'multiplex', 'all_layers', 'layers', 'family', 'future_versions', 'expression')
    if truth_scope not in scopes or truth_scope is not None and claim is None:
        raise ValueError('a truth request requires an explicit proposition scope and canonical claim')
    truth_claims = {'category': (*CLAIM_VOCABULARY[:2], 'gene_not_node_in_multiplex', 'gene_not_node_in_layer'),
        'multiplex': ('gene_not_node_in_multiplex',),
        'all_layers': ('absence_holds_in_other_layers', 'gene_not_node_in_multiplex'),
        'layers': ('absence_holds_in_other_layers',), 'family': ('absence_holds_in_other_layers',),
        'future_versions': ('absence_holds_in_other_versions',), 'expression': ('gene_not_expressed_in_context',)}
    if truth_scope is not None and claim not in truth_claims[truth_scope]:
        raise ValueError('the canonical support category does not match the exact truth proposition')
    if truth_scope == 'layers' and (not isinstance(truth_layer_ids, (list, tuple)) or not truth_layer_ids):
        raise ValueError('a named-layer truth request requires explicit registered layers')
    if truth_scope == 'family' and not isinstance(truth_family, str):
        raise ValueError('a family truth request requires its explicit family')
    if truth_layer_ids is not None and truth_scope != 'layers' or truth_family is not None and truth_scope != 'family':
        raise ValueError('truth selections must match the exact proposition scope')
    category, used, active = _inspect(view, arguments, observations)
    gene = genes[0]
    coverage = _coverage(genes, genes if category is not None else ())

    def proof(result=None, pending=None, status='resolved', issue=None, retained=used):
        return MembershipProof(result, deepcopy(pending), 'get_nodes_by_layer' if pending else None,
            tuple(retained), _provenance(retained), coverage, status, issue)

    if category is None and (classification_requested or require_membership is not None):
        return proof(pending=arguments)
    if require_membership is not None and (category == 'missing_gene_ids'
            or require_membership == 'layer' and category != 'layer_gene_ids'):
        return proof(status='not_found', issue='MEMBERSHIP_PRECONDITION_UNSATISFIED')
    supported = (['gene_not_node_in_multiplex'] if category == 'missing_gene_ids' else
        [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer'] if category == 'not_in_layer_gene_ids' else [])
    statement_true = None
    if truth_scope is not None:
        statement_true = _truth(view, arguments, active, category, claim, truth_scope, truth_layer_ids, truth_family)
        # Wider propositions can depend on records outside the selected layer.
        if truth_scope != 'category':
            used = _retain(observations, [*used, *active])
        if statement_true is None:
            return proof(status='invalid_input', issue='UNSUPPORTED_TRUTH_SCOPE', retained=used)
    result = _ordinary(arguments, genes, {gene: category}) if classification_requested else {
        'gene_id': gene, **{key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_id')}}
    result['claim_cards'] = [_card(arguments, supported, claim, statement_true)]
    return proof(result=result, retained=used)
