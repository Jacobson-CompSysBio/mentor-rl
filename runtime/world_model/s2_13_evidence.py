"""Assess layer meanings from qualified metadata and visible native records.

The planner reads no source inventories or adjacency. Family capability and
support for a particular observed pair remain separate result branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

from .contract import load_contract
from .s2_9_evidence import _canonical, _edge_key, _scope
from .s2_10_evidence import _edge_observation
from .s2_11_evidence import _inspect, _known_absent, _provenance, _registry_key
from .s2_11_evidence import calibrate as _graph_calibrate
from .s2_multiplex import EvidenceError, MultiplexRuntime, SelectionError
from .sources import GENE_ID_PATTERN


_CONTRACT = load_contract()
CLAIM_VOCABULARY = tuple(_CONTRACT['vocabularies']['layer_claims'])
FAMILIES = tuple(_CONTRACT['vocabularies']['layer_family'])
FAMILY_CARDS = MappingProxyType({family: MappingProxyType({
    'evidence_type': card['evidence_type'], 'supported_claims': tuple(card['supported_claims'])})
    for family, card in _CONTRACT['curriculum']['families']['S1.7']['family_cards'].items()})
del _CONTRACT


@dataclass(frozen=True)
class LayerProof:
    """Return scoped cards, an exact required edge call, or an explicit issue."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping, ...]
    provenance: dict[str, list]
    status: str = 'resolved'
    issue_code: str | None = None
    coverage: None = None


def _card(subject: Mapping, supported: Sequence[str], claim: str | None) -> dict:
    if claim is not None and claim not in CLAIM_VOCABULARY:
        raise ValueError('the biological claim is outside the unchanged S1 vocabulary')
    result = {'subject': deepcopy(dict(subject)), 'claim_vocabulary': 's1_layer_v1',
        'supported_claims': [item for item in CLAIM_VOCABULARY if item in supported],
        'unsupported_claims': [item for item in CLAIM_VOCABULARY if item not in supported]}
    if claim is not None:
        result.update(claim=claim, claim_supported=claim in supported)
    return result


def family_result(layer_family: str, *, claim: str | None = None) -> dict:
    """Return S1's conceptual family capability without a graph observation."""
    if layer_family not in FAMILIES:
        raise ValueError('a family lookup requires one of the six S1 families')
    meaning = FAMILY_CARDS[layer_family]
    return {'layer_family': layer_family, 'evidence_type': meaning['evidence_type'],
        'claim_cards': [_card({'kind': 'family', 'layer_family': layer_family},
                              meaning['supported_claims'], claim)]}


def _arguments(view, arguments):
    if not isinstance(arguments, Mapping) or set(arguments) != {'multiplex_id', 'layer_id', 'gene_ids'}:
        raise SelectionError('invalid_input', 'INVALID_SELECTION')
    if not arguments['multiplex_id'] or not arguments['layer_id']:
        raise SelectionError('invalid_input', 'MISSING_SCOPE')
    if arguments['multiplex_id'] != view.multiplex_id:
        raise SelectionError('not_found', 'UNKNOWN_MULTIPLEX')
    layer = arguments['layer_id']
    if not isinstance(layer, str) or layer not in view.layers:
        raise SelectionError('not_found', 'UNKNOWN_LAYER')
    genes = arguments['gene_ids']
    if (not isinstance(genes, list) or len(genes) != 2 or genes[0] == genes[1]
            or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in genes)):
        raise SelectionError('invalid_input', 'INVALID_PAIR')
    return deepcopy(dict(arguments))


def pair_subject(view, arguments: Mapping) -> dict:
    """Bind one pair to its registered native representation and current version."""
    source = _arguments(view, arguments)
    layer = source['layer_id']
    directed = view.layers[layer].directed
    return {'kind': 'pair', 'gene_ids': list(_edge_key(source['gene_ids'], directed)),
        'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
        'layer_id': layer, 'layer_family': MultiplexRuntime(view).family_of(layer),
        'directed': directed}


def _path_arguments(view, arguments):
    return {**deepcopy(dict(arguments)), 'direction': 'source_to_target'
            if view.layers[arguments['layer_id']].directed else 'undirected'}


def _records(view, observations):
    """Validate original acquisitions, including contrasts that prove no target fact."""
    result, identities = [], {}
    for observation in observations:
        if (not isinstance(observation, Mapping)
                or set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}
                or not isinstance(observation['arguments'], Mapping)
                or not isinstance(observation['result'], Mapping)):
            raise EvidenceError('an observation requires its exact original partial-result fields')
        provenance = observation['provenance']
        if not isinstance(provenance, Mapping) or set(provenance) != {'registry', 'graph'}:
            raise EvidenceError('an observation requires both original provenance channels')
        for records in provenance.values():
            if not isinstance(records, list):
                raise EvidenceError('each provenance channel must contain acquisition records')
            for record in records:
                identity = record.get('evidence_id') if isinstance(record, Mapping) else None
                if not isinstance(identity, str) or not identity.strip():
                    raise EvidenceError('each acquisition requires an explicit evidence ID')
                if identity in identities and _canonical(identities[identity]) != _canonical(observation):
                    raise EvidenceError('one evidence ID names different original acquisitions')
                identities[identity] = observation
        operation, source = observation['operation'], observation['arguments']
        if operation not in ('lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id',
                'get_edge', 'get_nodes_by_layer', 'get_gene_layers', 'check_multiplex_membership',
                'get_neighbors', 'get_neighbor_layers', 'induce_subgraph', 'shortest_path'):
            raise EvidenceError('the saved acquisition uses an unsupported operation')
        if observation['status'] not in ('resolved', 'not_found', 'ambiguous', 'invalid_input'):
            raise EvidenceError('a failed attempt supplies no acquisition evidence')
        if operation in ('lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id'):
            records = provenance['registry']
            if (provenance['graph'] or len(records) != 1
                    or set(records[0]) != {'registry', 'release', 'evidence_id'}
                    or records[0]['registry'] != 'ensembl' or not isinstance(records[0]['release'], str)
                    or not records[0]['release'].isdigit()):
                raise EvidenceError('a registry observation requires its original Ensembl provenance')
            if observation['status'] == 'resolved':
                _registry_key(operation, source, observation['result'])
        else:
            records = provenance['graph']
            if (provenance['registry'] or len(records) != 1
                    or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}
                    or records[0]['multiplex_id'] != source.get('multiplex_id')
                    or not isinstance(records[0]['graph_version'], str) or not records[0]['graph_version']):
                raise EvidenceError('a graph observation requires one consistent original acquisition')
            if operation == 'get_edge':
                current = (records[0]['multiplex_id'] == view.multiplex_id
                    and records[0]['graph_version'] == view.graph_version)
                if current:
                    metadata = view
                else:
                    directed = observation['result'].get('directed')
                    if (type(directed) is not bool or not isinstance(source.get('layer_id'), str)
                            or not source['layer_id']):
                        raise EvidenceError('a historical edge must retain its native representation')
                    metadata = SimpleNamespace(multiplex_id=source.get('multiplex_id'),
                        layers={source.get('layer_id'): SimpleNamespace(directed=directed)})
                _edge_observation(metadata, observation)
        if observation not in result:
            result.append(observation)
    return tuple(result)


def _matching(view, source, observation):
    if observation['operation'] != 'get_edge':
        return False
    records = observation['provenance']['graph']
    return (records[0]['multiplex_id'] == view.multiplex_id
        and records[0]['graph_version'] == view.graph_version
        and observation['arguments']['layer_id'] == source['layer_id']
        and observation['result']['gene_ids'] == pair_subject(view, source)['gene_ids'])


def _ordinary(view, source, records, facts):
    matching = next((item for item in records if _matching(view, source, item)), None)
    if matching is not None:
        return deepcopy(matching['result'])
    genes, layer = source['gene_ids'], source['layer_id']
    if not all(gene in facts.categories for gene in genes):
        return None
    pair = _edge_key(genes, view.layers[layer].directed)
    missing = sorted(gene for gene in genes if facts.categories[gene] == 'missing_gene_ids')
    outside = sorted(gene for gene in genes if facts.categories[gene] == 'not_in_layer_gene_ids')
    weight = facts.positives.get((layer, pair))
    exists = (layer, pair) in facts.positives
    if not missing and not outside and not exists and not _known_absent(view, _path_arguments(view, source), facts, genes):
        return None
    return {'gene_ids': list(pair), 'multiplex_id': view.multiplex_id, 'layer_id': layer,
        'directed': view.layers[layer].directed, 'edge_exists': None if missing else exists,
        'weight': weight if exists and not missing else None,
        'missing_gene_ids': missing, 'not_in_layer_gene_ids': outside}


def collect(view, arguments: Mapping, observations: Sequence[Mapping], *,
            acquisition_requested: bool = True, claim: str | None = None,
            subjects: Sequence[Mapping] | None = None, truth_kind: str | None = None,
            graph_direction: str | None = None, graph_claim: str = 'no_edge_recorded_in_scope',
            graph_truth_scope: str = 'scope') -> LayerProof:
    """Assess ordered subjects and acquire only a required missing edge result.

    Explicit subjects keep their order. The primary query is appended only if
    absent. Evidence-only tasks retain all assessed contrasts without new calls.
    """
    source = _arguments(view, arguments)
    if type(acquisition_requested) is not bool or truth_kind not in (None, 'biological', 'graph'):
        raise ValueError('acquisition and truth requests must be explicit')
    if claim is not None and claim not in CLAIM_VOCABULARY:
        raise ValueError('the biological claim is outside the unchanged S1 vocabulary')
    if truth_kind == 'biological' and claim is None:
        raise ValueError('a biological truth request requires its canonical claim')
    direction = _path_arguments(view, source)['direction']
    if graph_direction is not None and graph_direction != direction:
        raise SelectionError('invalid_input', 'DIRECTION_UNAVAILABLE')
    if subjects is not None and (not isinstance(subjects, (list, tuple))
                                or any(not isinstance(item, Mapping) for item in subjects)):
        raise ValueError('pair subjects require an explicit ordered selector list')
    targets, seen = [], set()
    for item in [*(subjects or ()), source]:
        selected = _arguments(view, item)
        subject = pair_subject(view, selected)
        key = _canonical(subject)
        if key not in seen:
            seen.add(key)
            targets.append((selected, subject))
    records = _records(view, observations)
    facts = [_inspect(view, _path_arguments(view, selected), records) for selected, _ in targets]
    primary = next(data for (selected, _), data in zip(targets, facts)
                   if pair_subject(view, selected) == pair_subject(view, source))
    ordinary = _ordinary(view, source, records, primary)
    if acquisition_requested:
        chosen = [item for item in records if item['operation'].startswith('lookup_human_')
                  or any(_matching(view, selected, item) for selected, _ in targets)]
        if ordinary is not None and not any(_matching(view, source, item) for item in chosen):
            chosen.extend(item for item in primary.used if item not in chosen)
        used = tuple(item for item in records if item in chosen)
    else:
        used = records

    def proof(result=None, pending=None, status='resolved', issue=None):
        return LayerProof(result, deepcopy(pending), 'get_edge' if pending else None,
                          used, _provenance(used), status, issue)

    if acquisition_requested and ordinary is None:
        return proof(pending=source)
    if acquisition_requested and ordinary['missing_gene_ids']:
        return proof(result=ordinary, status='not_found')
    if truth_kind == 'biological':
        return proof(status='invalid_input', issue='UNSUPPORTED_TRUTH_SCOPE')
    cards = []
    for (selected, subject), data in zip(targets, facts):
        supported = []
        if tuple(subject['gene_ids']) in data.arcs:
            supported.extend(FAMILY_CARDS[subject['layer_family']]['supported_claims'])
            if subject['layer_family'] == 'tf_target' and subject['directed']:
                supported.append('regulation_direction')
        cards.append(_card(subject, supported, claim))
    if truth_kind == 'graph':
        graph = _graph_calibrate(view, {**source, 'direction': direction}, records,
            path_requested=False, claim=graph_claim, truth_scope=graph_truth_scope)
        if graph.issue_code:
            return proof(status=graph.status, issue=graph.issue_code)
        cards.extend(graph.result['claim_cards'])
    result = deepcopy(ordinary) if acquisition_requested else {}
    result['claim_cards'] = cards
    return proof(result=result)
