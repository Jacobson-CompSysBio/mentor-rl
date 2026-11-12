"""Build singleton membership and calibration traces from visible evidence.

All three membership categories complete the classification. Explicit gates,
unknown identities, and failed acquisitions keep separate blocked branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_2_identity import RegistryContext, get_symbols
from .s2_3_corpus import gene_group, _provenance
from .s2_6_corpus import _global_observation
from .s2_9_corpus import _active_context as _graph_context
from .s2_11_evidence import _registry_key
from .s2_12_sampling import MembershipFact, fact_from_gene
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_gene_layers import GeneLayersRuntime
from .s2_graph_claims import CLAIM_VOCABULARY
from .s2_multiplex import EvidenceError, MultiplexRuntime
from .s2_neighbors import NeighborRuntime
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphIndex, SubgraphRuntime
from .source_index import check_source_index, content_sha256
from .sources import GENE_ID_PATTERN, file_sha256
from .tools import WorldModelToolRuntime


CONTRACT_PATH = Path(__file__).with_name('s2_12_v1.json')
VARIANTS = ('lookup', 'context', 'duplicate_context', 'superset_context',
    'layer_list_context', 'family_context', 'wrong_family_context', 'prefix_context',
    'prefix_positive_context', 'prefix_omission_context', 'global_context',
    'edge_context', 'neighbor_context', 'neighbor_target_context', 'induced_context',
    'edgeless_context', 'old_view_context', 'stale', 'wrong_layer_context', 'distractor_context',
    'symbol_context', 'symbol_lookup', 'symbol_case_lookup', 'id_context', 'id_lookup',
    'null_symbol_context', 'terminal', 'recovered', 'exhausted',
    'symbol_terminal', 'symbol_recovered', 'symbol_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'tool_unknown_layer', 'tool_unknown_multiplex',
    'missing_scope', 'missing_genes', 'invalid_gene_list', 'unresolved_identity',
    'ambiguous_identity', 'invalid_identity')
LABELS = ('layer_ids', 'old_present', 'registry_state', 'outgoing_count', 'incoming_count', 'self_loop')
MEMBERSHIP_POLICY = ('Each layer inventory is the union of both endpoint columns in its complete source file. '
    'The multiplex inventory is the union of all six layer inventories. No registry filtering is applied.')


def load_build_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load the isolated six-file membership overlay and frozen request bank."""
    from .s2_12_prompts import load_s2_12_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.12']
            or contract['training_ready'] is not False or contract['production_export_allowed'] is not False):
        raise ValueError('unsupported S2.12 build contract')
    contract['_requests'] = load_s2_12_requests(contract['request_bank'])
    for key in ('source_manifest', 'previous_source_manifest'):
        pin = contract[key]
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'the {key} differs from its pin')
    return contract


def _arguments(view, fact):
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': [fact.gene_id]}


def _active_context(view):
    return {**_graph_context(view), 'membership_policy': MEMBERSHIP_POLICY}


def truth_statement(view, fact, claim, truth_scope, truth_layer_ids=None, truth_family=None):
    """Bind an exact proposition to the singleton, graph version, and requested scope."""
    gene, scope = fact.gene_id, f'{view.multiplex_id} at graph version {view.graph_version}'
    if truth_scope == 'category':
        if claim == 'gene_not_node_in_layer':
            return f'{gene} belongs to multiplex {scope} and is absent from its layer {fact.layer_id}.'
        if claim == 'gene_not_node_in_multiplex':
            return f'{gene} is absent from multiplex {scope}.'
        if claim in CLAIM_VOCABULARY[:2]:
            subject = 'native edges' if claim == CLAIM_VOCABULARY[0] else 'native paths'
            return f'No {subject} in layer {fact.layer_id} of {scope} involve {gene}.'
    elif truth_scope == 'multiplex' and claim == 'gene_not_node_in_multiplex':
        return f'{gene} is absent from multiplex {scope}.'
    elif truth_scope == 'all_layers' and claim in ('absence_holds_in_other_layers', 'gene_not_node_in_multiplex'):
        return f'{gene} is absent from every registered layer of {scope}.'
    elif truth_scope == 'layers' and claim == 'absence_holds_in_other_layers':
        if (not isinstance(truth_layer_ids, list) or not truth_layer_ids
                or len(set(truth_layer_ids)) != len(truth_layer_ids)
                or any(layer not in view.layers for layer in truth_layer_ids)):
            raise ValueError('an exact layer-set proposition requires registered unique layers')
        return f'{gene} is absent from every layer in {canonical_json(truth_layer_ids)} of {scope}.'
    elif truth_scope == 'family' and claim == 'absence_holds_in_other_layers':
        families = {MultiplexRuntime(view).family_of(layer) for layer in view.layer_order}
        if truth_family not in families:
            raise ValueError('a family proposition requires a registered family')
        return f'{gene} is absent from every registered {truth_family} layer of {scope}.'
    elif truth_scope == 'future_versions' and claim == 'absence_holds_in_other_versions':
        return f'{gene} will remain absent from {fact.layer_id} in every future graph version of {view.multiplex_id}.'
    elif truth_scope == 'expression' and claim == 'gene_not_expressed_in_context':
        return f'{gene} is not biologically expressed in the context of {fact.layer_id}.'
    raise ValueError('the truth claim and scope do not name a reviewed proposition')


def validate_task(view, fact, question_kind='membership_card', claim=None, truth_scope=None,
                  truth_layer_ids=None, truth_family=None, statement=None, require_membership=None):
    """Keep list cards, support, truth, and explicit outer prerequisites separate."""
    from .s2_12_prompts import KINDS
    if question_kind not in KINDS or require_membership not in (None, 'multiplex', 'layer'):
        raise ValueError('unsupported calibrated membership task or gate')
    if not question_kind.startswith('membership') and require_membership is not None:
        raise ValueError('an outer prerequisite requires a requested classification')
    assessed, truth = question_kind.endswith('support') or question_kind.endswith('truth'), question_kind.endswith('truth')
    if assessed:
        defaults = {'expression': 'gene_not_expressed_in_context', 'future_versions': 'absence_holds_in_other_versions',
                    'all_layers': 'absence_holds_in_other_layers', 'layers': 'absence_holds_in_other_layers',
                    'family': 'absence_holds_in_other_layers', 'multiplex': 'gene_not_node_in_multiplex'}
        claim = claim or defaults.get(truth_scope, 'gene_not_node_in_multiplex'
            if fact.category == 'outside_multiplex' else 'gene_not_node_in_layer')
        if claim not in CLAIM_VOCABULARY:
            raise ValueError('the assessed claim is outside the closed vocabulary')
    elif claim is not None:
        raise ValueError('a list-only membership card does not assess one claim')
    if truth:
        truth_scope = truth_scope or ('multiplex' if claim == 'gene_not_node_in_multiplex' else 'category')
        if truth_scope == 'layers' and truth_layer_ids is None:
            truth_layer_ids = [layer for layer in view.layer_order if layer != fact.layer_id][:2]
        if truth_scope == 'family' and truth_family is None:
            truth_family = MultiplexRuntime(view).family_of(fact.layer_id)
        if truth_scope != 'layers' and truth_layer_ids is not None or truth_scope != 'family' and truth_family is not None:
            raise ValueError('the truth task contains an unrelated scope selector')
        expected = truth_statement(view, fact, claim, truth_scope, truth_layer_ids, truth_family)
        statement = expected if statement is None else statement
        if statement != expected:
            raise ValueError('the literal statement differs from its exact graph proposition')
    elif any(value is not None for value in (truth_scope, truth_layer_ids, truth_family, statement)):
        raise ValueError('a support question cannot silently request statement truth')
    return {'question_kind': question_kind, 'claim': claim, 'truth_scope': truth_scope,
            'truth_layer_ids': truth_layer_ids, 'truth_family': truth_family, 'statement': statement,
            'require_membership': require_membership}


@lru_cache(maxsize=2)
def _registry_rows(path, compressed_sha, logical_sha):
    """Read authoritative candidate details only when an ambiguous lookup needs them."""
    path = ROOT / path
    if file_sha256(path) != compressed_sha:
        raise ValueError('registry candidate details differ from their compressed pin')
    payload = check_source_index(path)
    if content_sha256(payload) != logical_sha:
        raise ValueError('registry candidate details differ from their logical pin')
    return payload['genes_by_id']


class _RegistryRows(Mapping):
    """Supply current symbols and optional source details to the unchanged S1 lookup."""

    def __init__(self, registry, details=None):
        self.registry, self.details = registry, details

    def __getitem__(self, gene):
        symbol = self.registry.symbols_by_id[gene]
        if self.details is None:
            return symbol, None, None
        values = self.details[gene]
        if values[0] != symbol:
            raise ValueError('candidate details disagree with current registry symbols')
        return values[:3]

    def __iter__(self):
        return iter(self.registry.symbols_by_id)

    def __len__(self):
        return len(self.registry.symbols_by_id)


def _identity_output(registry, pin, operation, raw, evidence_id):
    """Reuse S1 lookup rules while preserving raw call arguments and optional nulls."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('a registry lookup requires an explicit literal string')
    details = None
    if operation == 'lookup_human_gene_symbol' and len(registry.gene_ids_by_symbol.get(raw.strip().casefold(), ())) > 1:
        details = _registry_rows(pin['path'], pin['sha256'], pin['logical_sha256'])
    data = SimpleNamespace(_gene_ids_by_symbol=registry.gene_ids_by_symbol, _genes_by_id=_RegistryRows(registry, details))
    provenance = {'registry': [{'registry': 'ensembl', 'release': registry.release, 'evidence_id': evidence_id}], 'graph': []}
    method = WorldModelToolRuntime._lookup_symbol if operation == 'lookup_human_gene_symbol' else WorldModelToolRuntime._lookup_gene_id
    return method(data, raw.strip(), provenance)


def _identity_record(operation, source, response):
    return {'operation': operation, 'arguments': deepcopy(source), 'status': response['status'],
            'result': {key: deepcopy(value) for key, value in response.items() if key not in ('status', 'provenance')},
            'provenance': deepcopy(response['provenance'])}


def _plan(view, request, observations, task):
    """Resolve a literal identity before graph work, using only visible mappings."""
    from .s2_12_evidence import MembershipProof, collect
    raw = request['genes'][0]
    identities = []
    for item in observations:
        if item['operation'] in ('lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id') and item not in identities:
            identities.append(item)
    if GENE_ID_PATTERN.fullmatch(raw):
        gene = raw
    else:
        operation = 'lookup_human_ensembl_gene_id' if raw.strip().lower().startswith('ensg') else 'lookup_human_gene_symbol'
        field = 'gene_id' if operation.endswith('gene_id') else 'gene_symbol'
        matching = []
        for item in identities:
            if item['operation'] != operation:
                continue
            if item['status'] != 'resolved':
                if item['arguments'] == {field: raw}:
                    matching.append(item)
                continue
            key = _registry_key(operation, item['arguments'], item['result'])
            try:
                requested_key = _registry_key(operation, {field: raw}, item['result'])
            except EvidenceError:
                continue
            if requested_key == key:
                matching.append(item)
        if not matching:
            return MembershipProof(None, {field: raw}, operation, tuple(identities), _provenance(identities), None), identities
        chosen = matching[0]
        if chosen['status'] != 'resolved':
            return MembershipProof(None, None, None, tuple(identities), _provenance(identities), None,
                                   chosen['status'], 'UNRESOLVED_IDENTITY'), identities
        _registry_key(chosen['operation'], chosen['arguments'], chosen['result'])
        gene = chosen['result']['gene_id']
    arguments = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'], 'gene_ids': [gene]}
    proof = collect(view, arguments, observations, classification_requested=task['question_kind'].startswith('membership'),
        claim=task['claim'], truth_scope=task['truth_scope'], truth_layer_ids=task['truth_layer_ids'],
        truth_family=task['truth_family'], require_membership=task['require_membership'])
    used = []
    for item in observations:
        if (item in identities or item in proof.observations) and item not in used:
            used.append(item)
    return proof, used


@lru_cache(maxsize=12)
def _split_candidates(genes: frozenset[str], seed: int, split: str) -> tuple[str, ...]:
    """Reuse two fixed split candidates so excluding one query keeps the same first ID."""
    return tuple(sorted((gene for gene in genes if gene_group(gene, seed)[1] == split),
                        key=lambda gene: gene_group(gene, seed)[0])[:2])


def _context(view, fact, variant, row_id, index, registry, contract, old_view, seed=20261028):
    """Create original attributed source records for one declared evidence contrast."""
    observations, serial = [], 0
    nodes, memberships = NodesByLayerRuntime(view), GeneLayersRuntime(view)
    arguments = _arguments(view, fact)

    def add(operation, source, execute):
        nonlocal serial
        serial += 1
        observations.append(_observation(operation, source, execute(**source, evidence_id=f'ev_{row_id}_{serial}')))

    if variant in ('context', 'duplicate_context', 'superset_context', 'stale', 'wrong_layer_context', 'distractor_context'):
        source = deepcopy(arguments)
        if variant == 'superset_context':
            extra = next((gene for gene in _split_candidates(view.multiplex_nodes, seed, fact.split)
                          if gene != fact.gene_id), None)
            if extra is None:
                raise VariantUnavailable('no second same-split source gene exists')
            source['gene_ids'] = [extra, fact.gene_id]
        elif variant == 'wrong_layer_context':
            source['layer_id'] = next((layer for layer in view.layer_order if layer != fact.layer_id), None)
            if source['layer_id'] is None:
                raise VariantUnavailable('no second registered layer exists')
        elif variant == 'distractor_context':
            extra = next((gene for gene in _split_candidates(view.multiplex_nodes, seed, fact.split)
                          if gene != fact.gene_id), None)
            if extra is None:
                raise VariantUnavailable('no distinct same-split distractor exists')
            source['gene_ids'] = [extra]
        add('get_nodes_by_layer', source, nodes.get_nodes_by_layer)
        if variant == 'duplicate_context':
            observations.append(deepcopy(observations[0]))
        elif variant == 'stale':
            observations[0]['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('layer_list_context', 'family_context', 'wrong_family_context', 'prefix_context',
                     'prefix_positive_context', 'prefix_omission_context'):
        family = MultiplexRuntime(view).family_of(fact.layer_id) if variant == 'family_context' else None
        if variant == 'wrong_family_context':
            family = next((MultiplexRuntime(view).family_of(layer) for layer in view.layer_order
                if MultiplexRuntime(view).family_of(layer) != MultiplexRuntime(view).family_of(fact.layer_id)), None)
            if family is None:
                raise VariantUnavailable('no other registered family exists')
        prefix = variant.startswith('prefix')
        if prefix and len(fact.layer_ids) < 2:
            raise VariantUnavailable('a genuine containing-layer prefix omits a positive layer')
        if variant == 'prefix_positive_context' and fact.layer_ids[0] != fact.layer_id:
            raise VariantUnavailable('the target is not an explicitly returned top-one positive')
        if variant == 'prefix_omission_context' and fact.layer_ids[0] == fact.layer_id:
            raise VariantUnavailable('the target is shown in this prefix')
        source = memberships.arguments(fact.gene_id, {'kind': 'all', 'layer_ids': [], 'layer_family': family},
            mode='top_k' if prefix else 'complete', top_k=1 if prefix else None)
        add('get_gene_layers', source, memberships.get_gene_layers)
    elif variant == 'global_context':
        observations.append(_global_observation(view, [fact.gene_id], f'ev_{row_id}_global'))
    elif variant in ('edge_context', 'neighbor_context', 'neighbor_target_context'):
        native = view.layers[fact.layer_id]
        if variant == 'edge_context':
            sides = index.native_index.directions(fact.layer_id)
            choices = [(side, gene) for side in sides for gene, _ in index.native_index.neighbors(fact.layer_id, fact.gene_id, side)
                       if gene != fact.gene_id]
            if not choices:
                raise VariantUnavailable('a positive edge context needs a distinct native endpoint')
            side, gene = min(choices, key=lambda pair: pair[1])
            pair = [gene, fact.gene_id] if side == 'in' else [fact.gene_id, gene]
            source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': pair}
            add('get_edge', source, view.get_edge)
        else:
            gene = fact.gene_id
            direction = 'out' if native.directed else 'undirected'
            if variant == 'neighbor_target_context':
                incoming = [source for source, _ in index.native_index.neighbors(fact.layer_id, fact.gene_id,
                            'in' if native.directed else 'undirected') if source != fact.gene_id]
                if not incoming:
                    raise VariantUnavailable('a returned-neighbor context needs a distinct native source')
                gene = min(incoming, key=lambda item: len(index.native_index.neighbors(fact.layer_id, item, direction)))
            source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': gene,
                      'direction': direction, 'mode': 'complete', 'top_k': None, 'cursor': None}
            add('get_neighbors', source, NeighborRuntime(view, index=index.native_index).get_neighbors)
    elif variant in ('induced_context', 'edgeless_context'):
        if variant == 'edgeless_context' and (fact.category != 'member' or fact.self_loop):
            raise VariantUnavailable('an edgeless singleton must remain an actual loop-free layer member')
        runtime = SubgraphRuntime(view, index=index)
        source = runtime.arguments([fact.gene_id], fact.layer_id)
        add('induce_subgraph', source, runtime.induce_subgraph)
    elif variant == 'old_view_context':
        if old_view is None:
            raise VariantUnavailable('the old-view contrast requires its separate pinned inventory')
        source = {'multiplex_id': old_view.multiplex_id, 'layer_id': old_view.layer_order[0], 'gene_ids': [fact.gene_id]}
        add('get_nodes_by_layer', source, NodesByLayerRuntime(old_view).get_nodes_by_layer)
    elif variant in ('symbol_context', 'symbol_terminal', 'symbol_recovered', 'symbol_exhausted', 'id_context', 'null_symbol_context'):
        if variant.startswith('symbol'):
            try:
                raw = get_symbols([fact.gene_id], registry)[0]
            except ValueError as error:
                raise VariantUnavailable(str(error)) from error
            operation, field = 'lookup_human_gene_symbol', 'gene_symbol'
        else:
            if fact.registry_state == 'graph_only' or variant == 'null_symbol_context' and fact.registry_state != 'null_symbol':
                raise VariantUnavailable('this supplied ID context needs the matching authoritative registry identity')
            raw, operation, field = fact.gene_id + '.4', 'lookup_human_ensembl_gene_id', 'gene_id'
        serial += 1
        source = {field: raw}
        response = _identity_output(registry, contract['registry_index'], operation, raw, f'ev_{row_id}_{serial}')
        observations.append(_identity_record(operation, source, response))
    return observations, serial


def _blocked(request, issue, used, rejected=None):
    return {'issue_code': issue, 'request': deepcopy(request), 'rejected_arguments': deepcopy(rejected),
            'partial_results': list(used)}


def _answer(request, task, framing, proof, used, failure=None):
    if failure:
        operation, source, response = failure
        return {'status': 'error', 'provenance': _provenance(used), 'result': {'failed_tool': operation,
            'error_code': response['error_code'], 'arguments': deepcopy(source), 'partial_results': list(used),
            'coverage': proof.coverage}, 'continuation_decision': 'defer',
            'blocked_reason': 'The required acquisition must recover before the missing membership evidence is available.',
            'summary': 'The required acquisition failed. Retain successful partials and leave the attempted classification unchecked.'}
    if proof.issue_code:
        return {'status': proof.status, 'provenance': _provenance(used),
            'result': _blocked(request, proof.issue_code, used), 'continuation_decision': 'defer',
            'blocked_reason': 'The explicit outer membership prerequisite fails.' if proof.issue_code == 'MEMBERSHIP_PRECONDITION_UNSATISFIED'
                else 'The exact proposition needs adequate evidence for its requested scope.',
            'summary': 'Retain successful observations and defer the blocked outer task without an invented classification or truth verdict.'}
    if proof.result is None or proof.next_arguments is not None:
        raise ValueError('a required membership or calibrated result remains unresolved')
    return {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
            'summary': 'The requested membership and graph-evidence assessment are complete within their stated scope.'}


def _reason(answer, framing):
    result = answer['result']
    if answer['status'] == 'error':
        return ('The required acquisition failed and adds no evidence. Preserve successful identities and graph observations. '
                'Its attempted gene remains unchecked. Defer after the visible retry allowance ends.')
    if 'issue_code' in result:
        return ('The outer task has an unresolved prerequisite or an undecidable truth scope. '
                'Retain successful observations as original partial results. Unsupported does not mean false. Defer for the stated correction.')
    parts = ['Use the current multiplex version, registered layer, and authoritative endpoint inventories.']
    if 'layer_gene_ids' in result:
        parts.append(f"Layer members are {result['layer_gene_ids']}; layer exclusions are {result['not_in_layer_gene_ids']}; global exclusions are {result['missing_gene_ids']}.")
        parts.append('The three lists partition the singleton. Their positive count is not the size of the layer.')
        parts.append('All three categories complete this membership operation, including a checked global negative.')
    for card in result['claim_cards']:
        parts.append(f"For {card['subject']['gene_id']}, the supported graph concepts are {card['supported_claims']}.")
        if 'claim' in card:
            parts.append(f"The requested claim is {card['claim']}; its support verdict is {card['claim_supported']}.")
        if 'statement_true' in card:
            parts.append(f"The exact requested proposition has statement_true {card['statement_true']}.")
    parts.append('Every other vocabulary concept remains unsupported. Membership alone does not establish biological expression or a particular interaction.')
    parts.append('A bounded omission, zero outgoing count, or empty selected edge set does not establish node absence.')
    parts.append('The workflow step is complete, so continue.' if framing == 'step' else 'The standalone task is complete, so stop.')
    return ' '.join(parts)


def build_row(view: GraphView, fact: MembershipFact, framing: str, variant: str, variant_index: int,
              bank: Mapping, contract: Mapping, *, index: SubgraphIndex, registry: RegistryContext,
              old_view: GraphView | None = None, question_kind='membership_card', claim=None, truth_scope=None,
              truth_layer_ids=None, truth_family=None, statement=None, require_membership=None,
              seed: int = 20261028) -> dict:
    """Construct a calibrated singleton trace with exact identity and backend steps."""
    from .s2_12_prompts import render_s2_12_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported membership contrast or framing')
    task = validate_task(view, fact, question_kind, claim, truth_scope, truth_layer_ids, truth_family, statement, require_membership)
    raw = fact.gene_id
    if variant.startswith('symbol'):
        try:
            raw = get_symbols([fact.gene_id], registry)[0]
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
        if variant == 'symbol_case_lookup':
            raw = ' ' + raw.lower() + ' '
    elif variant in ('id_context', 'id_lookup', 'null_symbol_context'):
        if fact.registry_state == 'graph_only':
            raise VariantUnavailable('a versioned registry contrast needs an authoritative current identity')
        raw = fact.gene_id + '.4'
    request = {'genes': [raw], 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id}
    core = render_s2_12_request(contract['_requests'], split=fact.split, variant_index=variant_index, **request, **task)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's212_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, task, user])[:32]
    observations, serial = _context(view, fact, variant, row_id, index, registry, contract, old_view, seed)
    if not question_kind.startswith('membership') and not observations:
        raise VariantUnavailable('a claim-only assessment requires a supplied observation')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': _active_context(view), 'observations': observations})},
        {'role': 'user', 'content': user}]
    fault = next((kind for kind in ('terminal', 'recovered', 'exhausted') if variant.endswith(kind)), None)
    failure, injected = None, False
    while True:
        proof, used = _plan(view, request, observations, task)
        if proof.next_arguments is None:
            break
        operation, source = proof.next_tool, deepcopy(proof.next_arguments)

        def execute():
            nonlocal serial
            serial += 1
            identity = f'ev_{row_id}_{serial}'
            if operation.startswith('lookup_human_'):
                return _identity_output(registry, contract['registry_index'], operation, next(iter(source.values())), identity)
            if operation != 'get_nodes_by_layer':
                raise ValueError('a missing singleton classification must use the unchanged node tool')
            return NodesByLayerRuntime(view).get_nodes_by_layer(**source, evidence_id=identity)

        def append(response, thought):
            messages.extend([{'role': 'assistant', 'thinking': thought,
                'tool_call': {'name': operation, 'arguments': deepcopy(source)}},
                {'role': 'tool', 'name': operation, 'content': response}])

        if fault and operation == 'get_nodes_by_layer' and not injected:
            injected = True
            response = _failure(None if fault == 'terminal' else 1)
            append(response, 'The compatible visible evidence does not settle this singleton. Request only its missing current classification.')
            if fault != 'terminal':
                response = _failure(0) if fault == 'exhausted' else execute()
                append(response, 'The visible error grants one retry. Repeat the identical tool and arguments, then stop after recovery or exhaustion.')
            if response['status'] == 'error':
                failure = operation, source, response
                break
        else:
            response = execute()
            append(response, 'Resolve a literal identity when needed, then acquire only the missing current singleton membership. '
                   'A registry identity is not graph membership. Do not infer absence from a bounded display or another family.')
        observations.append(_identity_record(operation, source, response) if operation.startswith('lookup_human_')
                            else _observation(operation, source, response))
    if fault and not injected:
        raise VariantUnavailable('the supplied evidence already settles this classification before a failed acquisition')
    answer = _answer(request, task, framing, proof, used, failure)
    messages.append({'role': 'assistant', 'thinking': _reason(answer, framing), 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.12', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_id': fact.gene_id,
        'category': fact.category, **{key: getattr(fact, key) for key in LABELS},
        'framing': framing, 'variant': variant, **task, 'request': request, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][question_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view, fact, kind):
    request = {'genes': [fact.gene_id], 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'tool_unknown_multiplex': 'UNKNOWN_MULTIPLEX',
        'missing_scope': 'MISSING_SCOPE', 'missing_genes': 'MISSING_GENES',
        'invalid_gene_list': 'INVALID_GENE_LIST', 'unresolved_identity': 'UNRESOLVED_IDENTITY',
        'ambiguous_identity': 'UNRESOLVED_IDENTITY', 'invalid_identity': 'UNRESOLVED_IDENTITY'}
    if kind not in codes:
        raise ValueError('unsupported membership prerequisite diagnostic')
    if kind.endswith('unknown_layer'):
        request['layer_id'] = 'scPEN:brain:unregistered-context:astrocyte'
    elif kind.endswith('unknown_multiplex'):
        request['multiplex_id'] = 'unknown_membership_multiplex'
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_gene_list':
        request['genes'] = [fact.gene_id, fact.gene_id]
    elif kind in ('unresolved_identity', 'ambiguous_identity', 'invalid_identity'):
        request['genes'] = [{'unresolved_identity': 'HER2', 'ambiguous_identity': 'SHOX',
                             'invalid_identity': 'ENSGinvalid'}[kind]]
    status = ('ambiguous' if kind == 'ambiguous_identity' else 'invalid_input' if kind in
        ('missing_scope', 'missing_genes', 'invalid_gene_list', 'invalid_identity') else 'not_found')
    return request, status, codes[kind]


def _diagnostic_wording(kind):
    if kind in ('ambiguous_identity', 'invalid_identity'):
        return 'unresolved_identity'
    return kind.removeprefix('tool_')


def build_diagnostic(view: GraphView, fact: MembershipFact, framing: str, variant_index: int,
                     bank: Mapping, contract: Mapping, *, kind='missing_genes',
                     registry: RegistryContext | None = None, seed=20261028) -> dict:
    """Preserve exact singleton prerequisites, candidate details, and rejected calls."""
    from .s2_12_prompts import render_s2_12_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    wording = _diagnostic_wording(kind)
    core = render_s2_12_diagnostic_request(contract['_requests'], split=fact.split, kind=wording,
        variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's212_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32]
    active, used, rejected = _active_context(view), [], None
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    if kind.startswith('tool_'):
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'], 'gene_ids': request['genes']}
        response = NodesByLayerRuntime(view).get_nodes_by_layer(**rejected, evidence_id='unused_rejected_membership_scope')
        messages.extend([{'role': 'assistant', 'thinking': 'Ask the native resolver for the explicit requested scope before classifying an unknown representation.',
            'tool_call': {'name': 'get_nodes_by_layer', 'arguments': deepcopy(rejected)}},
            {'role': 'tool', 'name': 'get_nodes_by_layer', 'content': response}])
    elif kind.endswith('identity'):
        if registry is None:
            raise ValueError('identity diagnostics require the pinned current registry')
        raw = request['genes'][0]
        operation = 'lookup_human_ensembl_gene_id' if kind == 'invalid_identity' else 'lookup_human_gene_symbol'
        source = {'gene_id' if operation.endswith('gene_id') else 'gene_symbol': raw}
        response = _identity_output(registry, contract['registry_index'], operation, raw, f'ev_{row_id}_identity')
        if response['status'] != status:
            raise ValueError('the declared identity scenario disagrees with the complete pinned registry')
        used.append(_identity_record(operation, source, response))
        messages.extend([{'role': 'assistant', 'thinking': 'Resolve the literal input under the current registry. Preserve all candidates and do not choose by expected graph membership.',
            'tool_call': {'name': operation, 'arguments': source}},
            {'role': 'tool', 'name': operation, 'content': response}])
    answer = {'status': status, 'provenance': _provenance(used), 'result': _blocked(request, code, used, rejected),
        'continuation_decision': 'defer', 'blocked_reason': 'Resolve the identity, valid singleton input, or requested graph scope.',
        'summary': 'A membership prerequisite is unresolved. Retain the literal request and successful registry observations without a fabricated graph category.'}
    messages.append({'role': 'assistant', 'thinking': f'The raw request is {request}. Its prerequisite fails with {code}. '
        'Unknown or ambiguous identities are not graph-absence categories. Retain candidates and rejected arguments, then defer.', 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.12', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_id': fact.gene_id, 'category': fact.category,
        **{key: None for key in LABELS}, 'framing': framing, 'variant': kind,
        'question_kind': 'membership_card', 'claim': None, 'truth_scope': None, 'truth_layer_ids': None,
        'truth_family': None, 'statement': None, 'require_membership': None,
        'request': request, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping, view: GraphView, *, bank: Mapping, contract: Mapping,
                 registry: RegistryContext, index: SubgraphIndex | None = None,
                 old_view: GraphView | None = None) -> None:
    """Verify native acquisitions and replay exact visible membership and claim proofs."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_12_prompts import render_s2_12_request, render_s2_12_diagnostic_request
    if old_view is None:
        from .s2_sources import load_source_view
        old_view = load_source_view(ROOT / contract['previous_source_manifest']['path'])
    index = index or SubgraphIndex(view)
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('a singleton row requires one user turn and an explicit harness')
    for message in messages:
        fields = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != fields or message['role'] == 'assistant' and (not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if (messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']
            or meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.12'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'
            or meta['framing'] not in FRAMINGS):
        raise ValueError('the membership row differs from its build identity')
    fact = fact_from_gene(view, index, registry, old_view, meta['layer_id'], meta['gene_id'], meta['selection_seed'])
    if fact.split != meta['split'] or fact.split_unit != meta['split_unit'] or fact.category != meta['category']:
        raise ValueError('the source identity, membership category, or fixed gene split changed')
    final = messages[-1]['content']
    if (list(final) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or not isinstance(final['summary'], str) or not final['summary'].strip()
            or (final['continuation_decision'] == 'defer') != bool(final['blocked_reason'])
            or final['blocked_reason'] is not None and (not isinstance(final['blocked_reason'], str) or not final['blocked_reason'].strip())):
        raise ValueError('the final envelope, free summary, or blocker presence is invalid')
    diagnostic = meta['variant'] in DIAGNOSTICS
    request = meta['request']
    if set(request) != {'genes', 'multiplex_id', 'layer_id'}:
        raise ValueError('membership diagnostics preserve exactly three raw request fields')
    task = {key: meta[key] for key in ('question_kind', 'claim', 'truth_scope', 'truth_layer_ids', 'truth_family',
                                     'statement', 'require_membership')}
    if task != validate_task(view, fact, **task):
        raise ValueError('the explicit claim, truth proposition, or outer gate changed')
    wording = _diagnostic_wording(meta['variant'])
    templates = (contract['_requests']['diagnostic_requests'][meta['split']][wording] if diagnostic else
                 contract['_requests'][meta['split']][meta['question_kind']])
    for key, size in (('wrapper_index', len(bank[meta['split']][meta['framing']])), ('request_template_index', len(templates))):
        if type(meta[key]) is not int or not 0 <= meta[key] < size:
            raise ValueError('the request must select an audited wording and wrapper')
    if diagnostic:
        expected_request, status, code = _diagnostic_request(view, fact, meta['variant'])
        if request != expected_request or any(meta[key] is not None for key in LABELS):
            raise ValueError('the diagnostic changes its original inputs or private labels')
        if task != {'question_kind': 'membership_card', 'claim': None, 'truth_scope': None,
                    'truth_layer_ids': None, 'truth_family': None, 'statement': None, 'require_membership': None}:
            raise ValueError('a prerequisite diagnostic cannot assess an ordinary card')
        core = render_s2_12_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording,
            variant_index=meta['request_template_index'], **request)
    else:
        if meta['variant'] not in VARIANTS or any(canonical_json(meta[key]) != canonical_json(getattr(fact, key)) for key in LABELS):
            raise ValueError('the source labels or selected contrast changed')
        raw = fact.gene_id
        if meta['variant'].startswith('symbol'):
            raw = get_symbols([fact.gene_id], registry)[0]
            if meta['variant'] == 'symbol_case_lookup':
                raw = ' ' + raw.lower() + ' '
        elif meta['variant'] in ('id_context', 'id_lookup', 'null_symbol_context'):
            raw += '.4'
        if request != {'genes': [raw], 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id}:
            raise ValueError('the raw singleton, namespace, or scope changed')
        core = render_s2_12_request(contract['_requests'], split=meta['split'],
            variant_index=meta['request_template_index'], **request, **task)
    user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': user}:
        raise ValueError('the visible request differs from its frozen task wording')
    identity = ([contract['contract_version'], view.graph_version, user, meta['variant']] if diagnostic else
        [contract['contract_version'], view.graph_version, asdict(fact), meta['framing'], meta['variant'], task, user])
    if row['row_id'] != 's212_' + digest(identity)[:32]:
        raise ValueError('the row ID does not bind its source fact, task, contrast, and wording')
    context = json.loads(messages[1]['content'])
    active = _active_context(view)
    if meta['variant'] == 'tool_unknown_layer':
        active.pop('layers')
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != active:
        raise ValueError('the visible graph identity or explicit union policy changed')
    if diagnostic:
        if context['observations']:
            raise ValueError('a local prerequisite cannot invent supplied graph observations')
        used, rejected, middle = [], None, []
        if meta['variant'].startswith('tool_'):
            rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'], 'gene_ids': request['genes']}
            response = NodesByLayerRuntime(view).get_nodes_by_layer(**rejected, evidence_id='unused_rejected_membership_scope')
            middle = [{'role': 'assistant', 'tool_call': {'name': 'get_nodes_by_layer', 'arguments': rejected}},
                      {'role': 'tool', 'name': 'get_nodes_by_layer', 'content': response}]
        elif meta['variant'].endswith('identity'):
            operation = 'lookup_human_ensembl_gene_id' if meta['variant'] == 'invalid_identity' else 'lookup_human_gene_symbol'
            source = {'gene_id' if operation.endswith('gene_id') else 'gene_symbol': request['genes'][0]}
            response = _identity_output(registry, contract['registry_index'], operation, request['genes'][0],
                                        f"ev_{row['row_id']}_identity")
            used.append(_identity_record(operation, source, response))
            middle = [{'role': 'assistant', 'tool_call': {'name': operation, 'arguments': source}},
                      {'role': 'tool', 'name': operation, 'content': response}]
        if len(messages) != 4 + len(middle) or any(any(canonical_json(actual[key]) != canonical_json(value)
            for key, value in expected.items()) for actual, expected in zip(messages[3:-1], middle)):
            raise ValueError('the prerequisite changes its exact call or response')
        if (final['status'] != status or final['continuation_decision'] != 'defer'
                or final['result'] != _blocked(request, code, used, rejected) or final['provenance'] != _provenance(used)):
            raise ValueError('the diagnostic changes its structured prerequisite or original evidence')
        return
    expected_context, _ = _context(view, fact, meta['variant'], row['row_id'], index, registry, contract,
                                  old_view, meta['selection_seed'])
    if canonical_json(context['observations']) != canonical_json(expected_context):
        raise ValueError('the contrast differs from its original source observations')
    available, acquisitions = list(context['observations']), {}

    def verify(observation):
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected observation fields')
        for values in observation['provenance'].values():
            for record in values:
                key = record['evidence_id']
                if key in acquisitions and canonical_json(acquisitions[key]) != canonical_json(observation):
                    raise ValueError('one evidence ID names contradictory acquisitions')
                acquisitions[key] = observation
        operation, source = observation['operation'], observation['arguments']
        if operation.startswith('lookup_human_'):
            records = observation['provenance']['registry']
            if len(records) != 1:
                raise ValueError('each registry acquisition needs one original record')
            output = _identity_output(registry, contract['registry_index'], operation, next(iter(source.values())), records[0]['evidence_id'])
            expected = _identity_record(operation, source, output)
        else:
            records = observation['provenance']['graph']
            if len(records) != 1:
                raise ValueError('each graph acquisition needs one original record')
            selected, checked = view, deepcopy(observation)
            if meta['variant'] == 'old_view_context' and source['multiplex_id'] == old_view.multiplex_id:
                selected = old_view
            elif records[0]['graph_version'] != view.graph_version:
                if meta['variant'] != 'stale' or records[0]['graph_version'] != '0' * 64:
                    raise ValueError('undeclared incompatible graph evidence')
                checked['provenance']['graph'][0]['graph_version'] = view.graph_version
            if operation == 'check_multiplex_membership':
                expected = _global_observation(selected, source['gene_ids'], records[0]['evidence_id'])
            else:
                if operation == 'get_nodes_by_layer':
                    execute = NodesByLayerRuntime(selected).get_nodes_by_layer
                elif operation == 'get_gene_layers':
                    execute = GeneLayersRuntime(selected).get_gene_layers
                elif operation == 'get_edge':
                    execute = selected.get_edge
                elif operation == 'get_neighbors':
                    execute = NeighborRuntime(selected, index=index.native_index if selected is view else None).get_neighbors
                elif operation == 'induce_subgraph':
                    execute = SubgraphRuntime(selected, index=index if selected is view else None).induce_subgraph
                else:
                    raise ValueError('unsupported graph acquisition in this singleton corpus')
                expected = _observation(operation, source, execute(**source, evidence_id=records[0]['evidence_id']))
            observation = checked
        if canonical_json(observation) != canonical_json(expected):
            raise ValueError('a saved acquisition differs from its exact pinned source')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    responses, failure = [], None
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, used = _plan(view, request, available, task)
        call = message['tool_call']
        if proof.next_arguments is None or call != {'name': proof.next_tool, 'arguments': proof.next_arguments}:
            raise ValueError('the call precedes identity resolution or changes the required singleton acquisition')
        tool_message = messages[position + 1]
        if tool_message['role'] != 'tool' or tool_message['name'] != call['name']:
            raise ValueError('the response does not match its required call')
        response = tool_message['content']
        if previous_response and previous_response['status'] == 'error' and (previous_response != _failure(1) or call != previous_call):
            raise ValueError('the retry lacks visible allowance or changes its exact arguments')
        responses.append(response)
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('an undeclared failure invented evidence or a retry budget')
            failure = call['name'], call['arguments'], response
        else:
            record = (_identity_record(call['name'], call['arguments'], response) if call['name'].startswith('lookup_human_')
                      else _observation(call['name'], call['arguments'], response))
            verify(record)
            available.append(record)
            failure = None
        previous_call, previous_response = call, response
    fault = next((kind for kind in ('terminal', 'recovered', 'exhausted') if meta['variant'].endswith(kind)), None)
    errors = [response for response in responses if response['status'] == 'error']
    expected_errors = ([_failure(None)] if fault == 'terminal' else [_failure(1), _failure(0)]
                       if fault == 'exhausted' else [_failure(1)] if fault == 'recovered' else [])
    if errors != expected_errors or previous_response == _failure(1):
        raise ValueError('the fault schedule changed or an authorized retry remains unused')
    proof, used = _plan(view, request, available, task)
    expected = _answer(request, task, meta['framing'], proof, used, failure)
    if any(canonical_json(final[key]) != canonical_json(expected[key]) for key in
            ('status', 'result', 'provenance', 'continuation_decision')):
        raise ValueError('the final classification, claim, truth, gate, evidence, or task decision changed')


def build_corpus(output_dir: Path, *, counts: Mapping | None = None, smoke=False,
                 seed=20261028, contract_path: Path = CONTRACT_PATH, source_manifest: Path | None = None):
    """Export S2.12 after source, evidence, wording, and full token checks."""
    from .s2_12_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
