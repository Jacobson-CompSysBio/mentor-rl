"""Bind every reviewed S2.2 case to fresh, source-selected generator behavior.

This isolated extension leaves published builders and corpora unchanged. Case
bindings specify observable semantics rather than copying the design examples.
Additional identity and prerequisite branches are labelled separately, so they
cannot stand in for a numbered example with a different evidence requirement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import load_s2_prompt_banks, render_s2_prompt
from .s2_2_corpus import (
    MultiplexFact, VariantUnavailable, _active, _observation, _proofs, _reason,
    build_direction_diagnostic, build_row, load_build_contract as _load_base_contract, select_facts,
    validate_row,
)
from .s2_2_identity import (
    RegistryContext, get_symbols, load_registry_context,
    validate_identity_observations,
)
from .s2_corpus import ROOT, SPLITS, _pair_group, canonical_json, digest
from .s2_multiplex import MultiplexRuntime
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


EXTENSION_PATH = Path(__file__).with_name('s2_2_production_v1.json')
CONTRACT_PATH = EXTENSION_PATH
ALL = {'kind': 'all', 'layer_ids': [], 'layer_family': None}
TF = {'kind': 'all', 'layer_ids': [], 'layer_family': 'tf_target'}


@dataclass(frozen=True)
class CaseBinding:
    """One reviewed scope, predicate, framing, and source/evidence condition."""

    condition: str
    mode: str = 'complete'
    variant: str = 'lookup'
    framing: str = 'direct'
    query: str = 'any_direction'
    scope: str = 'all'


CASE_BINDINGS = {
    'S2.2.1': CaseBinding('multiple'),
    'S2.2.2': CaseBinding('one', framing='step'),
    'S2.2.3': CaseBinding('zero', mode='exists'),
    'S2.2.4': CaseBinding('symbols', variant='symbol_context', framing='step'),
    'S2.2.5': CaseBinding('outside'),
    'S2.2.6': CaseBinding('multiple', variant='partial', framing='step'),
    'S2.2.7': CaseBinding('positive', mode='exists', variant='reversed_context'),
    'S2.2.8': CaseBinding('undirected_witness', mode='exists', variant='single_edge_context', framing='step'),
    'S2.2.9': CaseBinding('multiple', mode='top_k'),
    'S2.2.10': CaseBinding('multiple', variant='bounded_context', framing='step'),
    'S2.2.11': CaseBinding('zero', mode='exists', framing='step'),
    'S2.2.12': CaseBinding('forward_only', query='source_to_target', scope='tf'),
    'S2.2.13': CaseBinding('reverse_only', framing='step', query='source_to_target', scope='tf'),
    'S2.2.14': CaseBinding('reciprocal_one'),
    'S2.2.15': CaseBinding('multiple', variant='terminal'),
    'S2.2.16': CaseBinding('multiple', variant='recovered', framing='step'),
    'S2.2.17': CaseBinding('multiple', variant='exhausted'),
    'S2.2.18': CaseBinding('subset', scope='undirected'),
    'S2.2.19': CaseBinding('direction_invalid', query='source_to_target'),
    'S2.2.20': CaseBinding('layer_absence', mode='exists', scope='absent_layer'),
    'S2.2.21': CaseBinding('positive', query='both_directions', scope='tf'),
    'S2.2.22': CaseBinding('later_directed', mode='exists', variant='later_witness'),
}
DIAGNOSTICS = ('missing_genes', 'invalid_pair', 'invalid_identity', 'missing_scope',
               'unknown_multiplex', 'unknown_layer', 'unknown_family',
               'invalid_selection', 'invalid_predicate', 'tool_unknown_layer')


def load_production_contract(path: Path = EXTENSION_PATH) -> dict[str, Any]:
    """Pin the historical contract and add only the isolated extension rules."""
    extension = json.loads(Path(path).read_text())
    if extension['schema_version'] != 'mentor-rl-s2-multiplex-production-extension-v1':
        raise ValueError('unsupported S2.2 production extension')
    pin = extension['base_contract']
    base = ROOT / pin['path']
    if file_sha256(base) != pin['sha256']:
        raise ValueError('the historical S2.2 contract changed')
    contract = _load_base_contract(base)
    contract['contract_version'] = extension['contract_version']
    contract['system_prompt'] += '\n\n' + extension['system_append']
    contract['tools'] = [*contract['tools'], extension['identity_tool']]
    contract['_diagnostic_leads'] = extension['diagnostic_leads']
    return contract


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Expose the standard family-loader API for the isolated extension."""
    return load_production_contract(path)


@lru_cache(maxsize=1)
def load_inputs() -> tuple[GraphView, dict, dict, RegistryContext]:
    """Load and verify the pinned native graph, wording, and identity-only index."""
    contract = load_production_contract()
    pin = contract['source_manifest']
    if file_sha256(ROOT / pin['path']) != pin['sha256']:
        raise ValueError('S2.2 graph-source manifest changed')
    return (load_source_view(ROOT / pin['path']), contract,
            load_s2_prompt_banks(contract['prompt_bank']),
            load_registry_context(contract['registry_index']))


def _scope(view: GraphView, binding: CaseBinding, pair: Sequence[str]) -> dict:
    if binding.scope == 'tf':
        return deepcopy(TF)
    if binding.scope == 'undirected':
        layers = [layer for layer in view.layer_order if not view.layers[layer].directed]
        if not layers or len(layers) == len(view.layer_order):
            raise VariantUnavailable('the view has no proper undirected subset')
        return {'kind': 'layers', 'layer_ids': layers, 'layer_family': None}
    if binding.scope == 'absent_layer':
        layers = [layer for layer in view.layer_order
                  if set(pair) - view.layers[layer].nodes and set(pair) <= view.multiplex_nodes]
        if not layers:
            raise VariantUnavailable('no selected layer distinguishes layer and global absence')
        return {'kind': 'layers', 'layer_ids': layers[-1:], 'layer_family': None}
    return deepcopy(ALL)


def _fact(view: GraphView, pair: Sequence[str], binding: CaseBinding, seed: int) -> MultiplexFact:
    scope = _scope(view, binding, pair)
    unit, split = _pair_group(tuple(pair), seed)
    result = MultiplexRuntime(view).get_edge_layers(view.multiplex_id, list(pair), scope,
        'any_direction' if binding.condition == 'direction_invalid' else binding.query,
        'complete', None, evidence_id='selection_only')['result']
    return MultiplexFact(split, tuple(pair), scope, binding.query,
                         result['supporting_layer_count'], unit)


def _condition(view: GraphView, fact: MultiplexFact, binding: CaseBinding,
               registry: RegistryContext) -> bool:
    """Check a source stratum independently of rendered reasoning and wording."""
    h, pair, condition = fact.support_count, fact.gene_ids, binding.condition
    if condition == 'outside':
        return h is None
    if condition == 'direction_invalid':
        return h is not None and any(not layer.directed for layer in view.layers.values())
    if h is None:
        return False
    if condition in ('multiple', 'one', 'zero', 'positive'):
        matches = {'multiple': h >= 2, 'one': h == 1, 'zero': h == 0, 'positive': h > 0}[condition]
        if matches and binding.variant in ('partial', 'terminal', 'recovered', 'exhausted'):
            runtime = MultiplexRuntime(view)
            first = runtime.resolve_scope(fact.layer_scope, fact.edge_query)[0]
            prefix = runtime.get_edge_layers(view.multiplex_id, list(pair),
                {'kind': 'layers', 'layer_ids': [first], 'layer_family': None}, fact.edge_query,
                'complete', None, evidence_id='selection_prefix')['result']
            # These reviewed branches retain an earlier positive observation;
            # a checked negative prefix has the same call plan but changes the
            # evidence scenario and cannot qualify their numbered bindings.
            return prefix['supporting_layer_count'] == 1
        return matches
    if condition == 'symbols':
        try:
            get_symbols(pair, registry)
        except ValueError:
            return False
        return h > 0
    if condition in ('forward_only', 'reverse_only'):
        reverse = MultiplexRuntime(view).get_edge_layers(view.multiplex_id, list(pair[::-1]),
            fact.layer_scope, 'source_to_target', 'complete', None, evidence_id='selection_reverse')['result']['supporting_layer_count']
        return (h > 0 and reverse == 0) if condition == 'forward_only' else (h == 0 and reverse > 0)
    if condition == 'reciprocal_one':
        return h == 1 and any(layer.directed and pair in layer.edges and pair[::-1] in layer.edges
                              for layer in view.layers.values())
    if condition == 'subset':
        total = MultiplexRuntime(view).get_edge_layers(view.multiplex_id, list(pair), ALL,
            'any_direction', 'complete', None, evidence_id='selection_all')['result']['supporting_layer_count']
        return h > 0 and total > h
    if condition == 'layer_absence':
        return h == 0 and bool(set(pair) - view.layers[fact.layer_scope['layer_ids'][0]].nodes)
    if condition == 'undirected_witness':
        return any(not layer.directed and tuple(sorted(pair)) in layer.edges for layer in view.layers.values())
    if condition == 'later_directed':
        return any(view.layers[layer].directed and
                   (pair in view.layers[layer].edges or pair[::-1] in view.layers[layer].edges)
                   for layer in view.layer_order[1:])
    raise ValueError('unknown reviewed source condition')


def _single_edge_context(row: dict, view: GraphView) -> None:
    """Use one native undirected witness without inventing all-layer coverage."""
    pair, meta = tuple(row['metadata']['gene_ids']), row['metadata']
    layer = next(layer for layer in view.layer_order if not view.layers[layer].directed
                 and tuple(sorted(pair)) in view.layers[layer].edges)
    arguments = {'multiplex_id': view.multiplex_id, 'layer_id': layer, 'gene_ids': list(pair)}
    response = view.get_edge(**arguments, evidence_id='ev_' + row['row_id'] + '_single')
    observations = [_observation('get_edge', arguments, response)]
    context = json.loads(row['messages'][1]['content'])
    context['observations'] = observations
    row['messages'][1]['content'] = canonical_json(context)
    batch = {'multiplex_id': view.multiplex_id, **{key: meta[key]
             for key in ('gene_ids', 'layer_scope', 'edge_query', 'mode', 'top_k')}}
    proofs, missing, _, provenance = _proofs(MultiplexRuntime(view), batch, observations)
    result = MultiplexRuntime(view).assemble_result(pair, meta['layer_scope'], meta['edge_query'],
        meta['mode'], meta['top_k'], proofs, missing)
    row['messages'][-1]['content'].update(result=result, provenance=provenance)
    row['messages'][-1]['thinking'] = _reason(result, meta['framing'])
    row['messages'][-1]['content']['summary'] = 'One attributed layer proves existence; the supporting-layer total remains unknown.'


def _fresh_identity(row: dict, *, retain_first: bool = False) -> None:
    """Move required registry acquisitions into the current turn before graph work."""
    context = json.loads(row['messages'][1]['content'])
    identities = context['observations']
    supplied, fresh = (identities[:1], identities[1:]) if retain_first else ([], identities)
    context['observations'] = supplied
    row['messages'][1]['content'] = canonical_json(context)
    middle = []
    for observation in fresh:
        middle.extend([
            {'role': 'assistant', 'thinking': f"Resolve the exact supplied symbol {observation['arguments']['gene_symbol']} before graph acquisition. Preserve its attributed canonical identity.",
             'tool_call': {'name': observation['operation'], 'arguments': deepcopy(observation['arguments'])}},
            {'role': 'tool', 'name': observation['operation'],
             'content': {key: deepcopy(observation[key]) for key in ('status', 'result', 'provenance')}},
        ])
    row['messages'][3:3] = middle
    # The original graph-call thought referred to supplied mappings. Describe
    # the actual visible acquisitions without changing its graph plan/target.
    for message in row['messages'][3 + len(middle):]:
        if 'tool_call' in message:
            message['thinking'] = (f"The registry evidence now resolves {row['metadata']['input_genes']} to {row['metadata']['gene_ids']} in input order. "
                                  'Identity alone does not prove multiplex membership. Acquire the requested scoped graph observation.')


def _tag(row: dict, case_id: str | None, behavior: str) -> dict:
    row['row_id'] = 's22prod_' + digest([row['row_id'], case_id, behavior])[:32]
    row['metadata'].update(catalog_case_id=case_id, production_behavior=behavior,
                           production_extension='s2_2_production_v1')
    return row


def _diagnostic_request(view: GraphView, pair: Sequence[str], kind: str) -> dict:
    request = {'genes': list(pair), 'multiplex_id': view.multiplex_id,
               'layer_scope': deepcopy(ALL), 'edge_query': 'any_direction', 'mode': 'complete', 'top_k': None}
    if kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_pair':
        request['genes'] = [pair[0], pair[0]]
    elif kind == 'invalid_identity':
        request['genes'] = [{'gene_id': pair[0]}, pair[1]]
    elif kind == 'missing_scope':
        request['layer_scope'] = None
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_s2_multiplex'
    elif kind in ('unknown_layer', 'tool_unknown_layer'):
        request['layer_scope'] = {'kind': 'layers', 'layer_ids': ['TFs:missing_context'], 'layer_family': None}
    elif kind == 'unknown_family':
        request['layer_scope']['layer_family'] = 'unknown_family'
    elif kind == 'invalid_selection':
        request.update(mode='top_k', top_k=0)
    elif kind == 'invalid_predicate':
        request['edge_query'] = {'direction': 'out'}
    else:
        raise ValueError('unsupported prerequisite diagnostic')
    return request


def _diagnostic_arguments(request: Mapping[str, Any]) -> dict:
    return {'gene_ids' if key == 'genes' else key: deepcopy(value) for key, value in request.items()}


def build_diagnostic(view: GraphView, pair: Sequence[str], kind: str, *, split: str,
                     contract: Mapping, bank: Mapping, variant_index: int = 0,
                     seed: int = 20261028) -> dict:
    """Generate literal malformed/scope rejections from the active resolver rules."""
    if _pair_group(tuple(pair), seed)[1] != split:
        raise ValueError('the diagnostic subject pair belongs to a different split')
    request = _diagnostic_request(view, pair, kind)
    arguments = _diagnostic_arguments(request)
    response = MultiplexRuntime(view).get_edge_layers(**arguments, evidence_id='unused_rejection')
    if response['provenance'] != {'registry': [], 'graph': []} or 'issue_code' not in response['result']:
        raise ValueError('the proposed diagnostic is not a real atomic rejection')
    lead = contract['_diagnostic_leads'][split]
    user, wrapper = render_s2_prompt(bank, split=split, framing='direct', variant_index=variant_index,
                                    request=lead + canonical_json(request))
    row_id = 's22diag_' + digest([view.graph_version, user, kind])[:32]
    active = _active(view)
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    result = {'issue_code': response['result']['issue_code'], 'request': request, 'partial_results': []}
    if kind == 'tool_unknown_layer':
        messages.extend([{'role': 'assistant', 'thinking': 'The visible layer catalogue is withheld. Ask the resolver for the explicit scope without substituting another layer.',
            'tool_call': {'name': 'get_edge_layers', 'arguments': arguments}},
            {'role': 'tool', 'name': 'get_edge_layers', 'content': response}])
        result['rejected_arguments'] = arguments
    messages.append({'role': 'assistant', 'thinking':
        f"The literal request is {request}. Its prerequisite fails with {result['issue_code']}. "
        'Preserve the raw request and any atomic rejected arguments. No graph observation was acquired, so do not infer edge absence, zero counts, or membership. Defer until the input is resolved.',
        'content': {'status': response['status'], 'provenance': {'registry': [], 'graph': []},
            'result': result, 'continuation_decision': 'defer',
            'blocked_reason': 'A valid pair, registered scope, and explicit selection are required.',
            'summary': 'The graph request has an unresolved prerequisite; no edge result was computed.'}})
    unit, _ = _pair_group(tuple(pair), seed)
    return _tag({'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': 'mentor-rl-world-model-s2-corpus-v1', 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.2', 'split': split, 'split_unit': unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'gene_ids': list(pair), 'request': request,
        'framing': 'direct', 'variant': kind, 'wrapper_index': wrapper, 'variant_index': variant_index,
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}, None, kind)


def build_case_rows(view: GraphView | None = None, contract: Mapping | None = None, *,
                    split: str = 'train', case_ids: Sequence[str] | None = None,
                    instances_per_case: int = 1, bank: Mapping | None = None,
                    registry: RegistryContext | None = None, seed: int = 20261028,
                    include_auxiliary: bool = True) -> list[dict]:
    """Generate distinct source-backed case witnesses without assigning production quotas.

    A scarce source stratum raises VariantUnavailable rather than silently
    dropping a case, changing its behavior, duplicating a row, or moving splits.
    """
    if view is None:
        view, default_contract, default_bank, default_registry = load_inputs()
        contract = default_contract if contract is None else contract
        bank = default_bank if bank is None else bank
        registry = default_registry if registry is None else registry
    contract = load_production_contract() if contract is None else contract
    bank = load_s2_prompt_banks(contract['prompt_bank']) if bank is None else bank
    registry = load_registry_context(contract['registry_index']) if registry is None else registry
    if split not in SPLITS or type(instances_per_case) is not int or instances_per_case < 1:
        raise ValueError('a source split and positive witness count are required')
    selected_ids = list(CASE_BINDINGS if case_ids is None else case_ids)
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(CASE_BINDINGS):
        raise ValueError('case IDs must be distinct reviewed S2.2 cases')
    facts, _ = select_facts(view, tuple(sorted(registry.symbols_by_id)),
                            dict.fromkeys(SPLITS, max(4, instances_per_case)), seed)
    pairs = sorted({fact.gene_ids for fact in facts if fact.split == split},
                   key=lambda pair: digest([seed, pair]))
    # Include both reference orders while retaining one unordered-pair split.
    pairs = list(dict.fromkeys(pair for source in pairs for pair in (source, source[::-1])))
    rows = []
    for case_id in selected_ids:
        binding = CASE_BINDINGS[case_id]
        matched = 0
        for pair in pairs:
            try:
                fact = _fact(view, pair, binding, seed)
            except VariantUnavailable:
                continue
            if not _condition(view, fact, binding, registry):
                continue
            variant = 'full_context' if binding.variant == 'single_edge_context' else binding.variant
            try:
                row = (build_direction_diagnostic(view, fact, binding.framing, matched, bank, contract, seed=seed)
                       if binding.condition == 'direction_invalid' else
                       build_row(view, fact, binding.mode, 1 if binding.mode == 'top_k' else None,
                           binding.framing, variant, matched, bank, contract, registry=registry, seed=seed))
            except VariantUnavailable:
                continue
            if binding.variant == 'single_edge_context':
                _single_edge_context(row, view)
            _tag(row, case_id, binding.variant)
            validate_case_row(row, view, bank=bank, contract=contract, registry=registry)
            rows.append(row)
            matched += 1
            if matched == instances_per_case:
                break
        if matched != instances_per_case:
            raise VariantUnavailable(f'{case_id}: only {matched} qualifying source witnesses in fixed {split} split')
    if include_auxiliary:
        symbol_source = next((row for row in rows if row['metadata']['catalog_case_id'] == 'S2.2.4'), None)
        if symbol_source is not None:
            for retain_first, behavior in ((False, 'fresh_identity'), (True, 'partial_identity')):
                row = deepcopy(symbol_source)
                _fresh_identity(row, retain_first=retain_first)
                _tag(row, None, behavior)
                validate_case_row(row, view, bank=bank, contract=contract, registry=registry)
                rows.append(row)
        base = next((pair for pair in pairs if set(pair) <= view.multiplex_nodes), None)
        if base is None:
            raise VariantUnavailable('no globally present diagnostic pair in the fixed split')
        for kind in DIAGNOSTICS:
            row = build_diagnostic(view, base, kind, split=split, contract=contract, bank=bank, seed=seed)
            validate_case_row(row, view, bank=bank, contract=contract, registry=registry)
            rows.append(row)
    return rows


def validate_case_row(row: Mapping, view: GraphView, *, bank: Mapping | None = None,
                      contract: Mapping | None = None, registry: RegistryContext | None = None) -> None:
    """Replay actual visible acquisitions and separately qualify numbered behavior.

    Fresh identity acquisitions are validated against the complete registry and
    their chronology before projecting them into the historical graph verifier.
    This projection changes no observation or target and cannot hide a graph
    call made before identity resolution.
    """
    from ..world_model_training import _validate_v3_message_sequence
    contract = load_production_contract() if contract is None else contract
    bank = load_s2_prompt_banks(contract['prompt_bank']) if bank is None else bank
    registry = load_registry_context(contract['registry_index']) if registry is None else registry
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if (meta.get('production_extension') != 's2_2_production_v1'
            or meta.get('tool_contract_version') != contract['contract_version']
            or meta.get('system_prompt_sha256') != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or messages[0] != {'role': 'system', 'content': contract['system_prompt']}
            or row['tools'] != contract['tools']):
        raise ValueError('the isolated production contract identity changed')
    behavior, case_id = meta['production_behavior'], meta['catalog_case_id']
    if case_id is not None and case_id not in CASE_BINDINGS:
        raise ValueError('unknown numbered S2.2 case binding')
    if case_id is None and behavior not in (*DIAGNOSTICS, 'fresh_identity', 'partial_identity'):
        raise ValueError('unknown auxiliary production behavior')
    if (_pair_group(tuple(meta['gene_ids']), meta['selection_seed']) != (meta['split_unit'], meta['split'])
            or meta['graph_version'] != view.graph_version or meta['question_family'] != 'S2.2'):
        raise ValueError('the subject split or active source identity changed')
    answer = messages[-1]['content']
    if (list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']
            or (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason'])):
        raise ValueError('the final structured envelope or defer blocker changed')
    for message in messages:
        fields = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                  {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                  {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != fields or message['role'] == 'assistant' and not message.get('thinking', '').strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if behavior in DIAGNOSTICS:
        if case_id is not None or meta['request'] != _diagnostic_request(view, meta['gene_ids'], behavior):
            raise ValueError('a prerequisite diagnostic changed its literal request or claims a numbered binding')
        request = meta['request']
        arguments = _diagnostic_arguments(request)
        rejection = MultiplexRuntime(view).get_edge_layers(**arguments, evidence_id='unused_rejection')
        active = _active(view)
        tool_rejection = behavior == 'tool_unknown_layer'
        if tool_rejection:
            active.pop('layers')
        if json.loads(messages[1]['content']) != {'active_graph': active, 'observations': []}:
            raise ValueError('a prerequisite rejection changed its visible catalogue or invented observations')
        expected_user = bank[meta['split']]['direct'][meta['wrapper_index']].format(
            request=contract['_diagnostic_leads'][meta['split']] + canonical_json(request))
        if messages[2]['content'] != expected_user:
            raise ValueError('the literal diagnostic inputs are not present in the visible request')
        expected_result = {'issue_code': rejection['result']['issue_code'], 'request': request, 'partial_results': []}
        if tool_rejection:
            expected_result['rejected_arguments'] = arguments
            if (len(messages) != 6 or messages[3].get('tool_call') != {'name': 'get_edge_layers', 'arguments': arguments}
                    or messages[4] != {'role': 'tool', 'name': 'get_edge_layers', 'content': rejection}):
                raise ValueError('the withheld-catalogue request lost its exact resolver rejection')
        elif len(messages) != 4:
            raise ValueError('a locally evident request rejection made an unnecessary acquisition')
        if (answer['status'] != rejection['status'] or answer['provenance'] != {'registry': [], 'graph': []}
                or answer['result'] != expected_result or answer['continuation_decision'] != 'defer'):
            raise ValueError('diagnostic invents graph output or loses the rejected request')
        return
    projected = deepcopy(row)
    if behavior in ('fresh_identity', 'partial_identity'):
        if case_id is not None:
            raise ValueError('fresh identity is auxiliary, not supplied-context case S2.2.4')
        context = json.loads(messages[1]['content'])
        identities = list(context['observations'])
        if len(identities) != (1 if behavior == 'partial_identity' else 0):
            raise ValueError('the fresh identity variant supplied the wrong prerequisite count')
        validate_identity_observations(identities, registry)
        cursor = 3
        for gene_id, symbol in zip(meta['gene_ids'], meta['input_genes']):
            if any(o['arguments']['gene_symbol'] == symbol and o['result']['gene_id'] == gene_id for o in identities):
                continue
            if cursor + 1 >= len(messages) or messages[cursor].get('tool_call') != {
                    'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': symbol}}:
                raise ValueError('a graph call preceded required identity resolution or symbol order changed')
            response = messages[cursor + 1]
            if (response.get('name') != 'lookup_human_gene_symbol' or response.get('role') != 'tool'
                    or set(response['content']) != {'status', 'result', 'provenance'}):
                raise ValueError('a registry response is detached from its exact call')
            observation = _observation('lookup_human_gene_symbol', {'gene_symbol': symbol}, response['content'])
            validate_identity_observations([observation], registry)
            if observation['result']['gene_id'] != gene_id:
                raise ValueError('the acquired symbol changes the canonical reference order')
            identities.append(observation)
            cursor += 2
        context['observations'] = identities
        projected['messages'][1]['content'] = canonical_json(context)
        projected['messages'] = projected['messages'][:3] + projected['messages'][cursor:]
    validate_row(projected, view, bank=bank, contract=contract, registry=registry)
    if case_id is not None:
        binding = CASE_BINDINGS[case_id]
        fact = _fact(view, meta['gene_ids'], binding, meta['selection_seed'])
        expected_variant = ('direction_invalid' if binding.condition == 'direction_invalid' else
                            'full_context' if binding.variant == 'single_edge_context' else binding.variant)
        expected_status = {'S2.2.5': 'not_found', 'S2.2.15': 'error', 'S2.2.17': 'error',
                           'S2.2.19': 'invalid_input'}.get(case_id, 'resolved')
        if (not _condition(view, fact, binding, registry)
                or meta['layer_scope'] != fact.layer_scope or meta['mode'] != binding.mode
                or meta['edge_query'] != binding.query or meta['framing'] != binding.framing
                or behavior != binding.variant or meta['variant'] != expected_variant
                or answer['status'] != expected_status):
            raise ValueError('the numbered binding changed its reviewed source condition, scope, or task')
        observations = json.loads(messages[1]['content'])['observations']
        calls = [message['tool_call'] for message in messages if 'tool_call' in message]
        variant = binding.variant
        if variant == 'single_edge_context':
            if (len(observations) != 1 or observations[0]['operation'] != 'get_edge'
                    or observations[0]['result']['directed'] or not observations[0]['result']['edge_exists'] or calls):
                raise ValueError('case S2.2.8 needs exactly one positive undirected native observation')
        elif variant == 'later_witness':
            if (len(observations) != 1 or observations[0]['operation'] != 'get_edge'
                    or not observations[0]['result']['directed'] or calls):
                raise ValueError('case S2.2.22 needs one later directed witness without acquisition')
        elif variant == 'symbol_context' and len(observations) != 2:
            raise ValueError('case S2.2.4 needs both supplied registry mappings')
        elif variant in ('partial', 'terminal', 'recovered', 'exhausted', 'bounded_context') and len(observations) != 1:
            raise ValueError('the reviewed remaining-scope branch needs its one independent saved batch')
        elif variant == 'lookup' and observations:
            raise ValueError('a fresh reviewed scan cannot be replaced by supplied graph evidence')
        if variant in ('partial', 'bounded_context', 'terminal') and len(calls) != 1:
            raise ValueError('the reviewed remaining-scope behavior needs exactly one required scan')
        if variant in ('recovered', 'exhausted') and len(calls) != 2:
            raise ValueError('the reviewed retry behavior needs the failed call and its authorized retry')
        if variant in ('partial', 'terminal', 'recovered', 'exhausted'):
            saved = observations[0]
            selected = MultiplexRuntime(view).resolve_scope(fact.layer_scope, fact.edge_query)
            subset = {'kind': 'layers', 'layer_ids': [selected[0]], 'layer_family': None}
            if (saved['operation'] != 'get_edge_layers' or saved['arguments']['layer_scope'] != subset
                    or saved['arguments']['mode'] != 'complete' or saved['arguments']['top_k'] is not None
                    or saved['result']['coverage']['unchecked_layer_ids']
                    or saved['result']['collection']['complete'] is not True
                    or saved['result']['supporting_layer_count'] != 1):
                raise ValueError('the reviewed prefix requires a complete positive one-layer subset acquisition')
        if variant == 'bounded_context':
            saved = observations[0]
            if (saved['operation'] != 'get_edge_layers' or saved['arguments']['layer_scope'] != fact.layer_scope
                    or saved['arguments']['mode'] != 'top_k' or saved['arguments']['top_k'] != 1
                    or not saved['result']['coverage']['unchecked_layer_ids']
                    or saved['result']['collection']['mode'] != 'top_k'
                    or saved['result']['collection']['returned_count'] != 1
                    or saved['result']['collection']['complete'] is not False
                    or saved['result']['collection']['total_count'] is not None):
                raise ValueError('the reviewed bounded record requires a genuine incomplete full-scope top-k acquisition')
        if variant in ('reversed_context',) and calls:
            raise ValueError('complete saved symmetric evidence cannot require a new graph call')
        if variant == 'reversed_context':
            if (len(observations) != 1 or observations[0]['operation'] != 'get_edge_layers'
                    or observations[0]['arguments']['gene_ids'] != list(fact.gene_ids[::-1])
                    or observations[0]['arguments']['mode'] != 'complete'
                    or observations[0]['arguments']['layer_scope'] != fact.layer_scope):
                raise ValueError('the reviewed reversed-reference case requires its complete reversed-pair acquisition')
