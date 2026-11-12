"""Build exact component partitions from visible native graph evidence.

Source labels validate construction only. A complete component collection can
follow from partial edges when every possible completion gives that partition.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_2_identity import RegistryContext, get_symbols, validate_identity_observations
from .s2_3_corpus import gene_group, _provenance
from .s2_9_corpus import (_active_context, _identity_rejection, _identity_partial,
    build_row as _graph_context_row, load_build_contract as _graph_contract, _transport as _graph_transport)
from .s2_9_sampling import fact_from_genes as graph_fact
from .s2_10_sampling import ComponentFact, fact_from_genes
from .s2_components import components, validate_connectivity
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphIndex, SubgraphRuntime
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_10_v1.json')
VARIANTS = ('lookup', 'reordered', 'duplicate_inputs', 'unqualified', 'context', 'duplicate_context',
    'superset_context', 'bridge_context', 'outside_route_context', 'subset_context', 'disjoint_context',
    'membership_context', 'membership_first', 'membership_gate', 'gate_context', 'native_context',
    'native_in_context', 'partial_native_context', 'bounded_context', 'spanning_context',
    'strong_witness_context', 'weak_witness_strong_context', 'insufficient_context', 'prefix_context',
    'final_page_only', 'paged', 'overlap', 'empty_page', 'unknown_total', 'symbol_context', 'symbol_terminal',
    'stale', 'wrong_layer', 'distractor', 'terminal', 'recovered', 'exhausted',
    'page_terminal', 'page_recovered', 'page_exhausted',
    'membership_terminal', 'membership_recovered', 'membership_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope', 'invalid_list',
               'missing_kind', 'invalid_kind', 'unavailable_kind', 'unresolved_identity', 'tool_unknown_layer')
LABELS = ('category_counts', 'edge_count', 'components', 'component_count', 'component_sizes',
          'single_component', 'isolated_count', 'loop_count', 'reciprocal_pairs')


def load_build_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load the isolated S2.10 overlay and frozen request bank."""
    from .s2_10_prompts import load_s2_10_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.10']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.10 build contract')
    contract['_requests'] = load_s2_10_requests(contract['request_bank'])
    pin = contract['context_source_contract']
    if file_sha256(ROOT / pin['path']) != pin['sha256']:
        raise ValueError('the saved-context source contract differs from its pin')
    return contract


def _arguments(view, fact, kind):
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
            'gene_ids': list(fact.gene_ids), 'connectivity_kind': kind}


def _request(arguments, genes, gate):
    return {'genes': list(genes), 'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
            'connectivity_kind': arguments['connectivity_kind'], 'require_all_in_layer': gate}


def _blocked(request, code, used, coverage, rejected=None):
    return {'issue_code': code, 'request': deepcopy(request), 'rejected_arguments': rejected,
            'partial_results': list(used), 'coverage': coverage}


def _plan(view, arguments, observations, gate, membership_first):
    from .s2_10_evidence import collect
    proof = collect(view, arguments, observations, require_all_in_layer=gate, membership_first=membership_first)
    identities = [item for item in observations if item['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _transport(fact, variant):
    source = 'prefix_context' if variant.startswith('page_') else variant
    return _graph_transport(fact, source)


def _bridge(view, index, fact, seed):
    """Find a real same-split intermediate that the outer request did not select."""
    if len(fact.gene_ids) != 2 or fact.category_counts[0] != 2 or fact.connectivity_kind == 'strong':
        raise VariantUnavailable('this outside-route contrast needs two undirected or weak members')
    sides = index.native_index.directions(fact.layer_id)
    neighbors = [{gene for side in sides for gene, _ in index.native_index.neighbors(fact.layer_id, query, side)}
                 for query in fact.gene_ids]
    pool = (neighbors[0] & neighbors[1]) - set(fact.gene_ids)
    candidates = [gene for gene in pool if gene_group(gene, seed)[1] == fact.split]
    if not candidates or fact.component_count != 2:
        raise VariantUnavailable('no separating source bridge exists for this selected set')
    return min(candidates, key=lambda gene: digest([seed, fact.split_unit, 'bridge', gene]))


def _reason(result, framing):
    text = (f"The requested kind is {result['connectivity_kind']} and native directedness is {result['directed']}. "
        f"The selected layer members are {result['layer_gene_ids']}. Their exact partition is {result['components']}. "
        f"There are {result['component_count']} groups. Layer exclusions are {result['not_in_layer_gene_ids']}; "
        f"global exclusions are {result['missing_gene_ids']}. "
        f"The all-input single_component value is {result['single_component']}. "
        'It requires one projected group and no excluded requested inputs. An exclusion can make it false without splitting that group. '
        'The final collection counts component groups, not acquired edge rows. Original graph pages keep their own completeness metadata. ')
    if result['layer_gene_count'] <= 1:
        text += 'Complete membership alone determines this zero- or one-member partition; no self-loop lookup is needed. '
    else:
        text += ('The visible positive partition agrees with every completion permitted by the checked graph evidence. '
                 'A spanning or strongly connected witness can suffice before all edges arrive; partial splits alone do not suffice. ')
    text += 'Use paths only through selected nodes. Recompute from restricted edges after removing a possible bridge. '
    if result['connectivity_kind'] == 'strong':
        text += 'Separate strong groups lack mutual reachability; a one-way path between them can still exist. '
    else:
        text += 'Ignoring arrows for weak connectivity does not change native directed evidence or combine its weights. '
    return text + ('The requested workflow step is complete, so continue.' if framing == 'step'
                   else 'The standalone request is complete, so stop.')


def build_row(view: GraphView, fact: ComponentFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, index: SubgraphIndex,
              registry: RegistryContext | None = None, question_kind: str = 'partition',
              require_all_in_layer: bool = False, membership_first: bool = False,
              seed: int = 20261028) -> dict:
    """Build a complete component answer, explicit gate failure, or backend deferral."""
    from .s2_10_prompts import render_s2_10_request
    if variant not in VARIANTS or framing not in FRAMINGS or question_kind not in ('partition', 'single_component'):
        raise ValueError('unsupported component contrast, framing, or question kind')
    if type(require_all_in_layer) is not bool or type(membership_first) is not bool:
        raise ValueError('gate and acquisition policy must be explicit booleans')
    if variant == 'reordered':
        fact = fact_from_genes(view, index, fact.layer_id, list(reversed(fact.gene_ids)), fact.connectivity_kind,
                               seed, split=fact.split, category=fact.category)
    if variant in ('membership_gate', 'gate_context'):
        require_all_in_layer = True
    if variant in ('membership_first', 'membership_terminal', 'membership_recovered', 'membership_exhausted'):
        membership_first = True
    raw_kind = fact.connectivity_kind
    if variant == 'unqualified':
        if view.layers[fact.layer_id].directed:
            raise VariantUnavailable('directed component requests require an explicit kind')
        raw_kind = None
    arguments = _arguments(view, fact, raw_kind)
    validate_connectivity(view, fact.layer_id, raw_kind)
    raw_genes = list(fact.gene_ids)
    if variant == 'duplicate_inputs':
        if not raw_genes:
            raise VariantUnavailable('duplicate inputs require a nonempty request')
        raw_genes += [raw_genes[0], raw_genes[-1]]
    if variant in ('symbol_context', 'symbol_terminal'):
        if registry is None:
            raise VariantUnavailable('supplied symbols require the pinned registry')
        try:
            raw_genes = list(get_symbols(fact.gene_ids, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = _request(arguments, raw_genes, require_all_in_layer)
    core = render_s2_10_request(contract['_requests'], split=fact.split, variant_index=variant_index,
                                question_kind=question_kind, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's210_' + digest([contract['contract_version'], view.graph_version, asdict(fact),
                              framing, variant, question_kind, membership_first, user])[:32]
    transport = _transport(fact, variant)
    runtime, nodes = SubgraphRuntime(view, index=index, **transport), NodesByLayerRuntime(view)
    observations, serial = [], 0

    def evidence_id():
        nonlocal serial
        serial += 1
        return f'ev_{row_id}_{serial}'

    basic = {'context', 'duplicate_context', 'superset_context', 'subset_context', 'disjoint_context',
        'membership_context', 'native_context', 'native_in_context', 'partial_native_context', 'bounded_context',
        'prefix_context', 'final_page_only', 'stale', 'wrong_layer', 'distractor', 'symbol_context', 'symbol_terminal', 'gate_context'}
    if variant in basic or variant.startswith('page_'):
        source_variant = ('prefix_context' if variant.startswith('page_') else
                          'symbol_context' if variant == 'symbol_terminal' else variant)
        previous = _graph_context_row(view, graph_fact(view, index, fact.layer_id, fact.gene_ids, seed,
            split=fact.split, category=fact.category), framing, source_variant, variant_index,
            bank, _graph_contract(ROOT / contract['context_source_contract']['path']),
            index=index, registry=registry, seed=seed)
        observations = json.loads(previous['messages'][1]['content'])['observations']
    elif variant in ('spanning_context', 'strong_witness_context', 'weak_witness_strong_context', 'insufficient_context'):
        edges = index.edges(fact.layer_id, fact.gene_ids)
        members = index.partition(fact.layer_id, fact.gene_ids)['layer_gene_ids']
        if len(members) < 2 or len(edges) < 2:
            raise VariantUnavailable('this contrast needs a genuine multi-member edge prefix')
        if variant == 'strong_witness_context' and fact.connectivity_kind != 'strong':
            raise VariantUnavailable('a strong witness contrast requires strong connectivity')
        if variant == 'spanning_context' and fact.connectivity_kind == 'strong':
            raise VariantUnavailable('a spanning contrast uses undirected or weak connectivity')
        if variant == 'weak_witness_strong_context' and fact.connectivity_kind != 'strong':
            raise VariantUnavailable('weak evidence contrasts with a requested strong partition')
        limit = None
        for k in range(1, len(edges)):
            current = components(members, edges[:k], fact.connectivity_kind)
            if variant == 'insufficient_context':
                choose = len(current) > 1
            elif variant == 'weak_witness_strong_context':
                choose = len(components(members, edges[:k], 'weak')) == 1 and len(current) > 1
            else:
                choose = len(current) == 1
            if choose:
                limit = k
                break
        if limit is None:
            raise VariantUnavailable('no qualifying proper source prefix exists')
        source = runtime.arguments(list(fact.gene_ids), fact.layer_id, mode='top_k', top_k=limit)
        observations.append(_observation('induce_subgraph', source, runtime.induce_subgraph(**source, evidence_id=evidence_id())))
    elif variant in ('bridge_context', 'outside_route_context'):
        bridge = _bridge(view, index, fact, seed)
        if variant == 'bridge_context':
            source = runtime.arguments([*fact.gene_ids, bridge], fact.layer_id)
            observations.append(_observation('induce_subgraph', source, runtime.induce_subgraph(**source, evidence_id=evidence_id())))
        else:
            if view.layers[fact.layer_id].directed:
                raise VariantUnavailable('this outside-route acquisition uses undirected edge records')
            for gene in fact.gene_ids:
                source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': [bridge, gene]}
                observations.append(_observation('get_edge', source, view.get_edge(**source, evidence_id=evidence_id())))
    proof, _ = _plan(view, arguments, observations, require_all_in_layer, membership_first)
    if variant in ('insufficient_context', 'weak_witness_strong_context') and proof.next_arguments is None:
        raise VariantUnavailable('this source prefix already proves the exact component partition')
    if variant in ('spanning_context', 'strong_witness_context') and proof.result is None:
        raise VariantUnavailable('the partial witness does not yet establish every required fact')
    failure_kind = variant.removeprefix('page_').removeprefix('membership_').removeprefix('symbol_')
    if variant.startswith('membership_') and failure_kind in ('terminal', 'recovered', 'exhausted') and proof.next_tool != 'get_nodes_by_layer':
        raise VariantUnavailable('the selected membership acquisition is unnecessary')
    context = {'active_graph': _active_context(view), 'observations': deepcopy(observations)}
    if membership_first:
        context['acquisition_policy'] = 'membership_first'
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    injected, failure = False, None
    while True:
        proof, used = _plan(view, arguments, observations, require_all_in_layer, membership_first)
        if proof.next_arguments is None:
            break
        pending, operation = proof.next_arguments, proof.next_tool
        thought = (f"The {fact.connectivity_kind} request needs the full selected-member partition for {list(fact.gene_ids)}. "
            'The current observations leave membership or possible component mergers unresolved. '
            'Acquire the required native evidence, with the complete input set fixed during edge pages. '
            'A provisional split or an excluded input does not supply the remaining exact partition.')

        def append(response, reasoning):
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                'tool_call': {'name': operation, 'arguments': deepcopy(pending)}},
                {'role': 'tool', 'name': operation, 'content': response}])

        execute = nodes.get_nodes_by_layer if operation == 'get_nodes_by_layer' else runtime.induce_subgraph
        should_fail = (not injected and failure_kind in ('terminal', 'recovered', 'exhausted')
            and (not variant.startswith('page_') or pending.get('cursor') is not None))
        if should_fail:
            injected = True
            response = _failure(None if failure_kind == 'terminal' else 1)
            append(response, thought)
            if failure_kind != 'terminal':
                response = _failure(0) if failure_kind == 'exhausted' else execute(**pending, evidence_id=evidence_id())
                append(response, 'The visible error supplies one retry. Repeat the identical tool and every argument, including the current cursor. '
                    'Retain successful partials and stop after recovery or exhaustion. Failed calls add no evidence.')
            if response['status'] == 'error':
                failure = operation, pending, response
                break
        else:
            response = execute(**pending, evidence_id=evidence_id())
            append(response, thought)
        observations.append(_observation(operation, pending, response))
    if failure_kind in ('terminal', 'recovered', 'exhausted') and not injected:
        raise VariantUnavailable('a sufficient component proof settles the selected failure first')
    if failure:
        operation, pending, response = failure
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': {'failed_tool': operation,
            'error_code': response['error_code'], 'arguments': pending, 'partial_results': used, 'coverage': proof.coverage},
            'continuation_decision': 'defer', 'blocked_reason': 'The required backend must recover before the remaining exact partition can be proved.',
            'summary': 'A required acquisition failed. Retain supported membership and native edge observations without guessed components.'}
        thought = 'No recovery allowance remains for the required call. Preserve every successful partial and truthful membership coverage. '
        thought += 'Even fully checked membership does not justify guessed component groups. Defer for backend recovery.'
    elif proof.issue_code:
        answer = {'status': 'not_found', 'provenance': _provenance(used),
            'result': _blocked(request, proof.issue_code, used, proof.coverage), 'continuation_decision': 'defer',
            'blocked_reason': 'The explicit all-members prerequisite fails; revise the requested set.',
            'summary': 'The complete membership partition fails the explicit gate. Retain it and defer before unnecessary graph retrieval.'}
        thought = 'The user explicitly requires all requested inputs to be layer members, and the complete classification excludes at least one. '
        thought += 'Preserve that partition and defer. Default component projection would not impose this prerequisite.'
    else:
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
            'summary': 'The full exact component partition and all-input Boolean follow from the visible compatible evidence.'}
        thought = ('The supplied evidence is sufficient; no new call is needed. ' if len(messages) == 3 else '') + _reason(proof.result, framing)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.10', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'connectivity_kind': fact.connectivity_kind, **{key: getattr(fact, key) for key in LABELS},
        'category': fact.category, 'framing': framing, 'variant': variant, 'question_kind': question_kind,
        'request': request, 'membership_first': membership_first, 'transport': transport, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][question_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view, fact, kind):
    request = {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
        'layer_id': fact.layer_id, 'connectivity_kind': None, 'require_all_in_layer': False}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'missing_genes': 'MISSING_GENES',
        'missing_scope': 'MISSING_SCOPE', 'invalid_list': 'INVALID_GENE_LIST',
        'missing_kind': 'CONNECTIVITY_KIND_REQUIRED', 'invalid_kind': 'INVALID_CONNECTIVITY_KIND',
        'unavailable_kind': 'CONNECTIVITY_KIND_UNAVAILABLE', 'unresolved_identity': 'UNRESOLVED_IDENTITY'}
    if kind not in codes:
        raise ValueError('unsupported component prerequisite diagnostic')
    if kind in ('unknown_layer', 'tool_unknown_layer'):
        request['layer_id'] = 'TFs:missing_context'
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_component_multiplex'
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'invalid_list':
        request['genes'] = fact.gene_ids[0] if fact.gene_ids else {'invalid': 'gene list'}
    elif kind == 'missing_kind':
        directed = next((layer for layer in view.layer_order if view.layers[layer].directed), None)
        if directed is None:
            raise VariantUnavailable('no directed representation needs a connectivity choice')
        request['layer_id'] = directed
    elif kind == 'invalid_kind':
        request['connectivity_kind'] = {'kind': 'weak'}
    elif kind == 'unavailable_kind':
        request['connectivity_kind'] = 'undirected' if view.layers[fact.layer_id].directed else 'weak'
    elif kind == 'unresolved_identity':
        request['genes'] = ['HER2', *list(fact.gene_ids[:1])]
        request['connectivity_kind'] = fact.connectivity_kind
    status = 'not_found' if kind in ('unknown_layer', 'tool_unknown_layer', 'unknown_multiplex', 'unresolved_identity') else 'invalid_input'
    return request, status, codes[kind]


def build_diagnostic(view: GraphView, fact: ComponentFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'missing_genes', registry: RegistryContext | None = None,
                     seed: int = 20261028) -> dict:
    """Preserve unresolved identities, raw kinds, and scopes without invented groups."""
    from .s2_10_prompts import render_s2_10_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    wording_kind = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    core = render_s2_10_diagnostic_request(contract['_requests'], split=fact.split, kind=wording_kind,
        variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's210_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32]
    active = _active_context(view)
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    used, rejected = [], None
    if kind == 'tool_unknown_layer':
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
            'gene_ids': request['genes'], 'mode': 'complete', 'top_k': None, 'cursor': None}
        response = SubgraphRuntime(view).induce_subgraph(**rejected, evidence_id='unused_rejected_component_scope')
        messages.extend([{'role': 'assistant', 'thinking':
            'The context withholds the layer catalogue. Ask the resolver for the requested scope before assuming a native connectivity kind.',
            'tool_call': {'name': 'induce_subgraph', 'arguments': deepcopy(rejected)}},
            {'role': 'tool', 'name': 'induce_subgraph', 'content': response}])
    elif kind == 'unresolved_identity':
        if registry is None:
            raise ValueError('the unresolved identity requires the pinned registry')
        symbol = request['genes'][0]
        response = _identity_rejection(registry, symbol, f'ev_{row_id}_identity')
        messages.extend([{'role': 'assistant', 'thinking':
            'A symbol lacks authoritative identity context. Resolve its exact raw spelling before establishing the complete canonical input domain.',
            'tool_call': {'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': symbol}}},
            {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}])
        used.append(_identity_partial(symbol, response))
    answer = {'status': status, 'provenance': _provenance(used), 'result': _blocked(request, code, used, None, rejected),
        'continuation_decision': 'defer', 'blocked_reason': 'The identity, graph scope, input list, or required connectivity kind is unresolved.',
        'summary': 'A request prerequisite is unresolved. No component partition or graph-absence conclusion was established.'}
    messages.append({'role': 'assistant', 'thinking': f"The raw request is {request}. Its prerequisite fails with {code}. "
        'Retain the exact original kind, including null or invalid JSON. Do not invent a directed default or excluded graph node. '
        'Preserve registry evidence and rejected arguments, then defer.', 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.10', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'connectivity_kind': fact.connectivity_kind, **{key: None for key in LABELS},
        'category': fact.category, 'framing': framing, 'variant': kind, 'question_kind': 'partition',
        'request': request, 'membership_first': False, 'transport': None, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: SubgraphIndex | None = None) -> None:
    """Replay every source acquisition, then grade only the visible exact proof."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_10_prompts import render_s2_10_request, render_s2_10_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.10 requires one user turn and explicit harness context')
    for message in messages:
        expected = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected or message['role'] == 'assistant' and (not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('unexpected fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the component overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.10'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('component row identity differs from its build contract')
    index = index or SubgraphIndex(view)
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['connectivity_kind'],
        meta['selection_seed'], split=meta['split'], category=meta['category'])
    if fact.split_unit != meta['split_unit'] or list(fact.gene_ids) != meta['gene_ids']:
        raise ValueError('the canonical domain differs from its fixed subject split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if ((answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason'])
            or answer['blocked_reason'] is not None and (not isinstance(answer['blocked_reason'], str) or not answer['blocked_reason'].strip())):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if (meta['framing'] not in FRAMINGS or meta['question_kind'] not in ('partition', 'single_component')
            or not diagnostic and meta['variant'] not in VARIANTS or type(meta['membership_first']) is not bool):
        raise ValueError('unsupported framing, question, variant, or acquisition policy')
    request = meta['request']
    if (set(request) != {'genes', 'multiplex_id', 'layer_id', 'connectivity_kind', 'require_all_in_layer'}
            or type(request['require_all_in_layer']) is not bool):
        raise ValueError('the raw request fields or membership gate changed')
    wording_kind = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
    templates = (contract['_requests']['diagnostic_requests'][meta['split']][wording_kind] if diagnostic
                 else contract['_requests'][meta['split']][meta['question_kind']])
    if any(type(meta[key]) is not int or not 0 <= meta[key] < size for key, size in (
            ('wrapper_index', len(bank[meta['split']][meta['framing']])), ('request_template_index', len(templates)))):
        raise ValueError('wrapper and request indices must select their audited banks')
    if diagnostic:
        expected_request, status, code = _diagnostic_request(view, fact, meta['variant'])
        if canonical_json(request) != canonical_json(expected_request):
            raise ValueError('the diagnostic changes the literal request')
        core = render_s2_10_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording_kind,
            variant_index=meta['request_template_index'], **request)
    else:
        if any(canonical_json(getattr(fact, key)) != canonical_json(meta[key]) for key in LABELS):
            raise ValueError('private source labels differ from the native partition')
        if (meta['variant'] == 'unqualified' and request['connectivity_kind'] is not None
                or meta['variant'] != 'unqualified' and request['connectivity_kind'] != fact.connectivity_kind):
            raise ValueError('the variant does not match the visible connectivity choice')
        if (request['multiplex_id'] != view.multiplex_id or request['layer_id'] != fact.layer_id
                or validate_connectivity(view, fact.layer_id, request['connectivity_kind']) != fact.connectivity_kind):
            raise ValueError('the raw scope or requested interpretation changed')
        expected_genes = list(fact.gene_ids)
        if meta['variant'] in ('symbol_context', 'symbol_terminal'):
            if registry is None:
                raise ValueError('supplied symbols need the pinned registry')
            expected_genes = list(get_symbols(fact.gene_ids, registry))
        elif meta['variant'] == 'duplicate_inputs':
            if not expected_genes:
                raise ValueError('duplicate inputs require a nonempty domain')
            expected_genes += [expected_genes[0], expected_genes[-1]]
        if request['genes'] != expected_genes or meta['transport'] != _transport(fact, meta['variant']):
            raise ValueError('the raw identities, order, duplicates, or transport changed')
        if meta['variant'] in ('membership_gate', 'gate_context') and request['require_all_in_layer'] is not True:
            raise ValueError('the explicit gate contrast drops its prerequisite')
        if meta['variant'] in ('membership_first', 'membership_terminal', 'membership_recovered', 'membership_exhausted') and not meta['membership_first']:
            raise ValueError('the classification-first contrast drops its visible policy')
        core = render_s2_10_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'],
            question_kind=meta['question_kind'], **request)
        arguments = _arguments(view, fact, request['connectivity_kind'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from audited wording or framing')
    context = json.loads(messages[1]['content'])
    active = _active_context(view)
    if meta['variant'] == 'tool_unknown_layer':
        active.pop('layers')
    expected_fields = {'active_graph', 'observations'} | ({'acquisition_policy'} if meta['membership_first'] else set())
    if (set(context) != expected_fields or context['active_graph'] != active
            or meta['membership_first'] and context['acquisition_policy'] != 'membership_first'):
        raise ValueError('the active graph or visible acquisition policy changed')
    if not diagnostic:
        context_variants = {'context', 'duplicate_context', 'superset_context', 'bridge_context', 'outside_route_context',
            'subset_context', 'disjoint_context', 'membership_context', 'gate_context', 'native_context',
            'native_in_context', 'partial_native_context', 'bounded_context', 'spanning_context',
            'strong_witness_context', 'weak_witness_strong_context', 'insufficient_context',
            'prefix_context', 'final_page_only', 'symbol_context', 'symbol_terminal', 'stale',
            'wrong_layer', 'distractor', 'page_terminal', 'page_recovered', 'page_exhausted'}
        if (fact.gene_ids and meta['variant'] in context_variants and not context['observations']
                or meta['variant'] not in context_variants and context['observations']):
            raise ValueError('the variant does not match the supplied evidence context')
        if (meta['variant'] == 'duplicate_context' and fact.gene_ids
                and len({canonical_json(item) for item in context['observations']}) == len(context['observations'])):
            raise ValueError('the duplicate-context contrast lacks a repeated observation')
    if diagnostic:
        if (context['observations'] or meta['transport'] is not None or meta['membership_first']
                or meta['question_kind'] != 'partition' or any(meta[key] is not None for key in LABELS)):
            raise ValueError('malformed prerequisite diagnostic')
        used, rejected, middle = [], None, []
        if meta['variant'] == 'tool_unknown_layer':
            rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
                'gene_ids': request['genes'], 'mode': 'complete', 'top_k': None, 'cursor': None}
            response = SubgraphRuntime(view).induce_subgraph(**rejected, evidence_id='unused_rejected_component_scope')
            middle = [{'role': 'assistant', 'tool_call': {'name': 'induce_subgraph', 'arguments': rejected}},
                      {'role': 'tool', 'name': 'induce_subgraph', 'content': response}]
        elif meta['variant'] == 'unresolved_identity':
            if registry is None:
                raise ValueError('unresolved identity requires the pinned registry')
            response = _identity_rejection(registry, request['genes'][0], f"ev_{row['row_id']}_identity")
            used.append(_identity_partial(request['genes'][0], response))
            middle = [{'role': 'assistant', 'tool_call': {'name': 'lookup_human_gene_symbol',
                'arguments': {'gene_symbol': request['genes'][0]}}},
                {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}]
        if len(messages) != 4 + len(middle):
            raise ValueError('the prerequisite adds or drops a required call')
        for actual, expected in zip(messages[3:-1], middle):
            if any(canonical_json(actual[key]) != canonical_json(value) for key, value in expected.items()):
                raise ValueError('the prerequisite changes its exact call or response')
        if (answer['status'] != status or answer['continuation_decision'] != 'defer'
                or canonical_json(answer['result']) != canonical_json(_blocked(request, code, used, None, rejected))
                or canonical_json(answer['provenance']) != canonical_json(_provenance(used))):
            raise ValueError('the diagnostic changes its prerequisite or evidence')
        return
    runtime = SubgraphRuntime(view, index=index, **meta['transport'])
    nodes = NodesByLayerRuntime(view)
    from .s2_neighbors import NeighborRuntime
    native = NeighborRuntime(view, index=index.native_index)
    available, evidence_ids = list(context['observations']), {}
    identities = [item for item in available if item['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] in ('symbol_context', 'symbol_terminal'):
        validate_identity_observations(identities, registry)
        if ([item['result']['gene_id'] for item in identities] != list(fact.gene_ids)
                or [item['arguments']['gene_symbol'] for item in identities] != request['genes']):
            raise ValueError('the supplied identity records are missing or changed')
    elif identities:
        raise ValueError('unexpected supplied registry identities')

    def verify(observation):
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected saved-observation fields')
        for records in observation['provenance'].values():
            for record in records:
                evidence_id = record['evidence_id']
                if evidence_id in evidence_ids and canonical_json(evidence_ids[evidence_id]) != canonical_json(observation):
                    raise ValueError('one evidence ID names conflicting acquisitions')
                evidence_ids[evidence_id] = observation
        tool, source = observation['operation'], observation['arguments']
        if tool == 'lookup_human_gene_symbol':
            return
        if tool not in ('induce_subgraph', 'get_nodes_by_layer', 'get_neighbors', 'get_edge') or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed supplied graph operation')
        subjects = source.get('gene_ids', [source['gene_id']] if 'gene_id' in source else [])
        if any(gene_group(gene, meta['selection_seed'])[1] != fact.split for gene in subjects):
            raise ValueError('a supplied query subject crossed split boundaries')
        graph = observation['provenance']['graph'][0]
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        execute = {'induce_subgraph': runtime.induce_subgraph, 'get_nodes_by_layer': nodes.get_nodes_by_layer,
                   'get_neighbors': native.get_neighbors, 'get_edge': view.get_edge}[tool]
        expected = _observation(tool, source, execute(**source, evidence_id=graph['evidence_id']))
        if canonical_json(observation) != canonical_json(expected):
            raise ValueError('an observation differs from its complete source or page schedule')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available, request['require_all_in_layer'], meta['membership_first'])
        call = message['tool_call']
        if proof.next_arguments is None or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments}):
            raise ValueError('a call changes the required graph domain, operation, or cursor')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes its arguments')
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('invalid fault schedule or invented failed-call evidence')
            consecutive_failures += 1
        else:
            if set(response) != {'status', 'provenance', 'result'}:
                raise ValueError('unexpected successful response fields')
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
            consecutive_failures = 0
        previous_call, previous_response = call, response
    proof, used = _plan(view, arguments, available, request['require_all_in_layer'], meta['membership_first'])
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from the visible supporting observations')
    if previous_response and previous_response['status'] == 'error':
        if proof.next_arguments is None or previous_response == _failure(1):
            raise ValueError('an error requires an unresolved terminal acquisition failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
            'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    elif proof.issue_code:
        expected = _blocked(request, proof.issue_code, used, proof.coverage)
        status, decision = 'not_found', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes sufficient proof of the full partition')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer changes the exact partition, Boolean, gate, or task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, input_sizes: Sequence[int] | None = None,
                 seed: int = 20261028, contract_path: Path = CONTRACT_PATH,
                 source_manifest: Path | None = None) -> dict:
    """Export the separate S2.10 corpus after source, wording, and token checks."""
    from .s2_10_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, input_sizes=input_sizes, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
