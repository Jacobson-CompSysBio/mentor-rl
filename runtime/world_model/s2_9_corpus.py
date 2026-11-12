"""Build native induced graphs from visible membership and complete edge proof.

Default projection retains explicit exclusions. The optional all-members gate
is a separate user prerequisite and never changes the native graph operation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _active, _observation
from .s2_2_identity import RegistryContext, get_symbols, identity_observations, validate_identity_observations
from .s2_3_corpus import gene_group, _provenance
from .s2_9_sampling import SubgraphFact, fact_from_genes
from .s2_corpus import SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_multiplex import MultiplexRuntime
from .s2_neighbors import NeighborRuntime
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphIndex, SubgraphRuntime


CONTRACT_PATH = Path(__file__).with_name('s2_9_v1.json')
VARIANTS = ('lookup', 'reordered', 'duplicate_inputs', 'context', 'duplicate_context',
    'superset_context', 'subset_context', 'disjoint_context', 'membership_context', 'membership_gate',
    'native_context', 'native_in_context', 'partial_native_context', 'positive_loop_context',
    'bounded_context', 'shorter_preview_context', 'prefix_context', 'final_page_only',
    'paged', 'overlap', 'empty_page', 'unknown_total', 'gate_context', 'symbol_context', 'symbol_terminal',
    'stale', 'wrong_layer', 'distractor', 'terminal', 'recovered', 'exhausted',
    'page_terminal', 'page_recovered', 'page_exhausted',
    'membership_terminal', 'membership_recovered', 'membership_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_genes', 'missing_scope',
    'invalid_list', 'invalid_selection', 'invalid_cursor', 'ranking_required', 'unresolved_identity', 'tool_unknown_layer')


def load_build_contract(path: Path = CONTRACT_PATH) -> dict:
    """Load the isolated S2.9 overlay and its frozen request bank."""
    from .s2_9_prompts import load_s2_9_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.9']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.9 build contract')
    contract['_requests'] = load_s2_9_requests(contract['request_bank'])
    return contract


def _arguments(view: GraphView, fact: SubgraphFact, mode='complete', top_k=None) -> dict:
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
            'mode': mode, 'top_k': top_k, 'cursor': None}


def _request(arguments: Mapping, genes: Any, require_all_in_layer: bool) -> dict:
    return {'genes': deepcopy(genes), **{key: deepcopy(arguments[key]) for key in (
        'multiplex_id', 'layer_id', 'mode', 'top_k', 'cursor')}, 'require_all_in_layer': require_all_in_layer}


def _active_context(view: GraphView) -> dict:
    runtime = MultiplexRuntime(view)
    return {**_active(view), 'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed,
        'family': runtime.family_of(layer)} for layer in view.layer_order]}


def _blocked_result(request, code, used, coverage, rejected=None):
    return {'issue_code': code, 'request': deepcopy(request), 'rejected_arguments': rejected,
            'partial_results': list(used), 'coverage': coverage}


def _plan(view, arguments, observations, require_all_in_layer):
    from .s2_9_evidence import collect
    proof = collect(view, arguments, observations, require_all_in_layer=require_all_in_layer)
    identities = [observation for observation in observations if observation['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _transport(fact: SubgraphFact, variant: str) -> dict:
    """Expose genuine edge pages while keeping the entire input list fixed."""
    paged = variant in ('prefix_context', 'final_page_only', 'paged', 'empty_page', 'unknown_total',
                        'page_terminal', 'page_recovered', 'page_exhausted')
    result = {'page_size': max(1, (fact.edge_count + 1) // 2) if paged else None,
        'disclose_total': variant not in ('unknown_total', 'final_page_only'), 'overlap': 0,
        'empty_page_at': 1 if variant == 'empty_page' else None}
    if variant in ('prefix_context', 'final_page_only', 'empty_page', 'page_terminal', 'page_recovered', 'page_exhausted') and fact.edge_count < 2:
        raise VariantUnavailable('this contrast needs multiple genuine edge pages')
    if variant == 'overlap':
        if fact.edge_count < 3:
            raise VariantUnavailable('overlapping pages require at least three edges')
        result.update(page_size=max(2, (fact.edge_count + 1) // 2), overlap=1)
    return result


def _reason(result, framing):
    text = (f"The layer is {result['layer_id']}, with directed={result['directed']}. "
        f"The full projection retains {result['layer_gene_ids']}, excludes layer-absent {result['not_in_layer_gene_ids']}, "
        f"and excludes globally missing {result['missing_gene_ids']}. "
        f"There are {result['layer_gene_count']} selected layer members and {result['edge_count']} returned native edges. "
        'Keep native source/target order in directed layers. Canonicalize only undirected endpoint pairs. '
        'Retain reciprocal arcs, their separate exact weights, self-loops, and selected members without an induced incident edge. ')
    if result['collection']['complete']:
        text += 'The visible proof settles every candidate pair, including loops, or establishes an empty projected edge domain. '
        text += 'The complete induced set supports isolation only inside this selected graph. '
    else:
        text += 'The explicit canonical edge preview is complete for the requested prefix. Its omitted edges remain unknown. '
        text += 'Do not infer selected-set isolation or a full graph from this bounded display. '
    text += 'Membership coverage and edge completeness describe different facts. An exclusion is not an evaluated negative edge predicate. '
    return text + ('The requested workflow step is complete, so continue.' if framing == 'step'
                   else 'The standalone request is complete, so stop.')


def build_row(view: GraphView, fact: SubgraphFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, index: SubgraphIndex,
              registry: RegistryContext | None = None, mode: str = 'complete', top_k: int | None = None,
              require_all_in_layer: bool = False, seed: int = 20261028) -> dict:
    """Build one complete projection, declared preview, or attributed prerequisite failure."""
    from .s2_9_prompts import render_s2_9_request
    if variant not in VARIANTS or framing not in FRAMINGS or type(require_all_in_layer) is not bool:
        raise ValueError('unsupported subgraph contrast, framing, or gate')
    if variant == 'reordered':
        fact = replace(fact, gene_ids=tuple(reversed(fact.gene_ids)))
    if variant in ('membership_gate', 'gate_context', 'membership_terminal', 'membership_recovered', 'membership_exhausted'):
        require_all_in_layer = True
    arguments = _arguments(view, fact, mode, top_k)
    runtime = SubgraphRuntime(view, index=index, **_transport(fact, variant))
    runtime.validate_arguments(arguments)
    raw_genes = list(fact.gene_ids)
    if variant == 'duplicate_inputs':
        if not raw_genes:
            raise VariantUnavailable('duplicate inputs require a nonempty request')
        raw_genes += [raw_genes[0], raw_genes[-1]]
    if variant in ('symbol_context', 'symbol_terminal'):
        if registry is None:
            raise VariantUnavailable('symbol context requires the pinned registry')
        try:
            raw_genes = list(get_symbols(fact.gene_ids, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = _request(arguments, raw_genes, require_all_in_layer)
    core = render_s2_9_request(contract['_requests'], split=fact.split, variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's29_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, request, user])[:32]
    transport = _transport(fact, variant)
    native = NeighborRuntime(view, index=index.native_index)
    nodes = NodesByLayerRuntime(view)
    observations, serial = [], 0

    def evidence_id():
        nonlocal serial
        serial += 1
        return f'ev_{row_id}_{serial}'

    def save_graph(source, *, first_only=False, last_only=False):
        if not source['gene_ids']:
            return
        pages = []
        while True:
            response = runtime.induce_subgraph(**source, evidence_id=evidence_id())
            pages.append(_observation('induce_subgraph', source, response))
            cursor = response['result']['collection']['next_cursor']
            if cursor is None or first_only:
                break
            source = {**source, 'cursor': cursor}
        observations.extend(pages[-1:] if last_only else pages)

    def save_membership():
        if not fact.gene_ids:
            return
        source = nodes.arguments(fact.layer_id, list(fact.gene_ids))
        observations.append(_observation('get_nodes_by_layer', source, nodes.get_nodes_by_layer(**source, evidence_id=evidence_id())))

    def save_native(gene, side, *, k=None):
        source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': gene, 'direction': side,
            'mode': 'top_k' if k else 'complete', 'top_k': k, 'cursor': None}
        observations.append(_observation('get_neighbors', source, native.get_neighbors(**source, evidence_id=evidence_id())))

    side = 'out' if view.layers[fact.layer_id].directed else 'undirected'
    if variant in ('symbol_context', 'symbol_terminal'):
        observations.extend(identity_observations(fact.gene_ids, row_id, registry))
    elif variant in ('context', 'duplicate_context', 'stale'):
        save_graph(arguments)
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
        elif variant == 'stale':
            for observation in observations:
                for record in observation['provenance']['graph']:
                    record['graph_version'] = '0' * 64
    elif variant == 'superset_context':
        pool = [gene for gene in view.multiplex_nodes if gene not in fact.gene_ids and gene_group(gene, seed)[1] == fact.split]
        if not pool:
            raise VariantUnavailable('no same-split source input can enlarge the context')
        extra = min(pool, key=lambda gene: digest([seed, fact.split_unit, 'superset', gene]))
        save_graph({**arguments, 'gene_ids': [*fact.gene_ids, extra], 'mode': 'complete', 'top_k': None})
    elif variant in ('subset_context', 'disjoint_context'):
        if len(fact.gene_ids) < 2:
            raise VariantUnavailable('a proper input subset requires at least two inputs')
        groups = [fact.gene_ids[:-1]] if variant == 'subset_context' else [fact.gene_ids[:len(fact.gene_ids)//2], fact.gene_ids[len(fact.gene_ids)//2:]]
        for group in groups:
            save_graph({**arguments, 'gene_ids': list(group), 'mode': 'complete', 'top_k': None})
    elif variant in ('membership_context', 'gate_context'):
        save_membership()
    elif variant in ('native_context', 'native_in_context', 'partial_native_context'):
        if variant == 'native_in_context':
            if not view.layers[fact.layer_id].directed:
                raise VariantUnavailable('incoming-only context requires a directed layer')
            side = 'in'
        genes = fact.gene_ids[:-1] if variant == 'partial_native_context' else fact.gene_ids
        if variant == 'partial_native_context' and len(fact.gene_ids) < 2:
            raise VariantUnavailable('partial input evidence requires multiple unique inputs')
        for gene in genes:
            save_native(gene, side)
    elif variant == 'positive_loop_context':
        if len(fact.gene_ids) != 1 or fact.loop_count != 1:
            raise VariantUnavailable('this proof requires an established singleton loop')
        rows = index.native_index.neighbors(fact.layer_id, fact.gene_ids[0], side)
        k = next(i for i, (neighbor, _) in enumerate(rows, 1) if neighbor == fact.gene_ids[0])
        if k >= len(rows):
            raise VariantUnavailable('the loop must be visible in a genuinely incomplete prefix')
        save_native(fact.gene_ids[0], side, k=k)
    elif variant == 'bounded_context':
        save_graph({**arguments, 'mode': 'top_k', 'top_k': 1})
    elif variant == 'shorter_preview_context':
        if mode != 'top_k' or fact.edge_count <= top_k + 1:
            raise VariantUnavailable('a shorter preview needs a larger incomplete source prefix')
        save_graph({**arguments, 'top_k': top_k + 1})
    elif variant in ('prefix_context', 'final_page_only', 'page_terminal', 'page_recovered', 'page_exhausted'):
        save_graph(arguments, first_only=variant != 'final_page_only', last_only=variant == 'final_page_only')
    elif variant == 'wrong_layer':
        other = next((layer for layer in view.layer_order if layer != fact.layer_id), None)
        if other is None:
            raise VariantUnavailable('no other registered layer exists')
        save_graph({**arguments, 'layer_id': other, 'mode': 'complete', 'top_k': None})
    elif variant == 'distractor':
        pool = [gene for gene in view.multiplex_nodes if gene not in fact.gene_ids and gene_group(gene, seed)[1] == fact.split]
        if not pool:
            raise VariantUnavailable('no same-split source distractor exists')
        gene = min(pool, key=lambda value: digest([seed, fact.split_unit, 'distractor', value]))
        save_graph({**arguments, 'gene_ids': [gene], 'mode': 'complete', 'top_k': None})
    proof, _ = _plan(view, arguments, observations, require_all_in_layer)
    kind = variant.removeprefix('page_').removeprefix('membership_').removeprefix('symbol_')
    if variant.startswith('membership_') and kind in ('terminal', 'recovered', 'exhausted') and proof.next_tool != 'get_nodes_by_layer':
        raise VariantUnavailable('a membership failure requires an unresolved explicit gate')
    context = {'active_graph': _active_context(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    injected, failure = False, None
    while True:
        proof, used = _plan(view, arguments, observations, require_all_in_layer)
        if proof.next_arguments is None:
            break
        pending, operation = proof.next_arguments, proof.next_tool
        thought = (f"The request selects {list(fact.gene_ids)} in {fact.layer_id}, mode {mode}. "
            'Complete the missing membership or edge proof. Projection keeps exclusions separate from retained nodes. '
            'A partial edge list does not prove absent edges or self-loops. Keep the full ordered input fixed during edge pagination.')

        def append(response, reasoning):
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                'tool_call': {'name': operation, 'arguments': deepcopy(pending)}},
                {'role': 'tool', 'name': operation, 'content': response}])

        execute = nodes.get_nodes_by_layer if operation == 'get_nodes_by_layer' else runtime.induce_subgraph
        should_fail = (not injected and kind in ('terminal', 'recovered', 'exhausted')
            and (not variant.startswith('page_') or pending.get('cursor') is not None))
        if should_fail:
            injected = True
            response = _failure(None if kind == 'terminal' else 1)
            append(response, thought)
            if kind != 'terminal':
                response = _failure(0) if kind == 'exhausted' else execute(**pending, evidence_id=evidence_id())
                append(response, 'The visible runtime error permits one retry. Repeat the identical tool and every argument, including any cursor. '
                    'Retain all earlier supported observations. Stop after recovery or exhaustion; failed calls add no evidence.')
            if response['status'] == 'error':
                failure = operation, pending, response
                break
        else:
            response = execute(**pending, evidence_id=evidence_id())
            append(response, thought)
        observations.append(_observation(operation, pending, response))
    if kind in ('terminal', 'recovered', 'exhausted') and not injected:
        raise VariantUnavailable('the selected failure is unnecessary after the visible proof')
    if failure:
        operation, pending, response = failure
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': {'failed_tool': operation,
            'error_code': response['error_code'], 'arguments': pending, 'partial_results': used, 'coverage': proof.coverage},
            'continuation_decision': 'defer', 'blocked_reason': 'The required backend must recover before the remaining graph proof can finish.',
            'summary': 'A required acquisition failed. Retain the supported membership and exact partial edge observations.'}
        thought = 'The required call failed without a remaining retry allowance. Preserve every successful partial and truthful membership coverage. '
        thought += 'Checked membership does not complete edge retrieval. Do not replace unavailable edges with an empty graph; defer.'
    elif proof.issue_code:
        answer = {'status': 'not_found', 'provenance': _provenance(used),
            'result': _blocked_result(request, proof.issue_code, used, proof.coverage),
            'continuation_decision': 'defer', 'blocked_reason': 'The explicit all-members prerequisite fails; revise the requested set.',
            'summary': 'The complete membership partition fails the explicit gate. Retain the classification and defer before graph construction.'}
        thought = 'The visible complete partition excludes at least one input, and the user explicitly requires every input to be a layer member. '
        thought += 'Keep the actual partition and defer. Ordinary default projection would have been a completed different request.'
    else:
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': proof.result,
            'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
            'summary': 'The requested native projection or declared edge preview follows from the visible compatible evidence.'}
        thought = ('The supplied observations are sufficient; no new call is needed. ' if len(messages) == 3 else '') + _reason(proof.result, framing)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.9', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'category_counts': list(fact.category_counts), 'edge_count': fact.edge_count, 'isolated_count': fact.isolated_count,
        'loop_count': fact.loop_count, 'reciprocal_pairs': fact.reciprocal_pairs, 'descending_arc_count': fact.descending_arc_count,
        'category': fact.category, 'framing': framing, 'variant': variant, 'request': request,
        'transport': transport, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][mode]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _identity_rejection(registry: RegistryContext, symbol: str, evidence_id: str) -> dict:
    """Return a current-symbol negative only when the full registry supports it."""
    if registry.gene_ids_by_symbol.get(symbol.strip().casefold(), ()):
        raise ValueError('the unresolved-symbol case is not negative in this pinned registry')
    return {'status': 'not_found', 'provenance': {'registry': [{'registry': 'ensembl',
        'release': registry.release, 'evidence_id': evidence_id}], 'graph': []},
        'gene_symbol': symbol, 'candidate_gene_ids': []}


def _identity_partial(symbol: str, response: Mapping) -> dict:
    return {'operation': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': symbol},
        'status': response['status'], 'result': {key: deepcopy(value) for key, value in response.items()
            if key not in ('status', 'provenance')}, 'provenance': deepcopy(response['provenance'])}


def _diagnostic_request(view: GraphView, fact: SubgraphFact, kind: str) -> tuple[dict, str, str]:
    request = _request(_arguments(view, fact), list(fact.gene_ids), False)
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'tool_unknown_layer': 'UNKNOWN_LAYER',
        'unknown_multiplex': 'UNKNOWN_MULTIPLEX', 'missing_genes': 'MISSING_GENES',
        'missing_scope': 'MISSING_SCOPE', 'invalid_list': 'INVALID_GENE_LIST',
        'invalid_selection': 'INVALID_SELECTION', 'invalid_cursor': 'INVALID_CURSOR',
        'ranking_required': 'SUBGRAPH_RANKING_REQUIRED', 'unresolved_identity': 'UNRESOLVED_IDENTITY'}
    if kind not in codes:
        raise ValueError('unsupported subgraph prerequisite diagnostic')
    if kind in ('unknown_layer', 'tool_unknown_layer'):
        request['layer_id'] = 'TFs:missing_context'
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_subgraph_multiplex'
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'invalid_list':
        request['genes'] = fact.gene_ids[0] if fact.gene_ids else {'invalid': 'gene list'}
    elif kind == 'invalid_selection':
        request.update(mode='top_k', top_k=0)
    elif kind == 'invalid_cursor':
        request['cursor'] = ''
    elif kind == 'ranking_required':
        request.update(mode='strongest', top_k=1)
    elif kind == 'unresolved_identity':
        request['genes'] = ['HER2', *list(fact.gene_ids[:1])]
    status = ('not_found' if kind in ('unknown_layer', 'tool_unknown_layer', 'unknown_multiplex', 'unresolved_identity')
              else 'ambiguous' if kind == 'ranking_required' else 'invalid_input')
    return request, status, codes[kind]


def build_diagnostic(view: GraphView, fact: SubgraphFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'missing_genes', registry: RegistryContext | None = None,
                     seed: int = 20261028) -> dict:
    """Preserve unresolved raw inputs without a fabricated membership or graph result."""
    from .s2_9_prompts import render_s2_9_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    wording_kind = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    core = render_s2_9_diagnostic_request(contract['_requests'], split=fact.split, kind=wording_kind,
        variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's29_' + digest([contract['contract_version'], view.graph_version, user, kind])[:32]
    active = _active_context(view)
    if kind == 'tool_unknown_layer':
        active.pop('layers')
    messages = [{'role': 'system', 'content': contract['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': active, 'observations': []})},
        {'role': 'user', 'content': user}]
    used, rejected = [], None
    if kind == 'tool_unknown_layer':
        rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
            'gene_ids': request['genes'], 'mode': request['mode'], 'top_k': request['top_k'], 'cursor': request['cursor']}
        response = SubgraphRuntime(view).induce_subgraph(**rejected, evidence_id='unused_rejected_subgraph_request')
        messages.extend([{'role': 'assistant', 'thinking':
            'The visible context withholds the layer catalogue. Ask the resolver for the requested scope without an invented replacement.',
            'tool_call': {'name': 'induce_subgraph', 'arguments': deepcopy(rejected)}},
            {'role': 'tool', 'name': 'induce_subgraph', 'content': response}])
    elif kind == 'unresolved_identity':
        if registry is None:
            raise ValueError('the unresolved identity requires the pinned current-symbol registry')
        symbol = request['genes'][0]
        response = _identity_rejection(registry, symbol, f'ev_{row_id}_identity')
        messages.extend([{'role': 'assistant', 'thinking':
            'A symbol lacks supplied authoritative identity evidence. Resolve that exact raw input before establishing the canonical graph domain.',
            'tool_call': {'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': symbol}}},
            {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}])
        used.append(_identity_partial(symbol, response))
    answer = {'status': status, 'provenance': _provenance(used),
        'result': _blocked_result(request, code, used, None, rejected), 'continuation_decision': 'defer',
        'blocked_reason': 'A valid canonical input list, registered scope, and explicit selection rule are required.',
        'summary': 'A request prerequisite is unresolved. No membership partition or induced graph was computed.'}
    messages.append({'role': 'assistant', 'thinking': f"The literal request is {request}. Its prerequisite fails with {code}. "
        'Preserve original JSON values, registry observations, and rejected arguments. '
        'An unresolved identity is not graph absence, and a missing list is not an explicit empty set. Defer.', 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.9', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
        'category_counts': None, 'edge_count': None, 'isolated_count': None, 'loop_count': None,
        'reciprocal_pairs': None, 'descending_arc_count': None,
        'category': fact.category, 'framing': framing, 'variant': kind, 'request': request,
        'transport': None, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][wording_kind]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: SubgraphIndex | None = None) -> None:
    """Replay source records, then grade the graph proof visible in this row."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_9_prompts import render_s2_9_request, render_s2_9_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.9 requires one user turn and explicit harness context')
    for message in messages:
        expected = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
            {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
            {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected or message['role'] == 'assistant' and (not isinstance(message['thinking'], str) or not message['thinking'].strip()):
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the subgraph overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.9'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('subgraph row identity differs from its build contract')
    index = index or SubgraphIndex(view)
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], meta['selection_seed'],
                            split=meta['split'], category=meta['category'])
    if fact.split_unit != meta['split_unit'] or list(fact.gene_ids) != meta['gene_ids']:
        raise ValueError('the canonical input domain differs from its fixed subject split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if ((answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason'])
            or answer['blocked_reason'] is not None and (not isinstance(answer['blocked_reason'], str) or not answer['blocked_reason'].strip())):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if meta['framing'] not in FRAMINGS or not diagnostic and meta['variant'] not in VARIANTS:
        raise ValueError('unsupported framing or subgraph contrast')
    request = meta['request']
    if (set(request) != {'genes', 'multiplex_id', 'layer_id', 'mode', 'top_k', 'cursor', 'require_all_in_layer'}
            or type(request['require_all_in_layer']) is not bool):
        raise ValueError('the literal request has invalid fields or gate type')
    wording_kind = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
    templates = (contract['_requests']['diagnostic_requests'][meta['split']][wording_kind] if diagnostic
                 else contract['_requests'][meta['split']].get(request['mode'], []))
    if any(type(meta[key]) is not int or not 0 <= meta[key] < size for key, size in (
            ('wrapper_index', len(bank[meta['split']][meta['framing']])), ('request_template_index', len(templates)))):
        raise ValueError('wrapper and request indices must select their audited banks')
    labels = ('category_counts', 'edge_count', 'isolated_count', 'loop_count', 'reciprocal_pairs', 'descending_arc_count')
    if diagnostic:
        expected_request, status, code = _diagnostic_request(view, fact, meta['variant'])
        if canonical_json(request) != canonical_json(expected_request):
            raise ValueError('the diagnostic changes the raw request')
        core = render_s2_9_diagnostic_request(contract['_requests'], split=meta['split'], kind=wording_kind,
            variant_index=meta['request_template_index'], **request)
    else:
        if any(canonical_json(getattr(fact, key)) != canonical_json(meta[key]) for key in labels):
            raise ValueError('private sampling labels differ from the complete source graph')
        arguments = _arguments(view, fact, request['mode'], request['top_k'])
        SubgraphRuntime(view).validate_arguments(arguments)
        if request['multiplex_id'] != view.multiplex_id or request['layer_id'] != fact.layer_id or request['cursor'] is not None:
            raise ValueError('the raw scope or new-selection cursor changed')
        expected_genes = list(fact.gene_ids)
        if meta['variant'] in ('symbol_context', 'symbol_terminal'):
            if registry is None:
                raise ValueError('the supplied symbol mapping needs the pinned registry')
            expected_genes = list(get_symbols(fact.gene_ids, registry))
        elif meta['variant'] == 'duplicate_inputs':
            expected_genes += [expected_genes[0], expected_genes[-1]]
        if request['genes'] != expected_genes or meta['transport'] != _transport(fact, meta['variant']):
            raise ValueError('the raw input order, duplicates, identity, or declared transport changed')
        if meta['variant'] in ('membership_gate', 'gate_context', 'membership_terminal', 'membership_recovered', 'membership_exhausted') and request['require_all_in_layer'] is not True:
            raise ValueError('the explicit gate contrast drops its prerequisite')
        core = render_s2_9_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'], **request)
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    active = _active_context(view)
    if meta['variant'] == 'tool_unknown_layer':
        active.pop('layers')
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != active:
        raise ValueError('the declared active graph changed')
    if diagnostic:
        if context['observations'] or meta['transport'] is not None or any(meta[key] is not None for key in labels):
            raise ValueError('malformed prerequisite diagnostic')
        used, rejected = [], None
        expected_middle = []
        if meta['variant'] == 'tool_unknown_layer':
            rejected = {'multiplex_id': request['multiplex_id'], 'layer_id': request['layer_id'],
                'gene_ids': request['genes'], 'mode': request['mode'], 'top_k': request['top_k'], 'cursor': request['cursor']}
            response = SubgraphRuntime(view).induce_subgraph(**rejected, evidence_id='unused_rejected_subgraph_request')
            expected_middle = [{'role': 'assistant', 'tool_call': {'name': 'induce_subgraph', 'arguments': rejected}},
                               {'role': 'tool', 'name': 'induce_subgraph', 'content': response}]
        elif meta['variant'] == 'unresolved_identity':
            if registry is None:
                raise ValueError('unresolved identity needs the pinned registry')
            response = _identity_rejection(registry, request['genes'][0], f"ev_{row['row_id']}_identity")
            used.append(_identity_partial(request['genes'][0], response))
            expected_middle = [{'role': 'assistant', 'tool_call': {'name': 'lookup_human_gene_symbol',
                'arguments': {'gene_symbol': request['genes'][0]}}},
                {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': response}]
        if len(messages) != 4 + len(expected_middle):
            raise ValueError('the prerequisite adds or drops a required call')
        for actual, expected in zip(messages[3:-1], expected_middle):
            if any(canonical_json(actual[key]) != canonical_json(value) for key, value in expected.items()):
                raise ValueError('the prerequisite changes its exact call or response')
        expected = _blocked_result(request, code, used, None, rejected)
        if (answer['status'] != status or answer['continuation_decision'] != 'defer'
                or canonical_json(answer['result']) != canonical_json(expected)
                or canonical_json(answer['provenance']) != canonical_json(_provenance(used))):
            raise ValueError('the diagnostic changes its prerequisite or evidence')
        return
    runtime = SubgraphRuntime(view, index=index, **meta['transport'])
    nodes, native = NodesByLayerRuntime(view), NeighborRuntime(view, index=index.native_index)
    available, evidence_ids = list(context['observations']), {}
    identities = [observation for observation in available if observation['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] in ('symbol_context', 'symbol_terminal'):
        validate_identity_observations(identities, registry)
        if ([observation['result']['gene_id'] for observation in identities] != list(fact.gene_ids)
                or [observation['arguments']['gene_symbol'] for observation in identities] != request['genes']):
            raise ValueError('supplied identity evidence is missing or changed')
    elif identities:
        raise ValueError('unexpected identity context')

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
        if tool not in ('induce_subgraph', 'get_nodes_by_layer', 'get_neighbors') or len(observation['provenance']['graph']) != 1:
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
                   'get_neighbors': native.get_neighbors}[tool]
        expected = _observation(tool, source, execute(**source, evidence_id=graph['evidence_id']))
        if canonical_json(observation) != canonical_json(expected):
            raise ValueError('an observation differs from its complete pinned source or page schedule')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available, request['require_all_in_layer'])
        call = message['tool_call']
        if proof.next_arguments is None or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments}):
            raise ValueError('a call changes the required input domain, operation, or cursor')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes the failed arguments')
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
    proof, used = _plan(view, arguments, available, request['require_all_in_layer'])
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from its visible supporting observations')
    if previous_response and previous_response['status'] == 'error':
        if proof.next_arguments is None or previous_response == _failure(1):
            raise ValueError('an error requires an unresolved terminal failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
            'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    elif proof.issue_code:
        expected = _blocked_result(request, proof.issue_code, used, proof.coverage)
        status, decision = 'not_found', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes sufficient membership and edge proof')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer changes the visible graph, preview, gate, or task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, input_sizes: Sequence[int] | None = None,
                 seed: int = 20261028, contract_path: Path = CONTRACT_PATH,
                 source_manifest: Path | None = None) -> dict:
    """Export the isolated S2.9 corpus with source, wording, and full token audits."""
    from .s2_9_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, input_sizes=input_sizes, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
