"""Build exact shared-neighbor traces from visible sets and pinned sources.

Each final result is a calculation over two ordered query roles. Primitive
neighbor acquisitions retain their own direction, weights, pages, and evidence.
The private sampler never supplies an unseen intersection or membership fact.
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
from .s2_6_corpus import _global_observation
from .s2_7_sampling import SharedFact, fact_from_queries, neighbor_ids
from .s2_corpus import SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure
from .s2_neighbors import NeighborIndex, NeighborRuntime
from .s2_sources import GraphView


CONTRACT_PATH = Path(__file__).with_name('s2_7_v1.json')
VARIANTS = ('lookup', 'swapped', 'first_context', 'second_context', 'context', 'duplicate_context',
            'bounded_context', 'containment_context', 'global_context', 'empty_context', 'symbol_context',
            'stale', 'wrong_direction', 'wrong_layer', 'distractor', 'paged', 'overlap', 'empty_page',
            'unknown_total', 'prefix_context', 'terminal', 'recovered', 'exhausted',
            'second_terminal', 'second_recovered', 'second_exhausted',
            'page_terminal', 'page_recovered', 'page_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_scope', 'missing_genes',
               'invalid_pair', 'invalid_directions', 'ranking_required')


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the isolated shared-neighbor overlay and its pinned frozen requests."""
    from .s2_7_prompts import load_s2_7_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.7']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.7 build contract')
    contract['_requests'] = load_s2_7_requests(contract['request_bank'])
    return contract


def _arguments(view: GraphView, fact: SharedFact) -> dict:
    """Preserve both user roles, including identical genes and their directions."""
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
            'queries': [{'gene_id': gene, 'direction': direction} for gene, direction in fact.queries]}


def _atomic(arguments: Mapping[str, Any]) -> list[dict]:
    """Expand final-only both roles and deduplicate identical primitive queries."""
    result = []
    for role in arguments['queries']:
        for direction in ('in', 'out') if role['direction'] == 'both' else (role['direction'],):
            query = {'gene_id': role['gene_id'], 'direction': direction}
            if query not in result:
                result.append(query)
    return result


def _native_arguments(arguments: Mapping[str, Any], query: Mapping[str, Any]) -> dict:
    """Start each required complete primitive selection at the null cursor."""
    return {'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'], **query,
            'mode': 'complete', 'top_k': None, 'cursor': None}


def _kind(directions: Sequence[str]) -> str:
    """Select frozen wording without changing the caller's two role bindings."""
    return 'undirected' if list(directions) == ['undirected', 'undirected'] else '_'.join(directions)


def _transport(view: GraphView, index: NeighborIndex, fact: SharedFact, variant: str) -> dict:
    """Declare real page schedules without clipping logical neighbor sets."""
    largest = max((len(index.neighbors(fact.layer_id, q['gene_id'], q['direction']))
                   for q in _atomic(_arguments(view, fact))), default=0)
    result = {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}
    if variant in ('paged', 'unknown_total', 'prefix_context', 'empty_page', 'page_terminal', 'page_recovered', 'page_exhausted'):
        result['page_size'] = max(1, (largest + 1) // 2)
    if variant == 'overlap':
        if largest < 3:
            raise VariantUnavailable('overlapping pages require at least three neighbors')
        result.update(page_size=max(2, (largest + 1) // 2), overlap=1)
    if variant == 'empty_page':
        if largest < 2:
            raise VariantUnavailable('an empty intermediate page requires multiple pages')
        result['empty_page_at'] = 1
    if variant == 'unknown_total':
        result['disclose_total'] = False
    return result


def _plan(view: GraphView, arguments: dict, observations: Sequence[dict]):
    """Keep the supplied registry mappings beside the source-blind set proof."""
    from .s2_7_evidence import collect
    proof = collect(view, arguments, observations)
    identities = [o for o in observations if o['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _blocked_result(request: dict, issue_code: str, used: Sequence[dict], coverage: dict | None) -> dict:
    """Preserve the raw blocked request without adding an uncomputed shared set."""
    return {'issue_code': issue_code, 'request': deepcopy(request), 'rejected_arguments': None,
            'partial_results': list(used), 'coverage': coverage}


def _reason(result: dict, framing: str, used: Sequence[dict]) -> str:
    """Explain the visible exact-intersection proof and truthful primitive coverage."""
    text = (f"The ordered query roles are {result['queries']} in {result['layer_id']}. "
            f"Their exact sorted intersection has {result['shared_neighbor_count']} IDs: {result['shared_neighbors']}. "
            f"The distinct input memberships are {result['input_memberships']}. "
            'Both global prerequisites are supported. Layer absence is a resolved empty set; unknown layer membership stays null. ')
    coverage = result['coverage']
    if coverage['unchecked_queries']:
        text += ('One complete role set is contained in visible positives of the other; unseen entries cannot add anything outside that complete set. '
                 if result['shared_neighbors'] else
                 'A complete role set is empty. This proves an empty intersection after both global prerequisites are supported. ')
        text += 'The unused or incomplete primitive lists remain unchecked; do not pretend they were fully retrieved. '
    else:
        text += 'The complete primitive sets supply each role, with incoming/outgoing unions where requested. Intersect their distinct IDs. '
    if result['queries'][0] == result['queries'][1]:
        text += 'Identical roles share one primitive acquisition while remaining two user roles. '
    text += 'Retain original acquisition evidence and source self-loops. Do not remove query endpoints or combine weights. '
    return text + ('This exact result supplies the workflow step, so continue.' if framing == 'step'
                   else 'This completes the standalone shared-neighbor question, so stop.')


def build_row(view: GraphView, fact: SharedFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, index: NeighborIndex,
              registry: RegistryContext | None = None, seed: int = 20261028) -> dict:
    """Build complete shared sets with sufficient evidence, exact pages, and retries."""
    from .s2_7_prompts import render_s2_7_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported S2.7 contrast or framing')
    if variant == 'swapped':
        fact = replace(fact, queries=tuple(reversed(fact.queries)), neighbor_counts=tuple(reversed(fact.neighbor_counts)))
    arguments = _arguments(view, fact)
    atomic = _atomic(arguments)
    raw_genes = [gene for gene, _ in fact.queries]
    directions = [direction for _, direction in fact.queries]
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('symbol context requires a verified registry')
        try:
            raw_genes = list(get_symbols(raw_genes, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    core = render_s2_7_request(contract['_requests'], split=fact.split, variant_index=variant_index,
        multiplex_id=view.multiplex_id, layer_id=fact.layer_id, genes=raw_genes, directions=directions)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's27_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, user])[:32]
    transport = _transport(view, index, fact, variant)
    runtime = NeighborRuntime(view, index=index, **transport)
    observations, serial = [], 0

    def acquire(source: dict) -> dict:
        nonlocal serial
        serial += 1
        return runtime.get_neighbors(**source, evidence_id=f'ev_{row_id}_{serial}')

    def save(query: dict, *, mode: str = 'complete', top_k: int | None = None,
             first_only: bool = False, layer_id: str | None = None) -> None:
        source = {**_native_arguments(arguments, query), 'mode': mode, 'top_k': top_k}
        if layer_id is not None:
            source['layer_id'] = layer_id
        while True:
            response = acquire(source)
            observations.append(_observation('get_neighbors', deepcopy(source), response))
            if response['status'] != 'resolved' or first_only:
                return
            cursor = response['result']['collection']['next_cursor']
            if cursor is None:
                return
            source = {**source, 'cursor': cursor}

    def role_queries(role: int) -> list[dict]:
        query = arguments['queries'][role]
        return [{'gene_id': query['gene_id'], 'direction': direction}
                for direction in (('in', 'out') if query['direction'] == 'both' else (query['direction'],))]

    if variant == 'symbol_context':
        observations.extend(identity_observations(tuple(dict.fromkeys(g for g, _ in fact.queries)), row_id, registry))
    elif variant in ('context', 'duplicate_context', 'stale', 'bounded_context'):
        for query in atomic:
            save(query, mode='top_k' if variant == 'bounded_context' else 'complete',
                 top_k=1 if variant == 'bounded_context' else None)
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
        if variant == 'stale':
            for observation in observations:
                observation['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('first_context', 'second_context', 'second_terminal', 'second_recovered', 'second_exhausted'):
        for query in role_queries(1 if variant == 'second_context' else 0):
            save(query)
    elif variant == 'containment_context':
        if any(direction == 'both' for direction in directions) or atomic[0] == atomic[-1]:
            raise VariantUnavailable('bounded containment uses two distinct primitive roles')
        found = False
        for first, second in ((0, 1), (1, 0)):
            complete = neighbor_ids(index, fact.layer_id, *fact.queries[first])
            ranked = index.neighbors(fact.layer_id, *fact.queries[second])
            rank = {gene: ordinal for ordinal, (gene, _) in enumerate(ranked, 1)}
            if complete and complete <= rank.keys():
                k = max(rank[gene] for gene in complete)
                if k < len(ranked):
                    save(role_queries(first)[0])
                    save(role_queries(second)[0], mode='top_k', top_k=k)
                    found = True
                    break
        if not found:
            raise VariantUnavailable('no exact nonempty containment proof from an incomplete ranked prefix')
    elif variant == 'global_context':
        genes = tuple(dict.fromkeys(g for g, _ in fact.queries))
        observations.append(_global_observation(view, genes, f'ev_{row_id}_global'))
    elif variant == 'empty_context':
        role = next((i for i, count in enumerate(fact.neighbor_counts) if count == 0), None)
        if role is None:
            raise VariantUnavailable('this contrast requires a source-supported empty role')
        for query in role_queries(role):
            save(query)
            if observations[-1]['result']['not_in_layer_gene_ids']:
                break
        genes = tuple(dict.fromkeys(g for g, _ in fact.queries))
        observations.append(_global_observation(view, genes, f'ev_{row_id}_global'))
    elif variant in ('prefix_context', 'page_terminal', 'page_recovered', 'page_exhausted'):
        query = next((q for q in atomic if len(index.neighbors(fact.layer_id, q['gene_id'], q['direction'])) > transport['page_size']), None)
        if query is None:
            raise VariantUnavailable('a saved page prefix requires an unfinished primitive list')
        save(query, first_only=True)
        if observations[-1]['status'] != 'resolved' or observations[-1]['result']['collection']['next_cursor'] is None:
            raise VariantUnavailable('the selected source list has no continuation')
    elif variant == 'wrong_direction':
        if directions[0] not in ('in', 'out'):
            raise VariantUnavailable('wrong-direction context requires a single directed role')
        query = {'gene_id': fact.queries[0][0], 'direction': 'out' if directions[0] == 'in' else 'in'}
        if query in atomic:
            raise VariantUnavailable('the opposite direction already supplies another requested role')
        save(query)
    elif variant == 'wrong_layer':
        layer = next((name for name in view.layer_order if name != fact.layer_id
                      and view.layers[name].directed == view.layers[fact.layer_id].directed), None)
        if layer is None:
            raise VariantUnavailable('no other registered layer with the same representation')
        save(atomic[0], layer_id=layer)
    elif variant == 'distractor':
        used_genes = {g for g, _ in fact.queries}
        query = atomic[0]
        pool = sorted((g for g in view.layers[fact.layer_id].nodes if g not in used_genes and gene_group(g, seed)[1] == fact.split
                       and len(index.neighbors(fact.layer_id, g, query['direction'])) <= 5),
                      key=lambda gene: digest([seed, fact.split_unit, 'distractor', gene]))
        if not pool:
            raise VariantUnavailable('no same-split readable distractor')
        save({**query, 'gene_id': pool[0]})
    proof, _ = _plan(view, arguments, observations)
    if variant in ('bounded_context', 'second_terminal', 'second_recovered', 'second_exhausted') and proof.next_arguments is None:
        raise VariantUnavailable('the supplied context already settles or blocks the intersection')
    if variant == 'containment_context' and (proof.result is None or not proof.coverage['unchecked_queries']):
        raise VariantUnavailable('containment must prove the exact set without completing the other list')
    context = {'active_graph': _active(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    kind = variant.removeprefix('second_').removeprefix('page_')
    injected, failed = False, None
    while True:
        proof, used = _plan(view, arguments, observations)
        if proof.next_arguments is None:
            break
        pending = proof.next_arguments
        thought = (f"The request asks for the exact intersection of {arguments['queries']} in {fact.layer_id}. "
                   f"The next required primitive is {pending['gene_id']} direction {pending['direction']}, cursor {pending['cursor']!r}. "
                   'Preserve the two role bindings. A bounded omission is unknown, and an empty opposite set does not remove global prerequisites. '
                   'Use the exact complete selection and retain any required pages.')

        def append(response: dict, reasoning: str) -> None:
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                              'tool_call': {'name': 'get_neighbors', 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': 'get_neighbors', 'content': response}])

        should_fail = (not injected and kind in ('terminal', 'recovered', 'exhausted')
                       and (not variant.startswith('page_') or pending['cursor'] is not None))
        if should_fail:
            injected = True
            response = _failure(None if kind == 'terminal' else 1)
            append(response, thought)
            if kind != 'terminal':
                response = _failure(0) if kind == 'exhausted' else acquire(pending)
                append(response, 'The runtime explicitly supplies retries_remaining=1. Repeat the identical tool and all seven arguments, '
                                 'including the opaque cursor. Preserve earlier observations and stop retries after recovery or exhaustion.')
            if response['status'] == 'error':
                failed = (pending, response)
                break
        else:
            response = acquire(pending)
            append(response, thought)
        observations.append(_observation('get_neighbors', deepcopy(pending), response))
    if kind in ('terminal', 'recovered', 'exhausted') and not injected:
        raise VariantUnavailable('a sufficient proof settled the request before the selected failure')
    request = {'genes': raw_genes, 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'directions': directions}
    if failed:
        pending, response = failed
        result = {'failed_tool': 'get_neighbors', 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': proof.coverage}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result, 'continuation_decision': 'defer',
                  'blocked_reason': 'The neighbor backend must recover before the remaining proof can be completed.',
                  'summary': 'A required neighbor acquisition failed; retain supported observations and primitive coverage.'}
        thought = (f"The required {pending['direction']} acquisition for {pending['gene_id']} failed. "
                   + ('The retry allowance is exhausted. ' if kind == 'exhausted' else 'No retry allowance was supplied. ')
                   + f"Keep {len(used)} attributed partials and coverage {proof.coverage}. "
                   'A partial or failed list cannot become an empty shared set. Defer for backend recovery.')
    elif proof.issue_code is not None:
        result = _blocked_result(request, proof.issue_code, used, proof.coverage)
        answer = {'status': 'not_found', 'provenance': _provenance(used), 'result': result, 'continuation_decision': 'defer',
                  'blocked_reason': 'A required query gene is absent from the active multiplex; its neighbor predicate is unavailable.',
                  'summary': 'Confirmed global absence blocks the shared-neighbor operation; no shared IDs or count are computed.'}
        thought = (f"The visible prerequisite fails with {proof.issue_code}. Keep the actual unavailable neighbor or global membership record. "
                   'An empty opposite set cannot make an undefined neighbor predicate available. Stop unnecessary acquisitions and defer.')
    else:
        result = proof.result
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
                  'summary': 'The exact shared-neighbor set follows from the visible compatible observations.'}
        thought = ('No new call is needed because the supplied proof is sufficient. ' if len(messages) == 3 else '') + _reason(result, framing, used)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.7', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'queries': arguments['queries'],
        'input_genes': raw_genes, 'directions': directions, 'neighbor_counts': list(fact.neighbor_counts),
        'shared_count': fact.shared_count, 'category': fact.category, 'framing': framing, 'variant': variant,
        'transport': transport, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][_kind(directions)]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view: GraphView, fact: SharedFact, kind: str) -> tuple[dict, str, str]:
    """Preserve literal missing and malformed role inputs before graph acquisition."""
    request = {'genes': [gene for gene, _ in fact.queries], 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'directions': [direction for _, direction in fact.queries]}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'unknown_multiplex': 'UNKNOWN_MULTIPLEX',
             'missing_scope': 'MISSING_SCOPE', 'missing_genes': 'MISSING_GENES',
             'invalid_pair': 'INVALID_QUERY_PAIR', 'invalid_directions': 'DIRECTION_UNAVAILABLE',
             'ranking_required': 'SHARED_NEIGHBOR_RANKING_REQUIRED'}
    if kind not in codes:
        raise ValueError('unsupported shared-neighbor prerequisite diagnostic')
    if kind == 'unknown_layer':
        request['layer_id'] = 'TFs:missing_context'
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_shared_multiplex'
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_pair':
        request['genes'] = request['genes'][:1]
    elif kind == 'invalid_directions':
        request['directions'] = (['undirected', 'undirected'] if view.layers[fact.layer_id].directed
                                 else ['out', 'out'])
    status = ('not_found' if kind in ('unknown_layer', 'unknown_multiplex') else
              'ambiguous' if kind == 'ranking_required' else 'invalid_input')
    return request, status, codes[kind]


def build_diagnostic(view: GraphView, fact: SharedFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'missing_genes', seed: int = 20261028) -> dict:
    """Reject an unresolved prerequisite without inventing an empty intersection."""
    from .s2_7_prompts import render_s2_7_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    core = render_s2_7_diagnostic_request(contract['_requests'], split=fact.split, kind=kind,
        variant_index=variant_index, **request)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    answer = {'status': status, 'provenance': {'registry': [], 'graph': []},
              'result': _blocked_result(request, code, [], None), 'continuation_decision': 'defer',
              'blocked_reason': 'Two valid roles, registered scope, and a defined selection rule are required.',
              'summary': 'A request prerequisite is unresolved; no shared set or shared count was computed.'}
    return {'row_id': 's27_' + digest([contract['contract_version'], view.graph_version, user, code])[:32],
        'messages': [{'role': 'system', 'content': contract['system_prompt']},
            {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
            {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking':
                f"The literal request is {request}. Its prerequisite fails with {code}. "
                'Keep the original JSON values and role order. A missing or malformed role is not an empty neighbor set. '
                'An undefined ranking cannot justify an invented selection. Make no graph call and defer.', 'content': answer}],
        'tools': contract['tools'], 'metadata': {
            'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
            'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
            'question_family': 'S2.7', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
            'graph_version': view.graph_version, 'layer_id': fact.layer_id,
            'queries': _arguments(view, fact)['queries'], 'input_genes': request['genes'],
            'directions': request['directions'], 'neighbor_counts': None, 'shared_count': None,
            'category': fact.category, 'framing': framing, 'variant': kind, 'transport': None,
            'wrapper_index': wrapper,
            'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][kind]),
            'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: NeighborIndex | None = None) -> None:
    """Replay source acquisitions and grade only the proof visible in the row.

    Private source counts validate sampling. They cannot complete a partial
    neighbor set, establish an unseen membership, or settle a failed request.
    Summary wording and blocker wording remain free prose.
    """
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_7_prompts import render_s2_7_request, render_s2_7_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.7 requires one user turn and explicit harness context')
    for message in messages:
        expected = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                    {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                    {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected or message['role'] == 'assistant' and not message['thinking'].strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the shared-neighbor overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.7'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('shared-neighbor row identity differs from the build contract')
    index = index or NeighborIndex(view)
    if (not isinstance(meta['queries'], list) or len(meta['queries']) != 2
            or any(set(query) != {'gene_id', 'direction'} for query in meta['queries'])):
        raise ValueError('two canonical query roles are required')
    fact = fact_from_queries(view, index, meta['layer_id'],
        [(query['gene_id'], query['direction']) for query in meta['queries']], meta['selection_seed'],
        split=meta['split'], category=meta['category'])
    if fact.split_unit != meta['split_unit']:
        raise ValueError('the ordered query roles differ from their fixed split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if ((answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason'])
            or answer['blocked_reason'] is not None and not isinstance(answer['blocked_reason'], str)):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if meta['framing'] not in FRAMINGS or not diagnostic and meta['variant'] not in VARIANTS:
        raise ValueError('unsupported framing or contrast')
    request_templates = (contract['_requests']['diagnostic_requests'][meta['split']][meta['variant']] if diagnostic
                         else contract['_requests'][meta['split']][_kind([direction for _, direction in fact.queries])])
    if any(type(meta[key]) is not int or not 0 <= meta[key] < size for key, size in (
            ('wrapper_index', len(bank[meta['split']][meta['framing']])),
            ('request_template_index', len(request_templates)))):
        raise ValueError('wrapper and request indices must select their audited banks')
    if diagnostic:
        request, _, _ = _diagnostic_request(view, fact, meta['variant'])
        if meta['input_genes'] != request['genes'] or meta['directions'] != request['directions']:
            raise ValueError('the diagnostic changes the literal role input')
        core = render_s2_7_diagnostic_request(contract['_requests'], split=meta['split'], kind=meta['variant'],
            variant_index=meta['request_template_index'], **request)
    else:
        if (canonical_json(list(fact.neighbor_counts)) != canonical_json(meta['neighbor_counts'])
                or canonical_json(fact.shared_count) != canonical_json(meta['shared_count'])):
            raise ValueError('private sampling labels differ from the source neighbor sets')
        genes = [gene for gene, _ in fact.queries]
        if meta['directions'] != [direction for _, direction in fact.queries]:
            raise ValueError('the raw directions change the canonical role bindings')
        if meta['variant'] == 'symbol_context':
            if registry is None or list(get_symbols(genes, registry)) != meta['input_genes']:
                raise ValueError('symbols do not uniquely identify the canonical roles')
        elif meta['input_genes'] != genes:
            raise ValueError('the raw IDs change ordered roles or duplicates')
        if meta['transport'] != _transport(view, index, fact, meta['variant']):
            raise ValueError('transport differs from the declared page contrast')
        core = render_s2_7_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'],
            multiplex_id=view.multiplex_id, layer_id=fact.layer_id, genes=meta['input_genes'], directions=meta['directions'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != _active(view):
        raise ValueError('active scope metadata changed')
    if diagnostic:
        if (context['observations'] or len(messages) != 4 or meta['transport'] is not None
                or meta['neighbor_counts'] is not None or meta['shared_count'] is not None):
            raise ValueError('malformed local prerequisite diagnostic')
        expected = build_diagnostic(view, fact, meta['framing'], meta['request_template_index'], bank, contract,
            kind=meta['variant'], seed=meta['selection_seed'])['messages'][-1]['content']
        if any(canonical_json(answer[key]) != canonical_json(expected[key])
               for key in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('the diagnostic changes the rejected request')
        return
    runtime = NeighborRuntime(view, index=index, **meta['transport'])
    arguments, available = _arguments(view, fact), list(context['observations'])
    identities = [observation for observation in available if observation['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        validate_identity_observations(identities, registry)
        distinct_genes = list(dict.fromkeys(gene for gene, _ in fact.queries))
        if ([observation['result']['gene_id'] for observation in identities] != distinct_genes
                or [observation['arguments']['gene_symbol'] for observation in identities] != list(get_symbols(distinct_genes, registry))):
            raise ValueError('required supplied identity evidence is missing or changed')
    elif identities:
        raise ValueError('unexpected identity context')
    evidence_ids = {}

    def verify(observation: Mapping[str, Any]) -> None:
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected saved-observation fields')
        for records in observation['provenance'].values():
            for record in records:
                evidence_id = record['evidence_id']
                if evidence_id in evidence_ids and canonical_json(evidence_ids[evidence_id]) != canonical_json(observation):
                    raise ValueError('one evidence ID names conflicting acquisitions')
                evidence_ids[evidence_id] = observation
        tool = observation['operation']
        if tool == 'lookup_human_gene_symbol':
            return
        if tool not in ('get_neighbors', 'check_multiplex_membership') or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed graph observation')
        source_args, graph = observation['arguments'], observation['provenance']['graph'][0]
        genes = source_args.get('gene_ids', [source_args.get('gene_id')])
        if any(gene_group(gene, meta['selection_seed'])[1] != meta['split'] for gene in genes):
            raise ValueError('a supplied query subject crossed split boundaries')
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        expected = (_global_observation(view, genes, graph['evidence_id']) if tool == 'check_multiplex_membership'
                    else _observation(tool, source_args, runtime.get_neighbors(**source_args, evidence_id=graph['evidence_id'])))
        if canonical_json(observation) != canonical_json(expected):
            raise ValueError('an observation differs from its pinned source or page schedule')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available)
        call = message['tool_call']
        if (proof.next_arguments is None
                or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments})):
            raise ValueError('a call changes the required primitive order, selection, or cursor')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes the failed arguments')
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('invalid fault schedule or invented failed-list evidence')
            consecutive_failures += 1
        else:
            if set(response) != {'status', 'provenance', 'result'}:
                raise ValueError('unexpected successful response fields')
            consecutive_failures = 0
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
        previous_call, previous_response = call, response
    proof, used = _plan(view, arguments, available)
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from the supporting visible proof')
    if previous_response and previous_response['status'] == 'error':
        if proof.next_arguments is None or previous_response == _failure(1):
            raise ValueError('an error requires an unresolved terminal acquisition failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    elif proof.issue_code is not None:
        request = {'genes': meta['input_genes'], 'multiplex_id': view.multiplex_id,
                   'layer_id': fact.layer_id, 'directions': meta['directions']}
        expected = _blocked_result(request, proof.issue_code, used, proof.coverage)
        status, decision = 'not_found', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes a sufficient exact-intersection proof')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer changes the visible intersection or task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028,
                 contract_path: Path = CONTRACT_PATH, source_manifest: Path | None = None) -> dict:
    """Export the isolated verification corpus with exact token and source audits."""
    from .s2_7_export import build_corpus as export
    return export(output_dir, counts=counts, smoke=smoke, seed=seed,
                  contract_path=contract_path, source_manifest=source_manifest)
