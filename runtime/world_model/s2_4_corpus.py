"""Build S2.4 maps from pinned sources and validate their visible evidence.

Query subjects use the S2.3 split policy across scopes and directions. Native
edges can occur in several splits; this policy does not claim edge separation.
Verification retains large maps and reports their actual token costs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import combinations
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import BUILDER_MODULES, load_s2_prompt_banks, load_s2_edge_requests, render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _active, _observation
from .s2_2_identity import RegistryContext, get_symbols, identity_observations, load_registry_context, validate_identity_observations
from .s2_3_corpus import gene_group, _provenance
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure, _jsonl, _write_immutable
from .s2_multiplex import MultiplexRuntime, SelectionError
from .s2_neighbor_layers import NeighborLayersRuntime
from .s2_neighbors import NeighborIndex, NeighborRuntime
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_4_v1.json')
VARIANTS = ('lookup', 'scope_order', 'context', 'complete_context', 'bounded_context', 'shorter_prefix_context',
            'projected_context', 'broader_bounded_context', 'duplicate_context', 'prefix_context',
            'paged', 'overlap', 'empty_page', 'unknown_total', 'terminal', 'recovered', 'exhausted',
            'native_context', 'native_in_context', 'native_terminal', 'native_recovered', 'native_exhausted',
            'symbol_context', 'stale')


@dataclass(frozen=True)
class MapFact:
    """One source-selected query before wording, transport, and fault contrasts."""

    split: str
    gene_id: str
    layer_scope: dict[str, Any]
    direction: str
    category: str
    unique_neighbor_count: int | None
    supporting_layer_counts: tuple[int, ...]
    split_unit: str


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the separate map overlay without changing earlier family contracts."""
    from .s2_4_prompts import load_s2_4_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.4']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.4 build contract')
    contract['_requests'] = load_s2_4_requests(contract['request_bank'])
    return contract


def scope_grid(runtime: NeighborLayersRuntime) -> list[tuple[dict, str]]:
    """Cover all explicit subsets and exhaustive registered family selectors."""
    layers = runtime.view.layer_order
    scopes = [{'kind': 'layers', 'layer_ids': list(group), 'layer_family': None}
              for size in range(1, len(layers) + 1) for group in combinations(layers, size)]
    scopes.extend({'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in (None, *runtime.multiplex.families))
    grid = []
    for scope in scopes:
        for direction in ('incident', 'in', 'out', 'undirected'):
            try:
                runtime.resolve_scope(scope, direction)
            except SelectionError:
                continue
            grid.append((scope, direction))
    return grid


def _stats(runtime: NeighborLayersRuntime, gene: str, scope: dict, direction: str) -> tuple:
    """Measure actual source strata without using them as runtime evidence."""
    view = runtime.view
    selected = runtime.resolve_scope(scope, direction)
    if gene not in view.multiplex_nodes:
        return 'outside_multiplex', None, (), ()
    supports: dict[str, set[str]] = defaultdict(set)
    features = set()
    present = [layer for layer in selected if gene in view.layers[layer].nodes]
    for layer in selected:
        sides = runtime.index.directions(layer) if direction == 'incident' else (direction,)
        by_side = {}
        for side in sides:
            entries = runtime.index.neighbors(layer, gene, side)
            by_side[side] = {neighbor for neighbor, _ in entries}
            for neighbor, weight in entries:
                supports[neighbor].add(layer)
                if neighbor == gene:
                    features.add('self_loop')
                if weight == 0:
                    features.add('zero_weight')
        if (by_side.get('in', set()) & by_side.get('out', set())) - {gene}:
            features.add('reciprocal_neighbor')
    support_counts = tuple(sorted({len(layers) for layers in supports.values()}))
    if any(n > 1 for n in support_counts):
        features.add('overlapping_support')
    category = ('empty_scope' if not selected else 'absent_all' if not present else
                'absent_some' if len(present) < len(selected) else 'present')
    return category, len(supports), support_counts, tuple(sorted(features))


def _fact(runtime: NeighborLayersRuntime, gene: str, scope: dict, direction: str, seed: int) -> MapFact:
    """Bind a measured query to the shared deterministic subject split."""
    unit, split = gene_group(gene, seed)
    category, count, supports, _ = _stats(runtime, gene, scope, direction)
    return MapFact(split, gene, deepcopy(scope), direction, category, count, supports, unit)


def select_facts(view: GraphView, index: NeighborIndex, registry_ids: Sequence[str],
                 counts: Mapping[str, int], seed: int, *, smoke: bool = False) -> tuple[list[MapFact], dict]:
    """Audit real count domains, then sample equally within each selected stratum.

    Smoke selects zero, the smallest positive count, a small pageable count,
    and the maximum in each split. Other count values remain explicit omissions.
    An explicit count recipe visits every attainable count and membership class.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts require positive integers for all three splits')
    runtime = NeighborLayersRuntime(view, index=index)
    outside = sorted(set(registry_ids) - view.multiplex_nodes)
    if not outside:
        raise ValueError('the registry has no globally missing query candidates')
    selected, domains, cells, features, omissions = {}, [], [], [], []

    def add(gene: str, scope: dict, direction: str) -> None:
        selected[(gene, canonical_json(scope), direction)] = _fact(runtime, gene, scope, direction, seed)

    for scope, direction in scope_grid(runtime):
        pools: dict[tuple, list[str]] = defaultdict(list)
        feature_pools: dict[tuple, list[str]] = defaultdict(list)
        histogram, support_histogram = Counter(), Counter()
        gene_stats = {}
        for gene in sorted(view.multiplex_nodes):
            category, n, supports, flags = _stats(runtime, gene, scope, direction)
            gene_stats[gene] = n
            split = gene_group(gene, seed)[1]
            pools[(split, category, n)].append(gene)
            histogram[n] += 1
            for support in supports:
                feature_pools[(split, f'support_count_{support}')].append(gene)
                support_histogram[support] += 1
            for feature in flags:
                feature_pools[(split, feature)].append(gene)
        domain = {'layer_scope': scope, 'direction': direction,
                  'selected_layer_ids': list(runtime.resolve_scope(scope, direction)),
                  'unique_neighbor_count_histogram': dict(sorted(histogram.items())),
                  'support_count_query_histogram': dict(sorted(support_histogram.items())),
                  'multiplex_members': len(view.multiplex_nodes)}
        domains.append(domain)
        for split in SPLITS:
            available = sorted({n for s, _, n in pools if s == split})
            positive = [n for n in available if n > 0]
            pageable = [n for n in positive if 2 <= n <= 8]
            targets = ({0, *positive[:1], *pageable[:1], *available[-1:]} if smoke else set(histogram))
            for (s, category, n), population in sorted(pools.items()):
                if s != split:
                    continue
                pool = sorted(population, key=lambda gene: gene_group(gene, seed)[0])
                requested = counts[split] if n in targets else 0
                for gene in pool[:requested]:
                    add(gene, scope, direction)
                cell = {'layer_scope': scope, 'direction': direction, 'split': split,
                        'category': category, 'unique_neighbor_count': n, 'split_population': len(pool),
                        'requested': requested, 'selected': min(requested, len(pool))}
                cells.append(cell)
                if requested > len(pool):
                    omissions.append({**cell, 'reason': 'insufficient subjects in this fixed split'})
            for (s, feature), population in sorted(feature_pools.items()):
                if s != split:
                    continue
                pool = sorted(population, key=lambda gene: (gene_stats[gene], gene_group(gene, seed)[0]))
                for gene in pool[:counts[split]]:
                    add(gene, scope, direction)
                features.append({'layer_scope': scope, 'direction': direction, 'split': split,
                                 'feature': feature, 'available': len(pool), 'selected': min(len(pool), counts[split])})
            pool = sorted((g for g in outside if gene_group(g, seed)[1] == split), key=lambda gene: gene_group(gene, seed)[0])
            for gene in pool[:counts[split]]:
                add(gene, scope, direction)
            if len(pool) < counts[split]:
                omissions.append({'layer_scope': scope, 'direction': direction, 'split': split,
                                  'category': 'outside_multiplex', 'requested': counts[split], 'selected': len(pool)})
    facts = sorted(selected.values(), key=lambda f: (SPLITS.index(f.split), canonical_json(f.layer_scope),
                                                   f.direction, f.category, f.split_unit))
    uncovered = []
    for domain in domains:
        actual = {f.unique_neighbor_count for f in facts if f.layer_scope == domain['layer_scope']
                  and f.direction == domain['direction'] and f.unique_neighbor_count is not None}
        uncovered.append({'layer_scope': domain['layer_scope'], 'direction': domain['direction'],
                          'unselected_count_values': sorted(set(domain['unique_neighbor_count_histogram']) - actual)})
    return facts, {'domains': domains, 'cells': cells, 'feature_cells': features, 'omissions': omissions,
                   'smoke_unselected_count_values': uncovered, 'selected_facts': len(facts),
                   'distinct_query_genes': len({f.gene_id for f in facts}),
                   'fact_selection_sha256': digest([asdict(f) for f in facts]),
                   'split_policy': 'SHA256(seed,query gene) bins 8:1:1 across all scopes and directions; same as S2.3',
                   'edge_fact_disjoint': False,
                   'scope_policy': 'every nonempty explicit subset and each exhaustive all/family selector in the pinned view',
                   'conditioning': 'equal requested quota per attainable count and membership class within scope/direction/split; feature supplements reported separately'}


def _arguments(view: GraphView, fact: MapFact, mode: str, top_k: int | None) -> dict:
    """Construct the exact seven-field request with manifest-ordered layer IDs."""
    runtime = MultiplexRuntime(view)
    scope = runtime._scope(fact.layer_scope, runtime.resolve_scope(fact.layer_scope))
    return {'multiplex_id': view.multiplex_id, 'gene_id': fact.gene_id, 'layer_scope': scope,
            'direction': fact.direction, 'mode': mode, 'top_k': top_k, 'cursor': None}


def _transport(fact: MapFact, variant: str) -> dict:
    """Declare transport contrasts independently of the request wording."""
    n = fact.unique_neighbor_count or 0
    transport = {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}
    if variant in ('paged', 'prefix_context', 'terminal', 'recovered', 'exhausted', 'unknown_total', 'empty_page'):
        transport['page_size'] = max(1, (n + 1) // 2)
    if variant == 'overlap':
        if n < 3:
            raise VariantUnavailable('overlap requires at least three unique neighbors')
        transport.update(page_size=max(2, (n + 1) // 2), overlap=1)
    if variant == 'empty_page':
        if n < 2:
            raise VariantUnavailable('an intermediate empty page requires a multi-page traversal')
        transport['empty_page_at'] = 1
    if variant == 'unknown_total':
        transport['disclose_total'] = False
    return transport


def _plan(view: GraphView, arguments: dict, observations: Sequence[dict]):
    """Retain validated identity evidence beside the purely visible graph proof."""
    from .s2_4_evidence import collect
    proof = collect(view, arguments, observations)
    identities = [o for o in observations if o['operation'] == 'lookup_human_gene_symbol']
    used = [*identities, *proof.observations]
    return proof, used


def _reason(result: dict, framing: str, used: Sequence[dict]) -> str:
    """Explain visible union support, selection coverage, and the user's purpose."""
    query = result['query_gene_id']
    parts = [f"The request concerns {query}, scope {result['layer_scope']}, direction {result['direction']}. "]
    if result['missing_gene_ids']:
        return ''.join(parts) + 'The observed global membership is absent. Adjacency and counts remain null. Preserve membership evidence and defer.'
    collection = result['collection']
    parts.append(f"The visible acquisitions supply {len(result['neighbors'])} unique neighbor rows. "
                 f"The full total is {result['unique_neighbor_count']} and collection.complete is {str(collection['complete']).lower()}. ")
    parts.append('Sort rows by canonical ID, then supports by manifest layer. Keep incoming edges before outgoing edges and preserve each weight. '
                 'A reciprocal pair counts as one layer support; a self-loop occurs once. Never add layer counts or compare weights across layers. ')
    if result['neighbors']:
        first = result['neighbors'][0]
        parts.append(f"The first row is {first['gene_id']} with {first['supporting_layer_count']} layer supports: {first['supporting_layers']}. ")
    parts.append(f"Layer absence is {result['layers_without_gene']}; layer coverage is {result['coverage']}. ")
    if any(o['operation'] == 'get_neighbors' for o in used):
        parts.append('Each required native list is complete. Regroup their visible union and calculate its distinct count before any canonical prefix. ')
    elif collection['mode'] == 'complete' and all(o['result'].get('unique_neighbor_count') is None for o in used if o['operation'] == 'get_neighbor_layers'):
        parts.append('No page supplied a total. The complete traversal establishes the total from the accumulated unique rows. ')
    if collection['mode'] == 'top_k':
        parts.append('The bounded traversal satisfies the requested prefix; omitted rows remain unknown unless full exhaustion is proved. ')
    parts.append('Retain the original acquisition evidence. ' + ('This completes the requested workflow step, so continue.' if framing == 'step'
                                                                else 'This completes the standalone request, so stop.'))
    return ''.join(parts)


def build_row(view: GraphView, fact: MapFact, mode: str, top_k: int | None, framing: str,
              variant: str, variant_index: int, bank: Mapping[str, Any], contract: Mapping[str, Any], *,
              index: NeighborIndex | None = None, registry: RegistryContext | None = None, seed: int = 20261028) -> dict:
    """Build a map trace with exact evidence reuse, page traversal, and retries."""
    from .s2_4_prompts import render_s2_4_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported S2.4 contrast or framing')
    if variant == 'scope_order':
        scope = fact.layer_scope
        if scope['kind'] != 'layers' or len(scope['layer_ids']) < 2:
            raise VariantUnavailable('scope-order contrasts require multiple explicit layers')
        fact = replace(fact, layer_scope={**scope, 'layer_ids': [*reversed(scope['layer_ids']), scope['layer_ids'][-1]]})
    transport = _transport(fact, variant)
    runtime = NeighborLayersRuntime(view, index=index, **transport)
    native = NeighborRuntime(view, index=runtime.index)
    arguments = _arguments(view, fact, mode, top_k)
    input_gene = fact.gene_id
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('supplied symbol context requires the pinned registry')
        try:
            input_gene = get_symbols([fact.gene_id], registry)[0]
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = render_s2_4_request(contract['_requests'], split=fact.split, direction=fact.direction, mode=mode,
        variant_index=variant_index, multiplex_id=view.multiplex_id, layer_scope=fact.layer_scope, gene=input_gene, top_k=top_k)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    row_id = 's24_' + digest([contract['contract_version'], view.graph_version, asdict(fact), mode, top_k, framing, variant, user])[:32]
    observations, serial = [], 0

    def acquire(tool: str, call_arguments: dict) -> dict:
        nonlocal serial
        serial += 1
        method = runtime.get_neighbor_layers if tool == 'get_neighbor_layers' else native.get_neighbors
        return method(**call_arguments, evidence_id=f'ev_{row_id}_{serial}')

    def save(tool: str, saved: dict, *, first_only: bool = False) -> None:
        while True:
            response = acquire(tool, saved)
            observations.append(_observation(tool, deepcopy(saved), response))
            if first_only or response['status'] != 'resolved' or response['result']['collection']['next_cursor'] is None:
                return
            saved = {**saved, 'cursor': response['result']['collection']['next_cursor']}

    if variant == 'symbol_context':
        observations.extend(identity_observations([fact.gene_id], row_id, registry))
    elif variant in ('context', 'complete_context', 'bounded_context', 'shorter_prefix_context', 'stale', 'duplicate_context'):
        saved = dict(arguments)
        if variant in ('complete_context', 'stale', 'duplicate_context'):
            saved.update(mode='complete', top_k=None)
        elif variant in ('bounded_context', 'shorter_prefix_context'):
            saved.update(mode='top_k', top_k=1 if variant == 'bounded_context' else (top_k or 1) + 1)
        if variant == 'shorter_prefix_context' and mode != 'top_k':
            raise VariantUnavailable('a shorter-prefix contrast requires a bounded request')
        save('get_neighbor_layers', saved)
        if variant == 'stale':
            for observation in observations:
                observation['provenance']['graph'][0]['graph_version'] = '0' * 64
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
    elif variant in ('projected_context', 'broader_bounded_context'):
        selected = set(runtime.resolve_scope(fact.layer_scope, fact.direction))
        eligible = [layer for layer in view.layer_order if fact.direction == 'incident'
                    or view.layers[layer].directed == (fact.direction in ('in', 'out'))]
        if not selected or selected == set(eligible) or fact.category == 'outside_multiplex':
            raise VariantUnavailable('scope projection requires a proper compatible subset and global membership')
        saved = {**arguments, 'layer_scope': {'kind': 'layers', 'layer_ids': eligible, 'layer_family': None},
                 'mode': 'complete' if variant == 'projected_context' else 'top_k',
                 'top_k': None if variant == 'projected_context' else 1}
        save('get_neighbor_layers', saved)
        if variant == 'broader_bounded_context' and observations[-1]['result']['collection']['complete']:
            raise VariantUnavailable('the broader context must be a genuinely incomplete bounded map')
    elif variant in ('prefix_context', 'terminal', 'recovered', 'exhausted'):
        save('get_neighbor_layers', arguments, first_only=True)
        if observations[-1]['status'] != 'resolved' or not observations[-1]['result']['collection']['next_cursor']:
            raise VariantUnavailable('this contrast requires an unfinished saved map page')
    elif variant.startswith('native_'):
        inputs = runtime.native_inputs(arguments)
        if not inputs or fact.category == 'outside_multiplex':
            raise VariantUnavailable('native reuse requires selected collections and a global member')
        if variant == 'native_in_context':
            entry = next((entry for entry in inputs if entry['direction'] == 'in'
                          and fact.gene_id in view.layers[entry['layer_id']].nodes), None)
            if entry is None or fact.direction != 'incident':
                raise VariantUnavailable('incoming context needs a directed member under an incident request')
        else:
            entry = inputs[0]
        save('get_neighbors', {'multiplex_id': view.multiplex_id, 'gene_id': fact.gene_id, **entry,
                               'mode': 'complete', 'top_k': None, 'cursor': None})
    proof, _ = _plan(view, arguments, observations)
    if variant in ('bounded_context', 'broader_bounded_context') and proof.next_arguments is None:
        raise VariantUnavailable('the saved bounded context already supplies this request')
    if variant in ('native_terminal', 'native_recovered', 'native_exhausted') and proof.next_arguments is None:
        raise VariantUnavailable('native context already supplies the whole map')
    context = {'active_graph': _active(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    failure_kind = variant.removeprefix('native_')
    injected, failed = False, None
    while True:
        proof, used = _plan(view, arguments, observations)
        if proof.next_arguments is None:
            break
        pending, tool = proof.next_arguments, proof.next_tool
        thought = (f"The request asks for {mode} multiplex neighbors of {fact.gene_id}, direction {fact.direction}, "
                   f"scope {fact.layer_scope}, K={top_k}. The next required acquisition is {tool} with {pending}. "
                   'Use only matching graph-version evidence. Each returned row needs complete selected-layer support. '
                   'Copy any returned cursor exactly and keep the selection fixed.')
        if variant == 'symbol_context':
            thought = f'The supplied registry observation uniquely maps {input_gene} to {fact.gene_id}. ' + thought

        def append(response: dict, reasoning: str) -> None:
            messages.extend([{'role': 'assistant', 'thinking': reasoning, 'tool_call': {'name': tool, 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': tool, 'content': response}])

        if not injected and failure_kind in ('terminal', 'recovered', 'exhausted'):
            injected = True
            response = _failure(None if failure_kind == 'terminal' else 1)
            append(response, thought)
            if failure_kind != 'terminal':
                response = _failure(0) if failure_kind == 'exhausted' else acquire(tool, pending)
                append(response, f'The failed {tool} call supplied retries_remaining=1. Repeat the identical arguments, including the cursor. '
                                 'Preserve earlier independent evidence; stop retries after recovery or exhaustion.')
            if response['status'] == 'error':
                failed = (tool, pending, response)
                break
        else:
            response = acquire(tool, pending)
            append(response, thought)
        observations.append(_observation(tool, deepcopy(pending), response))
    if failed:
        tool, pending, response = failed
        result = {'failed_tool': tool, 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': proof.coverage}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'defer', 'blocked_reason': 'A required neighbor acquisition is unavailable.',
                  'summary': 'The required acquisition failed; preserve the valid partial observations.'}
        thought = (f"The required acquisition for {fact.gene_id} failed. "
                   + ('The retry allowance is exhausted. ' if failure_kind == 'exhausted' else 'No retry allowance was supplied. ')
                   + f"Retain {len(used)} observations and coverage {proof.coverage}. "
                   'The failed call adds no evidence. An unfinished native list remains unchecked; map pages retain their actual collection bounds. '
                   'Do not invent a full-scope map or total. Defer for backend recovery.')
    else:
        result = proof.result
        missing = bool(result['missing_gene_ids'])
        answer = {'status': 'not_found' if missing else 'resolved', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'defer' if missing else 'continue' if framing == 'step' else 'stop',
                  'blocked_reason': 'The queried gene is outside the active multiplex.' if missing else None,
                  'summary': 'The queried gene has no multiplex membership; its map is unavailable.' if missing else
                             'The requested unique neighbors retain every selected-layer support and native edge weight.'}
        thought = ('Compatible saved observations suffice; no new call is required. ' if len(messages) == 3 else '') + _reason(result, framing, used)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.4', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': input_gene,
        'layer_scope': fact.layer_scope, 'direction': fact.direction, 'category': fact.category,
        'unique_neighbor_count': fact.unique_neighbor_count, 'supporting_layer_counts': list(fact.supporting_layer_counts),
        'mode': mode, 'top_k': top_k, 'ranking': None, 'framing': framing, 'variant': variant, 'transport': transport,
        'wrapper_index': wrapper, 'request_template_index': variant_index % len(contract['_requests'][fact.split][mode][fact.direction]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def build_diagnostic(view: GraphView, fact: MapFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     ranking: bool = False, seed: int = 20261028) -> dict:
    """Reject a visible representation mismatch or an undefined cross-layer rank."""
    from .s2_4_prompts import render_s2_4_request
    direction = 'incident' if ranking else 'in'
    if not ranking and not any(not view.layers[layer].directed for layer in MultiplexRuntime(view).resolve_scope(fact.layer_scope)):
        raise VariantUnavailable('direction diagnostics require a selected undirected layer')
    rank = 'strongest_unspecified' if ranking else None
    request = render_s2_4_request(contract['_requests'], split=fact.split, direction=direction, mode='complete',
        variant_index=variant_index, multiplex_id=view.multiplex_id, layer_scope=fact.layer_scope, gene=fact.gene_id, ranking=rank)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    status, code = ('ambiguous', 'NEIGHBOR_RANKING_AMBIGUOUS') if ranking else ('invalid_input', 'DIRECTION_UNAVAILABLE')
    result = {'issue_code': code, 'request': {'gene': fact.gene_id, 'multiplex_id': view.multiplex_id,
              'layer_scope': fact.layer_scope, 'direction': direction, 'mode': None if ranking else 'complete',
              'top_k': None, 'cursor': None, 'ranking': rank}, 'rejected_arguments': None, 'partial_results': []}
    answer = {'status': status, 'provenance': {'registry': [], 'graph': []}, 'result': result,
              'continuation_decision': 'defer', 'blocked_reason': 'A cross-layer ranking rule is required.' if ranking else
              'The selected scope includes an undirected representation.',
              'summary': 'The request needs a cross-layer ranking rule.' if ranking else 'The selected scope cannot supply incoming roles.'}
    thought = (f"The request concerns {fact.gene_id}, scope {fact.layer_scope}. " +
               ('The request asks for strongest multiplex neighbors without a ranking rule. Source weights have different meanings; '
                'do not compare or combine them. Preserve the unresolved selection and defer for a ranking rule.' if ranking else
                'The visible manifest contains an undirected layer. Incoming roles require directed representations throughout the selected scope. '
                'Do not omit a layer, invent direction, or call the graph. Preserve the request and defer for a compatible direction.'))
    return {'row_id': 's24_' + digest([view.graph_version, user, code])[:32],
            'messages': [{'role': 'system', 'content': contract['system_prompt']},
                         {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
                         {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking': thought, 'content': answer}],
            'tools': contract['tools'], 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.4', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': fact.gene_id,
                'layer_scope': fact.layer_scope, 'direction': direction, 'category': 'diagnostic',
                'unique_neighbor_count': None, 'supporting_layer_counts': [], 'mode': 'complete', 'top_k': None,
                'ranking': rank, 'framing': framing, 'variant': 'ranking_ambiguous' if ranking else 'direction_invalid',
                'transport': {}, 'wrapper_index': wrapper,
                'request_template_index': variant_index % len(contract['_requests']['ranking_requests'][fact.split] if ranking else
                                                             contract['_requests'][fact.split]['complete']['in']),
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, index: NeighborIndex | None = None,
                 bank: Mapping[str, Any], contract: Mapping[str, Any], registry: RegistryContext | None = None) -> None:
    """Bind requests, replay native acquisitions, then grade the visible proof.

    Source replay verifies the observations. The separate evidence planner
    determines the answer without consulting native edges or hidden counts.
    Summary and blocker sentences remain free text.
    """
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_4_prompts import render_s2_4_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(m['role'] == 'user' for m in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.4 requires one user turn and explicit harness context')
    for message in messages:
        expected_keys = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                         {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                         {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected_keys or message['role'] == 'assistant' and not message['thinking'].strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the pinned map contract')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.4'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('map row identity differs from its build contract')
    if gene_group(meta['gene_id'], meta['selection_seed']) != (meta['split_unit'], meta['split']):
        raise ValueError('the query subject crossed its fixed split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    request = render_s2_4_request(contract['_requests'], split=meta['split'], direction=meta['direction'], mode=meta['mode'],
        variant_index=meta['request_template_index'], multiplex_id=view.multiplex_id, layer_scope=meta['layer_scope'],
        gene=meta['input_gene'], top_k=meta['top_k'], ranking=meta['ranking'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=request)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from its audited request or framing')
    context = json.loads(messages[1]['content'])
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != _active(view):
        raise ValueError('active graph metadata differs from the pinned source')
    if meta['variant'] in ('direction_invalid', 'ranking_ambiguous'):
        if context['observations'] or len(messages) != 4 or meta['input_gene'] != meta['gene_id']:
            raise ValueError('malformed local diagnostic')
        fact = MapFact(meta['split'], meta['gene_id'], meta['layer_scope'], 'incident', 'diagnostic', None, (), meta['split_unit'])
        expected = build_diagnostic(view, fact, meta['framing'], meta['request_template_index'], bank, contract,
                                    ranking=meta['variant'] == 'ranking_ambiguous', seed=meta['selection_seed'])['messages'][-1]['content']
        if any(canonical_json(answer[k]) != canonical_json(expected[k]) for k in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('the diagnostic does not preserve the unresolved request')
        return
    runtime = NeighborLayersRuntime(view, index=index, **meta['transport'])
    native = NeighborRuntime(view, index=runtime.index)
    fact = _fact(runtime, meta['gene_id'], meta['layer_scope'], meta['direction'], meta['selection_seed'])
    if (fact.category != meta['category'] or fact.unique_neighbor_count != meta['unique_neighbor_count']
            or list(fact.supporting_layer_counts) != meta['supporting_layer_counts']):
        raise ValueError('private sampling labels differ from source facts')
    arguments = _arguments(view, fact, meta['mode'], meta['top_k'])
    available = list(context['observations'])
    identities = [o for o in available if o['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        if registry is None:
            raise ValueError('symbol validation requires the pinned registry')
        validate_identity_observations(identities, registry)
        if (len(identities) != 1 or identities[0]['result']['gene_id'] != fact.gene_id
                or get_symbols([fact.gene_id], registry)[0] != meta['input_gene']):
            raise ValueError('the supplied symbol does not uniquely identify the subject')
    elif identities or meta['input_gene'] != fact.gene_id:
        raise ValueError('unexpected identity context or changed canonical input')
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
        if tool not in ('get_neighbor_layers', 'get_neighbors') or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed graph observation')
        if gene_group(observation['arguments']['gene_id'], meta['selection_seed'])[1] != meta['split']:
            raise ValueError('a saved query subject crossed split boundaries')
        graph = observation['provenance']['graph'][0]
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        method = runtime.get_neighbor_layers if tool == 'get_neighbor_layers' else native.get_neighbors
        expected = method(**observation['arguments'], evidence_id=graph['evidence_id'])
        if any(canonical_json(observation[k]) != canonical_json(expected[k]) for k in ('status', 'result', 'provenance')):
            raise ValueError('an observation differs from its native source or declared transport')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available)
        call = message['tool_call']
        if proof.next_arguments is None or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments}):
            raise ValueError('a call differs from the required collection or returned cursor')
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
            consecutive_failures = 0
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
        previous_call, previous_response = call, response
    proof, used = _plan(view, arguments, available)
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from the relevant visible acquisitions')
    if answer['status'] == 'error':
        if (proof.next_arguments is None or not previous_response or previous_response['status'] != 'error'
                or previous_response == _failure(1)):
            raise ValueError('an error requires an unresolved terminal failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes completion of the requested acquisition')
        expected = proof.result
        missing = bool(expected['missing_gene_ids'])
        status = 'not_found' if missing else 'resolved'
        decision = 'defer' if missing else 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer exceeds visible evidence or changes task completion')


def _requests(fact: MapFact, smoke: bool = False) -> list[tuple[str, int | None]]:
    """Exercise real union-count boundaries without a hidden output cap."""
    n = fact.unique_neighbor_count or 0
    limits = {1} if smoke and n > 8 else {1, max(1, n // 2), max(1, n), n + 1}
    return [('complete', None), *[('top_k', k) for k in sorted(limits)]]


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None, smoke: bool = False,
                 seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write immutable source-verified maps, evaluation keys, and token audits.

    Explicit counts visit every attainable count group. Smoke retains complete
    maps at count extrema and reports unsampled counts and K values. Both
    schedules produce verification artifacts, without a production budget.
    """
    from .s2_2_prompts import load_s2_2_requests
    from .s2_3_prompts import load_s2_3_requests
    from .s2_3_tokens import load_tokenizer, audit_rows
    from .s2_4_prompts import validate_s2_4_rendered_prompts
    if (counts is None) != smoke:
        raise ValueError('choose explicit counts or smoke=True; no default generation budget exists')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('the source manifest differs from its pin')
    recipe = contract['context_recipe']
    if file_sha256(ROOT / recipe['path']) != recipe['sha256']:
        raise ValueError('the context reference differs from its pin')
    view = load_source_view(source_manifest)
    index = NeighborIndex(view)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_4_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']))
    facts, selection = select_facts(view, index, tuple(sorted(registry.symbols_by_id)), counts, seed, smoke=smoke)
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows = {split: [] for split in SPLITS}
    skipped = Counter()
    panel, order_panel = {}, {}
    # Select semantic contrasts by source counts before assigning any wording.
    for fact in facts:
        n = fact.unique_neighbor_count or 0
        key = (fact.split, fact.direction, fact.category)
        score = (0 if 2 <= n <= 8 else 1, n, canonical_json(fact.layer_scope), fact.split_unit)
        if key not in panel or score < panel[key][0]:
            panel[key] = (score, fact)
        if fact.layer_scope['kind'] == 'layers' and len(fact.layer_scope['layer_ids']) > 1:
            order_key = (fact.split, fact.direction)
            if order_key not in order_panel or score < order_panel[order_key][0]:
                order_panel[order_key] = (score, fact)
    panel_units = {(f.gene_id, canonical_json(f.layer_scope), f.direction) for _, f in panel.values()}
    order_units = {(f.gene_id, canonical_json(f.layer_scope), f.direction) for _, f in order_panel.values()}
    for fact in facts:
        for mode, top_k in _requests(fact, smoke):
            variants = ['lookup']
            if ((fact.gene_id, canonical_json(fact.layer_scope), fact.direction) in order_units
                    and (mode == 'complete' or top_k in (1, 2))):
                variants.append('scope_order')
            if ((fact.gene_id, canonical_json(fact.layer_scope), fact.direction) in panel_units
                    and (mode == 'complete' or top_k in (1, 2))):
                variants = list(VARIANTS)
            for variant in variants:
                for framing in FRAMINGS:
                    ordinal = len(rows[fact.split]) // 2
                    try:
                        row = build_row(view, fact, mode, top_k, framing, variant, ordinal, bank, contract,
                                        index=index, registry=registry, seed=seed)
                    except VariantUnavailable:
                        skipped[f'{variant}/{mode}'] += 1
                        continue
                    validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
                    rows[fact.split].append(row)
    for split in SPLITS:
        for ranking in (False, True):
            fact = next((f for f in facts if f.split == split and f.direction == 'incident'
                         and f.layer_scope['kind'] == 'layers'
                         and any(not view.layers[layer].directed for layer in f.layer_scope['layer_ids'])), None)
            if fact:
                for framing in FRAMINGS:
                    row = build_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, ranking=ranking, seed=seed)
                    validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
                    rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
             'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
             'bound_schedule': 'complete plus K=1,floor(count/2),count,count+1 (positive only); smoke counts above8 use complete and K1',
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    files = {}
    units = {split: {r['metadata']['split_unit'] for r in rows[split]} for split in SPLITS}
    if any(units[left] & units[right] for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate map row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, bounds = Counter(), defaultdict(Counter), defaultdict(set)
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta[k]) for k in ('direction', 'mode', 'category', 'variant'))] += 1
            if meta['mode'] == 'top_k':
                bounds[(canonical_json(meta['layer_scope']), meta['direction'], meta['unique_neighbor_count'])].add(meta['top_k'])
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(units[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_direction_mode_category_variant': {k: dict(sorted(v.items())) for k, v in sorted(joint.items())},
            'variants': dict(sorted(Counter(r['metadata']['variant'] for r in rows[split]).items())),
            'selected_bounds': [{'layer_scope': json.loads(scope), 'direction': direction, 'unique_neighbor_count': n,
                'selected_k': sorted(values), 'relationships': sorted({'unavailable' if n is None else 'below' if k < n else
                 'equal' if k == n else 'above' for k in values})}
                for (scope, direction, n), values in sorted(bounds.items(), key=lambda pair: str(pair[0]))]}
        if split != 'train':
            files[f'eval/{split}.questions.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                 'metadata': {k: row['metadata'][k] for k in ('schema_version', 'tool_contract_version', 'question_family', 'split')}}
                for row in rows[split]])
            files[f'eval/{split}.answers.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'expected_messages': row['messages'][3:], 'final_answer': row['messages'][-1]['content']}
                for row in rows[split]])
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    modules = ('s2_4_corpus', 's2_neighbor_layers', 's2_4_evidence', 's2_4_prompts', 's2_3_corpus', 's2_3_prompts',
               's2_3_tokens', 's2_neighbors', 's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex',
               's2_corpus', 's2_sources', 's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_4.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-4-manifest-v1', 'contract_version': contract['contract_version'],
                'implemented_families': ['S2.4'], 'purpose': 'builder_verification', 'training_ready': False,
                'production_export_allowed': False, 'generation_budget': None if smoke else counts,
                'sampling_schedule': 'representative_smoke' if smoke else 'every_attainable_count',
                'seed': seed, 'row_counts': {s: len(rows[s]) for s in SPLITS}, 'python_version': platform.python_version(),
                'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
                'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
                **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
                                                's23_request_bank', 'registry_index', 'tokenizer', 'context_recipe')},
                'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
                'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
                'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
                'transport_and_faults': 'declared deterministic pages, overlap, empty pages, unknown totals, retries and stale version 0x00; not deployed captures',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
