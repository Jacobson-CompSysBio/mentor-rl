"""Build S2.2 multiplex examples from pinned native records and visible proofs.

The selection inventory is private build metadata. Answers are assembled only
from saved observations and actual tool responses, including bounded coverage.
No production generation budget or training adapter is selected here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import heapq
from itertools import combinations
import json
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import BUILDER_MODULES, load_s2_prompt_banks, load_s2_edge_requests, render_s2_prompt
from .s2_2_prompts import load_s2_2_requests, render_s2_2_request, validate_s2_2_rendered_prompts
from .s2_2_identity import (
    RegistryContext, load_registry_context, get_symbols, identity_observations, validate_identity_observations,
)
from .s2_corpus import (
    ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest,
    _pair_group, _failure, _jsonl, _write_immutable,
)
from .s2_multiplex import CandidateInventory, MultiplexRuntime, EvidenceError, SelectionError
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_2_v1.json')
VARIANTS = ('lookup', 'context', 'full_context', 'partial', 'bounded_context', 'terminal',
            'recovered', 'exhausted', 'stale', 'later_witness', 'symbol_context',
            'reversed_context', 'wrong_pair', 'wrong_predicate', 'broader_context')


@dataclass(frozen=True)
class MultiplexFact:
    """A source-selected query before wording, retrieval, or failures are chosen."""

    split: str
    gene_ids: tuple[str, str]
    layer_scope: dict[str, Any]
    edge_query: str
    support_count: int | None
    split_unit: str


class VariantUnavailable(ValueError):
    """A context contrast would not exercise its declared evidence condition."""


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load independently pinned S2.2 semantics and frozen request wording."""
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION
            or contract['implemented_families'] != ['S2.2']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.2 build contract')
    contract['_requests'] = load_s2_2_requests(contract['request_bank'])
    return contract


def scope_grid(runtime: MultiplexRuntime) -> list[tuple[dict[str, Any], str]]:
    """Cover every explicit subset and every valid family for each predicate."""
    scopes = []
    layers = runtime.view.layer_order
    for size in range(1, len(layers) + 1):
        scopes.extend({'kind': 'layers', 'layer_ids': list(group), 'layer_family': None}
                      for group in combinations(layers, size))
    scopes.extend({'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in (None, *runtime.families))
    result = []
    for scope in scopes:
        for query in ('any_direction', 'source_to_target', 'both_directions'):
            try:
                runtime.resolve_scope(scope, query)
            except SelectionError:
                continue
            result.append((scope, query))
    return result


def select_facts(view: GraphView, registry_ids: Sequence[str], counts: Mapping[str, int],
                 seed: int) -> tuple[list[MultiplexFact], dict[str, Any]]:
    """Sample equally within attainable support counts without fabricating strata.

    Positive pairs are hashed once into the same split units used by S2.1.
    Small deterministic reservoirs per native support mask avoid rescanning the
    full pair population for every subset. Zeros are rejected from the complete
    global-node pair domain; their exact per-split populations are not claimed.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts must supply positive integers for all three splits')
    runtime, inventory = MultiplexRuntime(view), CandidateInventory(view)
    frequencies: Counter = Counter()
    reservoirs: dict[tuple[Any, str], list] = defaultdict(list)
    for masks, indices in inventory.mask_bins.items():
        for index in indices:
            pair = inventory.pairs[index]
            unit, split = _pair_group(pair, seed)
            frequencies[(masks, split)] += 1
            heap = reservoirs[(masks, split)]
            entry = (-int(unit, 16), index)
            if len(heap) < counts[split]:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    outside = sorted(set(registry_ids) - view.multiplex_nodes)
    if not outside:
        raise ValueError('the pinned registry supplies no globally missing endpoints')
    facts, cells, omissions = [], [], []
    for scope, query in scope_grid(runtime):
        layers = runtime.resolve_scope(scope, query)
        mask = sum(1 << view.layer_order.index(layer) for layer in layers)
        histogram = inventory.histogram(layers, query)
        for support, population in histogram.items():
            for split in SPLITS:
                candidates = []
                available = None if support == 0 else 0
                if support:
                    for masks in inventory.mask_bins:
                        for reverse in ((False, True) if query == 'source_to_target' else (False,)):
                            forward, backward = masks[::-1] if reverse else masks
                            if inventory._support(forward, backward, mask, query) != support:
                                continue
                            available += frequencies[(masks, split)]
                            for priority, index in reservoirs[(masks, split)]:
                                pair = inventory.pairs[index]
                                unit = f'{-priority:064x}'
                                if reverse or (query != 'source_to_target' and int(unit[-1], 16) % 2):
                                    pair = pair[::-1]
                                candidates.append((unit, pair))
                    candidates = sorted(set(candidates))[:counts[split]]
                else:
                    rng = random.Random(int(digest([seed, scope, query, split, 'zero']), 16))
                    seen = set()
                    if inventory.pair_universe_size <= 10000:
                        # Exact small-domain feasibility is useful for isolated
                        # fixtures and genuinely small future source views.
                        for pair in combinations(inventory.nodes, 2):
                            unit, actual_split = _pair_group(pair, seed)
                            if actual_split != split:
                                continue
                            for ordered in ((pair, pair[::-1]) if query == 'source_to_target' else (pair,)):
                                if inventory.query_support(ordered, layers, query) == 0:
                                    candidates.append((unit, ordered))
                        available = len(candidates)
                        candidates = sorted(candidates)[:counts[split]]
                    else:
                        for _ in range(max(10000, counts[split] * 1000)):
                            pair = tuple(rng.sample(inventory.nodes, 2))
                            key = pair if query == 'source_to_target' else tuple(sorted(pair))
                            unit, actual_split = _pair_group(pair, seed)
                            if (key in seen or actual_split != split
                                    or inventory.query_support(pair, layers, query)):
                                continue
                            seen.add(key)
                            candidates.append((unit, pair))
                            if len(candidates) == counts[split]:
                                break
                cell = {'layer_scope': scope, 'edge_query': query, 'scope_size': len(layers),
                        'support_count': support, 'split': split,
                        'natural_population': population, 'split_population': available,
                        'requested': counts[split], 'selected': len(candidates)}
                cells.append(cell)
                if len(candidates) < counts[split]:
                    omissions.append({**cell, 'reason': ('zero rejection limit reached; split feasibility remains unknown'
                                                        if available is None else 'insufficient candidates in the fixed pair split')})
                facts.extend(MultiplexFact(split, pair, scope, query, support, unit)
                             for unit, pair in candidates)
        # Membership failures are a separate stratum, never part of h=0.
        for split in SPLITS:
            rng = random.Random(int(digest([seed, scope, query, split, 'missing']), 16))
            selected = set()
            for _ in range(max(10000, counts[split] * 1000)):
                pair = (rng.choice(outside), rng.choice(inventory.nodes))
                if rng.randrange(2):
                    pair = pair[::-1]
                unit, actual_split = _pair_group(pair, seed)
                if actual_split != split or unit in selected:
                    continue
                selected.add(unit)
                facts.append(MultiplexFact(split, pair, scope, query, None, unit))
                if len(selected) == counts[split]:
                    break
            if len(selected) != counts[split]:
                omissions.append({'layer_scope': scope, 'edge_query': query, 'scope_size': len(layers),
                                  'support_count': None, 'split': split, 'requested': counts[split],
                                  'selected': len(selected), 'split_population': None,
                                  'reason': 'missing-membership rejection limit reached; split feasibility remains unknown'})
    facts.sort(key=lambda f: (SPLITS.index(f.split), canonical_json(f.layer_scope),
                             f.edge_query, -1 if f.support_count is None else f.support_count,
                             f.split_unit, f.gene_ids))
    audit = {'global_node_count': len(inventory.nodes),
             'unordered_pair_universe': inventory.pair_universe_size,
             'positive_unordered_pairs': len(inventory.pairs),
             'all_layer_any_direction_histogram': inventory.histogram(view.layer_order),
             'sampling': 'equal requested counts within each attainable scope/predicate/h cell',
             'zero_split_population': ('exact small-domain enumeration' if inventory.pair_universe_size <= 10000 else
                                       'not enumerated; rejection draws use all distinct multiplex nodes'),
             'scope_predicate_count': len(scope_grid(runtime)), 'cells': cells, 'omissions': omissions,
             'selected_facts': len(facts),
             'distinct_unordered_pairs': len({f.split_unit for f in facts}),
             'fact_selection_sha256': digest([asdict(f) for f in facts])}
    return facts, audit


def _arguments(view: GraphView, fact: MultiplexFact, mode: str, top_k: int | None) -> dict[str, Any]:
    return {'multiplex_id': view.multiplex_id, 'gene_ids': list(fact.gene_ids),
            'layer_scope': fact.layer_scope, 'edge_query': fact.edge_query,
            'mode': mode, 'top_k': top_k}


def _scope_text(scope: Mapping[str, Any]) -> str:
    if scope['kind'] == 'layers':
        return 'the explicit layer set [' + ', '.join(scope['layer_ids']) + ']'
    if scope['layer_family']:
        return 'all registered layers in family ' + scope['layer_family']
    return 'all registered layers'


def _active(view: GraphView) -> dict[str, Any]:
    return {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
            'layers': [{'layer_id': layer, 'directed': view.layers[layer].directed}
                       for layer in view.layer_order]}


def _observation(operation: str, arguments: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    return {'operation': operation, 'arguments': deepcopy(arguments),
            **{key: deepcopy(response[key]) for key in ('status', 'result', 'provenance')}}


def _proofs(runtime: MultiplexRuntime, arguments: Mapping[str, Any],
            observations: Sequence[Mapping[str, Any]]) -> tuple[list, list, list, dict]:
    """Extract compatible visible proofs; a stale graph version proves nothing."""
    rows, missing, used = [], [], []
    provenance = {'registry': [], 'graph': []}
    selected = runtime.resolve_scope(arguments['layer_scope'], arguments['edge_query'])
    for observation in observations:
        if observation['operation'] == 'lookup_human_gene_symbol':
            if observation['result']['gene_id'] in arguments['gene_ids']:
                for record in observation['provenance']['registry']:
                    if record not in provenance['registry']:
                        provenance['registry'].append(record)
            continue
        records = observation['provenance']['graph']
        if not records or any(record['graph_version'] != runtime.view.graph_version
                              or record['multiplex_id'] != runtime.view.multiplex_id for record in records):
            continue
        source = observation['result']
        if (source.get('multiplex_id') != arguments['multiplex_id']
                or set(source.get('gene_ids', [])) != set(arguments['gene_ids'])):
            continue
        if observation['operation'] == 'get_edge_layers':
            for row in source['layer_results'] or []:
                if row['layer_id'] not in selected:
                    continue
                try:
                    rows.append(runtime.requery_layer(row, arguments['edge_query'], arguments['gene_ids']))
                except EvidenceError:
                    pass  # Membership remains valid; this predicate is unsettled.
        elif observation['operation'] == 'get_edge':
            if source['layer_id'] not in selected:
                continue
            if not source['missing_gene_ids']:
                try:
                    rows.append(runtime.layer_from_edge(source, arguments['edge_query'], arguments['gene_ids']))
                except EvidenceError:
                    continue
        else:
            continue
        missing.extend(source['missing_gene_ids'])
        used.append(observation)
        for record in records:
            if record not in provenance['graph']:
                provenance['graph'].append(record)
    return rows, sorted(set(missing)), used, provenance


def _remaining(runtime: MultiplexRuntime, arguments: Mapping[str, Any],
               observations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Find required work from visible predicates, without querying the inventory."""
    rows, missing, used, _ = _proofs(runtime, arguments, observations)
    if not used:
        return dict(arguments)  # Even an empty family needs global membership.
    try:
        runtime.assemble_result(arguments['gene_ids'], arguments['layer_scope'],
                                arguments['edge_query'], arguments['mode'], arguments['top_k'], rows, missing)
        return None
    except EvidenceError:
        pass
    selected = runtime.resolve_scope(arguments['layer_scope'], arguments['edge_query'])
    checked = {row['layer_id'] for row in rows}
    remaining = [layer for layer in selected if layer not in checked]
    result = deepcopy(arguments)
    result['layer_scope'] = {'kind': 'layers', 'layer_ids': remaining, 'layer_family': None}
    if arguments['mode'] == 'top_k':
        positives = {row['layer_id'] for row in rows if row['edge_exists']}
        known_matches = [layer for layer in selected if layer in positives]
        if len(known_matches) >= arguments['top_k']:
            boundary = selected.index(known_matches[arguments['top_k'] - 1])
            result['layer_scope']['layer_ids'] = [layer for layer in selected[:boundary + 1] if layer not in checked]
        # Only matches before the first unknown layer can safely reduce the
        # fresh scan bound. Other saved witnesses may lie after an earlier match.
        for layer in selected:
            if layer not in checked:
                break
            result['top_k'] -= layer in positives
    return result


def _coverage(runtime: MultiplexRuntime, arguments: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = runtime.resolve_scope(arguments['layer_scope'], arguments['edge_query'])
    checked = {row['layer_id'] for row in rows}
    return {'operation': 'layer_edge_checks', 'requested_layer_ids': list(selected),
            'checked_layer_ids': [layer for layer in selected if layer in checked],
            'unchecked_layer_ids': [layer for layer in selected if layer not in checked]}


def _reason(result: Mapping[str, Any], framing: str) -> str:
    """Explain the actual observations, limits, and decision at S1 trace depth."""
    pair = ' and '.join(result['gene_ids'])
    if result['missing_gene_ids']:
        return (f"The membership checks place {', '.join(result['missing_gene_ids'])} outside {result['multiplex_id']}. "
                f'There is no evaluable {result["edge_query"]} predicate for {pair}, even if the selected family is empty. '
                'Retain the checked membership evidence; edge existence, layer lists, and counts remain null. '
                'A required endpoint is unavailable, so defer.')
    details = []
    for layer in result['layer_results']:
        connector = ' → ' if layer['directed'] else ' — '
        records = '; '.join(connector.join(edge['gene_ids']) + f" at weight {edge['weight']}" for edge in layer['edges']) or 'no positive observed edges'
        outside = ', '.join(layer['not_in_layer_gene_ids']) or 'neither endpoint'
        details.append(f"{layer['layer_id']} has predicate {str(layer['edge_exists']).lower()}: {records}; "
                       f"{outside} is outside this layer. {len(layer['coverage']['unchecked_pairs'])} requested orientations remain unchecked.")
    if not details:
        details.append('The valid family selects no registered layers, and both endpoints were checked as multiplex members.')
    count = result['supporting_layer_count']
    count_reason = (f'All selected layer predicates are known, so the exact supporting-layer count is {count}.'
                    if count is not None else
                    f"Unchecked layers {result['coverage']['unchecked_layer_ids']} leave the total unknown; the witnessed supports give only a lower bound.")
    if result['mode'] == 'exists':
        completion = ('A positive witness settles the Boolean without proving first position.' if result['edge_exists_any_layer']
                      else 'Exhaustive checked negatives settle the Boolean as false.')
    elif result['mode'] == 'top_k':
        completion = (f"Return the manifest-ordered prefix {result['supporting_layer_ids']}; its returned count is "
                      f"{result['collection']['returned_count']} and complete is {str(result['collection']['complete']).lower()}. "
                      'Earlier potential matches are checked. The bound ranks layer position, not weights across sources.')
    else:
        completion = f"The exhaustive list {result['supporting_layer_ids']} answers the selected scope only."
    ending = 'Continue the stated later workflow.' if framing == 'step' else 'The standalone request is complete, so stop.'
    return (f"For {pair}, apply {result['edge_query']} within {result['coverage']['requested_layer_ids']}. "
            + ' '.join(details) + ' Count distinct qualifying layers; reciprocal arcs retain separate native roles and weights but count once. '
            + count_reason + ' ' + completion + ' Preserve every supporting evidence record. ' + ending)


def build_row(view: GraphView, fact: MultiplexFact, mode: str, top_k: int | None,
              framing: str, variant: str, variant_index: int, bank: Mapping[str, Any],
              contract: Mapping[str, Any], *, seed: int = 20261028,
              registry: RegistryContext | None = None) -> dict[str, Any]:
    """Render a trace whose calls depend on available evidence and retrieval need."""
    if framing not in FRAMINGS or variant not in VARIANTS:
        raise ValueError('unsupported S2.2 framing or variant')
    runtime = MultiplexRuntime(view)
    arguments = _arguments(view, fact, mode, top_k)
    selected = runtime.resolve_scope(fact.layer_scope, fact.edge_query)
    input_genes = fact.gene_ids
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('symbol context needs a verified registry')
        try:
            input_genes = get_symbols(fact.gene_ids, registry)
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = render_s2_2_request(contract['_requests'], split=fact.split, mode=mode,
        edge_query=fact.edge_query, variant_index=variant_index, multiplex_id=view.multiplex_id,
        first=input_genes[0], second=input_genes[1], scope_text=_scope_text(fact.layer_scope), top_k=top_k)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing,
                                    variant_index=variant_index, request=request)
    row_id = 's22_' + digest([contract['contract_version'], view.graph_version, asdict(fact), mode, top_k, framing, variant, user])[:32]
    context = {'active_graph': _active(view), 'observations': []}
    observations = context['observations']

    def save_batch(saved_arguments: dict[str, Any]) -> None:
        response = runtime.get_edge_layers(**saved_arguments, evidence_id='ev_' + row_id + '_saved')
        observations.append(_observation('get_edge_layers', saved_arguments, response))

    if variant == 'context':
        save_batch(arguments)
    elif variant == 'symbol_context':
        observations.extend(identity_observations(fact.gene_ids, row_id, registry))
    elif variant == 'reversed_context':
        save_batch({**arguments, 'gene_ids': list(fact.gene_ids[::-1]), 'mode': 'complete', 'top_k': None})
    elif variant == 'broader_context':
        broader = [layer for layer in view.layer_order if fact.edge_query == 'any_direction' or view.layers[layer].directed]
        if len(broader) <= len(selected):
            raise VariantUnavailable('the selected scope is already the full eligible scope')
        save_batch({**arguments, 'layer_scope': {'kind': 'layers', 'layer_ids': broader, 'layer_family': None},
                    'mode': 'complete', 'top_k': None})
    elif variant == 'wrong_pair':
        others = sorted(view.multiplex_nodes - set(fact.gene_ids))
        if len(others) < 2:
            raise VariantUnavailable('a distractor needs a different real endpoint pair')
        rng = random.Random(int(digest([fact.split_unit, fact.layer_scope, fact.edge_query, 'distractor']), 16))
        distractor = None
        for _ in range(1000):
            pair = rng.sample(others, 2)
            if _pair_group(tuple(pair), seed)[1] == fact.split:
                distractor = pair
                break
        if distractor is None:
            raise VariantUnavailable('no alternate pair found in the same fixed split')
        save_batch({**arguments, 'gene_ids': distractor, 'mode': 'complete', 'top_k': None})
    elif variant == 'wrong_predicate':
        alternate = 'source_to_target' if fact.edge_query != 'source_to_target' else 'any_direction'
        try:
            runtime.resolve_scope(fact.layer_scope, alternate)
        except SelectionError as error:
            raise VariantUnavailable('alternate predicate is invalid in this scope') from error
        save_batch({**arguments, 'edge_query': alternate, 'mode': 'complete', 'top_k': None})
        if _remaining(runtime, arguments, observations) is None:
            raise VariantUnavailable('the alternate predicate already supplies sufficient native observations')
    elif variant in ('full_context', 'stale'):
        save_batch({**arguments, 'mode': 'complete', 'top_k': None})
        if variant == 'stale':
            # A declared stale-context fixture is not a historical source claim.
            observations[0]['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('partial', 'terminal', 'recovered', 'exhausted'):
        if len(selected) < 2 or fact.support_count is None:
            raise VariantUnavailable('partial context needs at least two layers and known endpoints')
        save_batch({**arguments, 'layer_scope': {'kind': 'layers', 'layer_ids': [selected[0]], 'layer_family': None},
                    'mode': 'complete', 'top_k': None})
        if _remaining(runtime, arguments, observations) is None:
            raise VariantUnavailable('the supplied prefix already completes this request')
    elif variant == 'bounded_context':
        if mode != 'complete' or fact.support_count is None or not selected:
            raise VariantUnavailable('bounded saved context contrasts with a complete request')
        save_batch({**arguments, 'mode': 'top_k', 'top_k': 1})
        if _remaining(runtime, arguments, observations) is None:
            raise VariantUnavailable('the bounded observation already exhausted this scope')
    elif variant == 'later_witness':
        if mode != 'exists' or fact.edge_query != 'any_direction' or not fact.support_count or len(selected) < 2:
            raise VariantUnavailable('later witness requires a positive any-direction existence query')
        found = None
        for layer_id in reversed(selected[1:]):
            layer = view.layers[layer_id]
            pairs = (fact.gene_ids, fact.gene_ids[::-1]) if layer.directed else (tuple(sorted(fact.gene_ids)),)
            for pair in pairs:
                if pair in layer.edges:
                    found = {'multiplex_id': view.multiplex_id, 'layer_id': layer_id, 'gene_ids': list(pair)}
                    break
            if found:
                break
        if found is None:
            raise VariantUnavailable('this pair has no later-layer witness')
        observations.append(_observation('get_edge', found, view.get_edge(**found, evidence_id='ev_' + row_id + '_saved')))
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    pending = _remaining(runtime, arguments, observations)
    pair_text = ' then '.join(fact.gene_ids)
    initial = (f'The reference order is {pair_text}; preserve those canonical IDs in graph arguments. '
               f'The requested {fact.edge_query} predicate selects {list(selected)} in {view.multiplex_id}. '
               f'The manifest identifies each native representation. The requested mode is {mode}'
               + (f' with K={top_k}' if top_k else '') + '. ')
    if observations:
        initial += (f'The supplied registry observations uniquely map {list(input_genes)} to {list(fact.gene_ids)} in that order. Preserve both registry records. '
                    if variant == 'symbol_context' else
                    'The saved pair differs from the requested pair, so its graph predicates cannot answer this request. '
                    if variant == 'wrong_pair' else
                    'The saved predicate differs. Reuse every native orientation that settles the requested layer predicate; acquire only layers whose requested predicate remains unproved. '
                    if variant == 'wrong_predicate' else
                    ('The saved reference order is reversed. This changes the source-to-target request, so acquire the requested orientation. '
                     if fact.edge_query == 'source_to_target' else
                     'The saved reference order is reversed, but this symmetric layer predicate is unchanged. Rebase coverage to the current reference order while preserving native edge roles and evidence. ')
                    if variant == 'reversed_context' else
                    'The saved graph version differs from the active version, so it cannot establish current memberships or edges. '
                    if variant == 'stale' else
                    'Reuse matching saved evidence with the active version, pair, scope, and predicate; preserve its actual acquisition record. ')
    if pending:
        initial += (f"Required evidence is still missing for {pending['layer_scope']}. Check that scope with mode "
                    f"{pending['mode']} and top_k={pending['top_k']}; the call checks both multiplex memberships. "
                    'A witness can finish existence, but a complete count needs all layer predicates and a bounded list needs the preceding manifest positions.')

    def call(thought: str, response: Mapping[str, Any]) -> None:
        messages.extend([{'role': 'assistant', 'thinking': thought,
                          'tool_call': {'name': 'get_edge_layers', 'arguments': deepcopy(pending)}},
                         {'role': 'tool', 'name': 'get_edge_layers', 'content': response}])

    failed = variant in ('terminal', 'exhausted')
    if pending:
        if variant in ('terminal', 'recovered', 'exhausted'):
            call(initial, _failure(None if variant == 'terminal' else 1))
            if variant != 'terminal':
                retry = (f"The scan of {pending['layer_scope']} failed without publishing partial graph work. "
                         'The response explicitly grants retries_remaining=1; retry the identical six arguments once. '
                         'The earlier independent observation remains valid.')
                if variant == 'exhausted':
                    call(retry, _failure(0))
                else:
                    response = runtime.get_edge_layers(**pending, evidence_id='ev_' + row_id + '_scan')
                    call(retry, response)
                    observations.append(_observation('get_edge_layers', pending, response))
        else:
            response = runtime.get_edge_layers(**pending, evidence_id='ev_' + row_id + '_scan')
            call(initial, response)
            observations.append(_observation('get_edge_layers', pending, response))
    rows, missing, used, provenance = _proofs(runtime, arguments, observations)
    if failed:
        result = {'failed_tool': 'get_edge_layers', 'error_code': 'BACKEND_UNAVAILABLE',
                  'arguments': pending, 'partial_results': used, 'coverage': _coverage(runtime, arguments, rows)}
        reason = (f"The required scan of {pending['layer_scope']} is unavailable. "
                  + ('The permitted retry is exhausted. ' if variant == 'exhausted' else 'No retry allowance was supplied. ')
                  + f"Keep the independent checked layers {result['coverage']['checked_layer_ids']} and their source records. "
                  + f"The original requested scope still has unchecked layers {result['coverage']['unchecked_layer_ids']}. "
                  'The failed batch supplies no additional coverage, count, or negative observations. Defer for backend recovery.')
        answer = {'status': 'error', 'provenance': provenance, 'result': result,
                  'continuation_decision': 'defer', 'blocked_reason': 'The required remaining-scope scan is unavailable.',
                  'summary': 'Saved layer observations remain valid; the required remaining scan failed.'}
    else:
        result = runtime.assemble_result(fact.gene_ids, fact.layer_scope, fact.edge_query, mode, top_k, rows, missing)
        reason = (initial + 'No new call is needed. ' if pending is None else '') + _reason(result, framing)
        status = 'not_found' if missing else 'resolved'
        answer = {'status': status, 'provenance': provenance, 'result': result,
                  'continuation_decision': 'defer' if missing else 'continue' if framing == 'step' else 'stop',
                  'blocked_reason': 'A required endpoint is outside the active multiplex.' if missing else None,
                  'summary': ('The pair cannot be evaluated because a required endpoint is outside the multiplex.' if missing else
                              f"The {mode} request is satisfied with {len(result['supporting_layer_ids'])} returned supporting layers; "
                              + (f"the exact selected-scope total is {result['supporting_layer_count']}." if result['supporting_layer_count'] is not None
                                 else 'the exact selected-scope total remains unknown.'))}
    messages.append({'role': 'assistant', 'thinking': reason, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'],
            'metadata': {'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                         'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                         'question_family': 'S2.2', 'split': fact.split, 'split_unit': fact.split_unit,
                         'selection_seed': seed,
                         'graph_version': view.graph_version, 'gene_ids': list(fact.gene_ids),
                         'input_genes': list(input_genes),
                         'layer_scope': fact.layer_scope, 'edge_query': fact.edge_query, 'mode': mode, 'top_k': top_k,
                         'support_count': fact.support_count, 'scope_size': len(selected),
                         'framing': framing, 'variant': variant, 'wrapper_index': wrapper,
                         'request_template_index': variant_index % len(contract['_requests'][fact.split][mode][fact.edge_query]),
                         'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def build_direction_diagnostic(view: GraphView, fact: MultiplexFact, framing: str,
                               variant_index: int, bank: Mapping[str, Any],
                               contract: Mapping[str, Any], *, seed: int = 20261028) -> dict[str, Any]:
    """Reject a visibly mixed-representation scope without making a graph call."""
    runtime = MultiplexRuntime(view)
    try:
        runtime.resolve_scope(fact.layer_scope, fact.edge_query)
    except SelectionError as error:
        if error.issue_code != 'DIRECTION_UNAVAILABLE':
            raise
    else:
        raise VariantUnavailable('diagnostic needs an undirected layer under a directed predicate')
    request = render_s2_2_request(contract['_requests'], split=fact.split, mode='complete',
        edge_query=fact.edge_query, variant_index=variant_index, multiplex_id=view.multiplex_id,
        first=fact.gene_ids[0], second=fact.gene_ids[1], scope_text=_scope_text(fact.layer_scope))
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    result = {'issue_code': 'DIRECTION_UNAVAILABLE',
              'request': {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
                          'layer_scope': fact.layer_scope, 'edge_query': fact.edge_query, 'mode': 'complete', 'top_k': None},
              'partial_results': []}
    answer = {'status': 'invalid_input', 'provenance': {'registry': [], 'graph': []}, 'result': result,
              'continuation_decision': 'defer', 'blocked_reason': 'The requested directed predicate includes an undirected layer.',
              'summary': 'The selected scope cannot support this directed predicate.'}
    undirected = [layer for layer in runtime.resolve_scope(fact.layer_scope) if not view.layers[layer].directed]
    return {'row_id': 's22_' + digest([asdict(fact), user, 'direction_invalid', view.graph_version])[:32],
            'messages': [{'role': 'system', 'content': contract['system_prompt']},
                         {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
                         {'role': 'user', 'content': user},
                         {'role': 'assistant', 'thinking':
                          f"The request applies {fact.edge_query} to {list(fact.gene_ids)} across {_scope_text(fact.layer_scope)}. "
                          f'The active manifest identifies {undirected} as undirected. This predicate requires every selected layer '
                          'to be directed, so the visible request is invalid before acquisition. Do not drop layers, reinterpret '
                          'an undirected edge as two arcs, or manufacture a negative result. Preserve the literal request, '
                          'return DIRECTION_UNAVAILABLE with empty provenance, and defer for a valid scope or predicate.', 'content': answer}],
            'tools': contract['tools'], 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.2', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'gene_ids': list(fact.gene_ids), 'layer_scope': fact.layer_scope,
                'edge_query': fact.edge_query, 'mode': 'complete', 'top_k': None, 'support_count': None,
                'scope_size': len(runtime.resolve_scope(fact.layer_scope)), 'framing': framing,
                'variant': 'direction_invalid', 'wrapper_index': wrapper,
                'request_template_index': variant_index % len(contract['_requests'][fact.split]['complete'][fact.edge_query]),
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}


@lru_cache(maxsize=1)
def _validation_wording() -> tuple[dict, dict]:
    """Load frozen default wording once for standalone row checks."""
    contract = load_build_contract()
    return load_s2_prompt_banks(contract['prompt_bank']), contract


@lru_cache(maxsize=1)
def _validation_registry() -> RegistryContext:
    """Load pinned identity context only when a standalone symbol row needs it."""
    return load_registry_context(load_build_contract()['registry_index'])


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any] | None = None,
                 contract: Mapping[str, Any] | None = None, registry: RegistryContext | None = None) -> None:
    """Replay visible acquisitions and reject hidden counts, altered evidence, or retries."""
    runtime = MultiplexRuntime(view)
    messages, meta = row['messages'], row['metadata']
    answer = messages[-1]['content']
    if bank is None or contract is None:
        default_bank, default_contract = _validation_wording()
        bank = default_bank if bank is None else bank
        contract = default_contract if contract is None else contract
    input_genes = meta.get('input_genes', meta['gene_ids'])
    if meta['variant'] == 'symbol_context':
        registry = _validation_registry() if registry is None else registry
        if list(get_symbols(meta['gene_ids'], registry)) != input_genes:
            raise ValueError('visible symbols do not uniquely map to the graph arguments')
    elif input_genes != meta['gene_ids']:
        raise ValueError('canonical query inputs were changed')
    request = render_s2_2_request(contract['_requests'], split=meta['split'], mode=meta['mode'],
        edge_query=meta['edge_query'], variant_index=meta['request_template_index'],
        multiplex_id=view.multiplex_id, first=input_genes[0], second=input_genes[1],
        scope_text=_scope_text(meta['layer_scope']), top_k=meta['top_k'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=request)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('visible request differs from the audited query or task framing')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('row system or tools differ from the pinned contract')
    if _pair_group(tuple(meta['gene_ids']), meta['selection_seed']) != (meta['split_unit'], meta['split']):
        raise ValueError('pair split metadata differs from the declared selection seed')
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    context = json.loads(messages[1]['content'])
    if context['active_graph'] != _active(view):
        raise ValueError('harness graph metadata differs from the pinned source')
    if meta['variant'] == 'direction_invalid':
        if (meta['mode'] != 'complete' or meta['top_k'] is not None or meta['support_count'] is not None
                or meta['scope_size'] != len(runtime.resolve_scope(meta['layer_scope']))):
            raise ValueError('direction diagnostic metadata does not match its request')
        fact = MultiplexFact(meta['split'], tuple(meta['gene_ids']), meta['layer_scope'],
                             meta['edge_query'], None, meta['split_unit'])
        # Frozen template indices may have different periods. Only compare the
        # semantic diagnostic envelope here; the user binding was checked above.
        expected = build_direction_diagnostic(view, fact, meta['framing'], 0, bank, contract,
                                              seed=meta['selection_seed'])['messages'][-1]['content']
        if (len(messages) != 4 or context['observations'] or not messages[-1].get('thinking')
                or any(answer[key] != expected[key] for key in ('status', 'provenance', 'result', 'continuation_decision'))):
            raise ValueError('invalid local direction diagnostic')
        return
    arguments = {key: meta[key] for key in ('gene_ids', 'layer_scope', 'edge_query', 'mode', 'top_k')}
    arguments = {'multiplex_id': view.multiplex_id, **arguments}
    # This separate source check validates audit labels, never supplies answer
    # evidence. The final answer is checked only against visible acquisitions.
    source_truth = runtime.get_edge_layers(**{**arguments, 'mode': 'complete', 'top_k': None}, evidence_id='audit_only')['result']
    if (type(meta['scope_size']) is not int or meta['scope_size'] != len(source_truth['coverage']['requested_layer_ids'])
            or canonical_json(meta['support_count']) != canonical_json(source_truth['supporting_layer_count'])):
        raise ValueError('sampling metadata differs from the native source stratum')
    available = list(context['observations'])
    evidence_ids = {}
    identities = [o for o in available if o['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        validate_identity_observations(identities, registry)
        if ([o['result']['gene_id'] for o in identities] != meta['gene_ids']
                or [o['arguments']['gene_symbol'] for o in identities] != input_genes):
            raise ValueError('supplied registry evidence does not resolve the requested symbols')
        for observation in identities:
            evidence_ids[observation['provenance']['registry'][0]['evidence_id']] = observation
    elif identities:
        raise ValueError('unexpected identity evidence')

    def check_observation(observation: Mapping[str, Any]) -> None:
        graph = observation['provenance']['graph']
        if len(graph) != 1:
            raise ValueError('each acquisition needs exactly one unchanged graph record')
        identity = graph[0]['evidence_id']
        if identity in evidence_ids and evidence_ids[identity] != observation:
            raise ValueError('one evidence ID names conflicting acquisitions')
        evidence_ids[identity] = observation
        if graph[0]['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph[0]['graph_version'] != '0' * 64:
                raise ValueError('undeclared stale observation')
            return
        operation = observation['operation']
        if operation not in ('get_edge', 'get_edge_layers'):
            raise ValueError('unsupported saved operation')
        if _pair_group(tuple(observation['arguments']['gene_ids']), meta['selection_seed'])[1] != meta['split']:
            raise ValueError('a saved graph pair crosses the declared split boundary')
        function = view.get_edge if operation == 'get_edge' else runtime.get_edge_layers
        expected = function(**observation['arguments'], evidence_id=graph[0]['evidence_id'])
        if any(canonical_json(observation[key]) != canonical_json(expected[key]) for key in ('status', 'result', 'provenance')):
            raise ValueError('visible observation differs from the pinned source')

    for observation in available:
        if observation['operation'] != 'lookup_human_gene_symbol':
            check_observation(observation)
    last_call = last_response = None
    call_count = 0
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if not message.get('thinking'):
            raise ValueError('assistant reasoning is missing')
        if 'tool_call' not in message:
            continue
        call = message['tool_call']
        pending = _remaining(runtime, arguments, available)
        if pending is None or call != {'name': 'get_edge_layers', 'arguments': pending}:
            raise ValueError('call does not match remaining required work')
        response_message = messages[index + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('tool response does not match its call')
        response = response_message['content']
        if last_call is not None:
            if (last_response['status'] != 'error' or 'retries_remaining=1' not in last_response.get('message', '')
                    or last_call != call or call_count >= 2):
                raise ValueError('retry lacks permission or changes its arguments')
        if response['status'] == 'error':
            if response['provenance'] != {'registry': [], 'graph': []} or 'result' in response:
                raise ValueError('a failed atomic batch publishes no graph evidence')
        else:
            observation = _observation(call['name'], call['arguments'], response)
            check_observation(observation)
            available.append(observation)
        last_call, last_response = call, response
        call_count += 1
    rows, missing, used, provenance = _proofs(runtime, arguments, available)
    if answer['provenance'] != provenance:
        raise ValueError('answer provenance does not match its visible supporting records')
    if answer['status'] == 'error':
        if not last_response or last_response['status'] != 'error' or _remaining(runtime, arguments, available) is None:
            raise ValueError('final error lacks an unresolved required failure')
        if 'retries_remaining=1' in last_response.get('message', ''):
            raise ValueError('the permitted retry was not attempted')
        expected = {'failed_tool': last_call['name'], 'error_code': last_response['error_code'],
                    'arguments': last_call['arguments'], 'partial_results': used,
                    'coverage': _coverage(runtime, arguments, rows)}
        if canonical_json(answer['result']) != canonical_json(expected) or answer['continuation_decision'] != 'defer':
            raise ValueError('failure loses saved evidence or invents batch coverage')
        return
    if not used:
        raise ValueError('answer has no visible membership or predicate evidence')
    expected = runtime.assemble_result(arguments['gene_ids'], arguments['layer_scope'],
        arguments['edge_query'], arguments['mode'], arguments['top_k'], rows, missing)
    status = 'not_found' if missing else 'resolved'
    decision = 'defer' if missing else 'continue' if meta['framing'] == 'step' else 'stop'
    if canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status or answer['continuation_decision'] != decision:
        raise ValueError('answer exceeds visible evidence or changes task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028,
                 source_manifest: Path | None = None, contract_path: Path | None = None) -> dict[str, Any]:
    """Write immutable S2.2 artifacts with explicit counts or a budget-free smoke run."""
    if (counts is None) != smoke:
        raise ValueError('choose explicit counts or smoke=True; no production budget is defaulted')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('S2.2 source manifest hash mismatch')
    view = load_source_view(source_manifest)
    runtime = MultiplexRuntime(view)
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    registry = load_registry_context(contract['registry_index'])
    wording = validate_s2_2_rendered_prompts(bank, contract['_requests'], load_s2_edge_requests(contract['s21_request_bank']))
    facts, selection = select_facts(view, tuple(sorted(registry.symbols_by_id)), counts, seed)
    rows = {split: [] for split in SPLITS}
    skipped: Counter = Counter()
    for fact in facts:
        size = len(runtime.resolve_scope(fact.layer_scope, fact.edge_query))
        requests = [('complete', None), ('exists', None), *[('top_k', k) for k in range(1, max(1, size) + 1)]]
        for mode, top_k in requests:
            for variant in VARIANTS:
                for framing in FRAMINGS:
                    # Round-robin independent of graph selection and bank size.
                    ordinal = len(rows[fact.split]) // len(FRAMINGS)
                    try:
                        row = build_row(view, fact, mode, top_k, framing, variant, ordinal, bank, contract, seed=seed, registry=registry)
                    except VariantUnavailable:
                        skipped[f'{variant}/{mode}'] += 1
                        continue
                    validate_row(row, view, bank=bank, contract=contract, registry=registry)
                    rows[fact.split].append(row)
    if any(not layer.directed for layer in view.layers.values()):
        for split in SPLITS:
            source_fact = next(f for f in facts if f.split == split and f.support_count is not None)
            for query in ('source_to_target', 'both_directions'):
                fact = MultiplexFact(split, source_fact.gene_ids,
                    {'kind': 'all', 'layer_ids': [], 'layer_family': None}, query, None, source_fact.split_unit)
                for framing in FRAMINGS:
                    row = build_direction_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, seed=seed)
                    validate_row(row, view, bank=bank, contract=contract)
                    rows[split].append(row)
    units = {split: {row['metadata']['split_unit'] for row in rows[split]} for split in SPLITS}
    if any(units[left] & units[right] for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]):
        raise ValueError('endpoint-pair contrasts crossed split boundaries')
    audit = {'source_semantics_passed': True, 'training_ready': False, 'tokenizer_audited': False,
             'purpose': 'builder_verification' if smoke else 'explicit_budget_build',
             'selection': selection, 'full_question_wording': wording, 'split_overlap_count': 0,
             'inapplicable_context_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    files = {}
    for split in SPLITS:
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage = Counter(f"{r['metadata']['framing']}:{r['metadata']['wrapper_index']}" for r in rows[split])
        joint = Counter('/'.join(str(r['metadata'][k]) for k in ('scope_size', 'support_count', 'top_k', 'mode', 'edge_query')) for r in rows[split])
        wrapper_cells: dict[str, Counter] = defaultdict(Counter)
        core_usage: Counter = Counter()
        for row in rows[split]:
            meta = row['metadata']
            wrapper_cells[f"{meta['framing']}:{meta['wrapper_index']}"]['/'.join(
                str(meta[key]) for key in ('mode', 'edge_query', 'support_count', 'variant'))] += 1
            core_usage[f"{meta['mode']}/{meta['edge_query']}/{meta['request_template_index']}"] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'distinct_pairs': len(units[split]),
                                 'wrapper_usage': dict(sorted(usage.items())), 's_h_k_mode_predicate': dict(sorted(joint.items())),
                                 'wrapper_by_mode_predicate_h_variant': {key: dict(sorted(value.items())) for key, value in sorted(wrapper_cells.items())},
                                 'request_template_usage': dict(sorted(core_usage.items())),
                                 'variants': dict(sorted(Counter(r['metadata']['variant'] for r in rows[split]).items()))}
        if split != 'train':
            questions, answers = [], []
            for row in rows[split]:
                questions.append({'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                                  'metadata': {key: row['metadata'][key] for key in
                                               ('schema_version', 'tool_contract_version', 'question_family', 'split')}})
                answers.append({'row_id': row['row_id'], 'expected_messages': row['messages'][3:],
                                'final_answer': row['messages'][-1]['content']})
            files[f'eval/{split}.questions.jsonl'] = _jsonl(questions)
            files[f'eval/{split}.answers.jsonl'] = _jsonl(answers)
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    implementation = ['runtime/world_model/' + name + '.py' for name in
                      ('s2_2_corpus', 's2_2_prompts', 's2_2_identity', 's2_multiplex', 's2_corpus', 's2_sources',
                       's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)]
    implementation.append('scripts/build_world_model_s2_2.py')
    manifest = {'schema_version': 'mentor-rl-world-model-s2-2-manifest-v1',
                'contract_version': contract['contract_version'], 'implemented_families': ['S2.2'],
                'training_ready': False, 'purpose': audit['purpose'], 'generation_budget': None if smoke else counts,
                'verification_queries_per_cell': 1 if smoke else None, 'seed': seed,
                'python_version': platform.python_version(), 'row_counts': {s: len(rows[s]) for s in SPLITS},
                'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
                'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
                'prompt_bank': contract['prompt_bank'], 'request_bank': contract['request_bank'],
                's21_request_bank': contract['s21_request_bank'], 'registry_index': contract['registry_index'],
                'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
                'fact_selection_sha256': selection['fact_selection_sha256'],
                'split_policy': 'S2.1-compatible unordered pair SHA256(seed,pair), bins 8:1:1; never reassign rare cells',
                'fault_injections': 'declared terminal/recovery/exhaustion schedules and stale version 0x00; not observed outages or historical graph versions',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: item['sha256'] for name, item in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
