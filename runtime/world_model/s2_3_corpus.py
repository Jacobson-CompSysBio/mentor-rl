"""Build source-backed S2.3 verification traces without clipping neighbor lists.

Query genes are grouped across layers and directions. This is query separation,
not edge-fact separation: adjacent genes may expose the same native edge. The
full source count domain is audited before sampling; production allocation and
the representation of oversized training examples remain explicit decisions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import BUILDER_MODULES, load_s2_prompt_banks, load_s2_edge_requests, render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _active, _observation
from .s2_2_identity import RegistryContext, get_symbols, identity_observations, load_registry_context, validate_identity_observations
from .s2_2_prompts import load_s2_2_requests
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure, _jsonl, _write_immutable
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_3_v1.json')
VARIANTS = ('lookup', 'context', 'complete_context', 'bounded_context', 'prefix_context',
            'paged', 'overlap', 'empty_page', 'unknown_total', 'terminal', 'recovered', 'exhausted',
            'second_terminal', 'second_recovered', 'second_exhausted', 'symbol_context',
            'stale', 'wrong_direction', 'wrong_layer', 'distractor')


@dataclass(frozen=True)
class NeighborFact:
    """One source-selected query before wording, pagination, or fault contrasts."""

    split: str
    gene_id: str
    layer_id: str
    direction: str
    category: str
    neighbor_counts: dict[str, int | None]
    split_unit: str


def gene_group(gene_id: str, seed: int = 20261028) -> tuple[str, str]:
    """Keep a queried gene and all its layer/direction contrasts in one split."""
    unit = digest({'seed': seed, 'gene_id': gene_id})
    bucket = int(unit[:16], 16) % 10
    return unit, 'train' if bucket < 8 else 'validation' if bucket == 8 else 'test'


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the separate S2.3 overlay and its independently pinned wording."""
    from .s2_3_prompts import load_s2_3_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.3']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.3 build contract')
    contract['_requests'] = load_s2_3_requests(contract['request_bank'])
    return contract


def _directions(view: GraphView, layer_id: str) -> tuple[str, ...]:
    return ('in', 'out') if view.layers[layer_id].directed else ('undirected',)


def _fact(view: GraphView, index: Any, gene: str, layer: str, direction: str, seed: int) -> NeighborFact:
    unit, split = gene_group(gene, seed)
    category = ('outside_multiplex' if gene not in view.multiplex_nodes else
                'outside_layer' if gene not in view.layers[layer].nodes else 'present')
    directions = ('in', 'out') if direction == 'both' else (direction,)
    counts = {d: None if category == 'outside_multiplex' else len(index.neighbors(layer, gene, d)) for d in directions}
    return NeighborFact(split, gene, layer, direction, category, counts, unit)


def select_facts(view: GraphView, index: Any, registry_ids: Sequence[str], counts: Mapping[str, int],
                 seed: int, *, smoke: bool = False) -> tuple[list[NeighborFact], dict[str, Any]]:
    """Audit all attainable counts, then draw real genes from fixed query splits.

    An explicit recipe samples every attainable count. Smoke mode exercises
    extrema, small counts, a median count and native ties/loops/zero weights.
    Counts absent from the smoke selection are reported rather than called
    covered. Neither mode invents missing integer count values.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts require positive integers for all three splits')
    selected: dict[tuple[str, str, str], NeighborFact] = {}
    domains, cells, omissions, feature_cells = [], [], [], []
    outside = sorted(set(registry_ids) - view.multiplex_nodes)
    if not outside:
        raise ValueError('the registry has no globally missing candidate IDs')

    def add(gene: str, layer: str, direction: str) -> None:
        selected[(gene, layer, direction)] = _fact(view, index, gene, layer, direction, seed)

    for layer_id, layer in view.layers.items():
        for direction in _directions(view, layer_id):
            pools: dict[tuple[str, int], list[str]] = defaultdict(list)
            features: dict[tuple[str, str], list[str]] = defaultdict(list)
            for gene in sorted(layer.nodes):
                neighbors = index.neighbors(layer_id, gene, direction)
                split = gene_group(gene, seed)[1]
                pools[(split, len(neighbors))].append(gene)
                weights = [weight for _, weight in neighbors]
                if len(weights) != len(set(weights)):
                    features[(split, 'equal_weight')].append(gene)
                if any(neighbor == gene for neighbor, _ in neighbors):
                    features[(split, 'self_loop')].append(gene)
                if any(weight == 0 for weight in weights):
                    features[(split, 'zero_weight')].append(gene)
                if layer.directed:
                    reverse = index.neighbors(layer_id, gene, 'out' if direction == 'in' else 'in')
                    if ({neighbor for neighbor, _ in neighbors} & {neighbor for neighbor, _ in reverse}) - {gene}:
                        features[(split, 'reciprocal_neighbor')].append(gene)
            histogram = index.histogram(layer_id, direction)
            domain = sorted(histogram)
            domain_record = {'layer_id': layer_id, 'direction': direction, 'histogram': histogram,
                             'members': len(layer.nodes), 'count_sum': sum(n * frequency for n, frequency in histogram.items()),
                             'minimum': min(domain), 'maximum': max(domain), 'attainable_count_values': len(domain)}
            domains.append(domain_record)
            for split in SPLITS:
                available_counts = sorted(n for s, n in pools if s == split)
                smoke_counts = ({*available_counts[:4], *available_counts[-1:],
                                 *available_counts[len(available_counts) // 2:len(available_counts) // 2 + 1]}
                                if smoke else set(domain))
                for n in domain:
                    pool = sorted(pools[(split, n)], key=lambda g: gene_group(g, seed)[0])
                    requested = counts[split] if n in smoke_counts else 0
                    drawn = pool[:requested]
                    cell = {'layer_id': layer_id, 'direction': direction, 'split': split, 'neighbor_count': n,
                            'natural_population': histogram[n], 'split_population': len(pool),
                            'requested': requested, 'selected': len(drawn)}
                    cells.append(cell)
                    if requested and len(drawn) < requested:
                        omissions.append({**cell, 'reason': 'insufficient genes in the fixed query split'})
                    for gene in drawn:
                        add(gene, layer_id, direction)
                for feature in ('equal_weight', 'self_loop', 'zero_weight', 'reciprocal_neighbor'):
                    pool = sorted(features[(split, feature)], key=lambda g: (len(index.neighbors(layer_id, g, direction)), gene_group(g, seed)[0]))
                    for gene in pool[:counts[split]]:
                        add(gene, layer_id, direction)
                    feature_cells.append({'layer_id': layer_id, 'direction': direction, 'split': split,
                                          'feature': feature, 'available': len(pool), 'selected': min(len(pool), counts[split])})
                for category, population in (
                    ('outside_layer', view.multiplex_nodes - layer.nodes), ('outside_multiplex', outside),
                ):
                    pool = sorted((g for g in population if gene_group(g, seed)[1] == split), key=lambda g: gene_group(g, seed)[0])
                    for gene in pool[:counts[split]]:
                        add(gene, layer_id, direction)
                    if len(pool) < counts[split]:
                        omissions.append({'layer_id': layer_id, 'direction': direction, 'split': split,
                                          'category': category, 'requested': counts[split], 'selected': len(pool),
                                          'reason': 'insufficient genes in the fixed query split'})
    # Compound requests reuse selected directional subjects. Their two counts
    # remain separate; there is no invented combined-neighbor ranking/count.
    for fact in list(selected.values()):
        if view.layers[fact.layer_id].directed:
            add(fact.gene_id, fact.layer_id, 'both')
    facts = sorted(selected.values(), key=lambda f: (SPLITS.index(f.split), view.layer_order.index(f.layer_id),
                   f.direction, f.category, f.split_unit))
    uncovered = []
    for entry in domains:
        actual = {f.neighbor_counts[entry['direction']] for f in facts if f.layer_id == entry['layer_id']
                  and f.direction == entry['direction'] and f.category == 'present'}
        uncovered.append({'layer_id': entry['layer_id'], 'direction': entry['direction'],
                          'unselected_count_values': sorted(set(entry['histogram']) - actual)})
    return facts, {'domains': domains, 'cells': cells, 'omissions': omissions, 'feature_cells': feature_cells,
                   'smoke_unselected_count_values': uncovered, 'selected_facts': len(facts),
                   'distinct_query_genes': len({f.gene_id for f in facts}),
                   'fact_selection_sha256': digest([asdict(f) for f in facts]),
                   'split_policy': 'SHA256(seed,query gene) bins 8:1:1 across all layers and directions',
                   'edge_fact_disjoint': False,
                   'edge_overlap_note': 'Neighbors can occur in multiple splits and earlier S2 pair panels; only query subjects and wording are held out.'}


def _arguments(view: GraphView, fact: NeighborFact, direction: str, mode: str, top_k: int | None) -> dict[str, Any]:
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': fact.gene_id,
            'direction': direction, 'mode': mode, 'top_k': top_k, 'cursor': None}


def _transport(fact: NeighborFact, variant: str) -> dict[str, Any]:
    largest = max((n or 0 for n in fact.neighbor_counts.values()), default=0)
    result = {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}
    if variant in ('paged', 'prefix_context', 'terminal', 'recovered', 'exhausted', 'unknown_total', 'empty_page'):
        result['page_size'] = max(1, (largest + 1) // 2)
    if variant == 'overlap':
        if largest < 3:
            raise VariantUnavailable('overlapping boundaries need at least three neighbors')
        result.update(page_size=max(2, (largest + 1) // 2), overlap=1)
    if variant == 'empty_page':
        if largest < 2:
            raise VariantUnavailable('an empty intermediate page needs a multi-page traversal')
        result['empty_page_at'] = 1
    if variant == 'unknown_total':
        result['disclose_total'] = False
    return result


def _provenance(observations: Sequence[Mapping[str, Any]]) -> dict[str, list]:
    result = {'registry': [], 'graph': []}
    for observation in observations:
        for channel in result:
            for record in observation['provenance'][channel]:
                if record not in result[channel]:
                    result[channel].append(record)
    return result


def _plan(runtime: Any, view: GraphView, fact: NeighborFact, mode: str, top_k: int | None,
          observations: Sequence[Mapping[str, Any]]) -> tuple[dict | None, dict | None, list, dict | None]:
    """Plan from visible collections only, sharing compound membership findings."""
    directions = ('in', 'out') if fact.direction == 'both' else (fact.direction,)
    identities = [o for o in observations if o['operation'] == 'lookup_human_gene_symbol']
    proofs = {direction: runtime.collect(_arguments(view, fact, direction, mode, top_k), observations) for direction in directions}
    used = list(identities)
    for proof in proofs.values():
        for observation in proof.observations:
            if observation not in used:
                used.append(observation)
    if fact.direction != 'both':
        proof = proofs[fact.direction]
        return proof.result, proof.next_arguments, used, None
    completed = {direction: proof.result for direction, proof in proofs.items() if proof.result is not None}
    missing = next(((direction, result) for direction, result in completed.items() if result['missing_gene_ids']), None)
    absent = next((result for result in completed.values() if result['not_in_layer_gene_ids']), None)
    if absent:
        completed = {direction: {**deepcopy(absent), 'direction': direction} for direction in directions}
    checked = [direction for direction in directions if direction in completed and not completed[direction]['missing_gene_ids']]
    coverage = {'operation': 'neighbor_directions', 'requested_directions': list(directions),
                'checked_directions': checked, 'unchecked_directions': [d for d in directions if d not in checked]}
    if missing or len(checked) == 2:
        return {'gene_id': fact.gene_id, 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
                'direction': 'both', 'incoming': completed.get('in'), 'outgoing': completed.get('out'),
                'coverage': coverage}, None, used, coverage
    direction = next(d for d in directions if d not in checked)
    return None, proofs[direction].next_arguments, used, coverage


def _reason(result: Mapping[str, Any], framing: str, used: Sequence[Mapping[str, Any]]) -> str:
    """Explain actual rank, membership, coverage and task completion concisely."""
    parts = [f"The query is {result['gene_id']} in {result['layer_id']} of {result['multiplex_id']}."]
    sides = [result] if result['direction'] != 'both' else [r for r in (result['incoming'], result['outgoing']) if r is not None]
    unavailable = False
    for side in sides:
        direction = side['direction']
        if side['missing_gene_ids']:
            parts.append(f"The membership check places {side['gene_id']} outside the multiplex. The {direction} list and collection remain null.")
            unavailable = True
            continue
        collection, neighbors = side['collection'], side['neighbors']
        if side['not_in_layer_gene_ids']:
            parts.append(f"The query belongs to the multiplex but is absent from this layer, which proves an empty {direction} list and total zero.")
        elif not neighbors:
            parts.append(f"The query is present in the layer. Its requested {direction} list is empty; this says nothing about an unrequested direction.")
        else:
            leading, trailing = neighbors[0], neighbors[-1]
            role = 'query → neighbor' if direction == 'out' else 'neighbor → query' if direction == 'in' else 'unordered query–neighbor edge'
            parts.append(f"For {direction}, {role} is the native interpretation. The accumulated ranked result has {len(neighbors)} distinct IDs: "
                         f"first {leading['gene_id']} at {leading['weight']}, last {trailing['gene_id']} at {trailing['weight']}. "
                         'Order weights descending, then canonical IDs ascending; equal weights do not enlarge a requested bound.')
            if any(entry['gene_id'] == result['gene_id'] for entry in neighbors):
                parts.append('The source self-loop is retained once in this directional list.')
        parts.append(f"The {collection['mode']} traversal has ended. The full total is "
                     + (str(collection['total_count']) if collection['total_count'] is not None else 'unknown')
                     + f" and complete is {str(collection['complete']).lower()}. "
                     'A final page alone need not contain the entire collection; the answer uses the compatible traversal from its initial page.')
        relevant = [o for o in used if o['operation'] == 'get_neighbors' and o['result'].get('direction') == direction]
        if collection['complete'] and relevant and all(o['result']['collection']['total_count'] is None for o in relevant if o['result']['collection']):
            parts.append(f"No page supplied a total; exhaustive unique-entry accumulation establishes {collection['total_count']} as a calculation over visible observations.")
    if result['direction'] == 'both':
        parts.append(f"Directional coverage is {result['coverage']}. Keep incoming and outgoing weights separate; a shared neighbor is not a license to add counts or invent an overall ranking.")
    parts.append('Preserve the original acquisition evidence and identical-entry deduplication. ' + (
        'The required global membership is missing, so defer without another direction lookup.' if unavailable else
        'This completes the requested workflow step, so continue.' if framing == 'step' else
        'This completes the standalone request, so stop.'))
    return ' '.join(parts)


def build_row(view: GraphView, fact: NeighborFact, mode: str, top_k: int | None, framing: str,
              variant: str, variant_index: int, bank: Mapping[str, Any], contract: Mapping[str, Any], *,
              index: Any = None, registry: RegistryContext | None = None, seed: int = 20261028) -> dict[str, Any]:
    """Build a fully bound neighbor traversal, preserving page evidence and retries."""
    from .s2_neighbors import NeighborRuntime
    from .s2_3_prompts import render_s2_3_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported S2.3 contrast or framing')
    transport = _transport(fact, variant)
    runtime = NeighborRuntime(view, index=index, **transport)
    index = runtime.index
    input_gene = fact.gene_id
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('supplied symbol context requires a verified registry')
        try:
            input_gene = get_symbols([fact.gene_id], registry)[0]
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = render_s2_3_request(contract['_requests'], split=fact.split, direction=fact.direction, mode=mode,
        variant_index=variant_index, multiplex_id=view.multiplex_id, layer_id=fact.layer_id, gene=input_gene, top_k=top_k)
    user, wrapper_index = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    row_id = 's23_' + digest([contract['contract_version'], view.graph_version, asdict(fact), mode, top_k, framing, variant, user])[:32]
    observations: list[dict] = []
    serial = 0

    def acquire(arguments: dict) -> dict:
        nonlocal serial
        serial += 1
        return runtime.get_neighbors(**arguments, evidence_id=f'ev_{row_id}_{serial}')

    def save(arguments: dict, *, first_only: bool = False) -> None:
        response = acquire(arguments)
        observations.append(_observation('get_neighbors', arguments, response))
        if first_only or response['status'] != 'resolved':
            return
        cursor = response['result']['collection']['next_cursor']
        while cursor is not None:
            arguments = {**arguments, 'cursor': cursor}
            response = acquire(arguments)
            observations.append(_observation('get_neighbors', arguments, response))
            cursor = response['result']['collection']['next_cursor']

    directions = ('in', 'out') if fact.direction == 'both' else (fact.direction,)
    if variant == 'symbol_context':
        observations.extend(identity_observations([fact.gene_id], row_id, registry))
    elif variant in ('context', 'complete_context', 'bounded_context', 'stale'):
        saved_mode = 'complete' if variant in ('complete_context', 'stale') else 'top_k' if variant == 'bounded_context' else mode
        saved_k = 1 if variant == 'bounded_context' else None if saved_mode == 'complete' else top_k
        for direction in directions:
            save(_arguments(view, fact, direction, saved_mode, saved_k))
            if observations[-1]['result']['missing_gene_ids'] or observations[-1]['result']['not_in_layer_gene_ids']:
                break
        if variant == 'stale':
            for observation in observations:
                observation['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant in ('prefix_context', 'terminal', 'recovered', 'exhausted'):
        direction = next((d for d in directions if (fact.neighbor_counts[d] or 0) > transport['page_size']
                          and (mode == 'complete' or top_k > transport['page_size'])), None)
        if direction is None:
            raise VariantUnavailable('this contrast requires an unfinished saved first page')
        if fact.direction == 'both' and direction == 'out':
            save(_arguments(view, fact, 'in', mode, top_k))
        save(_arguments(view, fact, direction, mode, top_k), first_only=True)
    elif variant == 'wrong_direction':
        if fact.direction not in ('in', 'out') or fact.category != 'present':
            raise VariantUnavailable('wrong-direction context needs a present directional query')
        save(_arguments(view, fact, 'out' if fact.direction == 'in' else 'in', mode, top_k))
    elif variant == 'wrong_layer':
        if fact.category == 'outside_multiplex':
            raise VariantUnavailable('a saved global-absence finding already settles membership across layers')
        candidates = [layer for layer in view.layer_order if layer != fact.layer_id
                      and view.layers[layer].directed == view.layers[fact.layer_id].directed]
        if not candidates:
            raise VariantUnavailable('no other layer with the same native representation')
        save({**_arguments(view, fact, directions[0], mode, top_k), 'layer_id': candidates[0]})
    elif variant == 'distractor':
        candidates = sorted((gene for gene in view.multiplex_nodes if gene != fact.gene_id and gene_group(gene, seed)[1] == fact.split),
                            key=lambda gene: digest([fact.split_unit, gene]))
        if not candidates:
            raise VariantUnavailable('no same-split distractor query gene')
        save({**_arguments(view, fact, directions[0], mode, top_k), 'gene_id': candidates[0]})
    if variant.startswith('second_') and (fact.direction != 'both' or fact.category != 'present'):
        raise VariantUnavailable('second-side failure needs a present compound query')
    if variant == 'bounded_context' and _plan(runtime, view, fact, mode, top_k, observations)[1] is None:
        raise VariantUnavailable('the saved bounded result already answers this request')
    context = {'active_graph': _active(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    failure_variant = variant.replace('second_', '')
    injected = False
    failed = None
    while True:
        result, pending, used, coverage = _plan(runtime, view, fact, mode, top_k, observations)
        if pending is None:
            break
        initial = (f"The request asks for {mode} neighbors of {fact.gene_id} in {fact.layer_id}, direction {fact.direction}"
                   + (f', with K={top_k} per requested directional list' if mode == 'top_k' else '') + '. '
                   f"Matching evidence still requires {pending['direction']} with cursor {pending['cursor']!r}. "
                   'Use only compatible gene, layer, graph version, direction and selection evidence. '
                   'A transport boundary does not finish an all-neighbors request; copy returned cursors and retain every needed page.')
        if variant == 'symbol_context':
            initial = f'The supplied registry observation uniquely maps {input_gene} to {fact.gene_id}; retain that evidence. ' + initial
        should_fail = not injected and failure_variant in ('terminal', 'recovered', 'exhausted') and (
            not variant.startswith('second_') or pending['direction'] == 'out')

        def append_call(thought: str, response: dict) -> None:
            messages.extend([{'role': 'assistant', 'thinking': thought,
                              'tool_call': {'name': 'get_neighbors', 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': 'get_neighbors', 'content': response}])

        if should_fail:
            injected = True
            response = _failure(None if failure_variant == 'terminal' else 1)
            append_call(initial, response)
            if failure_variant != 'terminal':
                response = _failure(0) if failure_variant == 'exhausted' else acquire(pending)
                append_call(f"The {pending['direction']} page failed without adding evidence. retries_remaining=1 explicitly permits one retry. "
                            'Repeat the identical seven arguments, including the opaque cursor; keep the earlier independent observations.', response)
            if response['status'] == 'error':
                failed = (pending, response)
                break
        else:
            response = acquire(pending)
            append_call(initial, response)
        observations.append(_observation('get_neighbors', pending, response))
    if failed:
        pending, response = failed
        result = {'failed_tool': 'get_neighbors', 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': coverage}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'defer', 'blocked_reason': 'A required neighbor page is unavailable.',
                  'summary': 'The required traversal failed; valid earlier observations remain available.'}
        thought = (f"The required {pending['direction']} traversal for {fact.gene_id} in {fact.layer_id} failed at cursor {pending['cursor']!r}. "
                   + ('The retry allowance is exhausted. ' if failure_variant == 'exhausted' else 'No retry allowance was supplied. ')
                   + f'Preserve the {len(used)} supported earlier records and coverage {coverage}. '
                   'The failed page proves no neighbors, absence, or total. A partial directional traversal remains unchecked in a compound request. Defer for backend recovery.')
    else:
        sides = [result] if fact.direction != 'both' else [r for r in (result['incoming'], result['outgoing']) if r is not None]
        missing = any(side['missing_gene_ids'] for side in sides)
        answer = {'status': 'not_found' if missing else 'resolved', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'defer' if missing else 'continue' if framing == 'step' else 'stop',
                  'blocked_reason': 'The queried gene is outside the active multiplex.' if missing else None,
                  'summary': 'The queried gene is outside the multiplex; its neighbors are unavailable.' if missing else
                             'The requested directional neighbor collection is supplied in rank order.'}
        thought = ('Compatible saved observations already satisfy this request; no new call is needed. '
                   if not any('tool_call' in message for message in messages) else '') + _reason(result, framing, used)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.3', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': input_gene,
        'layer_id': fact.layer_id, 'direction': fact.direction, 'category': fact.category,
        'neighbor_counts': fact.neighbor_counts, 'mode': mode, 'top_k': top_k,
        'framing': framing, 'variant': variant, 'transport': transport, 'wrapper_index': wrapper_index,
        'request_template_index': variant_index % len(contract['_requests'][fact.split][mode][fact.direction]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def build_direction_diagnostic(view: GraphView, fact: NeighborFact, framing: str, variant_index: int,
                               bank: Mapping[str, Any], contract: Mapping[str, Any], *, seed: int = 20261028) -> dict[str, Any]:
    """Reject incoming roles on a visibly undirected layer without an acquisition."""
    from .s2_3_prompts import render_s2_3_request
    if view.layers[fact.layer_id].directed:
        raise VariantUnavailable('the selected layer supports directed roles')
    request = render_s2_3_request(contract['_requests'], split=fact.split, mode='complete', direction='in',
        variant_index=variant_index, multiplex_id=view.multiplex_id, layer_id=fact.layer_id, gene=fact.gene_id)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    result = {'issue_code': 'DIRECTION_UNAVAILABLE',
              'request': {'gene': fact.gene_id, 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id,
                          'direction': 'in', 'mode': 'complete', 'top_k': None, 'cursor': None},
              'rejected_arguments': None, 'partial_results': []}
    answer = {'status': 'invalid_input', 'provenance': {'registry': [], 'graph': []}, 'result': result,
              'continuation_decision': 'defer', 'blocked_reason': 'The selected layer has no incoming representation.',
              'summary': 'The requested incoming roles are unavailable in this undirected layer.'}
    return {'row_id': 's23_' + digest([view.graph_version, fact.gene_id, fact.layer_id, user, 'direction_invalid'])[:32],
            'messages': [{'role': 'system', 'content': contract['system_prompt']},
                         {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
                         {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking':
                          f"The request asks for incoming neighbors of {fact.gene_id} in {fact.layer_id}. "
                          'The visible manifest says this layer is undirected, so source-to-target roles are unavailable. '
                          'Reject the request locally without inventing an empty list, reinterpreting edges, or calling a graph tool. '
                          'Preserve the original request and defer for a compatible direction.', 'content': answer}],
            'tools': contract['tools'], 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.3', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': fact.gene_id,
                'layer_id': fact.layer_id, 'direction': 'in', 'category': 'diagnostic', 'neighbor_counts': {},
                'mode': 'complete', 'top_k': None, 'framing': framing, 'variant': 'direction_invalid',
                'transport': {}, 'wrapper_index': wrapper,
                'request_template_index': variant_index % len(contract['_requests'][fact.split]['complete']['in']),
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}


@lru_cache(maxsize=1)
def _default_inputs() -> tuple[dict, dict]:
    """Load frozen wording once for standalone source checks."""
    contract = load_build_contract()
    return load_s2_prompt_banks(contract['prompt_bank']), contract


@lru_cache(maxsize=1)
def _default_registry() -> RegistryContext:
    """Read pinned identity evidence only when a symbol row needs it."""
    return load_registry_context(load_build_contract()['registry_index'])


def validate_row(row: Mapping[str, Any], view: GraphView, *, index: Any = None,
                 bank: Mapping[str, Any] | None = None, contract: Mapping[str, Any] | None = None,
                 registry: RegistryContext | None = None) -> None:
    """Bind visible requests, verify native pages, and grade only visible proofs."""
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_neighbors import NeighborRuntime
    from .s2_3_prompts import render_s2_3_request
    if bank is None or contract is None:
        default_bank, default_contract = _default_inputs()
        bank = default_bank if bank is None else bank
        contract = default_contract if contract is None else contract
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(message['role'] == 'user' for message in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.3 verification requires one current user turn and explicit harness context')
    for message in messages:
        expected_keys = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                         {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                         {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected_keys:
            raise ValueError('unexpected model-visible message fields')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the pinned S2.3 contract')
    if gene_group(meta['gene_id'], meta['selection_seed']) != (meta['split_unit'], meta['split']):
        raise ValueError('query gene crossed its fixed split')
    if meta['variant'] == 'symbol_context':
        registry = _default_registry() if registry is None else registry
        if get_symbols([meta['gene_id']], registry)[0] != meta['input_gene']:
            raise ValueError('the supplied symbol does not uniquely identify the graph subject')
    elif meta['input_gene'] != meta['gene_id']:
        raise ValueError('the canonical query input changed')
    request = render_s2_3_request(contract['_requests'], split=meta['split'], mode=meta['mode'], direction=meta['direction'],
        variant_index=meta['request_template_index'], multiplex_id=view.multiplex_id, layer_id=meta['layer_id'],
        gene=meta['input_gene'], top_k=meta['top_k'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=request)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != _active(view):
        raise ValueError('active graph metadata differs from the pinned source')
    fact = NeighborFact(meta['split'], meta['gene_id'], meta['layer_id'], meta['direction'],
                        meta['category'], meta['neighbor_counts'], meta['split_unit'])
    if meta['variant'] == 'direction_invalid':
        if (meta['mode'] != 'complete' or meta['top_k'] is not None or meta['category'] != 'diagnostic'
                or meta['neighbor_counts'] != {} or meta['direction'] != 'in' or context['observations']
                or len(messages) != 4 or not messages[-1].get('thinking')):
            raise ValueError('malformed direction diagnostic')
        expected = build_direction_diagnostic(view, fact, meta['framing'], 0, bank, contract,
                                              seed=meta['selection_seed'])['messages'][-1]['content']
        if any(canonical_json(answer[k]) != canonical_json(expected[k]) for k in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('direction diagnostic does not preserve the rejected request')
        return
    runtime = NeighborRuntime(view, index=index, **meta['transport'])
    source_fact = _fact(view, runtime.index, fact.gene_id, fact.layer_id, fact.direction, meta['selection_seed'])
    if canonical_json(asdict(fact)) != canonical_json(asdict(source_fact)):
        raise ValueError('sampling labels differ from the native source; labels cannot grade answers')
    available = list(context['observations'])
    identities = [observation for observation in available if observation['operation'] == 'lookup_human_gene_symbol']
    evidence_ids: dict[str, Mapping[str, Any]] = {}
    if meta['variant'] == 'symbol_context':
        validate_identity_observations(identities, registry)
        if (len(identities) != 1 or identities[0]['result']['gene_id'] != fact.gene_id
                or identities[0]['arguments']['gene_symbol'] != meta['input_gene']):
            raise ValueError('required supplied identity evidence is missing or incompatible')
    elif identities:
        raise ValueError('unexpected identity context')

    def verify(observation: Mapping[str, Any]) -> None:
        if set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise ValueError('unexpected saved-observation fields')
        for records in observation['provenance'].values():
            for record in records:
                evidence_id = record['evidence_id']
                if evidence_id in evidence_ids and canonical_json(evidence_ids[evidence_id]) != canonical_json(observation):
                    raise ValueError('one evidence ID names conflicting acquisitions')
                evidence_ids[evidence_id] = observation
        if observation['operation'] == 'lookup_human_gene_symbol':
            return
        if observation['operation'] != 'get_neighbors' or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed neighbor observation')
        if gene_group(observation['arguments']['gene_id'], meta['selection_seed'])[1] != meta['split']:
            raise ValueError('a saved query subject crosses split boundaries')
        graph = observation['provenance']['graph'][0]
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared stale graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        expected = runtime.get_neighbors(**observation['arguments'], evidence_id=graph['evidence_id'])
        if any(canonical_json(observation[k]) != canonical_json(expected[k]) for k in ('status', 'result', 'provenance')):
            raise ValueError('visible page differs from its pinned source or transport')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if not message.get('thinking'):
            raise ValueError('current-turn reasoning is missing')
        if 'tool_call' not in message:
            continue
        _, pending, _, _ = _plan(runtime, view, fact, meta['mode'], meta['top_k'], available)
        call = message['tool_call']
        if pending is None or canonical_json(call) != canonical_json({'name': 'get_neighbors', 'arguments': pending}):
            raise ValueError('call does not follow required scope, selection, or returned cursor')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('response does not match its call')
        response = response_message['content']
        if response['status'] != 'error' and set(response) != {'status', 'provenance', 'result'}:
            raise ValueError('unexpected successful tool-response fields')
        if previous_response is not None and previous_response['status'] == 'error':
            if (previous_response != _failure(1) or call != previous_call or consecutive_failures != 1):
                raise ValueError('retry lacks its exact allowance or changes arguments')
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('invalid atomic fault schedule or invented failed-page evidence')
            consecutive_failures += 1
        else:
            consecutive_failures = 0
            observation = _observation('get_neighbors', call['arguments'], response)
            verify(observation)
            available.append(observation)
        previous_call, previous_response = call, response
    expected, pending, used, coverage = _plan(runtime, view, fact, meta['mode'], meta['top_k'], available)
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from visible relevant acquisitions')
    if answer['status'] == 'error':
        if pending is None or not previous_response or previous_response['status'] != 'error' or previous_response == _failure(1):
            raise ValueError('error does not follow an unresolved required terminal failure')
        expected = {'failed_tool': 'get_neighbors', 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': coverage}
        decision, status = 'defer', 'error'
    else:
        if pending is not None or expected is None:
            raise ValueError('answer was produced before the requested traversal finished')
        sides = [expected] if fact.direction != 'both' else [s for s in (expected['incoming'], expected['outgoing']) if s is not None]
        missing = any(side['missing_gene_ids'] for side in sides)
        status = 'not_found' if missing else 'resolved'
        decision = 'defer' if missing else 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('answer exceeds visible evidence or changes task completion')


def _requests(fact: NeighborFact) -> list[tuple[str, int | None]]:
    """Exercise count boundaries and an interior K without creating a hidden cap."""
    counts = [n for n in fact.neighbor_counts.values() if n is not None]
    limits = {1}
    for n in counts:
        limits.update((max(1, n // 2), max(1, n), n + 1))
    return [('complete', None), *[('top_k', k) for k in sorted(limits)]]


def _bound_coverage(rows: Mapping[str, Sequence[Mapping[str, Any]]], domains: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Report selected K values and relationships without implying full coverage."""
    groups: dict[tuple, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for split, examples in rows.items():
        for row in examples:
            meta = row['metadata']
            if meta['mode'] != 'top_k' or meta['category'] != 'present':
                continue
            for direction, count in meta['neighbor_counts'].items():
                groups[(split, meta['layer_id'], direction)][count].add(meta['top_k'])
    result = []
    for domain in domains:
        for split in SPLITS:
            by_count = groups[(split, domain['layer_id'], domain['direction'])]
            chosen = sorted(set().union(*by_count.values())) if by_count else []
            unselected = sorted(set(range(1, domain['maximum'] + 2)) - set(chosen))
            ranges = []
            for value in unselected:
                if ranges and value == ranges[-1][1] + 1:
                    ranges[-1][1] = value
                else:
                    ranges.append([value, value])
            result.append({'split': split, 'layer_id': domain['layer_id'], 'direction': domain['direction'],
                           'selected_k': chosen, 'reviewed_k_range': [1, domain['maximum'] + 1],
                           'unselected_k_ranges': ranges,
                           'by_neighbor_count': {str(count): {'selected_k': sorted(values),
                               'relationships': sorted({'below' if k < count else 'equal' if k == count else 'above' for k in values})}
                               for count, values in sorted(by_count.items())}})
    return result


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None, smoke: bool = False,
                 seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write immutable verification artifacts with full source and token audits.

    Explicit counts select all attainable count groups. Smoke uses a declared
    representative schedule. Neither is a production training export: the
    generation budget, context representation, and S2 adapter remain pending.
    """
    from .s2_neighbors import NeighborIndex
    from .s2_3_prompts import validate_s2_3_rendered_prompts
    from .s2_3_tokens import load_tokenizer, audit_rows
    if (counts is None) != smoke:
        raise ValueError('choose explicit counts or smoke=True; there is no default generation budget')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('source manifest differs from its pin')
    context_recipe = contract['context_recipe']
    recipe_path = ROOT / context_recipe['path']
    if file_sha256(recipe_path) != context_recipe['sha256']:
        raise ValueError('context reference recipe differs from its pin')
    view = load_source_view(source_manifest)
    index = NeighborIndex(view)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_3_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']))
    facts, selection = select_facts(view, index, tuple(sorted(registry.symbols_by_id)), counts, seed, smoke=smoke)
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows = {split: [] for split in SPLITS}
    skipped: Counter = Counter()
    # Select a small semantic panel by native counts, independently of wording.
    panel: dict[tuple, NeighborFact] = {}
    for fact in facts:
        largest = max((n or 0 for n in fact.neighbor_counts.values()), default=0)
        key = (fact.split, fact.layer_id, fact.direction, fact.category)
        score = (0 if 2 <= largest <= 8 else 1, largest, fact.split_unit)
        prior = panel.get(key)
        prior_largest = max((n or 0 for n in prior.neighbor_counts.values()), default=0) if prior else None
        prior_score = (0 if 2 <= prior_largest <= 8 else 1, prior_largest, prior.split_unit) if prior else None
        if prior is None or score < prior_score:
            panel[key] = fact
    panel_units = {(f.layer_id, f.direction, f.gene_id) for f in panel.values()}
    for fact in facts:
        for mode, top_k in _requests(fact):
            variants = ['lookup']
            if (fact.layer_id, fact.direction, fact.gene_id) in panel_units and (mode == 'complete' or top_k in (1, 2)):
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
        fact = next((f for f in facts if f.split == split and not view.layers[f.layer_id].directed), None)
        if fact:
            for framing in FRAMINGS:
                row = build_direction_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, seed=seed)
                validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, context_recipe['max_length'])
    audit = {'source_semantics_passed': True, 'training_ready': False, 'production_export_allowed': False,
             'purpose': 'builder_verification', 'selection': selection, 'full_question_wording': wording,
             'bound_schedule': 'positive K at 1, floor(count/2), count, and count+1; zeros use1; compound sides remain separate',
             'bound_coverage': _bound_coverage(rows, selection['domains']),
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    files = {}
    units = {split: {row['metadata']['split_unit'] for row in rows[split]} for split in SPLITS}
    if any(units[left] & units[right] for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    for split in SPLITS:
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage: Counter = Counter()
        joint: dict[str, Counter] = defaultdict(Counter)
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta[k]) for k in ('direction', 'mode', 'category', 'variant'))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(units[split]),
                                 'wrapper_usage': dict(sorted(usage.items())),
                                 'wrapper_by_direction_mode_category_variant': {k: dict(sorted(v.items())) for k, v in sorted(joint.items())},
                                 'variants': dict(sorted(Counter(r['metadata']['variant'] for r in rows[split]).items()))}
        if split != 'train':
            files[f'eval/{split}.questions.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                 'metadata': {k: row['metadata'][k] for k in ('schema_version', 'tool_contract_version', 'question_family', 'split')}}
                for row in rows[split]])
            files[f'eval/{split}.answers.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'expected_messages': row['messages'][3:], 'final_answer': row['messages'][-1]['content']}
                for row in rows[split]])
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    modules = ('s2_3_corpus', 's2_neighbors', 's2_3_prompts', 's2_3_tokens', 's2_2_corpus', 's2_2_identity',
               's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources', 's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_3.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-3-manifest-v1', 'contract_version': contract['contract_version'],
                'implemented_families': ['S2.3'], 'purpose': 'builder_verification', 'training_ready': False,
                'production_export_allowed': False, 'generation_budget': None if smoke else counts,
                'sampling_schedule': 'representative_smoke' if smoke else 'every_attainable_count',
                'seed': seed, 'row_counts': {s: len(rows[s]) for s in SPLITS}, 'python_version': platform.python_version(),
                'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
                'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
                **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
                                                'registry_index', 'tokenizer', 'context_recipe')},
                'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
                'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'fact_selection_sha256': selection['fact_selection_sha256'],
                'tokenizer_render_date': tokens['render_date'],
                'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
                'transport_and_faults': 'declared deterministic page/overlap/empty-page/unknown-total and retry schedules, plus stale version 0x00 on real source records; not deployed captures or a historical graph version',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
