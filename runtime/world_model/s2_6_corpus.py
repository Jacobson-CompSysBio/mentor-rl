"""Build complete S2.6 partitions over finite, source-selected gene sets.

The verification recipe samples real category-count triples without imposing
a runtime input cap. Its audit reports larger unselected sizes and triples.
Every queried gene follows the S2.3–S2.5 subject split across layers and sets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import BUILDER_MODULES, load_s2_prompt_banks, load_s2_edge_requests, render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _active, _observation
from .s2_2_identity import RegistryContext, get_symbols, identity_observations, load_registry_context, validate_identity_observations
from .s2_3_corpus import gene_group, _provenance
from .s2_corpus import ROOT, SPLITS, FRAMINGS, SCHEMA_VERSION, canonical_json, digest, _failure, _jsonl, _write_immutable
from .s2_gene_layers import GeneLayersRuntime
from .s2_neighbors import NeighborIndex, NeighborRuntime
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_6_v1.json')
CATEGORIES = ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids')
VARIANTS = ('lookup', 'duplicates', 'reordered', 'context', 'larger_context', 'partial_context', 'overlapping_context',
            'duplicate_context', 'global_context', 'membership_context', 'bounded_membership_context', 'neighbor_context',
            'neighbor_zero', 'symbol_context', 'stale', 'wrong_layer', 'terminal', 'recovered', 'exhausted',
            'partial_terminal', 'partial_recovered', 'partial_exhausted', 'batched',
            'batched_terminal', 'batched_recovered', 'batched_exhausted')
DIAGNOSTICS = ('unknown_layer', 'unknown_multiplex', 'missing_scope', 'missing_genes', 'invalid_scalar', 'invalid_list')


@dataclass(frozen=True)
class NodesFact:
    """One finite source-selected set before input-order and wording contrasts."""

    split: str
    layer_id: str
    gene_ids: tuple[str, ...]
    category_counts: tuple[int, int, int]
    split_unit: str


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the isolated nodes overlay and its separately pinned request bank."""
    from .s2_6_prompts import load_s2_6_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.6']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.6 build contract')
    contract['_requests'] = load_s2_6_requests(contract['request_bank'])
    return contract


def _fact(view: GraphView, layer_id: str, gene_ids: Sequence[str], seed: int = 20261028,
          *, split: str | None = None) -> NodesFact:
    """Measure a finite set while keeping every subject in its fixed split."""
    genes = tuple(dict.fromkeys(gene_ids))
    subject_splits = {gene_group(gene, seed)[1] for gene in genes}
    if len(subject_splits) > 1 or split is not None and subject_splits and split not in subject_splits:
        raise ValueError('a classification set cannot cross fixed subject splits')
    split = next(iter(subject_splits)) if subject_splits else split or 'train'
    if split not in SPLITS or layer_id not in view.layers:
        raise ValueError('unknown set split or layer')
    nodes = view.layers[layer_id].nodes
    counts = (sum(gene in nodes for gene in genes),
              sum(gene in view.multiplex_nodes and gene not in nodes for gene in genes),
              sum(gene not in view.multiplex_nodes for gene in genes))
    unit = digest({'seed': seed, 'split': split, 'layer_id': layer_id, 'gene_ids': sorted(genes)})
    return NodesFact(split, layer_id, genes, counts, unit)


def _triple_rows(size: int, populations: Sequence[int], pattern: Sequence[int] | None = None):
    """Describe feasible count rows without enumerating a quadratic triple domain."""
    lower = [int(bool(value)) for value in pattern] if pattern is not None else [0, 0, 0]
    upper = [n if pattern is None or pattern[i] else 0 for i, n in enumerate(populations)]
    for a in range(lower[0], min(upper[0], size - lower[1] - lower[2]) + 1):
        low = max(lower[1], size - a - upper[2])
        high = min(upper[1], size - a - lower[2])
        if low <= high:
            yield a, low, high


def feasible_triple_count(size: int, populations: Sequence[int]) -> int:
    """Count all attainable category triples for one finite input size."""
    return sum(high - low + 1 for _, low, high in _triple_rows(size, populations))


def _triples(size: int, populations: Sequence[int], smoke: bool) -> list[tuple[int, int, int]]:
    """Use all small triples or declared extrema/median ranks for smoke coverage."""
    if not smoke or size <= 3:
        return [(a, b, size - a - b) for a, low, high in _triple_rows(size, populations) for b in range(low, high + 1)]
    selected = set()
    for pattern in product((0, 1), repeat=3):
        if not any(pattern):
            continue
        rows = list(_triple_rows(size, populations, pattern))
        count = sum(high - low + 1 for _, low, high in rows)
        for rank in {0, count // 2, count - 1} if count else ():
            for a, low, high in rows:
                width = high - low + 1
                if rank < width:
                    b = low + rank
                    selected.add((a, b, size - a - b))
                    break
                rank -= width
    return sorted(selected)


def _available_sets(populations: Sequence[int], triple: Sequence[int], quota: int) -> int:
    """Bound distinct-set availability by the requested quota for omission reports."""
    available = 1
    for population, chosen in zip(populations, triple):
        available *= math.comb(population, chosen)
        if available >= quota:
            return quota
    return available


def select_facts(view: GraphView, registry_ids: Sequence[str], counts: Mapping[str, int], seed: int, *,
                 input_sizes: Sequence[int], smoke: bool = False) -> tuple[list[NodesFact], dict]:
    """Sample disjoint category sets and report all unselected source ranges.

    Whole-graph and registry-conditioned populations remain separate. Smoke
    allocates a finite verification schedule; no schedule limits the runtime.
    Explicit quotas visit every feasible triple at the caller's chosen sizes.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    if (not input_sizes or any(type(n) is not int or n < 0 for n in input_sizes)):
        raise ValueError('input sizes require explicit nonnegative integers')
    sizes = sorted(set(input_sizes))
    registered = set(registry_ids)
    outside = registered - view.multiplex_nodes
    if not outside:
        raise ValueError('the registry has no globally absent candidates')
    universe = view.multiplex_nodes | outside
    groups = {gene: gene_group(gene, seed)[1] for gene in universe}
    facts, domains, cells, omissions = {}, [], [], []
    for layer_id, layer in view.layers.items():
        whole = (layer.nodes, view.multiplex_nodes - layer.nodes, outside)
        registry_pops = [len(pool & registered) for pool in whole]
        for split in SPLITS:
            pools = [sorted(gene for gene in pool if groups[gene] == split) for pool in whole]
            populations = [len(pool) for pool in pools]
            maximum = sum(populations)
            unselected_ranges, prior = [], -1
            for size in [*(n for n in sizes if n <= maximum), maximum + 1]:
                if size > prior + 1:
                    unselected_ranges.append([prior + 1, size - 1])
                prior = size
            domains.append({'layer_id': layer_id, 'split': split,
                'whole_view_category_populations': [len(pool) for pool in whole],
                'registered_category_populations': registry_pops,
                'split_category_populations': populations, 'attainable_input_size_range': [0, maximum],
                'selected_input_sizes': [n for n in sizes if n <= maximum],
                'unselected_input_size_ranges': unselected_ranges})
            for size in sizes:
                triples = _triples(size, populations, smoke)
                possible = feasible_triple_count(size, populations)
                cells.append({'layer_id': layer_id, 'split': split, 'input_size': size,
                              'feasible_triple_count': possible, 'selected_triples': [list(t) for t in triples],
                              'unselected_triple_count': possible - len(triples)})
                if not triples:
                    omissions.append({'layer_id': layer_id, 'split': split, 'input_size': size,
                                      'reason': 'input size exceeds the fixed-split candidate population'})
                for triple in triples:
                    quota = counts[split]
                    limit = _available_sets(populations, triple, quota)
                    chosen_sets, attempt = set(), 0
                    while len(chosen_sets) < limit:
                        rng = random.Random(int(digest([seed, layer_id, split, size, triple, attempt]), 16))
                        genes = [gene for pool, n in zip(pools, triple) for gene in rng.sample(pool, n)]
                        rng.shuffle(genes)
                        attempt += 1
                        key = tuple(sorted(genes))
                        if key in chosen_sets:
                            continue
                        chosen_sets.add(key)
                        fact = _fact(view, layer_id, genes, seed, split=split)
                        facts[(layer_id, split, key)] = fact
                    if limit < quota:
                        omissions.append({'layer_id': layer_id, 'split': split, 'input_size': size,
                                          'category_counts': list(triple), 'requested': quota, 'selected': limit,
                                          'reason': 'fewer distinct input sets exist in this cell'})
    selected = sorted(facts.values(), key=lambda f: (SPLITS.index(f.split), view.layer_order.index(f.layer_id),
                                                   len(f.gene_ids), f.category_counts, f.split_unit))
    return selected, {'domains': domains, 'cells': cells, 'omissions': omissions,
                      'selected_facts': len(selected), 'distinct_query_genes': len({g for f in selected for g in f.gene_ids}),
                      'input_sizes': sizes, 'fact_selection_sha256': digest([asdict(f) for f in selected]),
                      'split_policy': 'every nonempty set uses only SHA256(seed,gene) subjects from one S2.3/S2.4/S2.5 split; empty sets have a split-specific namespace',
                      'edge_fact_disjoint': False,
                      'candidate_policy': 'all graph nodes plus verified registry IDs outside the multiplex; no registry filter on graph membership',
                      'triple_policy': 'all feasible triples' if not smoke else 'all triples at sizes0..3; extrema and median ranks within each feasible presence pattern at larger selected sizes',
                      'conditioning': 'equal quota per selected size/category-count cell and fixed split; random real identities and input order independent of wording'}


def _arguments(view: GraphView, fact: NodesFact) -> dict:
    """Preserve canonical first-mention order in the exact three-field call."""
    return {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids)}


def _global_observation(view: GraphView, genes: Sequence[str], evidence_id: str) -> dict:
    """Supply an attributed global partition without a current-turn acquisition."""
    return {'operation': 'check_multiplex_membership',
            'arguments': {'multiplex_id': view.multiplex_id, 'gene_ids': list(genes)}, 'status': 'resolved',
            'result': {'multiplex_id': view.multiplex_id,
                       'present_gene_ids': sorted(g for g in genes if g in view.multiplex_nodes),
                       'absent_gene_ids': sorted(g for g in genes if g not in view.multiplex_nodes)},
            'provenance': {'registry': [], 'graph': [{'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
                'layer_scope': {'kind': 'all', 'layer_ids': [], 'layer_family': None}, 'evidence_id': evidence_id}]}}


def _plan(view: GraphView, arguments: dict, observations: Sequence[dict], batch_limit: int | None):
    """Retain unique registry mappings beside the source-blind graph proof."""
    from .s2_6_evidence import collect
    proof = collect(view, arguments, observations, batch_limit=batch_limit)
    identities = [o for o in observations if o['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _reason(result: dict, framing: str, used: Sequence[dict]) -> str:
    """Explain the complete visible partition without treating negatives as blockers."""
    coverage = result['coverage']
    counts = [len(result[name]) for name in CATEGORIES]
    text = (f"The request classifies {len(coverage['requested_gene_ids'])} unique supplied genes in {result['layer_id']}. "
            f"Visible evidence establishes {counts[0]} layer members, {counts[1]} multiplex members outside the layer, "
            f"and {counts[2]} genes outside the multiplex. The three groups form an exact disjoint partition. "
            'Sort each category by canonical ID, while coverage retains first-mention order. '
            f"The positive count is {result['layer_gene_count']}; it counts requested matches, not every node in the layer. ")
    if not coverage['requested_gene_ids']:
        text += 'The explicitly empty domain needs no lookup or graph evidence and says nothing about the full layer inventory. '
    else:
        first = coverage['requested_gene_ids'][0]
        category = next(name for name in CATEGORIES if first in result[name])
        text += f'The first requested ID, {first}, belongs to {category} in the visible partition. '
    if any(o['operation'] == 'get_neighbors' for o in used):
        text += 'The neighbor record supplies explicit membership fields; zero directional degree does not prove absence. '
    if any(o['operation'] == 'get_gene_layers' for o in used):
        text += 'Reuse only explicit positive layers or exhaustive covered-layer negatives from containing-layer records. '
    if len(used) > 1:
        text += 'Keep original acquisitions, deduplicate overlapping classifications, and calculate counts over the visible union. '
    return text + ('This complete classification supplies the workflow step, so continue.' if framing == 'step'
                   else 'This completes the standalone classification, so stop.')


def build_row(view: GraphView, fact: NodesFact, framing: str, variant: str, variant_index: int,
              bank: Mapping[str, Any], contract: Mapping[str, Any], *, registry: RegistryContext | None = None,
              index: NeighborIndex | None = None, seed: int = 20261028) -> dict:
    """Build complete partitions with supported context and optional explicit batches."""
    from .s2_6_prompts import render_s2_6_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported S2.6 contrast or framing')
    if variant == 'reordered':
        if len(fact.gene_ids) < 2:
            raise VariantUnavailable('input order contrasts need at least two IDs')
        fact = replace(fact, gene_ids=tuple(reversed(fact.gene_ids)))
    if variant != 'lookup' and not fact.gene_ids:
        raise VariantUnavailable('the explicit empty set needs no evidence contrast')
    raw_genes = list(fact.gene_ids)
    if variant == 'duplicates':
        raw_genes = [raw_genes[0], *raw_genes, raw_genes[0], raw_genes[-1]]
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('symbol context needs a verified registry')
        try:
            raw_genes = list(get_symbols(fact.gene_ids, registry))
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    batch_limit = None
    if variant.startswith('batched'):
        if len(fact.gene_ids) < 2:
            raise VariantUnavailable('complete batches require multiple inputs')
        batch_limit = max(1, (len(fact.gene_ids) + 1) // 2)
    core = render_s2_6_request(contract['_requests'], split=fact.split, variant_index=variant_index,
                             multiplex_id=view.multiplex_id, layer_id=fact.layer_id, genes=raw_genes)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    row_id = 's26_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, user])[:32]
    runtime = NodesByLayerRuntime(view)
    arguments = _arguments(view, fact)
    observations, serial = [], 0

    def acquire(call_arguments: dict) -> dict:
        nonlocal serial
        serial += 1
        return runtime.get_nodes_by_layer(**call_arguments, evidence_id=f'ev_{row_id}_{serial}')

    def save(genes: Sequence[str], *, layer_id: str | None = None) -> None:
        saved = {**arguments, 'layer_id': layer_id or fact.layer_id, 'gene_ids': list(genes)}
        observations.append(_observation('get_nodes_by_layer', saved, acquire(saved)))

    if variant == 'symbol_context':
        observations.extend(identity_observations(fact.gene_ids, row_id, registry))
    elif variant in ('context', 'duplicate_context', 'stale'):
        save(fact.gene_ids)
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
        if variant == 'stale':
            observations[0]['provenance']['graph'][0]['graph_version'] = '0' * 64
    elif variant == 'larger_context':
        extras = sorted((g for g in view.multiplex_nodes if g not in fact.gene_ids and gene_group(g, seed)[1] == fact.split),
                        key=lambda gene: digest([seed, fact.split_unit, 'larger_context', gene]))
        if not extras:
            raise VariantUnavailable('no same-split extra query subject')
        save([*reversed(fact.gene_ids), *extras[:2]])
    elif variant in ('partial_context', 'partial_terminal', 'partial_recovered', 'partial_exhausted'):
        if len(fact.gene_ids) < 2:
            raise VariantUnavailable('partial context requires multiple inputs')
        save(fact.gene_ids[:max(1, len(fact.gene_ids) // 2)])
    elif variant == 'overlapping_context':
        if len(fact.gene_ids) < 3:
            raise VariantUnavailable('overlapping records require at least three inputs')
        save(fact.gene_ids[:2])
        save(fact.gene_ids[1:])
    elif variant == 'global_context':
        observations.append(_global_observation(view, fact.gene_ids, f'ev_{row_id}_global'))
    elif variant in ('membership_context', 'bounded_membership_context'):
        member_runtime = GeneLayersRuntime(view)
        for ordinal, gene in enumerate(fact.gene_ids, 1):
            source = member_runtime.arguments(gene, {'kind': 'all', 'layer_ids': [], 'layer_family': None},
                                              mode='top_k' if variant == 'bounded_membership_context' else 'complete',
                                              top_k=1 if variant == 'bounded_membership_context' else None)
            response = member_runtime.get_gene_layers(**source, evidence_id=f'ev_{row_id}_membership_{ordinal}')
            observations.append(_observation('get_gene_layers', source, response))
    elif variant in ('neighbor_context', 'neighbor_zero'):
        native = NeighborRuntime(view, index=index)
        genes = list(fact.gene_ids)
        if variant == 'neighbor_zero':
            if not view.layers[fact.layer_id].directed:
                raise VariantUnavailable('a directional zero needs a directed layer')
            genes = [g for g in genes if g in view.layers[fact.layer_id].nodes and not native.index.neighbors(fact.layer_id, g, 'out')]
            if not genes:
                raise VariantUnavailable('this request has no present zero-outdegree subject')
        for ordinal, gene in enumerate(genes, 1):
            source = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_id': gene,
                      'direction': 'out' if view.layers[fact.layer_id].directed else 'undirected',
                      'mode': 'top_k', 'top_k': 1, 'cursor': None}
            response = native.get_neighbors(**source, evidence_id=f'ev_{row_id}_neighbor_{ordinal}')
            observations.append(_observation('get_neighbors', source, response))
    elif variant == 'wrong_layer':
        other = next((layer for layer in view.layer_order if layer != fact.layer_id), None)
        if other is None:
            raise VariantUnavailable('no alternative registered layer')
        save(fact.gene_ids, layer_id=other)
    proof, _ = _plan(view, arguments, observations, batch_limit)
    if variant in ('partial_terminal', 'partial_recovered', 'partial_exhausted') and proof.next_arguments is None:
        raise VariantUnavailable('partial context already completes the set')
    context = {'active_graph': _active(view), 'observations': deepcopy(observations)}
    if batch_limit is not None:
        context['batch_limit'] = batch_limit
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    kind = variant.removeprefix('partial_').removeprefix('batched_')
    injected, failed, successful_calls = False, None, 0
    while True:
        proof, used = _plan(view, arguments, observations, batch_limit)
        if proof.next_arguments is None:
            break
        pending = proof.next_arguments
        thought = (f"Classify all {len(fact.gene_ids)} unique supplied genes in {fact.layer_id}. "
                   f"Visible evidence still leaves {len(proof.coverage['unchecked_gene_ids'])} inputs unchecked. "
                   f"Call get_nodes_by_layer for the {len(pending['gene_ids'])} required IDs in first-mention order. "
                   'The batch must classify every supplied ID atomically; it has no cursor or default limit.')
        if variant == 'symbol_context':
            thought = 'The supplied unique registry mappings resolve every input symbol to its canonical ID. ' + thought
        if batch_limit is not None:
            thought += f' The visible harness explicitly limits each complete batch to {batch_limit} IDs.'

        def append(response: dict, reasoning: str) -> None:
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                              'tool_call': {'name': 'get_nodes_by_layer', 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': 'get_nodes_by_layer', 'content': response}])

        should_fail = (not injected and kind in ('terminal', 'recovered', 'exhausted')
                       and (not variant.startswith('batched_') or successful_calls > 0))
        if should_fail:
            injected = True
            response = _failure(None if kind == 'terminal' else 1)
            append(response, thought)
            if kind != 'terminal':
                response = _failure(0) if kind == 'exhausted' else acquire(pending)
                append(response, 'The failed atomic batch supplies retries_remaining=1. Repeat the identical tool and all three arguments. '
                                 'Keep earlier classifications; stop retries after recovery or exhaustion.')
            if response['status'] == 'error':
                failed = (pending, response)
                break
        else:
            response = acquire(pending)
            append(response, thought)
        successful_calls += 1
        observations.append(_observation('get_nodes_by_layer', deepcopy(pending), response))
    if failed:
        pending, response = failed
        result = {'failed_tool': 'get_nodes_by_layer', 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': proof.coverage}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result, 'continuation_decision': 'defer',
                  'blocked_reason': 'The membership backend must recover before the remaining inputs can be classified.',
                  'summary': 'The required atomic classification failed; preserve supported earlier classifications.'}
        thought = (f"The required batch for {fact.layer_id} failed. "
                   + ('The retry allowance is exhausted. ' if kind == 'exhausted' else 'No retry allowance was supplied. ')
                   + f"Coverage retains {len(proof.coverage['checked_gene_ids'])} checked and {len(proof.coverage['unchecked_gene_ids'])} unchecked IDs. "
                   'The failed batch adds no evidence. Do not assign unchecked inputs to empty negative placeholders. Defer for backend recovery.')
    else:
        result = proof.result
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
                  'summary': 'The complete classification separates layer members, other multiplex members, and globally absent inputs.'}
        thought = ('No new call is required; supplied evidence or the explicit empty domain suffices. ' if len(messages) == 3 else '') + _reason(result, framing, used)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.6', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids), 'input_genes': raw_genes,
        'input_size': len(fact.gene_ids), 'category_counts': list(fact.category_counts),
        'framing': framing, 'variant': variant, 'batch_limit': batch_limit, 'wrapper_index': wrapper,
        'request_template_index': variant_index % len(contract['_requests'][fact.split]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view: GraphView, fact: NodesFact, kind: str) -> tuple[dict, str, str]:
    """Keep the original rejected JSON shape separate from canonical graph input."""
    request = {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'unknown_multiplex': 'UNKNOWN_MULTIPLEX',
             'missing_scope': 'MISSING_SCOPE', 'missing_genes': 'MISSING_GENES',
             'invalid_scalar': 'INVALID_GENE_LIST', 'invalid_list': 'INVALID_GENE_LIST'}
    if kind not in codes:
        raise ValueError('unsupported nodes prerequisite diagnostic')
    if kind == 'unknown_layer':
        request['layer_id'] = 'TFs:missing_context'
    elif kind == 'unknown_multiplex':
        request['multiplex_id'] = 'unknown_nodes_multiplex'
    elif kind == 'missing_scope':
        request['layer_id'] = None
    elif kind == 'missing_genes':
        request['genes'] = None
    elif kind == 'invalid_scalar':
        request['genes'] = fact.gene_ids[0] if fact.gene_ids else 'ENSG00000000001'
    elif kind == 'invalid_list':
        request['genes'] = [fact.gene_ids[0] if fact.gene_ids else 'ENSG00000000001', {'invalid': 'gene'}]
    return request, 'not_found' if kind in ('unknown_layer', 'unknown_multiplex') else 'invalid_input', codes[kind]


def build_diagnostic(view: GraphView, fact: NodesFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'missing_genes', seed: int = 20261028) -> dict:
    """Reject a literal invalid prerequisite without a graph-negative partition."""
    from .s2_6_prompts import render_s2_6_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    core = render_s2_6_diagnostic_request(contract['_requests'], split=fact.split, kind=kind, variant_index=variant_index,
        multiplex_id=request['multiplex_id'], layer_id=request['layer_id'], genes=request['genes'])
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    answer = {'status': status, 'provenance': {'registry': [], 'graph': []},
              'result': {'issue_code': code, 'request': request, 'rejected_arguments': None, 'partial_results': []},
              'continuation_decision': 'defer', 'blocked_reason': 'A valid finite gene list and registered graph scope are required.',
              'summary': 'The request has an unresolved prerequisite; no gene classification was performed.'}
    return {'row_id': 's26_' + digest([contract['contract_version'], view.graph_version, user, code])[:32],
            'messages': [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
                {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking':
                    f"The literal request is {request}. The visible prerequisite fails with {code}. "
                    'A missing list is not an explicitly empty set. An invalid input or unregistered scope is not a checked membership negative. '
                    'Make no graph call, preserve the original JSON value and scope, and defer for the missing prerequisite.', 'content': answer}],
            'tools': contract['tools'], 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.6', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
                'input_genes': request['genes'], 'input_size': None, 'category_counts': None,
                'framing': framing, 'variant': kind, 'batch_limit': None, 'wrapper_index': wrapper,
                'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][kind]),
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: NeighborIndex | None = None) -> None:
    """Replay inventories and observations, then grade the visible complete proof.

    Source labels validate sampling only. Unknown inputs remain unchecked after
    failure; their private source categories cannot fill a missing observation.
    Summary and blocker wording remain free prose.
    """
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_6_prompts import render_s2_6_request, render_s2_6_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(m['role'] == 'user' for m in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.6 requires one user turn and explicit harness context')
    for message in messages:
        expected_keys = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                         {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                         {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected_keys or message['role'] == 'assistant' and not message['thinking'].strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the nodes overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.6'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('nodes row identity differs from the build contract')
    fact = _fact(view, meta['layer_id'], meta['gene_ids'], meta['selection_seed'], split=meta['split'])
    if fact.split_unit != meta['split_unit'] or list(fact.gene_ids) != meta['gene_ids']:
        raise ValueError('the ordered unique input domain differs from its fixed split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if diagnostic:
        request, _, _ = _diagnostic_request(view, fact, meta['variant'])
        if meta['input_genes'] != request['genes']:
            raise ValueError('the diagnostic changes the literal gene input')
        core = render_s2_6_diagnostic_request(contract['_requests'], split=meta['split'], kind=meta['variant'],
            variant_index=meta['request_template_index'], multiplex_id=request['multiplex_id'],
            layer_id=request['layer_id'], genes=request['genes'])
    else:
        if len(fact.gene_ids) != meta['input_size'] or list(fact.category_counts) != meta['category_counts']:
            raise ValueError('private sampling labels differ from the source inventories')
        if meta['variant'] == 'symbol_context':
            if registry is None or list(get_symbols(fact.gene_ids, registry)) != meta['input_genes']:
                raise ValueError('symbols do not uniquely identify the canonical inputs')
        elif (not isinstance(meta['input_genes'], list)
              or list(dict.fromkeys(meta['input_genes'])) != list(fact.gene_ids)):
            raise ValueError('raw input order does not resolve to the canonical first mentions')
        core = render_s2_6_request(contract['_requests'], split=meta['split'], variant_index=meta['request_template_index'],
            multiplex_id=view.multiplex_id, layer_id=fact.layer_id, genes=meta['input_genes'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=core)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    expected_keys = {'active_graph', 'observations'} | ({'batch_limit'} if meta['batch_limit'] is not None else set())
    if (set(context) != expected_keys or context['active_graph'] != _active(view)
            or context.get('batch_limit') != meta['batch_limit']):
        raise ValueError('scope metadata or the explicit batch constraint changed')
    if diagnostic:
        if context['observations'] or len(messages) != 4 or meta['batch_limit'] is not None:
            raise ValueError('malformed local prerequisite diagnostic')
        expected = build_diagnostic(view, fact, meta['framing'], meta['request_template_index'], bank, contract,
            kind=meta['variant'], seed=meta['selection_seed'])['messages'][-1]['content']
        if any(canonical_json(answer[k]) != canonical_json(expected[k]) for k in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('the diagnostic changes the rejected request')
        return
    runtime = NodesByLayerRuntime(view)
    arguments = _arguments(view, fact)
    available = list(context['observations'])
    identities = [o for o in available if o['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        validate_identity_observations(identities, registry)
        if (len(identities) != len(fact.gene_ids)
                or [o['result']['gene_id'] for o in identities] != list(fact.gene_ids)
                or [o['arguments']['gene_symbol'] for o in identities] != meta['input_genes']):
            raise ValueError('required supplied identity evidence is missing or changed')
    elif identities:
        raise ValueError('unexpected identity context')
    evidence_ids, native, memberships = {}, None, None

    def verify(observation: Mapping[str, Any]) -> None:
        nonlocal native, memberships
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
        if tool not in ('get_nodes_by_layer', 'get_neighbors', 'get_gene_layers', 'check_multiplex_membership') or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed graph observation')
        source_args = observation['arguments']
        genes = source_args.get('gene_ids', [source_args.get('gene_id')])
        if any(gene_group(gene, meta['selection_seed'])[1] != meta['split'] for gene in genes):
            raise ValueError('a supplied query subject crossed split boundaries')
        graph = observation['provenance']['graph'][0]
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        if tool == 'check_multiplex_membership':
            if canonical_json(observation) != canonical_json(_global_observation(view, genes, graph['evidence_id'])):
                raise ValueError('global context differs from the source inventory')
            return
        if tool == 'get_neighbors':
            if native is None:
                native = NeighborRuntime(view, index=index)
            expected = native.get_neighbors(**source_args, evidence_id=graph['evidence_id'])
        elif tool == 'get_gene_layers':
            if memberships is None:
                memberships = GeneLayersRuntime(view)
            expected = memberships.get_gene_layers(**source_args, evidence_id=graph['evidence_id'])
        else:
            expected = runtime.get_nodes_by_layer(**source_args, evidence_id=graph['evidence_id'])
        if any(canonical_json(observation[k]) != canonical_json(expected[k]) for k in ('status', 'result', 'provenance')):
            raise ValueError('an observation differs from its source inventory')

    for observation in available:
        verify(observation)
    previous_call = previous_response = None
    consecutive_failures = 0
    for position, message in enumerate(messages):
        if 'tool_call' not in message:
            continue
        proof, _ = _plan(view, arguments, available, meta['batch_limit'])
        call = message['tool_call']
        if proof.next_arguments is None or canonical_json(call) != canonical_json({'name': proof.next_tool, 'arguments': proof.next_arguments}):
            raise ValueError('a call changes the required ordered batch or scope')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes the failed batch')
        if response['status'] == 'error':
            if response not in (_failure(None), _failure(1), _failure(0)):
                raise ValueError('invalid fault schedule or invented failed-batch evidence')
            consecutive_failures += 1
        else:
            if set(response) != {'status', 'provenance', 'result'}:
                raise ValueError('unexpected successful response fields')
            consecutive_failures = 0
            observation = _observation(call['name'], call['arguments'], response)
            verify(observation)
            available.append(observation)
        previous_call, previous_response = call, response
    proof, used = _plan(view, arguments, available, meta['batch_limit'])
    if canonical_json(answer['provenance']) != canonical_json(_provenance(used)):
        raise ValueError('final evidence differs from supporting visible classifications')
    if answer['status'] == 'error':
        if (proof.next_arguments is None or not previous_response or previous_response['status'] != 'error'
                or previous_response == _failure(1)):
            raise ValueError('an error requires an unresolved terminal batch failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes complete input classification')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer changes the visible partition or task completion')


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None, smoke: bool = False,
                 input_sizes: Sequence[int] | None = None, seed: int = 20261028,
                 contract_path: Path | None = None, source_manifest: Path | None = None) -> dict[str, Any]:
    """Write immutable finite-set classifications and full source/token audits.

    Smoke is an explicit representative verification allocation. It reports
    unselected sizes and feasible triples, including larger source domains.
    Production quotas and context representation remain separate decisions.
    """
    from .s2_2_prompts import load_s2_2_requests
    from .s2_3_prompts import load_s2_3_requests
    from .s2_4_prompts import load_s2_4_requests
    from .s2_5_prompts import load_s2_5_requests
    from .s2_3_tokens import load_tokenizer, audit_rows
    from .s2_6_prompts import validate_s2_6_rendered_prompts
    if (counts is None) != smoke or smoke and input_sizes is not None or not smoke and input_sizes is None:
        raise ValueError('choose smoke alone or explicit quotas with input_sizes; no production budget is defaulted')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    sizes = contract['smoke_input_sizes'] if smoke else list(input_sizes)
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('the source manifest differs from its pin')
    recipe = contract['context_recipe']
    if file_sha256(ROOT / recipe['path']) != recipe['sha256']:
        raise ValueError('the context reference differs from its pin')
    view = load_source_view(source_manifest)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_6_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']), load_s2_4_requests(contract['s24_request_bank']),
        load_s2_5_requests(contract['s25_request_bank']))
    facts, selection = select_facts(view, tuple(sorted(registry.symbols_by_id)), counts, seed, input_sizes=sizes, smoke=smoke)
    index = NeighborIndex(view)
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped, panel = {split: [] for split in SPLITS}, Counter(), {}
    for fact in facts:
        if not 0 < len(fact.gene_ids) <= 8:
            continue
        presence = tuple(n > 0 for n in fact.category_counts)
        key = (fact.split, fact.layer_id, presence)
        score = (0 if len(fact.gene_ids) >= 3 else 1, len(fact.gene_ids), fact.split_unit)
        if key not in panel or score < panel[key][0]:
            panel[key] = (score, fact)
    panel_units = {f.split_unit for _, f in panel.values()}
    for fact in facts:
        variants = VARIANTS if fact.split_unit in panel_units else ('lookup',)
        for variant in variants:
            for framing in FRAMINGS:
                ordinal = len(rows[fact.split]) // 2
                try:
                    row = build_row(view, fact, framing, variant, ordinal, bank, contract,
                                    registry=registry, index=index, seed=seed)
                except VariantUnavailable:
                    skipped[variant] += 1
                    continue
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                rows[fact.split].append(row)
    for split in SPLITS:
        fact = next((f for f in facts if f.split == split and f.gene_ids), None)
        if fact:
            for kind in DIAGNOSTICS:
                for framing in FRAMINGS:
                    row = build_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, kind=kind, seed=seed)
                    validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                    rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
             'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    subjects = {split: {g for row in rows[split] for g in row['metadata']['gene_ids']} for split in SPLITS}
    if any(subjects[left] & subjects[right] for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate finite-set row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, size_histogram, count_histogram = Counter(), defaultdict(Counter), Counter(), Counter()
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta[k]) for k in ('input_size', 'category_counts', 'variant'))] += 1
            if meta['input_size'] is not None:
                size_histogram[meta['input_size']] += 1
                count_histogram['/'.join(map(str, meta['category_counts']))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_size_categories_variant': {k: dict(sorted(v.items())) for k, v in sorted(joint.items())},
            'input_size_histogram': dict(sorted(size_histogram.items())),
            'category_count_histogram': dict(sorted(count_histogram.items())),
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
    modules = ('s2_6_corpus', 's2_nodes_by_layer', 's2_6_evidence', 's2_6_prompts', 's2_5_evidence', 's2_5_prompts',
               's2_gene_layers', 's2_4_prompts', 's2_3_corpus', 's2_3_prompts', 's2_3_tokens', 's2_neighbors',
               's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources',
               's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_6.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-6-manifest-v1', 'contract_version': contract['contract_version'],
                'implemented_families': ['S2.6'], 'purpose': 'builder_verification', 'training_ready': False,
                'production_export_allowed': False, 'generation_budget': None if smoke else counts,
                'input_sizes': sizes, 'sampling_schedule': 'representative_smoke' if smoke else 'every_feasible_triple_at_explicit_sizes',
                'seed': seed, 'row_counts': {s: len(rows[s]) for s in SPLITS}, 'python_version': platform.python_version(),
                'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
                'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
                **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
                    's23_request_bank', 's24_request_bank', 's25_request_bank', 'registry_index', 'tokenizer', 'context_recipe')},
                'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
                'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
                'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
                'transport_and_faults': 'injected explicit batch limits, retry schedules and stale version0x00; no implicit runtime cap, pagination, or deployed captures',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
