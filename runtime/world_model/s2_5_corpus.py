"""Build S2.5 membership traces from pinned inventories and visible proofs.

The builder separates registry identity, global membership, and scoped layer
membership. A checked global negative completes this classification. Subject
splits match S2.3/S2.4; they do not claim separation of native graph facts.
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
from .s2_multiplex import MultiplexRuntime
from .s2_gene_layers import GeneLayersRuntime
from .s2_neighbors import NeighborIndex, NeighborRuntime
from .s2_sources import GraphView, load_source_view
from .sources import file_sha256


CONTRACT_PATH = Path(__file__).with_name('s2_5_v1.json')
VARIANTS = ('lookup', 'scope_order', 'context', 'complete_context', 'bounded_context', 'shorter_prefix_context',
            'projected_context', 'broader_bounded_context', 'partial_context', 'family_context', 'overlapping_context',
            'duplicate_context', 'prefix_context', 'paged', 'overlap', 'empty_page', 'unknown_total',
            'terminal', 'recovered', 'exhausted', 'page_terminal', 'page_recovered', 'page_exhausted',
            'partial_terminal', 'partial_recovered', 'partial_exhausted', 'symbol_context',
            'neighbor_context', 'neighbor_zero', 'global_context', 'stale', 'wrong_gene')
DIAGNOSTICS = ('unknown_layer', 'unknown_family', 'missing_scope', 'empty_scope', 'missing_limit', 'invalid_limit')


@dataclass(frozen=True)
class MembershipFact:
    """One source-selected classification before wording or transport contrasts."""

    split: str
    gene_id: str
    layer_scope: dict[str, Any]
    category: str
    present: bool
    layer_count: int
    split_unit: str


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the isolated membership overlay and independently pinned requests."""
    from .s2_5_prompts import load_s2_5_requests
    contract = json.loads(Path(path).read_text())
    if (contract['row_schema_version'] != SCHEMA_VERSION or contract['implemented_families'] != ['S2.5']
            or contract['training_ready'] is not False):
        raise ValueError('unsupported S2.5 build contract')
    contract['_requests'] = load_s2_5_requests(contract['request_bank'])
    return contract


def scope_grid(runtime: GeneLayersRuntime) -> list[dict]:
    """Enumerate real explicit subsets and exhaustive registered family selectors."""
    layers = runtime.view.layer_order
    return [*({'kind': 'layers', 'layer_ids': list(group), 'layer_family': None}
              for size in range(1, len(layers) + 1) for group in combinations(layers, size)),
            *({'kind': 'all', 'layer_ids': [], 'layer_family': family}
              for family in (None, *runtime.multiplex.families))]


def _fact(runtime: GeneLayersRuntime, gene: str, scope: dict, seed: int) -> MembershipFact:
    """Measure classification strata without using their labels to grade answers."""
    unit, split = gene_group(gene, seed)
    selected = runtime.resolve_scope(scope)
    present = gene in runtime.view.multiplex_nodes
    n = sum(gene in runtime.view.layers[layer].nodes for layer in selected)
    category = ('outside_multiplex' if not present else 'empty_scope' if not selected else
                'selected_absence' if n == 0 else 'positive')
    return MembershipFact(split, gene, deepcopy(scope), category, present, n, unit)


def select_facts(view: GraphView, registry_ids: Sequence[str], counts: Mapping[str, int],
                 seed: int) -> tuple[list[MembershipFact], dict]:
    """Sample actual count strata and membership patterns within fixed subjects.

    Whole-graph and registry-conditioned histograms remain separate. Registry
    absence never removes a genuine graph node. Globally absent candidates
    come from the verified registry, and have their own classification stratum.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts require positive integers for all three splits')
    runtime = GeneLayersRuntime(view)
    registered = set(registry_ids)
    outside = sorted(registered - view.multiplex_nodes)
    if not outside:
        raise ValueError('the registry has no globally absent query candidates')
    groups = {gene: gene_group(gene, seed) for gene in view.multiplex_nodes | set(outside)}
    outside_pools = {split: sorted((g for g in outside if groups[g][1] == split), key=lambda g: groups[g][0]) for split in SPLITS}
    selected, domains, cells, patterns, omissions = {}, [], [], [], []

    def add(gene: str, scope: dict) -> None:
        selected[(gene, canonical_json(scope))] = _fact(runtime, gene, scope, seed)

    for scope in scope_grid(runtime):
        layers = runtime.resolve_scope(scope)
        pools, pattern_pools = defaultdict(list), defaultdict(list)
        whole, registry_histogram = Counter(), Counter()
        for gene in sorted(view.multiplex_nodes):
            mask = tuple(layer for layer in layers if gene in view.layers[layer].nodes)
            n = len(mask)
            split = groups[gene][1]
            pools[(split, n)].append(gene)
            pattern_pools[(split, mask)].append(gene)
            whole[n] += 1
            if gene in registered:
                registry_histogram[n] += 1
        domains.append({'layer_scope': scope, 'selected_layer_ids': list(layers),
                        'whole_graph_count_histogram': dict(sorted(whole.items())),
                        'registered_graph_count_histogram': dict(sorted(registry_histogram.items())),
                        'whole_graph_members': len(view.multiplex_nodes),
                        'registered_graph_members': len(registered & view.multiplex_nodes),
                        'unregistered_graph_members': len(view.multiplex_nodes - registered),
                        'registry_outside_multiplex': len(outside)})
        for split in SPLITS:
            for n in sorted(whole):
                pool = sorted(pools[(split, n)], key=lambda gene: groups[gene][0])
                for gene in pool[:counts[split]]:
                    add(gene, scope)
                cell = {'layer_scope': scope, 'split': split, 'present': True, 'layer_count': n,
                        'split_population': len(pool), 'requested': counts[split], 'selected': min(len(pool), counts[split])}
                cells.append(cell)
                if len(pool) < counts[split]:
                    omissions.append({**cell, 'reason': 'insufficient subjects in the fixed split'})
            for (s, mask), population in sorted(pattern_pools.items()):
                if s != split:
                    continue
                pool = sorted(population, key=lambda gene: groups[gene][0])
                for gene in pool[:counts[split]]:
                    add(gene, scope)
                patterns.append({'layer_scope': scope, 'split': split, 'containing_layer_ids': list(mask),
                                 'split_population': len(pool), 'selected': min(len(pool), counts[split])})
            for registered_condition in (False, True):
                pool = sorted((g for g in view.multiplex_nodes if groups[g][1] == split
                               and (g in registered) == registered_condition), key=lambda g: groups[g][0])
                for gene in pool[:counts[split]]:
                    add(gene, scope)
            pool = outside_pools[split]
            for gene in pool[:counts[split]]:
                add(gene, scope)
            cells.append({'layer_scope': scope, 'split': split, 'present': False, 'layer_count': 0,
                          'split_population': len(pool), 'requested': counts[split], 'selected': min(len(pool), counts[split])})
            if len(pool) < counts[split]:
                omissions.append({'layer_scope': scope, 'split': split, 'present': False,
                                  'requested': counts[split], 'selected': len(pool)})
    facts = sorted(selected.values(), key=lambda f: (SPLITS.index(f.split), canonical_json(f.layer_scope), f.category, f.split_unit))
    return facts, {'domains': domains, 'cells': cells, 'membership_patterns': patterns, 'omissions': omissions,
                   'selected_facts': len(facts), 'distinct_query_genes': len({f.gene_id for f in facts}),
                   'fact_selection_sha256': digest([asdict(f) for f in facts]),
                   'split_policy': 'SHA256(seed,query gene) bins 8:1:1 across all scopes; same subjects as S2.3/S2.4',
                   'edge_fact_disjoint': False,
                   'conditioning': 'equal quota per attainable count and global presence; separate pattern and registry-coverage supplements',
                   'scope_policy': 'all nonempty explicit subsets and exhaustive all/family scopes in the pinned view'}


def _arguments(view: GraphView, fact: MembershipFact, mode: str, top_k: int | None) -> dict:
    """Normalize explicit layer IDs before the exact six-field membership call."""
    runtime = MultiplexRuntime(view)
    scope = runtime._scope(fact.layer_scope, runtime.resolve_scope(fact.layer_scope))
    return {'multiplex_id': view.multiplex_id, 'gene_id': fact.gene_id, 'layer_scope': scope,
            'mode': mode, 'top_k': top_k, 'cursor': None}


def _global_observation(view: GraphView, gene: str, evidence_id: str) -> dict:
    """Supply a source-backed global classification without a current-turn call."""
    return {'operation': 'check_multiplex_membership',
            'arguments': {'multiplex_id': view.multiplex_id, 'gene_ids': [gene]}, 'status': 'resolved',
            'result': {'multiplex_id': view.multiplex_id, 'present_gene_ids': [gene] if gene in view.multiplex_nodes else [],
                       'absent_gene_ids': [] if gene in view.multiplex_nodes else [gene]},
            'provenance': {'registry': [], 'graph': [{'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
                'layer_scope': {'kind': 'all', 'layer_ids': [], 'layer_family': None}, 'evidence_id': evidence_id}]}}


def _transport(fact: MembershipFact, variant: str) -> dict:
    """Declare page and count contrasts independently of request wording."""
    result = {'page_size': None, 'disclose_total': True, 'overlap': 0, 'empty_page_at': None}
    if variant in ('paged', 'prefix_context', 'page_terminal', 'page_recovered', 'page_exhausted', 'unknown_total', 'empty_page'):
        result['page_size'] = max(1, (fact.layer_count + 1) // 2)
    if variant == 'overlap':
        if fact.layer_count < 3:
            raise VariantUnavailable('overlap requires three containing layers')
        result.update(page_size=2, overlap=1)
    if variant == 'empty_page':
        if fact.layer_count < 2:
            raise VariantUnavailable('an empty intermediate page requires a multi-page traversal')
        result['empty_page_at'] = 1
    if variant == 'unknown_total':
        result['disclose_total'] = False
    return result


def _plan(view: GraphView, arguments: dict, observations: Sequence[dict]):
    """Keep unique identity evidence beside a source-blind membership proof."""
    from .s2_5_evidence import collect
    proof = collect(view, arguments, observations)
    identities = [o for o in observations if o['operation'] == 'lookup_human_gene_symbol']
    return proof, [*identities, *proof.observations]


def _reason(result: dict, framing: str, used: Sequence[dict]) -> str:
    """Explain visible global presence, scoped counts, and classification completion."""
    collection = result['collection']
    text = (f"The request concerns {result['gene_id']} in scope {result['layer_scope']}. "
            f"The visible global classification is present:{str(result['present']).lower()}. "
            f"Return {result['layer_ids']} in manifest order, with scoped total {result['layer_count']}. "
            f"Coverage is {result['coverage']}; collection.complete is {str(collection['complete']).lower()}. ")
    if not result['present']:
        text += ('Confirmed multiplex absence proves every selected layer negative by containment. '
                 'This is a completed membership classification, so return resolved rather than an unavailable neighbor operation. ')
    elif not result['coverage']['requested_layer_ids']:
        text += 'The valid family selects no registered layers. This empty selection proves no gene absence from an actual layer. '
    else:
        text += 'Scoped layer absence does not imply global absence. Membership comes from explicit inventories and observed membership fields, never from directional degree. '
    if collection['mode'] == 'top_k':
        text += 'The requested first-K selection is complete as a task; any omitted layer remains unknown without exhaustive compatible evidence. '
    if result['layer_count'] is None:
        text += 'The bounded evidence leaves the full count unavailable; keep both total fields null. '
    if len(used) > 1:
        text += 'Retain each original acquisition and deduplicate overlapping positive IDs; calculate an exact union count only from exhaustive compatible data. '
    return text + ('This classification supplies the requested workflow step, so continue.' if framing == 'step'
                   else 'This completes the standalone membership question, so stop.')


def build_row(view: GraphView, fact: MembershipFact, mode: str, top_k: int | None, framing: str,
              variant: str, variant_index: int, bank: Mapping[str, Any], contract: Mapping[str, Any], *,
              index: NeighborIndex | None = None, registry: RegistryContext | None = None, seed: int = 20261028) -> dict:
    """Build membership traces with attributed reuse, exact cursors, and retries."""
    from .s2_5_prompts import render_s2_5_request
    if variant not in VARIANTS or framing not in FRAMINGS:
        raise ValueError('unsupported S2.5 contrast or framing')
    if variant == 'scope_order':
        scope = fact.layer_scope
        if scope['kind'] != 'layers' or len(scope['layer_ids']) < 2:
            raise VariantUnavailable('scope order requires multiple explicit layers')
        fact = replace(fact, layer_scope={**scope, 'layer_ids': [*reversed(scope['layer_ids']), scope['layer_ids'][-1]]})
    transport = _transport(fact, variant)
    runtime = GeneLayersRuntime(view, **transport)
    arguments = _arguments(view, fact, mode, top_k)
    selected = runtime.resolve_scope(fact.layer_scope)
    input_gene = fact.gene_id
    if variant == 'symbol_context':
        if registry is None:
            raise VariantUnavailable('symbol context requires a verified registry')
        try:
            input_gene = get_symbols([fact.gene_id], registry)[0]
        except ValueError as error:
            raise VariantUnavailable(str(error)) from error
    request = render_s2_5_request(contract['_requests'], split=fact.split, mode=mode, variant_index=variant_index,
        multiplex_id=view.multiplex_id, layer_scope=fact.layer_scope, gene=input_gene, top_k=top_k)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=request)
    row_id = 's25_' + digest([contract['contract_version'], view.graph_version, asdict(fact), mode, top_k, framing, variant, user])[:32]
    observations, serial = [], 0

    def acquire(call_arguments: dict) -> dict:
        nonlocal serial
        serial += 1
        return runtime.get_gene_layers(**call_arguments, evidence_id=f'ev_{row_id}_{serial}')

    def save(saved: dict, *, first_only: bool = False) -> None:
        while True:
            response = acquire(saved)
            observations.append(_observation('get_gene_layers', deepcopy(saved), response))
            cursor = response['result']['collection']['next_cursor']
            if first_only or cursor is None:
                return
            saved = {**saved, 'cursor': cursor}

    if variant == 'symbol_context':
        observations.extend(identity_observations([fact.gene_id], row_id, registry))
    elif variant in ('context', 'complete_context', 'bounded_context', 'shorter_prefix_context', 'stale', 'duplicate_context'):
        saved = dict(arguments)
        if variant in ('complete_context', 'stale', 'duplicate_context'):
            saved.update(mode='complete', top_k=None)
        elif variant in ('bounded_context', 'shorter_prefix_context'):
            if variant == 'shorter_prefix_context' and mode != 'top_k':
                raise VariantUnavailable('a shorter prefix requires a bounded request')
            saved.update(mode='top_k', top_k=1 if variant == 'bounded_context' else top_k + 1)
        save(saved)
        if variant == 'stale':
            for observation in observations:
                observation['provenance']['graph'][0]['graph_version'] = '0' * 64
        if variant == 'duplicate_context':
            observations.extend(deepcopy(observations))
    elif variant in ('projected_context', 'broader_bounded_context'):
        if set(selected) == set(view.layer_order):
            raise VariantUnavailable('projection requires a proper subset')
        save({**arguments, 'layer_scope': {'kind': 'all', 'layer_ids': [], 'layer_family': None},
              'mode': 'complete' if variant == 'projected_context' else 'top_k',
              'top_k': None if variant == 'projected_context' else 1})
        if variant == 'broader_bounded_context' and observations[-1]['result']['collection']['complete']:
            raise VariantUnavailable('the broader bounded record must omit memberships')
    elif variant in ('partial_context', 'family_context', 'partial_terminal', 'partial_recovered', 'partial_exhausted', 'overlapping_context'):
        if len(selected) < 2:
            raise VariantUnavailable('partial scope reuse requires multiple selected layers')
        if variant == 'family_context':
            candidates = [{'kind': 'all', 'layer_ids': [], 'layer_family': family} for family in runtime.multiplex.families]
            scope = next((s for s in reversed(candidates) if 0 < len(runtime.resolve_scope(s)) < len(selected)
                          and set(runtime.resolve_scope(s)) <= set(selected)), None)
            if scope is None:
                raise VariantUnavailable('no proper nonempty family within the selected scope')
        else:
            size = 2 if variant == 'overlapping_context' else max(1, len(selected) // 2)
            scope = {'kind': 'layers', 'layer_ids': list(selected[:size]), 'layer_family': None}
        save({**arguments, 'layer_scope': scope, 'mode': 'complete', 'top_k': None})
        if variant == 'overlapping_context':
            if len(selected) < 3:
                raise VariantUnavailable('overlapping scopes require at least three layers')
            save({**arguments, 'layer_scope': {'kind': 'layers', 'layer_ids': list(selected[1:]), 'layer_family': None},
                  'mode': 'complete', 'top_k': None})
    elif variant in ('prefix_context', 'page_terminal', 'page_recovered', 'page_exhausted'):
        save(arguments, first_only=True)
        if observations[-1]['result']['collection']['next_cursor'] is None:
            raise VariantUnavailable('this contrast requires an unfinished first page')
    elif variant in ('neighbor_context', 'neighbor_zero'):
        if not selected:
            raise VariantUnavailable('no selected layer can supply neighbor context')
        native = NeighborRuntime(view, index=index)
        candidates = list(selected)
        if variant == 'neighbor_zero':
            candidates = [layer for layer in selected if view.layers[layer].directed and fact.gene_id in view.layers[layer].nodes
                          and not native.index.neighbors(layer, fact.gene_id, 'out')]
        if not candidates:
            raise VariantUnavailable('no compatible neighbor-membership context')
        layer = candidates[0]
        native_arguments = {'multiplex_id': view.multiplex_id, 'gene_id': fact.gene_id, 'layer_id': layer,
                            'direction': 'out' if view.layers[layer].directed else 'undirected',
                            'mode': 'top_k', 'top_k': 1, 'cursor': None}
        response = native.get_neighbors(**native_arguments, evidence_id=f'ev_{row_id}_neighbor')
        observations.append(_observation('get_neighbors', native_arguments, response))
    elif variant == 'global_context':
        observations.append(_global_observation(view, fact.gene_id, f'ev_{row_id}_global'))
    elif variant == 'wrong_gene':
        pool = sorted((gene for gene in view.multiplex_nodes if gene != fact.gene_id and gene_group(gene, seed)[1] == fact.split),
                      key=lambda gene: gene_group(gene, seed)[0])
        if not pool:
            raise VariantUnavailable('no same-split distractor subject')
        save({**arguments, 'gene_id': pool[0], 'mode': 'complete', 'top_k': None})
    proof, _ = _plan(view, arguments, observations)
    if variant in ('bounded_context', 'broader_bounded_context', 'partial_terminal', 'partial_recovered', 'partial_exhausted') and proof.next_arguments is None:
        raise VariantUnavailable('the supplied record already answers the request')
    context = {'active_graph': _active(view), 'observations': deepcopy(observations)}
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)}, {'role': 'user', 'content': user}]
    failure_kind = variant.removeprefix('page_').removeprefix('partial_')
    injected, failed = False, None
    while True:
        proof, used = _plan(view, arguments, observations)
        if proof.next_arguments is None:
            break
        pending = proof.next_arguments
        thought = (f"The request asks for {mode} memberships of {fact.gene_id}, scope {fact.layer_scope}, K={top_k}. "
                   f"The missing acquisition is get_gene_layers with {pending}. "
                   'Use explicit membership evidence from the active version. A checked selection does not make omitted positive IDs negative. '
                   'Follow the exact cursor and retain all supporting pages.')
        if variant == 'symbol_context':
            thought = f'The supplied registry record uniquely maps {input_gene} to {fact.gene_id}. ' + thought

        def append(response: dict, reasoning: str) -> None:
            messages.extend([{'role': 'assistant', 'thinking': reasoning,
                              'tool_call': {'name': 'get_gene_layers', 'arguments': deepcopy(pending)}},
                             {'role': 'tool', 'name': 'get_gene_layers', 'content': response}])

        if not injected and failure_kind in ('terminal', 'recovered', 'exhausted'):
            injected = True
            response = _failure(None if failure_kind == 'terminal' else 1)
            append(response, thought)
            if failure_kind != 'terminal':
                response = _failure(0) if failure_kind == 'exhausted' else acquire(pending)
                append(response, 'The runtime explicitly supplies retries_remaining=1. Repeat the identical six arguments and cursor. '
                                 'Preserve independent evidence and stop retries after recovery or exhaustion.')
            if response['status'] == 'error':
                failed = (pending, response)
                break
        else:
            response = acquire(pending)
            append(response, thought)
        observations.append(_observation('get_gene_layers', deepcopy(pending), response))
    if failed:
        pending, response = failed
        result = {'failed_tool': 'get_gene_layers', 'error_code': response['error_code'], 'arguments': pending,
                  'partial_results': used, 'coverage': proof.coverage}
        answer = {'status': 'error', 'provenance': _provenance(used), 'result': result, 'continuation_decision': 'defer',
                  'blocked_reason': 'A required membership acquisition is unavailable.',
                  'summary': 'The required acquisition failed; retain the independently supported membership observations.'}
        thought = (f"The required membership acquisition for {fact.gene_id} failed. "
                   + ('The retry allowance is exhausted. ' if failure_kind == 'exhausted' else 'No retry allowance was supplied. ')
                   + f"Keep {len(used)} earlier observations and coverage {proof.coverage}. "
                   'The failed call supplies no evidence. A partial positive list is not an exhaustive negative. Defer for backend recovery.')
    else:
        result = proof.result
        answer = {'status': 'resolved', 'provenance': _provenance(used), 'result': result,
                  'continuation_decision': 'continue' if framing == 'step' else 'stop', 'blocked_reason': None,
                  'summary': 'The requested scoped membership classification is complete.'}
        thought = ('Compatible supplied evidence suffices; no new call is required. ' if len(messages) == 3 else '') + _reason(result, framing, used)
    messages.append({'role': 'assistant', 'thinking': thought, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'], 'metadata': {
        'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
        'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
        'question_family': 'S2.5', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
        'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': input_gene,
        'layer_scope': fact.layer_scope, 'category': fact.category, 'present': fact.present, 'layer_count': fact.layer_count,
        'mode': mode, 'top_k': top_k, 'framing': framing, 'variant': variant, 'transport': transport,
        'wrapper_index': wrapper, 'request_template_index': variant_index % len(contract['_requests'][fact.split][mode]),
        'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def _diagnostic_request(view: GraphView, fact: MembershipFact, kind: str) -> tuple[dict, str, str]:
    """Keep literal rejected scopes and limits separate from valid graph arguments."""
    request = {'gene': fact.gene_id, 'multiplex_id': view.multiplex_id, 'layer_scope': deepcopy(fact.layer_scope),
               'mode': 'complete', 'top_k': None, 'cursor': None}
    codes = {'unknown_layer': 'UNKNOWN_LAYER', 'unknown_family': 'INVALID_LAYER_FAMILY',
             'missing_scope': 'MISSING_SCOPE', 'empty_scope': 'INVALID_SELECTION',
             'missing_limit': 'MISSING_LIMIT', 'invalid_limit': 'INVALID_SELECTION'}
    if kind not in codes:
        raise ValueError('unsupported membership diagnostic')
    if kind == 'unknown_layer':
        request['layer_scope'] = {'kind': 'layers', 'layer_ids': ['HumanNetV3:unknown_membership_layer'], 'layer_family': None}
    elif kind == 'unknown_family':
        request['layer_scope'] = {'kind': 'all', 'layer_ids': [], 'layer_family': 'unknown_membership_family'}
    elif kind == 'missing_scope':
        request['layer_scope'] = None
    elif kind == 'empty_scope':
        request['layer_scope'] = {'kind': 'layers', 'layer_ids': [], 'layer_family': None}
    else:
        request['mode'] = 'top_k'
        request['top_k'] = 0 if kind == 'invalid_limit' else None
    return request, 'not_found' if kind == 'unknown_layer' else 'invalid_input', codes[kind]


def build_diagnostic(view: GraphView, fact: MembershipFact, framing: str, variant_index: int,
                     bank: Mapping[str, Any], contract: Mapping[str, Any], *,
                     kind: str = 'unknown_layer', seed: int = 20261028) -> dict:
    """Reject an unresolved local prerequisite without inventing negative membership."""
    from .s2_5_prompts import render_s2_5_diagnostic_request
    request, status, code = _diagnostic_request(view, fact, kind)
    core = render_s2_5_diagnostic_request(contract['_requests'], split=fact.split, kind=kind, variant_index=variant_index,
        multiplex_id=view.multiplex_id, layer_scope=request['layer_scope'], gene=fact.gene_id, top_k=request['top_k'])
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing, variant_index=variant_index, request=core)
    answer = {'status': status, 'provenance': {'registry': [], 'graph': []},
              'result': {'issue_code': code, 'request': request, 'rejected_arguments': None, 'partial_results': []},
              'continuation_decision': 'defer', 'blocked_reason': 'The requested membership selection needs a valid scope or bound.',
              'summary': 'The request lacks a valid membership prerequisite; no graph classification was performed.'}
    return {'row_id': 's25_' + digest([contract['contract_version'], view.graph_version, user, code])[:32],
            'messages': [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json({'active_graph': _active(view), 'observations': []})},
                {'role': 'user', 'content': user}, {'role': 'assistant', 'thinking':
                    f"The request concerns {fact.gene_id}. Its literal selection is {request}. "
                    f"The visible prerequisite fails with {code}. An unregistered scope or invalid limit is not evidence of gene absence. "
                    'Make no graph call, preserve the rejected request, and defer for the actual missing prerequisite.', 'content': answer}],
            'tools': contract['tools'], 'metadata': {
                'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                'question_family': 'S2.5', 'split': fact.split, 'split_unit': fact.split_unit, 'selection_seed': seed,
                'graph_version': view.graph_version, 'gene_id': fact.gene_id, 'input_gene': fact.gene_id,
                'layer_scope': request['layer_scope'], 'category': 'diagnostic', 'present': None, 'layer_count': None,
                'mode': request['mode'], 'top_k': request['top_k'], 'framing': framing, 'variant': kind,
                'transport': {}, 'wrapper_index': wrapper,
                'request_template_index': variant_index % len(contract['_requests']['diagnostic_requests'][fact.split][kind]),
                'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView, *, bank: Mapping[str, Any], contract: Mapping[str, Any],
                 registry: RegistryContext | None = None, index: NeighborIndex | None = None) -> None:
    """Replay inventories and acquisitions, then grade only the visible proof.

    Private source labels audit sampling. They never supply a hidden count or
    global classification to the final answer. Summary and blocker text remain
    free prose; their wording is not an exact-match target.
    """
    from ..world_model_training import _validate_v3_message_sequence
    from .s2_5_prompts import render_s2_5_request, render_s2_5_diagnostic_request
    messages, meta = row['messages'], row['metadata']
    _validate_v3_message_sequence(messages)
    if sum(m['role'] == 'user' for m in messages) != 1 or messages[1]['role'] != 'harness':
        raise ValueError('S2.5 requires one user turn and explicit harness context')
    for message in messages:
        expected_keys = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                         {'role', 'thinking', 'content'} if message['role'] == 'assistant' else
                         {'role', 'name', 'content'} if message['role'] == 'tool' else {'role', 'content'})
        if set(message) != expected_keys or message['role'] == 'assistant' and not message['thinking'].strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    if messages[0] != {'role': 'system', 'content': contract['system_prompt']} or row['tools'] != contract['tools']:
        raise ValueError('system or tools differ from the membership overlay')
    if (meta['schema_version'] != SCHEMA_VERSION or meta['question_family'] != 'S2.5'
            or meta['tool_contract_version'] != contract['contract_version'] or meta['graph_version'] != view.graph_version
            or meta['system_prompt_sha256'] != hashlib.sha256(contract['system_prompt'].encode()).hexdigest()
            or meta['training_ready'] is not False or meta['supervision'] != 'current_turn_assistant_only'):
        raise ValueError('membership row identity differs from the build contract')
    if gene_group(meta['gene_id'], meta['selection_seed']) != (meta['split_unit'], meta['split']):
        raise ValueError('a query subject crossed its fixed split')
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if not isinstance(answer['summary'], str) or not answer['summary'].strip():
        raise ValueError('a readable summary is required')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    diagnostic = meta['variant'] in DIAGNOSTICS
    if diagnostic:
        request = render_s2_5_diagnostic_request(contract['_requests'], split=meta['split'], kind=meta['variant'],
            variant_index=meta['request_template_index'], multiplex_id=view.multiplex_id,
            layer_scope=meta['layer_scope'], gene=meta['input_gene'], top_k=meta['top_k'])
    else:
        request = render_s2_5_request(contract['_requests'], split=meta['split'], mode=meta['mode'],
            variant_index=meta['request_template_index'], multiplex_id=view.multiplex_id,
            layer_scope=meta['layer_scope'], gene=meta['input_gene'], top_k=meta['top_k'])
    expected_user = bank[meta['split']][meta['framing']][meta['wrapper_index']].format(request=request)
    if messages[2] != {'role': 'user', 'content': expected_user}:
        raise ValueError('the visible question differs from the audited request or framing')
    context = json.loads(messages[1]['content'])
    if set(context) != {'active_graph', 'observations'} or context['active_graph'] != _active(view):
        raise ValueError('active metadata differs from the pinned source view')
    if diagnostic:
        if context['observations'] or len(messages) != 4 or meta['input_gene'] != meta['gene_id']:
            raise ValueError('malformed local prerequisite diagnostic')
        fact = MembershipFact(meta['split'], meta['gene_id'], meta['layer_scope'], 'diagnostic', False, 0, meta['split_unit'])
        expected = build_diagnostic(view, fact, meta['framing'], meta['request_template_index'], bank, contract,
            kind=meta['variant'], seed=meta['selection_seed'])['messages'][-1]['content']
        if any(canonical_json(answer[k]) != canonical_json(expected[k]) for k in ('status', 'provenance', 'result', 'continuation_decision')):
            raise ValueError('the diagnostic changes the rejected request')
        return
    runtime = GeneLayersRuntime(view, **meta['transport'])
    fact = _fact(runtime, meta['gene_id'], meta['layer_scope'], meta['selection_seed'])
    if (fact.category != meta['category'] or fact.present is not meta['present'] or fact.layer_count != meta['layer_count']):
        raise ValueError('private sampling labels differ from authoritative node inventories')
    arguments = _arguments(view, fact, meta['mode'], meta['top_k'])
    available = list(context['observations'])
    identities = [o for o in available if o['operation'] == 'lookup_human_gene_symbol']
    if meta['variant'] == 'symbol_context':
        if registry is None:
            raise ValueError('symbol validation requires the verified registry')
        validate_identity_observations(identities, registry)
        if (len(identities) != 1 or identities[0]['result']['gene_id'] != fact.gene_id
                or get_symbols([fact.gene_id], registry)[0] != meta['input_gene']):
            raise ValueError('the supplied symbol does not uniquely identify the query')
    elif identities or meta['input_gene'] != fact.gene_id:
        raise ValueError('unexpected identity context or changed canonical input')
    evidence_ids = {}
    native = None

    def verify(observation: Mapping[str, Any]) -> None:
        nonlocal native
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
        if tool not in ('get_gene_layers', 'get_neighbors', 'check_multiplex_membership') or len(observation['provenance']['graph']) != 1:
            raise ValueError('unsupported or unattributed graph observation')
        genes = observation['arguments'].get('gene_ids', [observation['arguments'].get('gene_id')])
        if any(gene_group(gene, meta['selection_seed'])[1] != meta['split'] for gene in genes):
            raise ValueError('a saved query subject crossed split boundaries')
        graph = observation['provenance']['graph'][0]
        if graph['graph_version'] != view.graph_version:
            if meta['variant'] != 'stale' or graph['graph_version'] != '0' * 64:
                raise ValueError('undeclared incompatible graph context')
            observation = deepcopy(observation)
            observation['provenance']['graph'][0]['graph_version'] = view.graph_version
        if tool == 'check_multiplex_membership':
            if canonical_json(observation) != canonical_json(_global_observation(view, genes[0], graph['evidence_id'])):
                raise ValueError('global context differs from the authoritative multiplex inventory')
            return
        if tool == 'get_neighbors':
            if native is None:
                native = NeighborRuntime(view, index=index)
            expected = native.get_neighbors(**observation['arguments'], evidence_id=graph['evidence_id'])
        else:
            expected = runtime.get_gene_layers(**observation['arguments'], evidence_id=graph['evidence_id'])
        if any(canonical_json(observation[k]) != canonical_json(expected[k]) for k in ('status', 'result', 'provenance')):
            raise ValueError('an observation differs from its source inventory or declared transport')

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
            raise ValueError('a call changes the required scope, selection, or returned cursor')
        response_message = messages[position + 1]
        if response_message['role'] != 'tool' or response_message['name'] != call['name']:
            raise ValueError('a response does not match its call')
        response = response_message['content']
        if previous_response is not None and previous_response['status'] == 'error':
            if previous_response != _failure(1) or call != previous_call or consecutive_failures != 1:
                raise ValueError('a retry lacks allowance or changes failed arguments')
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
        raise ValueError('final evidence differs from relevant visible acquisitions')
    if answer['status'] == 'error':
        if (proof.next_arguments is None or not previous_response or previous_response['status'] != 'error'
                or previous_response == _failure(1)):
            raise ValueError('an error requires an unresolved terminal failure')
        expected = {'failed_tool': proof.next_tool, 'error_code': previous_response['error_code'],
                    'arguments': previous_call['arguments'], 'partial_results': used, 'coverage': proof.coverage}
        status, decision = 'error', 'defer'
    else:
        if proof.next_arguments is not None or proof.result is None:
            raise ValueError('the answer precedes completion of the requested classification')
        expected = proof.result
        status, decision = 'resolved', 'continue' if meta['framing'] == 'step' else 'stop'
    if (canonical_json(answer['result']) != canonical_json(expected) or answer['status'] != status
            or answer['continuation_decision'] != decision):
        raise ValueError('the answer exceeds visible membership evidence or changes task completion')


def _requests(fact: MembershipFact) -> list[tuple[str, int | None]]:
    """Exercise real membership-count boundaries without an implicit layer cap."""
    n = fact.layer_count
    return [('complete', None), *[('top_k', k) for k in sorted({1, max(1, n // 2), max(1, n), n + 1})]]


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None, smoke: bool = False,
                 seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write immutable classifications, evaluation keys, and full token audits.

    Smoke uses one query per attainable count and fixed split, plus reported
    membership-pattern supplements. An explicit quota visits the same real
    domains. Neither recipe authorizes a production training export.
    """
    from .s2_2_prompts import load_s2_2_requests
    from .s2_3_prompts import load_s2_3_requests
    from .s2_4_prompts import load_s2_4_requests
    from .s2_3_tokens import load_tokenizer, audit_rows
    from .s2_5_prompts import validate_s2_5_rendered_prompts
    if (counts is None) != smoke:
        raise ValueError('choose explicit counts or smoke=True; there is no default production budget')
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
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_5_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']), load_s2_4_requests(contract['s24_request_bank']))
    facts, selection = select_facts(view, tuple(sorted(registry.symbols_by_id)), counts, seed)
    # Only supplied neighbor context needs this index; membership itself uses inventories.
    index = NeighborIndex(view)
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()
    panels = {}
    scope_runtime = GeneLayersRuntime(view)
    for fact in facts:
        key = (fact.split, fact.category, fact.layer_scope['kind'])
        score = (0 if fact.layer_count >= 3 else 1, fact.layer_count,
                 len(scope_runtime.resolve_scope(fact.layer_scope)), canonical_json(fact.layer_scope), fact.split_unit)
        if key not in panels or score < panels[key][0]:
            panels[key] = (score, fact)
    panel_units = {(f.gene_id, canonical_json(f.layer_scope)) for _, f in panels.values()}
    zero_panels = {}
    for fact in facts:
        if not fact.present:
            continue
        for layer in scope_runtime.resolve_scope(fact.layer_scope):
            if (view.layers[layer].directed and fact.gene_id in view.layers[layer].nodes
                    and not index.neighbors(layer, fact.gene_id, 'out')):
                key = fact.split
                score = (len(scope_runtime.resolve_scope(fact.layer_scope)), fact.layer_count, fact.split_unit)
                if key not in zero_panels or score < zero_panels[key][0]:
                    zero_panels[key] = (score, fact)
    zero_units = {(f.gene_id, canonical_json(f.layer_scope)) for _, f in zero_panels.values()}
    for fact in facts:
        for mode, top_k in _requests(fact):
            variants = ['lookup']
            if ((fact.gene_id, canonical_json(fact.layer_scope)) in panel_units
                    and (mode == 'complete' or top_k in (1, 2))):
                variants = list(VARIANTS)
            if (fact.gene_id, canonical_json(fact.layer_scope)) in zero_units and 'neighbor_zero' not in variants:
                variants.append('neighbor_zero')
            for variant in variants:
                for framing in FRAMINGS:
                    ordinal = len(rows[fact.split]) // 2
                    try:
                        row = build_row(view, fact, mode, top_k, framing, variant, ordinal, bank, contract,
                                        index=index, registry=registry, seed=seed)
                    except VariantUnavailable:
                        skipped[f'{variant}/{mode}'] += 1
                        continue
                    validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                    rows[fact.split].append(row)
    for split in SPLITS:
        fact = next((f for f in facts if f.split == split and f.layer_scope['kind'] == 'all'
                     and f.layer_scope['layer_family'] is None), None)
        if fact:
            for kind in DIAGNOSTICS:
                for framing in FRAMINGS:
                    row = build_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, kind=kind, seed=seed)
                    validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                    rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
             'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
             'bound_schedule': 'complete plus positive K at1,floor(count/2),count,count+1; count0 uses1',
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    files = {}
    units = {split: {r['metadata']['split_unit'] for r in rows[split]} for split in SPLITS}
    if any(units[left] & units[right] for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate membership row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, bounds = Counter(), defaultdict(Counter), defaultdict(set)
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta[k]) for k in ('mode', 'category', 'variant'))] += 1
            if meta['mode'] == 'top_k' and meta['category'] != 'diagnostic':
                bounds[(canonical_json(meta['layer_scope']), meta['present'], meta['layer_count'])].add(meta['top_k'])
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(units[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_mode_category_variant': {k: dict(sorted(v.items())) for k, v in sorted(joint.items())},
            'variants': dict(sorted(Counter(r['metadata']['variant'] for r in rows[split]).items())),
            'selected_bounds': [{'layer_scope': json.loads(scope), 'present': present, 'layer_count': n,
                'selected_k': sorted(values), 'relationships': sorted({'below' if k < n else 'equal' if k == n else 'above' for k in values})}
                for (scope, present, n), values in sorted(bounds.items())]}
        if split != 'train':
            files[f'eval/{split}.questions.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                 'metadata': {k: row['metadata'][k] for k in ('schema_version', 'tool_contract_version', 'question_family', 'split')}}
                for row in rows[split]])
            files[f'eval/{split}.answers.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'expected_messages': row['messages'][3:], 'final_answer': row['messages'][-1]['content']}
                for row in rows[split]])
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    modules = ('s2_5_corpus', 's2_gene_layers', 's2_5_evidence', 's2_5_prompts', 's2_4_prompts', 's2_3_corpus',
               's2_3_prompts', 's2_3_tokens', 's2_neighbors', 's2_2_corpus', 's2_2_identity', 's2_2_prompts',
               's2_multiplex', 's2_corpus', 's2_sources', 's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_5.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-5-manifest-v1', 'contract_version': contract['contract_version'],
                'implemented_families': ['S2.5'], 'purpose': 'builder_verification', 'training_ready': False,
                'production_export_allowed': False, 'generation_budget': None if smoke else counts,
                'sampling_schedule': 'one_query_per_attainable_count' if smoke else 'explicit_quota_per_attainable_count',
                'seed': seed, 'row_counts': {s: len(rows[s]) for s in SPLITS}, 'python_version': platform.python_version(),
                'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
                'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
                **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
                                                's23_request_bank', 's24_request_bank', 'registry_index', 'tokenizer', 'context_recipe')},
                'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
                'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
                'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
                'transport_and_faults': 'declared pages, overlap, empty pages, unknown totals, retry schedules and stale version0x00; not deployed captures',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
