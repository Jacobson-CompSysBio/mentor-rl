"""Export exact paths and scoped claims with source and token checks.

All selected ordered pairs retain their full path results. A smaller source
panel adds evidence contrasts, claim questions, explicit truth, and failures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

from .s1_prompt_banks import BUILDER_MODULES, load_s2_edge_requests, load_s2_prompt_banks
from .s2_2_corpus import VariantUnavailable
from .s2_2_identity import load_registry_context
from .s2_2_prompts import load_s2_2_requests
from .s2_3_prompts import load_s2_3_requests
from .s2_3_corpus import gene_group
from .s2_3_tokens import audit_rows, load_tokenizer
from .s2_4_prompts import load_s2_4_requests
from .s2_5_prompts import load_s2_5_requests
from .s2_6_prompts import load_s2_6_requests
from .s2_7_prompts import load_s2_7_requests
from .s2_8_prompts import load_s2_8_requests
from .s2_9_prompts import load_s2_9_requests
from .s2_10_prompts import load_s2_10_requests
from .s2_11_prompts import validate_s2_11_rendered_prompts
from .s2_11_sampling import select_facts
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, canonical_json, digest
from .s2_paths import PathIndex
from .s2_sources import load_source_view
from .sources import file_sha256


BASELINE_KINDS = ('path', 'path_claims')
REQUEST_PINS = ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
                's23_request_bank', 's24_request_bank', 's25_request_bank', 's26_request_bank',
                's27_request_bank', 's28_request_bank', 's29_request_bank', 's210_request_bank')


def _fact_key(fact) -> tuple:
    """Keep every reverse query distinct while related facts share a split unit."""
    return fact.split, fact.layer_id, tuple(fact.gene_ids), fact.direction


def _zero_weight_availability(view, seed: int) -> dict:
    """Report actual zero-weight sources without creating an unattainable path bin."""
    result = {}
    for layer_id, layer in view.layers.items():
        counts, eligible = Counter(), Counter()
        for (source, target), weight in layer.edges.items():
            if weight != 0:
                continue
            counts['native_zero_weight_edges'] += 1
            if source == target:
                counts['zero_weight_self_loops'] += 1
                continue
            counts['distinct_endpoint_zero_edges'] += 1
            split = gene_group(source, seed)[1]
            if gene_group(target, seed)[1] == split:
                eligible[split] += 1
        result[layer_id] = {**{key: counts[key] for key in ('native_zero_weight_edges',
            'zero_weight_self_loops', 'distinct_endpoint_zero_edges')},
            'eligible_one_hop_edges_by_split': {split: eligible[split] for split in SPLITS}}
    return result


def _native_cost(view, index, fact) -> int:
    """Measure full native evidence without truncation before choosing a witness."""
    layer = view.layers[fact.layer_id]
    genes = {fact.gene_ids[0]} & layer.nodes
    if not layer.directed and fact.path_found is False and not any(fact.category_counts[1:]):
        genes = set(index.distances(fact.layer_id, fact.gene_ids[0]))
    direction = 'out' if layer.directed else 'undirected'
    return sum(len(index.native_index.neighbors(fact.layer_id, gene, direction)) for gene in genes)


def _semantic_panel(view, index, facts, registry=None) -> set[tuple]:
    """Cover attainable coarse outcomes and proof features with complete source facts."""
    candidates = []
    by_pair = {_fact_key(fact): fact for fact in facts}
    for fact in facts:
        layer, source, target = view.layers[fact.layer_id], *fact.gene_ids
        outcome = ('missing' if fact.path_found is None else 'no_path' if not fact.path_found else
                   'zero' if fact.hop_count == 0 else 'one' if fact.hop_count == 1 else
                   'two' if fact.hop_count == 2 else 'longer')
        category = lambda gene: 'member' if gene in layer.nodes else 'outside_layer' if gene in view.multiplex_nodes else 'missing'
        keys = {(fact.split, 'path', fact.direction, outcome),
                (fact.split, 'membership', category(source), category(target))}
        native = index.native_index.neighbors(fact.layer_id, source, 'out' if layer.directed else 'undirected') if source in layer.nodes else ()
        if (fact.tied_shortest_paths or 0) > 1:
            keys.add((fact.split, 'tied_shortest', fact.direction))
            if fact.hop_count == 2:
                keys.add((fact.split, 'two_hop_tie', fact.direction))
        reverse = by_pair.get((fact.split, fact.layer_id, tuple(reversed(fact.gene_ids)), fact.direction))
        if reverse is not None:
            if layer.directed and reverse.path_found != fact.path_found:
                keys.add((fact.split, 'reverse_reachability_difference'))
            if not layer.directed and fact.path_found and reverse.path_gene_ids != tuple(reversed(fact.path_gene_ids)):
                keys.add((fact.split, 'reverse_tie_witness_difference'))
        if source == target and source in layer.nodes:
            keys.add((fact.split, 'same_node_loop' if any(gene == source for gene, _ in native) else 'same_node_no_loop', fact.direction))
        if source == target and fact.category_counts[1]:
            keys.add((fact.split, 'same_node_outside_layer'))
        if source == target and fact.category_counts[2]:
            keys.add((fact.split, 'same_node_missing'))
        if fact.category_counts[2] > 1:
            keys.add((fact.split, 'multiple_missing'))
        if fact.hop_count == 1 and any(gene == target and weight == 0 for gene, weight in native):
            keys.add((fact.split, 'zero_weight_path', fact.direction))
        if fact.path_found is False and not any(fact.category_counts[1:]):
            if layer.directed and not native:
                keys.add((fact.split, 'empty_outgoing'))
            elif not layer.directed:
                keys.add((fact.split, 'closed_component'))
        if len(native) > 1:
            keys.add((fact.split, 'partial_native', fact.direction))
        if registry is not None and any(gene in view.multiplex_nodes and gene not in registry.symbols_by_id for gene in fact.gene_ids):
            keys.add((fact.split, 'graph_only_query'))
        score = (_native_cost(view, index, fact), len(fact.path_gene_ids or ()),
                 fact.layer_id, fact.gene_ids, fact.direction)
        candidates.append((score, fact, keys))
    uncovered = {key for _, _, keys in candidates for key in keys}
    panel = set()
    while uncovered:
        _, fact, keys = min(candidates, key=lambda item: (-len(item[2] & uncovered), item[0]))
        panel.add(_fact_key(fact))
        uncovered.difference_update(keys)
    return panel


def _claim_questions(fact, view, vocabulary) -> list[tuple[str, dict]]:
    """Select explicit claim and truth tasks independently of wrapper count."""
    from .s2_11_corpus import truth_statement
    rows = [('context', {'question_kind': 'claim_card', 'membership_cards': True})]
    for claim in vocabulary:
        rows.extend((variant, {'question_kind': kind, 'claim': claim, 'membership_cards': True})
                    for variant, kind in (('lookup', 'path_support'), ('context', 'claim_support')))
    if fact.path_found is not None:
        claims = ['no_path_recorded_in_scope']
        if fact.gene_ids[0] != fact.gene_ids[1]:
            claims.append('no_edge_recorded_in_scope')
        for claim in claims:
            for variant, kind in (('lookup', 'path_truth'), ('context', 'truth')):
                rows.append((variant, {'question_kind': kind, 'claim': claim, 'truth_scope': 'scope',
                    'statement': truth_statement(view, fact, claim, 'scope'), 'membership_cards': True}))
        rows.append(('context', {'question_kind': 'truth', 'claim': 'absence_holds_in_other_layers',
            'truth_scope': 'all_layers', 'statement': truth_statement(view, fact,
                'absence_holds_in_other_layers', 'all_layers'), 'membership_cards': True}))
    supplied = ['membership_context', 'subgraph_context', 'native_context']
    distinct = fact.gene_ids[0] != fact.gene_ids[1]
    if distinct and fact.path_found is not None:
        supplied.append('edge_context')
    if (fact.hop_count or 0) >= 2:
        supplied.append('route_context')
    for variant in supplied:
        rows.append((variant, {'question_kind': 'claim_card', 'membership_cards': True}))
        for claim in ('no_edge_recorded_in_scope', 'no_path_recorded_in_scope'):
            rows.append((variant, {'question_kind': 'claim_support', 'claim': claim, 'membership_cards': True}))
        if variant in ('edge_context', 'route_context'):
            rows.append((variant, {'question_kind': 'claim_support', 'claim': 'no_interaction_in_biology',
                                   'membership_cards': True}))
    if fact.hop_count == 0:
        rows.append(('membership_context', {'question_kind': 'path_support',
            'claim': 'no_edge_recorded_in_scope', 'membership_cards': True}))
    if fact.hop_count == 2 and (fact.tied_shortest_paths or 0) > 1:
        rows.append(('tie_route_context', {'question_kind': 'claim_support',
            'claim': 'no_edge_recorded_in_scope', 'membership_cards': True}))
    if fact.direction == 'undirected' and distinct:
        rows.append(('reversed_context', {'question_kind': 'claim_card', 'membership_cards': True}))
        for claim in ('no_edge_recorded_in_scope', 'no_path_recorded_in_scope'):
            rows.append(('reversed_context', {'question_kind': 'claim_support', 'claim': claim, 'membership_cards': True}))
            if fact.path_found is not None:
                rows.append(('reversed_context', {'question_kind': 'truth', 'claim': claim, 'truth_scope': 'scope',
                    'statement': truth_statement(view, fact, claim, 'scope'), 'membership_cards': True}))
    if fact.path_found is not None and any(layer != fact.layer_id
            and view.layers[layer].directed == view.layers[fact.layer_id].directed for layer in view.layer_order):
        rows.append(('wrong_layer', {'question_kind': 'truth', 'claim': 'absence_holds_in_other_layers',
            'truth_scope': 'all_layers', 'statement': truth_statement(view, fact,
                'absence_holds_in_other_layers', 'all_layers'), 'membership_cards': True}))
    return rows


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, source_window: int | None = None,
                 seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable files after full source, wording, token, and mask checks."""
    from .s2_11_corpus import (
        CONTRACT_PATH, DIAGNOSTICS, ROUTE_CONSTRUCTION_BUDGET, VARIANTS, build_diagnostic, build_row,
        load_build_contract, validate_row,
    )
    if (counts is None) != smoke:
        raise ValueError('choose smoke or explicit quotas; no production budget is defaulted')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    if set(counts) != set(SPLITS) or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError('counts require positive integers for all splits')
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
    loaders = (load_s2_edge_requests, load_s2_2_requests, load_s2_3_requests,
               load_s2_4_requests, load_s2_5_requests, load_s2_6_requests,
               load_s2_7_requests, load_s2_8_requests, load_s2_9_requests, load_s2_10_requests)
    earlier = [loader(contract[f's2{number}_request_bank']) for number, loader in enumerate(loaders, 1)]
    wording = validate_s2_11_rendered_prompts(bank, contract['_requests'], *earlier)
    index = PathIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke, source_window=source_window)
    panel = _semantic_panel(view, index, facts, registry)
    selection.update(semantic_panel_facts=len(panel), baseline_question_kinds=list(BASELINE_KINDS),
        alternate_route_construction_budget=dict(ROUTE_CONSTRUCTION_BUDGET),
        zero_weight_source_availability=_zero_weight_availability(view, seed),
        semantic_panel_policy='cover every attainable coarse path, endpoint membership, native proof, tie, and source feature by split; cover most remaining keys first, then prefer lower complete native evidence cost; alternate-route budgets limit source construction only; no observation truncation or runtime hop cap',
        semantic_panel_witnesses=[{'fact': asdict(fact), 'complete_native_neighbor_rows': _native_cost(view, index, fact)}
                                  for fact in facts if _fact_key(fact) in panel])
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()

    def append(fact, variant, **task):
        for framing in FRAMINGS:
            try:
                row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                    bank, contract, registry=registry, index=index, seed=seed, **task)
            except VariantUnavailable:
                skipped[variant] += 1
                continue
            validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
            rows[fact.split].append(row)

    for fact in facts:
        for question_kind in BASELINE_KINDS:
            append(fact, 'lookup', question_kind=question_kind)
        if _fact_key(fact) in panel:
            for variant in VARIANTS:
                if variant != 'lookup':
                    append(fact, variant)
            for variant, task in _claim_questions(fact, view, contract['claim_vocabulary']):
                append(fact, variant, **task)
    for split in SPLITS:
        fact = next((fact for fact in facts if fact.split == split), None)
        if fact is None:
            raise ValueError('each split requires a source pair for diagnostics')
        for kind in DIAGNOSTICS:
            for framing in FRAMINGS:
                row = build_diagnostic(view, fact, framing, len(rows[split]) // 2,
                    bank, contract, kind=kind, registry=registry, seed=seed)
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
        'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    subjects = {split: {gene for row in rows[split] for gene in row['metadata']['gene_ids']} for split in SPLITS}
    if any(subjects[left] & subjects[right] for rank, left in enumerate(SPLITS) for right in SPLITS[rank + 1:]):
        raise ValueError('query endpoints crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate path row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, hops, outcomes, memberships, directions, questions, claims, truth, membership_cards = (
            Counter(), defaultdict(Counter), Counter(), Counter(), Counter(), Counter(),
            Counter(), Counter(), Counter(), Counter())
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            direction = meta['request'].get('direction')
            directions[direction if isinstance(direction, str) else canonical_json(direction)] += 1
            questions[meta.get('question_kind', 'diagnostic')] += 1
            hops[canonical_json(meta.get('hop_count'))] += 1
            outcomes[canonical_json(meta.get('path_found'))] += 1
            memberships[canonical_json(meta.get('category_counts'))] += 1
            claims[canonical_json(meta.get('claim'))] += 1
            truth[canonical_json(meta.get('truth_scope'))] += 1
            membership_cards[canonical_json(meta.get('membership_cards', False))] += 1
            joint[key]['/'.join(canonical_json(value) for value in (direction, meta.get('hop_count'),
                meta.get('category_counts'), meta['variant'], meta.get('question_kind')))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_direction_hops_membership_variant_question': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
            'hop_count_histogram': dict(sorted(hops.items())), 'path_found_histogram': dict(sorted(outcomes.items())),
            'membership_histogram': dict(sorted(memberships.items())), 'direction_histogram': dict(sorted(directions.items())),
            'question_kind_histogram': dict(sorted(questions.items())), 'claim_histogram': dict(sorted(claims.items())),
            'truth_scope_histogram': dict(sorted(truth.items())), 'membership_cards_histogram': dict(sorted(membership_cards.items())),
            'variants': dict(sorted(Counter(row['metadata']['variant'] for row in rows[split]).items()))}
        if split != 'train':
            files[f'eval/{split}.questions.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                 'metadata': {key: row['metadata'][key] for key in ('schema_version', 'tool_contract_version', 'question_family', 'split')}}
                for row in rows[split]])
            files[f'eval/{split}.answers.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'expected_messages': row['messages'][3:], 'final_answer': row['messages'][-1]['content']}
                for row in rows[split]])
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    modules = ('s2_11_export', 's2_11_corpus', 's2_11_sampling', 's2_11_evidence', 's2_11_prompts', 's2_paths', 's2_graph_claims',
        's2_10_prompts', 's2_10_evidence', 's2_10_corpus', 's2_10_sampling', 's2_10_export', 's2_components',
        's2_9_evidence', 's2_9_prompts', 's2_9_corpus', 's2_9_sampling', 's2_9_export', 's2_subgraphs',
        's2_8_prompts', 's2_8_evidence', 's2_degrees', 's2_7_prompts', 's2_7_evidence',
        's2_6_evidence', 's2_6_prompts', 's2_6_corpus', 's2_5_evidence', 's2_5_prompts', 's2_4_evidence', 's2_4_prompts',
        's2_neighbor_layers', 's2_nodes_by_layer', 's2_gene_layers', 's2_3_corpus', 's2_3_prompts', 's2_3_tokens',
        's2_neighbors', 's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources',
        's1_prompt_banks', 's1_lookup_retry', 's1_membership_retry', 's1_overlay', 's0_corpus',
        'tools', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_11.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-11-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.11'], 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'generation_budget': None if smoke else counts,
        'sampling_schedule': selection['construction_policy'], 'source_window': selection['source_window'],
        'seed': seed, 'row_counts': {split: len(rows[split]) for split in SPLITS}, 'python_version': platform.python_version(),
        'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
        'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
        **{key: contract[key] for key in (*REQUEST_PINS, 'registry_index', 'tokenizer', 'context_recipe')},
        'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
        'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
        'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
        'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
        'transport_and_faults': 'injected required path failures with retained edge and registry observations, exact authorized retries, execution limits, stale versions, direction and scope contrasts; no deployed backend captures or runtime hop limit',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
