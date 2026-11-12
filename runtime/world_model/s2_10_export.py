"""Export exact induced components with frozen wording and actual token checks.

All measured source facts retain their full partitions. A separate small
source panel exercises visible evidence, membership gates, and failures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import BUILDER_MODULES, load_s2_edge_requests, load_s2_prompt_banks
from .s2_2_corpus import VariantUnavailable
from .s2_2_identity import load_registry_context
from .s2_2_prompts import load_s2_2_requests
from .s2_3_prompts import load_s2_3_requests
from .s2_3_tokens import audit_rows, load_tokenizer
from .s2_4_prompts import load_s2_4_requests
from .s2_5_prompts import load_s2_5_requests
from .s2_6_prompts import load_s2_6_requests
from .s2_7_prompts import load_s2_7_requests
from .s2_8_prompts import load_s2_8_requests
from .s2_9_prompts import load_s2_9_requests
from .s2_10_prompts import KINDS, validate_s2_10_rendered_prompts
from .s2_10_sampling import select_facts
from .s2_components import components
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, canonical_json, digest
from .s2_sources import load_source_view
from .s2_subgraphs import SubgraphIndex
from .sources import file_sha256


def _fact_key(fact) -> tuple:
    """Identify an input set and connectivity kind independently of wording."""
    return fact.split, fact.layer_id, tuple(sorted(fact.gene_ids)), fact.connectivity_kind


def _native_cost(view, index, fact) -> int:
    """Measure full native observations before choosing a readable witness."""
    return sum(len(index.native_index.neighbors(fact.layer_id, gene, direction))
               for gene in fact.gene_ids if gene in view.layers[fact.layer_id].nodes
               for direction in index.native_index.directions(fact.layer_id))


def _early_exact_witness(view, index, fact) -> bool:
    """Check if a proper canonical edge prefix already spans one exact group."""
    if fact.component_count != 1 or fact.category_counts[0] < 2 or fact.edge_count < 2:
        return False
    members = index.partition(fact.layer_id, fact.gene_ids)['layer_gene_ids']
    edges = index.edges(fact.layer_id, fact.gene_ids)
    return any(len(components(members, edges[:size], fact.connectivity_kind)) == 1
               for size in range(1, len(edges)))


def _weak_prefix_for_strong(view, index, fact) -> bool:
    """Identify a connected native prefix that leaves strong groups unsettled."""
    if fact.connectivity_kind != 'strong' or fact.category_counts[0] < 2 or fact.edge_count < 2:
        return False
    members = index.partition(fact.layer_id, fact.gene_ids)['layer_gene_ids']
    edges = index.edges(fact.layer_id, fact.gene_ids)
    return any(len(components(members, edges[:size], 'weak')) == 1
               and len(components(members, edges[:size], 'strong')) > 1
               for size in range(1, len(edges)))


def _semantic_panel(view, index, facts) -> set[tuple]:
    """Choose small witnesses by coarse outcomes, membership, kind, and structure.

    Detailed component-count and size strata belong to source allocation. This
    panel only chooses evidence contrasts, without truncating any observation.
    """
    candidates = []
    for fact in facts:
        if len(fact.gene_ids) > 8 or fact.edge_count > 8:
            continue
        outcome = 'zero' if not fact.component_count else 'one' if fact.component_count == 1 else 'multiple'
        keys = [(fact.split, 'groups', fact.connectivity_kind, outcome),
                (fact.split, 'membership', tuple(bool(value) for value in fact.category_counts))]
        for feature, present in (
                ('loop', fact.loop_count), ('reciprocal', fact.reciprocal_pairs),
                ('selected_isolate', fact.isolated_count),
                ('graph_only_query', fact.category == 'graph_only_query'),
                ('more_than_one_nontrivial_group', sum(size > 1 for size in fact.component_sizes) > 1),
                ('directed_chain', fact.category == 'directed_chain'),
                ('removed_bridge', fact.category == 'removed_bridge' and fact.connectivity_kind != 'strong'),
                ('outside_route', fact.category == 'removed_bridge' and fact.connectivity_kind == 'undirected')):
            if present:
                keys.append((fact.split, feature))
        if fact.edge_count >= 2:
            keys.append((fact.split, 'edge_pages', fact.connectivity_kind))
        if _early_exact_witness(view, index, fact):
            keys.append((fact.split, 'early_exact_witness', fact.connectivity_kind))
        if _weak_prefix_for_strong(view, index, fact):
            keys.append((fact.split, 'weak_prefix_for_strong'))
        score = (_native_cost(view, index, fact), len(fact.gene_ids), fact.edge_count,
                 fact.layer_id, fact.split_unit, fact.connectivity_kind)
        candidates.append((score, fact, set(keys)))
    # One witness can cover several features. Cover every attainable key
    # before choosing another fact, then prefer lower native evidence cost.
    uncovered = {key for _, _, keys in candidates for key in keys}
    panel = set()
    while uncovered:
        _, fact, keys = min(candidates, key=lambda item: (-len(item[2] & uncovered), item[0]))
        panel.add(_fact_key(fact))
        uncovered.difference_update(keys)
    return panel


def _gate_panel(facts) -> list:
    """Choose empty, passing, and failing explicit all-members prerequisites."""
    panel = {}
    for fact in facts:
        if len(fact.gene_ids) > 8 or fact.edge_count > 8:
            continue
        outcome = 'empty' if not fact.gene_ids else 'fails' if sum(fact.category_counts[1:]) else 'passes'
        key = fact.split, fact.connectivity_kind, outcome
        score = (len(fact.gene_ids), fact.edge_count, fact.layer_id, fact.split_unit)
        if key not in panel or score < panel[key][0]:
            panel[key] = score, fact
    return sorted((fact for _, fact in panel.values()), key=_fact_key)


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, input_sizes: Sequence[int] | None = None,
                 seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable verification files after complete source and token checks.

    The allocation measures a finite construction domain. It does not enumerate
    every combinatorial input set or select an implicit production budget.
    """
    from .s2_10_corpus import (
        CONTRACT_PATH, DIAGNOSTICS, VARIANTS, build_diagnostic, build_row,
        load_build_contract, validate_row,
    )
    if (counts is None) != smoke:
        raise ValueError('choose smoke or explicit quotas; no production budget is defaulted')
    counts = {split: 1 for split in SPLITS} if smoke else dict(counts)
    if set(counts) != set(SPLITS) or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    context_source = contract['context_source_contract']
    if file_sha256(ROOT / context_source['path']) != context_source['sha256']:
        raise ValueError('the saved graph context contract differs from its pin')
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('the source manifest differs from its pin')
    recipe = contract['context_recipe']
    if file_sha256(ROOT / recipe['path']) != recipe['sha256']:
        raise ValueError('the context reference differs from its pin')
    view = load_source_view(source_manifest)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_10_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']), load_s2_4_requests(contract['s24_request_bank']),
        load_s2_5_requests(contract['s25_request_bank']), load_s2_6_requests(contract['s26_request_bank']),
        load_s2_7_requests(contract['s27_request_bank']), load_s2_8_requests(contract['s28_request_bank']),
        load_s2_9_requests(contract['s29_request_bank']))
    index = SubgraphIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke, input_sizes=input_sizes)
    panel = _semantic_panel(view, index, facts)
    gates = _gate_panel(facts)
    selection.update(semantic_panel_facts=len(panel),
        baseline_question_kinds=list(KINDS),
        semantic_panel_policy='complete source sets with at most 8 inputs and 8 native edges, by split, coarse zero/one/multiple component outcome, connectivity kind, membership categories, and attainable proof/structure features; cover the most remaining feature keys per fact, then prefer lower complete native-neighborhood cost; no source observation truncation or runtime cap',
        semantic_panel_witnesses=[{'fact': asdict(fact), 'complete_native_neighbor_rows': _native_cost(view, index, fact)}
                                  for fact in facts if _fact_key(fact) in panel],
        gate_panel=[asdict(fact) for fact in gates])
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()
    for fact in facts:
        variants = VARIANTS if _fact_key(fact) in panel else ('lookup',)
        for variant in variants:
            question_kinds = KINDS if variant == 'lookup' else ('partition',)
            for question_kind in question_kinds:
                for framing in FRAMINGS:
                    try:
                        row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                            bank, contract, registry=registry, index=index, seed=seed,
                            question_kind=question_kind)
                    except VariantUnavailable:
                        skipped[variant] += 1
                        continue
                    validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                    rows[fact.split].append(row)
    for fact in gates:
        for framing in FRAMINGS:
            row = build_row(view, fact, framing, 'lookup', len(rows[fact.split]) // 2,
                bank, contract, registry=registry, index=index, require_all_in_layer=True, seed=seed)
            validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
            rows[fact.split].append(row)
    for split in SPLITS:
        fact = next((fact for fact in facts if fact.split == split and fact.gene_ids), None)
        if fact is None:
            raise ValueError('each split requires a nonempty source input for diagnostics')
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
        raise ValueError('query subjects crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate component row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, sizes, counts_histogram, group_sizes, kinds, questions, requirements = (
            Counter(), defaultdict(Counter), Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            kind = meta['request'].get('connectivity_kind')
            kinds[kind if isinstance(kind, str) else canonical_json(kind)] += 1
            questions[meta.get('question_kind', 'diagnostic')] += 1
            joint[key]['/'.join(str(value) for value in (kind, meta.get('component_count'),
                meta.get('category_counts'), meta['variant']))] += 1
            if meta.get('component_count') is not None:
                sizes[len(meta['gene_ids'])] += 1
                counts_histogram[meta['component_count']] += 1
                group_sizes[canonical_json(meta.get('component_sizes'))] += 1
            requirements[canonical_json(meta['request'].get('require_all_in_layer'))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_connectivity_components_membership_variant': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
            'input_size_histogram': dict(sorted(sizes.items())),
            'component_count_histogram': dict(sorted(counts_histogram.items())),
            'component_size_multiset_histogram': dict(sorted(group_sizes.items())),
            'connectivity_kind_histogram': dict(sorted(kinds.items())),
            'question_kind_histogram': dict(sorted(questions.items())),
            'membership_requirement_histogram': dict(sorted(requirements.items())),
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
    modules = ('s2_10_export', 's2_10_corpus', 's2_10_sampling', 's2_10_evidence', 's2_10_prompts', 's2_components',
        's2_9_corpus', 's2_9_sampling', 's2_9_evidence', 's2_9_prompts', 's2_9_export', 's2_subgraphs',
        's2_8_prompts', 's2_8_evidence', 's2_degrees',
        's2_7_prompts', 's2_7_evidence', 's2_6_corpus', 's2_6_evidence', 's2_6_prompts',
        's2_5_evidence', 's2_5_prompts', 's2_4_evidence', 's2_4_prompts', 's2_neighbor_layers',
        's2_nodes_by_layer', 's2_gene_layers', 's2_3_corpus', 's2_3_prompts', 's2_3_tokens', 's2_neighbors',
        's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources',
        's1_prompt_banks', 's1_lookup_retry', 's1_membership_retry', 's1_overlay', 's0_corpus',
        'tools', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_10.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-10-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.10'], 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'generation_budget': None if smoke else counts,
        'sampling_schedule': selection['construction_policy'], 'input_sizes': selection['input_sizes'],
        'seed': seed, 'row_counts': {split: len(rows[split]) for split in SPLITS}, 'python_version': platform.python_version(),
        'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
        'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
        **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
            's23_request_bank', 's24_request_bank', 's25_request_bank', 's26_request_bank', 's27_request_bank',
            's28_request_bank', 's29_request_bank', 'context_source_contract',
            'registry_index', 'tokenizer', 'context_recipe')},
        'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
        'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
        'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
        'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
        'transport_and_faults': 'injected edge pages, known/unknown totals, overlaps, retries, failures, stale versions, scope rejection, and explicit membership-first harness policies; no deployed backend captures or hidden input/component cap',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
