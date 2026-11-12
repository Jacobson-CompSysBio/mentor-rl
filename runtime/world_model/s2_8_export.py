"""Write isolated degree verification files with exact source and token audits.

All selected queries retain their complete population statistics. A small
source panel adds visible-evidence and failure contrasts without a degree cap.
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
from .s2_3_corpus import gene_group
from .s2_3_prompts import load_s2_3_requests
from .s2_3_tokens import audit_rows, load_tokenizer
from .s2_4_prompts import load_s2_4_requests
from .s2_5_prompts import load_s2_5_requests
from .s2_6_prompts import load_s2_6_requests
from .s2_7_prompts import load_s2_7_requests
from .s2_8_prompts import KINDS, validate_s2_8_rendered_prompts
from .s2_8_sampling import fact_from_arguments, select_facts
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, canonical_json, digest
from .s2_degrees import DegreeIndex, DegreeRuntime
from .s2_sources import load_source_view
from .sources import file_sha256


def _fact_key(fact) -> tuple:
    """Identify the complete source request independently of wording and policy."""
    return fact.gene_id, canonical_json(fact.layer_scope), fact.direction


def _semantic_panel(view, index, facts) -> set[tuple]:
    """Choose small real examples for directions, scope forms, and proof features."""
    panel = {}
    runtime = DegreeRuntime(view, index=index)
    for fact in facts:
        selected = runtime.resolve_scope(fact.layer_scope, fact.direction)
        count = fact.degree
        small = count is None or count <= 5
        in_scope = fact.membership['present_in_scope'] is True
        keys = []
        if small:
            outcome = ('missing' if not fact.membership['present_in_multiplex'] else
                       'outside' if not in_scope else 'zero' if count == 0 else 'positive')
            keys.append((fact.split, 'outcome', outcome))
            scope_kind = ('empty_family' if not selected else 'family' if fact.layer_scope['layer_family'] else
                          'all' if fact.layer_scope['kind'] == 'all' else
                          'single' if len(selected) == 1 else 'multiple')
            keys.append((fact.split, 'scope', scope_kind))
            if in_scope:
                keys.append((fact.split, 'direction', fact.direction))
            if count is not None and 3 <= count <= 5 and in_scope:
                keys.append((fact.split, 'page_proof'))
        if count is not None and 0 < count <= 64 and in_scope:
            native_sets = [set(neighbor for neighbor, _ in index.native_index.neighbors(layer, fact.gene_id, direction))
                           for layer in selected for direction in (
                               index.native_index.directions(layer) if fact.direction == 'incident' else (fact.direction,))]
            if any(fact.gene_id in neighbors for neighbors in native_sets):
                keys.append((fact.split, 'self_loop'))
            if sum(map(len, native_sets)) > count:
                keys.append((fact.split, 'overlap'))
        score = (count is None, count or 0, len(selected), canonical_json(fact.layer_scope), fact.split_unit)
        for key in keys:
            if key not in panel or score < panel[key][0]:
                panel[key] = score, fact
    return {_fact_key(fact) for _, fact in panel.values()}


def _policy_panel(view, index, facts, seed):
    """Select attainable equality witnesses from full source populations.

    Each split gets at most one witness per threshold. Empty or unavailable
    equalities remain explicit. The threshold-zero representative needs no
    equality because its purpose is an explicit low policy threshold.
    """
    runtime = DegreeRuntime(view, index=index)
    domains, groups, candidates = [], {}, {}
    existing = {_fact_key(fact): fact for fact in facts}
    for fact in facts:
        selected = runtime.resolve_scope(fact.layer_scope, fact.direction)
        if not selected:
            continue
        direction = ('undirected' if fact.direction == 'incident'
                     and all(not view.layers[layer].directed for layer in selected) else fact.direction)
        groups.setdefault((selected, direction), None)
    for selected, direction in sorted(groups):
        histogram = index.histogram(selected, direction)
        population = index.population(selected)
        size, lower, equalities = len(population), 0, {50: None, 100: None}
        for row in histogram:
            equal = row['node_count']
            if size == 1 or 2 * lower + equal == size:
                equalities[50] = row['degree']
            if size > 1 and lower == size - 1 and equal == 1:
                equalities[100] = row['degree']
            lower += equal
        available = {split: Counter() for split in SPLITS}
        for gene in population:
            unit, split = gene_group(gene, seed)
            degree = index.degree(selected, gene, direction)
            for threshold in (0, 50, 100):
                if threshold and degree != equalities[threshold]:
                    continue
                available[split][threshold] += 1
                key = split, threshold
                score = (len(selected), degree, selected, direction, unit)
                if key not in candidates or score < candidates[key][0]:
                    candidates[key] = score, gene, selected, direction
        domains.append({'resolved_layer_ids': list(selected), 'direction': direction,
            'population_count': size, 'exact_equality_degree_values': equalities,
            'available_policy_witnesses_by_split': {
                split: {str(threshold): available[split][threshold] for threshold in (0, 50, 100)}
                for split in SPLITS}})
    chosen, additions, omissions = [], [], []
    variants = {0: 'threshold_zero', 50: 'threshold_fifty', 100: 'threshold_hundred'}
    for split in SPLITS:
        for threshold in (0, 50, 100):
            candidate = candidates.get((split, threshold))
            if candidate is None:
                omissions.append({'split': split, 'threshold_percentile': threshold,
                    'reason': 'no eligible source member supplies the required exact equality' if threshold else
                              'no nonempty source reference population exists'})
                continue
            _, gene, selected, direction = candidate
            scope = {'kind': 'layers', 'layer_ids': list(selected), 'layer_family': None}
            key = gene, canonical_json(scope), direction
            fact = existing.get(key)
            if fact is None:
                fact = fact_from_arguments(view, index, gene, scope, direction, seed,
                    category='policy_equality' if threshold else 'policy_threshold')
                existing[key] = fact
                additions.append(fact)
            chosen.append((fact, variants[threshold]))
    return chosen, additions, {'method': 'one full-source exact equality witness per split at50/100 when attainable; one valid member at threshold0',
        'domains': domains, 'omissions': omissions,
        'selected': [{'fact': asdict(fact), 'variant': variant} for fact, variant in chosen]}


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028,
                 contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable files after source, wording, and full token checks.

    Smoke chooses recorded degree representatives from every declared domain.
    Explicit quotas cover each attainable degree. Neither filters rank cohorts.
    Production budgets and trainer support remain separate work.
    """
    from .s2_8_corpus import (
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
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('the source manifest differs from its pin')
    recipe = contract['context_recipe']
    if file_sha256(ROOT / recipe['path']) != recipe['sha256']:
        raise ValueError('the context reference differs from its pin')
    view = load_source_view(source_manifest)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    wording = validate_s2_8_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']), load_s2_4_requests(contract['s24_request_bank']),
        load_s2_5_requests(contract['s25_request_bank']), load_s2_6_requests(contract['s26_request_bank']),
        load_s2_7_requests(contract['s27_request_bank']))
    index = DegreeIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke)
    policy_panel, policy_additions, policy_audit = _policy_panel(view, index, facts, seed)
    selection['degree_domain_selected_facts'] = len(facts)
    facts = sorted([*facts, *policy_additions], key=lambda fact: (
        SPLITS.index(fact.split), canonical_json(fact.layer_scope), fact.direction, fact.category, fact.split_unit))
    selection.update(policy_panel=policy_audit, policy_source_additions=[asdict(fact) for fact in policy_additions],
        selected_facts=len(facts), fact_selection_sha256=digest([asdict(fact) for fact in facts]))
    panel = _semantic_panel(view, index, facts)
    selection['semantic_panel_facts'] = len(panel)
    selection['semantic_panel_policy'] = ('small source examples by query split, direction, membership outcome, '
        'scope form, page proof, self-loop, and overlap; every selected fact has both requested statistics')
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()
    constructed = set()
    for fact in facts:
        variants = VARIANTS if _fact_key(fact) in panel else ('lookup',)
        for variant in variants:
            for request_kind in KINDS:
                for framing in FRAMINGS:
                    try:
                        row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                            bank, contract, registry=registry, index=index, request_kind=request_kind, seed=seed)
                    except VariantUnavailable:
                        skipped[f'{variant}/{request_kind}'] += 1
                        continue
                    validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                    rows[fact.split].append(row)
                    constructed.add((_fact_key(fact), variant, request_kind, framing))
    for fact, variant in policy_panel:
        for framing in FRAMINGS:
            if (_fact_key(fact), variant, 'degree_and_hub', framing) in constructed:
                continue
            row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                bank, contract, registry=registry, index=index, request_kind='degree_and_hub', seed=seed)
            validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
            rows[fact.split].append(row)
    for split in SPLITS:
        fact = next((fact for fact in facts if fact.split == split and fact.rank_support is not None), None)
        if fact is None:
            raise ValueError('each split requires a population member for diagnostics')
        for kind in DIAGNOSTICS:
            for framing in FRAMINGS:
                row = build_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, kind=kind, seed=seed)
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
             'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    subjects = {split: {row['metadata']['gene_id'] for row in rows[split]} for split in SPLITS}
    if any(subjects[left] & subjects[right] for rank, left in enumerate(SPLITS) for right in SPLITS[rank + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate degree row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, degrees, policies, requested = Counter(), defaultdict(Counter), Counter(), Counter(), Counter()
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta.get(field)) for field in ('request_kind', 'direction', 'degree', 'variant'))] += 1
            if meta.get('degree') is not None:
                degrees[meta['degree']] += 1
            requested[meta['request_kind']] += 1
            policies[canonical_json(meta.get('hub_policy'))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_statistic_direction_degree_variant': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
            'degree_histogram': dict(sorted(degrees.items())), 'request_kind_histogram': dict(sorted(requested.items())),
            'hub_policy_histogram': dict(sorted(policies.items())),
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
    modules = ('s2_8_export', 's2_8_corpus', 's2_8_sampling', 's2_8_evidence', 's2_8_prompts', 's2_degrees',
        's2_7_prompts', 's2_6_corpus', 's2_6_evidence', 's2_6_prompts', 's2_5_evidence', 's2_5_prompts',
        's2_4_evidence', 's2_4_prompts', 's2_neighbor_layers', 's2_nodes_by_layer', 's2_gene_layers',
        's2_3_corpus', 's2_3_prompts', 's2_3_tokens', 's2_neighbors', 's2_2_corpus', 's2_2_identity',
        's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources', 's1_prompt_banks', 'sources',
        'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_8.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-8-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.8'], 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'generation_budget': None if smoke else counts,
        'sampling_schedule': 'recorded_degree_representatives' if smoke else 'explicit_quotas_per_attainable_degree_cell',
        'seed': seed, 'row_counts': {split: len(rows[split]) for split in SPLITS}, 'python_version': platform.python_version(),
        'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
        'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
        **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
            's23_request_bank', 's24_request_bank', 's25_request_bank', 's26_request_bank', 's27_request_bank',
            'registry_index', 'tokenizer', 'context_recipe', 'degree_policy_id', 'population_policy_id',
            'default_hub_policy', 'percentile_display')},
        'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
        'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
        'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
        'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
        'transport_and_faults': 'injected neighbor pages, known/unknown totals, stale versions, retry budgets, errors, and scope rejection; no deployed runtime captures or degree cap',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
