"""Write isolated S2.7 verification files with exact source and token audits.

All selected facts retain their full neighbor sets. A small semantic panel
adds evidence and fault contrasts without a runtime list-size limit.
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
from .s2_7_prompts import DIRECTION_PAIRS, validate_s2_7_rendered_prompts
from .s2_7_sampling import fact_from_queries, neighbor_ids, select_facts
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, digest
from .s2_neighbors import NeighborIndex
from .s2_sources import load_source_view
from .sources import file_sha256


def _direction_examples(view, index, facts, registry, seed):
    """Add small real examples for directed role pairs absent from the sampler."""
    additions = []
    present = {(fact.split, fact.layer_id, tuple(direction for _, direction in fact.queries)) for fact in facts}
    for layer_id, layer in view.layers.items():
        if not layer.directed:
            continue
        for split in SPLITS:
            pool = [gene for gene in sorted(layer.nodes & registry.symbols_by_id.keys())
                    if gene_group(gene, seed)[1] == split]
            if not pool:
                continue
            gene = min(pool, key=lambda gene: (
                not all(neighbor_ids(index, layer_id, gene, direction) for direction in ('in', 'out')),
                len(neighbor_ids(index, layer_id, gene, 'both')), gene_group(gene, seed)[0]))
            for directions in DIRECTION_PAIRS.values():
                if directions == ('undirected', 'undirected') or (split, layer_id, directions) in present:
                    continue
                fact = fact_from_queries(view, index, layer_id,
                    ((gene, directions[0]), (gene, directions[1])), seed, category='direction_pair')
                additions.append(fact)
                present.add((split, layer_id, directions))
    return additions


def _semantic_panel(facts):
    """Use small representatives for outcomes, roles, and special proof domains."""
    panel = {}
    for fact in facts:
        if max((count or 0 for count in fact.neighbor_counts), default=0) > 5:
            continue
        directions = tuple(direction for _, direction in fact.queries)
        keys = [('outcome', fact.split, fact.layer_id, fact.shared_count is None,
                 fact.shared_count is not None and fact.shared_count > 0)]
        if directions != ('undirected', 'undirected'):
            keys.append(('directions', fact.split, directions))
        if fact.category not in ('native_pair', 'repeated_role', 'both_union_pair', 'direction_pair'):
            keys.append(('feature', fact.split, fact.category))
        score = (max((count or 0 for count in fact.neighbor_counts), default=0),
                 sum(count or 0 for count in fact.neighbor_counts), fact.split_unit)
        for key in keys:
            if key not in panel or score < panel[key][0]:
                panel[key] = (score, fact)
    return {fact.split_unit for _, fact in panel.values()}


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028,
                 contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable files with untruncated lists and pinned token checks.

    The readable degree conditioning defines a verification population only.
    Full-source extrema retain complete source lists in ordinary lookup rows.
    Production quotas, context representation, and trainer support remain open.
    """
    from .s2_7_corpus import (
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
    wording = validate_s2_7_rendered_prompts(bank, contract['_requests'],
        load_s2_edge_requests(contract['s21_request_bank']), load_s2_2_requests(contract['s22_request_bank']),
        load_s2_3_requests(contract['s23_request_bank']), load_s2_4_requests(contract['s24_request_bank']),
        load_s2_5_requests(contract['s25_request_bank']), load_s2_6_requests(contract['s26_request_bank']))
    index = NeighborIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke)
    additions = _direction_examples(view, index, facts, registry, seed)
    facts = sorted([*facts, *additions], key=lambda fact: (
        SPLITS.index(fact.split), view.layer_order.index(fact.layer_id), fact.category, fact.split_unit, fact.queries))
    selection.update(direction_pair_additions=[asdict(fact) for fact in additions], selected_facts=len(facts),
                     fact_selection_sha256=digest([asdict(fact) for fact in facts]),
                     distinct_query_genes=len({gene for fact in facts for gene, _ in fact.queries}))
    panel = _semantic_panel(facts)
    selection['semantic_panel_facts'] = len(panel)
    selection['semantic_panel_policy'] = 'small source representatives by split, layer outcome, directed pair, and special feature; full-degree extrema use lookup only'
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()
    for fact in facts:
        variants = VARIANTS if fact.split_unit in panel else ('lookup',)
        for variant in variants:
            for framing in FRAMINGS:
                try:
                    row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                                    bank, contract, registry=registry, index=index, seed=seed)
                except VariantUnavailable:
                    skipped[variant] += 1
                    continue
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                rows[fact.split].append(row)
    for split in SPLITS:
        fact = next((fact for fact in facts if fact.split == split and fact.shared_count is not None), None)
        if fact is None:
            raise ValueError('each split requires a valid source pair for diagnostics')
        for kind in DIAGNOSTICS:
            for framing in FRAMINGS:
                row = build_diagnostic(view, fact, framing, len(rows[split]) // 2, bank, contract, kind=kind, seed=seed)
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
             'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
             'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    subjects = {split: {query['gene_id'] for row in rows[split] for query in row['metadata'].get('queries', [])}
                for split in SPLITS}
    if any(subjects[left] & subjects[right] for rank, left in enumerate(SPLITS) for right in SPLITS[rank + 1:]):
        raise ValueError('query subjects crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate shared-neighbor row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint, counts_histogram, direction_histogram = Counter(), defaultdict(Counter), Counter(), Counter()
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            joint[key]['/'.join(str(meta.get(field)) for field in ('directions', 'shared_count', 'variant'))] += 1
            if meta.get('shared_count') is not None:
                counts_histogram[meta['shared_count']] += 1
            directions = meta.get('directions')
            direction_histogram[json.dumps(directions, sort_keys=True)] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_directions_shared_count_variant': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
            'shared_count_histogram': dict(sorted(counts_histogram.items())),
            'direction_pair_histogram': dict(sorted(direction_histogram.items())),
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
    modules = ('s2_7_export', 's2_7_corpus', 's2_7_sampling', 's2_7_evidence', 's2_7_prompts',
               's2_6_corpus', 's2_nodes_by_layer', 's2_6_evidence', 's2_6_prompts', 's2_5_evidence', 's2_5_prompts',
               's2_gene_layers', 's2_4_prompts', 's2_3_corpus', 's2_3_prompts', 's2_3_tokens', 's2_neighbors',
               's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources',
               's1_prompt_banks', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    implementation = [f'runtime/world_model/{name}.py' for name in modules]
    implementation.extend(('runtime/world_model_training.py', 'scripts/build_world_model_s2_7.py'))
    manifest = {'schema_version': 'mentor-rl-world-model-s2-7-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.7'], 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'generation_budget': None if smoke else counts,
        'sampling_schedule': 'representative_smoke' if smoke else 'explicit_pair_quotas_per_declared_stratum',
        'seed': seed, 'row_counts': {split: len(rows[split]) for split in SPLITS}, 'python_version': platform.python_version(),
        'graph_version': view.graph_version, 'source_pins': dict(view.source_pins),
        'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
        **{key: contract[key] for key in ('prompt_bank', 'request_bank', 's21_request_bank', 's22_request_bank',
            's23_request_bank', 's24_request_bank', 's25_request_bank', 's26_request_bank', 'registry_index', 'tokenizer', 'context_recipe')},
        'implementation_sha256': {path: file_sha256(ROOT / path) for path in implementation},
        'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
        'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
        'split_policy': selection['split_policy'], 'edge_fact_disjoint': False,
        'transport_and_faults': 'injected complete-mode page boundaries, overlap, unknown totals, retries, errors, and incompatible metadata; no deployed runtime captures or list cap',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
