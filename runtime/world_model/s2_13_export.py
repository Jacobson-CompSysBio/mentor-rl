"""Export family meanings and native pair claims with source and token checks.

A compact semantic panel exercises observed support and scope contrasts. Every
selected source pair also retains its ordinary acquisition in both framings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping

from .s1_prompt_banks import load_s2_edge_requests, load_s2_prompt_banks
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
from .s2_10_prompts import load_s2_10_requests
from .s2_11_prompts import load_s2_11_requests
from .s2_12_prompts import load_s2_12_requests
from .s2_12_sources import load_source_view
from .s2_12_export import _implementation_paths as _previous_implementation_paths
from .s2_13_prompts import validate_s2_13_rendered_prompts
from .s2_13_evidence import CLAIM_VOCABULARY, FAMILY_CARDS, FAMILIES
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, canonical_json, digest
from .s2_multiplex import MultiplexRuntime
from .s2_sources import load_source_view as load_previous_view
from .s2_subgraphs import SubgraphIndex
from .sources import file_sha256


BASELINE_KINDS = ('edge_card',)
REQUEST_PINS = ('prompt_bank', 'request_bank', *(f's2{number}_request_bank' for number in range(1, 13)))


def _fact_key(fact) -> tuple:
    """Keep native query order and registered layers distinct within a pair split."""
    return fact.split, fact.layer_id, tuple(fact.gene_ids)


def _feature_keys(view, index, fact, registry=None, old_view=None) -> set[tuple]:
    """Describe attainable source controls without exposing their labels to the model."""
    family = fact.layer_family
    keys = {(fact.split, 'layer_category', fact.layer_id, fact.category),
            (fact.split, 'family_category', family, fact.category),
            (fact.split, 'native_orientation', fact.directed, fact.category, bool(fact.reverse_exists))}
    for state in fact.registry_states:
        keys.add((fact.split, 'registry_state', state, fact.category))
    pen = tuple(layer for layer in fact.positive_layers
                if MultiplexRuntime(view).family_of(layer) in ('bulk_pen', 'scpen'))
    if family in ('bulk_pen', 'scpen'):
        keys.add((fact.split, 'pen_recorded_contexts', pen, fact.category))
    if old_view is not None and any(gene in view.multiplex_nodes and gene not in old_view.multiplex_nodes
                                    for gene in fact.gene_ids):
        keys.add((fact.split, 'expanded_view_endpoint', fact.category))
    if family == 'ppi' and fact.category == 'no_edge':
        left = {gene for gene, _ in index.native_index.neighbors(fact.layer_id, fact.gene_ids[0], 'undirected')}
        right = {gene for gene, _ in index.native_index.neighbors(fact.layer_id, fact.gene_ids[1], 'undirected')}
        if (left & right) - set(fact.gene_ids):
            keys.add((fact.split, 'two_edge_route_without_direct_record'))
    return keys


def _semantic_panel(view, index, facts, registry=None, old_view=None) -> set[tuple]:
    """Cover source categories and contrasts with complete fixed-size edge records."""
    candidates = [((fact.layer_id, -(fact.weight or 0), tuple(fact.gene_ids)), fact,
                   _feature_keys(view, index, fact, registry, old_view)) for fact in facts]
    uncovered = {key for _, _, keys in candidates for key in keys}
    panel = set()
    while uncovered:
        _, fact, keys = min(candidates, key=lambda item: (-len(item[2] & uncovered), item[0]))
        panel.add(_fact_key(fact))
        uncovered.difference_update(keys)
    return panel


def _claim_questions(fact, view, vocabulary=CLAIM_VOCABULARY) -> list[tuple[str, dict]]:
    """Keep all concept checks independent of wording and preserve evidence-only limits."""
    from .s2_13_corpus import VARIANTS
    rows = [('lookup', {'question_kind': 'claim_card'}), ('context', {'question_kind': 'claim_card'})]
    for claim in vocabulary:
        rows.extend((variant, {'question_kind': kind, 'claim': claim})
                    for variant, kind in (('lookup', 'edge_support'), ('context', 'claim_support')))
    local = FAMILY_CARDS[fact.layer_family]['evidence_type']
    contrasts = ('reverse_context', 'family_context', 'wrong_layer_context', 'same_family_context',
                 'wrong_pair_context', 'stale', 'old_view_context', 'pen_contrast_context',
                 'two_context_context', 'reciprocal_context', 'route_context')
    for variant in contrasts:
        if variant in VARIANTS:
            rows.append((variant, {'question_kind': 'claim_card'}))
            for claim in dict.fromkeys((local, 'causal_effect')):
                if claim is not None:
                    rows.append((variant, {'question_kind': 'claim_support', 'claim': claim}))
    for claim in ('causal_effect', 'holds_in_other_contexts'):
        rows.append(('context', {'question_kind': 'biological_truth', 'claim': claim}))
    if fact.category != 'outside_multiplex':
        rows.append(('context', {'question_kind': 'graph_truth', 'claim': local,
                                'graph_claim': 'no_edge_recorded_in_scope', 'graph_truth_scope': 'scope'}))
    unique = {}
    for variant, task in rows:
        unique.setdefault(canonical_json([variant, task]), (variant, task))
    return list(unique.values())


def _implementation_paths() -> list[str]:
    """Pin every local import, including unchanged earlier source and token helpers."""
    new = [f'runtime/world_model/s2_13_{name}.py'
           for name in ('export', 'corpus', 'sampling', 'evidence', 'prompts')]
    return list(dict.fromkeys([*new, *_previous_implementation_paths(), 'scripts/build_world_model_s2_13.py']))


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke=False, seed=20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable files after complete source, wording, token, and mask checks."""
    if (counts is None) != smoke:
        raise ValueError('choose smoke or explicit pair quotas; no production budget is defaulted')
    counts = dict.fromkeys(SPLITS, 1) if smoke else dict(counts)
    if set(counts) != set(SPLITS) or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    from .s2_13_corpus import (
        CONTRACT_PATH, DIAGNOSTICS, VARIANTS, build_diagnostic, build_family_row,
        build_row, load_build_contract, validate_row,
    )
    from .s2_13_sampling import select_facts
    contract_path = Path(contract_path or CONTRACT_PATH)
    contract = load_build_contract(contract_path)
    source_manifest = Path(source_manifest or ROOT / contract['source_manifest']['path'])
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('the source manifest differs from its pin')
    previous = ROOT / contract['previous_source_manifest']['path']
    if file_sha256(previous) != contract['previous_source_manifest']['sha256']:
        raise ValueError('the previous source manifest differs from its pin')
    recipe = contract['context_recipe']
    if file_sha256(ROOT / recipe['path']) != recipe['sha256']:
        raise ValueError('the context reference differs from its pin')
    view, old_view = load_source_view(source_manifest), load_previous_view(previous)
    registry = load_registry_context(contract['registry_index'])
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    loaders = (load_s2_edge_requests, load_s2_2_requests, load_s2_3_requests,
        load_s2_4_requests, load_s2_5_requests, load_s2_6_requests, load_s2_7_requests,
        load_s2_8_requests, load_s2_9_requests, load_s2_10_requests, load_s2_11_requests, load_s2_12_requests)
    earlier = [loader(contract[f's2{number}_request_bank']) for number, loader in enumerate(loaders, 1)]
    wording = validate_s2_13_rendered_prompts(bank, contract['_requests'], *earlier)
    index = SubgraphIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke, old_view=old_view)
    panel = _semantic_panel(view, index, facts, registry, old_view)
    selection.update(semantic_panel_facts=len(panel), baseline_question_kinds=list(BASELINE_KINDS),
        semantic_panel_policy='cover attainable layer/category, native orientation, registry, PEN context, expanded-view, and route controls; preserve complete native observations',
        semantic_panel_witnesses=[asdict(fact) for fact in facts if _fact_key(fact) in panel],
        family_capability_concepts=list(FAMILIES), other_family_source_qualified=False)
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()

    def append(fact, variant, **task):
        for framing in FRAMINGS:
            try:
                row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                    bank, contract, index=index, registry=registry, old_view=old_view, seed=seed, **task)
            except VariantUnavailable:
                skipped[variant] += 1
                continue
            validate_row(row, view, bank=bank, contract=contract, index=index, registry=registry, old_view=old_view)
            rows[fact.split].append(row)

    for fact in facts:
        append(fact, 'lookup', question_kind='edge_card')
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
        for family in FAMILIES:
            for kind, claim in [('family_card', None), *(('family_support', claim) for claim in CLAIM_VOCABULARY)]:
                for framing in FRAMINGS:
                    row = build_family_row(view, family, split, framing, len(rows[split]) // 2,
                        bank, contract, question_kind=kind, claim=claim, seed=seed)
                    validate_row(row, view, bank=bank, contract=contract, index=index, registry=registry, old_view=old_view)
                    rows[split].append(row)
        for kind in DIAGNOSTICS:
            for framing in FRAMINGS:
                row = build_diagnostic(view, fact, framing, len(rows[split]) // 2,
                    bank, contract, kind=kind, registry=registry, seed=seed)
                validate_row(row, view, bank=bank, contract=contract, index=index, registry=registry, old_view=old_view)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
        'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    pairs = {split: {tuple(sorted(row['metadata']['gene_ids'])) for row in rows[split]
                     if len(row['metadata']['gene_ids']) == 2} for split in SPLITS}
    if any(pairs[left] & pairs[right] for rank, left in enumerate(SPLITS) for right in SPLITS[rank + 1:]):
        raise ValueError('unordered target pairs crossed split boundaries')
    files = {}
    fields = ('category', 'layer_family', 'question_kind', 'claim', 'graph_claim', 'variant')
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate specificity row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint = Counter(), defaultdict(Counter)
        histograms = {key: Counter() for key in fields}
        for row in rows[split]:
            meta = row['metadata']
            wrapper = f'{meta["framing"]}:{meta["wrapper_index"]}'
            usage[wrapper] += 1
            for field, histogram in histograms.items():
                histogram[canonical_json(meta.get(field))] += 1
            joint[wrapper]['/'.join(canonical_json(meta.get(field)) for field in
                ('layer_id', 'category', 'variant', 'question_kind', 'claim'))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_pairs': len(pairs[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_layer_category_variant_question_claim': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
            **{f'{key}_histogram': dict(sorted(histogram.items())) for key, histogram in histograms.items()}}
        if split != 'train':
            files[f'eval/{split}.questions.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'messages': row['messages'][:3], 'tools': row['tools'],
                 'metadata': {key: row['metadata'][key] for key in ('schema_version', 'tool_contract_version', 'question_family', 'split')}}
                for row in rows[split]])
            files[f'eval/{split}.answers.jsonl'] = _jsonl([
                {'row_id': row['row_id'], 'expected_messages': row['messages'][3:], 'final_answer': row['messages'][-1]['content']}
                for row in rows[split]])
    files['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    manifest = {'schema_version': 'mentor-rl-world-model-s2-13-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.13'], 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'generation_budget': None if smoke else counts,
        'sampling_schedule': selection['construction_policy'], 'seed': seed,
        'row_counts': {split: len(rows[split]) for split in SPLITS}, 'python_version': platform.python_version(),
        'graph_version': view.graph_version, 'previous_graph_version': old_view.graph_version,
        'source_pins': dict(view.source_pins), 'previous_source_pins': dict(old_view.source_pins),
        'source_manifest_sha256': file_sha256(source_manifest), 'contract_sha256': file_sha256(contract_path),
        **{key: contract[key] for key in (*REQUEST_PINS, 'previous_source_manifest', 'registry_index', 'tokenizer', 'context_recipe')},
        'implementation_sha256': {path: file_sha256(ROOT / path) for path in _implementation_paths()},
        'shared_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
        'fact_selection_sha256': selection['fact_selection_sha256'], 'tokenizer_render_date': tokens['render_date'],
        'split_policy': selection['split_policy'], 'query_pair_disjoint': True, 'query_gene_disjoint': False,
        'edge_fact_disjoint': False, 'family_capability_graph_scope_required': False,
        'transport_and_faults': 'injected required edge failures with original successful partials and exact visible retries; no deployed backend captures',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
