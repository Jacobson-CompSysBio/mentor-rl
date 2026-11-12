"""Export singleton membership and claim traces with exact source and token checks.

The finite source allocation measures all available membership categories.
A separate semantic panel supplies context, support, truth, and failure contrasts.
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
from .s2_3_tokens import audit_rows, load_tokenizer
from .s2_4_prompts import load_s2_4_requests
from .s2_5_prompts import load_s2_5_requests
from .s2_6_prompts import load_s2_6_requests
from .s2_7_prompts import load_s2_7_requests
from .s2_8_prompts import load_s2_8_requests
from .s2_9_prompts import load_s2_9_requests
from .s2_10_prompts import load_s2_10_requests
from .s2_11_prompts import load_s2_11_requests
from .s2_12_prompts import validate_s2_12_rendered_prompts
from .s2_12_sources import load_source_view
from .s2_corpus import FRAMINGS, ROOT, SPLITS, _jsonl, _write_immutable, canonical_json, digest
from .s2_multiplex import MultiplexRuntime
from .s2_sources import load_source_view as load_previous_view
from .s2_subgraphs import SubgraphIndex
from .sources import file_sha256


BASELINE_KINDS = ('membership_card',)
REQUEST_PINS = ('prompt_bank', 'request_bank', *(f's2{number}_request_bank' for number in range(1, 12)))


def _fact_key(fact) -> tuple:
    """Retain each singleton and layer as a separate source fact."""
    return fact.split, fact.layer_id, fact.gene_id


def _native_cost(view, index, fact) -> int:
    """Measure complete native evidence before a deterministic witness choice."""
    if fact.gene_id not in view.layers[fact.layer_id].nodes:
        return 0
    direction = 'out' if view.layers[fact.layer_id].directed else 'undirected'
    return len(index.native_index.neighbors(fact.layer_id, fact.gene_id, direction))


def _feature_keys(view, index, fact, registry=None, old_view=None) -> set[tuple]:
    """Describe attainable reviewer features without exposing them as model context."""
    gene, layer = fact.gene_id, view.layers[fact.layer_id]
    family = MultiplexRuntime(view).family_of(fact.layer_id)
    keys = {(fact.split, 'layer_category', fact.layer_id, fact.category),
            (fact.split, 'family_category', family, fact.category)}
    containing = tuple(layer_id for layer_id in view.layer_order if gene in view.layers[layer_id].nodes)
    keys.add((fact.split, 'containing_pattern', containing))
    if containing:
        keys.add((fact.split, 'top_one_positive' if containing[0] == fact.layer_id else 'top_one_omits_target', fact.category))
    if fact.category == 'member':
        keys.add((fact.split, 'native', layer.directed, 'empty' if _native_cost(view, index, fact) == 0 else 'nonempty'))
        if (gene, gene) in layer.edges:
            keys.add((fact.split, 'self_loop_member', layer.directed))
        if registry is not None and gene not in registry.symbols_by_id:
            keys.add((fact.split, 'graph_only_member'))
    state = getattr(fact, 'registry_state', None)
    if state is not None:
        keys.add((fact.split, 'registry_state', state, fact.category))
    if old_view is not None and gene in view.multiplex_nodes and gene not in old_view.multiplex_nodes:
        keys.add((fact.split, 'expanded_view_member', fact.category))
    if registry is not None and gene in registry.symbols_by_id:
        if registry.symbols_by_id[gene] is None:
            keys.add((fact.split, 'null_registry_symbol', fact.category))
    return keys


def _semantic_panel(view, index, facts, registry=None, old_view=None) -> set[tuple]:
    """Cover every attainable category and source feature with full observations."""
    candidates = [((_native_cost(view, index, fact), fact.layer_id, fact.gene_id), fact,
                   _feature_keys(view, index, fact, registry, old_view)) for fact in facts]
    uncovered = {key for _, _, keys in candidates for key in keys}
    panel = set()
    while uncovered:
        _, fact, keys = min(candidates, key=lambda item: (-len(item[2] & uncovered), item[0]))
        panel.add(_fact_key(fact))
        uncovered.difference_update(keys)
    return panel


def _claim_questions(fact, view, vocabulary) -> list[tuple[str, dict]]:
    """Bind observed support and exact truth independently of wrapper-bank size."""
    from .s2_12_corpus import VARIANTS, truth_statement
    rows = [('context', {'question_kind': 'claim_card'})]
    for claim in vocabulary:
        rows.extend((variant, {'question_kind': kind, 'claim': claim})
                    for variant, kind in (('lookup', 'membership_support'), ('context', 'claim_support')))

    def truth(variant, kind, claim, scope, layers=None, family=None):
        task = {'question_kind': kind, 'claim': claim, 'truth_scope': scope,
                'truth_layer_ids': layers, 'truth_family': family,
                'statement': truth_statement(view, fact, claim, scope,
                    truth_layer_ids=layers, truth_family=family)}
        rows.append((variant, task))

    claim = 'gene_not_node_in_multiplex' if fact.category == 'outside_multiplex' else 'gene_not_node_in_layer'
    scope = 'multiplex' if fact.category == 'outside_multiplex' else 'category'
    for variant, kind in (('lookup', 'membership_truth'), ('context', 'truth')):
        truth(variant, kind, claim, scope)
    if fact.category != 'outside_multiplex':
        truth('context', 'truth', 'absence_holds_in_other_layers', 'all_layers')
        family = MultiplexRuntime(view).family_of(fact.layer_id)
        truth('context', 'truth', 'absence_holds_in_other_layers', 'family', family=family)
        others = [layer for layer in view.layer_order if layer != fact.layer_id]
        if others:
            truth('context', 'truth', 'absence_holds_in_other_layers', 'layers', layers=others[:2])
    truth('context', 'truth', 'absence_holds_in_other_versions', 'future_versions')
    truth('context', 'truth', 'gene_not_expressed_in_context', 'expression')
    for gate in ('multiplex', 'layer'):
        rows.append(('context', {'question_kind': 'membership_support', 'claim': claim,
                                 'require_membership': gate}))
    supplied = [variant for variant in VARIANTS if variant.endswith('_context')
                and variant not in ('symbol_context', 'duplicate_context')]
    for variant in supplied:
        rows.append((variant, {'question_kind': 'claim_card'}))
        for selected in ('gene_not_node_in_layer', 'gene_not_expressed_in_context'):
            rows.append((variant, {'question_kind': 'claim_support', 'claim': selected}))
    for variant in ('wrong_layer_context', 'wrong_family_context', 'prefix_context',
                    'prefix_positive_context', 'prefix_omission_context', 'layer_list_context'):
        if variant in VARIANTS and fact.category != 'outside_multiplex':
            truth(variant, 'truth', 'absence_holds_in_other_layers', 'all_layers')
    if fact.category != 'outside_multiplex' and 'family_context' in VARIANTS:
        truth('family_context', 'truth', 'absence_holds_in_other_layers', 'family',
              family=MultiplexRuntime(view).family_of(fact.layer_id))
    unique = {}
    for variant, task in rows:
        unique.setdefault(canonical_json([variant, task]), (variant, task))
    return list(unique.values())


def _implementation_paths() -> list[str]:
    """Pin the full imported build closure, including unchanged earlier helpers."""
    modules = ('s2_12_export', 's2_12_corpus', 's2_12_sampling', 's2_12_evidence', 's2_12_prompts', 's2_12_sources',
        's2_11_export', 's2_11_corpus', 's2_11_sampling', 's2_11_evidence', 's2_11_prompts', 's2_paths', 's2_graph_claims',
        's2_10_prompts', 's2_10_evidence', 's2_10_corpus', 's2_10_sampling', 's2_10_export', 's2_components',
        's2_9_evidence', 's2_9_prompts', 's2_9_corpus', 's2_9_sampling', 's2_9_export', 's2_subgraphs',
        's2_8_prompts', 's2_8_evidence', 's2_degrees', 's2_7_prompts', 's2_7_evidence',
        's2_6_evidence', 's2_6_prompts', 's2_6_corpus', 's2_5_evidence', 's2_5_prompts', 's2_4_evidence', 's2_4_prompts',
        's2_neighbor_layers', 's2_nodes_by_layer', 's2_gene_layers', 's2_3_corpus', 's2_3_prompts', 's2_3_tokens',
        's2_neighbors', 's2_2_corpus', 's2_2_identity', 's2_2_prompts', 's2_multiplex', 's2_corpus', 's2_sources',
        's1_prompt_banks', 's1_lookup_retry', 's1_membership_retry', 's1_overlay', 's0_corpus',
        'tools', 'sources', 'source_index', 'contract', *BUILDER_MODULES)
    return [*(f'runtime/world_model/{name}.py' for name in dict.fromkeys(modules)),
            'runtime/world_model_training.py', 'scripts/build_world_model_s2_12.py']


def build_corpus(output_dir: Path, *, counts: Mapping[str, int] | None = None,
                 smoke: bool = False, seed: int = 20261028, contract_path: Path | None = None,
                 source_manifest: Path | None = None) -> dict[str, Any]:
    """Write nine immutable files after full source, wording, token, and mask checks."""
    from .s2_12_corpus import (
        CONTRACT_PATH, DIAGNOSTICS, VARIANTS, build_diagnostic, build_row, load_build_contract, validate_row,
    )
    from .s2_12_sampling import select_facts
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
               load_s2_4_requests, load_s2_5_requests, load_s2_6_requests,
               load_s2_7_requests, load_s2_8_requests, load_s2_9_requests,
               load_s2_10_requests, load_s2_11_requests)
    earlier = [loader(contract[f's2{number}_request_bank']) for number, loader in enumerate(loaders, 1)]
    wording = validate_s2_12_rendered_prompts(bank, contract['_requests'], *earlier)
    index = SubgraphIndex(view)
    facts, selection = select_facts(view, index, registry, counts, seed, smoke=smoke, old_view=old_view)
    panel = _semantic_panel(view, index, facts, registry, old_view)
    selection.update(semantic_panel_facts=len(panel), baseline_question_kinds=list(BASELINE_KINDS),
        semantic_panel_policy='cover every attainable singleton layer/category, containing-layer pattern, native membership, registry, and expanded-view source feature by split; prefer lower complete native evidence cost; never truncate observations',
        semantic_panel_witnesses=[{'fact': asdict(fact), 'complete_native_neighbor_rows': _native_cost(view, index, fact)}
                                  for fact in facts if _fact_key(fact) in panel])
    tokenizer = load_tokenizer(contract['tokenizer'])
    rows, skipped = {split: [] for split in SPLITS}, Counter()

    def append(fact, variant, **task):
        for framing in FRAMINGS:
            try:
                row = build_row(view, fact, framing, variant, len(rows[fact.split]) // 2,
                    bank, contract, registry=registry, index=index, old_view=old_view, seed=seed, **task)
            except VariantUnavailable:
                skipped[variant] += 1
                continue
            validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index, old_view=old_view)
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
            raise ValueError('each split requires a source singleton for diagnostics')
        for kind in DIAGNOSTICS:
            for framing in FRAMINGS:
                row = build_diagnostic(view, fact, framing, len(rows[split]) // 2,
                    bank, contract, kind=kind, registry=registry, seed=seed)
                validate_row(row, view, bank=bank, contract=contract, registry=registry, index=index, old_view=old_view)
                rows[split].append(row)
    tokens = audit_rows((row for split in SPLITS for row in rows[split]), tokenizer, recipe['max_length'])
    audit = {'source_semantics_passed': True, 'purpose': 'builder_verification', 'training_ready': False,
        'production_export_allowed': False, 'selection': selection, 'full_question_wording': wording,
        'tokenizer': tokens, 'inapplicable_contrasts': dict(sorted(skipped.items())), 'splits': {}}
    subjects = {split: {row['metadata']['gene_id'] for row in rows[split]} for split in SPLITS}
    if any(subjects[left] & subjects[right] for rank, left in enumerate(SPLITS) for right in SPLITS[rank + 1:]):
        raise ValueError('query singletons crossed split boundaries')
    files = {}
    for split in SPLITS:
        if len({row['row_id'] for row in rows[split]}) != len(rows[split]):
            raise ValueError('duplicate singleton row IDs')
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage, joint = Counter(), defaultdict(Counter)
        histograms = {key: Counter() for key in ('category', 'question_kind', 'claim', 'truth_scope', 'require_membership', 'variant')}
        for row in rows[split]:
            meta = row['metadata']
            key = f"{meta['framing']}:{meta['wrapper_index']}"
            usage[key] += 1
            for field, histogram in histograms.items():
                histogram[canonical_json(meta.get(field))] += 1
            joint[key]['/'.join(canonical_json(meta.get(field)) for field in
                ('layer_id', 'category', 'variant', 'question_kind', 'claim', 'truth_scope'))] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_genes': len(subjects[split]),
            'wrapper_usage': dict(sorted(usage.items())),
            'wrapper_by_layer_category_variant_question_claim_scope': {key: dict(sorted(values.items())) for key, values in sorted(joint.items())},
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
    manifest = {'schema_version': 'mentor-rl-world-model-s2-12-manifest-v1', 'contract_version': contract['contract_version'],
        'implemented_families': ['S2.12'], 'purpose': 'builder_verification', 'training_ready': False,
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
        'split_policy': selection['split_policy'], 'query_gene_disjoint': True, 'edge_fact_disjoint': False,
        'transport_and_faults': 'injected required membership failures with preserved independent evidence and exact authorized retries; no deployed backend captures',
        'remaining_coverage': contract['remaining_coverage'],
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({name: value['sha256'] for name, value in manifest['files'].items() if name.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
