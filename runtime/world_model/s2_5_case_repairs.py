"""Build family-context failure contrasts missing from the frozen S2.5 builder.

The original builder and artifacts stay reproducible. These additions acquire
real complete family observations before injecting a remaining-scope failure.
The existing validator replays sources and grades the source-blind proof.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

from . import s2_5_corpus as base
from .s2_2_corpus import VariantUnavailable
from .s2_audit_cases_3_6 import matches
from .s2_corpus import SPLITS, _failure, digest
from .s2_gene_layers import GeneLayersRuntime


CASE_IDS = ('S2.5.14', 'S2.5.15', 'S2.5.16')
REPAIR = 's2_5_family_failure_repairs_v1'
load_build_contract = base.load_build_contract
CONTRACT_PATH = base.CONTRACT_PATH
_VARIANTS = dict(zip(CASE_IDS, ('partial_terminal', 'partial_recovered', 'partial_exhausted')))
ALL = {'kind': 'all', 'layer_ids': [], 'layer_family': None}


def _visible(view):
    """Expose representation metadata without inventories or native edges."""
    return SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})


def build_row(view, fact, case_id, bank, contract, registry=None, *, index=None, seed=20261028,
              variant_index=0):
    """Acquire a true family record and fail or retry only its uncovered layers."""
    index = getattr(index, 'native_index', index)
    if case_id not in CASE_IDS or fact.layer_scope != ALL:
        raise ValueError('a repaired all-layer membership case is required')
    row = base.build_row(view, fact, 'complete', None, 'step', 'family_context', variant_index,
                         bank, contract, registry=registry, index=index, seed=seed)
    observations = json.loads(row['messages'][1]['content'])['observations']
    if (not observations or observations[0]['arguments']['layer_scope']['layer_family'] != 'tf_target'
            or not any(o['result'].get('layer_count', 0) > 0 for o in observations)):
        raise VariantUnavailable('the complete family context must contain positive membership')
    arguments = base._arguments(view, fact, 'complete', None)
    proof, used = base._plan(_visible(view), arguments, observations)
    if proof.next_arguments is None:
        raise VariantUnavailable('the family record must leave required layers unchecked')
    call = {'name': proof.next_tool, 'arguments': deepcopy(proof.next_arguments)}
    first = row['messages'][3]
    if first.get('tool_call') != call:
        raise ValueError('the source build disagrees with the visible remaining-scope proof')
    variant = _VARIANTS[case_id]
    failure = _failure(None if variant == 'partial_terminal' else 1)
    injected = [deepcopy(first), {'role': 'tool', 'name': call['name'], 'content': failure}]
    if variant == 'partial_recovered':
        row['messages'][3:3] = injected
    else:
        if variant == 'partial_exhausted':
            injected.extend([{'role': 'assistant', 'thinking':
                'The runtime supplied retries_remaining=1. Repeat the identical remaining-layer acquisition once; '
                'retain the original family observation and stop after exhaustion.', 'tool_call': deepcopy(call)},
                {'role': 'tool', 'name': call['name'], 'content': _failure(0)}])
        answer = {'status': 'error', 'provenance': base._provenance(used), 'result': {
            'failed_tool': call['name'], 'error_code': failure['error_code'], 'arguments': call['arguments'],
            'partial_results': used, 'coverage': proof.coverage}, 'continuation_decision': 'defer',
            'blocked_reason': 'A required remaining-layer membership acquisition is unavailable.',
            'summary': 'Retain the checked family memberships while the remaining layers await recovery.'}
        row['messages'][3:] = [*injected, {'role': 'assistant', 'thinking':
            f"The complete family record remains evidence for {fact.gene_id}. Coverage is {proof.coverage}. "
            'The failed acquisition adds no membership evidence. Preserve the original family scope and evidence; '
            'unchecked layers remain unknown. Defer for backend recovery.', 'content': answer}]
    row['metadata'].update(variant=variant, catalog_case_id=case_id, repair=REPAIR)
    row['row_id'] = 's25repair_' + digest([REPAIR, case_id, row['messages'][2]['content'], fact.split_unit])[:32]
    return row


def build_case_rows(view, index, bank, contract, registry=None, *, split, seed=20261028):
    """Select one small real subject per missing case within its original split."""
    index = getattr(index, 'native_index', index)
    if split not in SPLITS:
        raise ValueError('a fixed source split is required')
    runtime = GeneLayersRuntime(view)
    family_scope = {'kind': 'all', 'layer_ids': [], 'layer_family': 'tf_target'}
    family_layers = runtime.resolve_scope(family_scope)
    if not 0 < len(family_layers) < len(view.layer_order):
        raise VariantUnavailable('the reviewed TF family must be a proper nonempty scope')
    candidates = [gene for gene in view.multiplex_nodes if base.gene_group(gene, seed)[1] == split
                  and any(gene in view.layers[layer].nodes for layer in family_layers)]
    if not candidates:
        raise VariantUnavailable(f'no positive family membership in fixed {split} split')
    gene = min(candidates, key=lambda g: (sum(g in view.layers[layer].nodes for layer in view.layer_order),
                                         base.gene_group(g, seed)[0], g))
    fact = base._fact(runtime, gene, ALL, seed)
    rows = [build_row(view, fact, case_id, bank, contract, registry, index=index, seed=seed,
                      variant_index=number) for number, case_id in enumerate(CASE_IDS)]
    for row in rows:
        validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    return rows, {'split': split, 'candidate_population': len(candidates), 'selected_gene_id': gene,
        'family_scope': family_scope, 'family_layer_ids': list(family_layers),
        'case_ids': list(CASE_IDS), 'split_policy': 'unchanged gene_group(seed,gene)',
        'source_selection': 'complete native layer endpoint inventories; no registry filtering'}


def validate_row(row, view, *, index=None, bank, contract, registry=None):
    """Replay native inventories, then enforce exact family, fault, and case evidence."""
    index = getattr(index, 'native_index', index)
    meta = row['metadata']
    case_id = meta.get('catalog_case_id')
    if (set(row) != {'row_id', 'messages', 'tools', 'metadata'}
            or case_id not in CASE_IDS or meta.get('repair') != REPAIR or meta.get('variant') != _VARIANTS[case_id]
            or meta.get('layer_scope') != ALL or meta.get('mode') != 'complete' or meta.get('top_k') is not None
            or meta.get('framing') != 'step' or meta.get('transport') != base._transport(None, 'family_context')):
        raise ValueError('the repair identity or approved all-layer request changed')
    expected_id = 's25repair_' + digest([REPAIR, case_id, row['messages'][2]['content'], meta['split_unit']])[:32]
    if row['row_id'] != expected_id:
        raise ValueError('the repaired row identity differs from its case and visible question')
    base.validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    context = json.loads(row['messages'][1]['content'])['observations']
    if len(context) != 1 or context[0]['operation'] != 'get_gene_layers':
        raise ValueError('exactly one original complete family observation is required')
    saved = context[0]
    scope = saved['arguments']['layer_scope']
    runtime = GeneLayersRuntime(view)
    selected = runtime.resolve_scope(scope)
    if (scope['kind'] != 'all' or scope['layer_ids'] != [] or scope['layer_family'] != 'tf_target'
            or saved['arguments']['mode'] != 'complete' or saved['arguments']['top_k'] is not None
            or not 0 < len(selected) < len(view.layer_order) or saved['result']['layer_count'] <= 0
            or saved['result']['collection']['complete'] is not True):
        raise ValueError('the retained record is not a complete positive proper-family acquisition')
    errors = [m['content'] for m in row['messages'] if m['role'] == 'tool' and m['content']['status'] == 'error']
    expected = ([_failure(None)] if case_id == 'S2.5.14' else [_failure(1)] if case_id == 'S2.5.15'
                else [_failure(1), _failure(0)])
    if errors != expected or not matches(case_id, row):
        raise ValueError('the approved family failure/recovery/exhaustion case is not established')
    # Base replay already checks that only uncovered layers are acquired. Keep
    # this explicit guard so an expanded family scan cannot silently be added.
    for message in row['messages']:
        if 'tool_call' in message:
            requested = runtime.resolve_scope(message['tool_call']['arguments']['layer_scope'])
            if set(requested) & set(selected):
                raise ValueError('a remaining-scope call reacquired the saved family')
