"""Complete sparse diagnostic and weak-prefix production pools for S2.9/10.

The first production helper remains frozen. This extension adds genuine raw
query coordinates to its sparse diagnostic pools and reserves weakly sufficient
strong-connectivity contrasts from the broader insufficient-prefix case.
"""

from __future__ import annotations

from . import s2_production_cases_7_10 as original
from .s2_2_corpus import VariantUnavailable
from .s2_3_corpus import gene_group
from .s2_audit_cases_7_10 import matches


CASE_IDS = tuple(case_id for case_id in original.CASE_IDS if case_id.startswith(('S2.9.', 'S2.10.')))
INPUT_PATHS = original.INPUT_PATHS
DIAGNOSTICS = {'S2.9.19': 'unresolved_identity', 'S2.9.24': 'missing_genes',
    'S2.9.28': 'tool_unknown_layer', 'S2.10.16': 'missing_kind',
    'S2.10.29': 'tool_unknown_layer', 'S2.10.30': 'missing_genes', 'S2.10.31': 'unresolved_identity'}


def prepare_context(context, family):
    """Normalize the orchestrator's family number before sharing frozen caches."""
    if family in (9, 10, '9', '10'):
        family = f'S2.{int(family)}'
    if family not in ('S2.9', 'S2.10'):
        raise ValueError('the sparse-pool extension supports S2.9 and S2.10')
    return original.prepare_context(context, family)


def _diagnostic_seeds(case_id, context, split, pools):
    """Bind true fixed-split canonical inputs across the registered source layers."""
    number = int(case_id.split('.')[1])
    if number == 9:
        from .s2_9_corpus import build_diagnostic
        from .s2_9_sampling import fact_from_genes
    else:
        from .s2_10_corpus import build_diagnostic
        from .s2_10_sampling import fact_from_genes
    view, index, seed = context['view'], context['index'], context['seed']
    for layer in view.layer_order:
        genes = sorted(gene for gene in view.layers[layer].nodes if gene_group(gene, seed)[1] == split)[:8]
        kinds = ('weak', 'strong') if view.layers[layer].directed else ('undirected',)
        for gene in genes:
            for kind in kinds if number == 10 else (None,):
                kwargs = {'split': split, 'category': 'production_diagnostic_coordinate'}
                fact = (fact_from_genes(view, index, layer, [gene], seed, **kwargs) if number == 9 else
                        fact_from_genes(view, index, layer, [gene], kind, seed, **kwargs))
                for framing in ('direct', 'step'):
                    row = build_diagnostic(view, fact, framing, 0, context['bank'], context['contract'],
                        kind=DIAGNOSTICS[case_id], registry=context.get('registry'), seed=seed)
                    original._add_seed(pools, case_id, row, context)


def _weak_prefix_seeds(context, split, pools):
    """Vary actual input ordering while preserving the native weak-prefix proof."""
    from .s2_10_corpus import build_row
    from .s2_10_sampling import fact_from_genes
    view, index, seed = context['view'], context['index'], context['seed']
    seeds = [row for _, row in pools['S2.10.21', split].values()]
    for row in seeds:
        meta, genes = row['metadata'], row['metadata']['gene_ids']
        orders = {tuple(genes[offset:] + genes[:offset]) for offset in range(len(genes))}
        orders.update(tuple(reversed(order)) for order in list(orders))
        for order in sorted(orders):
            fact = fact_from_genes(view, index, meta['layer_id'], order, 'strong', seed,
                                   split=split, category='production_weak_prefix_order')
            try:
                candidate = build_row(view, fact, 'direct', 'weak_witness_strong_context', 0,
                    context['bank'], context['contract'], index=index, registry=context.get('registry'), seed=seed)
            except VariantUnavailable:
                continue
            original._add_seed(pools, 'S2.10.21', candidate, context)


def _specific_weak_prefix(row, context):
    """Recognize the scarce approved contrast without changing its predicate."""
    classified = dict(row)
    classified['_audit_source'] = context['_features'].for_row(row)
    return matches('S2.10.21', classified)


def production_candidates(case_id, context, *, split):
    """Extend only identified sparse cells, preserving the original exact cases."""
    if case_id not in CASE_IDS or split not in original.SPLITS:
        raise ValueError('a reviewed S2.9/10 case and fixed split are required')
    family = '.'.join(case_id.split('.')[:2])
    prepare_context(context, family)
    pools = original._published_seeds(context, family)
    augmented = context.setdefault('_sparse_pool_v2_cells', set())
    if (case_id, split) not in augmented:
        if case_id in DIAGNOSTICS:
            _diagnostic_seeds(case_id, context, split, pools)
        elif case_id == 'S2.10.21':
            _weak_prefix_seeds(context, split, pools)
        augmented.add((case_id, split))
    for row in original.production_candidates(case_id, context, split=split):
        if case_id == 'S2.10.19' and _specific_weak_prefix(row, context):
            continue
        yield row


def validate_row(row, context):
    """Keep the first helper's frozen wording and complete native source checks."""
    original.validate_row(row, context)
