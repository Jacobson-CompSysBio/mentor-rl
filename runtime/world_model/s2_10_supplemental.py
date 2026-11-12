"""Find faithful strong-prefix witnesses beyond the historical sampling window.

Search only native directed records whose subjects already belong to the
requested split. Strong-component decomposition removes irrelevant vertices;
no source arc, input split, historical builder, or published panel is changed.
"""

from __future__ import annotations

from copy import deepcopy
import json

from .s2_2_corpus import VariantUnavailable
from .s2_3_corpus import gene_group
from .s2_10_corpus import build_row, validate_row as _validate_base_row
from .s2_10_sampling import fact_from_genes
from .s2_components import components
from .s2_corpus import SPLITS


def strong_prefix(index, layer_id, gene_ids):
    """Return the first sufficient proper canonical prefix, or no witness.

    Connectivity grows monotonically as native arcs are added. Testing all but
    the last canonical arc first avoids scanning prefixes that cannot succeed.
    """
    genes = sorted(set(gene_ids))
    edges = index.edges(layer_id, genes)
    if len(genes) < 2 or len(edges) < 2 or len(components(genes, edges[:-1], 'strong')) != 1:
        return None
    lower, upper = 1, len(edges) - 1
    while lower < upper:
        middle = (lower + upper) // 2
        if len(components(genes, edges[:middle], 'strong')) == 1:
            upper = middle
        else:
            lower = middle + 1
    return lower


def _reduce_witness(index, layer_id, gene_ids):
    """Remove vertices only when recomputed native evidence keeps the proof.

    This deterministic reduction finds a smaller source witness; it makes no
    claim of globally minimum input size or token length.
    """
    genes, removed = sorted(gene_ids), []
    while True:
        for gene in genes:
            trial = [item for item in genes if item != gene]
            if strong_prefix(index, layer_id, trial) is not None:
                genes = trial
                removed.append(gene)
                break
        else:
            return genes, removed


def select_fact(view, index, *, split='validation', seed=20261028):
    """Select a real strong-prefix case without a finite vertex-window shortcut.

    Any suitable selected set lies inside a strong component. If removing a
    component's last canonical arc breaks it, a suitable subset must lie wholly
    in one resulting strong component: its own sufficient proper prefix cannot
    need that last arc. Repeating this decomposition searches beyond the old
    size-two/three and 128-vertex limits without enumerating arbitrary subsets.
    """
    if split not in SPLITS or type(seed) is not int:
        raise ValueError('a fixed source split and integer seed are required')
    domains, examined, candidates = [], [], []
    for layer_id in view.layer_order:
        layer = view.layers[layer_id]
        if not layer.directed:
            continue
        genes = sorted(gene for gene in layer.nodes if gene_group(gene, seed)[1] == split)
        edges = index.edges(layer_id, genes)
        groups = [group for group in components(genes, edges, 'strong') if len(group) > 1]
        domains.append({'layer_id': layer_id, 'same_split_members': len(genes),
                        'same_split_native_arcs': len(edges),
                        'nontrivial_strong_component_sizes': sorted(map(len, groups))})
        pending = sorted(groups, key=lambda group: (len(group), tuple(group)))
        while pending:
            group = pending.pop(0)
            native = index.edges(layer_id, group)
            prefix = strong_prefix(index, layer_id, group)
            record = {'layer_id': layer_id, 'gene_ids': group, 'native_arcs': len(native),
                      'proper_strong_prefix': prefix}
            examined.append(record)
            if prefix is not None:
                reduced, removed = _reduce_witness(index, layer_id, group)
                fact = fact_from_genes(view, index, layer_id, reduced, 'strong', seed,
                                       split=split, category='supplemental_strong_prefix')
                candidates.append((len(reduced), fact.edge_count, layer_id, tuple(reduced), fact,
                                   {'original_gene_ids': group, 'removed_gene_ids': removed}))
                continue
            children = [child for child in components(group, native[:-1], 'strong') if len(child) > 1]
            record['removed_last_native_arc'] = deepcopy(native[-1])
            record['child_strong_component_sizes'] = sorted(map(len, children))
            if any(len(child) >= len(group) for child in children):
                raise ValueError('failed strong-prefix decomposition did not make progress')
            pending.extend(children)
            pending.sort(key=lambda child: (len(child), tuple(child)))
    if not candidates:
        raise VariantUnavailable(f'no native proper strong-prefix witness in the fixed {split} split')
    chosen = min(candidates, key=lambda item: item[:4])
    fact, reduction = chosen[4:]
    prefix = strong_prefix(index, fact.layer_id, fact.gene_ids)
    report = {'split': split, 'seed': seed,
              'search': 'complete same-split directed inventories, recursive strong-component canonical-arc peeling',
              'source_domains': domains, 'examined_strong_components': examined,
              'qualifying_components': len(candidates), 'all_combinatorial_subsets_enumerated': False,
              'vertex_window': None, 'maximum_input_size': None,
              'selected_layer_id': fact.layer_id, 'selected_gene_ids': list(fact.gene_ids),
              'selected_native_arc_count': fact.edge_count, 'selected_prefix_arc_count': prefix,
              'reduction': reduction, 'reduction_is_globally_minimal': False}
    return fact, report


def validate_strong_witness(row, view, index, *, bank, contract, registry=None):
    """Check source truth and reject a weak, complete, or cross-split substitute."""
    _validate_base_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
    meta = row['metadata']
    if meta['variant'] != 'strong_witness_context' or meta['connectivity_kind'] != 'strong':
        raise ValueError('the supplemental case requires an explicit strong witness')
    observations = json.loads(row['messages'][1]['content'])['observations']
    if len(observations) != 1 or observations[0]['operation'] != 'induce_subgraph':
        raise ValueError('the proof needs one preserved native induced-graph acquisition')
    saved = observations[0]
    graph = saved['result']
    native = index.edges(meta['layer_id'], graph['layer_gene_ids'])
    if (saved['arguments']['mode'] != 'top_k' or graph['collection']['complete']
            or not 0 < len(graph['edges']) < len(native)
            or graph['edges'] != native[:len(graph['edges'])]
            or len(components(graph['layer_gene_ids'], graph['edges'], 'strong')) != 1):
        raise ValueError('saved evidence is not a genuinely partial, sufficient canonical strong prefix')
    if (any('tool_call' in message for message in row['messages'])
            or row['messages'][-1]['content']['result']['component_count'] != 1):
        raise ValueError('a sufficient strong prefix must settle one component without a new call')


def validate_row(row, view, *, index, bank, contract, registry=None):
    """Expose source replay and the supplemental strong-proof guard together."""
    validate_strong_witness(row, view, index, bank=bank, contract=contract, registry=registry)


def build_rows(view, index, bank, contract, registry=None, *, split='validation', seed=20261028):
    """Generate both framings from the same newly selected native strong proof."""
    fact, report = select_fact(view, index, split=split, seed=seed)
    rows = []
    for framing in ('direct', 'step'):
        row = build_row(view, fact, framing, 'strong_witness_context', 0, bank, contract,
                        index=index, registry=registry, seed=seed)
        validate_strong_witness(row, view, index, bank=bank, contract=contract, registry=registry)
        rows.append(row)
    return rows, report
