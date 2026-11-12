"""Check sparse-pool extensions without weakening native source validation."""

from collections import defaultdict
from copy import deepcopy
import importlib

import pytest

from runtime.world_model import s2_production_cases_9_10_v2 as sparse
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


@pytest.fixture(scope='module')
def contexts():
    """Provide many true fixed-split IDs and multiple native layer coordinates."""
    genes = [f'ENSG{ordinal:011d}' for ordinal in range(1, 81)]
    arcs = {(gene, genes[(ordinal + 1) % len(genes)]): .5 for ordinal, gene in enumerate(genes)}
    undirected = {tuple(sorted(pair)): value for pair, value in arcs.items()}
    view = GraphView('sparse_production_unit', '8' * 64, {
        'HumanNetV3:string_ppi': LayerGraph('HumanNetV3:string_ppi', False, undirected, frozenset(genes)),
        'TFs:brain': LayerGraph('TFs:brain', True, arcs, frozenset(genes)),
        'TFs:cd4+t_cells': LayerGraph('TFs:cd4+t_cells', True, arcs, frozenset(genes))})
    registry = RegistryContext('110', 'ensembl_homo_sapiens_110_sparse_unit',
                               {gene: f'SPARSE_UNIT_{ordinal}' for ordinal, gene in enumerate(genes)})
    index, result = SubgraphIndex(view), {}
    for number in (9, 10):
        base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
        contract = base.load_build_contract()
        context = {'view': view, 'index': index, 'registry': registry, 'seed': 20261028,
                   'bank': load_s2_prompt_banks(contract['prompt_bank']), 'contract': contract}
        sparse.prepare_context(context, number)
        result[number] = context
    return result


@pytest.mark.parametrize('number', [9, 10, '9', '10', 'S2.9', 'S2.10'])
def test_orchestrator_family_numbers_and_explicit_names_are_supported(contexts, number):
    family = int(str(number).split('.')[-1])
    sparse.prepare_context(contexts[family], number)


@pytest.mark.parametrize('case_id', tuple(sparse.DIAGNOSTICS))
def test_sparse_diagnostics_gain_real_literal_requests_and_replay_exactly(contexts, case_id):
    context = contexts[int(case_id.split('.')[1])]
    pools = defaultdict(dict)
    sparse._diagnostic_seeds(case_id, context, 'train', pools)
    seeds = [row for _, row in pools[case_id, 'train'].values()]
    assert len(seeds) >= 4
    rendered = [sparse.original._reword(row, context, 31, 3) for row in seeds]
    assert len({model_trace_sha256(row) for row in rendered}) >= 4
    for row in rendered[:4]:
        sparse.validate_row(row, context)
    altered = deepcopy(rendered[0])
    altered['messages'][-1]['content']['result']['issue_code'] = 'INVENTED_FAILURE'
    with pytest.raises(ValueError):
        sparse.validate_row(altered, context)


def test_native_input_reordering_keeps_source_split_and_strong_prefix_contrast(contexts):
    from runtime.world_model.s2_10_corpus import build_row
    from runtime.world_model.s2_10_sampling import fact_from_genes
    genes = [gene for gene in sorted(contexts[10]['view'].multiplex_nodes) if gene_group(gene)[1] == 'train'][:2]
    layer = 'TFs:brain'
    view = GraphView('sparse_prefix_order_unit', '7' * 64, {layer: LayerGraph(layer, True,
        {(genes[0], genes[1]): .6, (genes[1], genes[0]): .4}, frozenset(genes))})
    context = {key: value for key, value in contexts[10].items() if not key.startswith('_')}
    context.update(view=view, index=SubgraphIndex(view))
    sparse.prepare_context(context, 10)
    fact = fact_from_genes(view, context['index'], layer, genes, 'strong')
    row = build_row(view, fact, 'direct', 'weak_witness_strong_context', 0, context['bank'], context['contract'],
                    index=context['index'], registry=context['registry'])
    pools = defaultdict(dict)
    sparse.original._add_seed(pools, 'S2.10.21', row, context)
    assert sparse._specific_weak_prefix(row, context)
    sparse._weak_prefix_seeds(context, 'train', pools)
    seeds = [candidate for _, candidate in pools['S2.10.21', 'train'].values()]
    assert {tuple(candidate['metadata']['gene_ids']) for candidate in seeds} == {tuple(genes), tuple(reversed(genes))}
    assert {candidate['metadata']['split_unit'] for candidate in seeds} == {fact.split_unit}
    for seed in seeds:
        sparse.validate_row(sparse.original._reword(seed, context, 31, 15), context)
