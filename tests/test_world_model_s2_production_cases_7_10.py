"""Check frozen wording expansion without relaxing original native proofs."""

from copy import deepcopy
import importlib
import json

import pytest

from runtime.world_model import s2_production_cases_7_10 as production
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


@pytest.fixture(scope='module')
def contexts():
    """Use complete small native inventories with every original source split."""
    genes = [f'ENSG{ordinal:011d}' for ordinal in range(1, 81)]
    arcs = {(gene, gene): .9 for gene in genes}
    arcs.update({(gene, genes[(ordinal + 1) % len(genes)]): .5 for ordinal, gene in enumerate(genes)})
    undirected = {tuple(sorted(pair)): weight for pair, weight in arcs.items()}
    view = GraphView('production_wording_unit', 'f' * 64, {
        'HumanNetV3:string_ppi': LayerGraph('HumanNetV3:string_ppi', False, undirected, frozenset(genes)),
        'TFs:brain': LayerGraph('TFs:brain', True, arcs, frozenset(genes)),
        'TFs:cd4+t_cells': LayerGraph('TFs:cd4+t_cells', True, arcs, frozenset(genes))})
    index = SubgraphIndex(view)
    result = {}
    for number in (7, 8, 9, 10):
        base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
        contract = base.load_build_contract()
        context = {'view': view, 'index': index, 'bank': load_s2_prompt_banks(contract['prompt_bank']),
                   'contract': contract, 'registry': None, 'seed': 20261028}
        production.prepare_context(context, f'S2.{number}')
        result[number] = context
    return result


def shared_seed(context):
    """Bind a repeated exact role to the reviewed single-acquisition case."""
    from runtime.world_model.s2_7_corpus import build_row
    from runtime.world_model.s2_7_sampling import fact_from_queries
    gene = next(gene for gene in sorted(context['view'].multiplex_nodes) if gene_group(gene)[1] == 'train')
    fact = fact_from_queries(context['view'], context['index'].native_index,
        'HumanNetV3:string_ppi', [(gene, 'undirected')] * 2)
    row = build_row(context['view'], fact, 'direct', 'lookup', 0, context['bank'], context['contract'],
                    index=context['index'].native_index)
    row['metadata']['catalog_case_id'] = 'S2.7.23'
    return row


def test_wrapper_request_product_adds_real_wording_without_changing_evidence(contexts):
    context = contexts[7]
    seed = shared_seed(context)
    rows = [production._reword(seed, context, wrapper, request) for wrapper in range(32) for request in range(4)]
    assert len({model_trace_sha256(row) for row in rows}) == 128
    assert all(row['messages'][1] == seed['messages'][1] and row['messages'][3:] == seed['messages'][3:]
               for row in rows)
    for row in (rows[0], rows[7], rows[-1]):
        production.validate_row(row, context)


@pytest.mark.parametrize('change', ['question', 'wrapper', 'request', 'identity', 'answer', 'source_weight'])
def test_wording_normalization_cannot_hide_source_or_answer_changes(contexts, change):
    context = contexts[7]
    row = production._reword(shared_seed(context), context, 5, 2)
    if change == 'question':
        row['messages'][2]['content'] += ' Alter the request.'
    elif change == 'wrapper':
        row['metadata']['wrapper_index'] = 6
    elif change == 'request':
        row['metadata']['request_template_index'] = -1
    elif change == 'identity':
        row['row_id'] = 'unbound_row'
    elif change == 'answer':
        row['messages'][-1]['content']['result']['shared_neighbor_count'] += 1
    else:
        next(message for message in row['messages'] if message['role'] == 'tool')['content']['result']['neighbors'][0]['weight'] = 1.5
    with pytest.raises(ValueError):
        production.validate_row(row, context)


def test_free_summary_is_not_scored_exactly(contexts):
    context = contexts[7]
    row = production._reword(shared_seed(context), context, 5, 2)
    row['messages'][-1]['content']['summary'] = 'The supplied native records establish the exact shared set.'
    production.validate_row(row, context)


def test_combined_degree_repair_keeps_its_strict_original_validator(contexts):
    from runtime.world_model.s2_8_case_repairs import _build_case
    from runtime.world_model.s2_8_sampling import fact_from_arguments
    context = contexts[8]
    gene = next(gene for gene in sorted(context['view'].multiplex_nodes) if gene_group(gene)[1] == 'train')
    fact = fact_from_arguments(context['view'], context['_degree_index'], gene,
        {'kind': 'layers', 'layer_ids': ['TFs:brain'], 'layer_family': None}, 'out', category='case_repair')
    seed = _build_case(context['view'], context['_degree_index'], context['bank'], context['contract'], None,
                      fact, 'S2.8.22', 'TFs:cd4+t_cells', context['seed'])
    row = production._reword(seed, context, 31, 3)
    production.validate_row(row, context)
    altered = deepcopy(row)
    saved = json.loads(altered['messages'][1]['content'])
    saved['observations'][0]['result']['degree'] += 1
    altered['messages'][1]['content'] = json.dumps(saved)
    with pytest.raises(ValueError):
        production.validate_row(altered, context)


def test_new_subgraph_wording_preserves_a_real_incomplete_loop_page(contexts):
    from runtime.world_model.s2_9_loop_context import build_rows
    context = contexts[9]
    rows, _ = build_rows(context['view'], context['index'], context['bank'], context['contract'], split='test')
    seed = rows[0]
    seed['metadata']['catalog_case_id'] = 'S2.9.27'
    row = production._reword(seed, context, 7, 7)
    production.validate_row(row, context)
    assert row['messages'][1] == seed['messages'][1]
    assert row['messages'][3:] == seed['messages'][3:]
    assert production._template_count(row, context) == 8


def test_expanded_requests_keep_historical_bank_objects_unchanged(contexts):
    for number in (8, 9, 10):
        context = contexts[number]
        base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
        assert context['contract']['_requests'] == base.load_build_contract()['_requests']
        kinds = (('degree_only', 'degree_and_hub') if number == 8 else
                 ('complete', 'top_k') if number == 9 else ('partition',))
        for split, count in (('train', 16), ('validation', 8), ('test', 8)):
            for kind in kinds:
                assert len(context['_production_requests'][split][kind]) == count
                assert context['_production_requests'][split][kind][:4] == context['contract']['_requests'][split][kind]


def test_production_additions_obey_the_original_request_semantic_schemas(contexts):
    from runtime.world_model.s2_8_prompts import validate_s2_8_requests
    from runtime.world_model.s2_9_prompts import validate_s2_9_requests
    from runtime.world_model.s2_10_prompts import validate_s2_10_requests
    validate_s2_8_requests(contexts[8]['_production_requests'])
    validate_s2_9_requests(contexts[9]['_production_requests'])
    validate_s2_10_requests(contexts[10]['_production_requests'])


def test_live_source_proofs_exclude_only_the_user_approved_unattainable_cases():
    for case, expected in ((29, {'train': 13, 'validation': 0, 'test': 1}),
                           (30, {'train': 1, 'validation': 0, 'test': 0})):
        path = production.ROOT / f'diagnostics/s2_production_coverage/s2_8_{case}_source_capacity.json'
        report = json.loads(path.read_text())
        assert report['unique_qualifying_gene_counts'] == expected
        assert report['all_split_quotas_feasible'] is False


def test_missing_raw_genes_do_not_create_private_only_source_diversity(contexts):
    from runtime.world_model.s2_9_corpus import build_diagnostic
    from runtime.world_model.s2_9_sampling import fact_from_genes
    context = contexts[9]
    gene = next(gene for gene in sorted(context['view'].multiplex_nodes) if gene_group(gene)[1] == 'train')
    fact = fact_from_genes(context['view'], context['index'], 'TFs:brain', [gene])
    row = build_diagnostic(context['view'], fact, 'direct', 0, context['bank'], context['contract'], kind='missing_genes')
    private_change = deepcopy(row)
    private_change['metadata']['gene_ids'] = ['ENSG99999999999']
    assert production._source_key(row) == production._source_key(private_change)
    private_change['metadata']['request']['layer_id'] = 'TFs:cd4+t_cells'
    assert production._source_key(row) != production._source_key(private_change)


def test_specific_prefix_lower_bound_is_reserved_without_changing_frozen_predicates(contexts):
    from runtime.world_model.s2_7_corpus import build_row
    from runtime.world_model.s2_7_sampling import fact_from_queries
    from runtime.world_model.s2_audit_cases_7_10 import matches
    genes = sorted(contexts[7]['view'].multiplex_nodes)
    first, second = [gene for gene in genes if gene_group(gene)[1] == 'train'][:2]
    common = [gene for gene in genes if gene not in (first, second)][:2]
    layer = 'HumanNetV3:string_ppi'
    edges = {tuple(sorted((query, target))): .9 - ordinal / 10
             for query in (first, second) for ordinal, target in enumerate(common)}
    view = GraphView('prefix_reservation_unit', 'c' * 64,
        {layer: LayerGraph(layer, False, edges, frozenset((first, second, *common)))})
    context = dict(contexts[7], view=view, index=SubgraphIndex(view))
    for key in ('_features', '_degree_index'):
        context.pop(key)
    production.prepare_context(context, 'S2.7')
    fact = fact_from_queries(view, context['index'].native_index, layer,
                            [(first, 'undirected'), (second, 'undirected')])
    row = build_row(view, fact, 'direct', 'bounded_context', 0, context['bank'], context['contract'],
                    index=context['index'].native_index)
    assert matches('S2.7.12', row) and matches('S2.7.27', row)
    assert production._eligible('S2.7.27', row, context)
    assert not production._eligible('S2.7.12', row, context)
