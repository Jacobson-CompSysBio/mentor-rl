"""Check source-search coverage and faithful partial strong-connectivity proofs."""

from copy import deepcopy
import json

import pytest

from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_corpus import VariantUnavailable, _observation
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_10_corpus import load_build_contract
from runtime.world_model.s2_10_supplemental import (
    build_rows, select_fact, strong_prefix, validate_strong_witness,
)
from runtime.world_model.s2_components import components
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex, SubgraphRuntime


LAYER = 'TFs:brain'


def gene_pool(split, count):
    """Keep fixture subjects under the real fixed split policy."""
    genes = (f'ENSG{number:011d}' for number in range(1, 10000))
    selected = []
    for gene in genes:
        if gene_group(gene)[1] == split:
            selected.append(gene)
            if len(selected) == count:
                return selected
    raise AssertionError('fixture split domain too small')


def graph(edges, nodes=None):
    """Use declared native unit records, never training source substitutions."""
    members = set(nodes or ()) | {gene for pair in edges for gene in pair}
    view = GraphView('strong_prefix_unit', 'f' * 64, {
        LAYER: LayerGraph(LAYER, True, edges, frozenset(members))})
    return view, SubgraphIndex(view)


@pytest.fixture(scope='module')
def valid_case():
    """Create a three-cycle with a real redundant final canonical arc."""
    a, b, c, d = gene_pool('validation', 4)
    view, index = graph({(a, b): .1, (b, c): .2, (c, a): .3,
                         (c, b): 0, (c, d): .4, (d, a): .5})
    contract = load_build_contract()
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    return view, index, bank, contract, (a, b, c, d)


def test_recursive_search_finds_a_witness_when_full_scc_last_arc_is_essential(valid_case):
    view, index, _, _, (a, b, c, d) = valid_case
    assert strong_prefix(index, LAYER, [a, b, c, d]) is None
    fact, report = select_fact(view, index)
    assert fact.gene_ids == (a, b, c)
    assert fact.connectivity_kind == 'strong'
    assert (fact.edge_count, report['selected_prefix_arc_count']) == (4, 3)
    assert any(record.get('child_strong_component_sizes') for record in report['examined_strong_components'])
    assert all(gene_group(gene)[1] == 'validation' for gene in fact.gene_ids)


def test_search_has_no_128_vertex_cutoff():
    nodes = gene_pool('validation', 270)
    a, b, c = nodes[-3:]
    edges = {(nodes[n], nodes[n+1]): .8 for n in range(0, 260, 2)}
    edges.update({(target, source): .2 for source, target in list(edges)})
    edges.update({(a, b): .1, (b, c): .2, (c, a): .3, (c, b): .4})
    view, index = graph(edges)
    fact, report = select_fact(view, index)
    assert fact.gene_ids == (a, b, c)
    assert report['source_domains'][0]['same_split_members'] == 263
    assert report['vertex_window'] is None


def test_four_member_witness_is_not_lost_to_the_old_three_member_limit():
    a, b, c, d = gene_pool('validation', 4)
    view, index = graph({(a, b): .1, (b, c): .2, (c, d): .3, (d, a): .4, (d, b): .5})
    fact, report = select_fact(view, index)
    assert fact.gene_ids == (a, b, c, d)
    assert report['selected_prefix_arc_count'] == 4
    assert fact.edge_count == 5


def test_cyclic_full_graph_or_cross_split_pair_cannot_fill_the_gap():
    a, b, c = gene_pool('validation', 3)
    view, index = graph({(a, b): .1, (b, c): .2, (c, a): .3})
    with pytest.raises(VariantUnavailable):
        select_fact(view, index)
    train = gene_pool('train', 1)[0]
    view, index = graph({(a, train): .1, (train, a): .2, (a, a): .3})
    with pytest.raises(VariantUnavailable):
        select_fact(view, index)


def test_selection_and_both_framings_are_deterministic_and_keep_partial_evidence(valid_case):
    view, index, bank, contract, _ = valid_case
    first, report = build_rows(view, index, bank, contract)
    second, repeated = build_rows(view, index, bank, contract)
    assert first == second and report == repeated
    for row, decision in zip(first, ('stop', 'continue')):
        saved = json.loads(row['messages'][1]['content'])['observations'][0]
        assert saved['result']['collection']['complete'] is False
        assert len(components(saved['result']['layer_gene_ids'], saved['result']['edges'], 'strong')) == 1
        assert not any('tool_call' in message for message in row['messages'])
        final = row['messages'][-1]['content']
        assert final['continuation_decision'] == decision
        assert final['result']['component_count'] == 1
        assert final['result']['collection']['complete'] is True


@pytest.mark.parametrize('mode,limit', [('complete', None), ('top_k', 2)])
def test_complete_or_weak_saved_evidence_cannot_masquerade_as_partial_strong_proof(valid_case, mode, limit):
    view, index, bank, contract, _ = valid_case
    row = deepcopy(build_rows(view, index, bank, contract)[0][0])
    context = json.loads(row['messages'][1]['content'])
    saved = context['observations'][0]
    arguments = {**saved['arguments'], 'mode': mode, 'top_k': limit}
    evidence_id = saved['provenance']['graph'][0]['evidence_id']
    response = SubgraphRuntime(view, index=index).induce_subgraph(**arguments, evidence_id=evidence_id)
    context['observations'][0] = _observation('induce_subgraph', arguments, response)
    row['messages'][1]['content'] = json.dumps(context)
    with pytest.raises(ValueError):
        validate_strong_witness(row, view, index, bank=bank, contract=contract)


def test_source_weight_tampering_and_split_changes_are_rejected(valid_case):
    view, index, bank, contract, _ = valid_case
    original = build_rows(view, index, bank, contract)[0][0]
    tampered = deepcopy(original)
    context = json.loads(tampered['messages'][1]['content'])
    context['observations'][0]['result']['edges'][0]['weight'] = 99
    tampered['messages'][1]['content'] = json.dumps(context)
    with pytest.raises(ValueError):
        validate_strong_witness(tampered, view, index, bank=bank, contract=contract)
    tampered = deepcopy(original)
    tampered['metadata']['split'] = 'train'
    with pytest.raises(ValueError):
        validate_strong_witness(tampered, view, index, bank=bank, contract=contract)
