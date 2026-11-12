"""Check explicit late-family case bindings against actual generator fixtures."""

from copy import deepcopy
import json

import pytest

from runtime.world_model.s2_audit_cases_11_13 import CASE_BINDINGS, matches
from runtime.world_model.s2_production_cases import load_catalog
import test_world_model_s2_11_corpus as path_fixture
import test_world_model_s2_12_corpus as member_fixture
import test_world_model_s2_13_corpus as pair_fixture

PATH_PPI, BRAIN = path_fixture.PPI, path_fixture.BRAIN
MEMBER_PPI = member_fixture.PPI
PPI, CD4, BULK = pair_fixture.PPI, pair_fixture.CD4, pair_fixture.BULK


@pytest.fixture(scope='module')
def path_builder():
    """Reuse the native source motifs from the independently validated builder suite."""
    path_fixture.PathCorpusTests.setUpClass()
    result = path_fixture.PathCorpusTests()
    result.setUp()
    return result


@pytest.fixture(scope='module')
def member_builder():
    """Supply the source-backed singleton fixture and pinned request bank."""
    member_fixture.SingletonCorpusTests.setUpClass()
    result = member_fixture.SingletonCorpusTests()
    result.setUp()
    return result


@pytest.fixture(scope='module')
def pair_builder():
    """Supply actual generated biological cards, rather than hand-drafted targets."""
    pair_fixture.SpecificityCorpusTests.setUpClass()
    return pair_fixture.SpecificityCorpusTests()


def test_inventory_binds_every_reviewed_case_in_these_three_families():
    expected = {case.case_id for case in load_catalog() if case.family in ('S2.11', 'S2.12', 'S2.13')}
    assert set(CASE_BINDINGS) == expected
    assert len(CASE_BINDINGS) == 112
    assert all(binding['generator'] and binding['variants'] and binding['requirements']
               for binding in CASE_BINDINGS.values())


def test_negative_edge_cannot_be_substituted_for_a_path_search(path_builder):
    p = path_builder
    row = p.row([p.a, p.x], PATH_PPI, variant='edge_context')
    p.validate(row)
    assert matches('S2.11.8', row)
    changed = deepcopy(row)
    changed['messages'] = changed['messages'][:3] + [changed['messages'][-1]]
    assert not matches('S2.11.8', changed)
    changed = deepcopy(row)
    changed['messages'][-1]['content']['result']['hop_count'] = 1
    assert not matches('S2.11.8', changed)


def test_loop_and_loop_free_zero_hop_cases_need_different_visible_sources(path_builder):
    p = path_builder
    loop = p.row([p.a, p.a], BRAIN, variant='subgraph_context')
    no_loop = p.row([p.b, p.b], BRAIN, variant='subgraph_context')
    for row in (loop, no_loop):
        p.validate(row)
    assert matches('S2.11.15', loop)
    assert not matches('S2.11.16', loop)
    assert matches('S2.11.16', no_loop)
    assert not matches('S2.11.15', no_loop)


def test_claim_only_and_requested_membership_are_distinct_cases(member_builder):
    p = member_builder
    classification = p.row(p.pen, MEMBER_PPI, variant='context',
                           question_kind='membership_card')
    support = p.row(p.pen, MEMBER_PPI, variant='context',
                    question_kind='claim_support', claim='gene_not_expressed_in_context')
    assert matches('S2.12.12', classification)
    assert not matches('S2.12.20', classification)
    assert matches('S2.12.20', support)
    assert not matches('S2.12.12', support)


def test_bounded_membership_omission_cannot_pass_as_a_shown_positive(member_builder):
    p = member_builder
    omitted = p.row(p.a, CD4, variant='prefix_omission_context',
                    question_kind='membership_support', claim='gene_not_node_in_layer')
    assert matches('S2.12.15', omitted)
    assert not matches('S2.12.16', omitted)
    changed = deepcopy(omitted)
    context = json.loads(changed['messages'][1]['content'])
    context['observations'][0]['result']['collection']['complete'] = True
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.12.15', changed)


def test_family_capability_is_not_positive_pair_evidence(pair_builder):
    from runtime.world_model import s2_13_corpus as corpus
    p = pair_builder
    family = corpus.build_family_row(p.view, 'ppi', 'train', 'direct', 0, p.bank, p.contract)
    pair = p.row(question_kind='edge_support', claim='physical_interaction_recorded')
    p.validate(family)
    p.validate(pair)
    assert matches('S2.13.1', family)
    assert not matches('S2.13.7', family)
    assert matches('S2.13.7', pair)
    assert not matches('S2.13.1', pair)


def test_high_pen_weight_requires_source_relative_control_not_numeric_guess(pair_builder):
    p = pair_builder
    row = p.row(p.named['pen', 'train', '11'], BULK, variant='context',
                question_kind='claim_support', claim='causal_effect')
    p.validate(row)
    assert not matches('S2.13.16', row)
    row['_audit_source'] = {'high_weight_control': True}
    assert matches('S2.13.16', row)
    row['_audit_source']['high_weight_control'] = False
    assert not matches('S2.13.16', row)


def test_positive_route_does_not_become_direct_physical_interaction(pair_builder):
    p = pair_builder
    route = p.row(p.named[PPI, 'train', 'route'], PPI, variant='route_context',
                  question_kind='claim_support', claim='physical_interaction_recorded')
    p.validate(route)
    assert matches('S2.13.37', route)
    assert not matches('S2.13.27', route)
    route['messages'][-1]['content']['result']['claim_cards'][0]['claim_supported'] = True
    assert not matches('S2.13.37', route)
