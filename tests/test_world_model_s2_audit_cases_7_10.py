"""Check catalog audit predicates using actual builders and adversarial contexts."""

from copy import deepcopy
from collections import Counter
import json

import pytest

from runtime.world_model.s2_audit_cases_7_10 import CASE_BINDINGS, matches
from runtime.world_model.s2_production_cases import load_catalog
import test_world_model_s2_7_corpus as shared_fixture
import test_world_model_s2_8_corpus as degree_fixture
import test_world_model_s2_9_corpus as graph_fixture
import test_world_model_s2_10_corpus as component_fixture


BRAIN = shared_fixture.BRAIN


@pytest.fixture(scope='module')
def fixtures():
    """Reuse established native graph fixtures through their real row builders."""
    result = {}
    for number, cls in ((7, shared_fixture.WorldModelS27CorpusTests), (8, degree_fixture.DegreeCorpusTests),
                        (9, graph_fixture.SubgraphCorpusTests), (10, component_fixture.ComponentCorpusTests)):
        cls.setUpClass()
        instance = cls()
        instance.setUp()
        result[number] = instance
    return result


def test_every_reviewed_case_has_an_explicit_generator_binding():
    expected = {case.case_id for case in load_catalog() if case.family in ('S2.7', 'S2.8', 'S2.9', 'S2.10')}
    assert set(CASE_BINDINGS) == expected
    assert Counter(key.rsplit('.', 1)[0] for key in expected) == {'S2.7': 29, 'S2.8': 31, 'S2.9': 30, 'S2.10': 34}
    assert all(value['generator'].startswith('runtime.world_model.') and value['variants']
               and value['requirements'] for value in CASE_BINDINGS.values())


def test_shared_pair_adjacency_is_a_real_source_condition(fixtures):
    item = fixtures[7]._row()
    assert not matches('S2.7.1', item)
    assert not matches('S2.7.2', item)
    item['_audit_source'] = {'input_pair_adjacent': True}
    assert matches('S2.7.2', item)
    assert not matches('S2.7.1', item)


def test_shared_bounded_positive_is_not_an_exact_saved_intersection(fixtures):
    item = fixtures[7]._row(variant='bounded_context')
    assert matches('S2.7.27', item)
    changed = deepcopy(item)
    observations = json.loads(changed['messages'][1]['content'])['observations']
    for observation in observations:
        observation['result']['collection']['complete'] = True
    changed['messages'][1]['content'] = json.dumps({'observations': observations})
    assert not matches('S2.7.27', changed)


def test_containment_requires_a_complete_set_and_genuine_incomplete_other_prefix(fixtures):
    shared = fixtures[7]
    item = shared._row((shared.e, shared.a), variant='containment_context')
    assert matches('S2.7.28', item)
    changed = deepcopy(item)
    context = json.loads(changed['messages'][1]['content'])
    for observation in context['observations']:
        observation['result']['collection']['complete'] = True
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.7.28', changed)


def test_repeated_roles_do_not_substitute_for_opposite_directions(fixtures):
    shared = fixtures[7]
    repeated = shared._row((shared.a, shared.a), layer=BRAIN, directions=('out', 'out'))
    opposite = shared._row((shared.a, shared.a), layer=BRAIN, directions=('out', 'in'))
    assert matches('S2.7.23', repeated)
    assert not matches('S2.7.29', repeated)
    assert matches('S2.7.29', opposite)


def test_exact_threshold_predicate_uses_rank_support_not_the_display(fixtures):
    degree = fixtures[8]
    item = degree.row(variant='threshold_hundred')
    assert matches('S2.8.3', item)
    changed = deepcopy(item)
    changed['metadata']['variant'] = 'lookup'
    changed['metadata']['hub_policy']['threshold_percentile'] = 95
    support = {'population_count': 19033, 'lower_degree_count': 18077, 'equal_degree_count': 7}
    changed['metadata']['rank_support'] = support
    answer = changed['messages'][-1]['content']['result']
    answer.update(rank_support=support, hub_policy={'threshold_percentile': 95}, degree_percentile=95.0, hub_like=False)
    assert matches('S2.8.29', changed)
    assert not matches('S2.8.3', changed)
    answer['hub_like'] = True
    assert not matches('S2.8.29', changed)


def test_single_incompatibility_and_degree_only_retry_do_not_cover_joint_cases(fixtures):
    degree = fixtures[8]
    for variant in ('stale', 'wrong_scope', 'wrong_direction'):
        item = degree.row(selected={'kind': 'layers', 'layer_ids': [BRAIN], 'layer_family': None},
                          direction='out', variant=variant)
        assert not matches('S2.8.22', item)
    exhausted = degree.row(variant='rank_exhausted')
    assert not matches('S2.8.26', exhausted)
    assert CASE_BINDINGS['S2.8.22']['implementation_gap']
    assert CASE_BINDINGS['S2.8.26']['implementation_gap']


def test_histogram_evidence_uses_the_actual_degree_distribution_operation(fixtures):
    item = fixtures[8].row(variant='histogram_context')
    assert matches('S2.8.21', item)


def test_singleton_fresh_loop_and_positive_partial_loop_are_distinct(fixtures):
    graph = fixtures[9]
    fresh = graph.row([graph.a])
    partial = graph.row([graph.a], variant='positive_loop_context')
    assert matches('S2.9.9', fresh)
    assert not matches('S2.9.27', fresh)
    assert matches('S2.9.27', partial)
    assert not matches('S2.9.9', partial)


def test_generic_exhausted_graph_cannot_cover_preserved_neighbor_case(fixtures):
    item = fixtures[9].row(variant='exhausted')
    assert not matches('S2.9.21', item)
    assert CASE_BINDINGS['S2.9.21']['implementation_gap']
    assert CASE_BINDINGS['S2.9.16']['implementation_gap']


def test_strong_prefix_cannot_be_substituted_by_complete_context(fixtures):
    component = fixtures[10]
    item = component.row([component.a, component.b, component.c], BRAIN, 'strong', variant='strong_witness_context')
    assert matches('S2.10.20', item)
    changed = deepcopy(item)
    context = json.loads(changed['messages'][1]['content'])
    context['observations'][0]['result']['collection']['complete'] = True
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.10.20', changed)


def test_chain_predicate_uses_native_edges_instead_of_sampling_category(fixtures):
    component = fixtures[10]
    for kind, case in (('weak', 'S2.10.13'), ('strong', 'S2.10.14')):
        item = component.row([component.x, component.y, component.z], BRAIN, kind)
        item['metadata']['category'] = 'constructed'
        assert matches(case, item)
        changed = deepcopy(item)
        changed['metadata']['edge_count'] = 3
        assert not matches(case, changed)


def test_weak_saved_prefix_requires_actual_weak_connectivity(fixtures):
    component = fixtures[10]
    item = component.row([component.a, component.b, component.c], BRAIN, 'strong', variant='weak_witness_strong_context')
    assert matches('S2.10.21', item)
    changed = deepcopy(item)
    context = json.loads(changed['messages'][1]['content'])
    context['observations'][0]['result']['edges'] = []
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.10.21', changed)


def test_disjoint_batches_must_actually_omit_a_cross_batch_edge(fixtures):
    graph = fixtures[9]
    item = graph.row([graph.a, graph.b, graph.c], variant='disjoint_context')
    assert matches('S2.9.29', item)
    changed = deepcopy(item)
    changed['messages'][-1]['content']['result']['edges'] = []
    assert not matches('S2.9.29', changed)


def test_unknown_cases_raise_and_malformed_rows_never_count():
    with pytest.raises(KeyError):
        matches('S2.7.999', {})
    assert not matches('S2.7.1', {})
