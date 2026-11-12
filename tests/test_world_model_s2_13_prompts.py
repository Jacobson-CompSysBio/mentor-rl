"""Check exact task bindings, separate vocabularies, and frozen full questions."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT, load_contract
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_13_prompts import (
    BIOLOGICAL_CLAIMS, DIAGNOSTICS, FAMILIES, FIELDS, KINDS, TASK_FIELDS,
    load_s2_13_requests, render_s2_13_diagnostic_request, render_s2_13_request,
    validate_s2_13_rendered_prompts, validate_s2_13_requests,
)
from runtime.world_model.sources import file_sha256


DIRECTORY = REPOSITORY_ROOT / 'runtime/world_model'
REQUEST_PATH = DIRECTORY / 's2_13_v1_requests.json'
A, B = 'ENSG00000000001', 'ENSG00000000002'


@pytest.fixture
def requests():
    """Give each mutation test an independent request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the frozen wrappers and every earlier S2 request bank."""
    names = ('s2_v1_prompt_banks.json', 's2_v1_edge_requests.json',
             *(f's2_{number}_v1_requests.json' for number in range(2, 13)))
    return tuple(json.loads((DIRECTORY / name).read_text()) for name in names)


def values(kind):
    """Use explicit family values or literal ordered pair and truth inputs."""
    if kind.startswith('family_'):
        return {'layer_family': 'tf_target', **({'claim': 'regulation_direction'} if kind.endswith('support') else {})}
    return {'genes': [' raw symbol ', 'ENSG00000000002.4'], 'multiplex_id': 'ExampleView',
        'layer_id': 'TFs:brain', 'direction': 'source_to_target',
        'subjects': [{'multiplex_id': 'ExampleView', 'layer_id': 'TFs:brain', 'gene_ids': [B, A]}],
        'claim': 'causal_effect', 'graph_claim': 'no_edge_recorded_in_scope', 'graph_truth_scope': 'scope',
        'statement': 'The literal proposition has "quotes" and {braces}.'}


def test_overlay_retains_qualified_sources_tools_and_s1_family_meanings(requests):
    overlay = json.loads((DIRECTORY / 's2_13_v1.json').read_text())
    previous = json.loads((DIRECTORY / 's2_12_v1.json').read_text())
    shared = load_contract()
    assert load_s2_13_requests(overlay['request_bank']) == requests
    report = validate_s2_13_requests(requests)
    assert all(set(kinds.values()) == {4} for kinds in report['template_counts'].values())
    assert all(set(kinds.values()) == {4} for kinds in report['diagnostic_template_counts'].values())
    for key in ('source_manifest', 'previous_source_manifest', 'registry_index', 'prompt_bank',
                'tokenizer', 'context_recipe', 'scoring', 'tools',
                *(f's2{number}_request_bank' for number in range(1, 12))):
        assert overlay[key] == previous[key]
    assert overlay['s212_request_bank'] == previous['request_bank']
    assert overlay['claim_vocabulary'] == list(BIOLOGICAL_CLAIMS) == shared['vocabularies']['layer_claims']
    assert overlay['graph_claim_vocabulary'] == previous['claim_vocabulary']
    assert overlay['layer_families'] == list(FAMILIES) == shared['vocabularies']['layer_family']
    assert overlay['family_cards'] == shared['curriculum']['families']['S1.7']['family_cards']
    assert overlay['implemented_families'] == ['S2.13']
    assert overlay['training_ready'] is overlay['production_export_allowed'] is False
    edge = next(tool['function'] for tool in overlay['tools'] if tool['function']['name'] == 'get_edge')
    assert edge['parameters']['required'] == ['multiplex_id', 'layer_id', 'gene_ids']
    assert set(edge['parameters']['properties']) == {'multiplex_id', 'layer_id', 'gene_ids'}
    for marker in ('s1_layer_v1', 's2_graph_observations_v1', 'UNSUPPORTED_TRUTH_SCOPE',
                   'coverage:null', 'R-squared', 'importance', 'retries_remaining',
                   'no other-family layer is registered', 'Unsupported biological support does not mean false'):
        assert marker in overlay['system_prompt']


def test_all_thirteen_families_have_disjoint_full_questions(requests, prior_banks):
    report = validate_s2_13_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_13_rendered_questions'] == {'train': 3840, 'validation': 960, 'test': 960}
    assert report['cross_split_pairs_checked'] == 661929984
    assert report['maximum_cross_split_jaccard'] < .6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_request_cannot_enter_training(requests, prior_banks):
    requests['train']['claim_card'][0] = requests['validation']['claim_card'][0]
    validate_s2_13_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_13_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('kind', KINDS)
def test_every_task_binds_each_literal_once(requests, split, kind):
    inputs = values(kind)
    original = deepcopy(inputs)
    rendered = render_s2_13_request(requests, split=split, variant_index=5, question_kind=kind, **inputs)
    expected = requests[split][kind][1].format(
        **{key: json.dumps(inputs[key]) for key in TASK_FIELDS[kind]})
    assert rendered == expected and inputs == original


@pytest.mark.parametrize('family', FAMILIES)
def test_family_capability_needs_no_graph_or_pair(requests, family):
    rendered = render_s2_13_request(requests, split='test', variant_index=0,
        question_kind='family_card', layer_family=family)
    assert json.dumps(family) in rendered
    assert 'null' not in rendered and A not in rendered


@pytest.mark.parametrize('extra', [{'genes': [A, B]}, {'multiplex_id': 'M'}, {'layer_id': 'L'},
    {'direction': 'undirected'}, {'subjects': [{'multiplex_id': 'M', 'layer_id': 'L', 'gene_ids': [A, B]}]}])
def test_family_task_does_not_hide_pair_fields(requests, extra):
    with pytest.raises(PromptBankError, match='no pair or graph scope'):
        render_s2_13_request(requests, split='train', variant_index=0,
            question_kind='family_card', layer_family='other', **extra)


@pytest.mark.parametrize('genes', [None, 'TP53', [], ['A'], ['A', 'A'], ['A', 'B', 'C'], [None, 'B'], [True, 'B']])
def test_invalid_pair_structure_requires_a_diagnostic(requests, genes):
    with pytest.raises(PromptBankError, match='literal distinct pair'):
        render_s2_13_request(requests, split='train', variant_index=0,
            **{**values('edge_card'), 'genes': genes})


@pytest.mark.parametrize('subjects', [(), [{}], [{'multiplex_id': 'M', 'layer_id': 'L', 'gene_ids': ['x', 'y']}],
    [{'multiplex_id': 'M', 'layer_id': 'L', 'gene_ids': [A, A]}],
    [{'multiplex_id': None, 'layer_id': 'L', 'gene_ids': [A, B]}],
    [{'multiplex_id': 'M', 'layer_id': 'L', 'gene_ids': [A, B], 'directed': True}]])
def test_subject_order_contains_only_explicit_canonical_edge_selectors(requests, subjects):
    with pytest.raises(PromptBankError):
        render_s2_13_request(requests, split='train', variant_index=0,
            **{**values('edge_card'), 'subjects': subjects})


def test_subject_order_and_requested_direction_remain_literal(requests):
    inputs = values('claim_card')
    inputs['subjects'].append({'multiplex_id': 'ExampleView', 'layer_id': 'bulkPEN:brain:amygdala', 'gene_ids': [A, B]})
    rendered = render_s2_13_request(requests, split='validation', variant_index=2, question_kind='claim_card', **inputs)
    assert json.dumps(inputs['subjects']) in rendered
    assert json.dumps(inputs['direction']) in rendered


@pytest.mark.parametrize('kind', ['family_support', 'edge_support', 'claim_support', 'biological_truth'])
def test_canonical_biological_checks_reject_invented_concepts(requests, kind):
    with pytest.raises(PromptBankError, match='canonical concept'):
        render_s2_13_request(requests, split='train', variant_index=0,
            question_kind=kind, **{**values(kind), 'claim': 'activation'})


@pytest.mark.parametrize('change', [{'claim': 'causal_probability'}, {'graph_claim': 'activation'},
    {'graph_truth_scope': 'all_biology'}, {'statement': None}])
def test_graph_truth_keeps_canonical_vocabularies_and_exact_statement(requests, change):
    with pytest.raises(PromptBankError):
        render_s2_13_request(requests, split='validation', variant_index=0,
            question_kind='graph_truth', **{**values('graph_truth'), **change})


@pytest.mark.parametrize('kind', [*DIAGNOSTICS, 'tool_unknown_layer', 'tool_unknown_multiplex',
                                  'ambiguous_identity', 'invalid_identity'])
def test_diagnostics_preserve_all_four_raw_fields(requests, kind):
    inputs = {'genes': [' raw ', None], 'multiplex_id': None, 'layer_id': {'unknown': 4}, 'direction': ['raw']}
    rendered = render_s2_13_diagnostic_request(requests, split='validation', kind=kind, variant_index=1, **inputs)
    selected = kind.removeprefix('tool_')
    if selected in ('ambiguous_identity', 'invalid_identity'):
        selected = 'unresolved_identity'
    assert rendered == requests['diagnostic_requests']['validation'][selected][1].format(
        **{key: json.dumps(value) for key, value in inputs.items()})


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A',), {3: 'A'}])
def test_non_json_raw_values_are_rejected(requests, field, value):
    inputs = {'genes': [A, B], 'multiplex_id': 'M', 'layer_id': 'L', 'direction': None}
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_13_diagnostic_request(requests, split='test', kind='invalid_pair', variant_index=0,
            **{**inputs, field: value})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_templates_bind_each_declared_field_once(requests, change):
    template = requests['train']['graph_truth'][0]
    if change == 'missing':
        template = template.replace('{statement}', 'an unstated proposition')
    elif change == 'duplicate':
        template += ' {genes}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{statement}', '{statement!r}')
    else:
        template = template.replace('{statement}', '{statement:>20}')
    requests['train']['graph_truth'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_13_requests(requests)


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_13_requests(pin)


@pytest.mark.parametrize('args', [[], ['--pairs-per-stratum', '1', '0', '1'],
    ['--smoke', '--pairs-per-stratum', '1', '1', '1']])
def test_cli_requires_one_explicit_valid_budget(monkeypatch, tmp_path, args):
    fake = SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_13_corpus', fake)
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_13.py', '--output-dir', str(tmp_path), *args])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_13.py'), run_name='__main__')
    assert error.value.code == 2


@pytest.mark.parametrize('args,expected', [(['--smoke'], {'counts': None, 'smoke': True}),
    (['--pairs-per-stratum', '3', '2', '1'], {'counts': {'train': 3, 'validation': 2, 'test': 1}, 'smoke': False})])
def test_cli_passes_only_the_explicit_pair_budget(monkeypatch, tmp_path, args, expected):
    calls = []
    def build(output, **kwargs):
        calls.append((output, kwargs))
        return {'purpose': 'builder_verification', 'generation_budget': kwargs['counts'], 'row_counts': {},
                'panel_id': 'panel', 'training_ready': False, 'production_export_allowed': False}
    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_13_corpus',
                        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_13.py', '--output-dir', str(tmp_path), *args])
    runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_13.py'), run_name='__main__')
    assert calls == [(tmp_path, {**expected, 'seed': 20261028})]
