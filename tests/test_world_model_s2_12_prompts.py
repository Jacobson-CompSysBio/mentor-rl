"""Check literal singleton inputs, explicit truth scopes, and frozen full questions."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_12_prompts import (
    DIAGNOSTICS, FIELDS, KINDS, TASK_FIELDS, load_s2_12_requests,
    render_s2_12_diagnostic_request, render_s2_12_request,
    validate_s2_12_rendered_prompts, validate_s2_12_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_12_v1_requests.json'
VALUES = {'multiplex_id': 'MultiplexExample', 'layer_id': 'LayerExample',
          'genes': [' raw symbol '], 'claim': 'gene_not_node_in_layer',
          'truth_scope': 'category', 'truth_layer_ids': None, 'truth_family': None,
          'statement': 'The literal proposition includes "quoted" text and {braces}.',
          'require_membership': None}


@pytest.fixture
def requests():
    """Give each mutation test an independent request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the frozen wrappers and all eleven earlier request banks."""
    names = ('s2_v1_prompt_banks.json', 's2_v1_edge_requests.json',
             *(f's2_{number}_v1_requests.json' for number in range(2, 12)))
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in names)


def test_overlay_pins_both_views_and_keeps_all_approved_tools(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_12_v1.json').read_text())
    previous = json.loads((REQUEST_PATH.parent / 's2_11_v1.json').read_text())
    assert load_s2_12_requests(contract['request_bank']) == requests
    report = validate_s2_12_requests(requests)
    assert all(set(kinds.values()) == {4} for kinds in report['template_counts'].values())
    assert all(set(kinds.values()) == {4} for kinds in report['diagnostic_template_counts'].values())
    for key in ('registry_index', 'prompt_bank', 'tokenizer', 'context_recipe', 'scoring',
                *(f's2{number}_request_bank' for number in range(1, 11))):
        assert contract[key] == previous[key]
    assert contract['previous_source_manifest'] == previous['source_manifest']
    assert contract['source_manifest']['path'].endswith('s2_12_v1_sources.json')
    assert contract['s211_request_bank'] == previous['request_bank']
    assert contract['implemented_families'] == ['S2.12']
    assert contract['training_ready'] is contract['production_export_allowed'] is False
    tools = {item['function']['name']: item['function'] for item in contract['tools']}
    assert set(tools) == {'lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id',
        'check_multiplex_membership', 'get_gene_layers', 'get_nodes_by_layer', 'get_edge',
        'get_neighbors', 'induce_subgraph', 'shortest_path'}
    membership = tools['get_nodes_by_layer']['parameters']
    assert membership['required'] == ['multiplex_id', 'layer_id', 'gene_ids']
    assert set(membership['properties']) == {'multiplex_id', 'layer_id', 'gene_ids'}
    assert all(tool.get('description', '').strip() for tool in tools.values())
    assert len(contract['claim_vocabulary']) == 11
    for marker in ('UNSUPPORTED_TRUTH_SCOPE', 'MEMBERSHIP_PRECONDITION_UNSATISFIED',
                   'INVALID_GENE_LIST', 'UNRESOLVED_IDENTITY', 'Unsupported does not mean false',
                   'old four-file negative', 'null symbol', 'no direction selector'):
        assert marker in contract['system_prompt']


def test_all_twelve_families_have_disjoint_full_questions(requests, prior_banks):
    report = validate_s2_12_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_12_rendered_questions'] == {'train': 3072, 'validation': 768, 'test': 768}
    assert report['cross_split_pairs_checked'] == 522031104
    assert report['maximum_cross_split_jaccard'] < .6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_full_question_cannot_enter_training(requests, prior_banks):
    requests['train']['membership_card'][0] = requests['validation']['membership_card'][0]
    validate_s2_12_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_12_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('question_kind', KINDS)
def test_every_task_preserves_literal_subject_and_proposition(requests, split, question_kind):
    values = deepcopy(VALUES)
    original = deepcopy(values)
    rendered = render_s2_12_request(requests, split=split, variant_index=5,
        question_kind=question_kind, **values)
    expected = requests[split][question_kind][1].format(
        **{key: json.dumps(values[key]) for key in TASK_FIELDS[question_kind]})
    assert rendered == expected and values == original


@pytest.mark.parametrize('genes', [None, 'TP53', [], ['A', 'B'], [None], [True], ['']])
def test_malformed_singletons_require_a_diagnostic(requests, genes):
    with pytest.raises(PromptBankError, match='literal singleton list'):
        render_s2_12_request(requests, split='train', variant_index=0, **{**VALUES, 'genes': genes})


@pytest.mark.parametrize('kind', [*DIAGNOSTICS, 'tool_unknown_layer', 'tool_unknown_multiplex'])
def test_diagnostics_preserve_all_three_raw_json_values(requests, kind):
    values = {'multiplex_id': None, 'layer_id': {'raw': 'unknown'}, 'genes': [' raw ', False, None]}
    rendered = render_s2_12_diagnostic_request(requests, split='validation', kind=kind,
        variant_index=1, **values)
    selected = kind.removeprefix('tool_')
    assert rendered == requests['diagnostic_requests']['validation'][selected][1].format(
        **{key: json.dumps(value) for key, value in values.items()})


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A',), {3: 'A'}])
def test_diagnostics_reject_non_json_values(requests, field, value):
    values = {key: VALUES[key] for key in FIELDS}
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_12_diagnostic_request(requests, split='test', kind='invalid_gene_list',
            variant_index=0, **{**values, field: value})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_templates_bind_each_declared_field_once(requests, change):
    template = requests['train']['membership_truth'][0]
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
    requests['train']['membership_truth'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_12_requests(requests)


@pytest.mark.parametrize('scope,layers,family', [('layers', ['A', 'B'], None), ('family', None, 'scpen'),
    ('category', None, None), ('multiplex', None, None), ('all_layers', None, None),
    ('future_versions', None, None), ('expression', None, None)])
def test_truth_preserves_exact_scope_selectors(requests, scope, layers, family):
    rendered = render_s2_12_request(requests, split='train', variant_index=0, question_kind='truth',
        **{**VALUES, 'truth_scope': scope, 'truth_layer_ids': layers, 'truth_family': family})
    assert json.dumps(VALUES['statement']) in rendered
    assert json.dumps(scope) in rendered


@pytest.mark.parametrize('change', [{'statement': None}, {'truth_scope': None}, {'truth_scope': 'other_layer'},
    {'truth_scope': 'layers', 'truth_layer_ids': []}, {'truth_scope': 'family', 'truth_family': None},
    {'truth_layer_ids': ['A']}, {'truth_family': 'scpen'}])
def test_truth_cannot_hide_or_misplace_selectors(requests, change):
    with pytest.raises(PromptBankError):
        render_s2_12_request(requests, split='test', variant_index=0, question_kind='truth',
            **{**VALUES, **change})


@pytest.mark.parametrize('gate', ['multiplex', 'layer'])
def test_only_classification_forms_can_require_a_membership_prerequisite(requests, gate):
    for kind in KINDS:
        values = {**VALUES, 'require_membership': gate}
        if kind.startswith('membership_'):
            assert json.dumps(gate) in render_s2_12_request(requests, split='train', variant_index=0,
                question_kind=kind, **values)
        else:
            with pytest.raises(PromptBankError, match='claim-only'):
                render_s2_12_request(requests, split='train', variant_index=0, question_kind=kind, **values)


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_12_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Check CLI budgets without source reads or output writes."""
    calls = []
    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return dict.fromkeys(('purpose', 'generation_budget', 'row_counts', 'panel_id',
                              'training_ready', 'production_export_allowed'))
    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_12_corpus',
        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_12.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--genes-per-stratum', '0', '1', '1'],
    ['--genes-per-stratum', '1', '-1', '1'], ['--smoke', '--genes-per-stratum', '1', '1', '1']])
def test_cli_requires_one_explicit_positive_budget(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_12.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_explicit_gene_quotas_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_12.py', '--output-dir', str(tmp_path),
        '--genes-per-stratum', '2', '3', '4', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4}, 'smoke': False, 'seed': 7})]


def test_cli_smoke_has_no_implicit_production_budget(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_12.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls[0][1] == {'counts': None, 'smoke': True, 'seed': 20261028}
