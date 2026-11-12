"""Check exact path positions, scoped claim requests, and frozen full questions."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_11_prompts import (
    DIAGNOSTICS, FIELDS, KINDS, TASK_FIELDS, load_s2_11_requests,
    render_s2_11_diagnostic_request, render_s2_11_request,
    validate_s2_11_rendered_prompts, validate_s2_11_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_11_v1_requests.json'
VALUES = {'multiplex_id': 'MultiplexExample', 'layer_id': 'LayerExample',
          'genes': [' raw symbol ', 'ENSG00000141510'], 'direction': 'source_to_target',
          'claim': 'no_path_recorded_in_scope', 'truth_scope': 'scope',
          'statement': 'The literal proposition includes "quoted" text and {braces}.',
          'membership_cards': False}


@pytest.fixture
def requests():
    """Give each mutation test a separate request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the frozen wrappers and all ten earlier request banks."""
    names = ('s2_v1_prompt_banks.json', 's2_v1_edge_requests.json',
             *(f's2_{number}_v1_requests.json' for number in range(2, 11)))
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in names)


def test_overlay_preserves_source_pins_and_adds_only_native_path_operation(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_11_v1.json').read_text())
    previous = json.loads((REQUEST_PATH.parent / 's2_10_v1.json').read_text())
    assert load_s2_11_requests(contract['request_bank']) == requests
    report = validate_s2_11_requests(requests)
    assert all(list(kinds) == list(KINDS) and set(kinds.values()) == {4}
               for kinds in report['template_counts'].values())
    assert all(list(kinds) == list(DIAGNOSTICS) and set(kinds.values()) == {4}
               for kinds in report['diagnostic_template_counts'].values())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank',
                's25_request_bank', 's26_request_bank', 's27_request_bank', 's28_request_bank',
                's29_request_bank', 'scoring'):
        assert contract[key] == previous[key]
    assert contract['s210_request_bank'] == previous['request_bank']
    assert 'context_source_contract' not in contract
    assert contract['implemented_families'] == ['S2.11']
    assert contract['training_ready'] is contract['production_export_allowed'] is False
    tools = {item['function']['name']: item['function'] for item in contract['tools']}
    assert set(tools) == {'lookup_human_gene_symbol', 'lookup_human_ensembl_gene_id',
        'check_multiplex_membership', 'get_edge', 'get_neighbors', 'get_nodes_by_layer',
        'induce_subgraph', 'shortest_path'}
    path = tools['shortest_path']['parameters']
    assert path['required'] == ['multiplex_id', 'layer_id', 'gene_ids', 'direction']
    assert path['properties']['gene_ids']['minItems'] == path['properties']['gene_ids']['maxItems'] == 2
    assert path['properties']['direction']['enum'] == ['undirected', 'source_to_target']
    assert all(tool.get('description', '').strip() for tool in tools.values())
    assert all('hop_limit' not in tool['parameters']['properties'] for tool in tools.values())
    for marker in ('UNSUPPORTED_TRUTH_SCOPE', 'AMBIGUOUS_PATH_DIRECTION', 'INVALID_DIRECTION',
                   'DIRECTION_UNAVAILABLE', 'UNRESOLVED_IDENTITY', 'zero-hop',
                   'lexicographically smallest complete ID sequence', 'Unsupported does not mean false'):
        assert marker in contract['system_prompt']
    assert len(contract['claim_vocabulary']) == 11
    assert contract['claim_vocabulary'][:4] == ['no_edge_recorded_in_scope', 'no_path_recorded_in_scope',
        'gene_not_node_in_multiplex', 'gene_not_node_in_layer']


def test_all_eleven_families_have_disjoint_full_questions(requests, prior_banks):
    report = validate_s2_11_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_11_rendered_questions'] == {'train': 4096, 'validation': 1024, 'test': 1024}
    assert report['cross_split_pairs_checked'] == 422055936
    assert report['maximum_cross_split_jaccard'] < .6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_full_question_cannot_enter_training(requests, prior_banks):
    requests['train']['path'][0] = requests['validation']['path'][0]
    validate_s2_11_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_11_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('question_kind', KINDS)
def test_all_tasks_preserve_literals_and_order_without_deduplication(requests, split, question_kind):
    for direction in (None, 'undirected', 'source_to_target'):
        for membership in (False, True):
            values = {**deepcopy(VALUES), 'direction': direction, 'membership_cards': membership}
            original = deepcopy(values)
            rendered = render_s2_11_request(requests, split=split, variant_index=5,
                question_kind=question_kind, **values)
            expected = requests[split][question_kind][1].format(
                **{key: json.dumps(values[key]) for key in TASK_FIELDS[question_kind]})
            assert rendered == expected and values == original
            repeated = render_s2_11_request(requests, split=split, variant_index=0,
                question_kind=question_kind, **{**values, 'genes': [' raw symbol ', ' raw symbol ']})
            assert '[" raw symbol ", " raw symbol "]' in repeated


@pytest.mark.parametrize('genes', [None, 'TP53', [], ['TP53'], ['A', 'B', 'C'], [None, 'A'], [True, 'A'], ['', 'A']])
def test_malformed_pairs_require_a_diagnostic(requests, genes):
    with pytest.raises(PromptBankError, match='literal ordered endpoint pair'):
        render_s2_11_request(requests, split='train', variant_index=0, **{**VALUES, 'genes': genes})


@pytest.mark.parametrize('direction', ['', 'weak', 'strong', 'either', 1, True, [], {}])
def test_ordinary_direction_cannot_add_a_projection_or_coerce_a_value(requests, direction):
    with pytest.raises(PromptBankError, match='recognized direction'):
        render_s2_11_request(requests, split='train', variant_index=0, **{**VALUES, 'direction': direction})


@pytest.mark.parametrize('kind', [*DIAGNOSTICS, 'tool_unknown_layer'])
def test_diagnostics_preserve_all_four_original_json_values(requests, kind):
    values = {'multiplex_id': None, 'layer_id': {'raw': 'unknown'},
              'genes': [' raw ', False, None], 'direction': {'raw': 'undirected'}}
    rendered = render_s2_11_diagnostic_request(requests, split='validation', kind=kind,
        variant_index=1, **values)
    selected = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    assert rendered == requests['diagnostic_requests']['validation'][selected][1].format(
        **{key: json.dumps(value) for key, value in values.items()})


@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
def test_raw_diagnostics_reject_values_that_json_cannot_represent(requests, field, value):
    values = {key: VALUES[key] for key in FIELDS}
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_11_diagnostic_request(requests, split='test', kind='invalid_pair',
            variant_index=0, **{**values, field: value})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_templates_bind_each_declared_field_once(requests, change):
    template = requests['train']['path_truth'][0]
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
    requests['train']['path_truth'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_11_requests(requests)


@pytest.mark.parametrize('kind', ['path_support', 'path_truth', 'claim_support', 'truth'])
def test_assessed_cards_require_the_requested_claim(requests, kind):
    with pytest.raises(PromptBankError, match='explicit claim'):
        render_s2_11_request(requests, split='train', variant_index=0,
            question_kind=kind, **{**VALUES, 'claim': None})


@pytest.mark.parametrize('change', [{'statement': None}, {'statement': ''}, {'truth_scope': None}, {'truth_scope': 'other_layer'}])
def test_truth_needs_an_exact_proposition_and_its_scope(requests, change):
    with pytest.raises(PromptBankError, match='exact statement and scope'):
        render_s2_11_request(requests, split='train', variant_index=0,
            question_kind='truth', **{**VALUES, **change})


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_11_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate CLI budget checks from source reads and artifact writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return dict.fromkeys(('purpose', 'generation_budget', 'row_counts', 'panel_id',
                              'training_ready', 'production_export_allowed'))

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_11_corpus',
        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_11.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--pairs-per-stratum', '0', '1', '1'],
    ['--pairs-per-stratum', '1', '-1', '1'], ['--smoke', '--pairs-per-stratum', '1', '1', '1'],
    ['--smoke', '--source-window', '0'], ['--smoke', '--source-window', '-1']])
def test_cli_rejects_missing_or_conflicting_budgets(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_11.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_explicit_subject_budget_quotas_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_11.py', '--output-dir', str(tmp_path),
        '--pairs-per-stratum', '2', '3', '4', '--source-window', '16', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'source_window': 16, 'seed': 7})]


def test_cli_smoke_selects_no_implicit_production_budget(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_11.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls[0][1] == {'counts': None, 'smoke': True, 'source_window': None, 'seed': 20261028}
