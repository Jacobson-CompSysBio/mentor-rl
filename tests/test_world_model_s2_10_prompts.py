"""Check literal component requests, directed kinds, gates, and frozen wording."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_10_prompts import (
    DIAGNOSTICS, FIELDS, KINDS, load_s2_10_requests, render_s2_10_diagnostic_request,
    render_s2_10_request, validate_s2_10_rendered_prompts, validate_s2_10_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_10_v1_requests.json'
VALUES = {'multiplex_id': 'MultiplexExample', 'layer_id': 'LayerExample',
          'genes': [' raw symbol ', 'ENSG00000141510', ' raw symbol '],
          'connectivity_kind': 'weak', 'require_all_in_layer': False}


@pytest.fixture
def requests():
    """Give each mutation test its own component request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the unchanged wrappers and all nine earlier request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json', 's2_5_v1_requests.json',
        's2_6_v1_requests.json', 's2_7_v1_requests.json', 's2_8_v1_requests.json',
        's2_9_v1_requests.json',
    ))


def test_overlay_preserves_all_prior_pins_and_unchanged_native_tools(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_10_v1.json').read_text())
    previous = json.loads((REQUEST_PATH.parent / 's2_9_v1.json').read_text())
    assert load_s2_10_requests(contract['request_bank']) == requests
    report = validate_s2_10_requests(requests)
    assert all(list(kinds) == list(KINDS) and set(kinds.values()) == {4}
               for kinds in report['template_counts'].values())
    assert all(list(kinds) == list(DIAGNOSTICS) and set(kinds.values()) == {4}
               for kinds in report['diagnostic_template_counts'].values())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank',
                's25_request_bank', 's26_request_bank', 's27_request_bank', 's28_request_bank', 'scoring'):
        assert contract[key] == previous[key]
    assert contract['s29_request_bank'] == previous['request_bank']
    assert contract['context_source_contract'] == {
        'path': 'runtime/world_model/s2_9_v1.json',
        'sha256': file_sha256(REQUEST_PATH.parent / 's2_9_v1.json'),
    }
    assert contract['implemented_families'] == ['S2.10']
    assert contract['training_ready'] is contract['production_export_allowed'] is False
    assert contract['tools'] == previous['tools']
    tool = contract['tools'][0]['function']
    assert tool['name'] == 'induce_subgraph'
    assert tool['parameters']['required'] == ['multiplex_id', 'layer_id', 'gene_ids', 'mode', 'top_k', 'cursor']
    assert 'connectivity_kind' not in tool['parameters']['properties']
    assert all(isinstance(tool['function'].get('description'), str)
               and tool['function']['description'].strip() for tool in contract['tools'])
    for marker in ('MEMBERSHIP_PRECONDITION_UNSATISFIED', 'CONNECTIVITY_KIND_REQUIRED',
                   'INVALID_CONNECTIVITY_KIND', 'CONNECTIVITY_KIND_UNAVAILABLE',
                   'UNRESOLVED_IDENTITY', 'zero-length path', 'recompute', 'single_component:false'):
        assert marker in contract['system_prompt']


def test_all_ten_families_have_disjoint_full_questions(requests, prior_banks):
    report = validate_s2_10_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_10_rendered_questions'] == {'train': 2816, 'validation': 704, 'test': 704}
    assert report['cross_split_pairs_checked'] > 235929600
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_core_cannot_enter_training(requests, prior_banks):
    requests['train']['partition'][0] = requests['validation']['partition'][0]
    validate_s2_10_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_10_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('question_kind', KINDS)
@pytest.mark.parametrize('connectivity_kind', [None, 'undirected', 'weak', 'strong'])
@pytest.mark.parametrize('gate', [False, True])
def test_literal_fields_survive_every_question_kind(requests, split, question_kind, connectivity_kind, gate):
    values = {**deepcopy(VALUES), 'connectivity_kind': connectivity_kind, 'require_all_in_layer': gate}
    original = deepcopy(values)
    rendered = render_s2_10_request(requests, split=split, variant_index=5,
                                   question_kind=question_kind, **values)
    assert rendered == requests[split][question_kind][1].format(
        **{key: json.dumps(value) for key, value in values.items()})
    assert '[" raw symbol ", "ENSG00000141510", " raw symbol "]' in rendered
    assert values == original


def test_explicit_empty_list_differs_from_missing_input_and_kind_remains_null(requests):
    ordinary = render_s2_10_request(requests, split='train', variant_index=0,
                                    **{**VALUES, 'genes': [], 'connectivity_kind': None})
    missing = render_s2_10_diagnostic_request(requests, split='train', kind='missing_genes',
        variant_index=0, **{**VALUES, 'genes': None, 'connectivity_kind': None})
    assert '[]' in ordinary and 'null' in ordinary
    assert '[]' not in missing and missing.count('null') == 2


@pytest.mark.parametrize('genes', [None, 'TP53', [None], [True], ['']])
def test_nonlist_or_nonstring_inputs_require_a_diagnostic(requests, genes):
    with pytest.raises(PromptBankError, match='literal gene list'):
        render_s2_10_request(requests, split='train', variant_index=0, **{**VALUES, 'genes': genes})


@pytest.mark.parametrize('kind', ['', 'directed', 'Weak', 1, True, [], {}])
def test_ordinary_connectivity_is_recognized_without_coercion(requests, kind):
    with pytest.raises(PromptBankError, match='recognized connectivity kind'):
        render_s2_10_request(requests, split='train', variant_index=0,
                             **{**VALUES, 'connectivity_kind': kind})


@pytest.mark.parametrize('gate', [None, 0, 1, 'true', []])
def test_ordinary_gate_is_an_explicit_boolean(requests, gate):
    with pytest.raises(PromptBankError, match='explicit booleans'):
        render_s2_10_request(requests, split='train', variant_index=0,
                             **{**VALUES, 'require_all_in_layer': gate})


@pytest.mark.parametrize('kind', [*DIAGNOSTICS, 'tool_unknown_layer'])
def test_diagnostics_preserve_all_original_json(requests, kind):
    values = {'multiplex_id': None, 'layer_id': {'raw': 'unknown'},
              'genes': [' raw ', False, None], 'connectivity_kind': {'raw': 'weak'},
              'require_all_in_layer': 'true'}
    rendered = render_s2_10_diagnostic_request(requests, split='validation', kind=kind,
                                               variant_index=1, **values)
    selected = 'unknown_layer' if kind == 'tool_unknown_layer' else kind
    assert rendered == requests['diagnostic_requests']['validation'][selected][1].format(
        **{key: json.dumps(value) for key, value in values.items()})


@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
@pytest.mark.parametrize('field', FIELDS)
def test_renderers_require_actual_json_values(requests, value, field):
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_10_diagnostic_request(requests, split='test', kind='invalid_list',
            variant_index=0, **{**VALUES, field: value})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_exact_five_template_bindings_are_required(requests, change):
    template = requests['train']['partition'][0]
    if change == 'missing':
        template = template.replace('{genes}', 'these genes')
    elif change == 'duplicate':
        template += ' {genes}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{genes}', '{genes!r}')
    else:
        template = template.replace('{genes}', '{genes:>20}')
    requests['train']['partition'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_10_requests(requests)


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_10_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate CLI allocation checks from source reads and output writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return dict.fromkeys(('purpose', 'generation_budget', 'row_counts', 'panel_id',
                              'training_ready', 'production_export_allowed'))

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_10_corpus',
        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_10.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--sets-per-stratum', '0', '1', '1'],
    ['--sets-per-stratum', '1', '-1', '1'], ['--smoke', '--sets-per-stratum', '1', '1', '1'],
    ['--smoke', '--input-sizes', '-1']])
def test_cli_rejects_missing_or_conflicting_budgets(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_10.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_explicit_sizes_quotas_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_10.py', '--output-dir', str(tmp_path),
        '--sets-per-stratum', '2', '3', '4', '--input-sizes', '0', '3', '8', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'input_sizes': [0, 3, 8], 'seed': 7})]


def test_cli_smoke_has_no_implicit_production_budget(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_10.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls[0][1] == {'counts': None, 'smoke': True, 'input_sizes': None, 'seed': 20261028}
