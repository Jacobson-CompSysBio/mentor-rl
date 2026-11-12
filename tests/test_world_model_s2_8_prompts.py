"""Check literal degree inputs, policy selection, and frozen wording splits."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_8_prompts import (
    DIAGNOSTICS, FIELDS, KINDS, load_s2_8_requests, render_s2_8_diagnostic_request,
    render_s2_8_request, validate_s2_8_rendered_prompts, validate_s2_8_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_8_v1_requests.json'
VALUES = {'multiplex_id': 'MultiplexExample', 'gene': ' raw symbol ',
          'layer_scope': {'kind': 'layers', 'layer_ids': ['L2', 'L1'], 'layer_family': None},
          'direction': 'incident', 'request_kind': 'degree_and_hub',
          'hub_policy': {'policy_id': 'degree_midrank_v1', 'threshold_percentile': 95}}


@pytest.fixture
def requests():
    """Give each mutation test an independent degree bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read unchanged wrappers and all seven earlier S2 request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json', 's2_5_v1_requests.json',
        's2_6_v1_requests.json', 's2_7_v1_requests.json',
    ))


def test_overlay_declares_the_exact_degree_tool_and_preserves_prior_pins(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_8_v1.json').read_text())
    previous = json.loads((REQUEST_PATH.parent / 's2_7_v1.json').read_text())
    assert load_s2_8_requests(contract['request_bank']) == requests
    report = validate_s2_8_requests(requests)
    assert all(list(kinds) == list(KINDS) and set(kinds.values()) == {4}
               for kinds in report['template_counts'].values())
    assert all(list(kinds) == list(DIAGNOSTICS) and set(kinds.values()) == {4}
               for kinds in report['diagnostic_template_counts'].values())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank',
                's25_request_bank', 's26_request_bank', 'scoring'):
        assert contract[key] == previous[key]
    assert contract['s27_request_bank'] == previous['request_bank']
    assert contract['implemented_families'] == ['S2.8'] and contract['training_ready'] is False
    tool = contract['tools'][0]['function']
    assert tool['name'] == 'get_degree'
    assert tool['parameters']['required'] == [
        'multiplex_id', 'layer_scope', 'gene_id', 'direction', 'request_kind', 'hub_policy']
    assert tool['parameters']['additionalProperties'] is False
    assert contract['degree_policy_id'] == 'distinct_neighbor_union_v1'
    assert contract['population_policy_id'] == 'selected_layer_node_union_v1'
    assert contract['default_hub_policy'] == {'policy_id': 'degree_midrank_v1', 'threshold_percentile': 95}
    assert contract['percentile_display'] == {'decimal_places': 6, 'rounding': 'ROUND_HALF_UP'}
    assert 'OUTSIDE_REFERENCE_POPULATION' in contract['system_prompt']
    assert 'EMPTY_REFERENCE_POPULATION' in contract['system_prompt']


def test_full_questions_are_disjoint_across_all_eight_families(requests, prior_banks):
    report = validate_s2_8_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_8_rendered_questions'] == {'train': 2304, 'validation': 576, 'test': 576}
    assert report['cross_split_pairs_checked'] > 132710400
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_core_cannot_enter_training(requests, prior_banks):
    requests['train']['degree_only'][0] = requests['validation']['degree_only'][0]
    validate_s2_8_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_8_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('kind', KINDS)
def test_both_statistic_kinds_preserve_every_literal_input(requests, split, kind):
    values = deepcopy(VALUES)
    values['request_kind'] = kind
    if kind == 'degree_only':
        values['hub_policy'] = None
    original = deepcopy(values)
    text = render_s2_8_request(requests, split=split, variant_index=5, **values)
    assert text == requests[split][kind][1].format(**{key: json.dumps(value) for key, value in values.items()})
    assert '" raw symbol "' in text
    assert '"layer_ids": ["L2", "L1"]' in text
    assert values == original


@pytest.mark.parametrize('direction', ['in', 'out', 'incident', 'undirected'])
def test_direction_does_not_choose_the_statistic(requests, direction):
    values = {**VALUES, 'direction': direction}
    text = render_s2_8_request(requests, split='train', variant_index=0, **values)
    assert json.dumps(direction) in text and '"degree_and_hub"' in text


@pytest.mark.parametrize('threshold', [False, 95.0, 95.5, '95', -1, 101, None])
def test_hub_threshold_is_a_bounded_integer_without_coercion(requests, threshold):
    values = {**VALUES, 'hub_policy': {'policy_id': 'degree_midrank_v1', 'threshold_percentile': threshold}}
    with pytest.raises(PromptBankError, match='declared hub policy'):
        render_s2_8_request(requests, split='train', variant_index=0, **values)


@pytest.mark.parametrize('policy', [None, {}, {'policy_id': 'other', 'threshold_percentile': 95},
    {'policy_id': 'degree_midrank_v1', 'threshold_percentile': 95, 'extra': True}])
def test_hub_requests_require_the_declared_policy(requests, policy):
    with pytest.raises(PromptBankError, match='declared hub policy'):
        render_s2_8_request(requests, split='train', variant_index=0, **{**VALUES, 'hub_policy': policy})


def test_count_only_request_cannot_carry_a_hub_policy(requests):
    with pytest.raises(PromptBankError, match='declared hub policy'):
        render_s2_8_request(requests, split='train', variant_index=0, **{**VALUES, 'request_kind': 'degree_only'})


@pytest.mark.parametrize('scope', [None, 'all', {}, {'kind': 'layers', 'layer_ids': [], 'layer_family': None},
    {'kind': 'all', 'layer_ids': ['L'], 'layer_family': None},
    {'kind': 'layers', 'layer_ids': ['L', 'L'], 'layer_family': None}])
def test_invalid_scope_shapes_require_a_diagnostic(requests, scope):
    with pytest.raises(PromptBankError, match='typed layer scope'):
        render_s2_8_request(requests, split='train', variant_index=0, **{**VALUES, 'layer_scope': scope})


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_diagnostics_keep_missing_and_malformed_literal_values(requests, kind):
    values = {'multiplex_id': None, 'gene': [' raw ', False, None],
              'layer_scope': {'kind': 'all', 'layer_ids': ['L', 'L']}, 'direction': {'raw': 'sideways'},
              'request_kind': 3, 'hub_policy': {'threshold_percentile': '95'}}
    text = render_s2_8_diagnostic_request(requests, split='validation', kind=kind, variant_index=1, **values)
    assert text == requests['diagnostic_requests']['validation'][kind][1].format(
        **{key: json.dumps(value) for key, value in values.items()})


@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
@pytest.mark.parametrize('field', FIELDS)
def test_renderers_require_actual_json_values(requests, value, field):
    values = {**VALUES, field: value}
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_8_diagnostic_request(requests, split='test', kind='invalid_scope', variant_index=0, **values)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_template_bindings_cannot_change(requests, change):
    template = requests['train']['degree_only'][0]
    if change == 'missing':
        template = template.replace('{gene}', 'this gene')
    elif change == 'duplicate':
        template += ' {gene}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{gene}', '{gene!r}')
    else:
        template = template.replace('{gene}', '{gene:>20}')
    requests['train']['degree_only'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_8_requests(requests)


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_8_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate CLI budget checks from source reads and artifact writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {key: None for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                      'training_ready', 'production_export_allowed')}

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_8_corpus',
                        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_8.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--genes-per-stratum', '0', '1', '1'],
    ['--genes-per-stratum', '1', '-1', '1'], ['--smoke', '--genes-per-stratum', '1', '1', '1']])
def test_cli_rejects_missing_or_conflicting_budgets(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_8.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_separate_split_budgets_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_8.py', '--output-dir', str(tmp_path),
        '--genes-per-stratum', '2', '3', '4', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'seed': 7})]


def test_cli_selects_the_verification_schedule_only_in_smoke_mode(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_8.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls == [(tmp_path, {'counts': None, 'smoke': True, 'seed': 20261028})]
