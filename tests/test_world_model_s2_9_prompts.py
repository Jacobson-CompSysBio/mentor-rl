"""Check literal subgraph inputs, edge previews, gates, and frozen wording."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT, load_contract, tool_definitions
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_9_prompts import (
    DIAGNOSTICS, FIELDS, KINDS, load_s2_9_requests, render_s2_9_diagnostic_request,
    render_s2_9_request, validate_s2_9_rendered_prompts, validate_s2_9_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_9_v1_requests.json'
VALUES = {'multiplex_id': 'MultiplexExample', 'layer_id': 'LayerExample',
          'genes': [' raw symbol ', 'ENSG00000141510', ' raw symbol '],
          'mode': 'complete', 'top_k': None, 'cursor': None, 'require_all_in_layer': False}


@pytest.fixture
def requests():
    """Give each mutation test an independent subgraph request bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the unchanged wrappers and eight earlier request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json', 's2_5_v1_requests.json',
        's2_6_v1_requests.json', 's2_7_v1_requests.json', 's2_8_v1_requests.json',
    ))


def test_overlay_uses_exact_subgraph_arguments_and_preserves_prior_pins(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_9_v1.json').read_text())
    previous = json.loads((REQUEST_PATH.parent / 's2_8_v1.json').read_text())
    assert load_s2_9_requests(contract['request_bank']) == requests
    report = validate_s2_9_requests(requests)
    assert all(list(kinds) == list(KINDS) and set(kinds.values()) == {4}
               for kinds in report['template_counts'].values())
    assert all(list(kinds) == list(DIAGNOSTICS) and set(kinds.values()) == {4}
               for kinds in report['diagnostic_template_counts'].values())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank',
                's25_request_bank', 's26_request_bank', 's27_request_bank', 'scoring'):
        assert contract[key] == previous[key]
    assert contract['s28_request_bank'] == previous['request_bank']
    assert contract['implemented_families'] == ['S2.9'] and contract['training_ready'] is False
    tool = contract['tools'][0]['function']
    assert tool['name'] == 'induce_subgraph'
    assert tool['parameters']['required'] == ['multiplex_id', 'layer_id', 'gene_ids', 'mode', 'top_k', 'cursor']
    assert tool['parameters']['additionalProperties'] is False
    assert 'require_all_in_layer' not in tool['parameters']['properties']
    memberships = json.loads((REQUEST_PATH.parent / 's2_6_v1.json').read_text())
    assert contract['tools'][1] == memberships['tools'][0]
    original_lookup = next(tool['function'] for tool in tool_definitions(load_contract())
                           if tool['function']['name'] == 'lookup_human_gene_symbol')
    lookup = contract['tools'][2]['function']
    assert {key: lookup[key] for key in ('name', 'parameters')} == original_lookup
    assert all(isinstance(tool['function'].get('description'), str)
               and tool['function']['description'].strip() for tool in contract['tools'])
    for marker in ('MEMBERSHIP_PRECONDITION_UNSATISFIED', 'SUBGRAPH_RANKING_REQUIRED',
                   'UNRESOLVED_IDENTITY', 'initial null cursor', 'cross-batch'):
        assert marker in contract['system_prompt']


def test_all_nine_families_have_disjoint_full_questions(requests, prior_banks):
    report = validate_s2_9_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_9_rendered_questions'] == {'train': 2816, 'validation': 704, 'test': 704}
    assert report['cross_split_pairs_checked'] > 175509504
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_core_cannot_enter_training(requests, prior_banks):
    requests['train']['complete'][0] = requests['validation']['complete'][0]
    validate_s2_9_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_9_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('gate', [False, True])
def test_selection_and_gate_preserve_raw_gene_order_and_duplicates(requests, split, kind, gate):
    values = {**deepcopy(VALUES), 'mode': kind, 'top_k': 2 if kind == 'top_k' else None,
              'require_all_in_layer': gate}
    original = deepcopy(values)
    rendered = render_s2_9_request(requests, split=split, variant_index=5, **values)
    assert rendered == requests[split][kind][1].format(**{key: json.dumps(value) for key, value in values.items()})
    assert '[" raw symbol ", "ENSG00000141510", " raw symbol "]' in rendered
    assert values == original


def test_explicit_empty_list_remains_distinct_from_missing_input(requests):
    complete = render_s2_9_request(requests, split='train', variant_index=0, **{**VALUES, 'genes': []})
    missing = render_s2_9_diagnostic_request(requests, split='train', kind='missing_genes',
        variant_index=0, **{**VALUES, 'genes': None})
    assert '[]' in complete and 'null' in missing
    assert complete != missing


def test_ordinary_cursor_is_preserved_without_rebinding_the_selection(requests):
    rendered = render_s2_9_request(requests, split='test', variant_index=2,
        **{**VALUES, 'cursor': 'opaque_selection_cursor'})
    assert '"opaque_selection_cursor"' in rendered


@pytest.mark.parametrize('selection', [
    {'mode': 'complete', 'top_k': 1}, {'mode': 'top_k', 'top_k': None},
    {'mode': 'top_k', 'top_k': 0}, {'mode': 'top_k', 'top_k': -1},
    {'mode': 'top_k', 'top_k': True}, {'mode': 'top_k', 'top_k': 1.0},
    {'mode': 'top_k', 'top_k': '1'}, {'mode': 'strongest', 'top_k': 1},
])
def test_invalid_previews_require_a_diagnostic(requests, selection):
    with pytest.raises(PromptBankError, match='positive canonical prefix'):
        render_s2_9_request(requests, split='train', variant_index=0, **{**VALUES, **selection})


@pytest.mark.parametrize('genes', [None, 'TP53', [None], [True], ['']])
def test_nonlist_or_nonstring_inputs_require_a_diagnostic(requests, genes):
    with pytest.raises(PromptBankError, match='literal gene list'):
        render_s2_9_request(requests, split='train', variant_index=0, **{**VALUES, 'genes': genes})


@pytest.mark.parametrize('gate', [None, 0, 1, 'true', []])
def test_ordinary_membership_requirement_is_not_coerced(requests, gate):
    with pytest.raises(PromptBankError, match='explicit booleans'):
        render_s2_9_request(requests, split='train', variant_index=0, **{**VALUES, 'require_all_in_layer': gate})


@pytest.mark.parametrize('cursor', ['', False, 1, []])
def test_ordinary_cursor_requires_null_or_a_nonempty_string(requests, cursor):
    with pytest.raises(PromptBankError, match='nonempty cursor'):
        render_s2_9_request(requests, split='train', variant_index=0, **{**VALUES, 'cursor': cursor})


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_diagnostics_preserve_missing_and_malformed_json(requests, kind):
    values = {'multiplex_id': None, 'layer_id': {'raw': 'unknown'},
              'genes': [' raw ', False, None], 'mode': {'raw': 'ranked'},
              'top_k': '2', 'cursor': ['bad'], 'require_all_in_layer': 'true'}
    rendered = render_s2_9_diagnostic_request(requests, split='validation', kind=kind, variant_index=1, **values)
    assert rendered == requests['diagnostic_requests']['validation'][kind][1].format(
        **{key: json.dumps(value) for key, value in values.items()})


@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
@pytest.mark.parametrize('field', FIELDS)
def test_renderers_require_actual_json_values(requests, value, field):
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_9_diagnostic_request(requests, split='test', kind='invalid_list', variant_index=0,
            **{**VALUES, field: value})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_template_bindings_cannot_change(requests, change):
    template = requests['train']['complete'][0]
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
    requests['train']['complete'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_9_requests(requests)


def test_request_pin_rejects_changed_bytes(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_9_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate CLI allocation checks from source reads and output writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return dict.fromkeys(('purpose', 'generation_budget', 'row_counts', 'panel_id',
                              'training_ready', 'production_export_allowed'))

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_9_corpus',
        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_9.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--sets-per-stratum', '0', '1', '1'],
    ['--sets-per-stratum', '1', '-1', '1'], ['--smoke', '--sets-per-stratum', '1', '1', '1'],
    ['--smoke', '--input-sizes', '-1']])
def test_cli_rejects_missing_or_conflicting_budgets(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_9.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_explicit_sizes_quotas_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_9.py', '--output-dir', str(tmp_path),
        '--sets-per-stratum', '2', '3', '4', '--input-sizes', '0', '3', '8', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'input_sizes': [0, 3, 8], 'seed': 7})]


def test_cli_smoke_has_no_implicit_production_budget(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_9.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls[0][1] == {'counts': None, 'smoke': True, 'input_sizes': None, 'seed': 20261028}
