"""Check finite-list wording, literal diagnostic values, and held-out splits."""

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_6_prompts import (
    DIAGNOSTICS, load_s2_6_requests, render_s2_6_diagnostic_request,
    render_s2_6_request, validate_s2_6_rendered_prompts, validate_s2_6_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_6_v1_requests.json'
GENES = ['ENSG00000004897', ' raw_symbol ', 'ENSG00000001167', 'ENSG00000004897']


@pytest.fixture
def requests():
    """Give each mutation test an independent frozen classification bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read the unchanged wrappers and all five earlier S2 request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json', 's2_5_v1_requests.json',
    ))


def test_pinned_overlay_keeps_three_required_tool_arguments_and_no_input_cap(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_6_v1.json').read_text())
    assert load_s2_6_requests(contract['request_bank']) == requests
    report = validate_s2_6_requests(requests)
    assert report['template_counts'] == {split: 4 for split in ('train', 'validation', 'test')}
    assert all(count == 4 for kinds in report['diagnostic_template_counts'].values() for count in kinds.values())
    assert contract['implemented_families'] == ['S2.6'] and contract['training_ready'] is False
    assert contract['smoke_input_sizes'] == [0, 1, 2, 3, 8, 32, 256]
    assert len(contract['tools']) == 1
    tool = contract['tools'][0]['function']
    assert tool['name'] == 'get_nodes_by_layer'
    assert tool['parameters']['required'] == ['multiplex_id', 'layer_id', 'gene_ids']
    assert tool['parameters']['additionalProperties'] is False
    assert 'maxItems' not in tool['parameters']['properties']['gene_ids']
    assert 'batch_limit' in contract['system_prompt']
    prior = json.loads((REQUEST_PATH.parent / 's2_5_v1.json').read_text())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank', 'scoring'):
        assert contract[key] == prior[key]
    assert contract['s25_request_bank'] == prior['request_bank']


def test_full_questions_are_disjoint_across_all_six_families(requests, prior_banks):
    report = validate_s2_6_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_6_rendered_questions'] == {'train': 1792, 'validation': 448, 'test': 448}
    assert report['cross_split_pairs_checked'] > 47775744
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_earlier_heldout_wording_cannot_enter_training(requests, prior_banks):
    requests['train'][0] += ' Return the smaller set when availability is insufficient.'
    validate_s2_6_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_6_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_raw_gene_list_binds_once_without_identity_or_order_changes(requests, split):
    original = list(GENES)
    text = render_s2_6_request(requests, split=split, variant_index=5,
        multiplex_id='MultiplexExample', layer_id='LayerExample', genes=GENES)
    literal = json.dumps(GENES, ensure_ascii=False)
    assert text == requests[split][1].format(multiplex_id='MultiplexExample', layer_id='LayerExample', genes_json=literal)
    assert text.count(literal) == text.count('MultiplexExample') == text.count('LayerExample') == 1
    assert GENES == original


def test_explicit_empty_list_remains_an_ordinary_request(requests):
    text = render_s2_6_request(requests, split='train', variant_index=0,
        multiplex_id='M', layer_id='L', genes=[])
    assert '[]' in text and 'null' not in text


def test_renderer_does_not_impose_a_hidden_large_input_limit(requests):
    genes = [f'ENSG{index:011d}' for index in range(5000)]
    text = render_s2_6_request(requests, split='train', variant_index=0,
        multiplex_id='M', layer_id='L', genes=genes)
    assert json.dumps(genes) in text


@pytest.mark.parametrize('genes', [None, 'ENSG00000004897', True, {}, ['A', 2], ['A', None]])
def test_invalid_shapes_require_the_diagnostic_renderer(requests, genes):
    with pytest.raises(PromptBankError, match='gene list'):
        render_s2_6_request(requests, split='train', variant_index=0,
            multiplex_id='M', layer_id='L', genes=genes)


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_diagnostics_keep_original_json_and_missing_scope(requests, kind):
    genes = None if kind == 'missing_genes' else 'raw_gene' if kind == 'invalid_scalar' else ['B', None, 'B'] if kind == 'invalid_list' else GENES
    multiplex = 'UnknownMultiplex' if kind == 'unknown_multiplex' else None
    layer = 'UnknownLayer' if kind == 'unknown_layer' else None
    text = render_s2_6_diagnostic_request(requests, split='validation', kind=kind, variant_index=1,
        multiplex_id=multiplex, layer_id=layer, genes=genes)
    assert json.dumps(genes, ensure_ascii=False) in text
    assert ('UnknownMultiplex' if kind == 'unknown_multiplex' else 'null') in text
    assert ('UnknownLayer' if kind == 'unknown_layer' else 'null') in text


@pytest.mark.parametrize('genes', [False, 7, ' raw scalar ', ['B', 3, 'B'], {'genes': ['A']}, None])
def test_invalid_json_value_is_not_coerced_to_a_list(requests, genes):
    text = render_s2_6_diagnostic_request(requests, split='test', kind='invalid_scalar', variant_index=0,
        multiplex_id='M', layer_id='L', genes=genes)
    assert json.dumps(genes, ensure_ascii=False) in text


@pytest.mark.parametrize('genes', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
def test_diagnostics_require_actual_json_values(requests, genes):
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_6_diagnostic_request(requests, split='train', kind='invalid_scalar', variant_index=0,
            multiplex_id='M', layer_id='L', genes=genes)


@pytest.mark.parametrize('field', ['multiplex_id', 'layer_id'])
def test_ordinary_scope_cannot_be_missing(requests, field):
    values = {'multiplex_id': 'M', 'layer_id': 'L', 'genes': []}
    values[field] = None
    with pytest.raises(PromptBankError, match='scope bindings'):
        render_s2_6_request(requests, split='train', variant_index=0, **values)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_template_bindings_cannot_change(requests, change):
    template = requests['train'][0]
    if change == 'missing':
        template = template.replace('{genes_json}', 'the selected genes')
    elif change == 'duplicate':
        template += ' {genes_json}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{genes_json}', '{genes_json!r}')
    else:
        template = template.replace('{genes_json}', '{genes_json:>20}')
    requests['train'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_6_requests(requests)


@pytest.mark.parametrize('addition', [' Use outgoing degree.', ' Choose the strongest matches.', ' Sort by weight.', ' Use cursor pages.'])
def test_membership_cannot_be_replaced_by_ranking_or_paging(requests, addition):
    requests['train'][0] += addition
    with pytest.raises(PromptBankError, match='direction, ranking, or paging'):
        validate_s2_6_requests(requests)


@pytest.mark.parametrize('addition', [' List every node.', ' Return all nodes.', ' Give the layer inventory.'])
def test_finite_input_cannot_turn_into_a_full_layer_inventory(requests, addition):
    requests['train'][0] += addition
    with pytest.raises(PromptBankError, match='entire layer inventory'):
        validate_s2_6_requests(requests)


def test_unknown_diagnostic_kind_is_rejected(requests):
    with pytest.raises(PromptBankError, match='unknown S2.6 diagnostic'):
        render_s2_6_diagnostic_request(requests, split='train', kind='neighbors', variant_index=0,
            multiplex_id='M', layer_id='L', genes=GENES)


def test_request_file_tampering_fails_its_pin(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_6_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate argument checks from source reads and artifact writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {key: None for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                      'training_ready', 'production_export_allowed')}

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_6_corpus',
                        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_6.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [
    ['--requests-per-stratum', '1', '1', '1'],
    ['--requests-per-stratum', '0', '1', '1', '--input-sizes', '0'],
    ['--requests-per-stratum', '1', '1', '1', '--input-sizes', '-1'],
    ['--smoke', '--input-sizes', '2'],
])
def test_cli_rejects_missing_or_conflicting_verification_schedules(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_6.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_selected_sizes_and_separate_split_budgets(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_6.py', '--output-dir', str(tmp_path),
        '--requests-per-stratum', '2', '3', '4', '--input-sizes', '0', '7', '256'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'input_sizes': [0, 7, 256], 'seed': 20261028})]


def test_cli_uses_the_pinned_schedule_only_in_smoke_mode(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_6.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls == [(tmp_path, {'counts': None, 'smoke': True, 'input_sizes': None, 'seed': 20261028})]
