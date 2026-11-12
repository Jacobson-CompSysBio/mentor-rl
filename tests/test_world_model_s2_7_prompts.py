"""Check ordered neighbor roles, raw diagnostics, and frozen wording splits."""

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_7_prompts import (
    DIAGNOSTICS, DIRECTION_PAIRS, KINDS, load_s2_7_requests,
    render_s2_7_diagnostic_request, render_s2_7_request,
    validate_s2_7_rendered_prompts, validate_s2_7_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_7_v1_requests.json'
GENES = [' raw_symbol ', 'ENSG00000001167']


@pytest.fixture
def requests():
    """Give each mutation test an independent shared-neighbor bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read unchanged wrappers and all six earlier S2 request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json', 's2_5_v1_requests.json', 's2_6_v1_requests.json',
    ))


def test_overlay_keeps_the_exact_s23_neighbor_tool_and_all_prior_pins(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_7_v1.json').read_text())
    assert load_s2_7_requests(contract['request_bank']) == requests
    report = validate_s2_7_requests(requests)
    assert all(list(kinds) == list(KINDS) and set(kinds.values()) == {4}
               for kinds in report['template_counts'].values())
    assert all(list(kinds) == list(DIAGNOSTICS) and set(kinds.values()) == {4}
               for kinds in report['diagnostic_template_counts'].values())
    assert contract['implemented_families'] == ['S2.7'] and contract['training_ready'] is False
    s23 = json.loads((REQUEST_PATH.parent / 's2_3_v1.json').read_text())
    assert contract['tools'] == s23['tools']
    assert contract['tools'][0]['function']['parameters']['required'] == [
        'multiplex_id', 'layer_id', 'gene_id', 'direction', 'mode', 'top_k', 'cursor']
    previous = json.loads((REQUEST_PATH.parent / 's2_6_v1.json').read_text())
    for key in ('source_manifest', 'registry_index', 'prompt_bank', 'tokenizer', 'context_recipe',
                's21_request_bank', 's22_request_bank', 's23_request_bank', 's24_request_bank',
                's25_request_bank', 'scoring'):
        assert contract[key] == previous[key]
    assert contract['s26_request_bank'] == previous['request_bank']
    assert 'both direction is never a tool argument' in contract['system_prompt']
    assert 'SHARED_NEIGHBOR_RANKING_REQUIRED' in contract['system_prompt']


def test_full_questions_are_disjoint_across_all_seven_families(requests, prior_banks):
    report = validate_s2_7_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_7_rendered_questions'] == {'train': 4352, 'validation': 1088, 'test': 1088}
    assert report['cross_split_pairs_checked'] > 60000000
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_earlier_heldout_wording_cannot_enter_training(requests, prior_banks):
    requests['train']['undirected'][0] += ' Return the smaller set when availability is insufficient.'
    validate_s2_7_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_7_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('kind', KINDS)
def test_each_direction_pair_preserves_raw_order_and_exact_bindings(requests, split, kind):
    directions = list(DIRECTION_PAIRS[kind])
    genes = list(GENES)
    text = render_s2_7_request(requests, split=split, variant_index=5,
        multiplex_id='MultiplexExample', layer_id='LayerExample', genes=genes, directions=directions)
    assert text == requests[split][kind][1].format(
        multiplex_id='MultiplexExample', layer_id='LayerExample',
        gene_a=json.dumps(genes[0]), gene_b=json.dumps(genes[1]))
    assert text.count(json.dumps(genes[0])) == text.count(json.dumps(genes[1])) == 1
    assert text.count('MultiplexExample') == text.count('LayerExample') == 1
    assert genes == GENES and directions == list(DIRECTION_PAIRS[kind])


@pytest.mark.parametrize('directions', [['out', 'out'], ['in', 'out'], ['both', 'both']])
def test_repeated_gene_keeps_two_roles(requests, directions):
    text = render_s2_7_request(requests, split='train', variant_index=0,
        multiplex_id='M', layer_id='L', genes=['FOXP3', 'FOXP3'], directions=directions)
    assert text.count('"FOXP3"') == 2


@pytest.mark.parametrize('genes', [None, 'A', [], ['A'], ['A', 'B', 'C'], ['A', 2], ['A', None], ['', 'B']])
def test_invalid_pairs_require_the_diagnostic_renderer(requests, genes):
    with pytest.raises(PromptBankError, match='two gene strings'):
        render_s2_7_request(requests, split='train', variant_index=0,
            multiplex_id='M', layer_id='L', genes=genes, directions=['out', 'out'])


@pytest.mark.parametrize('directions', [None, 'out', [], ['out'], ['in', 'undirected'], ['sideways', 'out'], ['out', 2]])
def test_invalid_directions_require_the_diagnostic_renderer(requests, directions):
    with pytest.raises(PromptBankError, match='two compatible directions'):
        render_s2_7_request(requests, split='train', variant_index=0,
            multiplex_id='M', layer_id='L', genes=GENES, directions=directions)


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_diagnostics_keep_all_original_json_values_and_missing_scope(requests, kind):
    genes = ['B', None, 'B']
    directions = {'raw': ['out', 'in', 'out']}
    text = render_s2_7_diagnostic_request(requests, split='validation', kind=kind, variant_index=1,
        multiplex_id=None, layer_id=None, genes=genes, directions=directions)
    assert json.dumps(genes) in text and json.dumps(directions) in text
    assert text.count('null') == 3


@pytest.mark.parametrize('value', [False, 7, ' raw scalar ', ['B', 3, 'B'], {'genes': ['A']}, None])
def test_diagnostics_do_not_coerce_raw_input_shapes(requests, value):
    text = render_s2_7_diagnostic_request(requests, split='test', kind='invalid_pair', variant_index=0,
        multiplex_id='M', layer_id='L', genes=value, directions=value)
    assert text.count(json.dumps(value, ensure_ascii=False)) == 2


@pytest.mark.parametrize('value', [float('nan'), {'A'}, ('A', 'B'), {3: 'A'}])
@pytest.mark.parametrize('field', ['genes', 'directions'])
def test_diagnostics_require_actual_json_values(requests, value, field):
    values = {'genes': GENES, 'directions': ['out', 'out']}
    values[field] = value
    with pytest.raises(PromptBankError, match='JSON literal'):
        render_s2_7_diagnostic_request(requests, split='train', kind='invalid_pair', variant_index=0,
            multiplex_id='M', layer_id='L', **values)


@pytest.mark.parametrize('field', ['multiplex_id', 'layer_id'])
def test_ordinary_scope_cannot_be_missing(requests, field):
    values = {'multiplex_id': 'M', 'layer_id': 'L', 'genes': GENES, 'directions': ['out', 'out']}
    values[field] = None
    with pytest.raises(PromptBankError, match='scope bindings'):
        render_s2_7_request(requests, split='train', variant_index=0, **values)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_template_bindings_cannot_change(requests, change):
    template = requests['train']['undirected'][0]
    if change == 'missing':
        template = template.replace('{gene_a}', 'the first gene')
    elif change == 'duplicate':
        template += ' {gene_a}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{gene_a}', '{gene_a!r}')
    else:
        template = template.replace('{gene_a}', '{gene_a:>20}')
    requests['train']['undirected'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_7_requests(requests)


@pytest.mark.parametrize('addition', [' Choose the strongest matches.', ' Sort by weight.', ' Use cursor pages.', ' Return top-k.'])
def test_ordinary_intersection_cannot_add_weight_rank_or_paging(requests, addition):
    requests['train']['out_in'][0] += addition
    with pytest.raises(PromptBankError, match='ranking, weight, or paging'):
        validate_s2_7_requests(requests)


def test_unknown_diagnostic_kind_is_rejected(requests):
    with pytest.raises(PromptBankError, match='unknown S2.7 diagnostic'):
        render_s2_7_diagnostic_request(requests, split='train', kind='neighbors', variant_index=0,
            multiplex_id='M', layer_id='L', genes=GENES, directions=['out', 'out'])


def test_request_file_tampering_fails_its_pin(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_7_requests(pin)


@pytest.fixture
def cli(monkeypatch):
    """Isolate CLI argument checks from source reads and artifact writes."""
    calls = []

    def build(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return {key: None for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                      'training_ready', 'production_export_allowed')}

    monkeypatch.setitem(sys.modules, 'runtime.world_model.s2_7_corpus',
                        SimpleNamespace(SPLITS=('train', 'validation', 'test'), build_corpus=build))
    main = runpy.run_path(str(REPOSITORY_ROOT / 'scripts/build_world_model_s2_7.py'))['main']
    return main, calls


@pytest.mark.parametrize('arguments', [[], ['--pairs-per-stratum', '0', '1', '1'],
    ['--pairs-per-stratum', '1', '-1', '1'], ['--smoke', '--pairs-per-stratum', '1', '1', '1']])
def test_cli_rejects_missing_or_conflicting_budgets(cli, monkeypatch, tmp_path, arguments):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_7.py', '--output-dir', str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and calls == []


def test_cli_forwards_separate_split_budgets_and_seed(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_7.py', '--output-dir', str(tmp_path),
        '--pairs-per-stratum', '2', '3', '4', '--seed', '7'])
    main()
    assert calls == [(tmp_path, {'counts': {'train': 2, 'validation': 3, 'test': 4},
                                'smoke': False, 'seed': 7})]


def test_cli_selects_the_pinned_schedule_only_in_smoke_mode(cli, monkeypatch, tmp_path):
    main, calls = cli
    monkeypatch.setattr(sys, 'argv', ['build_world_model_s2_7.py', '--output-dir', str(tmp_path), '--smoke'])
    main()
    assert calls == [(tmp_path, {'counts': None, 'smoke': True, 'seed': 20261028})]
