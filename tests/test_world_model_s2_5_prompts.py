"""Check frozen membership requests, raw diagnostics, and split isolation."""

import json

import pytest

from runtime.world_model.contract import REPOSITORY_ROOT
from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_5_prompts import (
    DIAGNOSTICS, MODES, load_s2_5_requests, render_s2_5_diagnostic_request,
    render_s2_5_request, validate_s2_5_rendered_prompts, validate_s2_5_requests,
)
from runtime.world_model.sources import file_sha256


REQUEST_PATH = REPOSITORY_ROOT / 'runtime/world_model/s2_5_v1_requests.json'
SCOPE = {'kind': 'layers', 'layer_ids': ['LayerB', 'LayerA', 'LayerB'], 'layer_family': None}


@pytest.fixture
def requests():
    """Give each mutation test its own frozen membership bank."""
    return json.loads(REQUEST_PATH.read_text())


@pytest.fixture
def prior_banks():
    """Read neutral wrappers and the four preceding immutable request banks."""
    return tuple(json.loads((REQUEST_PATH.parent / name).read_text()) for name in (
        's2_v1_prompt_banks.json', 's2_v1_edge_requests.json', 's2_2_v1_requests.json',
        's2_3_v1_requests.json', 's2_4_v1_requests.json',
    ))


def test_pinned_bank_and_overlay_keep_all_six_tool_arguments(requests):
    contract = json.loads((REQUEST_PATH.parent / 's2_5_v1.json').read_text())
    assert load_s2_5_requests(contract['request_bank']) == requests
    report = validate_s2_5_requests(requests)
    assert report['template_counts'] == {split: {'complete': 4, 'top_k': 4} for split in ('train', 'validation', 'test')}
    assert all(count == 4 for kinds in report['diagnostic_template_counts'].values() for count in kinds.values())
    assert contract['implemented_families'] == ['S2.5'] and contract['training_ready'] is False
    assert len(contract['tools']) == 1
    tool = contract['tools'][0]['function']
    assert tool['name'] == 'get_gene_layers'
    assert tool['parameters']['required'] == ['multiplex_id', 'gene_id', 'layer_scope', 'mode', 'top_k', 'cursor']
    assert tool['parameters']['additionalProperties'] is False


def test_full_questions_remain_disjoint_across_all_five_families(requests, prior_banks):
    report = validate_s2_5_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])
    assert report['s2_5_rendered_questions'] == {'train': 2048, 'validation': 512, 'test': 512}
    assert report['cross_split_pairs_checked'] > 28901376
    assert report['maximum_cross_split_jaccard'] < 0.6
    assert report['shared_cross_split_5grams'] == 0


def test_heldout_wording_from_an_earlier_family_cannot_enter_training(requests, prior_banks):
    requests['train']['complete'][0] += ' Return the smaller set when availability is insufficient.'
    validate_s2_5_requests(requests)
    with pytest.raises(PromptBankError, match='share a 5-word run'):
        validate_s2_5_rendered_prompts(prior_banks[0], requests, *prior_banks[1:])


@pytest.mark.parametrize('mode', MODES)
def test_rendering_preserves_literal_scope_mentions_and_binds_once(requests, mode):
    text = render_s2_5_request(requests, split='train', mode=mode, variant_index=5,
        multiplex_id='MultiplexExample', layer_scope=SCOPE, gene='GeneExample', top_k=7 if mode == 'top_k' else None)
    values = {'multiplex_id': 'MultiplexExample', 'scope_text': 'layers [LayerB, LayerA, LayerB]', 'gene': 'GeneExample'}
    if mode == 'top_k':
        values['top_k'] = 7
    assert text == requests['train'][mode][1].format(**values)
    assert text.count('GeneExample') == 1
    assert 'LayerB, LayerA, LayerB' in text


@pytest.mark.parametrize('family,expected', [(None, 'all layers'), ('tf_target', 'all layers in family tf_target')])
def test_all_selectors_remain_exhaustive(requests, family, expected):
    text = render_s2_5_request(requests, split='test', mode='complete', variant_index=0,
        multiplex_id='M', layer_scope={'kind': 'all', 'layer_ids': [], 'layer_family': family}, gene='Q')
    assert expected in text


@pytest.mark.parametrize('mode,top_k', [('top_k', None), ('top_k', 0), ('top_k', -1),
                                     ('top_k', True), ('top_k', '7'), ('complete', 7)])
def test_ordinary_mode_and_bound_must_agree(requests, mode, top_k):
    with pytest.raises(PromptBankError, match='top_k'):
        render_s2_5_request(requests, split='train', mode=mode, variant_index=0,
            multiplex_id='M', layer_scope=SCOPE, gene='Q', top_k=top_k)


@pytest.mark.parametrize('scope', [None, {'kind': 'layers', 'layer_ids': [], 'layer_family': None}])
def test_ordinary_renderer_cannot_hide_missing_or_empty_selection(requests, scope):
    with pytest.raises(PromptBankError, match='scope'):
        render_s2_5_request(requests, split='train', mode='complete', variant_index=0,
            multiplex_id='M', layer_scope=scope, gene='Q')


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_diagnostic_renderer_preserves_invalid_inputs_without_call_validation(requests, kind):
    scope = ({'kind': 'layers', 'layer_ids': ['UnknownLayer'], 'layer_family': None} if kind == 'unknown_layer' else
             {'kind': 'all', 'layer_ids': [], 'layer_family': 'UnknownFamily'} if kind == 'unknown_family' else
             None if kind == 'missing_scope' else
             {'kind': 'layers', 'layer_ids': [], 'layer_family': None} if kind == 'empty_scope' else SCOPE)
    text = render_s2_5_diagnostic_request(requests, split='validation', kind=kind, variant_index=1,
        multiplex_id=None, layer_scope=scope, gene='raw_gene', top_k=0 if kind == 'invalid_limit' else None)
    assert 'raw_gene' in text and 'not supplied' in text
    if kind == 'unknown_layer':
        assert 'UnknownLayer' in text
    elif kind == 'unknown_family':
        assert 'UnknownFamily' in text
    elif kind == 'empty_scope':
        assert 'layers []' in text
    elif kind == 'invalid_limit':
        assert '0' in text


@pytest.mark.parametrize('value,literal', [(True, 'true'), (None, 'null'), ('bad limit', '"bad limit"')])
def test_invalid_limit_retains_its_json_literal(requests, value, literal):
    text = render_s2_5_diagnostic_request(requests, split='test', kind='invalid_limit', variant_index=0,
        multiplex_id='M', layer_scope=SCOPE, gene='Q', top_k=value)
    assert literal in text


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'conversion', 'format'])
def test_request_slots_cannot_change(requests, change):
    template = requests['train']['complete'][0]
    if change == 'missing':
        template = template.replace('{scope_text}', 'the active scope')
    elif change == 'duplicate':
        template += ' {gene}'
    elif change == 'extra':
        template += ' {outcome}'
    elif change == 'conversion':
        template = template.replace('{gene}', '{gene!r}')
    else:
        template = template.replace('{gene}', '{gene:>20}')
    requests['train']['complete'][0] = template
    with pytest.raises(PromptBankError, match='declared fields once'):
        validate_s2_5_requests(requests)


@pytest.mark.parametrize('addition', [' Use outgoing degree.', ' Choose the strongest layers.', ' Sort by weight.'])
def test_membership_cannot_be_replaced_by_direction_or_neighbor_ranking(requests, addition):
    requests['train']['complete'][0] += addition
    with pytest.raises(PromptBankError, match='direction, degree, or weight'):
        validate_s2_5_requests(requests)


def test_bounded_request_requires_manifest_order_and_exhaustion(requests):
    requests['train']['top_k'][0] = requests['train']['top_k'][0].replace('manifest order', 'degree order')
    with pytest.raises(PromptBankError):
        validate_s2_5_requests(requests)


def test_request_file_tampering_fails_its_pin(requests, tmp_path):
    path = tmp_path / 'requests.json'
    path.write_text(json.dumps(requests))
    pin = {'path': str(path), 'sha256': file_sha256(path)}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(PromptBankError, match='does not match its pin'):
        load_s2_5_requests(pin)
