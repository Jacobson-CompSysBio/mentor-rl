"""Adversarial checks for the independent corpus verifier and split namespaces."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_world_model_s2_production as verifier
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_corpus import _pair_group
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.s2_paths import PathIndex
from runtime.world_model.s2_production_adapter import tools_sha256
from runtime.world_model.s2_production_replay import adapt_replay_row
from runtime.world_model.contract import load_contract, tool_definitions


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'data/world_model_v4/diagnostics/s2_production_assembly_smoke_v1'


@pytest.fixture(scope='module')
def native():
    """Keep actual published factories in the test instead of guessing group formulas."""
    genes = [f'ENSG{value:011d}' for value in range(1, 100)
             if gene_group(f'ENSG{value:011d}')[1] == 'train'][:3]
    layer = 'HumanNetV3:string_ppi'
    view = GraphView('group_fixture', 'a' * 64, {layer: LayerGraph(layer, False,
        {(genes[0], genes[1]): .5, (genes[1], genes[2]): .6}, frozenset(genes))})
    return view, SubgraphIndex(view), genes, layer


@pytest.mark.parametrize('family', [6, 7, 9, 10, 11])
def test_composite_group_namespaces_match_the_actual_sampling_factories(native, family):
    view, index, genes, layer = native
    if family == 6:
        from runtime.world_model.s2_6_corpus import _fact
        fact = _fact(view, layer, genes[:2])
    elif family == 7:
        from runtime.world_model.s2_7_sampling import fact_from_queries
        fact = fact_from_queries(view, index.native_index, layer,
                                 [(gene, 'undirected') for gene in genes[:2]])
    elif family == 9:
        from runtime.world_model.s2_9_sampling import fact_from_genes
        fact = fact_from_genes(view, index, layer, genes[:2])
    elif family == 10:
        from runtime.world_model.s2_10_sampling import fact_from_genes
        fact = fact_from_genes(view, index, layer, genes[:2], 'undirected')
    else:
        from runtime.world_model.s2_11_sampling import fact_from_genes
        fact = fact_from_genes(view, PathIndex(view, index.native_index), layer, genes[:2])
    meta = json.loads(json.dumps(asdict(fact)))
    meta.update(question_family=f'S2.{family}', selection_seed=20261028)
    groups, exemption = verifier.query_groups(meta, 20261028)
    assert groups and exemption is None
    assert any(group[2] == fact.split_unit for group in groups)
    altered = {**meta, 'split_unit': gene_group(genes[0])[0]}
    with pytest.raises(ValueError, match='source group'):
        verifier.query_groups(altered, 20261028)
    with pytest.raises(ValueError, match='gene split'):
        verifier.query_groups({**meta, 'split': 'test'}, 20261028)


@pytest.mark.parametrize('family', [1, 2, 13])
def test_native_pair_policy_uses_pair_bins_independently_of_individual_gene_bins(native, family):
    _, _, genes, _ = native
    unit, split = _pair_group(tuple(genes[:2]), 20261028)
    meta = {'question_family': f'S2.{family}', 'selection_seed': 20261028,
            'gene_ids': genes[:2], 'split_unit': unit, 'split': split}
    groups, exempt = verifier.query_groups(meta, 20261028)
    assert groups == [('native_unordered_pair', 20261028, unit)] and exempt is None
    with pytest.raises(ValueError, match='assigned split'):
        verifier.query_groups({**meta, 'split': next(s for s in verifier.SPLITS if s != split)}, 20261028)


@pytest.mark.parametrize('family', [3, 4, 5, 8, 12])
def test_singleton_published_gene_namespace(native, family):
    _, _, genes, _ = native
    unit, split = gene_group(genes[0])
    meta = {'question_family': f'S2.{family}', 'selection_seed': 20261028,
            'gene_id': genes[0], 'split_unit': unit, 'split': split}
    assert verifier.query_groups(meta, 20261028) == ([('native_query_gene', 20261028, unit)], None)
    with pytest.raises(ValueError, match='split seed'):
        verifier.query_groups({**meta, 'selection_seed': 0}, 20261028)


def test_taxonomy_and_inputless_requests_have_no_invented_physical_group():
    meta = {'question_family': 'S2.13', 'selection_seed': 20261028, 'split': 'test',
            'category': 'family_capability', 'gene_ids': [], 'split_unit': 'synthetic_family'}
    assert verifier.query_groups(meta, 20261028) == ([], 'family_capability')
    with pytest.raises(ValueError, match='cannot hide'):
        verifier.query_groups({**meta, 'gene_ids': ['ENSG00000000001']}, 20261028)
    request = {'question_family': 'S2.9', 'selection_seed': 20261028, 'split': 'validation',
               'request': {'genes': None}, 'split_unit': 'diagnostic_anchor'}
    assert verifier.query_groups(request, 20261028) == ([], 'inputless_request')


def test_pin_checks_reject_changed_bytes_and_conflicting_path_aliases(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('{}\n')
    pin = verifier._reference(path)
    assert verifier.checked_pins([pin, pin]) == [pin]
    with pytest.raises(ValueError, match='conflicting input'):
        verifier.checked_pins([pin, {'path': str(path.parent / '.' / path.name), 'sha256': '0' * 64}])
    path.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='pinned input changed'):
        verifier.checked_pins([pin])


def legacy_history():
    """Use canonical legacy tools and a prior completed call before the last user."""
    contract = load_contract()
    system = contract['runtime']['system_prompt']
    provenance = {'registry': {'registry': 'ensembl', 'release': '116', 'evidence_id': 'old_evidence'}, 'graph': None}
    final = {'status': 'resolved', 'provenance': provenance,
        'result': {'gene_id': 'ENSG00000141510'}, 'continuation_decision': 'stop',
        'blocked_reason': None, 'summary': 'The identifier is resolved.'}
    row = {'row_id': 'source_legacy', 'messages': [
        {'role': 'system', 'content': system}, {'role': 'harness', 'content': '{}'},
        {'role': 'user', 'content': 'Resolve TP53.'},
        {'role': 'assistant', 'thinking': 'Prior private reasoning.',
         'tool_call': {'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': 'TP53'}}},
        {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': {'status': 'resolved',
            'gene_id': 'ENSG00000141510', 'provenance': provenance}},
        {'role': 'assistant', 'thinking': 'Prior final analysis.', 'content': deepcopy(final)},
        {'role': 'user', 'content': 'Give that same identifier from the existing conversation.'},
        {'role': 'assistant', 'thinking': 'The compatible supplied registry record is sufficient.', 'content': final}],
        'tools': tool_definitions(contract), 'metadata': {'step': 'S1', 'source_trace': 'S1.1.1',
            'question_family': 'human_symbol_to_ensembl', 'split_unit': 'old_group',
            'schema_version': contract['runtime']['row_schema_version'],
            'tool_contract_version': contract['runtime']['tool_contract_version'],
            'system_prompt_sha256': hashlib.sha256(system.encode()).hexdigest()}}
    return adapt_replay_row(row, 'S1')


def test_production_identity_ignores_only_prior_hidden_analysis():
    row = legacy_history()
    changed = deepcopy(row)
    changed['messages'][3]['thinking'] = 'A completely different hidden prior analysis.'
    assert verifier.model_trace_sha256(row) == verifier.model_trace_sha256(changed)
    assert verifier.source_trace_sha256(row) != verifier.source_trace_sha256(changed)
    changed['messages'][-1]['thinking'] = 'Different current reasoning.'
    assert verifier.model_trace_sha256(row) != verifier.model_trace_sha256(changed)


def test_scalar_and_evaluation_check_accepts_legacy_history_and_excludes_past_calls(monkeypatch):
    row = legacy_history()
    verifier.adapter_check(row, 'train', {})
    target = verifier.evaluation_target(row)
    assert target['expected_tool_calls'] == []
    altered = deepcopy(row)
    altered['messages'][-1]['thinking'] = 'Different current reasoning.'
    altered['messages'][-1]['content']['summary'] = 'A different summary.'
    assert verifier.evaluation_target(altered) == target
    old_target = verifier.evaluation_target
    monkeypatch.setattr(verifier, 'evaluation_target', lambda value: {
        **old_target(value), 'expected_tool_calls': [value['messages'][3]['tool_call']]})
    with pytest.raises(ValueError, match='historical calls'):
        verifier.adapter_check(row, 'train', {})


@pytest.mark.parametrize('mutation', ['free_summary', 'nonscalar_projection', 'changed_trace_order'])
def test_adapter_boundary_rejects_free_text_scoring_or_changed_projection(monkeypatch, mutation):
    row = legacy_history()
    if mutation == 'free_summary':
        old = verifier.evaluation_target
        monkeypatch.setattr(verifier, 'evaluation_target', lambda value: {
            **old(value), 'summary': value['messages'][-1]['content']['summary']})
    else:
        old = verifier.flatten_row
        def changed(value, **kwargs):
            result = old(value, **kwargs)
            if mutation == 'nonscalar_projection':
                result['fact_id'] = ['nested']
            else:
                result['trace_json'] = json.dumps(value, sort_keys=True, separators=(',', ':'))
            return result
        monkeypatch.setattr(verifier, 'flatten_row', changed)
    with pytest.raises(ValueError):
        verifier.adapter_check(row, 'train', {})


@pytest.fixture(scope='module')
def smoke_snapshot():
    """Freeze a tiny actual assembled corpus once; expensive lineage is tested separately."""
    if not (SMOKE / 'manifest.json').exists():
        pytest.skip('the separately qualified one-case assembly smoke is unavailable')
    manifest = json.loads((SMOKE / 'manifest.json').read_text())
    datasets = {split: [json.loads(line) for line in (SMOKE / f'{split}.jsonl').read_text().splitlines()]
                for split in verifier.SPLITS}
    receipts, _ = verifier.token_receipts(manifest['input_pins'])
    legacy = {'S0': {}, 'S1': {}}
    lengths = {}
    for split, rows in datasets.items():
        for row in rows:
            stage = row['metadata'].get('source_stage', 'S2')
            wire = verifier.model_trace_sha256(row)
            raw = verifier.source_trace_sha256(row)
            record = receipts[stage, raw if stage != 'S2' else wire]
            lengths[wire] = record['tokens'], record['assistant_tokens']
            if stage != 'S2':
                legacy[stage][row['metadata']['source_record_id']] = {
                    'row_id': row['row_id'], 'model_trace_sha256': wire,
                    'source_model_trace_sha256': raw, 'production_replay': row['metadata']['production_replay']}
    return manifest, datasets, receipts, legacy, lengths


@pytest.mark.parametrize('mutation', ['none', 'duplicate_id', 'duplicate_wire', 'legacy_heldout',
    'budget', 'missing_receipt', 'route', 'file_hash', 'qualified_metadata'])
def test_independent_whole_corpus_checks(tmp_path, monkeypatch, smoke_snapshot, mutation):
    manifest, datasets, receipts, legacy, lengths = deepcopy(smoke_snapshot)
    original_catalog = verifier.catalog_inputs
    def only_smoke_case():
        cases, answers, pin = original_catalog()
        return tuple(c for c in cases if c.case_id == 'S2.1.1'), {'S2.1.1': answers['S2.1.1']}, pin
    monkeypatch.setattr(verifier, 'catalog_inputs', only_smoke_case)
    monkeypatch.setattr(verifier, 'checked_pins', lambda pins: list(pins))
    monkeypatch.setattr(verifier, 'legacy_inventory', lambda: (legacy, {}, []))
    monkeypatch.setattr(verifier, 'retention_panels', lambda _: {
        'input_pins': [], 'file_count': 0, 'configs': [], 'test_panels': []})
    monkeypatch.setattr(verifier, 'token_receipts', lambda _: (receipts, []))
    monkeypatch.setattr(verifier, 'load_tokenizer', lambda _: object())
    def encode(row, _):
        length, assistant = lengths[verifier.model_trace_sha256(row)]
        return {'input_ids': [1] * length, 'completion_mask': [0] * (length - assistant) + [1] * assistant}
    monkeypatch.setattr(verifier, 'tokenize_trace', encode)
    rows = datasets['train']
    if mutation == 'duplicate_id':
        rows[1]['row_id'] = rows[0]['row_id']
    elif mutation == 'duplicate_wire':
        rows[1] = deepcopy(rows[0]); rows[1]['row_id'] = 'different_private_id'
    elif mutation == 'legacy_heldout':
        datasets['validation'][0] = deepcopy(next(row for row in rows if 'source_stage' in row['metadata']))
    elif mutation == 'budget':
        manifest['budget']['train'] += 1
    elif mutation == 'missing_receipt':
        row = next(row for row in rows if 'source_stage' not in row['metadata'])
        receipts.pop(('S2', verifier.model_trace_sha256(row)))
    elif mutation == 'route':
        manifest['contract_routes']['S2.1']['tools_sha256'] = '0' * 64
    elif mutation == 'qualified_metadata':
        row = next(row for row in rows if 'source_stage' not in row['metadata'])
        row['metadata']['wrapper_index'] += 1
    corpus = tmp_path / 'corpus'; corpus.mkdir()
    for split, data in datasets.items():
        payload = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in data).encode()
        (corpus / f'{split}.jsonl').write_bytes(payload)
        manifest['files'][f'{split}.jsonl'].update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    if mutation == 'file_hash':
        with (corpus / 'train.jsonl').open('ab') as stream:
            stream.write(b'\n')
    (corpus / 'manifest.json').write_text(json.dumps(manifest))
    if mutation == 'none':
        report = verifier.verify(corpus, tmp_path / 'audit')
        assert report['unique_model_traces'] == 768
        assert report['stage_counts']['S2'] == {'train': 512, 'validation': 64, 'test': 64}
    else:
        with pytest.raises(ValueError):
            verifier.verify(corpus, tmp_path / 'audit')
