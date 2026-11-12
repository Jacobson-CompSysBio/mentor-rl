"""Regress visible-wording capacity for fixed singleton identity diagnostics."""

from copy import deepcopy
import json

import pytest

from runtime.world_model import s2_production_12_identity as production
from runtime.world_model import s2_production_cases_11_13 as original
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_12_corpus import load_build_contract
from runtime.world_model.s2_12_sampling import fact_from_gene
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_corpus import ROOT
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


@pytest.fixture(scope='module')
def context():
    """Keep native graph facts small while using the real pinned identity registry."""
    genes = [f'ENSG{value:011d}' for value in range(1, 61)]
    layer = 'HumanNetV3:string_ppi'
    native = LayerGraph(layer, False, {tuple(sorted((left, right))): .5
        for left, right in zip(genes, genes[1:])}, frozenset(genes))
    view = GraphView('singleton_identity_fixture', 'b' * 64, {layer: native})
    contract = load_build_contract()
    return {'view': view, 'old_view': view, 'index': SubgraphIndex(view), 'contract': contract,
        'bank': load_s2_prompt_banks(contract['prompt_bank']),
        'registry': load_registry_context(contract['registry_index']), 'seed': 20261028}


@pytest.mark.parametrize('case_id', ['S2.12.27', 'S2.12.28'])
@pytest.mark.parametrize(('split', 'quota'), [('train', 512), ('validation', 64), ('test', 64)])
def test_fixed_symbol_diagnostic_has_a_full_quota_of_distinct_visible_wordings(context, case_id, split, quota):
    prepared = production.prepare_context(context)
    gene = next(gene for gene in sorted(context['view'].multiplex_nodes)
                if gene_group(gene, context['seed'])[1] == split)
    fact = fact_from_gene(context['view'], context['index'], context['registry'], context['old_view'],
                         'HumanNetV3:string_ppi', gene, context['seed'])
    from scripts.audit_world_model_s2_case_quotas import catalog_inputs
    case = next(c for c in catalog_inputs()[0] if c.case_id == case_id)
    framing = case.framing_hint or ('step' if case.continuation_decision == 'continue' else 'direct')
    rows = [original._build(prepared, case_id, fact, framing, wrapper, template)
        for template in range(16 if split == 'train' else 8)
        for wrapper in range(len(context['bank'][split][framing]))]
    assert len(rows) == len({model_trace_sha256(row) for row in rows}) == quota
    assert len({row['messages'][2]['content'] for row in rows}) == quota
    for row in (rows[0], rows[-1]):
        production.validate_row(row, context)
    changed = deepcopy(rows[-1])
    changed['metadata']['request_template_index'] = 0
    with pytest.raises(ValueError):
        production.validate_row(changed, context)


def test_only_identity_diagnostic_lists_change_and_original_first_four_survive():
    old = json.loads((ROOT / 'runtime/world_model/s2_12_v1_requests.json').read_text())
    new = json.loads((ROOT / production.REQUEST_BANK['path']).read_text())
    for split, size in (('train', 16), ('validation', 8), ('test', 8)):
        assert new[split] == old[split]
        assert new['diagnostic_requests'][split]['unresolved_identity'][:4] == old['diagnostic_requests'][split]['unresolved_identity']
        assert len(new['diagnostic_requests'][split]['unresolved_identity']) == size
        for kind, templates in old['diagnostic_requests'][split].items():
            if kind != 'unresolved_identity':
                assert new['diagnostic_requests'][split][kind] == templates
