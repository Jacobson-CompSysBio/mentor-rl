"""Fill sparse native neighbor quotas using independently bound frozen prose.

The five native loop subjects retain their original source splits. Only this
incoming-loop case uses the separately pinned larger request bank. The large
outgoing population case combines its existing four request cores and wrappers.
Other S2.3 cases retain the previous production generator and wording contract.
"""

from copy import deepcopy

from . import s2_production_cases_1_3 as previous
from . import s2_3_corpus as neighbors
from .s2_3_prompts import load_s2_3_requests, render_s2_3_request
from .s2_audit_cases_3_6 import matches
from .s2_corpus import ROOT, digest
from .sources import file_sha256


CASE_ID = 'S2.3.19'
CASE_IDS = (CASE_ID, 'S2.3.20')
EXTENSION = 's2_complete_incoming_loop_wording_v1'
HIGH_EXTENSION = 's2_high_degree_outgoing_wording_v1'
REQUEST_BANK = {'path': 'runtime/world_model/s2_production_3_requests_v1.json',
                'sha256': '523163c2cd1b516f1e682e1539846556a557207173f13d07dead925ff684f9ef'}
INPUT_PATHS = (REQUEST_BANK['path'],)


def prepare_context(context, family=3):
    """Verify and cache the local wording overlay without changing the base route."""
    if family != 3:
        raise ValueError('the incoming-loop extension serves only S2.3')
    if '_incoming_loop_contract' not in context:
        path = ROOT / REQUEST_BANK['path']
        if file_sha256(path) != REQUEST_BANK['sha256']:
            raise ValueError('the incoming-loop production wording differs from its pin')
        contract = deepcopy(context['contract'])
        contract['request_bank'] = dict(REQUEST_BANK)
        contract['_requests'] = load_s2_3_requests(REQUEST_BANK)
        context['_incoming_loop_contract'] = contract
    return context


def _rename(value, old, new):
    """Rename one opaque acquisition namespace everywhere while retaining reuse."""
    if isinstance(value, dict):
        return {key: _rename(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item, old, new) for item in value]
    return value.replace(old, new) if isinstance(value, str) else value


def _row_id(row, view):
    """Bind the new identity to actual source subjects and visible wording."""
    meta = row['metadata']
    return 's23prodloop_' + digest([meta['catalog_case_id'], meta['layer_id'], meta['gene_id'], meta['split'],
                                   row['messages'][2]['content'], view.graph_version])[:32]


def production_candidates(case_id, context, *, split):
    """Yield independent wording bindings for genuinely sparse native facts."""
    if case_id not in CASE_IDS:
        yield from previous.production_candidates(case_id, context, split=split)
        return
    if split not in ('train', 'validation', 'test'):
        raise ValueError('the incoming-loop case requires a fixed source split')
    if case_id == CASE_ID:
        prepare_context(context)
    view, bank, native = context['view'], context['bank'], context['index'].native_index
    contract = context['_incoming_loop_contract'] if case_id == CASE_ID else context['contract']
    mode, direction, bound = ('complete', 'in', None) if case_id == CASE_ID else ('top_k', 'out', 1)
    for fact in previous._neighbor_facts(case_id, context, split):
        if case_id == CASE_ID and (fact.direction != 'in' or (fact.gene_id, fact.gene_id) not in view.layers[fact.layer_id].edges):
            raise ValueError('a complete incoming native loop fact is required')
        if case_id != CASE_ID and (fact.direction != 'out' or len(native.neighbors(fact.layer_id, fact.gene_id, 'out')) < 731):
            raise ValueError('the large outgoing native population must contain at least731 neighbors')
        templates = contract['_requests'][split][mode][direction]
        for request_index in range(len(templates)):
            original = neighbors.build_row(view, fact, mode, bound, 'direct', 'lookup', request_index,
                bank, contract, index=native, registry=context['registry'], seed=context['seed'])
            core = render_s2_3_request(contract['_requests'], split=split, mode=mode, direction=direction,
                variant_index=request_index, multiplex_id=view.multiplex_id, layer_id=fact.layer_id, gene=fact.gene_id,
                top_k=bound)
            for wrapper, template in enumerate(bank[split]['direct']):
                row = deepcopy(original)
                row['messages'][2]['content'] = template.format(request=core)
                row['metadata'].update(catalog_case_id=case_id, wrapper_index=wrapper,
                    request_template_index=request_index,
                    production_extension=EXTENSION if case_id == CASE_ID else HIGH_EXTENSION)
                row = _rename(row, original['row_id'], _row_id(row, view))
                classified = dict(row)
                classified['_audit_source'] = {'native_neighbor_count': len(native.neighbors(fact.layer_id, fact.gene_id, direction))}
                if not matches(case_id, classified):
                    raise ValueError('the source loop no longer satisfies the frozen case condition')
                yield row


def validate_row(row, context):
    """Replay full native evidence and enforce the original exact case semantics."""
    case_id = row['metadata'].get('catalog_case_id')
    if case_id not in CASE_IDS or case_id != CASE_ID and row['metadata'].get('production_extension') is None:
        return previous.validate_row(row, context)
    if case_id == CASE_ID:
        prepare_context(context)
    meta, view = row['metadata'], context['view']
    extension = EXTENSION if case_id == CASE_ID else HIGH_EXTENSION
    mode, direction, bound = ('complete', 'in', None) if case_id == CASE_ID else ('top_k', 'out', 1)
    if (meta.get('production_extension') != extension or meta.get('direction') != direction
            or meta.get('mode') != mode or meta.get('top_k') != bound
            or meta.get('framing') != 'direct' or meta.get('variant') != 'lookup'
            or row['row_id'] != _row_id(row, view)):
        raise ValueError('the incoming-loop case identity or complete request changed')
    neighbors.validate_row(row, view, index=context['index'].native_index, bank=context['bank'],
        contract=context['_incoming_loop_contract'] if case_id == CASE_ID else context['contract'], registry=context['registry'])
    classified = dict(row)
    classified['_audit_source'] = {'native_neighbor_count': len(context['index'].native_index.neighbors(
        meta['layer_id'], meta['gene_id'], direction))}
    if not matches(case_id, classified):
        raise ValueError('the incoming-loop row no longer satisfies its frozen case condition')
