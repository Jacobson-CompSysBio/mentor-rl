"""Independently bind frozen strongest-neighbor requests and framing wrappers.

The old ranking repair reconstructs wrapper selection from the request index.
This adapter presents that validator with exactly the selected frozen wrapper,
then restores the public full-bank wrapper index without changing the trace.
All other S2.4–S2.6 production cases retain the preceding implementation.
"""

from copy import deepcopy

from . import s2_production_cases_4_6 as previous
from . import s2_4_case_repairs as repair


CASE_ID = 'S2.4.22'
EXTENSION = 's2_bounded_strongest_independent_wording_v1'
INPUT_PATHS = (repair.REQUEST_BANK['path'],)


def prepare_context(context, family=4):
    """Retain the existing source-selection cache and pinned bounded request bank."""
    previous.prepare_context(context, family)
    if '_repair_requests' not in context['contract']:
        context['contract'] = repair.load_build_contract()
    return context


def _narrow_bank(bank, split, wrapper):
    """Select one already audited wrapper for exact legacy validator projection."""
    if type(wrapper) is not int or not 0 <= wrapper < len(bank[split]['direct']):
        raise ValueError('the bounded strongest wrapper is outside its frozen bank')
    return {**bank, split: {**bank[split], 'direct': [bank[split]['direct'][wrapper]]}}


def production_candidates(case_id, context, *, split):
    """Yield independent original core/wrapper choices for the unresolved bound."""
    if case_id != CASE_ID:
        yield from previous.production_candidates(case_id, context, split=split)
        return
    if split not in ('train', 'validation', 'test'):
        raise ValueError('the bounded strongest case requires a fixed source split')
    prepare_context(context)
    facts = context['_production46']['facts']
    if (case_id, split) not in facts:
        facts[case_id, split] = previous._map_facts(22, context, split)
    for fact, _, _ in facts[case_id, split]:
        for request in range(len(context['contract']['_repair_requests'][split])):
            for wrapper in range(len(context['bank'][split]['direct'])):
                bank = _narrow_bank(context['bank'], split, wrapper)
                row = repair.build_row(context['view'], fact, CASE_ID, bank, context['contract'],
                    context['registry'], index=context['index'], seed=context['seed'], variant_index=request)
                row['metadata'].update(wrapper_index=wrapper, production_extension=EXTENSION)
                yield row


def validate_row(row, context):
    """Check the full-bank binding and retain the exact frozen ranking validator."""
    if row['metadata'].get('catalog_case_id') != CASE_ID:
        return previous.validate_row(row, context)
    prepare_context(context)
    meta = row['metadata']
    if meta.get('production_extension') != EXTENSION:
        raise ValueError('the independent bounded-ranking wording adapter is missing')
    bank = _narrow_bank(context['bank'], meta['split'], meta['wrapper_index'])
    projected = deepcopy(row)
    projected['metadata']['wrapper_index'] = 0
    repair.validate_row(projected, context['view'], index=context['index'], bank=bank,
        contract=context['contract'], registry=context['registry'])
