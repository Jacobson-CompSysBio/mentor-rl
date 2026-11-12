"""Expand fixed-symbol diagnostic wording without changing completed S2.12 rows.

The unresolved and ambiguous singleton requests contain one fixed raw symbol,
so changing private source facts cannot increase their distinct model traces.
This isolated overlay adds same-split request wording and delegates all source
semantics to the frozen production and family validators.
"""

import json

from . import s2_production_cases_11_13 as original
from .s2_12_prompts import validate_s2_12_requests
from .s2_audit_identity import model_trace_sha256
from .s2_corpus import ROOT
from .sources import file_sha256


REQUEST_BANK = {'path': 'runtime/world_model/s2_production_12_requests_v1.json',
                'sha256': '0a074ea2453266c4257401eb3405299d9508954114904aba3ddda42a916b7e4a'}
INPUT_PATHS = (*original.INPUT_PATHS, REQUEST_BANK['path'])
CASE_IDS = tuple(case for case in original.CASE_IDS if case.startswith('S2.12.'))


def prepare_context(context, family=12):
    """Replace only the cached S2.12 request data with its pinned extension."""
    prepared = original.prepare_context(context, family)
    if prepared['number'] != 12:
        raise ValueError('the singleton identity overlay applies only to S2.12')
    if '_identity_wording_ready' not in prepared:
        path = ROOT / REQUEST_BANK['path']
        if file_sha256(path) != REQUEST_BANK['sha256']:
            raise ValueError('the production singleton identity wording changed')
        requests = json.loads(path.read_text())
        validate_s2_12_requests(requests)
        prepared['contract'] = {**prepared['contract'], '_requests': requests,
                                'production_request_bank': dict(REQUEST_BANK)}
        prepared['_identity_wording_ready'] = True
    return prepared


def production_candidates(case_id, context, *, split):
    """Retain original candidates while independently expanding identity wording."""
    prepare_context(context)
    seen = set()
    for row in original.production_candidates(case_id, context, split=split):
        if case_id in ('S2.12.27', 'S2.12.28'):
            fingerprint = model_trace_sha256(row)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
        yield row


def validate_row(row, context):
    """Replay the original strict source, case, and wording validators."""
    prepare_context(context)
    original.validate_row(row, context)
