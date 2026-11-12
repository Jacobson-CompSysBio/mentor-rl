"""Keep reviewed S2 cases separate from generated behaviors and length exclusions.

The numbered design examples are not generator variant IDs. Production coverage
therefore requires an explicit mapping; a missing mapping never becomes an
excused context-length exclusion. Counts here describe complete, unique rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / 'agents/world_model_v3_curriculum/S2-question-classes.md'
SPLITS = ('train', 'validation', 'test')
DEFAULT_QUOTAS = {'train': 512, 'validation': 64, 'test': 64}
CASE_HEADING = re.compile(r'^### (S2\.(\d+)\.(\d+)): ([^\n]+)$', re.MULTILINE)
FINAL_ANSWER = re.compile(r'Final answer:\s*```json\s*(.*?)\s*```', re.DOTALL)
BEHAVIOR_FIELDS = (
    'variant', 'question_kind', 'claim', 'graph_claim', 'truth_scope',
    'graph_truth_scope', 'request_kind', 'mode', 'direction',
    'connectivity_kind', 'edge_query', 'framing', 'category', 'bucket',
    'require_membership', 'require_all_in_layer', 'membership_first',
)

# The initial edge builder accepts only lookup/context/terminal/recovered/
# exhausted variants. These design cases need capabilities absent from it;
# their omission is an implementation gap, not evidence of exceeding 8,192.
INITIAL_EDGE_GAPS = {
    **dict.fromkeys((f'S2.1.{n}' for n in (5, 15, 16)),
                    'The initial edge builder has no composed registry identity trace.'),
    **dict.fromkeys((f'S2.1.{n}' for n in (*range(8, 15), 19, 20, *range(23, 28))),
                    'The initial edge builder has no required native, partial, or incompatible context variant.'),
    'S2.1.28': 'The initial edge builder has no incompatible-direction diagnostic.',
    **dict.fromkeys((f'S2.1.{n}' for n in range(31, 42)),
                    'The initial edge builder has no pair-wide directionality operation.'),
}


@dataclass(frozen=True)
class CatalogCase:
    """One approved numbered example, without guessing a generator binding."""

    case_id: str
    family: str
    title: str
    line: int
    status: str
    continuation_decision: str
    framing_hint: str | None


def parse_catalog(text: str) -> tuple[CatalogCase, ...]:
    """Read case IDs and final outcomes; deferred examples have no framing guess.

    A title or a workflow-related word is insufficient to infer the framing of
    a blocked request. Only final stop/continue decisions provide that hint.
    """
    headings = list(CASE_HEADING.finditer(text))
    cases, seen = [], set()
    for number, heading in enumerate(headings):
        case_id, family, case_number, title = heading.groups()
        if case_id in seen or not 1 <= int(family) <= 13 or int(case_number) < 1:
            raise ValueError(f'duplicate or invalid approved case ID: {case_id}')
        seen.add(case_id)
        end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        answer = FINAL_ANSWER.search(text, heading.end(), end)
        if answer is None:
            raise ValueError(f'approved case has no final JSON answer: {case_id}')
        final = json.loads(answer.group(1))
        status, decision = final.get('status'), final.get('continuation_decision')
        if status not in ('resolved', 'not_found', 'ambiguous', 'invalid_input', 'error'):
            raise ValueError(f'approved case has an unknown status: {case_id}')
        if decision not in ('stop', 'continue', 'defer'):
            raise ValueError(f'approved case has an unknown completion decision: {case_id}')
        cases.append(CatalogCase(case_id, f'S2.{family}', title,
            text.count('\n', 0, heading.start()) + 1, status, decision,
            {'stop': 'direct', 'continue': 'step'}.get(decision)))
    if not cases:
        raise ValueError('the approved S2 catalog contains no numbered cases')
    return tuple(cases)


def load_catalog(path: Path = CATALOG_PATH) -> tuple[CatalogCase, ...]:
    """Read the current design catalog without changing or rewriting it."""
    return parse_catalog(Path(path).read_text())


def semantic_signature(row: Mapping[str, Any]) -> str:
    """Identify generated behavior without source identities, weights, or prose.

    These signatures are audit strata, not an inferred mapping to numbered
    examples. Two cases with the same signature still need an explicit finer
    generator distinction before independent per-case quotas can be claimed.
    """
    meta = row['metadata']
    family = meta.get('question_family')
    if not isinstance(family, str) or not re.fullmatch(r'S2\.(?:[1-9]|1[0-3])(?:\.single_edge)?', family):
        raise ValueError('a generated behavior requires an S2 question family')
    final = row['messages'][-1]['content']
    if not isinstance(final, Mapping):
        raise ValueError('a generated behavior requires a structured final answer')
    if (final.get('status') not in ('resolved', 'not_found', 'ambiguous', 'invalid_input', 'error')
            or final.get('continuation_decision') not in ('stop', 'continue', 'defer')):
        raise ValueError('a generated behavior requires its actual status and completion decision')
    result = final.get('result')
    result = result if isinstance(result, Mapping) else {}
    signature = {'question_family': family.replace('.single_edge', '')}
    signature.update({key: meta[key] for key in BEHAVIOR_FIELDS if key in meta})
    if 'directed' in meta or 'directed' in result:
        signature['directed'] = meta.get('directed', result.get('directed'))
    if str(meta.get('question_kind', '')).startswith('family_'):
        signature['layer_family'] = meta.get('layer_family', result.get('layer_family'))
    signature.update(status=final.get('status'), continuation_decision=final.get('continuation_decision'))
    return json.dumps(signature, sort_keys=True, separators=(',', ':'), allow_nan=False)


def coverage_report(cases: Sequence[CatalogCase], mappings: Mapping[str, Sequence[str]],
                    observations: Iterable[tuple[Mapping[str, Any], int]], *,
                    max_length: int = 8192, quotas: Mapping[str, int] | None = None,
                    length_excluded_case_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Report explicit case coverage using complete trace token counts.

    Overlength means every observed candidate for a mapped case exceeds the
    limit; it is not a proof that no smaller source instance could exist.
    The caller must explicitly name cases excluded from its production scope;
    an observed length gap alone is never permission to drop a behavior.
    Unmapped cases, absent candidates, overlapping mappings, and quotas with too
    few unique eligible rows prevent case-coverage readiness. Historical splits
    stay put. This check cannot establish trainer or full export readiness.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    if set(quotas) != set(SPLITS) or any(type(n) is not int or n < 1 for n in quotas.values()):
        raise ValueError('positive train, validation, and test quotas are required')
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('coverage requires unique catalog case IDs')
    if isinstance(length_excluded_case_ids, str):
        raise ValueError('length exclusions require a collection of case IDs')
    length_excluded_case_ids = set(length_excluded_case_ids)
    if length_excluded_case_ids - set(case_ids):
        raise ValueError('a length exclusion names an unknown catalog case')
    if set(mappings) - set(case_ids):
        raise ValueError('a generator mapping names an unknown catalog case')
    owners = defaultdict(list)
    normalized = {}
    catalog = {case.case_id: case for case in cases}
    for case_id, signatures in mappings.items():
        if isinstance(signatures, str) or not isinstance(signatures, Sequence):
            raise ValueError('each case mapping requires a sequence of signatures')
        if any(not isinstance(signature, str) or not signature for signature in signatures):
            raise ValueError('case mappings require nonempty signature strings')
        normalized[case_id] = tuple(dict.fromkeys(signatures))
        for signature in normalized[case_id]:
            behavior = json.loads(signature)
            expected = catalog[case_id]
            if (not isinstance(behavior, dict)
                    or json.dumps(behavior, sort_keys=True, separators=(',', ':'), allow_nan=False) != signature
                    or behavior.get('question_family') != expected.family
                    or behavior.get('status') != expected.status
                    or behavior.get('continuation_decision') != expected.continuation_decision
                    or expected.framing_hint is not None and behavior.get('framing') != expected.framing_hint):
                raise ValueError(f'generator binding changes the reviewed family, outcome, or framing: {case_id}')
            owners[signature].append(case_id)
    counts = defaultdict(Counter)
    token_extrema, seen = {}, set()
    for row, tokens in observations:
        if type(tokens) is not int or tokens < 1:
            raise ValueError('complete trace token counts must be positive integers')
        row_id = row.get('row_id')
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError('coverage counts require unique nonempty row IDs')
        seen.add(row_id)
        split = row['metadata'].get('split')
        if split not in SPLITS:
            raise ValueError('coverage must retain the declared train/validation/test split')
        signature = semantic_signature(row)
        counts[signature][split + '_candidates'] += 1
        counts[signature][split + ('_eligible' if tokens <= max_length else '_overlength')] += 1
        minimum, maximum = token_extrema.get(signature, (tokens, tokens))
        token_extrema[signature] = min(minimum, tokens), max(maximum, tokens)
    entries = []
    for case in cases:
        signatures = normalized.get(case.case_id, ())
        total = sum((counts[signature] for signature in signatures), Counter())
        candidates = sum(total[split + '_candidates'] for split in SPLITS)
        eligible = {split: total[split + '_eligible'] for split in SPLITS}
        if not signatures:
            status = 'unmapped'
        elif not candidates:
            status = 'unavailable'
        elif not sum(eligible.values()):
            status = 'overlength'
        else:
            status = 'supported'
        if case.case_id in length_excluded_case_ids and status != 'overlength':
            raise ValueError(f'length exclusion requires mapped candidates with none fitting: {case.case_id}')
        shortfalls = {split: max(0, quotas[split] - eligible[split]) for split in SPLITS}
        entry = {**asdict(case), 'coverage_status': status,
            'candidate_rows': candidates, 'eligible_rows': eligible,
            'overlength_rows': {split: total[split + '_overlength'] for split in SPLITS},
            'quota_shortfalls': shortfalls, 'generator_signatures': list(signatures),
            'excluded_for_length': case.case_id in length_excluded_case_ids}
        extrema = [token_extrema[signature] for signature in signatures if signature in token_extrema]
        entry['minimum_observed_tokens'] = min((value[0] for value in extrema), default=None)
        entry['maximum_observed_tokens'] = max((value[1] for value in extrema), default=None)
        if status == 'unmapped' and case.case_id in INITIAL_EDGE_GAPS:
            entry['implementation_gap'] = INITIAL_EDGE_GAPS[case.case_id]
        entries.append(entry)
    ambiguous = {signature: names for signature, names in owners.items() if len(names) > 1}
    retained = [entry for entry in entries if not entry['excluded_for_length']]
    blockers = [entry['case_id'] for entry in retained
                if entry['coverage_status'] != 'supported' or any(entry['quota_shortfalls'].values())]
    statuses = Counter(entry['coverage_status'] for entry in entries)
    return {'schema_version': 'mentor-rl-s2-production-case-coverage-v1',
        'max_length': max_length, 'quotas': quotas, 'catalog_cases': len(entries),
        'coverage_status_counts': dict(sorted(statuses.items())),
        'case_entries': entries, 'ambiguous_generator_mappings': ambiguous,
        'blocking_case_ids': blockers, 'observed_unique_rows': len(seen),
        'length_excluded_case_ids': sorted(length_excluded_case_ids),
        'length_exclusion_scope': 'explicit caller scope choice, with all observed candidates exceeding the complete-trace limit; no universal source-domain claim',
        'case_coverage_ready': bool(retained) and not blockers and not ambiguous,
        'training_ready': False}
