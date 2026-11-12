#!/usr/bin/env python3
"""Extend the frozen 399-case audit with disjoint, validated repair witnesses.

The thirteen repaired cases had no eligible baseline rows. This updater proves
that each new trace matches its intended case, matches no previously populated
case, and is absent from every baseline pool. Assigning all new traces is a
maximum allocation on this independent part of the original quota network.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import importlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import (
    SPLITS, _envelope, catalog_inputs, input_chunks, load_bindings, reference,
)
from scripts.build_world_model_s2_case_repairs import check_case, frozen_audit
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_tokens import load_tokenizer
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_audit_source import SourceFeatures
from runtime.world_model.s2_corpus import _write_immutable
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.sources import file_sha256


BASELINE = ROOT / 'data/world_model_v4/diagnostics/s2_case_quota_audit_v3'


def check_new_identity(case_id, matching_cases, fingerprint, previous, added, repaired_cases):
    """Require a fresh node whose whole neighborhood is previously empty."""
    if case_id not in matching_cases or not set(matching_cases) <= set(repaired_cases):
        raise ValueError('a repair must match its intended case and only previously empty repaired cases')
    if fingerprint in previous or fingerprint in added:
        raise ValueError('a repair cannot reuse an old or already credited model trace')
    added.add(fingerprint)


def _matrix(entries):
    """Render the same per-case comparison columns as the frozen audit."""
    output = io.StringIO(newline='')
    columns = ['case_id', 'family', 'title', 'coverage_status', 'generator', 'requirements', 'implementation_gap']
    columns.extend(f'{field}_{split}' for field in ('eligible', 'allocated', 'shortfall', 'source_groups') for split in SPLITS)
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for entry in entries:
        values = {key: entry[key] for key in ('case_id', 'family', 'title', 'coverage_status')}
        values.update({key: entry['binding'].get(key, '') for key in ('generator', 'requirements', 'implementation_gap')})
        for split in SPLITS:
            for label, field in (('eligible', 'eligible_unique_model_traces'), ('allocated', 'disjoint_quota_allocation'),
                                 ('shortfall', 'quota_shortfalls'), ('source_groups', 'observed_source_groups')):
                values[f'{label}_{split}'] = entry[field][split]
        writer.writerow(values)
    return output.getvalue().encode()


def extend_audit(repair_report_path, output_dir):
    """Verify the exact incremental proof and publish a new complete matrix."""
    repair_report_path = Path(repair_report_path)
    baseline_path = BASELINE / 'report.json'
    baseline, audit_pins = frozen_audit()
    repairs = json.loads(repair_report_path.read_text())
    cases, answers, catalog = catalog_inputs()
    bindings, matchers, binding_pins = load_bindings(cases)
    cases = {case.case_id: case for case in cases}
    entries = {entry['case_id']: deepcopy(entry) for entry in baseline['case_entries']}
    wanted = {case_id for case_id, entry in entries.items() if entry['binding'].get('implementation_gap')}
    if len(wanted) != 13 or set(repairs['repaired_case_ids']) != wanted or repairs['checked_rows'] != 39:
        raise ValueError('the exact thirteen-case scope changed')
    if catalog != baseline['catalog'] or catalog != repairs['catalog']:
        raise ValueError('the frozen reviewed catalog changed')
    for case_id in wanted:
        if any(entries[case_id]['eligible_unique_model_traces'].values()):
            raise ValueError('a repaired case already has eligible baseline nodes; a full flow recomputation is required')
    pins = [*audit_pins, *baseline['input_pins'], *repairs['input_pins'], *binding_pins,
            reference(baseline_path), reference(repair_report_path), reference(Path(__file__)),
            reference(BASELINE / 'allocation.jsonl'), reference(BASELINE / 'validated_witnesses.json')]
    for pin in pins:
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'a baseline or repair input changed: {pin["path"]}')
    print('checking repair trace identities against all frozen source pools', file=sys.stderr, flush=True)
    previous = set()
    chunks, old_pins = input_chunks()
    pins.extend(old_pins)
    for _, _, path, _, _, expected in chunks:
        if file_sha256(path) != expected['sha256']:
            raise ValueError('a frozen baseline pool changed')
        with path.open() as stream:
            for line in stream:
                previous.add(model_trace_sha256(json.loads(line)))
    recipe = json.loads((ROOT / 'runtime/world_model/s2_production_v1_8k.json').read_text())
    tokenizer = load_tokenizer(recipe['tokenizer'])
    allocation = [json.loads(line) for line in (BASELINE / 'allocation.jsonl').read_text().splitlines()]
    witnesses = json.loads((BASELINE / 'validated_witnesses.json').read_text())
    if len(allocation) != len({entry['model_trace_sha256'] for entry in allocation}):
        raise ValueError('the baseline allocation reuses a model trace')
    if any(entry['model_trace_sha256'] not in previous for entry in allocation):
        raise ValueError('a baseline allocation identity is absent from the pinned pools')
    contexts, views, indices, added, seen_cells = {}, {}, {}, set(), set()
    source_groups, overlap_groups = defaultdict(set), Counter()
    for record in repairs['families']:
        family, split = record['family'], record['split']
        if family not in contexts:
            number = int(family.split('.')[1])
            module = importlib.import_module(f'runtime.world_model.s2_{number}_case_repairs')
            base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
            contract = getattr(module, 'load_build_contract', base.load_build_contract)()
            source = ROOT / contract['source_manifest']['path']
            if source not in views:
                views[source] = load_source_view(source)
                indices[source] = SubgraphIndex(views[source])
            view, index = views[source], indices[source]
            contexts[family] = (module, contract, view, index,
                load_s2_prompt_banks(contract['prompt_bank']), load_registry_context(contract['registry_index']),
                SourceFeatures(view, index, seed=recipe['seed']))
        module, contract, view, index, bank, registry, features = contexts[family]
        path = ROOT / record['artifact']
        if file_sha256(path) != record['artifact_sha256']:
            raise ValueError('a repaired witness file changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for line_number, row in enumerate(rows, 1):
            case_id = row['metadata']['catalog_case_id']
            if (case_id not in wanted or row['metadata']['split'] != split
                    or row['metadata'].get('selection_seed') != recipe['seed'] or (case_id, split) in seen_cells):
                raise ValueError('a repaired case/split is duplicated or moved')
            seen_cells.add((case_id, split))
            module.validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
            checked = check_case(row, cases[case_id], answers[case_id], bindings[case_id],
                                 matchers[case_id], features, tokenizer)
            classified = dict(row)
            classified['_audit_source'] = features.for_row(row)
            matching = [case.case_id for case in cases.values() if case.family == family
                and row['metadata']['variant'] in bindings[case.case_id]['variants']
                and _envelope(case, row['messages'][-1]['content'], row['metadata'])
                and set(row['messages'][-1]['content']['result']) == set(answers[case.case_id]['result'])
                and matchers[case.case_id].matches(case.case_id, classified)]
            fingerprint = checked['model_trace_sha256']
            check_new_identity(case_id, matching, fingerprint, previous, added, wanted)
            entry = entries[case_id]
            if 'implementation_gap' in entry['binding']:
                entry['binding']['previous_implementation_gap'] = entry['binding'].pop('implementation_gap')
                entry['binding']['generator'] = f'{module.__name__}.build_case_rows'
                entry['binding']['repair_generator'] = reference(Path(module.__file__))
            entry['disjoint_quota_allocation'][split] += 1
            entry['quota_shortfalls'][split] -= 1
            if entry['quota_shortfalls'][split] < 0:
                raise ValueError('a repaired allocation exceeds its quota')
            for matched_case in matching:
                available = entries[matched_case]
                available['eligible_unique_model_traces'][split] += 1
                source_groups[matched_case, split].add(row['metadata'].get('split_unit'))
                raw = available['raw_match_counts'][split]
                raw['matching_raw_rows'] = raw.get('matching_raw_rows', 0) + 1
                raw['eligible_raw_rows'] = raw.get('eligible_raw_rows', 0) + 1
            if len(matching) > 1:
                overlap_groups[split, tuple(sorted(matching))] += 1
            entry['coverage_status'] = 'witnessed_quota_shortfall'
            reference_row = {'row_id': row['row_id'], 'path': str(path.relative_to(ROOT)),
                'line': line_number, 'tokens': checked['tokens'], 'wrapper_index': row['metadata'].get('wrapper_index'),
                'request_template_index': row['metadata'].get('request_template_index')}
            allocation.append({'case_id': case_id, 'split': split, 'model_trace_sha256': fingerprint, **reference_row})
            witnesses.append({'case_id': case_id, 'split': split, **reference_row})
    if seen_cells != {(case_id, split) for case_id in wanted for split in SPLITS}:
        raise ValueError('every repaired case needs a verified witness in all three splits')
    if len(allocation) != len({entry['model_trace_sha256'] for entry in allocation}):
        raise ValueError('the updated allocation reuses a model trace')
    for (case_id, split), groups in source_groups.items():
        entries[case_id]['observed_source_groups'][split] = len(groups)
    report = deepcopy(baseline)
    report.update(schema_version='mentor-rl-s2-case-quota-audit-v2', case_entries=list(entries.values()),
        coverage_status_counts=dict(Counter(entry['coverage_status'] for entry in entries.values())),
        incremental_proof='All new nodes are absent from every baseline pool and match only previously empty repaired cases. Every new node is assigned once to a matching case within quota, attaining the upper bound of all new nodes; the baseline maximum allocation is unchanged.',
        repaired_case_ids=sorted(wanted), baseline_audit=reference(baseline_path), repair_report=reference(repair_report_path),
        source_and_tokenizer_witnesses_checked=baseline['source_and_tokenizer_witnesses_checked'] + len(added),
        unique_model_traces=baseline['unique_model_traces'] + len(added))
    for name in ('scanned_rows', 'version_compatible_rows', 'eligible_raw_rows'):
        report['totals'][name] += len(added)
    for split in SPLITS:
        report['allocated_s2_rows'][split] += len(wanted)
        report['observed_unique_eligible_rows'][split] += len(wanted)
    report['overlapping_case_neighborhoods'].extend(
        {'split': split, 'case_ids': list(group), 'unique_model_traces': count}
        for (split, group), count in sorted(overlap_groups.items()))
    report['case_coverage_ready'] = (all(not any(entry['quota_shortfalls'].values()) for entry in entries.values())
                                     and not report['cross_split_duplicate_model_traces'])
    report['remaining_work'] = ['Generate missing sampled combinations and per-case production quotas in new directories.',
        'Assemble and audit all production/replay traces with verified training/evaluation adapters.']
    report['input_pins'] = list({pin['path']: pin for pin in pins}.values())
    for pin in report['input_pins']:
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('an audit input changed during the incremental proof')
    if catalog_inputs()[2] != catalog:
        raise ValueError('the reviewed catalog changed during the audit')
    files = {'report.json': (json.dumps(report, indent=2) + '\n').encode(),
        'case_matrix.csv': _matrix(report['case_entries']),
        'allocation.jsonl': ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in allocation).encode(),
        'validated_witnesses.json': (json.dumps(witnesses, indent=2) + '\n').encode()}
    _write_immutable(output_dir, files)
    return report


def main():
    """Publish an updated immutable matrix from a pinned repair report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = extend_audit(args.repair_report, args.output_dir)
    print(json.dumps({'repaired_cases': len(report['repaired_case_ids']),
        'coverage_status_counts': report['coverage_status_counts'], 'allocated_s2_rows': report['allocated_s2_rows']}, indent=2))


if __name__ == '__main__':
    main()
