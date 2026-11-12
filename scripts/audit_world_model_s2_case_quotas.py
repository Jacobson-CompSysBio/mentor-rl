#!/usr/bin/env python3
"""Audit all 399 S2 case bindings and observed 512/64/64 quota availability.

Counts describe pinned existing traces, not an enumeration of future generator
capacity. The audit never moves subjects, edits source rows, relaxes a case, or
counts one model trace toward more than one numbered case in a split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_context import load_verified_lengths, sha256
from runtime.world_model.s2_audit_allocation import allocate_groups
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_audit_source import SourceFeatures, nearest_below_threshold
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_corpus import _write_immutable
from runtime.world_model.s2_production_cases import CATALOG_PATH, CASE_HEADING, FINAL_ANSWER, load_catalog
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks


SPLITS = ('train', 'validation', 'test')
BINDING_MODULES = ('s2_audit_cases_1_2', 's2_audit_cases_3_6', 's2_audit_cases_7_10', 's2_audit_cases_11_13')


def reference(path):
    """Pin an audit dependency using a repository-relative name when possible."""
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_absolute() else str(path), 'sha256': sha256(path)}


def register_row_identity(seen, row, fingerprint, split):
    """Reject a reused row ID with different private data, wire, or split."""
    record = (fingerprint, split, hashlib.sha256(json.dumps(row, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
    previous = seen.get(row['row_id'])
    if previous is not None:
        if previous != record:
            raise ValueError('one row ID names contradictory content, metadata, or source splits')
        return False
    seen[row['row_id']] = record
    return True


def catalog_inputs():
    """Read approved cases and answers while excluding appended build records."""
    text = CATALOG_PATH.read_text()
    end = text.index('## Initial builder implementation')
    frozen = text[:end]
    headings = list(CASE_HEADING.finditer(frozen))
    answers = {}
    for i, heading in enumerate(headings):
        stop = headings[i + 1].start() if i + 1 < len(headings) else len(frozen)
        answers[heading.group(1)] = json.loads(FINAL_ANSWER.search(frozen, heading.end(), stop).group(1))
    cases = load_catalog()
    if len(cases) != 399 or set(answers) != {case.case_id for case in cases}:
        raise ValueError('the approved 399-case catalog identity changed')
    return cases, answers, {'path': str(CATALOG_PATH.relative_to(ROOT)),
        'approved_prefix_bytes': len(frozen.encode()), 'approved_prefix_sha256': hashlib.sha256(frozen.encode()).hexdigest()}


def load_bindings(cases):
    """Require one explicit, inspectable generator binding for every case."""
    bindings, owners, pins = {}, {}, []
    for name in BINDING_MODULES:
        module = importlib.import_module(f'runtime.world_model.{name}')
        pins.append(reference(Path(module.__file__)))
        for case_id, binding in module.CASE_BINDINGS.items():
            if case_id in bindings:
                raise ValueError(f'duplicate catalog binding: {case_id}')
            if not binding.get('requirements') or not binding.get('variants'):
                raise ValueError(f'a case needs explicit evidence requirements and variants: {case_id}')
            module_name, function = binding['generator'].rsplit('.', 1)
            if not callable(getattr(importlib.import_module(module_name), function, None)):
                raise ValueError(f'unknown generator entrypoint: {binding["generator"]}')
            bindings[case_id], owners[case_id] = binding, module
    if set(bindings) != {case.case_id for case in cases}:
        raise ValueError('bindings do not exactly cover the frozen catalog')
    return bindings, owners, pins


def input_chunks():
    """Verify the complete length-audited pools and the authoritative repairs."""
    chunks, pins = [], []
    path = ROOT / 'diagnostics/s2_production_8k/length_audit.json'
    prep = json.loads((ROOT / 'diagnostics/s2_production_8k/preparation_summary.json').read_text())
    if sha256(path) != prep['context_audit']['sha256']:
        raise ValueError('the reviewed length audit changed')
    report = json.loads(path.read_text())
    pins.append(reference(path))
    for family in report['families']:
        directory = ROOT / family['path']
        if sha256(directory / 'manifest.json') != family['manifest_sha256']:
            raise ValueError('a corpus manifest differs from the reviewed length-audit pin')
        cache = family.get('separate_token_cache')
        cache_path = ROOT / cache['path'] if cache else None
        if cache and sha256(cache_path) != cache['sha256']:
            raise ValueError('a separately pinned token cache changed')
        manifest, lengths = load_verified_lengths(directory, ROOT, cache_path)
        pins.append(reference(directory / 'manifest.json'))
        if cache_path is not None:
            pins.append(reference(cache_path))
        for split in SPLITS:
            chunks.append((family['family'], split, directory / f'{split}.jsonl', lengths,
                           'historical_verification', manifest['files'][f'{split}.jsonl']))
    repaired = ROOT / 'diagnostics/s2_generator_checks/final/report.json'
    repairs = json.loads(repaired.read_text())
    pins.append(reference(repaired))
    for family in repairs['families'][:2]:
        for split, record in family['splits'].items():
            path = ROOT / record['witness_file']
            if sha256(path) != record['witness_sha256']:
                raise ValueError('a repaired numbered-case witness changed')
            chunks.append((family['family'], split, path, None, 'repaired_case_witness', reference(path)))
    overlap = ROOT / 'diagnostics/s2_generator_checks/supplemental/report.json'
    gap = ROOT / 'data/world_model_v4/diagnostics/s2_context_gap_witnesses_v1/report.json'
    for report_path in (overlap, gap):
        pins.append(reference(report_path))
        for family in json.loads(report_path.read_text())['families']:
            for path_key, hash_key in (('selector_module', 'selector_sha256'), ('validator_module', 'validator_sha256'),
                                       ('base_generator', 'base_generator_sha256')):
                if path_key in family:
                    dependency = ROOT / family[path_key]
                    if sha256(dependency) != family[hash_key]:
                        raise ValueError(f'a supplemental generator or validator changed: {dependency}')
                    pins.append(reference(dependency))
            path = ROOT / family['artifact']
            if sha256(path) != family['artifact_sha256']:
                raise ValueError('a supplemental source witness changed')
            chunks.append((family['family'], family['split'], path, None, 'supplemental_witness', reference(path)))
    return chunks, pins


def _envelope(case, answer, meta):
    return (answer['status'] == case.status and answer['continuation_decision'] == case.continuation_decision
            and (case.framing_hint is None or meta['framing'] == case.framing_hint))


def audit(output_dir):
    """Write the complete matrix, exact disjoint allocation, and evidence pins."""
    cases, answers, catalog = catalog_inputs()
    bindings, owners, binding_pins = load_bindings(cases)
    code_pins = [reference(ROOT / f'runtime/world_model/{name}.py') for name in
                 ('s2_audit_source', 's2_audit_identity', 's2_audit_allocation')]
    code_pins.append(reference(Path(__file__)))
    chunks, input_pins = input_chunks()
    recipe_path = ROOT / 'runtime/world_model/s2_production_v1_8k.json'
    recipe = json.loads(recipe_path.read_text())
    quotas = recipe['per_retained_case']
    tokenizer = load_tokenizer(recipe['tokenizer'])
    by_family = defaultdict(list)
    for case in cases:
        by_family[case.family].append(case)
    rows_by_fingerprint, seen_ids = {}, {}
    raw_counts = {case.case_id: {split: Counter() for split in SPLITS} for case in cases}
    schema_examples, witnesses, groups_by_case = {}, {}, defaultdict(set)
    contexts, views, indexes, validation_contexts = {}, {}, {}, {}
    totals, legacy, cross_split_duplicates = Counter(), Counter(), []
    source_inputs, source_pins = {}, []
    for family, split, path, lengths, kind, pin in chunks:
        print(f'auditing {family} {split} ({kind})', file=sys.stderr, flush=True)
        number = int(family.split('.')[1])
        if family not in contexts:
            name = f's2_{number}_production' if number < 3 else f's2_{number}_corpus'
            module = importlib.import_module(f'runtime.world_model.{name}')
            contract = module.load_build_contract()
            source_path = ROOT / contract['source_manifest']['path']
            if sha256(source_path) != contract['source_manifest']['sha256']:
                raise ValueError('the case source manifest changed')
            if source_path not in views:
                loader = load_source_view
                if number >= 12:
                    from runtime.world_model.s2_12_sources import load_source_view as loader
                views[source_path] = loader(source_path)
                indexes[source_path] = SubgraphIndex(views[source_path])
            view, index = views[source_path], indexes[source_path]
            nearest = None
            if number == 8:
                selection = json.loads((ROOT / 'data/world_model_v4/diagnostics/s2_8_builder_v1/audit.json').read_text())['selection']
                nearest = nearest_below_threshold(selection)
            source_inputs[family] = SourceFeatures(view, index, seed=recipe['seed'], nearest=nearest)
            contexts[family] = module, contract, view, index
            source_pins.extend((reference(Path(module.__file__)), reference(module.CONTRACT_PATH), reference(source_path)))
        module, contract, view, index = contexts[family]
        current_ids = by_family[family]
        file_digest = hashlib.sha256()
        with path.open('rb') as stream:
            for line_number, line in enumerate(stream, 1):
                file_digest.update(line)
                row = json.loads(line)
                meta, final = row['metadata'], row['messages'][-1]['content']
                totals['scanned_rows'] += 1
                if meta['split'] != split or meta['question_family'].replace('.single_edge', '') != family:
                    raise ValueError('a source row crossed its family or split')
                if meta['tool_contract_version'] != contract['contract_version']:
                    if number >= 3:
                        raise ValueError('a candidate uses an unexpected generator version')
                    legacy[family, split] += 1
                    continue
                totals['version_compatible_rows'] += 1
                tokens = lengths[row['row_id']]['tokens'] if lengths is not None else len(tokenize_trace(row, tokenizer)['input_ids'])
                fingerprint = model_trace_sha256(row)
                if not register_row_identity(seen_ids, row, fingerprint, split):
                    totals['repeated_row_id_references'] += 1
                    continue
                if fingerprint in rows_by_fingerprint:
                    totals['duplicate_model_traces'] += 1
                    entry = rows_by_fingerprint[fingerprint]
                    if entry['split'] != split:
                        entry['cross_split'] = True
                        cross_split_duplicates.append({'first_row': entry['first_row'], 'second_row': row['row_id']})
                        continue
                else:
                    entry = {'split': split, 'first_row': row['row_id'], 'eligible': {}, 'overlength': set(),
                             'eligible_any': False, 'cross_split': False}
                    rows_by_fingerprint[fingerprint] = entry
                if tokens <= 8192:
                    entry['eligible_any'] = True
                    totals['eligible_raw_rows'] += 1
                else:
                    totals['overlength_raw_rows'] += 1
                potential = [case for case in current_ids if meta['variant'] in bindings[case.case_id]['variants']
                             and _envelope(case, final, meta)]
                if not potential:
                    continue
                classified = dict(row)
                classified['_audit_source'] = source_inputs[family].for_row(row)
                for case in potential:
                    case_id = case.case_id
                    try:
                        match = owners[case_id].matches(case_id, classified)
                    except Exception as error:
                        raise ValueError(f'binding predicate failed: {case_id} {row["row_id"]} {path}:{line_number}') from error
                    if not match:
                        continue
                    counts = raw_counts[case_id][split]
                    if set(final['result']) != set(answers[case_id]['result']):
                        counts['catalog_result_schema_mismatch'] += 1
                        schema_examples.setdefault(case_id, {'row_id': row['row_id'], 'path': str(path.relative_to(ROOT)),
                            'expected_keys': sorted(answers[case_id]['result']), 'actual_keys': sorted(final['result'])})
                        continue
                    counts['matching_raw_rows'] += 1
                    if tokens > 8192:
                        counts['overlength_raw_rows'] += 1
                        entry['overlength'].add(case_id)
                        continue
                    counts['eligible_raw_rows'] += 1
                    source_group = meta.get('split_unit')
                    groups_by_case[case_id, split].add(source_group)
                    record = {'row_id': row['row_id'], 'path': str(path.relative_to(ROOT)), 'line': line_number,
                              'tokens': tokens, 'wrapper_index': meta.get('wrapper_index'),
                              'request_template_index': meta.get('request_template_index', meta.get('wording_index'))}
                    previous = entry['eligible'].get(case_id)
                    if previous is None or (tokens, row['row_id']) < (previous['tokens'], previous['row_id']):
                        entry['eligible'][case_id] = record
                    old = witnesses.get((case_id, split))
                    if old is None or (tokens, row['row_id']) < (old[0], old[1]['row_id']):
                        witnesses[case_id, split] = tokens, row, record
        if file_digest.hexdigest() != pin['sha256']:
            raise ValueError(f'a corpus changed while its rows were read: {path}')
    allocated, availability, overlength = {}, defaultdict(Counter), defaultdict(Counter)
    overlap_groups = []
    for split in SPLITS:
        groups = defaultdict(list)
        for fingerprint, entry in rows_by_fingerprint.items():
            if entry['split'] != split or entry['cross_split']:
                continue
            neighborhood = tuple(sorted(entry['eligible']))
            for case_id in neighborhood:
                availability[case_id][split] += 1
            for case_id in entry['overlength'] - set(entry['eligible']):
                overlength[case_id][split] += 1
            if neighborhood:
                groups[neighborhood].append(fingerprint)
        allocated[split] = allocate_groups(groups, {case.case_id: quotas[split] for case in cases})
        overlap_groups.extend({'split': split, 'case_ids': list(group), 'unique_model_traces': len(rows)}
                              for group, rows in sorted(groups.items()) if len(group) > 1)
    validation = []
    for (case_id, split), (_, row, record) in sorted(witnesses.items()):
        family = '.'.join(case_id.split('.')[:2])
        module, contract, view, index = contexts[family]
        number = int(family.split('.')[1])
        if family not in validation_contexts:
            kwargs = {'contract': contract, 'registry': load_registry_context(contract['registry_index'])}
            if number != 2:
                kwargs['index'] = index.native_index if number <= 7 else index
                if number == 8:
                    from runtime.world_model.s2_degrees import DegreeIndex
                    kwargs['index'] = DegreeIndex(view, index=index.native_index)
                if number == 11:
                    from runtime.world_model.s2_paths import PathIndex
                    kwargs['index'] = PathIndex(view, index=index.native_index)
            if number >= 3:
                kwargs['bank'] = load_s2_prompt_banks(contract['prompt_bank'])
            if number >= 12:
                previous_path = ROOT / contract['previous_source_manifest']['path']
                kwargs['old_view'] = views.get(previous_path) or load_source_view(previous_path)
            validation_contexts[family] = kwargs
        kwargs = dict(validation_contexts[family])
        validator = module.validate_case_row if number < 3 else module.validate_row
        if row['metadata'].get('extension') == 's2_9_native_loop_page_v1':
            from runtime.world_model.s2_9_loop_context import validate_row as validator
            kwargs['index'] = index
        validator(row, view, **kwargs)
        if len(tokenize_trace(row, tokenizer)['input_ids']) != record['tokens']:
            raise ValueError('a selected witness differs from its complete-wire token record')
        validation.append({'case_id': case_id, 'split': split, **record})
    for pin in (*binding_pins, *code_pins, *input_pins, *source_pins):
        if sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'a pinned audit input changed during the audit; rerun: {pin["path"]}')
    entries, allocations = [], []
    for case in cases:
        case_id = case.case_id
        counts = {split: availability[case_id][split] for split in SPLITS}
        assigned = {split: len(allocated[split][case_id]) for split in SPLITS}
        shortfalls = {split: quotas[split] - assigned[split] for split in SPLITS}
        coverage = ('quota_available' if not any(shortfalls.values()) else
                    'witnessed_quota_shortfall' if sum(counts.values()) else
                    'observed_overlength_only' if sum(overlength[case_id].values()) else 'no_observed_match')
        entries.append({**asdict(case), 'binding': bindings[case_id], 'coverage_status': coverage,
            'eligible_unique_model_traces': counts, 'disjoint_quota_allocation': assigned,
            'quota_shortfalls': shortfalls,
            'overlength_unique_model_traces': {split: overlength[case_id][split] for split in SPLITS},
            'observed_source_groups': {split: len(groups_by_case[case_id, split]) for split in SPLITS},
            'raw_match_counts': {split: dict(raw_counts[case_id][split]) for split in SPLITS},
            'schema_mismatch_example': schema_examples.get(case_id),
            'source_population_capacity': 'not_enumerated; observed shortages do not prove generator capacity limits',
            'excluded_for_length': False})
        for split in SPLITS:
            for fingerprint in allocated[split][case_id]:
                allocations.append({'case_id': case_id, 'split': split, 'model_trace_sha256': fingerprint,
                                    **rows_by_fingerprint[fingerprint]['eligible'][case_id]})
    pins = [reference(recipe_path), *binding_pins, *input_pins, *source_pins, *code_pins]
    pins = list({pin['path']: pin for pin in pins}.values())
    report = {'schema_version': 'mentor-rl-s2-case-quota-audit-v1', 'catalog': catalog,
        'catalog_cases': len(cases), 'explicit_bindings': len(bindings), 'quotas': quotas, 'max_length': 8192,
        'count_scope': 'observed pinned model traces in the selected generator contracts; not future generation capacity',
        'duplicate_policy': 'ignore private metadata/row IDs; canonicalize opaque acquisition IDs while preserving their reuse',
        'allocation_policy': 'exact maximum-flow disjoint allocation of eligible unique model traces within fixed source splits',
        'coverage_status_counts': dict(Counter(entry['coverage_status'] for entry in entries)),
        'totals': dict(totals), 'unique_model_traces': len(rows_by_fingerprint),
        'required_s2_rows': {split: len(cases) * quotas[split] for split in SPLITS},
        'allocated_s2_rows': {split: sum(len(rows) for rows in allocated[split].values()) for split in SPLITS},
        'observed_unique_eligible_rows': {split: sum(item['split'] == split and item['eligible_any'] and not item['cross_split']
            for item in rows_by_fingerprint.values()) for split in SPLITS},
        'observed_unique_eligible_rows_outside_exact_catalog_bindings': sum(item['eligible_any'] and not item['eligible']
            for item in rows_by_fingerprint.values()),
        'legacy_rows_requiring_regeneration': [{'family': f, 'split': s, 'rows': n} for (f, s), n in sorted(legacy.items())],
        'overlapping_case_neighborhoods': overlap_groups, 'cross_split_duplicate_model_traces': cross_split_duplicates,
        'case_entries': entries, 'source_and_tokenizer_witnesses_checked': len(validation),
        'input_pins': pins, 'case_coverage_ready': all(not any(entry['quota_shortfalls'].values()) for entry in entries) and not cross_split_duplicates,
        'training_ready': False, 'production_export_allowed': False,
        'remaining_work': ['Resolve exact case/evidence gaps and catalog schema drift before quota generation.',
                           'Generate missing counts in new directories, preserving original subjects and held-out wording.',
                           'Assemble and tokenize final production/replay data and verify training/evaluation adapters.']}
    csv_out = io.StringIO(newline='')
    columns = ['case_id', 'family', 'title', 'coverage_status', 'generator', 'requirements', 'implementation_gap']
    columns.extend(f'{field}_{split}' for field in ('eligible', 'allocated', 'shortfall', 'source_groups') for split in SPLITS)
    writer = csv.DictWriter(csv_out, fieldnames=columns)
    writer.writeheader()
    for entry in entries:
        values = {key: entry[key] for key in ('case_id', 'family', 'title', 'coverage_status')}
        values.update({key: entry['binding'].get(key, '') for key in ('generator', 'requirements', 'implementation_gap')})
        for split in SPLITS:
            for label, field in (('eligible', 'eligible_unique_model_traces'), ('allocated', 'disjoint_quota_allocation'),
                                 ('shortfall', 'quota_shortfalls'), ('source_groups', 'observed_source_groups')):
                values[f'{label}_{split}'] = entry[field][split]
        writer.writerow(values)
    files = {'report.json': (json.dumps(report, indent=2) + '\n').encode(),
             'case_matrix.csv': csv_out.getvalue().encode(),
             'allocation.jsonl': ''.join(json.dumps(item, separators=(',', ':')) + '\n' for item in allocations).encode(),
             'validated_witnesses.json': (json.dumps(validation, indent=2) + '\n').encode()}
    _write_immutable(output_dir, files)
    return report


def main():
    """Require a new immutable report directory; never overwrite a corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output_dir)
    print(json.dumps({key: report[key] for key in ('catalog_cases', 'explicit_bindings', 'coverage_status_counts',
        'source_and_tokenizer_witnesses_checked', 'case_coverage_ready')}, indent=2))


if __name__ == '__main__':
    main()
