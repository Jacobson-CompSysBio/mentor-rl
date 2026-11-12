#!/usr/bin/env python3
"""Assemble only complete, pinned S2 quotas and training-only legacy replay.

SQLite provides deterministic ordering without keeping the complete corpus in
memory. Every selected trace is independently fingerprinted and checked against
its source-qualified shard before an atomic immutable corpus is published.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import catalog_inputs, reference
from runtime.world_model.s2_audit_identity import model_trace_sha256 as source_trace_sha256
from runtime.world_model.s2_production_identity import model_trace_sha256
from runtime.world_model.s2_corpus import canonical_json
from runtime.world_model.s2_production import load_recipe, production_budget
from runtime.world_model.s2_production_adapter import flatten_row, tools_sha256
from runtime.world_model.s2_production_replay import load_replay_pool, select_replay_rows, adapt_replay_row
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.sources import file_sha256


SPLITS = ('train', 'validation', 'test')


def checked_pins(pins):
    """Reject inconsistent declarations as well as changed files."""
    by_path = {}
    for pin in pins:
        previous = by_path.setdefault(pin['path'], pin['sha256'])
        if previous != pin['sha256']:
            raise ValueError(f'inconsistent immutable input versions: {pin["path"]}')
    for path, expected in by_path.items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f'an assembly input changed: {path}')
    return [{'path': path, 'sha256': value} for path, value in sorted(by_path.items())]


def read_shards(shard_root, case_ids, quotas):
    """Require one exact complete shard for every retained case and split."""
    reports = []
    for case_id in case_ids:
        family = case_id.split('.')[1]
        for split in SPLITS:
            path = shard_root / f's2_{family}' / case_id / split / 'report.json'
            if not path.exists():
                raise ValueError(f'production quota is incomplete: {case_id}/{split}')
            report = json.loads(path.read_text())
            if (report['case_id'] != case_id or report['split'] != split or report['quota'] != quotas[split]
                    or len(report['rows']) != quotas[split] or not report['source_replay_checked']
                    or not report['frozen_predicate_checked'] or report['truncation_applied']):
                raise ValueError('a shard does not establish its exact production quota')
            if any(not 0 < row['tokens'] <= 8192 or not 0 < row['assistant_tokens'] <= row['tokens']
                   for row in report['rows']):
                raise ValueError('a shard has an overlength or invalid supervision receipt')
            if file_sha256(ROOT / report['artifact']) != report['artifact_sha256']:
                raise ValueError('a source-qualified shard payload changed')
            reports.append((path, report))
    return reports


def replay_for_budget(replay_root, budget, recipe, lengths_path):
    """Retarget a qualified replay pool when explicit case exclusions reduce counts.

Selection is repeated against the original full training pools. Previously
measured rows reuse hash-bound token receipts; any newly selected row is
measured with the same pinned complete-wire tokenizer.
"""
    lengths = json.loads(lengths_path.read_text())
    tokenizer = None
    result, reports, pins = [], [], []
    for stage in ('S1', 'S0'):
        path = replay_root / stage.lower() / 'report.json'
        previous = json.loads(path.read_text())
        if file_sha256(ROOT / previous['artifact']) != previous['artifact_sha256']:
            raise ValueError('qualified replay changed')
        cache = {row['model_trace_sha256']: row for row in lengths['stages'][stage]['rows']}
        pool, source = load_replay_pool(ROOT, stage)
        if source != lengths['stages'][stage]['source_pool']:
            raise ValueError('replay length eligibility uses a different source population')
        seen, distinct = set(), []
        for row in pool:
            adapted = adapt_replay_row(row, stage)
            source_fingerprint = source_trace_sha256(adapted)
            fingerprint = model_trace_sha256(adapted)
            if source_fingerprint not in cache:
                raise ValueError('a replay source body lacks complete-wire eligibility evidence')
            if cache[source_fingerprint]['tokens'] > recipe['max_length']:
                continue
            if fingerprint not in seen:
                distinct.append(adapted)
                seen.add(fingerprint)
        selected, selection = select_replay_rows(distinct, budget['replay'][stage], stage=stage, seed=recipe['seed'])
        checks = []
        for row in selected:
            fingerprint = model_trace_sha256(row)
            check = cache.get(source_trace_sha256(row))
            if check is None:
                if tokenizer is None:
                    tokenizer = load_tokenizer(recipe['tokenizer'])
                encoded = tokenize_trace(row, tokenizer)
                check = {'row_id': row['row_id'], 'tokens': len(encoded['input_ids']),
                    'assistant_tokens': sum(encoded['completion_mask']), 'model_trace_sha256': fingerprint}
            if not 0 < check['tokens'] <= 8192 or check['assistant_tokens'] <= 0:
                raise ValueError('replay is not a fitting complete supervised trace')
            check = {**check, 'model_trace_sha256': fingerprint}
            checks.append(check)
            result.append((stage, row, check))
        reports.append({'stage': stage, 'source_pool': source, 'selection': selection,
                        'max_tokens': max(row['tokens'] for row in checks)})
        pins.extend([reference(path), *previous['input_pins'], reference(lengths_path), *lengths['input_pins']])
    return result, reports, pins


def assemble(shard_root, replay_root, scope_path, output_dir, lengths_path):
    """Publish the exact retained-case mixture after global identity verification."""
    if output_dir.exists():
        raise ValueError('the final corpus directory already exists; use immutable verification')
    recipe = load_recipe()
    cases, _, catalog = catalog_inputs()
    scope = json.loads(scope_path.read_text())
    excluded = {entry['case_id'] for entry in scope['exclusions']}
    if scope['catalog'] != catalog or scope['retained_cases'] != len(cases) - len(excluded):
        raise ValueError('the explicit production scope differs from the frozen catalog')
    case_ids = [case.case_id for case in cases if case.case_id not in excluded]
    if len(excluded) != len(scope['exclusions']) or excluded - {case.case_id for case in cases}:
        raise ValueError('invalid production case exclusions')
    budget = production_budget(len(case_ids), recipe)
    shards = read_shards(shard_root, case_ids, recipe['per_retained_case'])
    pins = [reference(scope_path), reference(Path(__file__)), reference(ROOT / 'runtime/world_model/s2_production_v1_8k.json'),
            reference(ROOT / 'scripts/audit_world_model_s2_case_quotas.py'),
            reference(ROOT / 'scripts/build_world_model_s2_production.py'),
            reference(ROOT / 'runtime/world_model/s2_production_adapter.py'),
            reference(ROOT / 'runtime/world_model/s2_production_identity.py')]
    for extra in ('scripts/build_world_model_s2_production_v2.py', 'runtime/world_model/s2_production_case_3_19.py'):
        if (ROOT / extra).exists():
            pins.append(reference(ROOT / extra))
    for exclusion in scope['exclusions']:
        pins.append(exclusion['evidence'])
        proof = json.loads((ROOT / exclusion['evidence']['path']).read_text())
        pins.extend(proof.get('input_pins', []))
    for path, report in shards:
        pins.extend([reference(path), *report['input_pins']])
    pins = checked_pins(pins)
    replay, replay_reports, replay_pins = replay_for_budget(replay_root, budget, recipe, lengths_path)
    pins = checked_pins([*pins, *replay_pins])
    routes, seen_ids, seen_traces, counts, case_counts = {}, set(), set(), Counter(), Counter()
    maxima, cases_report = Counter(), {case_id: {'case_id': case_id, 'counts': {}, 'source_groups': {},
                                              'wrapper_counts': {}} for case_id in case_ids}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mentor-s2-spool-') as temporary, tempfile.TemporaryDirectory(
            prefix='.s2-production-', dir=output_dir.parent) as staging:
        connection = sqlite3.connect(str(Path(temporary) / 'rows.sqlite'))
        connection.execute('CREATE TABLE rows (ordering TEXT PRIMARY KEY, split TEXT, payload TEXT)')

        def add(stage, row, check, split):
            fingerprint = model_trace_sha256(row)
            if fingerprint != check['model_trace_sha256'] or row['row_id'] != check['row_id']:
                raise ValueError('a selected row differs from its hash-bound token receipt')
            if fingerprint in seen_traces or row['row_id'] in seen_ids:
                raise ValueError('the assembled corpus reuses a model trace or row identity')
            seen_traces.add(fingerprint)
            seen_ids.add(row['row_id'])
            flatten_row(row, expected_split=split, contract_routes=routes)
            ordering = hashlib.sha256(f'{recipe["seed"]}:{fingerprint}'.encode()).hexdigest()
            connection.execute('INSERT INTO rows VALUES (?, ?, ?)',
                               (ordering, split, json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)))
            counts[stage, split] += 1
            maxima[split] = max(maxima[split], check['tokens'])

        for path, report in shards:
            case_id, split = report['case_id'], report['split']
            groups, wrappers = set(), Counter()
            with (ROOT / report['artifact']).open() as stream:
                for line_number, line in enumerate(stream):
                    if line_number >= len(report['rows']):
                        raise ValueError('a shard has more rows than its verified quota')
                    row = json.loads(line)
                    meta = row['metadata']
                    if meta.get('catalog_case_id') != case_id or meta.get('split') != split:
                        raise ValueError('a production row moved its approved case or split')
                    family = meta['question_family']
                    route = {'contract_version': meta['tool_contract_version'],
                        'system_prompt_sha256': meta['system_prompt_sha256'], 'tools_sha256': tools_sha256(row['tools'])}
                    if family in routes and routes[family] != route:
                        raise ValueError('a family mixes incompatible visible contracts')
                    routes[family] = route
                    add('S2', row, report['rows'][line_number], split)
                    case_counts[case_id, split] += 1
                    groups.add(meta.get('split_unit'))
                    wrappers[meta.get('wrapper_index')] += 1
            if case_counts[case_id, split] != recipe['per_retained_case'][split]:
                raise ValueError('a production case quota changed during assembly')
            cases_report[case_id]['counts'][split] = case_counts[case_id, split]
            cases_report[case_id]['source_groups'][split] = len(groups)
            cases_report[case_id]['wrapper_counts'][split] = dict(wrappers)
        for stage, row, check in replay:
            add(stage, row, check, 'train')
        expected = {('S2', split): budget['s2'][split] for split in SPLITS}
        expected.update({(stage, 'train'): budget['replay'][stage] for stage in ('S1', 'S0')})
        if dict(counts) != expected:
            raise ValueError('the assembled stage mixture differs from its exact budget')
        connection.commit()
        files = {}
        for split in SPLITS:
            target = Path(staging) / f'{split}.jsonl'
            digest, total = hashlib.sha256(), 0
            with target.open('wb') as stream:
                for (payload,) in connection.execute('SELECT payload FROM rows WHERE split=? ORDER BY ordering', (split,)):
                    data = (payload + '\n').encode()
                    stream.write(data)
                    digest.update(data)
                    total += 1
            files[target.name] = {'sha256': digest.hexdigest(), 'row_count': total, 'bytes': target.stat().st_size}
        connection.close()
        pins = checked_pins(pins)
        if catalog_inputs()[2] != catalog:
            raise ValueError('the approved case catalog changed during assembly')
        manifest = {'schema_version': 'mentor-rl-s2-production-corpus-v1', 'production_id': recipe['production_id'],
            'catalog': catalog, 'scope': reference(scope_path), 'retained_cases': len(case_ids),
            'excluded_case_ids': sorted(excluded), 'per_case_quotas': recipe['per_retained_case'],
            'budget': budget, 'case_entries': list(cases_report.values()), 'contract_routes': routes,
            'files': files, 'max_tokens': dict(maxima), 'max_length': 8192, 'truncation_applied': False,
            'whole_trace_source_and_supervision_checked': True, 'duplicate_model_traces': 0,
            'cross_split_duplicate_model_traces': 0, 'unique_model_traces': len(seen_traces),
            'replay': replay_reports, 'input_pins': pins, 'case_coverage_ready': True,
            'production_export_allowed': True, 'data_ready': True, 'training_ready': False,
            'wording_review': {'status': 'pending_human_review', 'requirement': 'AGENTS.md: LLM-drafted content is frozen, pinned, validated, and reviewed by a person.'},
            'submission_authorized': False,
            'ordering': 'SHA256(seed:canonical-model-trace-sha256)',
            'panel_id': hashlib.sha256(canonical_json({split: files[f'{split}.jsonl']['sha256']
                                                     for split in ('validation', 'test')}).encode()).hexdigest()}
        (Path(staging) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        os.rename(staging, output_dir)
    return manifest


def main():
    """Require explicit immutable shards, replay, scope, and corpus destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-root', type=Path, required=True)
    parser.add_argument('--replay-root', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--replay-lengths', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.shard_root, args.replay_root, args.scope, args.output_dir, args.replay_lengths)
    print(json.dumps({key: report[key] for key in ('retained_cases', 'budget', 'max_tokens', 'panel_id')}, indent=2))


if __name__ == '__main__':
    main()
