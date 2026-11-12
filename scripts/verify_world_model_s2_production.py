#!/usr/bin/env python3
"""Independently verify an assembled S2 corpus and its frozen retention panels.

Read every selected body, route it through the actual scalar adapter, and check
global model-wire identity and exact case/mix quotas. Upstream complete-wire
token receipts are bound to every row; representative rows are tokenized again.
This audit publishes evidence only and never launches a model job.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import catalog_inputs, _envelope
from runtime.world_model.s2_audit_identity import model_trace_sha256 as source_trace_sha256
from runtime.world_model.s2_production_identity import model_trace_sha256
from runtime.world_model.s2_production_adapter import flatten_row, evaluation_target
from runtime.world_model.s2_production_replay import load_replay_pool, adapt_replay_row
from runtime.world_model.s2_production import load_recipe, production_budget
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_corpus import _pair_group, digest, canonical_json, _write_immutable
from runtime.world_model.sources import file_sha256


SPLITS = ('train', 'validation', 'test')
RETENTION_CONFIGS = (
    'config/world_model_s1_120b_32n_test_v10_structured95_512n.json',
    'config/world_model_s1_120b_s0_forgetting_test_v10_512n.json',
)


def _path(path):
    """Resolve repository pins while allowing explicit external audit artifacts."""
    return ROOT / Path(path)


def _reference(path):
    """Record a readable path and its actual bytes."""
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        name = str(path)
    return {'path': name, 'sha256': file_sha256(path)}


def checked_pins(pins):
    """Check declared versions independently and reject conflicting aliases."""
    expected = {}
    for pin in pins:
        if not isinstance(pin, dict) or not {'path', 'sha256'} <= set(pin):
            raise ValueError('an input pin requires a path and SHA-256')
        path = _path(pin['path']).resolve()
        if path in expected and expected[path] != pin['sha256']:
            raise ValueError(f'conflicting input versions: {path}')
        expected[path] = pin['sha256']
    for path, value in expected.items():
        if file_sha256(path) != value:
            raise ValueError(f'a pinned input changed: {path}')
    return [_reference(path) for path in sorted(expected)]


def legacy_inventory():
    """Anchor replay to actual published training sources and all retention data."""
    indexes, sources, pins = {}, {}, []
    for stage in ('S1', 'S0'):
        pool, source = load_replay_pool(ROOT, stage)
        indexes[stage], sources[stage] = {}, source
        pins.append({'path': source['source_config'], 'sha256': source['source_config_sha256']})
        for item in source['sources']:
            pins.extend(({'path': item['manifest_path'], 'sha256': item['manifest_sha256']},
                         {'path': item['path'], 'sha256': item['sha256']}))
        for row in pool:
            adapted = adapt_replay_row(row, stage)
            indexes[stage][row['row_id']] = {
                'row_id': adapted['row_id'], 'model_trace_sha256': model_trace_sha256(adapted),
                'source_model_trace_sha256': source_trace_sha256(adapted),
                'production_replay': adapted['metadata']['production_replay']}
        del pool
    return indexes, sources, pins


def retention_panels(sources):
    """Verify 22 S1 held-out files, S0 corpus splits, and both frozen test panels."""
    pins, files, configs, panels = [], {}, [], []

    def load(path, expected=None):
        pin = _reference(_path(path))
        if expected is not None and pin['sha256'] != expected:
            raise ValueError(f'a retention manifest/config changed: {path}')
        pins.append(pin)
        return json.loads(_path(path).read_text())

    def retained(path, expected, category, rows):
        pin = _reference(_path(path))
        if pin['sha256'] != expected:
            raise ValueError(f'a frozen retention file changed: {path}')
        record = {**pin, 'category': category, 'rows': rows}
        previous = files.setdefault(pin['path'], record)
        if previous != record:
            raise ValueError('retention file declarations disagree')

    training_path = sources['S1']['source_config']
    training = load(training_path, sources['S1']['source_config_sha256'])
    aggregate_path = str(Path(training['corpus']['root']) / 'manifest.json')
    aggregate = load(aggregate_path, training['corpus']['manifest_sha256'])
    if training['corpus']['manifest_sha256'] != sources['S1']['aggregate_manifest_sha256']:
        raise ValueError('retention and replay use different S1 aggregates')
    for stage in ('S1', 'S0'):
        for item in sources[stage]['sources']:
            manifest = load(item['manifest_path'], item['manifest_sha256'])
            for split in ('validation', 'test'):
                name = f'{split}.jsonl'
                spec = manifest['files'][name]
                retained(str(Path(item['manifest_path']).with_name(name)), spec['sha256'],
                         f'{stage}_component', spec['row_count'])
    for split in ('validation', 'test'):
        spec = aggregate['files'][f'{split}.jsonl']
        retained(str(Path(aggregate_path).with_name(f'{split}.jsonl')), spec['sha256'],
                 'S1_aggregate', spec['row_count'])
    for config_path in RETENTION_CONFIGS:
        config = load(config_path)
        if (config['training']['config'] != training_path
                or config['training']['config_sha256'] != sources['S1']['source_config_sha256']
                or config['corpus']['manifest_sha256'] != training['corpus']['manifest_sha256']):
            raise ValueError('a retention config targets a different frozen training lineage')
        panel = config['test_panel']
        evaluator = load(str(Path(panel['evaluator_root']) / panel['evaluator_manifest']),
                         panel['evaluator_manifest_sha256'])
        stages = ('validation', 'test') if config['evaluation_contract'].startswith('held_out_s1') else ('test',)
        for split in stages:
            spec = evaluator[split]
            for kind in ('questions', 'answer_key'):
                retained(spec[f'{kind}_path'], spec[f'{kind}_sha256'],
                         'S1_evaluator' if len(stages) == 2 else 'S0_full_registry_evaluator', spec['row_count'])
        target = evaluator['test']
        if (target['questions_sha256'] != panel['questions_sha256']
                or target['answer_key_sha256'] != panel['answer_key_sha256']
                or target['row_count'] != panel['row_count'] or target['test_panel_id'] != panel['test_panel_id']):
            raise ValueError('the frozen retention panel identity or counts changed')
        configs.append(_reference(_path(config_path)))
        panels.append({'config': config_path, 'test_panel_id': panel['test_panel_id'],
                       'rows': panel['row_count'], 'evaluator_manifest_sha256': panel['evaluator_manifest_sha256']})
    pins = checked_pins([*pins, *files.values()])
    return {'files': list(files.values()), 'file_count': len(files),
            'category_counts': dict(Counter(item['category'] for item in files.values())),
            'configs': configs, 'test_panels': panels, 'input_pins': pins,
            'files_copied_or_changed': False, 'jobs_submitted': []}


def token_receipts(pins):
    """Index every selected-row measurement from pinned shard and replay audits."""
    records, artifacts = {}, []

    def add(record, case_id=None, split=None, stage=None):
        tokens, assistant = record.get('tokens'), record.get('assistant_tokens')
        if (type(tokens) is not int or type(assistant) is not int or not 0 < assistant <= tokens):
            raise ValueError('invalid upstream complete-wire token/supervision receipt')
        key = (stage or 'S2', record['model_trace_sha256'])
        value = {key: record[key] for key in ('row_id', 'tokens', 'assistant_tokens', 'model_trace_sha256')}
        value.update(case_id=case_id, split=split)
        if stage is None:
            value.update({field: record.get(field) for field in
                          ('split_unit', 'wrapper_index', 'request_template_index')})
        previous = records.setdefault(key, value)
        if previous != value:
            raise ValueError('the same source body has conflicting token receipts')

    for pin in pins:
        path = _path(pin['path'])
        if path.name != 'report.json':
            continue
        report = json.loads(path.read_text())
        if {'case_id', 'split', 'quota', 'artifact', 'artifact_sha256', 'rows'} <= set(report):
            if (not report.get('source_replay_checked') or not report.get('frozen_predicate_checked')
                    or report.get('truncation_applied') is not False
                    or len(report['rows']) != report['quota']
                    or file_sha256(_path(report['artifact'])) != report['artifact_sha256']):
                raise ValueError('a production shard lacks complete source/case validation')
            for row in report['rows']:
                add(row, report['case_id'], report['split'])
            artifacts.append({'path': report['artifact'], 'sha256': report['artifact_sha256']})
        elif report.get('schema_version') == 'mentor-rl-s2-full-replay-length-audit-v1':
            for stage, item in report['stages'].items():
                for row in item['rows']:
                    add(row, stage=stage)
    return records, artifacts


def query_groups(meta, seed):
    """Recompute each published source-group namespace without conflating policies."""
    family = int(meta['question_family'].split('.')[1])
    split = meta['split']
    if meta.get('selection_seed') != seed:
        raise ValueError('a production row changed its published split seed')
    if family == 13 and meta.get('category') == 'family_capability':
        if meta.get('gene_ids') or meta.get('gene_id'):
            raise ValueError('a taxonomy exemption cannot hide query genes')
        return [], 'family_capability'
    request = meta.get('request')
    if isinstance(request, dict):
        inputs = [request[key] for key in ('genes', 'gene_ids', 'gene_id', 'gene', 'queries') if key in request]
        if inputs and all(value is None or value == [] or value == '' for value in inputs):
            return [], 'inputless_request'
    if family in (1, 2, 13):
        genes = meta.get('gene_ids')
        if not isinstance(genes, list) or len(genes) != 2:
            raise ValueError('a native pair namespace requires two source endpoints')
        if any(not isinstance(gene, str) or re.fullmatch(r'ENSG[0-9]{11}', gene) is None for gene in genes):
            raise ValueError('a pair source namespace requires canonical gene IDs')
        unit, assigned = _pair_group(tuple(genes), seed)
        groups = [('native_unordered_pair', seed, unit)]
    else:
        if family == 7:
            roles = [tuple((query['gene_id'], query['direction'])) if isinstance(query, dict) else tuple(query)
                     for query in meta['queries']]
            genes = [role[0] for role in roles]
        else:
            genes = [meta['gene_id']] if 'gene_id' in meta else list(meta['gene_ids'])
        if any(not isinstance(gene, str) or re.fullmatch(r'ENSG[0-9]{11}', gene) is None for gene in genes):
            raise ValueError('a source query namespace requires canonical gene IDs')
        assigned = {gene_group(gene, seed)[1] for gene in genes}
        if len(assigned) > 1 or assigned and assigned != {split}:
            raise ValueError('query subjects crossed the published gene split')
        assigned = next(iter(assigned)) if assigned else split
        groups = [('native_query_gene', seed, gene_group(gene, seed)[0]) for gene in sorted(set(genes))]
        if family in (3, 4, 5, 8, 12):
            if len(genes) != 1:
                raise ValueError('a singleton source group changed its query shape')
            unit = gene_group(genes[0], seed)[0]
        elif family in (6, 9, 10):
            unit = digest({'seed': seed, 'split': split, 'layer_id': meta['layer_id'],
                           'gene_ids': sorted(set(genes))})
            groups.append(('native_input_gene_set', seed, unit))
        elif family == 7:
            unit = digest({'seed': seed, 'split': split, 'layer_id': meta['layer_id'], 'queries': sorted(roles)})
            groups.append(('native_role_pair', seed, unit))
        elif family == 11:
            unit = digest({'seed': seed, 'gene_ids': sorted(set(genes))})
            groups.append(('native_path_endpoints', seed, unit))
        else:
            raise ValueError('unknown source-group policy')
    if assigned != split or meta.get('split_unit') != unit:
        raise ValueError('the published source group or assigned split changed')
    return (groups, None) if genes else ([], 'empty_query')


def adapter_check(row, split, routes):
    """Check scalar training projection and current-turn structured evaluation."""
    flattened = flatten_row(row, expected_split=split, contract_routes=routes)
    expected_keys = {'trace_json', 'metadata_json', 'question_family', 'record_id', 'fact_id', 'prompt_form_id', 'split'}
    if set(flattened) != expected_keys or any(type(value) is not str for value in flattened.values()):
        raise ValueError('the training adapter must return exactly seven scalar strings')
    wire = json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if (flattened['trace_json'] != wire or json.loads(flattened['metadata_json']) != row['metadata']
            or flattened['record_id'] != row['row_id'] or flattened['fact_id'] != row['metadata']['split_unit']
            or flattened['split'] != split or flattened['prompt_form_id'] != split
            or flattened['question_family'] != row['metadata']['question_family']):
        raise ValueError('the scalar adapter changed its trace, identity, or source split')
    current_user = max(i for i, message in enumerate(row['messages']) if message['role'] == 'user')
    current = row['messages'][current_user + 1:]
    final = row['messages'][-1]['content']
    expected = {field: final[field] for field in ('status', 'result', 'continuation_decision', 'provenance')}
    expected.update(blocked_reason_required=final['continuation_decision'] == 'defer',
        expected_tool_calls=[message['tool_call'] for message in current if 'tool_call' in message],
        expected_tool_responses=[{'name': message['name'], 'content': message['content']}
                                 for message in current if message['role'] == 'tool'])
    if evaluation_target(row) != expected:
        raise ValueError('evaluation scores free prose or historical calls instead of current structured targets')


def verify(corpus_root, output_dir):
    """Independently audit an immutable corpus and publish verification receipts."""
    corpus_root, output_dir = Path(corpus_root), Path(output_dir)
    manifest_path = corpus_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    recipe = load_recipe()
    catalog, answers, catalog_pin = catalog_inputs()
    cases = {case.case_id: case for case in catalog}
    entries = {entry['case_id']: entry for entry in manifest['case_entries']}
    expected_budget = production_budget(manifest['retained_cases'], recipe)
    if (manifest.get('schema_version') != 'mentor-rl-s2-production-corpus-v1'
            or manifest.get('production_id') != recipe['production_id']
            or manifest.get('catalog') != catalog_pin or len(entries) != manifest['retained_cases']
            or len(entries) != len(manifest['case_entries']) or set(entries) - set(cases)
            or len(manifest['excluded_case_ids']) != len(set(manifest['excluded_case_ids']))
            or set(entries) & set(manifest['excluded_case_ids'])
            or set(entries) | set(manifest['excluded_case_ids']) != set(cases)
            or manifest['budget'] != expected_budget or manifest['per_case_quotas'] != recipe['per_retained_case']
            or manifest['max_length'] != 8192 or manifest['truncation_applied'] is not False
            or manifest.get('whole_trace_source_and_supervision_checked') is not True
            or manifest.get('case_coverage_ready') is not True
            or manifest.get('data_ready') is not True or manifest.get('training_ready') is not False
            or manifest.get('submission_authorized') is not False):
        raise ValueError('the corpus manifest changes its approved scope, budget, or data-only readiness')
    if set(manifest['files']) != {f'{split}.jsonl' for split in SPLITS}:
        raise ValueError('the production corpus requires exactly its three data files')
    pins = checked_pins([*manifest['input_pins'], manifest['scope'], _reference(manifest_path),
        _reference(Path(__file__)), _reference(ROOT / 'runtime/world_model/s2_production_identity.py'),
        _reference(ROOT / 'runtime/world_model/s2_production_adapter.py'),
        _reference(ROOT / 'runtime/world_model/s2_production_replay.py'),
        _reference(ROOT / 'runtime/world_model/s2_production_v1_8k.json')])
    scope = json.loads(_path(manifest['scope']['path']).read_text())
    if (scope['catalog'] != catalog_pin or scope['retained_cases'] != len(entries)
            or sorted(item['case_id'] for item in scope['exclusions']) != sorted(manifest['excluded_case_ids'])):
        raise ValueError('the corpus scope differs from its pinned exclusions')
    receipts, artifact_pins = token_receipts(pins)
    legacy, sources, legacy_pins = legacy_inventory()
    retention = retention_panels(sources)
    pins = checked_pins([*pins, *artifact_pins, *legacy_pins, *retention['input_pins']])
    tokenizer = load_tokenizer(recipe['tokenizer'])
    seen_ids, seen_traces, groups, sampled, replay_ids = set(), set(), {}, set(), set()
    counts, case_counts, maxima, exemptions = Counter(), Counter(), Counter(), Counter()
    declared_groups, wrappers, samples = {}, {}, []

    def sample(row, record, split, key):
        encoded = tokenize_trace(row, tokenizer)
        length, mask = len(encoded['input_ids']), encoded['completion_mask']
        if (length != record['tokens'] or sum(mask) != record['assistant_tokens']
                or not 0 < length <= 8192 or len(mask) != length or not any(mask)
                or any(type(flag) is not int or flag not in (0, 1) for flag in mask)):
            raise ValueError('an independently tokenized representative differs from its complete-wire receipt')
        samples.append({'sample': key, 'row_id': row['row_id'], 'split': split,
                        'tokens': length, 'assistant_tokens': sum(mask),
                        'model_trace_sha256': model_trace_sha256(row)})

    for split in SPLITS:
        path = corpus_root / f'{split}.jsonl'
        spec = manifest['files'][path.name]
        if file_sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('an assembled dataset differs from its manifest hash or byte size')
        pins.append(_reference(path))
        rows = 0
        max_row = max_record = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                meta, final = row['metadata'], row['messages'][-1]['content']
                fingerprint = model_trace_sha256(row)
                if row['row_id'] in seen_ids or fingerprint in seen_traces:
                    raise ValueError('the assembled corpus repeats a row ID or model-visible trace')
                seen_ids.add(row['row_id']); seen_traces.add(fingerprint)
                adapter_check(row, split, manifest['contract_routes'])
                stage = meta.get('source_stage', 'S2')
                if stage == 'S2':
                    case_id = meta['catalog_case_id']
                    if (case_id not in entries or not _envelope(cases[case_id], final, meta)
                            or set(final['result']) != set(answers[case_id]['result'])):
                        raise ValueError('a final row changed its approved case or answer envelope')
                    query, exempt = query_groups(meta, recipe['seed'])
                    for group in query:
                        if group in groups and groups[group] != split:
                            raise ValueError('a published query group crosses final dataset splits')
                        groups[group] = split
                    if exempt:
                        exemptions[exempt] += 1
                    key = ('S2', fingerprint)
                    record = receipts.get(key)
                    if record is None or record['case_id'] != case_id or record['split'] != split:
                        raise ValueError('a final S2 body has no matching source-qualified token receipt')
                    if (record['split_unit'] != meta['split_unit']
                            or record['wrapper_index'] != meta.get('wrapper_index')
                            or record['request_template_index'] != meta.get('request_template_index', meta.get('wording_index'))):
                        raise ValueError('a final S2 row changed its qualified source or wording metadata')
                    case_counts[case_id, split] += 1
                    declared_groups.setdefault((case_id, split), set()).add(meta['split_unit'])
                    wrappers.setdefault((case_id, split), Counter())[str(meta['wrapper_index'])] += 1
                    sample_key = f'{case_id}/{split}'
                else:
                    if stage not in ('S0', 'S1') or split != 'train':
                        raise ValueError('legacy replay leaked into a held-out dataset')
                    source_id = meta['source_record_id']
                    source = legacy[stage].get(source_id)
                    if (source is None or source['row_id'] != row['row_id']
                            or source['model_trace_sha256'] != fingerprint
                            or source['production_replay'] != meta['production_replay']
                            or (stage, source_id) in replay_ids):
                        raise ValueError('legacy replay differs from its pinned unique training source')
                    replay_ids.add((stage, source_id))
                    record = receipts.get((stage, source['source_model_trace_sha256']))
                    if record is None:
                        raise ValueError('an adapted legacy source lacks a raw-body length receipt')
                    sample_key = f'{stage}/train'
                if record['row_id'] != row['row_id'] or not 0 < record['tokens'] <= 8192:
                    raise ValueError('a final body changed its token receipt identity or budget')
                if record['tokens'] > maxima[split]:
                    maxima[split], max_row, max_record = record['tokens'], row, record
                if sample_key not in sampled:
                    sample(row, record, split, sample_key)
                    sampled.add(sample_key)
                counts[stage, split] += 1
                rows += 1
        if rows != spec['row_count'] or rows != expected_budget[split]:
            raise ValueError('a final dataset row count differs from its exact budget')
        if max_row is not None:
            sample(max_row, max_record, split, f'maximum/{split}')
    expected_counts = {('S2', split): expected_budget['s2'][split] for split in SPLITS}
    expected_counts.update({(stage, 'train'): expected_budget['replay'][stage] for stage in ('S1', 'S0')})
    if dict(counts) != expected_counts or (counts['S2', 'train'] * 5 != expected_budget['train'] * 4
            or counts['S1', 'train'] * 20 != expected_budget['train'] * 3
            or counts['S0', 'train'] * 20 != expected_budget['train']):
        raise ValueError('the final training mixture differs from exact 80/15/5')
    for case_id, entry in entries.items():
        for split, quota in recipe['per_retained_case'].items():
            if (case_counts[case_id, split] != quota or entry['counts'][split] != quota
                    or entry['source_groups'][split] != len(declared_groups[case_id, split])
                    or entry['wrapper_counts'][split] != dict(wrappers[case_id, split])):
                raise ValueError('a final per-case quota, source-group, or wording count changed')
    expected_panel = hashlib.sha256(canonical_json({split: manifest['files'][f'{split}.jsonl']['sha256']
        for split in ('validation', 'test')}).encode()).hexdigest()
    if (dict(maxima) != manifest['max_tokens'] or manifest['unique_model_traces'] != len(seen_traces)
            or manifest.get('duplicate_model_traces') != 0 or manifest.get('cross_split_duplicate_model_traces') != 0
            or manifest['panel_id'] != expected_panel):
        raise ValueError('the manifest token maxima, identity counts, or panel identity changed')
    if catalog_inputs()[2] != catalog_pin:
        raise ValueError('the approved catalog changed during verification')
    pins = checked_pins(pins)
    report = {'schema_version': 'mentor-rl-s2-production-independent-verification-v1',
        'corpus_manifest': _reference(manifest_path), 'retained_cases': len(entries),
        'budget': expected_budget, 'stage_counts': {stage: {split: counts[stage, split] for split in SPLITS}
            for stage in ('S2', 'S1', 'S0')}, 'unique_row_ids': len(seen_ids),
        'unique_model_traces': len(seen_traces), 'duplicate_model_traces': 0,
        'cross_split_duplicate_model_traces': 0, 'published_query_groups_checked': len(groups),
        'source_group_exemptions': dict(exemptions), 'source_splits_preserved': True,
        'all_rows_scalar_adapter_checked': True, 'current_turn_structured_evaluation_checked': True,
        'free_summary_reasoning_and_blocker_text_excluded': True,
        'upstream_complete_token_receipts_matched': len(seen_traces), 'max_tokens': dict(maxima),
        'tokenizer_representatives': samples, 'retention_files_preserved': retention['file_count'],
        'retention_configs': retention['configs'], 'retention_test_panels': retention['test_panels'],
        'input_pins': pins, 'data_verification_passed': True,
        'training_ready': False, 'wording_review': manifest.get('wording_review'), 'jobs_submitted': []}
    _write_immutable(output_dir, {'report.json': (json.dumps(report, indent=2) + '\n').encode(),
        'retention_panels.json': (json.dumps(retention, indent=2) + '\n').encode()})
    return report


def main():
    """Require explicit corpus and immutable audit destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.corpus_root, args.output_dir)
    print(json.dumps({key: report[key] for key in ('retained_cases', 'budget', 'max_tokens',
        'unique_model_traces', 'retention_files_preserved', 'data_verification_passed')}, indent=2))


if __name__ == '__main__':
    main()
