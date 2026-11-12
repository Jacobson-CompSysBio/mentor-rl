#!/usr/bin/env python3
"""Generate immutable source-qualified S2 case shards under the approved recipe.

Case shards can be built independently. Existing shards are accepted only when
their input pins and output hashes still match; assembling the final corpus is
a separate verification step after every retained quota has been satisfied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import catalog_inputs, load_bindings, reference, _envelope
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_audit_source import SourceFeatures
from runtime.world_model.s2_corpus import _jsonl, _write_immutable
from runtime.world_model.s2_production import load_recipe
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.sources import file_sha256


SPLITS = ('train', 'validation', 'test')


def generator_module(family):
    """Select a small production implementation group without changing old code."""
    group = '1_3' if family <= 3 else '4_6' if family <= 6 else '7_10' if family <= 10 else '11_13'
    return importlib.import_module(f'runtime.world_model.s2_production_cases_{group}')


def build_context(family, recipe, views, indices):
    """Load pinned native sources once and share immutable adjacency indices."""
    name = f's2_{family}_production' if family < 3 else f's2_{family}_case_repairs' if family in (4, 5, 8, 9, 10) else f's2_{family}_corpus'
    base = importlib.import_module(f'runtime.world_model.{name}')
    if not hasattr(base, 'load_build_contract'):
        base = importlib.import_module(f'runtime.world_model.s2_{family}_corpus')
    contract = base.load_build_contract()
    pin = contract['source_manifest']
    source = ROOT / pin['path']
    if file_sha256(source) != pin['sha256']:
        raise ValueError('the native source manifest changed')
    if source not in views:
        if family >= 12:
            from runtime.world_model.s2_12_sources import load_source_view as load_expanded
            views[source] = load_expanded(source)
        else:
            views[source] = load_source_view(source)
        indices[source] = SubgraphIndex(views[source])
    context = {'view': views[source], 'index': indices[source], 'contract': contract,
        'bank': load_s2_prompt_banks(contract['prompt_bank']),
        'registry': load_registry_context(contract['registry_index']), 'seed': recipe['seed']}
    previous = contract.get('previous_source_manifest')
    if previous:
        path = ROOT / previous['path']
        if file_sha256(path) != previous['sha256']:
            raise ValueError('the previous native source manifest changed')
        if path not in views:
            views[path] = load_source_view(path)
        context['old_view'] = views[path]
    module = generator_module(family)
    if hasattr(module, 'prepare_context'):
        module.prepare_context(context, family)
    context['features'] = SourceFeatures(context['view'], context['index'], seed=recipe['seed'])
    return module, context


def input_pins(module, context, family):
    """Pin loaded proof/rendering implementations and every declared new resource."""
    paths = {Path(__file__), ROOT / 'runtime/world_model/s2_production_v1_8k.json'}
    for loaded in tuple(sys.modules.values()):
        path = getattr(loaded, '__file__', None)
        if path and str(path).startswith(str(ROOT / 'runtime')) and str(path).endswith('.py'):
            paths.add(Path(path))
    paths.update(ROOT / path for path in getattr(module, 'INPUT_PATHS', ()))
    base = importlib.import_module(f'runtime.world_model.s2_{family}_production' if family < 3 else f'runtime.world_model.s2_{family}_corpus')
    paths.add(base.CONTRACT_PATH)
    paths.add(Path(module.__file__))
    for key, value in context['contract'].items():
        if isinstance(value, dict) and 'path' in value and 'sha256' in value:
            paths.add(ROOT / value['path'])
    return [reference(path) for path in sorted(paths)]


def qualify_case(row, case, expected, binding, matcher, context, tokenizer):
    """Measure the complete wire only after matching the approved exact behavior."""
    meta = row['metadata']
    final = row['messages'][-1]['content']
    if (meta.get('catalog_case_id') != case.case_id or meta.get('question_family') != case.family
            or meta.get('selection_seed') != context['seed'] or meta.get('split') not in SPLITS
            or meta.get('variant') not in binding['variants'] or not _envelope(case, final, meta)
            or set(final['result']) != set(expected['result'])):
        return None
    classified = dict(row)
    classified['_audit_source'] = context['features'].for_row(row)
    if not matcher.matches(case.case_id, classified):
        return None
    encoded = tokenize_trace(row, tokenizer)
    length, mask = len(encoded['input_ids']), encoded['completion_mask']
    if not length or len(mask) != length or not any(mask) or any(value not in (0, 1) for value in mask):
        raise ValueError('invalid whole-trace assistant supervision')
    return {'row_id': row['row_id'], 'tokens': length, 'assistant_tokens': sum(mask),
            'model_trace_sha256': model_trace_sha256(row), 'split_unit': meta.get('split_unit'),
            'wrapper_index': meta.get('wrapper_index'),
            'request_template_index': meta.get('request_template_index', meta.get('wording_index'))}


def verify_shard(directory, pins, quota):
    """Resume only an immutable shard built with the current exact inputs."""
    report = json.loads((directory / 'report.json').read_text())
    if report['quota'] != quota or report['input_pins'] != pins:
        raise ValueError(f'completed shard inputs changed; use a new output directory: {directory}')
    if file_sha256(directory / 'rows.jsonl') != report['artifact_sha256']:
        raise ValueError('a completed production shard changed')
    return report


def build_shards(output_dir, families, *, qualification=False, exclude=()):
    """Fill exact per-case quotas, failing visibly on any remaining shortfall."""
    output_dir = Path(output_dir)
    recipe = load_recipe()
    catalog, answers, catalog_pin = catalog_inputs()
    bindings, matchers, binding_pins = load_bindings(catalog)
    if set(exclude) - {case.case_id for case in catalog}:
        raise ValueError('unknown production case exclusion')
    tokenizer = load_tokenizer(recipe['tokenizer'])
    views, indices, reports, failures = {}, {}, [], []
    for family in families:
        module, context = build_context(family, recipe, views, indices)
        # Import lazy proof implementations before capturing implementation pins.
        for name in ((f's2_{family}_evidence', f's2_{family}_prompts') if family >= 4 else (f's2_{family}_prompts',)):
            if family >= 3:
                importlib.import_module(f'runtime.world_model.{name}')
        pins = [*input_pins(module, context, family), *binding_pins]
        observed = set()
        cases = [case for case in catalog if case.family == f'S2.{family}' and case.case_id not in exclude]
        for case in cases:
            for split in SPLITS:
                quota = 1 if qualification else recipe['per_retained_case'][split]
                directory = output_dir / f's2_{family}' / case.case_id / split
                if (directory / 'report.json').exists():
                    report = verify_shard(directory, pins, quota)
                    fingerprints = {entry['model_trace_sha256'] for entry in report['rows']}
                    if observed & fingerprints:
                        raise ValueError('resumed case shards reuse a model trace')
                    observed.update(fingerprints)
                    reports.append(report)
                    continue
                selected, checks, counts = [], [], Counter()
                for row in module.production_candidates(case.case_id, context, split=split):
                    counts['candidates'] += 1
                    if row['metadata'].get('split') != split:
                        raise ValueError('a production candidate moved its original source split')
                    module.validate_row(row, context)
                    checked = qualify_case(row, case, answers[case.case_id], bindings[case.case_id],
                                            matchers[case.case_id], context, tokenizer)
                    if checked is None:
                        counts['case_rejected'] += 1
                        continue
                    if checked['tokens'] > recipe['max_length']:
                        counts['overlength'] += 1
                        continue
                    fingerprint = checked['model_trace_sha256']
                    if fingerprint in observed:
                        counts['duplicate'] += 1
                        continue
                    observed.add(fingerprint)
                    selected.append(row)
                    checks.append(checked)
                    if len(selected) == quota:
                        break
                print(f'{case.case_id} {split}: {len(selected)}/{quota} {dict(counts)}', file=sys.stderr, flush=True)
                if len(selected) != quota:
                    failures.append({'case_id': case.case_id, 'split': split,
                        'selected': len(selected), 'quota': quota, 'rejections': dict(counts)})
                    continue
                if any(file_sha256(ROOT / pin['path']) != pin['sha256'] for pin in pins):
                    raise ValueError('a production input changed during generation')
                payload = _jsonl(selected)
                report = {'case_id': case.case_id, 'split': split, 'quota': quota,
                    'artifact': str(directory / 'rows.jsonl'), 'artifact_sha256': hashlib.sha256(payload).hexdigest(),
                    'rows': checks, 'rejections': dict(counts), 'input_pins': pins, 'catalog': catalog_pin,
                    'source_replay_checked': True, 'frozen_predicate_checked': True, 'truncation_applied': False}
                _write_immutable(directory, {'rows.jsonl': payload,
                    'report.json': (json.dumps(report, indent=2) + '\n').encode()})
                reports.append(report)
    if catalog_inputs()[2] != catalog_pin:
        raise ValueError('the approved catalog changed during production generation')
    result = {'schema_version': 'mentor-rl-s2-production-shards-v1', 'qualification_only': qualification,
        'families': families, 'excluded_case_ids': sorted(exclude), 'catalog': catalog_pin,
        'completed_cells': len(reports), 'rows': sum(len(report['rows']) for report in reports),
        'shortfalls': failures, 'case_quotas_satisfied': not failures,
        'max_tokens': max((entry['tokens'] for report in reports for entry in report['rows']), default=0)}
    path = output_dir / ('qualification' if qualification else 'production') / ('families_' + '_'.join(map(str, families)) + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('a completed family receipt changed; use a new output directory')
    path.write_bytes(data)
    return result


def main():
    """Require an explicit destination and retain source-scoped case exclusions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--families', type=int, nargs='+', default=list(range(1, 14)))
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--exclude-case-ids', nargs='*', default=[])
    args = parser.parse_args()
    if sorted(set(args.families)) != args.families or any(number not in range(1, 14) for number in args.families):
        parser.error('families must be unique ascending S2 family numbers')
    report = build_shards(args.output_dir, args.families, qualification=args.qualification, exclude=args.exclude_case_ids)
    print(json.dumps(report, indent=2))
    if report['shortfalls']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
