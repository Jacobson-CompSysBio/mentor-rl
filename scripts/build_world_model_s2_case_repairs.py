#!/usr/bin/env python3
"""Generate and qualify the thirteen exact generator combinations from the audit.

Each repaired case retains the reviewed behavior, source split, and original
evidence. The output is a small source-backed qualification pool, not the final
512/64/64 production allocation or permission to launch training.
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

from scripts.audit_world_model_s2_case_quotas import catalog_inputs, load_bindings, reference
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_audit_source import SourceFeatures
from runtime.world_model.s2_corpus import _jsonl, _write_immutable
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.sources import file_sha256


SPLITS = ('train', 'validation', 'test')
FAMILIES = (4, 5, 8, 9, 10)


def frozen_audit():
    """Check the historical audit against its already published receipt."""
    receipt_path = ROOT / 'diagnostics/s2_case_quota_audit/summary.json'
    receipt = json.loads(receipt_path.read_text())
    pins = [reference(receipt_path)]
    for field in ('authoritative_report', 'case_matrix', 'allocation', 'validated_witnesses'):
        pin = receipt[field]
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'the frozen audit differs from its published receipt: {field}')
        pins.append(pin)
    return json.loads((ROOT / receipt['authoritative_report']['path']).read_text()), pins


def check_case(row, case, expected_answer, binding, matcher, features, tokenizer):
    """Require the frozen exact case condition and a complete supervised 8k wire."""
    meta, final = row['metadata'], row['messages'][-1]['content']
    if (meta.get('catalog_case_id') != case.case_id or meta['question_family'] != case.family
            or meta['variant'] not in binding['variants'] or final['status'] != case.status
            or final['continuation_decision'] != case.continuation_decision
            or case.framing_hint is not None and meta['framing'] != case.framing_hint
            or set(final['result']) != set(expected_answer['result'])):
        raise ValueError(f'the repaired row changes its approved case: {case.case_id}')
    classified = dict(row)
    classified['_audit_source'] = features.for_row(row)
    if not matcher.matches(case.case_id, classified):
        raise ValueError(f'the repair does not satisfy the frozen audit requirements: {case.case_id}')
    encoded = tokenize_trace(row, tokenizer)
    length, mask = len(encoded['input_ids']), encoded['completion_mask']
    if not length or length > 8192 or len(mask) != length or not any(mask) or any(value not in (0, 1) for value in mask):
        raise ValueError('a repaired complete trace or supervision mask exceeds the approved scope')
    return {'case_id': case.case_id, 'row_id': row['row_id'], 'split': meta['split'],
            'tokens': length, 'assistant_tokens': sum(mask),
            'model_trace_sha256': model_trace_sha256(row)}


def build_repairs(output_dir):
    """Write immutable real-source witnesses and exact validator/input pins."""
    catalog, answers, catalog_pin = catalog_inputs()
    cases = {case.case_id: case for case in catalog}
    bindings, matchers, binding_pins = load_bindings(catalog)
    baseline, audit_pins = frozen_audit()
    wanted = {entry['case_id'] for entry in baseline['case_entries'] if entry['binding'].get('implementation_gap')}
    if len(wanted) != 13:
        raise ValueError('the frozen thirteen-case repair scope changed')
    recipe_path = ROOT / 'runtime/world_model/s2_production_v1_8k.json'
    recipe = json.loads(recipe_path.read_text())
    if recipe['max_length'] != 8192 or recipe['truncation_allowed'] is not False:
        raise ValueError('repairs require the approved complete-trace 8k scope')
    tokenizer = load_tokenizer(recipe['tokenizer'])
    views, indices, files, reports, observed = {}, {}, {}, [], Counter()
    pins = [*audit_pins, reference(recipe_path), reference(Path(__file__)), *binding_pins,
            *[reference(ROOT / path) for path in ('scripts/audit_world_model_s2_case_quotas.py',
                'runtime/world_model/s2_audit_source.py', 'runtime/world_model/s2_audit_identity.py',
                'runtime/world_model/s2_3_tokens.py', 'runtime/world_model_training.py')]]
    for number in FAMILIES:
        module = importlib.import_module(f'runtime.world_model.s2_{number}_case_repairs')
        base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
        expected = {case_id for case_id in wanted if cases[case_id].family == f'S2.{number}'}
        if set(module.CASE_IDS) != expected:
            raise ValueError(f'the S2.{number} repair constructor changes its exact case scope')
        loader = getattr(module, 'load_build_contract', base.load_build_contract)
        contract = loader()
        source = ROOT / contract['source_manifest']['path']
        if file_sha256(source) != contract['source_manifest']['sha256']:
            raise ValueError('a source manifest differs from the selected repair contract')
        if source not in views:
            views[source] = load_source_view(source)
            indices[source] = SubgraphIndex(views[source])
        view, index = views[source], indices[source]
        bank = load_s2_prompt_banks(contract['prompt_bank'])
        registry = load_registry_context(contract['registry_index'])
        features = SourceFeatures(view, index, seed=recipe['seed'])
        module_path = Path(module.__file__)
        pins.extend((reference(module_path), reference(Path(base.__file__)), reference(base.CONTRACT_PATH), reference(source)))
        manifest_path = ROOT / f'data/world_model_v4/diagnostics/s2_{number}_builder_v1/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        pins.append(reference(manifest_path))
        pins.extend(reference(ROOT / path) for path in manifest['implementation_sha256'])
        for dependency in module_path.parent.glob(module_path.stem + '*.json'):
            pins.append(reference(dependency))
        request_pin = getattr(module, 'REQUEST_BANK', None)
        if isinstance(request_pin, dict) and 'path' in request_pin:
            request_path = ROOT / request_pin['path']
            if file_sha256(request_path) != request_pin['sha256']:
                raise ValueError('the repaired request wording differs from its pin')
            pins.append(reference(request_path))
        for split in SPLITS:
            print(f'building S2.{number} exact repairs ({split})', file=sys.stderr, flush=True)
            rows, selection = module.build_case_rows(view, index, bank, contract, registry,
                                                     split=split, seed=recipe['seed'])
            if len(rows) != len(expected) or {row['metadata'].get('catalog_case_id') for row in rows} != expected:
                raise ValueError('every requested case requires one fresh witness per fixed split')
            checks = []
            for row in rows:
                if row['metadata']['split'] != split or row['metadata'].get('selection_seed') != recipe['seed']:
                    raise ValueError('a repair moved a source subject to another split')
                module.validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
                case_id = row['metadata']['catalog_case_id']
                check = check_case(row, cases[case_id], answers[case_id], bindings[case_id],
                                   matchers[case_id], features, tokenizer)
                checks.append(check)
                observed[check['model_trace_sha256']] += 1
            name = f's2_{number}/{split}.jsonl'
            files[name] = _jsonl(rows)
            reports.append({'family': f'S2.{number}', 'split': split, 'case_ids': sorted(expected),
                'artifact': str(output_dir / name), 'artifact_sha256': hashlib.sha256(files[name]).hexdigest(),
                'rows': checks, 'selection': selection,
                'generator': reference(module_path), 'validator': reference(module_path),
                'contract_version': contract['contract_version'], 'source_manifest': reference(source)})
    if len(observed) != 39 or any(count != 1 for count in observed.values()):
        raise ValueError('the repair witnesses must be 39 distinct model traces')
    pins = list({pin['path']: pin for pin in pins}.values())
    for pin in pins:
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'a repair input changed during generation: {pin["path"]}')
    if catalog_inputs()[2] != catalog_pin:
        raise ValueError('the reviewed catalog changed during generation')
    report = {'schema_version': 'mentor-rl-s2-thirteen-case-repairs-v1',
        'catalog': catalog_pin, 'repaired_case_ids': sorted(wanted), 'families': reports,
        'checked_rows': sum(len(entry['rows']) for entry in reports),
        'max_tokens': max(row['tokens'] for entry in reports for row in entry['rows']),
        'source_replay_checked': True, 'frozen_audit_predicates_checked': True,
        'current_turn_supervision_checked': True, 'source_splits_preserved': True,
        'truncation_applied': False, 'input_pins': pins,
        'training_ready': False, 'production_export_allowed': False,
        'remaining_work': ['Production case quotas, remaining sample coverage, corpus/replay assembly, and training/evaluation adapters.']}
    files['report.json'] = (json.dumps(report, indent=2) + '\n').encode()
    _write_immutable(output_dir, files)
    return report


def main():
    """Require an explicit immutable witness destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = build_repairs(args.output_dir)
    print(json.dumps({'repaired_cases': len(report['repaired_case_ids']), 'checked_rows': report['checked_rows'],
                      'max_tokens': report['max_tokens']}, indent=2))


if __name__ == '__main__':
    main()
