#!/usr/bin/env python3
"""Measure complete adapted replay pools before production quota allocation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import reference
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_production import load_recipe
from runtime.world_model.s2_production_replay import load_replay_pool, adapt_replay_row
from runtime.world_model.s2_corpus import _write_immutable
from runtime.world_model.sources import file_sha256


def audit(replay_root, output_dir):
    """Reuse immutable measured bodies and measure every remaining source row."""
    recipe = load_recipe()
    tokenizer = load_tokenizer(recipe['tokenizer'])
    stages, pins = {}, [reference(Path(__file__)), reference(ROOT / 'runtime/world_model/s2_production_replay.py'),
        reference(ROOT / 'runtime/world_model/s2_3_tokens.py'), reference(ROOT / 'runtime/world_model/s2_audit_identity.py'),
        reference(ROOT / 'runtime/world_model_training.py'), reference(ROOT / 'runtime/world_model/s2_production_v1_8k.json')]
    for stage in ('S1', 'S0'):
        previous_path = replay_root / stage.lower() / 'report.json'
        previous = json.loads(previous_path.read_text())
        if file_sha256(ROOT / previous['artifact']) != previous['artifact_sha256']:
            raise ValueError('the measured replay artifact changed')
        cache = {row['model_trace_sha256']: row for row in previous['rows']}
        pins.extend([reference(previous_path), *previous['input_pins']])
        pool, source = load_replay_pool(ROOT, stage)
        checked, seen = [], set()
        for row in pool:
            adapted = adapt_replay_row(row, stage)
            fingerprint = model_trace_sha256(adapted)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            check = cache.get(fingerprint)
            if check is None:
                encoded = tokenize_trace(adapted, tokenizer)
                length, mask = len(encoded['input_ids']), encoded['completion_mask']
                if not length or len(mask) != length or not any(mask) or any(value not in (0, 1) for value in mask):
                    raise ValueError('invalid historical whole-trace supervision')
                check = {'row_id': adapted['row_id'], 'tokens': length, 'assistant_tokens': sum(mask),
                         'model_trace_sha256': fingerprint}
            checked.append(check)
            if len(checked) % 5000 == 0:
                print(f'{stage}: {len(checked)}/{len(pool)} complete replay bodies', file=sys.stderr, flush=True)
        stages[stage] = {'source_pool': source, 'rows': checked,
            'eligible_rows': sum(row['tokens'] <= 8192 for row in checked),
            'overlength_rows': sum(row['tokens'] > 8192 for row in checked),
            'max_tokens': max(row['tokens'] for row in checked), 'raw_source_rows': len(pool)}
    pins = list({pin['path']: pin for pin in pins}.values())
    if any(file_sha256(ROOT / pin['path']) != pin['sha256'] for pin in pins):
        raise ValueError('a replay length-audit input changed')
    report = {'schema_version': 'mentor-rl-s2-full-replay-length-audit-v1', 'stages': stages,
              'input_pins': pins, 'max_length': 8192, 'truncation_applied': False}
    _write_immutable(output_dir, {'report.json': (json.dumps(report, indent=2) + '\n').encode()})
    return report


def main():
    """Publish full-pool eligibility separately from selected replay budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.replay_root, args.output_dir)
    print(json.dumps({stage: {key: item[key] for key in ('raw_source_rows', 'eligible_rows', 'overlength_rows', 'max_tokens')}
                      for stage, item in report['stages'].items()}, indent=2))


if __name__ == '__main__':
    main()
