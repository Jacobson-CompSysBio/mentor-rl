#!/usr/bin/env python3
"""Build distinct training-only legacy replay under the visible S2 adapter."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_case_quotas import reference
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_corpus import _jsonl, _write_immutable
from runtime.world_model.s2_production import load_recipe, production_budget
from runtime.world_model.s2_production_replay import (
    load_replay_pool, adapt_replay_row, select_replay_rows, validate_replay_adapter,
)
from runtime.world_model.sources import file_sha256


def build_replay(output_dir, retained_cases):
    """Select capped type/condition quotas without replacement or body duplicates."""
    recipe = load_recipe()
    budget = production_budget(retained_cases, recipe)
    tokenizer = load_tokenizer(recipe['tokenizer'])
    reports, fingerprints = [], set()
    for stage in ('S1', 'S0'):
        pool, source = load_replay_pool(ROOT, stage)
        distinct, counts = [], Counter()
        for row in pool:
            adapted = adapt_replay_row(row, stage)
            fingerprint = model_trace_sha256(adapted)
            if fingerprint in fingerprints:
                counts['duplicate_model_trace'] += 1
                continue
            fingerprints.add(fingerprint)
            distinct.append(adapted)
        selected, selection = select_replay_rows(distinct, budget['replay'][stage], stage=stage, seed=recipe['seed'])
        checked = []
        for row in selected:
            validate_replay_adapter(row, stage)
            encoded = tokenize_trace(row, tokenizer)
            length, mask = len(encoded['input_ids']), encoded['completion_mask']
            if not length or length > 8192 or len(mask) != length or not any(mask):
                raise ValueError('selected historical replay exceeds the complete 8k budget')
            checked.append({'row_id': row['row_id'], 'tokens': length, 'assistant_tokens': sum(mask),
                'source_record_id': row['metadata']['source_record_id'],
                'model_trace_sha256': model_trace_sha256(row)})
            if len(checked) % 2000 == 0:
                print(f'{stage}: checked {len(checked)}/{len(selected)}', file=sys.stderr, flush=True)
        payload = _jsonl(selected)
        files = {'rows.jsonl': payload}
        pins = [reference(Path(__file__)), reference(ROOT / 'runtime/world_model/s2_production_replay.py'),
            reference(ROOT / 'runtime/world_model/s2_audit_identity.py'), reference(ROOT / 'runtime/world_model/s2_3_tokens.py'),
            reference(ROOT / 'runtime/world_model_training.py'), reference(ROOT / 'runtime/world_model/s2_production_v1_8k.json'),
            reference(ROOT / source['source_config'])]
        for item in source['sources']:
            pins.extend(({'path': item['path'], 'sha256': item['sha256']},
                         {'path': item['manifest_path'], 'sha256': item['manifest_sha256']}))
        if any(file_sha256(ROOT / pin['path']) != pin['sha256'] for pin in pins):
            raise ValueError('a replay input changed during selection')
        report = {'schema_version': 'mentor-rl-s2-production-replay-shard-v1', 'stage': stage,
            'retained_cases': retained_cases, 'source_pool': source, 'selection': selection,
            'removed_duplicates': dict(counts), 'rows': checked,
            'artifact': str(output_dir / stage.lower() / 'rows.jsonl'),
            'artifact_sha256': hashlib.sha256(payload).hexdigest(),
            'max_tokens': max(row['tokens'] for row in checked), 'input_pins': pins,
            'source_only_train': True, 'legacy_contract_visible': True, 'truncation_applied': False}
        files['report.json'] = (json.dumps(report, indent=2) + '\n').encode()
        _write_immutable(output_dir / stage.lower(), files)
        reports.append(report)
    return reports


def main():
    """Require an explicit retained case count and immutable replay destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--retained-cases', type=int, required=True)
    args = parser.parse_args()
    reports = build_replay(args.output_dir, args.retained_cases)
    print(json.dumps([{key: report[key] for key in ('stage', 'max_tokens', 'selection')}
                      for report in reports], indent=2))


if __name__ == '__main__':
    main()
