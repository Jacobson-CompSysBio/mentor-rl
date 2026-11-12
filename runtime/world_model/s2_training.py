"""Bind mixed S2 training masks and exposure receipts to the frozen corpus.

Historical S0/S1 trainers remain untouched. This module records S2's multiple
family contracts and distinguishes unique training exposure from DP padding.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from runtime.world_model_training import V3_TRACE_FAMILIES, _v3_assistant_loss_annotations
from .s2_production_adapter import flatten_row, tokenize_row

S2_DATASET_ID = 's2_production_v1_8k'
S2_FAMILIES = frozenset(f'S2.{section}' for section in range(1, 14))
# These are the approved numbered-case maxima, including the five excluded cases.
S2_CASE_IDS = tuple(f'S2.{section}.{case}' for section, count in enumerate(
    (41, 22, 25, 25, 25, 25, 29, 31, 30, 34, 36, 36, 40), start=1)
    for case in range(1, count + 1))


def stable_hash(value: Any) -> str:
    """Hash an explicit JSON value with finite canonical encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_corpus_manifest(path: str | Path) -> dict[str, Any]:
    """Require the complete, untruncated S2 production corpus contract."""
    manifest = json.loads(Path(path).read_text())
    if (manifest.get('schema_version') != 'mentor-rl-s2-production-corpus-v1'
            or manifest.get('production_id') != S2_DATASET_ID
            or manifest.get('max_length') != 8192
            or manifest.get('truncation_applied') is not False
            or set(manifest.get('contract_routes', {})) != S2_FAMILIES):
        raise ValueError('The S2 corpus manifest has an unsupported identity or wire budget')
    return manifest


def tokenize_training_row(row, tokenizer, *, expected_split, contract_routes,
                          segment_loss_telemetry=True):
    """Keep the complete production wire and annotate only current-turn loss."""
    encoded = tokenize_row(row, tokenizer, expected_split=expected_split,
        contract_routes=contract_routes)
    if segment_loss_telemetry:
        segments, fields = _v3_assistant_loss_annotations(row, tokenizer,
            encoded['input_ids'], encoded['completion_mask'])
        encoded['loss_segments'], encoded['loss_field_masks'] = segments, fields
    return encoded


def validate_question_families(dataset_id, families, *, label):
    """Require all thirteen S2 families and the complete replay family set."""
    if dataset_id != S2_DATASET_ID:
        raise ValueError('Unknown S2 dataset identity')
    expected = S2_FAMILIES | V3_TRACE_FAMILIES if label == 'training' else S2_FAMILIES
    if frozenset(families) != expected:
        raise ValueError(f'Wrong {label} family coverage: {sorted(set(families) ^ expected)}')


def build_training_exposure_manifest(**args):
    """Record ordered logical rows and explicit deterministic padding separately."""
    status = args['status']
    completed = args.get('completed_global_step')
    total = args['total_steps']
    if status not in ('planned', 'snapshot', 'complete'):
        raise ValueError('Unknown exposure status')
    if (status == 'planned' and completed is not None
            or status == 'snapshot' and (type(completed) is not int or not 0 < completed <= total)
            or status == 'complete' and completed != total):
        raise ValueError('Exposure status disagrees with completed optimizer steps')
    rows = args['record_ids']
    n = len(rows)
    columns = ('record_ids', 'fact_ids', 'question_families', 'prompt_form_ids', 'source_stages')
    if not n or len(set(rows)) != n or any(len(args[name]) != n for name in columns):
        raise ValueError('Exposure identities must be aligned and unique')
    if set(args['prompt_form_ids']) != {'train'} or set(args['source_stages']) - {'S0', 'S1', 'S2'}:
        raise ValueError('Exposure includes a non-training row or unknown source stage')
    if set(args['question_families']) - (S2_FAMILIES | V3_TRACE_FAMILIES):
        raise ValueError('Exposure includes an unknown family')
    logical = list(args['consumed_indices'])
    padding = list(args['distributed_padding_indices'])
    if not logical or any(type(i) is not int or i < 0 or i >= n for i in logical + padding):
        raise ValueError('Exposure references an invalid training index')
    complete_coverage = set(logical) == set(range(n))
    if status != 'snapshot' and not complete_coverage:
        raise ValueError('Production training must expose every eligible row')
    if args['dataset_id'] != S2_DATASET_ID or args['exposure_scope'] != 'all_eligible_train_rows':
        raise ValueError('Exposure requires the frozen complete S2 production dataset')
    identity_keys = ('run_config_sha256', 'corpus_manifest_sha256', 'train_sha256',
        'tokenizer_arm_manifest_sha256', 'tokenizer_manifest_sha256')
    identity = {key: args[key] for key in identity_keys}
    for value in identity.values():
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
            raise ValueError('Exposure requires exact SHA-256 pins')
    def exposure(indices):
        return {'record_occurrences': len(indices),
            'record_sequence_sha256': stable_hash([rows[i] for i in indices]),
            'fact_sequence_sha256': stable_hash([args['fact_ids'][i] for i in indices]),
            'question_family_counts': dict(sorted(Counter(args['question_families'][i] for i in indices).items())),
            'source_stage_counts': dict(sorted(Counter(args['source_stages'][i] for i in indices).items()))}
    logical_receipt = exposure(logical)
    logical_receipt.update(unique_record_count=len(set(logical)),
        unique_fact_count=len({args['fact_ids'][i] for i in logical}),
        all_eligible_train_rows_exposed=complete_coverage)
    physical = exposure(logical + padding)
    physical['distributed_padding'] = exposure(padding) | {
        'policy': args['padding_policy'], 'included_in_logical_exposure': False,
        'record_ids': [rows[i] for i in padding]}
    payload = {'schema_version': 'mentor-rl-s2-training-exposure-v1', 'status': status,
        'run_id': args['run_id'], 'method_id': args['method_id'], 'identity': identity,
        'corpus': {'dataset_id': S2_DATASET_ID, 'eligible_train_rows': n,
            'record_sequence_sha256': stable_hash(rows),
            'family_contract_routes_sha256': args['family_contract_routes_sha256'],
            'source_stage_counts': dict(sorted(Counter(args['source_stages']).items()))},
        'schedule': {key: args[key] for key in ('seed', 'total_steps', 'num_train_epochs',
            'per_device_train_batch_size', 'gradient_accumulation_steps', 'data_parallel_size', 'preserve_order')},
        'exposure_contract': {'scope': 'all_eligible_train_rows',
            'all_eligible_train_rows_required': True, 'satisfied': complete_coverage},
        'logical_exposure': logical_receipt, 'physical_exposure': physical}
    payload['schedule'].update(completed_global_step=completed,
        global_batch_size=args['per_device_train_batch_size'] * args['gradient_accumulation_steps'] * args['data_parallel_size'])
    payload['manifest_sha256'] = stable_hash(payload)
    return payload


def capture_validation_snapshot(checkpoint_dir, output_dir, global_step, exposure_manifest_args):
    """Keep a small immutable adapter with its actual prefix exposure receipt."""
    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    args = dict(exposure_manifest_args)
    final = global_step == args['total_steps']
    if not final and not (checkpoint_dir / 'CHECKPOINT_COMPLETE').is_file():
        raise ValueError('A snapshot requires a complete checkpoint')
    state = json.loads((checkpoint_dir / 'training_state.json').read_text())
    if state['global_step'] != global_step or args['num_train_epochs'] != 1:
        raise ValueError('Snapshot step or single-epoch exposure changed')
    count = global_step * args['per_device_train_batch_size'] * args['gradient_accumulation_steps'] * args['data_parallel_size']
    args['consumed_indices'] = args['consumed_indices'][:count]
    if not final:
        args['distributed_padding_indices'] = []
    receipt = build_training_exposure_manifest(**args, status='complete' if final else 'snapshot', completed_global_step=global_step)
    destination = output_dir / 'diagnostic_snapshots' / f'step-{global_step:06d}'
    if destination.exists():
        metadata = json.loads((destination / 'snapshot_metadata.json').read_text())
        if metadata['exposure_sha256'] != stable_hash(receipt):
            raise ValueError('Refusing to replace an existing snapshot with different exposure')
        return destination
    temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
    temporary.mkdir(parents=True)
    names = ('adapter_config.json', 'adapter_model.safetensors', 'tp_adapter_manifest.json')
    for name in names:
        shutil.copyfile(checkpoint_dir / name, temporary / name)
    (temporary / 'training_exposure.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    metadata = {'schema_version': 'mentor-rl-s2-inference-snapshot-v1',
        'global_step': global_step, 'diagnostic_only': not final, 'promotion_eligible': final,
        'exposure_sha256': stable_hash(receipt), 'files': {
            name: hashlib.sha256((temporary / name).read_bytes()).hexdigest()
            for name in (*names, 'training_exposure.json')}}
    (temporary / 'snapshot_metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    (temporary / 'SNAPSHOT_COMPLETE').write_text(stable_hash(metadata) + '\n')
    for path in temporary.iterdir():
        path.chmod(0o444)
    temporary.rename(destination)
    destination.chmod(0o555)
    return destination
