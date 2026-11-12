#!/usr/bin/env python3
"""Build source-backed witnesses for the two remaining S2 context gaps.

The S2.9 extension validates a genuine later native neighbor page. S2.10
uses the existing component generator on a larger source-selected motif.
Published corpora, splits, wording banks, and graph interpretations stay fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_3_tokens import load_tokenizer, tokenize_trace
from runtime.world_model.s2_corpus import _jsonl, _write_immutable
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex
from runtime.world_model.sources import file_sha256


REPAIRS = ((9, 'test', 'positive_loop_context', 's2_9_loop_context'),
           (10, 'validation', 'strong_witness_context', 's2_10_supplemental'))


def check_witness(row: dict, *, family: int, split: str, variant: str, tokenizer) -> dict:
    """Require the intended no-call proof, preserved scope, and complete 8k wire."""
    meta, messages = row['metadata'], row['messages']
    if (meta['question_family'] != f'S2.{family}' or meta['split'] != split
            or meta['variant'] != variant or any('tool_call' in message for message in messages)):
        raise ValueError('the witness changes its required family, split, context, or no-call behavior')
    answer = messages[-1]['content']
    if (answer['status'] != 'resolved' or answer['blocked_reason'] is not None
            or answer['continuation_decision'] != ('continue' if meta['framing'] == 'step' else 'stop')):
        raise ValueError('the saved evidence does not complete the requested task')
    context = json.loads(messages[1]['content'])
    observations = context['observations']
    if len(observations) != 1 or observations[0]['result']['collection']['complete'] is not False:
        raise ValueError('the witness must retain a genuinely incomplete original collection')
    result = answer['result']
    if family == 9:
        if (len(meta['gene_ids']) != 1 or len(result['edges']) != 1
                or result['edges'][0]['gene_ids'] != meta['gene_ids'] * 2):
            raise ValueError('the positive loop must prove exactly the requested singleton graph')
    elif (result['connectivity_kind'] != 'strong' or result['component_count'] != 1
          or result['layer_gene_count'] < 2 or result['single_component'] is not True):
        raise ValueError('the prefix must prove one strong group over the selected layer members')
    encoded = tokenize_trace(row, tokenizer)
    tokens = len(encoded['input_ids'])
    mask = encoded['completion_mask']
    if (tokens > 8192 or not tokens or len(mask) != tokens or not any(mask)
            or any(value not in (0, 1) for value in mask)):
        raise ValueError('the complete witness or current-turn supervision exceeds the approved scope')
    return {'row_id': row['row_id'], 'framing': meta['framing'], 'tokens': tokens,
            'assistant_tokens': sum(mask)}


def build_witnesses(output_dir: Path) -> dict:
    """Generate, independently replay, and immutably publish both gap witnesses."""
    recipe_path = ROOT / 'runtime/world_model/s2_production_v1_8k.json'
    recipe = json.loads(recipe_path.read_text())
    if recipe['max_length'] != 8192 or recipe['truncation_allowed'] is not False:
        raise ValueError('the witnesses require the approved untruncated 8k recipe')
    tokenizer = load_tokenizer(recipe['tokenizer'])
    views, indices, files, reports = {}, {}, {}, []
    for family, split, variant, repair_name in REPAIRS:
        print(f'building S2.{family} {split} {variant}', file=sys.stderr, flush=True)
        base = importlib.import_module(f'runtime.world_model.s2_{family}_corpus')
        repair = importlib.import_module(f'runtime.world_model.{repair_name}')
        contract = base.load_build_contract()
        source = ROOT / contract['source_manifest']['path']
        if file_sha256(source) != contract['source_manifest']['sha256']:
            raise ValueError('the native source manifest differs from its pin')
        if source not in views:
            views[source] = load_source_view(source)
            indices[source] = SubgraphIndex(views[source])
        view, index = views[source], indices[source]
        bank = load_s2_prompt_banks(contract['prompt_bank'])
        registry = load_registry_context(contract['registry_index'])
        rows, selection = repair.build_rows(view, index, bank, contract, registry,
                                            split=split, seed=recipe['seed'])
        if len(rows) != 2 or {row['metadata']['framing'] for row in rows} != {'direct', 'step'}:
            raise ValueError('each repaired cell requires its direct and workflow witnesses')
        validator = getattr(repair, 'validate_row', base.validate_row)
        lengths = []
        for row in rows:
            validator(row, view, index=index, bank=bank, contract=contract, registry=registry)
            lengths.append(check_witness(row, family=family, split=split, variant=variant, tokenizer=tokenizer))
        name = f's2_{family}_{split}.jsonl'
        files[name] = _jsonl(rows)
        validator_path = Path(importlib.import_module(validator.__module__).__file__)
        reports.append({'family': f'S2.{family}', 'split': split, 'variant': variant,
            'rows': len(rows), 'row_lengths': lengths, 'selection': selection,
            'artifact': str(output_dir / name),
            'artifact_sha256': hashlib.sha256(files[name]).hexdigest(),
            'contract_path': str(base.CONTRACT_PATH.relative_to(ROOT)),
            'contract_sha256': file_sha256(base.CONTRACT_PATH),
            'source_manifest': str(source.relative_to(ROOT)), 'source_manifest_sha256': file_sha256(source),
            'base_generator': str(Path(base.__file__).relative_to(ROOT)),
            'base_generator_sha256': file_sha256(Path(base.__file__)),
            'selector_module': str(Path(repair.__file__).relative_to(ROOT)),
            'selector_sha256': file_sha256(Path(repair.__file__)),
            'validator_module': str(validator_path.relative_to(ROOT)),
            'validator_sha256': file_sha256(validator_path),
            'source_replay_passed': True, 'incomplete_context_proves_exact_result': True})
    report = {'schema_version': 'mentor-rl-s2-context-gap-witnesses-v1', 'seed': recipe['seed'],
        'purpose': 'Close two observed 8k generator context gaps with faithful native witnesses.',
        'recipe_sha256': file_sha256(recipe_path), 'tokenizer': recipe['tokenizer'],
        'entrypoint_sha256': file_sha256(Path(__file__)), 'max_length': 8192,
        'truncation_applied': False, 'source_splits_preserved': True,
        'current_turn_supervision_checked': True, 'families': reports,
        'checked_rows': sum(item['rows'] for item in reports),
        'max_tokens': max(row['tokens'] for item in reports for row in item['row_lengths']),
        'remaining_targeted_context_gaps': [], 'training_ready': False,
        'production_export_allowed': False,
        'remaining_work': ['Per-case production quotas, full corpus/replay assembly, and training/evaluation integration.']}
    files['report.json'] = (json.dumps(report, indent=2) + '\n').encode()
    _write_immutable(output_dir, files)
    return report


def main() -> None:
    """Require an explicit destination that can only be reproduced identically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = build_witnesses(args.output_dir)
    print(json.dumps({'checked_rows': report['checked_rows'], 'max_tokens': report['max_tokens'],
                      'report': str(args.output_dir / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
