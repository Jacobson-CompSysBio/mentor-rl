#!/usr/bin/env python3
"""Replay source-backed S2 generator witnesses without changing published data.

The existing fitting candidate pool supplies stratified witnesses for S2.3–13.
The repaired S2.1/2 generators supply new numbered-case witnesses. This check
establishes generator behavior, not quota coverage or production readiness.
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
from runtime.world_model.s2_degrees import DegreeIndex
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_paths import PathIndex
from runtime.world_model.s2_production_cases import CATALOG_PATH, CASE_HEADING, FINAL_ANSWER, load_catalog
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.s2_subgraphs import SubgraphIndex


SPLITS = ('train', 'validation', 'test')
FEATURES = ('question_kind', 'request_kind', 'claim', 'graph_claim', 'truth_scope',
            'graph_truth_scope', 'mode', 'direction', 'connectivity_kind', 'category', 'bucket')
POOL = ROOT / 'data/world_model_v4/diagnostics/s2_production_8k_candidates/eligible'


def sha256(path: Path) -> str:
    """Hash even large candidate files without retaining their bytes in memory."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def representative_rows(path: Path, pin: dict, split: str, family: str) -> list[dict]:
    """Cover every variant/framing pair and marginal task feature in one split.

    Prefer smaller faithful witnesses. This deliberately does not claim full
    Cartesian or numbered-case coverage for the historical candidate pools.
    """
    if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
        raise ValueError(f'candidate file differs from its pin: {path}')
    selected, seen, count = {}, set(), 0
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            meta = row['metadata']
            if (meta['split'] != split or meta['question_family'].replace('.single_edge', '') != family
                    or row['row_id'] in seen):
                raise ValueError(f'candidate identity or split changed: {path}')
            seen.add(row['row_id'])
            count += 1
            keys = [('variant_framing', meta['variant'], meta['framing'])]
            keys.extend((name, json.dumps(meta[name], sort_keys=True)) for name in FEATURES if name in meta)
            rank = (len(line), row['row_id'])
            for key in keys:
                if key not in selected or rank < selected[key][0]:
                    selected[key] = rank, row
    if count != pin['rows']:
        raise ValueError(f'candidate count differs from its pin: {path}')
    unique = {row['row_id']: row for _, row in selected.values()}
    return [unique[key] for key in sorted(unique)]


def check_tokens(row: dict, tokenizer) -> int:
    """Check the complete trace and nonempty current-turn supervision at 8k."""
    encoded = tokenize_trace(row, tokenizer)
    tokens = len(encoded['input_ids'])
    mask = encoded['completion_mask']
    if tokens > 8192 or len(mask) != tokens or not any(mask):
        raise ValueError(f'complete trace or supervision exceeds production scope: {row["row_id"]}')
    return tokens


def check_generators(output: Path, families: list[int], *, pool: Path = POOL,
                     supplemental: Path | None = None) -> dict:
    """Check pinned real sources, exact evidence replay, and actual tokenization."""
    recipe = json.loads((ROOT / 'runtime/world_model/s2_production_v1_8k.json').read_text())
    tokenizer = load_tokenizer(recipe['tokenizer'])
    manifest = json.loads((pool / 'manifest.json').read_text())
    records = {entry['family']: entry for entry in manifest['families']}
    catalog = {case.case_id: case for case in load_catalog()}
    catalog_text = CATALOG_PATH.read_text()
    headings = list(CASE_HEADING.finditer(catalog_text))
    answers = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(catalog_text)
        answers[heading.group(1)] = json.loads(FINAL_ANSWER.search(catalog_text, heading.end(), end).group(1))
    additions = json.loads(supplemental.read_text())['families'] if supplemental is not None else []
    views, indices, reports = {}, {}, []
    for number in families:
        print(f'checking S2.{number}', file=sys.stderr, flush=True)
        module_name = f's2_{number}_production' if number < 3 else f's2_{number}_corpus'
        module = importlib.import_module(f'runtime.world_model.{module_name}')
        contract = module.load_build_contract()
        source = ROOT / contract['source_manifest']['path']
        if sha256(source) != contract['source_manifest']['sha256']:
            raise ValueError(f'changed source manifest: {source}')
        if str(source) not in views:
            loader = load_source_view
            if number >= 12:
                from runtime.world_model.s2_12_sources import load_source_view as loader
            views[str(source)] = loader(source)
        view = views[str(source)]
        registry = load_registry_context(contract['registry_index'])
        bank = load_s2_prompt_banks(contract['prompt_bank'])
        index_class = (DegreeIndex if number == 8 else PathIndex if number == 11 else
                       SubgraphIndex if number in (9, 10, 12, 13) else NeighborIndex)
        index_key = str(source), index_class.__name__
        if index_key not in indices:
            indices[index_key] = index_class(view)
        kwargs = dict(contract=contract, registry=registry)
        if number == 1:
            kwargs['index'] = indices[index_key]
        if number >= 3:
            kwargs.update(bank=bank, index=indices[index_key])
        if number >= 12:
            previous = ROOT / contract['previous_source_manifest']['path']
            if sha256(previous) != contract['previous_source_manifest']['sha256']:
                raise ValueError('changed previous source manifest')
            if str(previous) not in views:
                views[str(previous)] = load_source_view(previous)
            kwargs['old_view'] = views[str(previous)]
        checked, lengths, variants, case_ids, split_reports = [], [], set(), set(), {}
        for split in SPLITS:
            if number < 3:
                rows = module.build_case_rows(view, contract, split=split, instances_per_case=1)
            else:
                record = records[f'S2.{number}']
                rows = representative_rows(pool / f's2_{number}' / f'{split}.jsonl',
                                           record['files'][f'{split}.jsonl'], split, f'S2.{number}')
                for addition in additions:
                    if addition['family'] != f'S2.{number}' or addition['split'] != split:
                        continue
                    artifact = ROOT / addition['artifact']
                    if (sha256(artifact) != addition['artifact_sha256']
                            or sha256(module.CONTRACT_PATH) != addition['contract_sha256']
                            or sha256(Path(module.__file__)) != addition['implementation_sha256']
                            or sha256(source) != addition['source_manifest_sha256']):
                        raise ValueError('supplemental generator witness inputs changed')
                    rows.extend(json.loads(line) for line in artifact.read_text().splitlines())
            split_variants, split_cases = set(), set()
            for row in rows:
                validator = module.validate_case_row if number < 3 else module.validate_row
                validator(row, view, **kwargs)
                lengths.append(check_tokens(row, tokenizer))
                meta = row['metadata']
                if meta['split'] != split:
                    raise ValueError('a generator moved a subject to another split')
                checked.append(row['row_id'])
                split_variants.add(meta['variant'])
                if meta.get('catalog_case_id'):
                    case = catalog[meta['catalog_case_id']]
                    final = row['messages'][-1]['content']
                    if ((final['status'], final['continuation_decision']) != (case.status, case.continuation_decision)
                            or case.framing_hint is not None and meta['framing'] != case.framing_hint):
                        raise ValueError(f'generator changed the approved case outcome: {case.case_id}')
                    if set(final['result']) != set(answers[case.case_id]['result']):
                        raise ValueError(f'generator changed the approved result fields: {case.case_id}')
                    split_cases.add(case.case_id)
            if len(checked) != len(set(checked)):
                raise ValueError('generator qualification requires distinct row IDs')
            if number < 3 and split_cases != set(module.CASE_BINDINGS):
                raise ValueError(f'numbered-case witnesses missing from S2.{number} {split}')
            variants.update(split_variants)
            case_ids.update(split_cases)
            split_reports[split] = {'rows': len(rows), 'variants': sorted(split_variants),
                                    'numbered_cases': sorted(split_cases),
                                    'unwitnessed_declared_variants': sorted(set(getattr(module, 'VARIANTS', ())) - split_variants)}
            if number < 3:
                artifact = output.parent / 'witnesses' / f's2_{number}' / f'{split}.jsonl'
                data = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                               for row in rows).encode()
                artifact.parent.mkdir(parents=True, exist_ok=True)
                if artifact.exists() and artifact.read_bytes() != data:
                    raise ValueError(f'preserve already generated witnesses; choose a new report directory: {artifact}')
                artifact.write_bytes(data)
                split_reports[split]['witness_file'] = str(artifact)
                split_reports[split]['witness_sha256'] = sha256(artifact)
        reports.append({'family': f'S2.{number}', 'checked_rows': len(checked), 'max_tokens': max(lengths),
                        'source_manifest_sha256': sha256(source),
                        'contract_sha256': sha256(module.CONTRACT_PATH),
                        'implementation_sha256': sha256(Path(module.__file__)),
                        'checked_row_ids_sha256': hashlib.sha256('\n'.join(checked).encode()).hexdigest(),
                        'numbered_cases': sorted(case_ids), 'splits': split_reports})
    report = {'schema_version': 'mentor-rl-s2-generator-checks-v1',
              'scope': 'fresh numbered-case witnesses for repaired early generators; stratified published witnesses for S2.3-13',
              'max_length': 8192, 'truncation_applied': False, 'source_replay_checked': True,
              'current_turn_supervision_checked': True, 'production_ready': False,
              'remaining_checks': ['Full numbered-case mapping and production quotas for all thirteen families.',
                                   'Final production and replay assembly with verified trainer/evaluator adapters.'],
              'candidate_manifest_sha256': sha256(pool / 'manifest.json'), 'families': reports,
              'supplemental_report_sha256': sha256(supplemental) if supplemental is not None else None,
              'qualification_implementation_sha256': sha256(Path(__file__)),
              'corrected_path_entrypoint_sha256': sha256(ROOT / 'scripts/build_world_model_s2_11_v2.py'),
              'checked_rows': sum(record['checked_rows'] for record in reports)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    """Require a report destination and permit focused family reruns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', type=int, choices=range(1, 14), default=list(range(1, 14)))
    parser.add_argument('--supplemental', type=Path, help='pinned fresh witness report for gaps in the published candidate selection')
    args = parser.parse_args()
    report = check_generators(args.output, list(dict.fromkeys(args.families)), supplemental=args.supplemental)
    print(json.dumps({'checked_rows': report['checked_rows'], 'families': len(report['families']),
                      'output': str(args.output), 'production_ready': False}, indent=2))


if __name__ == '__main__':
    main()
