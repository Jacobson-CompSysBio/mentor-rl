#!/usr/bin/env python3
"""Resume frozen shards and supply isolated sparse-case wording overlays.

The original builder and completed shards remain immutable. Existing shards
must still match every original dependency and their complete artifact hash;
adding the new helper to the loaded code set does not invalidate older cases.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_world_model_s2_production as original
from runtime.world_model import s2_production_case_3_19 as incoming_loop
from runtime.world_model import s2_production_case_4_22 as strongest_bound
from runtime.world_model.sources import file_sha256


_previous_generator = original.generator_module
_previous_context = original.build_context


def generator_module(family):
    """Select isolated extensions, importing S2.12 only when that family is built."""
    if family == 3:
        return incoming_loop
    if family == 4:
        return strongest_bound
    if family in (7, 8):
        from runtime.world_model import s2_production_context_7_10
        return s2_production_context_7_10
    if family in (9, 10):
        from runtime.world_model import s2_production_cases_9_10_v2
        return s2_production_cases_9_10_v2
    if family == 12:
        from runtime.world_model import s2_production_12_identity
        return s2_production_12_identity
    return _previous_generator(family)


def build_context(*args, **kwargs):
    """Retain the exhaustive rank support supplied by the degree generator."""
    module, context = _previous_context(*args, **kwargs)
    if '_features' in context:
        context['features'] = context['_features']
    return module, context


def verify_shard(directory, pins, quota):
    """Accept an unchanged completed shard under its original verified code pins."""
    report = json.loads((directory / 'report.json').read_text())
    if report.get('quota') != quota or len(report.get('rows', [])) != quota or not report.get('input_pins'):
        raise ValueError('a completed production shard has an invalid quota or receipt')
    for pin in report['input_pins']:
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'a completed shard input changed: {pin["path"]}')
    if file_sha256(directory / 'rows.jsonl') != report['artifact_sha256']:
        raise ValueError('a completed production shard changed')
    if report['case_id'] == incoming_loop.CASE_ID:
        recorded = {pin['path']: pin['sha256'] for pin in report['input_pins']}
        required = (*incoming_loop.INPUT_PATHS, str(Path(incoming_loop.__file__).relative_to(ROOT)))
        if any(recorded.get(path) != file_sha256(ROOT / path) for path in required):
            raise ValueError('the loop shard lacks the current pinned wording extension')
    if report['case_id'] == strongest_bound.CASE_ID:
        recorded = {pin['path']: pin['sha256'] for pin in report['input_pins']}
        required = (*strongest_bound.INPUT_PATHS, str(Path(strongest_bound.__file__).relative_to(ROOT)))
        if any(recorded.get(path) != file_sha256(ROOT / path) for path in required):
            raise ValueError('the ranking shard lacks the current pinned wording extension')
    return report


def _run(function, *args, **kwargs):
    """Scope the two substitutions to one explicit builder invocation."""
    previous_generator, previous_verify, previous_context = original.generator_module, original.verify_shard, original.build_context
    original.generator_module, original.verify_shard, original.build_context = generator_module, verify_shard, build_context
    try:
        return function(*args, **kwargs)
    finally:
        original.generator_module, original.verify_shard, original.build_context = previous_generator, previous_verify, previous_context


def build_shards(*args, **kwargs):
    """Use the original shard builder with the isolated incoming-loop extension."""
    return _run(original.build_shards, *args, **kwargs)


def main():
    """Expose the unchanged original command-line options through the new entrypoint."""
    _run(original.main)


if __name__ == '__main__':
    main()
