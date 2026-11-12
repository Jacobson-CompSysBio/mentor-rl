#!/usr/bin/env python3
"""Prove fixed-split source capacity for the reviewed near-95 rounding case."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_8_corpus import load_build_contract
from runtime.world_model.s2_8_sampling import scope_schedule
from runtime.world_model.s2_degrees import DegreeIndex
from runtime.world_model.s2_multiplex import MultiplexRuntime
from runtime.world_model.s2_sources import load_source_view
from runtime.world_model.sources import file_sha256


def audit():
    """Recompute every valid population without moving or filtering its genes."""
    contract = load_build_contract()
    source = ROOT / contract['source_manifest']['path']
    view = load_source_view(source)
    index = DegreeIndex(view)
    runtime = MultiplexRuntime(view)
    prior_path = ROOT / 'data/world_model_v4/diagnostics/s2_8_builder_v1/audit.json'
    prior = json.loads(prior_path.read_text())['selection']['domains']
    schedule = scope_schedule(view)
    if len(schedule) != len(prior):
        raise ValueError('the exhaustive valid domain schedule changed')
    findings, domains = [], []
    for ordinal, (scope, direction) in enumerate(schedule, 1):
        selected = runtime.resolve_scope(scope)
        population = index.population(selected)
        histogram = index.histogram(selected, direction)
        expected = next(item for item in prior
            if item['layer_scope'] == scope and item['direction'] == direction)
        if len(population) != expected['population_count'] or histogram != expected['degree_histogram']:
            raise ValueError('a full live source distribution differs from its frozen audit')
        lower, near = 0, []
        for cell in histogram:
            degree, equal = cell['degree'], cell['node_count']
            rank = Fraction(50) if len(population) == 1 else Fraction(
                100 * (2 * lower + equal - 1), 2 * (len(population) - 1))
            if Fraction(18999, 200) < rank < 95:
                genes = sorted(gene for gene in population if index.degree(selected, gene, direction) == degree)
                counts = Counter(gene_group(gene, 20261028)[1] for gene in genes)
                record = {'layer_scope': scope, 'direction': direction, 'degree': degree,
                    'exact_percentile': {'numerator': rank.numerator, 'denominator': rank.denominator},
                    'population_count': len(population), 'lower_degree_count': lower,
                    'equal_degree_count': equal,
                    'genes_by_split': {split: [gene for gene in genes if gene_group(gene, 20261028)[1] == split]
                                       for split in ('train', 'validation', 'test')},
                    'counts_by_split': {split: counts[split] for split in ('train', 'validation', 'test')}}
                near.append(record)
                findings.append(record)
            lower += equal
        domains.append({'layer_scope': scope, 'direction': direction, 'population_count': len(population),
                        'degree_histogram': histogram, 'qualifying_degree_values': len(near)})
        print(f'validated full degree domain {ordinal}/{len(schedule)}', file=sys.stderr, flush=True)
    unique = {split: sorted({gene for record in findings for gene in record['genes_by_split'][split]})
              for split in ('train', 'validation', 'test')}
    inputs = [source, prior_path, ROOT / 'runtime/world_model/s2_8_v1.json', Path(__file__),
        *[ROOT / f'runtime/world_model/{name}.py' for name in (
            's2_8_sampling', 's2_8_corpus', 's2_degrees', 's2_multiplex', 's2_neighbors',
            's2_sources', 's2_3_corpus', 's2_audit_cases_7_10')]]
    return {'schema_version': 'mentor-rl-s2-8-rounding-source-capacity-v1',
        'case_id': 'S2.8.29', 'graph_version': view.graph_version,
        'selection_seed': 20261028, 'predicate': '18999/200 < exact percentile < 95; threshold=95; hub_like=false',
        'all_valid_domains_enumerated': True, 'domains_checked': len(domains),
        'all_live_distributions_equal_frozen_audit': True,
        'qualifying_domains': findings, 'unique_qualifying_genes': unique,
        'unique_qualifying_gene_counts': {split: len(genes) for split, genes in unique.items()},
        'validation_source_capacity': len(unique['validation']),
        'all_split_quotas_feasible': bool(unique['validation']),
        'conclusion': 'The pinned graph and fixed subject splits provide no validation witness for this exact case.',
        'domains': domains,
        'input_pins': [{'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)} for path in inputs]}


if __name__ == '__main__':
    report = audit()
    output = ROOT / 'diagnostics/s2_production_coverage/s2_8_29_source_capacity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('the source proof already exists')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'path': str(output.relative_to(ROOT)),
                      'unique_qualifying_gene_counts': report['unique_qualifying_gene_counts']}, indent=2))
