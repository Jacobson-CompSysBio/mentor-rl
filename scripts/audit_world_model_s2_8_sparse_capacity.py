#!/usr/bin/env python3
"""Prove source/split capacity for the reviewed degree-one outgoing hub case."""

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
    """Enumerate every outgoing scope and measure every qualifying source gene."""
    contract = load_build_contract()
    source = ROOT / contract['source_manifest']['path']
    view = load_source_view(source)
    index, domains, findings = DegreeIndex(view), [], []
    for scope, direction in scope_schedule(view):
        if direction != 'out':
            continue
        selected = MultiplexRuntime(view).resolve_scope(scope)
        population = index.population(selected)
        histogram = index.histogram(selected, direction)
        lower = 0
        for cell in histogram:
            equal, degree = cell['node_count'], cell['degree']
            rank = Fraction(50) if len(population) == 1 else Fraction(
                100 * (2 * lower + equal - 1), 2 * (len(population) - 1))
            if degree == 1 and rank >= 95:
                genes = sorted(gene for gene in population if index.degree(selected, gene, direction) == 1)
                counts = Counter(gene_group(gene, 20261028)[1] for gene in genes)
                findings.append({'layer_scope': scope, 'direction': direction, 'degree': 1,
                    'exact_percentile': {'numerator': rank.numerator, 'denominator': rank.denominator},
                    'population_count': len(population), 'lower_degree_count': lower, 'equal_degree_count': equal,
                    'genes_by_split': {split: [gene for gene in genes if gene_group(gene, 20261028)[1] == split]
                                       for split in ('train', 'validation', 'test')},
                    'counts_by_split': {split: counts[split] for split in ('train', 'validation', 'test')}})
            lower += equal
        domains.append({'layer_scope': scope, 'direction': direction, 'population_count': len(population),
                        'degree_histogram': histogram})
    unique = {split: sorted({gene for record in findings for gene in record['genes_by_split'][split]})
              for split in ('train', 'validation', 'test')}
    paths = [source, ROOT / 'runtime/world_model/s2_8_v1.json', Path(__file__),
        *[ROOT / f'runtime/world_model/{name}.py' for name in (
            's2_8_sampling', 's2_8_corpus', 's2_degrees', 's2_multiplex', 's2_neighbors',
            's2_sources', 's2_3_corpus', 's2_audit_cases_7_10')]]
    return {'schema_version': 'mentor-rl-s2-8-sparse-source-capacity-v1',
        'case_id': 'S2.8.30', 'graph_version': view.graph_version, 'selection_seed': 20261028,
        'all_valid_outgoing_domains_enumerated': True, 'domains_checked': len(domains),
        'qualifying_domains': findings, 'unique_qualifying_genes': unique,
        'unique_qualifying_gene_counts': {split: len(genes) for split, genes in unique.items()},
        'all_split_quotas_feasible': bool(unique['validation'] and unique['test']),
        'conclusion': 'Only one training gene satisfies degree=1, outgoing direction, and exact rank at least95; no validation or test witness exists.',
        'domains': domains, 'input_pins': [{'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)} for path in paths]}


if __name__ == '__main__':
    result = audit()
    output = ROOT / 'diagnostics/s2_production_coverage/s2_8_30_source_capacity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('the source proof already exists')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'path': str(output.relative_to(ROOT)),
                      'unique_qualifying_gene_counts': result['unique_qualifying_gene_counts']}, indent=2))
