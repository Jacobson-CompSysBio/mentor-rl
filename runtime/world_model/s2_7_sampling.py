"""Select real shared-neighbor facts and audit each declared pair population.

The readable pair audit conditions on assigned registry symbols and degrees
one through five. Separate domains expose empty sets, missing prerequisites,
mixed directions, repeated roles, and full-source degree extrema. The runtime
has no degree ceiling. Query subjects retain the S2.3 fixed gene split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

from .s2_2_identity import RegistryContext
from .s2_3_corpus import gene_group
from .s2_corpus import SPLITS, digest
from .s2_neighbors import NeighborIndex
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class SharedFact:
    """Two ordered source roles before wording, evidence, and fault contrasts."""

    split: str
    layer_id: str
    queries: tuple[tuple[str, str], tuple[str, str]]
    neighbor_counts: tuple[int | None, int | None]
    shared_count: int | None
    split_unit: str
    category: str = 'native_pair'


def neighbor_ids(index: NeighborIndex, layer_id: str, gene: str, direction: str) -> frozenset[str]:
    """Read one native set or the exact incoming/outgoing union from the source."""
    directions = ('in', 'out') if direction == 'both' else (direction,)
    return frozenset(neighbor for native in directions for neighbor, _ in index.neighbors(layer_id, gene, native))


def fact_from_queries(view: GraphView, index: NeighborIndex, layer_id: str,
                      queries: Sequence[Sequence[str]], seed: int = 20261028,
                      split: str | None = None, category: str = 'native_pair') -> SharedFact:
    """Measure an ordered pair without crossing the existing query-subject split."""
    if layer_id not in view.layers or len(queries) != 2:
        raise ValueError('a shared fact requires one registered layer and two roles')
    allowed = (*index.directions(layer_id), 'both') if view.layers[layer_id].directed else ('undirected',)
    if any(len(role) != 2 or not isinstance(role[0], str) or not GENE_ID_PATTERN.fullmatch(role[0])
           or role[1] not in allowed for role in queries):
        raise ValueError('a shared fact requires canonical IDs and available directions')
    roles = tuple((role[0], role[1]) for role in queries)
    splits = {gene_group(gene, seed)[1] for gene, _ in roles}
    if len(splits) != 1 or split is not None and split not in splits:
        raise ValueError('a shared pair cannot cross fixed subject splits')
    actual_split = next(iter(splits))
    sets = [None if gene not in view.multiplex_nodes else neighbor_ids(index, layer_id, gene, direction)
            for gene, direction in roles]
    counts = tuple(None if values is None else len(values) for values in sets)
    shared = None if any(values is None for values in sets) else len(sets[0] & sets[1])
    unit = digest({'seed': seed, 'split': actual_split, 'layer_id': layer_id, 'queries': sorted(roles)})
    return SharedFact(actual_split, layer_id, roles, counts, shared, unit, category)


def _adjacent(layer: Any, first: str, second: str) -> bool:
    """Test endpoint adjacency without changing the requested neighbor direction."""
    return ((first, second) in layer.edges or (second, first) in layer.edges) if layer.directed else (
        tuple(sorted((first, second))) in layer.edges)


class _Reservoir:
    """Keep a deterministic uniform sample for every declared source cell."""

    def __init__(self, quota: Mapping[str, int], seed: int, domain: str) -> None:
        self.quota, self.seed, self.domain = quota, seed, domain
        self.counts: Counter = Counter()
        self.samples: dict[tuple, list] = defaultdict(list)
        self.random: dict[tuple, random.Random] = {}

    def add(self, key: tuple, roles: tuple) -> None:
        self.counts[key] += 1
        quota, sample = self.quota[key[0]], self.samples[key]
        if len(sample) < quota:
            sample.append(roles)
            return
        if key not in self.random:
            self.random[key] = random.Random(int(digest([self.seed, self.domain, key]), 16))
        rng = self.random[key]
        rank = rng.randrange(self.counts[key])
        if rank < quota:
            sample[rank] = roles


def _pair_domain(view: GraphView, index: NeighborIndex, layer_id: str, direction: str,
                 genes: Sequence[str], quotas: Mapping[str, int], seed: int,
                 add: Any, category: str) -> dict:
    """Audit unordered distinct pairs with sparse positive intersection counts."""
    genes = tuple(sorted(genes))
    positions = {gene: rank for rank, gene in enumerate(genes)}
    sets = [neighbor_ids(index, layer_id, gene, direction) for gene in genes]
    groups = [gene_group(gene, seed)[1] for gene in genes]
    postings: dict[str, list[int]] = defaultdict(list)
    for rank, neighbors in enumerate(sets):
        for neighbor in sorted(neighbors):
            postings[neighbor].append(rank)
    domain = f'{category}:{layer_id}:{direction}'
    reservoir = _Reservoir(quotas, seed, domain)
    global_histogram: Counter = Counter()
    split_histograms = {split: Counter() for split in SPLITS}
    adjacent_pairs = {tuple(sorted((positions[a], positions[b]))) for a, b in view.layers[layer_id].edges
                      if a != b and a in positions and b in positions}
    adjacent_global: Counter = Counter()
    adjacent_split = {split: Counter() for split in SPLITS}
    witnesses = _Reservoir(quotas, seed, f'witnesses:{domain}')
    for first, neighbors in enumerate(sets):
        shared_counts: Counter = Counter()
        for neighbor in sorted(neighbors):
            for second in postings[neighbor]:
                if second > first:
                    shared_counts[second] += 1
        for second, count in sorted(shared_counts.items()):
            global_histogram[count] += 1
            adjacent = (first, second) in adjacent_pairs
            if adjacent:
                adjacent_global[count] += 1
            if groups[first] == groups[second]:
                split = groups[first]
                split_histograms[split][count] += 1
                adjacent_split[split][count] += int(adjacent)
                reservoir.add((split, count, adjacent), ((genes[first], direction), (genes[second], direction)))
                if direction != 'both':
                    for a, b in ((first, second), (second, first)):
                        if count == len(sets[a]) < len(sets[b]):
                            prefix = {g for g, _ in index.neighbors(layer_id, genes[b], direction)[:-1]}
                            if sets[a] <= prefix:
                                witnesses.add((split, 'containment'), ((genes[a], direction), (genes[b], direction)))
                    if count > 1 and all(len(sets[rank]) > 1 for rank in (first, second)):
                        tops = [index.neighbors(layer_id, genes[rank], direction)[0][0] for rank in (first, second)]
                        if tops[0] == tops[1]:
                            witnesses.add((split, 'bounded_common_lower_bound'), ((genes[first], direction), (genes[second], direction)))
    total = len(genes) * (len(genes) - 1) // 2
    global_histogram[0] = total - sum(global_histogram.values())
    zero_adjacent = []
    for first, second in sorted(adjacent_pairs):
        if not sets[first] & sets[second]:
            adjacent_global[0] += 1
            zero_adjacent.append((first, second))
            if groups[first] == groups[second]:
                split = groups[first]
                adjacent_split[split][0] += 1
                reservoir.add((split, 0, True), ((genes[first], direction), (genes[second], direction)))
    cells, omissions = [], []
    for split in SPLITS:
        ranks = [rank for rank, group in enumerate(groups) if group == split]
        pair_count = len(ranks) * (len(ranks) - 1) // 2
        split_histograms[split][0] = pair_count - sum(split_histograms[split].values())
        nonadjacent_zero = split_histograms[split][0] - adjacent_split[split][0]
        key = (split, 0, False)
        reservoir.counts[key] = nonadjacent_zero
        needed = min(quotas[split], nonadjacent_zero)
        chosen = set()
        rng = random.Random(int(digest([seed, domain, split, 'zero']), 16))
        attempts = 0
        while len(chosen) < needed and attempts < max(1000, 20 * needed):
            first, second = sorted(rng.sample(ranks, 2))
            attempts += 1
            if (first, second) not in adjacent_pairs and not sets[first] & sets[second]:
                chosen.add((first, second))
        if len(chosen) < needed:
            # This path guarantees completion even when a zero cell is rare.
            for first, second in combinations(ranks, 2):
                if (first, second) not in adjacent_pairs and not sets[first] & sets[second]:
                    chosen.add((first, second))
                    if len(chosen) == needed:
                        break
        reservoir.samples[key] = [((genes[a], direction), (genes[b], direction)) for a, b in sorted(chosen)]
        for count in sorted(global_histogram):
            for adjacent in (False, True):
                key = (split, count, adjacent)
                population = (adjacent_split[split][count] if adjacent else
                              split_histograms[split][count] - adjacent_split[split][count])
                sample = reservoir.samples[key]
                cell = {'split': split, 'shared_count': count, 'adjacent': adjacent,
                        'population': population, 'requested': quotas[split], 'selected': len(sample)}
                cells.append(cell)
                if len(sample) < quotas[split]:
                    omissions.append({**cell, 'reason': 'insufficient pairs in this fixed-split cell'})
                for roles in sample:
                    add(layer_id, roles, category)
        for feature in ('containment', 'bounded_common_lower_bound'):
            key = (split, feature)
            for roles in witnesses.samples[key]:
                add(layer_id, roles, feature)
    return {'category': category, 'layer_id': layer_id, 'directions': [direction, direction],
            'candidate_policy': 'registry IDs with assigned symbols and requested set size one through five',
            'candidate_genes': len(genes), 'candidate_pairs': total,
            'histogram': dict(sorted(global_histogram.items())),
            'adjacent_histogram': dict(sorted(adjacent_global.items())),
            'split_candidate_genes': dict(Counter(groups)),
            'split_histograms': {s: dict(sorted(h.items())) for s, h in split_histograms.items()},
            'cross_split_pairs': total - sum(sum(h.values()) for h in split_histograms.values()),
            'cells': cells, 'omissions': omissions,
            'witness_cells': [{'split': split, 'feature': feature, 'population': witnesses.counts[(split, feature)],
                               'selected': len(witnesses.samples[(split, feature)])}
                              for split in SPLITS for feature in ('containment', 'bounded_common_lower_bound')],
            'audit_method': 'inverted source-neighbor postings enumerate positive intersections; zero counts equal all pairs minus positive pairs'}


def _mixed_domain(view: GraphView, index: NeighborIndex, layer_id: str, eligible: Sequence[str],
                  quotas: Mapping[str, int], seed: int, add: Any) -> dict:
    """Audit ordered out/in pairs under the same explicit readable conditioning."""
    first = [g for g in eligible if 1 <= len(index.neighbors(layer_id, g, 'out')) <= 5]
    second = [g for g in eligible if 1 <= len(index.neighbors(layer_id, g, 'in')) <= 5]
    sets_out = {g: neighbor_ids(index, layer_id, g, 'out') for g in first}
    sets_in = {g: neighbor_ids(index, layer_id, g, 'in') for g in second}
    histogram: Counter = Counter()
    reservoir = _Reservoir(quotas, seed, f'mixed:{layer_id}')
    for a in first:
        for b in second:
            count = len(sets_out[a] & sets_in[b])
            histogram[count] += 1
            split = gene_group(a, seed)[1]
            if gene_group(b, seed)[1] == split:
                reservoir.add((split, count, _adjacent(view.layers[layer_id], a, b)), ((a, 'out'), (b, 'in')))
    cells = []
    for split in SPLITS:
        for count in sorted(histogram):
            for adjacent in (False, True):
                key = (split, count, adjacent)
                sample = reservoir.samples[key]
                cells.append({'split': split, 'shared_count': count, 'adjacent': adjacent,
                              'population': reservoir.counts[key], 'requested': quotas[split], 'selected': len(sample)})
                for roles in sample:
                    add(layer_id, roles, 'mixed_out_in')
    return {'category': 'mixed_out_in', 'layer_id': layer_id, 'directions': ['out', 'in'],
            'candidate_policy': 'ordered out/in roles; each registry ID has an assigned symbol and one through five requested neighbors; repeated IDs allowed',
            'role_populations': [len(first), len(second)], 'candidate_pairs': len(first) * len(second),
            'histogram': dict(sorted(histogram.items())), 'cells': cells,
            'cross_split_pairs': sum(histogram.values()) - sum(reservoir.counts.values())}


def select_facts(view: GraphView, index: NeighborIndex, registry: RegistryContext,
                 counts: Mapping[str, int], seed: int, *, smoke: bool = False) -> tuple[list[SharedFact], dict]:
    """Audit real pair domains, then select equal quotas within fixed gene splits.

    Both recipes retain full conditioned histograms. Smoke changes only the
    declared quota caller; full-source repeated-role extrema remain visible.
    No source neighbor, returned canonical ID, or native weight is removed.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    eligible = sorted(gene for gene, symbol in registry.symbols_by_id.items() if symbol is not None)
    registered = set(registry.symbols_by_id)
    global_present = sorted(registered & view.multiplex_nodes)
    global_missing = sorted(registered - view.multiplex_nodes)
    if not global_missing:
        raise ValueError('the registry has no globally absent candidates')
    facts: dict[tuple, SharedFact] = {}
    domains, features, omissions = [], [], []

    def add(layer: str, roles: tuple, category: str) -> None:
        fact = fact_from_queries(view, index, layer, roles, seed, category=category)
        facts.setdefault((layer, fact.queries), fact)

    for layer_id, layer in view.layers.items():
        native = index.directions(layer_id)
        directions = (*native, 'both') if layer.directed else native
        for direction in directions:
            readable = [g for g in eligible if 1 <= len(neighbor_ids(index, layer_id, g, direction)) <= 5]
            domains.append(_pair_domain(view, index, layer_id, direction, readable, counts, seed, add,
                                        'both_union_pair' if direction == 'both' else 'native_pair'))
            layer_genes = sorted(layer.nodes)
            full_histogram = Counter(len(neighbor_ids(index, layer_id, g, direction)) for g in layer_genes)
            repeated_cells = []
            for split in SPLITS:
                pool = sorted((g for g in layer_genes if gene_group(g, seed)[1] == split),
                              key=lambda g: gene_group(g, seed)[0])
                by_count: dict[int, list[str]] = defaultdict(list)
                for gene in pool:
                    by_count[len(neighbor_ids(index, layer_id, gene, direction))].append(gene)
                chosen_counts = sorted({*range(0, 6), *([min(by_count), max(by_count)] if by_count else [])})
                for count in chosen_counts:
                    sample = by_count[count][:counts[split]]
                    repeated_cells.append({'split': split, 'neighbor_count': count,
                                           'population': len(by_count[count]), 'selected': len(sample)})
                    for gene in sample:
                        add(layer_id, ((gene, direction), (gene, direction)), 'repeated_role')
                positive = [g for g in pool if len(neighbor_ids(index, layer_id, g, direction)) > 0]
                zero = [g for g in pool if not neighbor_ids(index, layer_id, g, direction)]
                absent = sorted((g for g in registered & (view.multiplex_nodes - layer.nodes)
                                 if gene_group(g, seed)[1] == split), key=lambda g: gene_group(g, seed)[0])
                missing = sorted((g for g in global_missing if gene_group(g, seed)[1] == split),
                                 key=lambda g: gene_group(g, seed)[0])
                present = sorted((g for g in global_present if gene_group(g, seed)[1] == split),
                                 key=lambda g: (len(neighbor_ids(index, layer_id, g, direction)), gene_group(g, seed)[0]))
                for category, left, right in (('zero_direction', zero, positive),
                    ('outside_layer', absent, positive), ('outside_multiplex', missing, present)):
                    population = len(left) * len(right)
                    sample = min(counts[split], population)
                    for rank in range(sample):
                        add(layer_id, ((left[rank % len(left)], direction), (right[rank // len(left)], direction)), category)
                    features.append({'category': category, 'layer_id': layer_id, 'direction': direction,
                                     'split': split, 'role_populations': [len(left), len(right)],
                                     'candidate_pairs': population, 'shared_count': None if category == 'outside_multiplex' else 0,
                                     'requested': counts[split], 'selected': sample})
                    if sample < counts[split]:
                        omissions.append({**features[-1], 'reason': 'insufficient source pairs in the declared fixed-split domain'})
                for feature in ('self_loop', 'same_gene_mixed_directions'):
                    candidates = ([g for g in pool if g in neighbor_ids(index, layer_id, g, direction)]
                                  if feature == 'self_loop' else pool if layer.directed and direction == 'in' else [])
                    sample = candidates[:counts[split]]
                    if feature == 'self_loop':
                        for gene in sample:
                            add(layer_id, ((gene, direction), (gene, direction)), 'self_loop')
                    else:
                        for gene in sample:
                            add(layer_id, ((gene, 'out'), (gene, 'in')), feature)
                    features.append({'category': feature, 'layer_id': layer_id, 'direction': direction,
                                     'split': split, 'candidate_genes': len(candidates), 'selected': len(sample)})
            features.append({'category': 'repeated_role', 'layer_id': layer_id, 'direction': direction,
                             'candidate_policy': 'all actual layer members, including canonical source IDs absent from the registry; assigned symbols are not required',
                             'histogram': dict(sorted(full_histogram.items())), 'cells': repeated_cells,
                             'unselected_count_values': sorted(set(full_histogram) - {
                                 c['neighbor_count'] for c in repeated_cells if c['selected']})})
        if layer.directed:
            domains.append(_mixed_domain(view, index, layer_id, eligible, counts, seed, add))
    selected = sorted(facts.values(), key=lambda f: (SPLITS.index(f.split), view.layer_order.index(f.layer_id),
                                                    f.category, f.split_unit, f.queries))
    return selected, {'domains': domains, 'feature_domains': features, 'omissions': omissions,
        'selected_facts': len(selected), 'distinct_query_genes': len({g for f in selected for g, _ in f.queries}),
        'fact_selection_sha256': digest([asdict(fact) for fact in selected]),
        'split_policy': 'both ordered query subjects use the same SHA256(seed,gene) split from S2.3 through S2.6; role swaps retain a sorted two-role split unit',
        'edge_fact_disjoint': False,
        'edge_overlap_note': 'returned neighbors and native graph facts can occur in multiple splits and earlier S2 panels',
        'conditioning': 'equal quota per attainable shared-count/adjacency cell within each declared orientation and fixed split; independent of wording',
        'runtime_degree_cap': None, 'readable_conditioning': [1, 5], 'smoke': smoke,
        'remaining_domains': ['all unrestricted distinct pairs beyond readable conditioning',
            'all mixed-direction combinations beyond ordered out/in and both/both unions',
            'production allocation across all full-source shared-count values',
            'additional pinned graph versions, PEN/other layers, and explicit isolated-node inventories']}


def verify_approved_traces(document: Path, view: GraphView, index: NeighborIndex,
                           registry: RegistryContext, *, gtf_path: Path | None = None,
                           gtf_sha256: str | None = None) -> dict:
    """Check all approved S2.7 source records without assuming uniform harness pages.

    The report checks native weights, membership, totals, intersections, and
    coverage partitions. Opaque cursor boundaries and injected failures remain
    harness conditions. Incompatible graph versions supply no source truth.
    """
    text = Path(document).read_text()
    section = text[text.index('# S2.7:'):text.index('# S2.8:')]
    matches = list(re.finditer(r'^### S2\.7\.(\d+):', section, re.M))
    if [int(match[1]) for match in matches] != list(range(1, 30)):
        raise ValueError('the approved section must contain all twenty-nine cases')
    checked, stale, acquisitions, case_reports = Counter(), [], {}, []
    source_ids = set(re.findall(r'ENSG\d{11}', section))

    def visit(value: Any, case: int, arguments: dict | None = None) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, case)
            return
        if not isinstance(value, dict):
            return
        result = value.get('result')
        records = value.get('provenance', {}).get('graph', [])
        arguments = value.get('arguments', arguments)
        if isinstance(result, dict) and records:
            current = all(record['graph_version'] == view.graph_version for record in records)
            if not current:
                stale.append(case)
            elif 'neighbors' in result and 'gene_id' in result:
                layer_id, gene, direction = result['layer_id'], result['gene_id'], result['direction']
                if result['multiplex_id'] != view.multiplex_id or layer_id not in view.layers:
                    raise ValueError(f'S2.7.{case}: unregistered source scope')
                missing = [gene] if gene not in view.multiplex_nodes else []
                outside = [gene] if gene in view.multiplex_nodes and gene not in view.layers[layer_id].nodes else []
                if result['missing_gene_ids'] != missing or result['not_in_layer_gene_ids'] != outside:
                    raise ValueError(f'S2.7.{case}: incorrect source membership')
                if missing:
                    if result['neighbors'] is not None or result['collection'] is not None:
                        raise ValueError(f'S2.7.{case}: missing global membership has an ordinary list')
                else:
                    native = [{'gene_id': neighbor, 'weight': weight}
                              for neighbor, weight in index.neighbors(layer_id, gene, direction)]
                    rows, collection = result['neighbors'], result['collection']
                    positions = {row['gene_id']: rank for rank, row in enumerate(native)}
                    if any(row not in native for row in rows) or len({row['gene_id'] for row in rows}) != len(rows):
                        raise ValueError(f'S2.7.{case}: neighbor IDs or weights differ from the source')
                    if [positions[row['gene_id']] for row in rows] != sorted(positions[row['gene_id']] for row in rows):
                        raise ValueError(f'S2.7.{case}: native order differs from the source')
                    if collection['returned_count'] != len(rows) or collection['total_count'] not in (None, len(native)):
                        raise ValueError(f'S2.7.{case}: source collection counts differ')
                    if arguments and arguments['cursor'] is None:
                        if collection['mode'] == 'top_k' and rows != native[:arguments['top_k']]:
                            raise ValueError(f'S2.7.{case}: bounded source prefix differs')
                        if collection['mode'] == 'complete' and collection['complete'] and rows != native:
                            raise ValueError(f'S2.7.{case}: first complete source list differs')
                    checked['neighbor_result_objects'] += 1
                    checked['neighbor_entries'] += len(rows)
                checked['neighbor_membership_objects'] += 1
                for record in records:
                    key = record['evidence_id']
                    payload = {'result': result, 'provenance': value['provenance']}
                    if key in acquisitions and acquisitions[key] != payload:
                        raise ValueError(f'S2.7.{case}: one evidence ID has different source records')
                    acquisitions[key] = payload
            elif {'present_gene_ids', 'absent_gene_ids'} <= result.keys():
                if any(g not in view.multiplex_nodes for g in result['present_gene_ids']) or any(
                    g in view.multiplex_nodes for g in result['absent_gene_ids']):
                    raise ValueError(f'S2.7.{case}: global membership differs from the source')
                checked['global_membership_objects'] += 1
        if isinstance(result, dict) and 'shared_neighbors' in result:
            layer_id = result['layer_id']
            roles = [(role['gene_id'], role['direction']) for role in result['queries']]
            if len(roles) != 2 or any(g not in view.multiplex_nodes for g, _ in roles):
                raise ValueError(f'S2.7.{case}: ordinary result has unavailable roles')
            expected = sorted(neighbor_ids(index, layer_id, *roles[0]) & neighbor_ids(index, layer_id, *roles[1]))
            if result['shared_neighbors'] != expected or result['shared_neighbor_count'] != len(expected):
                raise ValueError(f'S2.7.{case}: shared intersection differs from the source')
            if result['collection'] != {'mode': 'complete', 'returned_count': len(expected), 'total_count': len(expected),
                                        'complete': True, 'next_cursor': None}:
                raise ValueError(f'S2.7.{case}: final collection counts differ')
            for member in result['input_memberships']:
                if member['present_in_multiplex'] is not True or member['present_in_layer'] not in (
                    None, member['gene_id'] in view.layers[layer_id].nodes):
                    raise ValueError(f'S2.7.{case}: final input membership differs')
            requested = list(dict.fromkeys((g, native) for g, direction in roles
                                           for native in (('in', 'out') if direction == 'both' else (direction,))))
            coverage = result['coverage']
            actual = lambda name: [(row['gene_id'], row['direction']) for row in coverage[name]]
            done, pending = actual('checked_queries'), actual('unchecked_queries')
            if (coverage['operation'] != 'neighbor_lists' or actual('requested_queries') != requested
                    or len(set(done)) != len(done) or len(set(pending)) != len(pending)
                    or set(done) & set(pending) or set(done) | set(pending) != set(requested)
                    or done != [role for role in requested if role in done]
                    or pending != [role for role in requested if role in pending]):
                raise ValueError(f'S2.7.{case}: primitive coverage does not partition the request')
            checked['final_intersections'] += 1
        for key, item in value.items():
            if key not in ('result', 'provenance'):
                visit(item, case)
        if isinstance(result, dict):
            for item in result.get('partial_results', []):
                visit(item, case)

    for rank, match in enumerate(matches):
        case = int(match[1])
        chunk = section[match.start():matches[rank + 1].start() if rank + 1 < len(matches) else len(section)]
        before, last_call = dict(checked), None
        blocks = re.findall(r'```json\n(.*?)\n```', chunk, re.S)
        for block in blocks:
            value = json.loads(block)
            if isinstance(value, dict) and value.get('tool') == 'get_neighbors':
                last_call = value['arguments']
                checked['current_turn_neighbor_calls'] += 1
            visit(value, case, last_call if isinstance(value, dict) and 'operation' not in value else None)
        case_reports.append({'case_id': case, 'json_blocks': len(blocks),
                             'checks': {key: count - before.get(key, 0) for key, count in checked.items()
                                        if count != before.get(key, 0)}})
    report = {'graph_version': view.graph_version, 'source_manifest_sha256': view.manifest_sha256,
        'approved_section_sha256': hashlib.sha256(section.encode()).hexdigest(),
        'registry_index_sha256': registry.source_index_sha256, 'registry_logical_sha256': registry.content_sha256,
        'case_count': len(matches), 'checks': dict(checked), 'unique_neighbor_acquisitions': len(acquisitions),
        'injected_incompatible_version_case_ids': sorted(set(stale)),
        'distinct_approved_source_ids': len(source_ids),
        'registered_approved_ids': len(source_ids & registry.symbols_by_id.keys()),
        'approved_ids_outside_registry': sorted(source_ids - registry.symbols_by_id.keys()),
        'approved_graph_ids_outside_registry': sorted(source_ids & view.multiplex_nodes - registry.symbols_by_id.keys()),
        'cases': case_reports,
        'scope': 'All approved current-version neighbor records, source membership assertions, native weights and order, totals, final exact intersections, and coverage partitions match their declared pinned source. Opaque page boundaries and errors are harness conditions. No deployed runtime or model accuracy is measured.'}
    if gtf_path is not None:
        if not gtf_sha256:
            raise ValueError('the original GTF requires its SHA-256 pin')
        actual = hashlib.sha256(Path(gtf_path).read_bytes()).hexdigest()
        if actual != gtf_sha256:
            raise ValueError('the original GTF differs from its pin')
        found = {}
        with gzip.open(gtf_path, 'rt') as handle:
            for line in handle:
                fields = line.rstrip('\n').split('\t')
                if len(fields) == 9 and fields[2] == 'gene':
                    attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
                    gene = attrs['gene_id']
                    if gene in source_ids:
                        found[gene] = attrs.get('gene_name')
        if any(registry.symbols_by_id[g] != symbol for g, symbol in found.items() if g in registry.symbols_by_id):
            raise ValueError('the original GTF differs from the approved registry symbols')
        report['original_gtf'] = {'path': str(gtf_path), 'sha256': actual, 'matching_gene_entries': len(found),
                                  'approved_ids_absent_from_gtf': sorted(source_ids - found.keys())}
    return report
