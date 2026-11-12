"""Generate source-backed S2.7–S2.10 production candidates in fixed splits.

Published traces provide reproducible source/evidence seeds. Additional native
searches cover omitted small facts. Independent combinations of the existing
frozen request templates and wrappers add wording diversity without changing
the evidence, tool calls, or structured answer of a validated seed.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import importlib
from itertools import combinations
import json

from .s2_2_corpus import VariantUnavailable
from .s2_3_corpus import gene_group
from .s2_audit_cases_7_10 import CASE_BINDINGS, matches
from .s2_audit_source import SourceFeatures, nearest_below_threshold
from .s2_corpus import ROOT, canonical_json, digest
from .s2_degrees import DegreeIndex
from .s2_multiplex import MultiplexRuntime
from .sources import file_sha256


PRODUCTION = 's2_7_10_frozen_wording_product_v1'
FAMILIES = ('S2.7', 'S2.8', 'S2.9', 'S2.10')
CASE_IDS = tuple(CASE_BINDINGS)
SPLITS = ('train', 'validation', 'test')
REQUEST_ADDITIONS = {'path': 'runtime/world_model/s2_production_requests_9_10_v1.json',
                    'sha256': '94a5c2f4c59c760db70805a93a1a60567da4b45ae303d201c0028369e670dc66'}
DEGREE_REQUEST_ADDITIONS = {'path': 'runtime/world_model/s2_production_8_requests_v1.json',
                          'sha256': '59441aafaf85e0a60ff082815b0027049167332a6cf302fb9752901a0078093e'}
INPUT_PATHS = (
    REQUEST_ADDITIONS['path'],
    DEGREE_REQUEST_ADDITIONS['path'],
    'data/world_model_v4/diagnostics/s2_case_quota_audit_v4/report.json',
    'data/world_model_v4/diagnostics/s2_case_quota_audit_v4/validated_witnesses.json',
    'data/world_model_v4/diagnostics/s2_8_builder_v1/audit.json',
    'scripts/audit_world_model_s2_context.py',
    *[f'data/world_model_v4/diagnostics/s2_{number}_builder_v1/{name}'
      for number in (7, 8, 9, 10) for name in ('manifest.json', 'audit.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl')],
    *sorted({witness['path'] for witness in json.loads((ROOT /
        'data/world_model_v4/diagnostics/s2_case_quota_audit_v4/validated_witnesses.json').read_text())
        if '.'.join(witness['case_id'].split('.')[:2]) in FAMILIES}),
)


def prepare_context(context, family):
    """Cache shared source indices and the unchanged reviewed case envelopes."""
    if family not in FAMILIES:
        raise ValueError('the production family is outside S2.7–S2.10')
    context.setdefault('seed', 20261028)
    if '_production_case_entries' not in context:
        report = json.loads((ROOT / 'data/world_model_v4/diagnostics/s2_case_quota_audit_v4/report.json').read_text())
        context['_production_case_entries'] = {entry['case_id']: entry for entry in report['case_entries']}
    if '_degree_index' not in context:
        context['_degree_index'] = DegreeIndex(context['view'], context['index'].native_index)
    if '_features' not in context:
        selection = json.loads((ROOT / 'data/world_model_v4/diagnostics/s2_8_builder_v1/audit.json').read_text())['selection']
        context['_features'] = SourceFeatures(context['view'], context['index'], seed=context['seed'],
                                             nearest=nearest_below_threshold(selection))
    context.setdefault('_production_seed_pools', {})
    if '_production_requests' not in context:
        path = ROOT / REQUEST_ADDITIONS['path']
        if file_sha256(path) != REQUEST_ADDITIONS['sha256']:
            raise ValueError('the production request additions differ from their frozen pin')
        additions = json.loads(path.read_text())
        requests = deepcopy(context['contract']['_requests'])
        if family == 'S2.8':
            degree_path = ROOT / DEGREE_REQUEST_ADDITIONS['path']
            if file_sha256(degree_path) != DEGREE_REQUEST_ADDITIONS['sha256']:
                raise ValueError('the production degree request additions differ from their pin')
            degree_additions = json.loads(degree_path.read_text())['s2_8']
            for split in SPLITS:
                for kind in ('degree_only', 'degree_and_hub'):
                    requests[split][kind].extend(degree_additions[split])
                for kind in ('invalid_direction', 'unknown_layer'):
                    requests['diagnostic_requests'][split][kind].extend(degree_additions[split])
        elif family == 'S2.9':
            for split in SPLITS:
                for kind in ('complete', 'top_k'):
                    requests[split][kind].extend(additions['s2_9'][split])
        elif family == 'S2.10':
            for split in SPLITS:
                requests[split]['partition'].extend(additions['s2_10'][split])
                requests['diagnostic_requests'][split]['unavailable_kind'].extend(additions['s2_10'][split])
        context['_production_requests'] = requests
    return context


def _eligible(case_id, row, context):
    """Keep exact approved source conditions before any wording expansion."""
    entry = context['_production_case_entries'][case_id]
    meta, answer = row['metadata'], row['messages'][-1]['content']
    if (meta['question_family'] != entry['family'] or meta['variant'] not in CASE_BINDINGS[case_id]['variants']
            or answer['status'] != entry['status'] or answer['continuation_decision'] != entry['continuation_decision']
            or entry['framing_hint'] is not None and meta['framing'] != entry['framing_hint']):
        return False
    classified = dict(row)
    classified['_audit_source'] = context['_features'].for_row(row)
    if case_id == 'S2.7.12' and matches('S2.7.27', classified):
        return False  # Reserve this scarce specific evidence state for case27.
    return matches(case_id, classified)


def _source_key(row):
    """Distinguish genuine query, transport and evidence contrasts from wording."""
    meta = row['metadata']
    if meta['question_family'] in ('S2.8', 'S2.9', 'S2.10'):
        number = int(meta['question_family'].split('.')[1])
        base = importlib.import_module(f'.s2_{number}_corpus', __package__)
        if meta['variant'] in base.DIAGNOSTICS:
            # Missing raw inputs hide their private sampling facts. Retaining
            # one seed per literal request prevents those facts crowding out
            # actual layer/kind diversity in the bounded source pool.
            return canonical_json({'family': meta['question_family'], 'variant': meta['variant'],
                                   'framing': meta['framing'], 'request': meta['request']})
    keys = ('layer_id', 'layer_scope', 'gene_id', 'gene_ids', 'queries', 'direction',
            'request_kind', 'hub_policy', 'variant', 'framing', 'question_kind',
            'membership_first', 'saved_layer_id', 'saved_gene_ids', 'native_direction')
    return canonical_json({key: meta.get(key) for key in keys})


def _add_seed(pools, case_id, row, context):
    """Retain small source instances independently of wrapper-bank size."""
    if not _eligible(case_id, row, context):
        return
    row = deepcopy(row)
    row['metadata']['catalog_case_id'] = case_id
    cell = pools[case_id, row['metadata']['split']]
    key, score = _source_key(row), len(canonical_json(row))
    if key not in cell or score < cell[key][0]:
        cell[key] = score, row
    if len(cell) > 24:
        del cell[max(cell, key=lambda identity: (cell[identity][0], identity))]


def _published_seeds(context, family):
    """Read frozen source pools once and retain bounded exact-case candidates."""
    prepare_context(context, family)
    cache = context['_production_seed_pools']
    if family in cache:
        return cache[family]
    number = int(family.split('.')[1])
    cases = [case_id for case_id in CASE_IDS if case_id.startswith(family + '.')]
    by_variant = defaultdict(list)
    for case_id in cases:
        for variant in CASE_BINDINGS[case_id]['variants']:
            by_variant[variant].append(case_id)
    pools = defaultdict(dict)
    directory = ROOT / f'data/world_model_v4/diagnostics/s2_{number}_builder_v1'
    from scripts.audit_world_model_s2_context import load_verified_lengths
    _, lengths = load_verified_lengths(directory, ROOT)
    for split in SPLITS:
        with (directory / f'{split}.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if lengths[row['row_id']]['tokens'] > 8192:
                    continue
                for case_id in by_variant[row['metadata']['variant']]:
                    _add_seed(pools, case_id, row, context)
    # The verified compact loop, strong-prefix, and combination repairs have
    # separate immutable artifacts. Read only referenced witness lines.
    witnesses = json.loads((ROOT / 'data/world_model_v4/diagnostics/s2_case_quota_audit_v4/validated_witnesses.json').read_text())
    wanted = defaultdict(dict)
    for witness in witnesses:
        if witness['case_id'] in cases and not witness['path'].startswith(str(directory.relative_to(ROOT)) + '/'):
            wanted[witness['path']][witness['line']] = witness['case_id']
    for path, lines in wanted.items():
        with (ROOT / path).open() as stream:
            for ordinal, line in enumerate(stream, 1):
                if ordinal in lines:
                    _add_seed(pools, lines[ordinal], json.loads(line), context)
                if ordinal >= max(lines):
                    break
    cache[family] = pools
    return pools


def _render(row, context, wrapper_index, request_index, *, original=False):
    """Bind independent indices to the original frozen request and wrapper banks."""
    meta = row['metadata']
    family = int(meta['question_family'].split('.')[1])
    prompts = importlib.import_module(f'.s2_{family}_prompts', __package__)
    base = importlib.import_module(f'.s2_{family}_corpus', __package__)
    diagnostic = meta['variant'] in base.DIAGNOSTICS
    kwargs = {'split': meta['split'], 'variant_index': request_index}
    kind = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
    if family == 7:
        request = {'multiplex_id': context['view'].multiplex_id, 'layer_id': meta['layer_id'],
                   'genes': meta['input_genes'], 'directions': meta['directions']}
        if diagnostic:
            from .s2_7_sampling import fact_from_queries
            fact = fact_from_queries(context['view'], context['index'].native_index, meta['layer_id'],
                [(q['gene_id'], q['direction']) for q in meta['queries']], context['seed'], split=meta['split'],
                category=meta['category'])
            request, _, _ = base._diagnostic_request(context['view'], fact, meta['variant'])
    else:
        request = dict(meta['request'])
        if family == 10 and not diagnostic:
            request['question_kind'] = meta['question_kind']
    render = getattr(prompts, f'render_s2_{family}_' + ('diagnostic_request' if diagnostic else 'request'))
    if diagnostic:
        kwargs['kind'] = kind
    requests = context['contract']['_requests'] if original else context['_production_requests']
    core = render(requests, **kwargs, **request)
    wrappers = context['bank'][meta['split']][meta['framing']]
    if type(wrapper_index) is not int or not 0 <= wrapper_index < len(wrappers):
        raise ValueError('a production wrapper index is outside its frozen split bank')
    return wrappers[wrapper_index].format(request=core)


def _template_count(row, context):
    """Use the exact audited request category for this literal request."""
    meta = row['metadata']
    family = int(meta['question_family'].split('.')[1])
    base = importlib.import_module(f'.s2_{family}_corpus', __package__)
    requests = context['_production_requests']
    if meta['variant'] in base.DIAGNOSTICS:
        kind = 'unknown_layer' if meta['variant'] == 'tool_unknown_layer' else meta['variant']
        return len(requests['diagnostic_requests'][meta['split']][kind])
    kind = (base._kind(meta['directions']) if family == 7 else meta['request_kind'] if family == 8
            else meta['request']['mode'] if family == 9 else meta['question_kind'])
    return len(requests[meta['split']][kind])


def _reword(seed, context, wrapper_index, request_index):
    """Change audited user wording while retaining every original source record."""
    row = deepcopy(seed)
    meta = row['metadata']
    meta['production_wording'] = {'schema_version': PRODUCTION, 'base_row_id': seed['row_id'],
        'base_wrapper_index': meta['wrapper_index'], 'base_request_template_index': meta['request_template_index']}
    meta.update(wrapper_index=wrapper_index, request_template_index=request_index)
    row['messages'][2]['content'] = _render(row, context, wrapper_index, request_index)
    row['row_id'] = 's2prod_' + digest([PRODUCTION, meta['catalog_case_id'], seed['row_id'],
                                       row['messages'][2]['content']])[:32]
    return row


def validate_row(row, context):
    """Validate frozen wording, then replay the unchanged original source proof."""
    meta = row['metadata']
    family = meta['question_family']
    prepare_context(context, family)
    binding = meta.get('production_wording')
    if (not isinstance(binding, dict) or set(binding) != {'schema_version', 'base_row_id',
            'base_wrapper_index', 'base_request_template_index'} or binding['schema_version'] != PRODUCTION):
        raise ValueError('production wording requires its exact frozen binding')
    if (meta.get('catalog_case_id') not in CASE_IDS or meta['selection_seed'] != context['seed']
            or not _eligible(meta['catalog_case_id'], row, context)):
        raise ValueError('the production row changes its reviewed source case')
    if type(meta['request_template_index']) is not int or not 0 <= meta['request_template_index'] < _template_count(row, context):
        raise ValueError('a production request index is outside its frozen bank')
    expected = _render(row, context, meta['wrapper_index'], meta['request_template_index'])
    expected_id = 's2prod_' + digest([PRODUCTION, meta['catalog_case_id'], binding['base_row_id'], expected])[:32]
    if row['messages'][2]['content'] != expected or row['row_id'] != expected_id:
        raise ValueError('production request wording or row identity changed')
    original = deepcopy(row)
    original['row_id'] = binding['base_row_id']
    original['metadata'].pop('production_wording')
    original['metadata'].update(wrapper_index=binding['base_wrapper_index'],
                                request_template_index=binding['base_request_template_index'])
    original['messages'][2]['content'] = _render(original, context, binding['base_wrapper_index'],
                                                binding['base_request_template_index'], original=True)
    number = int(family.split('.')[1])
    suffix = ('case_repairs' if meta.get('repair') else 'loop_context' if meta.get('extension') else
              'supplemental' if number == 10 and meta.get('variant') == 'strong_witness_context' else 'corpus')
    module = importlib.import_module(f'.s2_{number}_{suffix}', __package__)
    index = context['index'].native_index if number == 7 else context['_degree_index'] if number == 8 else context['index']
    module.validate_row(original, context['view'], index=index, bank=context['bank'],
                        contract=context['contract'], registry=context.get('registry'))


def production_candidates(case_id, context, *, split):
    """Yield deterministic exact-case source seeds under all frozen wordings."""
    if case_id not in CASE_IDS or split not in SPLITS:
        raise ValueError('a reviewed case and fixed split are required')
    family = '.'.join(case_id.split('.')[:2])
    pools = _published_seeds(context, family)
    _targeted_seeds(context, case_id, split, pools)
    seeds = [item[1] for item in sorted(pools[case_id, split].values(), key=lambda item: (item[0], _source_key(item[1])))]
    for request in range(max((_template_count(seed, context) for seed in seeds), default=0)):
        for seed in seeds:
            if request >= _template_count(seed, context):
                continue
            for wrapper in range(len(context['bank'][split][seed['metadata']['framing']])):
                yield _reword(seed, context, wrapper, request)


def _targeted_seeds(context, case_id, split, pools):
    """Add omitted small source constructions without weakening a case predicate."""
    completed = context.setdefault('_targeted_cells', set())
    if (case_id, split) in completed:
        return
    completed.add((case_id, split))
    if case_id.startswith('S2.7.'):
        _shared_targets(context, case_id, split, pools)
    elif case_id.startswith('S2.8.'):
        _degree_targets(context, case_id, split, pools)
    elif case_id == 'S2.9.23':
        _directed_native_targets(context, split, pools)


def _shared_targets(context, case_id, split, pools):
    """Search every native middle vertex, retaining small directed role lists."""
    if case_id in ('S2.7.4', 'S2.7.12', 'S2.7.22', 'S2.7.24', 'S2.7.27', 'S2.7.28'):
        _shared_context_targets(context, case_id, split, pools)
        return
    if case_id not in ('S2.7.15', 'S2.7.17', 'S2.7.29'):
        return
    from .s2_7_corpus import build_row
    from .s2_7_sampling import fact_from_queries
    view, index, seed = context['view'], context['index'].native_index, context['seed']
    candidates = set()
    for layer in view.layer_order:
        if not view.layers[layer].directed:
            continue
        degrees = {(gene, side): len(index.neighbors(layer, gene, side))
                   for gene in view.layers[layer].nodes for side in ('in', 'out')}
        if case_id == 'S2.7.29':
            for gene in view.layers[layer].nodes:
                if gene_group(gene, seed)[1] == split and all(degrees[gene, side] for side in ('in', 'out')):
                    candidates.add((degrees[gene, 'in'] + degrees[gene, 'out'], layer, gene, gene))
        else:
            for middle in view.layers[layer].nodes:
                left = sorted((gene for gene, _ in index.neighbors(layer, middle, 'in')
                               if gene_group(gene, seed)[1] == split), key=lambda gene: (degrees[gene, 'out'], gene))[:12]
                if case_id == 'S2.7.15':
                    pairs = combinations(left, 2)
                    second_side = 'out'
                else:
                    right = sorted((gene for gene, _ in index.neighbors(layer, middle, 'out')
                                    if gene_group(gene, seed)[1] == split), key=lambda gene: (degrees[gene, 'in'], gene))[:12]
                    pairs = ((first, second) for first in left for second in right if first != second)
                    second_side = 'in'
                for first, second in pairs:
                    candidates.add((degrees[first, 'out'] + degrees[second, second_side], layer, first, second))
    for _, layer, first, second in sorted(candidates)[:48]:
        sides = ('out', 'out') if case_id == 'S2.7.15' else ('out', 'in')
        fact = fact_from_queries(view, index, layer, [(first, sides[0]), (second, sides[1])], seed,
                                 split=split, category='full_native_small_lists')
        row = build_row(view, fact, 'direct', 'lookup', 0, context['bank'], context['contract'],
                        index=index, registry=context.get('registry'), seed=seed)
        _add_seed(pools, case_id, row, context)


def _shared_context_targets(context, case_id, split, pools):
    """Find source-backed identity, self-loop, prefix and containment witnesses."""
    from .s2_7_corpus import build_row, build_diagnostic
    from .s2_7_sampling import fact_from_queries
    view, index, seed = context['view'], context['index'].native_index, context['seed']
    roles = set()
    if case_id in ('S2.7.4', 'S2.7.12', 'S2.7.22'):
        for (other, cell_split), choices in list(pools.items()):
            if cell_split != split:
                continue
            for _, row in choices.values():
                meta = row['metadata']
                if meta['directions'] == ['undirected', 'undirected']:
                    pair = tuple((query['gene_id'], query['direction']) for query in meta['queries'])
                    roles.add((meta['layer_id'], pair))
    if case_id == 'S2.7.24':
        for layer in view.layer_order:
            native = view.layers[layer]
            if not native.directed:
                continue
            for first, second in native.edges:
                if first == second and gene_group(first, seed)[1] == split:
                    for target, _ in index.neighbors(layer, first, 'out'):
                        if target != first and gene_group(target, seed)[1] == split:
                            roles.add((layer, ((first, 'in'), (target, 'in'))))
    if case_id in ('S2.7.27', 'S2.7.28'):
        for layer in view.layer_order:
            for side in index.directions(layer):
                local = sorted(gene for gene in view.layers[layer].nodes if gene_group(gene, seed)[1] == split)
                if case_id == 'S2.7.27':
                    by_first = defaultdict(list)
                    for gene in local:
                        values = index.neighbors(layer, gene, side)
                        if len(values) > 1:
                            by_first[values[0][0]].append(gene)
                    for common, genes in by_first.items():
                        genes.sort(key=lambda gene: (len(index.neighbors(layer, gene, side)), gene))
                        for first, second in combinations(genes[:32], 2):
                            sets = [{gene for gene, _ in index.neighbors(layer, source, side)} for source in (first, second)]
                            if len(sets[0] & sets[1]) > 1:
                                roles.add((layer, ((first, side), (second, side))))
                else:
                    leaves = defaultdict(list)
                    for gene in local:
                        values = index.neighbors(layer, gene, side)
                        if len(values) == 1:
                            leaves[values[0][0]].append(gene)
                    for second in local:
                        values = index.neighbors(layer, second, side)
                        for common, _ in values[:min(64, len(values) - 1)]:
                            for first in leaves.get(common, ())[:4]:
                                if first != second:
                                    roles.add((layer, ((first, side), (second, side))))
    ordered = sorted(roles, key=lambda item: (
        sum(len(index.neighbors(item[0], gene, side)) for gene, side in item[1]), item))
    variant = {'S2.7.4': 'symbol_context', 'S2.7.12': 'bounded_context', 'S2.7.24': 'lookup',
               'S2.7.27': 'bounded_context', 'S2.7.28': 'containment_context'}.get(case_id)
    found = 0
    for layer, queries in ordered:
        fact = fact_from_queries(view, index, layer, queries, seed, split=split, category='production_native_context')
        try:
            if case_id == 'S2.7.22':
                row = build_diagnostic(view, fact, 'direct', 0, context['bank'], context['contract'],
                                       kind='unknown_layer', seed=seed)
            else:
                row = build_row(view, fact, 'step' if case_id == 'S2.7.4' else 'direct', variant, 0,
                    context['bank'], context['contract'], index=index, registry=context.get('registry'), seed=seed)
        except VariantUnavailable:
            continue
        if _eligible(case_id, row, context):
            _add_seed(pools, case_id, row, context)
            found += 1
            if found >= 24:
                break


def _degree_targets(context, case_id, split, pools):
    """Measure threshold and union contrasts against full unfiltered populations."""
    if case_id in ('S2.8.22', 'S2.8.26'):
        _degree_repair_targets(context, case_id, split, pools)
        return
    if case_id not in ('S2.8.3', 'S2.8.29', 'S2.8.30', 'S2.8.31'):
        return
    from .s2_8_corpus import build_row
    from .s2_8_sampling import fact_from_arguments, scope_schedule
    view, index, seed = context['view'], context['_degree_index'], context['seed']
    runtime = MultiplexRuntime(view)
    for scope, direction in scope_schedule(view):
        selected = runtime.resolve_scope(scope)
        if case_id == 'S2.8.31' and (direction != 'incident' or len(selected) != 1 or not view.layers[selected[0]].directed):
            continue
        if case_id == 'S2.8.30' and direction != 'out':
            continue
        population, lower, wanted = index.population(selected), 0, {}
        for cell in index.histogram(selected, direction):
            degree, equal = cell['degree'], cell['node_count']
            rank = Fraction(50) if len(population) == 1 else Fraction(100 * (2 * lower + equal - 1), 2 * (len(population) - 1))
            if (case_id == 'S2.8.3' and rank.denominator == 1
                    or case_id == 'S2.8.29' and Fraction(18999, 200) < rank < 95
                    or case_id == 'S2.8.30' and degree == 1 and rank >= 95
                    or case_id == 'S2.8.31'):
                wanted[degree] = int(rank) if case_id == 'S2.8.3' else 95
            lower += equal
        genes = sorted((gene for gene in population if gene_group(gene, seed)[1] == split
                        and index.degree(selected, gene, direction) in wanted),
                       key=lambda gene: (index.degree(selected, gene, direction), gene))[:24]
        for gene in genes:
            fact = fact_from_arguments(view, index, gene, scope, direction, seed, category='exact_production_statistic')
            try:
                row = build_row(view, fact, 'direct', 'union_counts' if case_id == 'S2.8.31' else 'lookup',
                    0, context['bank'], context['contract'], index=index, registry=context.get('registry'),
                    request_kind='degree_only' if case_id == 'S2.8.31' else 'degree_and_hub',
                    threshold=wanted[fact.degree], seed=seed)
            except VariantUnavailable:
                continue
            _add_seed(pools, case_id, row, context)


def _degree_repair_targets(context, case_id, split, pools):
    """Apply the reviewed combined evidence state to multiple genuine genes."""
    from .s2_8_case_repairs import _build_case
    from .s2_8_sampling import fact_from_arguments
    view, index, seed = context['view'], context['_degree_index'], context['seed']
    layers = [layer for layer in view.layer_order if view.layers[layer].directed == (case_id == 'S2.8.22')]
    candidates = []
    if case_id == 'S2.8.22':
        for active in layers:
            for saved in layers:
                if active == saved:
                    continue
                for gene in view.layers[active].nodes & view.layers[saved].nodes:
                    if (gene_group(gene, seed)[1] == split
                            and index.native_index.neighbors(active, gene, 'out')
                            and index.native_index.neighbors(saved, gene, 'in')):
                        candidates.append((active, saved, gene))
    else:
        for layer in layers:
            genes = sorted((gene for gene in view.layers[layer].nodes if gene_group(gene, seed)[1] == split),
                key=lambda gene: (len(index.native_index.neighbors(layer, gene, 'undirected')), gene))
            candidates.extend((layer, layer, gene) for gene in genes[:16]
                              if index.native_index.neighbors(layer, gene, 'undirected'))
    for active, saved, gene in sorted(candidates, key=lambda item: digest([seed, case_id, *item]))[:16]:
        scope = {'kind': 'layers', 'layer_ids': [active], 'layer_family': None}
        fact = fact_from_arguments(view, index, gene, scope, 'out' if case_id == 'S2.8.22' else 'undirected',
                                   seed, category='case_repair')
        row = _build_case(view, index, context['bank'], context['contract'], context.get('registry'),
                           fact, case_id, saved, seed)
        _add_seed(pools, case_id, row, context)


def _directed_native_targets(context, split, pools):
    """Choose two-member directed sets with the smallest full outgoing records."""
    from .s2_9_corpus import build_row
    from .s2_9_sampling import fact_from_genes
    view, index, seed = context['view'], context['index'], context['seed']
    for layer in view.layer_order:
        if not view.layers[layer].directed:
            continue
        genes = sorted((gene for gene in view.layers[layer].nodes if gene_group(gene, seed)[1] == split),
            key=lambda gene: (len(index.native_index.neighbors(layer, gene, 'out')), gene))[:24]
        for first, second in list(combinations(genes, 2))[:48]:
            fact = fact_from_genes(view, index, layer, [first, second], seed, split=split, category='small_native_directed')
            row = build_row(view, fact, 'direct', 'native_context', 0, context['bank'], context['contract'],
                            index=index, registry=context.get('registry'), seed=seed)
            _add_seed(pools, 'S2.9.23', row, context)
