"""Generate the reviewed path, membership, and specificity production cases.

Historical constructors remain frozen. This module selects their exact task
and source combinations, independently crosses frozen wrappers and requests,
and supplies the two-PEN context that the historical first-layer default did
not express. Only the new S2.13 request bank adds wording.
"""

from copy import deepcopy
from dataclasses import asdict, fields
import importlib
import json

from .s2_2_corpus import VariantUnavailable, _observation
from .s2_audit_cases_11_13 import CASE_BINDINGS, matches
from .s2_audit_source import SourceFeatures
from .s2_corpus import ROOT, canonical_json, digest
from .s2_paths import PathIndex
from .sources import file_sha256


CASE_IDS = tuple(CASE_BINDINGS)
REQUEST_BANK = {'path': 'runtime/world_model/s2_production_13_requests_v1.json',
                'sha256': 'c2ec8bac0b8fa77b295882353472ef941523bb709cf814e46820a01d3147f592'}
PATH_REQUEST_BANK = {'path': 'runtime/world_model/s2_production_11_requests_v1.json',
                    'sha256': 'bb07052ebb81a7aa74ea1e65349cf63caaad878c3440acf688c619eba0039344'}
INPUT_PATHS = (REQUEST_BANK['path'], PATH_REQUEST_BANK['path'], *(
    f'data/world_model_v4/diagnostics/s2_{number}_builder_v1/{name}'
    for number in (11, 12, 13) for name in ('manifest.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl')))

# Tasks preserve the approved questions; source predicates below choose their
# inputs independently of wrapper counts and template positions.
TASKS_11 = {
    1: ('path_claims', None), 2: ('path_support', 'no_interaction_in_biology'),
    3: ('path_support', 'absence_holds_in_other_layers'),
    4: ('path_truth', 'no_path_recorded_in_scope'), 5: ('path_truth', 'no_path_recorded_in_scope'),
    6: ('path_claims', None), 17: ('path_support', 'no_edge_recorded_in_scope'),
    18: ('claim_support', 'no_interaction_in_biology'), 19: ('claim_support', 'no_edge_recorded_in_scope'),
    20: ('truth', 'absence_holds_in_other_layers'), 36: ('truth', 'absence_holds_in_other_layers'),
}
TASKS_12 = {
    1: ('membership_support', 'gene_not_expressed_in_context'),
    2: ('membership_support', 'gene_not_expressed_in_context'),
    3: ('membership_support', 'absence_holds_in_other_layers'),
    4: ('membership_support', 'absence_holds_in_other_versions'),
    5: ('membership_truth', 'gene_not_node_in_layer'), 6: ('membership_truth', 'gene_not_node_in_layer'),
    7: ('membership_support', 'gene_not_expressed_in_context'),
    8: ('membership_support', 'gene_not_expressed_in_context'),
    9: ('membership_support', 'gene_not_expressed_in_context'),
    10: ('membership_support', 'gene_not_node_in_layer'),
    11: ('membership_support', 'gene_not_node_in_layer'),
    13: ('membership_support', 'gene_not_node_in_layer'),
    14: ('membership_support', 'gene_not_node_in_layer'),
    15: ('membership_support', 'gene_not_node_in_layer'),
    16: ('membership_support', 'gene_not_node_in_layer'),
    17: ('membership_support', 'gene_not_node_in_multiplex'),
    18: ('membership_support', 'gene_not_node_in_layer'),
    19: ('membership_support', 'gene_not_node_in_layer'),
    20: ('claim_support', 'gene_not_expressed_in_context'),
    21: ('claim_support', 'gene_not_node_in_layer'),
    22: ('membership_support', 'gene_not_node_in_layer'),
    23: ('membership_truth', 'gene_not_node_in_multiplex'),
    24: ('truth', 'absence_holds_in_other_layers'),
    25: ('truth', 'absence_holds_in_other_layers'),
    30: ('membership_support', 'gene_not_expressed_in_context'),
    32: ('membership_support', 'gene_not_node_in_layer'),
    34: ('membership_support', 'gene_not_node_in_layer'),
    36: ('truth', 'gene_not_expressed_in_context'),
}
TASKS_13 = {
    7: ('edge_support', 'physical_interaction_recorded'),
    8: ('edge_support', 'physical_interaction_recorded'),
    10: ('edge_support', 'global_coexpression'), 11: ('edge_support', 'regulation_direction'),
    12: ('edge_support', 'regulation_direction'), 13: ('claim_support', 'causal_effect'),
    14: ('edge_support', 'context_predictive_expression'),
    16: ('claim_support', 'causal_effect'), 17: ('claim_support', 'causal_effect'),
    18: ('claim_card', None), 19: ('edge_support', 'context_predictive_expression'),
    20: ('edge_support', 'context_predictive_expression'),
    21: ('claim_support', 'holds_in_other_contexts'),
    22: ('claim_support', 'physical_interaction_recorded'),
    23: ('claim_support', 'physical_interaction_recorded'),
    24: ('edge_support', 'physical_interaction_recorded'), 25: ('edge_support', 'global_coexpression'),
    26: ('claim_support', 'regulation_direction'), 27: ('claim_support', 'physical_interaction_recorded'),
    28: ('edge_support', 'physical_interaction_recorded'),
    29: ('edge_support', 'physical_interaction_recorded'), 30: ('graph_truth', None),
    31: ('biological_truth', 'causal_effect'), 33: ('edge_support', 'regulation_direction'),
    36: ('claim_support', 'cofunctional_association'),
    37: ('claim_support', 'physical_interaction_recorded'),
    38: ('claim_support', 'physical_interaction_recorded'),
    40: ('edge_support', 'physical_interaction_recorded'),
}


def prepare_context(context, family):
    """Cache native path support and the separately pinned production wording."""
    number = int(str(family).split('.')[-1])
    key = f'_production_{number}'
    if key in context:
        return context[key]
    prepared = dict(context)
    prepared['number'] = number
    prepared['module'] = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
    if number == 11:
        prepared['index'] = PathIndex(context['view'], index=context['index'].native_index)
        path = ROOT / PATH_REQUEST_BANK['path']
        if file_sha256(path) != PATH_REQUEST_BANK['sha256']:
            raise ValueError('the production path wording differs from its pin')
        from .s2_11_prompts import validate_s2_11_requests
        requests = json.loads(path.read_text())
        validate_s2_11_requests(requests)
        prepared['contract'] = {**context['contract'], '_requests': requests,
                                'production_request_bank': dict(PATH_REQUEST_BANK)}
    if number == 13:
        path = ROOT / REQUEST_BANK['path']
        if file_sha256(path) != REQUEST_BANK['sha256']:
            raise ValueError('the production specificity wording differs from its pin')
        from .s2_13_prompts import validate_s2_13_requests
        requests = json.loads(path.read_text())
        validate_s2_13_requests(requests)
        prepared['contract'] = {**context['contract'], '_requests': requests,
                                'production_request_bank': dict(REQUEST_BANK)}
    prepared['features'] = SourceFeatures(context['view'], context['index'], seed=context['seed'])
    context[key] = prepared
    return prepared


def _pool(context, split):
    """Reuse pinned historical selections; selected rows undergo fresh replay."""
    cache = context.setdefault('_facts', {})
    if split in cache:
        return cache[split]
    number, view, index, registry = (context[key] for key in ('number', 'view', 'index', 'registry'))
    sampling = importlib.import_module(f'runtime.world_model.s2_{number}_sampling')
    fact_type = getattr(sampling, {11: 'PathFact', 12: 'MembershipFact', 13: 'PairFact'}[number])
    source = ROOT / f'data/world_model_v4/diagnostics/s2_{number}_builder_v1/{split}.jsonl'
    manifest = json.loads(source.with_name('manifest.json').read_text())
    if file_sha256(source) != manifest['files'][f'{split}.jsonl']['sha256']:
        raise ValueError('the historical source-selection pool changed')
    keys, facts = set(), []
    with source.open() as stream:
        for line in stream:
            meta = json.loads(line)['metadata']
            if meta.get('variant') in context['module'].DIAGNOSTICS or meta.get('category') == 'family_capability':
                continue
            key = meta['layer_id'], tuple(meta.get('gene_ids', [meta.get('gene_id')]))
            if key in keys:
                continue
            keys.add(key)
            if number == 11:
                values = {field.name: meta[field.name] for field in fields(fact_type)}
                for name in ('gene_ids', 'category_counts', 'path_gene_ids'):
                    if values[name] is not None:
                        values[name] = tuple(values[name])
                fact = fact_type(**values)
            elif number == 12:
                fact = sampling.fact_from_gene(view, index, registry, context['old_view'],
                    meta['layer_id'], meta['gene_id'], context['seed'])
            else:
                fact = sampling.fact_from_pair(view, index, registry, context['old_view'],
                    meta['layer_id'], meta['gene_ids'], context['seed'])
            facts.append(fact)
            if number == 13 and fact.directed:
                reverse_key = fact.layer_id, fact.gene_ids[::-1]
                if reverse_key not in keys:
                    keys.add(reverse_key)
                    facts.append(sampling.fact_from_pair(view, index, registry, context['old_view'],
                        fact.layer_id, fact.gene_ids[::-1], context['seed']))
    # Small complete native contexts are preferred before wording is selected.
    def cost(fact):
        if number == 12:
            return (fact.outgoing_count or 0) + (fact.incoming_count or 0), fact.layer_id, fact.gene_id
        return len(getattr(fact, 'path_gene_ids', ()) or ()), fact.layer_id, fact.gene_ids
    cache[split] = sorted(facts, key=cost)
    return cache[split]


def _eligible(number, case, fact, context):
    """Filter complete source populations without consulting wording or outcomes."""
    view, index = context['view'], context['index']
    layer = view.layers[fact.layer_id]
    if number == 11:
        same = fact.gene_ids[0] == fact.gene_ids[1]
        members = not any(fact.category_counts[1:])
        negative = fact.path_found is False and members and not same
        if case in (1, 2, 3, 4, 11, 18, 28, 32, 34, 35, 36) and not negative:
            return False
        if case in (1, 35) and layer.directed or case in (9, 10, 34) and not layer.directed:
            return False
        if case in (5, 8, 9, 13, 14, 27, 29) and fact.hop_count != 2:
            return False
        if case == 6 and (not fact.category_counts[1] or fact.category_counts[2] or same):
            return False
        if case == 7 and (not fact.category_counts[2] or same):
            return False
        if case == 10:
            reverse = index.distances(fact.layer_id, fact.gene_ids[1])
            return negative and fact.gene_ids[0] in reverse
        if case == 12 and fact.hop_count != 1:
            return False
        if case in (15, 16, 17) and (not same or fact.hop_count != 0):
            return False
        if case in (15, 16):
            loop = (fact.gene_ids[0],) * 2 in layer.edges
            return loop == (case == 15)
        if case in (20, 21, 22) and fact.path_found is not True:
            return False
        if case == 29 and (fact.tied_shortest_paths or 0) <= 1:
            return False
        if case in (30, 31):
            return same and bool(fact.category_counts[1 if case == 30 else 2])
        if case == 33 and (fact.hop_count or 0) < 4:
            return False
        if case == 34:
            return not index.native_index.neighbors(fact.layer_id, fact.gene_ids[0], 'out')
        return True
    if number == 12:
        category = ('outside_multiplex' if case in (9, 17, 23) else 'member'
            if case in (6, 8, 10, 11, 15, 16, 18, 24, 34) else 'outside_layer')
        if case not in (26, 27, 28, 29, 31, 35) and fact.category != category:
            return False
        family = context['features'].multiplex.family_of(fact.layer_id)
        if case == 1 and family != 'scpen' or case == 2 and family != 'bulk_pen':
            return False
        if case == 3 and family not in ('bulk_pen', 'scpen'):
            return False
        if case == 10 and (not layer.directed or fact.outgoing_count != 0):
            return False
        if case == 11 and fact.self_loop:
            return False
        if case in (15, 21, 22) and (len(fact.layer_ids) < 2 or fact.layer_ids[0] == fact.layer_id):
            return False
        if case == 16 and (len(fact.layer_ids) < 2 or fact.layer_ids[0] != fact.layer_id):
            return False
        if case == 18 and fact.old_present:
            return False
        if case == 19 and fact.gene_id not in view.layers[view.layer_order[0]].nodes:
            return False
        if case == 34 and fact.registry_state != 'null_symbol':
            return False
        return True
    family = fact.layer_family
    families = {7: 'ppi', 8: 'ppi', 9: 'coexpression', 10: 'coexpression', 11: 'tf_target',
        12: 'tf_target', 13: 'tf_target', 14: 'bulk_pen', 15: 'scpen', 16: 'bulk_pen', 17: 'scpen',
        19: 'scpen', 20: 'bulk_pen', 22: 'ppi', 23: 'ppi', 24: 'ppi', 25: 'coexpression',
        26: 'tf_target', 27: 'ppi', 28: 'ppi', 29: 'ppi', 30: 'ppi', 31: 'bulk_pen',
        33: 'tf_target', 36: 'ppi', 37: 'ppi', 38: 'ppi', 40: 'ppi'}
    if case in families and family != families[case]:
        return False
    category = ('no_edge' if case in (8, 10, 12, 19, 20, 26, 37) else 'outside_layer' if case == 28
        else 'outside_multiplex' if case in (29, 38) else 'recorded')
    if case not in (22, 23, 32, 34, 35, 39) and fact.category != category:
        return False
    if case in (12, 26) and not fact.reverse_exists or case == 13 and not fact.reverse_exists:
        return False
    if case in (16, 17):
        return context['features'].for_row({'metadata': {
            'question_family': 'S2.13', 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
            'split': fact.split}}).get('high_weight_control') is True
    if case in (18, 21):
        return family in ('bulk_pen', 'scpen') and any(
            context['features'].multiplex.family_of(other) != family
            and context['features'].multiplex.family_of(other) in ('bulk_pen', 'scpen')
            for other in fact.positive_layers)
    return True


def _tasks(number, case):
    """Return only the task fields requested by the reviewed case."""
    table = {11: TASKS_11, 12: TASKS_12, 13: TASKS_13}[number]
    kind, claim = table.get(case, ({11: 'path', 12: 'membership_card', 13: 'edge_card'}[number], None))
    task = {'question_kind': kind, 'claim': claim}
    if number == 11:
        if case in (20, 36):
            task['truth_scope'] = 'all_layers'
        if case == 6:
            task['membership_cards'] = True
    if number == 12:
        if case in (5, 6):
            task['truth_scope'] = 'category'
        if case == 23:
            task['truth_scope'] = 'multiplex'
        if case in (24, 25, 36):
            task['truth_scope'] = {24: 'all_layers', 25: 'layers', 36: 'expression'}[case]
        if case == 33:
            task['require_membership'] = 'layer'
    return task


def _two_pen_row(context, fact, framing, variant_index, bank, task):
    """Keep two actual positive PEN contexts and their ordered subject cards."""
    base, view = context['module'], context['view']
    other = next(layer for layer in fact.positive_layers
        if context['features'].multiplex.family_of(layer) in ('bulk_pen', 'scpen')
        and context['features'].multiplex.family_of(layer) != fact.layer_family)
    subjects = [base._arguments(view, fact, other), base._arguments(view, fact)]
    selected = base.validate_task(view, fact, subjects=subjects, **task)
    request = {'genes': list(fact.gene_ids), 'multiplex_id': view.multiplex_id,
               'layer_id': fact.layer_id, 'direction': None}
    from .s2_13_prompts import render_s2_13_request
    from .s1_prompt_banks import render_s2_prompt
    core = render_s2_13_request(context['contract']['_requests'], split=fact.split,
        variant_index=variant_index, **request, **selected)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing=framing,
                                    variant_index=variant_index, request=core)
    row_id = 's213_' + digest([context['contract']['contract_version'], view.graph_version,
        asdict(fact), framing, 'two_context_context', selected, user])[:32]
    observations = [_observation('get_edge', source,
        view.get_edge(**source, evidence_id=f'ev_{row_id}_{ordinal}'))
        for ordinal, source in enumerate(subjects, 1)]
    proof, used = base._plan(view, request, observations, selected)
    if proof.next_arguments is not None or proof.issue_code:
        raise ValueError('two supplied PEN observations must settle this evidence-only task')
    answer = base._answer(request, framing, proof, used, observations=observations)
    return {'row_id': row_id, 'messages': [
        {'role': 'system', 'content': context['contract']['system_prompt']},
        {'role': 'harness', 'content': canonical_json({'active_graph': base._active_context(view),
                                                    'observations': observations})},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'thinking': base._reason(answer, framing), 'content': answer}],
        'tools': context['contract']['tools'],
        'metadata': base._metadata(view, fact, fact.split, framing, 'two_context_context', request,
            selected, wrapper, variant_index, context['contract'], context['seed'])}


def _build(context, case_id, fact, framing, wrapper, template):
    """Cross independent audited wording dimensions without changing source facts."""
    number, base, view = (context[key] for key in ('number', 'module', 'view'))
    case = int(case_id.split('.')[-1])
    split = fact.split if fact is not None else context['_family_split']
    bank = {**context['bank'], split: {**context['bank'][split],
        framing: [context['bank'][split][framing][wrapper]]}}
    variant = CASE_BINDINGS[case_id]['variants'][0]
    common = {'registry': context['registry'], 'seed': context['seed']}
    diagnostic = CASE_BINDINGS[case_id]['generator'].endswith('.build_diagnostic')
    if number == 13 and case <= 6:
        family, kind, claim = (
            ('ppi', 'family_card', None), ('coexpression', 'family_support', 'holds_in_other_contexts'),
            ('tf_target', 'family_support', 'regulation_direction'), ('bulk_pen', 'family_card', None),
            ('scpen', 'family_support', 'causal_effect'), ('other', 'family_card', None))[case - 1]
        row = base.build_family_row(view, family, split, framing, template, bank, context['contract'],
            question_kind=kind, claim=claim, seed=context['seed'])
    elif diagnostic:
        row = base.build_diagnostic(view, fact, framing, template, bank, context['contract'], kind=variant, **common)
    else:
        task = _tasks(number, case)
        if number == 13:
            common['old_view'] = context['old_view']
            if case == 23:
                wrong = next(pair for pair in view.layers[fact.layer_id].edges
                    if pair[0] != pair[1] and frozenset(pair) != frozenset(fact.gene_ids)
                    and base._pair_group(pair, context['seed'])[1] == split)
                task['subjects'] = [base._arguments(view, fact), base._arguments(view, fact, gene_ids=wrong)]
            elif case == 26:
                task['subjects'] = [base._arguments(view, fact),
                                    base._arguments(view, fact, gene_ids=fact.gene_ids[::-1])]
        elif number == 12:
            common['old_view'] = context['old_view']
        if number == 13 and case in (18, 21):
            row = _two_pen_row(context, fact, framing, template, bank, task)
        else:
            row = base.build_row(view, fact, framing, variant, template, bank, context['contract'],
                index=context['index'], **common, **task)
    row['metadata']['wrapper_index'] = wrapper
    row['metadata']['catalog_case_id'] = case_id
    return row


def _matching(case_id, row, context):
    """Require the frozen predicate and approved answer envelope before expansion."""
    if '_catalog' not in context:
        from scripts.audit_world_model_s2_case_quotas import catalog_inputs
        cases, answers, _ = catalog_inputs()
        context['_catalog'] = {case.case_id: (case, answers[case.case_id]) for case in cases}
    case, answer = context['_catalog'][case_id]
    final = row['messages'][-1]['content']
    enriched = {**row, '_audit_source': context['features'].for_row(row)}
    return (final['status'] == case.status and final['continuation_decision'] == case.continuation_decision
        and (case.framing_hint is None or row['metadata']['framing'] == case.framing_hint)
        and set(final['result']) == set(answer['result']) and matches(case_id, enriched))


def production_candidates(case_id, context, *, split):
    """Yield finite, distinct-wording candidates in the original fixed source split."""
    if case_id not in CASE_BINDINGS:
        raise ValueError('unknown S2.11–13 production case')
    number = int(case_id.split('.')[1])
    prepared = prepare_context(context, number)
    prepared['_family_split'] = split
    from scripts.audit_world_model_s2_case_quotas import catalog_inputs
    cases, _, _ = catalog_inputs()
    case = next(case for case in cases if case.case_id == case_id)
    framings = [case.framing_hint] if case.framing_hint else [
        'step' if case.continuation_decision == 'continue' else 'direct']
    seeds = []
    for fact in ([None] if number == 13 and int(case_id.split('.')[-1]) <= 6 else _pool(prepared, split)):
        if fact is not None and not _eligible(number, int(case_id.split('.')[-1]), fact, prepared):
            continue
        for framing in framings:
            try:
                row = _build(prepared, case_id, fact, framing, 0, 0)
            except (VariantUnavailable, StopIteration):
                continue
            if _matching(case_id, row, prepared):
                seeds.append((fact, framing))
        if len(seeds) >= 8:
            break
    if not seeds:
        return
    variant = CASE_BINDINGS[case_id]['variants'][0]
    diagnostic = CASE_BINDINGS[case_id]['generator'].endswith('.build_diagnostic')
    kind = seeds[0][0] and _tasks(number, int(case_id.split('.')[-1]))['question_kind']
    if number == 13 and seeds[0][0] is None:
        kind = 'family_card' if int(case_id.split('.')[-1]) in (1, 4, 6) else 'family_support'
    if diagnostic:
        wording = (prepared['module']._diagnostic_wording(variant) if number != 11 else
                   'unknown_layer' if variant == 'tool_unknown_layer' else variant)
        templates = prepared['contract']['_requests']['diagnostic_requests'][split][wording]
    else:
        templates = prepared['contract']['_requests'][split][kind]
    prepared.setdefault('_candidate_capacities', {})[case_id, split] = (
        len(seeds) * len(templates) * len(prepared['bank'][split][seeds[0][1]]))
    for template in range(len(templates)):
        for wrapper in range(len(prepared['bank'][split][seeds[0][1]])):
            for fact, framing in seeds:
                yield _build(prepared, case_id, fact, framing, wrapper, template)


def validate_row(row, context):
    """Replay strict historical semantics, plus the explicit two-PEN extension."""
    case_id = row['metadata'].get('catalog_case_id')
    if case_id not in CASE_BINDINGS:
        raise ValueError('the production row needs an exact reviewed case identity')
    number, case = (int(value) for value in case_id.split('.')[1:])
    prepared = prepare_context(context, number)
    if row['metadata'].get('selection_seed') != prepared['seed']:
        raise ValueError('a production row cannot change the fixed source-selection seed')
    if not _matching(case_id, row, prepared):
        raise ValueError('the production trace does not establish its reviewed case')
    plain = {**row, 'metadata': {key: value for key, value in row['metadata'].items() if key != 'catalog_case_id'}}
    if number == 13 and case in (18, 21):
        from .s2_13_sampling import fact_from_pair
        fact = fact_from_pair(prepared['view'], prepared['index'], prepared['registry'], prepared['old_view'],
            row['metadata']['layer_id'], row['metadata']['gene_ids'], prepared['seed'])
        expected = _build(prepared, case_id, fact, row['metadata']['framing'],
            row['metadata']['wrapper_index'], row['metadata']['request_template_index'])
        if (fact.split != row['metadata']['split'] or row['tools'] != expected['tools']
                or row['metadata'] != expected['metadata'] or row['row_id'] != expected['row_id']):
            raise ValueError('the two-PEN row changed its source split, task, or audited wording')
        for actual, wanted in zip(row['messages'], expected['messages']):
            if actual['role'] != 'assistant':
                if actual != wanted:
                    raise ValueError('the two-PEN observations differ from their complete native source')
            elif any(actual['content'][key] != wanted['content'][key]
                     for key in ('status', 'result', 'provenance', 'continuation_decision')):
                raise ValueError('the two-PEN cards or evidence attribution changed')
            elif (set(actual) != {'role', 'thinking', 'content'} or not isinstance(actual['thinking'], str)
                    or not actual['thinking'].strip() or set(actual['content']) != set(wanted['content'])
                    or not isinstance(actual['content']['summary'], str) or not actual['content']['summary'].strip()
                    or actual['content']['blocked_reason'] is not None):
                raise ValueError('the two-PEN final envelope or free-text fields are malformed')
        if len(row['messages']) != 4:
            raise ValueError('supplied two-PEN evidence needs no acquisition call')
    else:
        kwargs = {'index': prepared['index'], 'bank': prepared['bank'],
                  'contract': prepared['contract'], 'registry': prepared['registry']}
        if number >= 12:
            kwargs['old_view'] = prepared['old_view']
        prepared['module'].validate_row(plain, prepared['view'], **kwargs)


def audit_wording(bank):
    """Compare the new request/wrapper product against every frozen S2 bank."""
    from . import s2_13_prompts as prompts
    from .s1_prompt_banks import _tokens, _ngrams, PromptBankError
    from .s2_4_case_repairs import REQUEST_BANK as ranking_pin
    splits = ('train', 'validation', 'test')
    catalog = json.loads((ROOT / 'runtime/world_model/s2_13_v1.json').read_text())
    renderers = (prompts._rendered_variants, prompts._rendered_variants,
        prompts._neighbor_variants, prompts._map_variants, prompts._membership_variants,
        prompts._classification_variants, prompts._shared_variants, prompts._degree_variants,
        prompts._subgraph_variants, prompts._component_variants, prompts._path_variants,
        prompts._singleton_variants, prompts._specificity_variants)
    prior, pins = {split: [] for split in splits}, []
    for number, renderer in enumerate(renderers, 1):
        pin = catalog['request_bank' if number == 13 else f's2{number}_request_bank']
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('a frozen wording bank changed')
        requests = json.loads((ROOT / pin['path']).read_text())
        rendered = renderer(bank, requests, f'S2.{number}') if number < 3 else renderer(bank, requests)
        for split in splits:
            prior[split].extend(rendered[split])
        pins.append(dict(pin))
    for family in (1, 2):
        path = ROOT / f'runtime/world_model/s2_{family}_production_v1.json'
        production = json.loads(path.read_text())
        if family == 1:
            pin = production['request_bank']
            requests = json.loads((ROOT / pin['path']).read_text())
        else:
            pin = {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)}
            requests = production['diagnostic_leads']
        pins.append(dict(pin))
        for split in splits:
            cores = ([template.format(first='', second='', layer='', multiplex='')
                      for templates in requests[split].values() for template in templates]
                     if family == 1 else [requests[split].strip()])
            for framing in ('direct', 'step'):
                for wrapper in bank[split][framing]:
                    for core in cores:
                        words = _tokens(wrapper.format(request=core))
                        prior[split].append(('production', frozenset(words), _ngrams(words, 5)))
    requests = json.loads((ROOT / ranking_pin['path']).read_text())
    pins.append(dict(ranking_pin))
    for split in splits:
        for wrapper in bank[split]['direct']:
            for template in requests[split]:
                words = _tokens(wrapper.format(request=template.format(gene='', multiplex_id='', scope_text='', top_k='')))
                prior[split].append(('bounded-ranking', frozenset(words), _ngrams(words, 5)))
    current = prompts._specificity_variants(bank, json.loads((ROOT / REQUEST_BANK['path']).read_text()))
    new_pins = [dict(REQUEST_BANK)]

    def include_new(rendered, pin):
        for selected_split in splits:
            current[selected_split].extend(rendered[selected_split])
        new_pins.append(pin)

    for number, renderer in ((3, prompts._neighbor_variants), (11, prompts._path_variants)):
        path = ROOT / f'runtime/world_model/s2_production_{number}_requests_v1.json'
        requests = json.loads(path.read_text())
        validator = importlib.import_module(f'runtime.world_model.s2_{number}_prompts')
        getattr(validator, f'validate_s2_{number}_requests')(requests)
        include_new(renderer(bank, requests), {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)})
    additions_path = ROOT / 'runtime/world_model/s2_production_requests_9_10_v1.json'
    if additions_path.exists():
        additions = json.loads(additions_path.read_text())
        pins.append({'path': str(additions_path.relative_to(ROOT)), 'sha256': file_sha256(additions_path)})
        for number, renderer in ((9, prompts._subgraph_variants), (10, prompts._component_variants)):
            base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
            requests = deepcopy(base.load_build_contract()['_requests'])
            for split in splits:
                if number == 9:
                    for kind in ('complete', 'top_k'):
                        requests[split][kind].extend(additions[f's2_{number}'][split])
                else:
                    requests[split]['partition'].extend(additions[f's2_{number}'][split])
                    requests['diagnostic_requests'][split]['unavailable_kind'].extend(additions[f's2_{number}'][split])
            rendered = renderer(bank, requests)
            include_new(rendered, {'path': str(additions_path.relative_to(ROOT)), 'sha256': file_sha256(additions_path)})
    vocabulary = sorted({word for rows in (*prior.values(), *current.values()) for _, words, _ in rows for word in words})
    import numpy as np
    from scipy.sparse import csr_matrix
    columns = {word: index for index, word in enumerate(vocabulary)}

    def packed(rows):
        # Sparse integer products count set intersections exactly. Chunking the
        # resulting dense blocks bounds audit memory independently of bank size.
        words = sorted({tuple(sorted(words)) for _, words, _ in rows})
        indices, pointers = [], [0]
        for row in words:
            indices.extend(columns[word] for word in row)
            pointers.append(len(indices))
        return (csr_matrix((np.ones(len(indices), dtype=np.int16), indices, pointers),
                           shape=(len(words), len(vocabulary))), np.array([len(row) for row in words]))

    maximum, compared = 0., 0
    for split in splits:
        for other in splits:
            if split == other:
                continue
            reference = prior[other] + current[other]
            grams = {gram: label for label, _, runs in reference for gram in runs}
            for label, _, runs in current[split]:
                shared = runs & grams.keys()
                if shared:
                    gram = min(shared)
                    raise PromptBankError(f'new wording shares a held-out five-word run: {split}/{other}; '
                                          f'{label}; {grams[gram]}; {gram}')
            compared += len(current[split]) * len(reference)
            left, sizes = packed(current[split])
            right, other_sizes = packed(reference)
            for start in range(0, left.shape[0], 128):
                common = (left[start:start + 128] @ right.T).toarray()
                unions = sizes[start:start + 128, None] + other_sizes[None, :] - common
                value = float((common / unions).max())
                maximum = max(maximum, value)
                if value >= .6:
                    raise PromptBankError(f'new specificity wording overlaps held-out wording: {value:.6f}')
    return {'cross_split_comparisons': compared, 'maximum_cross_split_jaccard': maximum,
            'shared_cross_split_5grams': 0, 'request_bank': dict(REQUEST_BANK), 'reference_pins': pins,
            'new_request_banks': list({pin['path']: pin for pin in new_pins}.values()),
            'rendered_wordings': {split: len(rows) for split, rows in current.items()}}
