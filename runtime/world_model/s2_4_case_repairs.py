"""Add native failure partitions and bounded ranking diagnostics to frozen S2.4.

Existing prompts already specify atomic native-input failure coverage. This
extension supplies that coverage even when the next acquisition batches whole
missing layers, and preserves a positive bound in unresolved ranking requests.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
from types import SimpleNamespace

from . import s2_4_corpus as base
from .s1_prompt_banks import render_s2_prompt
from .s2_2_corpus import VariantUnavailable, _observation
from .s2_audit_cases_3_6 import matches
from .s2_corpus import ROOT, SPLITS, _failure, canonical_json, digest
from .s2_multiplex import MultiplexRuntime
from .s2_neighbor_layers import NeighborLayersRuntime
from .s2_neighbors import NeighborRuntime
from .sources import file_sha256


CASE_IDS = ('S2.4.14', 'S2.4.16', 'S2.4.20', 'S2.4.22')
REPAIR = 's2_4_native_failure_ranking_repairs_v1'
CONTRACT_PATH = base.CONTRACT_PATH
REQUEST_BANK = {'path': 'runtime/world_model/s2_4_case_repairs_requests_v1.json',
                'sha256': '33b211093430090f41e4a2235001b64df79839361748b73c71687c5889f75072'}
_VARIANTS = dict(zip(CASE_IDS, ('native_terminal', 'native_exhausted', 'native_terminal', 'ranking_ambiguous')))


def load_build_contract(path=CONTRACT_PATH):
    """Load unchanged graph semantics plus a separately pinned bounded ranking bank."""
    contract = base.load_build_contract(path)
    request_path = ROOT / REQUEST_BANK['path']
    if file_sha256(request_path) != REQUEST_BANK['sha256']:
        raise ValueError('the bounded ranking wording changed')
    requests = json.loads(request_path.read_text())
    if requests.get('schema_version') != 'mentor-rl-s2-4-bounded-ranking-requests-v1':
        raise ValueError('unsupported bounded ranking request schema')
    contract['_repair_requests'] = requests
    contract['repair_request_bank'] = dict(REQUEST_BANK)
    return contract


def _visible(view):
    """Limit proof decisions to manifest representation and model-visible evidence."""
    return SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                            for layer in view.layer_order})


def _native_coverage(view, arguments, observations):
    """Partition atomic native inputs using complete collections or layer absence."""
    from .s2_4_evidence import _native_inputs, _native_runtime, _native_proof
    visible = _visible(view)
    selected = MultiplexRuntime(visible).resolve_scope(arguments['layer_scope'])
    inputs = _native_inputs(visible, selected, arguments['direction'])
    native = _native_runtime(visible)
    checked, absent = [], set()
    for entry in inputs:
        pending = {'multiplex_id': arguments['multiplex_id'], 'gene_id': arguments['gene_id'], **entry,
                   'mode': 'complete', 'top_k': None, 'cursor': None}
        proof = _native_proof(native, pending, observations)
        if proof.result is not None:
            checked.append(entry)
            if proof.result['not_in_layer_gene_ids']:
                absent.add(entry['layer_id'])
    checked = [entry for entry in inputs if entry in checked or entry['layer_id'] in absent]
    return {'operation': 'neighbor_collections', 'requested_inputs': inputs,
            'checked_inputs': checked, 'unchecked_inputs': [entry for entry in inputs if entry not in checked]}


def _ranking_row(view, fact, bank, contract, *, seed, variant_index):
    """Preserve the requested strongest-two bound before any graph acquisition."""
    row = base.build_diagnostic(view, fact, 'direct', variant_index, bank, contract, ranking=True, seed=seed)
    requests = contract.get('_repair_requests') or load_build_contract()['_repair_requests']
    selected = MultiplexRuntime(view).resolve_scope(fact.layer_scope)
    if len(selected) < 2:
        raise ValueError('cross-layer ranking requires at least two selected layers')
    scope_text = 'layers [' + ', '.join(fact.layer_scope['layer_ids']) + ']'
    templates = requests[fact.split]
    core = templates[variant_index % len(templates)].format(gene=fact.gene_id, multiplex_id=view.multiplex_id,
                                                           scope_text=scope_text, top_k=2)
    user, wrapper = render_s2_prompt(bank, split=fact.split, framing='direct', variant_index=variant_index, request=core)
    row['messages'][2]['content'] = user
    row['messages'][-1]['content']['result']['request'].update(mode='top_k', top_k=2)
    row['messages'][-1]['thinking'] = (
        f"The request asks for the strongest two neighbors of {fact.gene_id} across {fact.layer_scope}. "
        'It supplies no cross-layer ranking rule. Preserve mode top_k and K=2 in the unresolved request; '
        'a canonical-ID prefix would answer a different question. Make no graph call and defer for that rule.')
    row['metadata'].update(mode='top_k', top_k=2, wrapper_index=wrapper,
                           request_template_index=variant_index % len(templates))
    return row


def build_row(view, fact, case_id, bank, contract, registry=None, *, index=None, seed=20261028,
              variant_index=0):
    """Construct a missing approved case from fresh source acquisitions."""
    index = getattr(index, 'native_index', index)
    if case_id not in CASE_IDS:
        raise ValueError('unsupported S2.4 repaired case')
    if case_id == 'S2.4.22':
        row = _ranking_row(view, fact, bank, contract, seed=seed, variant_index=variant_index)
    else:
        row = base.build_row(view, fact, 'complete', None, 'step', _VARIANTS[case_id], variant_index,
                             bank, contract, index=index, registry=registry, seed=seed)
        arguments = base._arguments(view, fact, 'complete', None)
        runtime = NeighborRuntime(view, index=index)
        selected = MultiplexRuntime(view).resolve_scope(fact.layer_scope)
        undirected = next((layer for layer in selected if not view.layers[layer].directed), None)
        if undirected is None:
            raise VariantUnavailable('positive undirected saved evidence is required')
        inputs = [{'layer_id': undirected, 'direction': 'undirected'}]
        if case_id == 'S2.4.20':
            directed = next((layer for layer in selected if view.layers[layer].directed), None)
            if directed is None:
                raise VariantUnavailable('incoming directed saved evidence is required')
            inputs.append({'layer_id': directed, 'direction': 'in'})
        observations = []
        for number, entry in enumerate(inputs):
            saved = {'multiplex_id': view.multiplex_id, 'gene_id': fact.gene_id, **entry,
                     'mode': 'complete', 'top_k': None, 'cursor': None}
            response = runtime.get_neighbors(**saved, evidence_id=f"ev_{row['row_id']}_repair_{number}")
            if not response['result']['neighbors'] or not response['result']['collection']['complete']:
                raise VariantUnavailable('the original native list must be complete and positive')
            observations.append(_observation('get_neighbors', saved, response))
        proof, used = base._plan(_visible(view), arguments, observations)
        if proof.next_arguments is None:
            raise VariantUnavailable('the original native observations must leave required work')
        row['messages'][1]['content'] = canonical_json({'active_graph': base._active(view), 'observations': observations})
        call = {'name': proof.next_tool, 'arguments': deepcopy(proof.next_arguments)}
        failure = _failure(1 if case_id == 'S2.4.16' else None)
        messages = [{'role': 'assistant', 'thinking':
            f"The request concerns {fact.gene_id} across {fact.layer_scope}. Reuse the complete native observations. "
            f"Only the uncovered acquisition {call} remains; failed work cannot invalidate earlier evidence.",
            'tool_call': call}, {'role': 'tool', 'name': call['name'], 'content': failure}]
        if case_id == 'S2.4.16':
            messages.extend([{'role': 'assistant', 'thinking':
                'The runtime explicitly supplies retries_remaining=1. Repeat the identical acquisition once, '
                'including its scope and cursor, and stop after exhaustion.', 'tool_call': deepcopy(call)},
                {'role': 'tool', 'name': call['name'], 'content': _failure(0)}])
        coverage = _native_coverage(view, arguments, observations)
        answer = {'status': 'error', 'provenance': base._provenance(used), 'result': {
            'failed_tool': call['name'], 'error_code': failure['error_code'], 'arguments': call['arguments'],
            'partial_results': used, 'coverage': coverage}, 'continuation_decision': 'defer',
            'blocked_reason': 'A required native neighbor collection remains unavailable.',
            'summary': 'Preserve the complete native observations while missing collections await recovery.'}
        messages.append({'role': 'assistant', 'thinking':
            f"The visible atomic native-input partition is {coverage}. Keep the {len(used)} original observations "
            'and their acquisition provenance. The failed call adds no evidence. Unchecked native directions remain '
            'unknown, so a full-scope union and support counts cannot be claimed. Defer for backend recovery.', 'content': answer})
        row['messages'][3:] = messages
    row['metadata'].update(catalog_case_id=case_id, repair=REPAIR)
    row['row_id'] = 's24repair_' + digest([REPAIR, case_id, row['messages'][2]['content'], fact.split_unit])[:32]
    return row


def build_case_rows(view, index, bank, contract, registry=None, *, split, seed=20261028):
    """Choose compact positive native facts in each unchanged subject split."""
    if split not in SPLITS:
        raise ValueError('a fixed source split is required')
    runtime = NeighborLayersRuntime(view, index=getattr(index, 'native_index', index))
    index = runtime.index
    subjects = [gene for gene in view.multiplex_nodes if base.gene_group(gene, seed)[1] == split]
    rows, selections = [], []
    for number, case_id in enumerate(CASE_IDS):
        candidates = []
        for pair in combinations(view.layer_order, 2):
            undirected = [layer for layer in pair if not view.layers[layer].directed]
            directed = [layer for layer in pair if view.layers[layer].directed]
            if len(undirected) != (1 if case_id == 'S2.4.20' else 2):
                continue
            for gene in subjects:
                n = len(index.neighbors(undirected[0], gene, 'undirected'))
                incoming = len(index.neighbors(directed[0], gene, 'in')) if directed else 0
                if n <= 0 or case_id == 'S2.4.20' and incoming <= 0:
                    continue
                candidates.append((n + incoming, base.gene_group(gene, seed)[0], pair, gene))
        if not candidates:
            raise VariantUnavailable(f'no real {case_id} source facts in fixed {split} split')
        _, _, pair, gene = min(candidates)
        scope = {'kind': 'layers', 'layer_ids': list(pair), 'layer_family': None}
        fact = base._fact(runtime, gene, scope, 'incident', seed)
        row = build_row(view, fact, case_id, bank, contract, registry, index=index, seed=seed, variant_index=number)
        validate_row(row, view, index=index, bank=bank, contract=contract, registry=registry)
        rows.append(row)
        selections.append({'case_id': case_id, 'candidate_population': len(candidates), 'gene_id': gene, 'layer_scope': scope})
    return rows, {'split': split, 'selections': selections, 'split_policy': 'unchanged gene_group(seed,gene)',
                  'source_selection': 'native positive collections; smallest saved row count, then fixed subject hash'}


def validate_row(row, view, *, index=None, bank, contract, registry=None):
    """Replay source responses and independently verify the repaired visible semantics."""
    index = getattr(index, 'native_index', index)
    meta = row['metadata']
    case_id = meta.get('catalog_case_id')
    ranking = case_id == 'S2.4.22'
    if (set(row) != {'row_id', 'messages', 'tools', 'metadata'}
            or case_id not in CASE_IDS or meta.get('repair') != REPAIR or meta.get('variant') != _VARIANTS[case_id]
            or meta.get('framing') != ('direct' if ranking else 'step') or meta.get('direction') != 'incident'
            or meta.get('mode') != ('top_k' if ranking else 'complete')
            or meta.get('top_k') != (2 if ranking else None) or meta.get('layer_scope', {}).get('kind') != 'layers'
            or meta.get('ranking') != ('strongest_unspecified' if ranking else None)):
        raise ValueError('the repair identity, case binding, or approved request changed')
    expected_id = 's24repair_' + digest([REPAIR, case_id, row['messages'][2]['content'], meta['split_unit']])[:32]
    if row['row_id'] != expected_id:
        raise ValueError('the repaired row identity differs from its case and visible question')
    selected = MultiplexRuntime(view).resolve_scope(meta['layer_scope'])
    directed_count = sum(view.layers[layer].directed for layer in selected)
    if len(selected) != 2 or directed_count != (1 if case_id == 'S2.4.20' else 0):
        raise ValueError('the approved two-layer native representation selection changed')
    checked = deepcopy(row)
    if ranking:
        fact = base.MapFact(meta['split'], meta['gene_id'], meta['layer_scope'], 'incident', 'diagnostic', None, (), meta['split_unit'])
        expected = _ranking_row(view, fact, bank, contract, seed=meta['selection_seed'], variant_index=meta['request_template_index'])
        if (row['messages'][:3] != expected['messages'][:3] or row['tools'] != expected['tools']
                or any(meta.get(key) != value for key, value in expected['metadata'].items())
                or any(canonical_json(row['messages'][-1]['content'][key]) != canonical_json(expected['messages'][-1]['content'][key])
                       for key in ('status', 'provenance', 'result', 'continuation_decision'))):
            raise ValueError('the bounded strongest request was changed or acquired graph evidence')
        original = base.build_diagnostic(view, fact, 'direct', meta['request_template_index'], bank, contract,
                                         ranking=True, seed=meta['selection_seed'])
        checked['metadata'].update(mode='complete', top_k=None)
        checked['messages'][2] = original['messages'][2]
        checked['messages'][-1]['content']['result'] = original['messages'][-1]['content']['result']
    else:
        observations = json.loads(row['messages'][1]['content'])['observations']
        if len(observations) != (2 if case_id == 'S2.4.20' else 1):
            raise ValueError('the original native context was removed or expanded')
        for observation in observations:
            if (observation['operation'] != 'get_neighbors' or observation['arguments']['mode'] != 'complete'
                    or not observation['result']['neighbors'] or observation['result']['collection']['complete'] is not True):
                raise ValueError('complete positive original native context is required')
        runtime = NeighborLayersRuntime(view, index=index)
        fact = base._fact(runtime, meta['gene_id'], meta['layer_scope'], meta['direction'], meta['selection_seed'])
        if meta['transport'] != base._transport(fact, _VARIANTS[case_id]):
            raise ValueError('the original native transport declaration changed')
        arguments = base._arguments(view, fact, meta['mode'], meta['top_k'])
        proof, _ = base._plan(_visible(view), arguments, observations)
        coverage = _native_coverage(view, arguments, observations)
        if row['messages'][-1]['content']['result']['coverage'] != coverage:
            raise ValueError('atomic coverage exceeds the complete visible native collections')
        # Source replay and the old planner retain their published behavior.
        # Normalize only the expected coverage before delegating that audit.
        checked['messages'][-1]['content']['result']['coverage'] = proof.coverage
        errors = [m['content'] for m in row['messages'] if m['role'] == 'tool' and m['content']['status'] == 'error']
        if errors != ([_failure(1), _failure(0)] if case_id == 'S2.4.16' else [_failure(None)]):
            raise ValueError('the approved terminal or exhausted fault schedule changed')
    base.validate_row(checked, view, index=index, bank=bank, contract=contract, registry=registry)
    if not matches(case_id, row):
        raise ValueError('the repaired trace does not establish its approved catalog case')


def audit_request_wording(contract=None, bank=None):
    """Check bounded ranking prose against every frozen and repaired S2 split."""
    from . import s2_13_prompts as prior_prompts
    from .s1_prompt_banks import _tokens, _ngrams, load_s2_prompt_banks, PromptBankError
    contract = load_build_contract() if contract is None else contract
    bank = load_s2_prompt_banks(contract['prompt_bank']) if bank is None else bank
    catalog_path = ROOT / 'runtime/world_model/s2_13_v1.json'
    catalog = json.loads(catalog_path.read_text())
    renderers = (prior_prompts._rendered_variants, prior_prompts._rendered_variants,
        prior_prompts._neighbor_variants, prior_prompts._map_variants, prior_prompts._membership_variants,
        prior_prompts._classification_variants, prior_prompts._shared_variants, prior_prompts._degree_variants,
        prior_prompts._subgraph_variants, prior_prompts._component_variants, prior_prompts._path_variants,
        prior_prompts._singleton_variants, prior_prompts._specificity_variants)
    prior, current, pins = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}, []
    for number, renderer in enumerate(renderers, 1):
        pin = catalog['request_bank' if number == 13 else f's2{number}_request_bank']
        path = ROOT / pin['path']
        if file_sha256(path) != pin['sha256']:
            raise ValueError('a frozen S2 request bank changed')
        pins.append(dict(pin))
        requests = json.loads(path.read_text())
        rendered = renderer(bank, requests, f'S2.{number}') if number < 3 else renderer(bank, requests)
        for split in SPLITS:
            prior[split].extend(rendered[split])
    # The earlier new S2.1 and S2.2 prose also belongs to the held-out boundary.
    for family in (1, 2):
        path = ROOT / f'runtime/world_model/s2_{family}_production_v1.json'
        production = json.loads(path.read_text())
        if family == 1:
            pin = production['request_bank']
            request_path = ROOT / pin['path']
            if file_sha256(request_path) != pin['sha256']:
                raise ValueError('a production request bank changed')
            requests = json.loads(request_path.read_text())
        else:
            pin = {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)}
            requests = production['diagnostic_leads']
        pins.append(dict(pin))
        for split in SPLITS:
            for framing in ('direct', 'step'):
                for index, wrapper in enumerate(bank[split][framing]):
                    cores = ([template.format(first='', second='', layer='', multiplex='')
                              for templates in requests[split].values() for template in templates]
                             if family == 1 else [requests[split].strip()])
                    for ordinal, core in enumerate(cores):
                        tokens = _tokens(wrapper.format(request=core))
                        prior[split].append((f'S2.{family}.production/{split}/{framing}/{index}/{ordinal}',
                                             frozenset(tokens), _ngrams(tokens, 5)))
    requests = contract.get('_repair_requests') or load_build_contract()['_repair_requests']
    for split in SPLITS:
        for index, wrapper in enumerate(bank[split]['direct']):
            for ordinal, template in enumerate(requests[split]):
                core = template.format(gene='', multiplex_id='', scope_text='', top_k='')
                tokens = _tokens(wrapper.format(request=core))
                current[split].append((f'S2.4.repair/{split}/{index}/{ordinal}', frozenset(tokens), _ngrams(tokens, 5)))
    vocabulary = sorted({word for rows in (*prior.values(), *current.values()) for _, words, _ in rows for word in words})
    bits = {word: 1 << index for index, word in enumerate(vocabulary)}

    def packed(rows):
        return {(words, sum(bits[word] for word in words), len(words)) for _, words, _ in rows}

    maximum, compared = 0., 0
    for split in SPLITS:
        for other in SPLITS:
            if split == other:
                continue
            reference = prior[other] + current[other]
            grams = {gram: label for label, _, runs in reference for gram in runs}
            for label, _, runs in current[split]:
                shared = runs & grams.keys()
                if shared:
                    raise PromptBankError(f'bounded ranking shares a held-out five-word run: {label}, {grams[min(shared)]}')
            compared += len(current[split]) * len(reference)
            for _, words, size in packed(current[split]):
                for _, other_words, other_size in packed(reference):
                    common = (words & other_words).bit_count()
                    score = common / (size + other_size - common)
                    maximum = max(maximum, score)
                    if score >= .6:
                        raise PromptBankError(f'bounded ranking overlaps held-out wording: Jaccard={score:.4f}')
    return {'scope': 'new bounded-ranking/direct-wrapper combinations versus all thirteen frozen banks and both production extensions',
        'masking': 'bound gene, multiplex, scope, and positive K only', 'cross_split_pairs_checked': compared,
        'maximum_cross_split_jaccard': round(maximum, 6), 'shared_cross_split_5grams': 0,
        'new_wordings': {s: len(rows) for s, rows in current.items()}, 'request_bank_pin': dict(REQUEST_BANK),
        'reference_request_pins': pins, 'frozen_pin_catalog_sha256': file_sha256(catalog_path)}
