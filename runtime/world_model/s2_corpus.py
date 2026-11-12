"""Build the first source-backed S2 component without changing S0/S1 artifacts.

This version implements single ordered or undirected edge checks. It uses the
approved S2 envelopes in a separate row contract; the historical trainer cannot
consume these rows until its S2 adapter and tokenizer checks are implemented.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .s1_prompt_banks import (
    BUILDER_MODULES, load_s2_prompt_banks, render_s2_prompt,
    validate_s2_prompt_banks, load_s2_edge_requests, render_s2_edge_request,
    validate_s2_rendered_prompts,
)
from .s2_sources import GraphView, load_source_view
from .source_index import content_sha256
from .sources import file_sha256


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path(__file__).with_name('s2_v1.json')
SCHEMA_VERSION = 'mentor-rl-world-model-s2-corpus-v1'
SPLITS = ('train', 'validation', 'test')
FRAMINGS = ('direct', 'step')
VARIANTS = ('lookup', 'context', 'terminal', 'recovered', 'exhausted')
BUCKETS = ('recorded', 'no_edge', 'outside_layer', 'outside_multiplex')
DEFAULT_COUNTS = {'train': 8, 'validation': 4, 'test': 4}


@dataclass(frozen=True)
class EdgeFact:
    """One graph query selected before wording or fault contrasts are assigned."""

    split: str
    layer_id: str
    gene_ids: tuple[str, str]
    bucket: str
    split_unit: str


def canonical_json(value: Any) -> str:
    """Serialize build identities independently of dictionary insertion order."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value: Any) -> str:
    """Hash a canonical JSON value for reproducible selection and artifact IDs."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_build_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the isolated S2 build contract, checking its implementation boundary."""
    contract = json.loads(Path(path).read_text())
    if contract['row_schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported S2 row schema')
    if contract['implemented_families'] != ['S2.1.single_edge']:
        raise ValueError('this builder implements only S2.1 single-edge queries')
    if contract['training_ready'] is not False:
        raise ValueError('this component has no historical-trainer adapter')
    contract['_edge_requests'] = load_s2_edge_requests(contract['request_bank'])
    return contract


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _registry_ids(pin: Mapping[str, str]) -> tuple[str, ...]:
    path = _path(pin['path'])
    if file_sha256(path) != pin['sha256']:
        raise ValueError('S2 registry index hash mismatch')
    with gzip.open(path, 'rt') as stream:
        payload = json.load(stream)
    if content_sha256(payload) != pin['logical_sha256']:
        raise ValueError('S2 registry logical hash mismatch')
    return tuple(sorted(payload['genes_by_id']))


def _pair_group(pair: tuple[str, str], seed: int) -> tuple[str, str]:
    # Ignore orientation and layer: all contrasts on the same endpoints stay
    # together, including future queries about the reversed edge or other layers.
    unit = digest({'seed': seed, 'gene_ids': sorted(pair)})
    bin_number = int(unit[:16], 16) % 10
    split = 'train' if bin_number < 8 else 'validation' if bin_number == 8 else 'test'
    return unit, split


def select_facts(
    view: GraphView,
    registry_ids: Sequence[str],
    counts: Mapping[str, int],
    seed: int,
) -> list[EdgeFact]:
    """Sample real query strata with pair-grouped splits, before choosing prompts.

    Counts are per layer, outcome bucket, and split. Rejection sampling selects
    checked negatives from full layer inventories. A missing coverage population
    is an error, never permission to invent a node or edge.
    """
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('counts must supply positive integers for all three splits')
    outside = sorted(set(registry_ids) - view.multiplex_nodes)
    facts: list[EdgeFact] = []
    for layer_id in view.layer_order:
        layer = view.layers[layer_id]
        nodes = sorted(layer.nodes)
        elsewhere = sorted(view.multiplex_nodes - layer.nodes)
        edge_pairs = list(layer.edges)
        if len(nodes) < 2 or not edge_pairs or not elsewhere or not outside:
            raise ValueError(f'insufficient populations for all four strata: {layer_id}')
        for bucket in BUCKETS:
            rng = random.Random(int(digest([seed, layer_id, bucket]), 16))
            selected: dict[str, list[EdgeFact]] = {split: [] for split in SPLITS}
            seen: set[tuple[str, str]] = set()
            for _ in range(max(10000, sum(counts.values()) * 1000)):
                if all(len(selected[s]) == counts[s] for s in SPLITS):
                    break
                if bucket == 'recorded':
                    pair = rng.choice(edge_pairs)
                    if not layer.directed and rng.randrange(2):
                        pair = pair[::-1]
                elif bucket == 'no_edge':
                    pair = tuple(rng.sample(nodes, 2))
                else:
                    excluded = rng.choice(elsewhere if bucket == 'outside_layer' else outside)
                    pair = (excluded, rng.choice(nodes))
                    if rng.randrange(2):
                        pair = pair[::-1]
                if pair[0] == pair[1]:
                    continue
                key = pair if layer.directed else tuple(sorted(pair))
                if bucket == 'no_edge' and key in layer.edges:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                unit, split = _pair_group(pair, seed)
                if len(selected[split]) < counts[split]:
                    selected[split].append(EdgeFact(split, layer_id, pair, bucket, unit))
            if any(len(selected[s]) != counts[s] for s in SPLITS):
                raise ValueError(f'cannot fill real query strata for {layer_id}/{bucket}')
            for split in SPLITS:
                facts.extend(selected[split])
    return sorted(facts, key=lambda f: (SPLITS.index(f.split), view.layer_order.index(f.layer_id), BUCKETS.index(f.bucket), f.gene_ids))


def _result_bucket(result: Mapping[str, Any]) -> str:
    if result['missing_gene_ids']:
        return 'outside_multiplex'
    if result['not_in_layer_gene_ids']:
        return 'outside_layer'
    return 'recorded' if result['edge_exists'] else 'no_edge'


def _request(view: GraphView, fact: EdgeFact, variant_index: int, contract: Mapping[str, Any]) -> str:
    first, second = fact.gene_ids
    return render_s2_edge_request(contract['_edge_requests'], fact.split,
        view.layers[fact.layer_id].directed, variant_index, multiplex_id=view.multiplex_id,
        layer_id=fact.layer_id, first=first, second=second)


def _failure(remaining: int | None) -> dict[str, Any]:
    message = 'The graph backend is unavailable.'
    if remaining is not None:
        message += f' retries_remaining={remaining}'
    return {'status': 'error', 'provenance': {'registry': [], 'graph': []},
            'error_code': 'BACKEND_UNAVAILABLE', 'message': message}


def _outcome_reason(result: Mapping[str, Any], framing: str) -> str:
    first, second = result['gene_ids']
    relation = f'{first} → {second}' if result['directed'] else f'{first} and {second}'
    scope = f"{result['layer_id']} in {result['multiplex_id']}"
    if result['missing_gene_ids']:
        missing = ', '.join(result['missing_gene_ids'])
        return (f'The membership check places {missing} outside the multiplex. The edge predicate for '
                f'{relation} in {scope} is unavailable, so edge_exists and weight remain null. '
                'Retain both endpoint membership findings and their graph evidence. The required '
                'edge prerequisite is absent, so defer even if this is part of a larger workflow.')
    if result['not_in_layer_gene_ids']:
        absent = ', '.join(result['not_in_layer_gene_ids'])
        fact = (f'{absent} belongs to the multiplex but is absent from {result["layer_id"]}. '
                f'Therefore no edge {relation} is recorded in this scope: edge_exists is false and weight is null. '
                'This is a completed scoped negative, not an unavailable predicate or a biological non-interaction claim.')
    elif result['edge_exists']:
        fact = (f'The matching record for {relation} in {scope} has weight {result["weight"]}. '
                'Both endpoints are layer members. Copy the recorded weight and native endpoint roles; '
                'the record does not establish causality or a weight comparison across other layers.')
    else:
        fact = (f'Both {first} and {second} are members of {scope}, but the requested edge is not recorded. '
                'Set edge_exists to false and weight to null. This does not rule out a longer path, '
                'an edge in another layer, or a biological relationship.')
    ending = ('The user asked for this result as an input to a later step; this step is complete, so continue the stated workflow.'
              if framing == 'step' else 'This answers the complete standalone request, so stop.')
    return fact + ' Keep the supporting graph evidence. ' + ending


def _summary(result: Mapping[str, Any]) -> str:
    pair = (' → ' if result['directed'] else ' and ').join(f'<GENE:{g}>' for g in result['gene_ids'])
    layer = f'<LAYER:{result["layer_id"]}>'
    if result['missing_gene_ids']:
        absent = ', '.join(f'<GENE:{g}>' for g in result['missing_gene_ids'])
        return f'The edge for {pair} in {layer} was not evaluated: {absent} is outside <MULTIPLEX:{result["multiplex_id"]}>.'
    if result['edge_exists']:
        return f'{layer} records the edge for {pair} at weight {result["weight"]}.'
    return f'No edge for {pair} is recorded in {layer}; the checked membership categories are retained.'


def build_row(
    view: GraphView,
    fact: EdgeFact,
    framing: str,
    variant: str,
    variant_index: int,
    bank: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one fully bound trace with visible evidence and explicit fault policy."""
    if framing not in FRAMINGS or variant not in VARIANTS:
        raise ValueError('unsupported framing or evidence/failure variant')
    user, wrapper_index = render_s2_prompt(bank, split=fact.split, framing=framing,
                                         variant_index=variant_index, request=_request(view, fact, variant_index, contract))
    row_id = 's2_' + digest([contract['contract_version'], view.graph_version, asdict(fact), framing, variant, user])[:32]
    arguments = {'multiplex_id': view.multiplex_id, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids)}
    observation = view.get_edge(**arguments, evidence_id='ev_' + row_id + '_edge')
    if _result_bucket(observation['result']) != fact.bucket:
        raise ValueError('selected fact does not match its source outcome stratum')
    context = {'active_graph': {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
               'layers': [{'layer_id': k, 'directed': view.layers[k].directed} for k in view.layer_order]},
               'observations': []}
    if variant == 'context':
        context['observations'].append({'operation': 'get_edge', 'arguments': arguments,
                                       'status': observation['status'], 'result': observation['result'],
                                       'provenance': observation['provenance']})
    messages = [{'role': 'system', 'content': contract['system_prompt']},
                {'role': 'harness', 'content': canonical_json(context)},
                {'role': 'user', 'content': user}]
    first, second = fact.gene_ids
    direction = f'the ordered edge {first} → {second}' if view.layers[fact.layer_id].directed else f'the undirected edge between {first} and {second}'
    initial = (f'The request asks about {direction} in {fact.layer_id} of {view.multiplex_id}. '
               f'The active manifest identifies the layer as {"directed" if view.layers[fact.layer_id].directed else "undirected"}. '
               'The canonical IDs can go directly to get_edge; their spelling alone does not prove graph membership. '
               'No matching edge observation is supplied, so check both endpoint memberships and the requested edge in one call.')

    def append_call(thinking: str, response: dict[str, Any]) -> None:
        messages.extend([{'role': 'assistant', 'thinking': thinking,
                          'tool_call': {'name': 'get_edge', 'arguments': dict(arguments)}},
                         {'role': 'tool', 'name': 'get_edge', 'content': response}])

    if variant == 'lookup':
        append_call(initial, observation)
    elif variant in ('terminal', 'recovered', 'exhausted'):
        append_call(initial, _failure(None if variant == 'terminal' else 1))
        if variant != 'terminal':
            retry = (f'The failed lookup of {direction} in {fact.layer_id} supplies retries_remaining=1. '
                     'It provides no new graph evidence. Use the one permitted retry with the identical multiplex, layer, and ordered gene_ids; do not change the question.')
            append_call(retry, observation if variant == 'recovered' else _failure(0))
    if variant in ('terminal', 'exhausted'):
        allowance = 'No retry allowance was supplied.' if variant == 'terminal' else 'The permitted retry also failed and retries_remaining=0.'
        reasoning = (f'The required lookup for {direction} in {fact.layer_id} failed. {allowance} '
                     'Neither endpoint membership nor edge existence was established by the failed calls. '
                     'Keep the exact failed arguments, omit uncomputed edge fields, and report no successful partial results. '
                     'The requested operation remains incomplete, so defer for backend recovery rather than returning false, zero, or an empty graph result.')
        answer = {'status': 'error', 'provenance': {'registry': [], 'graph': []},
                  'result': {'failed_tool': 'get_edge', 'error_code': 'BACKEND_UNAVAILABLE',
                             'arguments': arguments, 'partial_results': [], 'coverage': None},
                  'continuation_decision': 'defer', 'blocked_reason': 'The required graph backend is unavailable.',
                  'summary': 'The required edge lookup failed; no graph predicate or membership result was obtained.'}
    else:
        reasoning = _outcome_reason(observation['result'], framing)
        if variant == 'context':
            reasoning = (f'The saved get_edge observation matches {first}, {second}, {fact.layer_id}, '
                         'the active multiplex and graph version, and the native representation. It includes the weight '
                         'and both membership categories, so no new call is needed. ' + reasoning)
        elif variant == 'recovered':
            reasoning = 'The authorized retry returned a checked result; no further retry is needed. ' + reasoning
        status = observation['status']
        answer = {'status': status, 'provenance': observation['provenance'], 'result': observation['result'],
                  'continuation_decision': 'defer' if status == 'not_found' else 'continue' if framing == 'step' else 'stop',
                  'blocked_reason': 'A required endpoint is outside the active multiplex.' if status == 'not_found' else None,
                  'summary': _summary(observation['result'])}
    messages.append({'role': 'assistant', 'thinking': reasoning, 'content': answer})
    return {'row_id': row_id, 'messages': messages, 'tools': contract['tools'],
            'metadata': {'schema_version': SCHEMA_VERSION, 'tool_contract_version': contract['contract_version'],
                         'system_prompt_sha256': hashlib.sha256(contract['system_prompt'].encode()).hexdigest(),
                         'question_family': 'S2.1.single_edge', 'split': fact.split, 'split_unit': fact.split_unit,
                         'graph_version': view.graph_version, 'layer_id': fact.layer_id, 'gene_ids': list(fact.gene_ids),
                         'bucket': fact.bucket, 'framing': framing, 'variant': variant, 'wrapper_index': wrapper_index,
                         'request_template_index': variant_index % len(contract['_edge_requests'][fact.split]['directed' if view.layers[fact.layer_id].directed else 'undirected']),
                         'supervision': 'current_turn_assistant_only', 'training_ready': False}}


def validate_row(row: Mapping[str, Any], view: GraphView) -> None:
    """Check source copying, evidence visibility, retries, and task completion."""
    messages = row['messages']
    meta = row['metadata']
    answer = messages[-1]['content']
    if list(answer) != ['status', 'provenance', 'result', 'continuation_decision', 'blocked_reason', 'summary']:
        raise ValueError('invalid S2 final envelope')
    if (answer['continuation_decision'] == 'defer') != bool(answer['blocked_reason']):
        raise ValueError('blocker/continuation mismatch')
    context = json.loads(messages[1]['content'])
    expected_active = {'multiplex_id': view.multiplex_id, 'graph_version': view.graph_version,
                       'layers': [{'layer_id': k, 'directed': view.layers[k].directed} for k in view.layer_order]}
    if context['active_graph'] != expected_active:
        raise ValueError('harness graph metadata does not match the pinned view')
    expected_arguments = {'multiplex_id': view.multiplex_id, 'layer_id': meta['layer_id'], 'gene_ids': meta['gene_ids']}
    available = list(context['observations'])
    calls = []
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if not message.get('thinking'):
            raise ValueError('assistant reasoning was omitted')
        if 'tool_call' not in message:
            continue
        call = message['tool_call']
        if call != {'name': 'get_edge', 'arguments': expected_arguments}:
            raise ValueError('call changed the requested graph query')
        response = messages[index + 1]
        if response['role'] != 'tool' or response['name'] != call['name']:
            raise ValueError('tool response does not match its call')
        if calls:
            previous_call, previous = calls[-1]
            allowance = re.search(r'\bretries_remaining=(\d+)\b', previous.get('message', ''))
            if previous['status'] != 'error' or allowance is None or int(allowance[1]) != 1 or call != previous_call:
                raise ValueError('retry lacks permission or changes arguments')
        calls.append((call, response['content']))
        if response['content']['status'] == 'error' and response['content']['provenance'] != {'registry': [], 'graph': []}:
            raise ValueError('failed call contains graph evidence')
        if response['content']['status'] != 'error':
            available.append({'operation': call['name'], 'arguments': call['arguments'], **response['content']})
    for observation in available:
        if observation['operation'] != 'get_edge' or observation['arguments'] != expected_arguments:
            raise ValueError('saved acquisition inputs do not match the query')
        expected_observation = view.get_edge(**expected_arguments,
            evidence_id=observation['provenance']['graph'][0]['evidence_id'])
        if any(observation[k] != expected_observation[k] for k in ('status', 'provenance', 'result')):
            raise ValueError('visible observation differs from the active source')
    expected_calls = {'lookup': 1, 'context': 0, 'terminal': 1, 'recovered': 2, 'exhausted': 2}
    if len(calls) != expected_calls[meta['variant']]:
        raise ValueError('unexpected call count')
    if answer['status'] == 'error':
        if not calls or calls[-1][1]['status'] != 'error' or meta['variant'] not in ('terminal', 'exhausted'):
            raise ValueError('final error does not follow a terminal required failure')
        if list(answer['result']) != ['failed_tool', 'error_code', 'arguments', 'partial_results', 'coverage']:
            raise ValueError('failure result contains uncomputed outputs')
        if answer['provenance'] != {'registry': [], 'graph': []} or answer['result']['partial_results']:
            raise ValueError('failed calls invented supporting evidence')
        if answer['result']['arguments'] != calls[-1][0]['arguments']:
            raise ValueError('failed arguments were not retained')
        if (answer['continuation_decision'] != 'defer' or
                answer['result']['failed_tool'] != 'get_edge' or
                answer['result']['error_code'] != calls[-1][1]['error_code'] or
                answer['result']['coverage'] is not None):
            raise ValueError('failure outcome does not match the required lookup')
        return
    matching = [o for o in available if o.get('result') == answer['result'] and o.get('provenance') == answer['provenance']]
    if not matching:
        raise ValueError('final answer lacks visible supporting evidence')
    evidence_id = answer['provenance']['graph'][0]['evidence_id']
    expected = view.get_edge(view.multiplex_id, meta['layer_id'], meta['gene_ids'], evidence_id=evidence_id)
    if any(answer[key] != expected[key] for key in ('status', 'provenance', 'result')):
        raise ValueError('answer differs from the pinned graph')
    decision = 'defer' if expected['status'] == 'not_found' else 'continue' if meta['framing'] == 'step' else 'stop'
    if answer['continuation_decision'] != decision:
        raise ValueError('completion decision does not match task scope')


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode()


def _write_immutable(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        existing = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        if existing != set(files) or any((output / name).read_bytes() != data for name, data in files.items()):
            raise ValueError(f'refusing to replace different existing artifacts: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    try:
        for name, data in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_corpus(
    output_dir: Path,
    *,
    counts: Mapping[str, int] | None = None,
    seed: int = 20261028,
    source_manifest: Path | None = None,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Build deterministic rows, separate held-out questions/keys, and audit files."""
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    contract_path = CONTRACT_PATH if contract_path is None else Path(contract_path)
    contract = load_build_contract(contract_path)
    source_manifest = _path(contract['source_manifest']['path']) if source_manifest is None else Path(source_manifest)
    if file_sha256(source_manifest) != contract['source_manifest']['sha256']:
        raise ValueError('S2 source manifest hash mismatch')
    view = load_source_view(source_manifest)
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    facts = select_facts(view, _registry_ids(contract['registry_index']), counts, seed)
    rows = {split: [] for split in SPLITS}
    within_cell: Counter = Counter()
    for fact in facts:
        cell = (fact.split, fact.layer_id, fact.bucket)
        ordinal = within_cell[cell]
        within_cell[cell] += 1
        layer_index = view.layer_order.index(fact.layer_id)
        bucket_index = BUCKETS.index(fact.bucket)
        width = counts[fact.split]
        domain = width * len(view.layer_order)
        for framing in FRAMINGS:
            for contrast_index, variant in enumerate(VARIANTS):
                # Rotate layer blocks across outcome/variant cells. For the
                # default 32-fact cells, every wrapper appears once per cell.
                # The full integer index also spreads a larger future bank;
                # neither this allocation nor bank size selects graph facts.
                block = (layer_index + bucket_index + contrast_index) % len(view.layer_order)
                wording_index = ordinal + width * block + domain * (bucket_index * len(VARIANTS) + contrast_index)
                row = build_row(view, fact, framing, variant, wording_index, bank, contract)
                validate_row(row, view)
                rows[fact.split].append(row)
    units = {split: {r['metadata']['split_unit'] for r in rows[split]} for split in SPLITS}
    if any(units[a] & units[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise ValueError('query contrasts crossed split boundaries')
    audit: dict[str, Any] = {'source_semantics_passed': True, 'split_overlap_count': 0,
                           'fact_selection_sha256': digest([asdict(f) for f in facts]),
                           'reasoning': 'all current-turn assistant messages carry fact-bound reasoning',
                           'prompt_bank': validate_s2_prompt_banks(bank),
                           'full_question_wording': validate_s2_rendered_prompts(bank, contract['_edge_requests']),
                           'tokenizer_audited': False, 'training_ready': False, 'splits': {}}
    files: dict[str, bytes] = {}
    for split in SPLITS:
        files[f'{split}.jsonl'] = _jsonl(rows[split])
        usage = Counter((r['metadata']['framing'], r['metadata']['wrapper_index']) for r in rows[split])
        cells = Counter((r['metadata']['bucket'], r['metadata']['variant'], r['metadata']['framing']) for r in rows[split])
        wrapper_cells: dict[str, Counter] = defaultdict(Counter)
        for row in rows[split]:
            meta = row['metadata']
            wrapper_cells[f'{meta["framing"]}:{meta["wrapper_index"]}'][f'{meta["bucket"]}/{meta["variant"]}'] += 1
        audit['splits'][split] = {'rows': len(rows[split]), 'query_groups': len(units[split]),
                                'wrapper_usage': {f'{f}:{i}': n for (f, i), n in sorted(usage.items())},
                                'wrapper_by_outcome_variant': {k: dict(sorted(v.items())) for k, v in sorted(wrapper_cells.items())},
                                'coverage_cells': {'/'.join(k): v for k, v in sorted(cells.items())}}
        if split != 'train':
            questions, answers = [], []
            for row in rows[split]:
                # Outcome buckets and fault labels belong only in the answer key.
                question_metadata = {k: row['metadata'][k] for k in
                                     ('schema_version', 'tool_contract_version', 'question_family', 'split')}
                questions.append({'row_id': row['row_id'], 'messages': row['messages'][:3],
                                  'tools': row['tools'], 'metadata': question_metadata})
                answers.append({'row_id': row['row_id'], 'expected_messages': row['messages'][3:],
                                'final_answer': row['messages'][-1]['content']})
            files[f'eval/{split}.questions.jsonl'] = _jsonl(questions)
            files[f'eval/{split}.answers.jsonl'] = _jsonl(answers)
    files['audit.json'] = (json.dumps(audit, indent=2, ensure_ascii=False) + '\n').encode()
    implementation = [Path(__file__), ROOT / 'scripts/build_world_model_s2.py',
                      *[Path(__file__).with_name(name) for name in
                        ('s2_sources.py', 's1_prompt_banks.py', 'sources.py', 'source_index.py', 'contract.py',
                         *[name + '.py' for name in BUILDER_MODULES])]]
    manifest = {'schema_version': 'mentor-rl-world-model-s2-manifest-v1',
                'contract_version': contract['contract_version'], 'implemented_families': contract['implemented_families'],
                'training_ready': False, 'seed': seed, 'counts_per_layer_bucket': counts,
                'python_version': platform.python_version(),
                'row_counts': {s: len(rows[s]) for s in SPLITS}, 'graph_version': view.graph_version,
                'source_manifest_sha256': file_sha256(source_manifest), 'source_pins': dict(view.source_pins),
                'contract_sha256': file_sha256(contract_path), 'prompt_bank': contract['prompt_bank'],
                'request_bank': contract['request_bank'],
                'registry_index': contract['registry_index'],
                'wording_audit_contract_sha256': file_sha256(ROOT / 'runtime/world_model/contract.yaml'),
                'implementation_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in implementation},
                'fact_selection_sha256': audit['fact_selection_sha256'],
                'fault_injections': 'terminal, recovered, and exhausted variants are declared backend fault schedules, not observed outages',
                'split_policy': 'unordered endpoint pair across layers/orientations; SHA256(seed,pair) bins 8:1:1; per-stratum quotas',
                'remaining_coverage': contract['remaining_coverage'],
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in sorted(files.items())}}
    manifest['panel_id'] = digest({k: v['sha256'] for k, v in manifest['files'].items() if k.startswith('eval/')})
    files['manifest.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    _write_immutable(Path(output_dir), files)
    return manifest
