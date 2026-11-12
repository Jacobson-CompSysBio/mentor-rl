"""Bind numbered S2.3–S2.6 cases to explicit evidence and source conditions.

These predicates classify already validated traces for the production audit.
They do not generate facts, rewrite a trace, or establish source populations.
The caller separately checks the approved envelope and allocates each row to
at most one case. Native cutoff ties require the caller's pinned-source
``_audit_source`` features because an omitted tied neighbor is not visible in
a bounded response.
"""

from __future__ import annotations

import json


def _m(row):
    return row['metadata']


def _r(row):
    return row['messages'][-1]['content'].get('result') or {}


def _source(row):
    return row.get('_audit_source', {})


def _saved(row, operation=None):
    context = next((message['content'] for message in row['messages'] if message['role'] == 'harness'), '{}')
    observations = json.loads(context).get('observations', [])
    return [item for item in observations if operation is None or item.get('operation') == operation]


def _calls(row):
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def _collection(row):
    return _r(row).get('collection') or {}


def _complete(row):
    return _m(row).get('mode') == 'complete' and _collection(row).get('complete') is True


def _bounded(row):
    return (_m(row).get('mode') == 'top_k' and _collection(row).get('complete') is False
            and (_collection(row).get('returned_count') or 0) > 0)


def _partial_saved(row, operation):
    return any(item.get('result', {}).get('collection', {}).get('complete') is False
               and (item['result']['collection'].get('returned_count') or 0) > 0
               for item in _saved(row, operation) if item.get('result', {}).get('collection'))


def _pages(row, operation):
    return [item for item in _saved(row, operation) if item.get('status') == 'resolved'] + [
        {'arguments': {}, 'result': message['content'].get('result', {})}
        for message in row['messages'] if message['role'] == 'tool' and message.get('name') == operation
        and message['content'].get('status') == 'resolved']


def _retry(row, exhausted=False):
    errors = [message['content'] for message in row['messages'] if message['role'] == 'tool'
              and message['content'].get('status') == 'error']
    calls = _calls(row)
    return (len(errors) == (2 if exhausted else 1)
            and any(calls[index] == calls[index + 1] for index in range(len(calls) - 1))
            and 'retries_remaining=1' in errors[0].get('message', '')
            and (not exhausted or 'retries_remaining=0' in errors[-1].get('message', '')))


def _present3(row):
    return (_m(row).get('category') == 'present' and not _r(row).get('missing_gene_ids')
            and not _r(row).get('not_in_layer_gene_ids'))


def _single3(row, direction):
    return _m(row).get('direction') == direction and _present3(row)


def _loop3(row):
    gene = _m(row).get('gene_id')
    return sum(item.get('gene_id') == gene for item in _r(row).get('neighbors', []) or []) == 1


def _scope_count(row):
    known = _source(row).get('selected_layer_count')
    if known is not None:
        return known
    coverage = _r(row).get('coverage') or {}
    return len(coverage.get('requested_layer_ids', []))


def _all_scope(row):
    scope = _m(row).get('layer_scope') or {}
    return scope.get('kind') == 'all' and scope.get('layer_family') is None


def _explicit_scope(row):
    return (_m(row).get('layer_scope') or {}).get('kind') == 'layers'


def _family_scope(row):
    return (_m(row).get('layer_scope') or {}).get('layer_family') is not None


def _map_regular(row):
    return (_complete(row) and _m(row).get('direction') == 'incident' and _scope_count(row) >= 2
            and _explicit_scope(row) and not _r(row).get('missing_gene_ids'))


def _map_all_present(row):
    return _map_regular(row) and _r(row).get('layers_without_gene') == []


def _native_saved(row, direction=None, positive=False):
    meta = _m(row)
    scope = meta.get('layer_scope') or {}
    allowed = ([meta['layer_id']] if meta.get('layer_id') else
               scope.get('layer_ids') if scope.get('kind') == 'layers' else None)
    return any(item['arguments'].get('gene_id') == meta.get('gene_id')
               and (allowed is None or item['arguments'].get('layer_id') in allowed)
               and (direction is None or item['arguments'].get('direction') == direction)
               and item.get('result', {}).get('collection', {}).get('complete') is True
               and (not positive or bool(item['result'].get('neighbors')))
               for item in _saved(row, 'get_neighbors') if item.get('result', {}).get('collection'))


def _support_representations(row):
    return {support['directed'] for neighbor in _r(row).get('neighbors', []) or []
            for support in neighbor.get('supporting_layers', [])}


def _reciprocal_map(row):
    for neighbor in _r(row).get('neighbors', []) or []:
        for support in neighbor.get('supporting_layers', []):
            pairs = {tuple(edge['gene_ids']) for edge in support.get('edges', [])}
            if support.get('directed') and any(first != second and (second, first) in pairs for first, second in pairs):
                return True
    return False


def _loop_map(row):
    gene = _m(row).get('gene_id')
    return any(neighbor.get('gene_id') == gene and any(
        support.get('directed') and len(support.get('edges', [])) == 1
        and support['edges'][0].get('gene_ids') == [gene, gene]
        for support in neighbor.get('supporting_layers', []))
        for neighbor in _r(row).get('neighbors', []) or [])


def _layer_positive(row):
    return _r(row).get('present') is True and (_m(row).get('layer_count') or 0) > 0


def _membership_partial(row):
    return any(item.get('result', {}).get('collection', {}).get('complete') is True
               and 0 < len(item['result'].get('coverage', {}).get('requested_layer_ids', [])) < _scope_count(row)
               and (item['result'].get('layer_count') or 0) > 0
               and (item['arguments'].get('layer_scope') or {}).get('layer_family') is not None
               for item in _saved(row, 'get_gene_layers') if item.get('result', {}).get('collection'))


def _atomic_map_failure(row):
    coverage = _r(row).get('coverage') or {}
    return (coverage.get('operation') == 'neighbor_collections'
            and bool(coverage.get('checked_inputs')) and bool(coverage.get('unchecked_inputs'))
            and all(set(item) == {'layer_id', 'direction'}
                    for item in coverage.get('requested_inputs', [])))


def _counts6(row):
    return tuple(_m(row).get('category_counts') or (0, 0, 0))


def _mixed6(row, global_missing=False):
    first, second, third = _counts6(row)
    return first > 0 and second > 0 and (third > 0 if global_missing else third == 0)


def _partition_context(row, larger=False):
    requested = set(_m(row).get('gene_ids') or [])
    return any((requested < set(item['arguments'].get('gene_ids', [])) if larger else
                bool(set(item['arguments'].get('gene_ids', [])))
                and set(item['arguments'].get('gene_ids', [])) < requested)
               for item in _saved(row, 'get_nodes_by_layer'))


def _partial_coverage6(row):
    coverage = _r(row).get('coverage') or {}
    return bool(coverage.get('checked_gene_ids')) and bool(coverage.get('unchecked_gene_ids'))


def _bounded_membership_omits_layer(row):
    return any(item.get('result', {}).get('collection', {}).get('complete') is False
               and item['result'].get('present') is True
               and _m(row).get('layer_id') not in item['result'].get('layer_ids', [])
               for item in _saved(row, 'get_gene_layers') if item.get('result', {}).get('collection'))


def _global_negatives_reused(row):
    records = _saved(row, 'check_multiplex_membership')
    if len(records) != 1:
        return False
    present = set(records[0].get('result', {}).get('present_gene_ids', []))
    absent = set(records[0].get('result', {}).get('absent_gene_ids', []))
    calls = _calls(row)
    return (bool(present) and bool(absent) and bool(calls)
            and calls[0]['arguments'].get('gene_ids') == [gene for gene in _m(row)['gene_ids'] if gene in present])


_RULES = {}


def _add(family, number, variants, requirements, predicate, framing='direct', diagnostic=False):
    case_id = f'S2.{family}.{number}'
    _RULES[case_id] = (tuple(variants.split()), predicate, framing)
    method = ('build_direction_diagnostic' if family == 3 else 'build_diagnostic') if diagnostic else 'build_row'
    CASE_BINDINGS[case_id] = {'generator': f'runtime.world_model.s2_{family}_corpus.{method}',
                             'variants': tuple(variants.split()), 'requirements': requirements}


CASE_BINDINGS = {}

_add(3, 1, 'lookup', 'Fresh complete positive undirected list for a present layer member.', lambda r: _single3(r, 'undirected') and _complete(r) and bool(_r(r).get('neighbors')))
_add(3, 2, 'lookup', 'Same complete positive undirected operation as a workflow step.', lambda r: _single3(r, 'undirected') and _complete(r) and bool(_r(r).get('neighbors')), 'step')
_add(3, 3, 'lookup', 'Positive incomplete undirected top-K with a known full total.', lambda r: _single3(r, 'undirected') and _bounded(r) and _collection(r).get('total_count') is not None)
_add(3, 4, 'symbol_context', 'Supplied unique registry identity followed by a complete positive outgoing query.', lambda r: _single3(r, 'out') and _complete(r) and bool(_saved(r, 'lookup_human_gene_symbol')) and bool(_calls(r)) and bool(_r(r).get('neighbors')))
_add(3, 5, 'lookup', 'Known multiplex member outside an undirected layer; complete checked empty list.', lambda r: _m(r).get('category') == 'outside_layer' and _m(r).get('direction') == 'undirected' and _complete(r) and _r(r).get('neighbors') == [], 'step')
_add(3, 6, 'lookup', 'Globally missing subject leaves the undirected neighbor result unavailable.', lambda r: _m(r).get('category') == 'outside_multiplex' and _m(r).get('direction') == 'undirected' and _r(r).get('neighbors') is None)
_add(3, 7, 'bounded_context', 'Incomplete saved undirected list requires acquisition for a complete request.', lambda r: _single3(r, 'undirected') and _complete(r) and _partial_saved(r, 'get_neighbors') and bool(_calls(r)), 'step')
_add(3, 8, 'context complete_context', 'Complete matching outgoing native context supplies the positive list without calls.', lambda r: _single3(r, 'out') and _complete(r) and bool(_r(r).get('neighbors')) and _native_saved(r, 'out', True) and not _calls(r))
_add(3, 9, 'lookup', 'A complete positive incoming list preserves the native source-to-target roles.', lambda r: _single3(r, 'in') and _complete(r) and bool(_r(r).get('neighbors')))
_add(3, 10, 'lookup', 'A present directed node has a complete empty outgoing list.', lambda r: _single3(r, 'out') and _complete(r) and _r(r).get('neighbors') == [])
_add(3, 11, 'wrong_direction', 'Saved incoming context cannot settle the outgoing complete request.', lambda r: _single3(r, 'out') and _complete(r) and _native_saved(r, 'in') and bool(_calls(r)), 'step')
_add(3, 12, 'lookup', 'A compound complete directed request preserves both native directions and checks both.', lambda r: _single3(r, 'both') and _m(r).get('mode') == 'complete' and (_r(r).get('coverage') or {}).get('checked_directions') == ['in', 'out'])
_add(3, 13, 'paged', 'An undirected complete traversal actually accumulates two or more successful pages.', lambda r: _single3(r, 'undirected') and _complete(r) and len(_pages(r, 'get_neighbors')) >= 2, 'step')
_add(3, 14, 'terminal', 'A later undirected complete page fails while preserving a genuine earlier partial page.', lambda r: _m(r).get('direction') == 'undirected' and _m(r).get('mode') == 'complete' and _partial_saved(r, 'get_neighbors') and _r(r).get('arguments', {}).get('cursor') is not None, 'step')
_add(3, 15, 'recovered', 'The identical permitted later-page retry recovers a complete undirected traversal.', lambda r: _single3(r, 'undirected') and _complete(r) and _partial_saved(r, 'get_neighbors') and _retry(r), 'step')
_add(3, 16, 'exhausted', 'The permitted later-page retry is exhausted, retaining the original partial list.', lambda r: _m(r).get('direction') == 'undirected' and _m(r).get('mode') == 'complete' and _partial_saved(r, 'get_neighbors') and _retry(r, True), 'step')
_add(3, 17, 'unknown_total', 'An incomplete positive undirected bounded answer keeps the full total unknown.', lambda r: _single3(r, 'undirected') and _bounded(r) and _collection(r).get('total_count') is None)
_add(3, 18, 'lookup', 'A real equal-weight native tie straddles an incomplete undirected top-K cutoff with known total.', lambda r: _single3(r, 'undirected') and _bounded(r) and _collection(r).get('total_count') is not None and _source(r).get('cutoff_tie') is True)
_add(3, 19, 'lookup', 'A complete directed list includes its source self-loop exactly once.', lambda r: _m(r).get('direction') in ('in', 'out') and _present3(r) and _complete(r) and _loop3(r))
_add(3, 20, 'lookup', 'An incomplete outgoing bound on an illustration-scale native population of at least731 neighbors.', lambda r: _single3(r, 'out') and _bounded(r) and (_source(r).get('native_neighbor_count') or 0) >= 731)
_add(3, 21, 'context complete_context', 'A matching saved incomplete top-K list with known full total answers the same bound without calls.', lambda r: _m(r).get('direction') == 'undirected' and _bounded(r) and _collection(r).get('total_count') is not None and _partial_saved(r, 'get_neighbors') and not _calls(r))
_add(3, 22, 'second_terminal', 'Incoming completes before the outgoing operation fails in a compound request.', lambda r: _m(r).get('direction') == 'both' and (_r(r).get('coverage') or {}).get('checked_directions') == ['in'] and (_r(r).get('coverage') or {}).get('unchecked_directions') == ['out'], 'step')
_add(3, 23, 'direction_invalid', 'Local incoming-on-undirected representation rejection; no acquisition.', lambda r: _r(r).get('issue_code') == 'DIRECTION_UNAVAILABLE' and not _calls(r), diagnostic=True)
_add(3, 24, 'lookup', 'Positive undirected top-K has K strictly above the available count and completes.', lambda r: _single3(r, 'undirected') and _m(r).get('mode') == 'top_k' and _collection(r).get('complete') is True and (_collection(r).get('returned_count') or 0) > 0 and (_m(r).get('top_k') or 0) > (_collection(r).get('total_count') or 0))
_add(3, 25, 'unknown_total', 'A complete undirected traversal accumulates pages whose initial full total is unknown.', lambda r: _single3(r, 'undirected') and _complete(r) and len(_pages(r, 'get_neighbors')) >= 2 and all((p.get('result', {}).get('collection') or {}).get('total_count') is None for p in _pages(r, 'get_neighbors')))

_add(4, 1, 'scope_order', 'Normalize repeated/reversed explicit layers while all selected memberships are positive.', lambda r: _map_all_present(r))
_add(4, 2, 'lookup', 'Complete positive explicit multi-layer incident map as a workflow step.', lambda r: _map_all_present(r) and bool(_r(r).get('neighbors')), 'step')
_add(4, 3, 'lookup', 'Complete explicit multi-layer incident map has both present and absent selected layers.', lambda r: _map_regular(r) and 0 < len(_r(r).get('layers_without_gene') or []) < _scope_count(r))
_add(4, 4, 'symbol_context', 'Unique registry context supplies identity for a complete explicit multi-layer map.', lambda r: _map_regular(r) and bool(_saved(r, 'lookup_human_gene_symbol')) and bool(_calls(r)), 'step')
_add(4, 5, 'lookup', 'Known global member is absent from every selected explicit layer and yields a complete empty map.', lambda r: _map_regular(r) and len(_r(r).get('layers_without_gene') or []) == _scope_count(r) and _r(r).get('neighbors') == [], 'step')
_add(4, 6, 'lookup', 'Globally absent subject leaves the explicit multi-layer map unavailable.', lambda r: _m(r).get('category') == 'outside_multiplex' and _explicit_scope(r) and _scope_count(r) >= 2 and _r(r).get('neighbors') is None)
_add(4, 7, 'native_context', 'A complete positive undirected native list is reused while other selected layers are acquired.', lambda r: _map_regular(r) and _native_saved(r, 'undirected', True) and bool(_calls(r)), 'step')
_add(4, 8, 'context complete_context', 'A matching complete explicit multi-layer map needs no call.', lambda r: _map_regular(r) and bool(_saved(r, 'get_neighbor_layers')) and not _calls(r))
_add(4, 9, 'lookup', 'A complete explicit incident map contains actual undirected and directed supporting records.', lambda r: _map_all_present(r) and _support_representations(r) == {False, True})
_add(4, 10, 'lookup', 'A present subject in multiple directed layers has zero outgoing neighbors throughout.', lambda r: _m(r).get('mode') == 'complete' and _m(r).get('direction') == 'out' and _explicit_scope(r) and _scope_count(r) >= 2 and _r(r).get('layers_without_gene') == [] and _r(r).get('missing_gene_ids') == [] and _r(r).get('neighbors') == [])
_add(4, 11, 'lookup', 'A genuinely incomplete bounded incident row with known total retains both reciprocal arcs in one support.', lambda r: _m(r).get('direction') == 'incident' and _bounded(r) and _collection(r).get('total_count') is not None and _reciprocal_map(r))
_add(4, 12, 'bounded_context', 'An incomplete saved multiplex map requires acquisition for the complete explicit scope.', lambda r: _map_regular(r) and _partial_saved(r, 'get_neighbor_layers') and bool(_calls(r)), 'step')
_add(4, 13, 'context complete_context', 'A saved incomplete multiplex bound with known total answers the same request without calls.', lambda r: _m(r).get('direction') == 'incident' and _explicit_scope(r) and _scope_count(r) >= 2 and _bounded(r) and _collection(r).get('total_count') is not None and _partial_saved(r, 'get_neighbor_layers') and not _calls(r))
_add(4, 14, 'native_terminal', 'Earlier positive undirected layer evidence survives failure with atomic neighbor_collections checked/unchecked coverage.', lambda r: _m(r).get('direction') == 'incident' and _m(r).get('mode') == 'complete' and _native_saved(r, 'undirected', True) and _atomic_map_failure(r) and bool(_r(r).get('partial_results')), 'step')
_add(4, 15, 'native_recovered', 'A permitted missing-layer retry recovers while retaining an earlier positive native layer list.', lambda r: _map_regular(r) and _native_saved(r, 'undirected', True) and _retry(r), 'step')
_add(4, 16, 'native_exhausted', 'The missing-layer retry is exhausted with original positive undirected context and atomic native-input coverage.', lambda r: _m(r).get('direction') == 'incident' and _m(r).get('mode') == 'complete' and _native_saved(r, 'undirected', True) and _atomic_map_failure(r) and _retry(r, True), 'step')
_add(4, 17, 'unknown_total', 'An incomplete positive incident multiplex bound keeps its full total unknown.', lambda r: _m(r).get('direction') == 'incident' and _bounded(r) and _collection(r).get('total_count') is None)
_add(4, 18, 'paged', 'An explicit complete incident map actually accumulates multiple multiplex pages.', lambda r: _map_regular(r) and len(_pages(r, 'get_neighbor_layers')) >= 2)
_add(4, 19, 'native_in_context', 'Incoming native context leaves another orientation required by an incident map.', lambda r: _m(r).get('direction') == 'incident' and _m(r).get('mode') == 'complete' and _native_saved(r, 'in', True) and bool(_calls(r)), 'step')
_add(4, 20, 'native_terminal', 'Complete undirected and incoming directed evidence survive a missing-orientation failure with atomic native-input coverage.', lambda r: _m(r).get('direction') == 'incident' and _m(r).get('mode') == 'complete' and _native_saved(r, 'in', True) and _native_saved(r, 'undirected', True) and _atomic_map_failure(r) and bool(_r(r).get('partial_results')), 'step')
_add(4, 21, 'direction_invalid', 'Incoming roles on a scope containing an undirected representation are rejected locally.', lambda r: _r(r).get('issue_code') == 'DIRECTION_UNAVAILABLE' and not _calls(r), diagnostic=True)
_add(4, 22, 'ranking_ambiguous', 'An explicitly bounded strongest-neighbor request retains its positive K while deferring for a ranking rule.', lambda r: _r(r).get('issue_code') == 'NEIGHBOR_RANKING_AMBIGUOUS' and (_r(r).get('request') or {}).get('mode') == 'top_k' and ((_r(r).get('request') or {}).get('top_k') or 0) > 0 and not _calls(r), diagnostic=True)
_add(4, 23, 'lookup', 'A complete incident all-layer request covers the whole registered view, without a family filter.', lambda r: _all_scope(r) and _m(r).get('direction') == 'incident' and _complete(r) and _scope_count(r) >= 2 and not _r(r).get('missing_gene_ids'))
_add(4, 24, 'stale', 'A complete explicit incident map acquires current evidence rather than reusing a foreign graph version.', lambda r: _map_regular(r) and bool(_calls(r)) and bool(_saved(r)))
_add(4, 25, 'lookup', 'A directed native self-loop is retained once in the sole selected layer.', lambda r: _m(r).get('direction') == 'incident' and _explicit_scope(r) and _scope_count(r) == 1 and _complete(r) and _loop_map(r))

_add(5, 1, 'lookup', 'All-layer complete membership enumerates every containing layer for a global member.', lambda r: _all_scope(r) and _complete(r) and _layer_positive(r))
_add(5, 2, 'lookup', 'The same complete all-layer membership classification supplies a workflow step.', lambda r: _all_scope(r) and _complete(r) and _layer_positive(r), 'step')
_add(5, 3, 'lookup', 'A nonempty requested family contains the globally present subject in at least one layer.', lambda r: _family_scope(r) and _scope_count(r) > 0 and _complete(r) and _layer_positive(r))
_add(5, 4, 'lookup', 'A global member has no membership in a nonempty selected family.', lambda r: _family_scope(r) and _scope_count(r) > 0 and _complete(r) and _r(r).get('present') is True and _r(r).get('layer_ids') == [], 'step')
_add(5, 5, 'symbol_context', 'Supplied unique symbol identity precedes an all-layer membership acquisition.', lambda r: _all_scope(r) and _complete(r) and bool(_saved(r, 'lookup_human_gene_symbol')) and bool(_calls(r)), 'step')
_add(5, 6, 'lookup', 'Checked global absence completes all-layer membership as a standalone negative.', lambda r: _all_scope(r) and _complete(r) and _r(r).get('present') is False)
_add(5, 7, 'family_context', 'A complete positive family-only saved membership record leaves other all-layer memberships to query.', lambda r: _all_scope(r) and _complete(r) and bool(_calls(r)) and _membership_partial(r), 'step')
_add(5, 8, 'context complete_context', 'Complete matching all-layer membership context answers without a call.', lambda r: _all_scope(r) and _complete(r) and bool(_saved(r, 'get_gene_layers')) and not _calls(r))
_add(5, 9, 'projected_context', 'A complete full membership record supplies a nonempty requested family by projection.', lambda r: _family_scope(r) and _scope_count(r) > 0 and _complete(r) and not _calls(r) and any((o['arguments'].get('layer_scope') or {}).get('layer_family') is None and (o['arguments'].get('layer_scope') or {}).get('kind') == 'all' and (o['result'].get('collection') or {}).get('complete') is True for o in _saved(r, 'get_gene_layers')))
_add(5, 10, 'lookup', 'The complete all-layer membership list has exactly one containing layer.', lambda r: _all_scope(r) and _complete(r) and _layer_positive(r) and _m(r).get('layer_count') == 1)
_add(5, 11, 'lookup', 'All-layer first-K membership stops at an incomplete positive prefix with known full total.', lambda r: _all_scope(r) and _bounded(r) and _collection(r).get('total_count') is not None and _layer_positive(r))
_add(5, 12, 'bounded_context', 'A saved incomplete membership prefix cannot answer all containing layers.', lambda r: _all_scope(r) and _complete(r) and _partial_saved(r, 'get_gene_layers') and bool(_calls(r)), 'step')
_add(5, 13, 'context complete_context', 'A saved incomplete membership bound with known full total answers the same bound without calls.', lambda r: _all_scope(r) and _bounded(r) and _collection(r).get('total_count') is not None and _partial_saved(r, 'get_gene_layers') and not _calls(r))
_add(5, 14, 'partial_terminal', 'Earlier complete positive family membership survives failure in required remaining layers.', lambda r: _all_scope(r) and _m(r).get('mode') == 'complete' and _membership_partial(r) and bool(_r(r).get('partial_results')), 'step')
_add(5, 15, 'partial_recovered', 'Remaining-layer acquisition recovers after one permitted identical retry while retaining complete positive family evidence.', lambda r: _all_scope(r) and _complete(r) and _membership_partial(r) and _retry(r), 'step')
_add(5, 16, 'partial_exhausted', 'Remaining-layer retry is exhausted with earlier complete positive family membership preserved.', lambda r: _all_scope(r) and _m(r).get('mode') == 'complete' and _membership_partial(r) and _retry(r, True), 'step')
_add(5, 17, 'unknown_total', 'A genuine incomplete all-layer bounded membership list retains an unknown total.', lambda r: _all_scope(r) and _bounded(r) and _collection(r).get('total_count') is None)
_add(5, 18, 'paged', 'A complete all-layer membership list actually accumulates at least two pages.', lambda r: _all_scope(r) and _complete(r) and len(_pages(r, 'get_gene_layers')) >= 2)
_add(5, 19, 'neighbor_zero', 'A complete zero-outgoing native list proves positive membership in the sole selected directed layer.', lambda r: _scope_count(r) == 1 and _complete(r) and _r(r).get('layer_count') == 1 and not _calls(r) and any(o['arguments'].get('direction') == 'out' and o['result'].get('neighbors') == [] and o['result'].get('not_in_layer_gene_ids') == [] for o in _saved(r, 'get_neighbors')))
_add(5, 20, 'global_context', 'Saved authoritative global absence supplies every selected layer negative without calls.', lambda r: _all_scope(r) and _complete(r) and _r(r).get('present') is False and not _calls(r) and bool(_saved(r, 'check_multiplex_membership')))
_add(5, 21, 'lookup', 'A valid named family has no registered layers, with global membership still checked.', lambda r: _family_scope(r) and _scope_count(r) == 0 and _complete(r) and _m(r).get('category') == 'empty_scope')
_add(5, 22, 'unknown_layer', 'An unregistered explicit layer is a local prerequisite problem, not a membership negative.', lambda r: _r(r).get('issue_code') == 'UNKNOWN_LAYER' and not _calls(r), diagnostic=True)
_add(5, 23, 'stale', 'Foreign-version membership evidence requires fresh complete all-layer acquisition.', lambda r: _all_scope(r) and _complete(r) and bool(_saved(r)) and bool(_calls(r)))
_add(5, 24, 'lookup', 'A completed checked global negative supplies the stated workflow step.', lambda r: _all_scope(r) and _complete(r) and _r(r).get('present') is False, 'step')
_add(5, 25, 'lookup', 'All-layer bounded membership has K strictly above the positive full count and therefore completes.', lambda r: _all_scope(r) and _m(r).get('mode') == 'top_k' and _collection(r).get('complete') is True and _layer_positive(r) and (_m(r).get('top_k') or 0) > (_collection(r).get('total_count') or 0))

_add(6, 1, 'lookup', 'Mixed layer and other-multiplex membership, with no globally missing input.', lambda r: _mixed6(r))
_add(6, 2, 'lookup', 'Every requested ID belongs to the layer, as a workflow step.', lambda r: _counts6(r)[0] > 0 and _counts6(r)[1:] == (0, 0), 'step')
_add(6, 3, 'lookup', 'Every known multiplex input is outside the selected layer.', lambda r: _counts6(r)[0] == 0 and _counts6(r)[1] > 0 and _counts6(r)[2] == 0)
_add(6, 4, 'lookup', 'All three membership categories occur in a completed workflow partition.', lambda r: _mixed6(r, True), 'step')
_add(6, 5, 'symbol_context', 'Supplied unique registry mappings precede mixed layer membership classification.', lambda r: _mixed6(r) and bool(_saved(r, 'lookup_human_gene_symbol')) and bool(_calls(r)), 'step')
_add(6, 6, 'lookup', 'One requested gene is positively classified as a layer member.', lambda r: _counts6(r) == (1, 0, 0))
_add(6, 7, 'lookup', 'One globally present gene is outside the layer and completes a workflow step.', lambda r: _counts6(r) == (0, 1, 0), 'step')
_add(6, 8, 'global_context', 'Global-positive context for mixed layer members cannot settle layer membership.', lambda r: _mixed6(r) and bool(_saved(r, 'check_multiplex_membership')) and bool(_calls(r)), 'step')
_add(6, 9, 'larger_context', 'Restrict an authoritative larger complete classification to all three requested categories without calls.', lambda r: _mixed6(r, True) and _partition_context(r, True) and not _calls(r))
_add(6, 10, 'global_context', 'A saved singleton global negative proves layer absence without a new call.', lambda r: _counts6(r) == (0, 0, 1) and bool(_saved(r, 'check_multiplex_membership')) and not _calls(r))
_add(6, 11, 'partial_context', 'Reuse a proper complete input subset and query only remaining inputs in a three-category workflow.', lambda r: _mixed6(r, True) and _partition_context(r) and bool(_calls(r)), 'step')
_add(6, 12, 'partial_terminal', 'Required remaining inputs fail while checked subset classification survives.', lambda r: _mixed6(r, True) and _partition_context(r) and _partial_coverage6(r), 'step')
_add(6, 13, 'partial_recovered', 'An identical permitted retry completes the remaining three-category classification.', lambda r: _mixed6(r, True) and _partition_context(r) and _retry(r), 'step')
_add(6, 14, 'partial_exhausted', 'A permitted remaining-input retry is exhausted with checked and unchecked inputs distinct.', lambda r: _mixed6(r, True) and _partition_context(r) and _partial_coverage6(r) and _retry(r, True), 'step')
_add(6, 15, 'duplicates', 'Repeated literal inputs deduplicate in first-mention order before complete classification.', lambda r: len(_m(r).get('input_genes') or []) > len(_m(r).get('gene_ids') or []) and list(dict.fromkeys(_m(r)['input_genes'])) == _m(r)['gene_ids'])
_add(6, 16, 'neighbor_zero', 'A singleton zero-outgoing native result still proves positive layer membership without calls.', lambda r: _counts6(r) == (1, 0, 0) and _native_saved(r, 'out') and not _calls(r) and any(o['result'].get('neighbors') == [] and o['result'].get('not_in_layer_gene_ids') == [] for o in _saved(r, 'get_neighbors')))
_add(6, 17, 'stale', 'A foreign graph version cannot settle the current three-category classification.', lambda r: _mixed6(r, True) and bool(_saved(r, 'get_nodes_by_layer')) and bool(_calls(r)))
_add(6, 18, 'unknown_layer', 'An unregistered layer is a local scope diagnostic with no graph classification.', lambda r: _r(r).get('issue_code') == 'UNKNOWN_LAYER' and not _calls(r), diagnostic=True)
_add(6, 19, 'lookup', 'An explicitly empty finite input has an empty complete classification without calls.', lambda r: _m(r).get('input_size') == 0 and _m(r).get('input_genes') == [] and not _calls(r))
_add(6, 20, 'membership_context', 'A saved complete containing-layer list proves a singleton layer negative without calls.', lambda r: _counts6(r) == (0, 1, 0) and not _calls(r) and any((o['result'].get('collection') or {}).get('complete') is True and o['result'].get('present') is True and _m(r).get('layer_id') not in o['result'].get('layer_ids', []) for o in _saved(r, 'get_gene_layers')))
_add(6, 21, 'bounded_membership_context', 'Omission from an incomplete containing-layer prefix needs a fresh positive singleton membership lookup.', lambda r: _counts6(r) == (1, 0, 0) and _bounded_membership_omits_layer(r) and bool(_calls(r)))
_add(6, 22, 'global_context', 'Reuse global negatives and acquire only global-positive inputs for a three-category workflow.', lambda r: _mixed6(r, True) and _global_negatives_reused(r), 'step')
_add(6, 23, 'terminal', 'A cold atomic failure leaves every requested input unchecked and no partial evidence.', lambda r: bool(_m(r).get('gene_ids')) and not _saved(r) and not (_r(r).get('coverage') or {}).get('checked_gene_ids') and bool((_r(r).get('coverage') or {}).get('unchecked_gene_ids')))
_add(6, 24, 'missing_genes', 'A missing input list is preserved as null, distinct from an explicit empty list.', lambda r: _r(r).get('issue_code') == 'MISSING_GENES' and (_r(r).get('request') or {}).get('genes') is None and not _calls(r), diagnostic=True)
_add(6, 25, 'lookup', 'The mixed layer/other-multiplex partition with no global absence supplies a workflow step.', lambda r: _mixed6(r), 'step')

for _number in (14, 16, 20):
    CASE_BINDINGS[f'S2.4.{_number}']['implementation_gap'] = (
        'The builder emits layer_neighbor_maps failure coverage instead of the reviewed '
        'neighbor_collections partition over atomic layer/direction inputs.'
        + (' Its native fault variant also saves only one observation, not both the complete '
           'undirected layer and incoming directed evidence.' if _number == 20 else ''))
CASE_BINDINGS['S2.4.22']['implementation_gap'] = (
    'The ranking diagnostic constructor drops the requested bounded mode and positive K, '
    'emitting request.mode=null and request.top_k=null.')
for _number in (14, 15, 16):
    CASE_BINDINGS[f'S2.5.{_number}']['implementation_gap'] = (
        'Family-only saved membership context cannot currently be combined with the required '
        'terminal/recovered/exhausted remaining-scope fault variants; they save an explicit layer-ID subset.')


def matches(case_id, row):
    """Check the explicitly bound operation, evidence, source, and task framing."""
    variants, predicate, framing = _RULES[case_id]
    meta = _m(row)
    return (meta.get('question_family') == case_id.rsplit('.', 1)[0]
            and meta.get('variant') in variants and meta.get('framing') == framing
            and bool(predicate(row)))
