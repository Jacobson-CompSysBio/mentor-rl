"""Bind reviewed S2.7–S2.10 cases to explicit generator and evidence conditions.

These predicates classify source-validated candidate rows. They do not replace
the family validators or silently allocate one row to several production cases.
The audit caller checks approved outcomes/framing and resolves overlapping
candidate ownership before claiming quotas. Missing witnesses remain gaps.
"""

from __future__ import annotations

from fractions import Fraction
import json

from .s2_components import components


CASE_BINDINGS = {}
_PREDICATES = {}


def _bind(family, number, variants, requirements, predicate, *, generator=None):
    case = f'S2.{family}.{number}'
    CASE_BINDINGS[case] = {
        'generator': generator or f'runtime.world_model.s2_{family}_corpus.build_row',
        'variants': tuple(variants.split()), 'requirements': requirements,
    }
    _PREDICATES[case] = predicate


def _meta(row):
    return row['metadata']


def _result(row):
    return row['messages'][-1]['content']['result']


def _saved(row, operation=None):
    observations = json.loads(row['messages'][1]['content'])['observations']
    return [item for item in observations if operation is None or item['operation'] == operation]


def _calls(row, operation=None):
    calls = [item['tool_call'] for item in row['messages'] if 'tool_call' in item]
    return [item for item in calls if operation is None or item['name'] == operation]


def _responses(row, operation=None, success=True):
    return [item['content'] for item in row['messages'] if item['role'] == 'tool'
            and (operation is None or item['name'] == operation)
            and (not success or item['content']['status'] != 'error')]


def _source(row, name, default=None):
    source = row.get('_audit_source', _meta(row).get('_audit_source', {}))
    return source.get(name, default)


def _directed(row):
    if 'directed' in _result(row):
        return _result(row)['directed']
    return _source(row, 'directed')


def _complete(observation):
    return (observation.get('result', {}).get('collection') or {}).get('complete') is True


def _incomplete(observation):
    return (observation.get('result', {}).get('collection') or {}).get('complete') is False


def _no_calls(row):
    return not _calls(row)


def _empty_context(row):
    return not _saved(row)


def _native_saved(row):
    return _saved(row, 'get_neighbors')


def _saved_positives(row):
    return {item['gene_id'] for observation in _native_saved(row)
            for item in observation['result'].get('neighbors') or []}


def _paged_acquisitions(row, operation):
    return any(call['arguments'].get('cursor') is not None for call in _calls(row, operation))


def _error_count(row):
    return sum(message['role'] == 'tool' and message['content']['status'] == 'error'
               for message in row['messages'])


def _category_counts(row):
    return _meta(row).get('category_counts') or (0, 0, 0)


def _members(row):
    return _category_counts(row)[0]


def _excluded(row):
    return sum(_category_counts(row)[1:])


def _edge_count(row):
    return _meta(row).get('edge_count') or 0


def _partial_native(row):
    return any(_incomplete(item) for item in _native_saved(row))


def _partial_graph(row):
    return any(_incomplete(item) for item in _saved(row, 'induce_subgraph'))


def _directed_chain(row):
    if _members(row) != 3 or _excluded(row) or _directed(row) is not True or _edge_count(row) != 2:
        return False
    full = [response['result'] for response in _responses(row, 'induce_subgraph')
            if (response.get('result', {}).get('collection') or {}).get('complete') is True]
    return any(len(components(graph['layer_gene_ids'], graph['edges'], 'weak')) == 1
               and len(components(graph['layer_gene_ids'], graph['edges'], 'strong')) == 3 for graph in full)


def _saved_connected_prefix(row, kind):
    return any(_incomplete(observation) and len(observation['result'].get('layer_gene_ids', [])) >= 2
               and len(components(observation['result']['layer_gene_ids'], observation['result']['edges'], kind)) == 1
               for observation in _saved(row, 'induce_subgraph'))


def _saved_weak_not_strong(row):
    return any(_incomplete(observation) and len(observation['result'].get('layer_gene_ids', [])) >= 2
               and len(components(observation['result']['layer_gene_ids'], observation['result']['edges'], 'weak')) == 1
               and len(components(observation['result']['layer_gene_ids'], observation['result']['edges'], 'strong')) > 1
               for observation in _saved(row, 'induce_subgraph'))


def _disjoint_batches_miss_edge(row):
    saved = _saved(row, 'induce_subgraph')
    groups = [set(observation['arguments']['gene_ids']) for observation in saved]
    if (len(groups) < 2 or not all(_complete(observation) for observation in saved)
            or any(first & second for position, first in enumerate(groups) for second in groups[position + 1:])
            or set().union(*groups) != set(_meta(row)['gene_ids'])):
        return False
    return any(not any(set(edge['gene_ids']) <= group for group in groups)
               for edge in _result(row).get('edges', []))


def _saved_superset_bridge(row):
    selected = set(_meta(row)['gene_ids'])
    kind = _meta(row)['connectivity_kind']
    return any(_complete(observation) and set(observation['arguments']['gene_ids']) > selected
               and len(components(observation['result']['layer_gene_ids'], observation['result']['edges'], kind)) == 1
               for observation in _saved(row, 'induce_subgraph'))


def _scope_size(row):
    return _source(row, 'selected_layer_count', len(_result(row).get('resolved_layer_ids', [])))


def _rank(row):
    support = _meta(row).get('rank_support') or _result(row).get('rank_support')
    if not support or support['population_count'] < 1:
        return None
    n = support['population_count']
    return Fraction(50) if n == 1 else Fraction(
        100 * (2 * support['lower_degree_count'] + support['equal_degree_count'] - 1), 2 * (n - 1))


def _hub(row):
    return _meta(row).get('request_kind') == 'degree_and_hub'


def _degree_only(row):
    return _meta(row).get('request_kind') == 'degree_only'


def _threshold(row):
    policy = _meta(row).get('hub_policy') or _result(row).get('hub_policy') or {}
    return policy.get('threshold_percentile')


def _issue(row, code):
    return _result(row).get('issue_code') == code


def _shared_positive(row):
    return (_result(row).get('shared_neighbor_count') or 0) > 0


def _shared_empty(row):
    return _result(row).get('shared_neighbor_count') == 0


def _directions(row):
    return tuple(_meta(row).get('directions', ()))


def _roles(row):
    return _meta(row).get('queries', ())


def _distinct_genes(row):
    return len(_roles(row)) == 2 and _roles(row)[0]['gene_id'] != _roles(row)[1]['gene_id']


def _bounded_lower_bound(row):
    saved = _native_saved(row)
    if len(saved) < 2 or not all(_incomplete(item) for item in saved):
        return False
    sets = [{item['gene_id'] for item in observation['result'].get('neighbors') or []}
            for observation in saved]
    common = set.intersection(*sets)
    return bool(common) and len(common) < (_result(row).get('shared_neighbor_count') or 0) and bool(_calls(row))


def _bounded_omission(row):
    shared = set(_result(row).get('shared_neighbors') or [])
    return bool(shared) and any(_incomplete(item) and not shared <= {
        value['gene_id'] for value in item['result'].get('neighbors') or []}
        for item in _native_saved(row)) and bool(_calls(row))


def _containment(row):
    saved = _native_saved(row)
    complete = [{item['gene_id'] for item in o['result'].get('neighbors') or []}
                for o in saved if _complete(o)]
    bounded = [{item['gene_id'] for item in o['result'].get('neighbors') or []}
               for o in saved if _incomplete(o)]
    shared = set(_result(row).get('shared_neighbors') or [])
    return _no_calls(row) and bool(shared) and any(values == shared and values <= prefix
        for values in complete for prefix in bounded)


def _incompatible_degree(row):
    meta = _meta(row)
    for observation in _saved(row, 'get_degree'):
        source = observation['arguments']
        graph = observation['provenance']['graph']
        if (source['layer_scope'] != meta['layer_scope'] and source['direction'] != meta['direction']
                and graph and graph[0]['graph_version'] != meta['graph_version']):
            return bool(_calls(row, 'get_degree'))
    return False


def _native_union(row):
    saved = _native_saved(row)
    if len(saved) < 2 or not all(_complete(item) for item in saved):
        return False
    sets = [{item['gene_id'] for item in observation['result'].get('neighbors') or []}
            for observation in saved]
    union = set().union(*sets)
    return len(union) < sum(map(len, sets)) and len(union) == _result(row).get('degree')


# S2.7: identical final intersections can arise from different prerequisites
# and evidence states, so saved/native/failure conditions are explicit.
_bind(7, 1, 'lookup swapped', 'Two distinct undirected layer members, nonadjacent with a positive shared set; no saved context.',
      lambda r: _directions(r) == ('undirected', 'undirected') and _distinct_genes(r) and _source(r, 'input_pair_adjacent') is False
      and _shared_positive(r) and _empty_context(r))
_bind(7, 2, 'lookup swapped', 'Two distinct adjacent undirected layer members with a positive shared set; fresh acquisition.',
      lambda r: _directions(r) == ('undirected', 'undirected') and _distinct_genes(r) and _source(r, 'input_pair_adjacent') is True
      and _shared_positive(r) and _empty_context(r))
_bind(7, 3, 'lookup swapped', 'Two present undirected roles have complete nonempty lists and an empty intersection.',
      lambda r: _directions(r) == ('undirected', 'undirected') and _shared_empty(r)
      and all((count or 0) > 0 for count in _meta(r)['neighbor_counts']) and _empty_context(r))
_bind(7, 4, 'symbol_context', 'Two supplied unique registry mappings are retained before positive shared-set acquisition.',
      lambda r: len(_saved(r, 'lookup_human_gene_symbol')) == 2 and _shared_positive(r))
_bind(7, 5, 'lookup', 'A known multiplex input outside the layer yields a resolved empty intersection.',
      lambda r: _shared_empty(r) and any(item['present_in_multiplex'] is True and item['present_in_layer'] is False
                                       for item in _result(r)['input_memberships']))
_bind(7, 6, 'lookup', 'Fresh acquisition finds an input outside the multiplex and leaves the shared set unavailable.',
      lambda r: _issue(r, 'NO_MULTIPLEX_MEMBERSHIP') and _empty_context(r))
_bind(7, 7, 'first_context second_context', 'One saved complete positive-role list replaces one required call.',
      lambda r: len(_native_saved(r)) == 1 and _complete(_native_saved(r)[0])
      and _shared_positive(r) and len(_calls(r, 'get_neighbors')) == 1)
_bind(7, 8, 'context duplicate_context', 'Both complete saved role sets supply a positive exact intersection with no call.',
      lambda r: _shared_positive(r) and _distinct_genes(r) and _no_calls(r) and len(_native_saved(r)) >= 2
      and all(_complete(item) for item in _native_saved(r)))
_bind(7, 9, 'second_terminal', 'The second required native lookup fails once; a complete earlier role remains attributed.',
      lambda r: _error_count(r) == 1 and bool(_native_saved(r)) and _result(r).get('failed_tool') == 'get_neighbors')
_bind(7, 10, 'second_recovered', 'The missing role recovers after exactly one permitted identical retry.',
      lambda r: _error_count(r) == 1 and len(_calls(r, 'get_neighbors')) == 2 and bool(_native_saved(r)))
_bind(7, 11, 'second_exhausted', 'The missing role exhausts its single retry while retaining the earlier complete role.',
      lambda r: _error_count(r) == 2 and len(_calls(r, 'get_neighbors')) == 2 and bool(_native_saved(r)))
_bind(7, 12, 'bounded_context', 'An actual bounded omission becomes a shared entry after required acquisition.', _bounded_omission)
_bind(7, 13, 'paged overlap empty_page unknown_total prefix_context', 'A positive exact intersection requires following a real native continuation cursor.',
      lambda r: _shared_positive(r) and _paged_acquisitions(r, 'get_neighbors'))
_bind(7, 14, 'page_terminal', 'A later native page fails with an incomplete earlier page retained.',
      lambda r: _result(r).get('arguments', {}).get('cursor') is not None and _partial_native(r))
_bind(7, 15, 'lookup', 'Two distinct outgoing roles yield a positive common-target intersection.',
      lambda r: _directions(r) == ('out', 'out') and _roles(r)[0]['gene_id'] != _roles(r)[1]['gene_id']
      and _shared_positive(r) and _empty_context(r))
_bind(7, 16, 'lookup', 'Two distinct incoming roles yield positive common regulators without endpoint self-loop membership.',
      lambda r: _directions(r) == ('in', 'in') and _roles(r)[0]['gene_id'] != _roles(r)[1]['gene_id']
      and _shared_positive(r) and not ({q['gene_id'] for q in _roles(r)} & set(_result(r)['shared_neighbors']))
      and _empty_context(r))
_bind(7, 17, 'lookup', 'Distinct out/in roles yield a positive intermediate set.',
      lambda r: _directions(r) == ('out', 'in') and _roles(r)[0]['gene_id'] != _roles(r)[1]['gene_id']
      and _shared_positive(r) and _empty_context(r))
_bind(7, 18, 'lookup', 'Both directed roles use incoming/outgoing unions and a positive exact intersection.',
      lambda r: _directions(r) == ('both', 'both') and _distinct_genes(r) and _shared_positive(r) and _empty_context(r))
_bind(7, 19, 'empty_context', 'A saved complete empty role and both positive global prerequisites settle the empty result without a call.',
      lambda r: _shared_empty(r) and _no_calls(r) and any(_complete(o) and o['result'].get('neighbors') == [] for o in _native_saved(r))
      and all(item['present_in_multiplex'] is True for item in _result(r)['input_memberships']))
_bind(7, 20, 'wrong_direction', 'A saved opposite native direction cannot fill the requested directed role; acquire required evidence.',
      lambda r: bool(_native_saved(r)) and bool(_calls(r, 'get_neighbors')))
_bind(7, 21, 'stale', 'A saved previous graph version cannot settle the active shared-neighbor request.',
      lambda r: bool(_calls(r, 'get_neighbors')) and any(g['graph_version'] != _meta(r)['graph_version']
          for o in _saved(r) for g in o['provenance']['graph']))
_bind(7, 22, 'unknown_layer', 'Reject an unregistered layer rather than emitting a measured intersection.',
      lambda r: _issue(r, 'UNKNOWN_LAYER'), generator='runtime.world_model.s2_7_corpus.build_diagnostic')
_bind(7, 23, 'lookup', 'Identical query roles reuse exactly one native acquisition while retaining two roles.',
      lambda r: len(_roles(r)) == 2 and _roles(r)[0] == _roles(r)[1]
      and len(_calls(r, 'get_neighbors')) == 1 and _empty_context(r))
_bind(7, 24, 'lookup', 'A source self-loop makes one endpoint a member of the positive incoming intersection.',
      lambda r: _directions(r) == ('in', 'in') and _distinct_genes(r) and _shared_positive(r)
      and bool({q['gene_id'] for q in _roles(r)} & set(_result(r)['shared_neighbors'])) and _empty_context(r))
_bind(7, 25, 'lookup swapped', 'Two complete nonempty undirected roles have an exact empty intersection usable as a workflow step.',
      lambda r: _directions(r) == ('undirected', 'undirected') and _shared_empty(r)
      and all((count or 0) > 0 for count in _meta(r)['neighbor_counts']) and _empty_context(r))
_bind(7, 26, 'empty_context', 'An empty saved opposite role cannot override another input absent from the multiplex.',
      lambda r: _issue(r, 'NO_MULTIPLEX_MEMBERSHIP') and _no_calls(r)
      and any(_complete(o) and o['result'].get('neighbors') == [] for o in _native_saved(r)))
_bind(7, 27, 'bounded_context', 'Two incomplete prefixes share a visible lower bound smaller than the exact intersection.', _bounded_lower_bound)
_bind(7, 28, 'containment_context', 'A complete nonempty role is contained in an incomplete other prefix, proving the exact result without a call.', _containment)
_bind(7, 29, 'lookup', 'The same gene in distinct out/in roles requires separate native sets.',
      lambda r: _directions(r) == ('out', 'in') and _roles(r)[0]['gene_id'] == _roles(r)[1]['gene_id']
      and len(_calls(r, 'get_neighbors')) == 2 and _empty_context(r))

# S2.8: rank and threshold distinctions use exact rational midranks, never the
# rounded display; population/source enrichment is supplied by the audit.
_bind(8, 1, 'lookup', 'A fresh single-layer hub evaluation resolves true at the default95 threshold.',
      lambda r: _hub(r) and _scope_size(r) == 1 and _threshold(r) == 95 and _result(r).get('hub_like') is True and _empty_context(r))
_bind(8, 2, 'lookup', 'A fresh single-layer default-threshold hub evaluation resolves false.',
      lambda r: _hub(r) and _scope_size(r) == 1 and _threshold(r) == 95 and _result(r).get('hub_like') is False and _empty_context(r))
_bind(8, 3, 'threshold_zero threshold_fifty threshold_hundred lookup', 'Exact attainable midrank equals the explicit threshold and is hub-like by inclusive comparison.',
      lambda r: _hub(r) and _rank(r) is not None and _rank(r) == _threshold(r) and _result(r).get('hub_like') is True)
_bind(8, 4, 'lookup', 'Single-layer exact midrank is the nearest attainable same-query-split rank below95.',
      lambda r: _hub(r) and _scope_size(r) == 1 and _threshold(r) == 95
      and _source(r, 'nearest_below_95') is True and _result(r).get('hub_like') is False and _empty_context(r))
_bind(8, 5, 'lookup', 'An all-layer incident union is counted distinctly and ranked against its full union population.',
      lambda r: _hub(r) and _meta(r)['layer_scope'] == {'kind': 'all', 'layer_ids': [], 'layer_family': None}
      and _meta(r)['direction'] == 'incident' and _scope_size(r) > 1)
_bind(8, 6, 'symbol_context', 'A supplied unique symbol mapping supports a single-layer degree and hub request.',
      lambda r: _hub(r) and _scope_size(r) == 1 and len(_saved(r, 'lookup_human_gene_symbol')) == 1)
_bind(8, 7, 'lookup', 'Global membership with selected-layer absence keeps degree zero but defers hub rank.',
      lambda r: _hub(r) and _scope_size(r) == 1 and _issue(r, 'OUTSIDE_REFERENCE_POPULATION'))
_bind(8, 8, 'lookup', 'Global absence leaves the requested degree and hub prerequisites unavailable.',
      lambda r: _hub(r) and _issue(r, 'NO_MULTIPLEX_MEMBERSHIP'))
_bind(8, 9, 'native_context', 'Complete native neighbors establish degree but a requested population rank still requires acquisition.',
      lambda r: _hub(r) and bool(_native_saved(r)) and all(_complete(o) for o in _native_saved(r))
      and bool(_calls(r, 'get_degree')))
_bind(8, 10, 'context duplicate_context', 'A compatible saved exact degree and rank satisfy the hub request without calls.',
      lambda r: _hub(r) and _no_calls(r) and bool(_saved(r, 'get_degree')) and _rank(r) is not None)
_bind(8, 11, 'native_context', 'Complete single-layer native context answers a degree-only request without a new call.',
      lambda r: _degree_only(r) and _scope_size(r) == 1 and _no_calls(r) and bool(_native_saved(r))
      and all(_complete(o) for o in _native_saved(r)))
_bind(8, 12, 'bounded_total', 'An actually incomplete bounded native list has an authoritative total that answers degree alone.',
      lambda r: _degree_only(r) and _no_calls(r) and any(_incomplete(o)
          and o['result']['collection']['total_count'] is not None for o in _native_saved(r)))
_bind(8, 13, 'unknown_total', 'An incomplete native list with unknown total requires exact degree acquisition.',
      lambda r: _degree_only(r) and bool(_calls(r, 'get_degree')) and any(_incomplete(o)
          and o['result']['collection']['total_count'] is None for o in _native_saved(r)))
_bind(8, 14, 'lookup', 'A present directed-layer member has zero outgoing degree and a valid rank.',
      lambda r: _hub(r) and _scope_size(r) == 1 and _meta(r)['direction'] == 'out'
      and _result(r).get('degree') == 0 and _result(r).get('membership', {}).get('present_in_scope') is True)
_bind(8, 15, 'lookup native_context', 'A selected native self-loop contributes one distinct neighbor to degree-only counting.',
      lambda r: _degree_only(r) and _source(r, 'native_self_loop') is True and (_result(r).get('degree') or 0) >= 1)
_bind(8, 16, 'native_context', 'Complete native lists from multiple layers overlap and are deduplicated into an exact degree-only union.',
      lambda r: _degree_only(r) and _scope_size(r) > 1 and _no_calls(r) and _native_union(r))
_bind(8, 17, 'lookup', 'A nonempty exhaustive family scope supplies degree and hub evaluation.',
      lambda r: _hub(r) and _meta(r)['layer_scope']['kind'] == 'all'
      and _meta(r)['layer_scope']['layer_family'] is not None and _scope_size(r) > 0)
_bind(8, 18, 'lookup', 'A multiplex member outside a nonempty selected layer has a resolved degree-only zero.',
      lambda r: _degree_only(r) and _scope_size(r) == 1 and _result(r).get('degree') == 0
      and _result(r).get('membership', {}).get('present_in_scope') is False)
_bind(8, 19, 'lookup', 'A valid empty family has no rank reference population and defers requested hub evaluation.',
      lambda r: _hub(r) and _issue(r, 'EMPTY_REFERENCE_POPULATION'))
_bind(8, 20, 'lookup', 'A valid empty family still resolves the degree-only zero.',
      lambda r: _degree_only(r) and _scope_size(r) == 0 and _meta(r)['layer_scope']['layer_family'] is not None
      and _result(r).get('degree') == 0)
_bind(8, 21, 'histogram_context', 'Compatible exact degree and complete reference histogram establish rank without calls.',
      lambda r: _hub(r) and _no_calls(r) and any(o['result'].get('complete') is True
                                               for o in _saved(r, 'degree_distribution')))
_bind(8, 22, 'stale wrong_scope wrong_direction', 'One saved degree observation has wrong scope AND direction AND version; acquire the active statistic.', _incompatible_degree)
_bind(8, 23, 'new_threshold', 'A changed threshold reuses compatible exact degree/rank evidence without a graph call.',
      lambda r: _hub(r) and _no_calls(r) and bool(_saved(r, 'get_degree')) and _rank(r) is not None)
_bind(8, 24, 'terminal', 'A cold required degree/rank acquisition fails with no retry allowance.',
      lambda r: _hub(r) and _empty_context(r) and _error_count(r) == 1)
_bind(8, 25, 'recovered', 'A cold required degree/rank acquisition recovers after one authorized identical retry.',
      lambda r: _hub(r) and _empty_context(r) and _error_count(r) == 1 and len(_calls(r, 'get_degree')) == 2)
_bind(8, 26, 'rank_exhausted', 'Exhausted rank acquisition retains a complete native-neighbor observation proving exact degree.',
      lambda r: _hub(r) and _error_count(r) == 2 and bool(_native_saved(r))
      and all(_complete(o) for o in _native_saved(r)))
_bind(8, 27, 'invalid_direction', 'Reject unavailable native direction locally without a call.',
      lambda r: _issue(r, 'DIRECTION_UNAVAILABLE') and _no_calls(r), generator='runtime.world_model.s2_8_corpus.build_diagnostic')
_bind(8, 28, 'tool_unknown_layer', 'Unknown layer is checked by a resolver call and preserves atomic rejected arguments.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _result(r).get('rejected_arguments') is not None
      and len(_calls(r, 'get_degree')) == 1, generator='runtime.world_model.s2_8_corpus.build_diagnostic')
_bind(8, 29, 'lookup', 'Exact rank is below95 although its two-decimal display rounds to95.00; classify from exact support.',
      lambda r: _hub(r) and _threshold(r) == 95 and _rank(r) is not None
      and Fraction(18999, 200) < _rank(r) < 95 and _result(r).get('hub_like') is False)
_bind(8, 30, 'lookup', 'Sparse outgoing population gives degree one a hub-like exact rank.',
      lambda r: _hub(r) and _meta(r)['direction'] == 'out' and _result(r).get('degree') == 1
      and _result(r).get('hub_like') is True)
_bind(8, 31, 'union_counts', 'Separate in/out degree totals cannot settle an incident union; acquire exact degree.',
      lambda r: _degree_only(r) and _meta(r)['direction'] == 'incident' and _scope_size(r) == 1
      and {o['arguments']['direction'] for o in _saved(r, 'get_degree')} >= {'in', 'out'}
      and bool(_calls(r, 'get_degree')))

# S2.9: match the selected-set projection rather than full-layer isolation,
# and distinguish positive partial evidence from a complete acquisition.
_bind(9, 1, 'lookup reordered', 'Nonempty projection includes both layer exclusions and global exclusions.',
      lambda r: all(count > 0 for count in _category_counts(r)) and _empty_context(r))
_bind(9, 2, 'lookup reordered', 'All requested inputs are layer members with both an edge and a selected-set isolate.',
      lambda r: _members(r) >= 3 and _excluded(r) == 0 and _edge_count(r) > 0
      and (_meta(r).get('isolated_count') or 0) > 0 and _empty_context(r))
_bind(9, 3, 'lookup reordered', 'At least two selected layer members have no induced edges.',
      lambda r: _members(r) >= 2 and _edge_count(r) == 0 and _excluded(r) == 0 and _empty_context(r))
_bind(9, 4, 'symbol_context', 'Supplied unique symbol mappings precede a nonempty native directed projection.',
      lambda r: _directed(r) is True and _members(r) >= 2 and _edge_count(r) > 0
      and len(_saved(r, 'lookup_human_gene_symbol')) >= 2)
_bind(9, 5, 'native_context native_in_context', 'Complete native neighbor records cover every selected input and answer with no call.',
      lambda r: _members(r) >= 2 and _no_calls(r) and bool(_native_saved(r))
      and all(_complete(o) for o in _native_saved(r)))
_bind(9, 6, 'partial_native_context', 'Saved native neighbors leave an input unclassified and force acquisition.',
      lambda r: _members(r) >= 2 and bool(_native_saved(r)) and bool(_calls(r)))
_bind(9, 7, 'lookup', 'Native directed descending endpoint order is preserved for a one-way selected edge.',
      lambda r: _directed(r) is True and _members(r) == 2 and _edge_count(r) == 1
      and (_meta(r).get('descending_arc_count') or 0) > 0 and _empty_context(r))
_bind(9, 8, 'lookup', 'A selected reciprocal directed pair retains both separate arc records and weights.',
      lambda r: _directed(r) is True and _members(r) == 2 and (_meta(r).get('reciprocal_pairs') or 0) > 0 and _empty_context(r))
_bind(9, 9, 'lookup', 'A fresh singleton projection contains its one recorded native loop.',
      lambda r: _members(r) == 1 and _excluded(r) == 0 and _meta(r).get('loop_count') == 1 and _empty_context(r))
_bind(9, 10, 'lookup', 'A fresh present singleton has no native self-loop.',
      lambda r: _members(r) == 1 and _excluded(r) == 0 and _meta(r).get('loop_count') == 0 and _empty_context(r))
_bind(9, 11, 'lookup', 'A nonempty input list has no layer members and ordinary projection still resolves.',
      lambda r: _members(r) == 0 and _excluded(r) > 0 and _empty_context(r))
_bind(9, 12, 'lookup', 'An explicit empty list resolves an empty graph without any acquisition.',
      lambda r: _meta(r)['gene_ids'] == [] and _no_calls(r) and _meta(r)['request']['genes'] == [])
_bind(9, 13, 'duplicate_inputs', 'Repeated input identities produce one canonical node occurrence and native edges.',
      lambda r: len(_meta(r)['request']['genes']) > len(set(_meta(r)['request']['genes'])) and _members(r) > 0)
_bind(9, 14, 'lookup', 'An explicit first-K canonical edge preview is genuinely incomplete.',
      lambda r: _meta(r)['request']['mode'] == 'top_k' and _result(r).get('collection', {}).get('complete') is False
      and _result(r).get('edge_count', 0) > 0)
_bind(9, 15, 'bounded_context shorter_preview_context', 'A saved incomplete edge preview cannot settle a requested complete graph.',
      lambda r: _meta(r)['request']['mode'] == 'complete' and _partial_graph(r) and bool(_calls(r)))
_bind(9, 16, 'unknown_total context', 'An exact saved first-K preview with unknown whole-graph total settles the same bound without a call.',
      lambda r: _meta(r)['request']['mode'] == 'top_k' and _no_calls(r) and any(_incomplete(o)
          and o['result']['collection']['total_count'] is None and o['arguments']['mode'] == 'top_k'
          and o['arguments']['top_k'] == _meta(r)['request']['top_k']
          and set(o['arguments']['gene_ids']) == set(_meta(r)['gene_ids']) for o in _saved(r, 'induce_subgraph')))
_bind(9, 17, 'paged overlap empty_page prefix_context unknown_total', 'Complete selected edges require following an actual graph continuation cursor.',
      lambda r: _meta(r)['request']['mode'] == 'complete' and _paged_acquisitions(r, 'induce_subgraph'))
_bind(9, 18, 'page_terminal', 'A later edge page fails while earlier edge and complete membership observations remain retained.',
      lambda r: _result(r).get('failed_tool') == 'induce_subgraph'
      and _result(r).get('arguments', {}).get('cursor') is not None and _partial_graph(r))
_bind(9, 19, 'unresolved_identity', 'An unresolved registry symbol is an identity prerequisite diagnostic, not graph-global absence.',
      lambda r: _issue(r, 'UNRESOLVED_IDENTITY'), generator='runtime.world_model.s2_9_corpus.build_diagnostic')
_bind(9, 20, 'recovered', 'A cold graph acquisition recovers after one permitted retry.',
      lambda r: _empty_context(r) and _error_count(r) == 1 and len(_calls(r, 'induce_subgraph')) == 2)
_bind(9, 21, 'exhausted', 'Exhausted graph retry retains earlier supplied native neighbor evidence.',
      lambda r: _error_count(r) == 2 and bool(_native_saved(r))
      and _result(r).get('failed_tool') == 'induce_subgraph')
_bind(9, 22, 'superset_context', 'Restrict a complete induced superset and recompute the requested projection without a new call.',
      lambda r: _no_calls(r) and any(_complete(o) and set(o['arguments']['gene_ids']) > set(_meta(r)['gene_ids'])
                                  for o in _saved(r, 'induce_subgraph')))
_bind(9, 23, 'native_context', 'Complete outgoing lists alone cover a multi-member directed selected graph with no call.',
      lambda r: _directed(r) is True and _members(r) >= 2 and _no_calls(r) and bool(_native_saved(r))
      and all(_complete(o) and o['arguments']['direction'] == 'out' for o in _native_saved(r)))
_bind(9, 24, 'missing_genes', 'A missing input list is rejected without inventing the explicit empty-set result.',
      lambda r: _issue(r, 'MISSING_GENES') and _meta(r)['request']['genes'] is None,
      generator='runtime.world_model.s2_9_corpus.build_diagnostic')
_bind(9, 25, 'stale', 'A saved different graph version requires active selected-graph evidence.',
      lambda r: bool(_calls(r)) and any(g['graph_version'] != _meta(r)['graph_version'] for o in _saved(r) for g in o['provenance']['graph']))
_bind(9, 26, 'membership_gate gate_context', 'An explicit all-members prerequisite fails on actual exclusions before graph construction.',
      lambda r: _issue(r, 'MEMBERSHIP_PRECONDITION_UNSATISFIED') and _meta(r)['request']['require_all_in_layer'] is True
      and _excluded(r) > 0)
_bind(9, 27, 'positive_loop_context', 'A genuinely incomplete positive native page proves the singleton loop without a new call.',
      lambda r: _members(r) == 1 and _excluded(r) == 0 and _meta(r).get('loop_count') == 1
      and _partial_native(r) and _no_calls(r), generator='runtime.world_model.s2_9_loop_context.build_rows')
_bind(9, 28, 'tool_unknown_layer', 'A resolver rejects the requested unknown layer and retains its atomic rejected arguments.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _result(r).get('rejected_arguments') is not None,
      generator='runtime.world_model.s2_9_corpus.build_diagnostic')
_bind(9, 29, 'disjoint_context', 'Complete disjoint induced batches do not prove missing cross-batch edges, so acquire the whole projection.',
      lambda r: _members(r) >= 2 and bool(_calls(r)) and _disjoint_batches_miss_edge(r))
_bind(9, 30, 'symbol_terminal', 'Terminal graph failure retains the original supplied unique registry mappings.',
      lambda r: _error_count(r) == 1 and bool(_saved(r, 'lookup_human_gene_symbol'))
      and _result(r).get('failed_tool') == 'induce_subgraph')

# S2.10: classify actual selected component problems, not connectivity through
# unselected bridge nodes or completeness of unrelated full-layer collections.
_bind(10, 1, 'lookup reordered unqualified', 'At least two selected undirected members form one component with no excluded inputs.',
      lambda r: _meta(r)['connectivity_kind'] == 'undirected' and _members(r) >= 2 and _excluded(r) == 0
      and _meta(r).get('component_count') == 1 and _empty_context(r))
_bind(10, 2, 'outside_route_context', 'An unselected outside-set route cannot join two selected components.',
      lambda r: _meta(r).get('component_count', 0) >= 2 and bool(_calls(r, 'induce_subgraph'))
      and len(_saved(r, 'get_edge')) >= 2)
_bind(10, 3, 'lookup', 'One projected component with excluded requested inputs keeps all-input single_component false.',
      lambda r: _members(r) >= 1 and _excluded(r) > 0 and _meta(r).get('component_count') == 1
      and _result(r).get('single_component') is False and _empty_context(r))
_bind(10, 4, 'symbol_context', 'Supplied unique registry mappings precede an acquired connected selected graph.',
      lambda r: _members(r) >= 2 and _meta(r).get('component_count') == 1
      and bool(_saved(r, 'lookup_human_gene_symbol')))
_bind(10, 5, 'context duplicate_context', 'Matching complete saved induced graph supplies the multi-member partition without calls.',
      lambda r: _members(r) >= 2 and _no_calls(r) and bool(_saved(r, 'induce_subgraph'))
      and all(_complete(o) for o in _saved(r, 'induce_subgraph')))
_bind(10, 6, 'wrong_layer', 'An induced graph from another layer cannot fill current component evidence.',
      lambda r: bool(_calls(r)) and any(o['arguments'].get('layer_id') != _meta(r)['layer_id'] for o in _saved(r)))
_bind(10, 7, 'lookup', 'An all-member edgeless selected graph of at least two nodes yields singleton components.',
      lambda r: _members(r) >= 2 and _excluded(r) == 0 and _edge_count(r) == 0
      and _meta(r).get('component_count') == _members(r) and _empty_context(r))
_bind(10, 8, 'duplicate_inputs', 'Repeated IDs produce one canonical node and one occurrence in the component partition.',
      lambda r: len(_meta(r)['request']['genes']) > len(set(_meta(r)['request']['genes'])) and _members(r) > 0)
_bind(10, 9, 'lookup', 'An explicit empty component request produces zero groups and no acquisition.',
      lambda r: _meta(r)['request']['genes'] == [] and _no_calls(r))
_bind(10, 10, 'lookup', 'Every requested input is excluded while ordinary projection completes an empty component workflow.',
      lambda r: _members(r) == 0 and _excluded(r) > 0 and _empty_context(r))
_bind(10, 11, 'lookup', 'A present singleton without a loop is settled from membership alone.',
      lambda r: _members(r) == 1 and _excluded(r) == 0 and _meta(r).get('loop_count') == 0
      and not _calls(r, 'induce_subgraph') and _empty_context(r))
_bind(10, 12, 'lookup', 'A native singleton self-loop adds neither a node nor a component.',
      lambda r: _members(r) == 1 and _excluded(r) == 0 and _meta(r).get('loop_count') == 1
      and _meta(r).get('component_count') == 1 and _empty_context(r))
_bind(10, 13, 'lookup', 'A directed chain with at least three strong groups is one weak selected component.',
      lambda r: _meta(r)['connectivity_kind'] == 'weak' and _directed_chain(r)
      and _meta(r).get('component_count') == 1 and _empty_context(r))
_bind(10, 14, 'lookup', 'A directed chain has multiple strong singleton components despite weak connectivity.',
      lambda r: _meta(r)['connectivity_kind'] == 'strong' and _directed_chain(r)
      and _meta(r).get('component_count') == _members(r) and _members(r) >= 3 and _empty_context(r))
_bind(10, 15, 'lookup', 'A selected reciprocal directed pair forms one strong component.',
      lambda r: _meta(r)['connectivity_kind'] == 'strong' and _members(r) == 2
      and (_meta(r).get('reciprocal_pairs') or 0) > 0 and _meta(r).get('component_count') == 1 and _empty_context(r))
_bind(10, 16, 'missing_kind', 'A directed component request without a connectivity kind is rejected locally.',
      lambda r: _issue(r, 'CONNECTIVITY_KIND_REQUIRED') and _meta(r)['request']['connectivity_kind'] is None,
      generator='runtime.world_model.s2_10_corpus.build_diagnostic')
_bind(10, 17, 'unavailable_kind', 'Requesting an undirected interpretation of a native directed layer is rejected locally.',
      lambda r: _issue(r, 'CONNECTIVITY_KIND_UNAVAILABLE') and _meta(r)['request']['connectivity_kind'] == 'undirected'
      and _no_calls(r), generator='runtime.world_model.s2_10_corpus.build_diagnostic')
_bind(10, 18, 'spanning_context', 'A genuinely incomplete undirected or weak edge prefix already spans one exact selected component.',
      lambda r: _meta(r)['connectivity_kind'] != 'strong' and _saved_connected_prefix(r, _meta(r)['connectivity_kind']) and _no_calls(r)
      and _meta(r).get('component_count') == 1 and _members(r) >= 2)
_bind(10, 19, 'insufficient_context', 'A disconnected saved preview is insufficient and required acquisition supplies the exact partition.',
      lambda r: _partial_graph(r) and bool(_calls(r, 'induce_subgraph')) and _members(r) >= 2)
_bind(10, 20, 'strong_witness_context', 'A genuinely incomplete native directed prefix already strongly connects all selected members.',
      lambda r: _meta(r)['connectivity_kind'] == 'strong' and _saved_connected_prefix(r, 'strong') and _no_calls(r)
      and _meta(r).get('component_count') == 1 and _members(r) >= 2,
      generator='runtime.world_model.s2_10_supplemental.build_rows')
_bind(10, 21, 'weak_witness_strong_context', 'A weakly connected saved prefix cannot decide the requested strong partition.',
      lambda r: _meta(r)['connectivity_kind'] == 'strong' and _saved_weak_not_strong(r) and bool(_calls(r, 'induce_subgraph')))
_bind(10, 22, 'bridge_context', 'Restrict a complete saved superset and recompute components after removing a connecting node.',
      lambda r: _meta(r).get('component_count', 0) >= 2 and _no_calls(r)
      and _saved_superset_bridge(r))
_bind(10, 23, 'native_context', 'Complete undirected native neighbor lists settle a multi-member exact partition without calls.',
      lambda r: _meta(r)['connectivity_kind'] == 'undirected' and _members(r) >= 2 and _no_calls(r)
      and bool(_native_saved(r)) and all(_complete(o) and o['arguments']['direction'] == 'undirected' for o in _native_saved(r)))
_bind(10, 24, 'native_context', 'Complete outgoing native records settle a weak directed multi-member partition without calls.',
      lambda r: _meta(r)['connectivity_kind'] == 'weak' and _members(r) >= 2 and _no_calls(r)
      and bool(_native_saved(r)) and all(_complete(o) and o['arguments']['direction'] == 'out' for o in _native_saved(r)))
_bind(10, 25, 'paged overlap empty_page prefix_context unknown_total', 'Follow real induced-edge continuation pages before reporting the exact partition.',
      lambda r: _members(r) >= 2 and _paged_acquisitions(r, 'induce_subgraph'))
_bind(10, 26, 'page_terminal', 'A required later edge page fails while prior membership and incomplete edge evidence remain attributed.',
      lambda r: _result(r).get('failed_tool') == 'induce_subgraph'
      and _result(r).get('arguments', {}).get('cursor') is not None and _partial_graph(r))
_bind(10, 27, 'recovered', 'A required component graph acquisition recovers after one authorized retry.',
      lambda r: _error_count(r) == 1 and len(_calls(r, 'induce_subgraph')) == 2)
_bind(10, 28, 'membership_exhausted', 'Required remaining membership retries exhaust while earlier complete native neighbors are retained.',
      lambda r: _result(r).get('failed_tool') == 'get_nodes_by_layer' and _error_count(r) == 2
      and bool(_native_saved(r)) and all(_complete(o) for o in _native_saved(r)))
_bind(10, 29, 'tool_unknown_layer', 'An unknown requested layer receives an atomic tool rejection with literal rejected arguments.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _result(r).get('rejected_arguments') is not None,
      generator='runtime.world_model.s2_10_corpus.build_diagnostic')
_bind(10, 30, 'missing_genes', 'Missing genes are rejected distinctly from an explicit empty component set.',
      lambda r: _issue(r, 'MISSING_GENES') and _meta(r)['request']['genes'] is None,
      generator='runtime.world_model.s2_10_corpus.build_diagnostic')
_bind(10, 31, 'unresolved_identity', 'An unresolved registry identity remains a prerequisite fault rather than an excluded graph node.',
      lambda r: _issue(r, 'UNRESOLVED_IDENTITY'), generator='runtime.world_model.s2_10_corpus.build_diagnostic')
_bind(10, 32, 'stale', 'A saved previous graph version cannot establish the active component partition.',
      lambda r: bool(_calls(r)) and any(g['graph_version'] != _meta(r)['graph_version'] for o in _saved(r) for g in o['provenance']['graph']))
_bind(10, 33, 'membership_gate gate_context', 'An explicit all-members gate blocks a workflow on source-supported excluded inputs.',
      lambda r: _issue(r, 'MEMBERSHIP_PRECONDITION_UNSATISFIED') and _meta(r)['request']['require_all_in_layer'] is True and _excluded(r) > 0)
_bind(10, 34, 'symbol_terminal', 'A terminal graph failure preserves original supplied registry identities.',
      lambda r: _result(r).get('failed_tool') == 'induce_subgraph' and _error_count(r) == 1
      and bool(_saved(r, 'lookup_human_gene_symbol')))

for _case_id, _gap in {
    'S2.8.22': 'Each current incompatible-context variant changes one axis; no public variant combines wrong scope, direction, and version in one saved degree record.',
    'S2.8.26': 'The rank_exhausted variant retains a degree_only get_degree observation, not the reviewed complete native get_neighbors evidence.',
    'S2.9.16': 'The public builder cannot combine saved matching preview context with an unknown whole-graph total: context discloses totals and unknown_total performs a fresh traversal.',
    'S2.9.21': 'The exhausted graph variant starts without saved native neighbors; no public variant combines that context with exhausted graph retries.',
    'S2.10.17': 'The unavailable_kind diagnostic requests weak connectivity on an undirected layer; the reviewed opposite contrast requests undirected connectivity on a directed layer.',
    'S2.10.28': 'The membership_exhausted variant starts without the reviewed saved complete native neighbors; no public variant combines these states.',
}.items():
    CASE_BINDINGS[_case_id]['implementation_gap'] = _gap


def matches(case_id, row):
    """Test one explicit case condition against an already validated candidate.

    Malformed candidates cannot count toward coverage. Unknown case IDs remain
    programming errors. The caller separately gates the frozen outcome and
    framing and allocates distinct rows across overlapping compatible cases.
    """
    binding = CASE_BINDINGS[case_id]
    try:
        if (_meta(row)['question_family'] != case_id.rsplit('.', 1)[0]
                or _meta(row)['variant'] not in binding['variants']):
            return False
        return bool(_PREDICATES[case_id](row))
    except (KeyError, IndexError, TypeError, ValueError):
        return False
