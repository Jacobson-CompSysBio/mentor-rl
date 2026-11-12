"""Bind the 112 reviewed S2.11–13 cases to precise generator audit predicates.

These rules recognize behavior in validated source rows. They never infer a
numbered case from its title, count generator variants as cases, or make private
source facts visible to a model. Source enrichments are audit-only evidence.
The aggregate auditor separately checks reviewed outcomes, framing, and quotas.
"""

from __future__ import annotations

from functools import lru_cache
import json


CASE_BINDINGS = {}
_PREDICATES = {}


def _bind(family, number, variants, requirement, predicate, *, generator='build_row'):
    case_id = f'S2.{family}.{number}'
    CASE_BINDINGS[case_id] = {
        'generator': f'runtime.world_model.s2_{family}_corpus.{generator}',
        'variants': tuple(variants.split()), 'requirements': requirement,
    }
    _PREDICATES[case_id] = predicate


def _m(row):
    return row['metadata']


def _r(row):
    return row['messages'][-1]['content']['result']


def _source(row):
    return row.get('_audit_source', {})


@lru_cache(maxsize=32)
def _context(text):
    return json.loads(text).get('observations', [])


def _obs(row, operation=None):
    observations = _context(row['messages'][1]['content'])
    return observations if operation is None else [o for o in observations if o.get('operation') == operation]


def _calls(row, name=None):
    calls = [m['tool_call'] for m in row['messages'] if 'tool_call' in m]
    return calls if name is None else [c for c in calls if c.get('name') == name]


def _cards(row, claim=None):
    cards = _r(row).get('claim_cards', [])
    return cards if claim is None else [c for c in cards if c.get('claim') == claim]


def _assessment(row, claim, supported, *, truth=None):
    cards = _cards(row, claim)
    return bool(cards) and all(c.get('claim_supported') is supported
        and (truth is None or c.get('statement_true') is truth) for c in cards)


def _task(row, kind, claim=None, scope=None):
    meta = _m(row)
    return (meta.get('question_kind') == kind and meta.get('claim') == claim
            and (scope is None or meta.get('truth_scope') == scope))


def _no_calls(row):
    return not _calls(row)


def _fresh(row, name):
    return not _obs(row) and len(_calls(row, name)) == 1


def _same_endpoint(row):
    genes = _m(row).get('gene_ids', [])
    return len(genes) == 2 and genes[0] == genes[1]


def _members_only(row):
    return not _r(row).get('missing_gene_ids') and not _r(row).get('not_in_layer_gene_ids')


def _path(row, found, hops=None):
    result = _r(row)
    return (result.get('path_found') is found
            and (hops is None or result.get('hop_count') == hops))


def _ordinary_path(row):
    return _task(row, 'path') and 'path_found' in _r(row) and not _cards(row)


def _negative_path(row):
    return _path(row, False) and _members_only(row) and not _same_endpoint(row)


def _positive_edges(row):
    return [o for o in _obs(row, 'get_edge') if o.get('result', {}).get('edge_exists') is True]


def _negative_edge_context(row):
    return any(o.get('result', {}).get('edge_exists') is False for o in _obs(row, 'get_edge'))


def _saved_path(row, found):
    return any(o.get('result', {}).get('path_found') is found for o in _obs(row, 'shortest_path'))


def _family(row):
    meta = _m(row)
    if meta.get('layer_family'):
        return meta['layer_family']
    layer = meta.get('layer_id', '')
    return ('bulk_pen' if layer.startswith('bulkPEN:') else 'scpen' if layer.startswith('scPEN:')
            else 'ppi' if layer == 'HumanNetV3:string_ppi' else 'coexpression'
            if layer == 'HumanNetV3:coexpression' else 'tf_target' if layer.startswith('TFs:') else 'other')


def _has_registry(row, count=1):
    return len(_obs(row, 'lookup_human_gene_symbol')) >= count


def _exact_retry(row, name):
    calls = _calls(row, name)
    return len(calls) == 2 and calls[0] == calls[1]


def _issue(row, code):
    return _r(row).get('issue_code') == code


# S2.11: path existence, exact shortest witnesses, and scoped claim calibration.
_bind(11, 1, 'lookup', 'Fresh exhaustive undirected no-path result; list graph claim cards.',
      lambda r: _task(r, 'path_claims') and _negative_path(r) and _m(r).get('direction') == 'undirected'
      and _fresh(r, 'shortest_path'))
_bind(11, 2, 'lookup', 'Fresh checked no-path result leaves biological noninteraction unsupported.',
      lambda r: _task(r, 'path_support', 'no_interaction_in_biology') and _negative_path(r)
      and _fresh(r, 'shortest_path') and _assessment(r, 'no_interaction_in_biology', False))
_bind(11, 3, 'lookup', 'Fresh local no-path result does not support absence in other layers.',
      lambda r: _task(r, 'path_support', 'absence_holds_in_other_layers') and _negative_path(r)
      and _fresh(r, 'shortest_path') and _assessment(r, 'absence_holds_in_other_layers', False))
_bind(11, 4, 'lookup', 'Fresh no-path proof makes an exact current-scope absence proposition true.',
      lambda r: _task(r, 'path_truth', 'no_path_recorded_in_scope', 'scope') and _negative_path(r)
      and _fresh(r, 'shortest_path') and _assessment(r, 'no_path_recorded_in_scope', True, truth=True))
_bind(11, 5, 'lookup', 'A fresh exact two-hop result refutes scoped no-path truth.',
      lambda r: _task(r, 'path_truth', 'no_path_recorded_in_scope', 'scope') and _path(r, True, 2)
      and _fresh(r, 'shortest_path') and _assessment(r, 'no_path_recorded_in_scope', False, truth=False))
_bind(11, 6, 'lookup', 'Layer absence with global membership resolves no path and includes endpoint membership cards.',
      lambda r: _task(r, 'path_claims') and _m(r).get('membership_cards') is True
      and _path(r, False) and bool(_r(r).get('not_in_layer_gene_ids'))
      and not _r(r).get('missing_gene_ids') and not _same_endpoint(r)
      and len(_cards(r)) > 1 and _fresh(r, 'shortest_path'))
_bind(11, 7, 'lookup', 'A distinct missing multiplex endpoint leaves ordinary path fields unknown.',
      lambda r: _ordinary_path(r) and not _same_endpoint(r) and _path(r, None)
      and bool(_r(r).get('missing_gene_ids')) and _fresh(r, 'shortest_path'))
_bind(11, 8, 'edge_context', 'A saved direct-edge negative does not replace a requested two-hop shortest-path lookup.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and _negative_edge_context(r)
      and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 9, 'lookup', 'Fresh directed two-hop path preserves ordered source and destination.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and _m(r).get('direction') == 'source_to_target'
      and _fresh(r, 'shortest_path'))
_bind(11, 10, 'wrong_direction', 'Positive reversed saved path cannot answer the directed no-path request.',
      lambda r: _ordinary_path(r) and _negative_path(r) and _m(r).get('direction') == 'source_to_target'
      and _saved_path(r, True) and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 11, 'context', 'Complete matching saved no-path result completes the workflow without a call.',
      lambda r: _ordinary_path(r) and _negative_path(r) and _saved_path(r, False) and _no_calls(r))
_bind(11, 12, 'edge_context', 'A positive direct-edge observation establishes the exact one-hop path without a call.',
      lambda r: _ordinary_path(r) and _path(r, True, 1) and bool(_positive_edges(r)) and _no_calls(r))
_bind(11, 13, 'longer_route_context', 'A longer supplied route still requires the exact shorter shortest-path acquisition.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and len(_positive_edges(r)) >= 3
      and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 14, 'subgraph_context', 'Disconnected selected endpoints require a full-layer search that finds a two-hop route.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and len(_calls(r, 'shortest_path')) == 1
      and any(o['result'].get('edges') == [] and o['result'].get('layer_gene_count') == 2
              for o in _obs(r, 'induce_subgraph')))
_bind(11, 15, 'subgraph_context', 'A visible singleton self-loop does not change the zero-hop path answer.',
      lambda r: _ordinary_path(r) and _same_endpoint(r) and _path(r, True, 0) and _no_calls(r)
      and any(o['result'].get('edge_count') == 1 and o['result'].get('edges', [{}])[0].get('gene_ids')
              == _m(r)['gene_ids'] for o in _obs(r, 'induce_subgraph')))
_bind(11, 16, 'subgraph_context', 'A visible loop-free singleton still has a zero-hop path.',
      lambda r: _ordinary_path(r) and _same_endpoint(r) and _path(r, True, 0) and _no_calls(r)
      and any(o['result'].get('edges') == [] and o['result'].get('layer_gene_count') == 1
              for o in _obs(r, 'induce_subgraph')))
_bind(11, 17, 'membership_context', 'Membership proves zero hops while direct-edge absence remains unassessed.',
      lambda r: _task(r, 'path_support', 'no_edge_recorded_in_scope') and _same_endpoint(r)
      and _path(r, True, 0) and _no_calls(r) and bool(_obs(r, 'get_nodes_by_layer'))
      and _assessment(r, 'no_edge_recorded_in_scope', False))
_bind(11, 18, 'edge_context', 'Evidence-only biological support check retains the saved negative edge without a path lookup.',
      lambda r: _task(r, 'claim_support', 'no_interaction_in_biology') and _negative_edge_context(r)
      and _no_calls(r) and 'path_found' not in _r(r) and _assessment(r, 'no_interaction_in_biology', False))
_bind(11, 19, 'route_context longer_route_context', 'An arbitrary positive multihop route does not support direct-edge absence.',
      lambda r: _task(r, 'claim_support', 'no_edge_recorded_in_scope') and len(_positive_edges(r)) >= 2
      and _no_calls(r) and 'path_found' not in _r(r) and _assessment(r, 'no_edge_recorded_in_scope', False))
_bind(11, 20, 'context', 'An observed positive path refutes an all-layer no-path truth proposition without new calls.',
      lambda r: _task(r, 'truth', 'absence_holds_in_other_layers', 'all_layers') and _saved_path(r, True)
      and _no_calls(r) and _assessment(r, 'absence_holds_in_other_layers', False, truth=False))
_bind(11, 21, 'wrong_layer', 'A positive saved path from another layer still needs the requested full-layer path.',
      lambda r: _ordinary_path(r) and _path(r, True) and _saved_path(r, True)
      and len(_calls(r, 'shortest_path')) == 1 and any(o['arguments'].get('layer_id') != _m(r).get('layer_id')
              for o in _obs(r, 'shortest_path')))
_bind(11, 22, 'stale', 'A stale positive path cannot establish the required current positive path.',
      lambda r: _ordinary_path(r) and _path(r, True) and _saved_path(r, True)
      and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 23, 'tool_unknown_layer', 'Exact path resolver rejects an unknown layer and retains rejected arguments.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _r(r).get('rejected_arguments') is not None
      and len(_calls(r, 'shortest_path')) == 1, generator='build_diagnostic')
_bind(11, 24, 'invalid_pair', 'A malformed three-position path input is rejected before any call.',
      lambda r: _issue(r, 'INVALID_PAIR') and _no_calls(r), generator='build_diagnostic')
_bind(11, 25, 'missing_direction', 'A missing directed reference is ambiguous rather than guessed from endpoint order.',
      lambda r: _issue(r, 'AMBIGUOUS_PATH_DIRECTION') and _no_calls(r), generator='build_diagnostic')
_bind(11, 26, 'symbol_terminal', 'Terminal path failure preserves both supplied unique registry identities.',
      lambda r: _r(r).get('failed_tool') == 'shortest_path' and _has_registry(r, 2)
      and len(_calls(r, 'shortest_path')) == 1 and len(_r(r).get('partial_results', [])) == 2)
_bind(11, 27, 'recovered', 'One identical permitted retry recovers a positive two-hop workflow result.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and _exact_retry(r, 'shortest_path'))
_bind(11, 28, 'edge_exhausted', 'Exhausted identical path retries preserve an earlier checked negative edge.',
      lambda r: _r(r).get('failed_tool') == 'shortest_path' and _negative_edge_context(r)
      and _exact_retry(r, 'shortest_path') and bool(_r(r).get('partial_results')))
_bind(11, 29, 'tie_route_context', 'A noncanonical shortest-length source witness requires canonical tie acquisition.',
      lambda r: _ordinary_path(r) and _path(r, True, 2) and (_m(r).get('tied_shortest_paths') or 0) > 1
      and bool(_obs(r, 'induce_subgraph')) and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 30, 'lookup', 'A repeated endpoint absent from the layer resolves no layer path.',
      lambda r: _ordinary_path(r) and _same_endpoint(r) and _path(r, False)
      and bool(_r(r).get('not_in_layer_gene_ids')) and _fresh(r, 'shortest_path'))
_bind(11, 31, 'lookup', 'A repeated endpoint outside the multiplex leaves the path unknown.',
      lambda r: _ordinary_path(r) and _same_endpoint(r) and _path(r, None)
      and bool(_r(r).get('missing_gene_ids')) and _fresh(r, 'shortest_path'))
_bind(11, 32, 'execution_limit', 'An incomplete execution retains negative-edge evidence without claiming no path.',
      lambda r: _r(r).get('error_code') == 'SEARCH_INCOMPLETE' and _negative_edge_context(r)
      and len(_calls(r, 'shortest_path')) == 1)
_bind(11, 33, 'lookup', 'A fresh attainable route of at least four hops retains every witness vertex.',
      lambda r: _ordinary_path(r) and _path(r, True) and (_r(r).get('hop_count') or 0) >= 4
      and len(_r(r).get('path_gene_ids', [])) == _r(r)['hop_count'] + 1 and _fresh(r, 'shortest_path'))
_bind(11, 34, 'native_context', 'A complete empty outgoing list proves directed no-path without a new call.',
      lambda r: _ordinary_path(r) and _negative_path(r) and _m(r).get('direction') == 'source_to_target'
      and _no_calls(r) and any(o['arguments'].get('direction') == 'out' and o['result'].get('neighbors') == []
          and o['result'].get('collection', {}).get('complete') is True for o in _obs(r, 'get_neighbors')))
_bind(11, 35, 'closed_component_context', 'Complete native incidence closes an undirected component and proves full-layer no-path.',
      lambda r: _ordinary_path(r) and _negative_path(r) and _no_calls(r)
      and _m(r).get('direction') == 'undirected' and bool(_obs(r, 'get_neighbors'))
      and all(o['result'].get('collection', {}).get('complete') is True for o in _obs(r, 'get_neighbors')))
_bind(11, 36, 'context', 'A scoped no-path observation cannot decide unqualified universal truth.',
      lambda r: _task(r, 'truth', 'absence_holds_in_other_layers', 'all_layers') and _saved_path(r, False)
      and _issue(r, 'UNSUPPORTED_TRUTH_SCOPE') and _no_calls(r))


def _member(row):
    return _r(row).get('layer_gene_count') == 1 and _members_only(row)


def _layer_absent(row):
    return (_r(row).get('layer_gene_count') == 0 and bool(_r(row).get('not_in_layer_gene_ids'))
            and not _r(row).get('missing_gene_ids'))


def _membership(row, claim, category):
    result = _member(row) if category == 'member' else _layer_absent(row) if category == 'outside_layer' else bool(_r(row).get('missing_gene_ids'))
    return _task(row, 'membership_support', claim) and _m(row).get('category') == category and result


def _complete_layer_list(row):
    return any(o['result'].get('collection', {}).get('complete') is True for o in _obs(row, 'get_gene_layers'))


# S2.12: singleton classification, calibration, cross-context evidence, and gates.
_bind(12, 1, 'lookup', 'Fresh astrocyte/scPEN layer absence leaves expression unsupported.',
      lambda r: _family(r) == 'scpen' and _membership(r, 'gene_not_expressed_in_context', 'outside_layer')
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'gene_not_expressed_in_context', False))
_bind(12, 2, 'lookup', 'Fresh bulk PEN layer absence leaves expression unsupported for a workflow.',
      lambda r: _family(r) == 'bulk_pen' and _membership(r, 'gene_not_expressed_in_context', 'outside_layer')
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'gene_not_expressed_in_context', False))
_bind(12, 3, 'lookup', 'Local PEN absence does not support absence in another PEN layer.',
      lambda r: _family(r) in ('bulk_pen', 'scpen')
      and _membership(r, 'absence_holds_in_other_layers', 'outside_layer')
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'absence_holds_in_other_layers', False))
_bind(12, 4, 'lookup', 'Fresh layer absence does not support future-version absence.',
      lambda r: _membership(r, 'absence_holds_in_other_versions', 'outside_layer')
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'absence_holds_in_other_versions', False))
_bind(12, 5, 'lookup', 'Exact multiplex-member layer-absence category is true after classification.',
      lambda r: _task(r, 'membership_truth', 'gene_not_node_in_layer', 'category') and _layer_absent(r)
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'gene_not_node_in_layer', True, truth=True))
_bind(12, 6, 'lookup', 'Present layer membership refutes the exact layer-absence category.',
      lambda r: _task(r, 'membership_truth', 'gene_not_node_in_layer', 'category') and _member(r)
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'gene_not_node_in_layer', False, truth=False))
_bind(12, 7, 'layer_list_context', 'Complete containing-layer list supplies a layer negative and expression remains unsupported.',
      lambda r: _membership(r, 'gene_not_expressed_in_context', 'outside_layer') and _complete_layer_list(r)
      and _no_calls(r) and _assessment(r, 'gene_not_expressed_in_context', False))
_bind(12, 8, 'wrong_family_context', 'Another-family observation leaves target membership unknown until acquisition.',
      lambda r: _membership(r, 'gene_not_expressed_in_context', 'member')
      and bool(_obs(r, 'get_gene_layers')) and len(_calls(r, 'get_nodes_by_layer')) == 1)
_bind(12, 9, 'lookup', 'A checked global negative completes membership without inferring biological non-expression.',
      lambda r: _membership(r, 'gene_not_expressed_in_context', 'outside_multiplex')
      and _fresh(r, 'get_nodes_by_layer') and _assessment(r, 'gene_not_expressed_in_context', False))
_bind(12, 10, 'neighbor_context', 'A genuine complete zero-outgoing observation proves positive layer membership.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member') and _m(r).get('outgoing_count') == 0
      and _no_calls(r) and any(o['arguments'].get('direction') == 'out' and o['result'].get('neighbors') == []
          for o in _obs(r, 'get_neighbors')))
_bind(12, 11, 'edgeless_context', 'A complete edgeless singleton subgraph still proves positive layer membership.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member') and _no_calls(r)
      and any(o['result'].get('edges') == [] and o['result'].get('layer_gene_count') == 1
              for o in _obs(r, 'induce_subgraph')))
_bind(12, 12, 'context', 'Matching singleton classification supplies ordinary membership cards without a duplicate call.',
      lambda r: _task(r, 'membership_card') and _layer_absent(r) and _no_calls(r)
      and len(_obs(r, 'get_nodes_by_layer')) == 1)
_bind(12, 13, 'superset_context', 'Restrict a larger complete classified batch to the requested absent-layer singleton.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'outside_layer') and _no_calls(r)
      and any(len(o['arguments'].get('gene_ids', [])) > 1 for o in _obs(r, 'get_nodes_by_layer')))
_bind(12, 14, 'family_context', 'A complete matching-family layer list supplies an absent target-layer classification.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'outside_layer') and _complete_layer_list(r)
      and _no_calls(r) and _assessment(r, 'gene_not_node_in_layer', True))
_bind(12, 15, 'prefix_omission_context', 'A genuine top-one omission hides a positive target membership that must be acquired.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member')
      and len(_calls(r, 'get_nodes_by_layer')) == 1
      and any(o['result'].get('collection', {}).get('complete') is False for o in _obs(r, 'get_gene_layers')))
_bind(12, 16, 'prefix_positive_context', 'An explicitly returned top-one positive settles membership while the original list stays incomplete.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member') and _no_calls(r)
      and any(o['result'].get('collection', {}).get('complete') is False for o in _obs(r, 'get_gene_layers')))
_bind(12, 17, 'wrong_family_context', 'Global absence from another-family record proves missing multiplex membership.',
      lambda r: _membership(r, 'gene_not_node_in_multiplex', 'outside_multiplex')
      and bool(_obs(r, 'get_gene_layers')) and _no_calls(r) and _assessment(r, 'gene_not_node_in_multiplex', True))
_bind(12, 18, 'old_view_context', 'An old-view global negative cannot classify a newly present expanded-view member.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member') and _m(r).get('old_present') is False
      and bool(_obs(r, 'get_nodes_by_layer')) and len(_calls(r, 'get_nodes_by_layer')) == 1)
_bind(12, 19, 'wrong_layer_context', 'A positive saved classification in another layer cannot answer the target layer negative.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'outside_layer')
      and len(_calls(r, 'get_nodes_by_layer')) == 1
      and any(o['result'].get('layer_gene_count') == 1 for o in _obs(r, 'get_nodes_by_layer')))
_bind(12, 20, 'context', 'Evidence-only expression support uses a saved layer negative without a new classification.',
      lambda r: _task(r, 'claim_support', 'gene_not_expressed_in_context') and _m(r).get('category') == 'outside_layer'
      and _no_calls(r) and _assessment(r, 'gene_not_expressed_in_context', False))
_bind(12, 21, 'prefix_omission_context', 'Evidence-only omission does not invent absent-layer membership from a truncated list.',
      lambda r: _task(r, 'claim_support', 'gene_not_node_in_layer') and _m(r).get('category') == 'outside_layer'
      and _no_calls(r) and _assessment(r, 'gene_not_node_in_layer', False)
      and any(o['result'].get('collection', {}).get('complete') is False for o in _obs(r, 'get_gene_layers')))
_bind(12, 22, 'prefix_omission_context', 'The same incomplete omission needs a membership call when classification is required.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'outside_layer')
      and len(_calls(r, 'get_nodes_by_layer')) == 1 and _assessment(r, 'gene_not_node_in_layer', True)
      and any(o['result'].get('collection', {}).get('complete') is False for o in _obs(r, 'get_gene_layers')))
_bind(12, 23, 'lookup', 'Fresh global-absence truth uses the explicit multiplex category.',
      lambda r: _task(r, 'membership_truth', 'gene_not_node_in_multiplex', 'multiplex')
      and bool(_r(r).get('missing_gene_ids')) and _fresh(r, 'get_nodes_by_layer')
      and _assessment(r, 'gene_not_node_in_multiplex', True, truth=True))
_bind(12, 24, 'context', 'A positive saved membership refutes absence from every layer without a new call.',
      lambda r: _task(r, 'truth', 'absence_holds_in_other_layers', 'all_layers')
      and _m(r).get('category') == 'member' and _no_calls(r)
      and _assessment(r, 'absence_holds_in_other_layers', False, truth=False))
_bind(12, 25, 'context', 'A saved local negative cannot decide absence in two other explicitly named layers.',
      lambda r: _task(r, 'truth', 'absence_holds_in_other_layers', 'layers')
      and _m(r).get('category') == 'outside_layer' and len(_m(r).get('truth_layer_ids', [])) == 2
      and _issue(r, 'UNSUPPORTED_TRUTH_SCOPE') and _no_calls(r))
_bind(12, 26, 'tool_unknown_layer', 'The exact node resolver rejects a syntactically valid unregistered layer.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _r(r).get('rejected_arguments') is not None
      and len(_calls(r, 'get_nodes_by_layer')) == 1, generator='build_diagnostic')
_bind(12, 27, 'unresolved_identity', 'An unresolved symbol remains an identity blocker rather than graph absence.',
      lambda r: _issue(r, 'UNRESOLVED_IDENTITY') and len(_calls(r, 'lookup_human_gene_symbol')) == 1
      and not _calls(r, 'get_nodes_by_layer'), generator='build_diagnostic')
_bind(12, 28, 'ambiguous_identity', 'An explicit choice requirement retains ambiguous registry identity and defers.',
      lambda r: _issue(r, 'UNRESOLVED_IDENTITY') and len(_calls(r, 'lookup_human_gene_symbol')) == 1
      and not _calls(r, 'get_nodes_by_layer'), generator='build_diagnostic')
_bind(12, 29, 'symbol_terminal', 'Terminal singleton membership failure retains its resolved supplied registry identity.',
      lambda r: _r(r).get('failed_tool') == 'get_nodes_by_layer' and _has_registry(r)
      and len(_calls(r, 'get_nodes_by_layer')) == 1)
_bind(12, 30, 'symbol_recovered', 'Permitted identical membership retry recovers layer absence and expression support stays false.',
      lambda r: _membership(r, 'gene_not_expressed_in_context', 'outside_layer')
      and _has_registry(r) and _exact_retry(r, 'get_nodes_by_layer'))
_bind(12, 31, 'symbol_exhausted', 'Exhausted identical membership retry preserves identity and leaves membership unchecked.',
      lambda r: _r(r).get('failed_tool') == 'get_nodes_by_layer' and _has_registry(r)
      and _exact_retry(r, 'get_nodes_by_layer'))
_bind(12, 32, 'symbol_lookup', 'Fresh symbol resolution precedes a singleton classification and retains both evidence channels.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'outside_layer')
      and len(_calls(r, 'lookup_human_gene_symbol')) == 1 and len(_calls(r, 'get_nodes_by_layer')) == 1
      and _calls(r)[0]['name'] == 'lookup_human_gene_symbol')
_bind(12, 33, 'lookup', 'An explicit outer layer-membership prerequisite fails on a checked layer absence.',
      lambda r: _m(r).get('require_membership') == 'layer' and _m(r).get('category') == 'outside_layer'
      and _issue(r, 'MEMBERSHIP_PRECONDITION_UNSATISFIED') and len(_calls(r, 'get_nodes_by_layer')) == 1)
_bind(12, 34, 'null_symbol_context', 'Authoritative null-symbol ID identity still permits positive layer classification.',
      lambda r: _membership(r, 'gene_not_node_in_layer', 'member') and _m(r).get('registry_state') == 'null_symbol'
      and bool(_obs(r, 'lookup_human_ensembl_gene_id')) and len(_calls(r, 'get_nodes_by_layer')) == 1)
_bind(12, 35, 'tool_unknown_multiplex', 'An exact unknown-multiplex resolver rejection produces no calibration card.',
      lambda r: _issue(r, 'UNKNOWN_MULTIPLEX') and _r(r).get('rejected_arguments') is not None
      and len(_calls(r, 'get_nodes_by_layer')) == 1, generator='build_diagnostic')
_bind(12, 36, 'context', 'Layer absence does not yield a true/false biological expression verdict.',
      lambda r: _task(r, 'truth', 'gene_not_expressed_in_context', 'expression')
      and _m(r).get('category') == 'outside_layer' and _issue(r, 'UNSUPPORTED_TRUTH_SCOPE') and _no_calls(r))


def _edge_task(row, claim, positive):
    return _task(row, 'edge_support', claim) and _r(row).get('edge_exists') is positive


def _all_pair_cards(row, claim, supported):
    cards = _cards(row)
    return len(cards) >= 2 and all(c.get('claim') == claim and c.get('claim_supported') is supported for c in cards)


def _pen_context_families(row):
    families = []
    for observation in _positive_edges(row):
        layer = observation['arguments'].get('layer_id', '')
        families.append('bulk_pen' if layer.startswith('bulkPEN:') else 'scpen' if layer.startswith('scPEN:') else 'other')
    return families


# S2.13: family capability is distinct from observed native pair support.
for number, family, kind, claim in (
    (1, 'ppi', 'family_card', None), (2, 'coexpression', 'family_support', 'holds_in_other_contexts'),
    (3, 'tf_target', 'family_support', 'regulation_direction'), (4, 'bulk_pen', 'family_card', None),
    (5, 'scpen', 'family_support', 'causal_effect'), (6, 'other', 'family_card', None),
):
    _bind(13, number, 'taxonomy', f'Pure {family} capability {kind} with no pair acquisition or graph evidence.',
          lambda r, f=family, k=kind, c=claim: _family(r) == f and _task(r, k, c)
          and _no_calls(r) and not _obs(r) and _m(r).get('category') == 'family_capability',
          generator='build_family_row')
_bind(13, 7, 'lookup', 'Fresh positive PPI edge supports recorded physical interaction.',
      lambda r: _family(r) == 'ppi' and _edge_task(r, 'physical_interaction_recorded', True)
      and _fresh(r, 'get_edge') and _assessment(r, 'physical_interaction_recorded', True))
_bind(13, 8, 'lookup', 'Fresh checked PPI nonedge supplies no positive biological support.',
      lambda r: _family(r) == 'ppi' and _edge_task(r, 'physical_interaction_recorded', False)
      and _members_only(r) and _fresh(r, 'get_edge') and _assessment(r, 'physical_interaction_recorded', False))
_bind(13, 9, 'lookup', 'Fresh positive coexpression record has its source-defined biological meaning.',
      lambda r: _family(r) == 'coexpression' and _task(r, 'edge_card')
      and _r(r).get('edge_exists') is True and _fresh(r, 'get_edge'))
_bind(13, 10, 'lookup', 'Fresh coexpression negative leaves association unsupported rather than disproven.',
      lambda r: _family(r) == 'coexpression' and _edge_task(r, 'global_coexpression', False)
      and _members_only(r) and _fresh(r, 'get_edge') and _assessment(r, 'global_coexpression', False))
_bind(13, 11, 'lookup', 'Native directed TF positive supports exact source-target roles without causal claims.',
      lambda r: _family(r) == 'tf_target' and _m(r).get('directed') is True
      and _edge_task(r, 'regulation_direction', True) and _fresh(r, 'get_edge')
      and _assessment(r, 'regulation_direction', True))
_bind(13, 12, 'reverse_context', 'A positive reverse TF observation does not replace a requested negative orientation.',
      lambda r: _family(r) == 'tf_target' and _m(r).get('directed') is True
      and _edge_task(r, 'regulation_direction', False) and bool(_positive_edges(r))
      and len(_calls(r, 'get_edge')) == 1 and _assessment(r, 'regulation_direction', False))
_bind(13, 13, 'reciprocal_context', 'Two positive reciprocal TF observations do not establish causal feedback.',
      lambda r: _family(r) == 'tf_target' and _m(r).get('directed') is True
      and _task(r, 'claim_support', 'causal_effect') and len(_positive_edges(r)) == 2
      and _no_calls(r) and _all_pair_cards(r, 'causal_effect', False))
_bind(13, 14, 'lookup', 'A positive bulk PEN acquisition supports its supplied tissue context only.',
      lambda r: _family(r) == 'bulk_pen' and _edge_task(r, 'context_predictive_expression', True)
      and _fresh(r, 'get_edge') and _assessment(r, 'context_predictive_expression', True))
_bind(13, 15, 'lookup', 'Positive single-cell PEN acquisition retains exact cell and region context.',
      lambda r: _family(r) == 'scpen' and _task(r, 'edge_card') and _r(r).get('edge_exists') is True
      and _fresh(r, 'get_edge'))
_bind(13, 16, 'context', 'A maximum-weight native bulk record within its layer and fixed pair split does not support causal effect.',
      lambda r: _family(r) == 'bulk_pen' and _task(r, 'claim_support', 'causal_effect')
      and _source(r).get('high_weight_control') is True and bool(_positive_edges(r))
      and _no_calls(r) and _assessment(r, 'causal_effect', False))
_bind(13, 17, 'context', 'A maximum-weight native single-cell record within its layer and fixed pair split does not support causal effect.',
      lambda r: _family(r) == 'scpen' and _task(r, 'claim_support', 'causal_effect')
      and _source(r).get('high_weight_control') is True and bool(_positive_edges(r))
      and _no_calls(r) and _assessment(r, 'causal_effect', False))
_bind(13, 18, 'two_context_context', 'Two positive bulk and single-cell PEN contexts retain their distinct weight meanings.',
      lambda r: _task(r, 'claim_card') and {'bulk_pen', 'scpen'} <= set(_pen_context_families(r))
      and len(_cards(r)) == 2 and _no_calls(r))
_bind(13, 19, 'pen_contrast_context', 'A saved bulk positive does not replace a required single-cell negative acquisition.',
      lambda r: _family(r) == 'scpen' and _edge_task(r, 'context_predictive_expression', False)
      and 'bulk_pen' in _pen_context_families(r) and len(_calls(r, 'get_edge')) == 1)
_bind(13, 20, 'pen_contrast_context', 'A saved single-cell positive does not replace a required bulk negative acquisition.',
      lambda r: _family(r) == 'bulk_pen' and _edge_task(r, 'context_predictive_expression', False)
      and 'scpen' in _pen_context_families(r) and len(_calls(r, 'get_edge')) == 1)
_bind(13, 21, 'two_context_context', 'Two observed PEN contexts do not support transfer to unexamined contexts.',
      lambda r: _task(r, 'claim_support', 'holds_in_other_contexts')
      and {'bulk_pen', 'scpen'} <= set(_pen_context_families(r))
      and _all_pair_cards(r, 'holds_in_other_contexts', False) and _no_calls(r))
_bind(13, 22, 'family_context', 'Family metadata alone provides no physical-interaction evidence for a pair.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'physical_interaction_recorded')
      and not _obs(r) and _no_calls(r) and _assessment(r, 'physical_interaction_recorded', False))
_bind(13, 23, 'wrong_pair_context', 'A positive PPI observation supports only its own pair, not the different target pair.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'physical_interaction_recorded')
      and len(_positive_edges(r)) == 1 and len(_cards(r)) >= 2 and _no_calls(r)
      and {c.get('claim_supported') for c in _cards(r, 'physical_interaction_recorded')} == {False, True})
_bind(13, 24, 'old_view_context stale', 'Older-view pair evidence cannot replace a positive required current-view PPI acquisition.',
      lambda r: _family(r) == 'ppi' and _edge_task(r, 'physical_interaction_recorded', True)
      and bool(_obs(r, 'get_edge')) and len(_calls(r, 'get_edge')) == 1)
_bind(13, 25, 'wrong_layer_context', 'A saved PPI negative cannot replace a positive coexpression acquisition.',
      lambda r: _family(r) == 'coexpression' and _edge_task(r, 'global_coexpression', True)
      and _negative_edge_context(r) and len(_calls(r, 'get_edge')) == 1
      and any(o['arguments'].get('layer_id') == 'HumanNetV3:string_ppi' for o in _obs(r, 'get_edge')))
_bind(13, 26, 'reverse_context', 'Evidence-only reverse TF support returns separate supported and unsupported orientation cards.',
      lambda r: _family(r) == 'tf_target' and _m(r).get('directed') is True
      and _task(r, 'claim_support', 'regulation_direction') and bool(_positive_edges(r)) and _no_calls(r)
      and {c.get('claim_supported') for c in _cards(r, 'regulation_direction')} == {False, True})
_bind(13, 27, 'context', 'A matching positive PPI observation supports evidence-only physical interaction without duplicate acquisition.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'physical_interaction_recorded')
      and len(_positive_edges(r)) == 1 and len(_cards(r)) == 1 and _no_calls(r)
      and _assessment(r, 'physical_interaction_recorded', True))
_bind(13, 28, 'lookup', 'Known multiplex endpoint outside the layer yields a checked negative edge with no biological support.',
      lambda r: _family(r) == 'ppi' and _edge_task(r, 'physical_interaction_recorded', False)
      and bool(_r(r).get('not_in_layer_gene_ids')) and not _r(r).get('missing_gene_ids')
      and _fresh(r, 'get_edge') and _assessment(r, 'physical_interaction_recorded', False))
_bind(13, 29, 'lookup', 'A missing required multiplex endpoint leaves edge fields unavailable and emits no unassessed cards.',
      lambda r: _task(r, 'edge_support', 'physical_interaction_recorded') and bool(_r(r).get('missing_gene_ids'))
      and _r(r).get('edge_exists') is None and not _cards(r) and _fresh(r, 'get_edge'))
_bind(13, 30, 'context', 'An actual supplied positive PPI edge refutes an exact scoped graph no-edge statement.',
      lambda r: _family(r) == 'ppi' and _m(r).get('question_kind') == 'graph_truth'
      and _m(r).get('graph_claim') == 'no_edge_recorded_in_scope'
      and _m(r).get('graph_truth_scope') == 'scope' and bool(_positive_edges(r)) and _no_calls(r)
      and _assessment(r, 'no_edge_recorded_in_scope', False, truth=False))
_bind(13, 31, 'context', 'A bulk PEN positive cannot decide biological causal truth; retain partials and defer.',
      lambda r: _family(r) == 'bulk_pen' and _task(r, 'biological_truth', 'causal_effect')
      and bool(_positive_edges(r)) and _issue(r, 'UNSUPPORTED_TRUTH_SCOPE') and _no_calls(r))
_bind(13, 32, 'symbol_terminal', 'Terminal graph failure retains both unique supplied symbol identities.',
      lambda r: _r(r).get('failed_tool') == 'get_edge' and _has_registry(r, 2)
      and len(_calls(r, 'get_edge')) == 1)
_bind(13, 33, 'recovered', 'One identical retry recovers a directed TF positive and its source-target claim.',
      lambda r: _family(r) == 'tf_target' and _m(r).get('directed') is True
      and _edge_task(r, 'regulation_direction', True) and _exact_retry(r, 'get_edge')
      and _assessment(r, 'regulation_direction', True))
_bind(13, 34, 'partial_exhausted', 'Exhausted graph retries preserve an earlier completed positive graph observation.',
      lambda r: _r(r).get('failed_tool') == 'get_edge' and bool(_positive_edges(r))
      and _exact_retry(r, 'get_edge') and bool(_r(r).get('partial_results')))
_bind(13, 35, 'tool_unknown_layer', 'A syntactically valid unregistered layer yields exact resolver rejection with no edge evidence.',
      lambda r: _issue(r, 'UNKNOWN_LAYER') and _r(r).get('rejected_arguments') is not None
      and len(_calls(r, 'get_edge')) == 1, generator='build_diagnostic')
_bind(13, 36, 'context', 'A positive HumanNet PPI observation does not support integrated cofunction.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'cofunctional_association')
      and bool(_positive_edges(r)) and _no_calls(r) and _assessment(r, 'cofunctional_association', False))
_bind(13, 37, 'route_context', 'A positive two-edge PPI route is not a direct physical-interaction record for its endpoints.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'physical_interaction_recorded')
      and len(_positive_edges(r)) == 2 and _no_calls(r) and _assessment(r, 'physical_interaction_recorded', False))
_bind(13, 38, 'context', 'A supplied missing-multiplex edge prerequisite can still settle evidence-only physical support as unsupported.',
      lambda r: _family(r) == 'ppi' and _task(r, 'claim_support', 'physical_interaction_recorded')
      and _m(r).get('category') == 'outside_multiplex' and _no_calls(r)
      and any(o['result'].get('edge_exists') is None and o['result'].get('missing_gene_ids') for o in _obs(r, 'get_edge'))
      and _assessment(r, 'physical_interaction_recorded', False))
_bind(13, 39, 'unresolved_identity', 'An unresolved symbol blocks required acquisition without inventing graph absence.',
      lambda r: _issue(r, 'UNRESOLVED_IDENTITY') and len(_calls(r, 'lookup_human_gene_symbol')) == 1
      and not _calls(r, 'get_edge'), generator='build_diagnostic')
_bind(13, 40, 'lookup', 'Fresh successful positive PPI assessment supplies a workflow step.',
      lambda r: _family(r) == 'ppi' and _edge_task(r, 'physical_interaction_recorded', True)
      and _fresh(r, 'get_edge') and _assessment(r, 'physical_interaction_recorded', True))


def matches(case_id, row):
    """Recognize one reviewed case conservatively in an already validated row.

    Outcome and framing checks belong to the aggregate auditor as well. Missing
    source enrichment or malformed required evidence cannot establish coverage.
    """
    binding = CASE_BINDINGS.get(case_id)
    if binding is None:
        return False
    try:
        return (_m(row).get('question_family') == '.'.join(case_id.split('.')[:2])
                and _m(row).get('variant') in binding['variants'] and bool(_PREDICATES[case_id](row)))
    except (KeyError, TypeError, ValueError, IndexError):
        return False
