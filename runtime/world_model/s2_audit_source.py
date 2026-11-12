"""Supply pinned native facts for checking catalog-case source conditions.

These private audit features are never added to model messages or corpus files.
They qualify examples; the model's answer must still follow its visible proof.
"""

from fractions import Fraction
import json

from .s2_corpus import _pair_group
from .s2_multiplex import MultiplexRuntime, SelectionError


def nearest_below_threshold(selection, threshold=95):
    """Read exact complete-population ranks and attainable fixed-split degrees."""
    ranks = {}
    for domain in selection['domains']:
        histogram = domain['degree_histogram']
        size, lower, values = sum(item['node_count'] for item in histogram), 0, {}
        for item in histogram:
            degree, equal = item['degree'], item['node_count']
            values[degree] = Fraction(50) if size == 1 else Fraction(100 * (2 * lower + equal - 1), 2 * (size - 1))
            lower += equal
        ranks[json.dumps(domain['layer_scope'], sort_keys=True), domain['direction']] = values
    result = {}
    for cell in selection['cells']:
        key = json.dumps(cell['layer_scope'], sort_keys=True), cell['direction']
        values = ranks[key]
        below = [degree for degree in cell['available_degree_values'] if values[degree] < threshold]
        result[(*key, cell['split'])] = max(below, key=lambda degree: values[degree]) if below else None
    return result


class SourceFeatures:
    """Cache complete source features shared by many evidence/wording variants."""

    def __init__(self, view, index, *, seed=20261028, nearest=None):
        self.view, self.index, self.seed = view, index, seed
        self.nearest = nearest or {}
        self.multiplex = MultiplexRuntime(view)
        self.cache, self.weight_maxima = {}, None

    def _high_weights(self):
        """Find exact maximum weights per native layer and fixed pair split."""
        if self.weight_maxima is None:
            self.weight_maxima = {}
            for layer_id in self.view.layer_order:
                for pair, weight in self.view.layers[layer_id].edges.items():
                    split = _pair_group(pair, self.seed)[1]
                    key = layer_id, split
                    self.weight_maxima[key] = max(self.weight_maxima.get(key, weight), weight)
        return self.weight_maxima

    def for_row(self, row):
        """Return true native conditions without changing the trace being audited."""
        meta = row['metadata']
        family = meta['question_family'].replace('.single_edge', '')
        keys = ('gene_id', 'gene_ids', 'queries', 'layer_id', 'layer_scope', 'direction', 'top_k', 'degree', 'split')
        identity = family, json.dumps({name: meta.get(name) for name in keys}, sort_keys=True)
        if identity in self.cache:
            return self.cache[identity]
        view, result = self.view, {}
        layer_id, gene = meta.get('layer_id'), meta.get('gene_id')
        pair = meta.get('gene_ids')
        if family == 'S2.7' and meta.get('queries'):
            pair = [query['gene_id'] if isinstance(query, dict) else query[0] for query in meta['queries']]
        if layer_id in view.layers:
            layer = view.layers[layer_id]
            result['directed'] = layer.directed
            if gene is None and pair and len(set(pair)) == 1:
                gene = pair[0]
            if gene is not None:
                result['native_self_loop'] = (gene, gene) in layer.edges
                directions = self.index.native_index.directions(layer_id)
                counts = {side: len(self.index.native_index.neighbors(layer_id, gene, side)) for side in directions}
                result['native_neighbor_counts'] = counts
                side = meta.get('direction')
                active = directions if side in ('both', 'incident') else (side,) if side in directions else directions
                native = {neighbor for direction in active for neighbor, _ in self.index.native_index.neighbors(layer_id, gene, direction)}
                result['native_neighbor_count'] = len(native)
                bound = meta.get('top_k')
                result['cutoff_tie'] = False
                if type(bound) is int and bound > 0:
                    result['cutoff_tie'] = any(len(rows) > bound and rows[bound - 1][1] == rows[bound][1]
                        for rows in (self.index.native_index.neighbors(layer_id, gene, direction) for direction in active))
            if pair and len(pair) == 2:
                first, second = pair
                native_pair = tuple(pair) if layer.directed else tuple(sorted(pair))
                reverse = (second, first) if layer.directed else native_pair
                result.update(edge_exists=native_pair in layer.edges,
                    reverse_edge_exists=reverse in layer.edges,
                    native_weight=layer.edges.get(native_pair),
                    input_pair_adjacent=native_pair in layer.edges or reverse in layer.edges,
                    input_pair_has_self_loop=any((item, item) in layer.edges for item in pair))
                if family == 'S2.13' and native_pair in layer.edges:
                    split = _pair_group(tuple(pair), self.seed)[1]
                    result['high_weight_control'] = layer.edges[native_pair] == self._high_weights()[layer_id, split]
            if family in ('S2.9', 'S2.10') and pair is not None:
                result['native_edges'] = self.index.edges(layer_id, pair)
                result['input_category_counts'] = {key: len(value) for key, value in self.index.partition(layer_id, pair).items()}
        scope = meta.get('layer_scope')
        if scope is not None:
            try:
                selected = self.multiplex.resolve_scope(scope)
            except (SelectionError, ValueError, TypeError, KeyError):
                selected = ()
            result['selected_layer_count'] = len(selected)
            result['mixed_native_representations'] = len({view.layers[layer].directed for layer in selected}) > 1
            if len(selected) == 1:
                result['directed'] = view.layers[selected[0]].directed
            if gene is not None:
                result['native_self_loop'] = any((gene, gene) in view.layers[layer].edges for layer in selected)
                result['present_selected_layers'] = [layer for layer in selected if gene in view.layers[layer].nodes]
                result['absent_selected_layers'] = [layer for layer in selected if gene not in view.layers[layer].nodes]
            nearest = self.nearest.get((json.dumps(scope, sort_keys=True), meta.get('direction'), meta.get('split')))
            result['nearest_below_95'] = nearest is not None and meta.get('degree') == nearest
        self.cache[identity] = result
        return result
