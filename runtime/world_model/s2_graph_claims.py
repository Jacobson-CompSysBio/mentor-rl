"""Build closed graph-observation cards in the approved shared order."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


CLAIM_VOCABULARY = (
    'no_edge_recorded_in_scope', 'no_path_recorded_in_scope',
    'gene_not_node_in_multiplex', 'gene_not_node_in_layer', 'not_in_shown_top_k',
    'absence_holds_in_other_layers', 'absence_holds_in_other_versions',
    'genes_unrelated_in_all_contexts', 'no_interaction_in_biology',
    'relationship_experimentally_disproven', 'gene_not_expressed_in_context',
)


def claim_card(arguments: Mapping[str, Any], gene_ids: Sequence[str], supported: Sequence[str],
               *, claim: str | None = None, statement_true: bool | None = None) -> dict:
    """Partition the vocabulary and add only explicitly requested assessments."""
    permitted = set(CLAIM_VOCABULARY)
    if not set(supported) <= permitted or claim is not None and claim not in permitted:
        raise ValueError('a graph card requires approved canonical claims')
    if statement_true is not None and (type(statement_true) is not bool or claim is None):
        raise ValueError('an explicit truth verdict requires a canonical assessed claim')
    card = {'subject': {'gene_ids': list(gene_ids),
        **{key: deepcopy(arguments[key]) for key in ('multiplex_id', 'layer_id', 'direction')}},
        'claim_vocabulary': 's2_graph_observations_v1',
        'supported_claims': [name for name in CLAIM_VOCABULARY if name in supported],
        'unsupported_claims': [name for name in CLAIM_VOCABULARY if name not in supported]}
    if claim is not None:
        card.update(claim=claim, claim_supported=claim in supported)
    if statement_true is not None:
        card['statement_true'] = statement_true
    return card
