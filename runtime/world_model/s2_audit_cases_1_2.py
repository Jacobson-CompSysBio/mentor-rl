"""Use the repaired early generators' explicit, independently checked case IDs.

Historical S2.1/2 verification versions need regeneration under the repaired
contracts. Their old token counts cannot qualify rows of the selected versions.
"""

from dataclasses import asdict

from .s2_1_production import CASE_BINDINGS as EDGE_BINDINGS
from .s2_2_production import CASE_BINDINGS as MULTIPLEX_BINDINGS


CASE_BINDINGS = {
    **{case_id: {'generator': 'runtime.world_model.s2_1_production.build_case_rows',
                'variants': (binding['behavior'],),
                'requirements': f"Native predicate {binding['source_predicate']}; {binding['behavior']} evidence; {binding['framing']} framing.",
                'generator_parameters': dict(binding)} for case_id, binding in EDGE_BINDINGS.items()},
    **{case_id: {'generator': 'runtime.world_model.s2_2_production.build_case_rows',
                'variants': (('full_context',) if case_id == 'S2.2.8' else
                             ('direction_invalid',) if case_id == 'S2.2.19' else (binding.variant,)),
                'requirements': f"Native condition {binding.condition}; {binding.scope} scope; {binding.query}; {binding.mode}; {binding.variant} evidence; {binding.framing} framing.",
                'generator_parameters': asdict(binding)} for case_id, binding in MULTIPLEX_BINDINGS.items()},
}


def matches(case_id, row):
    """Require the selected repaired contract's explicit case and behavior label."""
    meta = row['metadata']
    if meta.get('catalog_case_id') != case_id:
        return False
    if case_id.startswith('S2.1.'):
        binding = EDGE_BINDINGS[case_id]
        return (meta.get('variant') == binding['behavior'] and meta.get('source_predicate') == binding['source_predicate']
                and meta.get('framing') == binding['framing'])
    binding = MULTIPLEX_BINDINGS[case_id]
    return meta.get('production_behavior') == binding.variant and meta.get('framing') == binding.framing
