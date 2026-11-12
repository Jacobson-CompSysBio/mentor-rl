"""Adapt integer family identifiers to the frozen S2.7–S2.10 generator API."""

from . import s2_production_cases_7_10 as original


INPUT_PATHS = original.INPUT_PATHS
CASE_IDS = original.CASE_IDS


def prepare_context(context, family):
    """Accept the production CLI's numeric family without changing source logic."""
    return original.prepare_context(context, f'S2.{family}' if isinstance(family, int) else family)


def production_candidates(case_id, context, *, split):
    """Retain the frozen generator's complete source and wording candidates."""
    yield from original.production_candidates(case_id, context, split=split)


def validate_row(row, context):
    """Use the unchanged family source and visible-evidence validator."""
    return original.validate_row(row, context)
