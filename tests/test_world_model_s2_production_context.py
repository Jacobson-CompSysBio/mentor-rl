"""Keep CLI family identifiers and exhaustive degree support bound correctly."""

import pytest

from runtime.world_model import s2_production_context_7_10 as compatibility
from scripts import build_world_model_s2_production_v2 as builder


@pytest.mark.parametrize('family,expected', [(7, 'S2.7'), (8, 'S2.8'), ('S2.9', 'S2.9')])
def test_integer_cli_families_reach_the_frozen_family_api(monkeypatch, family, expected):
    calls = []
    context = {'source': object()}

    def prepare(selected, selected_family):
        calls.append((selected, selected_family))
        return selected

    monkeypatch.setattr(compatibility.original, 'prepare_context', prepare)
    assert compatibility.prepare_context(context, family) is context
    assert calls == [(context, expected)]


def test_exhaustive_degree_rank_support_survives_cli_context_construction(monkeypatch):
    generic, exhaustive = object(), object()
    context = {'features': generic, '_features': exhaustive}
    monkeypatch.setattr(builder, '_previous_context', lambda *_args, **_kwargs: ('degree_module', context))
    module, prepared = builder.build_context(8, {}, {}, {})
    assert module == 'degree_module'
    assert prepared['features'] is exhaustive


def test_other_family_features_are_preserved(monkeypatch):
    features = object()
    context = {'features': features}
    monkeypatch.setattr(builder, '_previous_context', lambda *_args, **_kwargs: ('edge_module', context))
    assert builder.build_context(1, {}, {}, {})[1]['features'] is features
