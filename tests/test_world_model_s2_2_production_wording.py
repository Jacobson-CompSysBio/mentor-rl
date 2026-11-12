"""Check diagnostic prose separation after masking literal structured inputs."""

from copy import deepcopy

import pytest

from runtime.world_model.s1_prompt_banks import PromptBankError
from runtime.world_model.s2_2_production import load_build_contract
from runtime.world_model.s2_2_production_wording import audit_diagnostic_wording


def test_new_diagnostic_leads_are_separate_from_every_frozen_s2_family():
    report = audit_diagnostic_wording()
    assert report['cross_split_pairs_checked'] > 1_000_000
    assert report['shared_cross_split_5grams'] == 0
    assert report['maximum_cross_split_jaccard'] < .6
    assert len(report['request_bank_pins']) == 13
    assert report['production_wording_expansion_still_required']


def test_masking_structured_json_does_not_hide_a_copied_natural_language_lead():
    contract = deepcopy(load_build_contract())
    contract['_diagnostic_leads']['train'] = contract['_diagnostic_leads']['test']
    with pytest.raises(PromptBankError, match='5-word run'):
        audit_diagnostic_wording(contract)
