"""Audit new diagnostic prose against every frozen S2 request-bank split.

The literal JSON request is structured input, like the bound genes and scopes
removed by existing audits. Mask that entire payload while retaining the new
natural-language lead and neutral wrapper; otherwise shared JSON field names
would be mistaken for held-out prose leakage.
"""

from __future__ import annotations

import json
from typing import Mapping

from . import s2_13_prompts as prompts
from .s1_prompt_banks import (
    HELD_OUT_JACCARD_LIMIT, HELD_OUT_SHARED_NGRAM, PromptBankError,
    _ngrams, _tokens, load_s2_prompt_banks,
)
from .s2_2_production import ROOT, SPLITS, load_build_contract
from .sources import file_sha256


def audit_diagnostic_wording(contract: Mapping | None = None, bank: Mapping | None = None) -> dict:
    """Check all wrapper/lead combinations, masking only their bound JSON payload."""
    contract = load_build_contract() if contract is None else contract
    bank = load_s2_prompt_banks(contract['prompt_bank']) if bank is None else bank
    frozen_path = ROOT / 'runtime/world_model/s2_13_v1.json'
    frozen = json.loads(frozen_path.read_text())
    renderers = (prompts._rendered_variants, prompts._rendered_variants,
        prompts._neighbor_variants, prompts._map_variants, prompts._membership_variants,
        prompts._classification_variants, prompts._shared_variants, prompts._degree_variants,
        prompts._subgraph_variants, prompts._component_variants, prompts._path_variants,
        prompts._singleton_variants, prompts._specificity_variants)
    prior = {split: [] for split in SPLITS}
    source_pins = []
    for number, renderer in enumerate(renderers, 1):
        pin = frozen['request_bank' if number == 13 else f's2{number}_request_bank']
        path = ROOT / pin['path']
        if file_sha256(path) != pin['sha256']:
            raise ValueError('a frozen S2 request bank changed')
        source_pins.append(dict(pin))
        requests = json.loads(path.read_text())
        variants = renderer(bank, requests, f'S2.{number}') if number <= 2 else renderer(bank, requests)
        for split in SPLITS:
            prior[split].extend(variants[split])
    current = {}
    for split in SPLITS:
        current[split] = []
        for index, wrapper in enumerate(bank[split]['direct']):
            tokens = _tokens(wrapper.format(request=contract['_diagnostic_leads'][split].strip()))
            current[split].append((f'S2.2.production/{split}/direct/{index}',
                frozenset(tokens), _ngrams(tokens, HELD_OUT_SHARED_NGRAM)))
    vocabulary = {word for rows in (*prior.values(), *current.values()) for _, words, _ in rows for word in words}
    bits = {word: 1 << index for index, word in enumerate(sorted(vocabulary))}

    def packed(rows):
        return {(label, sum(bits[word] for word in words), len(words)) for label, words, _ in rows}

    maximum, compared = 0.0, 0
    for split in SPLITS:
        for other in SPLITS:
            if split == other:
                continue
            # Compare new prose to all prior families in other splits, and to
            # the new held-out leads. Rechecking historical-only pairs is not
            # necessary: their existing thirteen-family audit remains frozen.
            reference = prior[other] + current[other]
            grams = {gram: label for label, _, runs in reference for gram in runs}
            for label, _, runs in current[split]:
                shared = runs & grams.keys()
                if shared:
                    run = min(shared)
                    raise PromptBankError(f'new S2.2 diagnostic shares a held-out 5-word run: {label}, {grams[run]}: {" ".join(run)}')
            for label, words, size in packed(current[split]):
                for other_label, other_words, other_size in packed(reference):
                    compared += 1
                    common = (words & other_words).bit_count()
                    overlap = common / (size + other_size - common)
                    maximum = max(maximum, overlap)
                    if overlap >= HELD_OUT_JACCARD_LIMIT:
                        raise PromptBankError(f'new S2.2 diagnostic overlaps held-out wording: {label}, {other_label}: {overlap:.4f}')
    return {'scope': 'all new S2.2 diagnostic lead/direct-wrapper combinations against all thirteen frozen S2 families across splits',
        'masking': 'remove the complete literal structured JSON request; retain all natural-language lead and wrapper tokens',
        'new_wordings': {split: len(rows) for split, rows in current.items()},
        'prior_wordings': {split: len(rows) for split, rows in prior.items()},
        'cross_split_pairs_checked': compared, 'maximum_cross_split_jaccard': round(maximum, 6),
        'shared_cross_split_5grams': 0, 'request_bank_pins': source_pins,
        'frozen_pin_catalog_sha256': file_sha256(frozen_path),
        'production_wording_expansion_still_required': True}
