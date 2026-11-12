"""Audit production request extensions without editing frozen corpus generators."""

from copy import deepcopy
import importlib
import json

from .s2_corpus import ROOT
from .sources import file_sha256
from .s2_production_cases_11_13 import REQUEST_BANK


def audit_wording(bank):
    """Compare the new request/wrapper product against every frozen S2 bank."""
    from . import s2_13_prompts as prompts
    from .s1_prompt_banks import _tokens, _ngrams, PromptBankError
    from .s2_4_case_repairs import REQUEST_BANK as ranking_pin
    splits = ('train', 'validation', 'test')
    catalog = json.loads((ROOT / 'runtime/world_model/s2_13_v1.json').read_text())
    renderers = (prompts._rendered_variants, prompts._rendered_variants,
        prompts._neighbor_variants, prompts._map_variants, prompts._membership_variants,
        prompts._classification_variants, prompts._shared_variants, prompts._degree_variants,
        prompts._subgraph_variants, prompts._component_variants, prompts._path_variants,
        prompts._singleton_variants, prompts._specificity_variants)
    prior, pins = {split: [] for split in splits}, []
    for number, renderer in enumerate(renderers, 1):
        pin = catalog['request_bank' if number == 13 else f's2{number}_request_bank']
        if file_sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError('a frozen wording bank changed')
        requests = json.loads((ROOT / pin['path']).read_text())
        rendered = renderer(bank, requests, f'S2.{number}') if number < 3 else renderer(bank, requests)
        for split in splits:
            prior[split].extend(rendered[split])
        pins.append(dict(pin))
    for family in (1, 2):
        path = ROOT / f'runtime/world_model/s2_{family}_production_v1.json'
        production = json.loads(path.read_text())
        if family == 1:
            pin = production['request_bank']
            requests = json.loads((ROOT / pin['path']).read_text())
        else:
            pin = {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)}
            requests = production['diagnostic_leads']
        pins.append(dict(pin))
        for split in splits:
            cores = ([template.format(first='', second='', layer='', multiplex='')
                      for templates in requests[split].values() for template in templates]
                     if family == 1 else [requests[split].strip()])
            for framing in ('direct', 'step'):
                for wrapper in bank[split][framing]:
                    for core in cores:
                        words = _tokens(wrapper.format(request=core))
                        prior[split].append(('production', frozenset(words), _ngrams(words, 5)))
    requests = json.loads((ROOT / ranking_pin['path']).read_text())
    pins.append(dict(ranking_pin))
    for split in splits:
        for wrapper in bank[split]['direct']:
            for template in requests[split]:
                words = _tokens(wrapper.format(request=template.format(gene='', multiplex_id='', scope_text='', top_k='')))
                prior[split].append(('bounded-ranking', frozenset(words), _ngrams(words, 5)))
    current = prompts._specificity_variants(bank, json.loads((ROOT / REQUEST_BANK['path']).read_text()))
    new_pins = [dict(REQUEST_BANK)]

    def include_new(rendered, pin):
        for selected_split in splits:
            current[selected_split].extend(rendered[selected_split])
        new_pins.append(pin)

    for number, renderer in ((3, prompts._neighbor_variants), (11, prompts._path_variants),
                             (12, prompts._singleton_variants)):
        path = ROOT / f'runtime/world_model/s2_production_{number}_requests_v1.json'
        requests = json.loads(path.read_text())
        validator = importlib.import_module(f'runtime.world_model.s2_{number}_prompts')
        getattr(validator, f'validate_s2_{number}_requests')(requests)
        include_new(renderer(bank, requests), {'path': str(path.relative_to(ROOT)), 'sha256': file_sha256(path)})
    degree_path = ROOT / 'runtime/world_model/s2_production_8_requests_v1.json'
    degree_additions = json.loads(degree_path.read_text())
    from .s2_8_corpus import load_build_contract as degree_contract
    requests = deepcopy(degree_contract()['_requests'])
    for split in splits:
        for kind in ('degree_only', 'degree_and_hub'):
            requests[split][kind].extend(degree_additions['s2_8'][split])
        for kind in ('invalid_direction', 'unknown_layer'):
            requests['diagnostic_requests'][split][kind].extend(degree_additions['s2_8'][split])
    include_new(prompts._degree_variants(bank, requests),
                {'path': str(degree_path.relative_to(ROOT)), 'sha256': file_sha256(degree_path)})
    additions_path = ROOT / 'runtime/world_model/s2_production_requests_9_10_v1.json'
    if additions_path.exists():
        additions = json.loads(additions_path.read_text())
        pins.append({'path': str(additions_path.relative_to(ROOT)), 'sha256': file_sha256(additions_path)})
        for number, renderer in ((9, prompts._subgraph_variants), (10, prompts._component_variants)):
            base = importlib.import_module(f'runtime.world_model.s2_{number}_corpus')
            requests = deepcopy(base.load_build_contract()['_requests'])
            for split in splits:
                if number == 9:
                    for kind in ('complete', 'top_k'):
                        requests[split][kind].extend(additions[f's2_{number}'][split])
                else:
                    requests[split]['partition'].extend(additions[f's2_{number}'][split])
                    requests['diagnostic_requests'][split]['unavailable_kind'].extend(additions[f's2_{number}'][split])
            rendered = renderer(bank, requests)
            include_new(rendered, {'path': str(additions_path.relative_to(ROOT)), 'sha256': file_sha256(additions_path)})
    vocabulary = sorted({word for rows in (*prior.values(), *current.values()) for _, words, _ in rows for word in words})
    import numpy as np
    from scipy.sparse import csr_matrix
    columns = {word: index for index, word in enumerate(vocabulary)}

    def packed(rows):
        # Sparse integer products count set intersections exactly. Chunking the
        # resulting dense blocks bounds audit memory independently of bank size.
        words = sorted({tuple(sorted(words)) for _, words, _ in rows})
        indices, pointers = [], [0]
        for row in words:
            indices.extend(columns[word] for word in row)
            pointers.append(len(indices))
        return (csr_matrix((np.ones(len(indices), dtype=np.int16), indices, pointers),
                           shape=(len(words), len(vocabulary))), np.array([len(row) for row in words]))

    maximum, compared = 0., 0
    for split in splits:
        for other in splits:
            if split == other:
                continue
            reference = prior[other] + current[other]
            grams = {gram: label for label, _, runs in reference for gram in runs}
            for label, _, runs in current[split]:
                shared = runs & grams.keys()
                if shared:
                    gram = min(shared)
                    raise PromptBankError(f'new wording shares a held-out five-word run: {split}/{other}; '
                                          f'{label}; {grams[gram]}; {gram}')
            compared += len(current[split]) * len(reference)
            left, sizes = packed(current[split])
            right, other_sizes = packed(reference)
            for start in range(0, left.shape[0], 128):
                common = (left[start:start + 128] @ right.T).toarray()
                unions = sizes[start:start + 128, None] + other_sizes[None, :] - common
                value = float((common / unions).max())
                maximum = max(maximum, value)
                if value >= .6:
                    raise PromptBankError(f'new specificity wording overlaps held-out wording: {value:.6f}')
    return {'cross_split_comparisons': compared, 'maximum_cross_split_jaccard': maximum,
            'shared_cross_split_5grams': 0, 'request_bank': dict(REQUEST_BANK), 'reference_pins': pins,
            'new_request_banks': list({pin['path']: pin for pin in new_pins}.values()),
            'rendered_wordings': {split: len(rows) for split, rows in current.items()}}
