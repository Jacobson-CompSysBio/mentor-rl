"""Run this file to inspect the tokenizer audit reports for s0: plain, domain-bpe, and atomic+domain-bpe tokenization."""
import json
from pathlib import Path

root = Path("data/world_model_v2/sft/s0_human_identifier_tokenizers_v4")
pair = json.loads((root / "pair_manifest.json").read_text())

for method in pair["methods"]:
    audit = json.loads((root / method / "audit_report.json").read_text())
    assert audit["passed"] is True
    print(f"Method: {method} | Audit PASSED")
    assert audit["full_corpus_tokenized_rows"] == 85284
    assert audit["fit_value_round_trip_failures"] == 0
    assert audit["full_corpus_row_failures"] == 0
    assert audit["representation_failures"] == 0
    assert 0 < audit["maximum_sequence_tokens"] <= 1024
    print(method, audit["unused_model_rows_consumed"], "rows used")
