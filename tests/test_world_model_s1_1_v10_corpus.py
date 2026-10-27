"""Protect S1.1 v10's reasoning-only boundary and evidence-bound scope decisions."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from runtime.world_model import s1_1_corpus as symbols
from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import S1OverlayError, apply_s1_overlay, overlay_pin


OVERLAY = Path("runtime/world_model/s1_v10.yaml")
BASELINE = Path("data/world_model_v4/corpora/s1_v9/s1_1")


@pytest.fixture(scope="module")
def revision(tmp_path_factory):
    """Build the revision through normal checks while retaining its audit inputs."""

    contract, payload = apply_s1_overlay(load_contract(), OVERLAY)
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    root = tmp_path_factory.mktemp("s1_1_v10")
    captured = {}
    original = symbols.audit_rows

    def capture(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(symbols, "audit_rows", capture)
        manifest = symbols.build_corpus(
            source, root, contract=contract,
            overlay=overlay_pin(OVERLAY, payload), prompt_banks=payload["prompt_banks"],
        )
    return {"root": root, "manifest": manifest, "contract": contract,
            "payload": payload, "source": source, **captured}


def test_only_800_id_confirmation_final_thoughts_change(revision):
    """Preserve every prompt, identity, call, answer, row ID, and other form."""

    templates = revision["contract"]["curriculum"]["families"]["S1.1"]["train_id_confirmation_reasoning"]
    changes = Counter()
    for old_raw, new_raw in zip(
        (BASELINE / "train.jsonl").read_bytes().splitlines(),
        (revision["root"] / "train.jsonl").read_bytes().splitlines(), strict=True,
    ):
        old, new = json.loads(old_raw), json.loads(new_raw)
        if old["metadata"]["form"] != 6:
            assert old_raw == new_raw
            continue
        final = old["messages"][-1]["content"]
        framing = "step" if final["continuation_decision"] == "continue" else "direct"
        expected = deepcopy(old)
        lookup = old["messages"][-2]["content"]
        expected["messages"][-1]["thinking"] = templates[framing].format(
            gene_id=lookup["gene_id"], symbol=lookup["gene_symbol"],
            user_request=old["messages"][1]["content"],
        )
        assert new == expected
        assert old_raw != new_raw
        assert final["status"] == "resolved"
        assert final["result"] == {key: lookup[key] for key in ("gene_id", "gene_symbol")}
        assert final["provenance"] == lookup["provenance"]
        changes[framing] += 1
    assert changes == {"direct": 400, "step": 400}
    assert revision["manifest"]["train_id_confirmation_reasoning"] == templates
    assert revision["manifest"]["row_counts"] == {"train": 10000, "validation": 1250, "test": 1250}


def test_heldout_bytes_are_frozen(revision):
    """The new scope reasoning never changes validation or test content."""

    for split in ("validation", "test"):
        assert (revision["root"] / f"{split}.jsonl").read_bytes() == (
            BASELINE / f"{split}.jsonl"
        ).read_bytes()


@pytest.mark.parametrize("field", ["thinking", "user_request", "continuation_decision", "gene_id", "provenance"])
def test_audit_rejects_scope_or_evidence_tampering(revision, field):
    """The exact thought and the structured decision must retain their evidence."""

    rows, scenarios, counts = revision["args"]
    index = next(i for i, case in enumerate(scenarios["train"])
                 if case.form == 6 and case.framing == "step")
    row = deepcopy(rows["train"][index])
    final = row["messages"][-1]
    if field == "thinking":
        final["thinking"] = final["thinking"].replace("then continue.", "then stop.")
    elif field == "user_request":
        row["messages"][1]["content"] = "Confirm this ID as the whole request."
    elif field == "gene_id":
        final["content"]["result"]["gene_id"] = "ENSG00000000000"
    elif field == "provenance":
        final["content"]["provenance"] = None
    else:
        final["content"]["continuation_decision"] = "stop"
    changed = {**rows, "train": list(rows["train"])}
    changed["train"][index] = row
    with pytest.raises(symbols.CorpusBuildError, match="ID confirmation"):
        symbols.audit_rows(changed, scenarios, counts, **revision["kwargs"])


@pytest.mark.parametrize("version", [8, 9])
def test_historical_builds_remain_byte_identical(version, tmp_path):
    """An absent v10 setting retains the complete published trained corpus."""

    contract, payload = apply_s1_overlay(
        load_contract(), Path(f"runtime/world_model/s1_v{version}.yaml"),
    )
    source = Path(contract["source_index"]["output_directory"]) / (
        f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    )
    symbols.build_corpus(source, tmp_path, contract=contract, prompt_banks=payload["prompt_banks"])
    baseline = Path(f"data/world_model_v4/corpora/s1_v{version}/s1_1")
    for split in ("train", "validation", "test"):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (baseline / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize("replacement", [
    "Return {gene_id} with symbol {symbol} and continue.",
    "The request {user_request} resolves {gene_id} as {symbol.name}.",
    "The request {user_request} resolves {gene_id} as {symbol} in {multiplex_id}.",
])
def test_overlay_requires_exact_visible_bindings(tmp_path, replacement):
    """Reject unbound scope, attribute access, and invented evidence slots."""

    payload = yaml.safe_load(OVERLAY.read_text())
    payload["family_overrides"]["S1.1"]["train_id_confirmation_reasoning"]["step"] = replacement
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(S1OverlayError, match="ID-confirmation"):
        apply_s1_overlay(load_contract(), path)
