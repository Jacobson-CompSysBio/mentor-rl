"""Check that filtering writes whole original rows and preserves frozen inputs."""

import json

import pytest

from scripts import filter_world_model_s2_8k as filtering


def test_complete_candidate_pool_preserves_original_line_bytes_and_splits(tmp_path, monkeypatch):
    families, verified = [], {}
    expected_lines = {}
    for number in range(1, 14):
        family = f"S2.{number}"
        source = tmp_path / f"source_{number}"
        source.mkdir()
        (source / "manifest.json").write_text("{}\n")
        lengths, excluded, before = {}, [], {}
        for split in filtering.SPLITS:
            lines = []
            for suffix, tokens in (("exact", 8192), ("oversized", 8193)):
                row_id = f"{family}_{split}_{suffix}"
                row = {"row_id": row_id, "metadata": {"split": split, "question_family": family},
                       "messages": [{"role": "tool", "content": {"weight": .123456789}}]}
                raw = (json.dumps(row, separators=(", ", ": ")) + "\n").encode()
                lines.append(raw)
                lengths[row_id] = {"tokens": tokens}
                if tokens <= 8192:
                    expected_lines[(number, split)] = raw
                else:
                    excluded.append(row_id)
            before[split] = b"".join(lines)
            (source / f"{split}.jsonl").write_bytes(before[split])
        verified[source] = lengths
        families.append({"family": family, "path": source.name,
            "manifest_sha256": filtering.sha256(source / "manifest.json"),
            "splits": {split: {"retained": 1} for split in filtering.SPLITS},
            "excluded_row_ids": excluded})
    monkeypatch.setattr(filtering, "load_verified_lengths", lambda path, root, cache: ({}, verified[path]))
    report_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps({"max_length": 8192, "truncation_applied": False,
        "unmeasured_families": [], "families": families}))
    output = tmp_path / "eligible"
    summary = filtering.filter_candidates(report_path, output, root=tmp_path)
    assert summary["row_counts"] == {"train": 13, "validation": 13, "test": 13}
    assert summary["training_ready"] is False
    assert summary["original_row_bytes_preserved"] is True
    for number in range(1, 14):
        for split in filtering.SPLITS:
            assert (output / f"s2_{number}" / f"{split}.jsonl").read_bytes() == expected_lines[(number, split)]
            assert (tmp_path / f"source_{number}" / f"{split}.jsonl").read_bytes().count(b"\n") == 2
    with pytest.raises(FileExistsError):
        filtering.filter_candidates(report_path, output, root=tmp_path)


def test_unmeasured_families_cannot_be_silently_omitted(tmp_path):
    report_path = tmp_path / "audit.json"
    report_path.write_text(json.dumps({"max_length": 8192, "truncation_applied": False,
                                     "unmeasured_families": ["S2.3"]}))
    with pytest.raises(ValueError, match="all thirteen"):
        filtering.filter_candidates(report_path, tmp_path / "eligible", root=tmp_path)
    assert not (tmp_path / "eligible").exists()
