"""Apply the approved production budget to complete, untruncated S2 traces.

Length eligibility is separate from semantic coverage. A missing generator or
an underfilled source population must never be called an overlength behavior.
Historical artifacts stay immutable; all selection uses their original splits.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


RECIPE_PATH = Path(__file__).with_name("s2_production_v1_8k.json")
SPLITS = ("train", "validation", "test")


def load_recipe(path: Path = RECIPE_PATH) -> dict[str, Any]:
    """Load the approved 8k budget without modifying historical contracts."""
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema_version") != "mentor-rl-s2-production-recipe-v1":
        raise ValueError("unsupported production recipe")
    if (recipe.get("max_length") != 8192
            or recipe.get("length_unit") != "complete_rendered_trace"
            or recipe.get("truncation_allowed") is not False):
        raise ValueError("production requires complete traces at the approved 8192 limit")
    if recipe.get("per_retained_case") != dict(train=512, validation=64, test=64):
        raise ValueError("the approved question-case quotas changed")
    if recipe.get("training_mix") != dict(S2=.8, S1=.15, S0=.05):
        raise ValueError("the approved stage mixture changed")
    if recipe.get("replay_rows_per_retained_s2_case") != dict(S1=96, S0=32):
        raise ValueError("replay counts do not implement the approved mixture")
    return recipe


def production_budget(retained_cases: int, recipe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Recompute all totals after explicit, justified case exclusions."""
    if type(retained_cases) is not int or retained_cases < 0:
        raise ValueError("retained_cases must be a nonnegative integer")
    recipe = load_recipe() if recipe is None else recipe
    s2 = {split: retained_cases * recipe["per_retained_case"][split] for split in SPLITS}
    replay = {stage: retained_cases * count
              for stage, count in recipe["replay_rows_per_retained_s2_case"].items()}
    return {"retained_cases": retained_cases, "s2": s2, "replay": replay,
            "train": s2["train"] + sum(replay.values()),
            "validation": s2["validation"], "test": s2["test"]}


def partition_by_length(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = 8192,
    encode: Callable | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Keep fitting original rows and report full-wire exclusions without clipping.

    The default encoder measures system and tool definitions, supplied context,
    every current-turn reasoning segment, calls, tool responses, and the final
    answer. The injectable encoder supports focused boundary tests.
    """
    if type(max_length) is not int or max_length != 8192:
        raise ValueError("the production context limit is 8192")
    if encode is None:
        from .s2_3_tokens import tokenize_trace
        encode = tokenize_trace
    kept, excluded, seen = [], [], set()
    counts, by_split = [], Counter()
    for row in rows:
        row_id = row.get("row_id")
        metadata = row.get("metadata", {})
        split = metadata.get("split")
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError("length selection requires unique nonempty row IDs")
        if split not in SPLITS:
            raise ValueError("length selection must preserve an explicit source split")
        seen.add(row_id)
        encoded = encode(row, tokenizer)
        ids, mask = encoded["input_ids"], encoded["completion_mask"]
        if not ids or len(ids) != len(mask) or not any(mask) or any(x not in (0, 1) for x in mask):
            raise ValueError("invalid complete-trace supervision mask")
        count = len(ids)
        record = {"row_id": row_id, "split": split, "tokens": count,
                  "assistant_tokens": sum(mask), "eligible": count <= max_length}
        counts.append(record)
        by_split[(split, "eligible" if record["eligible"] else "excluded")] += 1
        if record["eligible"]:
            kept.append(row)
        else:
            excluded.append(record)
    return kept, {
        "max_length": max_length, "length_unit": "complete_rendered_trace",
        "truncation_applied": False, "rows": len(counts), "kept": len(kept),
        "excluded": excluded, "row_counts": counts,
        "max_retained_tokens": max((r["tokens"] for r in counts if r["eligible"]), default=0),
        "by_split": {split: {key: by_split[(split, key)] for key in ("eligible", "excluded")}
                     for split in SPLITS},
    }
