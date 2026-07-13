from __future__ import annotations

import unittest

from scripts.build_pretrajectory_sft_curriculum import (
    CurriculumBuilder,
    CurriculumExample,
    Family,
)


class CurriculumFactGroupingTests(unittest.TestCase):
    def test_conflicting_strong_groups_coalesce_before_split_assignment(self) -> None:
        family = Family(
            id=1,
            name="fixture_family",
            primary_stage="stage1_entity_schema",
            mixture_bucket="entity_normalization_schema",
            allowed_book_modes=("closed_book",),
            difficulty_source="fixture",
        )
        common = {
            "family": family,
            "book_mode": "closed_book",
            "task": "Return JSON.",
            "answer": {"value": 1},
            "evidence": None,
            "fact_payload": {"value": 1},
        }
        first = CurriculumExample(**common, strongest_group_id="entity:a")
        second = CurriculumExample(**common, strongest_group_id="entity:b")
        builder = CurriculumBuilder.__new__(CurriculumBuilder)
        builder.examples = [first, second]
        builder._effective_fact_groups = {}
        builder.coalesced_fact_group_count = 0

        count = builder.coalesce_oracle_fact_groups()

        self.assertEqual(count, 1)
        self.assertEqual(
            builder._effective_fact_groups[first.oracle_fact_id],
            f"oracle_fact_group:{first.oracle_fact_id}",
        )
        self.assertEqual(first.oracle_fact_id, second.oracle_fact_id)

    def test_unambiguous_strong_group_is_preserved(self) -> None:
        family = Family(
            id=1,
            name="fixture_family",
            primary_stage="stage1_entity_schema",
            mixture_bucket="entity_normalization_schema",
            allowed_book_modes=("closed_book",),
            difficulty_source="fixture",
        )
        example = CurriculumExample(
            family=family,
            book_mode="closed_book",
            task="Return JSON.",
            answer={"value": 1},
            evidence=None,
            fact_payload={"value": 1},
            strongest_group_id="entity:a",
        )
        builder = CurriculumBuilder.__new__(CurriculumBuilder)
        builder.examples = [example]
        builder._effective_fact_groups = {}
        builder.coalesced_fact_group_count = 0

        self.assertEqual(builder.coalesce_oracle_fact_groups(), 0)
        self.assertEqual(
            builder._effective_fact_groups[example.oracle_fact_id], "entity:a"
        )


if __name__ == "__main__":
    unittest.main()
