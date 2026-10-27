"""Build the canonical S1.8 direction-conventions corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from . import s1_6_corpus as s1_6
from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from .s1_overlay import apply_s1_overlay, overlay_pin
from .s1_prompt_banks import (
    load_prompt_banks,
    metric_names,
    prompt_bank_manifest,
    validate_metric_names,
    wrapper_prompts,
)
from .sources import file_sha256


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-8-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-8-manifest-v1"
CONTRACT_S1_8_SHA256 = (
    "966fdf1b4f5723d428910ee0641ff9ecb25570c7b8525dcb77a1449fb3059a29"
)
SPLITS = ("train", "validation", "test")

CorpusBuildError = s1_6.CorpusBuildError
split_row_counts = s1_6.split_row_counts

TRACE_BY_BUCKET = {
    "direction_lookup": 1,
    "true_false_distance_style": 2,
    "true_false_similarity_style": 3,
    "true_false_workflow_style": 4,
    "ambiguous_percentile": 5,
    "unrecognized_metric": 6,
}
BUCKET_BY_TRACE = {value: key for key, value in TRACE_BY_BUCKET.items()}
SEMANTIC_BUCKET_BY_FORM = {
    1: "direction_lookup",
    2: "true_false",
    3: "true_false",
    4: "true_false",
    5: "ambiguous_metric",
    6: "unrecognized_metric",
}

METRIC_CONVENTIONS = {
    "rank": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_rank_closer",
    },
    "score": {
        "lower_is_stronger": False,
        "meaning": "proximity",
        "rule": "higher_score_closer",
    },
    "spearman_distance": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_closer",
    },
    "pearson_distance": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_closer",
    },
    "spearman_correlation": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_correlation_more_similar",
    },
    "pearson_correlation": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_correlation_more_similar",
    },
    "dot_similarity": {
        "lower_is_stronger": False,
        "meaning": "similarity",
        "rule": "higher_similarity_more_similar",
    },
    "distance_percentile": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "lower_distance_percentile_closer",
    },
    "score_percentile": {
        "lower_is_stronger": False,
        "meaning": "proximity",
        "rule": "higher_score_percentile_closer",
    },
    "hop_count": {
        "lower_is_stronger": True,
        "meaning": "proximity",
        "rule": "fewer_hops_closer",
    },
    "empirical_p_value": {
        "lower_is_stronger": True,
        "meaning": "evidence",
        "rule": "lower_p_stronger_evidence",
    },
    "q": {
        "lower_is_stronger": True,
        "meaning": "evidence",
        "rule": "lower_p_stronger_evidence",
    },
    "enrichment_score": {
        "lower_is_stronger": False,
        "meaning": "evidence",
        "rule": "higher_enrichment_stronger_evidence",
    },
}
METRICS = tuple(METRIC_CONVENTIONS)

NAME_VARIANTS = {
    "train": {
        "rank": (
            "RWR rank",
            "random walk with restart rank",
            "RWR-LOE rank",
        ),
        "score": ("RWR score", "restart score", "visiting probability"),
        "spearman_distance": (
            "Spearman distance",
            "1 minus Spearman correlation",
            "Spearman rho distance",
        ),
        "pearson_distance": (
            "Pearson distance",
            "1 minus Pearson correlation",
            "Pearson r distance",
        ),
        "spearman_correlation": (
            "Spearman correlation",
            "Spearman rho",
            "rank correlation coefficient",
        ),
        "pearson_correlation": (
            "Pearson correlation",
            "Pearson r",
            "linear correlation coefficient",
        ),
        "dot_similarity": (
            "dot similarity",
            "RWR dot product",
            "vector dot similarity",
        ),
        "distance_percentile": (
            "distance percentile",
            "distance-based percentile",
            "percentile of distance",
        ),
        "score_percentile": (
            "score percentile",
            "score-based percentile",
            "percentile of score",
        ),
        "hop_count": ("path length", "number of hops", "graph hop count"),
        "empirical_p_value": (
            "p-value",
            "empirical p value",
            "permutation p-value",
        ),
        "q": ("FDR q-value", "q-value", "false discovery rate q value"),
        "enrichment_score": (
            "enrichment score",
            "pathway enrichment score",
            "set enrichment score",
        ),
    },
    "validation": {
        "rank": ("restart-walk rank", "rank from RWR", "seed-proximity rank"),
        "score": (
            "restart-walk score",
            "RWR visiting score",
            "walk visit probability",
        ),
        "spearman_distance": (
            "rank-correlation distance",
            "one minus Spearman rho",
            "Spearman dissimilarity",
        ),
        "pearson_distance": (
            "linear-correlation distance",
            "one minus Pearson r",
            "Pearson dissimilarity",
        ),
        "spearman_correlation": (
            "Spearman association coefficient",
            "rho rank correlation",
            "monotonic correlation coefficient",
        ),
        "pearson_correlation": (
            "Pearson association coefficient",
            "r linear correlation",
            "product-moment correlation",
        ),
        "dot_similarity": (
            "dot-product similarity",
            "inner-product similarity",
            "RWR vector dot product",
        ),
        "distance_percentile": (
            "percentile by distance",
            "distance rank percentile",
            "distance distribution percentile",
        ),
        "score_percentile": (
            "percentile by score",
            "score rank percentile",
            "score distribution percentile",
        ),
        "hop_count": (
            "edge count along the path",
            "shortest-path hop count",
            "path hop total",
        ),
        "empirical_p_value": (
            "empirical significance probability",
            "randomization p-value",
            "permutation-tail probability",
        ),
        "q": (
            "FDR-adjusted p-value",
            "false-discovery q-value",
            "adjusted q statistic",
        ),
        "enrichment_score": (
            "gene-set enrichment score",
            "overrepresentation enrichment score",
            "enrichment strength score",
        ),
    },
    "test": {
        "rank": ("random-walk rank", "restart proximity rank", "RWR ordering rank"),
        "score": (
            "random-walk restart score",
            "seed-relative visit probability",
            "walk arrival score",
        ),
        "spearman_distance": (
            "Spearman rank distance",
            "one minus rank correlation",
            "rho-based distance",
        ),
        "pearson_distance": (
            "Pearson linear distance",
            "one minus linear correlation",
            "r-based distance",
        ),
        "spearman_correlation": (
            "Spearman rank correlation",
            "rho similarity coefficient",
            "ordinal correlation coefficient",
        ),
        "pearson_correlation": (
            "Pearson linear correlation",
            "r similarity coefficient",
            "product moment coefficient",
        ),
        "dot_similarity": (
            "dot-product score",
            "vector inner product",
            "RWR inner product",
        ),
        "distance_percentile": (
            "percentile-ranked distance",
            "distance value percentile",
            "relative distance percentile",
        ),
        "score_percentile": (
            "percentile-ranked score",
            "score value percentile",
            "relative score percentile",
        ),
        "hop_count": ("hop distance", "path edge count", "shortest-path length"),
        "empirical_p_value": (
            "empirical probability value",
            "null-sample p-value",
            "simulation p-value",
        ),
        "q": (
            "false-discovery-rate value",
            "multiple-test q-value",
            "FDR-adjusted significance",
        ),
        "enrichment_score": (
            "functional enrichment score",
            "set-overlap enrichment score",
            "overrepresentation score",
        ),
    },
}

SHARED_ALIASES = {
    "train": {
        "distance": {
            "candidate_metrics": ("spearman_distance", "pearson_distance"),
            "lower_is_stronger": True,
            "meaning": "proximity",
            "rule": "lower_distance_closer",
        },
        "correlation": {
            "candidate_metrics": (
                "spearman_correlation",
                "pearson_correlation",
            ),
            "lower_is_stronger": False,
            "meaning": "similarity",
            "rule": "higher_correlation_more_similar",
        },
    },
    "validation": {
        "correlation-derived distance": {
            "candidate_metrics": ("spearman_distance", "pearson_distance"),
            "lower_is_stronger": True,
            "meaning": "proximity",
            "rule": "lower_distance_closer",
        },
        "correlation coefficient": {
            "candidate_metrics": (
                "spearman_correlation",
                "pearson_correlation",
            ),
            "lower_is_stronger": False,
            "meaning": "similarity",
            "rule": "higher_correlation_more_similar",
        },
    },
    "test": {
        "correlation-based dissimilarity": {
            "candidate_metrics": ("spearman_distance", "pearson_distance"),
            "lower_is_stronger": True,
            "meaning": "proximity",
            "rule": "lower_distance_closer",
        },
        "vector correlation": {
            "candidate_metrics": (
                "spearman_correlation",
                "pearson_correlation",
            ),
            "lower_is_stronger": False,
            "meaning": "similarity",
            "rule": "higher_correlation_more_similar",
        },
    },
}

AMBIGUOUS_INPUTS = {
    "train": (
        "percentile",
        "relative percentile",
        "percentile value",
        "percentile rank",
        "percentile measure",
        "percentile metric",
        "percentile result",
        "reported percentile",
    ),
    "validation": (
        "percentile statistic",
        "percentile reading",
        "percentile output",
        "percentile indicator",
    ),
    "test": (
        "percentile quantity",
        "percentile field",
        "percentile number",
        "percentile estimate",
    ),
}
AMBIGUOUS_CANDIDATES = ("distance_percentile", "score_percentile")

UNKNOWN_INPUTS = {
    "train": (
        "diffusion state distance",
        "cosine distance",
        "cosine similarity",
        "Jaccard distance",
        "Jaccard similarity",
        "Euclidean distance",
        "Manhattan distance",
        "Mahalanobis distance",
        "Kendall tau",
        "mutual information",
        "graph resistance distance",
        "commute time",
        "heat kernel distance",
        "node2vec similarity",
        "betweenness centrality",
        "closeness centrality",
        "degree centrality",
        "eigenvector centrality",
        "resource allocation index",
        "Adamic-Adar index",
    ),
    "validation": (
        "Hellinger distance",
        "Wasserstein distance",
        "Canberra distance",
        "Bray-Curtis dissimilarity",
        "Kulczynski similarity",
        "Katz index",
        "SimRank similarity",
        "preferential attachment index",
    ),
    "test": (
        "Chebyshev distance",
        "Minkowski distance",
        "Bhattacharyya distance",
        "Jensen-Shannon distance",
        "Sorensen-Dice similarity",
        "Tanimoto coefficient",
        "harmonic centrality",
        "neighborhood overlap",
    ),
}

POLARITIES = (
    "lower_stronger",
    "lower_weaker",
    "higher_stronger",
    "higher_weaker",
)
MASK_TOKEN = "<SOURCE_VALUE>"
MASKED_WORDING_TOLERANCE = 0.05

S1_8_SETTING_KEYS = (
    "dataset_id",
    "question_family",
    "book_mode",
    "output_directory",
    "assignment_seed",
    "train_rows",
    "allocations",
    "evaluation_allocations",
    "semantic_buckets",
    "source_trace_by_allocation",
    "result_shape",
    "metric_conventions",
    "name_variants",
    "shared_aliases",
    "name_policy",
    "true_false",
    "ambiguous_metric",
    "unrecognized_metric",
    "status_counts",
    "framing_counts",
)
TRAIN_SETTING_KEYS = (
    "train_variant_coverage", "train_metric_reasoning", "train_final_reasoning",
    "train_decorrelate", "train_targeted_v9",
)

PROMPT_TEMPLATES = {
    "train": {
        "direct": (
            "Answer this metric question.\n{request}",
            "Review this metric question.\n{request}",
            "Give the exact metric answer.\n{request}",
            "Resolve this metric request.\n{request}",
            "Check the stated metric.\n{request}",
            "Apply the direction convention.\n{request}",
            "Evaluate this metric request.\n{request}",
            "Return the metric result.\n{request}",
        ),
        "step": (
            "Answer this before the next step.\n{request}",
            "Check this before the workflow continues.\n{request}",
            "The next step needs this metric answer.\n{request}",
            "Resolve this before the next action.\n{request}",
            "Prepare this result for later work.\n{request}",
            "Apply the convention before analysis continues.\n{request}",
            "The workflow needs this metric result.\n{request}",
            "Verify this before work advances.\n{request}",
        ),
    },
    "validation": {
        "direct": (
            "Supply the exact metric answer.\n{request}",
            "Read and answer this metric question.\n{request}",
            "Return the requested direction result.\n{request}",
            "Evaluate the supplied metric request.\n{request}",
            "State the applicable direction rule.\n{request}",
            "Assess this metric convention.\n{request}",
            "Give the result for this metric.\n{request}",
            "Resolve the supplied metric question.\n{request}",
        ),
        "step": (
            "Answer this before analysis continues.\n{request}",
            "Prepare this metric result for later work.\n{request}",
            "The workflow needs this direction answer.\n{request}",
            "Check this metric before work advances.\n{request}",
            "Resolve the convention before the next task.\n{request}",
            "Supply this result for the next action.\n{request}",
            "Verify this direction before analysis advances.\n{request}",
            "Prepare the exact answer for the next step.\n{request}",
        ),
    },
    "test": {
        "direct": (
            "Find the exact metric answer.\n{request}",
            "Give the result for this direction question.\n{request}",
            "Read this metric request and answer it.\n{request}",
            "State the requested metric result.\n{request}",
            "Use the direction convention.\n{request}",
            "Decide this metric question.\n{request}",
            "Return the applicable metric answer.\n{request}",
            "Assess the stated direction rule.\n{request}",
        ),
        "step": (
            "Resolve this before analysis advances.\n{request}",
            "This metric result is needed for the next action.\n{request}",
            "Prepare this direction answer for later analysis.\n{request}",
            "The workflow first needs this metric result.\n{request}",
            "Check the convention before the next step.\n{request}",
            "Supply this answer before work continues.\n{request}",
            "Verify the metric rule for later use.\n{request}",
            "Resolve the direction before the workflow advances.\n{request}",
        ),
    },
}

DIRECTION_REQUESTS = (
    "For {input}, which values indicate the stronger relation: lower or higher?",
    "State whether lower or higher {input} values indicate the stronger relation.",
    "Which direction is stronger for {input}: lower values or higher values?",
    "Give the lower-versus-higher convention for {input}.",
    "For {input}, do lower values or higher values show the stronger relation?",
    "Identify the stronger value direction for {input}.",
    "Does {input} treat the lower or the higher value as stronger?",
    "Return the direction rule for {input}: lower is stronger or higher is stronger.",
)

TRUE_FALSE_REQUESTS = {
    2: (
        "True or false: {statement}.",
        "Is this true or false: {statement}?",
        "Evaluate this statement as true or false: {statement}.",
        "Mark this metric statement true or false: {statement}.",
        "Decide whether this is true or false: {statement}.",
        "Give true or false for this claim: {statement}.",
        "Does this statement hold: {statement}?",
        "Check the truth of this statement: {statement}.",
    ),
    3: (
        "Check this metric statement: {statement}. True or false?",
        "Assess this convention: {statement}. Is it true?",
        "Is this claim true: {statement}?",
        "Test the stated direction rule: {statement}. True or false?",
        "Review this claim and give true or false: {statement}.",
        "Is the following metric relation correct: {statement}?",
        "Judge this direction statement: {statement}. True or false?",
        "Determine whether this convention is correct: {statement}.",
    ),
    4: (
        "Check this convention before the next analysis step: {statement}. True or false?",
        "Before later work, decide whether this is true: {statement}.",
        "Verify this metric rule for the workflow: {statement}. True or false?",
        "For the next action, assess this statement: {statement}.",
        "Prepare the workflow by checking this claim: {statement}. True or false?",
        "Before analysis continues, is this convention correct: {statement}?",
        "The next step needs a truth check: {statement}. True or false?",
        "Confirm or reject this direction statement for later use: {statement}.",
    ),
    5: (
        "True or false: {statement}.",
        "Can you decide whether this is true: {statement}?",
        "Check this percentile claim: {statement}. True or false?",
        "Evaluate this direction statement: {statement}.",
        "Is this percentile convention correct: {statement}?",
        "Give true or false for this percentile claim: {statement}.",
        "Assess this statement before use: {statement}.",
        "Determine whether this claim holds: {statement}.",
    ),
    6: (
        "True or false: {statement}.",
        "Can you decide whether this is true: {statement}?",
        "Check this metric claim: {statement}. True or false?",
        "Evaluate this direction statement: {statement}.",
        "Is this metric convention correct: {statement}?",
        "Give true or false for this metric claim: {statement}.",
        "Assess this statement before use: {statement}.",
        "Determine whether this claim holds: {statement}.",
    ),
}


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one exact S1.8 scenario."""

    split: str
    form: int
    metric: str | None
    input_name: str
    name_kind: str
    lower_is_stronger: bool | None
    meaning: str | None
    rule: str | None
    statement_polarity: str | None
    statement_true: bool | None
    candidate_metrics: tuple[str, ...]
    framing: str
    template_index: int
    occurrence: int
    scenario_id: str
    request_index: int = 0
    train_reasoning: bool = False

    @property
    def source_trace(self) -> str:
        """Return the source trace label."""

        return f"S1.8.{self.form}"

    @property
    def semantic_bucket(self) -> str:
        """Return the semantic bucket."""

        return SEMANTIC_BUCKET_BY_FORM[self.form]


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fraction(value: Any) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def convention_card(metric: str) -> dict[str, Any]:
    """Return the exact direction card for one canonical metric."""

    if metric not in METRIC_CONVENTIONS:
        raise CorpusBuildError(f"unknown direction metric {metric!r}")
    return dict(METRIC_CONVENTIONS[metric])


def _frozen_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the settings with the v1 build location.

    An S1 overlay may move the output and enable training-only coverage or
    reasoning. The frozen-section hash protects the shared source settings.
    """

    return {
        **{
            key: value for key, value in settings.items()
            if key not in TRAIN_SETTING_KEYS
        },
        "output_directory": "data/world_model_v3/corpora/s1/s1_8",
    }


def _assert_exact_source_settings(settings: Mapping[str, Any]) -> None:
    """Fail if the frozen S1.8 source settings change."""

    source_keys = tuple(key for key in settings if key not in TRAIN_SETTING_KEYS)
    if source_keys != S1_8_SETTING_KEYS:
        raise CorpusBuildError("the S1.8 setting fields changed")
    if _stable_sha256(_frozen_settings(settings)) != CONTRACT_S1_8_SHA256:
        raise CorpusBuildError("the frozen S1.8 contract section changed")
    checks = {
        "metric_conventions": METRIC_CONVENTIONS,
        "name_variants": NAME_VARIANTS,
        "shared_aliases": SHARED_ALIASES,
    }
    for key, expected in checks.items():
        if key not in settings or _stable_sha256(settings[key]) != _stable_sha256(
            expected
        ):
            raise CorpusBuildError(f"the S1.8 {key} source changed")
    ambiguous = settings.get("ambiguous_metric", {})
    if tuple(ambiguous.get("candidate_metrics", ())) != AMBIGUOUS_CANDIDATES:
        raise CorpusBuildError("the S1.8 ambiguous candidates changed")
    for split in SPLITS:
        if tuple(ambiguous.get(f"{split}_inputs", ())) != AMBIGUOUS_INPUTS[split]:
            raise CorpusBuildError(f"the S1.8 {split} ambiguous inputs changed")
    unknown = settings.get("unrecognized_metric", {})
    for split in SPLITS:
        if tuple(unknown.get(f"{split}_inputs", ())) != UNKNOWN_INPUTS[split]:
            raise CorpusBuildError(f"the S1.8 {split} unknown inputs changed")
    true_false = settings.get("true_false", {})
    if tuple(true_false.get("polarity_order", ())) != POLARITIES:
        raise CorpusBuildError("the S1.8 polarity order changed")
    if tuple(settings.get("allocations", {})) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.8 training allocation fields changed")
    if tuple(settings.get("evaluation_allocations", {})) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.8 evaluation allocation fields changed")

    variant_sets = {
        split: {
            value.casefold()
            for values in NAME_VARIANTS[split].values()
            for value in values
        }
        for split in SPLITS
    }
    alias_sets = {
        split: {value.casefold() for value in SHARED_ALIASES[split]}
        for split in SPLITS
    }
    ambiguous_sets = {
        split: {value.casefold() for value in AMBIGUOUS_INPUTS[split]}
        for split in SPLITS
    }
    unknown_sets = {
        split: {value.casefold() for value in UNKNOWN_INPUTS[split]}
        for split in SPLITS
    }
    for pools in (variant_sets, alias_sets, ambiguous_sets, unknown_sets):
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                if pools[left] & pools[right]:
                    raise CorpusBuildError(
                        f"S1.8 source names overlap in {left} and {right}"
                    )


def trace_counts(total: int, settings: Mapping[str, Any]) -> dict[int, int]:
    """Return the exact S1.8 trace allocation for one split."""

    _assert_exact_source_settings(settings)
    allocations = settings["allocations"]
    evaluation = settings["evaluation_allocations"]
    train_total = int(settings["train_rows"])
    evaluation_total = sum(int(value) for value in evaluation.values())
    if total == train_total:
        selected = allocations
    elif total == evaluation_total:
        selected = evaluation
    else:
        selected = s1_6.largest_remainder_counts(
            total,
            {
                name: Fraction(int(value), train_total)
                for name, value in allocations.items()
            },
        )
    return {
        TRACE_BY_BUCKET[name]: int(selected[name])
        for name in TRACE_BY_BUCKET
    }


def _mode(split: str) -> str:
    return "train" if split == "train" else "evaluation"


def _balanced_counts(
    total: int,
    labels: Sequence[str],
    *,
    offset: int = 0,
) -> dict[str, int]:
    """Return counts that differ by at most one."""

    quotient, remainder = divmod(total, len(labels))
    counts = {label: quotient for label in labels}
    for index in range(remainder):
        counts[labels[(offset + index) % len(labels)]] += 1
    return counts


def _apportion_from_capacity(
    total: int,
    capacities: Mapping[str, int],
    *,
    offset: int = 0,
) -> dict[str, int]:
    """Allocate a total in proportion to fixed capacities."""

    capacity_total = sum(int(value) for value in capacities.values())
    if total < 0 or total > capacity_total or capacity_total <= 0:
        raise CorpusBuildError("an S1.8 capacity allocation is invalid")
    labels = tuple(capacities)
    counts = {
        label: int(capacities[label]) * total // capacity_total
        for label in labels
    }
    remaining = total - sum(counts.values())
    order = sorted(
        labels,
        key=lambda label: (
            -(int(capacities[label]) * total % capacity_total),
            (labels.index(label) - offset) % len(labels),
        ),
    )
    for label in order:
        if remaining == 0:
            break
        if counts[label] < int(capacities[label]):
            counts[label] += 1
            remaining -= 1
    if remaining:
        raise CorpusBuildError("an S1.8 capacity allocation did not reconcile")
    return counts


def _frame_counts(
    counts: Mapping[int, int],
    settings: Mapping[str, Any],
    *,
    split: str,
) -> dict[int, dict[str, int]]:
    """Allocate the exact global frame count across all forms."""

    mode = _mode(split)
    expected = {
        key: int(value)
        for key, value in settings["framing_counts"][mode].items()
    }
    if sum(expected.values()) != sum(counts.values()):
        raise CorpusBuildError(f"the {split} S1.8 frame total changed")
    direct = {form: int(value) // 2 for form, value in counts.items()}
    extra = expected["direct"] - sum(direct.values())
    odd_forms = [form for form, value in counts.items() if int(value) % 2]
    if extra < 0 or extra > len(odd_forms):
        raise CorpusBuildError(f"the {split} S1.8 frame split cannot reconcile")
    for form in odd_forms[:extra]:
        direct[form] += 1
    result = {
        form: {
            "direct": direct[form],
            "step": int(counts[form]) - direct[form],
        }
        for form in counts
    }
    if Counter(
        {
            frame: sum(values[frame] for values in result.values())
            for frame in ("direct", "step")
        }
    ) != Counter(expected):
        raise CorpusBuildError(f"the {split} S1.8 frame counts changed")
    return result


def _truth_for_polarity(lower_is_stronger: bool, polarity: str) -> bool:
    if polarity == "lower_stronger":
        return lower_is_stronger
    if polarity == "lower_weaker":
        return not lower_is_stronger
    if polarity == "higher_stronger":
        return not lower_is_stronger
    if polarity == "higher_weaker":
        return lower_is_stronger
    raise CorpusBuildError(f"unknown statement polarity {polarity!r}")


def _true_polarities(lower_is_stronger: bool) -> tuple[str, str]:
    if lower_is_stronger:
        return ("lower_stronger", "higher_weaker")
    return ("lower_weaker", "higher_stronger")


def _false_polarities(lower_is_stronger: bool) -> tuple[str, str]:
    if lower_is_stronger:
        return ("lower_weaker", "higher_stronger")
    return ("lower_stronger", "higher_weaker")


def _interleaved_labels(counts: Mapping[str, int]) -> list[str]:
    """Interleave exact label counts in mapping order."""

    remaining = {key: int(value) for key, value in counts.items()}
    labels: list[str] = []
    while any(remaining.values()):
        for key in counts:
            if remaining[key] > 0:
                labels.append(key)
                remaining[key] -= 1
    return labels


def _name_counts_for_form(
    settings: Mapping[str, Any],
    *,
    split: str,
    form: int,
) -> dict[str, int]:
    key = f"{_mode(split)}_form_counts"
    bucket = BUCKET_BY_TRACE[form]
    counts = {
        name: int(value)
        for name, value in settings["name_policy"][key][bucket].items()
    }
    if split == "train" and settings.get("train_variant_coverage"):
        # Replace canonical repeats without changing form, alias, or truth quotas.
        counts["canonical"] -= counts["metric_variant"]
        counts["metric_variant"] *= 2
    return counts


def _metric_specs(
    *,
    split: str,
    form: int,
    name_counts: Mapping[str, int],
    rotation: int,
    name_variants: Mapping[str, Mapping[str, Sequence[str]]] = NAME_VARIANTS,
    alias_cards: Mapping[str, Mapping[str, Mapping[str, Any]]] = SHARED_ALIASES,
) -> list[dict[str, Any]]:
    """Build balanced single-metric and shared-alias specifications."""

    single_total = int(name_counts["canonical"]) + int(
        name_counts["metric_variant"]
    )
    metric_counts = _balanced_counts(single_total, METRICS, offset=rotation)
    variant_counts = _apportion_from_capacity(
        int(name_counts["metric_variant"]),
        metric_counts,
        offset=rotation,
    )
    specs: list[dict[str, Any]] = []
    serial = 0
    for metric_index, metric in enumerate(METRICS):
        count = metric_counts[metric]
        variant_count = variant_counts[metric]
        variant_positions = {
            (index * count // variant_count) % count
            for index in range(variant_count)
        } if variant_count else set()
        variant_index = 0
        for occurrence in range(count):
            is_variant = occurrence in variant_positions
            if is_variant:
                values = name_variants[split][metric]
                input_name = values[(variant_index + form + metric_index) % len(values)]
                name_kind = "metric_variant"
                variant_index += 1
            else:
                input_name = metric
                name_kind = "canonical"
            card = METRIC_CONVENTIONS[metric]
            specs.append(
                {
                    "form": form,
                    "metric": metric,
                    "input_name": input_name,
                    "name_kind": name_kind,
                    "lower_is_stronger": bool(card["lower_is_stronger"]),
                    "meaning": str(card["meaning"]),
                    "rule": str(card["rule"]),
                    "candidate_metrics": (),
                    "serial": serial,
                }
            )
            serial += 1

    alias_count = int(name_counts["shared_alias"])
    alias_names = tuple(alias_cards[split])
    alias_counts = _balanced_counts(alias_count, alias_names, offset=form - 1)
    for alias in _interleaved_labels(alias_counts):
        card = alias_cards[split][alias]
        specs.append(
            {
                "form": form,
                "metric": None,
                "input_name": alias,
                "name_kind": "shared_alias",
                "lower_is_stronger": bool(card["lower_is_stronger"]),
                "meaning": str(card["meaning"]),
                "rule": str(card["rule"]),
                "candidate_metrics": tuple(card["candidate_metrics"]),
                "serial": serial,
            }
        )
        serial += 1
    return sorted(
        specs,
        key=lambda item: _stable_sha256(
            {
                "split": split,
                "form": form,
                "input": item["input_name"],
                "serial": item["serial"],
            }
        ),
    )


def _assign_true_false(
    specs: list[dict[str, Any]],
    *,
    true_target: int,
    form: int,
) -> None:
    """Assign all polarities with exact truth and name-kind margins."""

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_kind[str(spec["name_kind"])].append(spec)
    kind_true = {kind: len(values) // 2 for kind, values in by_kind.items()}
    extra = true_target - sum(kind_true.values())
    odd_kinds = [kind for kind, values in by_kind.items() if len(values) % 2]
    ordered_odd_kinds = sorted(
        odd_kinds,
        key=lambda kind: _stable_sha256({"form": form, "kind": kind}),
    )
    if extra < 0 or extra > len(ordered_odd_kinds):
        raise CorpusBuildError(f"form {form} cannot meet its truth count")
    for kind in ordered_odd_kinds[:extra]:
        kind_true[kind] += 1

    for kind_index, (kind, kind_values) in enumerate(by_kind.items()):
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for spec in kind_values:
            key = (
                spec["metric"],
                spec["input_name"] if spec["metric"] is None else None,
            )
            grouped[key].append(spec)

        base_by_group: dict[tuple[Any, ...], int] = {}
        remaining_by_group: dict[tuple[Any, ...], int] = {}
        base_true = 0
        for key, values in grouped.items():
            use_full_base = kind in ("canonical", "shared_alias")
            base = 4 if use_full_base and len(values) >= 4 else 0
            base_by_group[key] = base
            remaining_by_group[key] = len(values) - base
            base_true += base // 2
        remaining_true_target = kind_true[kind] - base_true
        remaining_total = sum(remaining_by_group.values())
        if remaining_true_target < 0 or remaining_true_target > remaining_total:
            raise CorpusBuildError(
                f"form {form} cannot balance truth for {kind}"
            )
        true_remaining = {
            key: value // 2 for key, value in remaining_by_group.items()
        }
        truth_extra = remaining_true_target - sum(true_remaining.values())
        odd_groups = [
            key for key, value in remaining_by_group.items() if value % 2
        ]
        ordered_odd_groups = sorted(
            odd_groups,
            key=lambda key: _stable_sha256(
                {"form": form, "kind": kind, "key": key}
            ),
        )
        if truth_extra < 0 or truth_extra > len(ordered_odd_groups):
            raise CorpusBuildError(
                f"form {form} cannot allocate truth for {kind}"
            )
        for key in ordered_odd_groups[:truth_extra]:
            true_remaining[key] += 1

        for group_index, (key, values) in enumerate(grouped.items()):
            lower = bool(values[0]["lower_is_stronger"])
            base = base_by_group[key]
            true_count = base // 2 + true_remaining[key]
            false_count = len(values) - true_count
            true_pair = _true_polarities(lower)
            false_pair = _false_polarities(lower)
            true_pair_counts = _balanced_counts(
                true_count - base // 2,
                true_pair,
                offset=kind_index + group_index,
            )
            false_pair_counts = _balanced_counts(
                false_count - base // 2,
                false_pair,
                offset=kind_index + group_index + 1,
            )
            polarity_counts = {
                polarity: (1 if base else 0)
                + true_pair_counts.get(polarity, 0)
                + false_pair_counts.get(polarity, 0)
                for polarity in POLARITIES
            }
            labels = _interleaved_labels(polarity_counts)
            ordered_values = sorted(
                values,
                key=lambda item: _stable_sha256(
                    {
                        "form": form,
                        "kind": kind,
                        "key": key,
                        "serial": item["serial"],
                    }
                ),
            )
            for spec, polarity in zip(ordered_values, labels, strict=True):
                spec["statement_polarity"] = polarity
                spec["statement_true"] = _truth_for_polarity(lower, polarity)


def _assign_frames(
    specs: list[dict[str, Any]],
    *,
    direct_target: int,
    form: int,
) -> None:
    """Spread both frames across the semantic outcome strata."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        if form == 1:
            key = (spec["name_kind"],)
        elif form in (2, 3, 4):
            key = (spec["name_kind"], spec["statement_true"])
        else:
            key = (spec["input_name"],)
        grouped[key].append(spec)
    direct_counts = {key: len(values) // 2 for key, values in grouped.items()}
    extra = direct_target - sum(direct_counts.values())
    odd_keys = [key for key, values in grouped.items() if len(values) % 2]
    ordered_odd = sorted(
        odd_keys,
        key=lambda key: _stable_sha256({"frame_form": form, "key": key}),
    )
    if extra < 0 or extra > len(ordered_odd):
        raise CorpusBuildError(f"form {form} cannot meet its frame count")
    for key in ordered_odd[:extra]:
        direct_counts[key] += 1
    for key, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: _stable_sha256(
                {
                    "frame_form": form,
                    "key": key,
                    "serial": item["serial"],
                    "input": item["input_name"],
                }
            ),
        )
        direct = direct_counts[key]
        for index, spec in enumerate(ordered):
            spec["framing"] = "direct" if index < direct else "step"

    if form not in (1, 2, 3, 4):
        return

    def stratum(item: Mapping[str, Any]) -> tuple[Any, ...]:
        if form == 1:
            return (item["name_kind"],)
        return (item["name_kind"], item["statement_true"])

    for metric in METRICS:
        metric_specs = [item for item in specs if item["metric"] == metric]
        present = {str(item["framing"]) for item in metric_specs}
        if present == {"direct", "step"}:
            continue
        current = next(iter(present))
        missing = "step" if current == "direct" else "direct"
        swapped = False
        for target in metric_specs:
            target_stratum = stratum(target)
            donors = [
                item
                for item in specs
                if item["framing"] == missing
                and stratum(item) == target_stratum
                and item["metric"] != metric
            ]
            for donor in donors:
                donor_metric = donor["metric"]
                if donor_metric is not None:
                    donor_specs = [
                        item for item in specs if item["metric"] == donor_metric
                    ]
                    after = {
                        str(item["framing"])
                        for item in donor_specs
                        if item is not donor
                    }
                    after.add(current)
                    if after != {"direct", "step"}:
                        continue
                target["framing"], donor["framing"] = (
                    donor["framing"],
                    target["framing"],
                )
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            raise CorpusBuildError(
                f"form {form} cannot cover both frames for {metric}"
            )


def _deferred_specs(
    *,
    split: str,
    form: int,
    count: int,
) -> list[dict[str, Any]]:
    """Build ambiguous or unknown metric specifications."""

    inputs = AMBIGUOUS_INPUTS[split] if form == 5 else UNKNOWN_INPUTS[split]
    input_counts = _balanced_counts(count, inputs, offset=form)
    specs = []
    serial = 0
    for input_name in _interleaved_labels(input_counts):
        polarity = POLARITIES[serial % len(POLARITIES)]
        specs.append(
            {
                "form": form,
                "metric": None,
                "input_name": input_name,
                "name_kind": "ambiguous" if form == 5 else "unknown",
                "lower_is_stronger": None,
                "meaning": None,
                "rule": None,
                "statement_polarity": polarity,
                "statement_true": None,
                "candidate_metrics": AMBIGUOUS_CANDIDATES if form == 5 else (),
                "serial": serial,
            }
        )
        serial += 1
    return specs


def _balance_training_variants(
    specs_by_form: Mapping[int, list[dict[str, Any]]],
    variants: Mapping[str, Sequence[str]],
) -> None:
    """Give each training name every polarity and both lookup framings.

    Truth assignments and their framing quotas stay intact. Names rotate
    within each metric and polarity, independently of wrapper bank length.
    Lookup frames balance within metrics while preserving their total.
    """

    lookups = {
        metric: [
            item for item in specs_by_form[1]
            if item["metric"] == metric and item["name_kind"] == "metric_variant"
        ]
        for metric in METRICS
    }
    direct_total = sum(
        item["framing"] == "direct" for items in lookups.values() for item in items
    )
    direct_counts = {metric: len(items) // 2 for metric, items in lookups.items()}
    extra = direct_total - sum(direct_counts.values())
    odd_metrics = [metric for metric, items in lookups.items() if len(items) % 2]
    if not 0 <= extra <= len(odd_metrics):
        raise CorpusBuildError("training variants cannot balance lookup frames")
    for metric in odd_metrics[:extra]:
        direct_counts[metric] += 1

    def content_order(item: Mapping[str, Any]) -> str:
        return _stable_sha256({"form": item["form"], "serial": item["serial"]})

    for metric_index, metric in enumerate(METRICS):
        names = variants[metric]
        ordered = sorted(lookups[metric], key=content_order)
        direct = direct_counts[metric]
        for framing, items in (("direct", ordered[:direct]), ("step", ordered[direct:])):
            if len(items) < len(names):
                raise CorpusBuildError(
                    f"training lookups cannot cover both frames for {metric}"
                )
            for index, item in enumerate(items):
                item["framing"] = framing
                item["input_name"] = names[(index + metric_index) % len(names)]
        for polarity_index, polarity in enumerate(POLARITIES):
            items = sorted(
                (
                    item for form in (2, 3, 4) for item in specs_by_form[form]
                    if item["metric"] == metric
                    and item["name_kind"] == "metric_variant"
                    and item["statement_polarity"] == polarity
                ),
                key=content_order,
            )
            if len(items) < len(names):
                raise CorpusBuildError(
                    f"training variants lack {polarity} capacity for {metric}"
                )
            for index, item in enumerate(items):
                item["input_name"] = names[
                    (index + metric_index + polarity_index) % len(names)
                ]


def _decorrelate_training_specs(
    specs_by_form: Mapping[int, list[dict[str, Any]]],
) -> None:
    """Cover deferred name/polarity pairs and repair canonical frame gaps.

    Deferred answers do not depend on polarity. Rotate it within each name
    and framing, then assign request wording in an independent content order.
    Canonical frame swaps retain all metric, polarity, truth and frame totals.
    """

    for form in (2, 3, 4):
        specs = [item for item in specs_by_form[form] if item["name_kind"] == "canonical"]
        cells = Counter(
            (item["metric"], item["statement_polarity"], item["framing"])
            for item in specs
        )
        for metric in METRICS:
            for polarity in POLARITIES:
                for framing in ("direct", "step"):
                    if cells[(metric, polarity, framing)]:
                        continue
                    target = next(item for item in specs if (
                        item["metric"] == metric and item["statement_polarity"] == polarity
                    ))
                    donors = [item for item in specs if (
                        item["framing"] == framing
                        and item["statement_true"] == target["statement_true"]
                        and cells[(item["metric"], item["statement_polarity"], framing)] > 1
                    )]
                    if not donors:
                        raise CorpusBuildError("training canonical polarity lacks a frame donor")
                    donor = min(donors, key=lambda item: (
                        item["metric"] != metric,
                        _stable_sha256({"canonical_frame_donor": item["serial"], "form": form}),
                    ))
                    for item in (target, donor):
                        item.setdefault("original_framing", item["framing"])
                        cells[(item["metric"], item["statement_polarity"], item["framing"])] -= 1
                    target["framing"], donor["framing"] = donor["framing"], target["framing"]
                    for item in (target, donor):
                        cells[(item["metric"], item["statement_polarity"], item["framing"])] += 1

    for form in (5, 6):
        by_name_frame: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in specs_by_form[form]:
            by_name_frame[(item["input_name"], item["framing"])].append(item)
        names = sorted({key[0] for key in by_name_frame})
        for key in sorted(by_name_frame):
            ordered = sorted(by_name_frame[key], key=lambda item: _stable_sha256({
                "deferred_polarity": item["serial"], "form": form,
            }))
            offset = names.index(key[0]) + (
                len(by_name_frame[(key[0], "direct")]) if key[1] == "step" else 0
            )
            for position, item in enumerate(ordered):
                item["statement_polarity"] = POLARITIES[(position + offset) % len(POLARITIES)]
        by_polarity_frame: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in specs_by_form[form]:
            by_polarity_frame[(item["statement_polarity"], item["framing"])].append(item)
        position = 0
        for key in sorted(by_polarity_frame):
            ordered = sorted(by_polarity_frame[key], key=lambda item: _stable_sha256({
                "deferred_request": item["serial"], "form": form,
            }))
            for item in ordered:
                item["request_index"] = position % len(TRUE_FALSE_REQUESTS[form])
                position += 1


def _target_training_specs(
    specs_by_form: Mapping[int, list[dict[str, Any]]],
    variants: Mapping[str, Sequence[str]],
) -> None:
    """Match training names across polarity and scope within the fixed quotas.

    The variant budget cannot give every name all eight polarity/scope cells.
    Cover every polarity first, then use the remaining slots for opposite-scope
    partners. Alias frame swaps preserve each form's truth and scope totals.
    """

    for metric_index, metric in enumerate(METRICS):
        names = tuple(variants[metric])
        for polarity_index, polarity in enumerate(POLARITIES):
            offset = metric_index + polarity_index * len(names) // len(POLARITIES)
            position = 0
            for framing in ("direct", "step"):
                cases = sorted(
                    (item for form in (2, 3, 4) for item in specs_by_form[form]
                     if item["name_kind"] == "metric_variant"
                     and item["metric"] == metric
                     and item["statement_polarity"] == polarity
                     and item["framing"] == framing),
                    key=lambda item: _stable_sha256({
                        "variant_partner": item["serial"], "form": item["form"],
                    }),
                )
                if len(cases) > len(names):
                    raise CorpusBuildError("variant scope exceeds the matched-name capacity")
                for item in cases:
                    item["input_name"] = names[(offset + position) % len(names)]
                    position += 1
            if position < len(names):
                raise CorpusBuildError("variant budget cannot cover each name's polarity")

    aliases = [item for form in (2, 3, 4) for item in specs_by_form[form]
               if item["name_kind"] == "shared_alias"]
    cells = Counter((item["input_name"], item["statement_polarity"], item["framing"])
                    for item in aliases)
    for name in sorted({item["input_name"] for item in aliases}):
        for polarity in POLARITIES:
            for framing in ("direct", "step"):
                if cells[(name, polarity, framing)]:
                    continue
                partners = [item for item in aliases if (
                    item["input_name"] == name and item["statement_polarity"] == polarity
                )]
                swaps = [(target, donor) for target in partners for donor in aliases if (
                    donor["form"] == target["form"]
                    and donor["statement_true"] == target["statement_true"]
                    and donor["framing"] == framing
                    and cells[(donor["input_name"], donor["statement_polarity"], framing)] > 1
                )]
                if not swaps:
                    raise CorpusBuildError("shared alias lacks an opposite-scope partner")
                target, donor = min(swaps, key=lambda pair: _stable_sha256({
                    "alias_partner": [(item["form"], item["serial"]) for item in pair],
                }))
                for item in (target, donor):
                    cells[(item["input_name"], item["statement_polarity"], item["framing"])] -= 1
                target["framing"], donor["framing"] = donor["framing"], target["framing"]
                for item in (target, donor):
                    cells[(item["input_name"], item["statement_polarity"], item["framing"])] += 1

    # The same name's first views share neutral request wording regardless of
    # polarity or scope. Extra views rotate through the remaining requests.
    for form, specs in specs_by_form.items():
        groups: dict[tuple[str, str | None, str], list[dict[str, Any]]] = defaultdict(list)
        for item in specs:
            item.pop("original_framing", None)
            groups[(item["input_name"], item["statement_polarity"], item["framing"])].append(item)
        requests = DIRECTION_REQUESTS if form == 1 else TRUE_FALSE_REQUESTS[form]
        for (name, _, _), cases in groups.items():
            meaning = "proximity" if form == 5 else cases[0]["meaning"]
            outcome = max(_strength_words(meaning), key=lambda value: len(value.split()))
            statement = f'a higher value for "{name}" means {outcome}'
            choices = [index for index, request in enumerate(requests)
                       if max(_sentence_word_counts(request.format(
                           input=name, statement=statement,
                       )), default=0) <= 20]
            if not choices:
                raise CorpusBuildError("no matched request satisfies the instruction limit")
            offset = int(_stable_sha256({"matched_request": name, "form": form}), 16) % len(choices)
            for index, item in enumerate(sorted(cases, key=lambda case: case["serial"])):
                item["request_index"] = choices[(offset + index) % len(choices)]


def _audit_training_matches(
    scenarios: Sequence[Scenario], variants: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Check joint name/polarity/scope coverage and the fixed-budget maximum."""

    truth_cases = [case for case in scenarios if case.form in (2, 3, 4)]
    canonical = {(case.metric, case.form, case.statement_polarity, case.framing)
                 for case in truth_cases if case.name_kind == "canonical"}
    expected_canonical = {(metric, form, polarity, framing)
                          for metric in METRICS for form in (2, 3, 4)
                          for polarity in POLARITIES for framing in ("direct", "step")}
    if canonical != expected_canonical:
        raise CorpusBuildError("targeted canonical training views are incomplete")
    cells: dict[str, dict[tuple[str, str | None], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for case in scenarios:
        if case.form != 1:
            cells[case.name_kind][case.input_name, case.statement_polarity].add(case.framing)
    for kind in ("shared_alias", "ambiguous", "unknown"):
        names = {name for name, _ in cells[kind]}
        if any(cells[kind].get((name, polarity)) != {"direct", "step"}
               for name in names for polarity in POLARITIES):
            raise CorpusBuildError(f"targeted {kind} name lacks an opposite-scope view")
    for names in variants.values():
        for name in names:
            frames = set().union(*(cells["metric_variant"].get((name, polarity), set())
                                   for polarity in POLARITIES))
            if frames != {"direct", "step"} or any(
                not cells["metric_variant"].get((name, polarity)) for polarity in POLARITIES
            ):
                raise CorpusBuildError("targeted variant lacks polarity or scope coverage")
    capacities = Counter((case.metric, case.statement_polarity) for case in truth_cases
                         if case.name_kind == "metric_variant")
    maximum = sum(count - len(variants[str(metric)])
                  for (metric, _), count in capacities.items())
    matched = sum(frames == {"direct", "step"}
                  for frames in cells["metric_variant"].values())
    if matched != maximum:
        raise CorpusBuildError("targeted variants do not maximize opposite-scope partners")
    return {
        "canonical_form_polarity_scope_cells": len(canonical),
        "shared_alias_polarity_scope_cells": sum(map(len, cells["shared_alias"].values())),
        "variant_opposite_scope_pairs": matched,
        "variant_maximum_opposite_scope_pairs": maximum,
        "ambiguous_polarity_scope_cells": sum(map(len, cells["ambiguous"].values())),
        "unknown_polarity_scope_cells": sum(map(len, cells["unknown"].values())),
        "pass": True,
    }


def select_scenarios(
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    framing_shares: Mapping[str, Any],
    settings: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
    name_variants: Mapping[str, Mapping[str, Sequence[str]]] = NAME_VARIANTS,
    alias_cards: Mapping[str, Mapping[str, Mapping[str, Any]]] = SHARED_ALIASES,
) -> dict[str, list[Scenario]]:
    """Select all deterministic S1.8 scenarios."""

    del framing_shares
    _assert_exact_source_settings(settings)
    scenarios_by_split: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        counts = {form: int(value) for form, value in counts_by_split[split].items()}
        if tuple(counts) != tuple(BUCKET_BY_TRACE):
            raise CorpusBuildError(f"the {split} S1.8 form fields changed")
        frame_counts = _frame_counts(counts, settings, split=split)
        specs_by_form: dict[int, list[dict[str, Any]]] = {}
        for form in (1, 2, 3, 4):
            name_counts = _name_counts_for_form(
                settings, split=split, form=form
            )
            if sum(name_counts.values()) != counts[form]:
                raise CorpusBuildError(
                    f"the {split} form {form} name count changed"
                )
            specs = _metric_specs(
                split=split,
                form=form,
                name_counts=name_counts,
                rotation=(form - 1) * 3,
                name_variants=name_variants,
                alias_cards=alias_cards,
            )
            if form == 1:
                for spec in specs:
                    spec["statement_polarity"] = None
                    spec["statement_true"] = None
            else:
                truth_key = f"{_mode(split)}_form_truth_counts"
                truth_counts = settings["true_false"][truth_key][
                    BUCKET_BY_TRACE[form]
                ]
                _assign_true_false(
                    specs,
                    true_target=int(truth_counts["true"]),
                    form=form,
                )
            specs_by_form[form] = specs
        specs_by_form[5] = _deferred_specs(
            split=split, form=5, count=counts[5]
        )
        specs_by_form[6] = _deferred_specs(
            split=split, form=6, count=counts[6]
        )

        for form, specs in specs_by_form.items():
            _assign_frames(
                specs,
                direct_target=frame_counts[form]["direct"],
                form=form,
            )
        balanced_variants = split == "train" and bool(settings.get("train_variant_coverage"))
        if balanced_variants:
            _balance_training_variants(specs_by_form, name_variants[split])
        decorrelate = split == "train" and bool(settings.get("train_decorrelate"))
        if decorrelate:
            _decorrelate_training_specs(specs_by_form)
        targeted = split == "train" and bool(settings.get("train_targeted_v9"))
        if targeted:
            _target_training_specs(specs_by_form, name_variants[split])

        occurrence_counts: Counter[tuple[Any, ...]] = Counter()
        template_counts: Counter[tuple[int, str]] = Counter()
        request_counts: Counter[int] = Counter()
        used_pairs: dict[tuple[Any, ...], set[tuple[int, int]]] = defaultdict(set)
        selected: list[Scenario] = []
        for form in BUCKET_BY_TRACE:
            for spec in specs_by_form[form]:
                # Sparse frame repairs must not shift wording on later rows.
                allocation_framing = spec.get("original_framing", spec["framing"])
                key = (
                    form,
                    spec["input_name"],
                    allocation_framing,
                    spec["statement_polarity"],
                )
                occurrence = occurrence_counts[key]
                occurrence_counts[key] += 1
                wrappers = prompts[split][spec["framing"]]
                requests = (
                    DIRECTION_REQUESTS
                    if form == 1
                    else TRUE_FALSE_REQUESTS[form]
                )
                template_key = (form, allocation_framing)
                preferred_template = template_counts[template_key] % len(wrappers)
                template_counts[template_key] += 1
                preferred_request = request_counts[form] % len(requests)
                request_counts[form] += 1
                if targeted or (decorrelate and form in (5, 6)):
                    preferred_request = spec["request_index"]
                available = [
                    (
                        (preferred_template + template_shift) % len(wrappers),
                        (preferred_request + request_shift) % len(requests),
                    )
                    for template_shift in range(len(wrappers))
                    # In v5, a wrapper collision may move the wrapper only.
                    # Request wording is a content choice, independent of bank size.
                    for request_shift in range(1 if balanced_variants or decorrelate else len(requests))
                    if (
                        (preferred_template + template_shift) % len(wrappers),
                        (preferred_request + request_shift) % len(requests),
                    )
                    not in used_pairs[key]
                ]
                if not available:
                    raise CorpusBuildError(
                        f"the S1.8 prompt space is exhausted for {key!r}"
                    )
                template_index, request_index = available[0]
                used_pairs[key].add((template_index, request_index))
                identity = {
                    "split": split,
                    "form": form,
                    "metric": spec["metric"],
                    "input_name": spec["input_name"],
                    "name_kind": spec["name_kind"],
                    "lower_is_stronger": spec["lower_is_stronger"],
                    "meaning": spec["meaning"],
                    "rule": spec["rule"],
                    "statement_polarity": spec["statement_polarity"],
                    "statement_true": spec["statement_true"],
                    "candidate_metrics": spec["candidate_metrics"],
                    "framing": spec["framing"],
                    "template_index": template_index,
                    "request_index": request_index,
                    "occurrence": occurrence,
                }
                scenario_id = f"s1_8_{split}_" + _stable_sha256(identity)[:24]
                selected.append(
                    Scenario(
                        **identity,
                        scenario_id=scenario_id,
                        train_reasoning=split == "train" and bool(
                            settings.get("train_metric_reasoning")
                        ),
                    )
                )
        selected.sort(
            key=lambda scenario: _stable_sha256(
                {"seed": seed, "split": split, "id": scenario.scenario_id}
            )
        )
        scenarios_by_split[split] = selected
    return scenarios_by_split


def _strength_words(meaning: str | None) -> tuple[str, str]:
    if meaning == "proximity":
        return ("stronger proximity", "weaker proximity")
    if meaning == "similarity":
        return ("greater similarity", "less similarity")
    if meaning == "evidence":
        return ("stronger evidence", "weaker evidence")
    return ("a stronger relation", "a weaker relation")


def _statement(scenario: Scenario) -> str:
    polarity = scenario.statement_polarity
    if polarity is None:
        raise CorpusBuildError("a true-or-false scenario lacks a polarity")
    meaning = "proximity" if scenario.form == 5 else scenario.meaning
    stronger, weaker = _strength_words(meaning)
    if polarity.startswith("lower_"):
        direction = "lower"
    else:
        direction = "higher"
    outcome = stronger if polarity.endswith("_stronger") else weaker
    return f'a {direction} value for "{scenario.input_name}" means {outcome}'


def _request(scenario: Scenario) -> str:
    if scenario.form == 1:
        return DIRECTION_REQUESTS[scenario.request_index].format(
            input=scenario.input_name
        )
    return TRUE_FALSE_REQUESTS[scenario.form][scenario.request_index].format(
        statement=_statement(scenario)
    )


def _prompt(
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
) -> str:
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    return template.format(request=_request(scenario))


def _direction_sentence(
    *,
    input_name: str,
    lower_is_stronger: bool,
    meaning: str,
    rule: str,
) -> str:
    special = {
        "lower_rank_closer": f'A lower value for "{input_name}" means closer to the seed.',
        "higher_score_closer": f'A higher value for "{input_name}" means closer to the seed.',
        "lower_distance_closer": f'A lower value for "{input_name}" means closer.',
        "higher_correlation_more_similar": f'A higher value for "{input_name}" means more similar.',
        "higher_similarity_more_similar": f'A higher value for "{input_name}" means more similar.',
        "lower_distance_percentile_closer": f'A lower value for "{input_name}" means closer.',
        "higher_score_percentile_closer": f'A higher value for "{input_name}" means closer.',
        "fewer_hops_closer": f'A lower value for "{input_name}" means fewer hops and closer nodes.',
        "lower_p_stronger_evidence": f'A lower value for "{input_name}" means stronger evidence.',
        "higher_enrichment_stronger_evidence": f'A higher value for "{input_name}" means stronger evidence.',
    }
    sentence = special.get(rule)
    if sentence is None:
        direction = "lower" if lower_is_stronger else "higher"
        stronger, _ = _strength_words(meaning)
        sentence = f'A {direction} value for "{input_name}" means {stronger}.'
    return sentence


def _legacy_thinking(scenario: Scenario) -> str:
    if scenario.form == 5:
        return (
            f'"{scenario.input_name}" can mean distance_percentile or '
            "score_percentile. A lower distance percentile means closer, but "
            "a higher score percentile means closer. The candidates disagree, "
            "so I will ask which metric is meant."
        )
    if scenario.form == 6:
        return (
            f'"{scenario.input_name}" is not in the convention set. I have no '
            "direction rule for it. I will ask for its convention."
        )
    direction = _direction_sentence(
        input_name=scenario.input_name,
        lower_is_stronger=bool(scenario.lower_is_stronger),
        meaning=str(scenario.meaning),
        rule=str(scenario.rule),
    )
    if scenario.name_kind == "shared_alias":
        first, second = scenario.candidate_metrics
        lead = (
            f'"{scenario.input_name}" can mean {first} or {second}. '
            "Both metrics use the same direction rule."
        )
    else:
        lead = (
            f'"{scenario.input_name}" maps to the canonical metric '
            f"{scenario.metric}."
        )
    if scenario.form == 1:
        return f"{lead} {direction} I will return this direction."
    truth = "true" if scenario.statement_true else "false"
    return f"{lead} {direction} The statement is {truth}."


def _metric_variant_explanation(metric: str) -> str:
    """Explain the semantic mapping without changing the fixed convention."""

    return {
        "rank": "Ordering positions use rank; a smaller position is closer to the seed.",
        "score": "Visitation mass and occupancy probabilities use score; greater mass is closer to the seed.",
        "spearman_distance": "Rho and ordinal association identify Spearman; complements use spearman_distance.",
        "pearson_distance": "R and linear association identify Pearson; complements use pearson_distance.",
        "spearman_correlation": "Rho and ordinal association use spearman_correlation, the similarity form.",
        "pearson_correlation": "R and linear association use pearson_correlation, the similarity form.",
        "dot_similarity": "Inner and scalar products use dot_similarity, including product names that contain score.",
        "distance_percentile": "Distance quantiles and centiles use distance_percentile and retain the distance direction.",
        "score_percentile": "Score quantiles and centiles use score_percentile and retain the score direction.",
        "hop_count": "A count of traversed path edges uses hop_count.",
        "empirical_p_value": "Empirical and null tail probabilities use empirical_p_value as an evidence measure.",
        "q": "Correcting significance for false discoveries uses q as an evidence measure.",
        "enrichment_score": "Overabundance and overrepresentation use enrichment_score as an evidence measure.",
    }[metric]


def _reasoning_bindings(scenario: Scenario) -> dict[str, str]:
    """Bind the supplied name and its convention to the reviewed training thought."""

    bindings = {"input_name": json.dumps(scenario.input_name, ensure_ascii=False)}
    if scenario.form in (5, 6):
        return bindings
    if scenario.name_kind == "shared_alias":
        candidates = json.dumps(list(scenario.candidate_metrics), ensure_ascii=False)
        name_resolution = (
            f"This name admits the candidate metrics {candidates}. "
            "Their direction, meaning, and rule agree, so the convention is resolved. "
            "The name does not select one candidate, so metric remains null."
        )
        candidate_schema = (
            f"Also include candidate_metrics with the ordered list {candidates}. "
            "These are the only result fields."
        )
    else:
        name_resolution = "This is a canonical metric name; preserve it without normalization."
        if scenario.name_kind == "metric_variant":
            name_resolution = (
                "This name identifies a single metric in the fixed convention set. "
                + _metric_variant_explanation(str(scenario.metric))
            )
        candidate_schema = "A single metric is identified, so omit candidate_metrics. These are the only result fields."
    bindings.update(
        name_resolution=name_resolution,
        metric=json.dumps(scenario.metric),
        lower_is_stronger=json.dumps(scenario.lower_is_stronger),
        meaning=json.dumps(scenario.meaning),
        rule=json.dumps(scenario.rule),
        direction_explanation=_direction_sentence(
            input_name=scenario.input_name,
            lower_is_stronger=bool(scenario.lower_is_stronger),
            meaning=str(scenario.meaning), rule=str(scenario.rule),
        ),
        candidate_schema=candidate_schema,
    )
    if scenario.form != 1:
        polarity = str(scenario.statement_polarity)
        direction = "lower" if polarity.startswith("lower_") else "higher"
        stronger, weaker = _strength_words(scenario.meaning)
        actual_stronger = (direction == "lower") == scenario.lower_is_stronger
        bindings.update(
            claim_direction=direction,
            claim_outcome=stronger if polarity.endswith("_stronger") else weaker,
            actual_outcome=stronger if actual_stronger else weaker,
            statement_true=json.dumps(scenario.statement_true),
        )
    return bindings


def _thinking(
    scenario: Scenario,
    *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
) -> str:
    """Explain exact name copying, the fixed card, truth, and task scope."""

    if scenario.split == "train" and final_reasoning:
        template = final_reasoning[scenario.form][scenario.framing]
        bindings = _reasoning_bindings(scenario)
        if "{outer_scope}" in template:
            wrapper = prompts[scenario.split][scenario.framing][scenario.template_index]
            bindings["outer_scope"] = json.dumps(
                wrapper.replace("{request}", "").strip(), ensure_ascii=False,
            )
        return template.format(**bindings)
    if not scenario.train_reasoning:
        return _legacy_thinking(scenario)
    copy = f'Keep the input name exactly as "{scenario.input_name}".'
    if scenario.form in (5, 6):
        return (
            f"{copy} {_legacy_thinking(scenario)} "
            "No convention is resolved, so statement_true is null and continuation_decision is defer."
        )
    if scenario.name_kind == "shared_alias":
        first, second = scenario.candidate_metrics
        lead = (
            f'"{scenario.input_name}" leaves {first} and {second} as candidates, so metric is null. '
            "Both candidates share the same fixed convention card."
        )
    else:
        lead = f'"{scenario.input_name}" identifies the canonical metric {scenario.metric}.'
        if scenario.name_kind == "metric_variant":
            lead += " " + _metric_variant_explanation(str(scenario.metric))
    lower = "true" if scenario.lower_is_stronger else "false"
    card = (
        f"The fixed card sets lower_is_stronger to {lower}, meaning to {scenario.meaning}, "
        f"and rule to {scenario.rule}."
    )
    direction = _direction_sentence(
        input_name=scenario.input_name,
        lower_is_stronger=bool(scenario.lower_is_stronger),
        meaning=str(scenario.meaning), rule=str(scenario.rule),
    )
    if scenario.form == 1:
        truth = "This is a direction lookup, so statement_true is null."
    else:
        polarity = str(scenario.statement_polarity)
        requested_direction = "lower" if polarity.startswith("lower_") else "higher"
        stronger, weaker = _strength_words(scenario.meaning)
        requested_outcome = stronger if polarity.endswith("_stronger") else weaker
        actual_stronger = (requested_direction == "lower") == scenario.lower_is_stronger
        actual_outcome = stronger if actual_stronger else weaker
        verdict = "true" if scenario.statement_true else "false"
        truth = (
            f"The claim pairs {requested_direction} values with {requested_outcome}. "
            f"The convention pairs {requested_direction} values with {actual_outcome}, "
            f"so statement_true is {verdict}."
        )
    if scenario.framing == "direct":
        scope = "Only this answer is requested, so continuation_decision is stop."
        if scenario.form == 4:
            scope = (
                "The workflow wording describes the question's use; the enclosing request asks only for this answer. "
                + scope
            )
    else:
        scope = "Checking this convention prepares subsequent work, so continuation_decision is continue."
    return f"{copy} {lead} {card} {direction} {truth} {scope}"


def _result(scenario: Scenario) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metric": scenario.metric,
        "input": scenario.input_name,
        "lower_is_stronger": scenario.lower_is_stronger,
        "meaning": scenario.meaning,
        "rule": scenario.rule,
        "statement_true": scenario.statement_true,
    }
    if scenario.candidate_metrics:
        result["candidate_metrics"] = list(scenario.candidate_metrics)
    return result


def _summary(scenario: Scenario) -> str:
    if scenario.form == 5:
        return (
            f'"{scenario.input_name}" can mean the distance percentile, where '
            "lower is closer, or the score percentile, where higher is closer. "
            "Say which percentile you mean to continue."
        )
    if scenario.form == 6:
        return (
            f'"{scenario.input_name}" is not a metric in the convention set. '
            "This answer does not set a direction. Give its convention to continue."
        )
    direction = _direction_sentence(
        input_name=scenario.input_name,
        lower_is_stronger=bool(scenario.lower_is_stronger),
        meaning=str(scenario.meaning),
        rule=str(scenario.rule),
    )
    if scenario.form == 1:
        return direction
    truth = "True." if scenario.statement_true else "False."
    return f"{truth} {direction}"


def _answer(scenario: Scenario) -> dict[str, Any]:
    if scenario.form == 5:
        status = "ambiguous"
        decision = "defer"
        first, second = scenario.candidate_metrics
        blocked_reason = (
            f'The name "{scenario.input_name}" can mean {first} or {second}, '
            "and the two have opposite directions."
        )
    elif scenario.form == 6:
        status = "not_found"
        decision = "defer"
        blocked_reason = (
            f'The metric "{scenario.input_name}" is not in the direction conventions.'
        )
    else:
        status = "resolved"
        decision = "stop" if scenario.framing == "direct" else "continue"
        blocked_reason = None
    return {
        "status": status,
        "provenance": None,
        "result": _result(scenario),
        "continuation_decision": decision,
        "blocked_reason": blocked_reason,
        "summary": _summary(scenario),
    }


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_8_" + _stable_sha256(
        {"schema_version": schema_version, "scenario": scenario.scenario_id}
    )[:24]


def _split_unit(scenario: Scenario) -> str:
    if scenario.name_kind == "canonical":
        return f"shared_metric_{scenario.metric}"
    digest = hashlib.sha256(scenario.input_name.casefold().encode("utf-8")).hexdigest()
    return f"heldout_metric_name_{digest[:20]}"


def build_row(
    scenario: Scenario,
    *,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Render one closed-book S1.8 row without a tool call."""

    schema_version = str(contract["runtime"]["row_schema_version"])
    if scenario.form == 5:
        blocked_reason_code = "ambiguous_metric"
    elif scenario.form == 6:
        blocked_reason_code = "unrecognized_metric"
    else:
        blocked_reason_code = None
    row = {
        "row_id": _row_id(scenario, schema_version),
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {"role": "user", "content": _prompt(scenario, prompts=prompts)},
            {
                "role": "assistant",
                "thinking": _thinking(
                    scenario, final_reasoning=final_reasoning, prompts=prompts,
                ),
                "content": _answer(scenario),
            },
        ],
        "tools": tool_definitions(contract) if tools is None else tools,
        "metadata": {
            "book_mode": "closed_book",
            "step": "S1",
            "question_family": "direction_conventions",
            "form": scenario.form,
            "source_trace": scenario.source_trace,
            "split_unit": _split_unit(scenario),
            "blocked_reason_code": blocked_reason_code,
            "schema_version": schema_version,
            "system_prompt_sha256": system_prompt_sha256(contract),
            "tool_contract_version": contract["runtime"][
                "tool_contract_version"
            ],
            "ensembl_release": contract["coordinate_system"][
                "ensembl_release"
            ],
            "identifier_registry_id": contract["coordinate_system"][
                "identifier_registry_id"
            ],
        },
    }
    return row, []


THINKING_FILTER_RULES = (
    "one thinking step per row",
    "thinking uses only the frozen convention table",
    "ambiguous inputs do not get a direction",
    "unknown inputs do not get a direction",
    "no tool call or tool-result claim",
    "status and decision agree with the answer",
)


def _final_answer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    assistants = [
        message
        for message in row["messages"]
        if message.get("role") == "assistant" and "content" in message
    ]
    if len(assistants) != 1 or not isinstance(assistants[0]["content"], Mapping):
        raise CorpusBuildError(f"row {row['row_id']} lacks one final answer")
    return assistants[0]["content"]


def _tool_calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def _current_user_message(row: Mapping[str, Any]) -> str:
    users = [message for message in row["messages"] if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise CorpusBuildError(f"row {row['row_id']} lacks one user message")
    return str(users[0]["content"])


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
) -> int:
    """Reject a thought that violates an S1.8 runtime rule."""

    if _tool_calls(row) or any(
        message.get("role") == "tool" for message in row["messages"]
    ):
        raise CorpusBuildError(f"row {row['row_id']} contains a tool action")
    assistants = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(assistants) != 1 or assistants[0].get("thinking") != _thinking(
        scenario, final_reasoning=final_reasoning, prompts=prompts,
    ):
        raise CorpusBuildError(f"row {row['row_id']} changes its exact thinking")
    thought = str(assistants[0]["thinking"])
    lowered = thought.casefold()
    if any(value in lowered for value in ("provenance", "tool call", "tool result")):
        raise CorpusBuildError(f"row {row['row_id']} discusses runtime fields")
    if _final_answer(row) != _answer(scenario):
        raise CorpusBuildError(f"row {row['row_id']} changes its exact answer")
    return 1


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    messages = row["messages"]
    last_user = max(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    prompt = messages[: last_user + 1]
    outcomes = [
        {"tool_name": message["name"], "result_status": message["content"]["status"]}
        for message in messages[last_user + 1 :]
        if message.get("role") == "tool"
    ]
    prompt_key = _canonical_json(prompt, sort_keys=False)
    outcome_key = _canonical_json(outcomes, sort_keys=False)
    combined = _canonical_json(
        {"prompt": prompt, "tool_outcomes": outcomes}, sort_keys=False
    )
    return prompt_key, outcome_key, combined


def duplicate_prompt_outcome_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Reject each repeated closed-book prompt and outcome."""

    split_results = {}
    global_prompts: set[str] = set()
    for split in SPLITS:
        prompts: set[str] = set()
        combined: set[str] = set()
        for row, _ in zip(
            rows_by_split[split], scenarios_by_split[split], strict=True
        ):
            prompt, _, signature = _prompt_and_outcome_signatures(row)
            if prompt in prompts or signature in combined:
                raise CorpusBuildError(
                    f"{split} repeats a prompt and closed-book outcome"
                )
            prompts.add(prompt)
            combined.add(signature)
        if global_prompts & prompts:
            raise CorpusBuildError(f"{split} repeats a prompt from another split")
        global_prompts.update(prompts)
        split_results[split] = {
            "row_count": len(rows_by_split[split]),
            "normalized_prompt_count": len(prompts),
            "normalized_prompt_tool_outcome_count": len(combined),
            "duplicate_prompt_tool_outcome_count": 0,
            "cross_split_duplicate_prompt_count": 0,
            "intended_success_failure_pair_count": 0,
            "pass": True,
        }
    return {
        "normalization": "retain the exact visible prompt and every tool outcome",
        "splits": split_results,
        "pass": True,
    }


def masked_visible_prompt(row: Mapping[str, Any], scenario: Scenario) -> str:
    """Mask the metric input but retain the visible prompt structure."""

    visible = _current_user_message(row)
    visible = re.sub(
        re.escape(scenario.input_name),
        MASK_TOKEN,
        visible,
        flags=re.IGNORECASE,
    )
    return visible


def _wording_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", value.casefold()))


def _binary_wording_classifier(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    masker: Callable[[Mapping[str, Any], Scenario], str],
    scope: str,
    gated: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Train and test one class-balanced word model."""

    labels = (False, True)
    token_counts: dict[bool, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[bool] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(
        rows_by_split["train"], scenarios_by_split["train"], strict=True
    ):
        if scenario.form not in (2, 3, 4):
            continue
        tokens = _wording_tokens(masker(row, scenario))
        label = bool(scenario.statement_true)
        token_counts[label].update(tokens)
        token_totals[label] += len(tokens)
        vocabulary.update(tokens)
    if not vocabulary:
        raise CorpusBuildError("the S1.8 wording classifier has no vocabulary")

    def predict(tokens: Sequence[str]) -> bool:
        known = Counter(token for token in tokens if token in vocabulary)
        scores = []
        for label in labels:
            denominator = token_totals[label] + len(vocabulary)
            score = sum(
                count * math.log((token_counts[label][token] + 1) / denominator)
                for token, count in known.items()
            )
            scores.append((score, int(label), label))
        return max(scores)[2]

    evaluations = {}
    passed_all = True
    for split in SPLITS:
        pairs = [
            (row, scenario)
            for row, scenario in zip(
                rows_by_split[split], scenarios_by_split[split], strict=True
            )
            if scenario.form in (2, 3, 4)
        ]
        distribution = Counter(bool(scenario.statement_true) for _, scenario in pairs)
        correct = sum(
            predict(_wording_tokens(masker(row, scenario)))
            == bool(scenario.statement_true)
            for row, scenario in pairs
        )
        accuracy = correct / len(pairs)
        baseline = max(distribution.values()) / len(pairs)
        apply_gate = gated and split != "train"
        threshold = baseline + tolerance if apply_gate else None
        passed = accuracy <= threshold if apply_gate else None
        if passed is False:
            passed_all = False
        evaluations[split] = {
            "row_count": len(pairs),
            "correct": correct,
            "accuracy": accuracy,
            "largest_outcome_baseline": baseline,
            "threshold": threshold,
            "gate_applied": apply_gate,
            "pass": passed,
        }
    result = {
        "algorithm": "class_balanced_multinomial_naive_bayes_v1",
        "scope": scope,
        "diagnostic_only": not gated,
        "mask_token": MASK_TOKEN,
        "absolute_tolerance": tolerance,
        "training_vocabulary_size": len(vocabulary),
        "splits": evaluations,
        "pass": passed_all if gated else None,
    }
    if gated and not passed_all:
        raise CorpusBuildError("masked user wording predicts the S1.8 truth value")
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
) -> dict[str, Any]:
    """Gate truth leakage after the metric name is masked."""

    return _binary_wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=masked_visible_prompt,
        scope="true_false_rows_with_metric_input_masked",
        gated=True,
        tolerance=tolerance,
    )


def full_visible_prompt_classifier_diagnostic(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Measure full prompt cues without a pass gate."""

    return _binary_wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=lambda row, _: _current_user_message(row),
        scope="full_visible_true_false_prompt",
        gated=False,
        tolerance=MASKED_WORDING_TOLERANCE,
    )


def _sentence_word_counts(text: str) -> list[int]:
    normalized = re.sub(r"<[^>]+>", " CODE ", text)
    normalized = re.sub(r"\S*:\S+", " CODE ", normalized)
    normalized = re.sub(r"\b[A-Za-z0-9]+_[A-Za-z0-9_-]+\b", " CODE ", normalized)
    return [
        len(re.findall(r"[A-Za-z]+|CODE", sentence))
        for sentence in re.split(r"[.!?]+", normalized)
        if sentence.strip()
    ]


def _prose_length_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Check generated prose against the repository sentence limit."""

    splits = {}
    for split in SPLITS:
        maximum_instruction = 0
        maximum_description = 0
        violations = 0
        for row in rows_by_split[split]:
            instructions = _sentence_word_counts(_current_user_message(row))
            maximum_instruction = max(
                maximum_instruction, max(instructions, default=0)
            )
            if any(count > 20 for count in instructions):
                violations += 1
            assistant = next(
                message
                for message in row["messages"]
                if message.get("role") == "assistant" and "content" in message
            )
            answer = assistant["content"]
            descriptions = [str(assistant["thinking"]), str(answer["summary"])]
            if answer["blocked_reason"] is not None:
                descriptions.append(str(answer["blocked_reason"]))
            counts = [
                count
                for description in descriptions
                for count in _sentence_word_counts(description)
            ]
            maximum_description = max(
                maximum_description, max(counts, default=0)
            )
            if any(count > 25 for count in counts):
                violations += 1
        if violations:
            raise CorpusBuildError(f"{split} violates the prose length limits")
        splits[split] = {
            "row_count": len(rows_by_split[split]),
            "maximum_instruction_words": maximum_instruction,
            "maximum_description_words": maximum_description,
            "violation_count": 0,
            "pass": True,
        }
    return {"splits": splits, "pass": True}


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = PROMPT_TEMPLATES,
    name_variants: Mapping[str, Mapping[str, Sequence[str]]] = NAME_VARIANTS,
    alias_cards: Mapping[str, Mapping[str, Mapping[str, Any]]] = SHARED_ALIASES,
) -> dict[str, Any]:
    """Audit rendered rows against all S1.8 invariants."""

    del framing_shares
    if contract is None:
        contract = load_contract()
    if settings is None:
        settings = contract["curriculum"]["families"]["S1.8"]
    _assert_exact_source_settings(settings)
    tools = tool_definitions(contract)
    row_ids: set[str] = set()
    noncanonical_split_units: dict[str, set[str]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    tool_call_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    prompt_template_counts: dict[str, Any] = {}
    semantic_bucket_counts: dict[str, dict[str, int]] = {}
    metric_counts: dict[str, dict[str, int]] = {}
    candidate_metric_counts: dict[str, dict[str, int]] = {}
    name_kind_counts: dict[str, dict[str, int]] = {}
    statement_truth_counts: dict[str, dict[str, int]] = {}
    statement_polarity_counts: dict[str, dict[str, int]] = {}
    metric_polarity_counts: dict[str, Any] = {}
    agreeing_alias_counts: dict[str, dict[str, int]] = {}
    ambiguous_input_counts: dict[str, dict[str, int]] = {}
    unknown_input_counts: dict[str, dict[str, int]] = {}
    continuation_counts: dict[str, dict[str, int]] = {}
    source_coverage: dict[str, Any] = {}
    per_form: dict[str, Any] = {}
    thinking: dict[str, Any] = {}
    structures: dict[str, Any] = {}

    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} row and scenario counts differ")
        expected_forms = {
            form: int(value) for form, value in expected_counts[split].items()
        }
        forms = Counter(scenario.form for scenario in scenarios)
        if {form: forms[form] for form in expected_forms} != expected_forms:
            raise CorpusBuildError(f"{split} trace counts changed")

        statuses: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        frames: Counter[str] = Counter()
        semantics: Counter[str] = Counter()
        metrics: Counter[str] = Counter()
        candidate_metrics: Counter[str] = Counter()
        name_kinds: Counter[str] = Counter()
        truths: Counter[str] = Counter()
        polarities: Counter[str] = Counter()
        aliases: Counter[str] = Counter()
        ambiguous_inputs: Counter[str] = Counter()
        unknown_inputs: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        templates: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        metric_polarities: dict[int, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        gates: dict[str, Counter[str]] = defaultdict(Counter)
        thinking_steps = 0
        no_calls = 0
        harnesses = 0
        split_units: set[str] = set()
        canonical_units: set[str] = set()
        used_inputs: set[str] = set()

        for row, scenario in zip(rows, scenarios, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            expected_row = build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                final_reasoning=settings.get("train_final_reasoning"),
            )[0]
            if row != expected_row:
                raise CorpusBuildError(f"row {row_id} differs from its exact scenario")
            form_key = scenario.source_trace
            gate = gates[form_key]
            gate["row_count"] += 1
            answer = _final_answer(row)
            result = answer["result"]
            expected_result_keys = [
                "metric",
                "input",
                "lower_is_stronger",
                "meaning",
                "rule",
                "statement_true",
            ]
            if scenario.candidate_metrics:
                expected_result_keys.append("candidate_metrics")
            if list(result) != expected_result_keys:
                raise CorpusBuildError(f"row {row_id} changes result key order")
            if answer["provenance"] is not None:
                raise CorpusBuildError(f"row {row_id} has closed-book provenance")
            if any(message.get("role") == "harness" for message in row["messages"]):
                harnesses += 1
            row_calls = _tool_calls(row)
            calls[str(len(row_calls))] += 1
            if row_calls or any(
                message.get("role") == "tool" for message in row["messages"]
            ):
                raise CorpusBuildError(f"row {row_id} makes a forbidden tool call")
            no_calls += 1
            thinking_steps += filter_row_thinking(
                row, scenario, final_reasoning=settings.get("train_final_reasoning"),
                prompts=prompts,
            )

            statuses[str(answer["status"])] += 1
            frames[scenario.framing] += 1
            semantics[scenario.semantic_bucket] += 1
            name_kinds[scenario.name_kind] += 1
            decisions[str(answer["continuation_decision"])] += 1
            templates[form_key][scenario.framing][
                str(scenario.template_index)
            ] += 1
            used_inputs.add(scenario.input_name)
            unit = str(row["metadata"]["split_unit"])
            if scenario.name_kind == "canonical":
                canonical_units.add(unit)
                if unit != f"shared_metric_{scenario.metric}":
                    raise CorpusBuildError(f"row {row_id} has a wrong shared unit")
            else:
                split_units.add(unit)
                if unit != _split_unit(scenario):
                    raise CorpusBuildError(f"row {row_id} has a wrong held-out unit")

            if scenario.metric is not None:
                metrics[scenario.metric] += 1
                card = convention_card(scenario.metric)
                if any(result[key] != value for key, value in card.items()):
                    raise CorpusBuildError(f"row {row_id} changes its metric card")
            for candidate in scenario.candidate_metrics:
                candidate_metrics[candidate] += 1
            if scenario.name_kind == "shared_alias":
                aliases[scenario.input_name] += 1
                if scenario.metric is not None or any(
                    result[key] is None
                    for key in ("lower_is_stronger", "meaning", "rule")
                ):
                    raise CorpusBuildError(
                        f"row {row_id} changes a shared-alias result"
                    )
            if scenario.form == 5:
                ambiguous_inputs[scenario.input_name] += 1
                if any(
                    result[key] is not None
                    for key in (
                        "metric",
                        "lower_is_stronger",
                        "meaning",
                        "rule",
                        "statement_true",
                    )
                ):
                    raise CorpusBuildError(
                        f"row {row_id} assigns a direction to an ambiguous name"
                    )
            elif scenario.form == 6:
                unknown_inputs[scenario.input_name] += 1
                if "candidate_metrics" in result or any(
                    result[key] is not None
                    for key in (
                        "metric",
                        "lower_is_stronger",
                        "meaning",
                        "rule",
                        "statement_true",
                    )
                ):
                    raise CorpusBuildError(
                        f"row {row_id} assigns a direction to an unknown metric"
                    )
            elif scenario.form == 1:
                if result["statement_true"] is not None:
                    raise CorpusBuildError(
                        f"row {row_id} assigns truth to a direction lookup"
                    )
            else:
                expected_truth = _truth_for_polarity(
                    bool(scenario.lower_is_stronger),
                    str(scenario.statement_polarity),
                )
                if result["statement_true"] is not expected_truth:
                    raise CorpusBuildError(f"row {row_id} changes statement truth")
                truth_key = "true" if expected_truth else "false"
                truths[truth_key] += 1
                polarity = str(scenario.statement_polarity)
                polarities[polarity] += 1
                if scenario.metric is not None:
                    metric_polarities[scenario.form][scenario.metric][polarity] += 1

            for key in (
                "status",
                "result",
                "continuation_decision",
                "blocked_reason",
            ):
                gate[f"exact_{key}_count"] += 1
            gate["null_provenance_count"] += 1
            gate["exact_call_count"] += 1

        expected_frame_counts = {
            key: int(value)
            for key, value in settings["framing_counts"][_mode(split)].items()
        }
        if dict(frames) != expected_frame_counts:
            raise CorpusBuildError(f"{split} framing counts changed")
        expected_status_counts = {
            key: int(value)
            for key, value in settings["status_counts"][_mode(split)].items()
        }
        if dict(statuses) != expected_status_counts:
            raise CorpusBuildError(f"{split} status counts changed")
        expected_name_counts = {
            key: int(value)
            for key, value in settings["name_policy"][
                f"{_mode(split)}_counts"
            ].items()
        }
        if split == "train" and settings.get("train_variant_coverage"):
            expected_name_counts = dict(sum(
                (Counter(_name_counts_for_form(settings, split=split, form=form))
                 for form in (1, 2, 3, 4)),
                Counter(),
            ))
        resolved_name_counts = {
            key: name_kinds[key]
            for key in ("canonical", "metric_variant", "shared_alias")
        }
        if resolved_name_counts != expected_name_counts:
            raise CorpusBuildError(f"{split} name-kind counts changed")
        expected_truth_counts = {
            key: int(value)
            for key, value in settings["true_false"][
                f"{_mode(split)}_truth_counts"
            ].items()
        }
        if dict(truths) != expected_truth_counts:
            raise CorpusBuildError(f"{split} truth counts changed")

        for form in (2, 3, 4):
            form_scenarios = [item for item in scenarios if item.form == form]
            for kind in ("canonical", "metric_variant", "shared_alias"):
                selected = [item for item in form_scenarios if item.name_kind == kind]
                kind_truth = Counter(bool(item.statement_true) for item in selected)
                if abs(kind_truth[True] - kind_truth[False]) > 1:
                    raise CorpusBuildError(
                        f"{split} form {form} leaks truth through {kind}"
                    )
                for truth in (False, True):
                    truth_frames = Counter(
                        item.framing
                        for item in selected
                        if bool(item.statement_true) is truth
                    )
                    if abs(truth_frames["direct"] - truth_frames["step"]) > 1:
                        raise CorpusBuildError(
                            f"{split} form {form} leaks truth through framing"
                        )
            for metric in METRICS:
                coverage = set(metric_polarities[form][metric])
                if coverage != set(POLARITIES):
                    raise CorpusBuildError(
                        f"{split} form {form} lacks polarities for {metric}"
                    )

        for form in BUCKET_BY_TRACE:
            form_key = f"S1.8.{form}"
            for frame in ("direct", "step"):
                if set(templates[form_key][frame]) != {
                    str(index) for index in range(len(prompts[split][frame]))
                }:
                    raise CorpusBuildError(
                        f"{split} form {form} {frame} lacks template coverage"
                    )

        required_inputs = {
            *METRICS,
            *(value for values in name_variants[split].values() for value in values),
            *alias_cards[split],
            *AMBIGUOUS_INPUTS[split],
            *UNKNOWN_INPUTS[split],
        }
        if not required_inputs <= used_inputs:
            missing = sorted(required_inputs - used_inputs)
            raise CorpusBuildError(f"{split} lacks S1.8 source inputs {missing!r}")
        if split == "train" and settings.get("train_variant_coverage"):
            for names in name_variants[split].values():
                for name in names:
                    cases = [case for case in scenarios if case.input_name == name]
                    name_polarities = {
                        case.statement_polarity for case in cases if case.form in (2, 3, 4)
                    }
                    name_frames = {case.framing for case in cases}
                    if name_polarities != set(POLARITIES) or name_frames != {"direct", "step"}:
                        raise CorpusBuildError(
                            f"training name {name!r} lacks polarity or frame coverage"
                        )

        noncanonical_split_units[split] = split_units
        status_counts[split] = dict(sorted(statuses.items()))
        tool_call_counts[split] = dict(sorted(calls.items()))
        framing_counts[split] = dict(sorted(frames.items()))
        semantic_bucket_counts[split] = dict(sorted(semantics.items()))
        metric_counts[split] = {metric: metrics[metric] for metric in METRICS}
        candidate_metric_counts[split] = dict(sorted(candidate_metrics.items()))
        name_kind_counts[split] = dict(sorted(name_kinds.items()))
        statement_truth_counts[split] = {
            key: truths[key] for key in ("true", "false")
        }
        statement_polarity_counts[split] = {
            polarity: polarities[polarity] for polarity in POLARITIES
        }
        metric_polarity_counts[split] = {
            f"S1.8.{form}": {
                metric: {
                    polarity: metric_polarities[form][metric][polarity]
                    for polarity in POLARITIES
                }
                for metric in METRICS
            }
            for form in (2, 3, 4)
        }
        agreeing_alias_counts[split] = dict(sorted(aliases.items()))
        ambiguous_input_counts[split] = dict(sorted(ambiguous_inputs.items()))
        unknown_input_counts[split] = dict(sorted(unknown_inputs.items()))
        continuation_counts[split] = dict(sorted(decisions.items()))
        prompt_template_counts[split] = {
            form: {
                frame: dict(sorted(values.items()))
                for frame, values in sorted(frame_values.items())
            }
            for form, frame_values in sorted(templates.items())
        }
        source_coverage[split] = {
            "canonical_metric_count": sum(metric in used_inputs for metric in METRICS),
            "metric_variant_count": sum(
                value in used_inputs
                for values in name_variants[split].values()
                for value in values
            ),
            "shared_alias_count": sum(
                value in used_inputs for value in alias_cards[split]
            ),
            "ambiguous_input_count": sum(
                value in used_inputs for value in AMBIGUOUS_INPUTS[split]
            ),
            "unknown_input_count": sum(
                value in used_inputs for value in UNKNOWN_INPUTS[split]
            ),
            "shared_split_unit_count": len(canonical_units),
            "heldout_split_unit_count": len(split_units),
            "pass": True,
        }
        per_form[split] = {}
        for form, count in expected_forms.items():
            form_key = f"S1.8.{form}"
            values = gates[form_key]
            form_gate = {
                "row_count": count,
                "exact_status_count": values["exact_status_count"],
                "exact_result_count": values["exact_result_count"],
                "exact_continuation_decision_count": values[
                    "exact_continuation_decision_count"
                ],
                "exact_blocked_reason_count": values[
                    "exact_blocked_reason_count"
                ],
                "null_provenance_count": values["null_provenance_count"],
                "exact_call_count": values["exact_call_count"],
            }
            form_gate["pass"] = all(
                value == count
                for key, value in form_gate.items()
                if key.endswith("_count") and key != "row_count"
            )
            per_form[split][form_key] = form_gate
        thinking[split] = {
            "row_count": len(rows),
            "thinking_step_count": thinking_steps,
            "pass": thinking_steps == len(rows),
        }
        structures[split] = {
            "row_count": len(rows),
            "harness_message_count": harnesses,
            "no_tool_call_count": no_calls,
            "shared_metric_split_unit_count": len(canonical_units),
            "heldout_name_split_unit_count": len(split_units),
            "pass": harnesses == 0 and no_calls == len(rows),
        }
        if split == "train" and settings.get("train_targeted_v9"):
            structures[split]["matched_views"] = _audit_training_matches(
                scenarios, name_variants[split],
            )

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = noncanonical_split_units[left] & noncanonical_split_units[right]
            if overlap:
                raise CorpusBuildError(
                    f"held-out metric names overlap in {left} and {right}"
                )

    duplicate = duplicate_prompt_outcome_audit(rows_by_split, scenarios_by_split)
    wording = masked_wording_classifier_audit(rows_by_split, scenarios_by_split)
    full_prompt = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    prose = _prose_length_audit(rows_by_split)
    all_form_gates = all(
        gate["pass"]
        for split_gates in per_form.values()
        for gate in split_gates.values()
    )
    return {
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
        "framing_counts": framing_counts,
        "prompt_template_counts": prompt_template_counts,
        "semantic_bucket_counts": semantic_bucket_counts,
        "metric_counts": metric_counts,
        "candidate_metric_counts": candidate_metric_counts,
        "name_kind_counts": name_kind_counts,
        "truth_counts": statement_truth_counts,
        "statement_truth_counts": statement_truth_counts,
        "statement_polarity_counts": statement_polarity_counts,
        "metric_polarity_counts": metric_polarity_counts,
        "agreeing_alias_counts": agreeing_alias_counts,
        "ambiguous_input_counts": ambiguous_input_counts,
        "unknown_input_counts": unknown_input_counts,
        "continuation_counts": continuation_counts,
        "source_coverage": source_coverage,
        "per_form_gates": {
            "splits": per_form,
            "direction_accuracy": 1.0,
            "true_false_accuracy": 1.0,
            "heldout_name_accuracy": 1.0,
            "ambiguous_defer_accuracy": 1.0,
            "unknown_defer_accuracy": 1.0,
            "direction_claim_on_defer_count": 0,
            "false_defer_count": 0,
            "pass": all_form_gates,
        },
        "prompt_structure_audit": {
            "splits": structures,
            "full_visible_prompt_classifier": full_prompt,
            "prose_length_audit": prose,
            "pass": all(item["pass"] for item in structures.values())
            and prose["pass"],
        },
        "masked_wording_classifier": wording,
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking,
            "pass": all(item["pass"] for item in thinking.values()),
        },
        "duplicate_prompt_tool_outcome": duplicate,
        "split_unit_sha256": {
            split: _stable_sha256(sorted(units))
            for split, units in noncanonical_split_units.items()
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_corpus(path: Path) -> None:
    validator = (
        REPOSITORY_ROOT
        / "agents"
        / "world_model_v3_curriculum"
        / "s1_kit"
        / "validate.py"
    )
    result = subprocess.run(
        [sys.executable, str(validator), str(path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CorpusBuildError(
            f"semantic validation failed for {path}:\n" + result.stdout + result.stderr
        )


def _configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _source_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        stored_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        stored_path = str(resolved)
    return {"path": stored_path, "sha256": file_sha256(resolved)}


def build_corpus(
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path = CONTRACT_PATH,
    validate: bool = True,
    overlay: Mapping[str, str] | None = None,
    prompt_banks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate S1.8, then publish its manifest last.

    A pinned prompt bank replaces training wrappers and names. The v5
    switches balance individual name coverage and explain exact convention
    cards and task scope. Validation and test retain the frozen inline lists
    and reasoning, so their rows do not change.
    """

    settings = contract["curriculum"]["families"]["S1.8"]
    _assert_exact_source_settings(settings)
    seed = int(settings["assignment_seed"])
    if prompt_banks is not None:
        prompts = wrapper_prompts(
            prompt_banks, PROMPT_TEMPLATES, bank_name="metric"
        )
        name_variants, alias_cards = metric_names(
            prompt_banks, NAME_VARIANTS, SHARED_ALIASES
        )
        if settings.get("train_variant_coverage"):
            validate_metric_names(
                name_variants["train"], load_prompt_banks(prompt_banks)["shared_aliases"],
            )
    else:
        prompts = PROMPT_TEMPLATES
        name_variants, alias_cards = NAME_VARIANTS, SHARED_ALIASES
    row_counts = split_row_counts(
        int(settings["train_rows"]), contract["curriculum"]["split_fractions"]
    )
    counts_by_split = {
        split: trace_counts(row_counts[split], settings) for split in SPLITS
    }
    scenarios_by_split = select_scenarios(
        counts_by_split,
        seed=seed,
        framing_shares=contract["curriculum"]["framing"],
        settings=settings,
        prompts=prompts,
        name_variants=name_variants,
        alias_cards=alias_cards,
    )
    tools = tool_definitions(contract)
    rows_by_split = {
        split: [
            build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                final_reasoning=settings.get("train_final_reasoning"),
            )[0]
            for scenario in scenarios_by_split[split]
        ]
        for split in SPLITS
    }
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        framing_shares=contract["curriculum"]["framing"],
        settings=settings,
        contract=contract,
        prompts=prompts,
        name_variants=name_variants,
        alias_cards=alias_cards,
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_8_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {split: temporary / f"{split}.jsonl" for split in SPLITS}
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)

        golden_traces = REPOSITORY_ROOT / "runtime/world_model/golden_traces.jsonl"
        builder_path = Path(__file__).resolve()
        s1_6_builder = REPOSITORY_ROOT / "runtime/world_model/s1_6_corpus.py"
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "section": "S1.8",
            "dataset_id": settings["dataset_id"],
            "seed": seed,
            "contract_s1_8_sha256": _stable_sha256(settings),
            "prompt_templates_sha256": _stable_sha256(prompts),
            "request_templates_sha256": _stable_sha256(
                {
                    "direction": DIRECTION_REQUESTS,
                    "true_false": TRUE_FALSE_REQUESTS,
                }
            ),
            "metric_conventions_sha256": _stable_sha256(METRIC_CONVENTIONS),
            "name_sources_sha256": _stable_sha256(
                {
                    "name_variants": name_variants,
                    "shared_aliases": alias_cards,
                    "ambiguous_inputs": AMBIGUOUS_INPUTS,
                    "unknown_inputs": UNKNOWN_INPUTS,
                }
            ),
            "scenario_assignment_sha256": {
                split: _stable_sha256(
                    [item.scenario_id for item in scenarios_by_split[split]]
                )
                for split in SPLITS
            },
            "row_counts": row_counts,
            "form_counts": {
                split: {
                    f"S1.8.{form}": int(count)
                    for form, count in counts_by_split[split].items()
                }
                for split in SPLITS
            },
            "semantic_bucket_counts": audit["semantic_bucket_counts"],
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "prompt_template_counts": audit["prompt_template_counts"],
            "metric_counts": audit["metric_counts"],
            "candidate_metric_counts": audit["candidate_metric_counts"],
            "name_kind_counts": audit["name_kind_counts"],
            "statement_truth_counts": audit["statement_truth_counts"],
            "statement_polarity_counts": audit["statement_polarity_counts"],
            "metric_polarity_counts": audit["metric_polarity_counts"],
            "agreeing_alias_counts": audit["agreeing_alias_counts"],
            "ambiguous_input_counts": audit["ambiguous_input_counts"],
            "unknown_input_counts": audit["unknown_input_counts"],
            "continuation_counts": audit["continuation_counts"],
            "sources": {
                "contract": _source_record(contract_path),
                "builder": _source_record(builder_path),
                "s1_6_builder": _source_record(s1_6_builder),
                "golden_traces": _source_record(golden_traces),
                "convention_policy": (
                    "use_only_the_closed_metric_table_and_split_specific_name_pools"
                ),
            },
            "source_coverage": audit["source_coverage"],
            "split_unit_sha256": audit["split_unit_sha256"],
            "per_form_gates": audit["per_form_gates"],
            "prompt_structure_audit": audit["prompt_structure_audit"],
            "masked_wording_classifier": audit["masked_wording_classifier"],
            "thinking_filter": audit["thinking_filter"],
            "duplicate_prompt_tool_outcome": audit[
                "duplicate_prompt_tool_outcome"
            ],
            "evidence": {
                "call_fields": ["evidence_id", "row_id", "producer"],
                "calls": [],
                "call_count": 0,
            },
            "files": {
                path.name: {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        expected_manifest_keys = {
            "schema_version",
            "corpus_schema_version",
            "section",
            "dataset_id",
            "seed",
            "contract_s1_8_sha256",
            "prompt_templates_sha256",
            "request_templates_sha256",
            "metric_conventions_sha256",
            "name_sources_sha256",
            "scenario_assignment_sha256",
            "row_counts",
            "form_counts",
            "semantic_bucket_counts",
            "status_counts",
            "tool_call_counts",
            "framing_counts",
            "prompt_template_counts",
            "metric_counts",
            "candidate_metric_counts",
            "name_kind_counts",
            "statement_truth_counts",
            "statement_polarity_counts",
            "metric_polarity_counts",
            "agreeing_alias_counts",
            "ambiguous_input_counts",
            "unknown_input_counts",
            "continuation_counts",
            "sources",
            "source_coverage",
            "split_unit_sha256",
            "per_form_gates",
            "prompt_structure_audit",
            "masked_wording_classifier",
            "thinking_filter",
            "duplicate_prompt_tool_outcome",
            "evidence",
            "files",
        }
        if set(manifest) != expected_manifest_keys:
            raise CorpusBuildError("the S1.8 manifest fields changed")
        if overlay is not None:
            manifest["s1_overlay"] = dict(overlay)
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if "train_final_reasoning" in settings:
            manifest["train_final_reasoning"] = dict(settings["train_final_reasoning"])
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        output_directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            (temporary / name).replace(output_directory / name)
        manifest_path.replace(output_directory / "manifest.json")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--overlay",
        type=Path,
        help="apply an S1 overlay such as runtime/world_model/s1_v3.yaml",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip semantic validation for a focused test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    overlay = None
    prompt_banks = None
    if args.overlay is not None:
        contract, payload = apply_s1_overlay(
            contract, args.overlay, base_contract_path=args.contract
        )
        overlay = overlay_pin(args.overlay, payload)
        prompt_banks = payload.get("prompt_banks")
    settings = contract["curriculum"]["families"]["S1.8"]
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    canonical_output = _configured_path(settings["output_directory"]).resolve()
    if args.no_validate and output_directory.resolve() == canonical_output:
        parser.error(
            "--no-validate requires an explicit noncanonical output directory"
        )
    manifest = build_corpus(
        output_directory,
        contract=contract,
        contract_path=args.contract,
        validate=not args.no_validate,
        overlay=overlay,
        prompt_banks=prompt_banks,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "row_counts": manifest["row_counts"],
                "files": manifest["files"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
