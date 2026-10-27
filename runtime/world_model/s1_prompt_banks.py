"""Load and check the frozen S1 training prompt banks.

The v1/v2 S1 builders train on four (S1.8, S1.9: eight) direct and step
wrappers per split, so each wrapper repeats hundreds of times and held-out
paraphrases fall outside the training vocabulary.  S1.8 also trains on three
names per metric and two shared aliases.  A prompt bank replaces only these
training lists.  Validation and test keep the builders' inline lists, so their
rows, and the S1 test panel built from them, stay byte-identical.

The banks were drafted with an LLM and are checked in as data.  No LLM runs at
build time; the builders read the pinned file, and this module rejects any
entry that could change a row's meaning or leak a held-out paraphrase.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import re
from typing import Any

from .contract import REPOSITORY_ROOT
from .sources import file_sha256


PROMPT_BANK_SCHEMA_VERSION = "mentor-rl-world-model-s1-prompt-banks-v1"
FRAMINGS = ("direct", "step")
MINIMUM_WRAPPERS = 32
MAXIMUM_WRAPPER_LENGTH = 160
REQUEST_SLOT = "{request}"
S1_3_ENDING = ": {request}."

# Every wrapper bank must avoid naming tools, schema fields, or outcomes.
COMMON_REVEALING = (
    r"tools?|lookups?|look\s+up|registry|registries|databases?|calls?|json|"
    r"schema|status|provenance|evidence|found|invalid|valid|malformed|"
    r"errors?|fail\w*|unknown|missing|defer\w*|absent|present|ambiguous|"
    r"true|false"
)


def _revealing(family_terms: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b({COMMON_REVEALING}|{family_terms})\b", re.IGNORECASE
    )


# A wrapper frames the request; the request payload stays the only cue for
# the input type and the case.  Wrappers are spread evenly over a family's
# cases, so a word that fits every case ("parse" in S1.6) is allowed; a word
# that names one case or an outcome is not.
IDENTITY_REVEALING = _revealing(
    r"symbols?|ids?|identifiers?|ensembl|ensg\w*|accessions?|multiplex\w*|"
    r"nodes?|mouse|species|checks?|harness|blocks?|versions?|transcripts?|"
    r"alias\w*|corresponding|whether"
)
ALIGNMENT_REVEALING = _revealing(
    r"same|different|identical|distinct|match\w*|alias\w*|symbols?|ids?|"
    r"ensembl|whether|external|seeds?|modules?"
)
LAYER_REVEALING = _revealing(r"filter\w*|famil\w*")
LAYER_FAMILY_REVEALING = _revealing(
    r"supported|unsupported|yes|no|claims?|famil\w*|edges?|lists?|listed"
)
METRIC_REVEALING = _revealing(
    r"higher|lower|stronger|weaker|closer|farther|percentile|distance|"
    r"similarity|correlation|ranks?|scores?|statements?|claims?"
)
MODULE_REVEALING = _revealing(r"clades?|seeds?")

# Step wrappers must say that a later step will use the answer; direct
# wrappers must not.  The cue is the only thing the wrapper is meant to teach.
# "the following request" points at the text, not at a later step, so
# "following" is not a cue; a held-out direct wrapper uses it.
LATER_STEP_CUES = re.compile(
    r"\b("
    r"next|later|afterwards?|after\s+that|downstream|pipeline|workflow|"
    r"stages?|steps?|proceed\w*|continu\w*|subsequent|"
    r"follow(?:s|ed|-up)?|feeds?|"
    r"larger|bigger|broader|upcoming|rest|remaining|prerequisite|"
    r"input\s+(?:for|to)|hand(?:s|ed|ing)?|builds?\s+on|moves?\s+on|move\s+to|"
    r"go(?:es)?\s+(?:on|further)|further|carr(?:y|ies)\s+on|then|ongoing|"
    r"intermediate|mid-\w+|multi-step|unblocks?|waiting|waits|paused|"
    r"groundwork|set\s+up|reus\w*|chain|advance\w*|moving|first|"
    r"depends?|relies|rely|part\s+of"
    r")\b",
    re.IGNORECASE,
)

# An identity wrapper without a colon runs straight into the request, which
# is either a noun phrase or a "whether" clause, so the verb must take both.
IDENTITY_LEAD_INS = (
    ": ",
    "tell me ",
    "find out ",
    "work out ",
    "figure out ",
    "determine ",
    "determine for me ",
    "establish ",
    "clarify ",
    "let me know ",
    "i would like to know ",
    "i want to know ",
)


@dataclass(frozen=True, slots=True)
class WrapperBank:
    """How one family renders, and what its wrappers may not say."""

    sections: tuple[str, ...]
    style: str
    revealing: re.Pattern[str]
    sample_requests: tuple[str, ...]
    subset_of: str | None = None
    # S1.6 joins wrapper and request in one sentence, and its builder caps
    # an instruction sentence at 20 words; its longest request has 10.
    maximum_wrapper_words: int | None = None


WRAPPER_BANKS = {
    # S1.1 and S1.2 share one bank.
    "identity": WrapperBank(
        sections=("S1.1", "S1.2"),
        style="identity",
        revealing=IDENTITY_REVEALING,
        sample_requests=(
            "the human Ensembl gene ID for TP53",
            "the human Ensembl gene ID for the mouse gene Trp53",
            "whether the human gene TP53 is a node in full_brain_multiplex_v1",
            "the human gene symbol corresponding to ENSG00000141510",
        ),
    ),
    # S1.3 wraps imperative requests and the user's multi-turn replies, so
    # its wrappers must end at the request.
    "identity_s1_3": WrapperBank(
        sections=("S1.3",),
        style="colon_end",
        revealing=IDENTITY_REVEALING,
        sample_requests=(
            "give the human Ensembl gene ID for TP53",
            "give the human Ensembl gene ID for TP53 in multiplex "
            "full_brain_multiplex_v1",
            "Use ENSG00000141510",
        ),
        subset_of="identity",
    ),
    "alignment": WrapperBank(
        sections=("S1.4",),
        style="colon_end",
        revealing=ALIGNMENT_REVEALING,
        sample_requests=(
            "whether ENSG00000141510 in module group_431bc528 and the typed "
            "seed TP53 name the same gene",
        ),
    ),
    "layer": WrapperBank(
        sections=("S1.6",),
        style="colon_end",
        revealing=LAYER_REVEALING,
        sample_requests=(
            "parse this layer tag: bulkPEN:brain:hippocampus",
            "filter three layers by exact layer family: scpen. Tags: "
            "HumanNetV3:cocitation, bulkPEN:brain:hippocampus",
        ),
        maximum_wrapper_words=10,
    ),
    "layer_family": WrapperBank(
        sections=("S1.7",),
        style="newline",
        revealing=LAYER_FAMILY_REVEALING,
        sample_requests=(
            "Which listed layers are in the scpen family? HumanNetV3:string_ppi",
        ),
    ),
    "metric": WrapperBank(
        sections=("S1.8",),
        style="newline",
        revealing=METRIC_REVEALING,
        sample_requests=(
            'Evaluate this statement as true or false: a lower value for '
            '"hop_count" means weaker proximity.',
        ),
    ),
    "module": WrapperBank(
        sections=("S1.9",),
        style="newline",
        revealing=MODULE_REVEALING,
        sample_requests=(
            "Parse module ID mentor_ev:full_brain_all_genes_v1:clade_001514.",
        ),
    ),
}
# S1.2 keeps its own held-out wrappers although it trains on the shared bank.
HELD_OUT_SOURCES = (
    ("s1_1_corpus", "IDENTITY_PROMPTS"),
    ("s1_2_corpus", "IDENTITY_PROMPTS"),
    ("s1_4_corpus", "ALIGNMENT_PROMPTS"),
    ("s1_6_corpus", "LAYER_PROMPTS"),
    ("s1_7_corpus", "FAMILY_PROMPTS"),
    ("s1_8_corpus", "PROMPT_TEMPLATES"),
    ("s1_9_corpus", "MODULE_PROMPTS"),
)
NAME_BANKS = ("metric_name_variants", "shared_aliases")
PROMPT_BANK_KEYS = [
    "schema_version",
    "description",
    *WRAPPER_BANKS,
    *NAME_BANKS,
]

HELD_OUT_JACCARD_LIMIT = 0.6
HELD_OUT_SHARED_NGRAM = 5
HELD_OUT_NAME_JACCARD_LIMIT = 0.67
# S1.8 caps a sentence at 20 words, and its longest request leaves room for
# a name of four words.
MAXIMUM_NEW_NAME_WORDS = 4
CLAIM_PHRASE_LISTS = (
    "_STATUS_CLAIM_PHRASES",
    "_DECISION_CLAIM_PHRASES",
    "_ANSWER_SCHEMA_PHRASES",
    "_PREMATURE_OUTCOME_PHRASES",
)
BUILDER_MODULES = (
    "s1_1_corpus",
    "s1_2_corpus",
    "s1_3_corpus",
    "s1_4_corpus",
    "s1_6_corpus",
    "s1_7_corpus",
    "s1_8_corpus",
    "s1_9_corpus",
)
# A shared alias stays generic; a name that picks one method is a variant.
METHOD_TOKENS = frozenset(
    {
        "spearman",
        "pearson",
        "rho",
        "r",
        "rank",
        "ranked",
        "linear",
        "monotonic",
        "ordinal",
        "moment",
        "kendall",
        "tau",
    }
)


# Optional training reasoning, by key. S1 v4 stated the S1.1.10 status rule
# (absent gives not_found). S1 v5 makes an absent gene resolved, so its
# sentence states the membership answer instead. S1 v5 also states, for
# S1.2.4 in each framing, that a gene without a symbol is still usable.
REASONING_KEYS = (
    "s1_1_absent_membership",
    "s1_2_no_symbol_step",
    "s1_2_no_symbol_direct",
)


class PromptBankError(ValueError):
    """Report an invalid, unpinned, or leaking S1 prompt bank."""


def validate_identity_completion_reasoning(
    text: Any, *, fields: Sequence[str] = ("symbol", "gene_id"),
) -> None:
    """Check formatting and declared slot counts without limiting reasoning length.

    A repeated field explicitly requires that many uses of its binding.
    """

    if (
        not isinstance(text, str) or text != text.strip()
        or "\n" in text or not text.endswith(".")
    ):
        raise PromptBankError("completion reasoning must be one trimmed line")
    slots = tuple("{" + field + "}" for field in fields)
    if any(text.count(slot) != slots.count(slot) for slot in slots):
        raise PromptBankError(
            "completion reasoning must bind each declared field occurrence: "
            + ", ".join(fields)
        )
    outside = text
    for slot in slots:
        outside = outside.replace(slot, "")
    if "{" in outside or "}" in outside:
        raise PromptBankError("completion reasoning contains an unsupported slot")


def _builder(name: str) -> Any:
    # Imported lazily because the builders import this module.
    return importlib.import_module(f"{__package__}.{name}")


def _normalized(text: str) -> str:
    return " ".join(text.replace(REQUEST_SLOT, " ").casefold().split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", _normalized(text)))


def _ngrams(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def _held_out(sources: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            template
            for module, constant in sources
            for split in ("validation", "test")
            for framing in FRAMINGS
            for template in getattr(_builder(module), constant)[split][framing]
        )
    )


def held_out_wordings() -> tuple[str, ...]:
    """Return every validation and test wrapper of every S1 family.

    Families share wording ("The workflow first needs this result."), so each
    training bank is checked against all of them.
    """

    return _held_out(HELD_OUT_SOURCES)


def held_out_identity_wordings() -> tuple[str, ...]:
    """Return the validation and test wrappers of S1.1, S1.2, and S1.3."""

    return _held_out(HELD_OUT_SOURCES[:2])


def claim_phrases() -> tuple[str, ...]:
    """Return the builders' answer-revealing phrase lists, flattened."""

    phrases: set[str] = set()
    for name in BUILDER_MODULES:
        module = _builder(name)
        for list_name in CLAIM_PHRASE_LISTS:
            value = getattr(module, list_name, None)
            if value is None:
                continue
            items = (
                [item for group in value.values() for item in group]
                if isinstance(value, Mapping)
                else list(value)
            )
            phrases.update(item.strip().casefold() for item in items if item.strip())
    return tuple(sorted(phrases))


def _style_problems(template: str, style: str) -> list[str]:
    problems = []
    if style == "newline":
        line, _, rest = template.partition("\n")
        if rest != REQUEST_SLOT or not line or line != line.strip():
            problems.append("must be one sentence, a newline, then {request}")
        if line[-1:] != ".":
            problems.append("must end its sentence with a period")
        return problems
    if "\n" in template or template != template.strip():
        problems.append("must be one trimmed line")
    if style == "colon_end":
        if not template.endswith(S1_3_ENDING):
            problems.append("must end at the request after a colon")
        return problems
    if template[-1:] not in {".", "?"}:
        problems.append("must end with a period or question mark")
    prefix = template.split(REQUEST_SLOT)[0].casefold()
    if not prefix.endswith(IDENTITY_LEAD_INS):
        problems.append("must reach the slot through a colon or a known verb")
    return problems


def _check_wrapper(
    template: str,
    framing: str,
    spec: WrapperBank,
    phrases: Sequence[str],
) -> list[str]:
    if not isinstance(template, str) or not template:
        return ["is not a non-empty string"]
    problems = []
    if template.count(REQUEST_SLOT) != 1:
        problems.append("must contain exactly one {request} slot")
    outside = template.replace(REQUEST_SLOT, "")
    if "{" in outside or "}" in outside:
        problems.append("contains a brace outside the {request} slot")
    if len(template) > MAXIMUM_WRAPPER_LENGTH:
        problems.append("is too long")
    if not template[:1].isupper():
        problems.append("must start with a capital letter")
    problems.extend(_style_problems(template, spec.style))
    if spec.maximum_wrapper_words is not None:
        words = len(re.findall(r"[A-Za-z]+", outside))
        if words > spec.maximum_wrapper_words:
            problems.append(
                f"has {words} words, above {spec.maximum_wrapper_words}"
            )
    revealing = spec.revealing.search(outside)
    if revealing:
        problems.append(f"names {revealing.group(0)!r}")
    lowered = " ".join(outside.casefold().split())
    claim = next((phrase for phrase in phrases if phrase in lowered), None)
    if claim is not None:
        problems.append(f"contains the answer phrase {claim!r}")
    cue = LATER_STEP_CUES.search(outside)
    if framing == "step" and cue is None:
        problems.append("has no later-step cue")
    if framing == "direct" and cue is not None:
        problems.append(f"implies a later step with {cue.group(0)!r}")
    for request in spec.sample_requests:
        if template.format(request=request).count(request) != 1:
            problems.append("does not render the request exactly once")
            break
    return problems


def validate_wrapper_bank(
    bank: Mapping[str, Sequence[str]],
    *,
    spec: WrapperBank,
    held_out: Iterable[str],
    phrases: Sequence[str] = (),
    name: str = "wrapper",
) -> dict[str, Any]:
    """Validate one training wrapper bank and return its audit record."""

    if list(bank) != list(FRAMINGS):
        raise PromptBankError(f"the {name} bank must list direct, then step")
    counts = {framing: len(bank[framing]) for framing in FRAMINGS}
    if counts["direct"] != counts["step"]:
        raise PromptBankError(f"the {name} bank framings differ in size")
    if counts["direct"] < MINIMUM_WRAPPERS:
        raise PromptBankError(
            f"the {name} bank needs at least {MINIMUM_WRAPPERS} wrappers "
            "per framing"
        )

    held_out = tuple(held_out)
    held_out_normalized = {_normalized(template) for template in held_out}
    held_out_tokens = [_tokens(template) for template in held_out]
    held_out_ngrams = set().union(
        *(_ngrams(tokens, HELD_OUT_SHARED_NGRAM) for tokens in held_out_tokens)
    )
    problems: list[str] = []
    seen: dict[str, str] = {}
    closest = 0.0
    for framing in FRAMINGS:
        for index, template in enumerate(bank[framing]):
            label = f"{name} {framing}[{index}] {template!r}"
            problems.extend(
                f"{label} {problem}"
                for problem in _check_wrapper(template, framing, spec, phrases)
            )
            normalized = _normalized(template)
            if normalized in seen:
                problems.append(f"{label} repeats {seen[normalized]}")
            seen.setdefault(normalized, label)
            if normalized in held_out_normalized:
                problems.append(f"{label} is a held-out wrapper")
            tokens = _tokens(template)
            if _ngrams(tokens, HELD_OUT_SHARED_NGRAM) & held_out_ngrams:
                problems.append(
                    f"{label} shares a {HELD_OUT_SHARED_NGRAM}-word run "
                    "with a held-out wrapper"
                )
            for other in held_out_tokens:
                overlap = _jaccard(tokens, other)
                closest = max(closest, overlap)
                if overlap >= HELD_OUT_JACCARD_LIMIT:
                    problems.append(
                        f"{label} is too close to a held-out wrapper "
                        f"(Jaccard {overlap:.2f})"
                    )
                    break
    if problems:
        raise PromptBankError(
            f"the {name} bank is invalid:\n  " + "\n  ".join(problems)
        )
    return {
        "wrapper_counts": counts,
        "held_out_wrappers_checked": len(held_out),
        "maximum_held_out_jaccard": round(closest, 4),
    }


def validate_identity_bank(
    bank: Mapping[str, Sequence[str]],
    *,
    held_out: Iterable[str],
) -> dict[str, Any]:
    """Validate an S1.1/S1.2 training bank (kept for focused tests)."""

    return validate_wrapper_bank(
        bank,
        spec=WRAPPER_BANKS["identity"],
        held_out=held_out,
        phrases=claim_phrases(),
        name="identity",
    )


def _name_key(value: str) -> str:
    return " ".join(re.sub(r"[-_]", " ", value.casefold()).split())


def _validated_names(
    payload: Mapping[str, Any], *, held_out_limit: float = HELD_OUT_NAME_JACCARD_LIMIT,
) -> dict[str, Any]:
    """Validate the S1.8 training names against every held-out name."""

    builder = _builder("s1_8_corpus")
    variants = payload["metric_name_variants"]
    aliases = payload["shared_aliases"]
    legacy_variants = builder.NAME_VARIANTS["train"]
    legacy_aliases = builder.SHARED_ALIASES["train"]
    problems: list[str] = []
    if list(variants) != list(builder.METRICS):
        raise PromptBankError("metric_name_variants must list every metric in order")
    for metric, names in variants.items():
        if list(names[: len(legacy_variants[metric])]) != list(
            legacy_variants[metric]
        ):
            problems.append(f"{metric} must start with its v2 training names")
        if len(names) <= len(legacy_variants[metric]):
            problems.append(f"{metric} adds no training names")
    if list(aliases)[: len(legacy_aliases)] != list(legacy_aliases) or any(
        aliases[alias] != alias for alias in legacy_aliases
    ):
        problems.append("shared_aliases must start with the v2 training aliases")
    for alias, family in aliases.items():
        if family not in legacy_aliases:
            problems.append(f"alias {alias!r} names an unknown family {family!r}")
        method = METHOD_TOKENS & set(_tokens(alias))
        if method:
            problems.append(f"alias {alias!r} names a method: {sorted(method)!r}")

    held_out = [
        value
        for split in ("validation", "test")
        for value in (
            *(name for names in builder.NAME_VARIANTS[split].values() for name in names),
            *builder.SHARED_ALIASES[split],
            *builder.AMBIGUOUS_INPUTS[split],
            *builder.UNKNOWN_INPUTS[split],
        )
    ]
    held_out_keys = {_name_key(value) for value in held_out}
    held_out_tokens = [_tokens(_name_key(value)) for value in held_out]
    held_out_ngrams = set().union(
        *(_ngrams(tokens, HELD_OUT_SHARED_NGRAM) for tokens in held_out_tokens)
    )
    legacy_names = {
        *(name for names in legacy_variants.values() for name in names),
        *legacy_aliases,
        *builder.METRICS,
        *builder.AMBIGUOUS_INPUTS["train"],
        *builder.UNKNOWN_INPUTS["train"],
    }
    # Canonical keys ("spearman_distance") are a separate input kind; v2
    # already trains on both the key and its spelled-out variant.
    training = [
        *(name for names in variants.values() for name in names),
        *aliases,
        *builder.AMBIGUOUS_INPUTS["train"],
        *builder.UNKNOWN_INPUTS["train"],
    ]
    seen: set[str] = set()
    closest = 0.0
    for value in training:
        key = _name_key(value)
        if not value or value != value.strip() or "\n" in value:
            problems.append(f"name {value!r} is not one trimmed line")
        if key in seen:
            problems.append(f"name {value!r} repeats another training name")
        seen.add(key)
        if key in held_out_keys:
            problems.append(f"name {value!r} is a held-out name")
        if value in legacy_names:
            continue
        if len(re.findall(r"[A-Za-z]+", value)) > MAXIMUM_NEW_NAME_WORDS:
            problems.append(
                f"name {value!r} has more than {MAXIMUM_NEW_NAME_WORDS} words"
            )
        if _ngrams(_tokens(key), HELD_OUT_SHARED_NGRAM) & held_out_ngrams:
            problems.append(f"name {value!r} shares a held-out 5-word run")
        for other in held_out_tokens:
            overlap = _jaccard(_tokens(key), other)
            closest = max(closest, overlap)
            if overlap >= held_out_limit:
                problems.append(
                    f"name {value!r} is too close to a held-out name "
                    f"(Jaccard {overlap:.2f})"
                )
                break
    if problems:
        raise PromptBankError(
            "the S1.8 name banks are invalid:\n  " + "\n  ".join(problems)
        )
    return {
        "metric_variant_counts": {
            metric: len(names) for metric, names in variants.items()
        },
        "shared_alias_count": len(aliases),
        "held_out_names_checked": len(held_out),
        "maximum_new_name_jaccard": round(closest, 4),
    }


def validate_metric_names(
    variants: Mapping[str, Sequence[str]], aliases: Mapping[str, str],
) -> dict[str, Any]:
    """Require revised metric names to stay below 0.6 held-out overlap.

    Published v3/v4 banks retain their original 0.67 boundary. Revised
    training banks use the repository's stricter wording boundary here.
    """

    return _validated_names(
        {"metric_name_variants": variants, "shared_aliases": aliases},
        held_out_limit=HELD_OUT_JACCARD_LIMIT,
    )


def load_prompt_banks(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read a pinned bank file and check its schema and digest."""

    if not isinstance(pin, Mapping) or list(pin) != ["path", "sha256"]:
        raise PromptBankError("a prompt-bank pin must list path, then sha256")
    path = (REPOSITORY_ROOT / str(pin["path"])).resolve()
    if not path.is_file():
        raise PromptBankError(f"the prompt bank {path} does not exist")
    if file_sha256(path) != pin["sha256"]:
        raise PromptBankError(f"the prompt bank {path} does not match its pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or list(payload) not in (
        PROMPT_BANK_KEYS,
        [*PROMPT_BANK_KEYS, "reasoning"],
        [*PROMPT_BANK_KEYS, "reasoning", "claim_paraphrases"],
    ):
        raise PromptBankError("the prompt-bank keys changed")
    reasoning = payload.get("reasoning", {})
    if not isinstance(reasoning, Mapping) or not set(reasoning) <= set(
        REASONING_KEYS
    ):
        raise PromptBankError("the prompt-bank reasoning keys changed")
    for key, text in reasoning.items():
        if (
            not isinstance(text, str)
            or not text
            or text != text.strip()
            or "\n" in text
            or "{" in text
            or "}" in text
        ):
            raise PromptBankError(f"the {key} reasoning must be nonempty, trimmed, and contain no slots or newlines")
    if payload["schema_version"] != PROMPT_BANK_SCHEMA_VERSION:
        raise PromptBankError("the prompt-bank schema changed")
    for name in WRAPPER_BANKS:
        bank = payload[name]
        if not isinstance(bank, Mapping) or not all(
            isinstance(bank.get(framing), list) for framing in FRAMINGS
        ):
            raise PromptBankError(f"the {name} bank must hold two wrapper lists")
    if not isinstance(payload["metric_name_variants"], Mapping) or not isinstance(
        payload["shared_aliases"], Mapping
    ):
        raise PromptBankError("the S1.8 name banks must be mappings")
    return payload


def load_choice_replies(pin: Mapping[str, Any]) -> tuple[str, ...]:
    """Load frozen, scope-neutral S1.3.8 selections without held-out wording."""

    if not isinstance(pin, Mapping) or list(pin) != ["path", "sha256"]:
        raise PromptBankError("a choice-reply pin must list path, then sha256")
    path = (REPOSITORY_ROOT / str(pin["path"])).resolve()
    if not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise PromptBankError("the choice-reply file does not match its pin")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or list(payload) != ["schema_version", "replies"]
        or payload["schema_version"] != "mentor-rl-world-model-s1-choice-replies-v1"
        or not isinstance(payload["replies"], list)
        or len(payload["replies"]) < 16
    ):
        raise PromptBankError("the choice-reply schema requires at least 16 replies")
    replies = payload["replies"]
    held_out = (*held_out_wordings(), *_builder("s1_3_corpus").CHOICE_REPLIES)
    held_tokens = [_tokens(value.replace("{gene_id}", "")) for value in held_out]
    seen = set()
    for reply in replies:
        if (
            not isinstance(reply, str) or reply != reply.strip()
            or "\n" in reply or not reply.endswith(".")
            or reply.count("{gene_id}") != 1
            or "{" in reply.replace("{gene_id}", "")
            or "}" in reply.replace("{gene_id}", "")
            or LATER_STEP_CUES.search(reply)
            or re.search(r"\b(stop|defer|only|standalone|complete\w*|finish\w*)\b", reply, re.I)
        ):
            raise PromptBankError("choice replies must select one ID without a scope cue")
        tokens = _tokens(reply.replace("{gene_id}", ""))
        normalized = " ".join(tokens)
        if normalized in seen:
            raise PromptBankError("choice replies must be distinct")
        seen.add(normalized)
        if any(
            _jaccard(tokens, other) >= HELD_OUT_JACCARD_LIMIT
            or _ngrams(tokens, HELD_OUT_SHARED_NGRAM) & _ngrams(other, HELD_OUT_SHARED_NGRAM)
            for other in held_tokens
        ):
            raise PromptBankError("choice replies copy held-out wording")
    return tuple(replies)


def _validated_banks(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every bank and return the audit records by bank name."""

    held_out = held_out_wordings()
    phrases = claim_phrases()
    audits: dict[str, Any] = {}
    for name, spec in WRAPPER_BANKS.items():
        bank = {framing: tuple(payload[name][framing]) for framing in FRAMINGS}
        audits[name] = validate_wrapper_bank(
            bank, spec=spec, held_out=held_out, phrases=phrases, name=name
        )
        if spec.subset_of is None:
            continue
        for framing in FRAMINGS:
            outside = [
                item
                for item in bank[framing]
                if item not in payload[spec.subset_of][framing]
            ]
            if outside:
                raise PromptBankError(
                    f"{name} {framing} wrappers are not in the "
                    f"{spec.subset_of} bank: {outside!r}"
                )
    audits["s1_8_names"] = _validated_names(payload)
    if "claim_paraphrases" in payload:
        audits["s1_7_claims"] = validate_claim_paraphrases(payload["claim_paraphrases"])
    return audits


def validate_claim_paraphrases(bank: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Check the frozen S1.7 claim phrases for coverage and held-out overlap.

    Phrases state the requested claim; they must not name a family, a
    verdict, or a tool. Their meanings are reviewed as frozen data.
    """

    builder = _builder("s1_7_corpus")
    if (
        not isinstance(bank, Mapping)
        or list(bank) != list(builder.CLAIM_VOCABULARY)
    ):
        raise PromptBankError("the S1.7 claim bank must keep the claim vocabulary")
    held_out = [
        phrase
        for split in ("validation", "test")
        for values in builder.CLAIM_PARAPHRASES[split].values()
        for phrase in values
    ]
    held_tokens = [_tokens(phrase) for phrase in held_out]
    held_runs = set().union(*(_ngrams(tokens, 5) for tokens in held_tokens))
    forbidden = re.compile(
        r"\b(ppi|tf_target|bulk_pen|scpen|HumanNetV3|"
        r"(?:other|coexpression)\s+family|"
        r"supports?|unsupported|yes|no|true|false|tools?|status|provenance)\b",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    closest = 0.0
    for claim, phrases in bank.items():
        if not isinstance(phrases, list) or len(phrases) != 32:
            raise PromptBankError(f"the {claim} bank must have 32 phrases")
        for phrase in phrases:
            if (
                not isinstance(phrase, str) or phrase != phrase.strip()
                or not phrase or re.search(r"[\n{}.!?:]", phrase)
                or len(re.findall(r"[A-Za-z]+", phrase)) > 9
                or forbidden.search(phrase)
            ):
                raise PromptBankError(f"invalid S1.7 claim phrase: {phrase!r}")
            normalized = _normalized(phrase)
            if normalized in seen:
                raise PromptBankError(f"repeated S1.7 claim phrase: {phrase!r}")
            seen.add(normalized)
            tokens = _tokens(phrase)
            if _ngrams(tokens, 5) & held_runs:
                raise PromptBankError(
                    f"S1.7 phrase shares a held-out 5-word run: {phrase!r}"
                )
            for held in held_tokens:
                overlap = _jaccard(tokens, held)
                closest = max(closest, overlap)
                if overlap >= HELD_OUT_JACCARD_LIMIT:
                    raise PromptBankError(
                        "S1.7 phrase is too close to held-out wording "
                        f"({overlap:.2f}): {phrase!r}"
                    )
    return {
        "phrase_counts": {claim: len(phrases) for claim, phrases in bank.items()},
        "held_out_phrases_checked": len(held_out),
        "maximum_held_out_jaccard": round(closest, 4),
    }


def claim_paraphrases(
    pin: Mapping[str, Any],
    legacy: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return revised S1.7 training phrases with the frozen held-out phrases."""

    bank = load_prompt_banks(pin).get("claim_paraphrases")
    if bank is not None:
        validate_claim_paraphrases(bank)
    return {
        split: {
            claim: tuple(phrases)
            for claim, phrases in (
                bank if split == "train" and bank is not None else values
            ).items()
        }
        for split, values in legacy.items()
    }


def wrapper_prompts(
    pin: Mapping[str, Any],
    legacy: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    bank_name: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return a builder bank: pinned training wrappers, legacy held-out ones."""

    if bank_name not in WRAPPER_BANKS:
        raise PromptBankError(f"unknown prompt bank {bank_name!r}")
    payload = load_prompt_banks(pin)
    _validated_banks(payload)
    return {
        "train": {
            framing: tuple(payload[bank_name][framing]) for framing in FRAMINGS
        },
        "validation": {
            framing: tuple(legacy["validation"][framing]) for framing in FRAMINGS
        },
        "test": {framing: tuple(legacy["test"][framing]) for framing in FRAMINGS},
    }


def identity_prompts(
    pin: Mapping[str, Any],
    legacy: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    bank_name: str = "identity",
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return the S1.1-S1.3 builder bank."""

    return wrapper_prompts(pin, legacy, bank_name=bank_name)


def metric_names(
    pin: Mapping[str, Any],
    legacy_variants: Mapping[str, Mapping[str, Sequence[str]]],
    legacy_aliases: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, dict[str, Any]]]:
    """Return S1.8 name variants and shared aliases with pinned training names.

    Each new training alias copies the convention card of the v2 alias it
    names, so the bank adds wording, not conventions.
    """

    payload = load_prompt_banks(pin)
    _validated_banks(payload)
    variants = {
        split: {
            metric: tuple(names) for metric, names in legacy_variants[split].items()
        }
        for split in ("validation", "test")
    }
    variants = {
        "train": {
            metric: tuple(names)
            for metric, names in payload["metric_name_variants"].items()
        },
        **variants,
    }
    aliases = {
        "train": {
            alias: legacy_aliases["train"][family]
            for alias, family in payload["shared_aliases"].items()
        },
        "validation": dict(legacy_aliases["validation"]),
        "test": dict(legacy_aliases["test"]),
    }
    return variants, aliases


def bank_reasoning(pin: Mapping[str, Any], key: str) -> str | None:
    """Return pinned training reasoning, or None when absent."""

    if key not in REASONING_KEYS:
        raise PromptBankError(f"unknown reasoning key {key!r}")
    return load_prompt_banks(pin).get("reasoning", {}).get(key)


def prompt_bank_manifest(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest record for a pinned bank and its audit."""

    payload = load_prompt_banks(pin)
    return {
        "path": str(pin["path"]),
        "sha256": str(pin["sha256"]),
        "schema_version": PROMPT_BANK_SCHEMA_VERSION,
        "audit": _validated_banks(payload),
    }
