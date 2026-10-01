"""Build S1 rows with the shared supervised trace contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from .world_model_contracts import (
    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
    SUPERVISED_EXPLANATION_SCHEMA_VERSION,
    WORLD_MODEL_ANSWER_SCHEMA_VERSION,
    ExplanationReceipt,
    SupervisedTraceRecord,
    WorldModelAnswer,
    serialize_world_model_answer,
    stable_json_sha256,
    text_sha256,
)
from .world_model_prompts import s1_shared_trace_prompt_contract
from .world_model_s0_tools import (
    ENSEMBL_PATTERN,
    ENSEMBL_TOOL_NAME,
    ENSEMBL_RELEASE,
    SYMBOL_TOOL_NAME,
    TOOL_CONTRACT_VERSION_V2,
    identifier_tool_definitions,
)


S1_SOURCE_REFS = (
    "agents/world_model_v2_curriculum/S1-coordinate-system.md",
)
S1_1_QUESTION_FAMILY = "s1_1_symbol_to_ensembl"
S1_1_FORMS = (
    "direct",
    "inside_task",
    "already_resolved",
    "absent_symbol",
)
S1_2_QUESTION_FAMILY = "s1_2_ensembl_to_symbol"
S1_2_FORMS = (
    "direct",
    "inside_task",
    "already_resolved",
    "version_suffix",
    "no_symbol",
    "invalid_input",
)
S1_2_TASK_CONTEXTS = ("direct", "inside_task")
S1_2_INVALID_KINDS = ("transcript", "mouse_gene", "malformed")
S1_3_QUESTION_FAMILY = "s1_3_ambiguous_symbol"
S1_3_FORMS = ("direct", "inside_task", "already_marked")
ENSEMBL_VERSION_PATTERN = re.compile(
    r"^(ENSG[0-9]{11})\.([1-9][0-9]*)$"
)
ENSEMBL_TRANSCRIPT_PATTERN = re.compile(r"^ENST[0-9]{11}$")
ENSEMBL_MOUSE_GENE_PATTERN = re.compile(r"^ENSMUSG[0-9]{11}$")


@dataclass(frozen=True)
class S1ExplanationTemplate:
    """Define one reviewed S1 explanation template."""

    template_id: str
    text: str
    rule_ids: tuple[str, ...]


def _templates(
    action_path: str,
    texts: tuple[str, str, str],
    rule_ids: tuple[str, ...],
) -> tuple[S1ExplanationTemplate, ...]:
    """Create three templates for one S1 action path."""

    return tuple(
        S1ExplanationTemplate(
            template_id=f"s1_1_{action_path}_v1_{index}",
            text=text,
            rule_ids=rule_ids,
        )
        for index, text in enumerate(texts, start=1)
    )


S1_1_EXPLANATION_TEMPLATES = {
    "symbol_lookup_direct": _templates(
        "symbol_lookup_direct",
        (
            (
                "The prompt gives only the symbol {gene_symbol}, so I will "
                "copy it into the pinned registry tool before I answer."
            ),
            (
                "A symbol does not establish one Ensembl ID, so I will look "
                "up {gene_symbol} in the pinned registry."
            ),
            (
                "The task needs an exact ID for {gene_symbol}, so I will use "
                "the registry tool before I make a mapping claim."
            ),
        ),
        (
            "s1_rule_resolve_symbol_with_registry",
            "s1_rule_copy_visible_input",
        ),
    ),
    "symbol_lookup_task": _templates(
        "symbol_lookup_task",
        (
            (
                "The next task step needs an exact ID for {gene_symbol}, so "
                "I will resolve the visible symbol first."
            ),
            (
                "The seed is given only as {gene_symbol}, so I will use the "
                "pinned registry before the task continues."
            ),
            (
                "I cannot start the next operation with an unverified symbol, "
                "so I will look up {gene_symbol}."
            ),
        ),
        (
            "s1_rule_resolve_symbol_with_registry",
            "s1_rule_copy_visible_input",
            "s1_rule_resolve_before_next_step",
        ),
    ),
    "symbol_lookup_absent": _templates(
        "symbol_lookup_absent",
        (
            (
                "The prompt gives only the symbol {gene_symbol}, so I will "
                "copy it into the pinned registry tool before I answer."
            ),
            (
                "A symbol does not establish one Ensembl ID, so I will look "
                "up {gene_symbol} in the pinned registry."
            ),
            (
                "The task needs an exact ID for {gene_symbol}, so I will use "
                "the registry tool before I make a mapping claim."
            ),
        ),
        (
            "s1_rule_resolve_symbol_with_registry",
            "s1_rule_copy_visible_input",
        ),
    ),
    "resolved_direct": _templates(
        "resolved_direct",
        (
            (
                "The lookup maps {gene_symbol} to {gene_id}, so I can return "
                "the verified gene and stop."
            ),
            (
                "The registry supplies one ID for {gene_symbol}, so the "
                "requested conversion is complete."
            ),
            (
                "The tool verifies {gene_id} for {gene_symbol}, so I can make "
                "that mapping claim and finish the task."
            ),
        ),
        (
            "s1_rule_use_verified_mapping",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "resolved_task": _templates(
        "resolved_task",
        (
            (
                "The lookup maps {gene_symbol} to {gene_id}, so I will keep "
                "the verified gene and continue the larger task."
            ),
            (
                "The registry resolves {gene_symbol} to one ID, so the next "
                "analysis step can use {gene_id}."
            ),
            (
                "The tool verifies {gene_id} for {gene_symbol}, so the task "
                "can continue with the Ensembl ID."
            ),
        ),
        (
            "s1_rule_use_verified_mapping",
            "s1_rule_continue_inside_task",
        ),
    ),
    "resolved_registry_block": _templates(
        "resolved_registry_block",
        (
            (
                "The labeled registry block already maps {gene_symbol} to "
                "{gene_id}, so no tool call is necessary."
            ),
            (
                "The prompt supplies an authoritative registry entry for "
                "{gene_symbol}, so I will use it and stop."
            ),
            (
                "The pinned registry block contains the required mapping, so "
                "I will not repeat the lookup."
            ),
        ),
        (
            "s1_rule_use_visible_registry_block",
            "s1_rule_avoid_unnecessary_tool_call",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "symbol_not_found": _templates(
        "symbol_not_found",
        (
            (
                "The registry has no match for {gene_symbol}, so I will make "
                "no ID claim and request another identifier."
            ),
            (
                "The lookup returns no Ensembl ID for {gene_symbol}, so I "
                "must defer without inventing a mapping."
            ),
            (
                "The tool cannot resolve {gene_symbol}, so no graph lookup "
                "can use it until the user supplies another identifier."
            ),
        ),
        (
            "s1_rule_make_no_unresolved_claim",
            "s1_rule_do_not_fabricate_gene_id",
            "s1_rule_defer_for_user_input",
        ),
    ),
}


def _s1_2_templates(
    action_path: str,
    texts: tuple[str, str, str],
    rule_ids: tuple[str, ...],
) -> tuple[S1ExplanationTemplate, ...]:
    """Create three S1.2 templates for one action path."""

    return tuple(
        S1ExplanationTemplate(
            template_id=f"s1_2_{action_path}_v1_{index}",
            text=text,
            rule_ids=rule_ids,
        )
        for index, text in enumerate(texts, start=1)
    )


S1_2_EXPLANATION_TEMPLATES = {
    "id_lookup_direct": _s1_2_templates(
        "id_lookup_direct",
        (
            (
                "The prompt gives one Ensembl gene ID, so I will look up "
                "{gene_id} before I name its symbol."
            ),
            (
                "A symbol is a registry fact, so I will look up {gene_id} "
                "before I answer."
            ),
            (
                "I need the registry entry for {gene_id}, so I will call "
                "the Ensembl ID tool."
            ),
        ),
        (
            "s1_rule_resolve_id_with_registry",
            "s1_rule_copy_visible_input",
        ),
    ),
    "id_lookup_task": _s1_2_templates(
        "id_lookup_task",
        (
            (
                "The next task step needs the symbol for {gene_id}, so I "
                "will check the pinned registry first."
            ),
            (
                "The seed uses an Ensembl ID, so I will resolve its symbol "
                "before the task continues."
            ),
            (
                "I will look up {gene_id} before I pass the gene to the "
                "next analysis step."
            ),
        ),
        (
            "s1_rule_resolve_id_with_registry",
            "s1_rule_resolve_before_next_step",
        ),
    ),
    "version_lookup": _s1_2_templates(
        "version_lookup",
        (
            (
                "The input has a version suffix, so I will remove it and "
                "look up {gene_id}."
            ),
            (
                "The registry tool accepts a canonical gene ID, so I will "
                "call it with {gene_id} and omit the suffix."
            ),
            (
                "The suffix does not change the gene, so I will normalize "
                "{input_gene_id} to {gene_id} before the lookup."
            ),
        ),
        (
            "s1_rule_remove_version_suffix",
            "s1_rule_call_with_canonical_gene_id",
        ),
    ),
    "no_symbol_lookup_direct": _s1_2_templates(
        "no_symbol_lookup_direct",
        (
            (
                "The prompt gives one Ensembl gene ID, so I will look up "
                "{gene_id} before I state whether it has a symbol."
            ),
            (
                "The registry can return a symbol or null, so I will look "
                "up {gene_id}."
            ),
            (
                "I cannot infer a symbol from the ID, so I will check the "
                "registry entry for {gene_id}."
            ),
        ),
        (
            "s1_rule_resolve_id_with_registry",
            "s1_rule_allow_null_symbol",
        ),
    ),
    "no_symbol_lookup_task": _s1_2_templates(
        "no_symbol_lookup_task",
        (
            (
                "The next step needs the registry result for {gene_id}, so "
                "I will check its symbol before I continue."
            ),
            (
                "The seed ID remains usable without a symbol, but I will "
                "check the registry before the task continues."
            ),
            (
                "I will look up {gene_id}; a null symbol will not prevent "
                "the next step from using the ID."
            ),
        ),
        (
            "s1_rule_resolve_id_with_registry",
            "s1_rule_allow_null_symbol",
            "s1_rule_continue_with_valid_gene_id",
        ),
    ),
    "resolved_direct": _s1_2_templates(
        "resolved_direct",
        (
            (
                "The registry maps {gene_id} to {gene_symbol}, so I can "
                "return the verified symbol and stop."
            ),
            (
                "The lookup verifies {gene_symbol} for {gene_id}, so the "
                "requested conversion is complete."
            ),
            (
                "The tool returns one symbol for {gene_id}, so I can make "
                "the mapping claim and finish."
            ),
        ),
        (
            "s1_rule_use_verified_mapping",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "resolved_task": _s1_2_templates(
        "resolved_task",
        (
            (
                "The registry maps {gene_id} to {gene_symbol}, so the larger "
                "task can continue with the verified gene."
            ),
            (
                "The lookup verifies {gene_symbol} for {gene_id}, so I will "
                "keep both values for the next step."
            ),
            (
                "The tool supplies the seed symbol {gene_symbol}, so the "
                "next analysis step can continue."
            ),
        ),
        (
            "s1_rule_use_verified_mapping",
            "s1_rule_continue_inside_task",
        ),
    ),
    "resolved_registry_block": _s1_2_templates(
        "resolved_registry_block",
        (
            (
                "The labeled registry block already maps {gene_id} to "
                "{gene_symbol}, so no tool call is necessary."
            ),
            (
                "The prompt supplies an authoritative registry entry for "
                "{gene_id}, so I will use it and stop."
            ),
            (
                "The pinned registry block contains the requested symbol, "
                "so I will not repeat the lookup."
            ),
        ),
        (
            "s1_rule_use_visible_registry_block",
            "s1_rule_avoid_unnecessary_tool_call",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "resolved_version": _s1_2_templates(
        "resolved_version",
        (
            (
                "The canonical lookup maps {gene_id} to {gene_symbol}, so I "
                "will return the gene without the input suffix."
            ),
            (
                "The registry verifies {gene_symbol} for {gene_id}; the "
                "version suffix does not change this gene."
            ),
            (
                "The normalized ID resolves to {gene_symbol}, so I can "
                "return the canonical gene and stop."
            ),
        ),
        (
            "s1_rule_use_verified_mapping",
            "s1_rule_return_canonical_gene_id",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "resolved_no_symbol_direct": _s1_2_templates(
        "resolved_no_symbol_direct",
        (
            (
                "The registry recognizes {gene_id} but returns no symbol, "
                "so I will keep the valid ID and stop."
            ),
            (
                "The lookup resolves the gene with a null symbol, so I will "
                "report that {gene_id} has no known symbol."
            ),
            (
                "The ID is valid even though its symbol is null, so I will "
                "answer with {gene_id} and make no symbol claim."
            ),
        ),
        (
            "s1_rule_null_symbol_is_resolved",
            "s1_rule_do_not_invent_symbol",
            "s1_rule_stop_when_task_complete",
        ),
    ),
    "resolved_no_symbol_task": _s1_2_templates(
        "resolved_no_symbol_task",
        (
            (
                "The registry recognizes {gene_id} with no symbol, so the "
                "larger task can continue with the valid ID."
            ),
            (
                "The null symbol is not a failure, so I will keep {gene_id} "
                "and continue the analysis."
            ),
            (
                "The ID is valid without a known symbol, so the next step "
                "can use {gene_id} without a label."
            ),
        ),
        (
            "s1_rule_null_symbol_is_resolved",
            "s1_rule_do_not_invent_symbol",
            "s1_rule_continue_with_valid_gene_id",
        ),
    ),
    "invalid_input": _s1_2_templates(
        "invalid_input",
        (
            (
                "The input {input_gene_id} {invalid_reason}, so I will not "
                "call the human gene lookup tool."
            ),
            (
                "{input_gene_id} {invalid_reason}. I must defer until the "
                "user gives a human Ensembl gene ID."
            ),
            (
                "This value {invalid_reason}, so I will request a "
                "human Ensembl gene ID without a tool call."
            ),
        ),
        (
            "s1_rule_validate_gene_id_before_call",
            "s1_rule_avoid_invalid_tool_call",
            "s1_rule_defer_for_user_input",
        ),
    ),
}


def _s1_3_templates(
    action_path: str,
    texts: tuple[str, str, str],
    rule_ids: tuple[str, ...],
) -> tuple[S1ExplanationTemplate, ...]:
    """Create three S1.3 templates for one action path."""

    return tuple(
        S1ExplanationTemplate(
            template_id=f"s1_3_{action_path}_v1_{index}",
            text=text,
            rule_ids=rule_ids,
        )
        for index, text in enumerate(texts, start=1)
    )


S1_3_EXPLANATION_TEMPLATES = {
    "symbol_lookup_direct": _s1_3_templates(
        "symbol_lookup_direct",
        tuple(
            template.text
            for template in S1_1_EXPLANATION_TEMPLATES[
                "symbol_lookup_direct"
            ]
        ),
        (
            "s1_rule_resolve_symbol_with_registry",
            "s1_rule_copy_visible_input",
        ),
    ),
    "symbol_lookup_task": _s1_3_templates(
        "symbol_lookup_task",
        tuple(
            template.text
            for template in S1_1_EXPLANATION_TEMPLATES[
                "symbol_lookup_task"
            ]
        ),
        (
            "s1_rule_resolve_symbol_with_registry",
            "s1_rule_copy_visible_input",
            "s1_rule_resolve_before_next_step",
        ),
    ),
    "ambiguous_direct": _s1_3_templates(
        "ambiguous_direct",
        (
            (
                "The lookup returns {candidate_count} candidates for "
                "{gene_symbol}, so I cannot select one and must defer."
            ),
            (
                "The registry does not give one ID for {gene_symbol}, so I "
                "will list every candidate and request a choice."
            ),
            (
                "The symbol {gene_symbol} is ambiguous across "
                "{candidate_count} IDs, so I will make no single-gene claim."
            ),
        ),
        (
            "s1_rule_list_all_ambiguous_candidates",
            "s1_rule_do_not_choose_candidate",
            "s1_rule_defer_for_user_input",
        ),
    ),
    "ambiguous_task": _s1_3_templates(
        "ambiguous_task",
        (
            (
                "The lookup leaves {gene_symbol} ambiguous across "
                "{candidate_count} IDs, so the larger task must defer."
            ),
            (
                "The task cannot continue because {gene_symbol} has "
                "{candidate_count} candidates, so I will request one ID."
            ),
            (
                "The registry returns multiple candidates for {gene_symbol}. "
                "I will list them all and defer the next task step."
            ),
        ),
        (
            "s1_rule_list_all_ambiguous_candidates",
            "s1_rule_do_not_choose_candidate",
            "s1_rule_defer_inside_task",
        ),
    ),
    "ambiguous_registry_block": _s1_3_templates(
        "ambiguous_registry_block",
        (
            (
                "The labeled registry block marks {gene_symbol} ambiguous "
                "with {candidate_count} candidates, so no tool call applies."
            ),
            (
                "The prompt supplies every registry candidate for "
                "{gene_symbol}, so I will not repeat the lookup or choose one."
            ),
            (
                "The authoritative block already shows the ambiguity, so I "
                "will list all candidates and defer without a tool call."
            ),
        ),
        (
            "s1_rule_use_visible_registry_block",
            "s1_rule_avoid_unnecessary_tool_call",
            "s1_rule_do_not_choose_candidate",
            "s1_rule_defer_for_user_input",
        ),
    ),
}


def _require_symbol(value: Any) -> str:
    """Return one nonempty gene symbol."""

    if not isinstance(value, str) or not value:
        raise ValueError("S1.1 requires one nonempty gene symbol")
    return value


def _require_gene_id(value: Any) -> str:
    """Return one canonical human Ensembl gene ID."""

    if not isinstance(value, str) or ENSEMBL_PATTERN.fullmatch(value) is None:
        raise ValueError("S1.1 requires one canonical Ensembl gene ID")
    return value


def _validate_explanation(explanation: str) -> None:
    """Check the S1 explanation sentence and word limits."""

    sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", explanation)
        if sentence
    ]
    word_counts = [
        len(re.findall(r"\b[\w'-]+\b", sentence))
        for sentence in sentences
    ]
    if (
        len(sentences) not in {1, 2}
        or sum(word_counts) > 40
        or any(count > 25 for count in word_counts)
    ):
        raise ValueError("An S1 explanation exceeds its size limit")


def _explanation(
    *,
    record_id: str,
    action_index: int,
    action_path: str,
    values: Mapping[str, Any],
    visible_input: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one deterministic S1 explanation and receipt."""

    templates = S1_1_EXPLANATION_TEMPLATES[action_path]
    template_index = int(
        stable_json_sha256(
            {"record_id": record_id, "action_index": action_index}
        )[:16],
        16,
    ) % len(templates)
    template = templates[template_index]
    explanation = template.text.format(**dict(values))
    _validate_explanation(explanation)
    receipt = ExplanationReceipt(
        explanation_contract_version=(
            SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ),
        explanation_template_id=template.template_id,
        explanation_template_sha256=text_sha256(template.text),
        explanation_rule_ids=template.rule_ids,
        explanation_source_refs=S1_SOURCE_REFS,
        explanation_input_sha256=stable_json_sha256(dict(visible_input)),
        explanation_sha256=text_sha256(explanation),
    )
    return explanation, receipt.to_dict()


def _resolved_answer(
    *,
    form: str,
    gene_symbol: str,
    gene_id: str,
) -> tuple[str, WorldModelAnswer]:
    """Return the final path and answer for one resolved symbol."""

    if form == "inside_task":
        return "resolved_task", WorldModelAnswer(
            status="resolved",
            result={"gene_id": gene_id, "gene_symbol": gene_symbol},
            continuation_decision="continue",
            summary=(
                f"<GENE:{gene_id}|{gene_symbol}> is the resolved human gene. "
                "The larger task can continue with this Ensembl ID."
            ),
        )
    if form == "already_resolved":
        action_path = "resolved_registry_block"
    else:
        action_path = "resolved_direct"
    return action_path, WorldModelAnswer(
        status="resolved",
        result={"gene_id": gene_id, "gene_symbol": gene_symbol},
        continuation_decision="stop",
        summary=f"<GENE:{gene_id}|{gene_symbol}> is the resolved human gene.",
    )


def build_s1_1_trace(
    *,
    record_id: str,
    question: str,
    form: str,
    gene_symbol: str,
    gene_id: str | None,
    tool_result: Mapping[str, Any] | None,
    split: str,
    provenance: Mapping[str, Any],
    identifier_registry_id: str,
) -> SupervisedTraceRecord:
    """Build one S1.1 symbol-to-Ensembl trace."""

    if form not in S1_1_FORMS:
        raise ValueError(f"The S1.1 form is invalid: {form!r}")
    gene_symbol = _require_symbol(gene_symbol)
    prompt = s1_shared_trace_prompt_contract()
    tools = tuple(
        identifier_tool_definitions(
            contract_version=TOOL_CONTRACT_VERSION_V2
        )
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": question},
    ]

    if form == "already_resolved":
        if tool_result is not None:
            raise ValueError("An already-resolved row cannot use a tool")
        resolved_id = _require_gene_id(gene_id)
        if (
            "registry" not in question.lower()
            or gene_symbol not in question
            or resolved_id not in question
        ):
            raise ValueError(
                "An already-resolved row needs a labeled registry block"
            )
        action_path, answer = _resolved_answer(
            form=form,
            gene_symbol=gene_symbol,
            gene_id=resolved_id,
        )
        explanation, receipt = _explanation(
            record_id=record_id,
            action_index=0,
            action_path=action_path,
            values={"gene_symbol": gene_symbol, "gene_id": resolved_id},
            visible_input={"question": question},
        )
    else:
        if not isinstance(tool_result, Mapping):
            raise ValueError("A lookup row requires one tool result")
        call_path = (
            "symbol_lookup_task"
            if form == "inside_task"
            else "symbol_lookup_absent"
            if form == "absent_symbol"
            else "symbol_lookup_direct"
        )
        call_explanation, call_receipt = _explanation(
            record_id=record_id,
            action_index=0,
            action_path=call_path,
            values={"gene_symbol": gene_symbol},
            visible_input={
                "question": question,
                "tool_name": SYMBOL_TOOL_NAME,
                "arguments": {"gene_symbol": gene_symbol},
            },
        )
        call_id = "call_" + stable_json_sha256(
            {"record_id": record_id, "action_index": 0}
        )[:16]
        messages.extend(
            [
                {
                    "role": "assistant",
                    "assistant_explanation": call_explanation,
                    "explanation_receipt": call_receipt,
                    "tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": SYMBOL_TOOL_NAME,
                            "arguments": {"gene_symbol": gene_symbol},
                        },
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": SYMBOL_TOOL_NAME,
                    "content": dict(tool_result),
                },
            ]
        )
        status = tool_result.get("status")
        if form == "absent_symbol":
            expected = {
                "status": "not_found",
                "gene_symbol": gene_symbol,
                "candidate_gene_ids": [],
            }
            if dict(tool_result) != expected or gene_id is not None:
                raise ValueError("An absent-symbol result is invalid")
            action_path = "symbol_not_found"
            answer = WorldModelAnswer(
                status="not_found",
                result={"gene_symbol": gene_symbol},
                continuation_decision="defer",
                summary=(
                    f"{gene_symbol} has no match in the Ensembl release 116 "
                    "registry, so no graph lookup can use it. Give another "
                    "symbol or Ensembl gene ID to continue."
                ),
            )
        else:
            if status != "resolved":
                raise ValueError("A resolved S1.1 tool result is invalid")
            resolved_id = _require_gene_id(gene_id)
            expected = {
                "status": "resolved",
                "gene_id": resolved_id,
                "gene_symbol": gene_symbol,
            }
            if dict(tool_result) != expected:
                raise ValueError("A resolved S1.1 tool result is invalid")
            action_path, answer = _resolved_answer(
                form=form,
                gene_symbol=gene_symbol,
                gene_id=resolved_id,
            )
        explanation, receipt = _explanation(
            record_id=record_id,
            action_index=1,
            action_path=action_path,
            values={
                "gene_symbol": gene_symbol,
                "gene_id": gene_id,
            },
            visible_input={
                "question": question,
                "tool_name": SYMBOL_TOOL_NAME,
                "arguments": {"gene_symbol": gene_symbol},
                "tool_result": dict(tool_result),
            },
        )

    messages.append(
        {
            "role": "assistant",
            "assistant_explanation": explanation,
            "explanation_receipt": receipt,
            "content": serialize_world_model_answer(answer),
        }
    )
    return SupervisedTraceRecord(
        record_id=record_id,
        metadata={
            "book_mode": "tool_call",
            "question_family": S1_1_QUESTION_FAMILY,
            "question_form": form,
            "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
            "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "system_prompt_sha256": prompt.system_prompt_sha256,
            "step": "S1.1",
            "species_taxon_id": "NCBITaxon:9606",
            "ensembl_release": ENSEMBL_RELEASE,
            "identifier_registry_id": identifier_registry_id,
            "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
            "tool_call_limit": 1,
            "requires_user_input": (
                answer.continuation_decision == "defer"
            ),
        },
        tools=tools,
        messages=tuple(messages),
        split=split,
        provenance=dict(provenance),
    )


def _require_s1_2_gene_id(value: Any) -> str:
    """Return one canonical ID for S1.2."""

    if not isinstance(value, str) or ENSEMBL_PATTERN.fullmatch(value) is None:
        raise ValueError("S1.2 requires one canonical Ensembl gene ID")
    return value


def _require_s1_2_input(value: Any) -> str:
    """Return one nonempty S1.2 input."""

    if not isinstance(value, str) or not value:
        raise ValueError("S1.2 requires one nonempty input")
    return value


def _s1_2_explanation(
    *,
    record_id: str,
    action_index: int,
    action_path: str,
    values: Mapping[str, Any],
    visible_input: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one deterministic S1.2 explanation and receipt."""

    templates = S1_2_EXPLANATION_TEMPLATES[action_path]
    template_index = int(
        stable_json_sha256(
            {"record_id": record_id, "action_index": action_index}
        )[:16],
        16,
    ) % len(templates)
    template = templates[template_index]
    explanation = template.text.format(**dict(values))
    _validate_explanation(explanation)
    receipt = ExplanationReceipt(
        explanation_contract_version=(
            SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ),
        explanation_template_id=template.template_id,
        explanation_template_sha256=text_sha256(template.text),
        explanation_rule_ids=template.rule_ids,
        explanation_source_refs=S1_SOURCE_REFS,
        explanation_input_sha256=stable_json_sha256(dict(visible_input)),
        explanation_sha256=text_sha256(explanation),
    )
    return explanation, receipt.to_dict()


def _s1_2_invalid_reason(invalid_kind: str) -> str:
    """Return the visible reason for one invalid ID."""

    reasons = {
        "transcript": "is a transcript ID, not a human gene ID",
        "mouse_gene": "is a mouse gene ID, not a human gene ID",
        "malformed": "does not match ENSG plus 11 digits",
    }
    try:
        return reasons[invalid_kind]
    except KeyError as error:
        raise ValueError("The S1.2 invalid input kind is invalid") from error


def _validate_s1_2_invalid_input(
    input_gene_id: str,
    invalid_kind: str,
) -> None:
    """Check one invalid S1.2 input and its declared kind."""

    patterns = {
        "transcript": ENSEMBL_TRANSCRIPT_PATTERN,
        "mouse_gene": ENSEMBL_MOUSE_GENE_PATTERN,
    }
    if invalid_kind in patterns:
        if patterns[invalid_kind].fullmatch(input_gene_id) is None:
            raise ValueError("The S1.2 invalid input does not match its kind")
        return
    if invalid_kind != "malformed":
        raise ValueError("The S1.2 invalid input kind is invalid")
    known_patterns = (
        ENSEMBL_PATTERN,
        ENSEMBL_VERSION_PATTERN,
        ENSEMBL_TRANSCRIPT_PATTERN,
        ENSEMBL_MOUSE_GENE_PATTERN,
    )
    if any(pattern.fullmatch(input_gene_id) for pattern in known_patterns):
        raise ValueError("The malformed S1.2 input has a valid known form")


def _s1_2_resolved_answer(
    *,
    form: str,
    task_context: str,
    gene_id: str,
    gene_symbol: str | None,
) -> tuple[str, WorldModelAnswer]:
    """Return one resolved S1.2 answer."""

    if gene_symbol is None:
        if form != "no_symbol":
            raise ValueError("Only the S1.2 no-symbol form can use null")
        if task_context == "inside_task":
            return "resolved_no_symbol_task", WorldModelAnswer(
                status="resolved",
                result={"gene_id": gene_id, "gene_symbol": None},
                continuation_decision="continue",
                summary=(
                    f"<GENE:{gene_id}> has no known symbol in the Ensembl "
                    "release 116 registry. The larger task can continue "
                    "with this Ensembl ID."
                ),
            )
        return "resolved_no_symbol_direct", WorldModelAnswer(
            status="resolved",
            result={"gene_id": gene_id, "gene_symbol": None},
            continuation_decision="stop",
            summary=(
                f"<GENE:{gene_id}> has no known symbol in the Ensembl "
                "release 116 registry."
            ),
        )

    if form == "inside_task":
        return "resolved_task", WorldModelAnswer(
            status="resolved",
            result={"gene_id": gene_id, "gene_symbol": gene_symbol},
            continuation_decision="continue",
            summary=(
                f"<GENE:{gene_id}|{gene_symbol}> is the resolved human "
                "gene. The larger task can continue with this gene."
            ),
        )
    if form == "already_resolved":
        action_path = "resolved_registry_block"
    elif form == "version_suffix":
        action_path = "resolved_version"
    else:
        action_path = "resolved_direct"
    return action_path, WorldModelAnswer(
        status="resolved",
        result={"gene_id": gene_id, "gene_symbol": gene_symbol},
        continuation_decision="stop",
        summary=f"<GENE:{gene_id}|{gene_symbol}> is the resolved human gene.",
    )


def build_s1_2_trace(
    *,
    record_id: str,
    question: str,
    form: str,
    input_gene_id: str,
    canonical_gene_id: str | None,
    gene_symbol: str | None,
    tool_result: Mapping[str, Any] | None,
    task_context: str,
    invalid_kind: str | None,
    split: str,
    provenance: Mapping[str, Any],
    identifier_registry_id: str,
) -> SupervisedTraceRecord:
    """Build one S1.2 Ensembl-to-symbol trace."""

    if form not in S1_2_FORMS:
        raise ValueError(f"The S1.2 form is invalid: {form!r}")
    if task_context not in S1_2_TASK_CONTEXTS:
        raise ValueError("The S1.2 task context is invalid")
    if form == "inside_task" and task_context != "inside_task":
        raise ValueError("An inside-task row needs its task context")
    if form not in {"inside_task", "no_symbol"} and task_context != "direct":
        raise ValueError("This S1.2 form must use the direct task context")

    input_gene_id = _require_s1_2_input(input_gene_id)
    prompt = s1_shared_trace_prompt_contract()
    tools = tuple(
        identifier_tool_definitions(
            contract_version=TOOL_CONTRACT_VERSION_V2
        )
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": question},
    ]

    if form == "invalid_input":
        if tool_result is not None or canonical_gene_id is not None:
            raise ValueError("An invalid S1.2 row cannot use a tool or ID")
        if gene_symbol is not None or invalid_kind is None:
            raise ValueError("An invalid S1.2 row has invalid target data")
        _validate_s1_2_invalid_input(input_gene_id, invalid_kind)
        invalid_reason = _s1_2_invalid_reason(invalid_kind)
        answer = WorldModelAnswer(
            status="invalid_input",
            result={
                "input": input_gene_id,
                "expected": "human_gene_id",
            },
            continuation_decision="defer",
            summary=(
                f"The input {invalid_reason}, so no symbol lookup applies. "
                "Give the human gene ID to continue."
            ),
        )
        explanation, receipt = _s1_2_explanation(
            record_id=record_id,
            action_index=0,
            action_path="invalid_input",
            values={
                "input_gene_id": input_gene_id,
                "invalid_reason": invalid_reason,
            },
            visible_input={
                "question": question,
                "input_gene_id": input_gene_id,
            },
        )
    elif form == "already_resolved":
        if tool_result is not None or invalid_kind is not None:
            raise ValueError("An already-resolved S1.2 row cannot use a tool")
        gene_id = _require_s1_2_gene_id(canonical_gene_id)
        symbol = _require_symbol(gene_symbol)
        if (
            input_gene_id != gene_id
            or "registry" not in question.lower()
            or gene_id not in question
            or symbol not in question
        ):
            raise ValueError(
                "An already-resolved S1.2 row needs a labeled registry block"
            )
        action_path, answer = _s1_2_resolved_answer(
            form=form,
            task_context=task_context,
            gene_id=gene_id,
            gene_symbol=symbol,
        )
        explanation, receipt = _s1_2_explanation(
            record_id=record_id,
            action_index=0,
            action_path=action_path,
            values={"gene_id": gene_id, "gene_symbol": symbol},
            visible_input={"question": question},
        )
    else:
        if invalid_kind is not None:
            raise ValueError("A resolved S1.2 row cannot have an invalid kind")
        gene_id = _require_s1_2_gene_id(canonical_gene_id)
        if form == "version_suffix":
            match = ENSEMBL_VERSION_PATTERN.fullmatch(input_gene_id)
            if match is None or match.group(1) != gene_id:
                raise ValueError("The S1.2 version suffix input is invalid")
            call_path = "version_lookup"
        else:
            if input_gene_id != gene_id:
                raise ValueError("The S1.2 input must match its canonical ID")
            if form == "inside_task":
                call_path = "id_lookup_task"
            elif form == "no_symbol" and task_context == "inside_task":
                call_path = "no_symbol_lookup_task"
            elif form == "no_symbol":
                call_path = "no_symbol_lookup_direct"
            else:
                call_path = "id_lookup_direct"
        if not isinstance(tool_result, Mapping):
            raise ValueError("A resolved S1.2 row needs one tool result")
        call_explanation, call_receipt = _s1_2_explanation(
            record_id=record_id,
            action_index=0,
            action_path=call_path,
            values={
                "input_gene_id": input_gene_id,
                "gene_id": gene_id,
            },
            visible_input={
                "question": question,
                "tool_name": ENSEMBL_TOOL_NAME,
                "arguments": {"gene_id": gene_id},
            },
        )
        call_id = "call_" + stable_json_sha256(
            {"record_id": record_id, "action_index": 0}
        )[:16]
        messages.extend(
            [
                {
                    "role": "assistant",
                    "assistant_explanation": call_explanation,
                    "explanation_receipt": call_receipt,
                    "tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": ENSEMBL_TOOL_NAME,
                            "arguments": {"gene_id": gene_id},
                        },
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": ENSEMBL_TOOL_NAME,
                    "content": dict(tool_result),
                },
            ]
        )
        symbol = None if gene_symbol is None else _require_symbol(gene_symbol)
        expected = {
            "status": "resolved",
            "gene_id": gene_id,
            "gene_symbol": symbol,
        }
        if dict(tool_result) != expected:
            raise ValueError("A resolved S1.2 tool result is invalid")
        action_path, answer = _s1_2_resolved_answer(
            form=form,
            task_context=task_context,
            gene_id=gene_id,
            gene_symbol=symbol,
        )
        explanation, receipt = _s1_2_explanation(
            record_id=record_id,
            action_index=1,
            action_path=action_path,
            values={"gene_id": gene_id, "gene_symbol": symbol},
            visible_input={
                "question": question,
                "tool_name": ENSEMBL_TOOL_NAME,
                "arguments": {"gene_id": gene_id},
                "tool_result": dict(tool_result),
            },
        )

    messages.append(
        {
            "role": "assistant",
            "assistant_explanation": explanation,
            "explanation_receipt": receipt,
            "content": serialize_world_model_answer(answer),
        }
    )
    metadata = {
        "book_mode": "tool_call",
        "question_family": S1_2_QUESTION_FAMILY,
        "question_form": form,
        "task_context": task_context,
        "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
        "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        "system_prompt_sha256": prompt.system_prompt_sha256,
        "step": "S1.2",
        "species_taxon_id": "NCBITaxon:9606",
        "ensembl_release": ENSEMBL_RELEASE,
        "identifier_registry_id": identifier_registry_id,
        "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
        "tool_call_limit": 1,
        "requires_user_input": answer.continuation_decision == "defer",
    }
    if invalid_kind is not None:
        metadata["invalid_input_kind"] = invalid_kind
    return SupervisedTraceRecord(
        record_id=record_id,
        metadata=metadata,
        tools=tools,
        messages=tuple(messages),
        split=split,
        provenance=dict(provenance),
    )


def _require_s1_3_candidates(value: Any) -> tuple[str, ...]:
    """Return one canonical S1.3 candidate set."""

    if (
        not isinstance(value, (list, tuple))
        or len(value) < 2
        or len(value) > 49
    ):
        raise ValueError("S1.3 requires between 2 and 49 candidate IDs")
    candidates = tuple(_require_gene_id(item) for item in value)
    if list(candidates) != sorted(set(candidates)):
        raise ValueError("The S1.3 candidate IDs must be unique and sorted")
    return candidates


def _s1_3_explanation(
    *,
    record_id: str,
    action_index: int,
    action_path: str,
    values: Mapping[str, Any],
    visible_input: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one deterministic S1.3 explanation and receipt."""

    templates = S1_3_EXPLANATION_TEMPLATES[action_path]
    template_index = int(
        stable_json_sha256(
            {"record_id": record_id, "action_index": action_index}
        )[:16],
        16,
    ) % len(templates)
    template = templates[template_index]
    explanation = template.text.format(**dict(values))
    _validate_explanation(explanation)
    receipt = ExplanationReceipt(
        explanation_contract_version=(
            SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ),
        explanation_template_id=template.template_id,
        explanation_template_sha256=text_sha256(template.text),
        explanation_rule_ids=template.rule_ids,
        explanation_source_refs=S1_SOURCE_REFS,
        explanation_input_sha256=stable_json_sha256(dict(visible_input)),
        explanation_sha256=text_sha256(explanation),
    )
    return explanation, receipt.to_dict()


def _ambiguous_answer(
    *,
    gene_symbol: str,
    candidate_gene_ids: tuple[str, ...],
) -> WorldModelAnswer:
    """Return one S1.3 ambiguous answer with all anchors."""

    anchors = ", ".join(
        f"<GENE:{gene_id}|{gene_symbol}>"
        for gene_id in candidate_gene_ids
    )
    return WorldModelAnswer(
        status="ambiguous",
        result={
            "gene_symbol": gene_symbol,
            "candidate_gene_ids": list(candidate_gene_ids),
        },
        continuation_decision="defer",
        summary=(
            f"{gene_symbol} matches {len(candidate_gene_ids)} Ensembl IDs: "
            f"{anchors}. No graph lookup can use {gene_symbol} until one ID "
            "is chosen. Choose one ID to continue."
        ),
    )


def build_s1_3_trace(
    *,
    record_id: str,
    question: str,
    form: str,
    gene_symbol: str,
    candidate_gene_ids: list[str] | tuple[str, ...],
    tool_result: Mapping[str, Any] | None,
    split: str,
    provenance: Mapping[str, Any],
    identifier_registry_id: str,
) -> SupervisedTraceRecord:
    """Build one S1.3 ambiguous-symbol trace."""

    if form not in S1_3_FORMS:
        raise ValueError(f"The S1.3 form is invalid: {form!r}")
    gene_symbol = _require_symbol(gene_symbol)
    candidates = _require_s1_3_candidates(candidate_gene_ids)
    prompt = s1_shared_trace_prompt_contract()
    tools = tuple(
        identifier_tool_definitions(
            contract_version=TOOL_CONTRACT_VERSION_V2
        )
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": question},
    ]
    values = {
        "gene_symbol": gene_symbol,
        "candidate_count": len(candidates),
    }

    if form == "already_marked":
        if tool_result is not None:
            raise ValueError("An already-marked S1.3 row cannot use a tool")
        if (
            "registry" not in question.lower()
            or gene_symbol not in question
            or any(gene_id not in question for gene_id in candidates)
        ):
            raise ValueError(
                "An already-marked S1.3 row needs a labeled registry block"
            )
        action_path = "ambiguous_registry_block"
        explanation, receipt = _s1_3_explanation(
            record_id=record_id,
            action_index=0,
            action_path=action_path,
            values=values,
            visible_input={"question": question},
        )
    else:
        if not isinstance(tool_result, Mapping):
            raise ValueError("An S1.3 lookup row requires one tool result")
        expected = {
            "status": "ambiguous",
            "gene_symbol": gene_symbol,
            "candidate_gene_ids": list(candidates),
        }
        if dict(tool_result) != expected:
            raise ValueError("An S1.3 tool result is invalid")
        call_path = (
            "symbol_lookup_task"
            if form == "inside_task"
            else "symbol_lookup_direct"
        )
        call_explanation, call_receipt = _s1_3_explanation(
            record_id=record_id,
            action_index=0,
            action_path=call_path,
            values=values,
            visible_input={
                "question": question,
                "tool_name": SYMBOL_TOOL_NAME,
                "arguments": {"gene_symbol": gene_symbol},
            },
        )
        call_id = "call_" + stable_json_sha256(
            {"record_id": record_id, "action_index": 0}
        )[:16]
        messages.extend(
            [
                {
                    "role": "assistant",
                    "assistant_explanation": call_explanation,
                    "explanation_receipt": call_receipt,
                    "tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": SYMBOL_TOOL_NAME,
                            "arguments": {"gene_symbol": gene_symbol},
                        },
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": SYMBOL_TOOL_NAME,
                    "content": dict(tool_result),
                },
            ]
        )
        action_path = (
            "ambiguous_task"
            if form == "inside_task"
            else "ambiguous_direct"
        )
        explanation, receipt = _s1_3_explanation(
            record_id=record_id,
            action_index=1,
            action_path=action_path,
            values=values,
            visible_input={
                "question": question,
                "tool_name": SYMBOL_TOOL_NAME,
                "arguments": {"gene_symbol": gene_symbol},
                "tool_result": dict(tool_result),
            },
        )

    answer = _ambiguous_answer(
        gene_symbol=gene_symbol,
        candidate_gene_ids=candidates,
    )
    messages.append(
        {
            "role": "assistant",
            "assistant_explanation": explanation,
            "explanation_receipt": receipt,
            "content": serialize_world_model_answer(answer),
        }
    )
    return SupervisedTraceRecord(
        record_id=record_id,
        metadata={
            "book_mode": "tool_call",
            "question_family": S1_3_QUESTION_FAMILY,
            "question_form": form,
            "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
            "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "system_prompt_sha256": prompt.system_prompt_sha256,
            "step": "S1.3",
            "species_taxon_id": "NCBITaxon:9606",
            "ensembl_release": ENSEMBL_RELEASE,
            "identifier_registry_id": identifier_registry_id,
            "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
            "tool_call_limit": 1,
            "requires_user_input": True,
        },
        tools=tools,
        messages=tuple(messages),
        split=split,
        provenance=dict(provenance),
    )
