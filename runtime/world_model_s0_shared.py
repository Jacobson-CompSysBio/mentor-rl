"""Build S0 rows with the shared supervised trace contract."""

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
from .world_model_prompts import s0_shared_trace_prompt_contract
from .world_model_s0_tools import (
    ENSEMBL_PATTERN,
    ENSEMBL_RELEASE,
    ENSEMBL_TOOL_NAME,
    SYMBOL_TOOL_NAME,
    TOOL_CONTRACT_VERSION_V2,
    identifier_tool_definitions,
)


S0_SOURCE_REFS = (
    "agents/world_model_v2_curriculum/S0-human-gene-identifiers.md",
    "agents/world_model_v2_curriculum/S1-coordinate-system.md",
)
S0_SHARED_ACTION_PATHS = (
    "symbol_lookup",
    "gene_id_lookup",
    "symbol_resolved",
    "symbol_ambiguous",
    "symbol_not_found",
    "gene_id_named",
    "gene_id_unnamed",
    "gene_id_not_found",
)


@dataclass(frozen=True)
class S0ExplanationTemplate:
    """Define one reviewed S0 explanation template."""

    template_id: str
    text: str
    rule_ids: tuple[str, ...]


def _templates(
    action_path: str,
    texts: tuple[str, str, str],
    rule_ids: tuple[str, ...],
) -> tuple[S0ExplanationTemplate, ...]:
    """Create three templates for one S0 action path."""

    return tuple(
        S0ExplanationTemplate(
            template_id=f"s0_{action_path}_v1_{index}",
            text=text,
            rule_ids=rule_ids,
        )
        for index, text in enumerate(texts, start=1)
    )


S0_EXPLANATION_TEMPLATES = {
    "symbol_lookup": _templates(
        "symbol_lookup",
        (
            (
                "The input gives the symbol {gene_symbol}, so I will use "
                "the registry tool before I make a mapping claim."
            ),
            (
                "A symbol does not prove its Ensembl ID, so I will look up "
                "{gene_symbol} in the pinned registry."
            ),
            (
                "The task needs an exact ID for {gene_symbol}, so I will "
                "copy the visible symbol into the lookup tool."
            ),
        ),
        (
            "s0_rule_resolve_symbol_with_registry",
            "s0_rule_copy_visible_input",
        ),
    ),
    "gene_id_lookup": _templates(
        "gene_id_lookup",
        (
            (
                "The input gives {gene_id}, so I will use the registry tool "
                "before I make a symbol claim."
            ),
            (
                "An Ensembl ID can lack a symbol, so I will look up "
                "{gene_id} without assuming a label."
            ),
            (
                "The task needs the registry entry for {gene_id}, so I will "
                "copy the visible ID into the lookup tool."
            ),
        ),
        (
            "s0_rule_resolve_gene_id_with_registry",
            "s0_rule_allow_null_symbol",
            "s0_rule_copy_visible_input",
        ),
    ),
    "symbol_resolved": _templates(
        "symbol_resolved",
        (
            (
                "The tool linked {gene_symbol} to {gene_id}, so I will "
                "return both verified identifiers and continue."
            ),
            (
                "The registry resolved one ID for {gene_symbol}, so I will "
                "return the verified gene object and continue."
            ),
            (
                "The result verifies {gene_id} for {gene_symbol}, so I can "
                "report this mapping and continue."
            ),
        ),
        (
            "s0_rule_use_verified_mapping",
            "s0_rule_continue_after_resolution",
        ),
    ),
    "symbol_ambiguous": _templates(
        "symbol_ambiguous",
        (
            (
                "The tool returned multiple IDs for {gene_symbol}, so I "
                "will return all candidates without a mapping claim and defer."
            ),
            (
                "The registry cannot select one ID for {gene_symbol}, so I "
                "will preserve every candidate and request an Ensembl ID."
            ),
            (
                "The symbol {gene_symbol} is ambiguous, so I will not choose "
                "a candidate and will defer for one exact ID."
            ),
        ),
        (
            "s0_rule_return_all_candidates",
            "s0_rule_do_not_select_ambiguous_id",
            "s0_rule_defer_for_user_input",
        ),
    ),
    "symbol_not_found": _templates(
        "symbol_not_found",
        (
            (
                "The tool found no registry match for {gene_symbol}, so I "
                "will make no mapping claim and request another identifier."
            ),
            (
                "The registry has no ID for {gene_symbol}, so I will return "
                "the empty result and defer to the user."
            ),
            (
                "The symbol {gene_symbol} is not found, so I will not invent "
                "an ID and will request a valid identifier."
            ),
        ),
        (
            "s0_rule_make_no_unresolved_claim",
            "s0_rule_defer_for_user_input",
        ),
    ),
    "gene_id_named": _templates(
        "gene_id_named",
        (
            (
                "The tool verified {gene_id} with symbol {gene_symbol}, so I "
                "will return both identifiers and continue."
            ),
            (
                "The registry supplies {gene_symbol} for {gene_id}, so I "
                "can report the verified gene object and continue."
            ),
            (
                "The result names {gene_symbol} for {gene_id}, so I will "
                "keep the verified label and continue."
            ),
        ),
        (
            "s0_rule_use_verified_mapping",
            "s0_rule_continue_after_resolution",
        ),
    ),
    "gene_id_unnamed": _templates(
        "gene_id_unnamed",
        (
            (
                "The registry confirms {gene_id} but supplies no symbol, so "
                "I will keep a null label and continue with the valid ID."
            ),
            (
                "The tool resolved {gene_id} with a null symbol, so I will "
                "not invent a label and will continue."
            ),
            (
                "The ID {gene_id} is valid without a known symbol, so I will "
                "return the null label and continue."
            ),
        ),
        (
            "s0_rule_allow_null_symbol",
            "s0_rule_do_not_invent_symbol",
            "s0_rule_continue_after_resolution",
        ),
    ),
    "gene_id_not_found": _templates(
        "gene_id_not_found",
        (
            (
                "The tool found no registry entry for {gene_id}, so I will "
                "make no gene claim and request a valid ID."
            ),
            (
                "The registry does not contain {gene_id}, so I will keep the "
                "symbol null and defer to the user."
            ),
            (
                "The ID {gene_id} is not found, so I will not invent a symbol "
                "and will request another Ensembl ID."
            ),
        ),
        (
            "s0_rule_make_no_unresolved_claim",
            "s0_rule_do_not_invent_symbol",
            "s0_rule_defer_for_user_input",
        ),
    ),
}


@dataclass(frozen=True)
class _S0ResultBranch:
    """Store the branch data for one S0 tool result."""

    action_path: str
    question_family: str
    answer: WorldModelAnswer


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    """Require one exact field set."""

    if set(value) != expected:
        raise ValueError(
            "The S0 tool result fields do not match its status."
        )


def _require_gene_id(value: Any) -> str:
    """Return one valid human Ensembl gene ID."""

    if not isinstance(value, str) or ENSEMBL_PATTERN.fullmatch(value) is None:
        raise ValueError("The S0 tool result has an invalid gene ID.")
    return value


def _require_symbol(value: Any) -> str:
    """Return one nonempty gene symbol."""

    if not isinstance(value, str) or not value:
        raise ValueError("The S0 tool result has an invalid gene symbol.")
    return value


def _classify_symbol_result(
    arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> _S0ResultBranch:
    """Build the answer branch for one symbol result."""

    if set(arguments) != {"gene_symbol"}:
        raise ValueError("A symbol lookup needs exactly one gene_symbol.")
    gene_symbol = _require_symbol(arguments["gene_symbol"])
    if tool_result.get("gene_symbol") != gene_symbol:
        raise ValueError("The S0 symbol result changed the input symbol.")

    status = tool_result.get("status")
    if status == "resolved":
        _require_exact_fields(
            tool_result,
            {"status", "gene_id", "gene_symbol"},
        )
        gene_id = _require_gene_id(tool_result["gene_id"])
        return _S0ResultBranch(
            action_path="symbol_resolved",
            question_family="human_symbol_to_ensembl",
            answer=WorldModelAnswer(
                status="resolved",
                result={
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                },
                continuation_decision="continue",
                summary=(
                    f"<GENE:{gene_id}|{gene_symbol}> is the resolved "
                    "human gene."
                ),
            ),
        )

    _require_exact_fields(
        tool_result,
        {"status", "gene_symbol", "candidate_gene_ids"},
    )
    candidate_gene_ids = tool_result.get("candidate_gene_ids")
    if not isinstance(candidate_gene_ids, list):
        raise ValueError("The S0 symbol result needs a candidate list.")
    candidate_gene_ids = [
        _require_gene_id(gene_id) for gene_id in candidate_gene_ids
    ]
    if candidate_gene_ids != sorted(set(candidate_gene_ids)):
        raise ValueError("The S0 candidate IDs are not canonical.")

    if status == "ambiguous" and len(candidate_gene_ids) >= 2:
        candidate_text = ", ".join(
            f"<GENE:{gene_id}>" for gene_id in candidate_gene_ids
        )
        return _S0ResultBranch(
            action_path="symbol_ambiguous",
            question_family="human_ambiguous_symbol",
            answer=WorldModelAnswer(
                status="ambiguous",
                result={
                    "gene_symbol": gene_symbol,
                    "candidate_gene_ids": candidate_gene_ids,
                },
                continuation_decision="defer",
                summary=(
                    f"The symbol {gene_symbol} has these candidate genes: "
                    f"{candidate_text}. Give one exact Ensembl gene ID to "
                    "continue."
                ),
            ),
        )

    if status == "not_found" and not candidate_gene_ids:
        return _S0ResultBranch(
            action_path="symbol_not_found",
            question_family="human_symbol_to_ensembl",
            answer=WorldModelAnswer(
                status="not_found",
                result={
                    "gene_symbol": gene_symbol,
                    "candidate_gene_ids": [],
                },
                continuation_decision="defer",
                summary=(
                    f"The registry has no match for the symbol {gene_symbol}. "
                    "Give another human gene symbol or Ensembl gene ID to "
                    "continue."
                ),
            ),
        )

    raise ValueError("The S0 symbol result has an invalid status or count.")


def _classify_gene_id_result(
    arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> _S0ResultBranch:
    """Build the answer branch for one gene ID result."""

    if set(arguments) != {"gene_id"}:
        raise ValueError("A gene ID lookup needs exactly one gene_id.")
    gene_id = _require_gene_id(arguments["gene_id"])
    _require_exact_fields(
        tool_result,
        {"status", "gene_id", "gene_symbol"},
    )
    if tool_result.get("gene_id") != gene_id:
        raise ValueError("The S0 gene ID result changed the input ID.")

    status = tool_result.get("status")
    gene_symbol = tool_result.get("gene_symbol")
    if status == "resolved" and isinstance(gene_symbol, str):
        gene_symbol = _require_symbol(gene_symbol)
        return _S0ResultBranch(
            action_path="gene_id_named",
            question_family="human_ensembl_to_symbol",
            answer=WorldModelAnswer(
                status="resolved",
                result={
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                },
                continuation_decision="continue",
                summary=(
                    f"<GENE:{gene_id}|{gene_symbol}> is the resolved "
                    "human gene."
                ),
            ),
        )

    if status == "resolved" and gene_symbol is None:
        return _S0ResultBranch(
            action_path="gene_id_unnamed",
            question_family="human_ensembl_to_symbol",
            answer=WorldModelAnswer(
                status="resolved",
                result={"gene_id": gene_id, "gene_symbol": None},
                continuation_decision="continue",
                summary=(
                    f"<GENE:{gene_id}> is a valid human Ensembl gene ID "
                    "with no known symbol in release 116."
                ),
            ),
        )

    if status == "not_found" and gene_symbol is None:
        return _S0ResultBranch(
            action_path="gene_id_not_found",
            question_family="human_ensembl_to_symbol",
            answer=WorldModelAnswer(
                status="not_found",
                result={"gene_id": gene_id, "gene_symbol": None},
                continuation_decision="defer",
                summary=(
                    f"The registry has no match for <GENE:{gene_id}>. "
                    "Give a valid human Ensembl gene ID to continue."
                ),
            ),
        )

    raise ValueError("The S0 gene ID result has an invalid status or symbol.")


def _classify_result(
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> _S0ResultBranch:
    """Build one S0 answer from a version 2 tool result."""

    if not isinstance(arguments, Mapping):
        raise TypeError("The S0 tool arguments must be one object.")
    if not isinstance(tool_result, Mapping):
        raise TypeError("The S0 tool result must be one object.")
    if tool_name == SYMBOL_TOOL_NAME:
        return _classify_symbol_result(arguments, tool_result)
    if tool_name == ENSEMBL_TOOL_NAME:
        return _classify_gene_id_result(arguments, tool_result)
    raise ValueError("The S0 shared trace has an unknown tool name.")


def _validate_explanation(explanation: str) -> None:
    """Check the S0 explanation word and sentence limits."""

    sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", explanation)
        if sentence
    ]
    if len(sentences) not in {1, 2}:
        raise ValueError("An S0 explanation must have one or two sentences.")
    word_counts = [len(re.findall(r"\b[\w'-]+\b", text)) for text in sentences]
    if sum(word_counts) > 40 or any(count > 25 for count in word_counts):
        raise ValueError("An S0 explanation exceeds its word limit.")


def _explanation(
    *,
    record_id: str,
    action_index: int,
    action_path: str,
    values: Mapping[str, Any],
    visible_input: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one deterministic explanation and its receipt."""

    templates = S0_EXPLANATION_TEMPLATES[action_path]
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
        explanation_source_refs=S0_SOURCE_REFS,
        explanation_input_sha256=stable_json_sha256(dict(visible_input)),
        explanation_sha256=text_sha256(explanation),
    )
    return explanation, receipt.to_dict()


def build_s0_shared_trace(
    *,
    record_id: str,
    question: str,
    question_family: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
    split: str,
    provenance: Mapping[str, Any],
    identifier_registry_id: str,
) -> SupervisedTraceRecord:
    """Build one S0 row with one tool call and one JSON answer."""

    branch = _classify_result(tool_name, arguments, tool_result)
    if branch.question_family != question_family:
        raise ValueError("The S0 question family does not match the result.")

    prompt = s0_shared_trace_prompt_contract()
    call_path = (
        "symbol_lookup"
        if tool_name == SYMBOL_TOOL_NAME
        else "gene_id_lookup"
    )
    values = dict(arguments)
    values.update(dict(tool_result))
    call_explanation, call_receipt = _explanation(
        record_id=record_id,
        action_index=0,
        action_path=call_path,
        values=arguments,
        visible_input={
            "question": question,
            "tool_name": tool_name,
            "arguments": dict(arguments),
        },
    )
    final_explanation, final_receipt = _explanation(
        record_id=record_id,
        action_index=1,
        action_path=branch.action_path,
        values=values,
        visible_input={
            "question": question,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "tool_result": dict(tool_result),
        },
    )
    call_id = "call_" + stable_json_sha256(
        {"record_id": record_id, "action_index": 0}
    )[:16]
    messages = (
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "assistant_explanation": call_explanation,
            "explanation_receipt": call_receipt,
            "tool_call": {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": dict(arguments),
                },
            },
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": dict(tool_result),
        },
        {
            "role": "assistant",
            "assistant_explanation": final_explanation,
            "explanation_receipt": final_receipt,
            "content": serialize_world_model_answer(branch.answer),
        },
    )
    return SupervisedTraceRecord(
        record_id=record_id,
        metadata={
            "book_mode": "tool_call",
            "question_family": question_family,
            "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
            "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
            "system_prompt_sha256": prompt.system_prompt_sha256,
            "step": "S0",
            "species_taxon_id": "NCBITaxon:9606",
            "ensembl_release": ENSEMBL_RELEASE,
            "identifier_registry_id": identifier_registry_id,
            "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
            "requires_user_input": (
                branch.answer.continuation_decision == "defer"
            ),
        },
        tools=tuple(
            identifier_tool_definitions(
                contract_version=TOOL_CONTRACT_VERSION_V2
            )
        ),
        messages=messages,
        split=split,
        provenance=dict(provenance),
    )
