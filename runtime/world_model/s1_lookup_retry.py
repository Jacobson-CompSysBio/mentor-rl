"""Teach one permitted registry-lookup retry without changing the legacy wire.

The runtime supplies the remaining allowance in the existing error message.
Only a separate diagnostic namespace enables the new evaluation fault plans.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from .contract import REPOSITORY_ROOT
from .s1_prompt_banks import validate_identity_completion_reasoning
from .sources import file_sha256
from .tools import GENE_ID_TOOL, SYMBOL_TOOL, ToolCallResult, WorldModelToolRuntime


RETRY_CASES = ("no_retry", "recovered", "exhausted")
DIAGNOSTIC_PREFIX = "wm_s1_lookup_retry_v1_"
ID_DIAGNOSTIC_PREFIX = "wm_s1_id_lookup_retry_v1_"


def budget_failure(output: Mapping[str, Any], remaining: int) -> dict[str, Any]:
    """Expose the harness-owned allowance without adding tool-result keys."""

    if type(remaining) is not int or remaining not in (0, 1) or dict(output) != {
        "status": "error", "provenance": None,
        "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable",
    }:
        raise ValueError("retry allowances require a registry backend failure")
    return {**output, "message": f"registry unreachable; retries_remaining={remaining}"}


def load_retry_policy(pin: Mapping[str, Any], *, tool_name: str = SYMBOL_TOOL) -> dict[str, Any]:
    """Validate the pinned symbol or ID policy, including exact cases and bindings."""

    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
        raise ValueError("lookup retry policy needs path and sha256")
    path = REPOSITORY_ROOT / pin["path"]
    if file_sha256(path) != pin["sha256"]:
        raise ValueError("lookup retry policy does not match its pin")
    policy = json.loads(path.read_text())
    if set(policy) != {
        "schema_version", "max_retries", "case_counts", "thinking", "summaries",
    }:
        raise ValueError("lookup retry policy keys changed")
    if tool_name not in {SYMBOL_TOOL, GENE_ID_TOOL}:
        raise ValueError("unsupported lookup retry tool")
    is_id = tool_name == GENE_ID_TOOL
    schema = "mentor-rl-s1-id-lookup-retries-v1" if is_id else "mentor-rl-s1-lookup-retries-v1"
    if policy["schema_version"] != schema:
        raise ValueError("unsupported lookup retry policy")
    if type(policy["max_retries"]) is not int or policy["max_retries"] != 1:
        raise ValueError("lookup policies permit exactly one retry")
    counts = {"no_retry": 400, "recovered": 200, "exhausted": 200} if is_id else {"no_retry": 200, "recovered": 100, "exhausted": 100}
    if policy["case_counts"] != counts:
        raise ValueError("lookup retry case counts changed")
    if (
        set(policy["thinking"]) != {*RETRY_CASES, "retry"}
        or set(policy["summaries"]) != {"no_retry", "exhausted"}
    ):
        raise ValueError("lookup retry wording cases changed")
    for case, text in policy["thinking"].items():
        fields = {
            "recovered": ("symbol", "gene_id"), "exhausted": ("symbol",),
        }.get(case, ())
        if is_id:
            fields = {"recovered": ("input", "symbol", "gene_id"), "retry": ()}.get(case, ("input",))
        validate_identity_completion_reasoning(text, fields=fields)
    for text in policy["summaries"].values():
        validate_identity_completion_reasoning(text, fields=("input",) if is_id else ("symbol",))
    return policy


def assign_retry_cases(rows: Sequence[Mapping[str, Any]], *, tool_name: str = SYMBOL_TOOL) -> dict[str, str]:
    """Balance fault outcomes using identity hashes, independent of wrappers."""

    def identity_key(row: Mapping[str, Any]) -> str:
        input_key = "gene_id" if tool_name == GENE_ID_TOOL else "gene_symbol"
        identity = {
            "split_unit": row["metadata"]["split_unit"],
            input_key: row["messages"][2]["tool_call"]["arguments"][
                input_key
            ].casefold(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    ordered = sorted(rows, key=identity_key)
    cycle = ("no_retry", "recovered", "no_retry", "exhausted")
    return {
        str(row["row_id"]): cycle[index % 4]
        for index, row in enumerate(ordered)
    }


def _lookup_shape(row: Mapping[str, Any]) -> tuple[str, str]:
    """Limit retry rendering and auditing to the two reviewed lookup forms."""

    meta = row["metadata"]
    if meta["source_trace"] == "S1.1.8" and meta["form"] == 8:
        return SYMBOL_TOOL, "gene_symbol"
    if meta["source_trace"] == "S1.2.11" and meta["form"] == 11:
        return GENE_ID_TOOL, "gene_id"
    raise ValueError("lookup retries apply only to S1.1.8 or S1.2.11")


def render_retry_case(
    original: Mapping[str, Any],
    *,
    case: str,
    runtime: WorldModelToolRuntime,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[list[str]]]:
    """Revise one legacy failure row using a visible budget and real tool evidence."""

    if case not in RETRY_CASES:
        raise ValueError("unsupported lookup retry case")
    tool_name, input_key = _lookup_shape(original)
    row = deepcopy(original)
    messages = row["messages"]
    call, first, final = messages[-3:]
    arguments = call["tool_call"]["arguments"]
    if call["tool_call"]["name"] != tool_name or set(arguments) != {input_key}:
        raise ValueError("retry case must start with the declared lookup")
    evidence = []
    if case != "no_retry":
        first["content"] = budget_failure(first["content"], 1)
        retry = runtime.execute(
            tool_name, arguments, inject_backend_failure=case == "exhausted",
        )
        if case == "exhausted":
            retry = ToolCallResult(budget_failure(retry.output, 0), None)
        messages[-1:] = [
            {
                "role": "assistant", "thinking": policy["thinking"]["retry"],
                "tool_call": deepcopy(call["tool_call"]),
            },
            {"role": "tool", "name": tool_name, "content": retry.output},
            final,
        ]
        if case == "recovered":
            output = retry.output
            if output["status"] != "resolved" or retry.evidence is None:
                raise ValueError("a retry recovery must resolve from the pinned registry")
            symbol, gene_id = output["gene_symbol"], output["gene_id"]
            final["thinking"] = policy["thinking"][case].format(
                symbol=symbol, gene_id=gene_id, input=arguments[input_key],
            )
            final["content"] = {
                "status": "resolved", "provenance": output["provenance"],
                "result": {"gene_id": gene_id, "gene_symbol": symbol},
                "continuation_decision": "stop", "blocked_reason": None,
                "summary": (
                    f"{symbol} is <GENE:{gene_id}|{symbol}> in Ensembl release "
                    f"{runtime.ensembl_release}."
                ),
            }
            row["metadata"]["blocked_reason_code"] = None
            evidence.append([
                str(retry.evidence["evidence_id"]), str(row["row_id"]), tool_name,
            ])
            return row, evidence
    value = arguments[input_key]
    final["thinking"] = policy["thinking"][case].format(symbol=value, input=value)
    final["content"]["summary"] = policy["summaries"][case].format(symbol=value, input=value)
    return row, evidence


def audit_retry_row(row: Mapping[str, Any], case: str) -> None:
    """Reject extra calls, changed arguments, invented evidence, or a wrong stop rule."""

    if case not in RETRY_CASES:
        raise ValueError("unsupported lookup retry case")
    tool_name, input_key = _lookup_shape(row)
    suffix = row["messages"][2:]
    expected_roles = ["assistant", "tool"] * (
        1 if case == "no_retry" else 2
    ) + ["assistant"]
    if [message["role"] for message in suffix] != expected_roles:
        raise ValueError("lookup retry trajectory shape changed")
    calls = [message["tool_call"] for message in suffix[:-1:2]]
    if any(call != calls[0] or call["name"] != tool_name or set(call["arguments"]) != {input_key} for call in calls):
        raise ValueError("lookup retry changed the tool or arguments")
    outputs = [message["content"] for message in suffix[1:-1:2]]
    base = {
        "status": "error", "provenance": None,
        "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable",
    }
    if outputs[0] != (base if case == "no_retry" else budget_failure(base, 1)):
        raise ValueError("lookup retry first allowance changed")
    final = suffix[-1]["content"]
    if case == "recovered":
        expected = {key: outputs[-1][key] for key in ("gene_id", "gene_symbol")}
        valid = (
            outputs[-1]["status"] == final["status"] == "resolved"
            and final["result"] == expected
            and final["provenance"] == outputs[-1]["provenance"]
            and final["continuation_decision"] == "stop"
            and final["blocked_reason"] is None
            and row["metadata"]["blocked_reason_code"] is None
        )
    else:
        expected = {
            "failed_tool": tool_name, "error_code": "BACKEND_UNAVAILABLE",
            input_key: calls[0]["arguments"][input_key],
        }
        valid = (
            outputs[-1] == (base if case == "no_retry" else budget_failure(base, 0))
            and final["status"] == "error" and final["result"] == expected
            and final["provenance"] is None
            and final["continuation_decision"] == "defer"
            and bool(final["blocked_reason"])
        )
    if not valid:
        raise ValueError("lookup retry final answer contradicts observed evidence or budget")


def diagnostic_retry_case(record_id: str) -> str | None:
    """Read an answer-hidden fault plan only from the separate diagnostic namespace."""

    prefix = next((p for p in (DIAGNOSTIC_PREFIX, ID_DIAGNOSTIC_PREFIX) if record_id.startswith(p)), None)
    if prefix is None:
        return None
    for case in RETRY_CASES:
        pattern = re.escape(prefix + case + "_") + r"[0-9a-f]{24}"
        if re.fullmatch(pattern, record_id):
            return case
    raise ValueError("invalid lookup retry diagnostic record ID")


def diagnostic_retry_tool(record_id: str) -> str | None:
    """Select the diagnostic's lookup tool without changing legacy fault plans."""

    if diagnostic_retry_case(record_id) is None:
        return None
    return GENE_ID_TOOL if record_id.startswith(ID_DIAGNOSTIC_PREFIX) else SYMBOL_TOOL
