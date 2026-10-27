"""One explicitly budgeted membership retry, retaining the successful lookup."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from .contract import REPOSITORY_ROOT
from .s1_lookup_retry import RETRY_CASES
from .s1_prompt_banks import validate_identity_completion_reasoning
from .sources import file_sha256
from .tools import MEMBERSHIP_TOOL, SYMBOL_TOOL, WorldModelToolRuntime

DIAGNOSTIC_PREFIX = "wm_s1_membership_retry_v1_"
BASE_FAILURE = {
    "status": "error", "provenance": None, "error_code": "BACKEND_UNAVAILABLE",
    "message": "graph service unreachable",
}


def budget_membership_failure(output: Mapping[str, Any], remaining: int) -> dict[str, Any]:
    """Add an explicit allowance to the existing graph error message."""
    if dict(output) != BASE_FAILURE or type(remaining) is not int or remaining not in (0, 1):
        raise ValueError("membership retry allowances require a graph backend failure")
    return {**output, "message": f"graph service unreachable; retries_remaining={remaining}"}


def diagnostic_membership_retry_case(record_id: str) -> str | None:
    """Confine the new fault schedule to its separate diagnostic namespace."""
    if not record_id.startswith(DIAGNOSTIC_PREFIX):
        return None
    for case in RETRY_CASES:
        if re.fullmatch(re.escape(DIAGNOSTIC_PREFIX + case + "_") + r"[0-9a-f]{24}", record_id):
            return case
    raise ValueError("invalid membership retry diagnostic record ID")


def load_membership_retry_policy(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Check the frozen wording, its bindings, and the approved allocation."""
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256"}:
        raise ValueError("membership retry policy needs path and sha256")
    path = REPOSITORY_ROOT / pin["path"]
    if file_sha256(path) != pin["sha256"]:
        raise ValueError("membership retry policy does not match its pin")
    policy = json.loads(path.read_text())
    if set(policy) != {"schema_version", "max_retries", "case_counts", "thinking", "summary_suffix"}:
        raise ValueError("membership retry policy keys changed")
    if policy["schema_version"] != "mentor-rl-s1-membership-retries-v1":
        raise ValueError("unsupported membership retry policy")
    if type(policy["max_retries"]) is not int or policy["max_retries"] != 1:
        raise ValueError("membership policies permit exactly one retry")
    if policy["case_counts"] != {"no_retry": 300, "recovered": 150, "exhausted": 150}:
        raise ValueError("membership retry case counts changed")
    if set(policy["thinking"]) != {"after_lookup", "retry", "no_retry", "exhausted"}:
        raise ValueError("membership retry wording cases changed")
    for case, text in policy["thinking"].items():
        fields = {"after_lookup": ("count", "symbol", "source", "multiplex_id"), "retry": ()}.get(
            case, ("multiplex_id", "error_code", "count", "symbol"))
        validate_identity_completion_reasoning(text, fields=fields)
    validate_identity_completion_reasoning(policy["summary_suffix"], fields=())
    return policy


def assign_membership_retry_cases(scenarios: Sequence[Any]) -> dict[str, str]:
    """Stratify by membership outcome; hash identities independently of wrappers."""
    groups = defaultdict(list)
    identities = set()
    for scenario in scenarios:
        identity = json.dumps([
            scenario.fact.fact_id, scenario.input_value.casefold(), scenario.occurrence,
            scenario.framing, scenario.multiplex_in_harness,
        ], separators=(",", ":"))
        if identity in identities:
            raise ValueError("membership retry assignment has a duplicate identity")
        identities.add(identity)
        groups[scenario.fact.membership_outcome].append((hashlib.sha256(("S1.3.10:" + identity).encode()).hexdigest(), scenario))
    cases = {}
    cycle = ("no_retry", "recovered", "no_retry", "exhausted")
    for group in groups.values():
        if len(group) % 4:
            raise ValueError("membership outcomes cannot support the approved retry shares")
        for index, (_, scenario) in enumerate(sorted(group, key=lambda item: item[0])):
            cases[scenario.scenario_id] = cycle[index % 4]
    return cases


def render_membership_retry_case(
    original: Mapping[str, Any], *, success: Mapping[str, Any], case: str,
    runtime: WorldModelToolRuntime, policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[list[str]]]:
    """Retain the lookup and use the actual retried membership evidence on recovery."""
    if case not in RETRY_CASES or original["metadata"]["source_trace"] != "S1.3.10":
        raise ValueError("membership retries apply only to S1.3.10")
    row = deepcopy(original)
    messages = row["messages"]
    lookup = next(m["content"] for m in messages if m.get("role") == "tool")
    call, failure, final = messages[-3:]
    arguments = call["tool_call"]["arguments"]
    bindings = dict(count=len(lookup["candidate_gene_ids"]), symbol=lookup["gene_symbol"],
                    multiplex_id=arguments["multiplex_id"], error_code="BACKEND_UNAVAILABLE")
    source = "harness" if any(m.get("role") == "harness" and "Active multiplex:" in m.get("content", "") for m in messages) else "task"
    call["thinking"] = policy["thinking"]["after_lookup"].format(source=source, **bindings)
    evidence = []
    if case != "no_retry":
        failure["content"] = budget_membership_failure(failure["content"], 1)
        retry = runtime.execute(MEMBERSHIP_TOOL, arguments, inject_backend_failure=case == "exhausted")
        output = budget_membership_failure(retry.output, 0) if case == "exhausted" else retry.output
        messages[-1:] = [
            {"role": "assistant", "thinking": policy["thinking"]["retry"], "tool_call": deepcopy(call["tool_call"])},
            {"role": "tool", "name": MEMBERSHIP_TOOL, "content": output}, final,
        ]
        if case == "recovered":
            if retry.evidence is None or output["status"] != "resolved":
                raise ValueError("membership recovery requires successful graph evidence")
            final.clear()
            final.update(deepcopy(success["messages"][-1]))
            for key in ("multiplex_id", "present_gene_ids", "absent_gene_ids"):
                if final["content"]["result"][key] != output[key]:
                    raise ValueError("membership recovery disagrees with its retained success")
            final["content"]["provenance"] = {
                "registry": deepcopy(lookup["provenance"]["registry"]),
                "graph": deepcopy(output["provenance"]["graph"]),
            }
            row["metadata"]["blocked_reason_code"] = success["metadata"]["blocked_reason_code"]
            evidence.append([str(retry.evidence["evidence_id"]), str(row["row_id"]), MEMBERSHIP_TOOL])
            return row, evidence
    final["thinking"] = policy["thinking"][case].format(**bindings)
    final["content"]["summary"] = final["content"]["summary"].replace(
        "Retry the check to continue.", policy["summary_suffix"])
    return row, evidence


def audit_membership_retry(calls: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]],
                           final: Mapping[str, Any], case: str, *, framing: str | None = None) -> None:
    """Reject unbudgeted or changed retries and answers unsupported by both tools."""
    count = 2 if case == "no_retry" else 3
    if case not in RETRY_CASES or len(calls) != count or len(outputs) != count:
        raise ValueError("membership retry trajectory shape changed")
    if [c["name"] for c in calls] != [SYMBOL_TOOL] + [MEMBERSHIP_TOOL] * (count - 1):
        raise ValueError("membership retry changed the tool sequence")
    lookup, first = outputs[:2]
    args = calls[1]["arguments"]
    expected_args = {"multiplex_id": args.get("multiplex_id"), "gene_ids": lookup.get("candidate_gene_ids")}
    if lookup.get("status") != "ambiguous" or args != expected_args or (count == 3 and calls[1] != calls[2]):
        raise ValueError("membership retry changed the candidates or arguments")
    if first != (BASE_FAILURE if case == "no_retry" else budget_membership_failure(BASE_FAILURE, 1)):
        raise ValueError("membership retry first allowance changed")
    expected = {k: lookup[k] for k in ("gene_symbol", "candidate_gene_ids", "candidate_details")}
    expected["multiplex_id"] = args["multiplex_id"]
    provenance = {"registry": lookup["provenance"]["registry"], "graph": None}
    status, decision = "ambiguous", "defer"
    if case == "recovered":
        output = outputs[-1]
        if output.get("status") != "resolved" or output.get("multiplex_id") != args["multiplex_id"]:
            raise ValueError("membership recovery lacks the requested successful check")
        present, absent = output["present_gene_ids"], output["absent_gene_ids"]
        candidates = lookup["candidate_gene_ids"]
        if (len(present) != len(set(present)) or len(absent) != len(set(absent))
                or set(present) & set(absent) or set(present + absent) != set(candidates)):
            raise ValueError("membership recovery changed the candidate partition")
        expected.update(present_gene_ids=present, absent_gene_ids=absent)
        provenance["graph"] = output["provenance"]["graph"]
        if not provenance["registry"] or not provenance["graph"]:
            raise ValueError("membership recovery must retain both evidence channels")
        if len(present) == 1:
            expected["gene_id"] = present[0]
            status = "resolved"
            decision = ("stop" if framing == "direct" else "continue") if framing else final["continuation_decision"]
            if decision not in ("stop", "continue"):
                raise ValueError("a unique membership recovery completes the request")
    else:
        expected.update(failed_tool=MEMBERSHIP_TOOL, error_code="BACKEND_UNAVAILABLE")
        if outputs[-1] != (BASE_FAILURE if case == "no_retry" else budget_membership_failure(BASE_FAILURE, 0)):
            raise ValueError("membership retry exhaustion allowance changed")
    blocked = final.get("blocked_reason")
    if (final.get("result") != expected or final.get("status") != status
            or final.get("continuation_decision") != decision or final.get("provenance") != provenance
            or (not isinstance(blocked, str) or not blocked.strip() if decision == "defer" else blocked is not None)):
        raise ValueError("membership retry final contradicts observed evidence or budget")


def audit_membership_retry_row(row: Mapping[str, Any], case: str, *, framing: str | None = None) -> None:
    """Check alternating actions and apply the shared semantic retry audit."""
    messages = row["messages"]
    start = next(i for i, m in enumerate(messages) if m.get("role") == "assistant")
    suffix = messages[start:]
    if [m["role"] for m in suffix] != ["assistant", "tool"] * (2 if case == "no_retry" else 3) + ["assistant"]:
        raise ValueError("membership retry action order changed")
    audit_membership_retry([m["tool_call"] for m in suffix[:-1:2]], [m["content"] for m in suffix[1:-1:2]],
                           suffix[-1]["content"], case, framing=framing)
