#!/usr/bin/env python3
"""Build a paired, training-owned S1 diagnostic separate from promotion."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.world_model import s1_3_corpus as identity
from runtime.world_model import s1_4_corpus as alignment
from runtime.world_model import s1_6_corpus as layers
from runtime.world_model import s1_7_corpus as claims
from runtime.world_model import s1_9_corpus as modules
from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s0_corpus import EvidenceSequence
from runtime.world_model.s1_overlay import S1_V8_OVERLAY_PATH, apply_s1_overlay, overlay_pin
from runtime.world_model.source_index import check_source_index
from runtime.world_model.tools import WorldModelToolRuntime
from scripts.build_world_model_s1_bundle import project_row
from scripts.world_model_s1_eval_common import (
    read_json, read_jsonl, sha256_file, stable_sha256, validate_answer_row,
    validate_question, write_json,
)

DEFAULT_OUTPUT = ROOT / "data/world_model_v4/diagnostics/s1_robustness_v1"
COMPONENTS = ("s1_3", "s1_4", "s1_6", "s1_7", "s1_9")
BANK_NAMES = dict(zip(COMPONENTS, ("identity_s1_3", "alignment", "layer", "layer_family", "module")))
AXIS_COUNTS = {"wording": 48, "task_scope": 48, "evidence_order": 32,
               "id_notation_case": 32, "survivor_position": 12}
RELATIONS = {"wording": "invariant", "task_scope": "scope_flip",
             "evidence_order": "invariant", "id_notation_case": "normalized_identity_invariant",
             "survivor_position": "source_backed_strata"}


def final(row):
    """Read the final native answer, before projection separates the key."""
    return row["messages"][-1]["content"]


def _unique(rows, count, key):
    selected, seen = [], set()
    for row in sorted(rows, key=lambda r: r["row_id"]):
        value = key(row)
        if value not in seen:
            selected.append(row)
            seen.add(value)
        if len(selected) == count:
            return selected
    raise ValueError(f"training sources provide only {len(selected)} of {count} requested examples")


def _nonfirst(row):
    result = final(row)["result"]
    return result["gene_id"] != result["candidate_gene_ids"][0]


def _layer_tags(row):
    result = final(row)["result"]
    if "layers" in result:
        return [item["layer_id"] for item in result["layers"]]
    tags = result["matching_layer_ids"] + result["other_layer_ids"]
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return sorted(tags, key=user.index)


def _base_sources(sources):
    """Cover the measured filter/claim defects and both identity outcomes."""
    chosen = []
    membership = [r for r in sources["s1_3"] if r["metadata"]["form"] == 5]
    for nonfirst in (False, True):
        chosen += [("s1_3", r) for r in _unique(
            [r for r in membership if _nonfirst(r) == nonfirst], 4,
            lambda r: final(r)["result"]["gene_symbol"])]
    for form in (3, 4):
        chosen += [("s1_4", r) for r in _alignment_sources(sources, form, 4)]
    for component, form in (("s1_6", 4), ("s1_7", 2)):
        rows = [r for r in sources[component] if r["metadata"]["form"] == form]
        strata = ((True, False, 2), (True, True, 2), (False, False, 2), (False, True, 2))
        for homogeneous, positive, count in strata:
            query_override = component == "s1_7" and homogeneous and positive
            eligible = [r for r in rows
                        if all(t.startswith("scPEN:") for t in _layer_tags(r)) == homogeneous
                        and bool(final(r)["result"]["matching_layer_ids"]) == (False if query_override else positive)]
            if homogeneous and not positive:
                eligible = [r for r in eligible if (
                    final(r)["result"].get("layer_family") == "bulk_pen"
                    or final(r)["result"].get("filter") == {"field": "layer_family", "value": "bulk_pen"})]
            selected = _unique(eligible, count, lambda r: stable_sha256(final(r)["result"]))
            chosen += [(component, {**r, "query_family_override": "scpen"} if query_override else r) for r in selected]
    for supported in (False, True):
        chosen += [("s1_7", r) for r in _unique(
            [r for r in sources["s1_7"] if r["metadata"]["form"] == 4
             and final(r)["result"]["claim_supported"] == supported], 4,
            lambda r: (final(r)["result"]["layer_id"], final(r)["result"]["claim"]))]
    for form in (1, 2):
        chosen += [("s1_9", r) for r in _unique(
            [r for r in sources["s1_9"] if r["metadata"]["form"] == form], 4,
            lambda r: final(r)["result"]["module_id"])]
    if len(chosen) != 48:
        raise ValueError("wording/scope source allocation changed")
    return chosen


def _alignment_sources(sources, form, count, *, versioned=False):
    rows = [r for r in sources["s1_4"] if r["metadata"]["form"] == form]
    if versioned or form == 3:
        rows = [r for r in rows if any("." in final(r)["result"][side]["input"]
                                     for side in ("first", "second"))]
    return _unique(rows, count, lambda r: tuple(final(r)["result"][side]["gene_id"]
                                               for side in ("first", "second")))


def _scenario(component, row, contract, genes, occurrence):
    """Recover only facts visible in a pinned training row; builders derive gold."""
    result = final(row)["result"]
    form = row["metadata"]["form"]
    unit = row["metadata"]["split_unit"]
    common = dict(split="train", form=form, framing="direct", template_index=0, occurrence=occurrence)
    if component == "s1_3":
        fact = identity.GeneFact(stable_sha256(result), unit, result["gene_symbol"],
            tuple(result["candidate_gene_ids"]), result["candidate_details"], tuple(result["present_gene_ids"]))
        return identity.Scenario(**common, fact=fact, input_value=fact.gene_symbol,
                                 multiplex_in_harness=bool(occurrence % 2))
    if component == "s1_4":
        mentions = {result[side]["item"]: result[side] for side in ("first", "second")}
        versions = {m["gene_id"]: int(m["input"].split(".")[1])
                    for m in mentions.values() if "." in m["input"]}
        def gene(gene_id):
            return alignment.GeneFact(gene_id, (unit,), genes[gene_id][0], gene_id, versions.get(gene_id))
        seed, module = mentions["seed_gene_set"], mentions["module"]
        harness = next(m["content"] for m in row["messages"] if m["role"] == "harness")
        ids = json.loads(re.search(r"gene_ids: (\[[^\n]+\])", harness).group(1))
        distractor = next(value for value in ids if value != module["input"])
        fact, target = (seed, module) if form == 3 else (module, seed)
        return alignment.Scenario(**common, fact=gene(fact["gene_id"]), target=gene(target["gene_id"]),
            distractor=gene(distractor), orientation=occurrence % 2, training_variant=occurrence % 2)
    if component == "s1_6":
        facts = [layers.parse_layer_tag(tag, contract) for tag in _layer_tags(row)]
        return layers.Scenario(**common, fact=facts[0], companions=tuple(facts),
            requested_field=result["filter"]["field"], filter_value=result["filter"]["value"], train_reasoning=True)
    if component == "s1_7":
        if form == 2:
            facts = tuple(layers.parse_layer_tag(tag, contract) for tag in _layer_tags(row))
            return claims.Scenario(**common, family=row.get("query_family_override", result["layer_family"]), fact=None,
                                   companions=facts, train_reasoning=True)
        fact = layers.parse_layer_tag(result["layer_id"], contract) if result["layer_id"] else None
        return claims.Scenario(**common, family=result["layer_family"], fact=fact,
            input_kind="layer_tag" if fact else "family_name", claim=result["claim"],
            claim_supported=result["claim_supported"], train_reasoning=True)
    return modules.Scenario(**common, module_id=result["module_id"], module_family=result["module_family"],
        split_unit=unit, request_index=0, scenario_id=stable_sha256([row["row_id"], occurrence]), train_reasoning=True)


def _render(component, scenario, *, contract, runtime, banks):
    settings = contract["curriculum"]["families"][component.replace("s1_", "S1.")]
    common = dict(contract=contract, tools=tool_definitions(contract),
                  prompts={"train": banks[BANK_NAMES[component]]})
    if component == "s1_3":
        return identity.build_row(scenario, runtime=runtime, **common,
            membership_answers=settings["membership_answers"], train_multiplex_reasoning=True,
            single_membership_reasoning=settings["train_single_membership_reasoning"])[0]
    if component == "s1_4":
        return alignment.build_row(scenario, runtime=runtime, **common)[0]
    builder = {"s1_6": layers, "s1_7": claims, "s1_9": modules}[component]
    return builder.build_row(scenario, **common, final_reasoning=settings.get("train_final_reasoning"))[0]


def _structured(row, *, normalize_inputs=False):
    value = deepcopy(final(row))
    value.pop("summary")
    value["blocked_reason"] = value["blocked_reason"] is not None
    def clean(item):
        if isinstance(item, dict):
            return {k: (str(v).split(".")[0].upper() if k == "input" and normalize_inputs else clean(v))
                    for k, v in item.items() if k != "evidence_id"}
        return [clean(v) for v in item] if isinstance(item, list) else item
    return clean(value), [m["tool_call"] for m in row["messages"] if "tool_call" in m]


def validate_pair(group, rows):
    """Fail when the declared relationship is not demonstrated by native gold."""
    a, b = [rows[m["record_id"]] for m in group["members"]]
    axis = group["axis"]
    relation = group["expected_relationship"]
    if relation != RELATIONS[axis] and not (axis == "evidence_order" and relation == "mention_swap"):
        raise ValueError("pair relationship disagrees with its axis")
    if axis == "survivor_position":
        if _nonfirst(a) or not _nonfirst(b):
            raise ValueError("survivor coverage must contain actual first and nonfirst facts")
        for row in (a, b):
            result = final(row)["result"]
            if result["candidate_gene_ids"] != sorted(result["candidate_gene_ids"]) or result["present_gene_ids"] != [result["gene_id"]]:
                raise ValueError("survivor coverage changed canonical membership")
        return
    left, calls_a = _structured(a, normalize_inputs=axis == "id_notation_case")
    right, calls_b = _structured(b, normalize_inputs=axis == "id_notation_case")
    if relation == "mention_swap":
        right["result"]["first"], right["result"]["second"] = right["result"]["second"], right["result"]["first"]
    if axis == "task_scope":
        if (left.pop("continuation_decision"), right.pop("continuation_decision")) != ("stop", "continue"):
            raise ValueError("task scope pair does not flip stop to continue")
    if left != right or calls_a != calls_b:
        raise ValueError(f"{axis} pair changed an undeclared structured field or call")
    if [m for m in a["messages"] if m["role"] in {"harness", "user"}] == [m for m in b["messages"] if m["role"] in {"harness", "user"}]:
        raise ValueError("paired public prompts are identical")


def build_panel(output_directory=DEFAULT_OUTPUT):
    """Pin training sources, render and validate all pairs, and publish atomically."""
    output_directory = Path(output_directory).resolve()
    for protected in (ROOT / "data/world_model_v4/corpora", ROOT / "data/world_model_v4/eval", ROOT / "runtime"):
        if output_directory.is_relative_to(protected):
            raise ValueError("diagnostic output cannot overwrite corpora, frozen evaluation, or runtime inputs")
    contract, overlay = apply_s1_overlay(load_contract(), S1_V8_OVERLAY_PATH)
    bank_path = ROOT / "runtime/world_model/s1_v5_prompt_banks.json"
    banks = read_json(bank_path)
    inputs = {}
    def pin(path):
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for path in (ROOT / "runtime/world_model/contract.yaml", S1_V8_OVERLAY_PATH, bank_path):
        pin(path)
    sources = {}
    for component in COMPONENTS:
        directory = ROOT / "data/world_model_v4/corpora/s1_v8" / component
        manifest = read_json(directory / "manifest.json")
        path = directory / "train.jsonl"
        if sha256_file(path) != manifest["files"]["train.jsonl"]["sha256"]:
            raise ValueError("training source rows do not match their manifest")
        pin(path); pin(directory / "manifest.json")
        sources[component] = read_jsonl(path)
    source_path = ROOT / contract["source_index"]["output_directory"] / f"world_model_source_index_{contract['source_index']['content_sha256']}.json.gz"
    pin(source_path)
    payload = check_source_index(source_path, contract)
    runtime = WorldModelToolRuntime(payload=payload, contract=contract, evidence_id_factory=EvidenceSequence())
    rows, groups = [], []

    def add(axis, index, members, *, coverage=None, relation=None):
        group_id = f"s1_robust_v1_{axis}_{index:03d}"
        group = dict(group_id=group_id, axis=axis, kind="coverage_group" if axis == "survivor_position" else "pair",
                     expected_relationship=relation or RELATIONS[axis], members=[], coverage=coverage or {})
        for role, component, original, scenario in members:
            row = _render(component, scenario, contract=contract, runtime=runtime, banks=banks)
            if relation == "mention_swap" and role == "b":
                before = alignment._context_group_id(members[0][3])
                after = alignment._context_group_id(scenario)
                for message in row["messages"]:
                    for key in ("content", "thinking"):
                        if isinstance(message.get(key), str):
                            message[key] = message[key].replace(after, before)
            row["row_id"] = "wm_" + group_id + "_" + role
            rows.append(row)
            group["members"].append(dict(record_id=row["row_id"], role=role,
                source_row_id=original["row_id"], source_component=component))
            if "query_family_override" in original:
                group["members"][-1]["derivation"] = {
                    "query_family_override": original["query_family_override"],
                    "description": "Keep the training-owned tags; the native family filter derives the answer for this query."}
        validate_pair(group, {r["row_id"]: r for r in rows[-2:]})
        groups.append(group)

    base = _base_sources(sources)
    for axis in ("wording", "task_scope"):
        for index, (component, original) in enumerate(base):
            scenario = _scenario(component, original, contract, payload["genes_by_id"], index)
            scenario = replace(scenario, template_index=(index + (7 if axis == "task_scope" else 0)) % 32,
                               framing="step" if axis == "wording" and index % 2 else "direct")
            other = replace(scenario, template_index=(index + 13) % 32) if axis == "wording" else replace(scenario, framing="step")
            add(axis, index, [("a", component, original, scenario), ("b", component, original, other)],
                coverage={"source_trace": original["metadata"]["source_trace"]})
    for axis in ("evidence_order", "id_notation_case"):
        originals = [r for form in (3, 4) for r in _alignment_sources(sources, form, 16, versioned=axis == "id_notation_case")]
        for index, original in enumerate(originals):
            scenario = _scenario("s1_4", original, contract, payload["genes_by_id"], index + (1000 if axis == "evidence_order" else 2000))
            scenario = replace(scenario, framing="step" if index % 2 else "direct", template_index=index % 32)
            if axis == "evidence_order":
                mention_swap = index % 16 >= 8
                other = replace(scenario, orientation=1 - scenario.orientation) if mention_swap else replace(scenario, training_variant=scenario.training_variant + 2)
            else:
                mentions = {final(original)["result"][side]["item"]: final(original)["result"][side] for side in ("first", "second")}
                style = (1 if "." in mentions["seed_gene_set"]["input"] else 3) if index % 2 else 2
                other = replace(scenario, training_variant=scenario.training_variant + 4 * style)
            add(axis, index, [("a", "s1_4", original, scenario), ("b", "s1_4", original, other)],
                relation="mention_swap" if axis == "evidence_order" and mention_swap else None)
    membership = [r for r in sources["s1_3"] if r["metadata"]["form"] == 5]
    nonfirst = _unique([r for r in membership if _nonfirst(r)], 6, lambda r: final(r)["result"]["gene_symbol"])
    first = _unique([r for r in membership if not _nonfirst(r)], 293, lambda r: final(r)["result"]["gene_symbol"])
    for index, original in enumerate(nonfirst):
        count = len(final(original)["result"]["candidate_gene_ids"])
        match = min(first, key=lambda r: (abs(len(final(r)["result"]["candidate_gene_ids"]) - count), r["row_id"]))
        matched_count = len(final(match)["result"]["candidate_gene_ids"]) == count
        for frame_index, framing in enumerate(("direct", "step")):
            pair_index = index * 2 + frame_index
            members = []
            for role, source in (("a", match), ("b", original)):
                scenario = _scenario("s1_3", source, contract, payload["genes_by_id"], pair_index)
                members.append((role, "s1_3", source, replace(scenario, framing=framing, template_index=(pair_index + 22) % 32)))
            add("survivor_position", pair_index, members, coverage={
                "matching_fields": ["framing", "multiplex_id", "single_survivor", "wrapper", "multiplex_placement"] + (["candidate_count"] if matched_count else []),
                "unmatched_fields": ["gene_symbol", "candidate_gene_ids", "survivor_gene_id"] + ([] if matched_count else ["candidate_count"]),
                "interpretation": "Different real source facts; matched coverage strata, not a same-fact counterfactual."})
    if Counter(g["axis"] for g in groups) != AXIS_COUNTS or len(rows) != 344:
        raise ValueError("robustness coverage changed")
    questions, answers = [], []
    for row in rows:
        question, answer = project_row(row["row_id"], row["messages"], row["tools"], row["metadata"], "test")
        questions.append(validate_question(question, contract))
        answers.append(validate_answer_row(answer, question, contract))
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robustness-", dir=output_directory.parent) as temporary:
        staging = Path(temporary)
        for name, records in (("rows.jsonl", rows), ("test_questions.jsonl", questions), ("test_answer_key.jsonl", answers)):
            # The semantic contract owns key order, including tool provenance.
            (staging / name).write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records))
        write_json(staging / "pairs.json", {"schema_version": "mentor-rl-s1-robustness-pairs-v1", "groups": groups})
        validation = subprocess.run([sys.executable, str(ROOT / "agents/world_model_v3_curriculum/s1_kit/validate.py"),
            "--overlay", str(S1_V8_OVERLAY_PATH), str(staging / "rows.jsonl")], cwd=ROOT, capture_output=True, text=True)
        if validation.returncode:
            raise ValueError("robustness semantic validation failed:\n" + validation.stdout[:10000] + validation.stderr)
        implementation = [Path(__file__).resolve(), *(ROOT / "runtime/world_model" / f"{c}_corpus.py" for c in COMPONENTS),
            ROOT / "runtime/world_model/tools.py", ROOT / "runtime/world_model/contract.py",
            ROOT / "runtime/world_model/source_index.py", ROOT / "scripts/build_world_model_s1_bundle.py"]
        manifest = dict(schema_version="mentor-rl-s1-robustness-diagnostic-v1", kind="robustness_v1",
            diagnostic_only=True, source_split="train", row_count=len(rows), group_count=len(groups),
            purpose="Paired robustness diagnostic using training-owned facts. Never combine with frozen promotion scores.",
            max_tool_calls=2, group_counts=AXIS_COUNTS, axis_counts={k: 2*v for k,v in AXIS_COUNTS.items()},
            family_counts=dict(Counter(q["metadata"]["question_family"] for q in questions)),
            source_trace_counts=dict(Counter(q["metadata"]["source_trace"] for q in questions)),
            inputs=inputs, s1_overlay=overlay_pin(S1_V8_OVERLAY_PATH, overlay),
            source_index_sha256=contract["source_index"]["content_sha256"],
            implementation_sha256={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in implementation},
            files={name: dict(sha256=sha256_file(staging/name), **({"group_count": len(groups)} if name == "pairs.json" else {"row_count": len(rows)}))
                   for name in ("rows.jsonl", "test_questions.jsonl", "test_answer_key.jsonl", "pairs.json")},
            test_panel_id=sha256_file(staging/"test_questions.jsonl"),
            validation={"rows": len(rows), "pairs": len(groups), "external_semantic_violations": 0})
        write_json(staging / "manifest.json", manifest)
        for path in staging.iterdir():
            existing = output_directory / path.name
            if existing.exists() and existing.read_bytes() != path.read_bytes():
                raise ValueError(f"refusing to overwrite a different published diagnostic: {existing}")
        output_directory.mkdir(exist_ok=True)
        for name in (*manifest["files"], "manifest.json"):
            shutil.copyfile(staging / name, output_directory / name)
    return manifest


def main():
    """Build this CPU-only diagnostic; never submit generation jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    manifest = build_panel(parser.parse_args().output_directory)
    print(json.dumps({k: manifest[k] for k in ("row_count", "group_count", "test_panel_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
