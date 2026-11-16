"""The stage page's scripts mark words, align turns, and draw charts (run with node).

The page joins compare.js, charts.js, and page.js into one script; this test
loads them the same way, in a sandbox that receives their exports.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "dashboard" / "templates"

CHECKS = """
const fs = require("fs");
const vm = require("vm");
const box = { exports: {} };
const source = ["compare.js", "charts.js", "failures.js", "page.js"].map((f) => fs.readFileSync(`${process.argv[1]}/${f}`, "utf8")).join("\\n");
vm.runInNewContext(source, { module: box, JSON, Object, Math, Set, Date, String, Number, Uint32Array, Array, RegExp });
const c = box.exports;
const assert = require("assert");
// Arrays made inside the sandbox have its own Array prototype, so compare their JSON.
const same = (actual, expected) => assert.strictEqual(JSON.stringify(actual), JSON.stringify(expected));

same(c.wordDiff("so stop now", "so continue now"),
  ["so <del>stop</del> now", "so <ins>continue</ins> now"]);

const ask = { role: "user", content: "ask" };
const answer = (thinking) => ({ role: "assistant", thinking, content: { status: "resolved" } });
const rows = c.turnRows([ask, answer("old reason")], [ask, answer("new reason")], true);
same(rows.map((r) => [r.label, r.changed]), [["User", false], ["Thinking", true], ["Answer", false]]);
assert.strictEqual(rows[1].left, "<del>old</del> reason");
assert.strictEqual(rows[1].right, "<ins>new</ins> reason");
assert.strictEqual(c.turnRows(null, [ask], true)[0].left, null);

const turns = c.blocks([ask, { role: "assistant", content: { status: "ambiguous" } }, ask, { role: "assistant", content: {} }]);
same(turns.map((t) => t.masked), [true, true, false, false]);
assert.strictEqual(c.pretty({ status: "resolved", ids: [1, 2] }), '{"status": "resolved", "ids": [1, 2]}');
assert.ok(c.pretty({ text: "x".repeat(120) }).includes('\\n  "text"'));

// Sorting by accuracy: ties keep catalog order, unscored types go last.
const scored = { version: "v2", gate: { types: { "S1.1.1": [9, 10], "S1.1.2": [5, 10], "S1.1.3": [9, 10] } } };
const listed = ["S1.1.1", "S1.1.2", "S1.1.3", "S1.1.4"];
same(c.sortByAccuracy(listed, scored, "worst"), ["S1.1.2", "S1.1.1", "S1.1.3", "S1.1.4"]);
same(c.sortByAccuracy(listed, scored, "best"), ["S1.1.1", "S1.1.3", "S1.1.2", "S1.1.4"]);
same(c.sortByAccuracy(listed, null, "worst"), listed);
assert.strictEqual(c.accuracyBasis({ version: "v1", gate: null }, scored), scored);

// Links use only letters, digits, and hyphens, so they survive being shared.
assert.strictEqual(c.anchor("v9_v10", "S1.3.8"), "v9-v10--s1-3-8");
assert.strictEqual(c.slug("S2.1.single_edge"), "s2-1-single-edge");

// Downloads: JSON with one corpus row per line, and Markdown grouped by question type.
const json = c.rowsJson([{ row_id: "a" }, { row_id: "b" }]);
same(JSON.parse(json), [{ row_id: "a" }, { row_id: "b" }]);
assert.strictEqual(json.split("\\n").length, 5);
const v = { version: "v2", corpus_root: "data/s1_v2", names: { "S1.1.1": "direct" },
  prompts: [{ system_prompt: "Be exact.", tools: [], families: ["S1.1"] }],
  rows: { r1: { messages: [ask, { role: "assistant", thinking: "look", content: { status: "resolved" } }], prompt: 0 } } };
const md = c.tracesMarkdown({ families: { "S1.1": { title: "Symbol to Ensembl ID" } }, type_text: { "S1.1.1": "One lookup." } },
  v, [{ type: "S1.1.1", outcome: "calls: 0 · resolved / stop", row_id: "r1", state: "edited" }], "S1 v2 gold traces");
for (const part of ["# S1 v2 gold traces", "## S1.1 Symbol to Ensembl ID", "### S1.1.1 direct", "One lookup.",
  "#### calls: 0 · resolved / stop · edited", "Row `r1`", "**User**\\n\\nask", "**Answer**\\n\\n```json", "## System prompt"]) {
  assert.ok(md.includes(part), `markdown has ${part}`);
}

// S2 gives each family its own system prompt: downloads put back the row's own.
const s2 = { version: "v1", corpus_root: "data/s2", names: {}, rows: { e: { messages: [ask], prompt: 0 }, p: { messages: [ask], prompt: 1 } },
  prompts: [{ system_prompt: "Edges.", tools: [], families: ["S2.1"] }, { system_prompt: "Paths.", tools: [], families: ["S2.11"] }] };
assert.strictEqual(c.fullRow(s2, "p").messages[0].content, "Paths.");
const s2md = c.tracesMarkdown({ families: {}, type_text: {} }, s2, [{ type: "S2.11.1", outcome: "x", row_id: "p" }], "S2 v1");
assert.ok(s2md.includes("## System prompt: S2.11\\n") && !s2md.includes("Edges."));
const shared = c.sharedView({ version: "v0", prompts: [s2.prompts[0]] }, s2);
assert.ok(shared.includes("System prompt · S2.1") && shared.includes('class="state added"'));

const tool = { type: "function", function: { name: "get_edge", parameters: {
  properties: { layer_id: { type: "string" }, gene_ids: { type: "array", items: { type: "string" } } }, required: ["layer_id"] } } };
assert.strictEqual(c.toolText([tool]), "get_edge(layer_id: string, gene_ids: list of strings (optional))");

const columns = c.columnChart({ label: "x", threshold: { value: 0.95, label: "gate 95%" },
  items: [{ label: "v1", value: null, tip: "none" }, { label: "v2", value: 0.9, tip: "v2: 90%" }] });
assert.ok(columns.includes("no data") && columns.includes("height:90%") && columns.includes("bottom:95%"));
const bars = c.barRows({ label: "x", threshold: null, series: { value: "v2", compare: "v1" },
  groups: [{ title: "S1.1", rows: [{ label: "S1.1.1", sublabel: "direct", value: 0.5, compare: 0.25, tip: "t" }] }] });
assert.ok(bars.includes("width:50%") && bars.includes('class="dot" style="left:25%"') && bars.includes('class="key dot"'));

// Histogram bins hold their lower edge; the last also holds 100%.
const binned = c.binRates([{ name: "a", value: 0 }, { name: "b", value: 0.95 }, { name: "c", value: 1 },
  { name: "d", value: 0.9499 }, { name: "e", value: 61 / 64 }], 0.05);
assert.strictEqual(binned.length, 20);
same(binned[0].items.map((i) => i.name), ["a"]);
same(binned[18].items.map((i) => i.name), ["d"]);
same(binned[19].items.map((i) => i.name), ["b", "c", "e"]);
assert.strictEqual(c.binRates([], 0.01).length, 100);
const hist = c.histogram({ label: "x", threshold: { value: 0.95, label: "gate 95%" },
  series: { value: "v2", compare: "v1", unit: "types" },
  bins: [{ lo: 0, hi: 0.5, count: 0, compare: 2, tip: "low" }, { lo: 0.5, hi: 1, count: 4, compare: 1, tip: "high" }] });
assert.ok(hist.includes('class="vthreshold" style="left:95%"') && hist.includes('class="key step"'));
assert.ok(hist.includes('class="hist-bar" style="height:100%"') && hist.includes("0,100 0,50 50,50 50,75 100,75 100,100"));
// An empty bin draws no bar; one with only a compare count can still take focus.
assert.strictEqual((hist.match(/class="hist-bar"/g) || []).length, 1);
assert.strictEqual((hist.match(/tabindex="0"/g) || []).length, 2);

// The Question types tab shows the spread across types, with a compare version and the gate's mark.
const gateOf = (rows) => ({ thresholds: { structured_answer_accuracy: 0.95 }, profile: "structured_95_v1",
  groups: { by: "question type", rows: Object.fromEntries(Object.entries(rows).map(([name, value]) =>
    [name, { rates: { structured_answer_accuracy: value }, total: 10 }])) } });
const typesData = { stage: "S1", current: "v2", families: {}, type_text: {}, versions: [
  { version: "v1", names: {}, gold: [], gate: gateOf({ "S1.1.1": 0.5, "S1.1.2": 1 }) },
  { version: "v2", names: {}, gold: [], gate: gateOf({ "S1.1.1": 0.9, "S1.1.2": 1, "S1.1.3": 1 }) }] };
const tab = c.typesView(typesData, { bins: "0.1" });
assert.ok(tab.includes("Spread across question types") && tab.includes('<option value="0.1" selected>'));
assert.ok(tab.includes("<b>v2</b>: 3 types · median 100.00% · lowest 90.00% · 2 at 100% · 1 below 95.00%"));
assert.ok(tab.includes("<b>v1</b>: 2 types · median 75.00%") && tab.includes("90% to 100%"));
assert.ok(tab.indexOf("Spread across question types") < tab.indexOf("Each question type"));
// Rates by question family (S0) have no histogram.
const families = c.typesView({ ...typesData, versions: [{ ...typesData.versions[1],
  gate: { ...typesData.versions[1].gate, groups: { ...typesData.versions[1].gate.groups, by: "question family" } } }] }, {});
assert.ok(!families.includes("Spread across"));

const table = c.changelogView({ stage: "S0", current: "v4", versions: [
  { version: "v3", summary: "Baseline.", gold: [], gate: { passed: false, run: "r3", metrics: { status_exact: 0.9518 } }, vs_previous: null },
  { version: "v4", summary: null, description: "More rows.", gold: [{}], gate: { passed: true, run: "r4", metrics: { status_exact: 0.9999 } },
    vs_previous: { version: "v3", reasons: ["the answer keys differ"] } },
] }, ["status_exact"]);
assert.ok(table.indexOf("<b>v4</b>") < table.indexOf("<b>v3</b>"), "newest version first");
assert.ok(table.includes("not on disk") && table.includes("99.99%") && table.includes("More rows."));
assert.ok(table.includes("vs v3: the answer keys differ"));
// Failures tab: cases filter by family and category and sort worst first; a case lists its rows.
const failed = { stage: "S2", families: { "S2.1": { title: "Edges" } }, type_text: {}, versions: [{ version: "v1", names: { "S2.1.2": "edge" } }],
  failures: { version: "v1", run: "r", profile: "report_only_v1", rows: 4, correct: 1,
    fields: { status: 3, result: 1, decision: 3, provenance: 3, blocker: 3, "tool calls": 2 },
    family_fields: { "S2.1": { rows: 4, correct: 1, status: 3, result: 1, decision: 3, provenance: 3, blocker: 3, "tool calls": 2 } },
    categories: { "final answer: result": 2, "invalid: wrong arguments at call 1": 1 },
    family_categories: { "S2.1": { "final answer: result": 2, "invalid: wrong arguments at call 1": 1 } },
    cases: {
      "S2.1.1": { rows: 2, correct: 1, fields: {}, categories: { "final answer: result": 1 }, paths: [], examples: [] },
      "S2.1.2": { rows: 2, correct: 0, fields: { status: 1, result: 0, decision: 1, provenance: 1, blocker: 1, "tool calls": 1 },
        categories: { "invalid: wrong arguments at call 1": 1, "final answer: result": 1 }, paths: [[".result.ids[] (length)", 1]],
        examples: [{ row_id: "r1", category: "final answer: result", failed: ["result"], error: null,
          messages: [{ role: "user", content: "ask" }, { role: "assistant", thinking: "t", content: { status: "resolved" } }],
          expected: { tool_calls: [], status: "resolved", result: { ids: ["A"] }, continuation_decision: "stop", provenance: {}, blocked_reason_required: false },
          diffs: [{ path: ".result.ids[]", kind: "length", expected: 1, got: 0, only_expected: ["A"], only_got: [] }], diff_count: 1 }] },
      "S2.1.3": { rows: 2, correct: 2, fields: {}, categories: {}, paths: [], examples: [] },
    } } };
same(c.failedCases(failed, { family: "all", category: "all", order: "worst" }), ["S2.1.2", "S2.1.1"]);
same(c.failedCases(failed, { family: "all", category: "invalid: wrong arguments at call 1", order: "worst" }), ["S2.1.2"]);
same(c.failedCases(failed, { family: "S2.9", category: "all", order: "catalog" }), []);
const view = c.failuresView(failed, { family: "all", category: "all", order: "worst" });
assert.ok(view.includes("25.00%") && view.includes('id="failures--s2-1-2"') && view.includes("edge") && !view.includes("S2.1.3"));
const body = c.caseBody(failed, "S2.1.2");
assert.ok(body.includes(".result.ids[]") && body.includes("only here") && body.includes("answer key") && body.includes("rows.jsonl"));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_scripts():
    subprocess.run(["node", "-e", CHECKS, str(TEMPLATES)], check=True)
