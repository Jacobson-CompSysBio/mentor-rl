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
const source = ["compare.js", "charts.js", "page.js"].map((f) => fs.readFileSync(`${process.argv[1]}/${f}`, "utf8")).join("\\n");
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

const tool = { type: "function", function: { name: "get_edge", parameters: {
  properties: { layer_id: { type: "string" }, gene_ids: { type: "array", items: { type: "string" } } }, required: ["layer_id"] } } };
assert.strictEqual(c.toolText([tool]), "get_edge(layer_id: string, gene_ids: list of strings (optional))");

const columns = c.columnChart({ label: "x", threshold: { value: 0.95, label: "gate 95%" },
  items: [{ label: "v1", value: null, tip: "none" }, { label: "v2", value: 0.9, tip: "v2: 90%" }] });
assert.ok(columns.includes("no data") && columns.includes("height:90%") && columns.includes("bottom:95%"));
const bars = c.barRows({ label: "x", threshold: null, series: { value: "v2", compare: "v1" },
  groups: [{ title: "S1.1", rows: [{ label: "S1.1.1", sublabel: "direct", value: 0.5, compare: 0.25, tip: "t" }] }] });
assert.ok(bars.includes("width:50%") && bars.includes('class="dot" style="left:25%"') && bars.includes('class="key dot"'));

const table = c.changelogView({ stage: "S0", current: "v4", versions: [
  { version: "v3", summary: "Baseline.", gold: [], gate: { passed: false, run: "r3", metrics: { status_exact: 0.9518 } }, vs_previous: null },
  { version: "v4", summary: null, description: "More rows.", gold: [{}], gate: { passed: true, run: "r4", metrics: { status_exact: 0.9999 } },
    vs_previous: { version: "v3", reasons: ["the answer keys differ"] } },
] }, ["status_exact"]);
assert.ok(table.indexOf("<b>v4</b>") < table.indexOf("<b>v3</b>"), "newest version first");
assert.ok(table.includes("not on disk") && table.includes("99.99%") && table.includes("More rows."));
assert.ok(table.includes("vs v3: the answer keys differ"));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_scripts():
    subprocess.run(["node", "-e", CHECKS, str(TEMPLATES)], check=True)
