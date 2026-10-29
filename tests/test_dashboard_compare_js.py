"""The stage page's comparison script marks words and aligns turns (run with node)."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "compare.js"

CHECKS = """
const assert = require("assert");
const c = require(process.argv[1]);

assert.deepStrictEqual(c.wordDiff("so stop now", "so continue now"),
  ["so <del>stop</del> now", "so <ins>continue</ins> now"]);

const ask = { role: "user", content: "ask" };
const answer = (thinking) => ({ role: "assistant", thinking, content: { status: "resolved" } });
const rows = c.turnRows([ask, answer("old reason")], [ask, answer("new reason")], true);
assert.deepStrictEqual(rows.map((r) => [r.label, r.changed]), [["User", false], ["Thinking", true], ["Answer", false]]);
assert.strictEqual(rows[1].left, "<del>old</del> reason");
assert.strictEqual(rows[1].right, "<ins>new</ins> reason");
assert.strictEqual(c.turnRows(null, [ask], true)[0].left, null);

const turns = c.blocks([ask, { role: "assistant", content: { status: "ambiguous" } }, ask, { role: "assistant", content: {} }]);
assert.deepStrictEqual(turns.map((t) => t.masked), [true, true, false, false]);

assert.strictEqual(c.pretty({ status: "resolved", ids: [1, 2] }), '{"status": "resolved", "ids": [1, 2]}');
assert.ok(c.pretty({ text: "x".repeat(120) }).includes('\\n  "text"'));

const table = c.changelogView({ stage: "S0", current: "v4", versions: [
  { version: "v3", summary: "Baseline.", gold: [], gate: { passed: false, run: "r3", metrics: { status_exact: 0.9518 } }, vs_previous: null },
  { version: "v4", summary: null, gold: [{}], gate: { passed: true, run: "r4", metrics: { status_exact: 0.9999 } },
    vs_previous: { version: "v3", reasons: ["the answer keys differ"] } },
] });
assert.ok(table.indexOf("<b>v4</b>") < table.indexOf("<b>v3</b>"), "newest version first");
assert.ok(table.includes("not on disk") && table.includes("99.99%") && table.includes("no summary yet"));
assert.ok(table.includes("vs v3: the answer keys differ"));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_compare_script():
    subprocess.run(["node", "-e", CHECKS, str(SCRIPT)], check=True)
