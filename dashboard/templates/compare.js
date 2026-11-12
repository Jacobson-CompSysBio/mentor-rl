// Traces and version comparison, built in the browser.
//
// The stage page embeds each version's gold traces once (dashboard/render.py),
// plus a small record per pair of versions saying how the traces pair up
// (dashboard/diff.py). Picking two versions renders that pair here, so the
// page does not hold a pre-rendered copy of every pair. charts.js draws the
// charts and page.js wires the tabs; render.py joins the three files.

const TOKEN = /\s+|[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu;
const CODE_LABELS = new Set(["Tool call", "Tool result", "Answer"]);

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inline(text) {
  // Escaped text with `code` spans, for the hand-written descriptions.
  return esc(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}

function pretty(value) {
  // Text as it is, short JSON on one line, longer JSON indented.
  if (typeof value === "string") return value;
  if (value === undefined) return "";
  const compact = JSON.stringify(value, null, 1).replace(/,\n\s*/g, ", ").replace(/\n\s*/g, "");
  return compact.length <= 100 ? compact : JSON.stringify(value, null, 2);
}

function blocks(messages) {
  // Labeled turns in the question-doc layout. In a multi-turn row, turns
  // before the last user message are context the model saw but was not
  // trained on, so they are marked masked.
  const users = messages.flatMap((m, i) => (m.role === "user" ? [i] : []));
  const lastUser = users.length > 1 ? users[users.length - 1] : -1;
  const turns = [];
  messages.forEach((m, i) => {
    const add = (label, value) => turns.push({ label, text: pretty(value), masked: i < lastUser, code: CODE_LABELS.has(label) });
    if (m.role === "harness") add("Context (from the harness)", m.content ?? "");
    else if (m.role === "user") add("User", m.content ?? "");
    else if (m.role === "tool") add("Tool result", m.content);
    else if (m.role === "assistant") {
      if (m.thinking) add("Thinking", m.thinking);
      if (m.tool_call) add("Tool call", `${m.tool_call.name}(${pretty(m.tool_call.arguments)})`);
      else add("Answer", m.content);
    }
  });
  return turns;
}

function diffOps(a, b) {
  // Align two sequences by their longest common subsequence. Returns steps
  // ["=", i, j] (same), ["-", i, null] (only in a), ["+", null, j] (only in b).
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let endA = a.length, endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) { endA--; endB--; }
  const ops = [];
  for (let k = 0; k < start; k++) ops.push(["=", k, k]);
  const n = endA - start, m = endB - start, width = m + 1;
  if (n * m > 4e6) {
    // Too large to align word by word: mark the whole middle as replaced.
    for (let i = start; i < endA; i++) ops.push(["-", i, null]);
    for (let j = start; j < endB; j++) ops.push(["+", null, j]);
  } else {
    const table = new Uint32Array((n + 1) * width);
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        table[i * width + j] = a[start + i] === b[start + j]
          ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
      }
    }
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (a[start + i] === b[start + j]) { ops.push(["=", start + i, start + j]); i++; j++; }
      else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) { ops.push(["-", start + i, null]); i++; }
      else { ops.push(["+", null, start + j]); j++; }
    }
    for (; i < n; i++) ops.push(["-", start + i, null]);
    for (; j < m; j++) ops.push(["+", null, start + j]);
  }
  for (let k = 0; k < a.length - endA; k++) ops.push(["=", endA + k, endB + k]);
  return ops;
}

function wordDiff(oldText, newText) {
  // Old text with removed words in <del>, new text with added words in <ins>.
  const a = oldText.match(TOKEN) || [], b = newText.match(TOKEN) || [];
  let left = "", right = "", removed = "", added = "";
  const flush = () => {
    if (removed) left += `<del>${esc(removed)}</del>`;
    if (added) right += `<ins>${esc(added)}</ins>`;
    removed = added = "";
  };
  for (const [op, i, j] of diffOps(a, b)) {
    if (op === "=") { flush(); left += esc(a[i]); right += esc(b[j]); }
    else if (op === "-") removed += a[i];
    else added += b[j];
  }
  flush();
  return [left, right];
}

function turnRows(oldMessages, newMessages, markWords) {
  // One row per turn: the old text on the left, the new on the right, null
  // where a side has no such turn. A regenerated trace is a different
  // question, so its words are not marked.
  const left = oldMessages ? blocks(oldMessages) : [], right = newMessages ? blocks(newMessages) : [];
  return diffOps(left.map((t) => t.label), right.map((t) => t.label)).map(([op, i, j]) => {
    const before = op === "+" ? null : left[i], after = op === "-" ? null : right[j];
    const turn = after || before;
    const changed = !before || !after || before.text !== after.text;
    let l = before ? esc(before.text) : null, r = after ? esc(after.text) : null;
    if (before && after && changed && markWords) [l, r] = wordDiff(before.text, after.text);
    return { label: turn.label, masked: turn.masked, code: turn.code, changed, left: l, right: r };
  });
}

// ---- Shared helpers ----

const number = (n) => n.toLocaleString("en-US");
const percent = (pair) => `${((100 * pair[0]) / pair[1]).toFixed(2)}%`;
const rate = (value) => `${(100 * value).toFixed(2)}%`;
const SPLITS = ["train", "validation", "test"];

function familyOf(type) {
  // S1.3.8 and S2.1.single_edge belong to S1.3 and S2.1; S0.4 belongs to S0.
  const match = type.match(/^(S\d+\.\d+)\./) || type.match(/^(S\d+)\./);
  return match ? match[1] : type;
}

function slug(text) {
  // A name safe in a shared #link: ASCII letters, digits, and hyphens (S1.3.8 becomes s1-3-8).
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function anchor(scope, type) {
  // The element id, and #link, of one question type in a pair or version view: v9-v10--s1-3-8.
  return `${slug(scope)}--${slug(type)}`;
}

function expandButtons(scope) {
  // Open or close every trace in one question type, or in the whole list.
  const button = (open, text) => `<button type="button" class="link" data-open="${open}" data-scope="${scope}">${text}</button>`;
  return `<span class="section-tools">${button(1, "expand all")} · ${button(0, "collapse all")}</span>`;
}

function downloadButtons(v, what, count, key, side) {
  // One version's gold traces (all, or those changed in a pair) as JSON or Markdown.
  const attrs = `type="button" class="link" data-version="${esc(v.version)}" data-what="${what}"` +
    (key ? ` data-pair="${esc(key)}" data-side="${side}"` : "");
  return `${what === "gold" ? "all gold traces" : "changed traces"} (${count}): ` +
    `<button ${attrs} data-format="json">JSON</button> · <button ${attrs} data-format="md">Markdown</button>`;
}

function toolText(tools) {
  // A version's tool list as readable signatures, so a change can be word-diffed.
  return (tools || []).map((tool) => {
    const f = tool.function || tool;
    const props = (f.parameters && f.parameters.properties) || {};
    const required = new Set((f.parameters && f.parameters.required) || []);
    const args = Object.entries(props).map(([name, spec]) =>
      `${name}: ${spec.type === "array" ? `list of ${(spec.items && spec.items.type) || "value"}s` : spec.type}` +
      (required.has(name) ? "" : " (optional)"));
    return `${f.name}(${args.join(", ")})` + (f.description ? `\n  ${f.description}` : "");
  }).join("\n\n");
}

// ---- Compare tab ----

const ORDERS = { worst: "worst accuracy first", best: "best accuracy first" };

function accuracyBasis(...versions) {
  // The newest of these versions whose gate run scores each question type, or null.
  return versions.slice().reverse().find((v) => v && v.gate && Object.keys(v.gate.types).length) || null;
}

function typeRate(basis, type) {
  const pair = basis && basis.gate.types[type];
  return pair && pair[1] ? pair[0] / pair[1] : null;
}

function sortByAccuracy(items, basis, order, typeOf = (item) => item) {
  // Order question types by the basis version's accuracy, worst or best first.
  // Types it does not score go last; ties keep catalog order.
  if (!basis || !(order in ORDERS)) return items;
  return items.map((item, index) => [item, typeRate(basis, typeOf(item)), index]).sort((x, y) => {
    if (x[1] === null || y[1] === null) return (x[1] === null) - (y[1] === null) || x[2] - y[2];
    return (order === "worst" ? x[1] - y[1] : y[1] - x[1]) || x[2] - y[2];
  }).map(([item]) => item);
}

function orderNote(basis, order) {
  // Says what the list is sorted by, or why it is not.
  if (!(order in ORDERS)) return "";
  return basis ? ` <span class="count">by ${esc(basis.version)} accuracy, ${ORDERS[order]}</span>`
    : ' <span class="count">in catalog order: no version here has per-type accuracy</span>';
}

function isChanged(type) {
  const { edited, regenerated, removed } = type.rows;
  return edited || regenerated || removed || Object.keys(type.gold).some((state) => state !== "identical");
}

function sideBySide(rows, leftHead, rightHead) {
  const cell = (side, html, head) =>
    `<div class="cell ${side}" data-side="${esc(head)}">${html === null ? '<span class="empty">—</span>' : html}</div>`;
  const body = rows.map((r) =>
    `<div class="pair-row${r.masked ? " masked" : ""}${r.code ? " code" : ""}${r.changed ? " changed" : ""}">` +
    `<div class="label">${esc(r.label)}${r.masked ? " · in context, masked" : ""}</div>` +
    cell("old", r.left, leftHead) + cell("new", r.right, rightHead) + "</div>").join("");
  return `<div class="pair"><div class="pair-head"><div>${esc(leftHead)}</div><div>${esc(rightHead)}</div></div>${body}</div>`;
}

function singleView(rows, head) {
  // One version's trace in a single column.
  return `<div class="pair single"><div class="pair-head"><div>${esc(head)}</div></div>` + rows.map((r) =>
    `<div class="pair-row${r.masked ? " masked" : ""}${r.code ? " code" : ""}">` +
    `<div class="label">${esc(r.label)}${r.masked ? " · in context, masked" : ""}</div><div class="cell">${r.right}</div></div>`).join("") + "</div>";
}

function traceView(trace, a, b) {
  // Identical traces show the newer row on both sides: it is the same row.
  const oldRow = trace.state === "identical" ? b.rows[trace.new] : trace.old && a.rows[trace.old];
  const newRow = trace.new && b.rows[trace.new];
  const rows = turnRows(oldRow && oldRow.messages, newRow && newRow.messages, trace.state === "edited");
  const head = (v, id) => `${v.version} · ${id ? `row ${id}` : "not present"}`;
  const note = trace.state === "regenerated" ? ' · <span class="fields">different questions; words not marked</span>' : "";
  const fields = trace.fields.length ? ` · <span class="fields">${esc(trace.fields.join(", "))}</span>` : "";
  return `<details class="trace ${trace.state}"${trace.state === "edited" ? " open" : ""}>` +
    `<summary><span class="state ${trace.state}">${trace.state}</span> ${esc(trace.outcome)}${fields}${note}</summary>` +
    sideBySide(rows, head(a, trace.old), head(b, trace.new)) + "</details>";
}

function promptPairs(a, b) {
  // Pair each system prompt of the newer version with the older one's for the same families.
  const find = (families) => a.prompts.find((p) => p.families.some((f) => families.includes(f))) || null;
  const pairs = b.prompts.map((p) => [find(p.families), p]);
  for (const p of a.prompts) if (!pairs.some(([left]) => left === p)) pairs.push([p, null]);
  return pairs;
}

function sharedLabel(v, prompt) {
  // "" when every row of the version shares one prompt; else the families that use this one.
  return v.prompts.length > 1 ? ` · ${prompt.families.join(", ")}` : "";
}

function sharedView(a, b) {
  // The system prompt and tool list that a version's rows share (one per family
  // in S2), which the traces below leave out. Opened and marked when they differ.
  return promptPairs(a, b).flatMap(([before, after]) => {
    const scope = sharedLabel(after ? b : a, after || before);
    return [[`System prompt${scope}`, before && before.system_prompt, after && after.system_prompt, false],
      [`Tool definitions${scope}`, before && toolText(before.tools), after && toolText(after.tools), true]];
  }).map(([label, before, after, code]) => {
    const l = before === null ? null : pretty(before), r = after === null ? null : pretty(after), changed = l !== r;
    const [left, right] = changed && l !== null && r !== null ? wordDiff(l, r) : [l === null ? null : esc(l), r === null ? null : esc(r)];
    const state = l === null ? "added" : r === null ? "removed" : changed ? "edited" : "identical";
    return `<details class="trace ${state}"${changed ? " open" : ""}><summary><span class="state ${state}">${state}</span> ${esc(label)}</summary>` +
      sideBySide([{ label, masked: false, code, changed, left, right }], a.version, b.version) + "</details>";
  }).join("");
}

function promptsView(v) {
  // One version's system prompts and tool lists, which its traces leave out.
  return `<h2>${v.prompts.length > 1 ? "System prompts and tools by family" : "Shared by every row"}</h2><div class="qtype">` +
    v.prompts.flatMap((p) => [[`System prompt${sharedLabel(v, p)}`, p.system_prompt, false], [`Tool definitions${sharedLabel(v, p)}`, toolText(p.tools), true]])
      .map(([label, text, code]) => `<details class="trace"><summary>${esc(label)}</summary>` +
        singleView([{ label, masked: false, code, right: esc(pretty(text)) }], v.version) + "</details>").join("") + "</div>";
}

function typeView(data, type, traces, a, b, key, showUnchanged) {
  const before = a.gate && a.gate.types[type.type], after = b.gate && b.gate.types[type.type];
  let accuracy = "";
  if (before && after) {
    const direction = after[0] > before[0] ? "up" : after[0] < before[0] ? "down" : "";
    accuracy = `<span class="acc ${direction}">${before[0]} → ${after[0]} of ${after[1]} correct ` +
      `(${percent(before)} → ${percent(after)})</span> · `;
  } else if (before || after) {
    const only = after || before, v = after ? b : a;
    accuracy = `<span class="acc">${v.version}: ${only[0]} of ${only[1]} correct (${percent(only)})</span> · `;
  }
  const r = type.rows;
  let rows = `${number(r.edited)} of ${number(r.total)} training rows edited`;
  if (r.regenerated) rows += `, ${number(r.regenerated)} regenerated`;
  if (r.removed) rows += `, ${number(r.removed)} removed`;
  const fields = Object.entries(type.fields).map(([label, count]) => `${esc(label)}: ${number(count)}`).join(" · ");
  const shown = traces.filter((t) => showUnchanged || t.state !== "identical");
  const hidden = traces.length - shown.length;
  const text = data.type_text[type.type];
  return `<article class="qtype${isChanged(type) ? "" : " unchanged"}" id="${anchor(key, type.type)}">` +
    `<h3>${esc(type.type)} <span class="name">${esc(type.name)}</span>${shown.length > 1 ? expandButtons("type") : ""}</h3>` +
    (text ? `<p class="type-text">${inline(text)}</p>` : "") +
    `<p class="meta">${accuracy}${rows}${fields ? `<br><span class="fields">${fields}</span>` : ""}</p>` +
    (shown.length ? "" : '<p class="muted">No gold trace changed. The edits counted above are in other rows of this type.</p>') +
    shown.map((t) => traceView(t, a, b)).join("") +
    (hidden ? `<p class="muted small">${hidden} unchanged gold trace${hidden > 1 ? "s" : ""} hidden.</p>` : "") +
    "</article>";
}

function splitsView(data, versions) {
  // Rows per split for one version, or two side by side, and per question type.
  const shown = versions.filter((v) => v.splits);
  if (!shown.length) return "";
  const head = shown.map((v) => `<th class="num">${esc(v.version)}</th>`).join("");
  const changedCell = (values) => (shown.length === 2 && values[0] !== values[1] ? " changed" : "");
  const row = (label, values) => `<tr class="${changedCell(values).trim()}"><td>${label}</td>` +
    values.map((n) => `<td class="num">${n === undefined ? "—" : number(n)}</td>`).join("") + "</tr>";
  const total = (v) => SPLITS.reduce((sum, split) => sum + (v.splits.rows[split] || 0), 0);
  let html = '<h2>Questions and splits</h2><div class="table-wrap"><table class="splits">' +
    `<thead><tr><th>Rows</th>${head}</tr></thead><tbody>` +
    SPLITS.map((split) => row(split, shown.map((v) => v.splits.rows[split]))).join("") +
    row("<b>total</b>", shown.map(total)) +
    [...new Set(shown.flatMap((v) => Object.keys(v.splits.replay)))]
      .map((stage) => row(`of train: ${esc(stage)} replay`, shown.map((v) => v.splits.replay[stage]))).join("") +
    "</tbody></table></div>";
  const types = [...new Set(shown.flatMap((v) => Object.keys(v.splits.types)))]
    .sort((x, y) => x.localeCompare(y, "en", { numeric: true }));
  if (types.length) {
    const cols = SPLITS.flatMap((split) => shown.map((v) => [split, v]));
    html += `<details class="table-view"><summary>Rows per question type (${types.length})</summary><div class="table-wrap"><table class="splits">` +
      `<thead><tr><th>Type</th>${cols.map(([split, v]) => `<th class="num">${split}${shown.length > 1 ? ` ${esc(v.version)}` : ""}</th>`).join("")}</tr></thead><tbody>` +
      types.map((type) => {
        const values = cols.map(([split, v]) => (v.splits.types[type] || {})[split]);
        const differs = shown.length === 2 && SPLITS.some((split) => (shown[0].splits.types[type] || {})[split] !== (shown[1].splits.types[type] || {})[split]);
        return `<tr class="${differs ? "changed" : ""}"><td class="mono">${esc(type)}</td>` +
          values.map((n) => `<td class="num">${n === undefined ? "—" : number(n)}</td>`).join("") + "</tr>";
      }).join("") + "</tbody></table></div></details>";
  }
  return html;
}

function pairView(data, a, b, pair, showUnchanged, order = "catalog") {
  const key = `${a.version}_${b.version}`;
  const parts = [];

  parts.push('<ul class="changes">' + pair.changes.map((c) =>
    `<li><b>${esc(c.version)}</b> ${c.summary ? inline(c.summary) : '<span class="muted">no summary yet: add <code>summary:</code> to this version in the registry</span>'}</li>`).join("") + "</ul>");

  const g = pair.gold, r = pair.rows;
  const frozen = Object.entries(pair.held_out_frozen).filter(([, same]) => !same).map(([name]) => name);
  const accuracy = a.gate && b.gate && a.gate.overall && b.gate.overall
    ? `<div class="tile-value">${percent(a.gate.overall)} → ${percent(b.gate.overall)}</div>` +
      `<div class="tile-note">gate ${a.gate.passed ? "passed" : "failed"} → ${b.gate.passed ? "passed" : "failed"}</div>`
    : '<div class="tile-value muted">Not available</div>';
  parts.push('<div class="tiles">' +
    `<div class="tile"><div class="tile-label">Gold traces</div><div class="tile-value">${g.edited || 0} edited</div>` +
    `<div class="tile-note">${g.regenerated || 0} regenerated · ${g.identical || 0} identical` +
    `${g.added ? ` · ${g.added} added` : ""}${g.removed ? ` · ${g.removed} removed` : ""}</div></div>` +
    `<div class="tile"><div class="tile-label">Training rows</div><div class="tile-value">${number(r.edited)} edited</div>` +
    `<div class="tile-note">${number(r.regenerated)} regenerated, of ${number(r.total)}</div></div>` +
    `<div class="tile"><div class="tile-label">Held-out splits</div>` +
    (frozen.length ? `<div class="tile-value bad">Changed</div><div class="tile-note">${esc(frozen.join(", "))}</div>`
      : '<div class="tile-value ok">Unchanged</div><div class="tile-note">validation and test are byte-identical</div>') +
    `</div><div class="tile"><div class="tile-label">Structured accuracy</div>${accuracy}</div></div>`);
  if (pair.not_comparable.length) {
    parts.push(`<p class="warn">These accuracies are not directly comparable: ${esc(pair.not_comparable.join("; "))}. ` +
      "Read the changes below as context, not as the effect of the data change.</p>");
  }
  if (data.notes[key]) parts.push(`<div class="note">${data.notes[key]}</div>`);
  const byFamily = a.prompts.length > 1 || b.prompts.length > 1;
  parts.push(`<h2>${byFamily ? "System prompts and tools by family" : "Shared by every row"}</h2><div class="qtype">${sharedView(a, b)}</div>`);

  const tracesByType = {};
  for (const t of pair.traces) (tracesByType[t.type] = tracesByType[t.type] || []).push(t);
  const basis = accuracyBasis(a, b);
  const types = sortByAccuracy(showUnchanged ? pair.types : pair.types.filter(isChanged), basis, order, (t) => t.type);
  parts.push(`<h2>${showUnchanged ? "Question types" : "Changed question types"} <span class="count">${types.length}</span>` +
    `${orderNote(basis, order)}</h2>`);
  parts.push('<div class="type-list">' + (types.length ? `<p class="small">${expandButtons("list")}</p>` : "") +
    '<nav class="chips">' + types.map((t) =>
      `<a class="${isChanged(t) ? "" : "unchanged"}" href="#${anchor(key, t.type)}">${esc(t.type)}</a>`).join("") + "</nav>" +
    types.map((t) => typeView(data, t, tracesByType[t.type] || [], a, b, key, showUnchanged)).join("") + "</div>");

  const movers = [];
  for (const t of pair.types) {
    const before = a.gate && a.gate.types[t.type], after = b.gate && b.gate.types[t.type];
    if (!isChanged(t) && before && after && before[0] !== after[0]) movers.push({ t, before, after, delta: after[0] - before[0] });
  }
  movers.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  if (movers.length) {
    parts.push(`<h2>Unchanged types whose accuracy moved <span class="count">${movers.length}</span></h2>` +
      '<p class="muted">No training row of these types changed. Their movement comes from the new training run, not from their own rows, so read the gains above against it.</p>' +
      `<div class="table-wrap"><table><thead><tr><th>Type</th><th>Case</th><th class="num">${esc(a.version)}</th><th class="num">${esc(b.version)}</th><th class="num">Change</th><th class="num">Of</th></tr></thead><tbody>` +
      movers.map((m) => `<tr><td class="mono">${esc(m.t.type)}</td><td>${esc(m.t.name)}</td><td class="num">${m.before[0]}</td>` +
        `<td class="num">${m.after[0]}</td><td class="num ${m.delta > 0 ? "ok" : "bad"}">${m.delta > 0 ? "+" : ""}${m.delta}</td><td class="num">${m.after[1]}</td></tr>`).join("") +
      "</tbody></table></div>");
  }

  parts.push(splitsView(data, [a, b]));

  const gate = (v) => (v.gate ? `${esc(v.version)} gate: <code>${esc(v.gate.report)}</code> (${esc(v.gate.profile)}, history ${esc(v.gate.history_policy)})<br>`
    : `${esc(v.version)}: no gate run<br>`);
  const corpus = (v) => `<code>${esc(v.repo_root === "." ? v.corpus_root : `${v.repo_root}/${v.corpus_root}`)}</code>`;
  parts.push("<footer><p>Gold traces: the first training row for each outcome (calls, status, decision, blocked reason) " +
    `of a question type, picked in ${esc(b.version)} and matched by row id in ${esc(a.version)}. ` +
    "Edit counts cover every training row except S0 replay. JSON downloads hold corpus rows, one per line; " +
    "Markdown downloads lay the same traces out by question type.</p>" +
    `<p>Corpora: ${corpus(a)} → ${corpus(b)}<br>${gate(a)}${gate(b)}` +
    `Built ${esc(pair.built_at)} at commit <code>${esc(pair.repo_commit)}</code>.</p></footer>`);
  return parts.join("");
}

function versionView(data, v, order = "catalog") {
  // The gold traces of the only version whose corpus is on disk.
  const missing = data.versions.filter((x) => !x.gold.length).map((x) => x.version);
  const byType = {};
  for (const g of v.gold) (byType[g.type] = byType[g.type] || []).push(g);
  const basis = accuracyBasis(v);
  const types = sortByAccuracy(Object.keys(byType), basis, order);
  return `<h2>${esc(v.version)} gold traces <span class="count">${v.gold.length}</span>${orderNote(basis, order)}</h2>` +
    `<p class="muted">Only one ${esc(data.stage)} corpus is on disk${missing.length ? ` (not ${esc(missing.join(", "))})` : ""}, ` +
    "so there is no pair of versions to compare. The Versions tab lists every version's gate results.</p>" +
    `<p class="downloads">Download ${esc(v.version)} ${downloadButtons(v, "gold", v.gold.length)}</p>` +
    `<div class="type-list"><p class="small">${expandButtons("list")}</p>` +
    types.map((type) => {
      const golds = byType[type], text = data.type_text[type], score = basis && basis.gate.types[type];
      return `<article class="qtype" id="${anchor(v.version, type)}"><h3>${esc(type)} <span class="name">${esc(v.names[type] || "")}</span>` +
        `${golds.length > 1 ? expandButtons("type") : ""}</h3>` +
        (text ? `<p class="type-text">${inline(text)}</p>` : "") +
        (score ? `<p class="meta"><span class="acc">${score[0]} of ${score[1]} correct (${percent(score)})</span></p>` : "") +
        golds.map((g, i) => `<details class="trace"${i === 0 ? " open" : ""}><summary>${esc(g.outcome)}</summary>` +
          singleView(turnRows(null, v.rows[g.row_id].messages, false), `${v.version} · row ${g.row_id}`) + "</details>").join("") +
        "</article>";
    }).join("") + "</div>" + promptsView(v) + splitsView(data, [v]);
}

function fullRow(version, rowId) {
  // Put back the system prompt and tool list the row shares with its version (or, in S2, its family).
  const row = version.rows[rowId], shared = version.prompts[row.prompt];
  return { row_id: rowId, messages: [{ role: "system", content: shared.system_prompt }, ...row.messages],
    tools: shared.tools, metadata: row.metadata };
}

function rowsJson(rows) {
  // A JSON array with one corpus row per line, so it still reads like the JSONL it came from.
  return "[\n" + rows.map((row) => JSON.stringify(row)).join(",\n") + "\n]\n";
}

function tracesMarkdown(data, v, items, heading) {
  // Gold traces as a Markdown document, grouped by family and question type.
  // items: [{type, outcome, row_id, state?}]; the system prompts and tools they use close it.
  const turn = (t) => `**${t.label}**${t.masked ? " _(in context, masked)_" : ""}\n\n` +
    (t.code ? "```" + (t.label === "Tool call" ? "text" : "json") + `\n${t.text}\n` + "```" : t.text);
  const byType = {};
  for (const item of items) (byType[item.type] = byType[item.type] || []).push(item);
  const lines = [`# ${heading}`, "",
    `${items.length} gold trace${items.length === 1 ? "" : "s"} from \`${v.corpus_root}\`, one training row each, ` +
    "without the system prompt and tool list that the rows share (both are at the end).", ""];
  let family = null;
  for (const type of Object.keys(byType)) {
    if (familyOf(type) !== family) {
      family = familyOf(type);
      lines.push(`## ${family} ${(data.families[family] || {}).title || ""}`.trim(), "");
    }
    lines.push(`### ${type} ${v.names[type] || ""}`.trim(), "");
    if (data.type_text[type]) lines.push(data.type_text[type], "");
    for (const item of byType[type]) {
      lines.push(`#### ${item.outcome}${item.state ? ` · ${item.state}` : ""}`, "", `Row \`${item.row_id}\``, "");
      lines.push(blocks(v.rows[item.row_id].messages).map(turn).join("\n\n"), "");
    }
  }
  for (const index of [...new Set(items.map((item) => v.rows[item.row_id].prompt))].sort((x, y) => x - y)) {
    const p = v.prompts[index], scope = sharedLabel(v, p).replace(" · ", ": ");
    lines.push(`## System prompt${scope}`, "", "```text", p.system_prompt || "", "```", "",
      `## Tools${scope}`, "", "```text", toolText(p.tools), "```", "");
  }
  return lines.join("\n");
}

// In a claude.ai artifact, files are saved through its downloads capability;
// opened from disk or a local server, through an ordinary download link.
const hosted = typeof window !== "undefined" && Boolean(window.claude && window.claude.use);
let downloadsApi = null;

async function save(filename, text, status) {
  if (hosted) {
    try {
      await downloadsApi.save({ filename, data: text });
      status.textContent = "";
    } catch (error) {
      status.textContent = error && error.code === "declined" ? "" : `${filename} was not saved: ${(error && (error.message || error.code)) || "unknown error"}.`;
    }
    return;
  }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: filename.endsWith(".md") ? "text/markdown" : "application/json" }));
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
