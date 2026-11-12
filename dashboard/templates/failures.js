// Failures tab: where one version's held-out answers go wrong (S2).
// dashboard/failures.py collects the data and render.py embeds it as
// data.failures; this draws it with compare.js's trace helpers. A case's
// failed rows are drawn only when the case is opened.

const FIELD_NAMES = ["status", "result", "decision", "provenance", "blocker", "tool calls"];
const LOW_RATE = 0.9; // rates below this are marked

function share(count, total) {
  return total ? count / total : 0;
}

function rateCell(count, total) {
  const value = share(count, total);
  return `<td class="num${value < LOW_RATE ? " bad" : ""}" data-tip="${number(count)} of ${number(total)} rows">${rate(value)}</td>`;
}

function failedCases(data, state) {
  // Cases with a failed row that match the chosen family and category, in the chosen order.
  const f = data.failures;
  const names = Object.keys(f.cases).filter((name) => {
    const c = f.cases[name];
    return c.correct < c.rows && (state.family === "all" || familyOf(name) === state.family) &&
      (state.category === "all" || c.categories[state.category]);
  }).sort(typeOrder);
  if (state.order === "catalog") return names;
  const key = (name) => {
    const c = f.cases[name];
    return state.category === "all" ? c.correct / c.rows : -c.categories[state.category] / c.rows;
  };
  return names.map((name, i) => [name, key(name), i]).sort((x, y) => x[1] - y[1] || x[2] - y[2]).map(([name]) => name);
}

function diffsView(row) {
  // Where the predicted result and provenance differ from the key, one JSON path per line.
  if (!row.diffs.length) return "";
  const value = (v) => (v === null || v === undefined ? '<span class="muted">—</span>' : `<code>${esc(pretty(v))}</code>`);
  const items = (list) => (list.length ? `; only here: ${list.map(value).join(", ")}` : "");
  return '<div class="table-wrap"><table class="diffs"><thead><tr><th>Path</th><th>Difference</th><th>Key</th><th>Model</th></tr></thead><tbody>' +
    row.diffs.map((d) => `<tr><td class="mono">${esc(d.path)}</td><td>${esc(d.kind)}</td>` +
      (d.kind === "length"
        ? `<td>${d.expected} items${items(d.only_expected)}</td><td>${d.got} items${items(d.only_got)}</td>`
        : `<td>${value(d.expected)}</td><td>${value(d.got)}</td>`) + "</tr>").join("") + "</tbody></table></div>" +
    (row.diff_count > row.diffs.length ? `<p class="muted small">${row.diff_count - row.diffs.length} more differences are not shown.</p>` : "");
}

function expectedView(row) {
  // The answer key: the ordered tool calls and the scored final fields.
  const e = row.expected;
  const calls = e.tool_calls.length ? e.tool_calls.map((c) => `${c.name}(${pretty(c.arguments)})`).join("\n") : "no tool call";
  const answer = { status: e.status, provenance: e.provenance, result: e.result, continuation_decision: e.continuation_decision,
    blocked_reason: e.blocked_reason_required ? "required, any wording" : null };
  return singleView([{ label: "Tool calls", code: true, right: esc(calls) },
    { label: "Answer", code: true, right: esc(pretty(answer)) }], "answer key");
}

function failedRowView(row, open) {
  return `<details class="trace"${open ? " open" : ""}><summary><span class="state">${esc(row.category)}</span> ` +
    `<span class="fields">wrong: ${esc(row.failed.join(", "))}</span> <span class="mono small muted">${esc(row.row_id)}</span></summary>` +
    (row.error ? `<p class="small">Harness stop: <code>${esc(row.error)}</code></p>` : "") + diffsView(row) +
    singleView(turnRows(null, row.messages, false), "what the model saw and did") + expectedView(row) + "</details>";
}

function caseBody(data, name) {
  // One case's field counts, failure categories, common differences, and example rows.
  const c = data.failures.cases[name];
  const failed = c.rows - c.correct;
  const fields = FIELD_NAMES.map((field) => `${esc(field)} ${c.fields[field]}`).join(" · ");
  const categories = Object.entries(c.categories).map(([label, n]) => `${esc(label)}: ${n}`).join("<br>");
  const paths = c.paths.map(([path, n]) => `<code>${esc(path)}</code> in ${n} row${n === 1 ? "" : "s"}`).join("<br>");
  return (data.type_text[name] ? `<p class="type-text">${inline(data.type_text[name])}</p>` : "") +
    `<p class="meta small">Correct of ${c.rows}: ${fields}</p>` +
    `<p class="small"><b>Failure categories</b><br>${categories}</p>` +
    (paths ? `<p class="small"><b>Most common differences in the result and provenance</b><br>${paths}</p>` : "") +
    `<p class="small muted">${c.examples.length} of ${failed} failed rows shown, one per category in turn. ` +
    `Every failed row is in <code>site_data/${esc(data.stage)}/failures/${esc(data.failures.version)}.rows.jsonl</code>.</p>` +
    c.examples.map((row, i) => failedRowView(row, i === 0)).join("");
}

function failuresView(data, state) {
  const f = data.failures;
  const names = (data.versions.find((v) => v.version === f.version) || {}).names || {};
  const families = Object.keys(f.family_fields).sort(typeOrder);
  const categories = Object.keys(f.categories);
  let html = `<h2>Held-out failures in ${esc(f.version)} <span class="count">${number(f.rows - f.correct)} of ${number(f.rows)} rows</span></h2>` +
    `<p class="muted small">Gate run <code>${esc(f.run)}</code> (${esc(f.profile)}). A row is correct only when all six structured fields ` +
    "match the key. The summary and the blocked-reason wording are never scored; only whether a blocked reason is present. " +
    "These are test-panel answers: do not draft training wording from them.</p>";

  html += '<div class="tiles"><div class="tile"><div class="tile-label">Structured accuracy</div>' +
    `<div class="tile-value">${rate(share(f.correct, f.rows))}</div><div class="tile-note">${number(f.correct)} of ${number(f.rows)} rows</div></div>` +
    FIELD_NAMES.map((field) => `<div class="tile"><div class="tile-label">${esc(field)}</div>` +
      `<div class="tile-value">${rate(share(f.fields[field], f.rows))}</div><div class="tile-note">${number(f.rows - f.fields[field])} rows wrong</div></div>`).join("") + "</div>";

  html += '<h3>Field accuracy by family</h3><div class="table-wrap"><table><thead><tr><th>Family</th><th>Name</th>' +
    '<th class="num">Rows</th><th class="num">Structured</th>' + FIELD_NAMES.map((field) => `<th class="num">${esc(field)}</th>`).join("") +
    "</tr></thead><tbody>" + families.map((family) => {
      const r = f.family_fields[family];
      return `<tr><td class="mono">${esc(family)}</td><td>${esc((data.families[family] || {}).title || "")}</td>` +
        `<td class="num">${number(r.rows)}</td>${rateCell(r.correct, r.rows)}${FIELD_NAMES.map((field) => rateCell(r[field], r.rows)).join("")}</tr>`;
    }).join("") + "</tbody></table></div>" +
    `<p class="muted small">Rates under ${rate(LOW_RATE)} are marked; hover a rate for its counts. An invalid trajectory fails every field of its row.</p>`;

  html += '<p class="controls">' +
    `<label>Family ${select("family", [["all", "all families"], ...families.map((x) => [x, x])], state.family)}</label> ` +
    `<label>Category ${select("category", [["all", "all categories"], ...categories.map((x) => [x, x])], state.category)}</label> ` +
    `<label>Order ${select("order", [["worst", state.category === "all" ? "worst accuracy first" : "most rows in the category first"], ["catalog", "catalog order"]], state.order)}</label></p>`;

  const counts = state.family === "all" ? f.categories : f.family_categories[state.family] || {};
  const total = Object.values(counts).reduce((sum, n) => sum + n, 0);
  html += `<h3>Failure categories${state.family === "all" ? "" : ` in ${esc(state.family)}`} <span class="count">${number(total)} rows</span></h3>` +
    '<div class="table-wrap"><table><thead><tr><th>Category</th><th class="num">Rows</th><th class="num">Of the failures</th></tr></thead><tbody>' +
    Object.entries(counts).map(([label, n]) => `<tr${label === state.category ? ' class="changed"' : ""}><td>${esc(label)}</td>` +
      `<td class="num">${number(n)}</td><td class="num">${rate(share(n, total))}</td></tr>`).join("") + "</tbody></table></div>" +
    '<p class="muted small">"Invalid" rows stopped before a final answer: the harness serves a tool result only for the expected call, ' +
    'so a wrong call ends the row. "Final answer" rows made every expected call and name the fields they got wrong.</p>';

  const shown = failedCases(data, state);
  html += `<h3>Cases <span class="count">${shown.length} with failures</span></h3><div class="type-list">` + shown.map((name) => {
    const c = f.cases[name];
    const top = Object.entries(c.categories)[0];
    return `<details class="qtype case" id="${anchor("failures", name)}" data-case="${esc(name)}"><summary>` +
      `<span class="mono">${esc(name)}</span> <span class="name">${esc(names[name] || "")}</span> · ` +
      `<span class="acc">${c.correct} of ${c.rows} correct (${rate(share(c.correct, c.rows))})</span>` +
      (top ? ` <span class="fields">· ${esc(top[0])}: ${top[1]}</span>` : "") +
      '</summary><div class="case-body"></div></details>';
  }).join("") + "</div>";
  return html;
}

function wireFailures(data, panel) {
  // Redraw on a filter change, and draw a case's rows when it is first opened.
  const state = { family: "all", category: "all", order: "worst" };
  panel.innerHTML = failuresView(data, state);
  panel.addEventListener("change", (event) => {
    if (!(event.target.name in state)) return;
    state[event.target.name] = event.target.value;
    panel.innerHTML = failuresView(data, state);
  });
  // toggle does not bubble, so listen in the capture phase.
  panel.addEventListener("toggle", (event) => {
    const body = event.target.querySelector && event.target.querySelector(":scope > .case-body");
    if (event.target.open && body && !body.innerHTML) body.innerHTML = caseBody(data, event.target.dataset.case);
  }, true);
}
