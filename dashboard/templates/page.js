// Tabs and the Versions, Question types, Tools, and Data sources panels.
// Uses compare.js and charts.js, which render.py places before this file.

// Rates the dashboard shows when a gate report has them (dashboard/gates.py HEADLINE).
const METRICS = {
  structured_answer_accuracy: "Structured answer accuracy", status_exact: "status exact", result_exact: "result exact",
  continuation_decision_exact: "decision exact", exact_tool_argument: "tool argument exact", structured_json_valid: "valid JSON",
  // S2 reports.
  structured_accuracy: "Structured accuracy", status_accuracy: "status", result_accuracy: "result",
  continuation_decision_accuracy: "decision", provenance_accuracy: "provenance",
  blocked_reason_presence_accuracy: "blocker presence", exact_tool_calls: "exact tool calls", valid_trajectory: "valid trajectory",
};
const NORMALIZES = {
  trim_spaces: "trims spaces", ignore_case: "ignores case", strip_numeric_version_suffix: "strips a numeric `.N` version suffix",
};

function select(name, options, chosen) {
  return `<select name="${name}">` + options.map(([value, text]) =>
    `<option value="${esc(value)}"${value === chosen ? " selected" : ""}>${esc(text)}</option>`).join("") + "</select>";
}

function verdict(gate) {
  return gate.passed === true ? "passed" : gate.passed === false ? "failed" : "no verdict";
}

// ---- Versions tab ----

function changelogView(data, shown) {
  // Every version, newest first, with its stated change, description, and gate results.
  const rows = data.versions.slice().reverse().map((v) => {
    const gate = v.gate
      ? `<span class="${v.gate.passed ? "ok" : "bad"}">${verdict(v.gate)}</span> <span class="mono small muted">${esc(v.gate.run)}</span>` +
        (v.vs_previous && v.vs_previous.reasons.length
          ? `<br><span class="small muted">vs ${esc(v.vs_previous.version)}: ${esc(v.vs_previous.reasons.join("; "))}</span>` : "")
      : '<span class="muted">no gate run</span>';
    const summary = v.summary ? inline(v.summary) : '<span class="muted">no summary yet</span>';
    const change = v.description ? `<details><summary>${summary}</summary><p>${inline(v.description)}</p></details>` : summary;
    const rates = shown.map((name) => `<td class="num">${v.gate && name in v.gate.metrics ? rate(v.gate.metrics[name]) : "—"}</td>`).join("");
    return `<tr><td><b>${esc(v.version)}</b>${v.version === data.current ? ' <span class="small muted">current</span>' : ""}</td>` +
      `<td>${change}</td><td>${v.gold.length ? `${v.gold.length} gold traces` : '<span class="muted">not on disk</span>'}</td>` +
      `<td>${gate}</td>${rates}</tr>`;
  }).join("");
  return `<h2>Versions <span class="count">${data.versions.length}</span></h2>` +
    '<p class="muted small">Open a change to read the longer description from <code>dashboard/descriptions</code>.</p>' +
    '<div class="table-wrap"><table class="changelog"><thead><tr><th>Version</th><th>Change</th><th>Corpus</th><th>Gate</th>' +
    shown.map((name) => `<th class="num">${METRICS[name]}</th>`).join("") + `</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function versionsView(data, metric) {
  // A column per version for one gate rate, with the gate's pass mark, then the changelog.
  const metrics = Object.keys(METRICS).filter((name) => data.versions.some((v) => v.gate && name in v.gate.metrics));
  let html = "";
  const chosen = metrics.includes(metric) ? metric : metrics[0];
  if (chosen) {
    // The pass mark of the newest version whose gate sets one for this rate.
    const marked = data.versions.slice().reverse().find((v) => v.gate && chosen in v.gate.thresholds);
    const threshold = marked && {
      value: marked.gate.thresholds[chosen],
      label: `${marked.gate.profile || `${marked.version} gate`} ${rate(marked.gate.thresholds[chosen])}`,
    };
    const items = data.versions.map((v) => {
      const value = v.gate && chosen in v.gate.metrics ? v.gate.metrics[chosen] : null;
      const tip = value === null ? `${v.version}: ${v.gate ? "this report has no such rate" : "no gate run"}`
        : `${v.version}: ${rate(value)}\n${verdict(v.gate)} · ${v.gate.run}\n${v.gate.profile || "no profile"} · history ${v.gate.history_policy}`;
      return { label: v.version, value, tip };
    });
    html += `<h2>${METRICS[chosen]} by version</h2>` +
      (metrics.length > 1 ? `<p class="controls"><label>Rate ${select("metric", metrics.map((m) => [m, METRICS[m]]), chosen)}</label></p>` : "") +
      columnChart({ items, threshold, label: `${METRICS[chosen]} by version` }) +
      '<p class="muted small">Each column is the version\'s pinned gate run. Runs do not all compare directly; the table notes where they differ.</p>';
  }
  return html + changelogView(data, metrics);
}

// ---- Question types tab ----

function typeOrder(x, y) {
  return x.localeCompare(y, "en", { numeric: true });
}

const BIN_WIDTHS = [["0.01", "1%"], ["0.02", "2%"], ["0.05", "5%"], ["0.1", "10%"]];
const types = (n) => `${number(n)} type${n === 1 ? "" : "s"}`;

function distributionView(v, compareTo, valueOf, metric, threshold, width) {
  // A histogram of one version's per-type rates from 0 to 100%, with another
  // version's as a step line, a summary line per version, and a table view.
  const scored = (x) => Object.keys(x.gate.groups.rows).sort(typeOrder)
    .map((name) => ({ name, value: valueOf(x, name) })).filter((item) => item.value !== null);
  const items = scored(v), others = compareTo ? scored(compareTo) : null;
  const mine = binRates(items, width), theirs = others && binRates(others, width);
  const pct = (x) => `${Math.round(x * 100)}%`;
  const range = (b, i) => (i === mine.length - 1 ? `${pct(b.lo)} to 100%` : `${pct(b.lo)} to under ${pct(b.hi)}`);
  const bins = mine.map((b, i) => {
    const compare = theirs ? theirs[i].items.length : null;
    const listed = b.items.slice(0, 8).map((item) => `${item.name} ${rate(item.value)}`);
    const tip = `${range(b, i)}\n${v.version}: ${types(b.items.length)}` +
      (theirs ? `\n${compareTo.version}: ${types(compare)}` : "") +
      (listed.length ? `\n\n${listed.join("\n")}${b.items.length > 8 ? `\n… and ${b.items.length - 8} more` : ""}` : "");
    return { lo: b.lo, hi: b.hi, count: b.items.length, compare, tip };
  });
  const spread = (x, list) => {
    const values = list.map((item) => item.value).sort((a, b) => a - b);
    if (!values.length) return `<b>${esc(x.version)}</b>: no type has this rate`;
    const half = Math.floor(values.length / 2);
    const median = values.length % 2 ? values[half] : (values[half - 1] + values[half]) / 2;
    const unscored = Object.keys(x.gate.groups.rows).length - values.length;
    return `<b>${esc(x.version)}</b>: ${types(values.length)} · median ${rate(median)} · lowest ${rate(values[0])} · ` +
      `${number(values.filter((value) => value === 1).length)} at 100%` +
      (threshold ? ` · ${number(values.filter((value) => value < threshold.value).length)} below ${rate(threshold.value)}` : "") +
      (unscored ? ` · ${types(unscored)} without this rate` : "");
  };
  return '<h3 class="sub">Spread across question types</h3>' +
    `<p class="controls"><label>Bin width ${select("bins", BIN_WIDTHS, String(width))}</label></p>` +
    `<p class="spread">${spread(v, items)}</p>` + (compareTo ? `<p class="spread">${spread(compareTo, others)}</p>` : "") +
    histogram({ bins, threshold, label: `How ${METRICS[metric]} spreads across question types`,
      series: { value: v.version, compare: compareTo && compareTo.version, unit: "question types per bin" } }) +
    '<details class="table-view"><summary>Show as table</summary><div class="table-wrap"><table>' +
    `<thead><tr><th>${esc(METRICS[metric])}</th><th class="num">${esc(v.version)} types</th>` +
    (compareTo ? `<th class="num">${esc(compareTo.version)} types</th>` : "") + "</tr></thead><tbody>" +
    bins.map((b, i) => `<tr><td>${range(b, i)}</td><td class="num">${number(b.count)}</td>` +
      (compareTo ? `<td class="num">${number(b.compare)}</td>` : "") + "</tr>").join("") +
    "</tbody></table></div></details>";
}

function typesView(data, state) {
  // Accuracy by question type for one version, with another as dots, then every type described.
  // Per-type rates also get a histogram of how they spread.
  const gated = data.versions.filter((v) => v.gate && Object.keys(v.gate.groups.rows).length);
  let html = "";
  if (gated.length) {
    const v = gated.find((x) => x.version === state.version) || gated.find((x) => x.version === data.current) || gated[gated.length - 1];
    const earlier = gated.filter((x) => data.versions.indexOf(x) < data.versions.indexOf(v));
    const compareTo = state.compare === "none" ? null
      : gated.find((x) => x.version === state.compare && x !== v) || earlier[earlier.length - 1] || null;
    const rowsOf = (x) => x.gate.groups.rows;
    const metrics = Object.keys(METRICS).filter((m) => Object.values(rowsOf(v)).some((r) => m in r.rates));
    const metric = metrics.includes(state.metric) ? state.metric : metrics[0];
    const byType = v.gate.groups.by === "question type";
    const names = Object.keys(rowsOf(v)).sort(typeOrder);
    const families = {};
    for (const name of names) {
      const family = byType ? familyOf(name) : data.stage;
      (families[family] = families[family] || []).push(name);
    }
    const valueOf = (x, name) => (x && rowsOf(x)[name] && metric in rowsOf(x)[name].rates ? rowsOf(x)[name].rates[metric] : null);
    const groups = Object.entries(families).map(([family, members]) => ({
      title: byType ? `${family} ${(data.families[family] || {}).title || ""}` : `${data.stage} question families`,
      rows: members.map((name) => {
        const value = valueOf(v, name), compare = valueOf(compareTo, name), total = rowsOf(v)[name].total;
        const sublabel = byType ? v.names[name] || "" : (data.families[name] || {}).title || "";
        const tip = `${name} ${sublabel}\n${v.version}: ${value === null ? "—" : rate(value)}${total ? ` of ${number(total)} rows` : ""}` +
          (compareTo ? `\n${compareTo.version}: ${compare === null ? "—" : rate(compare)}` : "");
        return { label: name, sublabel, value, compare, tip };
      }),
    }));
    const threshold = metric in v.gate.thresholds
      ? { value: v.gate.thresholds[metric], label: `${v.gate.profile || "gate"} ${rate(v.gate.thresholds[metric])} (overall)` } : null;
    html += `<h2>${METRICS[metric]} by ${v.gate.groups.by}</h2>` +
      '<p class="controls">' +
      `<label>Version ${select("version", gated.map((x) => [x.version, x.version]), v.version)}</label> ` +
      `<label>Compare with ${select("compare", [["none", "none"], ...gated.filter((x) => x !== v).map((x) => [x.version, x.version])], compareTo ? compareTo.version : "none")}</label> ` +
      (metrics.length > 1 ? `<label>Rate ${select("metric", metrics.map((m) => [m, METRICS[m]]), metric)}</label>` : "") + "</p>" +
      (byType ? distributionView(v, compareTo, valueOf, metric, threshold,
        BIN_WIDTHS.some(([w]) => w === state.bins) ? Number(state.bins) : 0.05) +
        '<h3 class="sub">Each question type</h3>' : "") +
      barRows({ groups, threshold, series: { value: v.version, compare: compareTo && compareTo.version },
        label: `${METRICS[metric]} by ${v.gate.groups.by}` }) +
      (threshold ? `<p class="muted small">The gate applies its pass mark to the overall rate; per-${byType ? "type" : "family"} rates are diagnostics.</p>` : "") +
      '<details class="table-view"><summary>Show as table</summary><div class="table-wrap"><table>' +
      `<thead><tr><th>${byType ? "Type" : "Family"}</th><th>Name</th><th class="num">${esc(v.version)}</th>` +
      (compareTo ? `<th class="num">${esc(compareTo.version)}</th><th class="num">Change</th>` : "") + "</tr></thead><tbody>" +
      groups.flatMap((g) => g.rows).map((r) => `<tr><td class="mono">${esc(r.label)}</td><td>${esc(r.sublabel)}</td>` +
        `<td class="num">${r.value === null ? "—" : rate(r.value)}</td>` +
        (compareTo ? `<td class="num">${r.compare === null ? "—" : rate(r.compare)}</td>` +
          `<td class="num">${r.value === null || r.compare === null ? "—" : `${r.value >= r.compare ? "+" : ""}${(100 * (r.value - r.compare)).toFixed(2)}`}</td>` : "") +
        "</tr>").join("") + "</tbody></table></div></details>";
  } else {
    html += '<p class="muted">No version of this stage has a gate run yet, so there are no accuracies to chart.</p>';
  }
  return html + catalogView(data);
}

function catalogView(data) {
  // Every question family and type, with its description, rows, and gold outcomes in one version.
  const v = data.versions.find((x) => x.version === data.current && x.gold.length) ||
    data.versions.slice().reverse().find((x) => x.gold.length);
  const outcomes = {};
  if (v) for (const g of v.gold) (outcomes[g.type] = outcomes[g.type] || []).push(g.outcome);
  const types = [...new Set([...Object.keys(data.type_text), ...Object.keys(outcomes),
    ...Object.keys((v && v.splits && v.splits.types) || {})])].sort(typeOrder);
  const families = Object.keys(data.families).filter((f) => /^S\d/.test(f));
  for (const type of types) if (!families.includes(familyOf(type))) families.push(familyOf(type));
  families.sort(typeOrder);
  const split = (type) => (v && v.splits && v.splits.types[type]) || null;
  return `<h2>Question types${v ? ` <span class="count">in ${esc(v.version)}</span>` : ""}</h2>` +
    '<p class="muted small">Descriptions come from <code>dashboard/descriptions</code>; rows and outcomes come from the data.</p>' +
    (v ? `<p class="downloads">Download ${esc(v.version)} ${downloadButtons(v, "gold", v.gold.length)}</p>` : "") +
    families.map((family) => {
      const info = data.families[family] || {};
      const members = types.filter((t) => familyOf(t) === family);
      return `<section class="family"><h3>${esc(family)} ${esc(info.title || "")}</h3>` +
        (info.description ? `<p class="type-text">${inline(info.description)}</p>` : "") +
        (members.length ? members.map((type) => {
          const counts = split(type);
          const rows = counts ? SPLITS.map((s) => `${s} ${counts[s] === undefined ? "—" : number(counts[s])}`).join(" · ") : "";
          return `<div class="catalog-type"><div><span class="mono">${esc(type)}</span> <span class="name">${esc((v && v.names[type]) || "")}</span></div>` +
            (data.type_text[type] ? `<p class="type-text">${inline(data.type_text[type])}</p>` : "") +
            (rows || outcomes[type] ? `<p class="meta small">${rows}${rows && outcomes[type] ? "<br>" : ""}` +
              (outcomes[type] ? `gold outcomes: ${esc(outcomes[type].join("; "))}` : "") + "</p>" : "") + "</div>";
        }).join("") : '<p class="muted small">No rows yet.</p>') + "</section>";
    }).join("");
}

// ---- Tools tab ----

function toolsView(data) {
  // The tools offered to the model, written up like a package's API reference.
  const t = data.tools;
  if (!t.tools.length && !t.other.length) return '<p class="muted">No tool definitions found for this stage.</p>';
  const params = (list) => (list.length
    ? '<table class="params"><thead><tr><th>Parameter</th><th>Type</th><th></th><th>Notes</th></tr></thead><tbody>' +
      list.map((p) => `<tr><td><code>${esc(p.name)}</code></td><td>${esc(p.type)}</td>` +
        `<td class="small muted">${p.required ? "required" : "optional"}</td><td>${p.notes.map(inline).join("; ")}</td></tr>`).join("") +
      "</tbody></table>" : '<p class="muted small">No parameters.</p>');
  const returns = (map) => (Object.keys(map).length
    ? '<table class="params"><thead><tr><th>Status</th><th>Result fields</th></tr></thead><tbody>' +
      Object.entries(map).map(([status, fields]) => `<tr><td><code>${esc(status)}</code></td>` +
        `<td>${(Array.isArray(fields) ? fields : Object.keys(fields)).map((f) => `<code>${esc(f)}</code>`).join(", ")}</td></tr>`).join("") +
      "</tbody></table>" : "");
  const signature = (tool) => `${tool.name}(${tool.parameters.map((p) => `${p.name}: ${p.type}${p.required ? "" : " = None"}`).join(", ")})`;
  const doc = (tool) => {
    const text = t.text[tool.name] || tool.description;
    const facts = [];
    if (tool.normalizes.length) facts.push(`Input handling: ${tool.normalizes.map((n) => inline(NORMALIZES[n] || n.replace(/_/g, " "))).join(", ")}.`);
    if (tool.canonical_format) facts.push(`Canonical format: <code>${esc(tool.canonical_format)}</code>.`);
    if (tool.candidate_details) facts.push(`Each candidate's details: ${Object.entries(tool.candidate_details).map(([k, kind]) => `<code>${esc(k)}</code> (${esc(kind)})`).join(", ")}.`);
    if (tool.provenance_channel) facts.push(`Provenance channel: <code>${esc(tool.provenance_channel)}</code>.`);
    return `<section class="tooldoc" id="tool-${slug(tool.name)}"><h3><code>${esc(signature(tool))}</code></h3>` +
      (tool.families ? `<p class="small muted">Offered in ${esc(tool.families.join(", "))}.</p>` : "") +
      (text ? `<p>${inline(text)}</p>` : "") + "<h4>Parameters</h4>" + params(tool.parameters) +
      (Object.keys(tool.returns).length ? `<h4>Returns</h4>${returns(tool.returns)}` : "") +
      (facts.length ? `<p class="small">${facts.join("<br>")}</p>` : "") + "</section>";
  };
  return `<h2>Tools <span class="count">${t.tools.length}</span></h2>` +
    `<p class="muted small">The tool definitions in ${esc(t.version || "the current version")}'s rows, with result fields from the stage contract.` +
    (t.differs_in.length ? ` They differ in ${esc(t.differs_in.join(", "))}; the Compare tab shows how.` : " Every version on disk offers the same definitions.") + "</p>" +
    (t.text._all ? `<p>${inline(t.text._all)}</p>` : "") +
    '<nav class="chips">' + t.tools.map((tool) => `<a href="#tool-${slug(tool.name)}">${esc(tool.name)}</a>`).join("") + "</nav>" +
    t.tools.map(doc).join("") +
    (t.error_result ? '<section class="tooldoc"><h3>Errors</h3><p>Any tool can fail with a transport error. The result then has ' +
      `${(t.error_result.keys || []).map((k) => `<code>${esc(k)}</code>`).join(", ")}, with <code>status</code> <code>${esc(t.error_result.status)}</code> and <code>provenance</code> null.</p></section>` : "") +
    (t.other.length ? "<h2>Defined but not offered</h2>" + t.other.map((tool) =>
      `<section class="tooldoc"><h3><code>${esc(signature(tool))}</code></h3><p>${esc(tool.note)}</p>` +
      (Object.keys(tool.returns).length ? `<h4>Returns</h4>${returns(tool.returns)}` : "") + "</section>").join("") : "");
}

// ---- Raw data sources tab ----

function sourcesView(data) {
  // Every file the stage contracts pin: path, SHA-256, and the facts recorded beside it.
  const s = data.sources;
  if (!s || !s.pins.length) return '<p class="muted">This stage has no contract with pinned sources.</p>';
  const label = (name) => esc(name.replace(/_/g, " ").replace(/\./g, " › "));
  const sha = (value) => (value ? `<code title="${esc(value)}">${esc(value.slice(0, 12))}…</code>` : '<span class="muted">—</span>');
  const facts = (map) => Object.entries(map).map(([k, value]) => `${esc(k.replace(/_/g, " "))}: ${esc(typeof value === "number" ? number(value) : value)}`).join("<br>");
  return `<h2>Raw data sources <span class="count">${s.pins.length}</span></h2>` +
    `<p class="muted small">Pinned in ${s.files.map((file) => `<code>${esc(file)}</code>`).join(", ")}. Hover a hash to see all of it.</p>` +
    '<div class="table-wrap"><table class="sources"><thead><tr><th>Source</th><th>Path</th><th>SHA-256</th><th>Details</th></tr></thead><tbody>' +
    s.pins.map((pin) => `<tr><td>${label(pin.name)}</td><td><code>${esc(pin.path)}</code></td><td>${sha(pin.sha256)}</td>` +
      `<td class="small">${facts(pin.facts)}</td></tr>`).join("") + "</tbody></table></div>";
}

// ---- Wiring ----

function setHash(hash) {
  // Put the current view in the address bar so it can be linked. A page opened
  // from disk (file://) may refuse with a SecurityError; the view changes anyway.
  try {
    history.replaceState(null, "", hash);
  } catch (error) {
    // Nothing to do: only the address bar stays as it was.
  }
}

function wireTooltips() {
  // One tooltip for every [data-tip] element, on hover and on keyboard focus.
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  document.body.appendChild(tip);
  const place = (el, x, y) => {
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(x + 14, window.innerWidth - box.width - 8)}px`;
    tip.style.top = `${Math.min(y + 14, window.innerHeight - box.height - 8)}px`;
  };
  document.addEventListener("pointermove", (event) => {
    const el = event.target.closest && event.target.closest("[data-tip]");
    if (el) place(el, event.clientX, event.clientY); else tip.hidden = true;
  });
  document.addEventListener("focusin", (event) => {
    const el = event.target.closest && event.target.closest("[data-tip]");
    if (el) { const box = el.getBoundingClientRect(); place(el, box.left, box.bottom); }
  });
  document.addEventListener("focusout", () => { tip.hidden = true; });
}

function start() {
  const data = JSON.parse(document.getElementById("dashboard-data").textContent);
  const versions = data.available;
  const byName = Object.fromEntries(data.versions.map((v) => [v.version, v]));
  const out = document.getElementById("pair");
  const panel = (tab) => document.getElementById(`tab-${tab}`);
  const tabs = [...document.querySelectorAll("[data-tab]")];
  // The address the Compare tab links to: the version pair it shows, once there is one.
  let compareHash = location.pathname + location.search;

  function showTab(tab) {
    for (const button of tabs) button.setAttribute("aria-selected", String(button.dataset.tab === tab));
    for (const button of tabs) panel(button.dataset.tab).hidden = button.dataset.tab !== tab;
  }
  for (const button of tabs) {
    button.addEventListener("click", () => {
      showTab(button.dataset.tab);
      setHash(button.dataset.tab === "compare" ? compareHash : `#${button.dataset.tab}`);
    });
  }

  panel("versions").innerHTML = versionsView(data);
  panel("versions").addEventListener("change", (event) => {
    if (event.target.name === "metric") panel("versions").innerHTML = versionsView(data, event.target.value);
  });
  const typesState = {};
  panel("types").innerHTML = typesView(data, typesState);
  panel("types").addEventListener("change", (event) => {
    typesState[event.target.name] = event.target.value;
    panel("types").innerHTML = typesView(data, typesState);
  });
  panel("tools").innerHTML = toolsView(data);
  panel("sources").innerHTML = sourcesView(data);
  if (data.failures) wireFailures(data, panel("failures"));
  wireTooltips();

  // Download buttons name their version and format, and for changed traces their pair and side.
  // In a claude.ai artifact they appear only once the downloads capability answers.
  const status = document.getElementById("save-status");
  if (hosted) {
    document.body.classList.add("saves-pending");
    window.claude.use("downloads").catch(() => null).then((api) => {
      downloadsApi = api;
      document.body.classList.remove("saves-pending");
      document.body.classList.toggle("saves-off", !api);
    });
  }
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("button[data-what]");
    if (!button) return;
    const { version, what, format, pair: key, side } = button.dataset;
    const v = byName[version];
    const items = what === "gold" ? v.gold.map((g) => ({ type: g.type, outcome: g.outcome, row_id: g.row_id }))
      : data.pairs[key].traces.filter((t) => t.state !== "identical" && t[side])
        .map((t) => ({ type: t.type, outcome: t.outcome, row_id: t[side], state: t.state }));
    const name = `${data.stage}_${key ? `${slug(key)}.` : ""}${version}.${what}`;
    const heading = what === "gold" ? `${data.stage} ${version} gold traces`
      : `${data.stage} ${data.pairs[key].old} → ${data.pairs[key].new}: changed gold traces in ${version}`;
    if (format === "md") save(`${name}.md`, tracesMarkdown(data, v, items, heading), status);
    else save(`${name}.json`, rowsJson(items.map((item) => fullRow(v, item.row_id))), status);
  });

  // "expand all" and "collapse all" open or close the traces of one question type, or of the whole list.
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest("button[data-open]");
    if (!button) return;
    const scope = button.closest(button.dataset.scope === "type" ? "article" : ".type-list");
    for (const details of scope.querySelectorAll("details.trace")) details.open = button.dataset.open === "1";
  });

  // Ordering by accuracy needs a version whose gate scores each question type.
  const order = document.getElementById("order-types");
  document.getElementById("order-control").hidden = !accuracyBasis(...data.versions);

  let show = null;
  if (data.default) {
    const form = document.getElementById("versions");
    const showUnchanged = document.getElementById("show-unchanged");
    let current = data.default;
    show = (old, next, scrollTo) => {
      const key = `${old}_${next}`, a = byName[old], b = byName[next], pair = data.pairs[key];
      current = [old, next];
      compareHash = `#${slug(key)}`;
      form.elements.old.value = old;
      form.elements.new.value = next;
      document.title = `${data.stage} ${old} → ${next}`;
      for (const [side, v] of [["old", a], ["new", b]]) {
        const changedRows = pair.traces.filter((t) => t.state !== "identical" && t[side]).length;
        document.getElementById(`downloads-${side}`).innerHTML = `Download ${esc(v.version)}<br>` +
          `${downloadButtons(v, "changed", changedRows, key, side)}<br>${downloadButtons(v, "gold", v.gold.length)}`;
      }
      out.innerHTML = pairView(data, a, b, pair, showUnchanged.checked, order.value);
      if (scrollTo) document.getElementById(scrollTo)?.scrollIntoView();
    };
    form.addEventListener("change", (event) => {
      let a = versions.indexOf(form.elements.old.value), b = versions.indexOf(form.elements.new.value);
      if (a >= b) {
        if (event.target.name === "old") { b = a + 1; } else { a = b - 1; }
      }
      setHash(`#${slug(`${versions[a]}_${versions[b]}`)}`);
      show(versions[a], versions[b]);
    });
    showUnchanged.addEventListener("change", () => show(...current));
    order.addEventListener("change", () => show(...current));
    show(...current);
  } else if (versions.length) {
    const only = byName[versions[versions.length - 1]];
    out.innerHTML = versionView(data, only, order.value);
    order.addEventListener("change", () => { out.innerHTML = versionView(data, only, order.value); });
  } else {
    out.innerHTML = '<p class="muted">No corpus of this stage is on disk yet.</p>';
  }

  // Links name a tab (#versions), a pair (#v9-v10), a question type in a pair or
  // version (#v9-v10--s1-3-8), or a tool (#tool-get-edge): letters, digits, and
  // hyphens only, so they survive being shared.
  function fromHash() {
    const hash = decodeURIComponent(location.hash.slice(1));
    const pair = Object.values(data.pairs).find((p) => slug(`${p.old}_${p.new}`) === hash.split("--")[0]);
    if (tabs.some((button) => button.dataset.tab === hash)) {
      showTab(hash);
    } else if (pair && show) {
      showTab("compare");
      show(pair.old, pair.new, hash.includes("--") ? hash : null);
    } else if (hash.startsWith("tool-")) {
      showTab("tools");
      document.getElementById(hash)?.scrollIntoView();
    } else if (hash.startsWith("failures--") && data.failures) {
      // A failing case: #failures--s2-1-28 opens it.
      showTab("failures");
      const item = document.getElementById(hash);
      if (item) { item.open = true; item.scrollIntoView(); }
    } else if (hash.includes("--")) {
      showTab("compare");
      document.getElementById(hash)?.scrollIntoView();
    }
  }
  window.addEventListener("hashchange", fromHash);
  showTab("compare");
  fromHash();
}

if (typeof module !== "undefined") {
  module.exports = {
    accuracyBasis, anchor, barRows, binRates, blocks, caseBody, catalogView, changelogView, columnChart, diffOps,
    distributionView, failedCases, histogram,
    failuresView, fullRow, pairView, pretty, rowsJson, sharedView, slug, sortByAccuracy, sourcesView, splitsView,
    toolText, toolsView, tracesMarkdown, turnRows, typesView, versionView, versionsView, wordDiff,
  };
} else {
  start();
}
