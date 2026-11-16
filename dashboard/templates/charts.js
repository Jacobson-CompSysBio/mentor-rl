// Bar charts and a histogram for the stage page, built as HTML so they fit any width.
//
// Rates run from 0 to 100% on one axis. Bars are at most 24px thick, with a
// rounded data end and a square base; gridlines are hairlines; a dashed line
// marks only a gate threshold. Each bar's whole column or row carries its
// numbers in data-tip, which page.js shows on hover and keyboard focus, and
// each chart has a table view beside it.

const AXIS = [0, 0.25, 0.5, 0.75, 1];

function columnChart({ items, threshold, label }) {
  // One column per item, in the given order. item: {label, value (0-1) or null, tip}.
  const grid = AXIS.map((t) => `<div class="grid-line" style="bottom:${t * 100}%"><span>${t * 100}%</span></div>`).join("");
  const line = threshold
    ? `<div class="threshold" style="bottom:${threshold.value * 100}%"><span>${esc(threshold.label)}</span></div>` : "";
  const cols = items.map((item) => {
    const empty = item.value === null || item.value === undefined;
    const marks = empty ? '<span class="col-empty">no data</span>'
      : `<span class="col-value" style="bottom:calc(${item.value * 100}% + 4px)">${rate(item.value)}</span>` +
        `<span class="col-bar" style="height:${item.value * 100}%"></span>`;
    return `<div class="col" tabindex="0" data-tip="${esc(item.tip)}">${marks}</div>`;
  }).join("");
  const labels = items.map((item) => `<div class="col-label">${esc(item.label)}</div>`).join("");
  return `<figure class="chart columns" aria-label="${esc(label)}"><div class="plot">${grid}${line}` +
    `<div class="cols">${cols}</div></div><div class="col-labels">${labels}</div></figure>`;
}

function barRows({ groups, threshold, series, label }) {
  // Horizontal bars, one row per item, grouped under headings. A row's
  // compare value, when there is one, is a dot on the same track.
  // group: {title, note, rows: [{label, sublabel, value, compare, tip}]}.
  const axis = '<div class="bar-row axis"><div class="bar-label"></div><div class="track">' +
    AXIS.map((t) => `<span style="left:${t * 100}%">${t * 100}%</span>`).join("") + "</div><div></div></div>";
  const legend = series.compare || threshold
    ? '<div class="legend">' +
      `<span><i class="key bar"></i>${esc(series.value)}</span>` +
      (series.compare ? `<span><i class="key dot"></i>${esc(series.compare)}</span>` : "") +
      (threshold ? `<span><i class="key thr"></i>${esc(threshold.label)}</span>` : "") + "</div>"
    : "";
  const row = (item) => {
    const fill = item.value === null || item.value === undefined ? ""
      : `<b class="fill" style="width:${item.value * 100}%"></b>`;
    const dot = item.compare === null || item.compare === undefined ? ""
      : `<i class="dot" style="left:${item.compare * 100}%"></i>`;
    const thr = threshold ? `<i class="thr" style="left:${threshold.value * 100}%"></i>` : "";
    return `<div class="bar-row" tabindex="0" data-tip="${esc(item.tip)}">` +
      `<div class="bar-label"><span class="mono">${esc(item.label)}</span> <span class="muted">${esc(item.sublabel || "")}</span></div>` +
      `<div class="track">${thr}${fill}${dot}</div>` +
      `<div class="bar-value">${item.value === null || item.value === undefined ? "—" : rate(item.value)}</div></div>`;
  };
  return `<figure class="chart bars" aria-label="${esc(label)}">${legend}${axis}` +
    groups.map((group) => `<div class="bar-group"><div class="bar-group-title">${esc(group.title)}` +
      (group.note ? ` <span class="muted">${esc(group.note)}</span>` : "") + "</div>" +
      group.rows.map(row).join("") + "</div>").join("") + "</figure>";
}

function binRates(items, width) {
  // Equal bins from 0 to 100%. A bin holds its lower edge, and the last one
  // also holds 100%. item: {name, value (0-1)}; each bin lists its items in order.
  const n = Math.round(1 / width);
  const bins = Array.from({ length: n }, (_, i) => ({ lo: i / n, hi: (i + 1) / n, items: [] }));
  for (const item of items) bins[Math.min(n - 1, Math.max(0, Math.floor(item.value * n + 1e-9)))].items.push(item);
  return bins;
}

function countTicks(top) {
  // Clean count ticks from 0, at most five, reaching at least top.
  for (let scale = 1; ; scale *= 10) {
    const step = [1, 2, 5].map((m) => m * scale).find((s) => Math.ceil(top / s) <= 4);
    if (step) return Array.from({ length: Math.ceil(top / step) + 1 }, (_, i) => i * step);
  }
}

function histogram({ bins, threshold, series, label }) {
  // How many items fall in each rate bin, over the same 0-100% axis as the
  // bars. Columns touch but for a 2px gap; a compare count, when there is
  // one, is a 2px step line on the same axes. Hovering a bin's whole column
  // shows its numbers. bin: {lo, hi, count, compare, tip}.
  const ticks = countTicks(Math.max(1, ...bins.map((b) => Math.max(b.count, b.compare || 0))));
  const top = ticks[ticks.length - 1];
  const y = (count) => (100 * count) / top;
  const grid = ticks.map((t) => `<div class="grid-line" style="bottom:${y(t)}%"><span>${number(t)}</span></div>`).join("");
  const line = threshold
    ? `<div class="vthreshold" style="left:${threshold.value * 100}%"><span>${esc(threshold.label)}</span></div>` : "";
  const cols = bins.map((b) => `<div class="hist-bin" style="left:${b.lo * 100}%;width:${(b.hi - b.lo) * 100}%"` +
    `${b.count || b.compare ? ' tabindex="0"' : ""} data-tip="${esc(b.tip)}">` +
    (b.count ? `<span class="hist-bar" style="height:${y(b.count)}%"></span>` : "") + "</div>").join("");
  // The step line runs along the baseline where a bin is empty.
  let step = "";
  if (series.compare) {
    const points = ["0,100"];
    for (const b of bins) points.push(`${b.lo * 100},${100 - y(b.compare || 0)}`, `${b.hi * 100},${100 - y(b.compare || 0)}`);
    points.push("100,100");
    step = '<svg class="hist-step" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' +
      `<polyline points="${points.join(" ")}" vector-effect="non-scaling-stroke"></polyline></svg>`;
  }
  const legend = series.compare || threshold
    ? '<div class="legend">' +
      `<span><i class="key hist"></i>${esc(series.value)}</span>` +
      (series.compare ? `<span><i class="key step"></i>${esc(series.compare)}</span>` : "") +
      (threshold ? `<span><i class="key thr"></i>${esc(threshold.label)}</span>` : "") + "</div>"
    : "";
  const xs = AXIS.map((t) => `<span style="left:${t * 100}%">${t * 100}%</span>`).join("");
  return `<figure class="chart hist" aria-label="${esc(label)}">${legend}` +
    `<div class="axis-title">${esc(series.unit)}</div>` +
    `<div class="plot">${grid}${line}<div class="hist-bins">${cols}</div>${step}</div>` +
    `<div class="hist-x">${xs}</div></figure>`;
}
