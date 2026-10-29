// Bar charts for the stage page, built as HTML so they fit any width.
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
