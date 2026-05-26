// ─────────────────────────────────────────────
// Constants & State
// ─────────────────────────────────────────────

const MODEL_PALETTE = [
  "#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6",
  "#8b5cf6","#ec4899","#14b8a6","#f97316","#84cc16",
  "#06b6d4","#a855f7",
];

const _modelColorCache = {};
let _modelColorIndex = 0;
function modelColor(name) {
  if (!_modelColorCache[name]) {
    _modelColorCache[name] = MODEL_PALETTE[_modelColorIndex % MODEL_PALETTE.length];
    _modelColorIndex++;
  }
  return _modelColorCache[name];
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const TASK_DISPLAY = {
  humaneval_plus:        "HumanEval+",
  mbpp_plus:             "MBPP+",
  bbh_cot_fewshot:       "BBH",
  gsm8k_cot_zeroshot:    "GSM8K",
  ifeval:                "IFEval",
};
function taskLabel(t) { return TASK_DISPLAY[t] || t; }

let _currentTab      = "overview";
let _sseSource       = null;
let _bannerInterval  = null;
let _lastSyncTime    = null;
let _modelFilter     = null; // null = all; Set<string> = allow-list by model_name
let _knownModels     = [];
let _pollFallbackInterval = null;
let _lastPolledTs    = null;

// ─────────────────────────────────────────────
// Plotly base layout
// ─────────────────────────────────────────────

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "#1a1d27",
  plot_bgcolor:  "#1a1d27",
  font: { color: "#e0e0e0", size: 11 },
  margin: { t: 40, l: 60, r: 20, b: 60 },
};

// ─────────────────────────────────────────────
// Core utilities
// ─────────────────────────────────────────────

function getFilters() {
  return {
    runtime:      document.getElementById("filter-runtime").value || null,
    quantization: document.getElementById("filter-quant").value || null,
    architecture: document.getElementById("filter-arch").value || null,
  };
}

async function fetchAPI(path, params = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
  );
  const qs = new URLSearchParams(clean).toString();
  const url = `/api${path}${qs ? "?" + qs : ""}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${url}`);
  return resp.json();
}

function emptyChart(divId, msg = "No data yet — run a benchmark first") {
  const el = document.getElementById(divId);
  if (!el) return;
  el.innerHTML = `<div class="empty-state">${msg}</div>`;
}

function modelLabel(row) {
  const name = row.model_name || row.config_name;
  return row.quantization && !name.toLowerCase().includes(row.quantization.toLowerCase())
    ? `${name} (${row.quantization})`
    : name;
}

function showSpinner(chartId) {
  const suffix = chartId.replace("chart-", "");
  const el = document.getElementById("spinner-" + suffix);
  if (el) el.classList.remove("hidden");
  const chart = document.getElementById(chartId);
  if (chart) Plotly.purge(chart);
}

function hideSpinner(chartId) {
  const suffix = chartId.replace("chart-", "");
  const el = document.getElementById("spinner-" + suffix);
  if (el) el.classList.add("hidden");
}

// Sort ascending — Plotly renders h-bar bottom-up, so ascending = best model at top
function sortedByValue(rows, key) {
  return [...rows].sort((a, b) => (a[key] ?? 0) - (b[key] ?? 0));
}

function applyModelFilter(rows) {
  if (!_modelFilter) return rows;
  return rows.filter(r => _modelFilter.has(r.model_name || r.config_name));
}

function updateLastSynced() {
  if (!_lastSyncTime) return;
  const el = document.getElementById("last-synced");
  if (!el) return;
  const secs = Math.round((Date.now() - _lastSyncTime) / 1000);
  el.textContent = secs < 5  ? "synced: just now"
                 : secs < 60 ? `synced: ${secs}s ago`
                 : `synced: ${Math.round(secs / 60)}m ago`;
}

// ─────────────────────────────────────────────
// Score cards (Overview tab)
// ─────────────────────────────────────────────

function renderScoreCards(data) {
  const container = document.getElementById("score-cards");
  if (!container) return;
  container.innerHTML = "";

  data.forEach(d => {
    const name  = modelLabel(d);
    const color = modelColor(name);

    const card = document.createElement("div");
    card.className = "score-card";

    const header = document.createElement("div");
    header.className = "score-card-header";
    header.textContent = name;
    header.style.borderBottomColor = color;
    header.style.color = color;
    card.appendChild(header);

    const metrics = [
      { label: "Recall",     value: d.recall_pass_rate, fmt: "pct" },
      { label: "HumanEval+", value: d.humaneval_plus,   fmt: "pct" },
      { label: "GSM8K",      value: d.gsm8k,            fmt: "pct" },
      { label: "IFEval",     value: d.ifeval,            fmt: "pct" },
      { label: "Speed (~8K)", value: d.gen_tps_8k,       fmt: "tps" },
    ];

    metrics.forEach(m => {
      const row   = document.createElement("div");
      row.className = "score-card-metric";
      const nameEl = document.createElement("span");
      nameEl.className = "metric-name";
      nameEl.textContent = m.label;
      const valEl = document.createElement("span");
      valEl.className = "metric-value";
      if (m.value == null) {
        valEl.textContent = "—";
        valEl.style.color = "#4b5563";
      } else if (m.fmt === "pct") {
        valEl.textContent = (m.value * 100).toFixed(1) + "%";
      } else {
        valEl.textContent = m.value.toFixed(1) + " t/s";
      }
      row.appendChild(nameEl);
      row.appendChild(valEl);
      card.appendChild(row);
    });

    container.appendChild(card);
  });
}

// ─────────────────────────────────────────────
// Auto-update (SSE + polling fallback)
// ─────────────────────────────────────────────

function showBanner() {
  clearInterval(_bannerInterval);
  let countdown = 60;
  const countEl = document.getElementById("banner-countdown");
  if (countEl) countEl.textContent = countdown;
  const banner = document.getElementById("refresh-banner");
  if (banner) banner.classList.remove("hidden");

  _bannerInterval = setInterval(() => {
    countdown--;
    if (countEl) countEl.textContent = countdown;
    if (countdown <= 0) {
      dismissBanner();
      renderCurrentTab();
    }
  }, 1000);
}

function dismissBanner() {
  clearInterval(_bannerInterval);
  _bannerInterval = null;
  const banner = document.getElementById("refresh-banner");
  if (banner) banner.classList.add("hidden");
}

function initSSE() {
  const dot = document.getElementById("sse-indicator");

  function connect() {
    if (_sseSource) _sseSource.close();
    _sseSource = new EventSource("/api/events");

    _sseSource.onopen = () => {
      if (dot) { dot.className = "sse-dot connected"; dot.title = "Live updates connected"; }
      _lastSyncTime = Date.now();
      updateLastSynced();
    };

    _sseSource.onmessage = (e) => {
      _lastSyncTime = Date.now();
      updateLastSynced();
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "update") {
          const banner = document.getElementById("refresh-banner");
          if (banner && banner.classList.contains("hidden")) showBanner();
        }
      } catch (_) {}
    };

    _sseSource.onerror = () => {
      if (dot) { dot.className = "sse-dot disconnected"; dot.title = "Live updates disconnected — polling active"; }
      _sseSource.close();
      _sseSource = null;
      startPollingFallback();
    };
  }

  connect();
}

function startPollingFallback() {
  if (_pollFallbackInterval) return;
  _pollFallbackInterval = setInterval(async () => {
    try {
      const r = await fetchAPI("/last-updated");
      _lastSyncTime = Date.now();
      updateLastSynced();
      const banner = document.getElementById("refresh-banner");
      if (_lastPolledTs && r.ts !== _lastPolledTs && banner && banner.classList.contains("hidden")) {
        showBanner();
      }
      _lastPolledTs = r.ts;
    } catch (_) {}
  }, 30_000);
}

// ─────────────────────────────────────────────
// Model filter panel
// ─────────────────────────────────────────────

function buildModelFilterPanel(models) {
  _knownModels = models;
  const panel = document.getElementById("model-filter-panel");
  if (!panel) return;
  panel.innerHTML = "";

  const allLabel = document.createElement("label");
  const allCb = document.createElement("input");
  allCb.type = "checkbox";
  allCb.id = "mf-all";
  allCb.checked = true;
  allLabel.appendChild(allCb);
  allLabel.appendChild(document.createTextNode(" All models"));
  panel.appendChild(allLabel);

  models.forEach(name => {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = name;
    cb.dataset.model = name;
    cb.checked = true;
    cb.addEventListener("change", updateModelFilter);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(" " + name));
    panel.appendChild(label);
  });

  allCb.addEventListener("change", (e) => {
    panel.querySelectorAll("input[data-model]").forEach(cb => { cb.checked = e.target.checked; });
    updateModelFilter();
  });
}

function updateModelFilter() {
  const checked = [...document.querySelectorAll("#model-filter-panel input[data-model]:checked")]
    .map(cb => cb.value);
  _modelFilter = checked.length === _knownModels.length ? null : new Set(checked);
  renderCurrentTab();
}

// ─────────────────────────────────────────────
// Tab 1: Overview — Radar + score cards
// ─────────────────────────────────────────────

async function renderOverview() {
  showSpinner("chart-radar");
  const f = getFilters();
  let data = await fetchAPI("/overview", f).catch(() => []);
  hideSpinner("chart-radar");
  data = applyModelFilter(data);

  if (!data.length) {
    emptyChart("chart-radar");
    const sc = document.getElementById("score-cards");
    if (sc) sc.innerHTML = "";
    return;
  }

  const tpsList = data.map(d => d.gen_tps_8k).filter(v => v != null);
  const maxTps  = tpsList.length ? Math.max(...tpsList) : 1;
  const opacity  = data.length > 5 ? 0.45 : 0.75;
  const lineWidth = data.length > 5 ? 1 : 2;

  const traces = data.map(d => {
    const name  = modelLabel(d);
    const color = modelColor(name);
    return {
      type: "scatterpolar",
      r: [
        d.recall_pass_rate ?? 0,
        d.humaneval_plus   ?? 0,
        d.gsm8k            ?? 0,
        d.ifeval           ?? 0,
        maxTps > 0 ? (d.gen_tps_8k ?? 0) / maxTps : 0,
      ],
      theta: ["Long Context\nRecall", "HumanEval+", "GSM8K", "IFEval", "Speed\n(norm)"],
      fill: "toself",
      name,
      opacity,
      line: { width: lineWidth, color },
      fillcolor: hexToRgba(color, opacity * 0.55),
    };
  });

  const scored = data.map(d => {
    const vals = [d.recall_pass_rate, d.humaneval_plus, d.gsm8k, d.ifeval].filter(v => v != null);
    return { name: modelLabel(d), avg: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0 };
  }).sort((a, b) => b.avg - a.avg);
  const subtitle = scored.length ? ` — top: ${scored[0].name}` : "";

  Plotly.newPlot("chart-radar", traces, {
    ...PLOTLY_LAYOUT_BASE,
    polar: {
      radialaxis: { range: [0, 1], color: "#4b5563", tickformat: ".0%", tickfont: { size: 9 } },
      bgcolor: "#1a1d27",
    },
    title: { text: `Model Overview${subtitle}`, font: { color: "#fff" } },
    showlegend: true,
    legend: { font: { color: "#e0e0e0" } },
  }, { responsive: true });

  renderScoreCards(data);
}

// ─────────────────────────────────────────────
// Tab 2: Coding — HumanEval+ / MBPP+
// ─────────────────────────────────────────────

async function renderCoding() {
  showSpinner("chart-coding-leaderboard");
  const f = getFilters();
  let data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "coding-standard" }).catch(() => []);
  hideSpinner("chart-coding-leaderboard");
  data = applyModelFilter(data);
  if (!data.length) return emptyChart("chart-coding-leaderboard");

  const pass1 = data.filter(d => d.metric && (d.metric.includes("pass@1") || d.metric.includes("pass_at_1")));
  const tasks  = [...new Set(pass1.map(d => d.task))];
  const heTask = tasks.find(t => t.includes("humaneval")) || tasks[0];
  const allModels = [...new Set(pass1.map(d => modelLabel(d)))];

  const modelsSorted = allModels.sort((a, b) => {
    const aVal = pass1.find(d => modelLabel(d) === a && d.task === heTask)?.value ?? 0;
    const bVal = pass1.find(d => modelLabel(d) === b && d.task === heTask)?.value ?? 0;
    return aVal - bVal; // ascending = best at top in Plotly h-bar
  });

  const traces = tasks.map(task => ({
    type: "bar", orientation: "h",
    x: modelsSorted.map(m => pass1.find(d => modelLabel(d) === m && d.task === task)?.value ?? 0),
    y: modelsSorted,
    name: taskLabel(task),
    text: modelsSorted.map(m => {
      const v = pass1.find(d => modelLabel(d) === m && d.task === task)?.value;
      return v != null ? `${(v * 100).toFixed(1)}%` : "";
    }),
    textposition: "outside",
    error_x: {
      type: "data",
      array: modelsSorted.map(m => pass1.find(d => modelLabel(d) === m && d.task === task)?.stderr ?? 0),
    },
  }));

  const chartEl = document.getElementById("chart-coding-leaderboard");
  chartEl.style.height = Math.max(350, modelsSorted.length * 48 + 100) + "px";

  Plotly.newPlot("chart-coding-leaderboard", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "Python Coding — EvalPlus (pass@1)", font: { color: "#fff" } },
    xaxis: { title: "pass@1", range: [0, 1.18], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 70 },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 3: Reasoning
// ─────────────────────────────────────────────

async function renderReasoning() {
  showSpinner("chart-reasoning-leaderboard");
  const f = getFilters();
  let data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "reasoning" }).catch(() => []);
  hideSpinner("chart-reasoning-leaderboard");
  data = applyModelFilter(data);
  if (!data.length) return emptyChart("chart-reasoning-leaderboard");

  function primaryMetric(task, metric) {
    if (task === "ifeval") return metric === "prompt_level_strict_acc,none";
    return metric && metric.includes("flexible");
  }
  // Drop BBH subtasks — keep only the top-level aggregate (bbh_cot_zeroshot)
  const BBH_SUBTASK = /^bbh_cot_zeroshot_.+/;
  const filtered = data.filter(d => primaryMetric(d.task, d.metric) && !BBH_SUBTASK.test(d.task));
  const tasks    = [...new Set(filtered.map(d => d.task))];
  const allModels = [...new Set(filtered.map(d => modelLabel(d)))];

  const modelsSorted = allModels.sort((a, b) => {
    const avg = name => {
      const vals = tasks.map(t => filtered.find(d => modelLabel(d) === name && d.task === t)?.value ?? 0);
      return vals.reduce((s, v) => s + v, 0) / (vals.length || 1);
    };
    return avg(a) - avg(b);
  });

  const traces = tasks.map(task => ({
    type: "bar", orientation: "h",
    x: modelsSorted.map(m => filtered.find(d => modelLabel(d) === m && d.task === task)?.value ?? 0),
    y: modelsSorted,
    name: taskLabel(task),
    text: modelsSorted.map(m => {
      const v = filtered.find(d => modelLabel(d) === m && d.task === task)?.value;
      return v != null ? `${(v * 100).toFixed(1)}%` : "";
    }),
    textposition: "outside",
  }));

  const chartEl = document.getElementById("chart-reasoning-leaderboard");
  chartEl.style.height = Math.max(350, modelsSorted.length * 48 + 100) + "px";

  Plotly.newPlot("chart-reasoning-leaderboard", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "Reasoning Benchmarks", font: { color: "#fff" } },
    xaxis: { title: "Score", range: [0, 1.18], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 70 },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 5: Long Context Recall
// ─────────────────────────────────────────────

async function renderRecallLeaderboard() {
  showSpinner("chart-recall-leaderboard");
  const f = getFilters();
  let data = await fetchAPI("/recall/leaderboard", f).catch(() => []);
  hideSpinner("chart-recall-leaderboard");

  if (f.architecture) data = data.filter(r => r.architecture === f.architecture);
  data = applyModelFilter(data);
  if (!data.length) return emptyChart("chart-recall-leaderboard");

  const sorted = sortedByValue(data, "pass_rate");

  const sel = document.getElementById("filter-depth-run");
  sel.innerHTML = '<option value="">select a run</option>';
  data.forEach(r => {
    const opt = document.createElement("option");
    opt.value = r.run_id;
    opt.textContent = `${modelLabel(r)} — ${r.corpus}`;
    sel.appendChild(opt);
  });

  Plotly.newPlot("chart-recall-leaderboard", [{
    type: "bar", orientation: "h",
    x: sorted.map(d => d.pass_rate ?? 0),
    y: sorted.map(d => `${modelLabel(d)} | ${d.corpus}`),
    text: sorted.map(d => `${((d.pass_rate ?? 0) * 100).toFixed(0)}%`),
    textposition: "outside",
    marker: { color: sorted.map(d => modelColor(modelLabel(d))) },
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Long Context Recall — Codeneedle", font: { color: "#fff" } },
    xaxis: { title: "Pass rate", range: [0, 1.2], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 70 },
  }, { responsive: true });
}

async function renderRecallDepth(runId) {
  if (!runId) return;
  showSpinner("chart-recall-depth");
  const data = await fetchAPI("/recall/depth", { run_id: runId }).catch(() => []);
  hideSpinner("chart-recall-depth");
  if (!data.length) return emptyChart("chart-recall-depth", "No depth data for this run");

  const maxLine = Math.max(...data.map(d => d.start_line || 0)) || 1;

  Plotly.newPlot("chart-recall-depth", [{
    type: "scatter", mode: "markers",
    x: data.map(d => (d.start_line || 0) / maxLine),
    y: data.map(d => (d.primary_matched || 0) / 20),
    text: data.map(d => d.function_name),
    hovertemplate: "<b>%{text}</b><br>depth: %{x:.0%}<br>matched: %{y:.0%}<extra></extra>",
    marker: {
      color: data.map(d => (d.pass_rate ?? 0) > 0 ? "#10b981" : "#ef4444"),
      size: 9,
      opacity: 0.85,
    },
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Recall Accuracy vs Context Depth", font: { color: "#fff" } },
    xaxis: { title: "Depth in file (0 = start, 1 = end)", range: [0, 1], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { title: "Lines matched / 20", range: [0, 1.05], color: "#9ca3af" },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 6: Speed
// ─────────────────────────────────────────────

async function renderSpeed() {
  showSpinner("chart-speed-curves");
  showSpinner("chart-speed-bar");
  const f = getFilters();
  let [curves, bar] = await Promise.all([
    fetchAPI("/speed/curves", f).catch(() => []),
    fetchAPI("/speed/comparison", { ...f, context_tokens: 8192 }).catch(() => []),
  ]);
  curves = applyModelFilter(curves);
  bar    = applyModelFilter(bar);
  hideSpinner("chart-speed-curves");
  hideSpinner("chart-speed-bar");

  if (!curves.length) {
    emptyChart("chart-speed-curves");
    emptyChart("chart-speed-bar");
    return;
  }

  const models = [...new Set(curves.map(d => modelLabel(d)))];
  const curveTraces = models.flatMap(m => {
    const rows  = curves.filter(d => modelLabel(d) === m).sort((a, b) => a.context_tokens - b.context_tokens);
    const color = modelColor(m);
    const yVals = rows.map(r => r.generation_tps_mean ?? r.overall_tps_mean ?? null);
    const sdVals = rows.map(r => r.generation_tps_stddev ?? r.overall_tps_stddev ?? 0);
    const hasStddev = sdVals.some(v => v > 0);

    const mainTrace = {
      type: "scatter", mode: "lines+markers",
      x: rows.map(r => r.context_tokens),
      y: yVals,
      name: m,
      connectgaps: false,
      line: { color },
      marker: { color, size: 6 },
    };

    if (!hasStddev) return [mainTrace];

    const xs = rows.map(r => r.context_tokens);
    const upper = yVals.map((v, i) => (v ?? 0) + sdVals[i]);
    const lower = yVals.map((v, i) => Math.max(0, (v ?? 0) - sdVals[i]));
    const bandTrace = {
      type: "scatter", mode: "none",
      x: [...xs, ...xs.slice().reverse()],
      y: [...upper, ...lower.slice().reverse()],
      fill: "toself",
      fillcolor: hexToRgba(color, 0.12),
      line: { color: "transparent" },
      showlegend: false,
      hoverinfo: "skip",
    };
    return [bandTrace, mainTrace];
  });

  Plotly.newPlot("chart-speed-curves", curveTraces, {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Generation Speed vs Context Size", font: { color: "#fff" } },
    xaxis: { title: "Context tokens", type: "log", color: "#9ca3af" },
    yaxis: { title: "tokens / sec", color: "#9ca3af" },
  }, { responsive: true });

  if (bar.length) {
    const sortedBar = sortedByValue(bar, "generation_tps_mean");
    Plotly.newPlot("chart-speed-bar", [{
      type: "bar", orientation: "h",
      x: sortedBar.map(d => d.generation_tps_mean ?? d.overall_tps_mean ?? 0),
      y: sortedBar.map(d => modelLabel(d)),
      text: sortedBar.map(d => {
        const v = d.generation_tps_mean ?? d.overall_tps_mean ?? 0;
        return `${v.toFixed(1)} t/s`;
      }),
      textposition: "outside",
      marker: { color: sortedBar.map(d => modelColor(modelLabel(d))) },
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: "Speed at ~8K Context (tokens/sec)", font: { color: "#fff" } },
      xaxis: { title: "tokens / sec", color: "#9ca3af" },
      yaxis: { color: "#9ca3af", automargin: true },
      margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 80 },
    }, { responsive: true });
  } else {
    emptyChart("chart-speed-bar");
  }
}

// ─────────────────────────────────────────────
// Tab 7: Quant Impact
// ─────────────────────────────────────────────

async function renderQuantImpact() {
  const modelConfig = document.getElementById("filter-quant-model").value;
  if (!modelConfig) return emptyChart("chart-quant-impact", "Select a model above");

  showSpinner("chart-quant-impact");
  const data = await fetchAPI("/quant-impact", { model_config: modelConfig }).catch(() => []);
  hideSpinner("chart-quant-impact");
  if (!data.length) return emptyChart("chart-quant-impact", "No quant comparison data yet");

  const quants  = [...new Set(data.map(d => d.quantization))].sort();
  const metrics = [...new Set(data.map(d => d.metric))];

  const traces = metrics.map(metric => ({
    type: "bar",
    x: quants,
    y: quants.map(q => {
      const r = data.find(d => d.quantization === q && d.metric === metric);
      return r ? r.value : 0;
    }),
    name: metric,
  }));

  Plotly.newPlot("chart-quant-impact", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: `Quantization Impact — ${modelConfig}`, font: { color: "#fff" } },
    xaxis: { title: "Quantization", color: "#9ca3af" },
    yaxis: { title: "Score", color: "#9ca3af" },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────

async function populateDropdowns() {
  const opts = await fetchAPI("/filter-options").catch(() => ({}));

  const fill = (id, vals) => {
    const sel = document.getElementById(id);
    const first = sel.querySelector("option");
    sel.innerHTML = "";
    sel.appendChild(first);
    (vals || []).forEach(v => {
      const opt = document.createElement("option");
      opt.value = opt.textContent = v;
      sel.appendChild(opt);
    });
  };

  fill("filter-runtime", opts.runtimes);
  fill("filter-quant",   opts.quantizations);
  fill("filter-arch",    opts.architectures);

  const qm = document.getElementById("filter-quant-model");
  const firstOpt = qm.querySelector("option");
  qm.innerHTML = "";
  qm.appendChild(firstOpt);
  (opts.model_configs || []).forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    qm.appendChild(opt);
  });

  buildModelFilterPanel(opts.model_configs || []);
}

function renderCurrentTab() {
  const renderers = {
    overview:  renderOverview,
    coding:    renderCoding,
    reasoning: renderReasoning,
    recall:    renderRecallLeaderboard,
    speed:     renderSpeed,
    quant:     renderQuantImpact,
  };
  const fn = renderers[_currentTab];
  if (fn) fn().catch(err => console.error("Chart error:", err));
}

async function init() {
  // Tab switching
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      _currentTab = btn.dataset.tab;
      renderCurrentTab();
    });
  });

  // Global filter dropdowns
  ["filter-runtime", "filter-quant", "filter-arch"].forEach(id => {
    document.getElementById(id).addEventListener("change", renderCurrentTab);
  });

  // Recall depth run selector
  document.getElementById("filter-depth-run").addEventListener("change", e => {
    renderRecallDepth(e.target.value);
  });

  // Quant model selector
  document.getElementById("filter-quant-model").addEventListener("change", renderQuantImpact);

  // Auto-refresh banner buttons
  document.getElementById("banner-refresh").addEventListener("click", () => {
    dismissBanner();
    renderCurrentTab();
  });
  document.getElementById("banner-cancel").addEventListener("click", dismissBanner);

  // Model filter panel toggle
  document.getElementById("model-filter-toggle").addEventListener("click", e => {
    e.stopPropagation();
    document.getElementById("model-filter-panel").classList.toggle("hidden");
  });
  document.addEventListener("click", e => {
    if (!e.target.closest("#filters")) {
      const panel = document.getElementById("model-filter-panel");
      if (panel) panel.classList.add("hidden");
    }
  });

  // Last-synced ticker
  setInterval(updateLastSynced, 10_000);

  await populateDropdowns();
  renderCurrentTab();
  initSSE();
}

document.addEventListener("DOMContentLoaded", init);
