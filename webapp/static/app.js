// ─────────────────────────────────────────────
// Constants & State
// ─────────────────────────────────────────────

const MODEL_PALETTE = [
  "#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6",
  "#8b5cf6","#ec4899","#14b8a6","#f97316","#84cc16",
  "#06b6d4","#a855f7",
];

// Colors reserved for pass/fail agreement markers in the recall comparison chart
const _RECALL_AGREE_COLORS = new Set(["#10b981", "#ef4444"]);
// Fallback palette that excludes those reserved colors
const _RECALL_MODEL_PALETTE = MODEL_PALETTE.filter(c => !_RECALL_AGREE_COLORS.has(c));
function _recallModelColor(label) {
  const base = modelColor(label);
  if (!_RECALL_AGREE_COLORS.has(base)) return base;
  // Clash — pick the next available safe color
  return _RECALL_MODEL_PALETTE[Math.abs([...label].reduce((h, c) => h * 31 + c.charCodeAt(0), 0)) % _RECALL_MODEL_PALETTE.length];
}

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
  bbh_cot_zeroshot:      "BBH",
  gsm8k_cot_zeroshot:    "GSM8K",
  ifeval:                "IFEval",
  multiple_py:           "Python (MultiPL)",
  multiple_js:           "JavaScript (MultiPL)",
  multiple_ts:           "TypeScript (MultiPL)",
  multiple_java:         "Java (MultiPL)",
  multiple_cpp:          "C++ (MultiPL)",
  multiple_rs:           "Rust (MultiPL)",
  multiple_go:           "Go (MultiPL)",
};
function taskLabel(t) {
  if (TASK_DISPLAY[t]) return TASK_DISPLAY[t];
  const bbh = t.match(/^bbh_cot_zeroshot_(.+)$/);
  if (bbh) return "BBH: " + bbh[1].replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  return t;
}

let _currentTab      = "overview";
let _sseSource       = null;
let _bannerInterval  = null;
let _lastSyncTime    = null;
let _modelFilter     = null; // null = all; Set<"model_name|quantization"> = allow-list
let _knownModelQuants = []; // [{model_name, quantization}]
let _pollFallbackInterval = null;
let _lastPolledTs    = null;
let _reasoningData   = [];
let _bbhModels            = []; // models currently shown in BBH drilldown
let _recallModels         = []; // keys currently shown in recall drilldown (max 2)
let _recallLeaderboardData = [];

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
  return rows.filter(r => {
    const key = (r.model_name || r.config_name || "") + "|" + (r.quantization || "");
    return _modelFilter.has(key);
  });
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

function buildModelFilterPanel(modelQuants) {
  _knownModelQuants = modelQuants;
  const panel = document.getElementById("model-filter-panel");
  if (!panel) return;
  panel.innerHTML = "";

  const allLabel = document.createElement("label");
  const allCb = document.createElement("input");
  allCb.type = "checkbox"; allCb.id = "mf-all"; allCb.checked = true;
  allLabel.appendChild(allCb);
  allLabel.appendChild(document.createTextNode(" All models"));
  panel.appendChild(allLabel);

  // Group by model_name
  const groups = {};
  modelQuants.forEach(({ model_name, quantization }) => {
    if (!groups[model_name]) groups[model_name] = [];
    groups[model_name].push(quantization);
  });

  Object.entries(groups).forEach(([modelName, quants]) => {
    if (quants.length === 1) {
      const key = modelName + "|" + quants[0];
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = true; cb.dataset.key = key;
      cb.addEventListener("change", updateModelFilter);
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + modelName + (quants[0] ? ` (${quants[0]})` : "")));
      panel.appendChild(label);
    } else {
      const parentLabel = document.createElement("label");
      parentLabel.className = "mf-parent";
      const parentCb = document.createElement("input");
      parentCb.type = "checkbox"; parentCb.checked = true; parentCb.dataset.parent = modelName;
      parentLabel.appendChild(parentCb);
      parentLabel.appendChild(document.createTextNode(" " + modelName));
      panel.appendChild(parentLabel);

      quants.forEach(quant => {
        const key = modelName + "|" + quant;
        const childLabel = document.createElement("label");
        childLabel.className = "mf-child";
        const childCb = document.createElement("input");
        childCb.type = "checkbox"; childCb.checked = true;
        childCb.dataset.key = key; childCb.dataset.parentModel = modelName;
        childCb.addEventListener("change", () => { _syncParent(modelName); updateModelFilter(); });
        childLabel.appendChild(childCb);
        childLabel.appendChild(document.createTextNode(" " + (quant || "unknown")));
        panel.appendChild(childLabel);
      });

      parentCb.addEventListener("change", e => {
        panel.querySelectorAll(`input[data-parent-model="${modelName}"]`).forEach(cb => { cb.checked = e.target.checked; });
        updateModelFilter();
      });
    }
  });

  allCb.addEventListener("change", e => {
    panel.querySelectorAll("input[data-key]").forEach(cb => { cb.checked = e.target.checked; });
    panel.querySelectorAll("input[data-parent]").forEach(cb => { cb.checked = e.target.checked; cb.indeterminate = false; });
    updateModelFilter();
  });
}

function _syncParent(modelName) {
  const panel = document.getElementById("model-filter-panel");
  const children = [...panel.querySelectorAll(`input[data-parent-model="${modelName}"]`)];
  const parentCb = panel.querySelector(`input[data-parent="${modelName}"]`);
  if (!parentCb) return;
  const n = children.filter(c => c.checked).length;
  parentCb.checked = n > 0;
  parentCb.indeterminate = n > 0 && n < children.length;
}

function updateModelFilter() {
  const all = [...document.querySelectorAll("#model-filter-panel input[data-key]")];
  const checked = all.filter(cb => cb.checked).map(cb => cb.dataset.key);
  _modelFilter = checked.length === all.length ? null : new Set(checked);
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
  const data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "coding-standard" }).catch(() => []);
  hideSpinner("chart-coding-leaderboard");
  const filtered = applyModelFilter(data);
  if (!filtered.length) return emptyChart("chart-coding-leaderboard");

  const pass1 = filtered.filter(d => d.metric && (d.metric.includes("pass@1") || d.metric.includes("pass_at_1")));
  const tasks  = [...new Set(pass1.map(d => d.task))];
  const heTask = tasks.find(t => t.includes("humaneval")) || tasks[0];
  const allModels = [...new Set(pass1.map(d => modelLabel(d)))];

  const modelsSorted = allModels.sort((a, b) => {
    const aVal = pass1.find(d => modelLabel(d) === a && d.task === heTask)?.value ?? 0;
    const bVal = pass1.find(d => modelLabel(d) === b && d.task === heTask)?.value ?? 0;
    return aVal - bVal;
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

function _primaryMetric(task, metric) {
  if (task === "ifeval") return metric === "prompt_level_strict_acc,none";
  return metric && metric.includes("flexible");
}
const _BBH_SUBTASK = /^bbh_cot_zeroshot_.+/;

async function renderReasoning() {
  showSpinner("chart-reasoning-leaderboard");
  document.getElementById("reasoning-drilldown")?.classList.add("hidden");
  _bbhModels = [];
  const f = getFilters();
  let data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "reasoning" }).catch(() => []);
  hideSpinner("chart-reasoning-leaderboard");
  data = applyModelFilter(data);
  _reasoningData = data;
  if (!data.length) return emptyChart("chart-reasoning-leaderboard");

  // Summary: top-level benchmarks only (no individual BBH subtasks)
  const filtered = data.filter(d =>
    _primaryMetric(d.task, d.metric) && !_BBH_SUBTASK.test(d.task)
  );
  const tasks     = [...new Set(filtered.map(d => d.task))];
  const allModels = [...new Set(filtered.map(d => modelLabel(d)))];

  const modelsSorted = allModels.sort((a, b) => {
    const avg = name => {
      const vals = tasks.map(t => filtered.find(d => modelLabel(d) === name && d.task === t)?.value ?? 0);
      return vals.reduce((s, v) => s + v, 0) / (vals.length || 1);
    };
    return avg(a) - avg(b);
  });

  const traces = tasks.map(task => {
    const isBBH = task.startsWith("bbh");
    return {
      type: "bar", orientation: "h",
      x: modelsSorted.map(m => filtered.find(d => modelLabel(d) === m && d.task === task)?.value ?? 0),
      y: modelsSorted,
      name: isBBH ? taskLabel(task) + "  ▶" : taskLabel(task),
      customdata: modelsSorted.map(m => task + "|" + m),
      text: modelsSorted.map(m => {
        const v = filtered.find(d => modelLabel(d) === m && d.task === task)?.value;
        const pct = v != null ? `${(v * 100).toFixed(1)}%` : "";
        return isBBH && pct ? pct + "  ▶" : pct;
      }),
      textposition: "outside",
      textfont: isBBH ? { size: 13 } : {},
    };
  });

  const chartEl = document.getElementById("chart-reasoning-leaderboard");
  chartEl.style.height = Math.max(350, modelsSorted.length * 48 + 100) + "px";
  Plotly.newPlot("chart-reasoning-leaderboard", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "Reasoning Benchmarks  ·  click BBH to expand", font: { color: "#fff" } },
    xaxis: { title: "Score", range: [0, 1.18], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 70 },
  }, { responsive: true });

  chartEl.style.cursor = "pointer";
  chartEl.on("plotly_click", d => {
    const raw = d.points[0]?.customdata;
    if (!raw || !raw.startsWith("bbh")) return;
    const clickedModel = raw.split("|").slice(1).join("|");
    const panel = document.getElementById("reasoning-drilldown");
    const isOpen = panel && !panel.classList.contains("hidden");
    if (isOpen && _bbhModels[0] === clickedModel) {
      panel.classList.add("hidden");
      Plotly.purge(document.getElementById("chart-reasoning-drilldown"));
      _bbhModels = [];
    } else {
      openReasoningDrilldown(clickedModel);
    }
  });
}

function openReasoningDrilldown(clickedModel) {
  const panel = document.getElementById("reasoning-drilldown");
  if (!panel) return;
  _bbhModels = [clickedModel];
  panel.classList.remove("hidden");
  renderBBHDrilldown();
}

function renderBBHDrilldown() {
  const subtaskData = _reasoningData.filter(d =>
    _BBH_SUBTASK.test(d.task) && _primaryMetric(d.task, d.metric)
  );
  if (!subtaskData.length) return;

  showSpinner("chart-reasoning-drilldown");

  // Update title
  const titleEl = document.getElementById("reasoning-drilldown-title");
  if (titleEl) titleEl.textContent = "BBH — " + _bbhModels.join(", ");

  // Populate add-model dropdown with models not yet shown
  const allAvailable = [...new Set(subtaskData.map(d => modelLabel(d)))];
  const addSel = document.getElementById("bbh-add-model");
  if (addSel) {
    addSel.innerHTML = '<option value="">＋ Add model...</option>';
    allAvailable.filter(m => !_bbhModels.includes(m)).forEach(m => {
      const opt = document.createElement("option");
      opt.value = opt.textContent = m;
      addSel.appendChild(opt);
    });
    addSel.style.display = allAvailable.length > _bbhModels.length ? "" : "none";
  }

  const tasksSorted = [...new Set(subtaskData.map(d => d.task))].sort((a, b) => {
    const val = (t, m) => subtaskData.find(d => d.task === t && modelLabel(d) === m)?.value ?? 0;
    return val(a, _bbhModels[0]) - val(b, _bbhModels[0]);
  });

  const traces = _bbhModels.map(m => ({
    type: "bar", orientation: "h",
    x: tasksSorted.map(t => subtaskData.find(d => modelLabel(d) === m && d.task === t)?.value ?? 0),
    y: tasksSorted.map(t => taskLabel(t)),
    name: m,
    text: tasksSorted.map(t => {
      const v = subtaskData.find(d => modelLabel(d) === m && d.task === t)?.value;
      return v != null ? `${(v * 100).toFixed(1)}%` : "";
    }),
    textposition: "outside",
    marker: { color: modelColor(m) },
  }));

  const drillEl = document.getElementById("chart-reasoning-drilldown");
  drillEl.style.height = Math.max(400, tasksSorted.length * 28 + 120) + "px";
  hideSpinner("chart-reasoning-drilldown");

  Plotly.newPlot("chart-reasoning-drilldown", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "BBH — Subtask Breakdown", font: { color: "#fff" } },
    xaxis: { title: "Score", range: [0, 1.18], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true, tickfont: { size: 10 } },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, l: 220, r: 70 },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 5: Long Context Recall
// ─────────────────────────────────────────────

async function renderRecallLeaderboard() {
  showSpinner("chart-recall-leaderboard");
  document.getElementById("recall-drilldown")?.classList.add("hidden");
  _recallModels = [];
  const f = getFilters();
  let data = await fetchAPI("/recall/leaderboard", f).catch(() => []);
  hideSpinner("chart-recall-leaderboard");

  if (f.architecture) data = data.filter(r => r.architecture === f.architecture);
  data = applyModelFilter(data);
  _recallLeaderboardData = data;
  if (!data.length) return emptyChart("chart-recall-leaderboard");

  const sorted = sortedByValue(data, "pass_rate");

  Plotly.newPlot("chart-recall-leaderboard", [{
    type: "bar", orientation: "h",
    x: sorted.map(d => d.pass_rate ?? 0),
    y: sorted.map(d => `${modelLabel(d)} | ${d.corpus}`),
    text: sorted.map(d => `${((d.pass_rate ?? 0) * 100).toFixed(0)}%  ▶`),
    textposition: "outside",
    textfont: { size: 13 },
    marker: { color: sorted.map(d => modelColor(modelLabel(d))) },
    customdata: sorted.map(d => JSON.stringify({ config_name: d.config_name, quantization: d.quantization, corpus: d.corpus })),
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Long Context Recall — Codeneedle  ·  click a bar to expand", font: { color: "#fff" } },
    xaxis: { title: "Pass rate", range: [0, 1.2], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 70 },
  }, { responsive: true });

  const recallChartEl = document.getElementById("chart-recall-leaderboard");
  recallChartEl.style.cursor = "pointer";
  recallChartEl.on("plotly_click", d => {
    const key = d.points[0]?.customdata;
    if (!key) return;
    const panel = document.getElementById("recall-drilldown");
    const isOpen = panel && !panel.classList.contains("hidden");
    if (isOpen && _recallModels[0] === key) {
      panel.classList.add("hidden");
      Plotly.purge(document.getElementById("chart-recall-depth"));
      _recallModels = [];
    } else {
      openRecallDrilldown(key);
    }
  });
}

function _recallKeyLabel(key) {
  const { config_name, quantization, corpus } = JSON.parse(key);
  const row = _recallLeaderboardData.find(r =>
    r.config_name === config_name && r.quantization === quantization && r.corpus === corpus
  );
  return row ? modelLabel(row) : (quantization ? `${config_name} (${quantization})` : config_name);
}

function openRecallDrilldown(key) {
  const panel = document.getElementById("recall-drilldown");
  if (!panel) return;
  _recallModels = [key];
  panel.classList.remove("hidden");
  renderRecallDepth();
}

async function renderRecallDepth() {
  if (!_recallModels.length) return;
  showSpinner("chart-recall-depth");

  const datasets = await Promise.all(_recallModels.map(k => {
    const { config_name, quantization, corpus } = JSON.parse(k);
    return fetchAPI("/recall/depth", { config_name, quantization, corpus }).catch(() => []);
  }));

  hideSpinner("chart-recall-depth");
  if (!datasets[0].length) return emptyChart("chart-recall-depth", "No depth data");

  // Update title
  const titleEl = document.getElementById("recall-drilldown-title");
  if (titleEl) titleEl.textContent = "Recall Depth — " + _recallModels.map(_recallKeyLabel).join(" vs ");

  // Populate add-model dropdown
  const addSel = document.getElementById("recall-add-model");
  if (addSel) {
    addSel.innerHTML = '<option value="">＋ Add model...</option>';
    if (_recallModels.length < 2) {
      _recallLeaderboardData
        .filter(r => {
          const k = JSON.stringify({ config_name: r.config_name, quantization: r.quantization, corpus: r.corpus });
          return !_recallModels.includes(k);
        })
        .forEach(r => {
          const k = JSON.stringify({ config_name: r.config_name, quantization: r.quantization, corpus: r.corpus });
          const opt = document.createElement("option");
          opt.value = k;
          opt.textContent = `${modelLabel(r)} | ${r.corpus}`;
          addSel.appendChild(opt);
        });
    }
    addSel.style.display = _recallModels.length < 2 && addSel.options.length > 1 ? "" : "none";
  }

  // Build sorted function list by start_line (shallow → deep in file)
  const allFuncs = [...new Set(datasets.flat().map(d => d.function_name))];
  const lineOf = {};
  datasets.flat().forEach(d => { lineOf[d.function_name] = d.start_line || 0; });
  allFuncs.sort((a, b) => lineOf[a] - lineOf[b]);

  const chartHeight = Math.max(300, allFuncs.length * 36 * (_recallModels.length === 1 ? 1 : 2) + 100);
  document.getElementById("chart-recall-depth").style.height = chartHeight + "px";

  if (_recallModels.length === 1) {
    const mapD = Object.fromEntries(datasets[0].map(d => [d.function_name, d]));
    const color1 = _recallModelColor(_recallKeyLabel(_recallModels[0]));
    Plotly.newPlot("chart-recall-depth", [{
      type: "bar", orientation: "h",
      y: allFuncs,
      x: allFuncs.map(fn => (mapD[fn]?.primary_matched ?? 0) / 20),
      marker: {
        color: allFuncs.map(fn => (mapD[fn]?.pass_rate ?? 0) > 0 ? "#10b981" : hexToRgba(color1, 0.45)),
        line: { color: allFuncs.map(fn => (mapD[fn]?.pass_rate ?? 0) > 0 ? "#10b981" : color1), width: 1.5 },
      },
      text: allFuncs.map(fn => mapD[fn] ? `${mapD[fn].primary_matched}/20` : ""),
      textposition: "outside",
      textfont: { color: "#9ca3af", size: 11 },
      customdata: allFuncs.map(fn => lineOf[fn]),
      hovertemplate: "<b>%{y}</b><br>Matched: %{x:.0%} (%{text})<br>At line %{customdata}<extra></extra>",
      showlegend: false,
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: "Lines Recalled per Function (sorted shallow → deep)", font: { color: "#e0e0e0", size: 13 } },
      height: chartHeight,
      margin: { l: 180, r: 60, t: 50, b: 50 },
      xaxis: { title: "Fraction matched (0–1)", range: [0, 1.15], color: "#9ca3af", tickformat: ".0%" },
      yaxis: { color: "#9ca3af", automargin: true, tickfont: { size: 11 } },
      bargap: 0.3,
    }, { responsive: true });
    return;
  }

  // Two-model comparison — grouped horizontal bars
  const [dataA, dataB] = datasets;
  const labelA = _recallKeyLabel(_recallModels[0]);
  const labelB = _recallKeyLabel(_recallModels[1]);
  const colorA = _recallModelColor(labelA);
  let colorB = _recallModelColor(labelB);
  if (colorB === colorA) {
    colorB = _RECALL_MODEL_PALETTE.find(c => c !== colorA) ?? colorB;
  }
  const mapA = Object.fromEntries(dataA.map(d => [d.function_name, d]));
  const mapB = Object.fromEntries(dataB.map(d => [d.function_name, d]));

  const mkBarTrace = (funcs, modelMap, color, label) => ({
    type: "bar", orientation: "h",
    y: funcs,
    x: funcs.map(fn => (modelMap[fn]?.primary_matched ?? 0) / 20),
    name: label,
    marker: {
      color: funcs.map(fn => (modelMap[fn]?.pass_rate ?? 0) > 0 ? color : hexToRgba(color, 0.35)),
      line: { color, width: 1 },
    },
    text: funcs.map(fn => modelMap[fn] ? `${modelMap[fn].primary_matched}/20` : "–"),
    textposition: "outside",
    textfont: { color: "#9ca3af", size: 10 },
    customdata: funcs.map(fn => lineOf[fn]),
    hovertemplate: "<b>%{y}</b><br>%{x:.0%} matched (%{text})<br>At line %{customdata}<extra>" + label + "</extra>",
  });

  Plotly.newPlot("chart-recall-depth", [
    mkBarTrace(allFuncs, mapA, colorA, labelA),
    mkBarTrace(allFuncs, mapB, colorB, labelB),
  ], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Lines Recalled per Function (sorted shallow → deep)", font: { color: "#e0e0e0", size: 13 } },
    barmode: "group",
    height: chartHeight,
    margin: { l: 180, r: 60, t: 50, b: 50 },
    xaxis: { title: "Fraction matched (0–1)", range: [0, 1.25], color: "#9ca3af", tickformat: ".0%" },
    yaxis: { color: "#9ca3af", automargin: true, tickfont: { size: 11 } },
    legend: { font: { color: "#e0e0e0", size: 11 }, bgcolor: "rgba(0,0,0,0)", x: 0.5, xanchor: "center", y: 1.04, orientation: "h" },
    bargap: 0.25,
    bargroupgap: 0.08,
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
  fill("filter-arch",    opts.architectures);

  buildModelFilterPanel(opts.model_quants || []);
}

function renderCurrentTab() {
  const renderers = {
    overview:  renderOverview,
    coding:    renderCoding,
    reasoning: renderReasoning,
    recall:    renderRecallLeaderboard,
    speed:     renderSpeed,
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
  ["filter-runtime", "filter-arch"].forEach(id => {
    document.getElementById(id).addEventListener("change", renderCurrentTab);
  });

  // Drilldown close buttons
  document.getElementById("reasoning-drilldown-close").addEventListener("click", () => {
    document.getElementById("reasoning-drilldown").classList.add("hidden");
    Plotly.purge(document.getElementById("chart-reasoning-drilldown"));
    _bbhModels = [];
  });

  document.getElementById("bbh-add-model").addEventListener("change", e => {
    const m = e.target.value;
    if (!m || _bbhModels.includes(m)) return;
    _bbhModels.push(m);
    renderBBHDrilldown();
  });
  document.getElementById("recall-drilldown-close").addEventListener("click", () => {
    document.getElementById("recall-drilldown").classList.add("hidden");
    Plotly.purge(document.getElementById("chart-recall-depth"));
    _recallModels = [];
  });

  document.getElementById("recall-add-model").addEventListener("change", e => {
    const k = e.target.value;
    if (!k || _recallModels.includes(k) || _recallModels.length >= 2) return;
    _recallModels.push(k);
    renderRecallDepth();
  });

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
