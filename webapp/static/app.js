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
let _recallCorpusFilter = null; // null = all corpora
const _runSources = {};  // run_id → EventSource
let _modelsActiveModel = null; // name of currently selected model config

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

async function fetchAPI(path, params = {}, method = "GET") {
  const url = `/api${path}`;
  let resp;
  if (method === "GET") {
    const clean = Object.fromEntries(
      Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
    );
    const qs = new URLSearchParams(clean).toString();
    resp = await fetch(`${url}${qs ? "?" + qs : ""}`);
  } else {
    resp = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  }
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
  const quant = row.quantization && !name.toLowerCase().includes(row.quantization.toLowerCase())
    ? ` (${row.quantization})` : "";
  const hw = row.hardware ? ` · ${row.hardware}` : "";
  return `${name}${quant}${hw}`;
}

function showSpinner(chartId) {
  const suffix = chartId.replace("chart-", "");
  const el = document.getElementById("spinner-" + suffix);
  if (el) el.classList.remove("hidden");
  const chart = document.getElementById(chartId);
  // Purge only divs Plotly actually plotted — this also clears their event
  // handlers so re-renders don't stack duplicate plotly_click listeners.
  if (chart && chart.data) Plotly.purge(chart);
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
    const key = (r.model_name || r.config_name || "") + "|" + (r.quantization || "") + "|" + (r.hardware || "");
    return _modelFilter.has(key);
  });
}

// Uniform client-side filtering — every tab passes its rows through here so
// the Runtime/Architecture dropdowns and the model filter behave identically
// everywhere (every query returns runtime + architecture columns).
function applyGlobalFilters(rows) {
  const f = getFilters();
  let out = rows;
  if (f.runtime)      out = out.filter(r => r.runtime === f.runtime);
  if (f.architecture) out = out.filter(r => r.architecture === f.architecture);
  return applyModelFilter(out);
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
      { label: "Recall",       value: d.recall_pass_rate,  fmt: "pct" },
      { label: "HumanEval+",  value: d.humaneval_plus,    fmt: "pct" },
      { label: "GSM8K",       value: d.gsm8k,             fmt: "pct" },
      { label: "IFEval",      value: d.ifeval,             fmt: "pct" },
      { label: "Tool Calling", value: d.toolcall_accuracy, fmt: "pct" },
      { label: "Speed (~8K)", value: d.gen_tps_8k,        fmt: "tps" },
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

let _sseReconnectTimer = null;

function initSSE() {
  const dot = document.getElementById("sse-indicator");

  function connect() {
    if (_sseSource) _sseSource.close();
    _sseSource = new EventSource("/api/events");

    _sseSource.onopen = () => {
      if (dot) { dot.className = "sse-dot connected"; dot.title = "Live updates connected"; }
      _lastSyncTime = Date.now();
      updateLastSynced();
      // Back on live updates — polling fallback no longer needed
      if (_pollFallbackInterval) { clearInterval(_pollFallbackInterval); _pollFallbackInterval = null; }
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
      if (dot) { dot.className = "sse-dot disconnected"; dot.title = "Live updates disconnected — polling active, retrying"; }
      _sseSource.close();
      _sseSource = null;
      startPollingFallback();
      // Keep trying to restore the live stream instead of polling forever
      clearTimeout(_sseReconnectTimer);
      _sseReconnectTimer = setTimeout(connect, 15_000);
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

  // Group by model_name; each entry is { quantization, hardware }
  const groups = {};
  modelQuants.forEach(({ model_name, quantization, hardware }) => {
    if (!groups[model_name]) groups[model_name] = [];
    groups[model_name].push({ quantization: quantization || "", hardware: hardware || "" });
  });

  Object.entries(groups).forEach(([modelName, entries]) => {
    const childLabel = entry => {
      const hw = entry.hardware ? ` · ${entry.hardware}` : "";
      return (entry.quantization || "unknown") + hw;
    };

    if (entries.length === 1) {
      const e = entries[0];
      const key = modelName + "|" + e.quantization + "|" + e.hardware;
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = true; cb.dataset.key = key;
      cb.addEventListener("change", updateModelFilter);
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + modelName + (e.quantization ? ` (${childLabel(e)})` : "")));
      panel.appendChild(label);
    } else {
      const parentLabel = document.createElement("label");
      parentLabel.className = "mf-parent";
      const parentCb = document.createElement("input");
      parentCb.type = "checkbox"; parentCb.checked = true; parentCb.dataset.parent = modelName;
      parentLabel.appendChild(parentCb);
      parentLabel.appendChild(document.createTextNode(" " + modelName));
      panel.appendChild(parentLabel);

      entries.forEach(e => {
        const key = modelName + "|" + e.quantization + "|" + e.hardware;
        const child = document.createElement("label");
        child.className = "mf-child";
        const childCb = document.createElement("input");
        childCb.type = "checkbox"; childCb.checked = true;
        childCb.dataset.key = key; childCb.dataset.parentModel = modelName;
        childCb.addEventListener("change", () => { _syncParent(modelName); updateModelFilter(); });
        child.appendChild(childCb);
        child.appendChild(document.createTextNode(" " + childLabel(e)));
        panel.appendChild(child);
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
  let data = await fetchAPI("/overview").catch(() => []);
  hideSpinner("chart-radar");
  data = applyGlobalFilters(data);

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
        d.recall_pass_rate   ?? 0,
        d.humaneval_plus     ?? 0,
        d.gsm8k              ?? 0,
        d.ifeval             ?? 0,
        d.toolcall_accuracy  ?? 0,
        maxTps > 0 ? (d.gen_tps_8k ?? 0) / maxTps : 0,
      ],
      theta: ["Long Context<br>Recall", "HumanEval+", "GSM8K", "IFEval", "Tool<br>Calling", "Speed<br>(norm)"],
      fill: "toself",
      name,
      opacity,
      line: { width: lineWidth, color },
      fillcolor: hexToRgba(color, opacity * 0.55),
    };
  });

  const scored = data.map(d => {
    const vals = [d.recall_pass_rate, d.humaneval_plus, d.gsm8k, d.ifeval, d.toolcall_accuracy].filter(v => v != null);
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
  const data = await fetchAPI("/lmeval/leaderboard", { suite: "coding-standard" }).catch(() => []);
  hideSpinner("chart-coding-leaderboard");
  const filtered = applyGlobalFilters(data);
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
  let data = await fetchAPI("/lmeval/leaderboard", { suite: "reasoning" }).catch(() => []);
  hideSpinner("chart-reasoning-leaderboard");
  data = applyGlobalFilters(data);
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
  let allData = await fetchAPI("/recall/leaderboard").catch(() => []);
  hideSpinner("chart-recall-leaderboard");
  allData = applyGlobalFilters(allData);

  // Build corpus filter bar
  const corpora = [...new Set(allData.map(r => r.corpus).filter(Boolean))].sort();
  const filterBar = document.getElementById("recall-corpus-filter");
  if (filterBar) {
    filterBar.innerHTML = '';
    if (corpora.length > 1) {
      [null, ...corpora].forEach(c => {
        const btn = document.createElement('button');
        btn.className = 'corpus-filter-btn' + (c === _recallCorpusFilter ? ' active' : '');
        btn.textContent = c ?? 'All';
        btn.addEventListener('click', () => {
          _recallCorpusFilter = c;
          renderRecallLeaderboard();
        });
        filterBar.appendChild(btn);
      });
    } else {
      _recallCorpusFilter = null;
    }
  }

  let data = _recallCorpusFilter ? allData.filter(r => r.corpus === _recallCorpusFilter) : allData;
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
    customdata: sorted.map(d => JSON.stringify({ config_name: d.config_name, quantization: d.quantization, corpus: d.corpus, hardware: d.hardware || "" })),
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

function _recallKey(r) {
  return JSON.stringify({ config_name: r.config_name, quantization: r.quantization, corpus: r.corpus, hardware: r.hardware || "" });
}

function _recallKeyLabel(key) {
  const { config_name, quantization, corpus, hardware } = JSON.parse(key);
  const row = _recallLeaderboardData.find(r =>
    r.config_name === config_name && r.quantization === quantization &&
    r.corpus === corpus && (r.hardware || "") === (hardware || "")
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
    const { config_name, quantization, corpus, hardware } = JSON.parse(k);
    return fetchAPI("/recall/depth", { config_name, quantization, corpus, hardware }).catch(() => []);
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
        .filter(r => !_recallModels.includes(_recallKey(r)))
        .forEach(r => {
          const opt = document.createElement("option");
          opt.value = _recallKey(r);
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

  // primary_matched is averaged across runs (may be fractional); primary_total
  // comes from the DB rather than assuming the 20-line default.
  const _fmtMatched = d => d ? `${Number.isInteger(d.primary_matched) ? d.primary_matched : d.primary_matched.toFixed(1)}/${d.primary_total ?? 20}` : "";
  const _matchedFrac = d => d ? (d.primary_matched ?? 0) / (d.primary_total || 20) : 0;

  if (_recallModels.length === 1) {
    const mapD = Object.fromEntries(datasets[0].map(d => [d.function_name, d]));
    const color1 = _recallModelColor(_recallKeyLabel(_recallModels[0]));
    Plotly.newPlot("chart-recall-depth", [{
      type: "bar", orientation: "h",
      y: allFuncs,
      x: allFuncs.map(fn => _matchedFrac(mapD[fn])),
      marker: {
        color: allFuncs.map(fn => (mapD[fn]?.pass_rate ?? 0) > 0 ? "#10b981" : hexToRgba(color1, 0.45)),
        line: { color: allFuncs.map(fn => (mapD[fn]?.pass_rate ?? 0) > 0 ? "#10b981" : color1), width: 1.5 },
      },
      text: allFuncs.map(fn => _fmtMatched(mapD[fn])),
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
    x: funcs.map(fn => _matchedFrac(modelMap[fn])),
    name: label,
    marker: {
      color: funcs.map(fn => (modelMap[fn]?.pass_rate ?? 0) > 0 ? color : hexToRgba(color, 0.35)),
      line: { color, width: 1 },
    },
    text: funcs.map(fn => _fmtMatched(modelMap[fn]) || "–"),
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
  let [curves, bar] = await Promise.all([
    fetchAPI("/speed/curves").catch(() => []),
    fetchAPI("/speed/comparison", { context_tokens: 8192 }).catch(() => []),
  ]);
  curves = applyGlobalFilters(curves);
  bar    = applyGlobalFilters(bar);
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
// Tab: Tool Calling
// ─────────────────────────────────────────────

async function renderToolCalling() {
  showSpinner("chart-toolcall-heatmap");
  let data = await fetchAPI("/toolcall/heatmap").catch(() => []);
  data = applyGlobalFilters(data);
  hideSpinner("chart-toolcall-heatmap");

  const container = document.getElementById("chart-toolcall-heatmap");
  // Purge any previous per-model plots so click handlers don't accumulate
  container.querySelectorAll("div").forEach(d => { if (d.data) Plotly.purge(d); });

  if (!data.length) {
    emptyChart("chart-toolcall-heatmap");
    document.getElementById("toolcall-breakdown-panel").classList.add("hidden");
    return;
  }

  container.innerHTML = "";
  container.style.height = "auto";

  // One heatmap per model config — same label format as every other tab
  const groups = new Map();
  data.forEach(d => {
    const label = modelLabel(d);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(d);
  });

  groups.forEach((rows, label) => {
    const toolCounts    = [...new Set(rows.map(r => r.tool_count))].sort((a, b) => a - b);
    const contextSizes  = [...new Set(rows.map(r => r.context_bytes))].sort((a, b) => a - b);
    // Label the padding axis in tokens — the meaningful "does it fit context"
    // unit — since padding now ranges up to ~900K tokens (multi-MB).
    const contextLabels = contextSizes.map(_paddingTokenLabel);
    const toolCountLabels = toolCounts.map(n => `${n} tools`);

    // z matrix: rows = context_bytes, cols = tool_count.
    // `accuracy` counts correct refusals on irrelevance questions as wins —
    // same math as the Overview radar.
    const z = contextSizes.map(cb =>
      toolCounts.map(tc => {
        const cell = rows.find(r => r.tool_count === tc && r.context_bytes === cb);
        return cell ? (cell.accuracy ?? cell.arg_accuracy ?? null) : null;
      })
    );
    const textVals = z.map(row => row.map(v => v !== null ? `${(v * 100).toFixed(1)}%` : "—"));

    const div = document.createElement("div");
    div.style.height = "420px";
    container.appendChild(div);

    Plotly.newPlot(div, [{
      type: "heatmap",
      z,
      x: toolCountLabels,
      y: contextLabels,
      text: textVals,
      texttemplate: "%{text}",
      colorscale: [[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]],
      zmin: 0, zmax: 1,
      showscale: true,
      colorbar: { title: "Accuracy", tickformat: ".0%", tickfont: { color: "#9ca3af" }, titlefont: { color: "#9ca3af" } },
      hovertemplate: "Tools: %{x}<br>Padding: %{y} tok<br>Accuracy: %{text}<extra></extra>",
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: `Tool-Calling Accuracy — ${label}  ·  click a cell for categories`, font: { color: "#fff" } },
      // Force categorical axes so a lone numeric-looking label (e.g. "0")
      // doesn't flip an axis to a continuous scale.
      xaxis: { title: "Tool count", color: "#9ca3af", type: "category" },
      yaxis: { title: "Context padding (est. tokens)", color: "#9ca3af", type: "category", autorange: "reversed" },
      margin: { ...PLOTLY_LAYOUT_BASE.margin, r: 80 },
    }, { responsive: true });

    div.style.cursor = "pointer";
    const modelRow = rows[0];
    div.on("plotly_click", async e => {
      const pt = e.points[0];
      // Resolve the cell by index; fall back to label lookup (never parse the
      // token label, which isn't a plain number).
      const pn = Array.isArray(pt.pointNumber) ? pt.pointNumber
               : Array.isArray(pt.pointIndex)  ? pt.pointIndex : null;
      const colIdx = pn ? pn[1] : toolCountLabels.indexOf(pt.x);
      const rowIdx = pn ? pn[0] : contextLabels.indexOf(pt.y);
      const tc = toolCounts[colIdx];
      const cb = contextSizes[rowIdx];
      if (tc !== undefined && cb !== undefined) {
        await _renderToolcallBreakdown(modelRow, label, tc, cb);
      }
    });
  });
}

async function _renderToolcallBreakdown(modelRow, label, tool_count, context_bytes) {
  const panel = document.getElementById("toolcall-breakdown-panel");
  panel.classList.remove("hidden");
  showSpinner("chart-toolcall-breakdown");

  const cbLabel = context_bytes === 0 ? "no padding" : `~${_paddingTokenLabel(context_bytes)} tok padding`;
  document.getElementById("toolcall-breakdown-title").textContent =
    `Category Breakdown — ${label} · ${tool_count} tools · ${cbLabel}`;

  const params = {
    model_config: modelRow.config_name,
    quantization: modelRow.quantization,
    hardware:     modelRow.hardware,
    tool_count,
    context_bytes,
  };
  const data = await fetchAPI("/toolcall/breakdown", params).catch(() => []);
  hideSpinner("chart-toolcall-breakdown");

  if (!data.length) { emptyChart("chart-toolcall-breakdown"); return; }

  const cats = [...new Set(data.map(r => r.category))];
  const catValue = c => {
    const r = data.find(d => d.category === c);
    if (!r) return null;
    return c === "irrelevance" ? (r.irrelevance_accuracy ?? null) : (r.arg_accuracy ?? null);
  };

  Plotly.newPlot("chart-toolcall-breakdown", [{
    type: "bar",
    name: label,
    x: cats,
    y: cats.map(catValue),
    marker: { color: modelColor(label) },
    text: cats.map(c => {
      const v = catValue(c);
      return v !== null && v !== undefined ? `${(v * 100).toFixed(1)}%` : "";
    }),
    textposition: "outside",
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Accuracy by Category (irrelevance = correct refusals)", font: { color: "#fff" } },
    xaxis: { color: "#9ca3af" },
    yaxis: { title: "Accuracy", tickformat: ".0%", range: [0, 1.15], color: "#9ca3af" },
    showlegend: false,
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab: Run
// ─────────────────────────────────────────────

let _runTabInitialized = false;
let _modelStatusInterval = null;

async function refreshModelStatus(modelName) {
  const row  = document.getElementById('model-status-row');
  const dot  = document.getElementById('model-status-dot');
  const text = document.getElementById('model-status-text');
  const runBtn = document.getElementById('run-btn');
  if (!row || !dot || !text || !runBtn) return;

  let status;
  try {
    status = await fetchAPI(`/config/models/${encodeURIComponent(modelName)}/status`);
  } catch (_) { return; }

  if (!status.is_local) {
    row.style.display = 'none';
    runBtn.disabled = false;
    runBtn.style.background = '';
    runBtn.style.color = '';
    runBtn.style.cursor = '';
    return;
  }

  row.style.display = 'flex';
  if (status.loaded) {
    dot.className = 'model-status-dot loaded';
    const quant = status.quantization ? ` · ${status.quantization}` : '';
    text.textContent = `${status.model_name}${quant}`;
    runBtn.disabled = false;
    runBtn.style.background = '';
    runBtn.style.color = '';
    runBtn.style.cursor = '';
  } else {
    dot.className = 'model-status-dot unloaded';
    text.textContent = 'No model loaded';
    runBtn.disabled = true;
    runBtn.style.background = '#374151';
    runBtn.style.color = '#6b7280';
    runBtn.style.cursor = 'not-allowed';
  }
}

// Corpus size labels — help judge whether a recall corpus fits in the model's
// context window before running it. Sizes come from the backend (byte count +
// token estimate); the token figure is approximate (~3.8 chars/token).
let _corpusInfo = {};  // name → { n_files, bytes, est_tokens }

function _fmtBytes(b) {
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024)    return Math.round(b / 1024) + ' KB';
  return b + ' B';
}
function _fmtTokens(t) {
  if (t >= 1000) return (t / 1000).toFixed(t >= 10000 ? 0 : 1).replace(/\.0$/, '') + 'K';
  return String(t);
}
function _corpusOptionLabel(c) {
  if (!c.bytes) return c.name;
  const files = c.n_files > 1 ? ` · ${c.n_files} files` : '';
  return `${c.name}  —  ~${_fmtTokens(c.est_tokens)} tok · ${_fmtBytes(c.bytes)}${files}`;
}
function _populateCorpusSelect(sel, corpora, keepCurrent) {
  _corpusInfo = {};
  const current = keepCurrent ? sel.value : null;
  sel.innerHTML = '';
  corpora.forEach(c => {
    _corpusInfo[c.name] = c;
    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = _corpusOptionLabel(c);
    sel.appendChild(opt);
  });
  if (current && sel.querySelector(`option[value="${CSS.escape(current)}"]`)) sel.value = current;
  _updateCorpusSizeHint();
}
function _updateCorpusSizeHint() {
  const sel  = document.getElementById('run-corpus');
  const hint = document.getElementById('run-corpus-size');
  if (!sel || !hint) return;
  const c = _corpusInfo[sel.value];
  if (!c || !c.bytes) { hint.textContent = ''; return; }
  const files = c.n_files > 1 ? ` across ${c.n_files} files` : '';
  hint.textContent = `≈ ${_fmtTokens(c.est_tokens)} tokens (${_fmtBytes(c.bytes)}${files}) — must fit the model's context`;
}

// Tool Calling run options — the panel is only meaningful when the toolcall
// suite is selected. Requests = BFCL categories (4) × tool counts × padding
// sizes × questions-per-category.
const _TOOLCALL_CATEGORIES = 4;
const _TOOLCALL_WARN_THRESHOLD = 1500;
// Padding filler is English prose (~4 chars/token), unlike the code corpora.
const _PADDING_CHARS_PER_TOKEN = 4;

// KB → "512 KB" / "3.4 MB"
function _fmtPaddingSize(kb) {
  if (kb < 1024) return `${kb} KB`;
  return `${(kb / 1024).toFixed(1).replace(/\.0$/, '')} MB`;
}
// bytes → estimated-token label ("0", "2K", "900K")
function _paddingTokenLabel(bytes) {
  return bytes === 0 ? "0" : _fmtTokens(Math.round(bytes / _PADDING_CHARS_PER_TOKEN));
}

// Annotate each padding chip with its size + approximate token count (derived
// from data-kb so the labels can't drift from the actual sizes).
function _labelPaddingChips() {
  document.querySelectorAll('#tc-padding .tc-chip').forEach(chip => {
    const kb = parseInt(chip.dataset.kb) || 0;
    if (kb === 0) { chip.textContent = '0 KB'; return; }
    const tok = Math.round(kb * 1024 / _PADDING_CHARS_PER_TOKEN);
    chip.innerHTML = `${_fmtPaddingSize(kb)} <span class="tc-chip-sub">· ~${_fmtTokens(tok)} tok</span>`;
  });
}

function _updateToolcallPanel() {
  const cb = document.querySelector('.suite-toggle input[value="toolcall"]');
  const panel = document.getElementById('toolcall-options');
  if (!panel) return;
  panel.classList.toggle('hidden', !(cb && cb.checked));
  _updateToolcallEstimate();
}

function _updateToolcallEstimate() {
  const est = document.getElementById('tc-estimate');
  if (!est) return;
  const pads  = document.querySelectorAll('#tc-padding .tc-chip.active').length;
  const tcs   = document.querySelectorAll('#tc-toolcounts .tc-chip.active').length;
  const limit = parseInt(document.getElementById('tc-limit')?.value) || 0;
  const total = _TOOLCALL_CATEGORIES * tcs * pads * limit;
  est.textContent =
    `→ ${_TOOLCALL_CATEGORIES} cat × ${tcs} tools × ${pads} pad × ${limit} = ${total.toLocaleString()} requests`;
  const big = total > _TOOLCALL_WARN_THRESHOLD;
  est.classList.toggle('warn', big);
  if (big) est.textContent += '  ⚠ large run — may take hours';
}

async function renderRun() {
  if (!_runTabInitialized) {
    _runTabInitialized = true;

    // Populate model dropdown
    const models = await fetchAPI("/config/models").catch(() => []);
    const modelSel = document.getElementById('run-model');
    if (modelSel) {
      modelSel.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        modelSel.appendChild(opt);
      });
      modelSel.addEventListener('change', () => refreshModelStatus(modelSel.value));
    }

    // Initial status check + 15s poll
    if (modelSel && modelSel.value) refreshModelStatus(modelSel.value);
    if (_modelStatusInterval) clearInterval(_modelStatusInterval);
    _modelStatusInterval = setInterval(() => {
      const sel = document.getElementById('run-model');
      if (sel && sel.value) refreshModelStatus(sel.value);
    }, 15000);

    // Populate corpus dropdown (with size labels + a fit hint)
    const corpora = await fetchAPI("/config/corpora").catch(() => []);
    const corpusSel = document.getElementById('run-corpus');
    if (corpusSel) {
      _populateCorpusSelect(corpusSel, corpora, false);
      corpusSel.addEventListener('change', _updateCorpusSizeHint);
    }

    // Populate architecture datalist
    const archs = await fetchAPI("/config/architectures").catch(() => ({}));
    const datalist = document.getElementById('arch-datalist');
    if (datalist) {
      const unique = [...new Set(Object.values(archs))];
      unique.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        datalist.appendChild(opt);
      });
    }

    // Suite toggle logic
    const allToggle = document.getElementById('run-all-toggle');
    if (allToggle) {
      allToggle.addEventListener('change', e => {
        document.querySelectorAll('.suite-toggle input[type=checkbox]:not(#run-all-toggle)').forEach(cb => {
          cb.checked = e.target.checked;
        });
      });
    }
    const syncAllToggle = () => {
      const all = [...document.querySelectorAll('.suite-toggle input[value]')];
      const checked = all.filter(c => c.checked).length;
      const allToggleEl = document.getElementById('run-all-toggle');
      if (allToggleEl) {
        if (checked === all.length) {
          allToggleEl.checked = true;
          allToggleEl.indeterminate = false;
        } else if (checked === 0) {
          allToggleEl.checked = false;
          allToggleEl.indeterminate = false;
        } else {
          allToggleEl.checked = false;
          allToggleEl.indeterminate = true;
        }
      }
    };
    document.querySelectorAll('.suite-toggle input[value]').forEach(cb => {
      cb.addEventListener('change', () => { syncAllToggle(); _updateToolcallPanel(); });
    });
    if (allToggle) allToggle.addEventListener('change', _updateToolcallPanel);
    syncAllToggle();  // initial state: toolcall unchecked → "All" indeterminate

    // Tool Calling options: reveal padding/tool-count/limit controls when the
    // toolcall suite is selected, with a live request-count estimate.
    document.querySelectorAll('#toolcall-options .tc-chip-group').forEach(group => {
      group.addEventListener('click', e => {
        const chip = e.target.closest('.tc-chip');
        if (!chip) return;
        // Keep at least one chip active per group
        if (chip.classList.contains('active') &&
            group.querySelectorAll('.tc-chip.active').length === 1) return;
        chip.classList.toggle('active');
        _updateToolcallEstimate();
      });
    });
    document.getElementById('tc-limit')?.addEventListener('input', _updateToolcallEstimate);
    _labelPaddingChips();    // annotate padding chips with token counts
    _updateToolcallPanel();  // initial visibility + estimate

    // Run button
    const runBtn = document.getElementById('run-btn');
    if (runBtn) {
      runBtn.addEventListener('click', async () => {
        const model = document.getElementById('run-model').value;
        const corpus = document.getElementById('run-corpus').value;
        const suites = [...document.querySelectorAll('.suite-toggle input[value]')]
          .filter(cb => cb.checked).map(cb => cb.value);
        const quantization = document.getElementById('run-quant').value || null;
        const architecture = document.getElementById('run-arch').value || null;
        if (!model || !suites.length) return;
        const body = { model, corpus, suites, quantization, architecture };
        if (suites.includes('toolcall')) {
          body.context_padding_kb = [...document.querySelectorAll('#tc-padding .tc-chip.active')]
            .map(c => parseInt(c.dataset.kb));
          body.tool_counts = [...document.querySelectorAll('#tc-toolcounts .tc-chip.active')]
            .map(c => parseInt(c.dataset.n));
          body.toolcall_limit = parseInt(document.getElementById('tc-limit').value) || 50;
        }
        try {
          const { run_id } = await fetchAPI("/run", body, "POST");
          const label = `${model} · ${suites.join('+')}`;
          spawnRunCard(run_id, label);
          document.getElementById('run-note').classList.add('hidden');
        } catch (e) {
          console.error('Run failed:', e);
        }
      });
    }

    // Custom corpus file upload
    const fileInput = document.getElementById('custom-file-input');
    if (fileInput) {
      fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const fd = new FormData();
        fd.append('file', file);
        try {
          await fetch('/api/config/corpora/upload', { method: 'POST', body: fd });
        } catch (err) {
          console.error('Upload failed:', err);
        }
        e.target.value = '';
        await _refreshCustomFiles();
        await _refreshCorpusDropdown();
      });
    }
    await _refreshCustomFiles();
  }

  // Check for active runs each time tab is shown — reconnect cards after reload
  try {
    const active = await fetchAPI("/run/active");
    const noteEl = document.getElementById('run-note');
    if (noteEl) {
      const hasRunning = Array.isArray(active) && active.some(r => r.status === 'running');
      if (hasRunning) noteEl.classList.remove('hidden');
      else noteEl.classList.add('hidden');
    }
    if (Array.isArray(active)) {
      active
        .filter(r => r.status === 'running' && !document.getElementById(`run-card-${r.run_id}`))
        .forEach(r => spawnRunCard(r.run_id, r.cmd_summary));
    }
  } catch (_) {}
}

async function _refreshCustomFiles() {
  const list = document.getElementById('custom-files-list');
  if (!list) return;
  const files = await fetchAPI("/config/corpora/custom").catch(() => []);
  list.innerHTML = '';
  if (!files.length) {
    const empty = document.createElement('span');
    empty.className = 'custom-files-empty';
    empty.textContent = 'No custom files — upload a source file to use it as a recall corpus';
    list.appendChild(empty);
    return;
  }
  files.forEach(f => {
    const row = document.createElement('div');
    row.className = 'custom-file-row';
    const size = f.bytes ? `~${_fmtTokens(f.est_tokens)} tok · ${_fmtBytes(f.bytes)}` : '';
    row.innerHTML = `<span class="custom-file-name">${f.filename}</span>` +
      `<span class="custom-file-size">${size}</span>` +
      `<button class="custom-file-delete" title="Remove">✕</button>`;
    row.querySelector('.custom-file-delete').addEventListener('click', async () => {
      await fetchAPI(`/config/corpora/custom/${f.name}`, {}, "DELETE");
      await _refreshCustomFiles();
      await _refreshCorpusDropdown();
    });
    list.appendChild(row);
  });
}

async function _refreshCorpusDropdown() {
  const corpora = await fetchAPI("/config/corpora").catch(() => []);
  const sel = document.getElementById('run-corpus');
  if (!sel) return;
  _populateCorpusSelect(sel, corpora, true);
}

function spawnRunCard(run_id, label) {
  const card = document.createElement('div');
  card.className = 'run-card';
  card.id = `run-card-${run_id}`;
  const startedAt = Date.now();
  card.innerHTML = `
    <div class="run-card-header">
      <span class="run-status-dot running" id="dot-${run_id}"></span>
      <span class="run-card-title">${label}</span>
      <span class="run-card-age" id="age-${run_id}">just now</span>
      <button class="run-cancel-btn" id="cancel-${run_id}">Cancel</button>
      <button class="run-dismiss-btn" id="dismiss-${run_id}" style="display:none">&#x2715;</button>
    </div>
    <pre class="run-log" id="log-${run_id}"></pre>
  `;
  document.getElementById('run-cards').prepend(card);

  // While running only Cancel is shown; ✕ appears once the run has finished
  // and only removes the card — it never cancels anything.
  const showDismiss = () => {
    const d = document.getElementById(`dismiss-${run_id}`);
    if (d) d.style.display = '';
  };

  // Age timer
  const ageInterval = setInterval(() => {
    const el = document.getElementById(`age-${run_id}`);
    if (!el) { clearInterval(ageInterval); return; }
    const s = Math.floor((Date.now() - startedAt) / 1000);
    el.textContent = s < 60 ? `${s}s ago` : `${Math.floor(s/60)}m ago`;
  }, 5000);

  // SSE log stream
  const src = new EventSource(`/api/run/${run_id}/logs`);
  _runSources[run_id] = src;
  src.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'log') {
      const log = document.getElementById(`log-${run_id}`);
      if (log) { log.textContent += data.line + '\n'; log.scrollTop = log.scrollHeight; }
    } else if (data.type === 'done') {
      src.close(); delete _runSources[run_id];
      const dot = document.getElementById(`dot-${run_id}`);
      if (dot) { dot.className = 'run-status-dot done'; }
      const cancelBtn = document.getElementById(`cancel-${run_id}`);
      if (cancelBtn) cancelBtn.style.display = 'none';
      const ageEl = document.getElementById(`age-${run_id}`);
      if (ageEl) ageEl.textContent = 'Done';
      showDismiss();
      clearInterval(ageInterval);
    }
  };
  src.onerror = () => {
    const dot = document.getElementById(`dot-${run_id}`);
    if (dot && dot.classList.contains('running')) dot.className = 'run-status-dot error';
    src.close(); delete _runSources[run_id];
    showDismiss();
  };

  // Cancel button — the only way to stop a run
  document.getElementById(`cancel-${run_id}`).addEventListener('click', async () => {
    await fetchAPI(`/run/${run_id}`, {}, "DELETE");
    if (_runSources[run_id]) { _runSources[run_id].close(); delete _runSources[run_id]; }
    const dot = document.getElementById(`dot-${run_id}`);
    if (dot) dot.className = 'run-status-dot cancelled';
    document.getElementById(`cancel-${run_id}`).style.display = 'none';
    showDismiss();
    clearInterval(ageInterval);
  });

  // Dismiss button — removes the card only, never touches the run
  document.getElementById(`dismiss-${run_id}`).addEventListener('click', () => {
    if (_runSources[run_id]) { _runSources[run_id].close(); delete _runSources[run_id]; }
    clearInterval(ageInterval);
    document.getElementById(`run-card-${run_id}`)?.remove();
  });
}

// ─────────────────────────────────────────────
// Tab: Models
// ─────────────────────────────────────────────

async function renderModels() {
  const models = await fetchAPI("/config/models").catch(() => []);
  const list = document.getElementById('model-list');
  list.innerHTML = '';
  models.forEach(m => {
    const item = document.createElement('div');
    item.className = 'model-list-item';
    item.dataset.name = m.name;
    item.textContent = m.name;
    item.addEventListener('click', () => {
      document.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
      _modelsActiveModel = m.name;
      openModelEditor(m.name);
    });
    list.appendChild(item);
  });
  document.getElementById('model-new-btn').onclick = () => {
    document.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('active'));
    _modelsActiveModel = null;
    openModelEditor(null);
  };
  // Restore previously selected model
  if (_modelsActiveModel) {
    const match = list.querySelector(`.model-list-item[data-name="${_modelsActiveModel}"]`);
    if (match) {
      match.classList.add('active');
      openModelEditor(_modelsActiveModel);
    }
  }
}

async function openModelEditor(name) {
  const editor = document.getElementById('model-editor');
  const data = name ? await fetchAPI(`/config/models/${name}`).catch(() => ({})) : {};
  const isNew = !name;
  let showingAdvanced = false;

  const FIELDS = [
    'base_url','api_key','api_key_file','api_key_env','runtime',
    'temperature','max_tokens','timeout','quantization',
    'architecture','hardware','lmeval_tokenizer','suppress_thinking',
    'prefill_no_think','relax_indent','runs_per_function',
    'stream_for_ttft','reasoning_effort','use_max_completion_tokens'
  ];
  const BOOL_FIELDS = new Set(['suppress_thinking','prefill_no_think','relax_indent',
                               'stream_for_ttft','use_max_completion_tokens']);

  const isLocal = name === 'local';
  const canRename = !isNew && !isLocal;

  function renderForm() {
    editor.innerHTML = `
      <div class="model-editor-title">${isNew ? 'New Model Config' : `Editing: ${name}`}</div>
      ${isNew ? '<div class="model-field"><label>Config name</label><input id="mf-name" type="text" placeholder="e.g. my-model"></div>' : ''}
      ${canRename ? '<div class="model-field"><label>Config name</label><input id="mf-rename" type="text" value="' + name + '"></div>' : ''}
      <div class="model-editor-grid">
        ${FIELDS.map(f => `
          <div class="model-field">
            <label>${f}</label>
            ${BOOL_FIELDS.has(f)
              ? `<select id="mf-${f}"><option value="">—</option><option value="true" ${data[f]===true?'selected':''}>true</option><option value="false" ${data[f]===false?'selected':''}>false</option></select>`
              : `<input id="mf-${f}" type="text" value="${data[f] ?? ''}" placeholder="${f}">`
            }
          </div>`).join('')}
      </div>
      <div class="model-editor-actions">
        <button id="me-save" class="run-btn">Save</button>
        <button id="me-advanced" class="detail-btn">Advanced (Raw TOML)</button>
        ${canRename ? '<button id="me-delete" class="detail-btn me-delete-btn">Delete</button>' : ''}
        <button id="me-cancel" class="detail-btn">Cancel</button>
      </div>
    `;
    document.getElementById('me-cancel').onclick = () => {
      editor.innerHTML = '<span class="empty-state">Select a config to edit</span>';
      document.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('active'));
    };
    document.getElementById('me-advanced').onclick = () => {
      showingAdvanced = !showingAdvanced;
      if (showingAdvanced) renderAdvanced();
      else renderForm();
    };
    document.getElementById('me-save').onclick = () => saveModel(false);
    if (canRename) {
      document.getElementById('me-delete').onclick = () => deleteModel();
    }
  }

  function renderAdvanced() {
    // Existing configs: show the actual file contents (comments included).
    // New configs: build a starter skeleton from any structured values.
    const tomlText = data._raw_toml ?? FIELDS.map(f => {
      const val = data[f];
      if (val === null || val === undefined || val === '') return null;
      if (typeof val === 'boolean') return `${f} = ${val}`;
      if (typeof val === 'number') return `${f} = ${val}`;
      return `${f} = "${val}"`;
    }).filter(Boolean).join('\n');
    editor.innerHTML = `
      <div class="model-editor-title">${isNew ? 'New Model Config' : `Editing: ${name}`} — Raw TOML</div>
      ${isNew ? `<div class="model-field"><label>Config name</label><input id="mf-name" type="text" placeholder="e.g. my-model" value="${data.name||''}"></div>` : ''}
      <textarea id="mf-toml" class="model-toml-textarea" rows="18">${tomlText}</textarea>
      <div class="model-editor-actions">
        <button id="me-save" class="run-btn">Save</button>
        <button id="me-advanced" class="detail-btn">Structured Form</button>
        <button id="me-cancel" class="detail-btn">Cancel</button>
      </div>
    `;
    document.getElementById('me-cancel').onclick = () => {
      editor.innerHTML = '<span class="empty-state">Select a config to edit</span>';
      document.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('active'));
    };
    document.getElementById('me-advanced').onclick = () => { showingAdvanced = false; renderForm(); };
    document.getElementById('me-save').onclick = () => saveModel(true);
  }

  async function saveModel(fromAdvanced) {
    let payload = {};
    const newName = isNew
      ? (document.getElementById('mf-name')?.value.trim())
      : canRename ? (document.getElementById('mf-rename')?.value.trim() || name) : name;
    if (!newName) { alert('Config name required'); return; }
    if (fromAdvanced) {
      payload = { _raw_toml: document.getElementById('mf-toml').value };
    } else {
      // Merge-on-save: blank field = delete that key; fields absent from the
      // form (name, stop, ...) are preserved by the backend.
      FIELDS.forEach(f => {
        const el = document.getElementById(`mf-${f}`);
        if (!el) return;
        const v = el.value.trim();
        if (v === '') { payload[f] = null; return; }
        if (BOOL_FIELDS.has(f)) payload[f] = v === 'true';
        else if (!isNaN(v)) payload[f] = Number(v);
        else payload[f] = v;
      });
    }
    try {
      if (isNew) {
        payload.name = newName;
        await fetchAPI("/config/models", payload, "POST");
      } else {
        if (canRename && newName !== name) {
          payload._rename = newName;   // backend renames the file, keeping every field
          _modelsActiveModel = newName;
        }
        await fetchAPI(`/config/models/${encodeURIComponent(name)}`, payload, "PUT");
      }
      await renderModels();
      editor.innerHTML = '<span class="empty-state">Saved!</span>';
    } catch (e) {
      alert('Save failed: ' + e.message);
    }
  }

  async function deleteModel() {
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;
    try {
      await fetchAPI(`/config/models/${name}`, {}, "DELETE");
      _modelsActiveModel = null;
      editor.innerHTML = '<span class="empty-state">Deleted.</span>';
      await renderModels();
    } catch (e) {
      alert('Delete failed: ' + e.message);
    }
  }

  renderForm();
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
  fill("filter-arch",    opts.architectures);

  buildModelFilterPanel(opts.model_quants || []);
}

function renderCurrentTab() {
  const renderers = {
    overview:  renderOverview,
    coding:    renderCoding,
    reasoning: renderReasoning,
    recall:    renderRecallLeaderboard,
    toolcall:  renderToolCalling,
    speed:     renderSpeed,
    run:       renderRun,
    models:    renderModels,
    data:      renderData,
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

  document.getElementById("toolcall-breakdown-close").addEventListener("click", () => {
    document.getElementById("toolcall-breakdown-panel").classList.add("hidden");
    Plotly.purge(document.getElementById("chart-toolcall-breakdown"));
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

// ─────────────────────────────────────────────
// Tab: Data
// ─────────────────────────────────────────────

let _dataActiveModel   = null;
let _dataRuns          = [];
let _dataSelected      = new Set();
let _dataLastIdx       = -1;
let _dataDetailKey     = null;
let _dataSubtype       = "all";
let _dataInitialized   = false;

const _DATA_SECTION_LABELS = { recall: "RECALL", coding: "CODING", reasoning: "REASONING", speed: "SPEED", toolcall: "TOOL CALLING" };

function _runGroup(r) {
  if (r.type === "recall")   return "recall";
  if (r.type === "speed")    return "speed";
  if (r.type === "toolcall") return "toolcall";
  if (r.type === "lmeval") {
    return (r.task_suite && r.task_suite.startsWith("coding")) ? "coding" : "reasoning";
  }
  return r.type;
}

function _fmtDate(iso) {
  const d = new Date(iso);
  const y  = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dy = String(d.getDate()).padStart(2, "0");
  const h  = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${y}-${mo}-${dy} ${h}:${mi}`;
}

async function renderData() {
  if (!_dataInitialized) {
    _dataInitialized = true;
    document.querySelectorAll(".data-subtab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".data-subtab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        _dataSubtype = btn.dataset.type;
        _dataDetailKey = null;
        _dataSelected.clear();
        _renderRunList();
        _updateDataToolbar();
      });
    });
    document.getElementById("data-edit-btn").addEventListener("click", _openDataEditModal);
    document.getElementById("data-delete-btn").addEventListener("click", _deleteDataSelected);
    document.getElementById("data-edit-save").addEventListener("click", _saveDataEdit);
    document.getElementById("data-edit-cancel").addEventListener("click", () => {
      document.getElementById("data-edit-modal").classList.add("hidden");
    });
    document.getElementById("data-run-list").addEventListener("click", e => {
      const row = e.target.closest(".data-run-row");
      if (!row) return;
      _onRunClick(row.dataset.key, parseInt(row.dataset.idx), e.shiftKey);
    });
  }

  const opts = await fetchAPI("/filter-options").catch(() => ({}));
  const modelQuants = opts.model_quants || [];
  _renderDataSidebar(modelQuants);

  if (_dataActiveModel) {
    const still = modelQuants.find(m =>
      m.model_name === _dataActiveModel.model_name &&
      m.quantization === _dataActiveModel.quantization &&
      m.hardware === _dataActiveModel.hardware
    );
    if (still) {
      await _loadRuns();
    } else {
      _dataActiveModel = null;
      document.getElementById("data-empty-state").classList.remove("hidden");
      document.getElementById("data-content").classList.add("hidden");
    }
  }
}

function _renderDataSidebar(modelQuants) {
  const list = document.getElementById("data-model-list");
  list.innerHTML = "";
  modelQuants.forEach(m => {
    const item = document.createElement("div");
    item.className = "data-model-item";
    if (
      _dataActiveModel &&
      _dataActiveModel.model_name === m.model_name &&
      _dataActiveModel.quantization === m.quantization &&
      _dataActiveModel.hardware === m.hardware
    ) item.classList.add("active");

    const nameSpan = document.createElement("span");
    nameSpan.className = "data-model-name";
    let label = `${m.model_name} · ${m.quantization}`;
    if (m.hardware) label += ` · ${m.hardware}`;
    nameSpan.textContent = label;

    const countSpan = document.createElement("span");
    countSpan.className = "data-model-count";
    countSpan.textContent = `${m.run_count} run${m.run_count !== 1 ? "s" : ""}`;

    item.appendChild(nameSpan);
    item.appendChild(countSpan);
    item.addEventListener("click", () => {
      _dataActiveModel = { model_name: m.model_name, quantization: m.quantization, hardware: m.hardware };
      document.querySelectorAll(".data-model-item").forEach(el => el.classList.remove("active"));
      item.classList.add("active");
      _loadRuns();
    });
    list.appendChild(item);
  });
}

async function _loadRuns() {
  document.getElementById("data-content").classList.remove("hidden");
  document.getElementById("data-empty-state").classList.add("hidden");
  const m = _dataActiveModel;
  const runs = await fetchAPI("/runs", {
    model_name:   m.model_name,
    quantization: m.quantization,
    hardware:     m.hardware,
    page_size:    200,
  }).catch(() => []);
  _dataRuns = runs;
  _dataSelected.clear();
  _dataDetailKey = null;
  _renderRunList();
}

function _renderRunList() {
  const container = document.getElementById("data-run-list");
  container.innerHTML = "";

  const filtered = _dataSubtype === "all"
    ? _dataRuns
    : _dataRuns.filter(r => _runGroup(r) === _dataSubtype);

  // Rows are indexed in VISUAL order (the "All" view groups by section, which
  // reorders them) so shift-click selects the rows you actually see between
  // the two clicks.
  let visualIdx = 0;

  function makeRow(r, flatIdx) {
    const key = `${r.type}:${r.run_id}`;
    const row = document.createElement("div");
    row.className = "data-run-row" + (_dataSelected.has(key) ? " selected" : "");
    row.dataset.key = key;
    row.dataset.idx = flatIdx;

    const check = document.createElement("span");
    check.className = "data-check-mark";
    check.textContent = _dataSelected.has(key) ? "✓" : "";

    const dateEl = document.createElement("span");
    dateEl.className = "data-run-date";
    dateEl.textContent = _fmtDate(r.created_at);

    const typeEl = document.createElement("span");
    typeEl.className = "data-run-type";
    const badge = document.createElement("span");
    badge.className = `data-type-badge ${r.type}`;
    badge.textContent = r.type;
    const detail = document.createElement("span");
    detail.className = "data-run-detail-str";
    if (r.type === "recall") detail.textContent = ` · ${r.corpus} · ${r.n_runs} run${r.n_runs !== 1 ? "s" : ""}`;
    else if (r.type === "lmeval") detail.textContent = ` · ${r.task_suite}`;
    else if (r.type === "toolcall") detail.textContent = ` · ${r.task_suite || "bfcl-v4"}`;
    typeEl.appendChild(badge);
    typeEl.appendChild(detail);

    row.appendChild(check);
    row.appendChild(dateEl);
    row.appendChild(typeEl);
    return row;
  }

  if (_dataSubtype === "all") {
    ["recall", "coding", "reasoning", "speed", "toolcall"].forEach(type => {
      const group = filtered.filter(r => _runGroup(r) === type);
      if (!group.length) return;
      const block = document.createElement("div");
      block.className = "data-section-block";
      const title = document.createElement("div");
      title.className = "data-section-title";
      title.textContent = _DATA_SECTION_LABELS[type];
      block.appendChild(title);
      group.forEach(r => block.appendChild(makeRow(r, visualIdx++)));
      container.appendChild(block);
    });
  } else {
    const block = document.createElement("div");
    block.className = "data-section-block";
    filtered.forEach((r, i) => block.appendChild(makeRow(r, i)));
    container.appendChild(block);
  }

  _updateDataToolbar();

  if (_dataSelected.size === 1 && _dataDetailKey) {
    setTimeout(() => _openDetail(_dataDetailKey), 0);
  }
}

function _onRunClick(key, idx, shiftKey) {
  if (shiftKey && _dataLastIdx !== -1) {
    const rows = document.querySelectorAll(".data-run-row");
    const lo = Math.min(_dataLastIdx, idx);
    const hi = Math.max(_dataLastIdx, idx);
    rows.forEach(row => {
      if (parseInt(row.dataset.idx) >= lo && parseInt(row.dataset.idx) <= hi) {
        _dataSelected.add(row.dataset.key);
      }
    });
    if (_dataSelected.size > 1) _dataDetailKey = null;
  } else {
    if (_dataSelected.size === 1 && _dataSelected.has(key)) {
      _dataSelected.clear();
      _dataDetailKey = null;
    } else {
      _dataSelected.clear();
      _dataSelected.add(key);
      _dataDetailKey = key;
    }
    _dataLastIdx = idx;
  }
  _renderRunList();
  _updateDataToolbar();
}

async function _openDetail(key) {
  document.querySelectorAll(".data-detail-row").forEach(el => el.remove());

  const colonIdx = key.indexOf(":");
  const type   = key.slice(0, colonIdx);
  const run_id = key.slice(colonIdx + 1);

  const anchor = document.querySelector(`.data-run-row[data-key="${key}"]`);
  if (!anchor) return;

  const detailDiv = document.createElement("div");
  detailDiv.className = "data-detail-row";
  const inner = document.createElement("div");
  inner.className = "data-detail-inner";
  inner.innerHTML = `<span style="color:#6b7280;font-size:0.78rem">Loading…</span>`;
  detailDiv.appendChild(inner);
  anchor.insertAdjacentElement("afterend", detailDiv);
  requestAnimationFrame(() => detailDiv.classList.add("open"));

  try {
    const data = await fetchAPI(`/runs/${type}/${run_id}`);
    _renderDetailContent(inner, type, data);
  } catch (_) {
    inner.innerHTML = `<span style="color:#ef4444;font-size:0.78rem">Failed to load detail.</span>`;
  }
}

function _renderDetailContent(el, type, rows) {
  if (!rows || !rows.length) {
    el.innerHTML = `<span style="color:#6b7280;font-size:0.78rem">No detail data</span>`;
    return;
  }
  let html = `<table class="data-detail-table">`;
  if (type === "recall") {
    html += `<thead><tr><th>Function</th><th>Pass Rate</th><th>Matched</th><th>Latency</th></tr></thead><tbody>`;
    rows.forEach(r => {
      const pr  = r.pass_rate != null ? `${(r.pass_rate * 100).toFixed(0)}%` : "—";
      const mat = `${r.primary_matched ?? "?"}/${r.primary_total ?? "?"}`;
      const lat = r.latency_mean_s != null ? `${r.latency_mean_s.toFixed(1)}s` : "—";
      html += `<tr><td>${r.function_name}</td><td>${pr}</td><td>${mat}</td><td>${lat}</td></tr>`;
    });
  } else if (type === "lmeval") {
    html += `<thead><tr><th>Task</th><th>Metric</th><th>Value</th></tr></thead><tbody>`;
    rows.forEach(r => {
      const val = `${(r.value * 100).toFixed(1)}%`;
      html += `<tr><td>${r.task}</td><td>${r.metric}</td><td>${val}</td></tr>`;
    });
  } else if (type === "speed") {
    html += `<thead><tr><th>Context</th><th>Gen TPS</th><th>Overall TPS</th><th>TTFT</th><th>N</th></tr></thead><tbody>`;
    rows.forEach(r => {
      const gen     = r.generation_tps_mean != null ? r.generation_tps_mean.toFixed(1) : "—";
      const overall = r.overall_tps_mean    != null ? r.overall_tps_mean.toFixed(1)    : "—";
      const ttft    = r.ttft_mean_s         != null ? `${r.ttft_mean_s.toFixed(2)}s`   : "—";
      html += `<tr><td>${r.context_tokens}</td><td>${gen}</td><td>${overall}</td><td>${ttft}</td><td>${r.n_samples}</td></tr>`;
    });
  } else if (type === "toolcall") {
    html += `<thead><tr><th>Category</th><th>Tools</th><th>N</th><th>Tool Acc</th><th>Arg Acc</th><th>Irr Acc</th></tr></thead><tbody>`;
    rows.forEach(r => {
      const tool = r.tool_accuracy        != null ? `${(r.tool_accuracy * 100).toFixed(0)}%`        : "—";
      const arg  = r.arg_accuracy         != null ? `${(r.arg_accuracy  * 100).toFixed(0)}%`        : "—";
      const irr  = r.irrelevance_accuracy != null ? `${(r.irrelevance_accuracy * 100).toFixed(0)}%` : "—";
      html += `<tr><td>${r.category}</td><td>${r.tool_count}</td><td>${r.n_questions}</td><td>${tool}</td><td>${arg}</td><td>${irr}</td></tr>`;
    });
  }
  html += `</tbody></table>`;
  el.innerHTML = html;
}

function _updateDataToolbar() {
  const n = _dataSelected.size;
  document.getElementById("data-count").textContent = n > 0 ? `${n} selected` : "";
  document.getElementById("data-edit-btn").disabled   = n === 0;
  document.getElementById("data-delete-btn").disabled = n === 0;
}

function _openDataEditModal() {
  const firstKey = [..._dataSelected][0];
  if (!firstKey) return;
  const colonIdx = firstKey.indexOf(":");
  const type   = firstKey.slice(0, colonIdx);
  const run_id = firstKey.slice(colonIdx + 1);
  const run = _dataRuns.find(r => r.type === type && r.run_id === run_id);
  if (!run) return;
  const n = _dataSelected.size;
  document.getElementById("data-edit-subtitle").textContent = `Editing ${n} run${n !== 1 ? "s" : ""}`;
  document.getElementById("data-edit-quant").value = run.quantization || "";
  document.getElementById("data-edit-hw").value    = run.hardware    || "";
  document.getElementById("data-edit-modal").classList.remove("hidden");
}

async function _saveDataEdit() {
  const runs = [..._dataSelected].map(k => {
    const colonIdx = k.indexOf(":");
    return { type: k.slice(0, colonIdx), run_id: k.slice(colonIdx + 1) };
  });
  // Blank = leave unchanged, for both fields (previously a blank hardware
  // silently cleared it on every selected run)
  const quant = document.getElementById("data-edit-quant").value.trim();
  const hw    = document.getElementById("data-edit-hw").value.trim();
  const body  = { runs };
  if (quant) body.quantization = quant;
  if (hw)    body.hardware = hw;

  try {
    const result = await fetchAPI("/runs/bulk-meta", body, "PATCH");
    document.getElementById("data-edit-modal").classList.add("hidden");
    _dataSelected.clear();
    const modelQuants = result.model_quants || [];
    if (_dataActiveModel) {
      const newQuant = quant || _dataActiveModel.quantization;
      const newHw    = hw    || _dataActiveModel.hardware;
      const still = modelQuants.find(m =>
        m.model_name === _dataActiveModel.model_name &&
        m.quantization === newQuant &&
        m.hardware === newHw
      );
      _dataActiveModel = still
        ? { model_name: still.model_name, quantization: still.quantization, hardware: still.hardware }
        : null;
    }
    _renderDataSidebar(modelQuants);
    if (_dataActiveModel) {
      await _loadRuns();
    } else {
      document.getElementById("data-empty-state").classList.remove("hidden");
      document.getElementById("data-content").classList.add("hidden");
    }
  } catch (e) {
    alert(`Save failed: ${e.message || e}`);
  }
}

async function _deleteDataSelected() {
  const n = _dataSelected.size;
  if (!confirm(`Delete ${n} selected run${n !== 1 ? "s" : ""}? This cannot be undone.`)) return;
  const runs = [..._dataSelected].map(k => {
    const colonIdx = k.indexOf(":");
    return { type: k.slice(0, colonIdx), run_id: k.slice(colonIdx + 1) };
  });
  try {
    await fetchAPI("/runs/bulk", { runs }, "DELETE");
    _dataSelected.clear();
    _dataDetailKey = null;
    const opts = await fetchAPI("/filter-options").catch(() => ({}));
    const modelQuants = opts.model_quants || [];
    _renderDataSidebar(modelQuants);
    if (_dataActiveModel) {
      const still = modelQuants.find(m =>
        m.model_name === _dataActiveModel.model_name &&
        m.quantization === _dataActiveModel.quantization &&
        m.hardware === _dataActiveModel.hardware
      );
      if (still) {
        await _loadRuns();
      } else {
        _dataActiveModel = null;
        document.getElementById("data-empty-state").classList.remove("hidden");
        document.getElementById("data-content").classList.add("hidden");
      }
    }
  } catch (e) {
    alert(`Delete failed: ${e.message || e}`);
  }
}
