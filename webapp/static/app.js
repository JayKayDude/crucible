// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "#1a1d27",
  plot_bgcolor:  "#1a1d27",
  font: { color: "#e0e0e0", size: 11 },
  margin: { t: 40, l: 60, r: 20, b: 60 },
};

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
  if (!el) { console.warn(`emptyChart: element #${divId} not found`); return; }
  el.innerHTML = `<div class="empty-state">${msg}</div>`;
}

function modelLabel(row) {
  const name = row.model_name || row.config_name;
  return row.quantization && !name.toLowerCase().includes(row.quantization.toLowerCase())
    ? `${name} (${row.quantization})`
    : name;
}

// ─────────────────────────────────────────────
// Tab 1: Overview — Radar chart
// ─────────────────────────────────────────────

async function renderOverview() {
  const f = getFilters();
  const data = await fetchAPI("/overview", f).catch(() => []);
  if (!data.length) return emptyChart("chart-radar");

  // Normalize gen_tps to 0–1 across all models
  const tpsList = data.map(d => d.gen_tps_8k).filter(v => v != null);
  const maxTps = tpsList.length ? Math.max(...tpsList) : 1;

  const traces = data.map(d => ({
    type: "scatterpolar",
    r: [
      d.recall_pass_rate   ?? 0,
      d.humaneval_plus     ?? 0,
      d.gsm8k              ?? 0,
      d.ifeval             ?? 0,
      maxTps > 0 ? (d.gen_tps_8k ?? 0) / maxTps : 0,
    ],
    theta: ["Long Context\nRecall", "HumanEval+", "GSM8K", "IFEval", "Speed\n(norm)"],
    fill: "toself",
    name: modelLabel(d),
    opacity: 0.8,
  }));

  Plotly.react("chart-radar", traces, {
    ...PLOTLY_LAYOUT_BASE,
    polar: { radialaxis: { range: [0, 1], color: "#4b5563" }, bgcolor: "#1a1d27" },
    title: { text: "Model Overview", font: { color: "#fff" } },
    showlegend: true,
    legend: { font: { color: "#e0e0e0" } },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 2: Coding — HumanEval+ / MBPP+
// ─────────────────────────────────────────────

async function renderCoding() {
  const f = getFilters();
  const data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "coding-standard" })
    .catch(() => []);
  if (!data.length) return emptyChart("chart-coding-leaderboard");

  // Filter to pass@1 metrics only (HumanEval+ uses "pass@1", MBPP+ uses "pass_at_1")
  const pass1 = data.filter(d => d.metric && (d.metric.includes("pass@1") || d.metric.includes("pass_at_1")));
  const tasks  = [...new Set(pass1.map(d => d.task))];
  const models = [...new Set(pass1.map(d => modelLabel(d)))];

  const traces = tasks.map(task => ({
    type: "bar", orientation: "h",
    x: models.map(m => {
      const r = pass1.find(d => modelLabel(d) === m && d.task === task);
      return r ? r.value : 0;
    }),
    y: models,
    name: task,
    error_x: {
      type: "data",
      array: models.map(m => {
        const r = pass1.find(d => modelLabel(d) === m && d.task === task);
        return r?.stderr ?? 0;
      }),
    },
  }));

  Plotly.react("chart-coding-leaderboard", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "Python Coding — EvalPlus (pass@1)", font: { color: "#fff" } },
    xaxis: { title: "pass@1", range: [0, 1], color: "#9ca3af" },
    yaxis: { color: "#9ca3af", automargin: true },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 3: Multi-Language Heatmap
// ─────────────────────────────────────────────

async function renderMultilang() {
  const f = getFilters();
  const data = await fetchAPI("/lmeval/multilang", f).catch(() => []);
  if (!data.length) return emptyChart("chart-multilang-heatmap");

  const langs  = ["js", "ts", "java", "cpp", "rs", "go", "py"];
  const models = data.map(d => modelLabel(d));
  const z      = data.map(d => langs.map(l => d[l] ?? null));
  const text   = z.map(row => row.map(v => v != null ? (v * 100).toFixed(1) + "%" : "N/A"));

  Plotly.react("chart-multilang-heatmap", [{
    type: "heatmap", z, x: langs, y: models,
    colorscale: "RdYlGn", zmin: 0, zmax: 1,
    text, texttemplate: "%{text}",
    colorbar: { tickcolor: "#9ca3af", tickfont: { color: "#e0e0e0" } },
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Multi-Language Coding — MultiPL-E (pass@1)", font: { color: "#fff" } },
    xaxis: { title: "Language", color: "#9ca3af" },
    yaxis: { color: "#9ca3af", automargin: true },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 4: Reasoning
// ─────────────────────────────────────────────

async function renderReasoning() {
  const f = getFilters();
  const data = await fetchAPI("/lmeval/leaderboard", { ...f, suite: "reasoning" })
    .catch(() => []);
  if (!data.length) return emptyChart("chart-reasoning-leaderboard");

  // Pick one primary metric per task to avoid cluttering the chart with duplicates.
  // IFEval: prompt_level_strict_acc. Others: flexible-extract exact_match.
  function primaryMetric(task, metric) {
    if (task === "ifeval") return metric === "prompt_level_strict_acc,none";
    return metric && metric.includes("flexible");
  }
  const filtered = data.filter(d => primaryMetric(d.task, d.metric));

  const tasks  = [...new Set(filtered.map(d => d.task))];
  const models = [...new Set(filtered.map(d => modelLabel(d)))];

  const traces = tasks.map(task => ({
    type: "bar", orientation: "h",
    x: models.map(m => {
      const r = filtered.find(d => modelLabel(d) === m && d.task === task);
      return r ? r.value : 0;
    }),
    y: models,
    name: task,
  }));

  Plotly.react("chart-reasoning-leaderboard", traces, {
    ...PLOTLY_LAYOUT_BASE,
    barmode: "group",
    title: { text: "Reasoning Benchmarks", font: { color: "#fff" } },
    xaxis: { title: "Score", color: "#9ca3af" },
    yaxis: { color: "#9ca3af", automargin: true },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 5: Long Context Recall
// ─────────────────────────────────────────────

async function renderRecallLeaderboard() {
  const f = getFilters();
  const data = await fetchAPI("/recall/leaderboard", f).catch(() => []);
  if (!data.length) return emptyChart("chart-recall-leaderboard");

  // Populate depth run selector
  const sel = document.getElementById("filter-depth-run");
  sel.innerHTML = '<option value="">select a run</option>';
  data.forEach(r => {
    const opt = document.createElement("option");
    opt.value = r.run_id;
    opt.textContent = `${modelLabel(r)} — ${r.corpus}`;
    sel.appendChild(opt);
  });

  Plotly.react("chart-recall-leaderboard", [{
    type: "bar", orientation: "h",
    x: data.map(d => d.pass_rate ?? 0),
    y: data.map(d => `${modelLabel(d)} | ${d.corpus}`),
    text: data.map(d => `${((d.pass_rate ?? 0) * 100).toFixed(0)}%`),
    textposition: "outside",
    marker: { color: "#6366f1" },
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Long Context Recall — Codeneedle", font: { color: "#fff" } },
    xaxis: { title: "Pass rate", range: [0, 1.1], color: "#9ca3af" },
    yaxis: { color: "#9ca3af", automargin: true },
  }, { responsive: true });
}

async function renderRecallDepth(runId) {
  if (!runId) return;
  const data = await fetchAPI("/recall/depth", { run_id: runId }).catch(() => []);
  if (!data.length) return emptyChart("chart-recall-depth", "No depth data for this run");

  const maxLine = Math.max(...data.map(d => d.start_line || 0)) || 1;

  Plotly.react("chart-recall-depth", [{
    type: "scatter", mode: "markers+lines",
    x: data.map(d => (d.start_line || 0) / maxLine),
    y: data.map(d => (d.primary_matched || 0) / 20),
    text: data.map(d => d.function_name),
    hovertemplate: "<b>%{text}</b><br>depth: %{x:.0%}<br>matched: %{y:.0%}<extra></extra>",
    line: { color: "#6366f1" },
    marker: { color: "#818cf8", size: 8 },
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Recall Accuracy vs Context Depth", font: { color: "#fff" } },
    xaxis: { title: "Depth in file (0=start, 1=end)", range: [0, 1], color: "#9ca3af" },
    yaxis: { title: "Lines matched / 20", range: [0, 1.05], color: "#9ca3af" },
  }, { responsive: true });
}

// ─────────────────────────────────────────────
// Tab 6: Speed
// ─────────────────────────────────────────────

async function renderSpeed() {
  const f = getFilters();
  const curves = await fetchAPI("/speed/curves", f).catch(() => []);
  const bar    = await fetchAPI("/speed/comparison", { ...f, context_tokens: 8192 }).catch(() => []);

  if (!curves.length) {
    emptyChart("chart-speed-curves");
    emptyChart("chart-speed-bar");
    return;
  }

  // Speed curves: one trace per model
  const models = [...new Set(curves.map(d => modelLabel(d)))];
  const curveTraces = models.map(m => {
    const rows = curves.filter(d => modelLabel(d) === m)
                       .sort((a, b) => a.context_tokens - b.context_tokens);
    return {
      type: "scatter", mode: "lines+markers",
      x: rows.map(r => r.context_tokens),
      y: rows.map(r => r.generation_tps_mean ?? r.overall_tps_mean ?? null),
      name: m,
      connectgaps: false,
    };
  });

  Plotly.react("chart-speed-curves", curveTraces, {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: "Generation Speed vs Context Size", font: { color: "#fff" } },
    xaxis: { title: "Context tokens", type: "log", color: "#9ca3af" },
    yaxis: { title: "tokens/sec", color: "#9ca3af" },
  }, { responsive: true });

  // Speed bar at ~8K
  if (bar.length) {
    Plotly.react("chart-speed-bar", [{
      type: "bar",
      x: bar.map(d => d.generation_tps_mean ?? d.overall_tps_mean ?? 0),
      y: bar.map(d => modelLabel(d)),
      orientation: "h",
      marker: { color: "#10b981" },
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: "Speed at ~8K Context (tokens/sec)", font: { color: "#fff" } },
      xaxis: { title: "tokens/sec", color: "#9ca3af" },
      yaxis: { color: "#9ca3af", automargin: true },
    }, { responsive: true });
  }
}

// ─────────────────────────────────────────────
// Tab 7: Quant Impact
// ─────────────────────────────────────────────

async function renderQuantImpact() {
  const modelConfig = document.getElementById("filter-quant-model").value;
  if (!modelConfig) return emptyChart("chart-quant-impact", "Select a model above");

  const data = await fetchAPI("/quant-impact", { model_config: modelConfig }).catch(() => []);
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

  Plotly.react("chart-quant-impact", traces, {
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

let _currentTab = "overview";

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

  // Populate quant model selector
  const qm = document.getElementById("filter-quant-model");
  const firstOpt = qm.querySelector("option");
  qm.innerHTML = "";
  qm.appendChild(firstOpt);
  (opts.model_configs || []).forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    qm.appendChild(opt);
  });
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

  // Filter changes re-render current tab
  ["filter-runtime", "filter-quant", "filter-arch"].forEach(id => {
    document.getElementById(id).addEventListener("change", renderCurrentTab);
  });

  // Depth run selector
  document.getElementById("filter-depth-run").addEventListener("change", e => {
    renderRecallDepth(e.target.value);
  });

  // Quant model selector
  document.getElementById("filter-quant-model").addEventListener("change", renderQuantImpact);

  await populateDropdowns();
  renderCurrentTab();
}

document.addEventListener("DOMContentLoaded", init);
