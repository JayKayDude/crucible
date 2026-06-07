# Session State

**Last Updated:** 2026-06-07
**Status:** Active development

## Current Position

Dashboard is fully functional at http://127.0.0.1:8000 with all tabs live:
Overview, Coding, Reasoning, Long Context, Speed, Run, Models, Data.

**Next task:** Run full suite on a second model for comparison.

## Benchmark Results — Gemma 4 E2B

| Model | Quant | HumanEval+ | BBH | GSM8K | IFEval | Recall (jquery) | Recall (http_server) | Speed @8K |
|---|---|---|---|---|---|---|---|---|
| google/gemma-4-e2b | Q4_K_M | 47.6% | 74.3% | 74.3% | 80.2% | 31.3% | — | 119.3 t/s |
| google/gemma-4-e2b | Q8_0 | 54.3% | 75.7% | 83.2% | 80.2% | 25.0% | 18.2% | 125.9 t/s |

## Dashboard Features Completed

### Data tab (2026-05-27)
- Split panel: model sidebar (left) with run count badges + run browser (right)
- All/Recall/Coding/Reasoning/Speed sub-tabs; All groups runs into card blocks by type
- Single-click: select row + detail table slides out (CSS max-height transition)
- Re-click deselects; shift-click range multi-select; detail hidden when 2+ selected
- Edit Selected: change quant/hardware → run moves to correct sidebar entry
- Delete Selected: cascade delete + orphan model_config cleanup

### Run tab (2026-06-06)
- Reconnects to in-progress runs after page reload (fetches `/api/run/active` on tab open)
- Custom Corpus Files panel: upload any source file → persisted to `fixtures/custom/`, auto-creates corpus TOML
- Uploaded files appear in corpus dropdown and Long Context Recall filter immediately
- Remove button deletes file + TOML

### Long Context Recall tab (2026-06-06)
- Corpus filter bar (All / http_server / jquery / custom...) — auto-populated from data, hidden if only one corpus
- Filter persists across re-renders, resets when global filters change

### Models tab (2026-06-07)
- Selected model persists when switching to other tabs and back
- Sidebar item style aligned to Data tab (purple left border on active, consistent hover)
- Delete and Rename support for non-`local` configs (Delete button + editable Config name field in editor)

### Run tab (2026-06-07)
- Loaded model detection: green dot + model name/quant shown below model dropdown for local configs
- Red dot + "No model loaded" + disabled Run button when no model is in VRAM
- Status hides for API models (always enabled); polls every 15s for live updates
- Corpus (Recall) selector moved to second row alongside Quantization override and Architecture

### General (2026-05-27)
- Run tab: launch benchmarks from UI, per-run cards with live SSE log streaming, Cancel, Dismiss
- Models tab: TOML config editor with structured form + raw TOML toggle, create/edit/delete
- Recall drilldown: horizontal grouped bar chart, per-function pass rates, 2-model comparison
- Hardware differentiation: `(model_name, runtime, quantization, hardware)` unique key
- Filter panel: hierarchical model filter with `Q8_0 · Apple M5` style entries
- Playwright MCP configured to use Chromium (not Chrome) — no sign-out side effect

## Key Architectural Facts

- `webapp/routes/run_routes.py` — run management; uses `sys.executable` (not `python3`) to spawn bench.py subprocesses so venv packages are always available; `run_state` dict on `app.state`; GET /api/run/active lists all
- `webapp/routes/config_routes.py` — model TOML CRUD + custom corpus upload/list/delete; `GET /api/config/models/{name}/status` returns loaded state for local models
- `webapp/routes/api.py` — main API; GET/DELETE/PATCH /api/runs/* for Data tab
- `webapp/main.py` — CORS allows GET/POST/PUT/PATCH/DELETE
- `bench/db.py` unique key: (model_name, runtime, quantization, hardware); hardware stored as `""` not NULL
- `bench/timing.py` — `detect_loaded_model()`: tries LM Studio `/api/v0/models` (filters `type=embeddings`, raises if nothing loaded), Ollama `/api/ps`, then `/v1/models` fallback
- Auto-detect quant/arch skipped if value already set in config (`if not model_cfg.quantization`)
- Custom corpus files: `fixtures/custom/` + `configs/corpora/{stem}.toml`; needs `python-multipart` installed
- `webapp/static/index.html` has `?v=5` cache buster on app.js script tag
- `requirements.txt` is the single requirements file (merged from former `requirements-extended.txt`)
- `.gitignore` excludes: `results/`, `*.db`, `.playwright-mcp/`, all Claude/AI session docs, Docker files

## Pending Work

1. Run full suite on a second model for comparison
2. Create `.secrets/anthropic.key` to enable claude-sonnet-4-6 benchmarks

## Key Commands

```bash
# Dashboard
source ".venv 2/bin/activate" && python3 bench.py serve   # → http://127.0.0.1:8000

# Run individual benchmarks
python3 bench.py speed --model local
python3 bench.py recall --corpus jquery --model local
python3 bench.py recall --corpus http_server --model local
python3 bench.py lmeval --suite reasoning --model local
python3 bench.py run-all --model local --corpus jquery

# Custom corpus via CLI (bypass webapp)
python3 bench.py recall --file path/to/myfile.js --model local
```

## Venv Location

Lives at `~/llm-benchmarker-venv` (outside iCloud). Symlinked as `.venv 2`. Always activate with `source ".venv 2/bin/activate"`.
