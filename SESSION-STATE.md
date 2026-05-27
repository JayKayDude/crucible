# Session State

**Last Updated:** 2026-05-27
**Status:** Active development — dashboard feature work in progress

## Current Position

Dashboard is fully functional at http://127.0.0.1:8000 with the following tabs live:
Overview, Coding, Reasoning, Long Context, Speed, Run, Models.

**Next task:** Implement the **Data tab** — per-run browser with shift-select, inline metadata editing (quant/hardware), single-run detail expand, and bulk delete.

## Benchmark Results — Gemma 4 E2B

| Model | Quant | HumanEval+ | BBH | GSM8K | IFEval | Recall (jquery) | Recall (http_server) | Speed @8K |
|---|---|---|---|---|---|---|---|---|
| google/gemma-4-e2b | Q4_K_M | 47.6% | 74.3% | 74.3% | 80.2% | 31.3% | — | 119.3 t/s |
| google/gemma-4-e2b | Q8_0 | 54.3% | 75.7% | 83.2% | 80.2% | 25.0% (jquery) | 18.2% (http_server) | 125.9 t/s |

## Dashboard Features Completed (2026-05-27)

- **Run tab**: launch benchmarks from UI, per-run cards with live SSE log streaming, Cancel, Dismiss
- **Models tab**: TOML config editor with structured form + raw TOML toggle, create/edit/delete
- **Recall drilldown**: horizontal grouped bar chart, per-function pass rates, 2-model comparison
- **Hardware differentiation**: `(model_name, runtime, quantization, hardware)` is the unique key for model_configs — different hardware = separate entries on all charts
- **Filter panel**: hierarchical model filter shows `Q8_0 · Apple M5` style entries when hardware is set
- **`modelLabel()`**: appends ` · {hardware}` suffix when hardware is non-empty
- **Bug fix**: `POST /api/run` 405 error — changed `@router.post("/")` to `@router.post("")` to avoid spa_fallback catch-all conflict

## Key Architectural Facts (dashboard)

- `webapp/routes/run_routes.py` — run management (POST /api/run, GET /api/run/{id}/logs SSE, DELETE /api/run/{id})
- `webapp/routes/config_routes.py` — model TOML CRUD (GET/PUT/POST/DELETE /api/config/models)
- `webapp/main.py` — CORS allows GET/POST/PUT/DELETE; `app.state.run_state = {}` initialized in `serve()`
- `bench/db.py` unique key migration: (model_name, runtime) → (model_name, runtime, quantization) → (model_name, runtime, quantization, hardware)
- `hardware` stored as `""` (empty string, not NULL) to work correctly in UNIQUE constraints
- `query_speed_curves`, `query_recall_leaderboard`, `query_lmeval_leaderboard`, `query_overview` all SELECT `mc.hardware`

## Pending Work

1. **Data tab** — per-run browser (see plan file)
2. Run full suite on a second model for comparison

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
```

## Venv Location

Lives at `~/llm-benchmarker-venv` (outside iCloud). Symlinked as `.venv 2`. Always activate with `source ".venv 2/bin/activate"`.
