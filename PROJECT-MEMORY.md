# PROJECT-MEMORY.md — LLM Benchmarker

Persistent architectural decisions and context. Claude reads this at session start.

## Core Architecture

- **Three benchmark types:** codeneedle (long-context recall), lm-eval (coding/reasoning), speed profiling (tokens/sec, TTFT)
- **Storage:** Unified SQLite DB at `results/benchmark.db` + per-run JSON in `results/`
- **Frontend:** FastAPI + Plotly.js, no build step
- **Config:** TOML per model in `configs/models/`, TOML per suite in `configs/lmeval/`

## Repository

- **GitHub:** https://github.com/JayKayDude/crucible
- **Gitignored:** `results/*.db`, `results/lmeval/`, `.venv 2/` — each user generates their own results

## Key Architectural Decisions

- **Subprocess over library for lm-eval:** Stability and decoupling — lm-eval has heavy deps that can conflict; subprocess isolates it cleanly.
- **`local.toml` with auto-detection:** Single config for all local models. At runtime, queries LM Studio's `/api/v0/models` (or `/v1/models` fallback) to discover the loaded model name and quantization — no manual editing needed when switching models.
- **Unified SQLite DB for all three benchmark types:** Single source of truth for recall, lm-eval, and speed results. Enables cross-benchmark comparisons in the dashboard.
- **FastAPI + Plotly.js with no build step:** Keeps the dashboard deployable without Node.js.
- **Per-task limits via split-mode subprocess:** lm-eval's `--limit` flag is global. When different tasks need different limits (e.g. BBH=75, GSM8K=300, IFEval=no limit), the runner groups tasks by limit value and runs each group as a separate subprocess. Results are merged into one `run_id` in the DB — the dashboard sees one reasoning run.
- **CLI `--limit` overrides TOML per-task limits:** When `--limit N` is passed on the command line, it overrides all per-task TOML limits in split mode. Useful for quick smoke tests.

## Model Config Conventions

- `local.toml` — universal auto-detect config; works for LM Studio, llama.cpp, Ollama, MLX
- Other TOMLs in `configs/models/` — for cloud APIs (Claude, GPT-5) or explicit per-model configs
- `lmeval_tokenizer` field: set to a HuggingFace model ID from the same family for lm-eval tokenizer. Auto-detected where possible via `bench/known_architectures.py`.
- Never hardcode API keys — use `api_key_env` (reads from env var) or `api_key_file`

## Runtime Notes

- Python 3.11+ required (uses `tomllib` from stdlib)
- LM Studio must be running on port 1234 for local benchmarks
- `suppress_thinking = true` strips `<think>...</think>` blocks before scoring
- `prefill_no_think = true` (Qwen-specific) prepends assistant message to suppress CoT
- Venv must be OUTSIDE iCloud Drive — iCloud reads each file at ~0.918s; lm-eval has 13k+ YAMLs → 3.5hr startup inside iCloud. Venv lives at `~/llm-benchmarker-venv`, symlinked as `.venv 2`.

## CLI Quick Reference

```bash
# Run individual benchmarks
python3 bench.py recall --corpus jquery --model local
python3 bench.py lmeval --suite coding-standard --model local
python3 bench.py lmeval --suite reasoning --model local
python3 bench.py speed --model local

# Run everything (recall → coding-standard → reasoning → speed)
python3 bench.py run-all --model local --corpus jquery

# Quick test (overrides all per-task TOML limits)
python3 bench.py lmeval --suite reasoning --model local --limit 3

# Dashboard
python3 bench.py serve   # → http://127.0.0.1:8000

# Import existing lm-eval output
python3 bench.py import-lmeval --path results/lmeval/some_run/ --model local
```

## Final Benchmark Suite

| Suite | Tasks | Limits | Notes |
|---|---|---|---|
| coding-standard | humaneval_plus, mbpp_plus | none | 0-shot; system instruction scoped to coding suites only |
| reasoning | bbh_cot_zeroshot, gsm8k_cot_zeroshot, ifeval | BBH=75/subtask, GSM8K=300, IFEval=all 541 | 0-shot; 2048 max_gen_toks; 3 separate subprocess calls |
| codeneedle (recall) | long-context recall | 16 functions sampled | jquery.js or http_server.py corpus |
| speed | tokens/sec across context sizes | 7 sizes: 1K–128K (128K capped at ~75K — fixture limit) | 3 samples per size |

**coding-multilang deferred:** MultiPL-E tasks don't exist in lm-eval. BigCode Harness is the correct tool but scores are heavily correlated with Python (HumanEval+) — low ROI. `--suite coding-multilang` still exists as an explicit option but is not included in `--suite all` or `run-all`.

## Key Parameter Decisions

- `max_gen_toks = 2048` — matches MMLU Pro's own default; enough for CoT reasoning; prevents thinking models from running for days
- `num_fewshot = 0` — use `cot_zeroshot` task variants designed for 0-shot; comparable across thinking and non-thinking models
- `--system_instruction` applied to coding suites only — reasoning tasks must NOT get the Python-only instruction
- **MMLU Pro removed** from reasoning suite — tests knowledge recall, not reasoning ability. Replaced with IFEval (instruction following).
- **BBH limit: 75/subtask** — 27 subtasks × 75 = 2,025 total questions. Overall MoE ±1.7% at p=0.5.
- **GSM8K limit: 300** — down from 1,319; ~1h at Gemma E2B speed. Acceptable MoE (±2.8%).
- **IFEval: no limit** — all 541 questions; fast and uniform response length.

## Dashboard Architecture (webapp/)

- **Routes:** `webapp/routes/api.py` (data queries), `run_routes.py` (run management), `config_routes.py` (TOML CRUD + custom corpus)
- **Run management:** `POST /api/run` spawns bench.py subprocess using `sys.executable` (not `python3`) so venv packages are always available; streams stdout via SSE at `GET /api/run/{id}/logs`; `GET /api/run/active` lists all in-memory runs; reconnect on page reload via `renderRun()` fetching `/active` on every tab visit
- **Config CRUD:** `GET/PUT/POST/DELETE /api/config/models` — hand-rolled TOML writer (no tomlkit); reads via stdlib `tomllib`
- **Model status endpoint:** `GET /api/config/models/{name}/status` — returns `{is_local, loaded, model_name, quantization}`; used by Run tab to show live loaded-model indicator and disable Run button when nothing is in VRAM
- **Models tab:** Delete and Rename supported for all configs except `local` (protected by `isLocal` flag in `openModelEditor`). Rename is implemented as POST new name + DELETE old name.
- **Custom corpora:** `POST /api/config/corpora/upload` saves file to `fixtures/custom/` and writes a corpus TOML; `GET /api/config/corpora/custom` lists them; `DELETE /api/config/corpora/custom/{name}` removes both. Requires `python-multipart` in venv.
- **CORS:** allows GET/POST/PUT/PATCH/DELETE in `webapp/main.py`
- **Route gotcha:** use `@router.post("")` not `@router.post("/")` — a `GET /{path:path}` catch-all in main.py intercepts the 307 redirect and returns 405

## Hardware Differentiation

- **Unique key:** `model_configs` is keyed on `(model_name, runtime, quantization, hardware)` — same model on different hardware = separate DB entries
- **`hardware` field:** stored as `""` (empty string, not NULL) — NULL != NULL in SQL UNIQUE constraints would create duplicates
- **All queries** (`query_speed_curves`, `query_recall_leaderboard`, `query_lmeval_leaderboard`, `query_overview`) SELECT `mc.hardware` and pass it to the frontend
- **`modelLabel(row)`** in app.js: appends ` · {hardware}` suffix when `row.hardware` is non-empty
- **Filter panel key:** `model_name|quantization|hardware` — hardware-differentiated entries show as `Q8_0 · Apple M5` in the filter list
- **`applyModelFilter`** matches on the same 3-part key
- **`get_speed_comparison`** key includes hardware: `(model_name, quantization, hardware)` — prevents hardware variants from overwriting each other in bar chart

## Quantization / Architecture Auto-Detection

- `detect_loaded_model()` in `bench/timing.py` tries LM Studio `/api/v0/models` → Ollama `/api/ps` → `/v1/models` in order
- LM Studio path filters out `type=embeddings` models and raises `RuntimeError` immediately if nothing has `state=loaded` (prevents fallthrough to `/v1/models` which would return the embedding model)
- Auto-detect is **skipped** if `quantization` (or `architecture`) is already set in the config file — `if not model_cfg.quantization`
- Cloud APIs (Anthropic, OpenAI) don't expose these endpoints; quant/arch stay `None` in the DB unless set manually in the TOML
- `lookup_architecture()` in `bench/known_architectures.py` does substring matching on model ID (e.g. "qwen" → "gated-delta"); Claude/GPT return `None`

## Playwright MCP

- Configured via `claude mcp add playwright -s local -- npx @playwright/mcp@latest --browser chromium`
- Uses Chromium (not Chrome) — no sign-out side effect, no "managed by admin" banner
- Config stored in `.claude.json` at project root (local scope)
- Always wait 2+ seconds after tab clicks before checking DOM state — async renders need time

## Key Implementation Fixes (from real runs)

- **venv inside iCloud = 3.5hr startup**: 13k+ YAML files at 0.918s each. Fix: venv at `~/llm-benchmarker-venv`.
- **HF_DATASETS_OFFLINE blocks first-time download**: removed the flag; datasets download once and cache.
- **max_gen_toks=8192 causes multi-day runs for thinking models**: hardcoded 2048 in lmeval_runner.py.
- **GSM8K 0% at 0-shot with standard variant**: standard task scores `#### N` format. Fix: `gsm8k_cot_zeroshot`.
- **BBH crashes with num_fewshot=8**: only 3 examples in fewshot split. Fix: `bbh_cot_zeroshot`.
- **system_instruction applied to all suites**: was sending Python-only instruction to math/reasoning tasks. Fixed.
- **MBPP+ scored 5%** with "output only body" instruction — needs complete function. Fixed to "do not restate signature".
- **lm_eval binary path**: use full path, not PATH-dependent. Priority search: alongside sys.executable → .venv/bin → ~/llm-benchmarker-venv/bin.
- **lm-eval --limit is global**: solved with split-mode subprocess grouping by limit value.
- **Per-request overhead dominates benchmark time**: LM Studio adds ~14s per request independent of token count. Reported t/s (generation only) is not a reliable predictor of total benchmark time.
