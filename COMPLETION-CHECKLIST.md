# COMPLETION-CHECKLIST.md — LLM Benchmarker

Track wave completion. Check off items as they are verified.

## Wave 1 — Foundation
- [x] codeneedle cloned into project root
- [x] `configs/lmeval/` directory created
- [x] `results/lmeval/` directory created
- [x] `webapp/` skeleton created (`__init__.py`, `routes/__init__.py`, `static/`)
- [x] `requirements-extended.txt` created
- [x] `configs/models/local.toml` created (universal auto-detect)
- [x] Governance docs created: AGENTS.md, CLAUDE.md, PROJECT-MEMORY.md, SESSION-STATE.md, LEARNING-LOG.md
- [x] `bench` package imports cleanly

## Wave 2 — Core Modules
- [x] `config.py` — TOML loader with auto-detect fields
- [x] `timing.py` — TTFT + tokens/sec measurement, LM Studio auto-detection
- [x] `db.py` — SQLite schema for all three benchmark types

## Wave 3 — Runner Integration
- [x] `runner.py` — multi-run averaging, timing adapter, DB write, extended JSON dump

## Wave 4 — Benchmark Runners
- [x] `lmeval_runner.py` — subprocess-based lm-eval integration with split-mode per-task limits
- [x] `speed_profiler.py` — tokens/sec and TTFT measurement loop

## Wave 5 — CLI
- [x] `bench.py` extended with `recall`, `lmeval`, `speed`, `import-lmeval`, `run-all`, `serve` subcommands
- [x] `--limit` CLI flag overrides all per-task TOML limits in split mode

## Wave 6 — Dashboard
- [x] FastAPI app with routes for results, comparisons, charts
- [x] Plotly.js frontend in `webapp/static/`
- [x] Multi-Language tab removed (coding-multilang deferred; tab was empty)

## Wave 7 — Integration
- [x] Smoke test passes (`python3 smoke_test.py`)
- [x] All module imports clean
- [x] All CLI subcommands exit 0 on `--help`
- [x] Config loading with new fields (`runs_per_function`, `quantization`, `architecture`, `task_limits`)
- [x] DB schema creation and `query_filter_options` returns all 8 keys
- [x] webapp Python files parse cleanly
- [x] lm-eval suite configs loadable (coding-standard, reasoning)
- [x] Governance docs updated with final decisions

## Reasoning Suite Refinement (2026-05-25)
- [x] MMLU Pro removed from reasoning suite (tests knowledge recall, not reasoning)
- [x] IFEval added (541-question instruction following, no limit)
- [x] BBH limit set to 75/subtask (27 × 75 = 2,025 total questions)
- [x] GSM8K limit set to 300 questions
- [x] Per-task limits implemented in `lmeval_runner.py` via split-mode subprocess
- [x] CLI `--limit` override works in split mode
- [x] `coding-multilang` removed from `--suite all` default sweep
- [x] README updated: timing table, suite descriptions, run-all sequence
- [x] Smoke test passed: 3-question run, all 3 subprocesses (BBH/GSM8K/IFEval) confirmed working
- [x] DB confirmed: smoke test results landed under single run_id

## Real Benchmark Runs — Gemma 4 E2B
- [x] coding-standard — HumanEval+ and MBPP+ (ran in earlier session)
- [x] reasoning full run — BBH 74.6%, GSM8K 74.3%, IFEval 80.2% prompt-strict ✓
- [x] recall benchmark — codeneedle jquery corpus (ran in earlier session)
- [x] speed profiler — 7 context sizes, 135.5→57.5 t/s ✓
- [x] dashboard verified with real data from all benchmark types ✓

## Shipping
- [x] `.gitignore` updated (results DB, lm-eval outputs, venv with space excluded)
- [x] GitHub repo created: https://github.com/JayKayDude/crucible
- [x] All code committed and pushed to main

## Benchmark Suite (Final)
- [x] coding-standard: humaneval_plus + mbpp_plus (0-shot, coding system instruction)
- [x] reasoning: bbh_cot_zeroshot (75/subtask) + gsm8k_cot_zeroshot (300) + ifeval (all 541)
- [x] multilingual coding: deferred — BigCode Harness needed; removed from default suite
- [x] codeneedle recall: jquery + http_server corpora available
- [x] speed profiler: 7 context sizes (1K–128K), 3 samples each
