# CLAUDE.md — LLM Benchmarker

## Project Description

Comprehensive local LLM evaluation platform. Combines lm-eval (coding/reasoning/multi-language benchmarks), codeneedle (long-context recall), and speed profiling. Tracks everything by model, quantization, and runtime. Stores results in SQLite, visualizes in FastAPI web dashboard.

## Quick Start for Claude

1. Read `SESSION-STATE.md` first — it tells you exactly where work left off.
2. Read `PROJECT-MEMORY.md` for architectural decisions.
3. Follow the wave plan in `SESSION-STATE.md` or the task list.
4. Update `SESSION-STATE.md` when done.

## Directory Layout

```
LLM Benchmarker/
  bench/              # benchmark harness (Python package)
  bench.py            # CLI entry point
  configs/
    models/           # per-model TOML configs
    lmeval/           # lm-eval task configs
    corpora/          # recall corpus configs
  fixtures/           # source files for recall + speed tests
  results/            # generated results (gitignored)
  webapp/             # FastAPI dashboard
    routes/
    static/
  requirements.txt    # all deps (fastapi, uvicorn, lm-eval, codeneedle)
```

## Key Rules
- Do NOT commit unless explicitly asked.
- Do NOT push to remote unless explicitly asked.
- Minimal changes — no over-engineering.
- Always run verification steps before reporting done.
