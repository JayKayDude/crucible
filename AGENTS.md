# AGENTS.md — LLM Benchmarker

## Project Description

Comprehensive local LLM evaluation platform. Combines lm-eval (coding/reasoning/multi-language benchmarks), codeneedle (long-context recall), and speed profiling. Tracks everything by model, quantization, and runtime. Stores results in SQLite, visualizes in FastAPI web dashboard.

## Agent Roles

### Claude Code (Primary)
- Implements all source files
- Follows wave-based delivery plan
- Reads SESSION-STATE.md before each session
- Updates SESSION-STATE.md at end of each session

### Governance MCP
- Evaluates planned actions before implementation
- Blocks S-Series violations
- Referenced via ai-governance MCP tools

## Conventions
- All benchmark configs live in `configs/models/`
- All results stored in `results/` (JSON per run) and SQLite DB
- No external build steps — FastAPI + Plotly.js only
- Python 3.11+ required
