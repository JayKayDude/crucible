# Session State

**Last Updated:** 2026-05-25
**Status:** Active — smoke test running, full reasoning run pending

## Current Position

Reasoning suite refinement complete. A 3-question smoke test (`--limit 3`) is running in the background against Gemma 4 E2B to confirm the new split-mode subprocess logic works. Full reasoning run will start after smoke test passes.

## Key Changes This Session (2026-05-25)

- **Reasoning suite finalized**: BBH=75/subtask, GSM8K=300, IFEval=all 541, MMLU Pro removed
- **Per-task limits implemented**: `lmeval_runner.py` now supports per-task question limits via `[tasks.limits]` in suite TOML. When limits differ across tasks, each task group runs as a separate `lm_eval` subprocess; results are merged into one `run_id` in the DB.
- **CLI `--limit` overrides TOML limits**: when `--limit N` is passed, it takes priority over per-task TOML limits (useful for quick tests).
- **coding-multilang removed from default suite**: `--suite all` now runs `coding-standard` + `reasoning` only. `--suite coding-multilang` still works as an explicit option.
- **README updated**: timing table added, MMLU Pro removed from suite descriptions, run-all sequence updated.

## All Fixes Applied (Cumulative)

- **lm_eval binary path**: searches venv candidates in priority order (alongside sys.executable → .venv/bin → ~/llm-benchmarker-venv/bin)
- **HF_DATASETS_OFFLINE removed**: datasets download on first run, cached locally
- **HuggingFace cache pinned out of iCloud**: `xattr -w com.apple.fileprovider.ignore#P 1 ~/.cache/huggingface`
- **reasoning num_fewshot=0**: BBH only has 3 examples; cot_zeroshot variants are designed for 0-shot
- **system instruction scoped to coding suites only**: reasoning tasks never get the Python-only instruction
- **MBPP+ system instruction**: "do not restate the function signature" (not "output only body")
- **GSM8K variant**: `gsm8k_cot_zeroshot` (scores "The answer is X" format; standard gsm8k scores 0% at 0-shot)
- **BBH variant**: `bbh_cot_zeroshot` (0-shot, flexible extraction)
- **max_gen_toks=2048**: matches MMLU Pro's own default, 4× faster than 8192
- **--limit flag**: CLI override; in split mode, overrides all per-task TOML limits
- **venv outside iCloud**: lives at `~/llm-benchmarker-venv`, symlinked as `.venv 2`
- **MMLU Pro removed**: tests knowledge recall, not reasoning ability — removed from reasoning suite
- **IFEval added**: 541-question instruction-following benchmark, no question limit

## Pending Work

1. **Confirm smoke test passes** — BBH + GSM8K + IFEval with `--limit 3`, three separate subprocess calls
2. **Run full reasoning suite** against Gemma 4 E2B (~10-15h; run overnight)
3. **Re-run coding-standard** against Gemma 4 E2B (previous run was Qwen 3.6 35B)
4. **Run recall benchmark** (codeneedle)
5. **Run speed profiler**
6. **Launch dashboard** and verify all charts render with real data

## Key Run Commands

```bash
# Full reasoning suite (overnight)
nohup ".venv 2/bin/python3" bench.py lmeval --suite reasoning --model local > /tmp/lmeval_reasoning.log 2>&1 &

# Monitor
tail -f /tmp/lmeval_reasoning.log

# Verify DB after run
".venv 2/bin/python3" -c "
import sqlite3
conn = sqlite3.connect('results/benchmark.db')
for row in conn.execute('SELECT task, COUNT(*) FROM lmeval_results GROUP BY task'):
    print(row)
"

# Launch dashboard
".venv 2/bin/python3" bench.py serve
```

## Venv Location

The venv lives at `~/llm-benchmarker-venv` (outside iCloud to avoid the 0.918s-per-file read penalty).
Symlinked into project as `.venv 2`.
Always use `".venv 2/bin/python3"` — NOT system Python.
