# Session State

**Last Updated:** 2026-05-25
**Status:** Complete — all benchmarks run, repo live on GitHub

## Current Position

All benchmarks complete for Gemma 4 E2B. Dashboard running at http://127.0.0.1:8000.
GitHub repo live at https://github.com/JayKayDude/crucible.

## Benchmark Results — Gemma 4 E2B (google/gemma-4-e2b, Q4_K_M)

| Benchmark | Score | Details |
|---|---|---|
| **HumanEval+** | TBD | coding-standard run in earlier session |
| **MBPP+** | TBD | coding-standard run in earlier session |
| **BBH** | **74.6%** ±0.84% | 2,025 questions (27 subtasks × 75) |
| **GSM8K** | **74.3%** ±2.5% | 300 questions |
| **IFEval** | **80.2%** prompt-strict / **81.3%** prompt-loose | 541 questions |
| **Recall** | TBD | codeneedle jquery corpus, earlier session |
| **Speed** | 135.5 t/s @ 1K → 57.5 t/s @ 64K | 7 context sizes, 3 samples each |

Speed curve: 986→135.5, 4K→127.6, 8K→119.3, 16K→102.6, 32K→80.7, 64K→57.5, 75K→62.3 t/s
Note: 128K context test capped at ~75K tokens (jquery.js fixture too short).

## Key Changes This Session (2026-05-25)

- **Reasoning suite finalized**: BBH=75/subtask, GSM8K=300, IFEval=541, MMLU Pro removed
- **Per-task limits**: `lmeval_runner.py` split-mode — each task group runs as a separate subprocess, merged under one `run_id`
- **CLI `--limit` override**: overrides all per-task TOML limits (useful for quick tests)
- **coding-multilang removed from default suite**: `--suite all` = coding-standard + reasoning only
- **Multi-Language tab removed from dashboard**: HTML + JS renderer entry removed
- **GitHub repo created**: https://github.com/JayKayDude/crucible (36 files, 3,729 insertions)
- **`.gitignore` updated**: excludes results DB, lm-eval outputs, venv with space in name

## All Fixes Applied (Cumulative)

- **lm_eval binary path**: searches venv candidates in priority order
- **HF_DATASETS_OFFLINE removed**: datasets download on first run, cached locally
- **HuggingFace cache pinned out of iCloud**: `xattr -w com.apple.fileprovider.ignore#P 1 ~/.cache/huggingface`
- **reasoning num_fewshot=0**: BBH cot_zeroshot variants designed for 0-shot
- **system instruction scoped to coding suites only**
- **GSM8K variant**: `gsm8k_cot_zeroshot` (scores "The answer is X" format)
- **BBH variant**: `bbh_cot_zeroshot` (0-shot, flexible extraction)
- **max_gen_toks=2048**: prevents thinking models from running for days
- **venv outside iCloud**: lives at `~/llm-benchmarker-venv`, symlinked as `.venv 2`
- **MMLU Pro removed**: tests knowledge recall, not reasoning ability
- **IFEval added**: 541-question instruction-following benchmark

## Pending Work (Next Session)

1. **Check coding-standard scores** for Gemma 4 E2B in DB (ran in earlier session)
2. **Run full suite on a second model** (load different model in LM Studio, run `run-all`)
3. **Verify Quant Impact tab** by running same model at different quantizations

## Key Commands

```bash
# Run full suite on any loaded model
".venv 2/bin/python3" bench.py run-all --model local --corpus jquery

# Individual benchmarks
".venv 2/bin/python3" bench.py lmeval --suite reasoning --model local
".venv 2/bin/python3" bench.py speed --model local
".venv 2/bin/python3" bench.py recall --corpus jquery --model local

# Dashboard
".venv 2/bin/python3" bench.py serve   # → http://127.0.0.1:8000
```

## Venv Location

The venv lives at `~/llm-benchmarker-venv` (outside iCloud — 0.918s-per-file read penalty inside iCloud).
Symlinked into project as `.venv 2`. Always use `".venv 2/bin/python3"`.
