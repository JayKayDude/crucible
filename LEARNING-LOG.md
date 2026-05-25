# LEARNING-LOG.md — LLM Benchmarker

Lessons learned during implementation. Append new entries; do not delete old ones.

## Wave 1 (2026-05-19)

- codeneedle already includes `configs/models/` with several pre-existing TOML configs for cloud and local models. New LM Studio TOMLs added alongside them.
- codeneedle `bench/` package imports cleanly with no extra dependencies beyond `requirements.txt`.
- `tomllib` is stdlib in Python 3.11+ — no extra dep needed for TOML loading.
- The project directory was nearly empty before Wave 1 (only `.claude/settings.local.json`). All governance files were created fresh rather than edited.

## First Real Benchmark Runs (2026-05-23 to 2026-05-24)

- **venv inside iCloud = 3.5 hour startup**: iCloud reads each file at ~0.918s. lm-eval has 13,533 YAML files. Fix: venv at `~/llm-benchmarker-venv` outside iCloud, symlinked into project.
- **HF_DATASETS_OFFLINE blocks first-time download**: reasoning datasets (GSM8K, BBH, MMLU Pro) weren't cached. Removed the offline env flag entirely — datasets download once, then cache.
- **HuggingFace cache evicted to iCloud**: user pressed "Store in iCloud" on ~/.cache/huggingface. Fix: `xattr -w com.apple.fileprovider.ignore#P 1 ~/.cache/huggingface` pins it local permanently.
- **max_gen_toks=8192 causes multi-day runs for thinking models**: Qwen 3.6 35B at 8192 tokens/question on MMLU Pro = 80+ hours. MMLU Pro's own task YAML sets 2048 as intended limit. Fix: hardcode 2048 in lmeval_runner.py.
- **GSM8K scores 0% at 0-shot with standard variant**: standard `gsm8k` task scores `#### number` format, learned only from few-shot examples. At 0-shot, thinking models output "The answer is X" which isn't matched. Fix: use `gsm8k_cot_zeroshot` which scores "The answer is X".
- **BBH crashes with num_fewshot=8**: BBH fewshot split only has 3 examples. Fix: use `bbh_cot_zeroshot` (designed for 0-shot, flexible extraction, prompts "Let's think step by step").
- **system_instruction applied to all suites**: "Output only Python code" was being sent to GSM8K/BBH/MMLU Pro, telling the model to answer math problems in Python. Fix: scope instruction to suites where `"coding" in suite_cfg.name`.
- **MBPP+ scored 5% due to over-restrictive instruction**: "output ONLY the indented body" made MBPP+ (which needs a complete function) produce ungradeable code. Fix: "do not restate the function signature" — prevents HumanEval stop-sequence bug without breaking MBPP+.
- **MultiPL-E tasks don't exist in lm-eval**: plan was wrong. `multiple_js`, `multiple_py` etc. are not in lm-eval 0.4.12 or main branch. BigCode Harness is the correct tool but deferred (low ROI).
- **--limit flag missing from bench.py**: added to lmeval subcommand for fast test runs without editing configs.
- **Gemma 4 2B is a thinking model**: despite being 2B, still generates heavy reasoning traces. Don't assume small = fast for thinking model families.

## Wave 2-6 Implementation (2026-05-19)

- lm-eval `--model` flag routing must use `local-completions` for LM Studio (not `lm-studio`)
- `statistics.stdev()` crashes on a list of length 1 — always guard with `len(v) > 1`
- `Plotly.react()` must be used (not `newPlot`) for in-place chart updates on filter change
- API key should NOT be passed in lm-eval `--model_args` (visible in `ps`); use env var `OPENAI_API_KEY` instead
- `subprocess.Popen` with `bufsize=1` + line iteration streams lm-eval output correctly without deadlock
- System Python 3.14 (Homebrew macOS) blocks `pip install` via PEP 668 — always use a venv
- `_aggregate_runs()` must accept and forward `relax_indent` param — hardcoding False breaks Gemma-style models
- Speed profiler: use `continue` not `break` on sample error so one failed sample doesn't abort the context size
- Qwen 3.6 35B used as code reviewer via `opencode run` — effective at catching subtle logic issues

## Reasoning Suite Calibration (2026-05-25)

- **Per-request overhead dominates benchmark time**: LM Studio adds ~14s per request independent of how many tokens are generated. A 2-token response takes almost as long as a 400-token one. Reported t/s (generation speed) is a poor predictor of total benchmark time — per-request overhead and model verbosity matter more.
- **BBH `--limit` applies per subtask**: With 27 subtasks, `--limit 75` gives 27 × 75 = 2,025 total questions. The overall BBH headline score MoE at 75/subtask is ±1.7% — acceptable for comparison purposes.
- **Small models can be slower than larger ones on reasoning**: Gemma 4 E2B (~140 t/s) was slower than expected because it (1) hits the 2048 token limit frequently and (2) per-request overhead dominates. High t/s numbers on tiny models don't translate to fast benchmarks.
- **lm-eval `--limit` is global, not per-task**: Cannot set different question counts for different tasks in one subprocess call. Solution: group tasks by limit value and run each group as a separate subprocess; merge results into one `run_id` in the DB.
- **MMLU Pro removed from reasoning suite**: MMLU Pro tests knowledge recall (memorized facts), not reasoning ability. BBH + GSM8K + IFEval better reflects actual reasoning without contamination from training data memorization.
- **CLI `--limit` should override TOML per-task limits**: Useful for quick smoke tests. The initial split-mode implementation ignored the CLI `--limit` when in split mode — discovered during testing, fixed immediately by using `effective_limit = limit if limit is not None else grp_limit`.
- **Model size tier predicts benchmark time better than t/s**: t/s and model quality/verbosity are correlated on real hardware. A 2B model at 150 t/s can take longer than a 14B MoE at 50 t/s because the 2B hits the token limit constantly and is more verbose per question.

## Speed Profiler & Shipping (2026-05-25)

- **128K speed test capped by fixture size**: `jquery.js` is ~75K tokens. The 128K context size test used ~75K tokens instead, labeled correctly in the DB but doesn't reflect true 128K behavior. If 128K speed matters, a larger fixture is needed.
- **Speed degrades ~2.4× from 1K to 64K on Gemma 4 E2B**: 135.5 t/s at 1K → 57.5 t/s at 64K. Fairly linear degradation on LM Studio — no sharp cliff, just steady decline with context length.
- **`.venv 2` (space in name) not caught by `.venv/` in .gitignore**: Had to add `.venv 2/` as an explicit separate entry. Git doesn't glob-expand spaces in ignore patterns.
- **`results/lmeval/` and `results/*.db` must be in .gitignore**: Each user generates their own benchmark results — committing them would pollute the repo and create multi-GB diffs. Only configs and code belong in version control.
- **GitHub repo name "crucible"**: Short, evocative, nothing obvious already owns it in the LLM space. Metaphor fits (crucible = vessel for testing under extreme conditions).
