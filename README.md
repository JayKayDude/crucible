# LLM Benchmarker

A comprehensive local LLM evaluation platform. Runs standardized coding and reasoning benchmarks, long-context recall tests, and speed profiling — all on your own hardware at the quantizations you actually use. Results are stored in a SQLite database and visualized in a local web dashboard.

Built on top of [codeneedle](https://github.com/alexziskind1/codeneedle).

---

## What It Tests

| Benchmark | What it measures | Tool |
|---|---|---|
| **Coding (Python)** | HumanEval+, MBPP+ — execution-based pass@1 | lm-eval |
| **Reasoning** | GSM8K (math), BBH (75q/subtask), IFEval (instruction following) | lm-eval |
| **Long Context Recall** | Reproduce a named function from a 80K+ token file verbatim | codeneedle |
| **Speed Profiling** | Prefill + generation tokens/sec across 1K–128K context sizes | custom |

---

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible local server)
- macOS / Linux

---

## Setup

### 1. Create a virtual environment

```bash
cd "LLM Benchmarker"
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Core codeneedle deps
pip install -r requirements.txt

# Extended deps (FastAPI dashboard + lm-eval)
pip install -r requirements-extended.txt
```

> **Note:** `lm-eval` installs many dependencies and may take a few minutes.

### 3. Configure your model

Model configs live in `configs/models/`. Two are included:

**`local.toml`** — universal auto-detect config. Load any model in LM Studio and use `--model local`. The tool automatically reads the model name and quantization from LM Studio's API — no manual editing needed.

**`qwen36-35b-lmstudio.toml`** — explicit config for Qwen 3.6 35B A3B at Q4_K_M.

To add a new model, copy an existing config:

```bash
cp configs/models/local.toml configs/models/my-model.toml
```

```toml
# configs/models/my-model.toml
name             = "llama-3.1-8b-instruct"   # must match the model ID in your server
base_url         = "http://127.0.0.1:1234"
api_key          = "lm-studio"
temperature      = 0.0
max_tokens       = 32768
timeout          = 600.0
runtime          = "local"
quantization     = "Q4_K_M"        # label shown in dashboard
architecture     = "standard"      # "standard", "gated-delta", "sliding-window", "moe"
hardware         = "M3 Max 36GB"   # optional label
lmeval_tokenizer = "meta-llama/Meta-Llama-3.1-8B"  # see Tokenizer section below
```

**For cloud APIs** (Claude, OpenAI):

```toml
# configs/models/claude-opus.toml
name        = "claude-opus-4-7"
base_url    = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"   # reads from environment variable
temperature = 0.0
max_tokens  = 8000
timeout     = 120.0
runtime     = "anthropic"
```

> Use `api_key_env` (reads from env var) or `api_key_file` (reads from file path). Never hardcode keys.

---

## Running Benchmarks

Make sure your model server is running before starting any benchmark.

### Long Context Recall

Tests whether the model can reproduce a named function verbatim from a large source file loaded into context. The key challenge: 16 functions sampled from across an 80K+ token file, testing whether recall degrades at different positions.

```bash
# Quick test (~2 min) — small Python HTTP server file
python3 bench.py recall --corpus http_server --model local

# Full test (~10–30 min) — jQuery, ~80K tokens
python3 bench.py recall --corpus jquery --model local

# Run each function 3 times and average results
python3 bench.py recall --corpus jquery --model local --runs 3
```

Results are saved to `results/<corpus>__<model>.json` and to `results/benchmark.db`.

### Coding Benchmarks (lm-eval)

Requires lm-eval (`pip install -r requirements-extended.txt`).

```bash
# Python coding — HumanEval+ and MBPP+ (~538 problems total)
python3 bench.py lmeval --suite coding-standard --model local

# Multi-language coding — 7 languages via MultiPL-E
python3 bench.py lmeval --suite coding-multilang --model local

# Reasoning — GSM8K (math), BBH, MMLU-Pro
python3 bench.py lmeval --suite reasoning --model local

# Run all three suites back to back
python3 bench.py lmeval --suite all --model local
```

> These take a long time on local models. `coding-standard` alone is ~538 problems. Plan for several hours per suite. Results are saved to `results/lmeval/` and imported into the DB automatically.

### Speed Profiling

Measures how inference speed changes as context grows from 1K to 128K tokens.

```bash
# Default: 7 context sizes (1K, 4K, 8K, 16K, 32K, 64K, 128K), 3 samples each
python3 bench.py speed --model local

# Custom context sizes and sample count
python3 bench.py speed --model local --context-sizes 1024,4096,8192,32768 --samples 5
```

Prints a live table:
```
Speed profiling 'local' across 7 context sizes
Samples per size: 3
   Context   Prefill t/s     Gen t/s   Overall t/s     TTFT
----------------------------------------------------------
      1024        3200.0        42.1           N/A      N/A
      4096        1800.0        41.3           N/A      N/A
      8192         950.0        40.1           N/A      N/A
     32768         240.0        35.2           N/A      N/A
```

> `Prefill t/s` and `Gen t/s` are only available when running against llama.cpp directly. LM Studio and cloud APIs show `N/A` for those and populate `Overall t/s` instead.

### Run Everything

Runs recall → coding-standard → reasoning → speed in sequence.

```bash
python3 bench.py run-all --model local --corpus jquery
```

This will take several hours on a large model. Kick it off overnight.

---

## Estimated Benchmark Time

Times below are for the full suite: **Coding** (HumanEval+ + MBPP+) + **Reasoning** (BBH 75q/subtask + GSM8K 300q + IFEval 541q) + **Speed Profiler**. Estimates assume `max_tokens = 2048` and a single concurrent request.

| Model size | Typical generation speed | Full suite (near worst case) |
|---|---|---|
| ~2B | 120–200 t/s | ~10h |
| 4–9B | 60–120 t/s | ~10–12h |
| 14–35B MoE | 40–70 t/s | ~12–15h |
| 27–35B dense | 25–40 t/s | ~15–20h |
| 70B+ | 10–20 t/s | ~22–28h |

> **Note:** Benchmark time can vary drastically between models of the same reported generation speed. Factors include quantization level, architecture (MoE vs dense), hardware, and how verbose the model's chain-of-thought is. Generation speed (t/s) alone is not a reliable predictor — per-request overhead and input context processing time contribute significantly. The table above reflects near-worst-case real-world measurements.

### Import Existing lm-eval Results

If you've already run lm-eval manually and have the output directory, import it:

```bash
python3 bench.py import-lmeval \
  --path results/lmeval/coding-standard__local__abc12345/ \
  --model local \
  --suite-name coding-standard
```

---

## Dashboard

```bash
python3 bench.py serve
```

Opens at **http://127.0.0.1:8000**

| Tab | What you see |
|---|---|
| **Overview** | Radar chart comparing all models across recall, HumanEval+, GSM8K, and speed |
| **Coding** | HumanEval+ and MBPP+ leaderboard with error bars |
| **Multi-Language** | Heatmap of pass@1 scores across 7 languages |
| **Reasoning** | GSM8K, BBH, IFEval leaderboard |
| **Long Context** | Recall leaderboard + depth degradation chart |
| **Speed** | Speed curves (tokens/sec vs context size) + bar chart at fixed context |
| **Quant Impact** | Select a model to compare Q4 vs Q8 vs F16 across all metrics side by side |

Use the filter bar at the top to filter by runtime, quantization, or architecture.

---

## Comparing Quantizations

The main reason this tool exists. To empirically measure how quantization affects real performance:

1. Load Q4_K_M in LM Studio → `python3 bench.py run-all --model local --corpus jquery`
2. Load Q8_0 in LM Studio → `python3 bench.py run-all --model local --corpus jquery`
3. `python3 bench.py serve` → open the **Quant Impact** tab

`local.toml` auto-detects the loaded model name and quantization from LM Studio's `/v1/models` API each time, so results are stored with the correct label without any manual editing.

Alternatively, create explicit configs for each quant if you want full control:

```bash
# configs/models/qwen36-q4.toml  →  quantization = "Q4_K_M"
# configs/models/qwen36-q8.toml  →  quantization = "Q8_0"

python3 bench.py run-all --model qwen36-q4 --corpus jquery
python3 bench.py run-all --model qwen36-q8 --corpus jquery
python3 bench.py serve
```

---

## Project Structure

```
LLM Benchmarker/
├── bench.py                    # Main CLI — all subcommands
├── bench/
│   ├── config.py               # Model config loading (TOML)
│   ├── runner.py               # Long-context recall benchmark runner
│   ├── scorer.py               # Line-level LCS scoring
│   ├── extract.py              # Function extraction (JS + Python)
│   ├── timing.py               # Per-runtime timing adapter
│   ├── db.py                   # Unified SQLite storage
│   ├── lmeval_runner.py        # lm-eval subprocess wrapper
│   ├── speed_profiler.py       # Speed curve measurement
│   └── known_architectures.py  # Model family → architecture lookup
├── configs/
│   ├── corpora/                # Recall test corpus configs
│   ├── models/                 # Model connection configs (one per model/quant)
│   └── lmeval/                 # lm-eval task suite configs
├── fixtures/                   # Source files used in recall + speed tests
├── results/
│   ├── benchmark.db            # All results (SQLite)
│   ├── *.json                  # Per-run recall JSON dumps
│   └── lmeval/                 # Raw lm-eval output directories
├── webapp/
│   ├── main.py                 # FastAPI app
│   ├── routes/api.py           # Dashboard API (10 routes)
│   └── static/                 # index.html, app.js, style.css
├── requirements.txt            # Core deps (codeneedle)
└── requirements-extended.txt   # FastAPI + lm-eval
```

---

## Adding a Custom Corpus

To test recall on your own large source file:

1. Put the file in `fixtures/` (`.js` or `.py`)
2. Create `configs/corpora/myfile.toml`:

```toml
[files]
directory = "fixtures"
glob      = "myfile.js"

[sample]
k    = 16   # number of functions to sample
seed = 42
```

3. Run: `python3 bench.py recall --corpus myfile --model local`

---

## lm-eval Tokenizer

lm-eval needs a tokenizer to count tokens in prompts before sending them to your local server. It downloads only the tokenizer files from HuggingFace (~a few MB) — **not** the model weights. After the first download it caches locally and works offline.

Set `lmeval_tokenizer` in your model config to a compatible HuggingFace model ID:

```toml
# Qwen models (3.5, 3.6, 2.5, etc.)
lmeval_tokenizer = "Qwen/Qwen2.5-0.5B"

# Gemma 4
lmeval_tokenizer = "google/gemma-3-27b-it"

# Llama 3.x  ← requires HuggingFace login (see note below)
lmeval_tokenizer = "meta-llama/Meta-Llama-3.1-8B"

# Mistral
lmeval_tokenizer = "mistralai/Mistral-7B-v0.1"

# DeepSeek
lmeval_tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Phi-3 / Phi-4
lmeval_tokenizer = "microsoft/Phi-3-mini-4k-instruct"
```

The key is picking a tokenizer from the **same model family** — you don't need an exact match, just the same tokenizer type. For Qwen 3.6 we use `Qwen/Qwen2.5-0.5B` because it's the smallest model in the family and the tokenizer is identical.

**For cloud APIs (Claude, OpenAI, Anthropic):** you don't need `lmeval_tokenizer` at all. Setting `runtime = "anthropic"` or `runtime = "openai"` routes lm-eval to use its native backends which handle tokenization internally.

**Llama models require a HuggingFace login** because Meta gates access. Before running lm-eval with a Llama tokenizer:
```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token when prompted
```
Get a token at https://huggingface.co/settings/tokens (free account). Then accept the Llama license at the model page on HuggingFace before downloading.

---

## Troubleshooting

**"No model loaded in LM Studio"** — Make sure a model is fully loaded in LM Studio before running. The tool pings `/v1/models` on startup and fails fast if nothing is there.

**"lm-eval not installed" / "FastAPI not installed"** — Activate your venv and run `pip install -r requirements-extended.txt`.

**lm-eval task not found** — Task names change between lm-eval versions. Check what's available: `lm_eval --tasks list | grep humaneval`. Update `configs/lmeval/*.toml` if needed.

**Speed profiler shows N/A for prefill/gen** — Expected for LM Studio and cloud APIs. Only raw llama.cpp exposes the prefill/generation split. `Overall t/s` is available for all runtimes.

**System Python blocked (PEP 668 error)** — Use a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.

**Benchmarks are very slow** — Long-context recall and lm-eval are inherently slow on local hardware. Use `--corpus http_server` instead of `jquery` for faster recall tests. For lm-eval, `coding-standard` runs faster than `coding-multilang` (fewer problems).
