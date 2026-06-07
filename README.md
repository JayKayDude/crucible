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
- [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible local server — Ollama, llama.cpp, vLLM)
- macOS / Linux

---

## Setup

### 1. Create a virtual environment

> **iCloud Drive users:** create the venv *outside* your iCloud folder to avoid a multi-hour startup caused by iCloud syncing lm-eval's 13,000+ files.

```bash
# Standard setup
cd "LLM Benchmarker"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# If your project is inside iCloud Drive
python3 -m venv ~/llm-benchmarker-venv
source ~/llm-benchmarker-venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `lm-eval` installs many dependencies and may take a few minutes.

### 3. Start the dashboard

```bash
python3 bench.py serve
```

Opens at **http://127.0.0.1:8000**

---

## Running Benchmarks

### Using the Dashboard (recommended)

1. Load a model in LM Studio (or start your local server)
2. Open the **Run** tab — the dashboard auto-detects which model is loaded and shows it next to the model config selector
3. Select your corpus, choose which benchmark suites to run, and click **▶ Run Benchmarks**
4. Live logs stream in the Run tab as benchmarks execute
5. Results appear in the dashboard tabs as they complete

The Run button is disabled if no model is detected as loaded, preventing wasted runs.

### Adding a Model Config

Model configs live in `configs/models/`. The included **`local.toml`** works for any local server — it auto-detects the loaded model name and quantization from the server API each time.

To add a config for a cloud API or an explicit per-model setup, use the **Models** tab in the dashboard or create a TOML manually:

```toml
# configs/models/my-model.toml
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
base_url    = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"   # reads from environment variable
temperature = 0.0
max_tokens  = 8000
timeout     = 120.0
```

> Use `api_key_env` (reads from env var) or `api_key_file` (reads from file path). Never hardcode keys.

### Adding a Custom Corpus

Upload any `.js` or `.py` source file via the **Custom Corpus Files** panel at the bottom of the Run tab. It appears in the corpus dropdown immediately.

Alternatively, place the file in `fixtures/` and create `configs/corpora/myfile.toml`:

```toml
[files]
directory = "fixtures"
glob      = "myfile.js"

[sample]
k    = 16   # number of functions to sample
seed = 42
```

---

## Dashboard Tabs

| Tab | What you see |
|---|---|
| **Overview** | Radar chart comparing all models across recall, coding, reasoning, and speed |
| **Coding** | HumanEval+ and MBPP+ leaderboard with error bars |
| **Reasoning** | GSM8K, BBH, IFEval leaderboard |
| **Long Context** | Recall leaderboard + depth degradation chart |
| **Speed** | Speed curves (tokens/sec vs context size) + bar chart at fixed context |
| **Run** | Launch benchmarks, view live logs, manage runs |
| **Models** | Create, edit, rename, and delete model configs |
| **Data** | Browse raw results, edit metadata, delete runs |

Use the filter bar at the top to filter by runtime, quantization, or architecture.

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

> **Note:** Benchmark time can vary drastically between models of the same reported generation speed. Factors include quantization level, architecture (MoE vs dense), hardware, and how verbose the model's chain-of-thought is. The table above reflects near-worst-case real-world measurements.

---

## Comparing Quantizations

The main reason this tool exists — empirically measure how quantization affects real performance:

1. Load Q4_K_M in LM Studio → run benchmarks via the Run tab
2. Load Q8_0 in LM Studio → run benchmarks again
3. Compare results across all tabs — `local.toml` auto-detects the quantization each time so results are stored with the correct label

---

## CLI Reference

All benchmarks can also be run directly from the terminal without the dashboard.

```bash
# Start the dashboard
python3 bench.py serve

# Long-context recall
python3 bench.py recall --corpus http_server --model local   # quick (~2 min)
python3 bench.py recall --corpus jquery --model local         # full (~10–30 min)

# Coding + reasoning benchmarks
python3 bench.py lmeval --suite coding-standard --model local
python3 bench.py lmeval --suite reasoning --model local

# Speed profiling
python3 bench.py speed --model local

# Run everything in sequence (recall → coding → reasoning → speed)
python3 bench.py run-all --model local --corpus jquery

# Import existing lm-eval output
python3 bench.py import-lmeval \
  --path results/lmeval/coding-standard__local__abc12345/ \
  --model local \
  --suite-name coding-standard
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
│   ├── timing.py               # Per-runtime timing + model detection
│   ├── db.py                   # Unified SQLite storage
│   ├── lmeval_runner.py        # lm-eval subprocess wrapper
│   ├── speed_profiler.py       # Speed curve measurement
│   └── known_architectures.py  # Model family → architecture lookup
├── configs/
│   ├── corpora/                # Recall test corpus configs
│   ├── models/                 # Model connection configs (one per model/quant)
│   └── lmeval/                 # lm-eval task suite configs
├── fixtures/                   # Source files used in recall + speed tests
├── webapp/
│   ├── main.py                 # FastAPI app
│   ├── routes/                 # API routes (benchmarks, config, run management)
│   └── static/                 # index.html, app.js, style.css
└── requirements.txt
```

---

## lm-eval Tokenizer

lm-eval needs a tokenizer to count tokens in prompts. It downloads only the tokenizer files from HuggingFace (~a few MB, not model weights) and caches them locally.

Set `lmeval_tokenizer` in your model config to a compatible HuggingFace model ID:

```toml
lmeval_tokenizer = "Qwen/Qwen2.5-0.5B"          # Qwen models
lmeval_tokenizer = "google/gemma-3-27b-it"        # Gemma 4
lmeval_tokenizer = "meta-llama/Meta-Llama-3.1-8B" # Llama 3.x (requires HF login)
lmeval_tokenizer = "mistralai/Mistral-7B-v0.1"    # Mistral
lmeval_tokenizer = "microsoft/Phi-3-mini-4k-instruct" # Phi
```

Pick a tokenizer from the **same model family** — exact match isn't required. For cloud APIs (`runtime = "anthropic"` or `"openai"`), `lmeval_tokenizer` is not needed.

**Llama models require a HuggingFace login:**
```bash
huggingface-cli login
```
Accept the Llama license at the model page on HuggingFace before downloading.

---

## Troubleshooting

**Run button is disabled** — No model detected as loaded. Load a model in LM Studio (or start your local server) and it will auto-detect within 15 seconds.

**lm-eval task not found** — Task names change between lm-eval versions. Check available tasks: `lm_eval --tasks list | grep humaneval`. Update `configs/lmeval/*.toml` if needed.

**Speed profiler shows N/A for prefill/gen** — Expected for LM Studio and cloud APIs. Only raw llama.cpp exposes the prefill/generation split. `Overall t/s` is available for all runtimes.

**Benchmarks are very slow** — Use `--corpus http_server` instead of `jquery` for faster recall tests (~2 min vs ~30 min). For lm-eval, start with `--suite coding-standard`.

**System Python blocked (PEP 668 error)** — Use a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
