"""lm-eval subprocess wrapper.

Runs lm-eval as an external process, streams output to terminal in real-time,
parses the results JSON, and imports into the unified SQLite DB.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Suite config
# ---------------------------------------------------------------------------

@dataclass
class LMEvalSuiteConfig:
    name: str
    description: str
    tasks: list[str]
    num_fewshot: int = 0
    batch_size: int = 1
    task_limits: dict = field(default_factory=dict)  # NEW: task_name -> limit int


def load_suite(suite_name_or_path: str) -> LMEvalSuiteConfig:
    """Load a suite config from configs/lmeval/<name>.toml or a direct path."""
    import tomllib
    from pathlib import Path

    path = Path(suite_name_or_path)
    if not path.exists():
        # try configs/lmeval/<name>.toml relative to CWD
        path = Path("configs") / "lmeval" / f"{suite_name_or_path}.toml"
    if not path.exists():
        raise FileNotFoundError(f"Suite config not found: {suite_name_or_path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return LMEvalSuiteConfig(
        name=raw["suite"]["name"],
        description=raw["suite"].get("description", ""),
        tasks=raw["tasks"]["include"],
        num_fewshot=raw["tasks"].get("num_fewshot", 0),
        batch_size=raw["tasks"].get("batch_size", 1),
        task_limits=raw["tasks"].get("limits", {}),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_lmeval_suite(
    model_cfg,
    suite_cfg: LMEvalSuiteConfig,
    output_dir: Path,
    db_path: Path,
    limit: int | None = None,
) -> str:
    """Run lm-eval suite as subprocess, stream output, import results to DB.

    Returns run_id (UUID).
    """
    from bench.db import get_db, upsert_model_config, insert_lmeval_run, insert_lmeval_result
    from bench.timing import detect_runtime

    run_id = str(uuid.uuid4())

    # Determine --model flag from runtime
    runtime = detect_runtime(
        getattr(model_cfg.client, "base_url", ""),
        getattr(model_cfg, "runtime", None),
    )
    lmeval_model_flag = _runtime_to_lmeval_model(runtime)

    # Safely get client config
    client = getattr(model_cfg, "client", model_cfg)
    model_name = getattr(client, "model", getattr(model_cfg, "name", "unknown"))
    base_url   = getattr(client, "base_url", "")
    api_key    = getattr(client, "api_key", None)

    # Resolve HuggingFace tokenizer ID
    hf_tokenizer = getattr(model_cfg, "lmeval_tokenizer", None)
    if not hf_tokenizer:
        try:
            from bench.timing import detect_loaded_model
            from bench.known_architectures import lookup_tokenizer
            detected = detect_loaded_model(base_url)
            actual_name = detected["name"]
            model_cfg.model_name = actual_name
            if not getattr(model_cfg, "quantization", None):
                model_cfg.quantization = detected.get("quantization")
            if not getattr(model_cfg, "architecture", None):
                model_cfg.architecture = detected.get("architecture")
            hf_tokenizer = lookup_tokenizer(actual_name)
            if hf_tokenizer:
                print(f"  Auto-detected tokenizer: {hf_tokenizer} (for {actual_name})")
            else:
                hf_tokenizer = actual_name
                print(f"  Warning: no known tokenizer for '{actual_name}'.")
                print(f"  Set lmeval_tokenizer in your model config to fix this.")
        except Exception:
            hf_tokenizer = model_name

    max_gen_toks = 2048

    model_args_parts = [
        f"model={hf_tokenizer}",
        f"base_url={base_url}/v1/chat/completions",
        "num_concurrent=1",
        "max_retries=3",
        "tokenized_requests=False",
        "max_length=32768",
    ]

    if runtime == "anthropic":
        model_args_parts = [f"model={model_name}"]

    import sys as _sys
    _venv_candidates = [
        Path(_sys.executable).parent / "lm_eval",
        Path(__file__).parent.parent / ".venv" / "bin" / "lm_eval",
        Path.home() / "llm-benchmarker-venv" / "bin" / "lm_eval",
    ]
    lm_eval_bin = next((str(p) for p in _venv_candidates if p.exists()), "lm_eval")

    import os as _os
    env = {**_os.environ, "HF_ALLOW_CODE_EVAL": "1"}

    # ── Split mode: per-task limits differ ───────────────────────────────────
    if suite_cfg.task_limits:
        # Group tasks by their per-task limit (None = unlimited)
        groups: dict = {}
        for task in suite_cfg.tasks:
            lim = suite_cfg.task_limits.get(task)
            groups.setdefault(lim, []).append(task)

        all_task_results: dict = {}
        lmeval_version: str | None = None

        for grp_limit, grp_tasks in groups.items():
            # CLI --limit overrides per-task TOML limits (useful for quick tests)
            effective_limit = limit if limit is not None else grp_limit
            # Short readable suffix for output dir (strip common suffixes)
            suffix = "_".join(
                t.replace("_cot_zeroshot", "").replace("_cot", "")
                for t in grp_tasks
            )
            grp_dir = output_dir / f"{suite_cfg.name}__{model_cfg.name}__{run_id[:8]}__{suffix}"
            grp_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nRunning lm-eval group [{', '.join(grp_tasks)}]"
                  f"{' (limit=' + str(effective_limit) + ')' if effective_limit else ''}")
            print(f"Output: {grp_dir}")
            print("-" * 60)

            grp_cmd = _build_cmd(
                lm_eval_bin, lmeval_model_flag, model_args_parts,
                grp_tasks, suite_cfg, grp_dir, effective_limit, max_gen_toks,
            )
            _exec_subprocess(grp_cmd, env)

            grp_data = _parse_lmeval_output(grp_dir)
            all_task_results.update(grp_data.get("results", {}))
            if lmeval_version is None:
                lmeval_version = grp_data.get("versions", {}).get("lm_eval")

        # Record the first group's output dir as the canonical path
        first_suffix = "_".join(
            t.replace("_cot_zeroshot", "").replace("_cot", "")
            for t in next(iter(groups.values()))
        )
        canonical_dir = output_dir / f"{suite_cfg.name}__{model_cfg.name}__{run_id[:8]}__{first_suffix}"

    # ── Single mode: no per-task limits (all other suites) ───────────────────
    else:
        run_output_dir = output_dir / f"{suite_cfg.name}__{model_cfg.name}__{run_id[:8]}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning lm-eval suite '{suite_cfg.name}' on model '{model_cfg.name}'")
        print(f"Tasks: {', '.join(suite_cfg.tasks)}")
        print(f"Output: {run_output_dir}")
        print("-" * 60)

        cmd = _build_cmd(
            lm_eval_bin, lmeval_model_flag, model_args_parts,
            suite_cfg.tasks, suite_cfg, run_output_dir, limit, max_gen_toks,
        )
        _exec_subprocess(cmd, env)

        results_data = _parse_lmeval_output(run_output_dir)
        all_task_results = results_data.get("results", {})
        lmeval_version = results_data.get("versions", {}).get("lm_eval")
        canonical_dir = run_output_dir

    # ── Write to DB (shared by both paths) ───────────────────────────────────
    conn = get_db(db_path)
    try:
        mc_id = upsert_model_config(conn, model_cfg)
        insert_lmeval_run(
            conn, mc_id, suite_cfg.name, suite_cfg.tasks,
            lmeval_version, str(canonical_dir), run_id,
        )

        for task_name, metrics in all_task_results.items():
            for metric_key, value in metrics.items():
                if metric_key.endswith(",stderr") or metric_key.endswith(",_stderr"):
                    continue
                if "_stderr" in metric_key:
                    continue
                if metric_key.startswith("sample_len"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                stderr_key = metric_key.rsplit(",", 1)[0] + ",stderr"
                stderr_val = metrics.get(stderr_key)
                if not isinstance(stderr_val, (int, float)):
                    stderr_val = None
                insert_lmeval_result(
                    conn, run_id, task_name, metric_key,
                    float(value), stderr_val, None,
                )
    finally:
        conn.close()

    print(f"\nlm-eval suite '{suite_cfg.name}' complete. Run ID: {run_id}")
    return run_id


def _build_cmd(
    lm_eval_bin: str,
    lmeval_model_flag: str,
    model_args_parts: list[str],
    tasks: list[str],
    suite_cfg: LMEvalSuiteConfig,
    run_output_dir,
    limit: int | None,
    max_gen_toks: int,
) -> list[str]:
    """Build the lm_eval CLI command for one task group."""
    cmd = [
        lm_eval_bin,
        "--model", lmeval_model_flag,
        "--model_args", ",".join(model_args_parts),
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(suite_cfg.num_fewshot),
        "--batch_size", str(suite_cfg.batch_size),
        "--output_path", str(run_output_dir),
        "--log_samples",
        "--apply_chat_template",
        "--confirm_run_unsafe_code",
        "--gen_kwargs", f"max_gen_toks={max_gen_toks}",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if "coding" in suite_cfg.name:
        cmd += [
            "--system_instruction",
            "Output only Python code. Do not restate or repeat the function signature. No markdown, no explanations.",
        ]
    return cmd


def _exec_subprocess(cmd: list[str], env: dict) -> None:
    """Run lm_eval subprocess with live stdout streaming. Raises on non-zero exit."""
    import subprocess as _subprocess
    process = _subprocess.Popen(
        cmd,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"lm_eval exited with code {process.returncode}. "
            f"Check output above for details."
        )


def import_lmeval_results(
    output_path: Path,
    model_cfg,
    db_path: Path,
    suite_name: str = "imported",
) -> str:
    """Import results from an existing lm-eval output directory into the DB.

    Useful for manually-run benchmarks.
    """
    from bench.db import get_db, upsert_model_config, insert_lmeval_run, insert_lmeval_result

    output_path = Path(output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"lm-eval output directory not found: {output_path}")

    results_data = _parse_lmeval_output(output_path)
    run_id = str(uuid.uuid4())

    conn = get_db(db_path)
    try:
        mc_id = upsert_model_config(conn, model_cfg)
        lmeval_version = results_data.get("versions", {}).get("lm_eval")

        # Detect tasks from results keys
        tasks_found = list(results_data.get("results", {}).keys())

        insert_lmeval_run(
            conn, mc_id, suite_name, tasks_found,
            lmeval_version, str(output_path), run_id,
        )

        for task_name, metrics in results_data.get("results", {}).items():
            for metric_key, value in metrics.items():
                if metric_key.endswith(",stderr") or metric_key.endswith(",_stderr"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                stderr_key = metric_key.rsplit(",", 1)[0] + ",stderr"
                stderr_val = metrics.get(stderr_key)
                if not isinstance(stderr_val, (int, float)):
                    stderr_val = None
                insert_lmeval_result(
                    conn, run_id, task_name, metric_key,
                    float(value), stderr_val, None,
                )
    finally:
        conn.close()

    print(f"Imported lm-eval results from {output_path}. Run ID: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runtime_to_lmeval_model(runtime: str) -> str:
    """Map our runtime string to lm-eval's --model flag value.

    LM Studio and most local servers only support /v1/chat/completions (chat
    endpoint), not the legacy /v1/completions endpoint. Use local-chat-completions
    for those. Raw llama.cpp supports both but chat is safer.
    """
    mapping = {
        "anthropic": "anthropic-completions",
        "openai":    "openai-completions",
        "local":     "local-chat-completions",
        "lm-studio": "local-chat-completions",
        "mlx":       "local-chat-completions",
        "llama.cpp": "local-chat-completions",
    }
    result = mapping.get(runtime)
    if result is None:
        print(f"  Warning: unknown runtime '{runtime}', defaulting to local-chat-completions")
        return "local-chat-completions"
    return result


def _parse_lmeval_output(output_dir: Path) -> dict:
    """Find and parse the results JSON written by lm-eval.

    Newer lm-eval versions write into a subdirectory named after the tokenizer
    (e.g. Qwen__Qwen2.5-0.5B/results_*.json). Search recursively.
    """
    result_files = sorted(output_dir.glob("results_*.json"))
    if not result_files:
        # Try one level deep (new lm-eval behaviour)
        result_files = sorted(output_dir.glob("*/results_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No results_*.json found in {output_dir}. "
            f"Did lm-eval complete successfully?"
        )
    # Use the most recent file (last alphabetically = latest ISO timestamp)
    try:
        data = json.loads(result_files[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse lm-eval results JSON at {result_files[-1]}: {exc}. "
            f"The file may be incomplete (interrupted run)."
        ) from exc
    return data
