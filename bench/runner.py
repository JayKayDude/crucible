"""Orchestrate a full benchmark run: extract targets, query the model, score, report."""
from __future__ import annotations

import json
import statistics as _stats
import uuid
from dataclasses import dataclass
from pathlib import Path

from .client import ClientConfig, chat_complete
from .extract import Source, extract, load_source_glob, stratified_sample
from .report import render_function, render_summary
from .scorer import FunctionScore, score
from .timing import complete_with_timing, warmup as kv_warmup, TimingResult


# Keeping the file FIRST and the tiny task suffix LAST is deliberate:
# llama.cpp / LM Studio / Ollama all reuse the KV cache for common prefix tokens,
# so across the 16 queries only the tail re-processes. Move the file and the
# cache is invalidated every request.
PROMPT_TEMPLATE = (
    "{file_contents}\n"
    "\n"
    "---\n"
    "\n"
    "Task: reproduce verbatim the first {n} lines of the body of the function named "
    "`{name}`{file_qualifier} from the source above — i.e., the {n} lines {anchor_phrase}.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY those lines, one per line, in original order.\n"
    "- Preserve original indentation and characters exactly.\n"
    "- Do NOT output the function signature or the line containing `{signature_marker}`.\n"
    "- Do NOT add commentary, line numbers, or markdown code fences.\n"
    "- If there are blank lines in the body, include them as blank lines.\n"
    "{thinking_suffix}"
)
# Per-language anchor phrasing — the source has no opening brace in Python,
# so saying "following the opening brace" confuses the model and produces
# off-by-N-line drift (emits the signature line, emits class-attr lines before
# the def, etc.). Pin the anchor to a marker the language actually has.
ANCHOR_PHRASE = {
    "js": "starting immediately after the line containing `function {name}(` "
          "or the assignment that introduces it (the line with the opening "
          "brace `{{`)",
    "py": "starting with the first body line after the `def {name}(...):` "
          "signature (including the docstring if present)",
}
SIGNATURE_MARKER = {
    "js": "function {name}(",
    "py": "def {name}(",
}
# Qwen3 (and other reasoning-enabled models) treat `/no_think` as a directive
# to skip chain-of-thought. Ignored by non-reasoning models. For a pure recall
# benchmark, reasoning wastes tokens and risks drift — so suppress by default.
NO_THINK_SUFFIX = "\n/no_think\n"


@dataclass
class _Run:
    function: str
    source_path: str | None
    prompt_chars: int
    response: str
    latency_s: float
    error: str | None = None


def _build_prompt(target, text: str, multi_file: bool, suppress_thinking: bool) -> str:
    anchor = ANCHOR_PHRASE[target.language].format(name=target.name)
    sig_marker = SIGNATURE_MARKER[target.language].format(name=target.name)
    file_qualifier = (
        f" in file `{target.source_path}`" if multi_file and target.source_path else ""
    )
    return PROMPT_TEMPLATE.format(
        file_contents=text,
        name=target.name,
        file_qualifier=file_qualifier,
        n=len(target.primary_lines),
        anchor_phrase=anchor,
        signature_marker=sig_marker,
        thinking_suffix=NO_THINK_SUFFIX if suppress_thinking else "",
    )


def _preflight_context_check(prompt: str, cfg) -> str | None:
    """Send the actual prompt with max_tokens=1 to detect context-too-small.

    Returns None on success, an error message string otherwise. Cheap because
    no real generation happens — the model only ingests the prompt and emits
    a single token. As a side benefit it warms the server's prefix KV cache
    for the rest of the run.

    Inherits the full request shape from `cfg` (so flags like
    `use_max_completion_tokens`, `reasoning_effort`, `prefill_no_think`,
    and `stop` apply) — otherwise the probe and the real queries would hit
    different server-side validation paths.

    `max_tokens=16` (not 1): some hosted APIs reject very small budgets
    with "Could not finish the message" before even processing the prompt.
    16 is still negligible cost-wise and finishes in a fraction of a second.
    """
    from dataclasses import replace

    # cfg may be a ClientConfig or a ModelConfig — get the inner ClientConfig
    client_cfg = getattr(cfg, "client", cfg)
    probe_cfg = replace(client_cfg, max_tokens=16)
    try:
        chat_complete(probe_cfg, system=None, user=prompt)
        return None
    except Exception as e:
        return str(e)


def _is_context_error(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in ("context length", "n_ctx", "n_keep", "too long", "exceeds"))


def _aggregate_runs(
    target,
    raw_runs: list[tuple[str, "TimingResult"]],
    score_fn,
    relax_indent: bool = False,
) -> dict:
    """Aggregate N (response, timing) pairs into one result dict."""
    from bench.scorer import score as _score

    scores = [_score(target.name, target.primary_lines, target.bonus_lines, resp, relax_indent=relax_indent)
              for resp, _ in raw_runs]
    timings = [t for _, t in raw_runs]

    best_idx = max(range(len(scores)), key=lambda i: scores[i].primary_matched)
    best_score = scores[best_idx]
    best_response = raw_runs[best_idx][0]

    def _mean(vals):
        v = [x for x in vals if x is not None]
        return _stats.mean(v) if v else None

    def _stdev(vals):
        v = [x for x in vals if x is not None]
        return _stats.stdev(v) if len(v) > 1 else 0.0

    return {
        "best_score": best_score,
        "best_response": best_response,
        "matched_mean":          _mean([s.primary_matched for s in scores]),
        "matched_stddev":        _stdev([s.primary_matched for s in scores]),
        "hallucinated_mean":     _mean([s.hallucinated for s in scores]),
        "pass_rate":             sum(s.passed for s in scores) / len(scores),
        "latency_mean_s":        _mean([t.latency_s for t in timings]),
        "latency_stddev_s":      _stdev([t.latency_s for t in timings]),
        "prefill_tps_mean":      _mean([t.prefill_tokens_per_s for t in timings]),
        "prefill_tps_stddev":    _stdev([x for x in [t.prefill_tokens_per_s for t in timings] if x is not None]),
        "generation_tps_mean":   _mean([t.generation_tokens_per_s for t in timings]),
        "generation_tps_stddev": _stdev([x for x in [t.generation_tokens_per_s for t in timings] if x is not None]),
        "overall_tps_mean":      _mean([t.overall_tokens_per_s for t in timings]),
        "ttft_mean_s":           _mean([t.ttft_s for t in timings]),
    }


def run_benchmark(
    source: Source,
    cfg,
    k: int = 16,
    seed: int = 42,
    dump_path: Path | None = None,
    function_filter: list[str] | None = None,
    suppress_thinking: bool = True,
    skip_preflight: bool = False,
    fail_fast_after: int | None = 2,
    relax_indent: bool = False,
) -> list[FunctionScore]:
    # Support both a raw ClientConfig (legacy CLI path) and a ModelConfig.
    # When a full ModelConfig is passed, extract the inner ClientConfig for
    # low-level API calls and keep the ModelConfig for capability flags.
    if hasattr(cfg, "client"):
        model_cfg = cfg          # ModelConfig
        client_cfg = cfg.client  # ClientConfig
    else:
        model_cfg = None
        client_cfg = cfg         # bare ClientConfig (legacy)

    # Auto-detect the actual loaded model name from the server so the DB row
    # is labeled with the real model ID (e.g. "google/gemma-4-e2b") not the
    # TOML file stem (e.g. "local").
    if model_cfg is not None and not getattr(model_cfg, "model_name", None):
        try:
            from bench.timing import detect_loaded_model
            detected = detect_loaded_model(getattr(client_cfg, "base_url", ""))
            model_cfg.model_name = detected["name"]
            if not getattr(model_cfg, "quantization", None):
                model_cfg.quantization = detected.get("quantization")
            if not getattr(model_cfg, "architecture", None):
                model_cfg.architecture = detected.get("architecture")
        except Exception:
            pass

    # Generate a unique run_id for this benchmark run
    run_id = str(uuid.uuid4())

    text = source.text
    total_lines = text.count("\n") + 1
    print(
        f"Source: {source.display_name}  ({len(text):,} chars, {total_lines:,} lines, "
        f"{len(source.files)} file(s))",
        flush=True,
    )
    print(
        f"Extracted {len(source.targets)} named functions with ≥20 body lines",
        flush=True,
    )

    if function_filter:
        wanted = {n for n in function_filter}
        chosen = [t for t in source.targets if t.name in wanted]
        missing = wanted - {t.name for t in chosen}
        if missing:
            print(f"WARNING: requested but not found: {sorted(missing)}", flush=True)
    else:
        chosen = stratified_sample(source.targets, total_lines, k=k, seed=seed)

    print(f"Selected {len(chosen)} target function(s):", flush=True)
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(
            f"  - {t.name}  line {t.start_line}  body_lines={len(t.body_lines)}{loc}",
            flush=True,
        )

    multi_file = len(source.files) > 1

    # Pre-flight: send the first real prompt with max_tokens=1 to check that
    # the loaded context is big enough. Misleading FAILs from context-too-small
    # are the easiest mistake to make with LM Studio (TTL-driven JIT reload at
    # default 4K context). Better to abort up front.
    if not skip_preflight:
        probe_prompt = _build_prompt(chosen[0], text, multi_file, suppress_thinking)
        print(
            f"\nPre-flight: probing context fit with a {len(probe_prompt):,}-char prompt "
            f"(max_tokens=1)...",
            flush=True,
        )
        err = _preflight_context_check(probe_prompt, cfg)
        if err is None:
            print("Pre-flight OK.", flush=True)
        elif _is_context_error(err):
            print(f"\n❌ pre-flight failed (context too small):\n   {err}\n", flush=True)
            print("The loaded model context is smaller than the prompt. The most common", flush=True)
            print("cause is LM Studio JIT-reloading at default 4K context after its TTL", flush=True)
            print("expired. Force-reload at the size you need:", flush=True)
            print(f"\n   lms unload {client_cfg.model}", flush=True)
            print(f"   lms load {client_cfg.model} --context-length 131072 --gpu max -y\n", flush=True)
            print("Re-run after the model is loaded. (Pass --skip-preflight to override.)", flush=True)
            raise SystemExit(2)
        else:
            print(f"\n❌ pre-flight failed: {err}\n", flush=True)
            print("The server is reachable but rejected the request for a non-context reason.", flush=True)
            print("Fix the server-side error or pass --skip-preflight to push past this check.", flush=True)
            raise SystemExit(2)

    # Determine multi-run settings
    n_runs = max(1, getattr(model_cfg, "runs_per_function", 1) if model_cfg else 1)
    if n_runs > 1 and getattr(client_cfg, "temperature", None) == 0.0:
        print(
            f"  Warning: runs_per_function={n_runs} but temperature=0.0 — results identical, only timing varies",
            flush=True,
        )

    # KV cache warmup before the main loop (only when doing multiple runs)
    if n_runs > 1 and chosen:
        warmup_prompt = _build_prompt(chosen[0], text, multi_file, suppress_thinking)
        kv_warmup(client_cfg, model_cfg, warmup_prompt)

    scores: list[FunctionScore] = []
    runs: list[_Run] = []
    all_result_dicts: list[dict] = []
    consecutive_errors = 0
    for i, t in enumerate(chosen, 1):
        prompt = _build_prompt(t, text, multi_file, suppress_thinking)
        system_msg = None
        user_msg = prompt
        print(
            f"\n[{i}/{len(chosen)}] `{t.name}` — prompt {len(prompt):,} chars, waiting on model...",
            flush=True,
        )

        request_error: str | None = None
        raw_runs: list[tuple[str, TimingResult]] = []

        for run_idx in range(n_runs):
            try:
                resp, timing = complete_with_timing(client_cfg, model_cfg or client_cfg, system_msg, user_msg)
            except Exception as e:
                request_error = str(e)
                print(f"  ERROR (run {run_idx + 1}/{n_runs}): {request_error}", flush=True)
                resp = ""
                timing = TimingResult(latency_s=0.0)
            raw_runs.append((resp, timing))

        # Use the first run's timing for the legacy latency display
        first_resp, first_timing = raw_runs[0]
        if first_resp is None:
            print(f"  response: None in {first_timing.latency_s:.1f}s", flush=True)
            raw_runs[0] = ("", first_timing)
        else:
            print(
                f"  response: {len(first_resp)} chars in {first_timing.latency_s:.1f}s"
                + (f" (x{n_runs} runs)" if n_runs > 1 else ""),
                flush=True,
            )

        # Aggregate all runs
        agg = _aggregate_runs(t, raw_runs, score, relax_indent=relax_indent)

        sc = agg["best_score"]

        # Empty content with no exception = HTTP 200 but the model produced
        # nothing. On reasoning models that's typically the CoT eating the
        # entire max_tokens budget. Treat as a non-recall error so it shows
        # as ERROR, not FAIL.
        score_error = request_error
        best_resp = agg["best_response"]
        if best_resp.strip() == "" and score_error is None:
            score_error = (
                "empty response (200 OK but no content; reasoning models often need "
                "more max_tokens — try --max-tokens 8000)"
            )
            print(f"  ⚠ {score_error}", flush=True)

        if score_error:
            sc.error = score_error
        scores.append(sc)
        runs.append(
            _Run(
                function=t.name,
                source_path=str(t.source_path) if t.source_path else None,
                prompt_chars=len(prompt),
                response=best_resp,
                latency_s=agg["latency_mean_s"] or 0.0,
                error=score_error,
            )
        )
        print(render_function(sc), flush=True)

        # Build the result dict for JSON dump and DB insertion
        result_dict = {
            "function":            sc.name,
            "source_file":         str(t.source_path) if t.source_path else None,
            "passed":              sc.passed,
            "error":               sc.error,
            "primary_matched":     sc.primary_matched,
            "primary_total":       sc.primary_total,
            "hallucinated":        sc.hallucinated,
            "bonus_matched":       sc.bonus_matched,
            "prompt_chars":        len(prompt),
            "response":            best_resp,
            # timing — keep legacy key + add mean/stddev keys
            "latency_s":           agg["latency_mean_s"],
            "latency_mean_s":      agg["latency_mean_s"],
            "latency_stddev_s":    agg["latency_stddev_s"],
            # multi-run aggregate stats
            "matched_mean":        agg["matched_mean"],
            "matched_stddev":      agg["matched_stddev"],
            "hallucinated_mean":   agg["hallucinated_mean"],
            "pass_rate":           agg["pass_rate"],
            "prefill_tps_mean":    agg["prefill_tps_mean"],
            "generation_tps_mean": agg["generation_tps_mean"],
            "overall_tps_mean":    agg["overall_tps_mean"],
            "ttft_mean_s":         agg["ttft_mean_s"],
            "start_line":          getattr(t, "start_line", None),
        }
        all_result_dicts.append(result_dict)

        # Fail-fast: if N queries in a row error, the rest will too. Bail.
        if score_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if (
            fail_fast_after is not None
            and consecutive_errors >= fail_fast_after
            and i < len(chosen)
        ):
            remaining = len(chosen) - i
            print(
                f"\n⚠ {consecutive_errors} consecutive ERROR results — aborting the "
                f"remaining {remaining} queries.",
                flush=True,
            )
            print(
                "  Same prompt size + same model + same params → same outcome. "
                "Likely fixes:",
                flush=True,
            )
            print(
                "    • Reasoning model burning the budget? bump --max-tokens (try 8000–12000)",
                flush=True,
            )
            print(
                "    • Server-side error? check logs and the per-query message above",
                flush=True,
            )
            print(
                "  Pass --no-fail-fast to disable this check and run every query anyway.",
                flush=True,
            )
            break

    if relax_indent:
        print("\n(scored with relax_indent=true — leading whitespace ignored on both sides)",
              flush=True)
    print(render_summary(scores), flush=True)

    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "files":         [str(p) for p in source.files],
            "model":         client_cfg.model,
            "base_url":      client_cfg.base_url,
            "temperature":   client_cfg.temperature,
            "max_tokens":    client_cfg.max_tokens,
            "relax_indent":  relax_indent,
            # new top-level metadata
            "run_id":        run_id,
            "model_config":  getattr(model_cfg, "name", None),
            "quantization":  getattr(model_cfg, "quantization", None),
            "architecture":  getattr(model_cfg, "architecture", None),
            "runtime":       getattr(model_cfg, "runtime", "openai-compat"),
            "hardware":      getattr(model_cfg, "hardware", None),
            "n_runs":        n_runs,
            "results":       all_result_dicts,
        }
        dump_path.write_text(json.dumps(payload, indent=2))
        print(f"\nResults dumped to {dump_path}", flush=True)

        # Write to SQLite DB alongside the JSON dump
        if model_cfg is not None:
            try:
                from bench.db import get_db, upsert_model_config, insert_recall_run, insert_recall_result

                db_path = dump_path.parent / "benchmark.db"
                conn = get_db(db_path)
                mc_id = upsert_model_config(conn, model_cfg)
                # corpus_name: derive from dump_path stem (pattern: <corpus>__<model>.json)
                stem = dump_path.stem
                corpus_name = stem.split("__")[0] if "__" in stem else stem
                insert_recall_run(conn, mc_id, corpus_name, model_cfg, run_id, str(dump_path))
                for result in all_result_dicts:
                    insert_recall_result(conn, run_id, result)
                conn.close()
                print(f"Results written to {db_path}", flush=True)
            except Exception as db_err:
                print(f"  Warning: DB write failed: {db_err}", flush=True)

    return scores


def source_from_single_file(path: Path) -> Source:
    """Convenience: build a Source from one file (for backwards-compat with the file CLI)."""
    targets = extract(path)
    text = path.read_text()
    from .extract import language_of
    return Source(files=[path], text=text, targets=targets, language=language_of(path))
