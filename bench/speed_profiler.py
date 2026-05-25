"""Speed profiler: measures inference speed across context sizes.

For each context size, sends a realistic prompt (sliced from real source code),
warms up the KV cache, then takes N timed samples. Produces a speed curve
showing how prefill and generation tokens/sec degrade with context length.
"""
from __future__ import annotations

import statistics
import uuid
from pathlib import Path


DEFAULT_CONTEXT_SIZES = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
_CHARS_PER_TOKEN_EST = 3.8   # rough estimate for code content
_FIXTURE_PATH = Path("fixtures/jquery.js")
_FIXTURE_FALLBACK = Path("fixtures/http_server.py")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_speed(
    model_cfg,
    db_path: Path,
    context_sizes: list[int] | None = None,
    n_samples: int = 3,
) -> str:
    """Measure inference speed across context sizes. Returns run_id."""
    from bench.timing import complete_with_timing, warmup as kv_warmup
    from bench.db import get_db, upsert_model_config, insert_speed_run, insert_speed_measurement

    sizes = context_sizes or DEFAULT_CONTEXT_SIZES
    fixture_text = _load_fixture()

    # Pre-flight: verify server is up
    _preflight(model_cfg)

    run_id = str(uuid.uuid4())
    client_cfg = model_cfg.client if hasattr(model_cfg, "client") else model_cfg

    # Auto-detect actual loaded model name so DB is labeled correctly
    if not getattr(model_cfg, "model_name", None):
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

    print(f"\nSpeed profiling '{model_cfg.name}' across {len(sizes)} context sizes")
    print(f"Samples per size: {n_samples}")
    print(f"{'Context':>10}  {'Prefill t/s':>12}  {'Gen t/s':>10}  {'Overall t/s':>12}  {'TTFT':>8}")
    print("-" * 60)

    measurements: list[dict] = []

    for ctx_size in sizes:
        prompt = _build_prompt_of_size(fixture_text, ctx_size)
        actual_tokens = int(len(prompt) / _CHARS_PER_TOKEN_EST)

        # KV cache warmup — discarded
        kv_warmup(client_cfg, model_cfg, prompt)

        # Timed samples
        samples = []
        for _ in range(n_samples):
            try:
                _, timing = complete_with_timing(
                    client_cfg, model_cfg,
                    system="You are a code assistant.",
                    user=prompt,
                )
                samples.append(timing)
            except Exception as exc:
                print(f"  ERROR (sample) at {ctx_size} tokens: {exc}")
                continue  # try remaining samples; if all fail, samples stays empty

        if not samples:
            print(f"{ctx_size:>10}  {'ERROR':>12}")
            continue

        measurements.append({"context_tokens": actual_tokens, "samples": samples})

        # Print progress row
        prefill = _safe_mean([s.prefill_tokens_per_s for s in samples])
        gen     = _safe_mean([s.generation_tokens_per_s for s in samples])
        overall = _safe_mean([s.overall_tokens_per_s for s in samples])
        ttft    = _safe_mean([s.ttft_s for s in samples])

        print(
            f"{actual_tokens:>10}  "
            f"{_fmt(prefill):>12}  "
            f"{_fmt(gen):>10}  "
            f"{_fmt(overall):>12}  "
            f"{_fmt_ttft(ttft):>8}"
        )

    # Write to DB
    conn = get_db(db_path)
    try:
        mc_id = upsert_model_config(conn, model_cfg)
        insert_speed_run(conn, mc_id, sizes, run_id)
        for m in measurements:
            insert_speed_measurement(conn, run_id, m["context_tokens"], m["samples"])
    finally:
        conn.close()

    print(f"\nSpeed profile complete. Run ID: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture() -> str:
    """Load the largest available fixture file for realistic prompts."""
    for path in (_FIXTURE_PATH, _FIXTURE_FALLBACK):
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(
        f"No fixture file found. Expected {_FIXTURE_PATH} or {_FIXTURE_FALLBACK}"
    )


def _build_prompt_of_size(fixture_text: str, target_tokens: int) -> str:
    """Slice fixture text to approximately target_tokens tokens, append task."""
    target_chars = int(target_tokens * _CHARS_PER_TOKEN_EST)
    # Leave ~50 tokens for the task question at the end
    content_chars = max(0, target_chars - int(50 * _CHARS_PER_TOKEN_EST))
    if len(fixture_text) < content_chars:
        actual = int(len(fixture_text) / _CHARS_PER_TOKEN_EST)
        print(
            f"  Warning: fixture too short for {target_tokens}-token prompt "
            f"(have ~{actual} tokens). Result may not reflect true context size."
        )
    content = fixture_text[:content_chars]
    return f"{content}\n\n---\nSummarize the above code in one sentence."


def _preflight(model_cfg) -> None:
    """Send a minimal request to verify the server is up before starting."""
    import httpx
    client_cfg = model_cfg.client if hasattr(model_cfg, "client") else model_cfg
    url = f"{client_cfg.base_url}/v1/models"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach model server at {client_cfg.base_url} — "
            f"is it running? ({exc})"
        ) from exc


def _safe_mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return statistics.mean(v) if v else None


def _fmt(val: float | None) -> str:
    return f"{val:.1f}" if val is not None else "N/A"


def _fmt_ttft(val: float | None) -> str:
    return f"{val*1000:.0f}ms" if val is not None else "N/A"
