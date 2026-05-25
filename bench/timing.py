"""Per-runtime timing adapter for LLM API calls.

Replaces direct chat_complete() calls in runner.py. Extracts structured
timing data from whatever the runtime provides:
- llama.cpp: response body has timings.prompt_ms / predicted_ms
- LM Studio / OpenAI / Anthropic / cloud: usage object + wall-clock fallback
- Streaming mode: SSE chunks, measure TTFT on first content delta
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from bench.known_architectures import lookup_architecture


@dataclass
class TimingResult:
    latency_s: float                        # wall-clock, always present
    prompt_tokens: int | None = None        # from usage or llama timings
    completion_tokens: int | None = None
    prefill_tokens_per_s: float | None = None   # llama.cpp only
    generation_tokens_per_s: float | None = None  # llama.cpp only
    overall_tokens_per_s: float | None = None   # usage-based fallback
    ttft_s: float | None = None             # streaming only


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete_with_timing(
    client_cfg,
    model_cfg,
    system: str,
    user: str,
) -> tuple[str, TimingResult]:
    """Send a chat completion and return (response_text, TimingResult).

    Dispatches to streaming or blocking path based on model_cfg.stream_for_ttft.
    """
    if getattr(model_cfg, "stream_for_ttft", False):
        return _streaming_complete(client_cfg, model_cfg, system, user)
    return _blocking_complete(client_cfg, model_cfg, system, user)


def warmup(client_cfg, model_cfg, prompt: str) -> None:
    """Send one throwaway request to warm the KV cache. Result discarded."""
    print("  warming up KV cache...", end=" ", flush=True)
    try:
        _blocking_complete(client_cfg, model_cfg, "You are a helpful assistant.", prompt)
    except Exception:
        pass  # warmup failure is non-fatal
    print("done")


def detect_runtime(base_url: str, explicit_runtime: str | None) -> str:
    """Determine runtime type from base URL or explicit override.

    Used for lm-eval --model flag routing and capability detection.
    """
    if explicit_runtime and explicit_runtime not in ("auto", "openai-compat"):
        return explicit_runtime
    url = base_url.lower()
    if "anthropic.com" in url:
        return "anthropic"
    if "openai.com" in url:
        return "openai"
    if "127.0.0.1" in url or "://localhost" in url or "/localhost:" in url:
        return "local"
    return "openai-compat"


def detect_loaded_model(base_url: str) -> dict:
    """Detect the currently loaded model, quantization, and architecture.

    Tries LM Studio's /api/v0/models first (has explicit quantization + arch fields).
    Falls back to /v1/models + regex parsing for non-LM Studio runtimes (Ollama, llama.cpp).
    Returns dict: name, quantization (may be None), architecture (may be None).
    """
    # --- LM Studio v0 API (has explicit quantization + arch) ---
    try:
        resp = httpx.get(f"{base_url}/api/v0/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            loaded = [m for m in models if m.get("state") == "loaded"] or models[:1]
            if loaded:
                m = loaded[0]
                return {
                    "name":         m["id"],
                    "quantization": m.get("quantization"),
                    "architecture": lookup_architecture(m["id"]),
                }
    except Exception:
        pass

    # --- Ollama: /api/ps shows currently running model with quantization ---
    try:
        resp = httpx.get(f"{base_url}/api/ps", timeout=5.0)
        if resp.status_code == 200:
            running = resp.json().get("models", [])
            if running:
                m = running[0]
                quant = m.get("details", {}).get("quantization_level") or _parse_quantization(m["name"])
                return {
                    "name":         m["name"],
                    "quantization": quant,
                    "architecture": lookup_architecture(m["name"]),
                }
    except Exception:
        pass

    # --- Standard /v1/models fallback (llama.cpp, generic OpenAI-compat) ---
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach {base_url}/v1/models — is the server running? ({exc})"
        ) from exc

    if not models:
        raise RuntimeError("No model is loaded.")

    model_id: str = models[0]["id"]
    return {
        "name":         model_id,
        "quantization": _parse_quantization(model_id),  # works for llama.cpp filenames
        "architecture": lookup_architecture(model_id),
    }


# ---------------------------------------------------------------------------
# Payload builder (shared by blocking + streaming paths)
# ---------------------------------------------------------------------------

def _build_payload(client_cfg, model_cfg, messages: list[dict]) -> dict:
    """Build the OpenAI-compatible chat completions request payload."""
    # Handle thinking suppression
    processed_messages = list(messages)
    if getattr(model_cfg, "suppress_thinking", False):
        # Append /no_think to last user message
        for i in range(len(processed_messages) - 1, -1, -1):
            if processed_messages[i]["role"] == "user":
                processed_messages[i] = dict(processed_messages[i])
                if not processed_messages[i]["content"].endswith("/no_think"):
                    processed_messages[i]["content"] += " /no_think"
                break

    if getattr(model_cfg, "prefill_no_think", False):
        processed_messages = list(processed_messages) + [
            {"role": "assistant", "content": "<think>\n</think>\n\n"}
        ]

    payload: dict = {
        "model": client_cfg.model,
        "messages": processed_messages,
        "temperature": client_cfg.temperature,
        "stop": client_cfg.stop or [],
    }

    # max_tokens vs max_completion_tokens (OpenAI GPT-5+ quirk)
    if getattr(client_cfg, "use_max_completion_tokens", False):
        payload["max_completion_tokens"] = client_cfg.max_tokens
    else:
        payload["max_tokens"] = client_cfg.max_tokens

    # reasoning_effort (for models that support it)
    if getattr(client_cfg, "reasoning_effort", None):
        payload["reasoning_effort"] = client_cfg.reasoning_effort

    return payload


# ---------------------------------------------------------------------------
# Blocking path
# ---------------------------------------------------------------------------

def _blocking_complete(
    client_cfg, model_cfg, system: str, user: str
) -> tuple[str, TimingResult]:
    messages = _build_messages(system, user)
    payload = _build_payload(client_cfg, model_cfg, messages)

    headers = {"Content-Type": "application/json"}
    if getattr(client_cfg, "api_key", None):
        headers["Authorization"] = f"Bearer {client_cfg.api_key}"

    url = f"{client_cfg.base_url}/v1/chat/completions"

    t0 = time.monotonic()
    resp = httpx.post(url, json=payload, headers=headers, timeout=client_cfg.timeout)
    latency_s = time.monotonic() - t0

    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty response from model (status {resp.status_code})")

    content = choices[0]["message"].get("content") or ""
    timing = _extract_timing(data, latency_s)
    return content, timing


# ---------------------------------------------------------------------------
# Streaming path (for TTFT measurement)
# ---------------------------------------------------------------------------

def _streaming_complete(
    client_cfg, model_cfg, system: str, user: str
) -> tuple[str, TimingResult]:
    messages = _build_messages(system, user)
    payload = {**_build_payload(client_cfg, model_cfg, messages), "stream": True}

    headers = {"Content-Type": "application/json"}
    if getattr(client_cfg, "api_key", None):
        headers["Authorization"] = f"Bearer {client_cfg.api_key}"

    url = f"{client_cfg.base_url}/v1/chat/completions"

    t0 = time.monotonic()
    ttft_s: float | None = None
    chunks: list[str] = []

    with httpx.stream(
        "POST", url, json=payload, headers=headers, timeout=client_cfg.timeout
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta_content = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if delta_content:
                if ttft_s is None:
                    ttft_s = time.monotonic() - t0
                chunks.append(delta_content)

    latency_s = time.monotonic() - t0
    content = "".join(chunks)

    # Streaming doesn't give token counts — estimate from content length
    ct = max(1, len(content.split()))
    overall = ct / latency_s if latency_s > 0 else None

    return content, TimingResult(
        latency_s=latency_s,
        prompt_tokens=None,
        completion_tokens=None,
        prefill_tokens_per_s=None,
        generation_tokens_per_s=None,
        overall_tokens_per_s=overall,
        ttft_s=ttft_s,
    )


# ---------------------------------------------------------------------------
# Timing extraction from response body
# ---------------------------------------------------------------------------

def _extract_timing(data: dict, latency_s: float) -> TimingResult:
    """Extract timing info from API response. Tries llama.cpp timings first,
    falls back to usage object."""
    timings = data.get("timings")
    if timings and timings.get("prompt_n"):
        prompt_n  = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 1)
        pred_n    = timings.get("predicted_n", 0)
        pred_ms   = timings.get("predicted_ms", 1)
        return TimingResult(
            latency_s=latency_s,
            prompt_tokens=prompt_n,
            completion_tokens=pred_n,
            prefill_tokens_per_s=prompt_n / (prompt_ms / 1000) if prompt_ms else None,
            generation_tokens_per_s=pred_n / (pred_ms / 1000) if (pred_n and pred_ms) else None,
            overall_tokens_per_s=None,
            ttft_s=None,
        )

    # Fallback: usage object (LM Studio, OpenAI, Anthropic, MLX, etc.)
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    overall = ct / latency_s if (ct and latency_s > 0) else None
    return TimingResult(
        latency_s=latency_s,
        prompt_tokens=pt,
        completion_tokens=ct,
        prefill_tokens_per_s=None,
        generation_tokens_per_s=None,
        overall_tokens_per_s=overall,
        ttft_s=None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(system: str, user: str) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def _parse_quantization(model_id: str) -> str | None:
    """Extract quantization from model ID string.

    Handles patterns like: q4_k_m, Q4_K_M, q8_0, Q8_0, f16, F16, bf16, BF16,
    q4_k_xl, q3_k_s, q5_k_m, iq4_xs, etc.
    """
    match = re.search(
        r'\b(iq\d[\w]*|q\d[\w_]*|[bf]16)\b',
        model_id,
        re.IGNORECASE,
    )
    return match.group(0).upper() if match else None
