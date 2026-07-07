"""Minimal OpenAI-compatible chat-completions client."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from bench.timing import _build_payload  # noqa: F401 — re-exported for backwards compat


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 6000  # generous — reasoning models can spend most of this on CoT
    timeout: float = 600.0
    # Reasoning-suppression techniques. Each works on a different subset of
    # reasoning models — see configs/CONFIG_README.md for the matrix. Combine
    # freely; harmless flags are just ignored by models that don't recognize them.
    reasoning_effort: str | None = None    # sends `reasoning_effort: <value>` in request body (e.g. "none", "low")
    prefill_no_think: bool = False         # appends an assistant message containing `<think>\n</think>\n\n`
    stop: list[str] | None = None          # stop sequences sent to the server; useful for models that parrot the prompt back (Gemma 4)
    use_max_completion_tokens: bool = False  # send `max_completion_tokens` instead of `max_tokens` (required by OpenAI GPT-5 family)


def chat_complete(cfg: ClientConfig, system: str | None, user: str) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if cfg.prefill_no_think:
        # Pre-filling the assistant turn with an empty think block is the only
        # technique that reliably skips CoT on Qwen3.5/3.6 — the model sees
        # </think> and continues from there with the actual answer.
        messages.append({"role": "assistant", "content": "<think>\n</think>\n\n"})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": False,
    }
    if cfg.use_max_completion_tokens:
        payload["max_completion_tokens"] = cfg.max_tokens
    else:
        payload["max_tokens"] = cfg.max_tokens
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if cfg.stop:
        payload["stop"] = cfg.stop

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=cfg.timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
    return data["choices"][0]["message"]["content"]


def chat_complete_tools(
    cfg: ClientConfig,
    tools: list[dict],
    messages: list[dict],
    model_cfg=None,
) -> tuple[list[dict] | None, str | None]:
    """Call the model with tool schemas. Returns (tool_calls, text_content).

    tool_calls is a list of {"name": str, "arguments": dict} dicts, or None if
    the model produced a plain text response (e.g. no tool call, or irrelevance).
    Models that don't support tool calling will raise or return a text response —
    both cases result in tool_calls=None, scoring as 0.

    Passing model_cfg applies the same reasoning-suppression stack as the
    recall/speed suites (suppress_thinking, prefill_no_think, stop) so results
    stay comparable across benchmarks.
    """
    payload = {
        **_build_payload(cfg, model_cfg, messages),
        "stream": False,
        "tools": [{"type": "function", "function": t} for t in tools],
        "tool_choice": "auto",
    }

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=cfg.timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()

    msg = data["choices"][0]["message"]
    raw_calls = msg.get("tool_calls") or []
    if not raw_calls:
        return None, msg.get("content")

    parsed: list[dict] = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            import json as _json
            args = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        parsed.append({"name": name, "arguments": args})
    return parsed, msg.get("content")
