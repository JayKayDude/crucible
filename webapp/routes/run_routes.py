"""Run management endpoints for the LLM Benchmarker dashboard."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from asyncio.subprocess import PIPE, STDOUT
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# Project root: webapp/routes/ -> webapp/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_state(request: Request) -> dict:
    return request.app.state.run_state


def _toml_read(path: Path) -> dict[str, Any]:
    """Read a TOML file using stdlib tomllib."""
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _toml_write(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a TOML file using a simple hand-rolled serializer."""
    lines: list[str] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, bool):
            lines.append(f'{key} = {"true" if value else "false"}')
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Request body schema
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    model: str
    corpus: str
    suites: list[str]
    quantization: str | None = None
    architecture: str | None = None


# ---------------------------------------------------------------------------
# Background task: stream subprocess output into run_state
# ---------------------------------------------------------------------------

async def _stream_process(
    run_id: str,
    process: asyncio.subprocess.Process,
    run_state: dict,
    tmp_toml: Path | None,
) -> None:
    try:
        assert process.stdout is not None
        while True:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            run_state[run_id]["logs"].append(line)

        await process.wait()
        if process.returncode == 0:
            run_state[run_id]["status"] = "done"
        else:
            # Don't overwrite "cancelled"
            if run_state[run_id]["status"] == "running":
                run_state[run_id]["status"] = "error"
    finally:
        if tmp_toml is not None and tmp_toml.exists():
            try:
                tmp_toml.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# POST / — start a run
# ---------------------------------------------------------------------------

@router.post("")
async def start_run(body: RunRequest, request: Request) -> dict:
    run_state = _run_state(request)
    run_id = uuid.uuid4().hex

    model_name = body.model
    tmp_toml: Path | None = None

    # If quantization or architecture overrides are provided, write a temp TOML
    if body.quantization or body.architecture:
        base_toml = _PROJECT_ROOT / "configs" / "models" / f"{body.model}.toml"
        if not base_toml.exists():
            raise HTTPException(status_code=404, detail=f"Model config not found: {body.model}.toml")
        base_data = _toml_read(base_toml)
        if body.quantization:
            base_data["quantization"] = body.quantization
        if body.architecture:
            base_data["architecture"] = body.architecture
        tmp_model_name = f"{body.model}__tmp_{run_id}"
        tmp_toml = _PROJECT_ROOT / "configs" / "models" / f"{tmp_model_name}.toml"
        _toml_write(tmp_toml, base_data)
        model_name = tmp_model_name

    # Build the command string
    all_suites = {"recall", "coding", "reasoning", "speed"}
    suites = set(body.suites)

    if all_suites <= suites:
        # All four — use run-all
        cmd_str = f"{sys.executable} bench.py run-all --model {model_name} --corpus {body.corpus}"
    else:
        parts: list[str] = []
        if "recall" in suites:
            parts.append(f"{sys.executable} bench.py recall --model {model_name} --corpus {body.corpus}")
        if "coding" in suites:
            parts.append(f"{sys.executable} bench.py lmeval --suite all --model {model_name}")
        if "reasoning" in suites:
            parts.append(f"{sys.executable} bench.py lmeval --suite reasoning --model {model_name}")
        if "speed" in suites:
            parts.append(f"{sys.executable} bench.py speed --model {model_name}")
        if not parts:
            raise HTTPException(status_code=400, detail="No valid suites specified")
        cmd_str = " && ".join(parts)

    process = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=PIPE,
        stderr=STDOUT,
        cwd=str(_PROJECT_ROOT),
    )

    run_state[run_id] = {
        "process": process,
        "logs": [],
        "status": "running",
        "cmd_summary": cmd_str,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Fire-and-forget background task
    asyncio.create_task(_stream_process(run_id, process, run_state, tmp_toml))

    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# GET /active — list all run entries
# ---------------------------------------------------------------------------

@router.get("/active")
def get_active_runs(request: Request) -> list[dict]:
    run_state = _run_state(request)
    return [
        {
            "run_id": run_id,
            "status": info["status"],
            "cmd_summary": info["cmd_summary"],
            "started_at": info["started_at"],
        }
        for run_id, info in run_state.items()
    ]


# ---------------------------------------------------------------------------
# GET /{run_id}/logs — SSE log stream
# ---------------------------------------------------------------------------

@router.get("/{run_id}/logs")
async def stream_logs(run_id: str, request: Request):
    run_state = _run_state(request)

    if run_id not in run_state:
        raise HTTPException(status_code=404, detail="Run not found")

    async def generator():
        try:
            sent_index = 0
            last_heartbeat = asyncio.get_event_loop().time()

            while True:
                if await request.is_disconnected():
                    break

                info = run_state.get(run_id)
                if info is None:
                    break

                logs = info["logs"]
                # Replay / tail buffered lines
                while sent_index < len(logs):
                    line = logs[sent_index]
                    payload = json.dumps({"type": "log", "line": line})
                    yield f"data: {payload}\n\n"
                    sent_index += 1

                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= 3.0:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now

                # If process is finished and we've sent all lines, send done
                if info["status"] in ("done", "error", "cancelled") and sent_index >= len(logs):
                    yield f'data: {{"type":"done","status":"{info["status"]}"}}\n\n'
                    break

                await asyncio.sleep(0.5)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# DELETE /{run_id} — cancel a run
# ---------------------------------------------------------------------------

@router.delete("/{run_id}")
async def cancel_run(run_id: str, request: Request) -> dict:
    run_state = _run_state(request)

    if run_id not in run_state:
        raise HTTPException(status_code=404, detail="Run not found")

    info = run_state[run_id]
    process: asyncio.subprocess.Process = info["process"]
    try:
        process.terminate()
    except ProcessLookupError:
        pass  # Already exited
    info["status"] = "cancelled"

    return {"ok": True}
