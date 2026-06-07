"""API routes for the LLM Benchmarker dashboard."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from bench.db import (
    delete_runs,
    get_db,
    patch_run_meta,
    query_filter_options,
    query_lmeval_leaderboard,
    query_overview,
    query_quant_impact,
    query_recall_depth,
    query_recall_leaderboard,
    query_run_detail,
    query_speed_curves,
)


class _RunRef(BaseModel):
    type: str
    run_id: str

class _BulkRunBody(BaseModel):
    runs: list[_RunRef]

class _BulkMetaBody(BaseModel):
    runs: list[_RunRef]
    quantization: str | None = None
    hardware: str | None = None

_LAST_UPDATED_SQL = """
SELECT MAX(ts) AS ts FROM (
    SELECT MAX(created_at) AS ts FROM recall_runs
    UNION ALL SELECT MAX(created_at) FROM lmeval_runs
    UNION ALL SELECT MAX(created_at) FROM speed_runs
)
"""

router = APIRouter()


def _db(request: Request):
    """Get DB connection from app state."""
    db_path: Path = request.app.state.db_path
    return get_db(db_path)


# ---------------------------------------------------------------------------
# Filter options (populates all dropdowns)
# ---------------------------------------------------------------------------

@router.get("/filter-options")
def get_filter_options(request: Request) -> dict[str, list]:
    conn = _db(request)
    try:
        return query_filter_options(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Overview (radar chart)
# ---------------------------------------------------------------------------

@router.get("/overview")
def get_overview(
    request: Request,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        rows = query_overview(conn)
        # Filter client-side dimensions if specified
        if runtime:
            rows = [r for r in rows if r.get("runtime") == runtime]
        if quantization:
            rows = [r for r in rows if r.get("quantization") == quantization]
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recall benchmark
# ---------------------------------------------------------------------------

@router.get("/recall/leaderboard")
def get_recall_leaderboard(
    request: Request,
    corpus: str | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
    architecture: str | None = None,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        rows = query_recall_leaderboard(conn, corpus=corpus, runtime=runtime, quantization=quantization)
        if architecture:
            rows = [r for r in rows if r.get("architecture") == architecture]
        return rows
    finally:
        conn.close()


@router.get("/recall/depth")
def get_recall_depth(
    request: Request,
    config_name: str,
    quantization: str,
    corpus: str,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        return query_recall_depth(conn, config_name, quantization, corpus)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# lm-eval benchmark
# ---------------------------------------------------------------------------

@router.get("/lmeval/leaderboard")
def get_lmeval_leaderboard(
    request: Request,
    suite: str | None = None,
    task: str | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        return query_lmeval_leaderboard(
            conn, suite=suite, task=task, runtime=runtime, quantization=quantization
        )
    finally:
        conn.close()


@router.get("/lmeval/multilang")
def get_lmeval_multilang(
    request: Request,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    """Multi-language heatmap data: one row per model with a column per language."""
    conn = _db(request)
    try:
        # Fetch all multilang results (MultiPL-E tasks)
        multilang_tasks = ["multiple_js", "multiple_ts", "multiple_java",
                           "multiple_cpp", "multiple_rs", "multiple_go", "multiple_py"]
        all_rows = query_lmeval_leaderboard(
            conn, suite="coding-multilang", runtime=runtime, quantization=quantization
        )
        # Pivot: group by (config_name, quantization) → dict of lang → score
        pivot: dict[tuple, dict] = {}
        for row in all_rows:
            key = (row["model_name"], row.get("quantization"))
            if key not in pivot:
                pivot[key] = {
                    "model_name": row["model_name"],
                    "config_name": row["config_name"],
                    "quantization": row.get("quantization"),
                    "runtime": row.get("runtime"),
                }
            task = row["task"]
            lang = task.replace("multiple_", "")  # "multiple_js" → "js"
            if task in multilang_tasks and "pass@1" in row.get("metric", ""):
                pivot[key][lang] = row["value"]
        return list(pivot.values())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Quant impact
# ---------------------------------------------------------------------------

@router.get("/quant-impact")
def get_quant_impact(
    request: Request,
    model_config: str,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        return query_quant_impact(conn, model_config)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Speed profiler
# ---------------------------------------------------------------------------

@router.get("/speed/curves")
def get_speed_curves(
    request: Request,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        return query_speed_curves(conn, runtime=runtime, quantization=quantization)
    finally:
        conn.close()


@router.get("/speed/comparison")
def get_speed_comparison(
    request: Request,
    context_tokens: int = 8192,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    """Speed bar chart at a fixed context size."""
    conn = _db(request)
    try:
        all_curves = query_speed_curves(conn, runtime=runtime, quantization=quantization)
        # Find closest available context size for each model
        result = {}
        for row in all_curves:
            key = (row["model_name"], row.get("quantization"), row.get("hardware", ""))
            current_dist = abs(row["context_tokens"] - context_tokens)
            if key not in result or abs(result[key]["context_tokens"] - context_tokens) > current_dist:
                result[key] = row
        return list(result.values())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

@router.get("/runs")
def get_runs(
    request: Request,
    type: str | None = None,
    model_name: str | None = None,
    hardware: str | None = None,
    quantization: str | None = None,
    page: int = 0,
    page_size: int = 50,
) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        offset = page * page_size

        def _where() -> tuple[str, list]:
            clauses: list[str] = []
            params: list = []
            if model_name is not None:
                clauses.append("mc.model_name = ?")
                params.append(model_name)
            if hardware is not None:
                clauses.append("mc.hardware = ?")
                params.append(hardware)
            if quantization is not None:
                clauses.append("mc.quantization = ?")
                params.append(quantization)
            return ("AND " + " AND ".join(clauses) if clauses else ""), params

        wsql, wp = _where()

        if type == "recall":
            rows = conn.execute(
                f"""SELECT rr.run_id, 'recall' AS type, mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware, rr.corpus, rr.created_at, rr.n_runs
                    FROM recall_runs rr JOIN model_configs mc ON mc.id=rr.model_config_id
                    WHERE 1=1 {wsql}
                    ORDER BY rr.created_at DESC LIMIT ? OFFSET ?""",
                (*wp, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        if type == "lmeval":
            rows = conn.execute(
                f"""SELECT lr.run_id, 'lmeval' AS type, mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware, lr.task_suite, lr.created_at
                    FROM lmeval_runs lr JOIN model_configs mc ON mc.id=lr.model_config_id
                    WHERE 1=1 {wsql}
                    ORDER BY lr.created_at DESC LIMIT ? OFFSET ?""",
                (*wp, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        if type == "speed":
            rows = conn.execute(
                f"""SELECT sr.run_id, 'speed' AS type, mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware, sr.created_at
                    FROM speed_runs sr JOIN model_configs mc ON mc.id=sr.model_config_id
                    WHERE 1=1 {wsql}
                    ORDER BY sr.created_at DESC LIMIT ? OFFSET ?""",
                (*wp, page_size, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        # type is None — merge all three tables
        rows = conn.execute(
            f"""SELECT * FROM (
                    SELECT rr.run_id, 'recall' AS type, mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware,
                           rr.corpus AS corpus, NULL AS task_suite, rr.created_at, rr.n_runs
                    FROM recall_runs rr JOIN model_configs mc ON mc.id=rr.model_config_id
                    WHERE 1=1 {wsql}
                    UNION ALL
                    SELECT lr.run_id, 'lmeval', mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware,
                           NULL, lr.task_suite, lr.created_at, NULL
                    FROM lmeval_runs lr JOIN model_configs mc ON mc.id=lr.model_config_id
                    WHERE 1=1 {wsql}
                    UNION ALL
                    SELECT sr.run_id, 'speed', mc.config_name, mc.model_name,
                           mc.quantization, mc.hardware,
                           NULL, NULL, sr.created_at, NULL
                    FROM speed_runs sr JOIN model_configs mc ON mc.id=sr.model_config_id
                    WHERE 1=1 {wsql}
                ) ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (*wp, *wp, *wp, page_size, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.get("/runs/{run_type}/{run_id}")
def get_run_detail(request: Request, run_type: str, run_id: str) -> list[dict[str, Any]]:
    conn = _db(request)
    try:
        return query_run_detail(conn, run_type, run_id)
    finally:
        conn.close()


@router.delete("/runs/bulk")
def bulk_delete_runs(request: Request, body: _BulkRunBody) -> dict:
    conn = _db(request)
    try:
        delete_runs(conn, [r.model_dump() for r in body.runs])
        return {"ok": True}
    finally:
        conn.close()


@router.patch("/runs/bulk-meta")
def bulk_patch_run_meta(request: Request, body: _BulkMetaBody) -> dict[str, Any]:
    conn = _db(request)
    try:
        patch_run_meta(conn, [r.model_dump() for r in body.runs], body.quantization, body.hardware)
        from bench.db import query_filter_options
        opts = query_filter_options(conn)
        return {"ok": True, "model_quants": opts["model_quants"]}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Live update endpoints
# ---------------------------------------------------------------------------

@router.get("/last-updated")
def get_last_updated(request: Request) -> dict:
    conn = _db(request)
    try:
        row = conn.execute(_LAST_UPDATED_SQL).fetchone()
        return {"ts": row["ts"] or ""}
    finally:
        conn.close()


@router.get("/events")
async def sse_events(request: Request):
    """Server-Sent Events stream — pushes an update event when new benchmark results land."""
    db_path: Path = request.app.state.db_path

    async def generator():
        last_ts = None
        try:
            while True:
                if await request.is_disconnected():
                    break
                conn = get_db(db_path)
                try:
                    row = conn.execute(_LAST_UPDATED_SQL).fetchone()
                    current_ts = row["ts"] or ""
                finally:
                    conn.close()

                if last_ts is None:
                    last_ts = current_ts
                elif current_ts != last_ts:
                    last_ts = current_ts
                    yield f'data: {{"type":"update","ts":"{current_ts}"}}\n\n'

                yield ": heartbeat\n\n"
                await asyncio.sleep(5)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
