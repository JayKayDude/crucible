"""Structured progress markers for the dashboard run viewer.

Each runner emits one line per step via `emit(...)`; the webapp's run stream
parser (webapp/routes/run_routes.py) pulls these out of stdout into a separate
progress channel and keeps them out of the raw log. Keep the line format stable
— it is a contract between the two.

Markers are only printed when BENCH_PROGRESS is set in the environment (the
dashboard sets it when it launches a run), so plain CLI usage stays clean.
"""
from __future__ import annotations

import json
import os

PROGRESS_PREFIX = "@@PROGRESS "

_enabled: bool | None = None


def _is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = bool(os.environ.get("BENCH_PROGRESS"))
    return _enabled


def emit(phase: str, done: int, total: int, label: str = "") -> None:
    """Print a progress marker for `phase` (recall/coding/reasoning/speed/toolcall)."""
    if not _is_enabled():
        return
    print(
        PROGRESS_PREFIX + json.dumps(
            {"phase": phase, "done": done, "total": total, "label": label}
        ),
        flush=True,
    )
