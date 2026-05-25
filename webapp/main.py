"""FastAPI application for the LLM Benchmarker dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.routes.api import router as api_router

_HERE = Path(__file__).parent
STATIC_DIR = _HERE / "static"

app = FastAPI(title="LLM Benchmarker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{path:path}")
def spa_fallback(path: str):
    """Return index.html for any unmatched path (SPA routing)."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"error": "Frontend not built yet"}


def serve(db_path: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the dashboard server."""
    import uvicorn
    app.state.db_path = db_path
    print(f"Dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
