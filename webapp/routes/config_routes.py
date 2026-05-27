"""Model config and corpus CRUD endpoints for the LLM Benchmarker dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Project root: webapp/routes/ -> webapp/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "configs" / "models"
_CORPORA_DIR = _PROJECT_ROOT / "configs" / "corpora"

# All recognized TOML config fields
_CONFIG_FIELDS = (
    "base_url",
    "api_key",
    "api_key_file",
    "runtime",
    "temperature",
    "max_tokens",
    "timeout",
    "quantization",
    "architecture",
    "hardware",
    "lmeval_tokenizer",
    "suppress_thinking",
    "stream_for_ttft",
    "reasoning_effort",
    "use_max_completion_tokens",
)


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------

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
            if value == "":
                continue
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_model_file(path: Path) -> dict[str, Any]:
    """Parse a model TOML file and return a dict with the name and all known fields."""
    try:
        raw = _toml_read(path)
    except Exception:
        raw = {}
    result: dict[str, Any] = {"name": path.stem}
    for field in _CONFIG_FIELDS:
        if field in raw:
            result[field] = raw[field]
    return result


def _model_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.toml"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ModelConfigBody(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    api_key_file: str | None = None
    runtime: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    quantization: str | None = None
    architecture: str | None = None
    hardware: str | None = None
    lmeval_tokenizer: str | None = None
    suppress_thinking: bool | None = None
    stream_for_ttft: bool | None = None
    reasoning_effort: str | None = None
    use_max_completion_tokens: bool | None = None


class CreateModelBody(ModelConfigBody):
    name: str


# ---------------------------------------------------------------------------
# GET /models — list all model configs (excluding __tmp_ files)
# ---------------------------------------------------------------------------

@router.get("/models")
def list_models() -> list[dict]:
    if not _MODELS_DIR.exists():
        return []
    configs = []
    for toml_path in sorted(_MODELS_DIR.glob("*.toml")):
        if "__tmp_" in toml_path.stem:
            continue
        configs.append(_parse_model_file(toml_path))
    return configs


# ---------------------------------------------------------------------------
# GET /models/{name} — get a single model config
# ---------------------------------------------------------------------------

@router.get("/models/{name}")
def get_model(name: str) -> dict:
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {name}")
    return _parse_model_file(path)


# ---------------------------------------------------------------------------
# PUT /models/{name} — update (overwrite) a model config
# ---------------------------------------------------------------------------

@router.put("/models/{name}")
def update_model(name: str, body: ModelConfigBody) -> dict:
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {name}")
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    _toml_write(path, data)
    return _parse_model_file(path)


# ---------------------------------------------------------------------------
# POST /models — create a new model config
# ---------------------------------------------------------------------------

@router.post("/models")
def create_model(body: CreateModelBody) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    path = _model_path(name)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Model config already exists: {name}")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in body.model_dump(exclude={"name"}).items() if v is not None}
    _toml_write(path, data)
    return _parse_model_file(path)


# ---------------------------------------------------------------------------
# DELETE /models/{name} — delete a model config
# ---------------------------------------------------------------------------

@router.delete("/models/{name}")
def delete_model(name: str) -> dict:
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {name}")
    # Refuse if this is the only config
    existing = [p for p in _MODELS_DIR.glob("*.toml") if "__tmp_" not in p.stem]
    if len(existing) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only model config")
    path.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /corpora — list corpus config names
# ---------------------------------------------------------------------------

@router.get("/corpora")
def list_corpora() -> list[str]:
    if not _CORPORA_DIR.exists():
        return []
    return sorted(p.stem for p in _CORPORA_DIR.glob("*.toml"))


# ---------------------------------------------------------------------------
# GET /architectures — return KNOWN_ARCHITECTURES dict
# ---------------------------------------------------------------------------

@router.get("/architectures")
def list_architectures() -> dict[str, str]:
    from bench.known_architectures import KNOWN_ARCHITECTURES
    return KNOWN_ARCHITECTURES
