"""Model config and corpus CRUD endpoints for the LLM Benchmarker dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()

# Project root: webapp/routes/ -> webapp/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "configs" / "models"
_CORPORA_DIR  = _PROJECT_ROOT / "configs" / "corpora"
_CUSTOM_DIR   = _PROJECT_ROOT / "fixtures" / "custom"

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
# GET /models/{name}/status — check if the model's server has a model loaded
# ---------------------------------------------------------------------------

@router.get("/models/{name}/status")
def get_model_status(name: str) -> dict:
    from bench.timing import detect_loaded_model, detect_runtime
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {name}")
    cfg = _parse_model_file(path)
    base_url = cfg.get("base_url", "")
    runtime = detect_runtime(base_url, cfg.get("runtime"))
    if runtime != "local":
        return {"is_local": False, "loaded": True, "model_name": None, "quantization": None}
    try:
        info = detect_loaded_model(base_url)
        return {"is_local": True, "loaded": True, "model_name": info["name"], "quantization": info.get("quantization")}
    except Exception:
        return {"is_local": True, "loaded": False, "model_name": None, "quantization": None}


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
# GET /corpora/custom — list user-uploaded corpus files
# ---------------------------------------------------------------------------

@router.get("/corpora/custom")
def list_custom_corpora() -> list[dict]:
    if not _CUSTOM_DIR.exists():
        return []
    return sorted(
        [{"name": f.stem, "filename": f.name} for f in _CUSTOM_DIR.iterdir() if f.is_file()],
        key=lambda x: x["name"],
    )


# ---------------------------------------------------------------------------
# POST /corpora/upload — upload a source file as a new corpus
# ---------------------------------------------------------------------------

@router.post("/corpora/upload")
async def upload_corpus_file(file: UploadFile) -> dict:
    filename = Path(file.filename).name  # strip any path components
    stem = Path(filename).stem
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid filename")
    # Refuse to shadow a built-in corpus
    toml_path = _CORPORA_DIR / f"{stem}.toml"
    if toml_path.exists() and not (_CUSTOM_DIR / filename).exists():
        raise HTTPException(status_code=409, detail=f"A corpus named '{stem}' already exists")
    _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CUSTOM_DIR / filename
    dest.write_bytes(await file.read())
    toml_path.write_text(
        f'[files]\ndirectory = "fixtures/custom"\nglob = "{filename}"\nlimit = 1\n\n[sample]\nk = 16\nseed = 42\n',
        encoding="utf-8",
    )
    return {"name": stem, "filename": filename}


# ---------------------------------------------------------------------------
# DELETE /corpora/custom/{name} — remove a user-uploaded corpus file
# ---------------------------------------------------------------------------

@router.delete("/corpora/custom/{name}")
def delete_custom_corpus(name: str) -> dict:
    if not _CUSTOM_DIR.exists():
        raise HTTPException(status_code=404, detail="No custom files found")
    matches = [f for f in _CUSTOM_DIR.iterdir() if f.stem == name and f.is_file()]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Custom file not found: {name}")
    for f in matches:
        f.unlink()
    toml_path = _CORPORA_DIR / f"{name}.toml"
    if toml_path.exists():
        toml_path.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /architectures — return KNOWN_ARCHITECTURES dict
# ---------------------------------------------------------------------------

@router.get("/architectures")
def list_architectures() -> dict[str, str]:
    from bench.known_architectures import KNOWN_ARCHITECTURES
    return KNOWN_ARCHITECTURES
