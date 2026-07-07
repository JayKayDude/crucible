"""Model config and corpus CRUD endpoints for the LLM Benchmarker dashboard."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter()

# Project root: webapp/routes/ -> webapp/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "configs" / "models"
_CORPORA_DIR  = _PROJECT_ROOT / "configs" / "corpora"
_CUSTOM_DIR   = _PROJECT_ROOT / "fixtures" / "custom"

# TOML config fields editable from the dashboard form. Saves MERGE onto the
# existing file — fields not in this list (notably `name`, the model ID sent
# to the server, and `stop`) are preserved, never dropped.
_CONFIG_FIELDS = (
    "base_url",
    "api_key",
    "api_key_file",
    "api_key_env",
    "runtime",
    "temperature",
    "max_tokens",
    "timeout",
    "quantization",
    "architecture",
    "hardware",
    "lmeval_tokenizer",
    "suppress_thinking",
    "prefill_no_think",
    "relax_indent",
    "runs_per_function",
    "stream_for_ttft",
    "reasoning_effort",
    "use_max_completion_tokens",
)

_NAME_RE = re.compile(r"[A-Za-z0-9._\-]+")


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name or not _NAME_RE.fullmatch(name) or name.startswith("."):
        raise HTTPException(status_code=400, detail=f"Invalid config name: {name!r}")
    return name


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------

def _toml_read(path: Path) -> dict[str, Any]:
    """Read a TOML file using stdlib tomllib."""
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _toml_serialize_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value == "":
            return None
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        items = [_toml_serialize_value(v) for v in value]
        if any(i is None for i in items):
            return None
        return "[" + ", ".join(items) + "]"
    return None


def _toml_write(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a TOML file using a simple hand-rolled serializer."""
    lines: list[str] = []
    for key, value in data.items():
        if value is None:
            continue
        serialized = _toml_serialize_value(value)
        if serialized is not None:
            lines.append(f"{key} = {serialized}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_model_file(path: Path) -> dict[str, Any]:
    """Parse a model TOML file and return a dict with the name and all known fields."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        raw_text = ""
    try:
        raw = _toml_read(path)
    except Exception:
        raw = {}
    result: dict[str, Any] = {"name": path.stem, "_raw_toml": raw_text}
    for field in _CONFIG_FIELDS:
        if field in raw:
            result[field] = raw[field]
    return result


def _model_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.toml"


# ---------------------------------------------------------------------------
# Save helpers
#
# Request bodies are plain dicts:
#   {"_raw_toml": "..."}          — write the raw text verbatim (validated),
#                                    preserving comments and unknown fields
#   {"_rename": "new-name", ...}  — save under a new file name, delete the old
#   {field: value, ...}           — MERGE onto the existing file: null/"" deletes
#                                    the key, absent keys are left untouched
# ---------------------------------------------------------------------------

def _write_raw_toml(target: Path, raw_toml: str) -> None:
    import tomllib
    try:
        tomllib.loads(raw_toml)
    except tomllib.TOMLDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid TOML: {exc}")
    if not raw_toml.endswith("\n"):
        raw_toml += "\n"
    target.write_text(raw_toml, encoding="utf-8")


def _merge_fields(existing: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for field in _CONFIG_FIELDS:
        if field not in body:
            continue
        value = body[field]
        if value is None or value == "":
            merged.pop(field, None)
        else:
            merged[field] = value
    return merged


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
# PUT /models/{name} — update a model config (merge, raw TOML, or rename)
# ---------------------------------------------------------------------------

@router.put("/models/{name}")
def update_model(name: str, body: dict[str, Any]) -> dict:
    path = _model_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model config not found: {name}")

    new_name = body.get("_rename")
    target = path
    if new_name and new_name != name:
        new_name = _validate_name(new_name)
        target = _model_path(new_name)
        if target.exists():
            raise HTTPException(status_code=409, detail=f"Model config already exists: {new_name}")

    raw_toml = body.get("_raw_toml")
    if raw_toml is not None:
        _write_raw_toml(target, raw_toml)
    else:
        try:
            existing = _toml_read(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Existing config is not valid TOML: {exc}")
        _toml_write(target, _merge_fields(existing, body))

    if target != path:
        path.unlink()
    return _parse_model_file(target)


# ---------------------------------------------------------------------------
# POST /models — create a new model config
# ---------------------------------------------------------------------------

@router.post("/models")
def create_model(body: dict[str, Any]) -> dict:
    name = _validate_name(str(body.get("name", "")))
    path = _model_path(name)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Model config already exists: {name}")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_toml = body.get("_raw_toml")
    if raw_toml is not None:
        _write_raw_toml(path, raw_toml)
    else:
        _toml_write(path, _merge_fields({}, body))
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
# Corpus size estimation
# ---------------------------------------------------------------------------

# Rough chars-per-token for source code; matches speed_profiler's estimate.
# Deliberately on the low side so the token count errs high (safer for a
# "will it fit in context?" judgement).
_CHARS_PER_TOKEN = 3.8


def _corpus_size(toml_path: Path) -> dict:
    """Byte + estimated-token size of the files a corpus config would load.

    Mirrors bench.extract.load_source_glob's selection (glob → sort → limit)
    but only stats file sizes; it never reads contents. est_tokens is the
    dominant term of the recall prompt (the task wording adds <1 KB)."""
    empty = {"n_files": 0, "bytes": 0, "est_tokens": 0}
    try:
        raw = _toml_read(toml_path)
    except Exception:
        return empty
    files = raw.get("files") or {}
    directory, glob = files.get("directory"), files.get("glob")
    if not directory or not glob:
        return empty
    d = Path(directory)
    if not d.is_absolute():
        d = _PROJECT_ROOT / d
    try:
        paths = sorted(p for p in d.glob(glob) if p.is_file())
    except OSError:
        return empty
    limit = files.get("limit")
    if isinstance(limit, int):
        paths = paths[:limit]
    total = sum(p.stat().st_size for p in paths)
    return {"n_files": len(paths), "bytes": total,
            "est_tokens": round(total / _CHARS_PER_TOKEN)}


# ---------------------------------------------------------------------------
# GET /corpora — list corpus configs with their size labels
# ---------------------------------------------------------------------------

@router.get("/corpora")
def list_corpora() -> list[dict]:
    if not _CORPORA_DIR.exists():
        return []
    return [
        {"name": p.stem, **_corpus_size(p)}
        for p in sorted(_CORPORA_DIR.glob("*.toml"))
    ]


# ---------------------------------------------------------------------------
# GET /corpora/custom — list user-uploaded corpus files (with sizes)
# ---------------------------------------------------------------------------

@router.get("/corpora/custom")
def list_custom_corpora() -> list[dict]:
    if not _CUSTOM_DIR.exists():
        return []
    out = []
    for f in _CUSTOM_DIR.iterdir():
        if not f.is_file():
            continue
        b = f.stat().st_size
        out.append({"name": f.stem, "filename": f.name, "bytes": b,
                    "est_tokens": round(b / _CHARS_PER_TOKEN)})
    return sorted(out, key=lambda x: x["name"])


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
