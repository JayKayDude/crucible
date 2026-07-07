"""Tool-calling benchmark harness using BFCL V4 data.

Tests two stress dimensions:
  - Tool count N: accuracy as N tools are injected (correct + N-1 dummies)
  - Context padding: accuracy as filler text grows before the user message
"""
from __future__ import annotations

import json
import random
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .client import ClientConfig, chat_complete_tools


# ---------------------------------------------------------------------------
# BFCL V4 data locations
# ---------------------------------------------------------------------------

_BFCL_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

# (category_key, question_filename, answer_filename_or_None)
_CATEGORY_FILES = [
    ("simple",      "BFCL_v4_simple_python.json",  "possible_answer/BFCL_v4_simple_python.json"),
    ("multiple",    "BFCL_v4_multiple.json",        "possible_answer/BFCL_v4_multiple.json"),
    ("parallel",    "BFCL_v4_parallel.json",        "possible_answer/BFCL_v4_parallel.json"),
    ("irrelevance", "BFCL_v4_irrelevance.json",     None),  # no answer file — just check no call
]


def download_bfcl_data(fixtures_dir: Path) -> None:
    """Download BFCL V4 question + answer NDJSON files into fixtures/bfcl/."""
    out_dir = fixtures_dir / "bfcl"
    out_dir.mkdir(parents=True, exist_ok=True)

    files_to_fetch = []
    for _, q_file, a_file in _CATEGORY_FILES:
        files_to_fetch.append(q_file)
        if a_file:
            files_to_fetch.append(a_file)

    for filename in files_to_fetch:
        dest = out_dir / filename.replace("possible_answer/", "answers_")
        if dest.exists():
            print(f"  already exists: {dest.name}", flush=True)
            continue
        url = f"{_BFCL_BASE}/{filename}"
        print(f"  downloading {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  saved → {dest.name}", flush=True)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}", flush=True)
            raise


# ---------------------------------------------------------------------------
# NDJSON parsing
# ---------------------------------------------------------------------------

def _load_ndjson(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Schema normalisation
# BFCL uses non-standard type names and extra fields that break LM Studio's
# jinja template. Clean everything to standard OpenAI JSON Schema before sending.
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "dict":    "object",
    "float":   "number",
    "double":  "number",
    "long":    "integer",
    "short":   "integer",
    "byte":    "integer",
    "char":    "string",
    "tuple":   "array",
    "set":     "array",
}

# Fields allowed on a JSON Schema property node per OpenAI spec
_ALLOWED_PROPERTY_KEYS = {
    "type", "description", "enum", "items", "properties",
    "required", "default", "minimum", "maximum",
    "minLength", "maxLength", "pattern", "additionalProperties",
}


def _normalize_schema(schema: Any, is_property: bool = False) -> Any:
    if isinstance(schema, list):
        return [_normalize_schema(x) for x in schema]
    if not isinstance(schema, dict):
        return schema

    out: dict = {}
    for k, v in schema.items():
        # Strip BFCL-specific non-standard keys from property nodes
        if is_property and k not in _ALLOWED_PROPERTY_KEYS:
            continue
        if k == "type" and isinstance(v, str):
            out[k] = _TYPE_MAP.get(v, v)
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _normalize_schema(pv, is_property=True) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _normalize_schema(v, is_property=True)
        elif k == "additionalProperties":
            out[k] = _normalize_schema(v, is_property=True)
        else:
            out[k] = _normalize_schema(v)
    return out


def _normalize_function(fn: dict) -> dict:
    """Return a clean OpenAI-compatible function schema from a BFCL entry."""
    return {
        "name":        fn.get("name", ""),
        "description": fn.get("description", ""),
        "parameters":  _normalize_schema(fn.get("parameters", {"type": "object", "properties": {}})),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bfcl_category(
    fixtures_dir: Path,
    category: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Return list of {id, messages, functions, ground_truth} for a category."""
    bfcl_dir = fixtures_dir / "bfcl"

    # Find the config entry for this category
    entry = next((e for e in _CATEGORY_FILES if e[0] == category), None)
    if entry is None:
        raise ValueError(f"Unknown category: {category}")
    _, q_filename, a_filename = entry

    q_path = bfcl_dir / q_filename
    if not q_path.exists():
        raise FileNotFoundError(
            f"BFCL data not found: {q_path}\n"
            "Run: python3 bench.py toolcall --download-data"
        )

    # Load answers index
    answers: dict[str, list[dict]] = {}
    if a_filename:
        a_path = bfcl_dir / a_filename.replace("possible_answer/", "answers_")
        if a_path.exists():
            for item in _load_ndjson(a_path):
                item_id = item.get("id", "")
                gt = item.get("ground_truth", [])
                # gt is [{func_name: {arg: [vals]}}] — keep as-is for scoring
                answers[item_id] = gt

    questions = _load_ndjson(q_path)
    items = []
    for q in questions:
        item_id = q.get("id", "")
        # question is [[{role,content},...]] — take first turn
        turns = q.get("question", [])
        messages = turns[0] if turns else []
        functions = [_normalize_function(fn) for fn in q.get("function", [])]
        items.append({
            "id":           item_id,
            "messages":     messages,
            "functions":    functions,
            "ground_truth": answers.get(item_id, []),
        })

    if limit and len(items) > limit:
        rng = random.Random(seed)
        items = rng.sample(items, limit)

    return items


# ---------------------------------------------------------------------------
# Tool pool for dummy injection
# ---------------------------------------------------------------------------

def _build_tool_pool(fixtures_dir: Path) -> list[dict]:
    """Collect unique normalised function schemas from all categories."""
    seen: set[str] = set()
    pool: list[dict] = []
    for cat, _, _ in _CATEGORY_FILES:
        try:
            items = load_bfcl_category(fixtures_dir, cat, limit=None)
        except (FileNotFoundError, ValueError):
            continue
        for item in items:
            for fn in item["functions"]:
                name = fn.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    pool.append(fn)
    return pool


# ---------------------------------------------------------------------------
# Context padding
# ---------------------------------------------------------------------------

_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
)


def _make_filler(target_bytes: int) -> str:
    if target_bytes <= 0:
        return ""
    reps = (target_bytes // len(_FILLER)) + 1
    return (_FILLER * reps)[:target_bytes]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_call(
    predicted: list[dict] | None,
    ground_truth: list[dict],   # [{func_name: {arg: [acceptable_vals]}}]
    category: str,
) -> dict:
    """Returns {tool_match, arg_match, no_call}.

    All expected calls must be matched, order-independent — the parallel
    category expects several calls in one turn, so checking only the first
    would never actually validate parallel calling.

    tool_match: every expected function name appears among the predictions.
    arg_match:  every expected call is matched by a distinct predicted call
                with acceptable arguments, and there are no extra calls.
    """
    if category == "irrelevance":
        no_call = not predicted
        return {"tool_match": False, "arg_match": False, "no_call": no_call}

    expected: list[tuple[str, dict]] = [
        (next(iter(entry)), entry[next(iter(entry))])
        for entry in ground_truth
        if isinstance(entry, dict) and entry
    ]
    if not expected or not predicted:
        return {"tool_match": False, "arg_match": False, "no_call": False}

    # Greedy assignment: each expected call consumes one distinct predicted
    # call, preferring candidates whose arguments also match.
    remaining = list(predicted)
    args_ok = True
    for exp_name, exp_args in expected:
        name_hits = [p for p in remaining if p.get("name", "").lower() == exp_name.lower()]
        if not name_hits:
            return {"tool_match": False, "arg_match": False, "no_call": False}
        full = next((p for p in name_hits if _args_match(p.get("arguments", {}), exp_args)), None)
        chosen = full if full is not None else name_hits[0]
        if full is None:
            args_ok = False
        remaining.remove(chosen)

    arg_match = args_ok and not remaining  # extra spurious calls fail arg accuracy
    return {"tool_match": True, "arg_match": arg_match, "no_call": False}


def _args_match(predicted: dict, expected: dict) -> bool:
    """Each expected arg has a list of acceptable values. Empty string means optional."""
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        return False
    for key, acceptable_vals in expected.items():
        if not isinstance(acceptable_vals, list):
            acceptable_vals = [acceptable_vals]
        # If empty string is in acceptable_vals, the arg is optional — skip if missing
        has_empty = "" in acceptable_vals
        if key not in predicted:
            if has_empty:
                continue   # optional arg omitted — OK
            return False
        pred_val = predicted[key]
        non_empty_vals = [v for v in acceptable_vals if v != ""]
        if not non_empty_vals:
            continue  # fully optional
        if not any(_val_match(pred_val, exp_val) for exp_val in non_empty_vals):
            return False
    return True


def _val_match(pred: Any, exp: Any) -> bool:
    if pred == exp:
        return True
    try:
        if isinstance(exp, (int, float)) and isinstance(pred, str):
            return type(exp)(pred) == exp
        if isinstance(pred, (int, float)) and isinstance(exp, str):
            return type(pred)(exp) == pred
    except (ValueError, TypeError):
        pass
    if isinstance(pred, str) and isinstance(exp, str):
        return pred.strip().lower() == exp.strip().lower()
    if isinstance(pred, list) and isinstance(exp, list):
        return pred == exp
    return False


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_toolcall_benchmark(
    cfg,
    fixtures_dir: Path,
    db_path: Path,
    categories: list[str] | None = None,
    tool_counts: list[int] | None = None,
    context_padding_kb: list[int] | None = None,
    limit: int = 50,
    seed: int = 42,
    full: bool = False,
) -> None:
    from bench.db import get_db, upsert_model_config, insert_toolcall_run, insert_toolcall_result
    from bench.timing import detect_loaded_model

    if hasattr(cfg, "client"):
        model_cfg = cfg
        client_cfg: ClientConfig = cfg.client
    else:
        model_cfg = None
        client_cfg = cfg

    if model_cfg is not None and not getattr(model_cfg, "model_name", None):
        try:
            detected = detect_loaded_model(getattr(client_cfg, "base_url", ""))
            model_cfg.model_name = detected["name"]
            if not getattr(model_cfg, "quantization", None):
                model_cfg.quantization = detected.get("quantization")
            if not getattr(model_cfg, "architecture", None):
                model_cfg.architecture = detected.get("architecture")
        except Exception:
            pass

    cats = categories or [e[0] for e in _CATEGORY_FILES]
    t_counts = tool_counts or [5, 10, 25, 50]
    pad_sizes = context_padding_kb or ([0, 8, 32, 64] if full else [0])

    print(f"\nTool-calling benchmark — categories: {cats}", flush=True)
    print(f"Tool counts: {t_counts}  Context padding (KB): {pad_sizes}", flush=True)

    all_items: dict[str, list[dict]] = {}
    for cat in cats:
        try:
            items = load_bfcl_category(fixtures_dir, cat, limit=limit, seed=seed)
            all_items[cat] = items
            print(f"  {cat}: {len(items)} questions loaded", flush=True)
        except FileNotFoundError as e:
            print(f"  {cat}: SKIP — {e}", flush=True)

    if not all_items:
        print("No BFCL data. Run: python3 bench.py toolcall --download-data", flush=True)
        return

    pool = _build_tool_pool(fixtures_dir)
    print(f"  tool pool: {len(pool)} unique schemas for dummy injection\n", flush=True)

    run_id = str(uuid.uuid4())
    conn = get_db(db_path)
    mc_id = upsert_model_config(conn, model_cfg) if model_cfg else None
    insert_toolcall_run(conn, mc_id, "bfcl-v4", list(all_items.keys()), run_id)
    conn.close()

    from bench.progress import emit as emit_progress

    rng = random.Random(seed)
    total_cells = len(all_items) * len(t_counts) * len(pad_sizes)
    cell_idx = 0

    for cat, items in all_items.items():
        for tool_count in t_counts:
            for pad_kb in pad_sizes:
                pad_bytes = pad_kb * 1024
                filler = _make_filler(pad_bytes)

                cell_idx += 1
                n_tool_match = n_arg_match = n_no_call = n_scored = n_errors = 0
                label = f"{cat} | N={tool_count} | pad={pad_kb}KB"
                emit_progress("toolcall", cell_idx, total_cells, label)
                print(f"[{label}] {len(items)} questions ...", flush=True)

                for item in items:
                    correct_fns = item["functions"]
                    correct_names = {fn["name"] for fn in correct_fns}
                    dummy_pool = [f for f in pool if f["name"] not in correct_names]
                    need = max(0, tool_count - len(correct_fns))
                    dummies = rng.sample(dummy_pool, min(need, len(dummy_pool)))
                    tool_list = correct_fns + dummies
                    rng.shuffle(tool_list)

                    messages = list(item["messages"])
                    if filler and messages:
                        first = messages[0]
                        messages = [
                            {"role": first["role"],
                             "content": filler + "\n\n" + first.get("content", "")},
                            *messages[1:],
                        ]

                    try:
                        tool_calls, _ = chat_complete_tools(client_cfg, tool_list, messages,
                                                            model_cfg=model_cfg)
                    except Exception as e:
                        # Errors are excluded from scoring — a failed request must
                        # not count as a correct "declined to call" on irrelevance.
                        print(f"  ERROR: {e}", flush=True)
                        n_errors += 1
                        continue

                    result = _score_call(tool_calls, item["ground_truth"], cat)
                    n_tool_match += result["tool_match"]
                    n_arg_match  += result["arg_match"]
                    n_no_call    += result["no_call"]
                    n_scored     += 1

                tool_acc = n_tool_match / n_scored if n_scored else 0.0
                arg_acc  = n_arg_match  / n_scored if n_scored else 0.0
                irr_acc  = (n_no_call / n_scored) if (n_scored and cat == "irrelevance") else None

                print(
                    f"  tool={tool_acc:.1%}  arg={arg_acc:.1%}"
                    + (f"  irr={irr_acc:.1%}" if irr_acc is not None else "")
                    + (f"  ({n_errors} errored, excluded)" if n_errors else ""),
                    flush=True,
                )

                conn = get_db(db_path)
                insert_toolcall_result(conn, run_id, cat, tool_count, pad_bytes,
                                       n_scored, tool_acc, arg_acc, irr_acc)
                conn.close()

    print(f"\nDone. run_id={run_id}", flush=True)
