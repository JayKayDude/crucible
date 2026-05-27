"""Unified SQLite storage for all benchmark types.

All three benchmark runners (recall, lm-eval, speed) write to a single
database at results/benchmark.db. WAL mode enables concurrent reads
from the FastAPI dashboard while a benchmark is running.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS model_configs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    config_name     TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    quantization    TEXT NOT NULL DEFAULT '',
    architecture    TEXT,
    runtime         TEXT NOT NULL DEFAULT 'openai-compat',
    base_url        TEXT,
    hardware        TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(model_name, runtime, quantization)
);

CREATE TABLE IF NOT EXISTS recall_runs (
    run_id          TEXT PRIMARY KEY,
    model_config_id INTEGER NOT NULL REFERENCES model_configs(id),
    corpus          TEXT NOT NULL,
    n_runs          INTEGER NOT NULL DEFAULT 1,
    temperature     REAL,
    max_tokens      INTEGER,
    relax_indent    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    json_path       TEXT
);

CREATE TABLE IF NOT EXISTS recall_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL REFERENCES recall_runs(run_id),
    function_name           TEXT NOT NULL,
    source_file             TEXT,
    start_line              INTEGER,
    prompt_chars            INTEGER,
    primary_matched         INTEGER,
    primary_total           INTEGER,
    hallucinated            INTEGER,
    bonus_matched           INTEGER,
    passed                  INTEGER,
    error                   TEXT,
    matched_mean            REAL,
    matched_stddev          REAL,
    hallucinated_mean       REAL,
    pass_rate               REAL,
    latency_mean_s          REAL,
    latency_stddev_s        REAL,
    prefill_tps_mean        REAL,
    prefill_tps_stddev      REAL,
    generation_tps_mean     REAL,
    generation_tps_stddev   REAL,
    overall_tps_mean        REAL,
    ttft_mean_s             REAL,
    best_response           TEXT
);

CREATE TABLE IF NOT EXISTS lmeval_runs (
    run_id          TEXT PRIMARY KEY,
    model_config_id INTEGER NOT NULL REFERENCES model_configs(id),
    task_suite      TEXT NOT NULL,
    tasks_run       TEXT NOT NULL,
    lmeval_version  TEXT,
    created_at      TEXT NOT NULL,
    output_path     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lmeval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES lmeval_runs(run_id),
    task            TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL NOT NULL,
    stderr          REAL,
    n_samples       INTEGER
);

CREATE TABLE IF NOT EXISTS speed_runs (
    run_id          TEXT PRIMARY KEY,
    model_config_id INTEGER NOT NULL REFERENCES model_configs(id),
    created_at      TEXT NOT NULL,
    context_sizes   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speed_measurements (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL REFERENCES speed_runs(run_id),
    context_tokens          INTEGER NOT NULL,
    n_samples               INTEGER NOT NULL DEFAULT 3,
    prefill_tps_mean        REAL,
    prefill_tps_stddev      REAL,
    generation_tps_mean     REAL,
    generation_tps_stddev   REAL,
    overall_tps_mean        REAL,
    overall_tps_stddev      REAL,
    ttft_mean_s             REAL,
    ttft_stddev_s           REAL,
    raw_samples             TEXT
);

CREATE INDEX IF NOT EXISTS idx_recall_results_run   ON recall_results(run_id);
CREATE INDEX IF NOT EXISTS idx_lmeval_results_run   ON lmeval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_speed_measurements_run ON speed_measurements(run_id);
CREATE INDEX IF NOT EXISTS idx_recall_runs_model    ON recall_runs(model_config_id);
CREATE INDEX IF NOT EXISTS idx_lmeval_runs_model    ON lmeval_runs(model_config_id);
CREATE INDEX IF NOT EXISTS idx_speed_runs_model     ON speed_runs(model_config_id);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate model_configs unique key from (model_name, runtime) to (model_name, runtime, quantization)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_configs'"
    ).fetchone()
    if not row:
        return
    if "UNIQUE(model_name, runtime, quantization)" in row["sql"]:
        return  # already migrated

    conn.executescript("""
        PRAGMA foreign_keys=OFF;
        BEGIN;
        CREATE TABLE model_configs_new (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            config_name     TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            quantization    TEXT NOT NULL DEFAULT '',
            architecture    TEXT,
            runtime         TEXT NOT NULL DEFAULT 'openai-compat',
            base_url        TEXT,
            hardware        TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE(model_name, runtime, quantization)
        );
        INSERT INTO model_configs_new
            SELECT id, config_name, model_name, COALESCE(quantization,''),
                   architecture, runtime, base_url, hardware, created_at
            FROM model_configs;
        DROP TABLE model_configs;
        ALTER TABLE model_configs_new RENAME TO model_configs;
        COMMIT;
        PRAGMA foreign_keys=ON;
    """)


def get_db(db_path: Path) -> sqlite3.Connection:
    """Open DB, create schema if new, enable WAL and FK enforcement."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _migrate_schema(conn)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Model config registry
# ---------------------------------------------------------------------------

def upsert_model_config(conn: sqlite3.Connection, model_cfg) -> int:
    """Insert or update model_configs row. Keyed on (model_name, runtime, quantization). Returns row id."""
    model_name = getattr(model_cfg, "model_name", model_cfg.name)
    runtime = getattr(model_cfg, "runtime", "openai-compat")
    quantization = getattr(model_cfg, "quantization", None) or ""
    conn.execute(
        """
        INSERT INTO model_configs
            (config_name, model_name, quantization, architecture, runtime, base_url, hardware, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name, runtime, quantization) DO UPDATE SET
            config_name  = excluded.config_name,
            architecture = excluded.architecture,
            base_url     = excluded.base_url,
            hardware     = excluded.hardware
        """,
        (
            model_cfg.name,
            model_name,
            quantization,
            getattr(model_cfg, "architecture", None),
            runtime,
            getattr(model_cfg.client, "base_url", None) if hasattr(model_cfg, "client") else None,
            getattr(model_cfg, "hardware", None),
            _now(),
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM model_configs WHERE model_name = ? AND runtime = ? AND quantization = ?",
        (model_name, runtime, quantization),
    ).fetchone()
    return row["id"]


# ---------------------------------------------------------------------------
# Recall benchmark
# ---------------------------------------------------------------------------

def insert_recall_run(
    conn: sqlite3.Connection,
    model_config_id: int,
    corpus: str,
    cfg,
    run_id: str,
    json_path: str | None,
) -> str:
    conn.execute(
        """
        INSERT INTO recall_runs
            (run_id, model_config_id, corpus, n_runs, temperature, max_tokens,
             relax_indent, created_at, json_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            model_config_id,
            corpus,
            getattr(cfg, "runs_per_function", 1),
            getattr(cfg.client, "temperature", None) if hasattr(cfg, "client") else None,
            getattr(cfg.client, "max_tokens", None) if hasattr(cfg, "client") else None,
            int(getattr(cfg, "relax_indent", False)),
            _now(),
            json_path,
        ),
    )
    conn.commit()
    return run_id


def insert_recall_result(conn: sqlite3.Connection, run_id: str, result: dict) -> None:
    conn.execute(
        """
        INSERT INTO recall_results (
            run_id, function_name, source_file, start_line, prompt_chars,
            primary_matched, primary_total, hallucinated, bonus_matched, passed, error,
            matched_mean, matched_stddev, hallucinated_mean, pass_rate,
            latency_mean_s, latency_stddev_s, prefill_tps_mean, prefill_tps_stddev,
            generation_tps_mean, generation_tps_stddev, overall_tps_mean, ttft_mean_s,
            best_response
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?
        )
        """,
        (
            run_id,
            result.get("function"),
            result.get("source_file"),
            result.get("start_line"),
            result.get("prompt_chars"),
            result.get("primary_matched"),
            result.get("primary_total"),
            result.get("hallucinated"),
            result.get("bonus_matched"),
            int(bool(result.get("passed"))),
            result.get("error"),
            result.get("matched_mean"),
            result.get("matched_stddev"),
            result.get("hallucinated_mean"),
            result.get("pass_rate"),
            result.get("latency_mean_s") or result.get("latency_s"),
            result.get("latency_stddev_s"),
            result.get("prefill_tps_mean"),
            result.get("prefill_tps_stddev"),
            result.get("generation_tps_mean"),
            result.get("generation_tps_stddev"),
            result.get("overall_tps_mean"),
            result.get("ttft_mean_s"),
            result.get("response") or result.get("best_response"),
        ),
    )
    conn.commit()


def query_recall_leaderboard(
    conn: sqlite3.Connection,
    corpus: str | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if corpus:
        where.append("rr.corpus = ?")
        params.append(corpus)
    if runtime:
        where.append("mc.runtime = ?")
        params.append(runtime)
    if quantization:
        where.append("mc.quantization = ?")
        params.append(quantization)

    rows = conn.execute(
        f"""
        SELECT
            mc.model_name,
            mc.config_name,
            mc.quantization,
            mc.runtime,
            mc.architecture,
            rr.corpus,
            COUNT(DISTINCT rr.run_id)    AS n_runs,
            AVG(res.pass_rate)           AS pass_rate,
            AVG(res.matched_mean)        AS matched_mean,
            AVG(res.generation_tps_mean) AS generation_tps_mean,
            AVG(res.overall_tps_mean)    AS overall_tps_mean
        FROM recall_runs rr
        JOIN model_configs mc ON mc.id = rr.model_config_id
        LEFT JOIN recall_results res ON res.run_id = rr.run_id
        WHERE {" AND ".join(where)}
        GROUP BY mc.id, rr.corpus
        ORDER BY pass_rate DESC
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def query_recall_depth(
    conn: sqlite3.Connection,
    config_name: str,
    quantization: str,
    corpus: str,
) -> list[dict]:
    """Average per-function recall results across all runs for a given model+quant+corpus."""
    rows = conn.execute(
        """
        SELECT
            res.function_name,
            res.start_line,
            AVG(res.primary_matched) AS primary_matched,
            AVG(res.latency_mean_s)  AS latency_mean_s,
            AVG(res.pass_rate)       AS pass_rate
        FROM recall_results res
        JOIN recall_runs rr ON rr.run_id = res.run_id
        JOIN model_configs mc ON mc.id = rr.model_config_id
        WHERE mc.config_name = ? AND mc.quantization = ? AND rr.corpus = ?
          AND res.start_line IS NOT NULL
        GROUP BY res.function_name
        ORDER BY res.start_line
        """,
        (config_name, quantization, corpus),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# lm-eval benchmark
# ---------------------------------------------------------------------------

def insert_lmeval_run(
    conn: sqlite3.Connection,
    model_config_id: int,
    suite: str,
    tasks: list[str],
    lmeval_version: str | None,
    output_path: str,
    run_id: str | None = None,
) -> str:
    run_id = run_id or _new_run_id()
    conn.execute(
        """
        INSERT INTO lmeval_runs
            (run_id, model_config_id, task_suite, tasks_run, lmeval_version, created_at, output_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, model_config_id, suite, json.dumps(tasks), lmeval_version, _now(), output_path),
    )
    conn.commit()
    return run_id


def insert_lmeval_result(
    conn: sqlite3.Connection,
    run_id: str,
    task: str,
    metric: str,
    value: float,
    stderr: float | None,
    n_samples: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO lmeval_results (run_id, task, metric, value, stderr, n_samples)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, task, metric, value, stderr, n_samples),
    )
    conn.commit()


def query_lmeval_leaderboard(
    conn: sqlite3.Connection,
    suite: str | None = None,
    task: str | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if suite:
        where.append("lr.task_suite = ?")
        params.append(suite)
    if task:
        where.append("res.task = ?")
        params.append(task)
    if runtime:
        where.append("mc.runtime = ?")
        params.append(runtime)
    if quantization:
        where.append("mc.quantization = ?")
        params.append(quantization)

    rows = conn.execute(
        f"""
        SELECT
            mc.model_name,
            mc.config_name,
            mc.quantization,
            mc.runtime,
            mc.architecture,
            lr.task_suite,
            res.task,
            res.metric,
            AVG(res.value)      AS value,
            AVG(res.stderr)     AS stderr,
            SUM(res.n_samples)  AS n_samples
        FROM lmeval_runs lr
        JOIN model_configs mc ON mc.id = lr.model_config_id
        JOIN lmeval_results res ON res.run_id = lr.run_id
        WHERE {" AND ".join(where)}
        GROUP BY mc.id, lr.task_suite, res.task, res.metric
        ORDER BY mc.config_name, res.task
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def query_quant_impact(conn: sqlite3.Connection, model_config: str) -> list[dict]:
    """Return all metrics for a model across different quantizations."""
    rows = conn.execute(
        """
        SELECT mc.model_name, mc.quantization, res.task AS metric_source, res.metric, res.value
        FROM lmeval_runs lr
        JOIN model_configs mc ON mc.id = lr.model_config_id
        JOIN lmeval_results res ON res.run_id = lr.run_id
        WHERE mc.model_name LIKE ?
        UNION ALL
        SELECT mc.model_name, mc.quantization, 'recall_' || rr.corpus AS metric_source,
               'pass_rate', AVG(res2.pass_rate)
        FROM recall_runs rr
        JOIN model_configs mc ON mc.id = rr.model_config_id
        JOIN recall_results res2 ON res2.run_id = rr.run_id
        WHERE mc.model_name LIKE ?
        GROUP BY mc.quantization, rr.corpus
        ORDER BY mc.quantization
        """,
        (f"%{model_config}%", f"%{model_config}%"),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Speed profiler
# ---------------------------------------------------------------------------

def insert_speed_run(
    conn: sqlite3.Connection,
    model_config_id: int,
    context_sizes: list[int],
    run_id: str | None = None,
) -> str:
    run_id = run_id or _new_run_id()
    conn.execute(
        """
        INSERT INTO speed_runs (run_id, model_config_id, created_at, context_sizes)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, model_config_id, _now(), json.dumps(context_sizes)),
    )
    conn.commit()
    return run_id


def insert_speed_measurement(
    conn: sqlite3.Connection,
    run_id: str,
    context_tokens: int,
    timing_samples: list,
) -> None:
    import statistics

    def _mean(vals):
        v = [x for x in vals if x is not None]
        return statistics.mean(v) if v else None

    def _stdev(vals):
        v = [x for x in vals if x is not None]
        return statistics.stdev(v) if len(v) > 1 else 0.0

    prefill  = [t.prefill_tokens_per_s for t in timing_samples]
    gen      = [t.generation_tokens_per_s for t in timing_samples]
    overall  = [t.overall_tokens_per_s for t in timing_samples]
    ttft     = [t.ttft_s for t in timing_samples]

    raw = json.dumps([
        {"latency_s": t.latency_s, "prefill_tps": t.prefill_tokens_per_s,
         "gen_tps": t.generation_tokens_per_s, "overall_tps": t.overall_tokens_per_s,
         "ttft_s": t.ttft_s}
        for t in timing_samples
    ])

    conn.execute(
        """
        INSERT INTO speed_measurements (
            run_id, context_tokens, n_samples,
            prefill_tps_mean, prefill_tps_stddev,
            generation_tps_mean, generation_tps_stddev,
            overall_tps_mean, overall_tps_stddev,
            ttft_mean_s, ttft_stddev_s,
            raw_samples
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, context_tokens, len(timing_samples),
            _mean(prefill), _stdev(prefill),
            _mean(gen), _stdev(gen),
            _mean(overall), _stdev(overall),
            _mean(ttft), _stdev(ttft),
            raw,
        ),
    )
    conn.commit()


def query_speed_curves(
    conn: sqlite3.Connection,
    model_config_id: int | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if model_config_id:
        where.append("sr.model_config_id = ?")
        params.append(model_config_id)
    if runtime:
        where.append("mc.runtime = ?")
        params.append(runtime)
    if quantization:
        where.append("mc.quantization = ?")
        params.append(quantization)

    rows = conn.execute(
        f"""
        SELECT
            mc.model_name,
            mc.config_name,
            mc.quantization,
            mc.runtime,
            sm.context_tokens,
            AVG(sm.prefill_tps_mean)      AS prefill_tps_mean,
            AVG(sm.prefill_tps_stddev)    AS prefill_tps_stddev,
            AVG(sm.generation_tps_mean)   AS generation_tps_mean,
            AVG(sm.generation_tps_stddev) AS generation_tps_stddev,
            AVG(sm.overall_tps_mean)      AS overall_tps_mean,
            AVG(sm.ttft_mean_s)           AS ttft_mean_s
        FROM speed_runs sr
        JOIN model_configs mc ON mc.id = sr.model_config_id
        JOIN speed_measurements sm ON sm.run_id = sr.run_id
        WHERE {" AND ".join(where)}
        GROUP BY mc.id, sm.context_tokens
        ORDER BY mc.config_name, sm.context_tokens
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Cross-benchmark queries
# ---------------------------------------------------------------------------

def query_overview(conn: sqlite3.Connection) -> list[dict]:
    """One row per model_config with recall pass_rate, humaneval_plus, gsm8k,
    ifeval, and gen_tps at ~8K context. For radar chart."""
    rows = conn.execute(
        """
        SELECT
            mc.model_name,
            mc.config_name,
            mc.quantization,
            mc.runtime,
            mc.architecture,
            (SELECT AVG(res.pass_rate)
             FROM recall_runs rr2
             JOIN recall_results res ON res.run_id = rr2.run_id
             WHERE rr2.model_config_id = mc.id) AS recall_pass_rate,
            (SELECT res.value
             FROM lmeval_runs lr2
             JOIN lmeval_results res ON res.run_id = lr2.run_id
             WHERE lr2.model_config_id = mc.id
               AND res.task = 'humaneval_plus'
               AND res.metric LIKE 'pass@1%'
             ORDER BY lr2.created_at DESC LIMIT 1) AS humaneval_plus,
            (SELECT res.value
             FROM lmeval_runs lr2
             JOIN lmeval_results res ON res.run_id = lr2.run_id
             WHERE lr2.model_config_id = mc.id
               AND res.task = 'gsm8k_cot_zeroshot'
               AND res.metric LIKE '%flexible%'
             ORDER BY lr2.created_at DESC LIMIT 1) AS gsm8k,
            (SELECT res.value
             FROM lmeval_runs lr2
             JOIN lmeval_results res ON res.run_id = lr2.run_id
             WHERE lr2.model_config_id = mc.id
               AND res.task = 'ifeval'
               AND res.metric = 'prompt_level_strict_acc,none'
             ORDER BY lr2.created_at DESC LIMIT 1) AS ifeval,
            (SELECT COALESCE(sm.generation_tps_mean, sm.overall_tps_mean)
             FROM speed_runs sr2
             JOIN speed_measurements sm ON sm.run_id = sr2.run_id
             WHERE sr2.model_config_id = mc.id
               AND sm.context_tokens BETWEEN 6000 AND 10000
             ORDER BY sr2.created_at DESC LIMIT 1) AS gen_tps_8k
        FROM model_configs mc
        ORDER BY mc.config_name
        """
    ).fetchall()
    return [dict(r) for r in rows]


def query_filter_options(conn: sqlite3.Connection) -> dict:
    def _distinct(table: str, col: str) -> list:
        return [
            r[0] for r in conn.execute(
                f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col}"
            ).fetchall()
        ]

    model_quants = [
        {"model_name": r[0], "quantization": r[1]}
        for r in conn.execute(
            "SELECT DISTINCT model_name, quantization FROM model_configs ORDER BY model_name, quantization"
        ).fetchall()
    ]
    return {
        "corpora":       _distinct("recall_runs", "corpus"),
        "suites":        _distinct("lmeval_runs", "task_suite"),
        "tasks":         _distinct("lmeval_results", "task"),
        "runtimes":      _distinct("model_configs", "runtime"),
        "architectures": _distinct("model_configs", "architecture"),
        "hardware":      _distinct("model_configs", "hardware"),
        "model_quants":  model_quants,
    }
