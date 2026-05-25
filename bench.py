#!/usr/bin/env python3
"""Positional recall benchmark — CLI entry.

Tests an LLM's ability to reproduce the first N lines of a named function
inside a large source corpus loaded into context. Measures positional recall,
not just named-entity lookup.

Source selection (extract / recall / rescore):
    --corpus NAME      a config under configs/corpora/, or a path to one
    --file PATH        single source file (.js/.mjs/.cjs/.py)

Model selection (recall only):
    --model NAME       a config under configs/models/, OR a raw model identifier
                       (raw names get sane defaults; create a config for control)

Subcommands:
    extract        list functions the extractor would test
    recall         run the positional recall benchmark (alias: run)
    run            alias for recall
    rescore        re-score a previous dump without re-querying
    lmeval         run lm-eval harness suites
    speed          measure inference speed across context sizes
    import-lmeval  import existing lm-eval output directory into DB
    run-all        run recall + all lm-eval suites + speed in sequence
    serve          start the web dashboard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


# --- source resolution ---------------------------------------------------


def _resolve_source(args: argparse.Namespace):
    """Return (Source, CorpusConfig|None) from --corpus or --file."""
    from bench.extract import load_source_glob
    from bench.runner import source_from_single_file

    if getattr(args, "corpus", None):
        from bench.config import load_corpus

        corpus = load_corpus(args.corpus)
        src = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
        return src, corpus
    if getattr(args, "file", None):
        return source_from_single_file(Path(args.file)), None
    raise SystemExit("error: pass either --corpus NAME or --file PATH")


# --- extract -------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    from bench.extract import stratified_sample

    source, corpus = _resolve_source(args)

    if args.show:
        match = next((t for t in source.targets if t.name == args.show), None)
        if match is None:
            print(f"function {args.show!r} not found")
            return 1
        loc = f"  ({match.source_path})" if match.source_path else ""
        print(f"# {match.name} — start_line={match.start_line}  body_lines={len(match.body_lines)}{loc}")
        print(f"# -- primary (first {len(match.primary_lines)}) --")
        for i, l in enumerate(match.primary_lines, 1):
            print(f"{i:>3}| {l}")
        if match.bonus_lines:
            print(f"# -- bonus (next {len(match.bonus_lines)}) --")
            for i, l in enumerate(match.bonus_lines, len(match.primary_lines) + 1):
                print(f"{i:>3}| {l}")
        return 0

    total_lines = source.text.count("\n") + 1
    print(
        f"{len(source.targets)} function(s) with ≥20 body lines across "
        f"{len(source.files)} file(s) ({len(source.text):,} chars, {total_lines:,} lines)"
    )
    k = args.k if args.k is not None else (corpus.sample_k if corpus else 16)
    seed = args.seed if args.seed is not None else (corpus.sample_seed if corpus else 42)
    if args.all:
        chosen = source.targets
    else:
        chosen = stratified_sample(source.targets, total_lines, k=k, seed=seed)
        print(f"stratified sample of {len(chosen)}:")
    for t in chosen:
        loc = f"  ({t.source_path.name})" if t.source_path else ""
        print(f"  {t.name:<40}  line={t.start_line:>6}  body_lines={len(t.body_lines)}{loc}")
    return 0


# --- recall (was: run) ---------------------------------------------------


def cmd_recall(args: argparse.Namespace) -> int:
    from bench.config import auto_dump_path, load_model
    from bench.runner import run_benchmark

    source, corpus = _resolve_source(args)

    if not args.model:
        raise SystemExit("error: --model is required (a name in configs/models/, a path, or a raw model id)")
    model, model_from_file = load_model(args.model)
    if not model_from_file:
        print(
            f"  (no model config '{args.model}' found; using as raw model identifier with defaults)",
            file=sys.stderr,
        )

    # --runs override
    if getattr(args, "runs", None):
        model.runs_per_function = args.runs

    # CLI overrides — applied on top of whichever source the model came from.
    if args.base_url:
        model.client.base_url = args.base_url
    if args.api_key:
        model.client.api_key = args.api_key
    if args.temperature is not None:
        model.client.temperature = args.temperature
    if args.max_tokens is not None:
        model.client.max_tokens = args.max_tokens
    if args.timeout is not None:
        model.client.timeout = args.timeout
    suppress_thinking = model.suppress_thinking and not args.think

    if corpus is not None:
        k = args.k if args.k is not None else corpus.sample_k
        seed = args.seed if args.seed is not None else corpus.sample_seed
    else:
        k = args.k if args.k is not None else 16
        seed = args.seed if args.seed is not None else 42

    if args.dump:
        dump_path = Path(args.dump)
    elif corpus is not None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(corpus, model, DEFAULT_RESULTS_DIR)
    else:
        # --file mode: derive corpus stem from filename
        from bench.config import CorpusConfig

        synthetic_corpus = CorpusConfig(
            name=Path(args.file).stem,
            directory=Path(args.file).parent,
            glob=Path(args.file).name,
            limit=1,
            sample_k=k,
            sample_seed=seed,
        )
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dump_path = auto_dump_path(synthetic_corpus, model, DEFAULT_RESULTS_DIR)

    # Indent-tolerant scoring: take from model config, allow CLI overrides in either direction.
    relax_indent = model.relax_indent
    if args.relax_indent:
        relax_indent = True
    if args.strict_indent:
        relax_indent = False

    fn_filter = args.function if args.function else None
    scores = run_benchmark(
        source=source,
        cfg=model,
        k=k,
        seed=seed,
        dump_path=dump_path,
        function_filter=fn_filter,
        suppress_thinking=suppress_thinking,
        skip_preflight=args.skip_preflight,
        fail_fast_after=None if args.no_fail_fast else args.fail_fast_after,
        relax_indent=relax_indent,
    )
    passed = sum(1 for s in scores if s.passed)
    return 0 if passed == len(scores) else 1


# keep the old name around so cmd_run_all can call it directly
cmd_run = cmd_recall


# --- rescore --------------------------------------------------------------


def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-score a previous run's dump without re-querying the model."""
    import json

    from bench.extract import load_source_glob
    from bench.report import render_function, render_summary
    from bench.runner import source_from_single_file
    from bench.scorer import score

    dump = json.loads(Path(args.dump).read_text())
    if args.corpus:
        from bench.config import load_corpus

        corpus = load_corpus(args.corpus)
        source = load_source_glob(corpus.directory, corpus.glob, corpus.limit)
    elif args.file:
        source = source_from_single_file(Path(args.file))
    else:
        files = dump.get("files") or ([dump["source"]] if dump.get("source") else [])
        if len(files) == 1 and Path(files[0]).is_file():
            source = source_from_single_file(Path(files[0]))
        else:
            raise SystemExit(
                "error: dump references a missing or multi-file corpus; "
                "pass --corpus NAME or --file PATH to re-locate it"
            )

    # Honor original dump's relax_indent unless overridden on the CLI.
    relax_indent = bool(dump.get("relax_indent", False))
    if args.relax_indent:
        relax_indent = True
    if args.strict_indent:
        relax_indent = False

    targets = {t.name: t for t in source.targets}
    scores = []
    for r in dump["results"]:
        t = targets.get(r["function"])
        if t is None:
            print(f"skip: {r['function']} not found in source", file=sys.stderr)
            continue
        sc = score(
            t.name, t.primary_lines, t.bonus_lines,
            r.get("response", ""), relax_indent=relax_indent,
        )
        if r.get("error"):
            sc.error = r["error"]
        scores.append(sc)
        print(render_function(sc))
    if relax_indent:
        print("\n(scored with relax_indent=true — leading whitespace ignored on both sides)")
    print(render_summary(scores))
    return 0


# --- lmeval ---------------------------------------------------------------


def cmd_lmeval(args: argparse.Namespace) -> int:
    try:
        from bench.lmeval_runner import run_lmeval_suite, load_suite
    except ImportError:
        print("Error: lm-eval not installed. Run: pip install -r requirements-extended.txt")
        return 1

    from bench.config import load_model

    model, _ = load_model(args.model)
    db_path = Path(getattr(args, "db", "results/benchmark.db"))
    output_dir = Path("results/lmeval")

    suites_to_run = ["coding-standard", "reasoning"] \
        if args.suite == "all" else [args.suite]

    limit = getattr(args, "limit", None)
    for suite_name in suites_to_run:
        suite = load_suite(suite_name)
        run_lmeval_suite(model, suite, output_dir, db_path, limit=limit)

    return 0


# --- speed ----------------------------------------------------------------


def cmd_speed(args: argparse.Namespace) -> int:
    from bench.speed_profiler import profile_speed, DEFAULT_CONTEXT_SIZES
    from bench.config import load_model

    model, _ = load_model(args.model)
    db_path = Path(getattr(args, "db", "results/benchmark.db"))

    sizes = None
    if getattr(args, "context_sizes", None):
        sizes = [int(x.strip()) for x in args.context_sizes.split(",")]

    n_samples = getattr(args, "samples", 3)
    profile_speed(model, db_path, context_sizes=sizes, n_samples=n_samples)
    return 0


# --- import-lmeval --------------------------------------------------------


def cmd_import_lmeval(args: argparse.Namespace) -> int:
    try:
        from bench.lmeval_runner import import_lmeval_results
    except ImportError:
        print("Error: lm-eval not installed. Run: pip install -r requirements-extended.txt")
        return 1

    from bench.config import load_model

    model, _ = load_model(args.model)
    db_path = Path(getattr(args, "db", "results/benchmark.db"))
    suite_name = getattr(args, "suite_name", "imported")
    import_lmeval_results(Path(args.path), model, db_path, suite_name)
    return 0


# --- run-all --------------------------------------------------------------


def cmd_run_all(args: argparse.Namespace) -> int:
    try:
        from bench.lmeval_runner import run_lmeval_suite, load_suite
    except ImportError:
        print("Error: lm-eval not installed. Run: pip install -r requirements-extended.txt")
        return 1
    from bench.speed_profiler import profile_speed
    from bench.config import load_model

    model, _ = load_model(args.model)
    db_path = Path(getattr(args, "db", "results/benchmark.db"))

    # 1. Recall
    print("\n=== Step 1/5: Recall benchmark ===")
    args_copy = type("A", (), {
        "model": args.model,
        "corpus": args.corpus,
        "file": None,
        "runs": getattr(args, "runs", None),
        "db": str(db_path),
        "base_url": None,
        "api_key": None,
        "temperature": None,
        "max_tokens": None,
        "timeout": None,
        "k": None,
        "seed": None,
        "dump": None,
        "function": None,
        "think": False,
        "skip_preflight": False,
        "fail_fast_after": 2,
        "no_fail_fast": False,
        "relax_indent": False,
        "strict_indent": False,
    })()
    cmd_recall(args_copy)

    # 2-3. lm-eval suites (multilang deferred — MultiPL-E not in lm-eval 0.4.x)
    output_dir = Path("results/lmeval")
    for i, suite_name in enumerate(["coding-standard", "reasoning"], 2):
        print(f"\n=== Step {i}/4: lm-eval {suite_name} ===")
        suite = load_suite(suite_name)
        run_lmeval_suite(model, suite, output_dir, db_path)

    # 4. Speed
    print("\n=== Step 4/4: Speed profiler ===")
    profile_speed(model, db_path)

    return 0


# --- serve ----------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from webapp.main import serve
    except ImportError:
        print("Error: FastAPI not installed. Run: pip install -r requirements-extended.txt")
        return 1

    db_path = Path(getattr(args, "db", "results/benchmark.db"))
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    print(f"Dashboard: http://{host}:{port}")
    serve(db_path=db_path, host=host, port=port)
    return 0


# --- argparse -------------------------------------------------------------


def _add_recall_args(p_recall: argparse.ArgumentParser) -> None:
    """Attach all recall/run arguments to a subparser."""
    src_grp = p_recall.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_recall.add_argument(
        "--model", required=True,
        help="model config name (configs/models/<name>.toml), a path, or a raw model identifier",
    )
    p_recall.add_argument("--base-url", default=None, help="overrides model config")
    p_recall.add_argument("--api-key", default=None)
    p_recall.add_argument("--temperature", type=float, default=None)
    p_recall.add_argument("--max-tokens", type=int, default=None)
    p_recall.add_argument("--timeout", type=float, default=None)
    p_recall.add_argument("-k", type=int, default=None, help="overrides corpus.sample.k")
    p_recall.add_argument("--seed", type=int, default=None)
    p_recall.add_argument(
        "--runs", type=int, default=None, metavar="N",
        help="override runs_per_function from model config",
    )
    p_recall.add_argument(
        "--dump", default=None,
        help="JSON path for full results (default: results/<corpus>__<model>.json)",
    )
    p_recall.add_argument("--function", action="append", help="repeatable; overrides sampling")
    p_recall.add_argument("--think", action="store_true", help="allow chain-of-thought (default: suppress)")
    p_recall.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the context-fit pre-flight probe (not recommended)",
    )
    p_recall.add_argument(
        "--fail-fast-after", type=int, default=2, metavar="N",
        help="abort the run after N consecutive ERROR results (default: 2)",
    )
    p_recall.add_argument(
        "--no-fail-fast", action="store_true",
        help="disable fail-fast; run every query even if they're all erroring",
    )
    p_recall.add_argument(
        "--relax-indent", action="store_true",
        help="ignore leading whitespace when matching (overrides model config to true)",
    )
    p_recall.add_argument(
        "--strict-indent", action="store_true",
        help="enforce verbatim indentation (overrides model config to false)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ------------------------------------------------------------
    p_ex = sub.add_parser("extract", help="list functions the extractor would test")
    src_grp = p_ex.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="corpus config name (configs/corpora/<name>.toml) or path")
    src_grp.add_argument("--file", help="single source file")
    p_ex.add_argument("-k", type=int, default=None, help="override corpus sample.k")
    p_ex.add_argument("--seed", type=int, default=None, help="override corpus sample.seed")
    p_ex.add_argument("--all", action="store_true", help="list every extracted function, not a sample")
    p_ex.add_argument("--show", metavar="NAME", help="print expected primary+bonus lines for one function")
    p_ex.set_defaults(func=cmd_extract)

    # --- recall (canonical) -------------------------------------------------
    p_recall = sub.add_parser(
        "recall",
        help="run the positional recall benchmark against an OpenAI-compatible endpoint",
    )
    _add_recall_args(p_recall)
    p_recall.set_defaults(func=cmd_recall)

    # --- run (alias for recall) ---------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="alias for recall — run the benchmark against an OpenAI-compatible endpoint",
    )
    _add_recall_args(p_run)
    p_run.set_defaults(func=cmd_recall)

    # --- rescore ------------------------------------------------------------
    p_rs = sub.add_parser("rescore", help="re-score a previous --dump without re-querying")
    p_rs.add_argument("dump", help="path to JSON dump from a prior `recall`/`run`")
    src_grp = p_rs.add_mutually_exclusive_group()
    src_grp.add_argument("--corpus", help="re-locate corpus via this config")
    src_grp.add_argument("--file", help="re-locate corpus from a single file")
    p_rs.add_argument(
        "--relax-indent", action="store_true",
        help="ignore leading whitespace when matching (overrides dump's setting)",
    )
    p_rs.add_argument(
        "--strict-indent", action="store_true",
        help="enforce verbatim indentation (overrides dump's setting)",
    )
    p_rs.set_defaults(func=cmd_rescore)

    # --- lmeval -------------------------------------------------------------
    p_lm = sub.add_parser(
        "lmeval",
        help="run lm-eval harness suites (requires lm-eval installation)",
    )
    p_lm.add_argument(
        "--suite", required=True,
        help="suite name (coding-standard | coding-multilang | reasoning | all)",
    )
    p_lm.add_argument(
        "--model", required=True,
        help="model config name or raw identifier",
    )
    p_lm.add_argument(
        "--db", default="results/benchmark.db",
        help="SQLite DB path (default: results/benchmark.db)",
    )
    p_lm.add_argument(
        "--limit", type=int, default=None,
        help="limit number of examples per task (for testing only)",
    )
    p_lm.set_defaults(func=cmd_lmeval)

    # --- speed --------------------------------------------------------------
    p_sp = sub.add_parser(
        "speed",
        help="measure inference speed across context sizes",
    )
    p_sp.add_argument(
        "--model", required=True,
        help="model config name or raw identifier",
    )
    p_sp.add_argument(
        "--context-sizes", default=None, metavar="SIZES",
        help="comma-separated list of token counts (default: 1024,4096,8192,16384,32768,65536,131072)",
    )
    p_sp.add_argument(
        "--samples", type=int, default=3, metavar="N",
        help="timed samples per context size (default: 3)",
    )
    p_sp.add_argument(
        "--db", default="results/benchmark.db",
        help="SQLite DB path (default: results/benchmark.db)",
    )
    p_sp.set_defaults(func=cmd_speed)

    # --- import-lmeval ------------------------------------------------------
    p_imp = sub.add_parser(
        "import-lmeval",
        help="import an existing lm-eval output directory into the DB",
    )
    p_imp.add_argument(
        "--path", required=True,
        help="path to lm-eval output directory containing results_*.json",
    )
    p_imp.add_argument(
        "--model", required=True,
        help="model config name or raw identifier",
    )
    p_imp.add_argument(
        "--suite-name", default="imported",
        help="label to use for this suite in the DB (default: imported)",
    )
    p_imp.add_argument(
        "--db", default="results/benchmark.db",
        help="SQLite DB path (default: results/benchmark.db)",
    )
    p_imp.set_defaults(func=cmd_import_lmeval)

    # --- run-all ------------------------------------------------------------
    p_all = sub.add_parser(
        "run-all",
        help="run recall + all lm-eval suites + speed profiler in sequence",
    )
    p_all.add_argument(
        "--model", required=True,
        help="model config name or raw identifier",
    )
    p_all.add_argument(
        "--corpus", required=True,
        help="corpus config name for the recall step",
    )
    p_all.add_argument(
        "--db", default="results/benchmark.db",
        help="SQLite DB path (default: results/benchmark.db)",
    )
    p_all.add_argument(
        "--runs", type=int, default=None, dest="runs",
        help="override runs_per_function for the recall step",
    )
    p_all.set_defaults(func=cmd_run_all)

    # --- serve --------------------------------------------------------------
    p_srv = sub.add_parser(
        "serve",
        help="start the web dashboard (requires FastAPI installation)",
    )
    p_srv.add_argument(
        "--db", default="results/benchmark.db",
        help="SQLite DB path (default: results/benchmark.db)",
    )
    p_srv.add_argument(
        "--host", default="127.0.0.1",
        help="bind host (default: 127.0.0.1)",
    )
    p_srv.add_argument(
        "--port", type=int, default=8000,
        help="bind port (default: 8000)",
    )
    p_srv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
