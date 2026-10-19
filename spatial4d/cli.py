"""Command-line configuration and report presentation for Spatial4D scoring.

No judge key means offline rule scoring. A configured judge never falls back
to rules after an API/extraction failure. No platform packages are required.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from .config import RESULT_DIR, RULES_VERSION
from .data import guard_output_path
from .evaluation import calculate_directory
from .judge import JudgeClient


def build_parser(default_result_dir=RESULT_DIR):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=Path(default_result_dir), type=Path, help="Folder containing result_*.json arrays; defaults to RESULT_DIR")
    parser.add_argument("--qa-dir", type=Path, help="Matching QA JSON folder; restores question/options for old results by task+id")
    parser.add_argument("--output", type=Path, help="Optional JSON report including per-case parsing diagnostics")
    parser.add_argument("--require-all-tasks", action="store_true", help="Refuse to report a partial-task benchmark")
    parser.add_argument("--judge-api-key", "--api-key", dest="judge_api_key",
                        help="Enable judge scoring; prefer SPATIAL4D_JUDGE_API_KEY to avoid shell-history exposure")
    parser.add_argument("--judge-model", default=os.environ.get("SPATIAL4D_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--judge-base-url", default=os.environ.get("SPATIAL4D_JUDGE_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--rule-based", action="store_true", help="Force offline rules even if a dedicated judge key exists")
    parser.add_argument("--judge-timeout", type=float, default=60., help="Per-request timeout in seconds")
    parser.add_argument("--judge-max-attempts", type=int, default=3, help="Total attempts per judged sample, including the first")
    parser.add_argument("--judge-qps", type=float, default=2., help="Maximum HTTP requests per second (serial execution)")
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-cache", type=Path, help="Optional JSONL cache; resume completed judge responses without repaying")
    return parser


def create_judge(args):
    """Never reuse the inference OPENAI_API_KEY for paid judging implicitly."""
    key = args.judge_api_key if args.judge_api_key is not None else os.environ.get("SPATIAL4D_JUDGE_API_KEY", "")
    if args.rule_based or not str(key).strip():
        return None
    return JudgeClient(key, args.judge_model, args.judge_base_url, args.judge_timeout,
                       args.judge_max_attempts, args.judge_qps, args.judge_max_tokens, args.judge_cache)


def print_report(report):
    print(f"Rules: {RULES_VERSION}; mode={report['scoring_mode']}; samples={report['samples']}")
    print(f"{'Task':<26} {'N':>6} {'Metric':>9} {'Score %':>10}")
    for task, result in report["task_scores"].items():
        print(f"{task:<26} {result['samples']:>6} {result['metric']:>9} {result['score_percent']:>10.2f}")
    label = "TASK MACRO (18/18)" if report["benchmark_complete"] else f"PARTIAL TASK MACRO ({report['evaluated_tasks']}/18)"
    print(f"{label}: {report['macro_task_score_percent']:.2f}%")
    print("Failure groups:", json.dumps(report["failure_groups"], ensure_ascii=False))
    if report["judge"]["enabled"]:
        print("Judge usage:", json.dumps(report["judge"], ensure_ascii=False))
    if report["context_missing_cases"]:
        print(f"WARNING: {report['context_missing_cases']} cases lack question/options context; use --qa-dir for full validation.", file=sys.stderr)
    if report["missing_tasks"]:
        print("WARNING: partial benchmark; missing tasks: " + ", ".join(report["missing_tasks"]), file=sys.stderr)
    if report["empty_files"]:
        print("WARNING: skipped empty files: " + ", ".join(report["empty_files"]), file=sys.stderr)


def main(argv=None, *, default_result_dir=RESULT_DIR):
    parser = build_parser(default_result_dir)
    args = parser.parse_args(argv)
    try:
        if args.output:
            guard_output_path(args.output, args.result_dir, args.qa_dir)
        if args.judge_cache:
            guard_output_path(args.judge_cache, args.result_dir, args.qa_dir)
            if args.output and args.output.resolve() == args.judge_cache.resolve():
                raise ValueError("Judge cache and report must use different paths")
        judge = create_judge(args)
        if judge is not None:
            print(f"Judge enabled: model={judge.summary()['model']}; failed judging will NOT fall back to rules.", file=sys.stderr)
        report = calculate_directory(args.result_dir, args.qa_dir, args.require_all_tasks, judge)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0
