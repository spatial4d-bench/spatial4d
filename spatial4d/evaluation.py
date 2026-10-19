"""Per-case processing, pure aggregation and directory orchestration.

As in VSI-Bench, case processing and task aggregation have separate APIs. The
Spatial4D parsing rules, judge dispatch and task-macro semantics are preserved.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .config import NA_QUESTION_TYPES, OPTION_LABELS, PLATFORM_SOURCE_SHA256, RULES_VERSION, TASKS, TASK_SPECS
from .data import guard_output_path, load_qa_context, load_result_records, options_dict, resolve_context, task_name
from .judge import (_build_mca_judge_prompt, _build_na_extract_prompt, _build_option_str,
                    _extract_gt_letters, _extract_letter, _extract_letter_sequence,
                    _is_source_number, _numbers_in_text, _parse_finite_number)
from .rules import CaseScore, decimal_number, extract_options, letter_sequence, mra_score, score_case


def _judge_score(raw, ground_truth, task, question, choices, judge):
    """v3 dispatch: all MCA; NA only if the original strict parser fails."""
    numeric = task in NA_QUESTION_TYPES
    if numeric and _parse_finite_number(raw) is not None:
        return score_case(raw, ground_truth, task, question, choices), {
            "judge_mode": "na_direct", "failed_reason": None, "judge_called": False}
    if numeric:
        prompt = _build_na_extract_prompt(question, raw)
    else:
        choices = {k: v for k, v in choices.items() if k in OPTION_LABELS}
        prompt = _build_mca_judge_prompt(question, _build_option_str(choices), raw, task == "route_plan")
    reply = judge.request(prompt)
    meta = {"judge_mode": "na_judge_extract" if numeric else "mca", "judge_called": True,
            "judge_prompt": prompt, "judge_response": reply.text, "judge_attempts": reply.attempts,
            "judge_cache_hit": reply.cache_hit, "failed_reason": reply.failed_reason,
            "judge_error": reply.error}
    gold = str(decimal_number(ground_truth)) if numeric else _extract_gt_letters(str(ground_truth), task == "route_plan")
    if reply.failed_reason:
        return CaseScore(task, 0., "judge_error", "judge", None, gold, [reply.failed_reason]), meta
    if reply.text.strip() == "Z":
        # v3 treats a valid Z as a normal no-match, not a service failure.
        return CaseScore(task, 0., "no_match", "judge", None, gold), meta
    if numeric:
        candidate = _parse_finite_number(reply.text)
        if candidate is None:
            reason = "judge_unparseable_numeric"
        elif not _is_source_number(candidate, _numbers_in_text(raw)):
            reason = "judge_value_not_in_source"
        else:
            result = score_case(reply.text.strip(), ground_truth, task, question, choices)
            result.parse_method = "na_judge_extract"
            return result, meta
        meta["failed_reason"] = reason
        return CaseScore(task, 0., "numeric_extraction_failed", "judge", None, gold, [reason]), meta
    letter = _extract_letter(reply.text) if task != "route_plan" else None
    labels = _extract_letter_sequence(reply.text) if task == "route_plan" else ([letter] if letter else [])
    if not labels:
        meta["failed_reason"] = "judge_unparseable_choice"
        return CaseScore(task, 0., "judge_unparseable_choice", "judge", None, gold), meta
    if task == "route_plan" and len(labels) != len(gold):
        meta["local_validation_reason"] = "answer_length_mismatch"
    return CaseScore(task, float(bool(labels) and labels == gold), "ok", "judge", labels, gold), meta


def extract_and_format(ans, gt, is_choice, question_type=None, question="", options=None):
    """Keep the ZIP helper name, preserving order, repeats and numeric precision.

    For full single-choice/route validation pass question_type and context.
    The legacy boolean-only choice form uses an ordered sequence fallback.
    The command-line scorer always supplies the actual task.
    """
    task = task_name(question_type) if question_type else ("route_plan" if is_choice else "object_counting")
    choices = options_dict(options) if options is not None else extract_options(question)
    case = score_case(ans, gt, task, question, choices)
    if is_choice:
        prediction = ",".join(case.prediction_normalized) if case.parse_status == "ok" else ""
        return prediction, ",".join(case.ground_truth_normalized)
    return (str(case.prediction_normalized) if case.parse_status == "ok" else ""), str(case.ground_truth_normalized)


def calc_score(p, g, is_choice):
    """Score already-extracted values without sorting, deduplication or rounding."""
    if is_choice:
        valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        prediction, target = letter_sequence(p, valid), letter_sequence(g, valid)
        return float(bool(prediction) and prediction == target)
    try:
        return mra_score(p, g)
    except (ValueError, TypeError):
        return 0.0


@dataclass(frozen=True)
class PreparedRecord:
    """Validated case context. Preparation never sends a model request."""

    task: str
    id: Any
    raw_answer: str
    ground_truth: Any
    question: str
    choices: dict[str, str]
    context_available: bool
    inference_failed: bool
    judge_enabled: bool


def prepare_record(row, task, qa_context=None, *, judge_enabled=False):
    if not isinstance(row, dict) or "model_answer" not in row or "ground_truth" not in row:
        raise ValueError("Every result must contain model_answer and ground_truth")
    task = task_name(task)
    if "is_choice" in row:
        choice = row["is_choice"]
        if not isinstance(choice, bool):
            if judge_enabled:
                raise ValueError(f"is_choice conflicts with question_type={task}")
            raise ValueError("is_choice must be a JSON boolean")
        if choice != (task not in NA_QUESTION_TYPES):
            raise ValueError(f"is_choice conflicts with question_type={task}")
    question, choices, has_context = resolve_context(
        row, task, qa_context or {}, prefer_explicit=judge_enabled)
    raw = "" if row["model_answer"] is None else str(row["model_answer"])
    status = str(row.get("inference_status", "")).casefold()
    failed = status in {"failed", "error", "失败"} or bool(row.get("inference_error")) or raw.lstrip().startswith("Error:")
    if judge_enabled:
        # Validate reference and every required context before any paid call.
        score_case("", row["ground_truth"], task, question, choices)
        needs_judge = bool(raw.strip()) and not failed and (
            task not in NA_QUESTION_TYPES or _parse_finite_number(raw) is None)
        if needs_judge and not question.strip():
            raise ValueError(f"Judge requires question context for task={task}, id={row.get('id')}; supply --qa-dir")
        if needs_judge and task not in NA_QUESTION_TYPES:
            valid = {k for k in choices if k in OPTION_LABELS}
            gold = _extract_gt_letters(str(row["ground_truth"]), task == "route_plan")
            if not valid or not gold or not all(k in valid for k in gold):
                raise ValueError(f"Judge requires valid A-F options/GT for task={task}, id={row.get('id')}; supply --qa-dir")
    return PreparedRecord(task, row.get("id"), raw, row["ground_truth"], question,
                          choices, has_context, bool(failed), judge_enabled)


def process_results(record: PreparedRecord, judge=None):
    """Score one prepared case; only an explicitly supplied judge can call HTTP."""
    if record.judge_enabled != (judge is not None):
        raise ValueError("Prepare the record with the same judge mode used for processing")
    task, raw, failed = record.task, record.raw_answer, record.inference_failed
    meta = {"judge_mode": "rule_based", "judge_called": False, "failed_reason": None}
    if judge is not None and not failed and raw.strip():
        case, meta = _judge_score(raw, record.ground_truth, task, record.question, record.choices, judge)
    else:
        case = score_case("" if failed else raw, record.ground_truth, task, record.question, record.choices)
    if failed:
        case.parse_status = "invalid"
        case.parse_method = "inference_status"
        case.notes = ["inference_failure"]
        meta["failed_reason"] = "inference_failure"
    group = ("inference_failure" if failed else "empty_answer" if not raw.strip()
             else "answer_no_match" if case.parse_status == "no_match"
             else "judge_failure" if meta.get("judge_called") and meta.get("failed_reason")
             else "parsed" if case.parse_status == "ok" else "parsing_failure")
    return {"id": record.id, "raw_answer": raw, "ground_truth": record.ground_truth,
            **asdict(case), **meta, "failure_group": group, "context_available": record.context_available}


def score_record(row, task, qa_context=None, judge=None):
    """Compatibility API for scoring a result row directly."""
    return process_results(prepare_record(row, task, qa_context, judge_enabled=judge is not None), judge)


def _preflight_judge_record(row, task, context):
    """Compatibility helper: validate a judge row without making requests."""
    prepare_record(row, task, context, judge_enabled=True)


def missing_tasks(tasks):
    return [task for task in TASKS if task not in tasks]


def _check_coverage(missing, require_all_tasks):
    if missing and require_all_tasks:
        raise ValueError("Missing tasks: " + ", ".join(missing))


def aggregate_results(cases, *, require_all_tasks=False):
    """Pure task-macro aggregation; failed/empty answers stay in the denominator.

    Full runs average 18 task means equally. Partial runs are explicitly marked
    and average only present tasks, exactly as the original CLI did.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("No cases; no macro score can be reported")
    grouped = defaultdict(list)
    for record in cases:
        grouped[task_name(record["task"])].append(record)
    missing = missing_tasks(grouped)
    _check_coverage(missing, require_all_tasks)
    task_scores = {}
    for task in TASKS:
        if task not in grouped:
            continue
        rows = grouped[task]
        task_scores[task] = {
            "samples": len(rows), "metric": TASK_SPECS[task].metric,
            "score_percent": math.fsum(r["score"] for r in rows) / len(rows) * 100,
            "failure_groups": dict(Counter(r["failure_group"] for r in rows)),
            "parse_status_counts": dict(Counter(r["parse_status"] for r in rows)),
        }
    return {
        "samples": len(cases), "evaluated_tasks": len(task_scores), "benchmark_complete": not missing,
        "missing_tasks": missing,
        "context_missing_cases": sum(not r["context_available"] for r in cases),
        "macro_task_score_percent": math.fsum(t["score_percent"] for t in task_scores.values()) / len(task_scores),
        "failure_groups": dict(Counter(r["failure_group"] for r in cases)),
        "task_scores": task_scores, "cases": cases,
    }


def calculate_directory(directory, qa_dir=None, require_all_tasks=False, judge=None):
    """Load → prepare all cases → check coverage → score → aggregate.

    No judge request is made until every row and the required coverage pass
    preflight. Source inputs are read-only.
    """
    records, empty_files = load_result_records(directory)
    if judge is not None and judge.cache_path:
        guard_output_path(judge.cache_path, directory, qa_dir)
    before_stats = judge.summary() if judge is not None else None
    context = load_qa_context(qa_dir)
    prepared = [prepare_record(item.row, item.task, context, judge_enabled=judge is not None)
                for item in records]
    _check_coverage(missing_tasks({item.task for item in prepared}), require_all_tasks)
    cases = []
    for source, item in zip(records, prepared):
        result = process_results(item, judge)
        result.update(source_file=source.source_file, source_position=source.source_position)
        cases.append(result)
    summary = aggregate_results(cases, require_all_tasks=require_all_tasks)
    judge_stats = judge.summary() if judge is not None else {"enabled": False}
    if judge is not None:
        for key in ("api_requests", "judged_samples", "cache_hits"):
            judge_stats[key] -= before_stats[key]
        judge_stats["usage"] = {k: v - before_stats["usage"].get(k, 0) for k, v in judge_stats["usage"].items()}
    return {
        "rules_version": RULES_VERSION, "platform_source_sha256": PLATFORM_SOURCE_SHA256,
        "scoring_mode": "judge" if judge is not None else "rule_based", "judge": judge_stats,
        "new_model_calls": judge_stats.get("api_requests", 0),
        **summary, "empty_files": empty_files,
    }
