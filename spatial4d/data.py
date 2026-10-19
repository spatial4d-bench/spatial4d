"""Result/QA adapters and input/output safety checks; no scoring or HTTP."""
from __future__ import annotations
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import OPTION_LABELS, TASKS
from .rules import canonical_text, extract_options, normalize_task


@dataclass(frozen=True)
class ResultRecord:
    """One input row with its location, before parsing or scoring."""

    row: dict
    task: str
    source_file: str
    source_position: int


def load_result_records(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Result directory not found: {directory}")
    files = sorted(path for path in directory.glob("result_*.json") if path.is_file())
    if not files:
        raise ValueError(f"No result_*.json files in {directory}")
    records, empty_files, seen = [], [], set()
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON result array: {path}")
        if not data:
            empty_files.append(path.name)
            continue
        for position, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Non-object result in {path}, row {position}")
            task = task_name(row["question_type"]) if row.get("question_type") else filename_task(path)
            if row.get("id") is not None:
                key = (task, str(row["id"]))
                if key in seen:
                    raise ValueError(f"Duplicate task/id across result files: {key}")
                seen.add(key)
            records.append(ResultRecord(row, task, path.name, position))
    if not records:
        raise ValueError("All result files are empty; no macro score can be reported")
    return records, empty_files


def task_name(value):
    task = normalize_task(value)
    if task not in TASKS:
        raise ValueError(f"Unknown task {value!r}; supply question_type or a canonical result_<task>.json filename")
    return task


def filename_task(path):
    stem = Path(path).stem
    return task_name(stem.removeprefix("result_"))


def options_dict(raw):
    if raw is None or raw == [] or raw == {}:
        return {}
    if isinstance(raw, dict):
        result = {}
        for key, value in raw.items():
            label = str(key).strip()
            if not re.fullmatch(r"[A-Z]", label):
                raise ValueError(f"Invalid option label {key!r}")
            description = str(value).strip()
            description = re.sub(r"^" + label + r"\.\s+", "", description)
            result[label] = description
        return result
    if isinstance(raw, (list, tuple)):
        # Full labeled options are the benchmark format; also accept an
        # explicitly ordered array of unlabeled option descriptions.
        if all(isinstance(v, str) and not re.match(r"^\s*[A-Z]\.\s+", v) for v in raw):
            if len(raw) > 26:
                raise ValueError("More than 26 unlabeled options")
            return {chr(65+i): value.strip() for i, value in enumerate(raw)}
        raw = "\n".join(str(v) for v in raw)
    if not isinstance(raw, str):
        raise ValueError("Options must be a string, list or letter-to-text mapping")
    text = raw.strip()
    if text.casefold() in {"", "none", "null", "[]"}:
        return {}
    if text[0] in "[{":
        for decoder in (json.loads, ast.literal_eval):
            try:
                decoded = decoder(text)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(decoded, (list, dict)):
                return options_dict(decoded)
    markers = list(re.finditer(r"(?<!\w)([A-Z])\.\s+", text))
    if not markers or text[:markers[0].start()].strip(" \t\r\n['\""):
        raise ValueError("Cannot parse the supplied options")
    result = {}
    for i, marker in enumerate(markers):
        end = markers[i+1].start() if i+1 < len(markers) else len(text)
        description = text[marker.end():end].strip()
        label = marker.group(1)
        if label in result:
            raise ValueError(f"Duplicate option label {label}")
        result[label] = description
    return result


def read_qa_rows(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("id"), list):
        size = len(payload["id"])
        if not all(isinstance(values, list) and len(values) == size for values in payload.values()):
            raise ValueError(f"QA columns have inconsistent lengths: {path}")
        rows = [{key: values[i] for key, values in payload.items()} for i in range(size)]
    else:
        raise ValueError(f"Expected QA rows or column-oriented QA JSON: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Non-object QA row in {path}")
    return rows


def load_qa_context(directory):
    if directory is None:
        return {}
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"QA directory not found: {directory}")
    context = {}
    for path in sorted(directory.glob("*.json")):
        for row in read_qa_rows(path):
            task = task_name(row["question_type"]) if row.get("question_type") else filename_task(path)
            if "id" not in row:
                raise ValueError(f"QA row without id: {path}")
            key = (task, str(row["id"]))
            if key in context:
                raise ValueError(f"Duplicate QA task/id: {key}")
            # Deliberately do not import QA ground_truth into prediction parsing.
            context[key] = {k: row[k] for k in ("question", "options", "dataset", *OPTION_LABELS) if k in row}
    return context


def resolve_context(row, task, qa_context, prefer_explicit=False):
    saved = qa_context.get((task, str(row.get("id"))), {})
    question = row.get("question")
    question = saved.get("question", "") if question in (None, "") else question
    question = "" if question is None else str(question)
    raw_options = row.get("options")
    if raw_options in (None, "", []):
        raw_options = saved.get("options")
    explicit = options_dict(raw_options)
    if prefer_explicit:
        # v3: fill missing A-F from the question, then prefer reference fields.
        for source in (saved, row):
            explicit.update({k: str(source[k]) for k in OPTION_LABELS
                             if source.get(k) is not None and str(source[k]).strip()})
    from_question = extract_options(question)
    for key in explicit.keys() & from_question.keys():
        if not prefer_explicit and canonical_text(explicit[key]) != canonical_text(from_question[key]):
            raise ValueError(f"Option {key} conflicts between saved question and options")
    choices = {**from_question, **explicit}
    return question, choices, bool(question.strip() or choices)


def guard_output_path(path, directory, qa_dir=None):
    path, directory = Path(path), Path(directory)
    inputs = {p.resolve() for p in directory.glob("result_*.json")}
    if qa_dir:
        inputs.update(p.resolve() for p in Path(qa_dir).glob("*.json"))
    if (path.resolve() in inputs or path.parent.resolve() == directory.resolve()
            and path.name.startswith("result_") and path.suffix == ".json"
            or qa_dir and path.parent.resolve() == Path(qa_dir).resolve() and path.suffix == ".json"):
        raise ValueError("Output/cache must not overwrite or masquerade as an input result/QA file")
