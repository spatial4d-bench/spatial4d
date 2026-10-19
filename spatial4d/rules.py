"""Pure answer parsing and rule scoring; no files, clients or model calls."""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from .config import ALIASES, MCA_QUESTION_TYPES, NA_QUESTION_TYPES, TASKS


THRESHOLDS = tuple(Decimal(i) / 100 for i in range(50, 100, 5))
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER_RE = re.compile(r"(?<![\w.])" + NUMBER + r"(?![\w.])")
OPTION_LINE = re.compile(r"(?m)^\s*([A-Z])\.\s+([^\n]+)")
SMALL_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
SMALL_WORDS.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))
WORD_RE = re.compile(r"\b(?:" + "|".join(SMALL_WORDS) + r"|hundred|thousand)(?:[- ]+(?:and[- ]+)?(?:" + "|".join(SMALL_WORDS) + r"|hundred|thousand))*\b", re.I)


@dataclass
class Parsed:
    value: Any = None
    status: str = "unparseable"
    method: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    task: str
    score: float
    parse_status: str
    parse_method: str
    prediction_normalized: Any
    ground_truth_normalized: Any
    notes: list[str] = field(default_factory=list)


def normalize_task(task):
    task = str(task or "").strip().lower()
    if task.startswith("spatial_memory_"):
        return "spatial_memory"
    return ALIASES.get(task, task)


def strip_format(text):
    text = str(text or "").strip().replace("−", "-")
    text = re.sub(r"^```(?:json|text|plaintext)?\s*|\s*```$", "", text, flags=re.I)
    return text.strip().strip("`*_ \t\r\n")


def canonical_text(text):
    # Keep signs and decimal punctuation: different numeric options must stay distinct.
    text = strip_format(text).casefold().replace("’", "'").replace("→", " -> ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:!?\"'()")


def extract_options(question):
    body = str(question or "").replace("\\n", "\n").split("Please provide your answer", 1)[0]
    return {m.group(1): m.group(2).strip() for m in OPTION_LINE.finditer(body)}


def letter_sequence(value, valid):
    if isinstance(value, (list, tuple)):
        result = []
        for entry in value:
            item = letter_sequence(entry, valid)
            if item is None:
                return None
            result.extend(item)
        return result
    text = strip_format(value).strip(" .;()")
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        quoted = list(re.finditer(r"'([^']*)'|\"([^\"]*)\"", inner))
        residue = re.sub(r"'[^']*'|\"[^\"]*\"", "", inner)
        if quoted and not residue.strip(" ,\t\r\n"):
            return letter_sequence([m.group(1) if m.group(1) is not None else m.group(2) for m in quoted], valid)
    if len(text) == 1 and text.upper() in valid:
        return [text.upper()]
    if re.fullmatch(r"[A-Z](?:\s*[,，;；、\s]\s*[A-Z])+", text):
        letters = re.findall(r"[A-Z]", text)
        return letters if all(x in valid for x in letters) else None
    if re.fullmatch(r"[A-Z]{2,}", text):
        return list(text) if all(x in valid for x in text) else None
    return None


def decimal_number(value):
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def word_number(text):
    words = re.split(r"[\s-]+", text.casefold())
    total = current = 0
    for word in words:
        if word == "and":
            if not any(w in words for w in ("hundred", "thousand")):
                return None
        elif word in SMALL_WORDS:
            # "two three" and "two and three" are alternatives, not 5.
            if current % 100 in range(1, 20):
                return None
            current += SMALL_WORDS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
        else:
            return None
    return Decimal(total + current)


def units(text):
    lower = text.casefold()
    if re.search(r"square\s+met(?:er|re)s?|m\s*(?:²|\^\s*2|2\b)", lower):
        return "m2"
    if re.search(r"centimet(?:er|re)s?|\bcm\b|(?<=\d)cm\b", lower):
        return "cm"
    if re.search(r"millimet(?:er|re)s?|\bmm\b|(?<=\d)mm\b", lower):
        return "mm"
    if re.search(r"\bmet(?:er|re)s?\b|\bm\b|(?<=\d)m\b", lower):
        return "m"
    return None


REASON_TAG = re.compile(r"<\s*(/?)\s*(think|analysis|reasoning)\s*>", re.I)


ANSWER_TAG = re.compile(r"<\s*(answer|final_answer)\s*>(.*?)<\s*/\s*\1\s*>", re.I | re.S)


ANSWER_HEADING = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(final[ \t]+answer|answer|最终答案|最终回答|答案)[ \t]*[:：=][ \t]*", re.I)


STRUCT_LABEL = re.compile(r"^([A-Z])(?:[.)][ \t]*|[ \t]+[-–—][ \t]*|[ \t]*\()(.+?)\)?$")


STRUCT_INDEX = re.compile(r"^(?:(step|blank)\s*)?(\d+)\s*(?:[.):=]|→|->|[-–—])\s*(.*)$", re.I)


UNIT_TOKEN = r"(?:square\s+met(?:er|re)s?|centimet(?:er|re)s?|millimet(?:er|re)s?|met(?:er|re)s?|cm|mm|m²|m\^2|m2|m)"


def _reject(reason, status="invalid", method="structure"):
    return Parsed(status=status, method=method, notes=[reason])


def _clean_reasoning(text):
    stack, output = [], []
    cursor, seen_open, seen_close = 0, False, False
    for match in REASON_TAG.finditer(text):
        closing, tag = bool(match.group(1)), match.group(2).lower()
        if not closing:
            if not stack:
                output.append(text[cursor:match.start()])
            stack.append(tag)
            seen_open = True
        elif stack:
            if stack.pop() != tag:
                return None, "mismatched_reasoning_tags"
            if not stack:
                cursor = match.end()
                seen_close = True
        elif not seen_open and not seen_close:
            # Chat templates may prefill an opening reasoning token. Its explicit
            # closing token still defines a boundary; text before it is not scored.
            output = []
            cursor = match.end()
            seen_close = True
        else:
            return None, "unmatched_reasoning_tag"
    if stack:
        return None, "unclosed_reasoning"
    output.append(text[cursor:])
    return "".join(output).strip(), None


def _presentation(text):
    text = str(text).strip().replace("−", "-")
    text = re.sub(r"^```(?:json|text|plaintext)?\s*|\s*```$", "", text, flags=re.I)
    return text.replace("**", "").replace("__", "").replace("`", "").strip()


def _structured_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def answer_regions(raw):
    """Return all equal-priority explicit scopes, never the first/last guess.

    Priority: message final-content field -> reasoning cleanup -> JSON answer
    -> answer tags/boxes -> anchored answer headings -> entire cleaned text.
    """
    if isinstance(raw, dict) and "content" in raw and not any(k in raw for k in ("answer", "final_answer")):
        raw = raw["content"]
    if isinstance(raw, (dict, list)):
        text = json.dumps(raw, ensure_ascii=False)
    else:
        text = "" if raw is None else str(raw).strip()
    if not text:
        return "", [], "empty_answer"
    text, error = _clean_reasoning(text)
    if error:
        return "", [], error
    text = _presentation(text)
    if not text:
        return "", [], "missing_final_answer"
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        keys = [k for k in ("answer", "final_answer") if k in obj]
        if keys:
            return text, [_structured_value(obj[k]) for k in keys], "json_answer"
        # Explicit route mappings are validated by the route parser.
        if obj and all(str(k).isdigit() for k in obj):
            return text, [text], "json_mapping"
    if isinstance(obj, list):
        return text, [text], "json_sequence"
    tagged = list(ANSWER_TAG.finditer(text))
    if tagged:
        return text, [m.group(2).strip() for m in tagged], "answer_tag"
    if re.search(r"<\s*(?:answer|final_answer)\s*>", text, re.I):
        return "", [], "unclosed_answer_tag"
    boxes = list(re.finditer(r"\\boxed\{([^{}]*)\}", text))
    if boxes:
        return text, [m.group(1).strip() for m in boxes], "boxed_answer"
    markers = list(ANSWER_HEADING.finditer(text))
    if markers:
        final = [m for m in markers if m.group(1).lower().startswith("final") or m.group(1).startswith("最终")]
        active = final or markers
        scopes = []
        for m in active:
            following = next((n.start() for n in markers if n.start() > m.start()), len(text))
            scopes.append(text[m.end():following].strip())
        return text, scopes, "answer_heading"
    return text, [text], "unmarked_text"


def _scope_failure(origin):
    if origin in {"empty_answer", "missing_final_answer", "unclosed_reasoning", "unclosed_answer_tag"}:
        return _reject(origin, "incomplete", "scope")
    return _reject(origin, "invalid", "scope")


def _consensus(parsed, origin):
    if not parsed:
        return _scope_failure(origin)
    failures = [p for p in parsed if p.status != "ok"]
    if failures:
        # A failed explicit scope cannot disappear because another one parses.
        result = failures[0]
        if len(parsed) > 1:
            result = _reject("unresolved_multiple_answer_scopes", "ambiguous")
    else:
        values = {json.dumps(p.value, sort_keys=True) for p in parsed}
        result = parsed[0] if len(values) == 1 else _reject("conflicting_answer_scopes", "ambiguous")
    result.method = origin + ":" + result.method
    return result


def _choice_atom(text, options):
    text = _presentation(text)
    text = re.sub(r"^(?:[.。]\s+|[-*•·]\s+)", "", text)
    for key, option in options.items():
        if canonical_text(text) == canonical_text(option):
            matches = [k for k, v in options.items() if canonical_text(v) == canonical_text(text)]
            return Parsed(matches, "ok", "option_text") if len(matches) == 1 else _reject("duplicate_option_text", "ambiguous")
    bare = re.fullmatch(r"[\[('\"]?([A-Z])[\])'\"]?[.)]?", text)
    if bare:
        return Parsed([bare.group(1)], "ok", "label") if bare.group(1) in options else _reject("option_out_of_range")
    # Weak punctuation is only a label separator when followed by a literal
    # option. Do not turn arbitrary prose such as "I think ..." into a label.
    weak = re.fullmatch(r"([A-Z])(?:[,，][ \t]*|[ \t]+)(.+)", text)
    if weak:
        label, description = weak.groups()
        actual = [k for k,v in options.items() if canonical_text(description)==canonical_text(v)]
        if actual:
            if label not in options:
                return _reject("option_out_of_range")
            if label not in actual:
                return _reject("letter_content_conflict")
            return Parsed([label], "ok", "exact_labeled_option")
    match = STRUCT_LABEL.fullmatch(text)
    if match:
        label, description = match.groups()
        if re.match(r"^[A-Z]\s*\(", text) and not re.fullmatch(r"[A-Z]\s*\([^()]*\)[.!]?", text):
            return None  # Parenthesized option mentions embedded in prose.
        if label in options and canonical_text(description) == canonical_text(options[label]):
            return Parsed([label], "ok", "exact_labeled_option")
        if len(list(re.finditer(r"(?<![\w.])[A-Z][.)](?:[ \t]+|$)", text))) > 1:
            return _reject("multiple_inline_options", "ambiguous")
        if label not in options:
            return _reject("option_out_of_range")
        actual = [k for k, v in options.items() if canonical_text(description) == canonical_text(v)]
        if actual and label not in actual:
            return _reject("letter_content_conflict")
        return Parsed([label], "ok", "label_line")
    return None


def _inline_route_fields(text, options):
    """Tokenize typed option fields plus optional step metadata, not prose."""
    pattern = re.compile(
        r"(?<![\w.])(?:(?:(?i:for\s+)?(?i:step|blank)\s+)?(\d+)\s*(?:[.:=,]|→|->|[-–—])\s*)?"
        r"([A-Z])(?:[.)][ \t]+|[,，][ \t]*|[ \t]*\(|[ \t]+(?=[a-z]))"
    )
    markers = list(pattern.finditer(text))
    if not markers or markers[0].start() != 0:
        return None
    fields = []
    annotation = re.compile(r"\(\s*(?:(?i:for)\s+)?(?i:step|blank)\s+(\d+)\s*\)")
    for i, marker in enumerate(markers):
        end = markers[i+1].start() if i+1<len(markers) else len(text)
        description = text[marker.end():end].strip()
        description = re.sub(r"(?:\s*[,;，；、—–-]\s*|\s+)(?i:and\s+then|then|and)\s*$", "", description)
        description = description.strip().rstrip(",;，；、→- ")
        ids = [int(m.group(1)) for m in annotation.finditer(description)]
        description = annotation.sub("", description).strip()
        description = description.strip().rstrip(",;，；、→- ")
        label = marker.group(2)
        if description and canonical_text(description) != canonical_text(options.get(label, "")):
            actual = [k for k,v in options.items() if canonical_text(description)==canonical_text(v)]
            if actual and label not in actual:
                return _reject("letter_content_conflict")
            # A parenthesized annotation after a full literal option does not
            # supply another answer. Reject annotations naming another option.
            annotated = re.fullmatch(r"(.+?)\s*\([^()]*\)[.!]?", description)
            parenthesized = marker.group().rstrip().endswith("(") and re.fullmatch(r"[^()]*\)[.!]?", description)
            if annotated and label in options and canonical_text(annotated.group(1))==canonical_text(options[label]):
                annotation_text = description[len(annotated.group(1)):]
                if re.search(r"\b[A-Z]\b", annotation_text):
                    return None
            elif not parenthesized:
                return None
        if label not in options:
            return _reject("option_out_of_range")
        if marker.group(1):
            ids.append(int(marker.group(1)))
        if len(set(ids))>1:
            return _reject("conflicting_step_metadata","ambiguous")
        fields.append((ids[0] if ids else None,label))
    return fields


def _parallel_route_fields(text, options):
    """An explicit letter tuple plus an equally long literal description tuple.

    Both halves independently state the same ordered answer. Never duplicate
    one label because the question has multiple blanks.
    """
    head = re.match(r"[A-Z\s,;，；]+", text)
    for split in range(1, head.end()+1 if head else 1):
        prefix, suffix = text[:split].strip(), text[split:].strip()
        labels = letter_sequence(prefix, set(options))
        if not labels or len(labels)<2 or len(labels)>len(text)//2:
            continue
        suffix = suffix.strip(" ()[]")
        descriptions = re.split(r"\s*[,;，；]\s*", suffix)
        if len(descriptions)!=len(labels):
            continue
        actual = [[k for k,v in options.items() if canonical_text(part)==canonical_text(v)] for part in descriptions]
        if all(len(keys)==1 for keys in actual):
            if [keys[0] for keys in actual]!=labels:
                return _reject("letter_content_conflict")
            return [(None,label) for label in labels]
    return None


def _inline_option_block(text, options):
    """Decode a sequence of FULL options, not letter mentions in prose."""
    markers = list(re.finditer(r"(?<![\w.])([A-Z])[.)][ \t]+", text))
    if len(markers) < 2 or markers[0].start() != 0:
        return None
    labels = []
    for i, marker in enumerate(markers):
        end = markers[i+1].start() if i+1 < len(markers) else len(text)
        description = text[marker.end():end].strip().rstrip(",;，；、→- ")
        label = marker.group(1)
        if label not in options or canonical_text(description) != canonical_text(options[label]):
            return None
        labels.append(label)
    return labels


def _block_entries(paragraph, options, blank_steps=None):
    """Only pure answer blocks qualify. Explanatory paragraphs do not."""
    entries = []
    lines = [re.sub(r"^\s*[-*•·]\s+", "", _presentation(line)) for line in paragraph.splitlines() if line.strip()]
    # A standalone heading is presentation, but only if the following lines
    # form a fully typed block. Arbitrary prose below it still rejects the block.
    has_heading = len(lines)>1 and lines[0].endswith((":", "："))
    if has_heading:
        lines = lines[1:]
    indexed_lines = [STRUCT_INDEX.fullmatch(line) for line in lines]
    actual_numbering = bool(blank_steps) and indexed_lines and all(indexed_lines) and set(blank_steps) <= {int(m.group(2)) for m in indexed_lines}
    for line in lines:
        if not line:
            continue
        # A colon at line end is a heading, not a self-contained answer.
        if line.endswith((":", "：")):
            return None
        fields = _parallel_route_fields(line, options) or _inline_route_fields(line, options)
        if fields is None and blank_steps and re.search(r"[:：]", line):
            # Inline prose heading + an explicit complete step-to-answer map.
            # Only actual question step IDs qualify, not arbitrary letter mentions.
            suffix = re.split(r"[:：]", line, maxsplit=1)[1].strip()
            candidate = _inline_route_fields(suffix, options)
            if isinstance(candidate,list) and {n for n,_ in candidate}==set(blank_steps):
                fields = candidate
        if isinstance(fields, Parsed):
            return fields
        if fields:
            entries.extend((n,v) for n,v in fields if not actual_numbering or n in blank_steps)
            continue
        indexed = STRUCT_INDEX.fullmatch(line)
        number = int(indexed.group(2)) if indexed else None
        if actual_numbering and number not in blank_steps:
            continue  # A supplied route step is not an answer to a blank.
        content = indexed.group(3) if indexed else line
        # Compact indexed mappings: 1. A, 5. B (no prose/action inference).
        pairs = list(re.finditer(r"(\d+)\s*[.:=]\s*([A-Z])\b", line))
        residue = re.sub(r"\d+\s*[.:=]\s*[A-Z]\b", "", line)
        if len(pairs) > 1 and not residue.strip(" ,;，；\t"):
            if not all(m.group(2) in options for m in pairs):
                return _reject("option_out_of_range")
            entries.extend((int(m.group(1)), m.group(2)) for m in pairs)
            continue
        sequence = _inline_option_block(content, options)
        if sequence:
            if number is not None:
                return _reject("indexed_multiple_values", "ambiguous")
            entries.extend((None, value) for value in sequence)
            continue
        parsed = _choice_atom(content, options)
        if parsed is None:
            sequence = letter_sequence(content, set(options))
            if sequence:
                entries.extend((number, value) for value in sequence)
                continue
            return None
        if parsed.status != "ok":
            return parsed
        if has_heading and parsed.method=="label_line":
            return None  # A heading above annotated alternatives is not an answer list.
        if parsed.method == "label_line" and re.search(r"[:：]", content):
            return None  # Option-heading plus an annotation, not an answer atom.
        entries.append((number, parsed.value[0]))
    return entries or None


def _copied_option_list(paragraph, options):
    """A verbatim full option list is context, not N asserted answers."""
    found = []
    ends_in_list = False
    for raw in paragraph.splitlines():
        line = re.sub(r"^\s*[-*•·]\s+", "", _presentation(raw))
        if not line:
            continue
        match = re.fullmatch(r"([A-Z])[.)]\s+(.+)", line)
        ends_in_list = bool(match and match.group(1) in options and canonical_text(match.group(2))==canonical_text(options[match.group(1)]))
        if ends_in_list:
            found.append(match.group(1))
    return ends_in_list and len(found)==len(options) and set(found)==set(options) and len(options)>1


def _embedded_option_signature(paragraph, options):
    """Match a literal full option at paragraph end, independent of wording.

    A reference such as 'ruling out C' or '(B)' is not a complete option
    signature. There is no whitelist of conclusion phrases or model names.
    """
    matches = []
    for match in re.finditer(r"(?<![\w.])([A-Z])[.)][ \t]+", paragraph):
        label, description = match.group(1), paragraph[match.end():].strip()
        prefix = re.split(r"[.!?;\n]", paragraph[:match.start()])[-1]
        if re.search(r"\b(?:not|never|neither|without|if|unless|whether|might|could)\b", prefix, re.I):
            continue
        if label in options:
            actual = [k for k,v in options.items() if canonical_text(description) == canonical_text(v)]
            if actual and label not in actual:
                return _reject("letter_content_conflict")
            if actual == [label]:
                matches.append(label)
    return Parsed([matches[0]], "ok", "literal_option_signature") if len(set(matches)) == 1 else None


def _route_block(entries, blank_steps):
    numbers = [number for number, _ in entries]
    labels = [label for _, label in entries]
    expected = len(blank_steps) or None
    if any(n is not None for n in numbers):
        if not all(n is not None for n in numbers):
            return _reject("mixed_numbering", "ambiguous")
        if blank_steps and set(numbers) == set(blank_steps):
            mapping = {n: {label for number,label in entries if number == n} for n in blank_steps}
            if any(len(v) != 1 for v in mapping.values()):
                return _reject("conflicting_blank_values", "ambiguous")
            return Parsed([next(iter(mapping[n])) for n in blank_steps], "ok", "numbered_blanks")
        if numbers != list(range(1,len(numbers)+1)):
            return _reject("unresolved_step_numbering", "incomplete")
    return _validate_sequence(labels, True, expected, "answer_block")


def _choice_scope(text, options, route, blank_steps):
    text = _presentation(text)
    if not text:
        return _reject("missing_choice_answer", "incomplete")
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    expected = len(blank_steps) if blank_steps else (None if route else 1)
    if isinstance(obj, dict):
        if not route or not blank_steps or set(obj) != {str(n) for n in blank_steps}:
            return _reject("invalid_blank_mapping")
        atoms = [_choice_atom(str(obj[str(n)]), options) for n in blank_steps]
        if not all(p and p.status == "ok" for p in atoms):
            return _reject("invalid_mapping_value")
        return Parsed([p.value[0] for p in atoms], "ok", "blank_mapping")
    if isinstance(obj, list):
        atoms = [_choice_atom(str(v), options) if isinstance(v,str) else None for v in obj]
        if not obj or not all(p and p.status == "ok" for p in atoms):
            return _reject("invalid_answer_array")
        return _validate_sequence([p.value[0] for p in atoms], route, expected, "answer_array")
    sequence = letter_sequence(text, set(options))
    if sequence:
        return _validate_sequence(sequence, route, expected, "compact_sequence")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    answer_blocks, embedded, option_lists = [], [], []
    for paragraph in paragraphs:
        # Do not mistake the last option of a copied list for a conclusion,
        # even when the list shares its paragraph with a descriptive heading.
        if _copied_option_list(paragraph, options) and (not route or len(options)!=expected):
            option_lists.append(paragraph)
            continue
        entries = _block_entries(paragraph, options, blank_steps if route else None)
        if isinstance(entries, Parsed):
            # An isolated malformed answer is not silently discarded.
            answer_blocks.append(entries)
        elif entries:
            labels = [value for _,value in entries]
            # A complete copied option list is not an assertion. If nothing
            # else answers the question, it still produces an ambiguous result.
            if not route and len(labels)>1 and set(labels)==set(options):
                option_lists.append(entries)
                continue
            answer_blocks.append(entries)
        elif not route:
            signature = _embedded_option_signature(paragraph, options)
            if signature:
                embedded.append(signature)
    if not route:
        candidates = [p if isinstance(p,Parsed) else
                      (Parsed([p[0][1]],"ok","repeated_single_answer") if len({v for _,v in p})==1 else
                       _validate_sequence([v for _,v in p],False,1,"answer_block")) for p in answer_blocks]
        candidates.extend(embedded)
        if candidates:
            return _consensus(candidates,"assertions")
        if option_lists:
            return _reject("option_list_without_answer","ambiguous")
        return _reject("no_structural_choice_candidate")
    # A route sequence can be split into one-option paragraphs. Join only when
    # there is no complete block to contradict and exactly N explicit answers;
    # never manufacture a missing action or duplicate a single label.
    unnumbered = [block for block in answer_blocks if isinstance(block,list) and all(n is None for n,_ in block)]
    if expected and len(unnumbered)==len(answer_blocks) and all(len(block)<expected for block in unnumbered) and sum(map(len,unnumbered))==expected:
        return Parsed([value for block in unnumbered for _,value in block],"ok","split_answer_sequence")
    candidates, numbered_fragments = [], []
    for block in answer_blocks:
        if isinstance(block,Parsed):
            candidates.append(block)
            continue
        if blank_steps and all(n is not None for n,_ in block) and len(block)<len(blank_steps):
            numbered_fragments.extend(block)
        else:
            candidates.append(_route_block(block,blank_steps))
    if numbered_fragments:
        candidates.append(_route_block(numbered_fragments,blank_steps))
    if candidates:
        return _consensus(candidates,"assertions")
    return _reject("no_structural_choice_candidate")


def _validate_sequence(sequence, route, expected, method):
    if not route and len(sequence) != 1:
        return _reject("multiple_choice_candidates", "ambiguous", method)
    if expected is not None and len(sequence) != expected:
        return Parsed(sequence, "incomplete" if len(sequence) < expected else "invalid", method,
                      ["wrong_blank_count", f"expected={expected};actual={len(sequence)}"])
    return Parsed(sequence, "ok", method)


def parse_choice(raw, options=None, route=False, question=""):
    options = options or extract_options(question) or {k: k for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    blank_steps = [int(n) for n in re.findall(r"(\d+)\.\s*\[please fill in\]", question, flags=re.I)]
    _, scopes, origin = answer_regions(raw)
    return _consensus([_choice_scope(s, options, route, blank_steps) for s in scopes], origin)


def _number_scope(text, question):
    text = _presentation(text)
    if not text:
        return _reject("missing_numeric_answer", "incomplete")
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = text
    if isinstance(obj, (dict, list, bool)) or obj is None:
        return _reject("numeric_answer_must_be_scalar")
    if isinstance(obj, str) and obj != text:
        text = _presentation(obj)
    # An isolated scalar line is a typed answer block, regardless of position.
    # All such blocks must agree. Never take the first/last arbitrary number.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    atom = re.compile(r"(?:(?:approximately|approx\.?|about|around|roughly)\s+|[≈~]\s*)?(?:" + NUMBER + "|" + WORD_RE.pattern + r")\s*(?:" + UNIT_TOKEN + r")?[.!]?", re.I)
    if atom.fullmatch(text):
        # Sentence punctuation is not another decimal point. Restrict this
        # cleanup to a complete typed scalar, never arbitrary reasoning text.
        text = text.rstrip(".!")
    if len(lines) > 1:
        scalars = [line for line in lines if atom.fullmatch(line)]
        if scalars:
            block_units = {units(line) for line in scalars} - {None}
            if len(block_units) > 1:
                return _reject("mixed_units", "ambiguous")
            return _consensus([_number_scope(line, question) for line in scalars], "scalar_blocks")
    if re.search(r"\b(?:nan|inf|infinity)\b|%", text, re.I):
        return _reject("nonfinite_or_unsupported_numeric_unit")
    expected_unit = units(question)
    unit_matches = list(re.finditer(r"(?<![A-Za-z])" + UNIT_TOKEN + r"(?![A-Za-z])", text, flags=re.I))
    present = {units(m.group()) for m in unit_matches}
    if len(present) > 1:
        return _reject("mixed_units", "ambiguous")
    if expected_unit and present and present != {expected_unit}:
        return _reject("unit_mismatch")
    numeric_text = text
    for match in reversed(unit_matches):
        numeric_text = numeric_text[:match.start()] + " " + numeric_text[match.end():]
    if re.search(NUMBER + r"\s*(?:[-–—~]|\bto\b|至)\s*" + NUMBER, numeric_text, re.I):
        return _reject("numeric_range", "ambiguous")
    values = [decimal_number(m.group()) for m in NUMBER_RE.finditer(numeric_text)]
    values += [word_number(m.group()) for m in WORD_RE.finditer(numeric_text)]
    values = [v for v in values if v is not None]
    if not values:
        return _reject("no_numeric_candidate")
    if len(set(values)) != 1:
        return _reject("multiple_numeric_candidates", "ambiguous")
    if values[0] < 0:
        return _reject("negative_quantity")
    return Parsed(str(values[0]), "ok", "unique_numeric_value")


def parse_number(raw, question=""):
    _, scopes, origin = answer_regions(raw)
    return _consensus([_number_scope(s, question) for s in scopes], origin)


def ground_truth_labels(value, route):
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if isinstance(value, (list, tuple)):
        labels = []
        for entry in value:
            labels.extend(ground_truth_labels(entry, True))
        return labels
    seq = letter_sequence(value, valid)
    if seq is not None:
        if not route and len(seq) != 1:
            raise ValueError("Single-choice ground truth contains multiple letters")
        return seq
    labels = re.findall(r"(?m)^\s*(?:\*{1,2})?([A-Z])\.\s+", str(value or ""))
    if not labels or (not route and len(labels) != 1):
        raise ValueError("Invalid choice ground truth")
    return labels


def mra_score(pred, target, thresholds=THRESHOLDS):
    p, g = decimal_number(pred), decimal_number(target)
    if g is None or g < 0:
        raise ValueError("MRA ground truth must be finite and non-negative")
    if p is None:
        return 0.
    if g == 0:
        return float(p == 0)
    error = abs(p - g)
    return sum(error <= abs(g) * (1 - t) for t in thresholds) / len(thresholds)


def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    thresholds = []
    current, stop, step = Decimal(str(start)), Decimal(str(end)), Decimal(str(interval))
    if step <= 0:
        raise ValueError("MRA interval must be positive")
    while current <= stop:
        thresholds.append(current)
        current += step
    preds, targets = list(pred), list(target)
    if len(preds) != len(targets) or not preds or not thresholds:
        raise ValueError("MRA requires equal nonempty arrays and valid thresholds")
    return math.fsum(mra_score(p, g, thresholds) for p, g in zip(preds, targets)) / len(preds)


def score_case(prediction, reference, task, question="", choices=None):
    task = normalize_task(task)
    if task not in TASKS:
        raise ValueError(f"Unknown question type: {task}")
    if task in NA_QUESTION_TYPES:
        gt = decimal_number(reference)
        if gt is None or gt < 0:
            raise ValueError("Invalid numerical ground truth")
        parsed = parse_number(prediction, question)
        score = mra_score(parsed.value, gt) if parsed.status == "ok" else 0.
        gold = str(gt)
    else:
        gold = ground_truth_labels(reference, task == "route_plan")
        parsed = parse_choice(prediction, choices, task == "route_plan", question)
        score = float(parsed.status == "ok" and parsed.value == gold)
    return CaseScore(task, score, parsed.status, parsed.method, parsed.value, gold, parsed.notes)


def _rule_score(pred, gt, q_type, question="", choices=None):
    return score_case(pred, gt, q_type, question, choices).score
