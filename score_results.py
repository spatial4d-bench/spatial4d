"""Score Spatial4D model answers with rules or an optional judge.

Implementation lives in spatial4d/. Scoring helpers are re-exported here;
importing this file never starts evaluation or model requests.
"""
from spatial4d import cli as _cli
from spatial4d.config import RESULT_DIR as _DEFAULT_RESULT_DIR
from spatial4d.config import (
    RULES_VERSION, PLATFORM_SOURCE_SHA256, OPTION_LABELS, FAIL_MSG,
    TaskSpec, TASK_SPECS, MCA_QUESTION_TYPES, NA_QUESTION_TYPES, TASKS,
    ALIASES,
)
from spatial4d.rules import (
    THRESHOLDS, NUMBER, NUMBER_RE, OPTION_LINE, SMALL_WORDS,
    WORD_RE, Parsed, CaseScore, normalize_task, strip_format,
    canonical_text, extract_options, letter_sequence, decimal_number, word_number,
    units, REASON_TAG, ANSWER_TAG, ANSWER_HEADING, STRUCT_LABEL,
    STRUCT_INDEX, UNIT_TOKEN, _reject, _clean_reasoning, _presentation,
    _structured_value, answer_regions, _scope_failure, _consensus, _choice_atom,
    _inline_route_fields, _parallel_route_fields, _inline_option_block, _block_entries, _copied_option_list,
    _embedded_option_signature, _route_block, _choice_scope, _validate_sequence, parse_choice,
    _number_scope, parse_number, ground_truth_labels, mra_score, mean_relative_accuracy,
    score_case, _rule_score,
)
from spatial4d.judge import (
    _LETTER_RE, _STRICT_NUMBER_RE, _build_option_str, _build_mca_judge_prompt, _build_na_extract_prompt,
    _parse_finite_number, _numbers_in_text, _is_source_number, _extract_letter, _extract_letter_sequence,
    _extract_gt_letters, JudgeConfigurationError, JudgeReply, _NoJudgeRedirect, JudgeClient,
)
from spatial4d.data import (
    ResultRecord, load_result_records, task_name, filename_task, options_dict,
    read_qa_rows, load_qa_context, resolve_context, guard_output_path,
)
from spatial4d.evaluation import (
    _judge_score, extract_and_format, calc_score, PreparedRecord, prepare_record,
    process_results, score_record, _preflight_judge_record, missing_tasks, _check_coverage,
    aggregate_results, calculate_directory,
)

# Keep this legacy editable setting: main() forwards its current value.
# It is also available centrally as spatial4d.config.RESULT_DIR.
RESULT_DIR = _DEFAULT_RESULT_DIR

# Backward-compatible name for callers of the old safety helper.
_guard_output = guard_output_path


def main(argv=None):
    return _cli.main(argv, default_result_dir=RESULT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
