"""Public pipeline contracts; pure synthetic cases and no model calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import score_results as entrypoint
from spatial4d import config, data, evaluation, rules
from spatial4d.cli import build_parser, create_judge
from spatial4d.judge import JudgeReply


class EvaluationTests(unittest.TestCase):
    def row(self, **overrides):
        return {"id": "synthetic", "ground_truth": "F", "model_answer": "F",
                "question": "Choose an action.\nA. turn left\nF. go forward", **overrides}

    def test_registry_is_the_single_task_definition(self):
        self.assertEqual(len(config.TASK_SPECS), 18)
        self.assertEqual(config.TASKS, sorted(config.TASK_SPECS))
        self.assertEqual(len(config.MCA_QUESTION_TYPES), 14)
        self.assertEqual(len(config.NA_QUESTION_TYPES), 4)
        for task in config.TASKS:
            self.assertEqual(config.TASK_SPECS[task].metric,
                             "MRA" if task in config.NA_QUESTION_TYPES else "accuracy")
        with self.assertRaises(TypeError):
            config.TASK_SPECS["unexpected"] = config.TaskSpec("unexpected", "MRA")

    def test_prepare_does_not_call_judge_or_mutate_inputs(self):
        row = self.row(options={"A": "turn left", "F": "go forward"})
        before = deepcopy(row)
        with patch("spatial4d.judge.JudgeClient.request", side_effect=AssertionError("no requests")):
            prepared = evaluation.prepare_record(row, "action_prediction", judge_enabled=True)
        self.assertEqual(row, before)
        self.assertEqual(prepared.choices, row["options"])

    def test_prepared_and_direct_case_apis_agree(self):
        for task in config.TASKS:
            numeric = task in config.NA_QUESTION_TYPES
            row = self.row(ground_truth="100" if numeric else "F", model_answer="55" if numeric else "F")
            with self.subTest(task=task):
                prepared = evaluation.prepare_record(row, task)
                self.assertEqual(evaluation.process_results(prepared), evaluation.score_record(row, task))

    def test_processing_rejects_mismatched_preparation_mode(self):
        prepared = evaluation.prepare_record(self.row(), "action_prediction", judge_enabled=True)
        with self.assertRaisesRegex(ValueError, "same judge mode"):
            evaluation.process_results(prepared)

    def test_judge_processing_keeps_GT_out_of_prompt(self):
        class FakeJudge:
            def request(self, prompt):
                self.prompt = prompt
                return JudgeReply("F")

        judge = FakeJudge()
        prepared = evaluation.prepare_record(self.row(), "action_prediction", judge_enabled=True)
        result = evaluation.process_results(prepared, judge)
        self.assertEqual(result["score"], 1)
        self.assertTrue(result["judge_called"])
        self.assertNotIn("ground_truth", judge.prompt)

    def test_aggregation_has_no_file_or_network_dependencies(self):
        cases = [evaluation.score_record(self.row(), "action_prediction")]
        before = deepcopy(cases)
        with patch.object(Path, "read_text", side_effect=AssertionError("no disk reads")), \
                patch("spatial4d.judge.JudgeClient.request", side_effect=AssertionError("no requests")):
            summary = evaluation.aggregate_results(iter(cases))
        self.assertEqual(cases, before)
        self.assertEqual(summary["macro_task_score_percent"], 100)
        self.assertFalse(summary["benchmark_complete"])

    def test_full_macro_equals_mean_of_18_task_means(self):
        cases = []
        for task in config.TASKS:
            numeric = task in config.NA_QUESTION_TYPES
            cases.append(evaluation.score_record(
                self.row(model_answer="55" if numeric else "F", ground_truth="100" if numeric else "F"), task))
        summary = evaluation.aggregate_results(cases, require_all_tasks=True)
        self.assertAlmostEqual(summary["macro_task_score_percent"], (14 * 100 + 4 * 20) / 18)
        self.assertEqual(list(summary["task_scores"]), config.TASKS)

    def test_failure_stays_in_task_denominator(self):
        cases = [evaluation.score_record(self.row(model_answer=raw), "action_prediction")
                 for raw in ("F", None, "Error: HTTP 500")]
        summary = evaluation.aggregate_results(cases)
        self.assertAlmostEqual(summary["macro_task_score_percent"], 100 / 3)
        self.assertEqual(summary["failure_groups"], {"parsed": 1, "empty_answer": 1, "inference_failure": 1})

    def test_aggregation_rejects_empty_unknown_or_required_missing_tasks(self):
        with self.assertRaises(ValueError):
            evaluation.aggregate_results([])
        case = evaluation.score_record(self.row(), "action_prediction")
        with self.assertRaisesRegex(ValueError, "Missing tasks"):
            evaluation.aggregate_results([case], require_all_tasks=True)
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            evaluation.aggregate_results([{**case, "task": "unknown"}])

    def test_data_loading_preserves_provenance_and_empty_file_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "result_action_prediction.json").write_text(json.dumps([self.row()]), encoding="utf-8")
            (path / "result_object_counting.json").write_text("[]", encoding="utf-8")
            records, empty = data.load_result_records(path)
            self.assertEqual((records[0].task, records[0].source_position), ("action_prediction", 0))
            self.assertEqual(empty, ["result_object_counting.json"])
            report = evaluation.calculate_directory(path)
            self.assertEqual(report["cases"][0]["source_file"], "result_action_prediction.json")
            self.assertEqual(report["empty_files"], empty)

    def test_context_preparation_does_not_import_qa_GT(self):
        context = {("action_prediction", "synthetic"): {"question": self.row()["question"], "ground_truth": "A"}}
        row = {"id": "synthetic", "ground_truth": "F", "model_answer": "F"}
        before = deepcopy((row, context))
        prepared = evaluation.prepare_record(row, "action_prediction", context)
        self.assertEqual(evaluation.process_results(prepared)["score"], 1)
        self.assertEqual((row, context), before)

    def test_entrypoint_helpers_are_the_same_implementations(self):
        self.assertIs(entrypoint.score_case, rules.score_case)
        self.assertIs(entrypoint.score_record, evaluation.score_record)
        self.assertIs(entrypoint.options_dict, data.options_dict)
        self.assertIs(entrypoint._guard_output, data.guard_output_path)

    def test_entrypoint_editable_RESULT_DIR_is_forwarded(self):
        with patch.object(entrypoint, "RESULT_DIR", "custom-results"), \
                patch("spatial4d.cli.main", return_value=0) as run:
            self.assertEqual(entrypoint.main(["--rule-based"]), 0)
        run.assert_called_once_with(["--rule-based"], default_result_dir="custom-results")

    def test_parser_and_factory_do_not_implicitly_enable_paid_calls(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-only", "SPATIAL4D_JUDGE_API_KEY": ""}):
            args = build_parser("custom-results").parse_args(["--judge-model", "another-model"])
            self.assertEqual(args.result_dir, Path("custom-results"))
            self.assertIsNone(create_judge(args))

    def test_core_modules_have_no_inference_or_cli_imports(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("rules", "config", "data", "judge"):
            tree = ast.parse((root / "spatial4d" / f"{name}.py").read_text(encoding="utf-8"))
            imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
            self.assertFalse(imports & {"openai", "cv2", "numpy", "cli", "evaluation", "score_results", "run_inference"}, name)


if __name__ == "__main__":
    unittest.main()
