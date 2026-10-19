"""I/O, aggregation, failure and compatibility tests; no inference requests."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spatial4d import config as rules
from score_results import calculate_directory, main, score_record
from spatial4d.data import load_qa_context, options_dict

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_options_formats(self):
        expected = {"A": "turn left", "F": "go forward"}
        for raw in [expected, ["A. turn left", "F. go forward"],
                    '["A. turn left", "F. go forward"]', "['A. turn left', 'F. go forward']",
                    "A. turn left\nF. go forward", "A. turn left F. go forward"]:
            self.assertEqual(options_dict(raw), expected)
        self.assertEqual(options_dict(["red", "blue"]), {"A": "red", "B": "blue"})
        self.assertEqual(options_dict(None), {})
        with self.assertRaises(ValueError):options_dict("A. red A. blue")

    def test_route_sequence_is_not_sorted_or_deduplicated(self):
        row = {"id": "a", "question": "1. [please fill in]\n5. [please fill in]",
               "options": {"A": "turn left", "B": "turn right"}, "is_choice": True,
               "ground_truth": "AB", "model_answer": "BA"}
        self.assertEqual(score_record(row, "route_plan")["score"], 0)
        row.update(ground_truth="BB", model_answer="B")
        self.assertEqual(score_record(row, "route_plan")["score"], 0)
        row["model_answer"] = "BB"
        self.assertEqual(score_record(row, "route_plan")["score"], 1)

    def test_numeric_precision_boundary_and_not_last_number(self):
        row = {"is_choice": False, "ground_truth": "100", "model_answer": "55"}
        self.assertEqual(score_record(row, "object_counting")["score"], .2)
        row.update(ground_truth="1.04", model_answer="1.095")
        self.assertEqual(score_record(row, "object_size_est")["score"], .9)
        row.update(ground_truth="8", model_answer="First 3, then 8")
        self.assertEqual(score_record(row, "object_counting")["score"], 0)
        row["model_answer"] = "First 3.\nFinal Answer: 8"
        self.assertEqual(score_record(row, "object_counting")["score"], 1)

    def test_error_containing_a_number_cannot_get_credit(self):
        row = {"ground_truth": "500", "model_answer": "Error: HTTP 500"}
        self.assertEqual(score_record(row, "object_counting")["failure_group"], "inference_failure")
        row.update(model_answer="500", inference_status="failed")
        self.assertEqual(score_record(row, "object_counting")["score"], 0)

    def test_zero_and_empty_preserved(self):
        row = {"ground_truth": "0", "model_answer": 0}
        self.assertEqual(score_record(row, "object_counting")["score"], 1)
        row["model_answer"] = None
        self.assertEqual(score_record(row, "object_counting")["failure_group"], "empty_answer")

    def test_units_and_context_conflicts(self):
        row = {"ground_truth": "100", "question": "Give centimeters", "model_answer": "1 m"}
        self.assertEqual(score_record(row, "abs_dist_est")["notes"], ["unit_mismatch"])
        row = {"ground_truth": "A", "model_answer": "A", "question": "A. red\nB. blue",
               "options": {"A": "blue", "B": "red"}}
        with self.assertRaises(ValueError):score_record(row, "action_prediction")

    def test_choice_type_is_not_inferred_from_answer(self):
        with self.assertRaises(ValueError):
            score_record({"ground_truth": "A", "model_answer": "A", "is_choice": False}, "action_prediction")
        with self.assertRaises(ValueError):
            score_record({"ground_truth": "A", "model_answer": "A", "is_choice": "False"}, "action_prediction")

    def test_old_results_can_restore_question_by_id_without_using_qa_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "route_plan.json"
            path.write_text(json.dumps({"id": ["example"], "question": ["1. [please fill in]\n5. [please fill in]"],
                "options": [{"A": "turn left", "B": "turn right"}], "ground_truth": ["BB"]}), encoding="utf-8")
            context = load_qa_context(folder)
            row = {"id": "example", "model_answer": "5. B. turn right\n1. A. turn left", "ground_truth": "AB"}
            result = score_record(row, "route_plan", context)
            self.assertEqual(result["prediction_normalized"], ["A", "B"])
            self.assertEqual(result["score"], 1)
            self.assertNotIn("ground_truth", context[("route_plan", "example")])


class AggregationTests(unittest.TestCase):
    def write(self, root, filename, rows):
        (Path(root)/filename).write_text(json.dumps(rows), encoding="utf-8")

    def test_macro_is_equal_task_weight_not_micro_or_file_weight(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write(folder, "result_shard1.json", [{"id": "n1", "question_type": "object_counting", "model_answer": "2", "ground_truth": "2"}])
            self.write(folder, "result_shard2.json", [{"id": "n2", "question_type": "object_counting", "model_answer": "3", "ground_truth": "3"}])
            self.write(folder, "result_action_prediction.json", [{"id": "c", "model_answer": "B", "ground_truth": "A"}])
            report = calculate_directory(folder)
            self.assertEqual(report["macro_task_score_percent"], 50)
            self.assertEqual(report["task_scores"]["object_counting"]["samples"], 2)
            self.assertFalse(report["benchmark_complete"])
            with self.assertRaises(ValueError):calculate_directory(folder, require_all_tasks=True)

    def test_all_failures_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write(folder, "result_object_counting.json", [{"id": "ok", "model_answer": "2", "ground_truth": "2"},
                {"id": "empty", "model_answer": "", "ground_truth": "2"},
                {"id": "error", "model_answer": "Error: 2", "ground_truth": "2"}])
            report = calculate_directory(folder)
            self.assertEqual(report["samples"], 3)
            self.assertAlmostEqual(report["macro_task_score_percent"], 100/3)
            self.assertEqual(report["failure_groups"], {"parsed": 1, "empty_answer": 1, "inference_failure": 1})

    def test_empty_and_unknown_inputs_cannot_report_fake_zero_macro(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):calculate_directory(folder)
            self.write(folder, "result_route_plan.json", [])
            with self.assertRaises(ValueError):calculate_directory(folder)
            self.write(folder, "result_unknown.json", [{"model_answer": "A", "ground_truth": "A"}])
            with self.assertRaises(ValueError):calculate_directory(folder)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            row = {"id": "same", "question_type": "object_counting", "model_answer": "2", "ground_truth": "2"}
            self.write(folder, "result_a.json", [row]);self.write(folder, "result_b.json", [row])
            with self.assertRaises(ValueError):calculate_directory(folder)

    def test_full_eighteen_task_report(self):
        with tempfile.TemporaryDirectory() as folder:
            for task in rules.TASKS:
                numeric = task in rules.NA_QUESTION_TYPES
                row = {"id": task, "question_type": task, "model_answer": "2" if numeric else "A", "ground_truth": "2" if numeric else "A"}
                self.write(folder, f"result_{task}.json", [row])
            report = calculate_directory(folder, require_all_tasks=True)
            self.assertEqual(report["macro_task_score_percent"], 100)
            self.assertEqual(report["evaluated_tasks"], 18)
            self.assertTrue(report["benchmark_complete"])

    def test_cli_output_and_input_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write(folder, "result_object_counting.json", [{"id": "x", "model_answer": "2", "ground_truth": "2"}])
            output = Path(folder)/"metrics.json"
            run = subprocess.run([sys.executable, str(ROOT/"score_results.py"), "--rule-based", "--result-dir", folder, "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("PARTIAL TASK MACRO (1/18)", run.stdout)
            self.assertEqual(json.loads(output.read_text())["macro_task_score_percent"], 100)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--result-dir", folder, "--output", str(Path(folder)/"result_object_counting.json")])
