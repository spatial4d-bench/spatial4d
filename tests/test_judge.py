"""Synthetic judge tests. No requests are made to real model services."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import score_results as scoring



class API(BaseHTTPRequestHandler):
    calls = []
    replies = deque()

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append({'path': self.path, 'body': payload, 'authorization': self.headers.get('Authorization')})
        status, answer = self.replies.popleft() if self.replies else (200, 'F')
        if isinstance(answer, dict):
            data = answer
        elif status == 200:
            data = {'choices': [{'message': {'content': answer}}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 2, 'total_tokens': 14}}
        else:
            data = {'error': {'message': 'Synthetic failure; no actual model'}}
        encoded = json.dumps(data).encode('utf-8')
        self.send_response(status)
        if status == 302:
            self.send_header('Location', '/must-not-follow')
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), API)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='spatial4d-judge-fixtures-')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.results = self.folder / 'results'
        self.qa = self.folder / 'qa'
        self.results.mkdir()
        self.qa.mkdir()
        self.base = f'http://127.0.0.1:{self.server.server_port}/v1'
        API.calls.clear()
        API.replies.clear()

    def client(self, **kwargs):
        settings = {'base_url': self.base, 'qps': 100000}
        settings.update(kwargs)
        return scoring.JudgeClient('synthetic-only-no-real-key', **settings)

    def row(self, task='action_prediction', **kwargs):
        numeric = task in scoring.NA_QUESTION_TYPES
        row = dict(id='synthetic-' + task, question='Give centimeters' if numeric else
                   'Choose.\nA. left\nB. right\nF. forward',
                   question_type=task, options=None if numeric else ['A. left', 'B. right', 'F. forward'],
                   ground_truth='100' if numeric else 'F', model_answer='55' if numeric else 'F',
                   is_choice=not numeric)
        if task == 'route_plan':
            row.update(question='1. [please fill in]\n5. [please fill in]\nA. left\nB. right\nF. forward',
                       ground_truth='FB', model_answer='FB')
        row.update(kwargs)
        return row

    def write(self, task, rows):
        path = self.results / f'result_{task}.json'
        path.write_text(json.dumps(rows), encoding='utf-8')
        return path

    def cli(self, *args, env=None, no_site=False):
        clean_env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        for name in ('SPATIAL4D_JUDGE_API_KEY', 'SPATIAL4D_JUDGE_MODEL', 'SPATIAL4D_JUDGE_BASE_URL'):
            clean_env.pop(name, None)
        clean_env.update(env or {})
        return subprocess.run([sys.executable, *(['-S'] if no_site else []),
                               str(ROOT / 'score_results.py'), '--result-dir', str(self.results), *args],
                              capture_output=True, text=True, encoding='utf-8', env=clean_env, timeout=30)

    def test_no_key_uses_rules_without_network_or_dependencies(self):
        self.write('action_prediction', [self.row()])
        output = self.folder / 'rules.json'
        run = self.cli('--output', str(output), env={'OPENAI_API_KEY': 'unrelated-inference-key'}, no_site=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['scoring_mode'], 'rule_based')
        self.assertEqual(report['new_model_calls'], 0)
        self.assertEqual(report['macro_task_score_percent'], 100)
        self.assertEqual(API.calls, [])

    def test_judge_model_alone_does_not_enable_paid_calls(self):
        self.write('action_prediction', [self.row()])
        run = self.cli('--judge-model', 'gpt-4o')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('mode=rule_based', run.stdout)
        self.assertEqual(API.calls, [])

    def test_api_key_alias_and_default_gpt4o_use_real_http(self):
        self.write('action_prediction', [self.row()])
        output = self.folder / 'judge.json'
        run = self.cli('--api-key', 'synthetic-only-no-real-key', '--judge-base-url', self.base,
                       '--judge-qps', '100000', '--output', str(output), no_site=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(API.calls), 1)
        self.assertEqual(API.calls[0]['path'], '/v1/chat/completions')
        self.assertEqual(API.calls[0]['authorization'], 'Bearer synthetic-only-no-real-key')
        self.assertEqual(API.calls[0]['body']['model'], 'gpt-4o')
        self.assertIs(API.calls[0]['body']['store'], False)
        report = json.loads(output.read_text())
        self.assertEqual(report['new_model_calls'], 1)
        self.assertEqual(report['macro_task_score_percent'], 100)
        self.assertNotIn('synthetic-only-no-real-key', run.stdout + run.stderr + output.read_text())

    def test_dedicated_key_environment_enables_judge(self):
        self.write('action_prediction', [self.row()])
        run = self.cli('--judge-base-url', self.base, '--judge-model', 'custom-served-model',
                       env={'SPATIAL4D_JUDGE_API_KEY': 'synthetic-env-key'})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(API.calls[0]['body']['model'], 'custom-served-model')

    def test_force_rules_overrides_available_key(self):
        self.write('action_prediction', [self.row()])
        run = self.cli('--rule-based', '--judge-base-url', 'not-even-a-valid-url',
                       env={'SPATIAL4D_JUDGE_API_KEY': 'synthetic-env-key'})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(API.calls, [])

    def test_explicit_empty_key_disables_environment_key(self):
        self.write('action_prediction', [self.row()])
        run = self.cli('--api-key', '', env={'SPATIAL4D_JUDGE_API_KEY': 'synthetic-env-key'})
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(API.calls, [])

    def test_explicit_choice_still_uses_judge_not_rules(self):
        API.replies.append((200, 'B'))
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(len(API.calls), 1)
        self.assertEqual(result['score'], 0)
        self.assertIsNone(result['failed_reason'])

    def test_pure_numerical_answer_never_calls_judge(self):
        result = scoring.score_record(self.row('object_counting'), 'object_counting', judge=self.client())
        self.assertEqual(result['score'], .2)
        self.assertEqual(result['judge_mode'], 'na_direct')
        self.assertEqual(API.calls, [])

    def test_formatted_numbers_call_judge_and_preserve_provenance(self):
        for raw, reply, expected in [('100 cm', '100', 1), ('Two', '2.0', 1), ('100cm', '77', 0)]:
            with self.subTest(raw=raw):
                API.replies.append((200, reply))
                row = self.row('object_counting', model_answer=raw, ground_truth='2' if raw == 'Two' else '100')
                result = scoring.score_record(row, 'object_counting', judge=self.client())
                self.assertEqual(result['score'], expected)
                if expected == 0:
                    self.assertEqual(result['failed_reason'], 'judge_value_not_in_source')
        self.assertEqual(len(API.calls), 3)

    def test_judge_cannot_invent_number_from_gt(self):
        API.replies.append((200, '777777'))
        row = self.row('abs_dist_est', model_answer='100 cm', ground_truth='777777')
        result = scoring.score_record(row, 'abs_dist_est', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_value_not_in_source')
        self.assertNotIn('777777', API.calls[0]['body']['messages'][0]['content'])

    def test_F_fills_from_question_without_preprocessor(self):
        row = self.row(options=['A. left', 'B. right'])
        result = scoring.score_record(row, 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 1)
        self.assertIn('F. forward', API.calls[0]['body']['messages'][0]['content'])

    def test_explicit_options_have_v3_priority_in_judge_mode(self):
        row = self.row(options={'A': 'left', 'B': 'right', 'F': 'explicit alternative'})
        result = scoring.score_record(row, 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 1)
        self.assertIn('F. explicit alternative', API.calls[0]['body']['messages'][0]['content'])
        with self.assertRaises(ValueError):
            scoring.score_record(row, 'action_prediction')

    def test_flat_A_F_reference_fields_can_supply_options(self):
        row = self.row(question='Choose the direction', options=None, A='left', F='forward')
        self.assertEqual(scoring.score_record(row, 'action_prediction', judge=self.client())['score'], 1)

    def test_route_order_repeats_and_completeness(self):
        for reply, expected in [('FFB', 1), ('FBF', 0), ('FB', 0), ('F', 0)]:
            with self.subTest(reply=reply):
                API.replies.append((200, reply))
                row = self.row('route_plan', ground_truth='FFB', model_answer='FFB',
                               question='1. [please fill in]\n3. [please fill in]\n5. [please fill in]\nA. left\nB. right\nF. forward')
                result = scoring.score_record(row, 'route_plan', judge=self.client())
                self.assertEqual(result['score'], expected)
                self.assertEqual(result['parse_status'], 'ok')

    def test_Z_is_normal_no_match_not_judge_failure(self):
        API.replies.append((200, 'Z'))
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failure_group'], 'answer_no_match')
        self.assertIsNone(result['failed_reason'])
        self.assertEqual(len(API.calls), 1)

    def test_invalid_choice_does_not_fall_back_to_correct_raw_answer(self):
        API.replies.append((200, '???'))
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_unparseable_choice')

    def test_empty_reply_retries_v3_temperatures_then_stays_zero(self):
        API.replies.extend([(200, None)] * 3)
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_empty_response')
        self.assertEqual([c['body']['temperature'] for c in API.calls], [0, .5, 1])
        self.assertEqual(result['judge_attempts'], 3)
        API.calls.clear()
        API.replies.extend([(200, None)] * 9 + [(200, 'F')])
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client(max_attempts=10))
        self.assertEqual(result['score'], 1)
        self.assertEqual(len(API.calls), 10)
        self.assertTrue(all(0 <= c['body']['temperature'] <= 2 for c in API.calls))

    def test_transient_HTTP_failure_retries_then_recovers(self):
        API.replies.extend([(500, None), (429, None), (200, 'F')])
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 1)
        self.assertEqual(result['judge_attempts'], 3)

    def test_exhausted_HTTP_failures_stay_in_denominator(self):
        self.write('action_prediction', [self.row(id='one'), self.row(id='two', question='Choose again.\nA. left\nF. forward')])
        API.replies.extend([(200, 'F')] + [(500, None)] * 3)
        report = scoring.calculate_directory(self.results, judge=self.client())
        self.assertEqual(report['samples'], 2)
        self.assertEqual(report['macro_task_score_percent'], 50)
        self.assertEqual(report['new_model_calls'], 4)
        self.assertEqual(report['failure_groups'], {'parsed': 1, 'judge_failure': 1})

    def test_auth_or_model_error_stops_instead_of_retrying_benchmark(self):
        for code in (400, 401, 403, 404, 422):
            with self.subTest(code=code):
                API.calls.clear()
                API.replies.append((code, None))
                with self.assertRaises(scoring.JudgeConfigurationError):
                    scoring.score_record(self.row(), 'action_prediction', judge=self.client())
                self.assertEqual(len(API.calls), 1)

    def test_redirect_never_forwards_bearer(self):
        API.replies.append((302, None))
        with self.assertRaises(scoring.JudgeConfigurationError):
            scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(len(API.calls), 1)

    def test_timeout_is_audited_without_rule_fallback(self):
        client = self.client()
        with patch.object(client._opener, 'open', side_effect=socket.timeout):
            result = scoring.score_record(self.row(), 'action_prediction', judge=client)
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_transport_error')
        self.assertEqual(client.api_requests, 3)

    def test_invalid_response_schema_is_audited(self):
        API.replies.extend([(200, {'choices': [{'message': {'content': ['not a text string']}}]})] * 3)
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_invalid_response')

    def test_missing_choices_is_invalid_schema_not_empty_completion(self):
        API.replies.extend([(200, {'error': 'synthetic body error'})] * 3)
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client())
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_invalid_response')

    def test_incomplete_HTTP_body_is_transport_failure_not_crash(self):
        client = self.client()
        with patch.object(client._opener, 'open', side_effect=IncompleteRead(b'', 10)):
            result = scoring.score_record(self.row(), 'action_prediction', judge=client)
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['failed_reason'], 'judge_transport_error')

    def test_key_echo_is_redacted_from_audit_and_cache(self):
        API.replies.append((200, 'synthetic-only-no-real-key'))
        cache = self.folder / 'cache.jsonl'
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        self.assertNotIn('synthetic-only-no-real-key', json.dumps(result) + cache.read_text())
        self.assertIn('[REDACTED]', result['judge_response'])

    def test_persistent_cache_avoids_repeat_paid_requests(self):
        cache = self.folder / 'cache.jsonl'
        first = scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        second = scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        self.assertEqual(first['score'], second['score'])
        self.assertTrue(second['judge_cache_hit'])
        self.assertEqual(second['judge_attempts'], 0)
        self.assertEqual(len(API.calls), 1)

    def test_cache_identity_changes_with_model_and_prompt(self):
        cache = self.folder / 'cache.jsonl'
        scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache, model='another'))
        scoring.score_record(self.row(question='New question\nA. left\nF. forward'), 'action_prediction',
                             judge=self.client(cache_path=cache))
        self.assertEqual(len(API.calls), 3)

    def test_existing_cache_without_final_newline_appends_safely(self):
        cache = self.folder / 'cache.jsonl'
        scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        cache.write_bytes(cache.read_bytes().rstrip(b'\n'))
        scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache, model='other'))
        self.assertEqual(len(cache.read_text().splitlines()), 2)
        restored = self.client(cache_path=cache)
        result = scoring.score_record(self.row(), 'action_prediction', judge=restored)
        self.assertTrue(result['judge_cache_hit'])

    def test_api_failure_is_not_persisted_as_successful_cache(self):
        cache = self.folder / 'cache.jsonl'
        API.replies.extend([(500, None)] * 3)
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        self.assertEqual(result['score'], 0)
        self.assertFalse(cache.exists())
        result = scoring.score_record(self.row(), 'action_prediction', judge=self.client(cache_path=cache))
        self.assertEqual(result['score'], 1)

    def test_malformed_cache_is_rejected_before_request(self):
        cache = self.folder / 'cache.jsonl'
        cache.write_text('{broken', encoding='utf-8')
        with self.assertRaises(scoring.JudgeConfigurationError):
            self.client(cache_path=cache)
        self.assertEqual(API.calls, [])

    def test_cache_cannot_overwrite_input_or_report(self):
        source = self.write('action_prediction', [self.row()])
        before = source.read_bytes()
        run = self.cli('--api-key', 'synthetic', '--judge-cache', str(source), '--judge-base-url', self.base)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(source.read_bytes(), before)
        same = self.folder / 'same.jsonl'
        run = self.cli('--api-key', 'synthetic', '--judge-cache', str(same), '--output', str(same), '--judge-base-url', self.base)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(API.calls, [])

    def test_new_report_cannot_pollute_QA_JSON_directory(self):
        self.write('action_prediction', [self.row()])
        run = self.cli('--qa-dir', str(self.qa), '--output', str(self.qa / 'new_report.json'))
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse((self.qa / 'new_report.json').exists())
        self.assertEqual(API.calls, [])

    def test_missing_context_in_later_file_blocks_all_requests(self):
        self.write('action_prediction', [self.row()])
        self.write('state_change_det', [self.row('state_change_det', question='', options=None)])
        with self.assertRaisesRegex(ValueError, 'context'):
            scoring.calculate_directory(self.results, judge=self.client())
        self.assertEqual(API.calls, [])

    def test_full_benchmark_requirement_checked_before_any_request(self):
        self.write('action_prediction', [self.row()])
        with self.assertRaisesRegex(ValueError, 'Missing tasks'):
            scoring.calculate_directory(self.results, require_all_tasks=True, judge=self.client())
        self.assertEqual(API.calls, [])

    def test_duplicate_ids_block_all_requests(self):
        self.write('action_prediction', [self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            scoring.calculate_directory(self.results, judge=self.client())
        self.assertEqual(API.calls, [])

    def test_empty_or_inference_failed_answer_never_calls_judge(self):
        for overrides in ({'model_answer': ''}, {'inference_status': 'failed'}, {'model_answer': 'Error: HTTP 500'}):
            with self.subTest(overrides=overrides):
                result = scoring.score_record(self.row(**overrides), 'action_prediction', judge=self.client())
                self.assertEqual(result['score'], 0)
        self.assertEqual(API.calls, [])

    def test_old_results_can_restore_context_without_using_QA_GT(self):
        self.write('action_prediction', [{'id': 'old', 'model_answer': 'F', 'ground_truth': 'F', 'is_choice': True}])
        (self.qa / 'action_prediction.json').write_text(json.dumps([dict(self.row(id='old'), ground_truth='A')]), encoding='utf-8')
        report = scoring.calculate_directory(self.results, self.qa, judge=self.client())
        self.assertEqual(report['macro_task_score_percent'], 100)
        self.assertEqual(report['cases'][0]['ground_truth'], 'F')

    def test_full_18_task_judge_roundtrip_has_expected_macro(self):
        for task in scoring.TASKS:
            row = self.row(task)
            row['question'] += '\nSynthetic task: ' + task
            self.write(task, [row])
            if task not in scoring.NA_QUESTION_TYPES:
                API.replies.append((200, 'FB' if task == 'route_plan' else 'F'))
        report = scoring.calculate_directory(self.results, require_all_tasks=True, judge=self.client())
        self.assertEqual(report['samples'], 18)
        self.assertEqual(report['new_model_calls'], 14)
        self.assertEqual(report['judge']['judged_samples'], 14)
        self.assertAlmostEqual(report['macro_task_score_percent'], (14 * 100 + 4 * 20) / 18)

    def test_reused_client_reports_per_run_not_cumulative_usage(self):
        self.write('action_prediction', [self.row()])
        client = self.client()
        one = scoring.calculate_directory(self.results, judge=client)
        two = scoring.calculate_directory(self.results, judge=client)
        self.assertEqual(one['new_model_calls'], 1)
        self.assertEqual(two['new_model_calls'], 0)
        self.assertEqual(two['judge']['cache_hits'], 1)
        self.assertEqual(two['judge']['usage']['total_tokens'], 0)

    def test_bad_endpoint_or_settings_rejected(self):
        for kwargs in ({'timeout': 0}, {'qps': 0}, {'max_attempts': 0}, {'max_tokens': 0},
                       {'timeout': float('nan')}, {'qps': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(scoring.JudgeConfigurationError):
                self.client(**kwargs)
        for base in ('http://example.invalid/v1', 'https://name:secret@example.invalid/v1',
                     'https://example.invalid/v1?api_key=secret'):
            with self.assertRaises(scoring.JudgeConfigurationError):
                scoring.JudgeClient('synthetic', base_url=base)


if __name__ == "__main__":
    unittest.main()
