"""Synthetic regressions carried over from the v2.5 rule audit."""

import unittest

from spatial4d import rules as m

OPTIONS = {'A': 'turn left', 'B': 'turn right', 'F': 'go forward'}

QUESTION = '1. [please fill in]\n5. [please fill in]\n' + '\n'.join(f'{k}. {v}' for k,v in OPTIONS.items())



class StructuralTests(unittest.TestCase):

    def test_choice_formats_generalize_under_letter_permutation(self):
        for label, content in OPTIONS.items():
            for raw in [label, '(' + label + ')', '[' + label + ']', label + '. ' + content, '**' + label + '. ' + content + '**', '. ' + label + '. ' + content, {'answer': label}, {'final_answer': label}, {'reasoning_content': 'Considering another option', 'content': label}, f'<think>A. turn left</think>\n{label}', f'<reasoning>Unrelated text.</reasoning>\n{label}', f'prefilled reasoning </think>\n{label}', f'<answer>{label}</answer>', f'\\boxed{{{label}}}', f'Answer: {label}', f'Final Answer: {label}', f'最终答案：{label}']:
                with self.subTest(raw=raw):
                    p = m.parse_choice(raw, OPTIONS)
                    self.assertEqual((p.status, p.value), ('ok', [label]))

    def test_literal_option_signatures_without_conclusion_whitelist(self):
        for raw in ['The correct statement is B. turn right.', 'Therefore, the person is going to B. turn right.', 'The correct location is B. turn right.', '结论如下：B. turn right.', 'Voici mon choix : B. turn right.']:
            self.assertEqual(m.parse_choice(raw, OPTIONS).value, ['B'])
        for raw in ['Do not select B. turn right.', 'If I choose B. turn right.', 'Anything B. unrecognized action.']:
            self.assertNotEqual(m.parse_choice(raw, OPTIONS).status, 'ok')
        p = m.parse_choice('Both blanks require the same action: turn right.\n\nB. turn right', OPTIONS, True, QUESTION)
        self.assertEqual(p.status, 'incomplete')
        self.assertEqual(p.value, ['B'])

    def test_unmarked_candidates_are_not_first_or_last_wins(self):
        for raw in ['A. turn left\nB. turn right', 'A. turn left\n\nExplanation.\n\nB. turn right', 'A. turn left B. turn right']:
            self.assertEqual(m.parse_choice(raw, OPTIONS).status, 'ambiguous')
        self.assertEqual(m.parse_choice('A. turn left\n\nExplanation.\n\nA. turn left', OPTIONS).value, ['A'])
        self.assertEqual(m.parse_choice('B. turn right\n\nI actually mean A. turn left.', OPTIONS).status, 'ambiguous')

    def test_prose_alternatives_do_not_override_isolated_answers(self):
        for raw in ['Other options (A), (F) were discussed.\n\nB. turn right', 'B. turn right\n\nThe discussion mentions option A and F.', '- A. turn left: explanation of an alternative.\n- F. go forward: another alternative.\n\nB. turn right']:
            self.assertEqual(m.parse_choice(raw, OPTIONS).value, ['B'])
        copied = 'A. turn left\nB. turn right\nF. go forward\n\nB. turn right'
        self.assertEqual(m.parse_choice(copied, OPTIONS).value, ['B'])

    def test_explicit_scopes_require_consensus(self):
        for raw in ['<answer>A</answer><answer>B</answer>', 'Final Answer: A\nFinal Answer: B', {'answer': 'A', 'final_answer': 'B'}]:
            self.assertEqual(m.parse_choice(raw, OPTIONS).status, 'ambiguous')
        self.assertEqual(m.parse_choice('<answer>A</answer><answer>A. turn left</answer>', OPTIONS).value, ['A'])
        self.assertEqual(m.parse_choice('Answer: A\nFinal Answer: B', OPTIONS).value, ['B'])

    def test_reasoning_tags_are_balanced_and_never_mined(self):
        for raw in ['<think>A', '<think>A</analysis>B', '<answer>A']:
            self.assertNotEqual(m.parse_choice(raw, OPTIONS).status, 'ok')
        self.assertEqual(m.parse_choice('<think>A<analysis>F</analysis></think>B', OPTIONS).value, ['B'])
        self.assertEqual(m.parse_choice('<think>A</think>', OPTIONS).status, 'incomplete')

    def test_routes_require_complete_order_and_preserve_repeats_F(self):
        for raw in ['BF', 'B,F', 'B F', ['B', 'F'], {'answer': ['B', 'F']}, 'B. turn right F. go forward', 'B. turn right, F. go forward', 'B. turn right\nF. go forward', '1. B. turn right\n2. F. go forward', '5. F. go forward\n1. B. turn right', {'1': 'B', '5': 'F'}, 'Step 5: F. go forward\n\nExplanation.\n\nStep 1: B. turn right']:
            p = m.parse_choice(raw, OPTIONS, True, QUESTION)
            self.assertEqual((p.status, p.value), ('ok', ['B', 'F']), raw)
        self.assertEqual(m.parse_choice(['F', 'F'], OPTIONS, True, QUESTION).value, ['F', 'F'])
        self.assertEqual(m.parse_choice('B', OPTIONS, True, QUESTION).status, 'incomplete')
        self.assertEqual(m.parse_choice(['B', 'F', 'A'], OPTIONS, True, QUESTION).status, 'invalid')
        self.assertEqual(m.score_case('FB', 'B. turn right\nF. go forward', 'route_plan', QUESTION, OPTIONS).score, 0.0)

    def test_route_block_grammar_invariant_to_explanation_position(self):
        blocks = ['B. turn right, F. go forward', '1. B (turn right)\n5. F (go forward)', '- Step 1 → B. turn right\n- Step 5 → F. go forward', '1. B, 5. F', '1. B – turn right\n5. F – go forward']
        for block in blocks:
            for raw in [block, block + '\n\nExplanation mentions A and B.', 'Explanation mentions A and B.\n\n' + block]:
                p = m.parse_choice(raw, OPTIONS, True, QUESTION)
                self.assertEqual((p.status, p.value), ('ok', ['B', 'F']), raw)
        raw = 'B. turn right\nF. go forward\n\n1. B. turn right\n5. F. go forward'
        self.assertEqual(m.parse_choice(raw, OPTIONS, True, QUESTION).value, ['B', 'F'])

    def test_numbered_full_route_and_split_answer_paragraphs(self):
        q = '1. [please fill in]\n2. continue through door\n3. [please fill in]\n4. approach target\n'
        raw = '1. F. go forward\n2. continue through door\n3. B. turn right\n4. approach target\n\nExplanation.'
        self.assertEqual(m.parse_choice(raw, OPTIONS, True, q).value, ['F', 'B'])
        raw = 'F. go forward — explanation of first action.\n\nB. turn right — explanation of second action.'
        self.assertEqual(m.parse_choice(raw, OPTIONS, True, q).value, ['F', 'B'])
        self.assertEqual(m.parse_choice('F. go forward', OPTIONS, True, q).status, 'incomplete')

    def test_inline_step_metadata_and_connectors(self):
        q = '3. [please fill in]\n7. [please fill in]'
        for raw in ['3. F. go forward, 7. B. turn right', '3 – F. go forward; 7 – B. turn right', '3. F. go forward and 7. B. turn right', 'F. go forward (step 3); B. turn right (step 7)', 'F. go forward (for step 3), B. turn right (for step 7)', 'B. turn right (step 7) and F. go forward (step 3)']:
            p = m.parse_choice(raw, OPTIONS, True, q)
            self.assertEqual((p.status, p.value), ('ok', ['F', 'B']), raw)

    def test_routes_do_not_merge_conflicting_or_partial_blocks(self):
        for raw in ['B. turn right\nF. go forward\n\nExplanation.\n\nF. go forward\nB. turn right', 'B. turn right\nF. go forward\n\nExplanation.\n\nB. turn right']:
            self.assertEqual(m.parse_choice(raw, OPTIONS, True, QUESTION).status, 'ambiguous')
        raw = 'B. turn right\nF. go forward\n\nExplanation.\n\nB. turn right\nF. go forward'
        self.assertEqual(m.parse_choice(raw, OPTIONS, True, QUESTION).value, ['B', 'F'])

    def test_numeric_representation_and_units(self):
        for raw, value in [('1,234.5cm', '1234.5'), ('1.2e2 cm', '1.2E+2'), ('Two', '2'), ('There are eight boxes.', '8'), ('8 square meters', '8'), ('8 m²', '8'), ('8 m^2', '8'), ('8 m2', '8')]:
            p = m.parse_number(raw)
            self.assertEqual(p.status, 'ok', raw)
            self.assertEqual(m.decimal_number(p.value), m.decimal_number(value))
        self.assertEqual(m.parse_number('8cm', 'Give meters').notes, ['unit_mismatch'])
        self.assertEqual(m.parse_number('8 m and 8 cm').status, 'ambiguous')
        self.assertEqual(m.parse_number(0).value, '0')
        self.assertEqual(m.parse_number({'answer': 0}).value, '0')
        self.assertEqual(m.parse_choice({'answer': None}, OPTIONS).status, 'incomplete')

    def test_numeric_no_first_last_or_range_midpoint(self):
        for raw in ['First 3, then final 8.', 'The first part is 3. The total area is 8.', '8–10', '8 to 10', '29,99', '8 or 9']:
            self.assertEqual(m.parse_number(raw).status, 'ambiguous', raw)
        self.assertEqual(m.parse_number('3, 8.\nFinal Answer: 8').value, '8')
        self.assertEqual(m.parse_number('<think>3 and 4</think>8').value, '8')
        self.assertEqual(m.parse_number('8\n8').value, '8')
        for raw in ['NaN', 'Infinity', '-1', '30%']:
            self.assertEqual(m.parse_number(raw).status, 'invalid')

    def test_numeric_structural_blocks_are_position_invariant(self):
        for raw in ['8\n\nA derivation mentioning 3 and 5.', 'A derivation mentioning 3 and 5.\n\n8', '8\n\nA derivation mentioning 3 and 5.\n\n8']:
            self.assertEqual(m.parse_number(raw).value, '8')
        self.assertEqual(m.parse_number('8\n\nA derivation mentioning 3 and 5.\n\n9').status, 'ambiguous')
        self.assertEqual(m.parse_number('A derivation in 2 meters.\n\n200', 'Give centimeters').value, '200')
        self.assertEqual(m.parse_number('A derivation in 2 meters.\n\n2 m', 'Give centimeters').notes, ['unit_mismatch'])
        for raw in [{'answer': [8, 8]}, {'answer': True}, {'answer': {'value': 8}}]:
            self.assertEqual(m.parse_number(raw).status, 'invalid')
        for prefix in ['Approximately', 'About', 'Roughly', 'Around']:
            raw = 'A rectangle with dimensions 2 m and 4 m.\n\n' + prefix + ' 8 square meters'
            self.assertEqual(m.parse_number(raw, 'Give square meters').value, '8')

    def test_prediction_parsing_is_independent_of_correct_answer(self):
        for gt in ['A. turn left', 'B. turn right', 'F. go forward']:
            c = m.score_case('B. turn right', gt, 'action_prediction', choices=OPTIONS)
            self.assertEqual(c.prediction_normalized, ['B'])
            self.assertEqual(c.parse_status, 'ok')

class RouteAuditTests(unittest.TestCase):

    def test_label_separators_permuted_options_and_step_numbers(self):
        for labels in [('A', 'B'), ('F', 'C'), ('D', 'E')]:
            a, b = labels
            opts = {a: 'go downstairs', b: 'turn left', 'Z': 'move forward'}
            for first, last in [(1, 5), (3, 8), (12, 14)]:
                q = f'{first}. [please fill in]\n{last}. [please fill in]'
                for raw in [f'{a}, go downstairs; {b}, turn left', f'{a} go downstairs, {b} turn left', f'{first}. {a} (go downstairs), {last}. {b} (turn left)', f'{a}. go downstairs (step {first}), then {b}. turn left (step {last}).', f'{a}. go downstairs (step {first}) — and for step {last}, {b}. turn left.', f'{a}, {b} (go downstairs, turn left)', f'Une rubrique:\n- Step {last}: {b}. turn left\n- Step {first}: {a}. go downstairs', f'一种标题:\n{first}. {a}, go downstairs\n{last}. {b}, turn left', f'A heading: {first} → {a}. go downstairs, and {last} → {b}. turn left.', f'Answers:\nStep {first}: {a}. go downstairs (to reach the hall)\nStep {last}: {b}. turn left (to face the door)']:
                    with self.subTest(raw=raw):
                        p = m.parse_choice(raw, opts, True, q)
                        self.assertEqual((p.status, p.value), ('ok', [a, b]))

    def test_parallel_tuple_preserves_repeats_and_checks_both_halves(self):
        opts = {'F': 'turn left', 'B': 'turn right'}
        q = '3. [please fill in]\n9. [please fill in]'
        for raw in ['F, F turn left, turn left', 'F, F (turn left, turn left)']:
            self.assertEqual(m.parse_choice(raw, opts, True, q).value, ['F', 'F'])
        for raw in ['F, B (turn right, turn left)', 'F, F (turn left, turn right)']:
            self.assertNotEqual(m.parse_choice(raw, opts, True, q).status, 'ok')

    def test_conflicts_and_annotated_alternatives_are_not_filled_in(self):
        opts = {'A': 'turn left', 'B': 'turn right', 'F': 'go forward'}
        q = '1. [please fill in]\n5. [please fill in]'
        for raw in ['A, turn right; B, turn right', 'A turn right, B turn right', 'A. turn left (step 1), then B. turn right (step 3)', 'A. turn left\nB. turn right\n\nA. turn left', 'Both blanks require turn left.\n\nA. turn left', '<think>1. A\n5. B</think>\nA. turn left', 'Final Answer: A', 'Final Answer: A B\nFinal Answer: A']:
            self.assertNotEqual(m.parse_choice(raw, opts, True, q).status, 'ok', raw)
        raw = 'Checking the other options:\n- A – This is not the event.\n- F – Another alternative.\n\nB. turn right'
        self.assertEqual(m.parse_choice(raw, opts).value, ['B'])
        raw = '- A (self-tying) and F (rigid) are alternatives.\n- B (another case) is discussed.\n\nB. turn right'
        self.assertEqual(m.parse_choice(raw, opts).value, ['B'])

    def test_copied_option_list_with_context_is_not_a_conclusion(self):
        opts = {'A': 'turn left', 'B': 'turn right', 'F': 'go forward'}
        copied = 'The options are:\nA. turn left\nB. turn right\nF. go forward'
        self.assertEqual(m.parse_choice(copied + '\n\nB. turn right', opts).value, ['B'])
        self.assertEqual(m.parse_choice(copied + '\nThe result is B. turn right', opts).value, ['B'])
        self.assertNotEqual(m.parse_choice(copied, opts).status, 'ok')
        q = '1. [please fill in]\n5. [please fill in]'
        self.assertEqual(m.parse_choice(copied + '\n\n1. B\n5. A', opts, True, q).value, ['B', 'A'])

    def test_repeated_single_answer_is_consensus_not_multiselect(self):
        opts = {'A': 'the red object', 'F': 'the blue object'}
        self.assertEqual(m.parse_choice('the blue object\nF. the blue object', opts).value, ['F'])
        self.assertNotEqual(m.parse_choice('the red object\nF. the blue object', opts).status, 'ok')

    def test_new_formats_do_not_depend_on_reference_or_model_name(self):
        opts = {'A': 'turn left', 'B': 'turn right'}
        q = '1. [please fill in]\n5. [please fill in]'
        for gt in ['AA', 'AB', 'BA', 'BB']:
            c = m.score_case('A, turn left; B, turn right', gt, 'route_plan', q, opts)
            self.assertEqual(c.prediction_normalized, ['A', 'B'])
            self.assertEqual(c.parse_status, 'ok')

    def test_scalar_sentence_punctuation_does_not_hide_decimal_answer(self):
        for value in ['3000.0', '1.25', '1.2e3', '0']:
            for suffix in ['.', '!', ' cm.']:
                p = m.parse_number('Final Answer: ' + value + suffix)
                self.assertEqual(p.status, 'ok')
                self.assertEqual(m.decimal_number(p.value), m.decimal_number(value))
        self.assertNotEqual(m.parse_number('Final Answer: 1.25.\nFinal Answer: 2.5.').status, 'ok')
