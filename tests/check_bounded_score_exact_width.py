"""Synthetic correctness contracts for the additive exact-width engine.

These deterministic fixtures are not live-agent study results or timing claims.
"""
from fractions import Fraction as Q
import random
import unittest

from cbp.bounded_score import IdentitySetBounds, ScoreRow
from cbp.bounded_score.exact_width import ExactWidthBounds, ExactWidthRefusal


class ExactWidthChecks(unittest.TestCase):
    def engines(self, roster, F, eta, rows=(), **kwargs):
        reference = IdentitySetBounds(roster, F, eta)
        width = ExactWidthBounds(roster, F, eta, **kwargs)
        reference.add_rows(rows)
        width.add_rows(rows)
        return reference, width

    def assert_parity(self, reference, width, cutoffs=(0, 5, 7, 9, 10)):
        self.assertEqual(width.model_inconsistent, reference.model_inconsistent)
        for producer in reference.roster:
            for cutoff in cutoffs:
                self.assertEqual(width.count_bounds(producer, cutoff),
                                 reference.count_bounds(producer, cutoff),
                                 (producer, cutoff, reference.rows))

    def test_empty_inputs_and_zero_budget(self):
        for m, F in ((2, 0), (6, 0), (6, 2), (10, 4)):
            ref, width = self.engines(tuple(map(str, range(m))), F, '1/2')
            self.assert_parity(ref, width)
            self.assertFalse(hasattr(width, 'candidates'))
            self.assertFalse(hasattr(width, 'intervals'))
            self.assertEqual(width.plan()['feasibility'].elimination_width, 0)
        ref, width = self.engines('ab', 0, 0,
                                 [ScoreRow('a', 't', 'a', {'b': 5})])
        self.assert_parity(ref, width)
        self.assertEqual(width.count_bounds('a', 5), (1, 1))

    def test_endpoint_clamping_and_exact_equalities(self):
        roster = tuple('abcdef')
        rows = [ScoreRow('a', 'low', 'low', {'b': 0, 'c': '1/2'}),
                ScoreRow('a', 'touch', 'touch', {'b': '9/2', 'c': '11/2'}),
                ScoreRow('a', 'high', 'high', {'b': '19/2', 'c': 10}),
                ScoreRow('d', 'empty', 'empty', {})]
        ref, width = self.engines(roster, 2, '1/2', rows)
        self.assert_parity(ref, width, (0, Q(1, 100), Q(1, 2), 1, 4,
                                       Q(499, 100), 5, Q(501, 100), 6, 9,
                                       Q(19, 2), Q(999, 100), 10))
        self.assertEqual(width.count_bounds('a', 0), (3, 3))
        self.assertEqual(width.count_bounds('d', 10), (0, 1))
        self.assertEqual(width.plan('a', 5)['feasibility'].constraint_factor_count, 0)

    def test_missing_invalid_empty_products_repeated_scopes(self):
        rows = [ScoreRow('a', 'bad', 'bad', {'b': 'NaN', 'c': 11,
                                           'd': True, 'e': 5.0, 'f': None}),
                ScoreRow('a', 'empty', 'empty', {}),
                ScoreRow('a', 'one', 'one', {'b': '7/1'}),
                ScoreRow('a', 'two', 'two', {'b': 7}),
                ScoreRow('a', 'three', 'three', {'b': 7})]
        ref, width = self.engines('abcdef', 2, 0, rows)
        self.assert_parity(ref, width)
        plan = width.plan('a', 7)['lower']
        self.assertEqual(plan.objective_factor_count, 1)
        self.assertEqual(plan.objective_multiplicity, 3)
        self.assertEqual(plan.factor_scopes, (('b',),))
        self.assertEqual(width.count_bounds('a', 7), (0, 5))

    def test_global_infeasibility_applies_at_zero_and_absent_producer(self):
        rows = [ScoreRow('a', 't', 't', {'b': 0, 'c': 5, 'd': 10})]
        ref, width = self.engines('abcd', 1, 0, rows)
        self.assert_parity(ref, width)
        self.assertTrue(width.model_inconsistent)
        self.assertIsNone(width.count_bounds('d', 0))
        ref, width = self.engines('abcd', 0, 0, rows)
        self.assert_parity(ref, width)
        self.assertEqual(width.diagnostics['feasibility'].status, 'infeasible')

    def test_disconnected_cardinality_and_isolated_producer(self):
        rows = [ScoreRow('f', 'one', 'one', {'b': 0, 'c': 10}),
                ScoreRow('f', 'two', 'two', {'d': 0, 'e': 10}),
                ScoreRow('a', 'query', 'query', {'b': 5, 'c': 5})]
        ref, width = self.engines('abcdef', 2, 0, rows)
        self.assert_parity(ref, width)
        self.assertEqual(width.count_bounds('a', 5), (1, 1))
        self.assertIn('a', width.diagnostics['lower'].elimination_order)
        self.assertEqual(width.diagnostics['lower'].elimination_width, 1)

    def test_strict_joint_gain_and_touching_variant(self):
        roster = tuple(map(str, range(1, 11)))
        rows = []
        for t in range(2):
            for producer in roster:
                reports = {j: '5' for j in roster if j != producer}
                if producer == '1':
                    reports.update({str(j): ('4.4' if (j <= 6) == (t == 0)
                                             else '5.6') for j in range(3, 11)})
                rows.append(ScoreRow(producer, str(t), f'{producer}-{t}', reports))
        ref, width = self.engines(roster, 4, '0.5', rows)
        self.assert_parity(ref, width)
        self.assertEqual(width.count_bounds('1', 5), (1, 1))
        self.assertEqual(ref.taskwise_count_bounds('1', 5), (0, 2))
        ref, width = self.engines(roster, 4, '0.6', rows)
        self.assert_parity(ref, width)
        self.assertEqual(width.count_bounds('1', 5), (1, 2))

    def test_cross_producer_update_invalidates_old_bounds(self):
        roster = tuple(map(str, range(1, 11)))
        first = ScoreRow('1', 't', 'a', {'2': 5, **{
            str(j): '4.4' if j <= 6 else '5.6' for j in range(3, 11)}})
        ref, width = self.engines(roster, 4, '0.5', [first])
        self.assertEqual(width.count_bounds('1', 5), (0, 1))
        second = ScoreRow('2', 't', 'b', {
            j: 4 if int(j) <= 6 else 6 for j in roster if j != '2'})
        ref.add_rows([second])
        width.add_rows([second])
        self.assertEqual(dict(width.diagnostics), {})
        self.assert_parity(ref, width)
        self.assertEqual(width.count_bounds('1', 5), (0, 0))

    def test_large_monomials_simplified_before_graph_construction(self):
        roster = tuple(map(str, range(40)))
        rows = [ScoreRow('0', 'high', 'high', {j: 9 for j in roster[1:]}),
                ScoreRow('0', 'low', 'low', {j: 1 for j in roster[1:]})]
        width = ExactWidthBounds(roster, 2, 0, max_width=0)
        width.add_rows(rows)
        self.assertEqual(width.count_bounds('0', 5), (1, 1))
        for name in ('lower', 'upper'):
            plan = width.diagnostics[name]
            self.assertEqual(plan.factor_scopes, ())
            self.assertEqual(plan.elimination_width, 0)
            self.assertEqual(plan.simplified_monomials, 1)
        self.assertLess(width.diagnostics['lower'].table_states_created, 200)

    def test_guards_refuse_not_approximate_or_inconsistent(self):
        rows = [ScoreRow('a', 't', 't', {'b': 5, 'c': 5})]
        width = ExactWidthBounds('abcdef', 2, 0, max_width=0)
        width.add_rows(rows)
        self.assertFalse(width.model_inconsistent)
        with self.assertRaises(ExactWidthRefusal) as raised:
            width.count_bounds('a', 5)
        self.assertEqual(raised.exception.diagnostics.objective, 'lower')
        self.assertEqual(raised.exception.diagnostics.status, 'refused')
        self.assertFalse(width.model_inconsistent)
        self.assertEqual(width.plan('a', 5)['lower'].elimination_width, 1)
        width = ExactWidthBounds('abcd', 1, 0, max_states=0)
        with self.assertRaises(ExactWidthRefusal):
            _ = width.model_inconsistent
        self.assertEqual(width.diagnostics['feasibility'].status, 'refused')

    def test_explicit_order_and_actual_state_counters(self):
        rows = [ScoreRow('a', 't', 't', {'b': 0, 'c': 5, 'd': 10})]
        roster = tuple('abcdef')
        ref, width = self.engines(roster, 2, 0, rows,
                                 elimination_order=tuple(reversed(roster)))
        self.assert_parity(ref, width)
        for diagnostic in width.diagnostics.values():
            self.assertEqual(diagnostic.elimination_order, tuple(reversed(roster)))
            self.assertLessEqual(diagnostic.feasible_states_created,
                                 diagnostic.table_states_created)
            self.assertLessEqual(diagnostic.table_states_created,
                                 diagnostic.estimated_total_states)
        with self.assertRaises(ValueError):
            ExactWidthBounds(roster, 2, 0, elimination_order='abcdea')

    def test_validation_matches_reference_and_addition_is_atomic(self):
        bad_constructor = [('aabc', 1, 0), ('abc', 1, 0), ('abcd', True, 0),
                           ('abcd', -1, 0), ('abcd', 1, 0.5),
                           ('abcd', 1, 'NaN'), ('abcd', 1, 11)]
        for args in bad_constructor:
            for cls in (IdentitySetBounds, ExactWidthBounds):
                with self.assertRaises(ValueError):
                    cls(*args)
        for option in ({'max_width': True}, {'max_states': -1}, {'max_states': 1.0}):
            with self.assertRaises(ValueError):
                ExactWidthBounds('abcd', 1, 0, **option)
        report = {'b': 5}
        ref, width = self.engines('abcd', 1, 0,
                                 [ScoreRow('a', 'first', 'first', report)])
        report['b'] = 0
        self.assertEqual(width.rows[0].reports['b'], 5)
        with self.assertRaises(TypeError):
            width.rows[0].reports['b'] = Q(0)
        invalid_rows = [None, ScoreRow('z', 't', 't', {}),
                        ScoreRow('a', '', 't', {}),
                        ScoreRow('a', 't', 't', {}, 1),
                        ScoreRow('a', 't', 't', [('b', 5)]),
                        ScoreRow('a', 't', 't', {'a': 5}),
                        ScoreRow('a', 't', 't', {'z': 5}),
                        ScoreRow('a', 'first', 'new', {}),
                        ScoreRow('b', 'new', 'first', {})]
        for invalid in invalid_rows:
            for engine in (ref, width):
                before = list(engine.rows)
                with self.assertRaises(ValueError):
                    engine.add_rows([ScoreRow('c', 'fresh', 'fresh', {'a': 0, 'b': 10}),
                                     invalid])
                self.assertEqual(engine.rows, before)
        self.assert_parity(ref, width)
        for engine in (ref, width):
            for producer, cutoff in (('z', 5), ('a', 5.0), ('a', -1), ('a', 11)):
                with self.assertRaises(ValueError):
                    engine.count_bounds(producer, cutoff)
            with self.assertRaises(ValueError):
                engine.add_rows([ScoreRow('a', 'x', 'x', {}),
                                 ScoreRow('a', 'x', 'y', {})])
            with self.assertRaises(ValueError):
                engine.add_rows([ScoreRow('a', 'x', 'same', {}),
                                 ScoreRow('b', 'y', 'same', {})])

    def test_deterministic_random_exact_parity_m2_through_m10(self):
        rng = random.Random(887301)
        for m in range(2, 11):
            roster = tuple(map(str, range(m)))
            for F in range((m - 2) // 2 + 1):
                for mode in range(4):
                    eta = rng.choice((Q(0), Q(1, 2), Q(3, 5), Q(5), Q(10)))
                    rows, boundaries = [], {Q(0), Q(5), Q(7), Q(9), Q(10)}
                    excluded = set(rng.sample(roster, F))
                    for task in range(rng.randrange(1, 7)):
                        producer = rng.choice(roster)
                        reports = {}
                        center = rng.choice((Q(0), Q(5), Q(9), Q(10)))
                        for reviewer in roster:
                            if reviewer == producer or rng.random() < 0.3:
                                continue
                            if mode == 0:
                                value = rng.choice((None, 'bad', True, 5.0, 11,
                                                    '4.4', '4.5', 5, '5.5', '5.6'))
                            elif mode == 1:
                                value = center
                            elif mode == 2:
                                value = rng.randrange(11) if reviewer in excluded else center
                            else:
                                value = Q(rng.randrange(21), 2)
                            reports[reviewer] = value
                            if type(value) is int or isinstance(value, Q):
                                boundaries.update(c for c in (value - eta, value + eta)
                                                  if 0 <= c <= 10)
                        rows.append(ScoreRow(producer, str(task), f'{m}-{F}-{mode}-{task}',
                                             reports, bool(task % 2)))
                    ref, width = self.engines(roster, F, eta, rows,
                                             elimination_order=(roster[::-1] if mode == 3 else None))
                    self.assert_parity(ref, width, sorted(boundaries))
                    for diagnostic in width.diagnostics.values():
                        self.assertLessEqual(diagnostic.table_states_created,
                                             diagnostic.estimated_total_states)


if __name__ == '__main__':
    unittest.main()
