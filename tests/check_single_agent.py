import math
import unittest
from dataclasses import replace
from cbp.single_agent.protocol import Assumptions, Candidate, InMemoryRegistry, Market, Output, Quote
from cbp.single_agent.experiment import fixture_config, run_experiment, run_suite, SyntheticExecutor


class SingleAgentChecks(unittest.TestCase):
    def make(self, f=2, beta=0., eta=0., drift=0., cap=100, threshold=0.5, k=None):
        configs = tuple(fixture_config(i, threshold) for i in range(6))
        assumptions = Assumptions(f, beta, eta, drift, "exact fixture grader", "frozen period", "constant fixture")
        candidates = tuple(Candidate(c, Quote("v1", i+1., "mock", "one fixed call"), assumptions) for i,c in enumerate(configs))
        registry = InMemoryRegistry(configs)
        executor = SyntheticExecutor({c.agent_id: 1. for c in configs}, 7, "expected_quality")
        return Market(candidates, tuple(c.agent_id for c in configs), registry, executor,
                      delta=.05, max_probes=cap, seed=7, committee_size=k)

    def test_derived_bound(self):
        m = self.make(beta=.05, eta=.05)
        self.assertAlmostEqual(m.summary("agent-0")["rho"], .1925)
        self.assertEqual(self.make().summary("agent-0")["rho"], 0)
        self.assertAlmostEqual(self.make(k=3).summary("agent-0")["rho"], .3)

    def test_outside_majority_rejected(self):
        with self.assertRaises(ValueError): self.make(f=3)

    def test_underdeclared_adversary_is_not_a_valid_experiment(self):
        with self.assertRaisesRegex(ValueError, "outside assumptions"):
            run_experiment(max_bad=2, actual_bad=3, max_probes=1)
        r = run_experiment(max_bad=2, actual_bad=1, max_probes=1)
        self.assertEqual(r["actual_bad_identities"], ["agent-0"])

    def test_invalid_reports_terminal(self):
        for score in (None, math.nan, math.inf, -0.1, 1.1, True):
            m = self.make()
            s = m.probe("agent-0", "unit-task-v1", lambda *args: score)
            self.assertEqual((s["status"], s["n"]), ("abstained", 0))
            with self.assertRaises(ValueError): m.probe("agent-0", "unit-task-v1", lambda *args: 1.)
            self.assertEqual(m.certified_list(), [])

    def test_abstention_preserves_prefix(self):
        m = self.make()
        m.probe("agent-0", "unit-task-v1", lambda *args: 1.)
        s = m.probe("agent-0", "unit-task-v1", lambda *args: None)
        self.assertEqual((s["status"], s["n"]), ("abstained", 1))
        self.assertEqual(len(m.events), 2)

    def test_replay_terminal(self):
        m = self.make()
        class Replay:
            def invoke(self, *args): return Output("same-id", 1., 1.)
        m.executor = Replay()
        m.probe("agent-0", "unit-task-v1", lambda *args: 1.)
        self.assertEqual(m.probe("agent-0", "unit-task-v1", lambda *args: 1.)["status"], "abstained")

    def test_all_certified_quotes_and_explicit_choice(self):
        r = run_experiment(max_probes=100, selected_agent="agent-5")
        self.assertGreaterEqual(len(r["certified_candidates"]), 2)
        self.assertEqual(r["formal_execution"]["agent_id"], "agent-5")
        self.assertEqual(r["metrics"]["payments"]["peer_review"], 0)
        self.assertEqual(r["metrics"]["false_certifications"], 0)
        with self.assertRaises(ValueError): run_experiment(max_probes=1, selected_agent="agent-0")

    def test_version_change_invalidates_listing(self):
        m = self.make()
        for _ in range(100):
            if m.summary("agent-0")["status"] != "pending": break
            m.probe("agent-0", "unit-task-v1", lambda *args: 1.)
        self.assertEqual(len(m.certified_list()), 1)
        c = m.candidates[0].config
        m.registry._configs[c.ref] = replace(c, model_version="changed")
        self.assertEqual(m.certified_list(), [])
        with self.assertRaises(ValueError): m.execute_selected("agent-0", "unit-task-v1")

    def test_trial_and_events_bind_versions(self):
        m = self.make()
        m.probe("agent-0", "unit-task-v1", lambda *args: 1.)
        e = m.events[0]
        self.assertEqual(e["config_ref"]["version"], "v1")
        self.assertEqual(e["identity_period"], "frozen period")
        self.assertEqual(len(e["committee"]), 5)
        self.assertNotIn("agent-0", e["committee"])
        e["reports"].clear()
        self.assertEqual(len(m.events[0]["reports"]), 5)

    def test_frozen_config_and_required_assumptions(self):
        with self.assertRaises(TypeError): Assumptions(2)
        with self.assertRaises(ValueError): replace(fixture_config(0), parameters=(("api_key", "secret"),))
        with self.assertRaises(ValueError): replace(fixture_config(0), parameters=[("temperature", "0")])
        with self.assertRaises(ValueError): Assumptions(2, -1, 0, 0, "x", "y", "z")

    def test_seeded_suite_and_low_cap(self):
        a = run_suite(seeds=(3,), max_probes=1)
        self.assertEqual(a, run_suite(seeds=(3,), max_probes=1))
        self.assertEqual(len(a["cases"]), 15)
        self.assertTrue(all(c["metrics"]["certified_count"] == 0 for c in a["cases"]))
        selective = run_experiment(scenario="selective", max_bad=2, max_probes=10)
        self.assertGreater(selective["metrics"]["abstentions"], 0)


if __name__ == "__main__": unittest.main()
