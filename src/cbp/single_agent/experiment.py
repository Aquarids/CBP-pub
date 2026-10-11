"""Reproducible controlled fixtures, not real API/blockchain/calibration evidence."""
from dataclasses import asdict
import random
from .protocol import Assumptions, Candidate, ConfigRef, InMemoryRegistry, Market, Output, Quote, ServiceConfig

SCENARIOS = ("honest", "badmouth", "inflate", "selective", "onoff")


def fixture_config(i, threshold=0.5, target="expected_quality"):
    return ServiceConfig(ConfigRef("mock-chain", f"mock://agent-{i}", "v1"), f"agent-{i}",
                         "mock://uniform-service", "api-v1", f"fixture-model-{i}-v1",
                         "fixed synthetic system", "fixed synthetic task", (), "unit-task-v1",
                         "bounded-score-v1", "synthetic-unit-domain", "iid-v1", "reset-every-call",
                         1, "fixed seeded distribution", "no retries", "stop on failure",
                         "exact synthetic quality; no learned evaluator", target, threshold,
                         (("temperature", "fixed"),))


class SyntheticExecutor:
    """Truth exists only in this experimental fixture, never in Market."""
    def __init__(self, qualities, seed, target):
        self.qualities, self.target = dict(qualities), target
        self.rng = random.Random(seed)
        self.calls = 0

    def invoke(self, config, task):
        if task != "unit-task-v1":
            raise ValueError("Outside frozen synthetic task domain")
        self.calls += 1
        theta = self.qualities[config.agent_id]
        quality = float(self.rng.random() < theta) if self.target == "success_probability" else theta
        return Output(f"synthetic-output-{self.calls}", quality, 1.)


def run_experiment(*, seed=7, scenario="honest", max_bad=0, max_probes=256,
                   beta=0., eta=0., drift=0., threshold=0.5, delta=0.05,
                   target="expected_quality", selected_agent=None, actual_bad=None):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown score-report scenario")
    # Explicitly fixed six-agent cohort; F=3 is outside the theorem scope.
    configs = tuple(fixture_config(i, threshold, target) for i in range(6))
    roster = tuple(c.agent_id for c in configs)
    qualities = dict(zip(roster, (0.15, 0.35, 0.6, 0.8, 0.95, 1.0)))
    assumptions = Assumptions(max_bad, beta, eta, drift,
                              "synthetic exact grader; user-supplied conservative bounds",
                              f"synthetic-cohort-seed-{seed}",
                              "fixture constant execution policy and same task distribution")
    candidates = tuple(Candidate(c, Quote("quote-v1", float(6-i), "mock_credit",
                                        "one invocation; no retry; fixed total"), assumptions)
                       for i, c in enumerate(configs))
    executor = SyntheticExecutor(qualities, seed, target)
    registry = InMemoryRegistry(configs)
    market = Market(candidates, roster, registry, executor, delta=delta,
                    max_probes=max_probes, seed=seed)
    actual_bad = max_bad if actual_bad is None else actual_bad
    if isinstance(actual_bad, bool) or not isinstance(actual_bad, int) or not 0 <= actual_bad <= max_bad:
        raise ValueError("Actual bad identities exceed declared F or are invalid; outside assumptions")
    bad = set(roster[:actual_bad])
    review_attempts = 0

    def review(identity, config, output, history):
        nonlocal review_attempts
        review_attempts += 1
        truth = output.value
        if identity not in bad or scenario == "honest":
            return truth
        if scenario == "badmouth":
            return 0.
        if scenario == "inflate":
            return 1.
        if scenario == "selective":
            # Outcome-dependent refusal must end the candidate, never be retried.
            return None if truth >= threshold else truth
        return 1. if len(history) % 2 else 0.

    anytime_covered = True
    while True:
        unresolved = [r for r in roster if market.summary(r)["status"] == "pending"]
        if not unresolved:
            break
        # Predictable least-sampled scheduling; no future outcome access.
        agent = min(unresolved, key=lambda r: (market.summary(r)["n"], r))
        current = market.probe(agent, "unit-task-v1", review)
        anytime_covered = anytime_covered and current["lower"] <= qualities[agent] <= current["upper"]
    certified = market.certified_list()
    summaries = [market.summary(r) for r in roster]
    covered = [s["lower"] <= qualities[s["agent_id"]] <= s["upper"] for s in summaries]
    qualified = [r for r in roster if qualities[r] >= threshold]
    formal = None
    if selected_agent is not None:
        output = market.execute_selected(selected_agent, "unit-task-v1")
        formal = {"selected_by": "explicit_caller", "agent_id": selected_agent,
                  "artifact_id": output.artifact_id, "quality_experiment_only": output.value,
                  "resource_units": output.resource_units}
    return {
        "schema_version": "single-agent-experiment-v1", "seed": seed, "scenario": scenario,
        "registry": registry.kind, "execution": "synthetic_constant_policy_no_live_API",
        "participants": 6, "max_bad_eligible": max_bad, "actual_bad_identities": sorted(bad),
        "assumptions": asdict(assumptions), "delta": delta, "threshold": threshold,
        "target": target, "max_probes": max_probes,
        "certified_candidates": certified, "summaries": summaries,
        "metrics": {
            "false_certifications": sum(qualities[c["agent_id"]] < threshold for c in certified),
            "simultaneous_interval_coverage": anytime_covered,
            "terminal_interval_coverage": all(covered),
            "candidate_interval_coverage": sum(covered)/len(covered),
            "qualified_candidate_coverage": sum(c["agent_id"] in qualified for c in certified)/len(qualified) if qualified else None,
            "certified_count": len(certified), "rounds_attempted": len(market.events),
            "rounds_completed": sum(s["n"] for s in summaries),
            "rounds_by_candidate": {s["agent_id"]: s["n"] for s in summaries},
            "inconclusive": sum(s["status"] == "inconclusive" for s in summaries),
            "abstentions": sum(s["status"] == "abstained" for s in summaries),
            "resources": {"probe_units": sum(e["resource_units"] for e in market.events),
                          "review_attempts": review_attempts, "review_units": review_attempts*0.1,
                          "formal_units": formal["resource_units"] if formal else 0,
                          "units_source": "modeled_fixture_weights", "real_cost": None},
            "payments": {"probe": 0, "peer_review": 0, "formal": 0,
                         "note": "No payment system; quotes are offers, not transfers"}},
        "ground_truth_experiment_only": qualities,
        "formal_execution": formal, "events": market.events,
        "limitations": ["Conditional assumptions are not empirically validated",
                        "No actual blockchain adapter, live API or payment execution",
                        "Seeded randomness is not adversary-resistant deployment randomness",
                        "Coverage in a small run is not a proof or calibrated empirical rate"]}


def run_suite(*, seeds=(7,), max_probes=256, **kwargs):
    results = []
    for seed in seeds:
        for scenario in SCENARIOS:
            for max_bad in (0, 1, 2):
                result = run_experiment(seed=seed, scenario=scenario, max_bad=max_bad,
                                        max_probes=max_probes, **kwargs)
                # Keep suite summaries compact; single experiment retains raw trial records.
                results.append({k: result[k] for k in ("seed", "scenario", "max_bad_eligible", "metrics")})
    return {"schema_version": "single-agent-suite-v1", "kind": "synthetic_fixture",
            "seeds": list(seeds), "max_probes": max_probes, "parameters": kwargs,
            "cases": results, "out_of_scope": "n=6,F=3 rejected: n-1 <= 2F"}
