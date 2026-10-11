#!/usr/bin/env python3
"""Run offline CBP synthetic fixtures: python exp/runner/run_cbp.py.

Uses only the standard library, small in-memory task pools, and public quotes
in arbitrary fixture units. Authentication, a fixed honest identity set, bounded
honest-score error, and frozen potential outputs remain caller assumptions.
This smoke verifies code paths, not service calibration or scientific results.
"""
import json
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cbp.bounded_score import BoundedScoreMarket, IdentitySetBounds, ScoreRow
from cbp.bounded_score.exact_width import ExactWidthBounds
from cbp.market_comparison.buyer_policy import strict_offers


ROSTER = ("fixture-a", "fixture-b", "fixture-c", "fixture-d")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def market():
    return BoundedScoreMarket(
        candidates=[{"agent_id":a, "quote":i+1} for i,a in enumerate(ROSTER)],
        task_ids=("fixture-task-1", "fixture-task-2"),
        task_category="synthetic-fixture", F=1, eta="0.25",
        rho="0.5", epsilon="0.1", delta="0.05", looks=(2,), seed=7,
        aggregation="joint_counts",
    )


def run_smoke():
    relative_out = Path("runs") / "smoke"
    out = ROOT / relative_out
    out.mkdir(parents=True, exist_ok=True)
    # Use the unchanged public CLI and its original synthetic provider/reviewer
    # fixture, including certified user selection and formal execution.
    command = [sys.executable, "-m", "cbp", "smoke", "--seed", "7",
               "--max-probes", "64", "--select", "agent-5",
               "--output", str(relative_out / "single-agent.json")]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                       PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=60)
    (out / "normal-cli.log").write_text(completed.stdout + completed.stderr)
    require(completed.returncode == 0, "Original public CLI smoke failed")
    service = json.loads((out / "single-agent.json").read_text())
    require(service["formal_execution"]["agent_id"] == "agent-5",
            "Original certified service execution was not reached")
    require(service["metrics"]["rounds_completed"] > 0,
            "Original service probes were not executed")
    require(service["metrics"]["resources"]["review_attempts"] > 0,
            "Original peer reviewer callback was not invoked")

    protocol = market()
    scores = dict(zip(ROSTER, (10,8,6,2)))
    try:
        protocol.record_round({})
    except ValueError:
        pass
    else:
        raise RuntimeError("Incomplete rounds must be rejected")
    require(protocol.round == 0 and not protocol.bounds.rows,
            "Rejected round must not mutate the protocol")
    while not protocol.stopped:
        tasks = protocol.next_round()
        protocol.record_round({a:{
            "task_id":t, "artifact_id":f"{a}:{t}",
            "reports":{j:scores[a] for j in ROSTER if j!=a},
            "deadline_passed":True,
        } for a,t in tasks.items()})
    publications = protocol.publications()
    require(protocol.round == protocol.N, "Fixture must reach the census")
    require([publications[a]["certified_grade"] for a in ROSTER] == list("ABCD"),
            "Unexpected census grades")
    offers = protocol.candidate_list()
    require([c["agent_id"] for c in offers] == list(ROSTER[:3]),
            "All and only strictly qualified candidates must be listed")
    require([c["quote"] for c in offers] == [1,2,3], "Quotes must be preserved")
    ranked = strict_offers(publications, level="C", rho="0.5", base_price=1,
                           kappas={a:i+1 for i,a in enumerate(ROSTER)})
    require([c["agent_id"] for c in ranked] == list(ROSTER[:3]),
            "Original demand qualification and price ranking differ")
    require([c["rank_score"] for c in ranked] == [Fraction(1),Fraction(1,2),Fraction(1,3)],
            "Original capability-to-price ranking was not applied")

    exact = ExactWidthBounds(ROSTER,1,"0.25")
    exact.add_rows(protocol.bounds.rows)
    for a in ROSTER:
        for cutoff in (5,7,9):
            require(exact.count_bounds(a,cutoff) == protocol.bounds.count_bounds(a,cutoff),
                    "Exact-width solver differs from identity-set enumeration")

    missing = market()
    while not missing.stopped:
        missing.record_round({a:{
            "task_id":t, "artifact_id":f"missing:{a}:{t}",
            "reports":{}, "deadline_passed":True,
        } for a,t in missing.next_round().items()})
    require(not missing.candidate_list(), "Missing reports must not certify")
    require(all(p["status"] == "budget_ambiguity" for p in missing.publications().values()),
            "Missing evidence must remain ambiguous")
    require(missing.bounds.count_bounds(ROSTER[0],5) == (0,2),
            "Missing sampled tasks must remain in the denominator")

    inconsistent = IdentitySetBounds(ROSTER,1,0)
    inconsistent.add_rows([ScoreRow(ROSTER[0],"fixture-conflict","fixture-conflict-artifact",
                                      dict(zip(ROSTER[1:],(0,5,10))))])
    require(inconsistent.model_inconsistent and inconsistent.count_bounds(ROSTER[0],5) is None,
            "Inconsistent identity models must fail closed")

    return {
        "fixture":True, "status":"PASS", "certified_candidates":offers,
        "demand_ranked_offers":ranked,
        "original_cli":{
            "command":["python", *command[1:]],
            "certified_agents":[c["agent_id"] for c in service["certified_candidates"]],
            "formal_execution":service["formal_execution"],
            "probe_trials":service["metrics"]["rounds_completed"],
            "review_attempts":service["metrics"]["resources"]["review_attempts"],
        },
        "publications":publications, "modeled_resources":protocol.costs,
        "checks":["original_public_CLI", "original_provider_probe_and_peer_review",
                  "original_certified_formal_execution", "original_demand_price_ranking",
                  "atomic_rounds", "census_ABCD", "all_certified_quotes",
                  "exact_width_matches_enumeration", "missing_reports_ambiguous",
                  "inconsistent_model_fails_closed"],
        "real_usage":None, "real_cost":None, "peer_review_money_transfers":0,
    }


def encode(value):
    if isinstance(value,Fraction):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


if __name__ == "__main__":
    payload = json.dumps(run_smoke(),default=encode,indent=2,allow_nan=False)
    (ROOT / "runs" / "smoke" / "smoke.json").write_text(payload + "\n")
    print(payload)
