"""Benchmark-informed synthetic potential outcomes, isolated from assessment.

No actual model execution. ABCD labels and independent heldout outcomes are
simulated; binary benchmark labels cannot establish a real ABCD profile.
"""
import hashlib
import json
import random
from pathlib import Path
from .protocol import GRADES, LEVELS, METHODS, Vote, MultigradeMarket, meets

ATTACKS = ("selection_badmouth", "boost_others", "random_grades", "adjacent_threshold_flip", "history_adaptive", "withhold")


def read_calibration(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    profiles = value["profiles"]
    if not profiles:
        raise ValueError("Calibration needs profiles")
    for p in profiles:
        probs = p["cumulative_probabilities"]
        if set(probs) != set(LEVELS) or not 0 <= probs["A"] <= probs["B"] <= probs["C"] <= 1:
            raise ValueError("Nested synthetic A/B/C probabilities required")
    return value, hashlib.sha256(raw).hexdigest()


def make_world(*, calibration, m=20, N=512, heldout_size=2048, seed=7, f=6):
    if type(f) is not int or not 0 <= f <= m or min(N,heldout_size) < 1:
        raise ValueError("Invalid world dimensions")
    rng = random.Random(seed)
    roster = [f"agent-{i:02d}" for i in range(m)]
    faulty = set(random.Random(seed+47291).sample(roster,f))
    tasks = [f"probe-{i:05d}" for i in range(N)]
    heldout = [f"heldout-{i:05d}" for i in range(heldout_size)]
    truth, holdtruth, candidates, profiles, costs = {}, {}, [], {}, {}
    for i,a in enumerate(roster):
        p = calibration["profiles"][i % len(calibration["profiles"])]
        probs = p["cumulative_probabilities"]
        profiles[a] = {"profile_id":p["profile_id"],"synthetic_cumulative_probabilities":probs}
        # Finite quota world makes below/on/above-threshold cases reproducible.
        cumulative = [round(N*probs[l]) for l in LEVELS]+[N]
        counts = [cumulative[0]]+[cumulative[j]-cumulative[j-1] for j in range(1,4)]
        labels = [g for g,c in zip(GRADES,counts) for _ in range(c)]
        random.Random(seed+719*i+11).shuffle(labels)
        truth[a] = dict(zip(tasks,labels))
        # Independent same-category heldout draws; no quality-switching strategy.
        hrng = random.Random(seed+991*i+900001)
        holdtruth[a] = {}
        for task in heldout:
            u = hrng.random()
            holdtruth[a][task] = next((l for l in LEVELS if u < probs[l]),"D")
        cost = round(rng.uniform(.01,.10),6)
        quote = round(cost+rng.uniform(.01,.12),6)
        costs[a] = cost
        candidates.append({"agent_id":a,"quote":quote})
    return {"task_category":calibration["task_category"],"tasks":tasks,"heldout_tasks":heldout,"candidates":candidates,"faulty":sorted(faulty),"truth":truth,"heldout_truth":holdtruth,"profiles":profiles,"private_costs":costs,"seed":seed}


def reported_grade(*, true_grade, attack, reviewer, producer, round_number, public_history, rng):
    """Only current output and public past; no future labels/order or coalitions."""
    if attack == "selection_badmouth":
        return "D"
    if attack == "boost_others":
        return "A"  # Boosts every other producer, never a known accomplice set.
    if attack == "random_grades":
        return rng.choice(GRADES)
    if attack == "adjacent_threshold_flip":
        return {"A":"B","B":"C","C":"B","D":"C"}[true_grade]
    if attack == "history_adaptive":
        history = public_history.get(producer,{})
        return "D" if any(s.get("lower",0) >= .5 for s in history.values()) else "A"
    if attack == "withhold":
        return None
    raise ValueError("Unknown noncollusive attack")


def run_world(world, *, F=9, rho=.50, delta=.05, looks=(16,32,64,128,256,512), method="sequential_ci", attack="selection_badmouth", raw=True):
    market = MultigradeMarket(candidates=world["candidates"],task_ids=world["tasks"],task_category=world["task_category"],F=F,rho=rho,delta=delta,looks=looks,seed=world["seed"]+700000,method=method)
    rngs = {a:random.Random(world["seed"]+100003*i+390) for i,a in enumerate(market.roster)}
    fault_set = set(world["faulty"])
    snapshots = []
    violation = None
    while not market.stopped:
        outputs = {}
        for a,task in market.next_round().items():
            artifact = f"output:{market.round+1}:{a}:{task}"
            grade = world["truth"][a][task]
            votes = []
            for reviewer in market.roster:
                if reviewer == a:
                    continue
                report = grade if reviewer not in fault_set else reported_grade(true_grade=grade,attack=attack,reviewer=reviewer,producer=a,round_number=market.round+1,public_history=market.summaries,rng=rngs[reviewer])
                if report is not None:
                    votes.append(Vote(reviewer,a,task,artifact,market.round+1,market.rubric,report))
            outputs[a] = {"task_id":task,"artifact_id":artifact,"votes":votes}
        try:
            market.record_round(outputs)
        except ValueError as e:
            if "Conflicting certificates" not in str(e):
                raise
            violation = str(e)
            break
        if market.round in market.looks and (method not in ("fixed_ci", "median_point_fixed") or market.round == market.looks[-1]):
            snapshots.append(json.loads(json.dumps(market.summaries)))
    realized = {a:{l:sum(meets(g,l) for g in world["truth"][a].values())/market.N for l in LEVELS} for a in market.roster}
    heldout = {a:{l:sum(meets(g,l) for g in world["heldout_truth"][a].values())/len(world["heldout_tasks"]) for l in LEVELS} for a in market.roster}
    false_cert = [(a,l) for a in market.roster for l in LEVELS if market.summaries[a][l]["status"] == "certified" and realized[a][l] < rho]
    false_exclusion = [(a,l) for a in market.roster for l in LEVELS if market.summaries[a][l]["status"] == "excluded" and realized[a][l] >= rho]
    any_false_cert = sorted({(a,l) for snap in snapshots for a in market.roster for l in LEVELS if snap[a][l]["status"] == "certified" and realized[a][l] < rho})
    any_false_exclusion = sorted({(a,l) for snap in snapshots for a in market.roster for l in LEVELS if snap[a][l]["status"] == "excluded" and realized[a][l] >= rho})
    correct = sum(s["status"] == ("certified" if realized[a][l]>=rho else "excluded") for a,levels in market.summaries.items() for l,s in levels.items())
    covered = all(s[a][l]["lower"] <= realized[a][l] <= s[a][l]["upper"] for s in snapshots for a in market.roster for l in LEVELS)
    exact_grades = {}
    for a, levels in market.summaries.items():
        predicted = next((l for i,l in enumerate(LEVELS) if levels[l]["status"] == "certified" and all(levels[h]["status"] == "excluded" for h in LEVELS[:i])), None)
        if all(s["status"] == "excluded" for s in levels.values()):
            predicted = "D"
        truth_grade = next((l for l in LEVELS if realized[a][l] >= rho),"D")
        exact_grades[a] = {"predicted":predicted,"true":truth_grade}
    certified = market.candidate_list()
    # Economics are descriptive modeled costs; quotes/costs independent of ability.
    outputs_n = market.round*len(market.roster)
    metrics = {"any_look_false_certifications":any_false_cert,"any_look_false_exclusions":any_false_exclusion,"threshold_decision_accuracy":correct/(3*len(market.roster)),"exact_classification_accuracy":sum(g["predicted"] == g["true"] for g in exact_grades.values())/len(market.roster),"all_thresholds_decided_fraction":sum(all(s["status"] != "unresolved" for s in levels.values()) for levels in market.summaries.values())/len(market.roster),"pending_output_count":sum(e["certificate"] is None for e in market.events),"false_certifications":false_cert,"false_exclusions":false_exclusion,"simultaneous_coverage":covered if snapshots and method in ("sequential_ci","fixed_ci") else None,"unresolved_levels":sum(s["status"] == "unresolved" for d in market.summaries.values() for s in d.values()),"rounds":market.round,"probe_executions":outputs_n,"review_requests":outputs_n*(len(market.roster)-1),"reviews_received":sum(sum(len(group) for group in e["vote_groups"].values()) for e in market.events),"certified_agents":len(certified),"modeled_probe_cost":market.round*sum(world["private_costs"].values()),"modeled_review_cost":sum(sum(len(group) for group in e["vote_groups"].values()) for e in market.events)*.0001,"peer_money_transfers":0,"formal_executions":0,"assumption_violation":violation}
    return {"schema":"cbp-multigrade-v1","config":{"m":len(market.roster),"F":F,"actual_f":len(fault_set),"market_bad_fraction":len(fault_set)/len(market.roster),"honest_target_peer_bad_fraction":len(fault_set)/(len(market.roster)-1),"guarantee_regime":len(fault_set)<=F,"guarantee_applicable":len(fault_set)<=F and method in ("sequential_ci","fixed_ci"),"N":market.N,"looks":list(looks),"rho":rho,"delta":delta,"tail_alpha":market.alpha,"method":method,"attack":attack,"seed":world["seed"],"category":world["task_category"],"stopping":"whole-market all levels decided or horizon","authentication":"trusted simulator identity binding, not cryptographic implementation","guarantee_applies_to_method":method in ("sequential_ci","fixed_ci")},"metrics":metrics,"exact_grades":exact_grades,"summaries":market.summaries,"candidates":certified,"realized_pool_rates":realized,"heldout_rates":heldout,"events":market.events if raw else [],"simulator_provenance":{"profiles":world["profiles"],"faulty_identities":world["faulty"],"private_costs":world["private_costs"],"probe_truth":world["truth"] if raw else None,"heldout_truth":world["heldout_truth"] if raw else None,"probe_task_ids":world["tasks"],"heldout_task_ids":world["heldout_tasks"],"calibration_status":"synthetic ABCD, no real per-item multigrade labels","actual_currency":False}}
