"""Deterministic recorded-artifact experiments, deliberately not live certificates."""
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from statistics import median

from ..single_agent.experiment import fixture_config
from ..admission.committees import bias_bound
from ..single_agent.protocol import Assumptions, Candidate, ConfigRef, InMemoryRegistry, Market, Output, Quote
from .schema import digest, load_dataset, split_tasks, write_jsonl
from .finite_pool import interval as finite_interval
from .grading import build_judge_request, FileGradeProvider, SyntheticNoisyGradeProvider, AttackGradeProvider, load_evidence_gaps

PLAN_VERSION = "replay-v5-evidence-pending"
LEGACY_PLAN_VERSION = "replay-v4-mutual-producer-roster"
SUPPORTED_PLAN_VERSIONS = {PLAN_VERSION, LEGACY_PLAN_VERSION}

SCENARIOS = ("honest", "badmouth", "inflate", "onoff", "selective")


def make_plan(dataset, *, seeds=(7,), scenarios=("honest",), reviewers=0, max_bad=0,
              max_probes=256, threshold=.5, delta=.05, beta=1., eta=1., drift=0.,
              quotes=None, provider="file", calibration_fraction=.2, heldout_fraction=.2,
              pointwise_beta=None, pointwise_basis=None, honest_noise=.15,
              shared_difficulty=.2, reviewer_bias_scale=.1, honest_abstain_probability=0., grading_label="recorded_unverified", rubric_id="ABCD-v2-evidence"):
    if rubric_id not in {"ABCD-v1", "ABCD-v2-evidence"}:
        raise ValueError("Unknown grading contract")
    if rubric_id == "ABCD-v2-evidence" and honest_abstain_probability:
        raise ValueError("ABCD-only contract does not simulate an abstention grade")
    if not isinstance(grading_label, str) or not grading_label.strip():
        raise ValueError("A nonempty grading provenance label is required")
    for name, value in (("max_probes", max_probes),):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(reviewers) is not int or reviewers != 0:
        raise ValueError("Mutual producer market permits no dedicated reviewers; use reviewers=0")
    if type(max_bad) is not int or max_bad < 0:
        raise ValueError("Nonnegative integer F required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Unique integer seeds required")
    if not scenarios or any(s not in SCENARIOS for s in scenarios) or len(set(scenarios)) != len(scenarios):
        raise ValueError("Unknown or duplicate attack scenario")
    if provider not in {"file", "synthetic"}:
        raise ValueError("Provider must be file or explicitly synthetic")
    synthetic_config = {"noise": honest_noise, "shared_difficulty": shared_difficulty, "reviewer_bias_scale": reviewer_bias_scale, "abstain_probability": honest_abstain_probability}
    SyntheticNoisyGradeProvider(**synthetic_config)  # Validate generator parameters; no calls or truth.
    assumptions = Assumptions(max_bad, beta, eta, drift, "external sensitivity assumptions; not estimated guarantees", "recorded-frozen-roster", "static replay; no deployed stationarity claim")
    if not 0 < delta < 1 or isinstance(delta, bool) or not 0 <= threshold <= 1 or isinstance(threshold, bool):
        raise ValueError("Invalid delta/threshold")
    if pointwise_beta is not None:
        finite_interval([], pool_size=1, candidates=1, delta=delta, pointwise_beta=pointwise_beta)
        if not isinstance(pointwise_basis, str) or not pointwise_basis.strip():
            raise ValueError("Finite-pool mode requires an explicit external pointwise calibration basis")
        if len({t["task_group"] for t in dataset["tasks"].values()}) != len(dataset["tasks"]):
            raise ValueError("Finite-pool mode requires one normalized task per task group")
    agents = sorted(dataset["agents"])
    roster = list(agents)
    if len(set(roster)) != len(roster) or len(roster)-1 <= 2*max_bad:
        raise ValueError("Unique roster and strict honest eligible majority required")
    quotes = {aid: {"amount": 0., "currency": "replay_units", "scope": "configured replay quote; no payment"} for aid in agents} if quotes is None else quotes
    if not isinstance(quotes, dict) or set(quotes) != set(agents):
        raise ValueError("Quote configuration must cover exactly every candidate")
    for q in quotes.values():
        if not isinstance(q, dict) or set(q) != {"amount", "currency", "scope"}:
            raise ValueError("Each quote must contain exactly amount, currency, scope")
        Quote("replay-v1", **q)
    plans = []
    for seed in seeds:
        split = split_tasks(dataset["tasks"], seed, calibration_fraction, heldout_fraction)
        probe = sorted((t for t, s in split.items() if s == "probe"))
        orders = {}
        for aid in agents:
            agent_order = list(probe)
            random.Random(int(digest([seed, aid, "probe-order"]), 16)).shuffle(agent_order)
            orders[aid] = agent_order[:max_probes]
        plans.append({"seed": seed, "split": split, "probe_order_by_agent": orders, "probe_pool_size": len(probe)})
    body = {"version": PLAN_VERSION if rubric_id == "ABCD-v2-evidence" else LEGACY_PLAN_VERSION, "task_type": dataset["task_type"], "rubric_id": rubric_id, "dataset_fingerprint": dataset["fingerprint"], "seeds": list(seeds),
            "scenarios": list(scenarios), "roster": roster, "malicious_reviewers": agents[:max_bad], "reviewer_roster_scope": "producer_agents_only_self_excluded",
            "max_bad": max_bad, "max_probes": max_probes, "threshold": threshold, "delta": delta,
            "beta": beta, "eta": eta, "drift": drift, "pointwise_beta": pointwise_beta, "pointwise_basis": pointwise_basis, "quotes": quotes, "provider": provider, "synthetic_config": synthetic_config, "grading_label": "explicit_synthetic_not_quality" if provider == "synthetic" else grading_label, "splits": plans}
    return {**body, "plan_id": digest(body)}


def validate_mutual_roster(dataset, plan):
    """Reject renamed/rehashed legacy plans before exporting any peer requests."""
    expected_rubric = "ABCD-v2-evidence" if plan.get("version") == PLAN_VERSION else "ABCD-v1"
    if plan.get("rubric_id") != expected_rubric:
        raise ValueError("Frozen plan version and grading contract mismatch")
    agents = sorted(dataset["agents"])
    f = plan.get("max_bad")
    if (plan.get("roster") != agents
            or plan.get("reviewer_roster_scope") != "producer_agents_only_self_excluded"):
        raise ValueError("Frozen mutual roster must contain exactly the producer agents")
    if type(f) is not int or f < 0 or len(agents)-1 <= 2*f:
        raise ValueError("Frozen mutual roster requires strict honest eligible majority")
    if plan.get("malicious_reviewers") != agents[:f]:
        raise ValueError("Frozen malicious reviewers must be the preregistered producer subset")


def requests_for(dataset, plan):
    if plan.get("version") not in SUPPORTED_PLAN_VERSIONS:
        raise ValueError("Unsupported replay plan version; preserve old evidence and preregister a fresh per-agent-stream plan")
    validate_mutual_roster(dataset, plan)
    requests = {}
    for split in plan["splits"]:
        for aid in sorted(dataset["agents"]):
            tids = set(split["probe_order_by_agent"][aid]) | {t for t, s in split["split"].items() if s == "calibration"}
            for tid in sorted(tids):
                artifact = dataset["pairs"][aid, tid]
                for reviewer in plan["roster"]:
                    if reviewer == aid:
                        continue
                    request = build_judge_request(dataset["tasks"][tid], artifact, reviewer, rubric_id=plan["rubric_id"])
                    requests[request["request_id"]] = request
    return [requests[k] for k in sorted(requests)]


def preflight(dataset, plan):
    requests = requests_for(dataset, plan)
    availability = {aid: {status: sum(a["agent_id"] == aid and a["status"] == status for a in dataset["artifacts"].values()) for status in ("completed", "failed", "missing")} for aid in dataset["agents"]}
    shared_available = sum(all(dataset["pairs"][aid, tid]["status"] != "missing" for aid in dataset["agents"]) for tid in dataset["tasks"])
    shared_labeled = sum(all(dataset["pairs"][aid, tid]["artifact_id"] in dataset["truth"] for aid in dataset["agents"]) for tid in dataset["tasks"])
    n = len(plan["roster"])-1
    k = n if n % 2 else n-1
    rho = bias_bound(n, plan["max_bad"], k, plan["beta"], plan["eta"], plan["drift"])["bias_floor"]
    power = []
    for split in plan["splits"]:
        m = min(plan["max_probes"], split["probe_pool_size"])
        radius = math.sqrt(math.log(2*len(dataset["agents"])*m*(m+1)/plan["delta"])/(2*m))
        lower = max(0., 1.-rho-radius)
        power.append({"seed":split["seed"], "distinct_probe_cap":m, "probe_pool_size":split["probe_pool_size"],
                      "stochastic_radius_at_cap":radius, "assumed_bias_floor":rho,
                      "best_possible_lower_at_cap":lower, "adaptive_crossing_possible_at_cap":lower >= plan["threshold"],
                      "warning":"Increasing rounds beyond the fixed pool does not add independent data. This checks adaptive diagnostic power only, not an optional finite-pool certificate."})
    return {"plan_id": plan["plan_id"], "dataset_fingerprint": dataset["fingerprint"],
            "power_check":power, "nonempty_context_tasks":sum(bool(t["context"].strip()) for t in dataset["tasks"].values()),
            "task_type": plan["task_type"], "rubric_id": plan["rubric_id"], "tasks": len(dataset["tasks"]), "agents": len(dataset["agents"]), "coverage": dataset["coverage"], "artifact_status_by_agent": availability, "shared_nonmissing_tasks": shared_available, "shared_labeled_tasks": shared_labeled,
            "requests": len(requests), "cases": len(plan["seeds"])*len(plan["scenarios"]),
            "split_counts": [{"seed": p["seed"], **{s: sum(v == s for v in p["split"].values()) for s in ("calibration", "probe", "heldout")}, "capped_probe_count_by_agent": {a: len(order) for a, order in p["probe_order_by_agent"].items()}} for p in plan["splits"]],
            "benchmark_labels": len(dataset["truth"]), "provider": plan["provider"], "live_api_calls": 0, "finite_pool_mode": plan.get("pointwise_beta") is not None,
            "formal_theorem_applicable": False, "limitation": "Static finite replay does not satisfy live conditional-IID task assumptions. CBP bounds are protocol diagnostics only."}


def wilson(successes, count, z=1.96):
    if not count:
        return {"estimate": None, "lower": 0., "upper": 1., "n": 0, "coverage_guarantee": False}
    p = successes/count
    den = 1+z*z/count
    center = (p+z*z/(2*count))/den
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return {"estimate": p, "lower": max(0., center-half), "upper": min(1., center+half), "n": count, "coverage_guarantee": False}


def calibration(dataset, split, grades):
    result = {}
    for reviewer in sorted({r for _, r in grades}):
        fp = fn = neg = pos = abstained = absent_labels = 0
        for artifact in dataset["artifacts"].values():
            if split[artifact["task_id"]] != "calibration" or (artifact["artifact_id"], reviewer) not in grades:
                continue
            label = dataset["truth"].get(artifact["artifact_id"])
            if label is None:
                absent_labels += 1
                continue
            value = grades[artifact["artifact_id"], reviewer]
            if value is None:
                abstained += 1
                continue
            if label["success"]:
                pos += 1
                fn += value == 0
            else:
                neg += 1
                fp += value == 1
        result[reviewer] = {"false_positive": wilson(fp, neg), "false_negative": wilson(fn, pos), "abstained": abstained, "missing_labels": absent_labels}
    return {"reviewers": result, "status": "empirical_only_not_global_beta_eta", "interval": "95% Wilson descriptive; correlated task/agent observations are not independent trials", "note": "Abstentions reported separately; do not infer unconditional error from answered-only estimates"}


class ReplayExecutor:
    def __init__(self, dataset):
        self.dataset = dataset
    def invoke(self, config, task):
        row = self.dataset["pairs"][config.agent_id, task]
        return Output(row["artifact_id"], row["content"], 0.)


def run_case(dataset, plan, split, provider, scenario):
    grades = {}
    for request in requests_for(dataset, {**plan, "splits": [split]}):
        response = provider.grade(request)
        if response is None:
            raise ValueError("Pending grade reached execution")
        grades[request["artifact"]["artifact_id"], request["reviewer_id"]] = response.binary
    configs = []
    for i, (aid, agent) in enumerate(sorted(dataset["agents"].items())):
        base = fixture_config(i, plan["threshold"])
        configs.append(replace(base, ref=ConfigRef("offline-replay", aid, digest(agent)), api="recorded-artifact", api_version="replay-v1", agent_id=aid, model_version=agent["model"] or "unspecified", system_prompt="Read-only replay adapter; original system label: "+(agent["system"] or "unspecified"), task_prompt="Retrieve fixed recorded artifact by task_id; never run the original producer", input_schema="task_id-v1", output_schema="recorded-artifact-v1", reset_rule="fixed immutable files; no execution state", sampling_rule="preregistered uniform task permutation without replacement", retry_rule="no producer retries", failure_rule="preserve explicit failed and missing recordings", task_domain=plan["task_type"], distribution_version=dataset["fingerprint"], evaluation_rule=("ABCD-v1; A/B pass; C/D fail; abstention terminates" if plan["rubric_id"] == "ABCD-v1" else "ABCD-v2-evidence; A/B pass; C/D fail; insufficient evidence remains pending"), target="success_probability", parameters=(("recorded_config_digest", digest(agent)),)))
    assumptions = Assumptions(plan["max_bad"], plan["beta"], plan["eta"], plan["drift"], "external sensitivity assumptions; calibration descriptive only", "frozen-replay-roster", "static replay; theorem inapplicable")
    candidates = [Candidate(c, Quote("replay-v1", **plan["quotes"][c.agent_id]), assumptions) for c in configs]
    market = Market(candidates, plan["roster"], InMemoryRegistry(configs), ReplayExecutor(dataset), delta=plan["delta"], max_probes=max(1, min(plan["max_probes"], split["probe_pool_size"])), seed=split["seed"])
    baselines = {}
    finite_results = []
    for aid in sorted(dataset["agents"]):
        values = {"mean": [], "single_grader": [], "median_fixed": []}
        eligible = sorted(set(plan["roster"])-{aid})
        # Same frozen odd committee and order for all fixed baselines; all see identical reports.
        k = len(eligible) if len(eligible)%2 else len(eligible)-1
        committee = random.Random(int(digest([split["seed"], aid, "fixed-committee"]), 16)).sample(eligible, k)
        single = committee[0]
        finite_scores, finite_attempted, finite_status = [], [], "pending"
        finite_enabled = plan.get("pointwise_beta") is not None
        for tid in split["probe_order_by_agent"][aid]:
            artifact = dataset["pairs"][aid, tid]
            row = [grades[artifact["artifact_id"], r] for r in committee]
            values["mean"].append(None if any(v is None for v in row) else sum(row)/len(row))
            values["median_fixed"].append(None if any(v is None for v in row) else median(row))
            values["single_grader"].append(grades[artifact["artifact_id"], single])
            if finite_enabled and finite_status == "pending":
                finite_attempted.append(tid)
                if any(v is None for v in row):
                    finite_status = "abstained"
                else:
                    finite_scores.append(float(median(row)))
                    fi = finite_interval(finite_scores, pool_size=split["probe_pool_size"], candidates=len(configs), delta=plan["delta"], pointwise_beta=plan["pointwise_beta"])
                    finite_status = "finite_pool_certified" if fi["lower"] >= plan["threshold"] else "below_threshold" if fi["upper"] < plan["threshold"] else "pending"
            if market.summary(aid)["status"] == "pending":
                market.probe(aid, tid, lambda reviewer, config, output, history: grades[output.artifact_id, reviewer])
        if finite_enabled:
            fi = finite_interval(finite_scores, pool_size=split["probe_pool_size"], candidates=len(configs), delta=plan["delta"], pointwise_beta=plan["pointwise_beta"])
            finite_results.append({**fi, "agent_id": aid, "task_type": plan["task_type"], "rubric_id": plan["rubric_id"], "threshold": plan["threshold"], "attempted_task_ids": finite_attempted, "status": "inconclusive" if finite_status == "pending" else finite_status, "quote": plan["quotes"][aid], "target": "equally_weighted_fixed_probe_pool_mean_only", "assumptions": {"pointwise_beta": plan["pointwise_beta"], "basis": plan["pointwise_basis"], "strict_honest_majority_every_complete_item": True, "uniform_without_replacement": True}, "delta": plan["delta"], "delta_scope": "per_seed_scenario_case_not_campaign", "conditional_on_external_pointwise_assumptions": True})
        baselines[aid] = {name: {"mean": sum(v)/len(v) if v and None not in v else None, "n": len(v), "abstained": sum(x is None for x in v), "threshold_crossing": bool(v and None not in v and sum(v)/len(v) >= plan["threshold"]), "guarantee": None} for name, v in values.items()}
    heldout = {}
    for aid in sorted(dataset["agents"]):
        rows = [dataset["pairs"][aid, tid] for tid, s in split["split"].items() if s == "heldout"]
        labels = [dataset["truth"].get(a["artifact_id"]) for a in rows]
        known = [x["success"] for x in labels if x is not None]
        heldout[aid] = {"success_rate": sum(known)/len(known) if known and len(known) == len(rows) else None, "known_labels": len(known), "total": len(rows), "missing_labels": len(rows)-len(known), "failed_artifacts": sum(a["status"] == "failed" for a in rows), "missing_artifacts": sum(a["status"] == "missing" for a in rows)}
    summaries = [{**market.summary(aid), "status": "empirical_threshold_crossing" if market.summary(aid)["status"] == "certified" else "empirical_"+market.summary(aid)["status"], "quote": plan["quotes"][aid], "identity": dataset["agents"][aid], "configuration_scope": "Frozen replay adapter plus recorded producer metadata; original execution configuration not verified", "offered_probe_count": len(split["probe_order_by_agent"][aid]), "offered_task_ids": split["probe_order_by_agent"][aid], "task_type": plan["task_type"], "rubric_id": plan["rubric_id"], "threshold": plan["threshold"]} for aid in sorted(dataset["agents"])]
    return {"seed": split["seed"], "task_type": plan["task_type"], "rubric_id": plan["rubric_id"], "threshold": plan["threshold"], "scenario": scenario, "provider": plan["provider"], "grading_label": plan["grading_label"], "formal_theorem_applicable": False, "guarantees": [], "certified_candidates": [], "candidate_diagnostics": summaries, "empirical_candidates": [s for s in summaries if s["status"] == "empirical_threshold_crossing"], "finite_pool_results": finite_results, "finite_pool_certified_candidates": [f for f in finite_results if f["status"] == "finite_pool_certified"], "offered_probe_count_by_agent": {a: len(order) for a, order in split["probe_order_by_agent"].items()}, "probe_order_by_agent": split["probe_order_by_agent"], "comparison_scope": "Each agent has its own preregistered task permutation, shared by that agent across methods and scenarios; agents may overlap. Matched prerecorded stream, not matched consumed budget: adaptive consumes a prefix, fixed baselines the full offered stream. Requests are eagerly pregraded; consumed counts do not measure actual API savings.", "calibration": calibration(dataset, split["split"], grades), "heldout": heldout, "baselines": baselines, "events": [{**e, "task_id": dataset["artifacts"][e["artifact_id"]]["task_id"] if e.get("artifact_id") in dataset["artifacts"] else None} for e in market.events], "payments": {"peer_review": 0, "real_currency": 0}, "resource_accounting": "Replay spends no producer resources; historical costs are metadata, quotes are separate scenario inputs"}


def run(dataset_dir, output_dir, plan, responses=None, evidence_gaps=None):
    if plan.get("version") not in SUPPORTED_PLAN_VERSIONS:
        raise ValueError("Unsupported replay plan version; start a fresh per-agent-stream plan")
    if plan.get("plan_id") != digest({k: v for k, v in plan.items() if k != "plan_id"}):
        raise ValueError("Preregistered plan integrity mismatch")
    dataset = load_dataset(dataset_dir)
    if dataset["task_type"] != plan["task_type"] or plan["rubric_id"] not in {"ABCD-v1", "ABCD-v2-evidence"}:
        raise ValueError("Category or rubric does not match the frozen market scope")
    if dataset["fingerprint"] != plan["dataset_fingerprint"]:
        raise ValueError("Dataset changed after preregistration")
    validate_mutual_roster(dataset, plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume plan mismatch; use a new output directory")
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2)+"\n")
    requests = requests_for(dataset, plan)
    write_jsonl(output/"requests.jsonl", requests)
    if plan["provider"] == "file":
        base = FileGradeProvider(responses)
        request_ids = {r["request_id"] for r in requests}
        if set(base.responses) - request_ids:
            raise ValueError("Response file contains IDs outside this exact preregistered request set")
        missing = [r for r in requests if base.grade(r) is None]
        gaps = load_evidence_gaps(evidence_gaps, requests)
        unresolved_gaps = [gaps[r["request_id"]] for r in missing if r["request_id"] in gaps]
        cached_path = output/"accepted-responses.json"
        previous = json.loads(cached_path.read_text()) if cached_path.exists() else {}
        current = {key: value.to_dict() for key, value in base.responses.items()}
        if any(current.get(key) != value for key, value in previous.items()):
            raise ValueError("Resume cannot remove or modify an already accepted response")
        cached_path.write_text(json.dumps(current, sort_keys=True, indent=2)+"\n")
        if missing:
            write_jsonl(output/"pending_requests.jsonl", missing)
            result = {"status": "pending_evidence" if unresolved_gaps else "pending_grades", "missing": len(missing), "requests_path": str(output/"pending_requests.jsonl"), "plan_id": plan["plan_id"]}
            if unresolved_gaps:
                write_jsonl(output/"pending_evidence.jsonl", unresolved_gaps)
                result["evidence_gaps"] = len(unresolved_gaps)
                result["note"] = "Unfinished review work; no producer failure or terminal abstention inferred. Repair changed context in a fresh frozen plan."
            (output/"status.json").write_text(json.dumps(result, indent=2)+"\n")
            return result
    cases = []
    for split in plan["splits"]:
        base = FileGradeProvider(responses) if plan["provider"] == "file" else SyntheticNoisyGradeProvider(seed=split["seed"], **plan["synthetic_config"])
        for scenario in plan["scenarios"]:
            provider = AttackGradeProvider(base, mode=scenario, malicious_reviewers=plan["malicious_reviewers"], seed=split["seed"])
            result = run_case(dataset, plan, split, provider, scenario)
            case_path = output/f"case-{split['seed']}-{scenario}.json"
            serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
            if case_path.exists() and case_path.read_text() != serialized:
                raise ValueError("Resume result differs from existing evidence")
            temporary = case_path.with_suffix(".json.tmp")
            temporary.write_text(serialized)
            temporary.replace(case_path)
            cases.append(result)
    if plan["version"] == PLAN_VERSION:
        write_jsonl(output/"pending_requests.jsonl", [])
        if (output/"pending_evidence.jsonl").exists():
            write_jsonl(output/"pending_evidence.jsonl", [])
    result = {"status": "complete", "plan_id": plan["plan_id"], "cases": len(cases), "formal_theorem_applicable": False, "provider": plan["provider"]}
    (output/"status.json").write_text(json.dumps(result, indent=2)+"\n")
    return result
