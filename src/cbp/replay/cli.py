"""Offline-first replay CLI; acquisition and live grading are never implicit."""
import argparse
import csv
import json
from pathlib import Path
from .schema import load_dataset
from .runner import SCENARIOS, make_plan, preflight, run


def summarize(directory, destination):
    directory, destination = Path(directory), Path(destination)
    plan = json.loads((directory/"plan.json").read_text())
    rows = []
    for seed in plan["seeds"]:
        for scenario in plan["scenarios"]:
            path = directory/f"case-{seed}-{scenario}.json"
            if not path.exists():
                raise ValueError(f"Incomplete campaign: {path.name} is missing")
            case = json.loads(path.read_text())
            for candidate in case["candidate_diagnostics"]:
                aid = candidate["agent_id"]
                heldout = case["heldout"][aid]
                rows.append({"seed": seed, "scenario": scenario, "task_type": case["task_type"], "rubric_id": case["rubric_id"], "threshold": case["threshold"], "agent_id": aid, "method": "cbp_adaptive", "status": candidate["status"], "score": candidate["mean"], "n": candidate["n"], "heldout_success_rate": heldout["success_rate"], "heldout_missing_labels": heldout["missing_labels"], "provider": case["provider"], "grading_label": case["grading_label"], "formal_theorem_applicable": False, "selected": candidate["status"] == "empirical_threshold_crossing", "heldout_below_threshold_selection": candidate["status"] == "empirical_threshold_crossing" and heldout["success_rate"] is not None and heldout["success_rate"] < plan["threshold"]})
                for name, baseline in case["baselines"][aid].items():
                    rows.append({"seed": seed, "scenario": scenario, "task_type": case["task_type"], "rubric_id": case["rubric_id"], "threshold": case["threshold"], "agent_id": aid, "method": name, "status": "empirical_threshold_crossing" if baseline["threshold_crossing"] else "abstained" if baseline["abstained"] else "below_threshold", "score": baseline["mean"], "n": baseline["n"], "heldout_success_rate": heldout["success_rate"], "heldout_missing_labels": heldout["missing_labels"], "provider": case["provider"], "grading_label": case["grading_label"], "formal_theorem_applicable": False, "selected": baseline["threshold_crossing"], "heldout_below_threshold_selection": baseline["threshold_crossing"] and heldout["success_rate"] is not None and heldout["success_rate"] < plan["threshold"]})
            for candidate in case.get("finite_pool_results", []):
                heldout = case["heldout"][candidate["agent_id"]]
                selected = candidate["status"] == "finite_pool_certified"
                rows.append({"seed": seed, "scenario": scenario, "task_type": case["task_type"], "rubric_id": case["rubric_id"], "threshold": case["threshold"], "agent_id": candidate["agent_id"], "method": "cbp_finite_pool_adaptive", "status": candidate["status"], "score": candidate["mean"], "n": candidate["n"], "heldout_success_rate": heldout["success_rate"], "heldout_missing_labels": heldout["missing_labels"], "provider": case["provider"], "grading_label": case["grading_label"], "formal_theorem_applicable": False, "selected": selected, "heldout_below_threshold_selection": selected and heldout["success_rate"] is not None and heldout["success_rate"] < plan["threshold"]})
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"replay-metrics.json").write_text(json.dumps({"plan_id": plan["plan_id"], "rows": rows, "note": "Replay diagnostics; optional finite-pool results remain conditional and probe-pool-only. Heldout mismatch is not a finite-pool false-certificate measurement. No live deployment quality guarantee"}, indent=2, allow_nan=False)+"\n")
    with (destination/"replay-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["seed"])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "destination": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recorded-artifact CBP replay; offline by default, no real payments")
    parser.add_argument("command", choices=["preflight", "run", "resume", "summarize", "fetch-public", "enrich-context", "grade-pivot"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/replay"))
    parser.add_argument("--responses", type=Path, help="Recorded human/model ABCD grades; missing rows pause cleanly")
    parser.add_argument("--evidence-gaps", type=Path, help="Separate needs_evidence records; unfinished work, never a grade")
    parser.add_argument("--provider", choices=["file", "synthetic"], default="file")
    parser.add_argument("--grading-label", default="recorded_unverified", help="Disclose manual, recorded-model, or explicitly mock grading provenance")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=["honest"])
    parser.add_argument("--reviewers", type=int, default=0, help="Must be zero: only producer agents mutually review; no external reviewers")
    parser.add_argument("--max-bad", type=int, default=0)
    parser.add_argument("--max-probes", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--eta", type=float, default=1.)
    parser.add_argument("--drift", type=float, default=0.)
    parser.add_argument("--honest-noise", type=float, default=.15, help="Synthetic independent request noise standard deviation")
    parser.add_argument("--shared-difficulty", type=float, default=.2, help="Synthetic shared task difficulty standard deviation")
    parser.add_argument("--reviewer-bias-scale", type=float, default=.1, help="Synthetic persistent reviewer bias standard deviation")
    parser.add_argument("--pointwise-beta", type=float, help="Explicit optional finite-pool per-item deterministic honest grading error bound")
    parser.add_argument("--pointwise-basis", help="External justification for pointwise bound, never inferred from calibration estimates")
    parser.add_argument("--quotes", type=Path, help="JSON keyed by agent ID with amount/currency/scope; not historic cost")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report only; never grade or write")
    parser.add_argument("--summary-output", type=Path, default=Path("docs/metrics/replay"))
    parser.add_argument("--allow-live", action="store_true", help="Explicit opt-in for only grade-pivot; may incur provider charges")
    parser.add_argument("--pivot-config", type=Path, help="Explicit existing dotenv file; never copied to judge requests")
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--max-requests", type=int, default=1, help="Maximum new live reviews, default 1; only grade-pivot")
    parser.add_argument("--task-limit", type=int, default=12)
    parser.add_argument("--context-budget", type=int, default=45000)
    parser.add_argument("--context-cache", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "grade-pivot":
            if args.requests is None or args.responses is None or args.pivot_config is None:
                parser.error("grade-pivot requires --requests, --responses and --pivot-config")
            if args.dry_run:
                result = {"status": "dry_run", "live_api_calls": 0, "max_requests": args.max_requests}
            else:
                from .grading import grade_pivot_requests
                result = grade_pivot_requests(args.requests, args.responses, args.pivot_config, allow_live=args.allow_live, max_requests=args.max_requests)
        elif args.command == "enrich-context":
            if args.dataset is None:
                parser.error("enrich-context requires --dataset")
            if args.dry_run:
                result = {"status":"dry_run", "network":"explicit pinned public base-source download", "context_budget":args.context_budget}
            else:
                from .context import enrich_dataset
                result = enrich_dataset(args.dataset, args.output, budget=args.context_budget, cache_dir=args.context_cache)
        elif args.command == "fetch-public":
            if args.dry_run:
                result = {"status": "dry_run", "network": "explicit public dataset download", "task_limit": args.task_limit}
            else:
                from .public_data import fetch_swebench_public
                result = fetch_swebench_public(args.output, task_limit=args.task_limit)
        elif args.command == "summarize":
            result = {"status": "dry_run", "destination": str(args.summary_output)} if args.dry_run else summarize(args.output, args.summary_output)
        else:
            if args.dataset is None:
                parser.error("--dataset is required")
            dataset = load_dataset(args.dataset)
            if args.command == "resume":
                plan = json.loads((args.output/"plan.json").read_text())
                if plan["dataset_fingerprint"] != dataset["fingerprint"]:
                    raise ValueError("Dataset fingerprint mismatch")
            else:
                quotes = json.loads(args.quotes.read_text()) if args.quotes else None
                plan = make_plan(dataset, seeds=tuple(args.seeds), scenarios=tuple(args.scenarios), reviewers=args.reviewers, max_bad=args.max_bad, max_probes=args.max_probes, threshold=args.threshold, delta=args.delta, beta=args.beta, eta=args.eta, drift=args.drift, quotes=quotes, provider=args.provider, pointwise_beta=args.pointwise_beta, pointwise_basis=args.pointwise_basis, honest_noise=args.honest_noise, shared_difficulty=args.shared_difficulty, reviewer_bias_scale=args.reviewer_bias_scale, grading_label=args.grading_label)
            result = preflight(dataset, plan) if args.command == "preflight" or args.dry_run else run(args.dataset, args.output, plan, args.responses, args.evidence_gaps)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"replay: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
