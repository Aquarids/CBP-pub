"""Current single-agent, offline certification entry point."""
import argparse
import json
from pathlib import Path
from .single_agent.experiment import SCENARIOS, run_experiment, run_suite


def main(argv=None):
    import sys
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective and effective[0] == "multigrade":
        from .multigrade.cli import main as multigrade_main
        return multigrade_main(effective[1:])
    if effective and effective[0] == "replay":
        from .replay.cli import main as replay_main
        return replay_main(effective[1:])
    parser = argparse.ArgumentParser(description="CBP: certify single agents; show all certified quotes; user selects")
    parser.add_argument("command", choices=["smoke", "experiment", "suite"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--scenario", choices=SCENARIOS, default="honest")
    parser.add_argument("--max-bad", type=int, default=0)
    parser.add_argument("--max-probes", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.)
    parser.add_argument("--eta", type=float, default=0.)
    parser.add_argument("--drift", type=float, default=0.)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--target", choices=["expected_quality", "success_probability"], default="expected_quality")
    parser.add_argument("--select", help="Explicit caller choice for one synthetic formal invocation")
    parser.add_argument("--output", type=Path, default=Path("runs/single-agent/result.json"))
    args = parser.parse_args(argv)
    kwargs = dict(max_probes=args.max_probes, beta=args.beta, eta=args.eta,
                  drift=args.drift, threshold=args.threshold, delta=args.delta, target=args.target)
    if args.command == "suite":
        if args.select:
            parser.error("--select is only valid for an individual experiment")
        result = run_suite(seeds=tuple(args.seeds), **kwargs)
    else:
        result = run_experiment(seed=args.seed, scenario=args.scenario, max_bad=args.max_bad,
                                selected_agent=args.select, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "metrics": result.get("metrics"),
                      "certified_agents": [c["agent_id"] for c in result.get("certified_candidates", [])],
                      "cases": len(result.get("cases", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
