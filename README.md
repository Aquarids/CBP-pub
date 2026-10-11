# Capability Bidding Protocol: Review Artifact

This artifact contains the original CBP method implementations and a small,
offline execution example. Scientific experiment plans, datasets, comparative
studies, and paper sources are not included.

CBP processes peer assessments into conditional capability certificates and
service offers. The bounded-score implementation retains a common honest-identity
hypothesis across sampled tasks, computes threshold-success count bounds, and
publishes finite-pool capability intervals and tolerant grades at preregistered
assessment rounds. Demand-specific qualification and capability-to-price ranking
are implemented separately from certification. The single-agent reference
interface binds configurations, probes services, collects peer reviews, lists all
certified quotes, and executes an explicitly selected certified service.

These interfaces use distinct statistical assumptions. Their certificates are
conditional on the declared identity, scoring, execution, and sampling model.
This artifact does not establish calibration for live providers, truthful bids,
or unconditional service quality. It provides no deployed blockchain or payment
system.

## Requirements and installation

Python 3.10 or newer is required. The core methods and smoke example use only the
Python standard library. Run the following commands from the repository root:

```sh
python -m pip install -e .
```

Installation is optional for the standalone runner. Live-model and public-data
integrations are optional and are not used by the smoke example or contract tests.

## Offline smoke example

```sh
python exp/runner/run_cbp.py
```

The runner invokes the original `python -m cbp smoke` interface with six synthetic
providers, a fixed local task, and synthetic reviewer callbacks. It exercises
configuration resolution, probing, certification, and formal invocation of an
explicitly selected certified service. A separate four-provider, two-task fixture
calls the original bounded-score API and demand-price ranking function. It also
checks missing-review ambiguity, inconsistent-model refusal, and exact-count
solver agreement. These fixtures do not change the protocol implementations.

The runner prints a JSON result with `status: "PASS"` after its checks succeed.
Outputs are written under the ignored `runs/smoke/` directory:

- `smoke.json`: check results, offers, ranking, and modeled request counts;
- `single-agent.json`: original reference-interface output and synthetic events;
- `normal-cli.log`: original CLI output.

All inputs are constructed in memory. No dataset download, live API call, or
currency transfer occurs. Resource counters are fixture units; unknown real usage
and monetary cost are reported as `null`.

## Contract checks

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p 'check_*.py'
```

The retained tests cover bounded-score count constraints and certification,
exact-count solver parity and resource guards, single-agent configuration and
review boundaries, and admission/controller contracts. Passing these synthetic
checks is not an empirical quality evaluation or a validation of live-provider
assumptions.

## Source layout

- `src/cbp/bounded_score/`: joint-identity inference and exact-count solver;
- `src/cbp/single_agent/`: configuration-bound service reference interface;
- `src/cbp/admission/`: admission, committee, reviewer, and resource interfaces;
- `src/cbp/market_comparison/buyer_policy.py`: demand qualification and price ranking;
- `src/cbp/multigrade/`, `src/cbp/replay/`, and selected `src/cbp/core/` modules:
  original statistical and public-entry-point dependencies;
- `exp/runner/`: standalone synthetic smoke;
- `tests/`: small synthetic contract checks.

Some retained modules contain historical variants or fixture helpers imported by
the original methods and entry points. They remain in their original locations
to preserve source fidelity. Historical experiment inputs are not supplied, so
those experiment commands are outside the review smoke workflow.
