"""Small deterministic wiring check, not empirical validation of the assumptions."""
from dataclasses import asdict
from .protocol import AdmissionConfig, AdmissionState, Cell, Review
from .providers import Artifact
from .ledger import ResourceLedger, MarketLedger, Bid, choose_bid, private_bid_policy


def run_smoke(max_probes=128):
    config = AdmissionConfig(max_probes_per_cell=max_probes)
    cells = (Cell("strong", "scripted-v1", "copy"), Cell("weak", "scripted-v1", "copy"))
    state = AdmissionState(cells, frozenset({"reviewer-a", "reviewer-b", "reviewer-c"}), config)
    resources = ResourceLedger()
    # Known synthetic score oracle solely for checking software plumbing.
    # It does not constitute independent real-world quality measurement.
    for i in range(max_probes * len(cells)):
        cell = state.next_cell("copy", "threshold")
        if cell is None:
            break
        score = 1.0 if cell.provider_id == "strong" else 0.0
        resources.record(cell.provider_id, "probe", Artifact("fixture", "scripted-fixture", 1))
        reviews = []
        for reviewer in sorted(state.reviewer_ids):
            resources.record(reviewer, "review", Artifact(str(score), "scripted-fixture", .1))
            reviews.append(Review(reviewer, score, "scripted-fixture"))
        state.observe(cell, f"probe-{i}", reviews)
    estimates = {c.provider_id: state.summary(c) for c in cells}
    lower = {name: x["lower"] for name, x in estimates.items() if x["status"] == "admitted"}
    bids = [private_bid_policy("strong", .15, .1, 1), private_bid_policy("weak", .05, .05, 1)]
    winner = choose_bid(bids, lower, value=1, budget=1)
    money = MarketLedger(1, {"strong": 0, "weak": 0})
    if winner:
        resources.record(winner.provider_id, "formal", Artifact("completed fixture", "scripted-fixture", 1))
        money.settle_formal("formal-1", winner)
    return {"schema_version": 1, "scope": "deterministic offline plumbing smoke; no empirical robustness or incentive claim",
            "config": asdict(config), "estimates": estimates, "ranking": state.rank("copy"),
            "selected": winner.provider_id if winner else None, "resource_metrics": resources.summary(),
            "market_ledger": money.summary(), "provider_contract_kinds": ["human", "program", "model"],
            "exercised_provider_kind": "program (scripted fixture)"}
