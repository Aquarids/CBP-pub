"""Separate voluntary resource records from formal-task mock money transfers."""
from dataclasses import dataclass, field
from .protocol import finite
from .providers import Artifact


@dataclass
class ResourceLedger:
    events: list[dict] = field(default_factory=list)

    def record(self, participant: str, activity: str, artifact: Artifact):
        if not participant or activity not in {"probe", "review", "formal"}:
            raise ValueError("Participant and valid activity required")
        self.events.append({"participant": participant, "activity": activity,
                            "resource_units": artifact.resource_units,
                            "elapsed_seconds": artifact.elapsed_seconds, "tokens": artifact.tokens,
                            "external_monetary_cost": artifact.monetary_cost, "currency": artifact.currency, "source": artifact.source})

    def summary(self):
        currencies = {e["currency"] for e in self.events if e["external_monetary_cost"] is not None}
        return {"events": len(self.events),
                "counts": {kind: sum(e["activity"] == kind for e in self.events) for kind in ("probe", "review", "formal")},
                "modeled_resource_units": sum(e["resource_units"] for e in self.events),
                "external_monetary_cost": None if len(currencies) != 1 or any(e["external_monetary_cost"] is None for e in self.events) else sum(e["external_monetary_cost"] for e in self.events),
                "currency": next(iter(currencies)) if len(currencies) == 1 else None,
                "known_cost_subtotals_by_currency": {currency: sum(e["external_monetary_cost"] for e in self.events if e["currency"] == currency and e["external_monetary_cost"] is not None) for currency in sorted(currencies)},
                "tokens": None if any(e["tokens"] is None for e in self.events) else sum(e["tokens"] for e in self.events)}


@dataclass(frozen=True)
class Bid:
    provider_id: str
    amount: float

    def __post_init__(self):
        if not self.provider_id:
            raise ValueError("Provider identity required")
        finite(self.amount, "bid")


def choose_bid(bids: list[Bid], admitted_lower_bounds: dict[str, float], value: float, budget: float):
    """Pay-as-bid selection by conservative modeled surplus; no IC claim.

    The market never receives or verifies private execution costs.
    The caller supplies bounds for admitted candidates only.
    """
    finite(value, "value"); finite(budget, "budget")
    if len({b.provider_id for b in bids}) != len(bids):
        raise ValueError("At most one bid per provider")
    for bound in admitted_lower_bounds.values():
        finite(bound, "lower bound", upper=1)
    eligible = [(value * admitted_lower_bounds[b.provider_id] - b.amount, b)
                for b in bids if b.provider_id in admitted_lower_bounds and b.amount <= budget]
    eligible = [(surplus, b) for surplus, b in eligible if surplus > 0]
    return None if not eligible else sorted(eligible, key=lambda pair: (-pair[0], pair[1].provider_id))[0][1]


@dataclass
class MarketLedger:
    user_balance: float
    provider_balances: dict[str, float]
    transfers: list[dict] = field(default_factory=list)

    def __post_init__(self):
        finite(self.user_balance, "user_balance")
        self.provider_balances = dict(self.provider_balances)
        for balance in self.provider_balances.values():
            finite(balance, "provider balance")

    def settle_formal(self, task_id: str, bid: Bid):
        if not task_id or any(t["task_id"] == task_id for t in self.transfers):
            raise ValueError("Formal task settlement must have a unique ID")
        if bid.provider_id not in self.provider_balances or bid.amount > self.user_balance:
            raise ValueError("Unknown provider or insufficient user balance")
        self.user_balance -= bid.amount
        self.provider_balances[bid.provider_id] += bid.amount
        self.transfers.append({"task_id": task_id, "kind": "formal_task", "to": bid.provider_id, "amount": bid.amount})

    def summary(self):
        return {"currency": "mock_credit", "user_balance": self.user_balance,
                "provider_balances": dict(self.provider_balances), "transfers": list(self.transfers),
                "probe_transfers": 0, "peer_review_transfers": 0}


def private_bid_policy(provider_id: str, private_cost_bound: float, markup: float, budget: float):
    """Participant-local feasibility policy; not a truthful-cost mechanism."""
    finite(private_cost_bound, "private_cost_bound")
    finite(markup, "markup"); finite(budget, "budget")
    amount = private_cost_bound + markup
    return None if amount > budget else Bid(provider_id, amount)
