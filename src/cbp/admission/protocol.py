"""Anytime bounded-score estimation for fixed, preregistered capability cells.

Validity depends on fresh probes and a conditional mean bias bound. Median
aggregation alone does not establish that bound or truthful reviewer behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median


def finite(value, name, lower=0.0, upper=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


@dataclass(frozen=True, order=True)
class Cell:
    provider_id: str
    configuration_id: str
    skill: str

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (self.provider_id, self.configuration_id, self.skill)):
            raise ValueError("Cell identities and skill must be nonempty strings")


@dataclass(frozen=True)
class Review:
    reviewer_id: str
    score: float
    source: str

    def __post_init__(self):
        if not self.reviewer_id or not self.source:
            raise ValueError("Reviewer identity and provenance required")
        finite(self.score, "score", upper=1)


@dataclass(frozen=True)
class AdmissionConfig:
    delta: float = 0.05
    bias_floor: float = 0.0
    threshold: float = 0.5
    epsilon: float = 0.1
    max_probes_per_cell: int = 128
    committee_size: int = 3

    def __post_init__(self):
        finite(self.delta, "delta", upper=1)
        if self.delta in (0, 1):
            raise ValueError("delta must lie strictly between 0 and 1")
        for name in ("bias_floor", "threshold", "epsilon"):
            finite(getattr(self, name), name, upper=1)
        for name in ("max_probes_per_cell", "committee_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.committee_size % 2 != 1:
            raise ValueError("Use a preregistered odd-size committee")


@dataclass
class AdmissionState:
    cells: tuple[Cell, ...]
    reviewer_ids: frozenset[str]
    config: AdmissionConfig = field(default_factory=AdmissionConfig)
    _scores: dict = field(init=False, repr=False)
    _terminal: dict = field(default_factory=dict, init=False, repr=False)
    _probe_ids: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        self.cells = tuple(self.cells)
        self.reviewer_ids = frozenset(self.reviewer_ids)
        if not self.cells or len(set(self.cells)) != len(self.cells):
            raise ValueError("Declare unique capability cells before collecting data")
        if any(not isinstance(x, str) or not x for x in self.reviewer_ids):
            raise ValueError("Reviewer roster needs valid identities")
        for cell in self.cells:
            if len(self.reviewer_ids - {cell.provider_id}) < self.config.committee_size:
                raise ValueError("Insufficient non-self reviewers")
        self._scores = {cell: [] for cell in self.cells}

    def observe(self, cell: Cell, probe_id: str, reviews: list[Review]):
        if cell in self._terminal:
            raise ValueError("Cell is terminal")
        if cell not in self._scores:
            raise ValueError("Cell was not preregistered")
        if not probe_id or (cell, probe_id) in self._probe_ids:
            raise ValueError("Each cell requires fresh, unique probe IDs")
        if len(self._scores[cell]) >= self.config.max_probes_per_cell:
            raise ValueError("Participant's declared probe cap exhausted")
        ids = [r.reviewer_id for r in reviews]
        if len(ids) != self.config.committee_size or len(set(ids)) != len(ids):
            raise ValueError("Need the declared number of distinct reviewers")
        if cell.provider_id in ids or not set(ids) <= self.reviewer_ids:
            raise ValueError("Self review and unknown reviewers are prohibited")
        self._scores[cell].append(float(median(r.score for r in reviews)))
        self._probe_ids.add((cell, probe_id))
        return self.summary(cell)

    def summary(self, cell: Cell):
        scores = self._scores[cell]
        m = len(scores)
        if not m:
            return {"n": 0, "mean": None, "radius": None, "lower": 0.0, "upper": 1.0, "status": self._terminal.get(cell, "pending"), "guarantee": "conditional_assumptions"}
        radius = self.config.bias_floor + math.sqrt(math.log(2 * len(self.cells) * m * (m + 1) / self.config.delta) / (2 * m))
        mean = sum(scores) / m
        lower, upper = max(0.0, mean - radius), min(1.0, mean + radius)
        status = ("admitted" if lower >= self.config.threshold else "rejected" if upper < self.config.threshold
                  else "inconclusive" if m == self.config.max_probes_per_cell else "pending")
        return {"n": m, "mean": mean, "radius": radius, "lower": lower, "upper": upper, "status": self._terminal.get(cell, status), "guarantee": "conditional_assumptions"}

    def rank(self, skill: str):
        candidates = [c for c in self.cells if c.skill == skill]
        if not candidates:
            raise ValueError("Unknown skill")
        summaries = {c: self.summary(c) for c in candidates}
        if any(c in self._terminal for c in candidates):
            return {"status": "inconclusive", "winner": None, "guarantee": "conditional_assumptions"}
        if any(x["n"] == 0 for x in summaries.values()):
            return {"status": "pending", "winner": None, "guarantee": "conditional_assumptions"}
        best = max(candidates, key=lambda c: (summaries[c]["lower"], c.provider_id))
        rivals = [summaries[c]["upper"] for c in candidates if c != best]
        if not rivals or summaries[best]["lower"] >= max(rivals) - self.config.epsilon:
            return {"status": "epsilon_resolved", "winner": best.provider_id, "winning_cell": {"provider_id": best.provider_id, "configuration_id": best.configuration_id, "skill": best.skill}, "guarantee": "conditional_assumptions"}
        exhausted = all(x["n"] == self.config.max_probes_per_cell for x in summaries.values())
        return {"status": "inconclusive" if exhausted else "pending", "winner": None, "guarantee": "conditional_assumptions"}

    def accuracy(self, cell: Cell):
        result = self.summary(cell)
        if cell in self._terminal:
            return {**result, "decision": self._terminal[cell]}
        if self.config.bias_floor >= self.config.epsilon:
            return {**result, "decision": "resolution_floor"}
        if result["radius"] is not None and result["radius"] <= self.config.epsilon:
            return {**result, "decision": "estimate_certified"}
        return {**result, "decision": "inconclusive" if result["n"] >= self.config.max_probes_per_cell else "pending"}

    def next_cell(self, skill: str, mode="threshold"):
        """Predictable heuristic; no globally optimal sample-complexity claim."""
        candidates = [c for c in self.cells if c.skill == skill and c not in self._terminal]
        if not candidates:
            return None
        if mode not in {"threshold", "ranking", "accuracy"}:
            raise ValueError("Unknown sampling mode")
        summaries = {c: self.summary(c) for c in candidates}
        available = [c for c in candidates if summaries[c]["n"] < self.config.max_probes_per_cell]
        if mode == "threshold":
            available = [c for c in available if summaries[c]["status"] == "pending"]
        elif mode == "accuracy":
            available = [c for c in available if self.accuracy(c)["decision"] == "pending"]
        else:
            if self.rank(skill)["status"] != "pending":
                return None
            if all(summaries[c]["n"] for c in candidates):
                leader = max(candidates, key=lambda c: (summaries[c]["lower"], c.provider_id))
                others = [c for c in candidates if c != leader]
                challenger = max(others, key=lambda c: (summaries[c]["upper"], c.provider_id)) if others else leader
                focus = [c for c in available if c in (leader, challenger)]
                available = focus or available
        return min(available, key=lambda c: (summaries[c]["n"], c)) if available else None
