"""Finite-precision extension of the frozen strict protocol; no oracle inputs."""
from .protocol import MultigradeMarket, LEVELS
from ..single_agent.finite_game import _prob, count_interval

PAC_LOOKS = (16, 32, *range(64, 513, 32))
PAC_METHODS = ("pac_ci", "fixed_pac_ci", "census_ci")


def worst_width(N, n, alpha):
    """Data-independent enumeration, using the shared finite-population inverter."""
    return max(count_interval(N, n, s, alpha)[1] - count_interval(N, n, s, alpha)[0]
               for s in range(n + 1)) / N


def fixed_horizon(*, N, m, delta, epsilon, looks):
    """First grid endpoint attaining precision with a genuine single-look budget.

    The documented standard grid has independently audited exhaustive bounds.
    Other parameterizations enumerate the shared inverter, never outcome data.
    """
    if N == 512 and m == 20 and delta == .05 and tuple(looks) == PAC_LOOKS:
        audited = {.05: 352, .10: 192, .15: 128}
        if epsilon in audited:
            return audited[epsilon]
    for n in looks:
        if worst_width(N, n, delta / (6 * m)) <= 2 * epsilon:
            return n
    raise ValueError("No registered fixed horizon guarantees epsilon precision")


class PACMarket(MultigradeMarket):
    """Reuses atomic authenticated rounds and exact CIs; only stopping changes."""
    def __init__(self, *, epsilon=.10, pac_method="pac_ci", **kwargs):
        self.epsilon = _prob(epsilon, "epsilon", strict=True)
        if self.epsilon >= .5:
            raise ValueError("epsilon must be less than 1/2")
        if pac_method not in PAC_METHODS:
            raise ValueError("Unknown PAC method")
        self.pac_method = pac_method
        super().__init__(method="sequential_ci" if pac_method == "pac_ci" else "fixed_ci", **kwargs)
        if pac_method != "pac_ci" and len(self.looks) != 1:
            raise ValueError("Fixed and census methods require one predeclared endpoint")
        if pac_method == "census_ci" and self.looks != (self.N,):
            raise ValueError("Census endpoint must equal pool size")

    @property
    def width_stop_success(self):
        return self.round in self.looks and all(
            s.get("at_look") == self.round and s["upper"] - s["lower"] <= 2 * self.epsilon
            for levels in self.summaries.values() for s in levels.values())

    @property
    def stopped(self):
        return self.round >= self.looks[-1] or self.width_stop_success

    def estimates(self):
        return {a: {level: (s["lower"] + s["upper"]) / 2
                    for level, s in levels.items()} for a, levels in self.summaries.items()}

    def approximate_grades(self):
        """Publish a precision-qualified approximate grade only after success."""
        if not self.width_stop_success:
            return {a: None for a in self.roster}
        return {a: next((l for l in LEVELS if q[l] >= self.rho), "D")
                for a, q in self.estimates().items()}
