"""Oracle-free stopping variants with separate precision and grade-only goals.

The existing PAC implementation is frozen. This extension reuses authenticated
atomic rounds and uniform sampling, and changes the registered interval/stopping
rule. WSR means the uniform-total prior/posterior ratio confidence sequence of
Waudby-Smith and Ramdas (2020); no fixed-look multiplicity is charged to that CS.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb
from .pac import PACMarket
from .protocol import GRADES, LEVELS
from ..single_agent.finite_game import _int, _prob

OPTIMIZED_METHODS = ("fixed_pac_ci", "pac_ci", "intersect_pac_ci",
                     "direct_pac_grade", "direct_intersect_pac_grade", "wsr_cs_precision", "wsr_cs_grade")


@lru_cache(maxsize=100000)
def _wsr_count_interval(N, n, s, numerator, denominator):
    cutoff = numerator * comb(N, n)
    scale = (n + 1) * denominator
    retained = [K for K in range(s, N - n + s + 1)
                if comb(K, s) * comb(N-K, n-s) * scale > cutoff]
    return None if not retained else (retained[0], retained[-1])


def wsr_count_interval(N, n, s, alpha):
    """Integer-exact PMF > alpha/(n+1), returning the retained total's hull."""
    _int(N, "population", 1); _int(n, "sample size"); _int(s, "positives")
    _prob(float(alpha), "stream alpha", strict=True)
    if s > n or n > N:
        raise ValueError("Require 0 <= positives <= draws <= population")
    alpha = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
    return _wsr_count_interval(N, n, s, alpha.numerator, alpha.denominator)


def admissible_grades(levels, rho, epsilon):
    """Grades whose threshold decisions exclude every epsilon-severe error."""
    if any(s["lower"] > s["upper"] for s in levels.values()):
        return ()
    L = {l: Fraction(str(levels[l]["lower"])) for l in LEVELS}
    U = {l: Fraction(str(levels[l]["upper"])) for l in LEVELS}
    rho, epsilon = Fraction(str(rho)), Fraction(str(epsilon))
    return tuple(g for g, ok in zip(GRADES, (
        L["A"] >= rho-epsilon,
        L["B"] >= rho-epsilon and U["A"] < rho+epsilon,
        L["C"] >= rho-epsilon and U["B"] < rho+epsilon,
        U["C"] < rho+epsilon)) if ok)


def choose_admissible(levels, rho, epsilon):
    admissible = admissible_grades(levels, rho, epsilon)
    if not admissible:
        return None
    midpoint_grade = next((l for l in LEVELS
                           if (levels[l]["lower"]+levels[l]["upper"])/2 >= rho), "D")
    rank = GRADES.index(midpoint_grade)
    # Ordinal rank is a deterministic tie rule, not a cardinal quality metric.
    return min(admissible, key=lambda g: (abs(GRADES.index(g)-rank), -GRADES.index(g)))


class OptimizedMarket(PACMarket):
    def __init__(self, *, optimization_method="intersect_pac_ci", **kwargs):
        if optimization_method not in OPTIMIZED_METHODS:
            raise ValueError("Unknown optimization method")
        self.optimization_method = optimization_method
        self.empty_intersection = False
        self.first_eligible_at = {}
        super().__init__(pac_method="fixed_pac_ci" if optimization_method == "fixed_pac_ci" else "pac_ci", **kwargs)
        self.stream_alpha = Fraction(str(self.delta))/(3*len(self.roster))

    @property
    def grade_only(self):
        return self.optimization_method in ("direct_pac_grade", "direct_intersect_pac_grade", "wsr_cs_grade")

    @property
    def success(self):
        if self.empty_intersection or self.round not in self.looks:
            return False
        if not all(s.get("at_look") == self.round for ls in self.summaries.values() for s in ls.values()):
            return False
        if self.grade_only:
            return all(admissible_grades(ls, self.rho, self.epsilon) for ls in self.summaries.values())
        return self.width_stop_success

    @property
    def width_stop_success(self):
        return not self.empty_intersection and super().width_stop_success

    @property
    def stopped(self):
        return self.empty_intersection or self.round >= self.looks[-1] or self.success

    def record_round(self, outputs):
        prior = self.summaries
        previous = {a:{l:dict(s) for l,s in ls.items()} for a,ls in prior.items()} if self.round+1 in self.looks else None
        super().record_round(outputs)
        if self.round not in self.looks:
            return
        intersect = self.optimization_method not in ("pac_ci", "fixed_pac_ci", "direct_pac_grade")
        use_wsr = self.optimization_method.startswith("wsr_")
        for a, levels in self.summaries.items():
            for l, s in levels.items():
                if use_wsr:
                    # Union over all completions preserves pending samples in n.
                    intervals = [wsr_count_interval(self.N,self.round,k,self.stream_alpha)
                                 for k in range(s["positives"],s["positives"]+s["pending"]+1)]
                    intervals = [v for v in intervals if v is not None]
                    lo, hi = (min(v[0] for v in intervals)/self.N,max(v[1] for v in intervals)/self.N) if intervals else (1.,0.)
                else:
                    lo,hi = s["lower"],s["upper"]
                if intersect:
                    lo,hi = max(lo,previous[a][l]["lower"]),min(hi,previous[a][l]["upper"])
                s.update(lower=lo,upper=hi,status="empty" if lo>hi else "certified" if lo>=self.rho else "excluded" if hi<self.rho else "unresolved")
                self.empty_intersection |= lo > hi
            if choose_admissible(levels,self.rho,self.epsilon) is not None and a not in self.first_eligible_at:
                self.first_eligible_at[a] = self.round
        for event in self.events[-len(self.roster):]:
            event["bounds"] = {l:dict(s) for l,s in self.summaries[event["producer"]].items()}

    def approximate_grades(self):
        if not self.success:
            return {a:None for a in self.roster}
        if self.grade_only:
            return {a:choose_admissible(ls,self.rho,self.epsilon) for a,ls in self.summaries.items()}
        return super().approximate_grades()
