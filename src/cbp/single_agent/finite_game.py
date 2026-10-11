"""Truth-free finite-pool certification with ideal correct peer verification.

This module never receives latent ability, private costs or census labels. It
assumes a fixed potential outcome for every candidate/task, uniform task draws,
and at most F faulty identities. Honest peers implement the same exact binary
acceptance predicate. This is a simulation protocol, not real LLM calibration.
"""
from __future__ import annotations

from functools import lru_cache
import math
import random

from ..replay.exact_binary import hypergeom_tail


def _int(x, name, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return x


def _prob(x, name, *, strict=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite probability")
    if not (0 < x < 1 if strict else 0 <= x <= 1):
        raise ValueError(f"{name} outside probability range")
    return float(x)


def count_interval(N, n, s, alpha):
    """Exact hypergeometric inversion; alpha is allocated to EACH tail.

    'Exact' means discrete inversion, not arbitrary precision certification.
    Binary searches use monotone tails and exact integer binomial coefficients.
    """
    _int(N, "population", 1); _int(n, "sample size"); _int(s, "positives")
    _prob(alpha, "tail alpha", strict=True)
    if s > n or n > N:
        raise ValueError("Require 0 <= positives <= draws <= population")
    return _count_interval(N,n,s,float(alpha))


@lru_cache(maxsize=100000)
def _count_interval(N,n,s,alpha):
    if n == N:
        return s, s
    lo, hi = 0, N
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom_tail(population=N, successes=mid, draws=n, observed=s) > alpha:
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = 0, N
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom_tail(population=N, successes=mid, draws=n, observed=s, upper=False) > alpha:
            lo = mid
        else:
            hi = mid - 1
    return lower, lo


class FinitePoolMarket:
    """List every certified candidate with its quote; selection is external.

    Missing/invalid peer reports are pending evidence, not a fifth grade, a
    producer failure, or a skipped task. Three matching binary reports suffice
    for a five-peer/F=2 committee; majority of the first three does NOT suffice.
    """
    def __init__(self, *, candidates, task_ids, threshold, delta, looks, seed,
                 max_faulty=2, method="robust_exact"):
        self.candidates = tuple(dict(c) for c in candidates)
        self.roster = tuple(c["agent_id"] for c in self.candidates)
        self.task_ids = tuple(task_ids)
        if len(self.roster) < 2 or len(set(self.roster)) != len(self.roster):
            raise ValueError("Unique participant roster required")
        if any(not isinstance(x, str) or not x for x in (*self.roster, *self.task_ids)):
            raise ValueError("Nonempty public identities required")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Unique finite task pool required")
        for c in self.candidates:
            if set(c) != {"agent_id", "quote"}:
                raise ValueError("Protocol accepts only public identity and quote")
            if isinstance(c["quote"], bool) or not isinstance(c["quote"], (int,float)) or not math.isfinite(c["quote"]) or c["quote"] < 0:
                raise ValueError("Finite nonnegative quote required")
        self.N = len(self.task_ids)
        self.threshold = _prob(threshold, "threshold")
        self.delta = _prob(delta, "delta", strict=True)
        self.max_faulty = _int(max_faulty, "fault bound")
        if len(self.roster)-1 <= 2*self.max_faulty:
            raise ValueError("Require nonself committee size > 2F")
        self.looks = tuple(looks)
        if not self.looks or any(type(n) is not int or n < 1 or n > self.N for n in self.looks) or tuple(sorted(set(self.looks))) != self.looks:
            raise ValueError("Preregister distinct increasing looks within the pool")
        if method not in {"robust_exact", "fixed_budget_exact", "ordinary_mean", "median_point", "self_report", "never_certify"}:
            raise ValueError("Unknown experimental protocol method")
        self.method = method
        self.alpha = self.delta/(2*len(self.roster)*(1 if method == "fixed_budget_exact" else len(self.looks)))
        self._orders = {}
        for i, aid in enumerate(self.roster):
            order = list(self.task_ids)
            random.Random(seed + 1000003*i).shuffle(order)
            self._orders[aid] = tuple(order)
        self._scores = {a: [] for a in self.roster}
        self._seen = {a: set() for a in self.roster}
        self._pending_reports = {a: {} for a in self.roster}
        self._summary = {a: {"agent_id":a,"n":0,"positive":0,"lower":0.,"upper":1.,"status":"pending"} for a in self.roster}
        self.events = []

    def summary(self, aid):
        return dict(self._summary[aid])

    def next_task(self, aid):
        if self._summary[aid]["status"] not in {"pending", "needs_evidence"}:
            return None
        return self._orders[aid][len(self._scores[aid])]

    def record(self, aid, task_id, artifact_id, reports, *, own_grade=None):
        if task_id != self.next_task(aid):
            raise ValueError("Task must be next in candidate's random permutation")
        if not isinstance(artifact_id,str) or not artifact_id or artifact_id in self._seen[aid]:
            raise ValueError("Fresh artifact identity required")
        peers = set(self.roster)-{aid}
        if not isinstance(reports, dict) or not set(reports) <= peers:
            raise ValueError("Reports must come from nonself registered identities")
        valid = {r:g for r,g in reports.items() if g in ("A","B","C","D")}
        pending = self._pending_reports[aid]
        if pending and pending["artifact_id"] != artifact_id:
            raise ValueError("Pending task cannot switch artifacts")
        prior = pending.get("reports", {})
        if any(r in prior and prior[r] != g for r,g in valid.items()):
            raise ValueError("Conflicting duplicate reviewer report")
        merged = {**prior, **valid}
        votes = [int(g in ("A","B")) for g in merged.values()]
        required = self.max_faulty+1
        yes, no = sum(votes), len(votes)-sum(votes)
        if max(yes,no) < required:
            self._pending_reports[aid] = {"artifact_id":artifact_id,"reports":merged}
            self._summary[aid]["status"] = "needs_evidence"
            return self.summary(aid)
        if yes >= required and no >= required:
            raise ValueError("Conflicting quorums violate the declared fault assumption")
        median_score = int(yes >= required)
        if self.method == "ordinary_mean":
            if len(merged) != len(peers):
                self._pending_reports[aid] = {"artifact_id":artifact_id,"reports":merged}
                self._summary[aid]["status"] = "needs_evidence"
                return self.summary(aid)
            score = sum(votes)/len(votes)
        elif self.method == "self_report":
            if own_grade not in ("A","B","C","D"):
                raise ValueError("Self-report baseline requires explicit own grade")
            score = int(own_grade in ("A","B"))
        else:
            score = median_score
        self._seen[aid].add(artifact_id)
        self._pending_reports[aid] = {}
        self._scores[aid].append(score)
        n, s = len(self._scores[aid]), sum(self._scores[aid])
        lower, upper = self._summary[aid]["lower"], self._summary[aid]["upper"]
        status = "pending"
        at_look = n in self.looks and (self.method != "fixed_budget_exact" or n == self.looks[-1])
        if at_look:
            if self.method in {"robust_exact","fixed_budget_exact"}:
                L,U = count_interval(self.N,n,int(s),self.alpha)
                lower,upper = L/self.N,U/self.N
            elif self.method == "never_certify":
                lower,upper = 0.,1.
            else:
                lower=upper=s/n
            if self.method != "never_certify" and (self.method != "fixed_budget_exact" or n == self.looks[-1]):
                status = "certified" if lower >= self.threshold else "below_threshold" if upper < self.threshold else "pending"
            if n == self.looks[-1] and status == "pending":
                status = "inconclusive"
        self._summary[aid] = {"agent_id":aid,"n":n,"positive":s,"lower":lower,"upper":upper,"status":status}
        self.events.append({"agent_id":aid,"task_id":task_id,"artifact_id":artifact_id,
                            "reports":merged,"score":score,"at_look":at_look,
                            **self._summary[aid]})
        return self.summary(aid)

    def certified_list(self):
        return [{**self.summary(c["agent_id"]),"quote":c["quote"]} for c in self.candidates
                if self._summary[c["agent_id"]]["status"] == "certified"]

    def select(self, aid):
        if aid not in {x["agent_id"] for x in self.certified_list()}:
            raise ValueError("Explicit choice must name a certified candidate")
        return dict(next(c for c in self.candidates if c["agent_id"] == aid))
