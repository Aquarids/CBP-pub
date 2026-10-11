"""Oracle-free authenticated-peer ABCD certification for one fixed task category.

Authentication is an input trust boundary, not implemented cryptography. The
caller must authenticate identities and bind messages to their contexts. No
ability, benchmark labels, execution costs, or simulator world enters this API.
"""
from dataclasses import dataclass
import math
import random

from ..single_agent.finite_game import count_interval, _int, _prob

GRADES = ("A", "B", "C", "D")
LEVELS = GRADES[:3]
METHODS = ("sequential_ci", "fixed_ci", "certificate_point", "median_point", "median_point_fixed", "raw_vote_point")


def meets(grade, level):
    """Nested rubric membership, never numeric grade averaging."""
    if grade not in GRADES or level not in LEVELS:
        raise ValueError("Unknown rubric grade/level")
    return grade in GRADES[:GRADES.index(level)+1]


@dataclass(frozen=True)
class Vote:
    reviewer: str
    producer: str
    task_id: str
    artifact_id: str
    round_number: int
    rubric: str
    grade: str


def grade_certificate(votes, *, roster, producer, task_id, artifact_id,
                      round_number, rubric, F):
    """F+1 matching distinct nonself votes; None means pending, never D."""
    _int(F, "F")
    if producer not in roster or len(set(roster)) != len(roster):
        raise ValueError("Invalid roster/producer")
    seen = set()
    groups = {g: [] for g in GRADES}
    for vote in votes:
        if not isinstance(vote, Vote):
            raise ValueError("Authenticated context-bound Vote required")
        if vote.reviewer not in roster or vote.reviewer == producer or vote.reviewer in seen:
            raise ValueError("Unknown, self, or duplicate reviewer")
        if (vote.producer, vote.task_id, vote.artifact_id, vote.round_number, vote.rubric) != (producer, task_id, artifact_id, round_number, rubric):
            raise ValueError("Vote context mismatch")
        if vote.grade not in GRADES:
            raise ValueError("Invalid grade")
        seen.add(vote.reviewer)
        groups[vote.grade].append(vote.reviewer)
    winners = [g for g in GRADES if len(groups[g]) >= F+1]
    if len(winners) > 1:
        raise ValueError("Conflicting certificates: declared assumptions violated")
    return None if not winners else {"grade": winners[0], "reviewers": groups[winners[0]], "required": F+1}


class MultigradeMarket:
    def __init__(self, *, candidates, task_ids, task_category, rubric="nested-abcd-v1",
                 F, rho=.50, delta=.05, looks, seed=0, method="sequential_ci"):
        self.candidates = tuple(dict(c) for c in candidates)
        for c in self.candidates:
            if set(c) != {"agent_id", "quote"}:
                raise ValueError("Only public agent_id and quote may enter assessment")
            if type(c["quote"]) not in (float, int) or not math.isfinite(c["quote"]) or c["quote"] < 0:
                raise ValueError("Finite nonnegative own quote required")
        self.roster = tuple(c["agent_id"] for c in self.candidates)
        self.tasks = tuple(task_ids)
        if any(not isinstance(a,str) or not a for a in (*self.roster,*self.tasks)) or len(set(self.roster)) != len(self.roster) or not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Unique nonempty public identities and task pool required")
        self.F = _int(F,"F")
        if len(self.roster) < 2*self.F+2:
            raise ValueError("Require m >= 2F+2; F is a declared conservative bound")
        if not isinstance(task_category,str) or not task_category or not isinstance(rubric,str) or not rubric:
            raise ValueError("Fixed category and rubric required")
        self.task_category, self.rubric = task_category, rubric
        self.N = len(self.tasks)
        self.rho, self.delta = _prob(rho,"rho"), _prob(delta,"delta",strict=True)
        if self.rho == 0:
            raise ValueError("rho must be positive")
        self.looks = tuple(looks)
        if not self.looks or any(type(n) is not int or not 1 <= n <= self.N for n in self.looks) or tuple(sorted(set(self.looks))) != self.looks:
            raise ValueError("Preregister increasing looks within finite pool")
        if method not in METHODS:
            raise ValueError("Unknown method")
        self.method = method
        self.alpha = self.delta/(2*len(self.roster)*3*(1 if method == "fixed_ci" else len(self.looks)))
        self._orders = {}
        for i,a in enumerate(self.roster):
            order = list(self.tasks)
            random.Random(seed+1000003*i).shuffle(order)
            self._orders[a] = order
        self.round = 0
        self._grades = {a: [] for a in self.roster}
        self._raw = {a: [] for a in self.roster}
        self._artifacts = set()
        self.events = []
        self.summaries = {a:{l:{"lower":0.,"upper":1.,"status":"unresolved"} for l in LEVELS} for a in self.roster}

    @property
    def stopped(self):
        return self.round >= self.looks[-1] or (self.round > 0 and all(s["status"] != "unresolved" for levels in self.summaries.values() for s in levels.values()))

    def next_round(self):
        if self.stopped:
            return None
        return {a:self._orders[a][self.round] for a in self.roster}

    def record_round(self, outputs):
        """Exactly one output per agent. Validate entire round before mutation.

        outputs maps agent to {task_id, artifact_id, votes}. Missing reports leave
        the sampled output pending; its unknown grade remains in the denominator.
        Late report repair is not implemented in this synchronous reference runner.
        """
        tasks = self.next_round()
        if tasks is None or set(outputs) != set(self.roster):
            raise ValueError("A live round requires one output from every agent")
        staged = []
        new_ids = set()
        for a in self.roster:
            output = outputs[a]
            if set(output) != {"task_id","artifact_id","votes"} or output["task_id"] != tasks[a]:
                raise ValueError("Output must match next sampled task and public schema")
            artifact = output["artifact_id"]
            if not isinstance(artifact,str) or not artifact or artifact in self._artifacts or artifact in new_ids:
                raise ValueError("Fresh unique artifact required")
            new_ids.add(artifact)
            votes = tuple(output["votes"])
            try:
                cert = grade_certificate(votes, roster=self.roster, producer=a, task_id=tasks[a], artifact_id=artifact, round_number=self.round+1, rubric=self.rubric,F=self.F)
            except ValueError as error:
                if self.method not in ("median_point", "median_point_fixed", "raw_vote_point") or "Conflicting certificates" not in str(error):
                    raise
                cert = None
            staged.append((a,artifact,votes,cert))
        self.round += 1
        self._artifacts.update(new_ids)
        at_look = self.round in self.looks and (self.method not in ("fixed_ci", "median_point_fixed") or self.round == self.looks[-1])
        for a,artifact,votes,cert in staged:
            grade = cert["grade"] if cert else None
            if self.method in ("median_point", "median_point_fixed"):
                # Ordinal lower median; with even peers ties favor the better grade.
                grade = sorted((v.grade for v in votes), key=GRADES.index)[(len(votes)-1)//2] if len(votes) == len(self.roster)-1 else None
            self._grades[a].append(grade)
            # Full fixed committee denominator: absent votes are unknown, not removed.
            self._raw[a].append({l:(sum(meets(v.grade,l) for v in votes),len(self.roster)-1-len(votes)) for l in LEVELS})
            if at_look:
                for level in LEVELS:
                    positives = sum(g is not None and meets(g,level) for g in self._grades[a])
                    pending = self._grades[a].count(None)
                    if self.method in ("sequential_ci","fixed_ci"):
                        L = count_interval(self.N,self.round,positives,self.alpha)[0]
                        U = count_interval(self.N,self.round,positives+pending,self.alpha)[1]
                        lo,hi = L/self.N,U/self.N
                    elif self.method in ("certificate_point", "median_point", "median_point_fixed"):
                        lo,hi = positives/self.round,(positives+pending)/self.round
                    else:
                        den = self.round*(len(self.roster)-1)
                        lo = sum(x[level][0] for x in self._raw[a])/den
                        hi = sum(sum(x[level]) for x in self._raw[a])/den
                    self.summaries[a][level] = {"lower":lo,"upper":hi,"status":"certified" if lo >= self.rho else "excluded" if hi < self.rho else "unresolved","positives":positives,"pending":pending,"n":self.round,"at_look":self.round}
            self.events.append({"round":self.round,"producer":a,"task_id":tasks[a],"artifact_id":artifact,"vote_groups":{g:[v.reviewer for v in votes if v.grade == g] for g in GRADES},"rubric":self.rubric,"certificate":cert,"assessed_grade":grade,"at_look":at_look,"bounds":{l:dict(s) for l,s in self.summaries[a].items()} if at_look else None})

    def candidate_list(self):
        """All candidates with any certified level; no quote or budget filtering."""
        return [{**c,"certified_levels":[l for l in LEVELS if self.summaries[c["agent_id"]][l]["status"] == "certified"]} for c in self.candidates if any(s["status"] == "certified" for s in self.summaries[c["agent_id"]].values())]
