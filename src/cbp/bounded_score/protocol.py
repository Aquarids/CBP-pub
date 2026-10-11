"""Bounded honest-score reference protocol. No latent truths enter this API.

Authentication, frozen potential outputs, and a common honest-report deadline
are caller trust boundaries. Scores are exact decimal strings or integers;
floats are rejected rather than silently rounding a threshold comparison.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from functools import lru_cache
from types import MappingProxyType
from collections.abc import Mapping

from ..multigrade.protocol import MultigradeMarket
from ..multigrade.ordinal import exact_count_interval

CUTOFFS = {'A': Fraction(9), 'B': Fraction(7), 'C': Fraction(5)}


def exact(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError('Exact integer, decimal/rational string, or Fraction required')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError('Invalid exact number') from e


def score(value):
    value = exact(value)
    if not 0 <= value <= 10:
        raise ValueError('Score outside [0,10]')
    return value


@lru_cache(maxsize=100000)
def _endpoints(r, eta):
    return max(Fraction(0),r-eta), min(Fraction(10),r+eta)


@dataclass(frozen=True)
class ScoreRow:
    producer: str
    task_id: str
    artifact_id: str
    reports: object
    deadline_passed: bool = True


class IdentitySetBounds:
    """One fixed candidate honest set across every producer and sampled task.

    Missing or malformed scores contribute no constraint. The sampled row
    always remains in the denominator. Deadline flags document finalization;
    they never convert nonresponse into a quality label. Inputs are authenticated maps,
    so duplicates, unknown identities and self-reports are schema failures.
    """
    def __init__(self, roster, F, eta):
        self.roster = tuple(roster)
        if len(set(self.roster)) != len(self.roster) or any(not isinstance(x,str) or not x for x in self.roster):
            raise ValueError('Unique nonempty identity strings required')
        if type(F) is not int or F < 0 or len(self.roster) < 2*F+2:
            raise ValueError('Require m >= 2F+2 and integer F >= 0')
        self.F, self.eta = F, exact(eta)
        if not 0 <= self.eta <= 10:
            raise ValueError('eta must be in [0,10]')
        self.candidates = tuple(combinations(self.roster, len(self.roster)-F))
        self.rows = []
        self.intervals = {H: [] for H in self.candidates}
        self._keys, self._artifacts = set(), set()
        self._standard_counts = {H: {a: {c:[0,0] for c in CUTOFFS.values()} for a in self.roster} for H in self.candidates}
        self._count_cache = {}
        self._taskwise_cache = {}

    def normalize(self, row):
        if not isinstance(row, ScoreRow) or row.producer not in self.roster:
            raise ValueError('Known producer and ScoreRow required')
        if type(row.deadline_passed) is not bool or any(not isinstance(x,str) or not x for x in (row.task_id,row.artifact_id)):
            raise ValueError('Nonempty task/artifact and explicit deadline flag required')
        if not isinstance(row.reports, Mapping):
            raise ValueError('Authenticated report Mapping required; pair lists are ambiguous')
        reports = dict(row.reports)
        if any(j not in self.roster or j == row.producer for j in reports):
            raise ValueError('Unknown or self reviewer')
        clean = {}
        for j, value in reports.items():
            try: clean[j] = score(value)
            except ValueError: clean[j] = None
        return ScoreRow(row.producer,row.task_id,row.artifact_id,MappingProxyType(clean),row.deadline_passed)

    def interval(self, H, row):
        lo, hi = Fraction(0), Fraction(10)
        for j in H:
            if j == row.producer: continue
            r = row.reports.get(j)
            if r is None:
                continue
            lower,upper = _endpoints(r,self.eta)
            lo, hi = max(lo,lower), min(hi,upper)
        return (lo,hi) if lo <= hi else None

    def add_rows(self, rows):
        """Atomic addition, including fixed-set infeasibility as an explicit alarm."""
        rows = [self.normalize(r) for r in rows]
        keys = [(r.producer,r.task_id) for r in rows]
        arts = [r.artifact_id for r in rows]
        if len(set(keys)) != len(keys) or self._keys.intersection(keys) or len(set(arts)) != len(arts) or self._artifacts.intersection(arts):
            raise ValueError('Duplicate sampled task or artifact')
        updated = {}
        updated_counts = {}
        for H, old in self.intervals.items():
            new = [self.interval(H,r) for r in rows]
            if all(i is not None for i in new):
                updated[H] = old + new
                counts=self._standard_counts[H]
                for row,(lo,hi) in zip(rows,new):
                    for c,pair in counts[row.producer].items():
                        pair[0] += lo>=c
                        pair[1] += hi>=c
                updated_counts[H]=counts
        self.rows.extend(rows)
        self._keys.update(keys); self._artifacts.update(arts)
        self.intervals = updated
        self._standard_counts=updated_counts
        self._count_cache.clear()

    @property
    def model_inconsistent(self): return not self.intervals

    def count_bounds(self, producer, cutoff):
        if producer not in self.roster: raise ValueError('Unknown producer')
        cutoff = score(cutoff)
        if self.model_inconsistent: return None
        if cutoff in CUTOFFS.values():
            return min(v[producer][cutoff][0] for v in self._standard_counts.values()),max(v[producer][cutoff][1] for v in self._standard_counts.values())
        key=(producer,cutoff)
        if key in self._count_cache: return self._count_cache[key]
        ix = [i for i,r in enumerate(self.rows) if r.producer == producer]
        result = (min(sum(v[i][0] >= cutoff for i in ix) for v in self.intervals.values()),
                max(sum(v[i][1] >= cutoff for i in ix) for v in self.intervals.values()))
        self._count_cache[key]=result
        return result

    def taskwise_interval(self, row):
        """Exact hull of local compatibility via sorted interval windows.

        With v valid nonself scores, a size-(m-F) candidate honest set can
        include the producer and every unconstrained reviewer. It therefore
        needs only h=max(0,v-F) valid constraints. Intersections of every
        contiguous h-score window cover exactly the feasible union: any
        compatible h-subset can be replaced by a contiguous block between its
        extreme scores. Only the enclosing hull is needed for threshold counts.
        """
        values=sorted(r for r in row.reports.values() if r is not None)
        h=max(0,len(values)-self.F)
        if h==0: return Fraction(0),Fraction(10)
        windows=[]
        for k in range(len(values)-h+1):
            lo=_endpoints(values[k+h-1],self.eta)[0]
            hi=_endpoints(values[k],self.eta)[1]
            if lo<=hi: windows.append((lo,hi))
        if not windows: return None
        return min(lo for lo,hi in windows),max(hi for lo,hi in windows)

    def taskwise_count_bounds(self, producer, cutoff):
        """Strong comparator: independent exact row-compatible honest sets."""
        cutoff = score(cutoff)
        a=b=0
        for i,r in enumerate(self.rows):
            if r.producer != producer: continue
            if i not in self._taskwise_cache:
                self._taskwise_cache[i]=self.taskwise_interval(r)
            interval=self._taskwise_cache[i]
            if interval is None: return None
            a += interval[0] >= cutoff
            b += interval[1] >= cutoff
        return a,b


def numeric_points(row, F):
    """Uncertified task descriptors; invalid/missing values are not zero."""
    values=[]
    for value in row.reports.values():
        try: values.append(score(value))
        except ValueError: pass
    values.sort()
    if not values: return {'mean':None,'median':None,'trimmed_mean':None}
    n=len(values); middle=(values[(n-1)//2]+values[n//2])/2
    trimmed=values[F:n-F] if n > 2*F else []
    return {'mean':sum(values)/n,'median':middle,
            'trimmed_mean':sum(trimmed)/len(trimmed) if trimmed else None}


def median_interval(row, roster, eta):
    """Certified comparator only for a complete valid nonself report vector."""
    if set(row.reports) != set(roster)-{row.producer}: return Fraction(0),Fraction(10)
    try: values=sorted(score(v) for v in row.reports.values())
    except ValueError: return Fraction(0),Fraction(10)
    n=len(values); z=(values[(n-1)//2]+values[n//2])/2; eta=exact(eta)
    return max(Fraction(0),z-eta),min(Fraction(10),z+eta)


class BoundedScoreMarket(MultigradeMarket):
    """Complete atomic rounds, public own quotes, and registered-look inference.

    Reuses the historical sampler and discrete hypergeometric inverter without
    modifying either. CI tails, alpha, endpoints and decisions use exact rational
    arithmetic. Point letters remain descriptive.
    """
    def __init__(self, *, eta='0.5', epsilon='0.1', aggregation='joint_counts', **kwargs):
        rho=Fraction(str(kwargs.get('rho', '0.5')))
        delta=Fraction(str(kwargs.get('delta', '0.05')))
        if not 0<rho<=1 or not 0<delta<1: raise ValueError('Require rho in(0,1], delta in(0,1)')
        epsilon=exact(epsilon)
        if not 0<epsilon<Fraction(1,2): raise ValueError('Require 0<epsilon<1/2')
        kwargs['rho'],kwargs['delta']=float(rho),float(delta)
        super().__init__(**kwargs)
        self.rho,self.delta,self.epsilon=rho,delta,epsilon
        self.alpha=delta/(2*len(self.roster)*3*(1 if self.method=='fixed_ci' else len(self.looks)))
        if self.method not in ('sequential_ci','fixed_ci'):
            raise ValueError('Only registered-look CI methods supported')
        if self.method == 'fixed_ci' and len(self.looks) != 1:
            raise ValueError('Fixed CI requires one endpoint')
        if aggregation not in ('joint_counts','joint_hulls','taskwise','median_interval'):
            raise ValueError('Unknown certified aggregation')
        self.aggregation = aggregation
        self.bounds = IdentitySetBounds(self.roster,self.F,eta)
        self.point_scores = {a:[] for a in self.roster}
        self._hull_cache = {}
        self._median_cache = {}

    @property
    def model_inconsistent(self):
        """Each comparator uses only its own model-compatibility information."""
        if self.aggregation in ('joint_counts','joint_hulls'):
            return self.bounds.model_inconsistent
        if self.aggregation == 'median_interval': return False
        return any(self.bounds.taskwise_count_bounds(a,5) is None for a in self.roster)

    @property
    def method_inconsistent(self): return self.model_inconsistent

    @property
    def stopped(self):
        if self.model_inconsistent: return True
        return self.round >= self.looks[-1] or (self.round in self.looks and all(self.certified_grade(a) is not None for a in self.roster))

    def record_round(self, outputs):
        tasks=self.next_round()
        if tasks is None or set(outputs) != set(self.roster):
            raise ValueError('Live complete round required')
        rows=[]
        for a in self.roster:
            o=outputs[a]
            if set(o) != {'task_id','artifact_id','reports','deadline_passed'} or o['task_id'] != tasks[a] or o['deadline_passed'] is not True:
                raise ValueError('Context-bound finalized report rows required')
            rows.append(ScoreRow(a,o['task_id'],o['artifact_id'],o['reports'],True))
        self.bounds.add_rows(rows)
        self.round+=1
        self._hull_cache.clear()
        for r in rows: self.point_scores[r.producer].append(numeric_points(r,self.F)['median'])
        if self.model_inconsistent:
            self.summaries={a:{l:{'lower':Fraction(0),'upper':Fraction(1),'status':'model_inconsistent','at_look':None} for l in CUTOFFS} for a in self.roster}
            return
        if self.round not in self.looks: return
        for a in self.roster:
            for l,c in CUTOFFS.items():
                low,high=self.sample_count_bounds(a,c)
                L=exact_count_interval(self.N,self.round,low,self.alpha)[0]
                U=exact_count_interval(self.N,self.round,high,self.alpha)[1]
                lo,hi=Fraction(L,self.N),Fraction(U,self.N)
                self.summaries[a][l]={'lower':lo,'upper':hi,'status':'certified' if lo>=self.rho else 'excluded' if hi<self.rho else 'unresolved','certain':low,'possible':high,'n':self.round,'at_look':self.round}

    def sample_count_bounds(self, a, c):
        if self.aggregation == 'joint_counts': return self.bounds.count_bounds(a,c)
        if self.aggregation == 'taskwise': return self.bounds.taskwise_count_bounds(a,c)
        indices=[i for i,r in enumerate(self.bounds.rows) if r.producer==a]
        if self.aggregation == 'joint_hulls':
            if a not in self._hull_cache:
                self._hull_cache[a]=[(min(v[i][0] for v in self.bounds.intervals.values()),max(v[i][1] for v in self.bounds.intervals.values())) for i in indices]
            intervals=self._hull_cache[a]
        else:
            for i in indices:
                if i not in self._median_cache:
                    self._median_cache[i]=median_interval(self.bounds.rows[i],self.roster,self.bounds.eta)
            intervals=[self._median_cache[i] for i in indices]
        return sum(lo>=c for lo,hi in intervals),sum(hi>=c for lo,hi in intervals)

    def certified_grade(self, a):
        """Highest admissible tolerant letter; not strict market qualification."""
        levels=self.summaries[a]
        if levels['A'].get('at_look') is None: return None
        low,high=self.rho-self.epsilon,self.rho+self.epsilon
        if levels['A']['lower']>=low: return 'A'
        if levels['B']['lower']>=low and levels['A']['upper']<high: return 'B'
        if levels['C']['lower']>=low and levels['B']['upper']<high: return 'C'
        if levels['C']['upper']<high: return 'D'
        return None

    def tolerance_candidate_list(self):
        return [{**c,'tolerance_qualified_levels':[l for l,s in self.summaries[c['agent_id']].items() if s.get('at_look') is not None and s['lower']>=self.rho-self.epsilon], 'proportion_floor':self.rho-self.epsilon} for c in self.candidates if any(s.get('at_look') is not None and s['lower']>=self.rho-self.epsilon for s in self.summaries[c['agent_id']].values())]

    def publications(self):
        result={}
        for a in self.roster:
            values=self.point_scores[a]
            point=next((l for l,c in CUTOFFS.items() if values and Fraction(sum(v is not None and v>=c for v in values),len(values))>=self.rho),'D')
            grade=self.certified_grade(a) if not self.model_inconsistent else None
            result[a]={'point_grade':point,'certified_grade':grade,'status':'model_inconsistent' if self.model_inconsistent else 'certified' if grade else 'budget_ambiguity' if self.round>=self.looks[-1] else 'pending','point_valid_tasks':sum(v is not None for v in values),'certificate_contract':'epsilon_tolerant_cumulative_grade','rho':self.rho,'epsilon':self.epsilon,'cumulative_bounds':self.summaries[a]}
        return result

    @property
    def costs(self):
        m=len(self.roster)
        return {'rounds':self.round,'probe_executions':m*self.round,'review_requests':m*(m-1)*self.round}
