"""Outcome-independent fixed ordinal decisions and reserved-risk capped variant.

New refinement module; the earlier comparison's scientific sources are frozen.
Empirical ordinal decisions have no coordinate-CI or strict-offer interface.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb
from .optimized import OptimizedMarket, choose_admissible
from .protocol import LEVELS, meets
from .pac import PAC_LOOKS
from ..single_agent.finite_game import _int, _prob, count_interval

REFINED_METHODS = ("fixed_width_unrestricted", "fixed_grade_unrestricted", "direct_ordinal", "capped_ordinal")


def fraction(x):
    return x if isinstance(x,Fraction) else Fraction(str(x))


def ceiling(x):
    x=fraction(x)
    return -(-x.numerator//x.denominator)


def validate(N,m,rho,epsilon,delta):
    _int(N,"N",1);_int(m,"m",1)
    _prob(float(rho),"rho",strict=False);_prob(float(epsilon),"epsilon",strict=True);_prob(float(delta),"delta",strict=True)
    if rho<=0 or epsilon>=.5:raise ValueError("Require rho>0 and epsilon<.5")


@lru_cache(maxsize=20000)
def ordinal_risk(N,m,n,rho,epsilon):
    """Exact union risk of empirical nested ordinal rule at one fixed endpoint."""
    _int(N,"N",1);_int(m,"m",1);_int(n,"n",1)
    if n>N:raise ValueError("n exceeds population")
    rho,epsilon=fraction(rho),fraction(epsilon)
    if not 0<rho<=1 or not 0<epsilon<1:raise ValueError("Invalid threshold/tolerance")
    c=ceiling(n*rho);lo=rho-epsilon;hi=rho+epsilon
    denominator=comb(N,n);numerator=0
    if lo>0:
        K=ceiling(N*lo)-1
        numerator+=sum(comb(K,s)*comb(N-K,n-s) for s in range(max(c,n+K-N,0),min(n,K)+1))
    if hi<=1:
        K=ceiling(N*hi)
        numerator+=sum(comb(K,s)*comb(N-K,n-s) for s in range(max(0,n+K-N),min(n,K,c-1)+1))
    return min(Fraction(1),Fraction(m*numerator,denominator))


@lru_cache(maxsize=256)
def ordinal_horizon(N,m,rho,epsilon,delta):
    validate(N,m,rho,epsilon,delta)
    rho,epsilon,delta=map(fraction,(rho,epsilon,delta))
    for n in range(1,N+1):
        if ordinal_risk(N,m,n,rho,epsilon)<=delta:return n
    raise AssertionError("Census must suffice")


def _count_order(n):
    # A central failing witness rejects n cheaply; sufficiency still checks ALL s.
    first=(n//2,(n+1)//2)
    return tuple(dict.fromkeys((*first,*range(n+1))))


def ci_condition(N,m,n,rho,epsilon,delta,goal):
    rho,epsilon,delta=map(fraction,(rho,epsilon,delta))
    alpha=float(delta/(6*m))
    for s in _count_order(n):
        low,high=count_interval(N,n,s,alpha)
        if goal=="width":
            if Fraction(high-low,N)>2*epsilon:return False
        elif goal=="grade":
            if not (Fraction(low,N)>=rho-epsilon or Fraction(high,N)<rho+epsilon):return False
        else:raise ValueError("Unknown CI design goal")
    return True


@lru_cache(maxsize=256)
def ci_horizon(N,m,rho,epsilon,delta,goal):
    validate(N,m,rho,epsilon,delta)
    for n in range(1,N+1):
        if ci_condition(N,m,n,rho,epsilon,delta,goal):return n
    raise AssertionError("Census must suffice")


@lru_cache(maxsize=100000)
def exact_count_interval(N,n,s,alpha):
    """Exact rational hypergeometric-tail inversion for the residual early budget."""
    alpha=fraction(alpha)
    if not 0<alpha<1 or not 0<=s<=n<=N:raise ValueError("Invalid exact interval arguments")
    if n==N:return s,s
    denominator=comb(N,n)
    def retained(K,upper):
        low=max(0,n+K-N,s if upper else 0)
        high=min(n,K,n if upper else s)
        numerator=sum(comb(K,x)*comb(N-K,n-x) for x in range(low,high+1))
        return numerator*alpha.denominator>denominator*alpha.numerator
    lo,hi=0,N
    while lo<hi:
        mid=(lo+hi)//2
        if retained(mid,True):hi=mid
        else:lo=mid+1
    lower=lo;lo,hi=0,N
    while lo<hi:
        mid=(lo+hi+1)//2
        if retained(mid,False):lo=mid
        else:hi=mid-1
    return lower,lo


def exact_levels(levels,N):
    return {l:{"lower":Fraction(s.get("lower_count",round(s["lower"]*N)),N),
               "upper":Fraction(s.get("upper_count",round(s["upper"]*N)),N)} for l,s in levels.items()}


class OrdinalMarket(OptimizedMarket):
    def __init__(self, *, refinement_method="direct_ordinal", early_grid=None, **kwargs):
        if refinement_method not in REFINED_METHODS:raise ValueError("Unknown refinement")
        self.refinement_method=refinement_method
        N=len(kwargs["task_ids"]);m=len(kwargs["candidates"])
        rho=kwargs.get("rho",.5);epsilon=kwargs.get("epsilon",.1);delta=kwargs.get("delta",.05)
        validate(N,m,rho,epsilon,delta)
        if early_grid is None:early_grid=tuple(n for n in PAC_LOOKS if n<=N)
        if refinement_method in ("fixed_width_unrestricted","fixed_grade_unrestricted"):
            self.endpoint=ci_horizon(N,m,rho,epsilon,delta,"width" if refinement_method=="fixed_width_unrestricted" else "grade")
            self.fallback_risk=None;self.early_risk=Fraction(0);self.early_looks=()
        else:
            self.endpoint=ordinal_horizon(N,m,rho,epsilon,delta)
            self.fallback_risk=ordinal_risk(N,m,self.endpoint,fraction(rho),fraction(epsilon))
            self.early_risk=fraction(delta)-self.fallback_risk
            if any(type(n) is not int or n<1 or n>N for n in early_grid) or tuple(sorted(set(early_grid)))!=tuple(early_grid):raise ValueError("Register increasing early grid within pool")
            self.early_looks=tuple(n for n in early_grid if n<self.endpoint) if refinement_method=="capped_ordinal" and self.early_risk>0 else ()
        looks=(*self.early_looks,self.endpoint)
        super().__init__(optimization_method="fixed_pac_ci" if refinement_method=="fixed_width_unrestricted" else "direct_pac_grade",looks=looks,**kwargs)
        # Reuse authentication/counting only for ordinal methods. Their early
        # CIs are recomputed exactly below; no extra full-delta CI family exists.
        if self.is_ordinal:self.method="certificate_point"
        if self.early_looks:
            self.alpha=float(self.early_risk/(6*m*len(self.early_looks)))

    @property
    def is_ordinal(self):return self.refinement_method in ("direct_ordinal","capped_ordinal")

    @property
    def grade_only(self):return self.refinement_method!="fixed_width_unrestricted"

    @property
    def complete_prefix(self):return all(len(gs)==self.round and None not in gs for gs in self._grades.values())

    @property
    def success(self):
        if self.is_ordinal and self.round==self.endpoint:return self.complete_prefix
        if self.empty_intersection or self.round not in self.looks or not all(s.get("at_look")==self.round for ls in self.summaries.values() for s in ls.values()):return False
        if self.grade_only:return all(choose_admissible(exact_levels(ls,self.N),fraction(self.rho),fraction(self.epsilon)) is not None for ls in self.summaries.values())
        return self.width_stop_success

    @property
    def stopped(self):
        if self.is_ordinal:return self.round>=self.endpoint or self.success
        return super().stopped

    @property
    def width_stop_success(self):
        if self.is_ordinal or self.empty_intersection or self.round not in self.looks:return False
        return all(s.get("at_look")==self.round and Fraction(round((s["upper"]-s["lower"])*self.N),self.N)<=2*fraction(self.epsilon) for ls in self.summaries.values() for s in ls.values())

    def estimates(self):
        if self.is_ordinal and self.round==self.endpoint:
            return {a:{l:(sum(g is not None and meets(g,l) for g in gs)+.5*gs.count(None))/self.round for l in LEVELS} for a,gs in self._grades.items()}
        return super().estimates()

    def approximate_grades(self):
        if not self.success:return {a:None for a in self.roster}
        if self.is_ordinal and self.round==self.endpoint:
            c=ceiling(self.round*fraction(self.rho))
            return {a:next((l for l in LEVELS if sum(meets(g,l) for g in gs)>=c),"D") for a,gs in self._grades.items()}
        if self.grade_only:return {a:choose_admissible(exact_levels(ls,self.N),fraction(self.rho),fraction(self.epsilon)) for a,ls in self.summaries.items()}
        return super().approximate_grades()

    def record_round(self,outputs):
        previous_failure=self.empty_intersection
        previous_eligibility=dict(self.first_eligible_at)
        super().record_round(outputs)
        if self.is_ordinal:self.first_eligible_at=previous_eligibility
        if self.round not in self.early_looks:return
        exact_alpha=self.early_risk/(6*len(self.roster)*len(self.early_looks))
        actual_failure=False
        for a,levels in self.summaries.items():
            for l,summary in levels.items():
                lo=exact_count_interval(self.N,self.round,summary["positives"],exact_alpha)[0]
                hi=exact_count_interval(self.N,self.round,summary["positives"]+summary["pending"],exact_alpha)[1]
                summary.update(lower=lo/self.N,upper=hi/self.N,lower_count=lo,upper_count=hi,
                    status="empty" if lo>hi else "certified" if Fraction(lo,self.N)>=fraction(self.rho) else "excluded" if Fraction(hi,self.N)<fraction(self.rho) else "unresolved")
                actual_failure |= lo>hi
        self.empty_intersection=previous_failure or actual_failure
        if not self.empty_intersection:
            for a,ls in self.summaries.items():
                if a not in self.first_eligible_at and choose_admissible(exact_levels(ls,self.N),fraction(self.rho),fraction(self.epsilon)) is not None:self.first_eligible_at[a]=self.round
        for event in self.events[-len(self.roster):]:event["bounds"]={l:dict(v) for l,v in self.summaries[event["producer"]].items()}

    def candidate_list(self):
        return None if self.is_ordinal else super().candidate_list()
