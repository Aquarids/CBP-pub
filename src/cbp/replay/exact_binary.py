"""Exact finite-pool binary bounds under conditional peer-error assumptions.

This optional inference engine is not a calibration proof. True scores must be frozen
potential outcomes on one finite pool, sampled uniformly without replacement.
At every sampled task, conditional on the past, the aggregate report's error
probability must be at most q. Dependence among peers is permitted. All candidate
and look sample sizes are fixed before observing outcomes; unused budget is not recycled.
Reports may be adaptive. “Exact” refers to discrete distribution inversion;
probabilities are computed with ordinary floating-point numerical precision.
"""
from __future__ import annotations

import math


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _probability(value, name, *, open_interval=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")
    if not (0 < value < 1 if open_interval else 0 <= value <= 1):
        raise ValueError(f"{name} outside probability range")
    return float(value)


def peer_error_bound(*, peers=5, faulty=2, eta):
    """Markov bound for a full odd committee; no independent-peer assumption."""
    _integer(peers, "peers", 1)
    _integer(faulty, "faulty")
    eta = _probability(eta, "eta")
    if peers % 2 != 1 or faulty * 2 >= peers:
        raise ValueError("A strict honest majority in an odd full committee is required")
    return min(1., (peers - faulty) * eta / ((peers + 1) // 2 - faulty))


def binomial_error_allowance(n, q, alpha):
    """Smallest B with P[Bin(n,q)>B] <= alpha, using direct finite sums."""
    _integer(n, "n")
    q = _probability(q, "q")
    alpha = _probability(alpha, "alpha", open_interval=True)
    if q == 0 or n == 0:
        return 0
    if q == 1:
        return n
    masses = [math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                       + k*math.log(q)+(n-k)*math.log1p(-q)) for k in range(n+1)]
    # Sum the upper tail directly, avoiding catastrophic cancellation near zero.
    for bound in range(n+1):
        if math.fsum(masses[bound+1:]) <= alpha:
            return bound
    return n


def hypergeom_tail(*, population, successes, draws, observed, upper=True):
    """P[X >= observed] (upper) or P[X <= observed] for Hypergeom(N,K,m)."""
    for value, name in ((population, "population"), (successes, "successes"),
                        (draws, "draws"), (observed, "observed")):
        _integer(value, name)
    if population < 1 or successes > population or draws > population or observed > draws:
        raise ValueError("Invalid hypergeometric arguments")
    lo, hi = max(0, draws-(population-successes)), min(draws, successes)
    indices = range(max(lo, observed), hi+1) if upper else range(lo, min(hi, observed)+1)
    denominator = math.comb(population, draws)
    return sum(math.comb(successes, x)*math.comb(population-successes, draws-x)
               for x in indices) / denominator


def exact_binary_interval(scores, *, pool_size, candidates, looks, delta, q, lower_only=False):
    """Simultaneous two-sided CI at preregistered looks for all candidates.

    Each candidate/look receives delta/(3*C*J) for aggregate error, lower
    sampling tail, and upper sampling tail. lower_only uses delta/(2*C*J)
    for just aggregate error and lower sampling tail, returning upper=1. q is an assumption, not an estimate.
    This function cannot establish random sampling, unique task identities,
    frozen outcomes, preregistration, or conditional calibration for the caller.
    """
    for value, name in ((pool_size, "pool_size"), (candidates, "candidates"), (looks, "looks")):
        _integer(value, name, 1)
    delta = _probability(delta, "delta", open_interval=True)
    q = _probability(q, "q")
    if len(scores) > pool_size or any(type(x) is not int or x not in (0, 1) for x in scores):
        raise ValueError("At most N binary integer observations are required")
    if type(lower_only) is not bool:
        raise ValueError("lower_only must be boolean")
    n, positive = len(scores), sum(scores)
    alpha = delta/((2 if lower_only else 3)*candidates*looks)
    errors = binomial_error_allowance(n, q, alpha)
    low_sample, high_sample = max(0, positive-errors), min(n, positive+errors)
    lower = next(k for k in range(pool_size+1)
                 if hypergeom_tail(population=pool_size, successes=k, draws=n,
                                   observed=low_sample) > alpha)
    upper = pool_size if lower_only else next(k for k in range(pool_size, -1, -1)
                 if hypergeom_tail(population=pool_size, successes=k, draws=n,
                                   observed=high_sample, upper=False) > alpha)
    return {"n": n, "pool_size": pool_size, "positive": positive,
            "q_assumed": q, "lower_only": lower_only, "tail_budget": alpha, "error_allowance": errors,
            "lower_successes": lower, "upper_successes": upper,
            "lower": lower/pool_size, "upper": upper/pool_size}
