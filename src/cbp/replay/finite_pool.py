"""Optional finite-pool inference under explicit pointwise grading assumptions.

This is not the live protocol theorem. Hoeffding (1963), Theorem 4 / Section 6,
permits the same bounded-mean tail bound for uniform sampling without replacement.
The union bound here is over K candidates and at most N prefixes each.
"""
import math


def interval(scores, *, pool_size, candidates, delta, pointwise_beta):
    n = len(scores)
    if type(pool_size) is not int or pool_size < 1 or type(candidates) is not int or candidates < 1:
        raise ValueError("Positive finite pool and candidate count required")
    if not 0 < delta < 1 or isinstance(delta, bool) or not 0 <= pointwise_beta <= 1 or isinstance(pointwise_beta, bool):
        raise ValueError("Invalid finite-pool probability or pointwise error bound")
    if n > pool_size or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise ValueError("Finite pool accepts at most N distinct bounded observations")
    if not n:
        return {"n": 0, "pool_size": pool_size, "lower": 0., "upper": 1., "mean": None, "sampling_radius": None}
    mean = sum(scores)/n
    radius = 0. if n == pool_size else math.sqrt(math.log(2*candidates*pool_size/delta)/(2*n))
    completion_lower = sum(max(0., x-pointwise_beta) for x in scores)/pool_size
    completion_upper = (sum(min(1., x+pointwise_beta) for x in scores)+pool_size-n)/pool_size
    return {"n": n, "pool_size": pool_size, "mean": mean, "sampling_radius": radius,
            "lower": max(0., mean-pointwise_beta-radius, completion_lower),
            "upper": min(1., mean+pointwise_beta+radius, completion_upper)}
