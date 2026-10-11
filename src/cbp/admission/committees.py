"""Conditional median calibration bound; inputs are assumptions, not inferred truth."""
import math
import random
from .protocol import finite


def bias_bound(eligible_count, max_bad_eligible, committee_size, beta, eta, drift=0):
    for name, value in (("eligible_count", eligible_count), ("max_bad_eligible", max_bad_eligible), ("committee_size", committee_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    n, f, k = eligible_count, max_bad_eligible, committee_size
    if not 1 <= k <= n or k % 2 != 1 or f > n:
        raise ValueError("Require odd 1 <= committee size <= eligible count, and F <= N")
    for name, value in (("beta", beta), ("eta", eta), ("drift", drift)):
        finite(value, name, upper=1)
    h = (k + 1) // 2
    p = 0.0
    for b in range(max(0, k - (n-f)), min(k, f) + 1):
        mass = math.comb(f, b) * math.comb(n-f, k-b) / math.comb(n, k)
        p += mass * (1 if b >= h else min(1.0, (k-b) * eta / (h-b)))
    return {"median_failure_bound": p, "bias_floor": min(1.0, beta + (1-beta)*p + drift),
            "status": "conditional_assumptions"}


def sample_committee(roster, producer_id, size, rng: random.Random):
    eligible = sorted(set(roster) - {producer_id})
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size % 2 != 1 or size > len(eligible):
        raise ValueError("Invalid odd committee size")
    # Seeded RNG is for reproducibility; no deployment unpredictability claim.
    return tuple(rng.sample(eligible, size))
