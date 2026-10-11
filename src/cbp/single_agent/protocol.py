"""Conditional, anytime single-agent certificates. No budget allocation or truth oracle.

Registry and executor are trust boundaries. The included registry is a mock, not
an on-chain implementation. Calibration and stability are supplied assumptions.
"""
from dataclasses import dataclass, asdict
import math
import random
from statistics import median
from typing import Protocol
from ..admission.committees import bias_bound, sample_committee
from ..admission.protocol import finite


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class ConfigRef:
    chain_id: str
    address: str
    version: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            nonempty(value, key)


@dataclass(frozen=True)
class ServiceConfig:
    ref: ConfigRef
    agent_id: str
    api: str
    api_version: str
    model_version: str
    system_prompt: str
    task_prompt: str
    tools: tuple[str, ...]
    input_schema: str
    output_schema: str
    task_domain: str
    distribution_version: str
    reset_rule: str
    resource_limit: int
    sampling_rule: str
    retry_rule: str
    failure_rule: str
    evaluation_rule: str
    target: str
    threshold: float
    parameters: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if not isinstance(self.ref, ConfigRef):
            raise ValueError("A versioned registry reference is required")
        for key, value in asdict(self).items():
            if key not in {"ref", "tools", "resource_limit", "threshold", "parameters"}:
                nonempty(value, key)
        finite(self.threshold, "threshold", upper=1)
        if self.target not in {"expected_quality", "success_probability"}:
            raise ValueError("Unknown target")
        if isinstance(self.resource_limit, bool) or not isinstance(self.resource_limit, int) or self.resource_limit < 1:
            raise ValueError("Positive integer resource limit required")
        if not isinstance(self.tools, tuple) or any(not isinstance(t, str) or not t for t in self.tools):
            raise ValueError("Tools must be an immutable tuple of versioned descriptions")
        if not isinstance(self.parameters, tuple) or any(not isinstance(p, tuple) or len(p) != 2 or any(not isinstance(v, str) or not v for v in p) for p in self.parameters):
            raise ValueError("Parameters must be immutable string pairs")
        keys = [p[0] for p in self.parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate parameter")
        if any(any(word in k.lower() for word in ("secret", "password", "token", "api_key", "credential")) for k in keys):
            raise ValueError("Credentials must never be stored in public configuration")


@dataclass(frozen=True)
class Quote:
    version: str
    amount: float
    currency: str
    scope: str

    def __post_init__(self):
        finite(self.amount, "quote amount")
        for name in ("version", "currency", "scope"):
            nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class Assumptions:
    max_bad_eligible: int
    beta: float
    eta: float
    drift: float
    calibration_basis: str
    identity_period: str
    stability_basis: str

    def __post_init__(self):
        if isinstance(self.max_bad_eligible, bool) or not isinstance(self.max_bad_eligible, int) or self.max_bad_eligible < 0:
            raise ValueError("F must be a nonnegative integer")
        for name in ("beta", "eta", "drift"):
            finite(getattr(self, name), name, upper=1)
        for name in ("calibration_basis", "identity_period", "stability_basis"):
            nonempty(getattr(self, name), name)


class Registry(Protocol):
    """A real adapter must resolve immutable historical contents, never 'latest'."""
    def resolve(self, ref: ConfigRef) -> ServiceConfig: ...


class InMemoryRegistry:
    kind = "in_memory_mock_not_onchain"

    def __init__(self, configs):
        self._configs = {}
        for config in configs:
            if config.ref in self._configs:
                raise ValueError("Duplicate immutable reference")
            self._configs[config.ref] = config

    def resolve(self, ref):
        return self._configs[ref]


@dataclass(frozen=True)
class Output:
    artifact_id: str
    value: object
    resource_units: float

    def __post_init__(self):
        nonempty(self.artifact_id, "artifact_id")
        finite(self.resource_units, "resource_units")


class Executor(Protocol):
    """Trusted implementation follows every frozen field; API honesty is assumed.

    No probe/formal flag is supplied. Deployment must also prevent side channels.
    """
    def invoke(self, config: ServiceConfig, task: object) -> Output: ...


@dataclass(frozen=True)
class Candidate:
    config: ServiceConfig
    quote: Quote
    assumptions: Assumptions


class Market:
    def __init__(self, candidates, roster, registry: Registry, executor: Executor,
                 *, delta, max_probes, seed, committee_size=None):
        self.candidates = tuple(candidates)
        self.roster = tuple(roster)
        if not self.candidates or len({c.config.agent_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Preregister unique single-agent candidates")
        if len(set(self.roster)) != len(self.roster) or any(not isinstance(r, str) or not r for r in self.roster):
            raise ValueError("Freeze unique reviewer identities")
        finite(delta, "delta", upper=1)
        if not 0 < delta < 1:
            raise ValueError("Require 0 < delta < 1")
        if isinstance(max_probes, bool) or not isinstance(max_probes, int) or max_probes < 1:
            raise ValueError("Positive probe cap required")
        self.delta, self.max_probes = delta, max_probes
        self.registry, self.executor = registry, executor
        self._rng = random.Random(seed)  # Synthetic reproducibility, not deployment security.
        self._scores = {c.config.agent_id: [] for c in self.candidates}
        self._terminal, self._artifacts, self._events = {}, set(), []
        self._bounds, self._sizes = {}, {}
        for c in self.candidates:
            aid = c.config.agent_id
            if aid not in self.roster:
                raise ValueError("Producer must be in the frozen participant roster")
            n = len(self.roster) - 1
            if n <= 2 * c.assumptions.max_bad_eligible:
                raise ValueError("Current strict-majority scope requires n-1 > 2F")
            k = committee_size if committee_size is not None else n if n % 2 else n - 1
            self._bounds[aid] = bias_bound(n, c.assumptions.max_bad_eligible, k,
                                         c.assumptions.beta, c.assumptions.eta, c.assumptions.drift)
            self._sizes[aid] = k
            self._resolve(c)

    @property
    def events(self):
        import copy
        return copy.deepcopy(self._events)

    def _candidate(self, agent_id):
        return next(c for c in self.candidates if c.config.agent_id == agent_id)

    def _resolve(self, candidate):
        resolved = self.registry.resolve(candidate.config.ref)
        if resolved != candidate.config:
            raise ValueError("Registry contents or version changed; recertification required")
        return resolved

    def summary(self, agent_id):
        c = self._candidate(agent_id)
        scores = self._scores[agent_id]
        m = len(scores)
        rho = self._bounds[agent_id]["bias_floor"]
        radius = rho + math.sqrt(math.log(2 * len(self.candidates) * m * (m + 1) / self.delta) / (2 * m)) if m else None
        mean = sum(scores) / m if m else None
        lower, upper = (max(0., mean-radius), min(1., mean+radius)) if m else (0., 1.)
        status = "certified" if m and lower >= c.config.threshold else "below_threshold" if upper < c.config.threshold else "inconclusive" if m >= self.max_probes else "pending"
        return {"agent_id": agent_id, "n": m, "mean": mean, "rho": rho, "radius": radius,
                "lower": lower, "upper": upper, "status": self._terminal.get(agent_id, status)}

    def probe(self, agent_id, task, reviewer):
        """One indivisible trial; refusal/invalid/missing report terminates the unit.

        reviewer(identity, frozen_config, output, public_history) returns a score
        or None. It must itself account for timeouts; asynchronous transports are
        not provided. Calibration applies to potential complete reports.
        """
        c = self._candidate(agent_id)
        if self.summary(agent_id)["status"] != "pending":
            raise ValueError("Candidate is terminal or already resolved")
        event = {"event_version": "single-agent-v1", "trial": len(self._scores[agent_id])+1,
                 "agent_id": agent_id, "config_ref": asdict(c.config.ref),
                 "distribution_version": c.config.distribution_version,
                 "identity_period": c.assumptions.identity_period, "reports": {},
                 "resource_units": 0., "status": "started"}
        try:
            config = self._resolve(c)
            output = self.executor.invoke(config, task)
            if not isinstance(output, Output):
                raise ValueError("Executor returned invalid output")
            event["resource_units"] = output.resource_units
            if output.artifact_id in self._artifacts:
                raise ValueError("Replayed artifact")
            self._artifacts.add(output.artifact_id)
            event["artifact_id"] = output.artifact_id
            committee = sample_committee(self.roster, agent_id, self._sizes[agent_id], self._rng)
            event["committee"] = committee
            history = tuple((e["agent_id"], e["trial"], e["status"], tuple(e["reports"].items())) for e in self._events)
            for identity in committee:
                score = reviewer(identity, config, output, history)
                if score is None:
                    raise ValueError("Missing or refused report")
                finite(score, "score", upper=1)
                event["reports"][identity] = float(score)
            self._scores[agent_id].append(float(median(event["reports"].values())))
            event["status"] = "completed"
        except Exception as exc:
            self._terminal[agent_id] = "abstained"
            event["status"] = "abstained"
            event["reason"] = f"{type(exc).__name__}: {exc}"
        self._events.append(event)
        return self.summary(agent_id)

    def certified_list(self):
        results = []
        for c in self.candidates:
            s = self.summary(c.config.agent_id)
            if s["status"] != "certified":
                continue
            try:
                self._resolve(c)
            except Exception:
                self._terminal[c.config.agent_id] = "configuration_unavailable"
                continue
            results.append({**s, "certificate_version": "single-agent-v1", "config": asdict(c.config),
                            "quote": asdict(c.quote), "assumptions": asdict(c.assumptions),
                            "registered_candidates": len(self.candidates), "delta": self.delta,
                            "roster": self.roster, "committee_size": self._sizes[c.config.agent_id],
                            "guarantee": "conditional_simultaneous_anytime_soundness"})
        return results

    def execute_selected(self, agent_id, task):
        """Selection is supplied by the user; this method never selects or pays."""
        if agent_id not in {r["agent_id"] for r in self.certified_list()}:
            raise ValueError("User selection lacks a valid current certificate")
        return self.executor.invoke(self._resolve(self._candidate(agent_id)), task)
