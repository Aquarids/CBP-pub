"""Exact bounded-score counts by width-parameterized Boolean elimination.

This additive engine leaves ``IdentitySetBounds`` and all market/statistical
code unchanged.  A Boolean x_j says that reviewer j is excluded; exactly F
reviewers are excluded globally.  Factor tables enumerate only local Boolean
scopes, never all size-(m-F) candidate honest sets.

``max_width`` and ``max_states`` are optional conservative resource guards.
The latter bounds the sum of dense input, bucket-work, outgoing and final
cardinality-table capacities in one solve, not wall time or peak Python bytes.
A guard raises ``ExactWidthRefusal``; it never returns approximate endpoints or
mislabels a resource refusal as model inconsistency.  Feasibility is evaluated
lazily on ``model_inconsistent`` or ``count_bounds`` after an atomic addition.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType

from .protocol import IdentitySetBounds, exact, score


@dataclass(frozen=True)
class EliminationDiagnostics:
    """Reproducible plan and actual state counters for one exact optimization.

    Width is the largest separator size under the reported order.  The graph
    is the primal graph of *all* effective factors, including objective scopes.
    ``table_states_created`` counts dense input/outgoing/final table cells;
    ``feasible_states_created`` counts their reachable cells.  Work-buffer
    cells are included in the conservative estimate, not the actual counters.
    ``convolution_pairs`` counts reachable budget pairs considered in products.
    """
    objective: str
    factor_scopes: tuple
    elimination_order: tuple
    elimination_width: int
    estimated_total_states: int
    max_table_states: int
    simplified_monomials: int = 0
    constraint_factor_count: int = 0
    objective_factor_count: int = 0
    objective_multiplicity: int = 0
    objective_constant: int = 0
    table_states_created: int = 0
    feasible_states_created: int = 0
    convolution_pairs: int = 0
    status: str = 'planned'
    refusal_reason: object = None


class ExactWidthRefusal(RuntimeError):
    """Requested exact computation exceeded a configured conservative guard."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class _Spec:
    scope: tuple
    # None is a hard clause: forbid all-zero assignment; otherwise a monomial.
    weight: object = None


@dataclass(frozen=True)
class _Problem:
    specs: tuple
    constant: int = 0
    simplified: int = 0
    impossible: bool = False


@dataclass
class _Factor:
    scope: tuple
    table: dict


class ExactWidthBounds:
    """Drop-in count-bound interface, without the reference engine's H list.

    ``elimination_order``, if supplied, must contain every roster identity
    exactly once.  Otherwise deterministic min-fill chooses an order separately
    for each effective factor graph, breaking ties by degree then roster order.
    ``diagnostics`` exposes the latest solve of each objective.  ``plan`` exposes
    the same graph/width estimates without materializing any factor tables.
    """
    def __init__(self, roster, F, eta, *, elimination_order=None,
                 max_width=None, max_states=None):
        self.roster = tuple(roster)
        # Preserve the reference public-input validation and exact-number rules.
        if (len(set(self.roster)) != len(self.roster)
                or any(not isinstance(x, str) or not x for x in self.roster)):
            raise ValueError('Unique nonempty identity strings required')
        if type(F) is not int or F < 0 or len(self.roster) < 2 * F + 2:
            raise ValueError('Require m >= 2F+2 and integer F >= 0')
        self.F, self.eta = F, exact(eta)
        if not 0 <= self.eta <= 10:
            raise ValueError('eta must be in [0,10]')
        for name, limit in (('max_width', max_width), ('max_states', max_states)):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError(f'{name} must be a nonnegative integer or None')
        self.max_width, self.max_states = max_width, max_states
        self._index = {identity: i for i, identity in enumerate(self.roster)}
        if elimination_order is None:
            self._order = None
        else:
            order = tuple(elimination_order)
            if len(order) != len(self.roster) or set(order) != set(self.roster):
                raise ValueError('Elimination order must list every roster identity once')
            self._order = tuple(self._index[j] for j in order)
        self.rows = []
        self._rows_by_producer = {a: [] for a in self.roster}
        self._clauses = ()
        self._plan_cache = {}
        self._keys, self._artifacts, self._conflicts = set(), set(), set()
        self._inconsistent = None
        self._count_cache, self._diagnostics = {}, {}

    def normalize(self, row):
        """Use the identical immutable reference schema; construct no H list."""
        return IdentitySetBounds.normalize(self, row)

    def add_rows(self, rows):
        """Atomically normalize and add rows; retain malformed/missing scores."""
        normalized = [self.normalize(row) for row in rows]
        keys = [(row.producer, row.task_id) for row in normalized]
        artifacts = [row.artifact_id for row in normalized]
        if (len(set(keys)) != len(keys) or self._keys.intersection(keys)
                or len(set(artifacts)) != len(artifacts)
                or self._artifacts.intersection(artifacts)):
            raise ValueError('Duplicate sampled task or artifact')
        conflicts = set()
        for row in normalized:
            valid = sorted((value, self._index[j]) for j, value in row.reports.items()
                           if value is not None)
            tolerance = 2 * self.eta
            if not valid or valid[-1][0] - valid[0][0] <= tolerance:
                continue
            values = [value for value, _ in valid]
            # Enumerate only actual incompatible pairs after one sorted scan.
            for i, (r_u, u) in enumerate(valid):
                start = bisect_right(values, r_u + tolerance, lo=i + 1)
                for _, v in valid[start:]:
                    conflicts.add(tuple(sorted((u, v))))
        # No state is changed until the entire input batch has passed validation.
        self.rows.extend(normalized)
        for row in normalized:
            self._rows_by_producer[row.producer].append(row)
        self._keys.update(keys)
        self._artifacts.update(artifacts)
        self._conflicts.update(conflicts)
        self._clauses = tuple(_Spec(scope) for scope in sorted(self._conflicts))
        self._plan_cache.clear()
        self._inconsistent = None
        self._count_cache.clear()
        self._diagnostics.clear()

    @property
    def diagnostics(self):
        return MappingProxyType(dict(self._diagnostics))

    def _problem(self, producer=None, cutoff=None, objective='feasibility'):
        # F=0 fixes all variables to zero.  Its clauses are either absent or
        # immediately impossible; do not construct a spurious dense graph.
        clauses = () if self.F == 0 else self._clauses
        impossible = self.F == 0 and bool(self._conflicts)
        if objective == 'feasibility':
            return _Problem(clauses, impossible=impossible)
        rows = self._rows_by_producer[producer]
        if cutoff == 0:
            return _Problem(clauses, len(rows), impossible=impossible)
        monomials = Counter()
        simplified = 0
        constant = len(rows) if objective == 'lower' else 0
        sign = -1 if objective == 'lower' else 1
        threshold = cutoff + self.eta if objective == 'lower' else cutoff - self.eta
        for row in rows:
            if objective == 'lower':
                scope = tuple(sorted(self._index[j] for j, r in row.reports.items()
                                     if r is not None and r >= threshold))
            else:
                scope = tuple(sorted(self._index[j] for j, r in row.reports.items()
                                     if r is not None and r < threshold))
            # A product of more than F distinct exclusion bits is always zero
            # on the exact-F slice.  Apply this BEFORE graph construction.
            if len(scope) > self.F:
                simplified += 1
            elif not scope:  # The empty product is exactly one.
                constant += sign
            else:
                monomials[scope] += sign
        specs = clauses + tuple(_Spec(scope, weight)
                                for scope, weight in sorted(monomials.items()))
        return _Problem(specs, constant, simplified, impossible)

    def _plan(self, problem, objective):
        cache_key = problem, objective
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]
        graph = {v: set() for v in range(len(self.roster))}
        for spec in problem.specs:
            for u, v in combinations(spec.scope, 2):
                graph[u].add(v)
                graph[v].add(u)
        order, buckets = [], []
        width = 0
        for step in range(len(self.roster)):
            if self._order is None:
                def priority(v):
                    neighbors = graph[v]
                    fill = sum(w not in graph[u]
                               for u, w in combinations(sorted(neighbors), 2))
                    return fill, len(neighbors), v
                variable = min(graph, key=priority)
            else:
                variable = self._order[step]
            neighbors = graph[variable]
            scope = tuple(sorted(neighbors | {variable}))
            buckets.append(scope)
            width = max(width, len(neighbors))
            for u, v in combinations(sorted(neighbors), 2):
                graph[u].add(v)
                graph[v].add(u)
            for neighbor in neighbors:
                graph[neighbor].remove(variable)
            del graph[variable]
            order.append(variable)
        capacities = [(1 << len(spec.scope)) * (self.F + 1)
                      for spec in problem.specs]
        for scope in buckets:
            capacities.extend(((1 << len(scope)) * (self.F + 1),
                               (1 << (len(scope) - 1)) * (self.F + 1)))
        capacities.append(self.F + 1)
        diagnostics = EliminationDiagnostics(
            objective=objective,
            factor_scopes=tuple(tuple(self.roster[v] for v in spec.scope)
                                for spec in problem.specs),
            elimination_order=tuple(self.roster[v] for v in order),
            elimination_width=width,
            estimated_total_states=sum(capacities),
            max_table_states=max(capacities),
            simplified_monomials=problem.simplified,
            constraint_factor_count=sum(spec.weight is None for spec in problem.specs),
            objective_factor_count=sum(spec.weight is not None for spec in problem.specs),
            objective_multiplicity=sum(abs(spec.weight) for spec in problem.specs
                                       if spec.weight is not None),
            objective_constant=problem.constant)
        result = diagnostics, tuple(order)
        self._plan_cache[cache_key] = result
        return result

    def plan(self, producer=None, cutoff=None):
        """Return immutable static diagnostics without solving or enforcing guards.

        With no arguments, plan feasibility.  With a producer and cutoff, plan
        feasibility and both count objectives, including simplified scopes.
        """
        if producer is None and cutoff is None:
            names = ('feasibility',)
        else:
            if producer not in self.roster:
                raise ValueError('Unknown producer')
            cutoff = score(cutoff)
            names = ('feasibility', 'lower', 'upper')
        return MappingProxyType({name: self._plan(
            self._problem(producer, cutoff, name), name)[0] for name in names})

    def _solve(self, problem, objective, maximize=False):
        diagnostics, order = self._plan(problem, objective)
        reason = None
        if self.max_width is not None and diagnostics.elimination_width > self.max_width:
            reason = (f'Elimination width {diagnostics.elimination_width} exceeds '
                      f'max_width={self.max_width}')
        if (self.max_states is not None
                and diagnostics.estimated_total_states > self.max_states):
            reason = (f'Conservative state bound {diagnostics.estimated_total_states} '
                      f'exceeds max_states={self.max_states}')
        if reason is not None:
            diagnostics = replace(diagnostics, status='refused', refusal_reason=reason)
            self._diagnostics[objective] = diagnostics
            raise ExactWidthRefusal(reason, diagnostics)
        if problem.impossible:
            self._diagnostics[objective] = replace(diagnostics, status='infeasible')
            return None
        F = self.F
        zero = (0,) + (None,) * F
        created = reachable = pairs = 0

        def record(table):
            nonlocal created, reachable
            created += len(table) * (F + 1)
            reachable += sum(value is not None for vector in table.values()
                             for value in vector)

        def better(value, previous):
            return previous is None or (value > previous if maximize else value < previous)

        def convolve(left, right):
            nonlocal pairs
            result = [None] * (F + 1)
            for a, lhs in enumerate(left):
                if lhs is None:
                    continue
                for b in range(F - a + 1):
                    rhs = right[b]
                    if rhs is None:
                        continue
                    pairs += 1
                    value = lhs + rhs
                    if better(value, result[a + b]):
                        result[a + b] = value
            return tuple(result)

        rank = {variable: i for i, variable in enumerate(order)}
        buckets = {variable: [] for variable in order}
        scalar_factors = []

        def schedule(factor):
            if factor.scope:
                first = min(factor.scope, key=rank.__getitem__)
                buckets[first].append(factor)
            else:
                scalar_factors.append(factor)

        for spec in problem.specs:
            table = {}
            for assignment in range(1 << len(spec.scope)):
                # Boundary exclusion bits also consume the global budget,
                # although they are charged only when eliminated later.
                if assignment.bit_count() > F:
                    continue
                if spec.weight is None and assignment == 0:
                    continue
                value = (spec.weight if assignment == (1 << len(spec.scope)) - 1
                         and spec.weight is not None else 0)
                table[assignment] = (value,) + (None,) * F
            record(table)
            schedule(_Factor(spec.scope, table))

        for variable in order:
            bucket = buckets.pop(variable)
            scope = tuple(sorted({variable}.union(*(set(f.scope) for f in bucket))))
            separator = tuple(v for v in scope if v != variable)
            positions = {v: i for i, v in enumerate(scope)}
            projections = [tuple(positions[v] for v in factor.scope) for factor in bucket]
            separator_positions = tuple(positions[v] for v in separator)
            result = {}
            for assignment in range(1 << len(scope)):
                boundary_count = assignment.bit_count()
                if boundary_count > F:
                    continue
                vector = zero
                for factor, projection in zip(bucket, projections):
                    key = sum(((assignment >> source) & 1) << target
                              for target, source in enumerate(projection))
                    incoming = factor.table.get(key)
                    if incoming is None:
                        vector = None
                        break
                    vector = convolve(vector, incoming)
                    if all(value is None for value in vector):
                        vector = None
                        break
                if vector is None:
                    continue
                bit = (assignment >> positions[variable]) & 1
                key = sum(((assignment >> source) & 1) << target
                          for target, source in enumerate(separator_positions))
                outgoing = result.setdefault(key, [None] * (F + 1))
                # k counts already-eliminated bits, disjoint across the bucket.
                # The still-live boundary bits must leave enough remaining budget.
                for k in range(F - boundary_count + 1):
                    value = vector[k]
                    if value is not None and better(value, outgoing[k + bit]):
                        outgoing[k + bit] = value
            table = {key: tuple(values) for key, values in result.items()
                     if any(value is not None for value in values)}
            record(table)
            schedule(_Factor(separator, table))

        # Each disconnected component returns a scalar cardinality polynomial.
        # Convolve them: enforcing F separately in each component is incorrect.
        final = zero
        for factor in scalar_factors:
            assert not factor.scope
            incoming = factor.table.get(0)
            if incoming is None:
                final = (None,) * (F + 1)
                break
            final = convolve(final, incoming)
        record({0: final})
        value = final[F]
        self._diagnostics[objective] = replace(
            diagnostics, table_states_created=created,
            feasible_states_created=reachable, convolution_pairs=pairs,
            status='infeasible' if value is None else 'exact')
        return None if value is None else value + problem.constant

    @property
    def model_inconsistent(self):
        if self._inconsistent is None:
            self._inconsistent = self._solve(
                self._problem(), 'feasibility') is None
        return self._inconsistent

    def count_bounds(self, producer, cutoff):
        if producer not in self.roster:
            raise ValueError('Unknown producer')
        cutoff = score(cutoff)
        if self.model_inconsistent:
            return None
        key = producer, cutoff
        if key not in self._count_cache:
            lower = self._solve(self._problem(producer, cutoff, 'lower'), 'lower')
            upper = self._solve(self._problem(producer, cutoff, 'upper'),
                                'upper', maximize=True)
            # The hard constraints are identical in all three optimizations.
            assert lower is not None and upper is not None
            self._count_cache[key] = lower, upper
        return self._count_cache[key]
