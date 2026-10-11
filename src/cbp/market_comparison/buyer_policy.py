"""Registered-price buyer policy over published lower bounds only.

P_i(d)=B(d)*kappa_i is a quote, not verified execution cost or auction payment.
Strict eligibility uses L_i(d)>=rho(d); optional ranking uses L_i(d)/P_i(d).
This postprocessing neither modifies a certificate nor promises incentives.
"""
from collections.abc import Mapping
from ..bounded_score.protocol import exact


def strict_offers(publications, *, level, rho, base_price, kappas):
    """Return all eligible offers in exact score order, with ID tie breaking.

    A registered look is required. Fail-closed publications never qualify.
    Kappas must cover the roster exactly and remain fixed by the caller's plan.
    Inputs intentionally exclude outcome truths, private costs, or fault labels.
    """
    if level not in ('A', 'B', 'C'):
        raise ValueError('Require cumulative grade A, B, or C')
    rho, base_price = exact(rho), exact(base_price)
    if not 0 < rho <= 1 or base_price <= 0:
        raise ValueError('Require rho in (0,1] and positive base price')
    if not isinstance(kappas, Mapping) or set(kappas) != set(publications):
        raise ValueError('Kappas must match the entire candidate roster')
    prices = {}
    for agent, kappa in kappas.items():
        kappa = exact(kappa)
        if kappa <= 0:
            raise ValueError('All registered kappa values must be positive')
        prices[agent] = base_price * kappa
    offers = []
    for agent, publication in publications.items():
        if publication.get('status') == 'model_inconsistent':
            continue
        bound = publication['cumulative_bounds'][level]
        lower, upper = exact(bound['lower']), exact(bound['upper'])
        if not 0 <= lower <= upper <= 1:
            raise ValueError('Invalid cumulative interval')
        if bound.get('at_look') is None or bound.get('status') == 'model_inconsistent':
            continue
        if lower >= rho:
            offers.append({'agent_id': agent, 'level': level, 'lower': lower,
                           'quote': prices[agent], 'rank_score': lower / prices[agent]})
    return sorted(offers, key=lambda offer: (-offer['rank_score'], offer['agent_id']))
