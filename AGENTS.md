# Project conventions

Use src/cbp for Python modules, tests for checks and small programmatic contract
fixtures, configs for inputs, exp/scripts for runnable experiment helpers, and
docs for methods/setup and compact metrics. Raw experimental records, generated
reports and execution logs belong only in ignored runs/. Historical documents
belong in ignored runs/archive/, not docs/history/. Local replay recordings in
tests/fixtures/ are ignored and excluded from source packages; tests must work
without them, using synthetic temporary inputs labeled as fixtures. Never overwrite or
mislabel archived evidence as a fresh run. No live API calls or real currency in
tests. Keep aq-llm-pivot as an optional thin adapter. Unknown usage/cost is null.

The active research direction is Capability Bidding Protocol with volunteer resource
cost for admission and zero peer-review money transfers. Formal task bids remain
free; modeled cost constraints belong only in each participant’s private bid policy.
The active single-agent market lists every certified candidate with its quote for user selection. It does not optimize budget, utility or agent composition; private costs are not verified. Do not claim
truthful incentives, unconditional quality certification, or universal provider
comparability. v1/v3 code, proofs and metrics are historical comparison material.
