# Prior-Art Ledger — CFEDemands (`cfe`)

> Standing repo ledger (per the `prior-art-ledger` skill): the machinery,
> definitions, and invariants already in force, so a task neither reinvents what
> exists nor contradicts a local definition. Living + git-tracked — edit in place;
> the commit history is the journal. `§N` are the citations used in code comments
> and verification ("OK, anchored on §N"). For a specific task, restate it and its
> reuse decisions at the top of §1/§5.

**Search tier used:** ripgrep over the Org source of truth (`Empirics/*.org`) + the tangled `cfe/*.py`.

## §1 Repo, restated
CFEDemands (import `cfe`, pypi `CFEDemands`) estimates the **Constant Frisch
Elasticity (CFE)** demand system: from consumption-expenditure panels it recovers
household **marginal utility of expenditure λ** (welfare `w = -log λ`) and
good-specific **Frisch elasticities β**. It is the engine nearly every research repo
calls. Lower-level demand/curvature math is delegated to **`consumerdemands`** (the
`Demands` repo). Literate: the Org is the source of truth; `cfe/*.py` are gitignored
tangled artifacts.

## §2 Existing machinery — REUSE these, don't reinvent
Locations cite the Org source of truth (R.org = `Empirics/regression.org`,
E.org = `Empirics/cfe_estimation.org`); the tangled `cfe/*.py` mirror them.

| symbol | where | what it does | tested? |
|--------|-------|--------------|---------|
| `cfe.regression.Regression` | R.org:1218 | estimation object (construct from `y`,`d`; runs `prepare_data` in `__init__`) | `cfe/test/test_regression.py` |
| `Regression.get_beta()` | R.org:1332 | Frisch elasticities β (+ bootstrap SE) | test_regression |
| `Regression.get_w()` | R.org:1466 | welfare `w = -log λ` | test_regression |
| `Regression.relative_risk_aversion()` | R.org:1952 / bound 1971 | **R_λ(x) curvature**, returns a callable; delegates to `consumerdemands.demands.relative_risk_aversion` | not directly |
| `Regression.indirect_utility()` | R.org:1921 / bound 1969 | indirect utility (Marshallian/Frischian) via `consumerdemands` | not directly |
| `Regression.predicted_expenditures()` | R.org:1658 | predicted expenditures | test_regression |
| `Regression.graph_beta()` | R.org:1761 | canonical β plot | — |
| `w_var(e,beta,cov=...)` · `w_cov(e,beta)` | R.org:696 · 899 | welfare-uncertainty inference; `cov='hc2'` is leverage-corrected (HC0 biased low); `w_cov` → factored `WCov` for generated-regressor propagation | `cfe/test/test_w_var.py` |
| `cfe.read_pickle(fn)` · `Regression.to_pickle(fn)` | R.org:1980 · 1288 | (de)serialize a fitted result; on-disk artifacts are **`.rgsn`** | — |
| `cfe.dgp` (`prices`,`expenditures`,`geometric_brownian`) | E.org | synthetic test-data generator (pandas; post-xarray) | `cfe/stochastic_test/` |

Curvature / prudence / RRA and all demand evaluation **live in `consumerdemands`**
(`../Demands/consumerdemands/{_core,frischian,marshallian}.py`); `cfe` wraps them.
Do **not** hand-roll R_λ / CV² / share-weighted curvature from primitives.

## §3 Definitions & conventions in force
- **λ** = marginal utility of expenditure; **w = -log λ** (welfare; bigger = better / less scarce).
- **β_j are FRISCH elasticities, NOT income elasticities.** Identified up to scale; a single β_j above/below 1 is **not** meaningful in isolation — only the **dispersion** across goods is interpretable.
- **Indices:** i households · t periods · m markets · j goods · k characteristics; cell `c=(i,t,m)`.
- **Results are `.rgsn` files, read with `cfe.read_pickle`, never raw `pickle.load`.** Don't give a results file a `.pickle` extension.
- **Literate:** edit the `.org`, then `make tangle`; `cfe/*.py` and `cfe/test/*.py` are gitignored build artifacts — editing them directly is wrong (`make clean` deletes them).
- Decomposition `y = pi_tm + A(r)_tmj + beta_j w_itm + gamma d + e`, with **A(r) ⊥ beta by construction** (welfare already orthogonal to relative prices).
- Default branch **`master`**; `gh` Projects-classic GraphQL is broken → use `gh api` REST.

## §4 Invariants & assumptions — the landmines
- **Estimate λ from FOOD ITEMS ONLY** (non-food has different recall/error variance — matters for the factor-analytic inference). Estimate **one** demand system across panels so λ's are commensurable.
- **`prepare_data` inclusion:** drops goods with `count() <= min_obs` (30); keeps households with items `> min_prop_items * n_goods` (0.1). A **categorical index explodes dimensionality → empty join** (`assert ... "Join of y & d is empty."`): **coerce categorical indices to str/int upstream.** `alltm` defaults `True` in `__init__`, `False` in the bare `prepare_data`.
- **Do NOT re-add global-scale "rectification" (freeing `delta != 1`).** Investigated and rejected (issue #5 / PR #8): `estimate_pi` imposes `delta=1`; freeing it is a generated-regressor problem whose attenuation bias exceeds the effect.
- **Validating an SE for an estimated (generated) regressor needs a coverage Monte Carlo**, not a delta-method width check — correct first-order spread can coexist with ~0% coverage. This sank the delta-rectification prototype.

## §5 Reuse decision (standing guidance)
- Curvature / RRA / prudence / indirect utility → `Regression.relative_risk_aversion()` · `.indirect_utility()` (→ `consumerdemands`). **Never** hand-rolled.
- Read/write fitted results → `cfe.read_pickle` · `to_pickle` (`.rgsn`).
- Demand evaluation → `consumerdemands` via `cfe.demands`.
- Welfare-uncertainty / generated-regressor SEs → `w_var(cov='hc2')` · `w_cov`.

## §6 Open questions / known debt
- Two `FIXME`s in `cfe_estimation.org` (dividing by a random variable; precision-weighted cross-market mean) — long-standing, not urgent.
- CI still on bitbucket-pipelines (rest of the ecosystem is on GitHub Actions).
