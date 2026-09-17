# Prior-Art Ledger — CFEDemands (`cfe`)

> Standing repo ledger (per the `prior-art-ledger` skill): the machinery,
> definitions, and invariants already in force, so a task neither reinvents what
> exists nor contradicts a local definition. Living + git-tracked — edit in place;
> the commit history is the journal. `§N` are the citations used in code comments
> and verification ("OK, anchored on §N"). For a specific task, restate it and its
> reuse decisions at the top of §1/§5.

**Search tier used:** ripgrep over the Org source of truth (`Empirics/*.org`) + the tangled `cfe/*.py`.

## §1 Repo, restated
Current task (2026-09-16, branch `feature/prepare-data-scoring`): repair
`prepare_data`'s ineffective early support filter and irreversible household
deletion, expose preparation exclusions, and score additional households using
the fitted CFE parameters without refitting. Preserve the existing `get_w`
estimator, normalization, and the complete-covariance goods-selection heuristic.
Design context: `../LSMS_Library/SkunkWorks/cfe_aggregation.org`, especially
"Preparation and scoring in CFEDemands". Baseline after tangling: 42 tests pass.
Implemented and reviewed on 2026-09-16: 67 tests pass (25 new), with the same
pre-existing divide-by-zero log warning in `test_artificial_data`. Reproduce:

```sh
(cd Empirics; ../tangle.sh regression.org)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD" .venv/bin/python -m pytest cfe/test/ -q
```

Checked on Python 3.11 / pandas 3.0.0 / NumPy 2.4.1. A separate probe exercised
the legacy pandas `stack` fallback; this was not a full older-pandas suite.

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
| `cfe.regression.Regression` | R.org:1285 | estimation object (construct from `y`,`d`; runs `prepare_data` in `__init__`) | `cfe/test/test_regression.py` |
| `Regression.get_beta()` | R.org:1404 | Frisch elasticities β (+ bootstrap SE) | test_regression |
| `Regression.get_w()` | R.org:1538 | welfare `w = -log λ` | test_regression |
| `Regression.relative_risk_aversion()` | R.org:2116 / bound 2135 | **R_λ(x) curvature**, returns a callable; delegates to `consumerdemands.demands.relative_risk_aversion` | not directly |
| `Regression.indirect_utility()` | R.org:2066 / bound 2133 | indirect utility (Marshallian/Frischian) via `consumerdemands` | not directly |
| `Regression.predicted_expenditures()` | R.org:1822 | predicted expenditures | test_regression |
| `Regression.graph_beta()` | R.org:1925 | canonical β plot | — |
| `w_var(e,beta,cov=...)` · `w_cov(e,beta)` | R.org:698 · 901 | welfare-uncertainty inference; `cov='hc2'` is leverage-corrected (HC0 biased low); `w_cov` → factored `WCov` for generated-regressor propagation | `cfe/test/test_w_var.py` |
| `cfe.read_pickle(fn)` · `Regression.to_pickle(fn)` | R.org:2144 · 1359 | (de)serialize a fitted result, including preparation diagnostics/settings; on-disk artifacts are **`.rgsn`** | `test_preparation_scoring.py`, including legacy dictionaries |
| `cfe.dgp` (`prices`,`expenditures`,`geometric_brownian`) | E.org | synthetic test-data generator (pandas; post-xarray) | `cfe/stochastic_test/` |
| `_prepare_inputs`, `prepare_data` | R.org:1030,1070 / `code:data_preparation` | normalizes input, sums duplicate expenditures stably in log space, matches characteristics, filters rows/goods, reports exclusions | `test_regression`, `test_preparation_scoring.py` |
| `drop_columns_wo_covariance` | E.org:278 | greedy complete-covariance support filter; admits marginal counts **equal** to `min_obs` | `test_drop_columns_wo_cov.py` |
| `Md_generator`, `Ed`, `Regression.get_gamma_d` | R.org:229,354,1511 | linear controls use fitting-sample centering and an overwritten `Constant` column; `gamma` retains coefficients | `test_regression`, frozen-control tests in `test_preparation_scoring.py` |
| `Mpi`, `estimate_w`, `Regression.get_w` | R.org:152,575,1538 | market centering then joint market intercept and household-score fit | `test_regression`, `test_w_var`, known-score and rescoring tests in `test_preparation_scoring.py` |
| `Regression.score_w` | R.org:1563 | scores additional households with fitted loadings, demographic transformation, and market intercept; defaults `min_goods` to the fit's own item-count cutoff; returns support diagnostics on request | `test_preparation_scoring.py` |
| `predict_y`, `Regression.get_predicted_log_expenditures` | R.org:999,1695 | predicts expenditures for fitted welfare; does not infer welfare for additional households | `test_regression` |

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
- **`prepare_data` inclusion:** the early marginal filter and covariance helper both admit `count() >= min_obs` (30). Keep households with items `> min_prop_items * n_goods` (0.1), reconsidering the original matched rows whenever retained goods change. `alltm` defaults `True` in `__init__`, `False` in bare `prepare_data`. Entirely unavailable rows remain in diagnostics but cannot introduce a required market. Group categorical indices with `observed=True`; unused categories must not expand the sample. A finite zero log remains observed; non-finite logs are unavailable.
- **`estimate_w` coefficient extraction:** its joint design concatenates `(t,m)` and `(i,t,m)` columns. Pandas can truncate the mixed labels, so extract household coefficients from the solver array by position and attach `B.columns`. The solver and estimator are unchanged; a multiple-market test recovers known scores directly.
- **Scoring is conditional on a fitted model.** Preserve beta scale/sign, demographic coefficients and their training centering, and the fitted market intercept. The `Ar` returned by `estimate_w` is a *post-fit* mean of residuals by `(t,m,j)`, not the joint `(t,m)` intercept. For the existing estimator, the latter plus the `Mpi` centering is recovered from the training mean of `y - gamma_d - beta*w` by `(t,m)`. Subtracting `Ar` again can change training scores under missing goods.
- **Scoring inherits the fitting cutoff.** `prepare_data` keeps households with `count > min_prop_items*len(goods)`, so the equivalent count is `floor(min_prop_items*J)+1`; `score_w` defaults `min_goods` to exactly that, computed from the same floating-point product, and raises if the fit does not record `min_prop_items`. Scores below that cutoff are conditional point estimates with no precision attached: on Uganda, sd(w) is 0.70 in the fitting sample against 1.11/1.44/2.15 at three/two/one good. Do not restore `min_goods=1` as the default (issue #11); a per-cell scoring variance is the outstanding piece.
- **Initial scoring scope:** linear numeric demographic controls, existing fitted goods and market-periods. Unsupported control methods must fail explicitly, and unsupported observations must have reported reasons. The scoring data must not estimate or recenter any common parameter. Conditional support diagnostics do not claim to include parameter-estimation or partition-selection uncertainty.
- **Do NOT re-add global-scale "rectification" (freeing `delta != 1`).** Investigated and rejected (issue #5 / PR #8): `estimate_pi` imposes `delta=1`; freeing it is a generated-regressor problem whose attenuation bias exceeds the effect.
- **Validating an SE for an estimated (generated) regressor needs a coverage Monte Carlo**, not a delta-method width check — correct first-order spread can coexist with ~0% coverage. This sank the delta-rectification prototype.

## §5 Reuse decision (standing guidance)
- **Extend** `prepare_data`: shared input normalization, inclusive support boundary, monotone removal of goods with households reconsidered from the original matched input, optional structured diagnostics. Preserve the default two-value return.
- **Reuse** `drop_columns_wo_covariance`: no replacement maximum-clique or welfare-optimal search in this task. Its heuristic remains a statistical-policy choice.
- **New** `Regression.score_w`: the observed-good projection with *fixed* controls and market offsets. `estimate_w` jointly fits nuisance parameters and `predict_y` predicts the opposite direction, so neither is an out-of-sample scoring interface. Derive the offset from stored fitting observations and scores so existing `.rgsn` fits remain usable; do not change the fitting estimator to obtain a scorer.
- **Reuse** existing `w_var` / `w_cov` for their documented fitted-model inference; the first scoring interface reports point estimates and support/exclusion diagnostics. Broader scoring uncertainty belongs to the statistical evaluation work.
- Curvature / RRA / prudence / indirect utility → `Regression.relative_risk_aversion()` · `.indirect_utility()` (→ `consumerdemands`). **Never** hand-rolled.
- Read/write fitted results → `cfe.read_pickle` · `to_pickle` (`.rgsn`).
- Demand evaluation → `consumerdemands` via `cfe.demands`.
- Welfare-uncertainty / generated-regressor SEs → `w_var(cov='hc2')` · `w_cov`.

Verification: **OK (anchored on §2, §4, §5)**. Preparation reuses the greedy
covariance helper; scoring uses the existing fitted normalization and control
centering without refitting common parameters. No existing out-of-sample scoring
implementation was found in the formula/identifier search. Tests cover household
restoration, inclusive support, finite zero logs, stable duplicate sums, empty
input axes, training-score agreement, batch invariance, missing controls,
intercept-only controls, and `.rgsn` persistence. Synthetic scoring uses seed 451;
training agreement is checked at rtol=1e-8, atol=1e-9. No new uncertainty estimator
or coverage claim is introduced.

## §6 Open questions / known debt
- The semantic-curation proposal is at
  `../LSMS_Library/SkunkWorks/semantic_curation_proposal.org`.
  Selecting a statistically optimal fitting subset, generalized weighting,
  scoring unseen markets, and transferring categorical/K-means controls remain
  separate work; none are silently approximated by this scoring interface.
- Two `FIXME`s in `cfe_estimation.org` (dividing by a random variable; precision-weighted cross-market mean) — long-standing, not urgent.
- CI still on bitbucket-pipelines (rest of the ecosystem is on GitHub Actions).
