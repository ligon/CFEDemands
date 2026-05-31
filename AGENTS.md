# CFEDemands — notes for agents

## Literate programming: edit the Org, not the `.py`
- **Source of truth is the Org files**, tangled to Python:
  - `Empirics/cfe_estimation.org`, `Empirics/regression.org`, `Files/input_files.org`
- Tangle with `./tangle.sh` (see `Makefile`). The tangled outputs —
  `cfe/*.py`, `cfe/test/*.py`, `cfe/stochastic_test/*.py` — are **gitignored
  build artifacts**. Never hand-edit them; edit the `.org` and re-tangle.
  `make clean` deletes them.
- **Tests are tangled too.** e.g. `cfe/test/test_w_var.py` comes from an Org src
  block in `regression.org` marked `:tangle ../cfe/test/test_w_var.py`. To add or
  change a test, edit that Org block.
- Loop: edit `.org` → `(cd Empirics; ../tangle.sh regression.org)` →
  `poetry run pytest cfe/test/`. Or `make tangle test`.
- See the `orgmode` skill for Org conventions.

## Environment
- `poetry` project; if the venv is empty run `poetry install` (deps include
  `ConsumerDemands`). On HPC, `module avail` may surface a usable Python.
- Tests: `poetry run pytest cfe/test/`. Build synthetic test data with `cfe.dgp`
  (`prices`, `expenditures`, `geometric_brownian`).

## Git / GitHub
- **Default branch is `master`** (not `main`).
- `gh pr edit` / `gh issue view --comments` can fail on a deprecated
  "Projects (classic)" GraphQL field. Use REST instead:
  `gh api -X POST repos/ligon/CFEDemands/pulls ...`,
  `gh api -X PATCH .../pulls/N ...`, `gh api .../issues/N/comments ...`.

## The CFE model (enough to navigate `regression.org`)
- `w_itm = -log(lambda_itm)` welfare weights; rank-1 factor of residualized log
  expenditures `X_cj = beta_j w_c + eps`, cell `c=(i,t,m)`, good `j`; `beta_j` =
  Frisch elasticities; `w_c = (beta'X_c)/(beta'beta)` (within-cell projection
  over the observed goods).
- Decomposition `y = pi_tm + A(r)_tmj + beta_j w_itm + gamma*d + e`. By
  construction `A(r) ⊥ beta` — the welfare term is already orthogonal to
  relative prices. Period welfare `w_bar_t` is recoverable from the (t,m,j)
  fixed effects (the `beta`-loading); it needs `beta` to vary across goods.
- `w`-uncertainty inference: `w_var(e, beta, cov=...)` — HC0 default is biased
  low (projection leverage); `cov='hc2'` is the leverage-corrected one — and the
  factored `w_cov(e, beta) -> WCov` (`.var/.matvec/.todense`) for propagating
  `w`-uncertainty into downstream generated-regressor problems. `min_goods=1`
  opts single-good cells into the homoskedastic forms.
- **Do not re-add global-scale "rectification" (freeing `delta != 1`).** It was
  investigated and rejected: `estimate_pi` correctly imposes `delta=1`; the
  welfare/price orthogonality is already built in; freeing `delta` is a
  generated-regressor problem with an attenuation bias larger than the effect.
  See issue #5 and the PR #8 history.

## A methodological gotcha worth keeping
- Validating a standard error for an estimator that uses an *estimated*
  regressor (a generated regressor) needs a **coverage Monte Carlo**, not just a
  delta-method width check. An estimator can have the correct first-order
  *spread* (matches `std/eps` as `eps -> 0`) yet ~0% interval coverage, because
  the *point* estimate carries an O(noise-variance) attenuation bias that does
  not shrink relative to the se as N grows. This is exactly what sank the
  `delta`-rectification prototype.
