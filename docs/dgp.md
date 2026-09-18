# Synthetic Pricing Data-Generating Process

**Status:** specification frozen for calibration; constants pending
**Scope:** customer population, pricing policy, promo assignment, randomized arm, conversion
**Out of scope (deferred, see §13):** churn, post-treatment variables, cancellation text

---

## 1. Purpose

This document defines the data-generating process (DGP) for the hybrid subscription pricing project: which variables exist, how they cause one another, which equations produce them, and which are observable versus oracle-only.

The data are synthetic so that the true causal price effect is known. That makes it possible to score causal estimators against ground truth instead of against predictive fit alone.

Primary causal question:

> What is the causal effect of changing subscription price on conversion probability?

Historical prices are deliberately confounded by both measured and unmeasured variables. A randomized pricing arm provides a benchmark in which the assigned price is independent of all of them.

Nothing in this document is implemented in the production generator until the calibration protocol in §10 is complete and the constants in §14 are filled in.

---

## 2. Population and exogenous variables

These are drawn independently of everything else. Stating them matters: the centering convention in §6.5 and the interpretation of baseline conversion both depend on these distributions.

| Variable | Distribution |
|---|---|
| `segment` | uniform over $\{\texttt{price\_sensitive}, \texttt{price\_resilient}\}$ |
| `channel` | uniform over $\{\texttt{organic}, \texttt{affiliate}, \texttt{paid\_search}\}$ |
| `tier` | uniform over $\{\texttt{basic}, \texttt{premium}\}$ |
| `signup_week` | uniform over $0, \dots, n_{\mathrm{weeks}} - 1$ |
| $D_w$ (measured demand) | $\mathcal{N}(1, \sigma_D)$, clipped to $[0.75, 1.25]$, one draw per week |
| $H_w$ (hidden shock) | $\mathcal{N}(0, 1)$, one draw per week |
| $R_i$ (randomized-arm membership) | complete random assignment: exactly $\lceil r N \rceil$ users drawn uniformly without replacement, independent of all covariates |

Three consequences worth stating explicitly:

- Segment, channel, and tier are mutually independent. If any correlation is introduced later (for example, paid search skewing toward price-sensitive users), the centering convention in §6.5 must be revisited.
- $\sigma_H = 1.0$ is fixed, not calibrated. Only the products $\alpha_H \sigma_H$, $\gamma_H \sigma_H$, and $\delta_H \sigma_H$ affect the process, so fixing the scale makes the sensitivity parameters uniquely identified. Note the asymmetry of units: $D_w - 1$ has a standard deviation of roughly $\sigma_D$ (about $0.08$), while $H_w$ has one. The coefficients $\delta_D$ and $\delta_H$ are therefore not directly comparable.
- Arm assignment is **complete randomization**, not Bernoulli. The arm size is fixed, so membership is weakly dependent across users while remaining independent of every covariate.

---

## 3. Treatment, outcome, and true effects

**Treatment** — log price relative to the user's tier reference price:

$$
T_i = \log\left(\frac{P_i}{P^{\mathrm{ref}}_{t(i)}}\right)
$$

Using a tier-specific reference keeps the segment price coefficient interpretable across tiers.

**Outcome** — binary conversion:

$$
Y_i \sim \operatorname{Bernoulli}(p_i)
$$

**Structural heterogeneous treatment coefficients:**

| Segment | $\beta^{\mathrm{struct}}$ |
|---|---:|
| `price_sensitive` | $-2.0$ |
| `price_resilient` | $-0.8$ |

A more negative coefficient indicates greater sensitivity to price increases. The old segment name `premium` is retired because `premium` is also a subscription tier.

These are **logit-scale coefficients**, not elasticities. For a user with conversion probability $p$, the conversion elasticity with respect to price is

$$
\frac{\partial p}{\partial P}\cdot\frac{P}{p} = \beta\,(1-p),
$$

which varies across users. Any reported elasticity must state the value of $p$ at which it is evaluated.

### 3.1 Structural versus randomized-target coefficients

The configured $\beta_s$ values are **structural conditional coefficients** in the conversion DGP.

They are not necessarily identical to the coefficient obtained by fitting an observable logistic regression that omits an outcome predictor such as the hidden shock. Logistic regression coefficients are non-collapsible: even when an omitted variable is independent of randomized treatment, marginalizing over that variable can move the fitted treatment coefficient toward zero.

The DGP therefore distinguishes:

- $\beta_s^{\mathrm{struct}}$ — the coefficient used directly by the generator;
- $\beta_s^{\mathrm{rand}}$ — the population coefficient targeted by the observable randomized-arm regression after marginalizing over the hidden shock.

Calibration derives and records $\beta_s^{\mathrm{rand}}$ (§10.4) rather than assuming the two are numerically identical. Randomization removes confounding; it does not make logistic regression collapsible.

---

## 4. Causal graph

The observational pricing mechanism contains both measured and unmeasured common causes of price and conversion.

```text
segment ──────────────────────────────────────────────→ conversion

channel ──→ promo ──→ price ──────────────────────────→ conversion
   └──────────────────────────────────────────────────→ conversion

demand  ──→ base price ──→ price ─────────────────────→ conversion
   └──────→ promo
   └──────────────────────────────────────────────────→ conversion

shock   ──→ base price ──→ price ─────────────────────→ conversion   [HIDDEN]
   └──────→ promo
   └──────────────────────────────────────────────────→ conversion

tier    ──→ tier reference price ──→ price

R_i = 1 ──→ price   (bypasses base price, promo, demand, shock)
```

For users in the observational arm ($R_i = 0$), price is generated by the historical pricing mechanism and therefore inherits dependence on demand, channel, promotions, and the hidden shock.

For users in the randomized arm ($R_i = 1$), that mechanism is bypassed: the price multiplier is drawn independently of demand, channel, promotions, and the hidden shock, as specified in §6.4.

Backdoor paths in the observational arm:

| Path | Adjustable by the modeling path? |
|---|---|
| $\text{price} \leftarrow \text{demand} \rightarrow \text{conversion}$ | yes |
| $\text{price} \leftarrow \text{promo} \leftarrow \text{channel} \rightarrow \text{conversion}$ | yes |
| $\text{price} \leftarrow \text{shock} \rightarrow \text{conversion}$ | **no** — the shock is oracle-only |

---

## 5. Observable and oracle variables

**Observable** (may appear in the modeling database): `user_id`, `segment`, `channel`, `tier`, `signup_week`, `demand_index`, `observed_price`, `is_randomized`, `conversion_outcome`, `acquisition_cost`, `marginal_cost`.

**Observable but not a valid outcome-model control** (§5.1): `promo`, `promo_depth`.

**Oracle only** (separate database, §12): `hidden_shock`, true conversion probabilities, structural and randomized-target $\beta$ values, all configured DGP constants, and any future ground truth needed by the simulator or the decision-regret evaluation.

### 5.1 Why promo and promo depth are not outcome-model controls

`promo` and `promo_depth` are components of the treatment-assignment mechanism, not pre-treatment confounders that belong in the Day 2 outcome adjustment set.

In the DGP, promo assignment is influenced by several causes of treatment:

```text
demand ──→
channel ──→ promo
shock  ──→
```

`promo` is therefore a **collider** of measured demand, channel, and the hidden shock. Conditioning on it can induce associations among those causes, including with the unobserved shock — the very variable the modeling path cannot adjust for.

In addition, `promo_depth` directly determines the observed treatment once the weekly base price is known.

These variables remain available in the raw observable tables for diagnostics and propensity analysis, but the modeling-facing outcome view excludes them. Day 2 outcome models adjust for the pre-treatment common causes — demand and channel — not for the intermediate treatment-assignment variables.

---

## 6. Generating equations

### 6.1 Weekly base price

For week $w$ and tier $t$:

$$
P^{\mathrm{base}}_{w,t} = P^{\mathrm{ref}}_{t}\left[\,1 + \alpha_D\,(D_w - 1) + \alpha_H H_w\,\right]
$$

### 6.2 Promo assignment (observational arm only)

$$
\operatorname{logit} P(A_i = 1) = b_{C(i)} - \gamma_D\,(D_{w(i)} - 1) - \gamma_H H_{w(i)}
$$

Weak demand and negative shocks increase promotional activity. The channel intercepts $b_C$ set the baseline propensity.

### 6.3 Promo depth

Conditional on $A_i = 1$, depth $d_i$ is drawn uniformly from $\{0.05, 0.10, 0.15\}$, independent of everything else; otherwise $d_i = 0$.

$$
P_i = P^{\mathrm{base}}_{w(i),\,t(i)}\,(1 - d_i)
$$

This user-level draw is what creates overlap: two users in the same week, channel, and tier can face different prices.

### 6.4 Randomized arm

$$
P_i = P^{\mathrm{ref}}_{t(i)} M_i, \qquad M_i \sim \operatorname{Uniform}\{0.85,\ 0.925,\ 1.00,\ 1.075,\ 1.15\}
$$

The multiplier $M_i$ is drawn independently of demand, shock, channel, week, segment, and tier.

Raw price depends on tier by construction, since $P_i = P^{\mathrm{ref}}_{t(i)} M_i$. The quantity that must be independent of covariates is the multiplier — equivalently $\log(P_i / P^{\mathrm{ref}}_{t(i)})$ — not the raw price.

Randomized users have `promo = false` and `promo_depth = 0` by construction. These are sentinel values, not behavior, and §5.1 already bars both fields from models.

### 6.5 Conversion

$$
\operatorname{logit}(p_i) = \operatorname{logit}\!\left(\pi_{s(i)}\right)
+ \beta^{\mathrm{struct}}_{s(i)} \log\!\left(\frac{P_i}{P^{\mathrm{ref}}_{t(i)}}\right)
+ \delta_D\,(D_{w(i)} - 1)
+ \delta_H H_{w(i)}
+ \delta_{C(i)}
$$

where $s(i)$ is the segment, $t(i)$ the tier, $C(i)$ the channel, and $\pi_s$ the segment baseline conversion probability, subject to the centering constraint

$$
\delta_{\mathrm{organic}} + \delta_{\mathrm{affiliate}} + \delta_{\mathrm{paid\_search}} = 0,
$$

which is valid because channels are drawn uniformly (§2). With centering, $\pi_s$ is the conversion probability for segment $s$ at the tier reference price, reference demand, zero shock, and average channel effect.

---

## 7. Parameters

**Fixed by design (not swept):**

| Parameter | Value |
|---|---|
| $\sigma_H$ | $1.0$ |
| $\beta^{\mathrm{struct}}_{\texttt{price\_sensitive}}$ | $-2.0$ |
| $\beta^{\mathrm{struct}}_{\texttt{price\_resilient}}$ | $-0.8$ |
| promo depth support | $\{0.05,\ 0.10,\ 0.15\}$ |
| randomized multipliers | $\{0.85,\ 0.925,\ 1.00,\ 1.075,\ 1.15\}$ |
| $P^{\mathrm{ref}}_{\texttt{basic}}$ | $19.99$ |
| $P^{\mathrm{ref}}_{\texttt{premium}}$ | $29.99$ |

**Calibrated (swept, §10):** $b_C$, $\gamma_D$, $\gamma_H$, $\alpha_D$, $\alpha_H$, $\delta_D$, $\delta_H$, $\delta_C$, $\pi_s$, $\sigma_D$.

**Derived and frozen during calibration (§10.4):** $\beta^{\mathrm{rand}}_s$.

---

## 8. Random streams

Each stream is derived via `SeedSequence([master_seed, stream_id])`. Existing IDs are never renumbered, so previously generated data stays reproducible.

| Stream | ID |
|---|---:|
| user generation | 0 |
| weekly demand | 1 |
| price assignment | 2 |
| conversion outcome | 3 |
| churn outcome (legacy) | 4 |
| hidden demand shock | 5 |
| promo assignment | 6 |
| promo depth | 7 |
| randomized-arm membership | 8 |

---

## 9. Identification requirements

### 9.1 Overlap (positivity)

Observational treatment must not be deterministic given the measured confounders. Cells are $\texttt{channel} \times \texttt{tier} \times \texttt{demand tercile}$, where terciles are computed over users, not weeks.

For every cell above the minimum population (§10.1):

$$
0.05 < P(A_i = 1) < 0.95
$$

$$
\operatorname{SD}\left[\log\!\left(\frac{P_i}{P^{\mathrm{ref}}_{t(i)}}\right)\right] \ \geq\ \tau_{\mathrm{overlap}}, \qquad \text{initial target } \tau_{\mathrm{overlap}} = 0.04
$$

Tier is included in the cell definition so that differing tier reference prices cannot masquerade as within-tier price variation.

A promo rate of exactly $0$ or $1$ in an eligible cell fails the candidate immediately for that seed.

### 9.1.1 Price validity

Every generated base price and observed price must be strictly positive:

$$
P^{\mathrm{base}}_{w,t} > 0 \qquad\text{and}\qquad P_i > 0.
$$

$H_w$ is unbounded and enters the base price linearly, so a large $\alpha_H$ can in principle drive a price to zero or below. Any candidate producing a non-positive base or observed price fails immediately for that seed, and the check runs **before** any regression is fitted.

The production generator asserts the same condition, so a bad constant cannot silently emit a negative price outside calibration.

### 9.2 Randomization integrity

Within the randomized arm, the multiplier $M_i$ — equivalently $\log(P_i / P^{\mathrm{ref}}_{t(i)})$ — must be independent of segment, channel, tier, demand, shock, and week.

Raw price is **not** independent of tier, and must not be: it scales with the tier reference price by design.

Arm membership is independent of every pre-treatment covariate **by construction**, because users are selected uniformly without replacement. Complete randomization guarantees covariate balance in expectation, not exact balance in every realized sample. Calibration therefore checks realized balance against documented tolerances rather than requiring equality between arms.

Both arm-assignment integrity and randomized-treatment integrity are tested, because either can be broken accidentally while refactoring price assignment.

---

## 10. Calibration protocol

Calibration selects constants before the production generator is touched. Its goal is to guarantee, before Day 2 begins, that the naive, controlled, and randomized estimates will visibly differ.

### 10.1 Population and seeds

- $20{,}000$ users per run; the final candidate is re-confirmed at full size (§11.3).
- Three fixed seeds per candidate, never averaged inside the script: one CSV row per $\texttt{candidate} \times \texttt{seed}$.
- Minimum cell population for overlap eligibility: documented with the frozen constants. Cells below it are reported but cannot fail a candidate.

### 10.2 Regression specifications

All fits use the same heterogeneous price specification, so the only differences are the arm and the controls. The notation `log_price_ratio * segment` expands to

```text
log_price_ratio + segment + log_price_ratio:segment
```

| Fit | Arm | Specification |
|---|---|---|
| naive | observational | `conversion ~ log_price_ratio * segment` |
| controlled | observational | `... + demand + channel + tier` |
| oracle diagnostic | observational | `... + demand + channel + tier + hidden_shock` |
| randomized | randomized | `... + demand + channel + tier` |
| oracle randomized | randomized | `... + demand + channel + tier + hidden_shock` |

The observable randomized fit intentionally omits `hidden_shock`, because the production modeling path cannot access it. Randomization removes confounding by the shock, since treatment is independent of it, but the omission can still move a logistic coefficient through non-collapsibility.

The oracle-randomized fit serves a different purpose: it contains the complete structural conversion specification, so it should recover $\beta_s^{\mathrm{struct}}$ directly.

Estimation uses `statsmodels.Logit` throughout — unpenalized, with standard errors. Scikit-learn's `LogisticRegression` regularizes by default, which shrinks coefficients toward zero and is indistinguishable from the attenuation being measured.

With `price_sensitive` as the reference segment:

$$
\hat\beta_{\texttt{price\_sensitive}} = \hat\theta_{\texttt{log\_price\_ratio}}
$$

$$
\hat\beta_{\texttt{price\_resilient}} = \hat\theta_{\texttt{log\_price\_ratio}} + \hat\theta_{\texttt{log\_price\_ratio}:\texttt{segment[price\_resilient]}}
$$

The standard error of the derived coefficient must come from the fitted covariance matrix,

$$
\operatorname{SE}(\hat\beta_1 + \hat\beta_2) = \sqrt{\operatorname{Var}(\hat\beta_1) + \operatorname{Var}(\hat\beta_2) + 2\operatorname{Cov}(\hat\beta_1, \hat\beta_2)},
$$

never by summing individual standard errors.

### 10.3 The oracle diagnostic

The oracle fits exist only in calibration and evaluation, never in the modeling path. The purpose of the observational oracle fit:

> If adding the true hidden shock returns the estimates to ground truth, the residual bias in the controlled fit is genuinely caused by hidden confounding.

If the oracle-adjusted fit does not recover $\beta^{\mathrm{struct}}$ within tolerance, calibration fails. Investigate finite-sample instability, separation, implementation errors, or a mismatch between the fitted specification and the actual conversion DGP before changing any DGP constant.

### 10.4 Randomized estimand and non-collapsibility

The observable randomized regression identifies the causal effect without confounding, but its logistic coefficient need not numerically equal the structural generator coefficient when `hidden_shock` is omitted.

Calibration therefore records two quantities per segment: $\beta_s^{\mathrm{struct}}$, used directly in the conversion DGP, and $\beta_s^{\mathrm{rand}}$, the population coefficient corresponding to the observable randomized regression.

$\beta_s^{\mathrm{rand}}$ is derived during the full-size calibration run from a high-precision synthetic randomized population — at least $10^6$ users across three seeds, fitted with the observable randomized specification — and is then frozen in §14 together with its standard error. Because it is itself an estimate, its recovery criterion carries the same precision floor as every other SE-based rule.

The quantity

$$
\beta_s^{\mathrm{rand}} - \beta_s^{\mathrm{struct}}
$$

is recorded as the **non-collapsibility attenuation** for that segment.

Evaluation targets:

| Fit | Target |
|---|---|
| observable randomized | $\beta_s^{\mathrm{rand}}$ |
| oracle randomized | $\beta_s^{\mathrm{struct}}$ |
| oracle diagnostic (observational) | $\beta_s^{\mathrm{struct}}$ |
| naive (observational) | $\beta_s^{\mathrm{struct}}$ |
| controlled (observational) | $\beta_s^{\mathrm{rand}}$ — see below |

The controlled fit omits `hidden_shock` exactly as the observable randomized fit does, so its gap from $\beta^{\mathrm{struct}}$ contains both hidden confounding and the same non-collapsibility attenuation. Scoring its residual against $\beta^{\mathrm{rand}}$ isolates the confounding component, which is the claim the criterion exists to support.

One caveat is documented rather than assumed away: the attenuation factor depends on the distribution of treatment, and the observational arm has narrower price support than the randomized arm. $\beta^{\mathrm{rand}}$ is therefore a close but not exact attenuation benchmark for the observational arm. The calibration artifact records the controlled error against **both** targets, so this choice can be revisited without re-running the sweep.

The three headline gaps, in the order they are presented:

$$
\underbrace{\hat\beta^{\mathrm{naive}} - \hat\beta^{\mathrm{ctrl}}}_{\text{measured confounding removed}}, \qquad
\underbrace{\hat\beta^{\mathrm{ctrl}} - \beta^{\mathrm{rand}}}_{\text{hidden confounding remaining}}, \qquad
\underbrace{\beta^{\mathrm{rand}} - \beta^{\mathrm{struct}}}_{\text{non-collapsibility}}
$$

---

## 11. Acceptance criteria

Each criterion must hold **on every calibration seed**.

### 11.1 Behavior

| Metric | Target |
|---|---|
| overall conversion rate | $0.30 - 0.50$ |
| promo rate, organic | $\approx 0.15$ (soft) |
| promo rate, affiliate | $\approx 0.30$ (soft) |
| promo rate, paid search | $\approx 0.50$ (soft) |
| minimum eligible-cell log-price SD | $\geq \tau_{\mathrm{overlap}}$ |
| failed eligible overlap cells | $0$ |
| non-positive base or observed prices | $0$ |

Priority when these conflict:

1. **overlap** — an identification requirement;
2. **observable confounding** — the point of the exercise;
3. **plausible conversion rates** — realism;
4. **channel promo targets** — decoration.

A paid-search promo rate of $0.42$ with healthy within-cell variation beats $0.50$ with near-deterministic assignment.

### 11.2 Statistical behavior

Errors are defined against the target appropriate to each fit (§10.4).

For fits evaluated against the structural coefficient:

$$
\varepsilon_s^{\mathrm{struct}} = \hat\beta_s - \beta_s^{\mathrm{struct}}
$$

For fits evaluated against the randomized target:

$$
\varepsilon_s^{\mathrm{rand}} = \hat\beta_s - \beta_s^{\mathrm{rand}}
$$

| Criterion | Rule |
|---|---|
| naive bias, signed | $\varepsilon_s^{\mathrm{naive,struct}} \geq +0.5$ |
| controlled improves | $\lvert \varepsilon_s^{\mathrm{ctrl,struct}} \rvert < \lvert \varepsilon_s^{\mathrm{naive,struct}} \rvert$ |
| controlled residual is real | $\varepsilon_s^{\mathrm{ctrl,rand}} \geq +0.2$ and $\varepsilon_s^{\mathrm{ctrl,rand}} / \operatorname{SE}^{\mathrm{ctrl}}_s \geq 2$ |
| oracle observational recovers structural $\beta$ | $\lvert \varepsilon_s^{\mathrm{oracle,struct}} \rvert \leq 2\operatorname{SE}^{\mathrm{oracle}}_s$ and $\operatorname{SE}^{\mathrm{oracle}}_s \leq \tau_{\mathrm{SE}}$ |
| randomized recovers randomized target | $\lvert \varepsilon_s^{\mathrm{rand\text{-}fit,rand}} \rvert \leq 2\operatorname{SE}^{\mathrm{rand\text{-}fit}}_s$ and $\operatorname{SE}^{\mathrm{rand\text{-}fit}}_s \leq \tau_{\mathrm{SE}}$ |
| oracle randomized recovers structural $\beta$ | $\lvert \varepsilon_s^{\mathrm{oracle\text{-}rand,struct}} \rvert \leq 2\operatorname{SE}^{\mathrm{oracle\text{-}rand}}_s$ and $\operatorname{SE}^{\mathrm{oracle\text{-}rand}}_s \leq \tau_{\mathrm{SE}}$ |

Two notes on why the criteria are shaped this way:

- **The sign is part of the contract.** Both confounding paths push the same direction — high demand and positive shocks raise price and raise conversion — so the naive estimate is biased upward, meaning less negative than the truth. An error of the same magnitude in the opposite direction indicates a broken DGP, not a passing candidate.
- **Every SE-based rule carries a precision floor $\tau_{\mathrm{SE}}$.** Without it, an underpowered fit passes by being too noisy to contradict anything, and the problem worsens as samples grow: $\operatorname{SE} \to 0$ while a systematic gap can remain. $\tau_{\mathrm{SE}}$ is set from the pilot and recorded in §14; a starting value of $0.15$ per segment is reasonable.

### 11.3 Robustness

- All criteria hold across three seeds.
- The selected candidate is re-run at the full known-answer population and re-passes every criterion before its constants are frozen.
- Separation or regression failure marks the candidate failed and is recorded; it never terminates the sweep.
- Any candidate producing a non-positive price is recorded as failed before regression fitting.

---

## 12. Data separation

### 12.1 Two databases

| Path | Contents |
|---|---|
| `data/pricing.duckdb` | observable data only |
| `data/pricing_oracle.duckdb` | hidden shock, true probabilities, structural and randomized-target coefficients, configured constants, future oracle churn and regret quantities |

Oracle quantities exist because Day 4 needs them to compute the oracle action and decision regret. They are kept, not discarded — just unreachable by accident.

The modeling path must never open, attach, or join the oracle database. Evaluation code may do so deliberately.

### 12.2 Modeling-facing view

Models read `user_price_exposure`, not raw tables. The view exposes an explicit column allowlist and excludes the hidden shock, all true probabilities and coefficients, `promo` and `promo_depth` (§5.1), and every future oracle quantity.

Two tests enforce this: the observable database contains no oracle table, and the view's column set matches the allowlist exactly.

---

## 13. Deferred extensions

Anything here changes the DGP and therefore **requires recalibration**. Nothing in this section is implemented before §14 is filled in.

### 13.1 Post-treatment variables

`checkout_visit` is a mediator,

$$
\text{price} \rightarrow \text{checkout\_visit} \rightarrow \text{conversion},
$$

and `coupon_redeemed` is intended to demonstrate collider bias, with a candidate structure

$$
\text{promo} \rightarrow \text{coupon\_redeemed} \leftarrow \text{purchase\_intent} \rightarrow \text{conversion},
$$

where purchase intent is hidden.

One consequence must be settled before implementation: **once a mediator carries part of the price effect, the configured $\beta^{\mathrm{struct}}$ is no longer the total effect.** The extension must state which estimand it refers to and compute the total effect numerically from the oracle, since there is no closed form on the logit scale. The direction of collider bias is not assumed in advance.

### 13.2 Churn

The churn code currently in the repository predates this specification and is **not approved by it**. Its outputs are legacy synthetic data and must not be used for causal estimation or the simulator.

The redesign: for converted users, a monthly hazard driven by price, tier, segment, and tenure; a sampled churn month in $1, \dots, 12$ or NULL; `churned_within_12_months` derived from it. Non-converters get no churn record rather than a fabricated zero probability. The hidden cancellation reason is generated at churn time and stored as oracle ground truth for the later text-extraction exercise.

---

## 14. Frozen constants

*Filled in after calibration completes. Until then, the DGP is specified but not calibrated.*

| Item | Value |
|---|---|
| $b_{\texttt{organic}},\ b_{\texttt{affiliate}},\ b_{\texttt{paid\_search}}$ | TBD |
| $\gamma_D,\ \gamma_H$ | TBD |
| $\alpha_D,\ \alpha_H$ | TBD |
| $\delta_D,\ \delta_H$ | TBD |
| $\delta_C$ (centered) | TBD |
| $\pi_s$ (segment baseline conversion) | TBD |
| $\sigma_D$ | TBD |
| $\beta^{\mathrm{rand}}_{\texttt{price\_sensitive}}$ (with SE) | TBD |
| $\beta^{\mathrm{rand}}_{\texttt{price\_resilient}}$ (with SE) | TBD |
| non-collapsibility attenuation by segment | TBD |
| realized arm-balance tolerances | TBD |
| $\tau_{\mathrm{overlap}}$ | TBD |
| $\tau_{\mathrm{SE}}$ | TBD |
| minimum cell population | TBD |
| calibration population and seeds | TBD |
| high-precision population used to derive $\beta^{\mathrm{rand}}$ | TBD |
| final fixture: $n_{\mathrm{users}}$, randomization rate $r$ | TBD |
| selected `candidate_id` | TBD |

Also record: the naive, controlled, oracle, randomized, and oracle-randomized estimates per segment at full size, with their standard errors; and the commit hash of the calibration script that produced them.

---

## 15. Calibration artifacts

### 15.1 `artifacts/dgp_calibration.csv`

One row per $\texttt{candidate} \times \texttt{seed}$:

```text
candidate_id, seed,
<all swept parameters>,
overall_conversion_rate,
promo_rate_organic, promo_rate_affiliate, promo_rate_paid_search,
min_cell_promo_rate, max_cell_promo_rate, min_cell_log_price_sd,
n_valid_cells, n_failed_overlap_cells,
n_nonpositive_prices,
{naive|controlled|oracle|randomized|oracle_randomized}_{beta|se}_{segment},
{naive|controlled|oracle|randomized|oracle_randomized}_error_struct_{segment},
{naive|controlled|oracle|randomized|oracle_randomized}_error_rand_{segment},
{...}_standardized_error_{segment},
randomized_target_beta_price_sensitive,
randomized_target_beta_price_resilient,
noncollapsibility_attenuation_price_sensitive,
noncollapsibility_attenuation_price_resilient,
<per-criterion pass/fail flags>, verdict
```

### 15.2 `artifacts/dgp_overlap_cells.csv`

One row per $\texttt{candidate} \times \texttt{seed} \times \texttt{channel} \times \texttt{tier} \times \texttt{demand tercile}$:

```text
candidate_id, seed, channel, tier, demand_tercile,
cell_population, promo_rate,
mean_log_price_ratio, sd_log_price_ratio,
overlap_pass, failure_reason
```

This makes a failed candidate diagnosable by filtering rather than by re-running the sweep.

Both artifacts must be reproducible from the committed calibration script and its recorded seeds.

---

## 16. Day 1 done checklist

**Identification**

1. Observational treatment has adequate within-cell overlap.
2. Promo assignment is non-deterministic in every eligible cell.
3. Channel affects both treatment and conversion.
4. Measured demand affects both treatment and conversion.
5. The hidden shock affects both treatment and conversion.
6. Randomized-arm membership is independent of pre-treatment variables by construction, and realized arm balance is within the documented tolerances.
7. Within the randomized arm, the price multiplier is independent of every historical pricing driver.

**Statistical behavior**

8. The naive estimate is biased upward past the signed threshold, per segment.
9. Measured controls reduce that bias.
10. Residual bias against the randomized target remains and is at least $2\operatorname{SE}$ from zero.
11. The observational oracle fit recovers each $\beta^{\mathrm{struct}}$ within tolerance.
12. The observable randomized fit recovers each frozen randomized-arm target within $2\operatorname{SE}$, with $\operatorname{SE} \leq \tau_{\mathrm{SE}}$.
13. The oracle-randomized fit recovers each $\beta^{\mathrm{struct}}$ within $2\operatorname{SE}$, with $\operatorname{SE} \leq \tau_{\mathrm{SE}}$.

**Robustness**

14. Every criterion holds on all three calibration seeds.
15. The selected candidate passes a full-size confirmation run.
16. Separation and regression failures are recorded as failures, never silently passed.
17. Candidates producing non-positive prices are failed before regression fitting.

**Leakage prevention**

18. Oracle data live in a separate database.
19. No modeling code opens or attaches it.
20. The modeling-facing view matches its allowlist exactly, with promo fields excluded.

**Documentation**

21. §14 is filled in, including $\beta^{\mathrm{rand}}_s$, the attenuation per segment, $\tau_{\mathrm{overlap}}$, $\tau_{\mathrm{SE}}$, and the SE derivation behind the recovery tolerance.
22. Both calibration artifacts are reproducible from the committed script.

Only then do the Day 2 estimators get written.
