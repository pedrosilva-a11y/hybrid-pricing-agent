# Synthetic Pricing Data-Generating Process

**Status:** calibrated and frozen — see §14
**Selected candidate:** `balanced`
**Scope:** customer population, pricing policy, promo assignment, randomized arm, conversion
**Out of scope (deferred, see §13):** churn, post-treatment variables, cancellation text

---

## 1. Purpose

This document defines the data-generating process (DGP) for the hybrid subscription pricing project: which variables exist, how they cause one another, which equations produce them, and which are observable versus oracle-only.

The data are synthetic so that the true causal price effect is known. That makes it possible to score causal estimators against ground truth instead of against predictive fit alone.

Primary causal question:

> What is the causal effect of changing subscription price on conversion probability?

Historical prices are deliberately confounded by both measured and unmeasured variables. A randomized pricing arm provides a benchmark in which the assigned price is independent of all of them.

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
- $\sigma_H = 1.0$ is fixed, not calibrated. Only the products $\alpha_H \sigma_H$, $\gamma_H \sigma_H$, and $\delta_H \sigma_H$ affect the process, so fixing the scale makes the sensitivity parameters uniquely identified. Note the asymmetry of units: $D_w - 1$ has a standard deviation of roughly $\sigma_D$ ($0.10$, see §14), while $H_w$ has one. The coefficients $\delta_D$ and $\delta_H$ are therefore not directly comparable.
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

At the frozen constants the two differ by $+0.020$ and $+0.018$ — within one standard error of zero. The distinction was measured, not assumed away, and turned out to be immaterial at this shock strength.

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

Conditional on $A_i = 1$, depth $d_i$ is drawn uniformly from $\{0.05, 0.10, 0.20\}$, independent of everything else; otherwise $d_i = 0$.

$$
P_i = P^{\mathrm{base}}_{w(i),\,t(i)}\,(1 - d_i)
$$

This user-level draw is what creates overlap: two users in the same week, channel, and tier can face different prices. The support was widened from $\{0.05, 0.10, 0.15\}$ during calibration because the minimum within-cell log-price SD sat at $0.043$ against a threshold of $0.04$ — too thin a margin to survive an unlucky seed.

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
| promo depth support | $\{0.05,\ 0.10,\ 0.20\}$ |
| randomized multipliers | $\{0.85,\ 0.925,\ 1.00,\ 1.075,\ 1.15\}$ |
| $P^{\mathrm{ref}}_{\texttt{basic}}$ | $19.99$ |
| $P^{\mathrm{ref}}_{\texttt{premium}}$ | $29.99$ |

**Calibrated (swept, §10):** $b_C$, $\gamma_D$, $\gamma_H$, $\alpha_D$, $\alpha_H$, $\delta_D$, $\delta_H$, $\delta_C$, $\pi_s$, $\sigma_D$ — frozen values in §14.

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

Stream 0 uses `np.random.default_rng` as of the calibration commit. The earlier `random.Random` implementation produced a different population for the same master seed; the two are not interchangeable, and the switch was made deliberately before the constants were frozen.

---

## 9. Identification requirements

### 9.1 Overlap (positivity)

Observational treatment must not be deterministic given the measured confounders. Cells are $\texttt{channel} \times \texttt{tier} \times \texttt{demand tercile}$, where terciles are computed over users, not weeks.

For every cell above the minimum population (§10.1):

$$
0.05 < P(A_i = 1) < 0.95
$$

$$
\operatorname{SD}\left[\log\!\left(\frac{P_i}{P^{\mathrm{ref}}_{t(i)}}\right)\right] \ \geq\ \tau_{\mathrm{overlap}}
$$

Tier is included in the cell definition so that differing tier reference prices cannot masquerade as within-tier price variation.

A promo rate of exactly $0$ or $1$ in an eligible cell fails the candidate immediately for that seed.

Terciles are computed with `labels=False, duplicates="drop"`, followed by an explicit check that exactly three bins resulted. Duplicate quantile edges must not silently produce mislabeled bins, but a genuinely degenerate demand draw must fail loudly.

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

- $60{,}000$ users over $1{,}000$ weeks per pilot run; the selected candidate is re-confirmed at full size (§11.3).
- Three fixed seeds per candidate, never averaged inside the script: one CSV row per $\texttt{candidate} \times \texttt{seed}$.
- Minimum cell population for overlap eligibility: $100$. Cells below it are reported but cannot fail a candidate.

The week count matters more than it appears. Demand and the hidden shock vary by week, so the effective sample for the confounding path is the number of weeks, not the number of users. At $240$ weeks the paired hidden-confounding gap swung by $0.70$ across seeds; at $1{,}000$ weeks it swings by $0.17$. Standard errors are clustered on week for the same reason.

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

The pilot fits only the three observational specifications; confirmation adds the two randomized ones (see §11.3).

Estimation uses `statsmodels.Logit` throughout — unpenalized, with standard errors clustered on week. Scikit-learn's `LogisticRegression` regularizes by default, which shrinks coefficients toward zero and is indistinguishable from the attenuation being measured.

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

$\beta_s^{\mathrm{rand}}$ is derived from a high-precision synthetic randomized population — $1{,}200{,}000$ users across three seeds at $3{,}000$ weeks, fitted with the observable randomized specification — and frozen in §14 together with its standard error. Because it is itself an estimate, its recovery criterion carries the same precision floor as every other SE-based rule, and the target's own SE enters the comparison.

The quantity $\beta_s^{\mathrm{rand}} - \beta_s^{\mathrm{struct}}$ is recorded as the **non-collapsibility attenuation**.

### 10.4.1 Paired hidden-confounding estimator

Scoring the controlled fit against $\beta^{\mathrm{rand}}$ compares estimates from two independent data draws, so each seed's realized demand and shock values enter the difference and never cancel. The paired contrast avoids this:

$$
\text{paired gap} = \hat\beta^{\mathrm{ctrl}} - \hat\beta^{\mathrm{oracle}}
$$

Both fits use the same rows of the same data and differ only by the inclusion of $H_w$, so a seed-level shift differences out. Measured on the confirmation runs, the paired gap varies by $0.17$ across seeds where the $\beta^{\mathrm{rand}}$ version varies by $0.46$.

The paired gap still contains the specification effect of adding $H_w$. The randomized arm isolates that effect alone, since treatment there is independent of the shock:

$$
\text{specification effect} = \hat\beta^{\mathrm{rand\text{-}fit}} - \hat\beta^{\mathrm{oracle\text{-}rand}}
$$

$$
\text{corrected hidden confounding} = \left(\hat\beta^{\mathrm{ctrl}} - \hat\beta^{\mathrm{oracle}}\right) - \left(\hat\beta^{\mathrm{rand\text{-}fit}} - \hat\beta^{\mathrm{oracle\text{-}rand}}\right)
$$

Caveat: the specification effect depends on the treatment distribution, and the randomized arm has wider price support than the observational arm, so the subtraction is close but not exact. At the frozen constants the specification effect is approximately zero (measured: $-0.002$ to $+0.007$), so the correction is immaterial here — but it is computed rather than assumed.

### 10.4.2 Evaluation targets

| Fit | Target |
|---|---|
| naive (observational) | $\beta_s^{\mathrm{struct}}$ |
| controlled (observational) | paired against the oracle fit (§10.4.1); error against $\beta^{\mathrm{rand}}$ recorded as a diagnostic |
| oracle diagnostic (observational) | $\beta_s^{\mathrm{struct}}$ |
| observable randomized | $\beta_s^{\mathrm{rand}}$ |
| oracle randomized | $\beta_s^{\mathrm{struct}}$ |

The three headline gaps, in the order they are presented:

$$
\underbrace{\hat\beta^{\mathrm{naive}} - \hat\beta^{\mathrm{ctrl}}}_{\text{measured confounding removed}}, \qquad
\underbrace{\hat\beta^{\mathrm{ctrl}} - \hat\beta^{\mathrm{oracle}}}_{\text{hidden confounding remaining}}, \qquad
\underbrace{\beta^{\mathrm{rand}} - \beta^{\mathrm{struct}}}_{\text{non-collapsibility}}
$$

---

## 11. Acceptance criteria

Each criterion must hold **on every seed**.

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

Errors are defined against the target appropriate to each fit (§10.4.2), with $\varepsilon_s^{\mathrm{struct}} = \hat\beta_s - \beta_s^{\mathrm{struct}}$ and $\varepsilon_s^{\mathrm{rand}} = \hat\beta_s - \beta_s^{\mathrm{rand}}$.

| Criterion | Rule | Phase |
|---|---|---|
| naive bias, signed | $\varepsilon_s^{\mathrm{naive,struct}} \geq +0.50$ | both |
| controlled improves | $\lvert \varepsilon_s^{\mathrm{ctrl,struct}} \rvert < \lvert \varepsilon_s^{\mathrm{naive,struct}} \rvert$ | both |
| paired hidden confounding | $\hat\beta^{\mathrm{ctrl}}_s - \hat\beta^{\mathrm{oracle}}_s \geq 0.20$ | both |
| corrected hidden confounding | corrected gap $\geq 0.20$ | confirmation |
| oracle observational recovers $\beta^{\mathrm{struct}}$ | $\lvert \varepsilon_s^{\mathrm{oracle,struct}} \rvert \leq 3\operatorname{SE}^{\mathrm{oracle}}_s$ and $\operatorname{SE}^{\mathrm{oracle}}_s \leq \tau_{\mathrm{SE}}$ | confirmation |
| randomized recovers $\beta^{\mathrm{rand}}$ | $\lvert \varepsilon_s^{\mathrm{rand\text{-}fit,rand}} \rvert \leq 3\sqrt{(\operatorname{SE}^{\mathrm{rand\text{-}fit}}_s)^2 + (\operatorname{SE}^{\mathrm{target}}_s)^2}$ and $\operatorname{SE}^{\mathrm{rand\text{-}fit}}_s \leq \tau_{\mathrm{SE}}$ | confirmation |
| oracle randomized recovers $\beta^{\mathrm{struct}}$ | $\lvert \varepsilon_s^{\mathrm{oracle\text{-}rand,struct}} \rvert \leq 3\operatorname{SE}^{\mathrm{oracle\text{-}rand}}_s$ and $\operatorname{SE}^{\mathrm{oracle\text{-}rand}}_s \leq \tau_{\mathrm{SE}}$ | confirmation |

Three notes on why the criteria are shaped this way:

- **The sign is part of the contract.** Both confounding paths push the same direction — high demand and positive shocks raise price and raise conversion — so the naive estimate is biased upward, meaning less negative than the truth. An error of the same magnitude in the opposite direction indicates a broken DGP, not a passing candidate.
- **Every SE-based rule carries a precision floor $\tau_{\mathrm{SE}}$.** Without it, an underpowered fit passes by being too noisy to contradict anything, and the problem worsens as samples grow: $\operatorname{SE} \to 0$ while a systematic gap can remain.
- **The tolerance is 3 SE, not 2, because of multiple comparisons.** Confirmation runs six recovery checks per seed across three seeds. At 2 SE, eighteen simultaneous tests pass together only about 40% of the time even when the DGP is correct. At 3 SE that rises to roughly 95%. The precision floor is what keeps the test meaningful; widening the tolerance without it would not be acceptable.

### 11.3 Robustness

- All criteria hold across three seeds.
- The selected candidate is re-run at the full known-answer population and re-passes every criterion before its constants are frozen.
- Separation or regression failure marks the candidate failed and is recorded; it never terminates the sweep.
- Any candidate producing a non-positive price is recorded as failed before regression fitting.
- Pilot and confirmation gate different criteria. The pilot gates point-estimate geometry only — naive bias, controlled improvement, paired hidden confounding — while SE-based recovery is confirmation-only, because a pilot-sized arm cannot meet $\tau_{\mathrm{SE}}$ and would reject every candidate for lack of power rather than for behavior.

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

Anything here changes the DGP and therefore **requires recalibration**.

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

Selected candidate: **`balanced`**.

| Item | Value |
|---|---|
| $b_{\texttt{organic}},\ b_{\texttt{affiliate}},\ b_{\texttt{paid\_search}}$ | $-1.73,\ -0.85,\ 0.0$ |
| $\gamma_D,\ \gamma_H$ (promo) | $5.0,\ 0.45$ |
| $\alpha_D,\ \alpha_H$ (price) | $0.20,\ 0.025$ |
| $\delta_D,\ \delta_H$ (conversion) | $5.0,\ 0.10$ |
| $\delta_C$ (centered: organic, affiliate, paid search) | $-0.10,\ 0.0,\ +0.10$ |
| $\pi_s$ (baseline conversion) | $0.35$ sensitive, $0.40$ resilient |
| $\sigma_D$ | $0.10$ |
| $\beta^{\mathrm{rand}}_{\texttt{price\_sensitive}}$ | $-1.9796$ (SE $0.0261$, seed SD $0.0446$) |
| $\beta^{\mathrm{rand}}_{\texttt{price\_resilient}}$ | $-0.7825$ (SE $0.0254$, seed SD $0.0698$) |
| non-collapsibility attenuation | $+0.020$ and $+0.018$ — within one SE of zero |
| $\tau_{\mathrm{overlap}}$ | $0.04$ |
| $\tau_{\mathrm{SE}}$ | $0.15$ |
| minimum cell population | $100$ |
| pilot run | $60{,}000$ users, $1{,}000$ weeks, $r = 0.10$, seeds $(11, 22, 33)$ |
| $\beta^{\mathrm{rand}}$ derivation | $1{,}200{,}000$ users total, $3{,}000$ weeks, $r = 1.0$, seeds $(101, 202, 303)$ |
| final fixture | $150{,}000$ users, $2{,}000$ weeks, $r = 0.50$, seeds $(11, 22, 33)$ |
| recovery tolerance | $3\operatorname{SE}$ (see §11.2) |
| calibration script commit | TBD |

### 14.1 Confirmation results

Per seed $(11 / 22 / 33)$:

| Quantity | `price_sensitive` | `price_resilient` |
|---|---|---|
| removal (naive − controlled) | $1.944 / 1.631 / 1.768$ | $1.930 / 1.608 / 1.803$ |
| paired hidden confounding | $0.546 / 0.712 / 0.630$ | $0.545 / 0.727 / 0.629$ |
| specification effect | $-0.002 / +0.005 / +0.003$ | $-0.002 / +0.007 / +0.001$ |
| corrected hidden confounding | $0.548 / 0.707 / 0.627$ | $0.547 / 0.719 / 0.628$ |
| controlled $-\ \beta^{\mathrm{rand}}$ (diagnostic) | $0.719 / 0.671 / 0.255$ | $0.653 / 0.646 / 0.368$ |
| controlled SE | $0.133 / 0.136 / 0.133$ | $0.135 / 0.134 / 0.136$ |
| oracle SE | $0.142 / 0.144 / 0.140$ | $0.144 / 0.143 / 0.143$ |
| randomized SE | $0.109 / 0.106 / 0.103$ | $0.103 / 0.102 / 0.101$ |
| oracle-randomized SE | $0.108 / 0.106 / 0.103$ | $0.103 / 0.102 / 0.101$ |

All three seeds pass every confirmation criterion.

The diagnostic row is retained deliberately: its spread of $0.46$ against the paired estimator's $0.17$ is the evidence for §10.4.1.

---

## 15. Calibration artifacts

### 15.1 `artifacts/dgp_calibration.csv`

One row per $\texttt{candidate} \times \texttt{seed}$:

```text
candidate_id, seed, evaluation_phase,
n_users, n_weeks, randomization_rate,
<all swept parameters>,
overall_conversion_rate,
promo_rate_{channel}, promo_rate_error_{channel},
min_cell_promo_rate, max_cell_promo_rate, min_cell_log_price_sd,
n_valid_cells, n_failed_overlap_cells, n_ineligible_cells,
n_nonpositive_prices, n_nonfinite_prices,
randomized_price_corr_{demand_index|hidden_shock|week},
randomized_price_spread_{segment|channel|tier},
{naive|controlled|oracle|randomized|oracle_randomized}_{beta|se}_{segment},
{...}_error_struct_{segment}, {...}_error_rand_{segment},
{...}_standardized_error_{segment},
{segment}_naive_minus_controlled,
{segment}_controlled_minus_oracle,
{segment}_randomized_minus_oracle_randomized,
{segment}_corrected_hidden_confounding,
{segment}_controlled_minus_beta_rand,
{segment}_beta_rand_minus_beta_struct,
randomized_target_beta_{segment}, randomized_target_se_{segment},
randomized_target_seed_sd_{segment},
noncollapsibility_attenuation_{segment},
<per-criterion pass/fail flags>, passed, verdict, failures
```

Confirmation runs write the same schema to `artifacts/dgp_confirmation.csv`.

### 15.2 `artifacts/dgp_overlap_cells.csv`

One row per $\texttt{candidate} \times \texttt{seed} \times \texttt{channel} \times \texttt{tier} \times \texttt{demand tercile}$:

```text
candidate_id, seed, channel, tier, demand_tercile,
cell_population, promo_rate,
mean_log_price_ratio, sd_log_price_ratio,
overlap_eligible, overlap_pass, failure_reason
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
10. The paired hidden-confounding gap is at least $0.20$ on every seed and segment.
11. The corrected gap, net of the specification effect, is at least $0.20$ (confirmation).
12. The observational oracle fit recovers each $\beta^{\mathrm{struct}}$ within $3\operatorname{SE}$, with $\operatorname{SE} \leq \tau_{\mathrm{SE}}$.
13. The observable randomized fit recovers $\beta^{\mathrm{rand}}$ within $3\operatorname{SE}$ of the combined SE, and the oracle-randomized fit recovers $\beta^{\mathrm{struct}}$ within $3\operatorname{SE}$.

**Robustness**

14. Every criterion holds on all three seeds.
15. The selected candidate passes a full-size confirmation run.
16. Separation and regression failures are recorded as failures, never silently passed.
17. Candidates producing non-positive prices are failed before regression fitting.

**Leakage prevention**

18. Oracle data live in a separate database.
19. No modeling code opens or attaches it.
20. The modeling-facing view matches its allowlist exactly, with promo fields excluded.

**Documentation**

21. §14 is filled in, including $\beta^{\mathrm{rand}}_s$, the attenuation per segment, $\tau_{\mathrm{overlap}}$, $\tau_{\mathrm{SE}}$, and the reasoning behind the recovery tolerance.
22. Both calibration artifacts are reproducible from the committed script.

Items 1–17 and 21–22 are satisfied as of the confirmation run recorded in §14.1. Items 18–20 remain open: they depend on the production generator adopting the frozen constants and on the oracle/observable split being implemented and tested.

Only then do the Day 2 estimators get written.
