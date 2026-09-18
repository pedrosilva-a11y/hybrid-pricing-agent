"""Calibrate the synthetic pricing data-generating process."""

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]
from numpy.random import SeedSequence
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import (  # type: ignore[import-untyped]
    PerfectSeparationError,
)

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Random Stream Identifiers
# ---------------------------------------------------------------------------

USER_GENERATION_STREAM: Final = 0
WEEKLY_DEMAND_STREAM: Final = 1
PRICE_ASSIGNMENT_STREAM: Final = 2
CONVERSION_OUTCOME_STREAM: Final = 3
HIDDEN_SHOCK_STREAM: Final = 5
PROMO_ASSIGNMENT_STREAM: Final = 6
PROMO_DEPTH_STREAM: Final = 7
RANDOMIZED_ARM_MEMBERSHIP_STREAM: Final = 8

# ---------------------------------------------------------------------------
# User Generation
# ---------------------------------------------------------------------------

ALLOWED_ACQUISITION_CHANNELS: Final = (
    "organic",
    "affiliate",
    "paid_search",
)
ALLOWED_SUBSCRIPTION_TIERS: Final = (
    "basic",
    "premium",
)

CHANNEL_KEY: Final = "channel"
SEGMENT_KEY: Final = "segment"
SEGMENTS: Final = (
    "price_sensitive",
    "price_resilient",
)
SIGNUP_WEEK_KEY: Final = "signup_week"
TIER_KEY: Final = "tier"
USER_ID_KEY: Final = "user_id"

# ---------------------------------------------------------------------------
# Weekly Conditions
# ---------------------------------------------------------------------------

DEMAND_HIGH_BOUND: Final = 1.25
DEMAND_INDEX_KEY: Final = "demand_index"
DEMAND_LOW_BOUND: Final = 0.75
DEMAND_MEAN: Final = 1.0
HIDDEN_SHOCK_KEY: Final = "hidden_shock"
WEEK_KEY: Final = "week"

# ---------------------------------------------------------------------------
# Randomized Arm
# ---------------------------------------------------------------------------

IS_RANDOMIZED_KEY: Final = "is_randomized"

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

BASE_PRICE_KEY: Final = "base_price"
OBSERVED_PRICE_KEY: Final = "observed_price"
PROMO_DEPTH_KEY: Final = "promo_depth"
PROMO_DEPTHS: Final = (
    0.05,
    0.10,
    0.20,
)
PROMO_KEY: Final = "promo"

RANDOMIZED_PRICE_MULTIPLIERS: Final = (
    0.85,
    0.925,
    1.00,
    1.075,
    1.15,
)

REFERENCE_PRICE_BY_TIER: Final = {
    "basic": 19.99,
    "premium": 29.99,
}

REFERENCE_PRICE_KEY: Final = "reference_price"

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

CONVERSION_PROB_KEY: Final = "conversion_probability"
CONVERSION_OUTCOME_KEY: Final = "conversion_outcome"
LOG_PRICE_RATIO_KEY: Final = "log_price_ratio"

STRUCTURAL_BETA_BY_SEGMENT: Final = {
    "price_sensitive": -2.0,
    "price_resilient": -0.8,
}

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

DEMAND_TERCILE_KEY: Final = "demand_tercile"

N_USERS_DIAGNOSTIC_KEY: Final = "n_users"
PROMO_RATE_DIAGNOSTIC_KEY: Final = "promo_rate"
LOG_PRICE_SD_DIAGNOSTIC_KEY: Final = "log_price_sd"

PROMO_RATE_TARGET_BY_CHANNEL: Final = {
    "organic": 0.15,
    "affiliate": 0.30,
    "paid_search": 0.50,
}


@dataclass(frozen=True, slots=True)
class DgpCandidate:
    """Candidate parameters for data-generating process calibration.

    Attributes:
        candidate_id: Unique identifier for the candidate parameter configuration.
        promo_bias_organic: Promo logit intercept for organic users.
        promo_bias_affiliate: Promo logit intercept for affiliate users.
        promo_bias_paid_search: Promo logit intercept for paid-search users.
        promo_demand_sensitivity: Demand effect on promo assignment.
        promo_shock_sensitivity: Hidden-shock effect on promo assignment.
        price_demand_sensitivity: Demand effect on base price.
        price_shock_sensitivity: Hidden-shock effect on base price.
        baseline_conversion_price_sensitive: Baseline conversion for
            price-sensitive users.
        baseline_conversion_price_resilient: Baseline conversion for
            price-resilient users.
        demand_conversion_sensitivity: Effect of measured demand on
            conversion logits.
        shock_conversion_sensitivity: Effect of hidden shock on
            conversion logits.
        channel_conversion_organic: Organic-channel conversion-logit effect.
        channel_conversion_affiliate: Affiliate-channel conversion-logit effect.
        channel_conversion_paid_search: Paid-search conversion-logit effect.
        demand_std_dev: Standard deviation of measured weekly demand.
    """

    candidate_id: str

    promo_bias_organic: float
    promo_bias_affiliate: float
    promo_bias_paid_search: float

    promo_demand_sensitivity: float
    promo_shock_sensitivity: float

    price_demand_sensitivity: float
    price_shock_sensitivity: float

    baseline_conversion_price_sensitive: float
    baseline_conversion_price_resilient: float

    demand_conversion_sensitivity: float
    shock_conversion_sensitivity: float

    channel_conversion_organic: float
    channel_conversion_affiliate: float
    channel_conversion_paid_search: float

    demand_std_dev: float


@dataclass(frozen=True, slots=True)
class PriceEffectEstimate:
    """Estimated segment-specific price effects.

    Attributes:
        price_sensitive_beta: Price coefficient for price-sensitive users.
        price_sensitive_se: Standard error for the price-sensitive coefficient.
        price_resilient_beta: Price coefficient for price-resilient users.
        price_resilient_se: Standard error for the price-resilient coefficient.
    """

    price_sensitive_beta: float
    price_sensitive_se: float
    price_resilient_beta: float
    price_resilient_se: float


# ---------------------------------------------------------------------------
# Randomness Helpers
# ---------------------------------------------------------------------------


def derive_seed(master_seed: int, stream: int) -> int:
    """Derive a deterministic seed for an independent random stream."""
    seed_sequence = SeedSequence([master_seed, stream])
    return int(seed_sequence.generate_state(1)[0])


def sigmoid(value: np.ndarray) -> FloatArray:
    """Transform logits into probabilities."""
    result = 1.0 / (1.0 + np.exp(-value))
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# DGP Simulation
# ---------------------------------------------------------------------------


def generate_population(
    n_users: int,
    n_weeks: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the synthetic customer population.

    Stream 0 intentionally uses NumPy ``default_rng``. This implementation is part
    of the reproducibility contract once calibration is frozen.

    Args:
        n_users: Number of synthetic users to generate.
        n_weeks: Number of simulated weeks available for signup.
        seed: Master seed used for reproducible user generation.

    Returns:
        DataFrame containing the generated user population.
    """
    rng = np.random.default_rng(derive_seed(seed, USER_GENERATION_STREAM))

    segments = np.asarray(SEGMENTS, dtype=object)
    channels = np.asarray(ALLOWED_ACQUISITION_CHANNELS, dtype=object)
    tiers = np.asarray(ALLOWED_SUBSCRIPTION_TIERS, dtype=object)

    return pd.DataFrame(
        {
            USER_ID_KEY: np.arange(n_users, dtype=np.int64),
            SIGNUP_WEEK_KEY: rng.integers(
                0,
                n_weeks,
                size=n_users,
                dtype=np.int64,
            ),
            SEGMENT_KEY: rng.choice(
                segments,
                size=n_users,
            ),
            CHANNEL_KEY: rng.choice(
                channels,
                size=n_users,
            ),
            TIER_KEY: rng.choice(
                tiers,
                size=n_users,
            ),
        }
    )


def generate_weekly_conditions(
    n_weeks: int,
    demand_std_dev: float,
    seed: int,
) -> pd.DataFrame:
    """Generate measured demand and hidden market shocks by week.

    Args:
        n_weeks: Number of simulated weeks to generate.
        demand_std_dev: Standard deviation of weekly measured demand.
        seed: Master seed used for reproducible weekly conditions.

    Returns:
        DataFrame containing weekly demand indices and hidden shocks.
    """
    demand_rng = np.random.default_rng(derive_seed(seed, WEEKLY_DEMAND_STREAM))

    shock_rng = np.random.default_rng(derive_seed(seed, HIDDEN_SHOCK_STREAM))

    raw_demand = demand_rng.normal(
        loc=DEMAND_MEAN,
        scale=demand_std_dev,
        size=n_weeks,
    )

    simulated_demand = np.clip(
        raw_demand,
        DEMAND_LOW_BOUND,
        DEMAND_HIGH_BOUND,
    )

    simulated_hidden_shock = shock_rng.normal(
        loc=0.0,
        scale=1.0,
        size=n_weeks,
    )

    return pd.DataFrame(
        {
            WEEK_KEY: np.arange(n_weeks, dtype=np.int64),
            DEMAND_INDEX_KEY: simulated_demand,
            HIDDEN_SHOCK_KEY: simulated_hidden_shock,
        }
    )


def assign_arm(
    population: pd.DataFrame,
    randomization_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Assign users to observational or randomized pricing arms.

    Args:
        population: Generated synthetic user population.
        randomization_rate: Fraction of users assigned to randomized pricing.
        seed: Master seed used for reproducible arm assignment.

    Returns:
        Population DataFrame with randomized-arm membership added.
    """
    rng = random.Random(
        derive_seed(
            seed,
            RANDOMIZED_ARM_MEMBERSHIP_STREAM,
        )
    )

    n_users = len(population)
    n_randomized = math.ceil(randomization_rate * n_users)

    randomized_positions = set(
        rng.sample(
            range(n_users),
            k=n_randomized,
        )
    )

    assigned_population = population.copy()

    assigned_population[IS_RANDOMIZED_KEY] = [
        position in randomized_positions for position in range(n_users)
    ]

    return assigned_population


def calculate_base_prices(
    data: pd.DataFrame,
    candidate: DgpCandidate,
) -> pd.DataFrame:
    """Calculate tier-specific base prices.

    Args:
        data: User data containing tier and weekly market conditions.
        candidate: Candidate DGP parameters controlling base-price policy.

    Returns:
        DataFrame with reference and base prices added.
    """
    data[REFERENCE_PRICE_KEY] = data[TIER_KEY].map(REFERENCE_PRICE_BY_TIER)

    data[BASE_PRICE_KEY] = data[REFERENCE_PRICE_KEY] * (
        1.0
        + candidate.price_demand_sensitivity * (data[DEMAND_INDEX_KEY] - DEMAND_MEAN)
        + candidate.price_shock_sensitivity * data[HIDDEN_SHOCK_KEY]
    )

    return data


def calculate_promo_probabilities(
    data: pd.DataFrame,
    candidate: DgpCandidate,
) -> np.ndarray:
    """Calculate promo probabilities from channel, demand, and hidden shock.

    Args:
        data: User data containing channel and weekly market conditions.
        candidate: Candidate DGP parameters controlling promo assignment.

    Returns:
        Promo-assignment probability for each user.
    """
    promo_bias_by_channel = {
        "organic": candidate.promo_bias_organic,
        "affiliate": candidate.promo_bias_affiliate,
        "paid_search": candidate.promo_bias_paid_search,
    }

    promo_logits = (
        data[CHANNEL_KEY].map(promo_bias_by_channel)
        - candidate.promo_demand_sensitivity * (data[DEMAND_INDEX_KEY] - DEMAND_MEAN)
        - candidate.promo_shock_sensitivity * data[HIDDEN_SHOCK_KEY]
    )

    return sigmoid(promo_logits.to_numpy())


def assign_promotions(
    data: pd.DataFrame,
    promo_probabilities: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Assign promotions and promo depths to observational users.

    Args:
        data: User data containing randomized-arm membership.
        promo_probabilities: Promo probability for each user.
        seed: Master seed used for reproducible promo assignment.

    Returns:
        DataFrame with promo assignment and promo depth added.
    """
    promo_rng = np.random.default_rng(
        derive_seed(
            seed,
            PROMO_ASSIGNMENT_STREAM,
        )
    )

    promo_depth_rng = np.random.default_rng(
        derive_seed(
            seed,
            PROMO_DEPTH_STREAM,
        )
    )

    promo_draws = promo_rng.random(len(data))

    is_observational = ~data[IS_RANDOMIZED_KEY]

    data[PROMO_KEY] = (promo_draws < promo_probabilities) & is_observational

    data[PROMO_DEPTH_KEY] = 0.0

    promoted_mask = data[PROMO_KEY]
    n_promoted = int(promoted_mask.sum())

    data.loc[
        promoted_mask,
        PROMO_DEPTH_KEY,
    ] = promo_depth_rng.choice(
        PROMO_DEPTHS,
        size=n_promoted,
    )

    return data


def assign_observational_prices(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate prices from base price and promotional depth.

    Args:
        data: User data containing base prices and promo depths.

    Returns:
        DataFrame with observational prices assigned.
    """
    data[OBSERVED_PRICE_KEY] = data[BASE_PRICE_KEY] * (1.0 - data[PROMO_DEPTH_KEY])

    return data


def assign_randomized_prices(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Replace prices for randomized users with experimental prices.

    Args:
        data: User data containing randomized-arm membership and
            reference prices.
        seed: Master seed used for randomized price assignment.

    Returns:
        DataFrame with randomized experimental prices assigned.
    """
    price_rng = np.random.default_rng(
        derive_seed(
            seed,
            PRICE_ASSIGNMENT_STREAM,
        )
    )

    randomized_mask = data[IS_RANDOMIZED_KEY]

    n_randomized = int(randomized_mask.sum())

    randomized_multipliers = price_rng.choice(
        RANDOMIZED_PRICE_MULTIPLIERS,
        size=n_randomized,
    )

    data.loc[
        randomized_mask,
        OBSERVED_PRICE_KEY,
    ] = (
        data.loc[
            randomized_mask,
            REFERENCE_PRICE_KEY,
        ].to_numpy()
        * randomized_multipliers
    )

    return data


def assign_prices(
    population: pd.DataFrame,
    weekly_conditions: pd.DataFrame,
    candidate: DgpCandidate,
    seed: int,
) -> pd.DataFrame:
    """Generate observational and randomized prices for each user.

    Args:
        population: Population including randomized-arm membership.
        weekly_conditions: Weekly measured demand and hidden shocks.
        candidate: Candidate DGP parameters controlling price and promotion.
        seed: Master seed used for reproducible price assignment.

    Returns:
        User-level DataFrame containing pricing and promotion variables.
    """
    data = population.merge(
        weekly_conditions,
        left_on=SIGNUP_WEEK_KEY,
        right_on=WEEK_KEY,
        how="left",
        validate="many_to_one",
    )

    data = calculate_base_prices(
        data,
        candidate,
    )

    promo_probabilities = calculate_promo_probabilities(
        data,
        candidate,
    )

    data = assign_promotions(
        data,
        promo_probabilities,
        seed,
    )

    data = assign_observational_prices(data)

    return assign_randomized_prices(
        data,
        seed,
    )


def validate_prices(
    data: pd.DataFrame,
) -> list[str]:
    """Validate price values before downstream outcome calculations.

    Args:
        data: User-level data containing generated pricing variables.

    Returns:
        List of validation failures. Empty means valid prices.
    """
    validation_failures: list[str] = []

    price_columns = [
        REFERENCE_PRICE_KEY,
        BASE_PRICE_KEY,
        OBSERVED_PRICE_KEY,
    ]

    missing_columns = [column for column in price_columns if column not in data.columns]

    if missing_columns:
        validation_failures.append(f"missing required price columns: {missing_columns}")
        return validation_failures

    if not np.isfinite(data[price_columns]).all().all():
        validation_failures.append("found missing or infinite price values")

    if not (data[price_columns] > 0).all().all():
        validation_failures.append("found zero or negative price values")

    return validation_failures


def calculate_conversion_probabilities(
    data: pd.DataFrame,
    candidate: DgpCandidate,
) -> pd.DataFrame:
    """Calculate structural conversion probabilities for each user.

    Args:
        data: User data containing price, segment, channel, demand,
            and hidden shock.
        candidate: Candidate DGP parameters controlling conversion behavior.

    Returns:
        DataFrame with log-price ratios and conversion probabilities added.
    """
    baseline_conversion_by_segment = {
        "price_sensitive": (candidate.baseline_conversion_price_sensitive),
        "price_resilient": (candidate.baseline_conversion_price_resilient),
    }

    channel_effect_by_channel = {
        "organic": (candidate.channel_conversion_organic),
        "affiliate": (candidate.channel_conversion_affiliate),
        "paid_search": (candidate.channel_conversion_paid_search),
    }

    baseline_conversion = data[SEGMENT_KEY].map(baseline_conversion_by_segment)

    baseline_logits = np.log(baseline_conversion / (1.0 - baseline_conversion))

    data[LOG_PRICE_RATIO_KEY] = np.log(
        data[OBSERVED_PRICE_KEY] / data[REFERENCE_PRICE_KEY]
    )

    structural_beta = data[SEGMENT_KEY].map(STRUCTURAL_BETA_BY_SEGMENT)

    channel_effect = data[CHANNEL_KEY].map(channel_effect_by_channel)

    conversion_logits = (
        baseline_logits
        + structural_beta * data[LOG_PRICE_RATIO_KEY]
        + candidate.demand_conversion_sensitivity
        * (data[DEMAND_INDEX_KEY] - DEMAND_MEAN)
        + candidate.shock_conversion_sensitivity * data[HIDDEN_SHOCK_KEY]
        + channel_effect
    )

    data[CONVERSION_PROB_KEY] = sigmoid(conversion_logits.to_numpy())

    return data


def sample_conversions(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Sample conversion outcomes from structural probabilities.

    Args:
        data: User data containing structural conversion probabilities.
        seed: Master seed used for reproducible conversion outcomes.

    Returns:
        DataFrame with sampled binary conversion outcomes added.
    """
    rng = np.random.default_rng(
        derive_seed(
            seed,
            CONVERSION_OUTCOME_STREAM,
        )
    )

    conversion_draws = rng.random(len(data))

    data[CONVERSION_OUTCOME_KEY] = (
        conversion_draws < data[CONVERSION_PROB_KEY]
    ).astype(int)

    return data


def simulate_candidate(
    candidate: DgpCandidate,
    n_users: int,
    n_weeks: int,
    randomization_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Simulate one candidate data-generating process.

    Args:
        candidate: Candidate DGP parameters used for simulation.
        n_users: Number of synthetic users.
        n_weeks: Number of simulated weeks.
        randomization_rate: Fraction assigned to randomized pricing.
        seed: Master seed.

    Returns:
        Simulated data and any validation failures.
    """
    population = generate_population(
        n_users=n_users,
        n_weeks=n_weeks,
        seed=seed,
    )

    weekly_conditions = generate_weekly_conditions(
        n_weeks=n_weeks,
        demand_std_dev=(candidate.demand_std_dev),
        seed=seed,
    )

    population = assign_arm(
        population=population,
        randomization_rate=randomization_rate,
        seed=seed,
    )

    data = assign_prices(
        population=population,
        weekly_conditions=weekly_conditions,
        candidate=candidate,
        seed=seed,
    )

    validation_failures = validate_prices(data)

    if validation_failures:
        return (
            data,
            validation_failures,
        )

    data = calculate_conversion_probabilities(
        data=data,
        candidate=candidate,
    )

    data = sample_conversions(
        data=data,
        seed=seed,
    )

    return data, []


# ---------------------------------------------------------------------------
# Calibration Diagnostics
# ---------------------------------------------------------------------------

EvaluationPhase = Literal["pilot", "confirmation"]

CELL_POPULATION_KEY: Final = "cell_population"
MEAN_LOG_PRICE_RATIO_KEY: Final = "mean_log_price_ratio"
SD_LOG_PRICE_RATIO_KEY: Final = "sd_log_price_ratio"
OVERLAP_ELIGIBLE_KEY: Final = "overlap_eligible"
OVERLAP_PASS_KEY: Final = "overlap_pass"
OVERLAP_FAILURE_REASON_KEY: Final = "failure_reason"


@dataclass(frozen=True, slots=True)
class RandomizedTargetEstimate:
    """Population target for the observable randomized regression.

    Attributes:
        beta: Estimated population coefficient.
        se: Standard error of the across-seed mean estimate.
        seed_sd: Across-seed standard deviation of the fitted coefficients.
    """

    beta: float
    se: float
    seed_sd: float


def count_nonpositive_prices(data: pd.DataFrame) -> float:
    """Count non-positive prices, or return NaN when prices are unavailable."""
    required_columns = (BASE_PRICE_KEY, OBSERVED_PRICE_KEY)

    if any(column not in data.columns for column in required_columns):
        return math.nan

    values = data[list(required_columns)].to_numpy(dtype=float)
    return float((values <= 0.0).sum())


def count_nonfinite_prices(data: pd.DataFrame) -> float:
    """Count non-finite prices, or return NaN when prices are unavailable."""
    required_columns = (BASE_PRICE_KEY, OBSERVED_PRICE_KEY)

    if any(column not in data.columns for column in required_columns):
        return math.nan

    values = data[list(required_columns)].to_numpy(dtype=float)
    return float((~np.isfinite(values)).sum())


def calculate_overlap_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate treatment-overlap diagnostics across observational cells.

    Args:
        data: Simulated user-level data containing pricing variables.

    Returns:
        Cell-level population, promo rate, and log-price variation. The returned
        frame contains all channel x tier x demand-tercile combinations, including
        cells with zero observed users.
    """
    observational_data = data.loc[~data[IS_RANDOMIZED_KEY]].copy()

    tercile_codes = pd.Series(
        pd.qcut(
            observational_data[DEMAND_INDEX_KEY],
            q=3,
            labels=False,
            duplicates="drop",
        ),
        index=observational_data.index,
        dtype="Int64",
    )

    if tercile_codes.nunique(dropna=True) != 3:
        raise ValueError("expected three demand terciles after duplicate-edge handling")

    observational_data[DEMAND_TERCILE_KEY] = tercile_codes.map(
        {
            0: "low",
            1: "medium",
            2: "high",
        }
    )

    grouped = (
        observational_data.groupby(
            [CHANNEL_KEY, TIER_KEY, DEMAND_TERCILE_KEY],
            observed=True,
        )
        .agg(
            cell_population=(USER_ID_KEY, "size"),
            promo_rate=(PROMO_KEY, "mean"),
            mean_log_price_ratio=(LOG_PRICE_RATIO_KEY, "mean"),
            sd_log_price_ratio=(LOG_PRICE_RATIO_KEY, "std"),
        )
        .reset_index()
    )

    expected_index = pd.MultiIndex.from_product(
        [
            ALLOWED_ACQUISITION_CHANNELS,
            ALLOWED_SUBSCRIPTION_TIERS,
            ("low", "medium", "high"),
        ],
        names=[CHANNEL_KEY, TIER_KEY, DEMAND_TERCILE_KEY],
    )

    diagnostics = (
        grouped.set_index([CHANNEL_KEY, TIER_KEY, DEMAND_TERCILE_KEY])
        .reindex(expected_index)
        .reset_index()
    )

    diagnostics[CELL_POPULATION_KEY] = (
        diagnostics[CELL_POPULATION_KEY].fillna(0).astype(int)
    )

    return diagnostics


def annotate_overlap_diagnostics(
    diagnostics: pd.DataFrame,
    min_cell_population: int,
    min_log_price_sd: float = 0.04,
) -> pd.DataFrame:
    """Annotate overlap cells with eligibility and pass/fail diagnostics.

    Cells below the minimum population are reported but are not eligible to fail
    a candidate.
    """
    annotated = diagnostics.copy()

    eligible = annotated[CELL_POPULATION_KEY] >= min_cell_population
    promo_failure = eligible & (
        (annotated[PROMO_RATE_DIAGNOSTIC_KEY] <= 0.05)
        | (annotated[PROMO_RATE_DIAGNOSTIC_KEY] >= 0.95)
        | annotated[PROMO_RATE_DIAGNOSTIC_KEY].isna()
    )
    variation_failure = eligible & (
        annotated[SD_LOG_PRICE_RATIO_KEY].isna()
        | (annotated[SD_LOG_PRICE_RATIO_KEY] < min_log_price_sd)
    )

    annotated[OVERLAP_ELIGIBLE_KEY] = eligible
    annotated[OVERLAP_PASS_KEY] = eligible & ~promo_failure & ~variation_failure
    annotated[OVERLAP_FAILURE_REASON_KEY] = ""

    annotated.loc[~eligible, OVERLAP_FAILURE_REASON_KEY] = "below_min_population"
    annotated.loc[promo_failure, OVERLAP_FAILURE_REASON_KEY] = (
        "promo_rate_out_of_bounds"
    )
    annotated.loc[
        variation_failure & ~promo_failure,
        OVERLAP_FAILURE_REASON_KEY,
    ] = "insufficient_price_variation"
    annotated.loc[
        variation_failure & promo_failure,
        OVERLAP_FAILURE_REASON_KEY,
    ] = "promo_rate_out_of_bounds;insufficient_price_variation"

    return annotated


def validate_overlap(diagnostics: pd.DataFrame) -> list[str]:
    """Validate only eligible observational overlap cells."""
    validation_failures: list[str] = []

    failed_cells = diagnostics[OVERLAP_ELIGIBLE_KEY] & ~diagnostics[OVERLAP_PASS_KEY]

    if failed_cells.any():
        validation_failures.append(
            f"{int(failed_cells.sum())} eligible cells fail overlap requirements"
        )

    return validation_failures


def calculate_behavioral_diagnostics(data: pd.DataFrame) -> dict[str, float]:
    """Calculate high-level behavioral diagnostics."""
    observational_data = data.loc[~data[IS_RANDOMIZED_KEY]]

    diagnostics = {
        "overall_conversion_rate": float(data[CONVERSION_OUTCOME_KEY].mean())
    }

    for channel in ALLOWED_ACQUISITION_CHANNELS:
        channel_data = observational_data.loc[
            observational_data[CHANNEL_KEY] == channel
        ]

        promo_rate = float(channel_data[PROMO_KEY].mean())
        diagnostics[f"promo_rate_{channel}"] = promo_rate
        diagnostics[f"promo_rate_error_{channel}"] = (
            promo_rate - PROMO_RATE_TARGET_BY_CHANNEL[channel]
        )

    return diagnostics


def calculate_behavioral_criteria(
    diagnostics: dict[str, float],
) -> dict[str, bool]:
    """Calculate hard behavioral criterion flags."""
    conversion_rate = diagnostics["overall_conversion_rate"]
    return {
        "criterion_conversion_rate_pass": 0.30 <= conversion_rate <= 0.50,
    }


def calculate_randomization_diagnostics(data: pd.DataFrame) -> dict[str, float]:
    """Calculate randomized-price independence diagnostics."""
    randomized_data = data.loc[data[IS_RANDOMIZED_KEY]]

    diagnostics: dict[str, float] = {}

    for column in (DEMAND_INDEX_KEY, HIDDEN_SHOCK_KEY, WEEK_KEY):
        correlation = randomized_data[LOG_PRICE_RATIO_KEY].corr(randomized_data[column])
        diagnostics[f"randomized_price_corr_{column}"] = float(correlation)

    for column in (SEGMENT_KEY, CHANNEL_KEY, TIER_KEY):
        group_means = randomized_data.groupby(
            column,
            observed=True,
        )[LOG_PRICE_RATIO_KEY].mean()

        diagnostics[f"randomized_price_spread_{column}"] = float(
            group_means.max() - group_means.min()
        )

    return diagnostics


def validate_randomization_integrity(
    data: pd.DataFrame,
    randomization_rate: float,
) -> list[str]:
    """Validate deterministic randomized-arm invariants."""
    validation_failures: list[str] = []

    randomized_data = data.loc[data[IS_RANDOMIZED_KEY]]
    expected_randomized = math.ceil(randomization_rate * len(data))

    if len(randomized_data) != expected_randomized:
        validation_failures.append(
            "randomized-arm population does not match complete randomization"
        )

    if randomized_data.empty:
        validation_failures.append("randomized arm is empty")
        return validation_failures

    if randomized_data[PROMO_KEY].any():
        validation_failures.append("randomized users received promotions")

    if (randomized_data[PROMO_DEPTH_KEY] != 0.0).any():
        validation_failures.append("randomized users have non-zero promo depth")

    randomized_multipliers = (
        randomized_data[OBSERVED_PRICE_KEY] / randomized_data[REFERENCE_PRICE_KEY]
    ).to_numpy(dtype=float)

    allowed_multipliers = np.asarray(
        RANDOMIZED_PRICE_MULTIPLIERS,
        dtype=float,
    )

    multiplier_is_valid = np.isclose(
        randomized_multipliers[:, None],
        allowed_multipliers[None, :],
        rtol=0.0,
        atol=1e-12,
    ).any(axis=1)

    if not multiplier_is_valid.all():
        validation_failures.append("randomized prices contain unsupported multipliers")

    return validation_failures


# ---------------------------------------------------------------------------
# Statistical Estimation
# ---------------------------------------------------------------------------


def build_regression_design(
    data: pd.DataFrame,
    include_controls: bool,
    include_hidden_shock: bool,
) -> pd.DataFrame:
    """Build a logistic-regression design matrix."""
    design = pd.DataFrame(index=data.index)
    design["const"] = 1.0
    design[LOG_PRICE_RATIO_KEY] = data[LOG_PRICE_RATIO_KEY].astype(float)
    design["segment_price_resilient"] = (data[SEGMENT_KEY] == "price_resilient").astype(
        float
    )
    design["price_x_resilient"] = (
        design[LOG_PRICE_RATIO_KEY] * design["segment_price_resilient"]
    )

    if include_controls:
        design["demand_centered"] = (data[DEMAND_INDEX_KEY] - DEMAND_MEAN).astype(float)
        design["channel_affiliate"] = (data[CHANNEL_KEY] == "affiliate").astype(float)
        design["channel_paid_search"] = (data[CHANNEL_KEY] == "paid_search").astype(
            float
        )
        design["tier_premium"] = (data[TIER_KEY] == "premium").astype(float)

    if include_hidden_shock:
        design[HIDDEN_SHOCK_KEY] = data[HIDDEN_SHOCK_KEY].astype(float)

    return design


def fit_price_effects(
    data: pd.DataFrame,
    randomized: bool,
    include_controls: bool,
    include_hidden_shock: bool,
) -> PriceEffectEstimate:
    """Fit one logistic model with week-clustered standard errors."""
    model_data = data.loc[data[IS_RANDOMIZED_KEY] == randomized].copy()

    if model_data.empty:
        raise ValueError("regression arm is empty")

    design = build_regression_design(
        data=model_data,
        include_controls=include_controls,
        include_hidden_shock=include_hidden_shock,
    )
    outcome = model_data[CONVERSION_OUTCOME_KEY].astype(float)
    groups = model_data[WEEK_KEY].to_numpy()

    result: Any = sm.Logit(
        outcome,
        design,
    ).fit(
        disp=False,
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )

    if not bool(result.mle_retvals.get("converged", False)):
        raise ValueError("logistic regression did not converge")

    params: FloatArray = np.asarray(result.params, dtype=np.float64)
    covariance: FloatArray = np.asarray(result.cov_params(), dtype=np.float64)

    column_names = list(design.columns)
    price_index = column_names.index(LOG_PRICE_RATIO_KEY)
    interaction_index = column_names.index("price_x_resilient")

    sensitive_beta = float(params[price_index])
    sensitive_variance = float(covariance[price_index, price_index])
    sensitive_se = math.sqrt(max(sensitive_variance, 0.0))

    resilient_beta = float(params[price_index] + params[interaction_index])
    resilient_variance = float(
        covariance[price_index, price_index]
        + covariance[interaction_index, interaction_index]
        + 2.0 * covariance[price_index, interaction_index]
    )
    resilient_se = math.sqrt(max(resilient_variance, 0.0))

    return PriceEffectEstimate(
        price_sensitive_beta=sensitive_beta,
        price_sensitive_se=sensitive_se,
        price_resilient_beta=resilient_beta,
        price_resilient_se=resilient_se,
    )


def fit_models(
    data: pd.DataFrame,
    phase: EvaluationPhase,
) -> dict[str, PriceEffectEstimate]:
    """Fit the regression specifications required for one calibration phase.

    Pilot runs fit only the three observational specifications. Confirmation adds
    the observable-randomized and oracle-randomized fits.
    """
    models = {
        "naive": fit_price_effects(
            data=data,
            randomized=False,
            include_controls=False,
            include_hidden_shock=False,
        ),
        "controlled": fit_price_effects(
            data=data,
            randomized=False,
            include_controls=True,
            include_hidden_shock=False,
        ),
        "oracle": fit_price_effects(
            data=data,
            randomized=False,
            include_controls=True,
            include_hidden_shock=True,
        ),
    }

    if phase == "confirmation":
        models["randomized"] = fit_price_effects(
            data=data,
            randomized=True,
            include_controls=True,
            include_hidden_shock=False,
        )
        models["oracle_randomized"] = fit_price_effects(
            data=data,
            randomized=True,
            include_controls=True,
            include_hidden_shock=True,
        )

    return models


def get_segment_estimate(
    estimate: PriceEffectEstimate,
    segment: str,
) -> tuple[float, float]:
    """Return coefficient and standard error for one segment."""
    if segment == "price_sensitive":
        return estimate.price_sensitive_beta, estimate.price_sensitive_se

    if segment == "price_resilient":
        return estimate.price_resilient_beta, estimate.price_resilient_se

    raise ValueError(f"unsupported segment: {segment}")


def distribute_population(total_n_users: int, n_parts: int) -> list[int]:
    """Split a total population as evenly as possible across independent seeds."""
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    base, remainder = divmod(total_n_users, n_parts)
    return [base + (1 if index < remainder else 0) for index in range(n_parts)]


def derive_randomized_targets(
    candidate: DgpCandidate,
    total_n_users: int,
    n_weeks: int,
    seeds: tuple[int, ...],
) -> dict[str, RandomizedTargetEstimate]:
    """Estimate marginalized observable-randomized population targets.

    The total population is distributed across seeds rather than repeated per seed.
    A large number of weeks is used so marginalization is over many independent
    market-shock realizations instead of many copies of only a few weekly shocks.
    """
    if not seeds:
        raise ValueError("at least one randomized-target seed is required")

    users_by_seed = distribute_population(total_n_users, len(seeds))
    betas_by_segment: dict[str, list[float]] = {segment: [] for segment in SEGMENTS}
    ses_by_segment: dict[str, list[float]] = {segment: [] for segment in SEGMENTS}

    for seed, n_users in zip(seeds, users_by_seed, strict=True):
        data, failures = simulate_candidate(
            candidate=candidate,
            n_users=n_users,
            n_weeks=n_weeks,
            randomization_rate=1.0,
            seed=seed,
        )

        if failures:
            raise ValueError(f"randomized-target simulation failed: {failures}")

        estimate = fit_price_effects(
            data=data,
            randomized=True,
            include_controls=True,
            include_hidden_shock=False,
        )

        for segment in SEGMENTS:
            beta, se = get_segment_estimate(estimate, segment)
            betas_by_segment[segment].append(beta)
            ses_by_segment[segment].append(se)

    targets: dict[str, RandomizedTargetEstimate] = {}

    for segment in SEGMENTS:
        betas = np.asarray(betas_by_segment[segment], dtype=np.float64)
        ses = np.asarray(ses_by_segment[segment], dtype=np.float64)
        n_estimates = len(betas)

        target_beta = float(np.mean(betas))
        target_se = float(math.sqrt(float(np.sum(ses**2))) / n_estimates)
        seed_sd = float(np.std(betas, ddof=1)) if n_estimates > 1 else math.nan

        targets[segment] = RandomizedTargetEstimate(
            beta=target_beta,
            se=target_se,
            seed_sd=seed_sd,
        )

    return targets


def calculate_model_diagnostics(
    models: dict[str, PriceEffectEstimate],
    randomized_targets: dict[str, RandomizedTargetEstimate],
) -> dict[str, float]:
    """Flatten estimates, errors, and uncertainty-aware standardized errors."""
    diagnostics: dict[str, float] = {}
    model_names = (
        "naive",
        "controlled",
        "oracle",
        "randomized",
        "oracle_randomized",
    )
    randomized_target_models = {"controlled", "randomized"}

    for model_name in model_names:
        estimate = models.get(model_name)

        for segment in SEGMENTS:
            if estimate is None:
                diagnostics[f"{model_name}_beta_{segment}"] = math.nan
                diagnostics[f"{model_name}_se_{segment}"] = math.nan
                diagnostics[f"{model_name}_error_struct_{segment}"] = math.nan
                diagnostics[f"{model_name}_error_rand_{segment}"] = math.nan
                diagnostics[f"{model_name}_standardized_error_{segment}"] = math.nan
                continue

            beta, se = get_segment_estimate(estimate, segment)
            structural_beta = STRUCTURAL_BETA_BY_SEGMENT[segment]
            randomized_target = randomized_targets[segment]

            error_struct = beta - structural_beta
            error_rand = beta - randomized_target.beta

            if model_name in randomized_target_models:
                target_error = error_rand
                denominator = math.sqrt(se**2 + randomized_target.se**2)
            else:
                target_error = error_struct
                denominator = se

            standardized_error = (
                target_error / denominator
                if denominator > 0.0
                else math.copysign(math.inf, target_error)
            )

            diagnostics[f"{model_name}_beta_{segment}"] = beta
            diagnostics[f"{model_name}_se_{segment}"] = se
            diagnostics[f"{model_name}_error_struct_{segment}"] = error_struct
            diagnostics[f"{model_name}_error_rand_{segment}"] = error_rand
            diagnostics[f"{model_name}_standardized_error_{segment}"] = (
                standardized_error
            )

    return diagnostics


def calculate_effect_gaps(
    models: dict[str, PriceEffectEstimate],
    randomized_targets: dict[str, RandomizedTargetEstimate],
) -> dict[str, float]:
    """Calculate headline causal calibration gaps without phase assumptions."""
    diagnostics: dict[str, float] = {}

    naive_estimate = models.get("naive")
    controlled_estimate = models.get("controlled")
    oracle_estimate = models.get("oracle")
    randomized_estimate = models.get("randomized")
    oracle_randomized_estimate = models.get("oracle_randomized")

    for segment in SEGMENTS:
        structural_beta = STRUCTURAL_BETA_BY_SEGMENT[segment]
        randomized_beta = randomized_targets[segment].beta

        diagnostics[f"{segment}_beta_rand_minus_beta_struct"] = (
            randomized_beta - structural_beta
        )

        if naive_estimate is None or controlled_estimate is None:
            diagnostics[f"{segment}_naive_minus_controlled"] = math.nan
            diagnostics[f"{segment}_controlled_minus_beta_rand"] = math.nan
            diagnostics[f"{segment}_controlled_minus_oracle"] = math.nan
            diagnostics[f"{segment}_randomized_minus_oracle_randomized"] = math.nan
            diagnostics[f"{segment}_corrected_hidden_confounding"] = math.nan
            continue

        naive_beta, _ = get_segment_estimate(
            naive_estimate,
            segment,
        )

        controlled_beta, _ = get_segment_estimate(
            controlled_estimate,
            segment,
        )

        diagnostics[f"{segment}_naive_minus_controlled"] = naive_beta - controlled_beta

        diagnostics[f"{segment}_controlled_minus_beta_rand"] = (
            controlled_beta - randomized_beta
        )

        oracle_beta = math.nan

        if oracle_estimate is None:
            diagnostics[f"{segment}_controlled_minus_oracle"] = math.nan
        else:
            oracle_beta, _ = get_segment_estimate(
                oracle_estimate,
                segment,
            )

            diagnostics[f"{segment}_controlled_minus_oracle"] = (
                controlled_beta - oracle_beta
            )

        if randomized_estimate is None or oracle_randomized_estimate is None:
            diagnostics[f"{segment}_randomized_minus_oracle_randomized"] = math.nan
            diagnostics[f"{segment}_corrected_hidden_confounding"] = math.nan
            continue

        randomized_beta_hat, _ = get_segment_estimate(
            randomized_estimate,
            segment,
        )

        oracle_randomized_beta, _ = get_segment_estimate(
            oracle_randomized_estimate,
            segment,
        )

        randomized_oracle_gap = randomized_beta_hat - oracle_randomized_beta

        diagnostics[f"{segment}_randomized_minus_oracle_randomized"] = (
            randomized_oracle_gap
        )

        if oracle_estimate is None:
            diagnostics[f"{segment}_corrected_hidden_confounding"] = math.nan
            continue

        diagnostics[f"{segment}_corrected_hidden_confounding"] = (
            controlled_beta - oracle_beta - randomized_oracle_gap
        )

    return diagnostics


def calculate_regression_criteria(
    models: dict[str, PriceEffectEstimate],
    randomized_targets: dict[str, RandomizedTargetEstimate],
    tau_se: float,
    phase: EvaluationPhase,
) -> dict[str, bool]:
    """Calculate criterion flags relevant to the requested calibration phase."""
    criteria: dict[str, bool] = {}

    for segment in SEGMENTS:
        structural_beta = STRUCTURAL_BETA_BY_SEGMENT[segment]
        randomized_target = randomized_targets[segment]

        naive_beta, _ = get_segment_estimate(
            models["naive"],
            segment,
        )

        controlled_beta, _ = get_segment_estimate(
            models["controlled"],
            segment,
        )

        oracle_beta, oracle_se = get_segment_estimate(
            models["oracle"],
            segment,
        )

        naive_error_struct = naive_beta - structural_beta
        controlled_error_struct = controlled_beta - structural_beta

        paired_hidden_gap = controlled_beta - oracle_beta

        criteria[f"criterion_naive_bias_{segment}_pass"] = naive_error_struct >= 0.50

        criteria[f"criterion_controlled_improves_{segment}_pass"] = abs(
            controlled_error_struct
        ) < abs(naive_error_struct)

        criteria[f"criterion_paired_hidden_confounding_{segment}_pass"] = (
            paired_hidden_gap >= 0.20
        )

        if phase != "confirmation":
            continue

        randomized_beta_hat, randomized_se = get_segment_estimate(
            models["randomized"],
            segment,
        )

        oracle_randomized_beta, oracle_randomized_se = get_segment_estimate(
            models["oracle_randomized"],
            segment,
        )

        randomized_oracle_gap = randomized_beta_hat - oracle_randomized_beta

        corrected_hidden_gap = paired_hidden_gap - randomized_oracle_gap

        criteria[f"criterion_corrected_hidden_confounding_{segment}_pass"] = (
            corrected_hidden_gap >= 0.20
        )

        criteria[f"criterion_oracle_observational_{segment}_pass"] = (
            abs(oracle_beta - structural_beta) <= 3.0 * oracle_se
            and oracle_se <= tau_se
        )

        randomized_difference_se = math.sqrt(randomized_se**2 + randomized_target.se**2)

        criteria[f"criterion_randomized_{segment}_pass"] = (
            abs(randomized_beta_hat - randomized_target.beta)
            <= 3.0 * randomized_difference_se
            and randomized_se <= tau_se
        )

        criteria[f"criterion_oracle_randomized_{segment}_pass"] = (
            abs(oracle_randomized_beta - structural_beta) <= 3.0 * oracle_randomized_se
            and oracle_randomized_se <= tau_se
        )

    return criteria


def regression_failures_for_phase(
    criteria: dict[str, bool],
    phase: EvaluationPhase,
) -> list[str]:
    """Return statistical failures that gate the requested phase."""
    required_prefixes: tuple[str, ...] = (
        "criterion_naive_bias_",
        "criterion_controlled_improves_",
        "criterion_paired_hidden_confounding_",
    )

    if phase == "confirmation":
        required_prefixes = required_prefixes + (
            "criterion_corrected_hidden_confounding_",
            "criterion_oracle_observational_",
            "criterion_randomized_",
            "criterion_oracle_randomized_",
        )

    failures: list[str] = []

    for criterion_name, passed in criteria.items():
        if criterion_name.startswith(required_prefixes) and not passed:
            failures.append(criterion_name.removesuffix("_pass"))

    return failures


# ---------------------------------------------------------------------------
# Candidate Evaluation
# ---------------------------------------------------------------------------


def candidate_record(candidate: DgpCandidate) -> dict[str, object]:
    """Convert all swept candidate parameters into artifact columns."""
    return dict(asdict(candidate))


def randomized_target_record(
    targets: dict[str, RandomizedTargetEstimate],
) -> dict[str, float]:
    """Flatten randomized target estimates for artifact output."""
    record: dict[str, float] = {}

    for segment in SEGMENTS:
        estimate = targets[segment]
        record[f"randomized_target_beta_{segment}"] = estimate.beta
        record[f"randomized_target_se_{segment}"] = estimate.se
        record[f"randomized_target_seed_sd_{segment}"] = estimate.seed_sd
        record[f"noncollapsibility_attenuation_{segment}"] = (
            estimate.beta - STRUCTURAL_BETA_BY_SEGMENT[segment]
        )

    return record


def prepare_overlap_artifact(
    diagnostics: pd.DataFrame,
    candidate_id: str,
    seed: int,
) -> pd.DataFrame:
    """Prepare cell diagnostics for the detailed overlap artifact."""
    artifact = diagnostics.copy()
    artifact.insert(0, "seed", seed)
    artifact.insert(0, "candidate_id", candidate_id)
    return artifact[
        [
            "candidate_id",
            "seed",
            CHANNEL_KEY,
            TIER_KEY,
            DEMAND_TERCILE_KEY,
            CELL_POPULATION_KEY,
            PROMO_RATE_DIAGNOSTIC_KEY,
            MEAN_LOG_PRICE_RATIO_KEY,
            SD_LOG_PRICE_RATIO_KEY,
            OVERLAP_ELIGIBLE_KEY,
            OVERLAP_PASS_KEY,
            OVERLAP_FAILURE_REASON_KEY,
        ]
    ]


def evaluate_candidate(
    candidate: DgpCandidate,
    randomized_targets: dict[str, RandomizedTargetEstimate],
    n_users: int,
    n_weeks: int,
    randomization_rate: float,
    seed: int,
    min_cell_population: int,
    phase: EvaluationPhase,
    min_log_price_sd: float = 0.04,
    tau_se: float = 0.15,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate one candidate under one seed and return both artifacts."""
    result = candidate_record(candidate)
    result.update(
        {
            "seed": seed,
            "evaluation_phase": phase,
            "n_users": n_users,
            "n_weeks": n_weeks,
            "randomization_rate": randomization_rate,
        }
    )
    result.update(randomized_target_record(randomized_targets))

    data, failures = simulate_candidate(
        candidate=candidate,
        n_users=n_users,
        n_weeks=n_weeks,
        randomization_rate=randomization_rate,
        seed=seed,
    )

    result["n_nonpositive_prices"] = count_nonpositive_prices(data)
    result["n_nonfinite_prices"] = count_nonfinite_prices(data)
    result["criterion_price_valid_pass"] = not failures

    if failures:
        result["passed"] = False
        result["verdict"] = "fail"
        result["failures"] = " | ".join(failures)
        return result, pd.DataFrame()

    try:
        overlap = calculate_overlap_diagnostics(data)
    except ValueError as error:
        result["criterion_overlap_pass"] = False
        result["passed"] = False
        result["verdict"] = "fail"
        result["failures"] = f"overlap calculation failed: {error}"
        return result, pd.DataFrame()

    overlap = annotate_overlap_diagnostics(
        diagnostics=overlap,
        min_cell_population=min_cell_population,
        min_log_price_sd=min_log_price_sd,
    )
    overlap_failures = validate_overlap(overlap)
    failures.extend(overlap_failures)

    eligible_cells = overlap[OVERLAP_ELIGIBLE_KEY]
    failed_eligible_cells = eligible_cells & ~overlap[OVERLAP_PASS_KEY]
    passing_eligible_cells = eligible_cells & overlap[OVERLAP_PASS_KEY]

    result["n_valid_cells"] = int(passing_eligible_cells.sum())
    result["n_failed_overlap_cells"] = int(failed_eligible_cells.sum())
    result["n_ineligible_cells"] = int((~eligible_cells).sum())
    result["criterion_overlap_pass"] = not overlap_failures

    eligible_overlap = overlap.loc[eligible_cells]
    if eligible_overlap.empty:
        result["min_cell_promo_rate"] = math.nan
        result["max_cell_promo_rate"] = math.nan
        result["min_cell_log_price_sd"] = math.nan
    else:
        result["min_cell_promo_rate"] = float(
            eligible_overlap[PROMO_RATE_DIAGNOSTIC_KEY].min()
        )
        result["max_cell_promo_rate"] = float(
            eligible_overlap[PROMO_RATE_DIAGNOSTIC_KEY].max()
        )
        result["min_cell_log_price_sd"] = float(
            eligible_overlap[SD_LOG_PRICE_RATIO_KEY].min()
        )

    behavioral_diagnostics = calculate_behavioral_diagnostics(data)
    result.update(behavioral_diagnostics)
    behavioral_criteria = calculate_behavioral_criteria(behavioral_diagnostics)
    result.update(behavioral_criteria)

    if not behavioral_criteria["criterion_conversion_rate_pass"]:
        failures.append("criterion_conversion_rate")

    randomization_diagnostics = calculate_randomization_diagnostics(data)
    result.update(randomization_diagnostics)
    randomization_failures = validate_randomization_integrity(
        data=data,
        randomization_rate=randomization_rate,
    )
    result["criterion_randomization_integrity_pass"] = not randomization_failures
    failures.extend(randomization_failures)

    try:
        models = fit_models(data, phase)
    except (
        ValueError,
        np.linalg.LinAlgError,
        PerfectSeparationError,
    ) as error:
        failures.append(f"regression fitting failed: {error}")
        result["passed"] = False
        result["verdict"] = "fail"
        result["failures"] = " | ".join(failures)
        return result, prepare_overlap_artifact(overlap, candidate.candidate_id, seed)

    result.update(calculate_model_diagnostics(models, randomized_targets))
    result.update(calculate_effect_gaps(models, randomized_targets))

    regression_criteria = calculate_regression_criteria(
        models=models,
        randomized_targets=randomized_targets,
        tau_se=tau_se,
        phase=phase,
    )
    result.update(regression_criteria)
    failures.extend(regression_failures_for_phase(regression_criteria, phase))

    result["passed"] = not failures
    result["verdict"] = "pass" if not failures else "fail"
    result["failures"] = " | ".join(failures)

    return result, prepare_overlap_artifact(overlap, candidate.candidate_id, seed)


# ---------------------------------------------------------------------------
# Calibration Sweep
# ---------------------------------------------------------------------------


def run_calibration(
    candidates: list[DgpCandidate],
    seeds: tuple[int, ...],
    n_users: int,
    n_weeks: int,
    randomization_rate: float,
    min_cell_population: int,
    target_total_n_users: int,
    target_n_weeks: int,
    target_seeds: tuple[int, ...],
    output_path: Path,
    overlap_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pilot calibration sweep across fixed seeds."""
    records: list[dict[str, object]] = []
    overlap_records: list[pd.DataFrame] = []

    for candidate in candidates:
        randomized_targets = derive_randomized_targets(
            candidate=candidate,
            total_n_users=target_total_n_users,
            n_weeks=target_n_weeks,
            seeds=target_seeds,
        )

        for seed in seeds:
            record, overlap = evaluate_candidate(
                candidate=candidate,
                randomized_targets=randomized_targets,
                n_users=n_users,
                n_weeks=n_weeks,
                randomization_rate=randomization_rate,
                seed=seed,
                min_cell_population=min_cell_population,
                phase="pilot",
            )
            records.append(record)

            if not overlap.empty:
                overlap_records.append(overlap)

    results = pd.DataFrame(records)
    overlap_results = (
        pd.concat(overlap_records, ignore_index=True)
        if overlap_records
        else pd.DataFrame()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    overlap_results.to_csv(overlap_output_path, index=False)

    return results, overlap_results


def summarize_candidate_passes(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether candidates passed every pilot seed."""
    return (
        results.groupby("candidate_id", observed=True)
        .agg(
            seeds_evaluated=("seed", "count"),
            seeds_passed=("passed", "sum"),
            passed_all_seeds=("passed", "all"),
        )
        .reset_index()
    )


def confirm_candidate(
    candidate: DgpCandidate,
    seeds: tuple[int, ...],
    n_users: int,
    n_weeks: int,
    randomization_rate: float,
    min_cell_population: int,
    target_total_n_users: int,
    target_n_weeks: int,
    target_seeds: tuple[int, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, RandomizedTargetEstimate],
]:
    """Run full-size certification for one selected candidate.

    The caller should use at least one million randomized-target users in total
    across multiple seeds and thousands of weeks.
    """
    randomized_targets = derive_randomized_targets(
        candidate=candidate,
        total_n_users=target_total_n_users,
        n_weeks=target_n_weeks,
        seeds=target_seeds,
    )

    records: list[dict[str, object]] = []
    overlap_records: list[pd.DataFrame] = []

    for seed in seeds:
        record, overlap = evaluate_candidate(
            candidate=candidate,
            randomized_targets=randomized_targets,
            n_users=n_users,
            n_weeks=n_weeks,
            randomization_rate=randomization_rate,
            seed=seed,
            min_cell_population=min_cell_population,
            phase="confirmation",
        )
        records.append(record)

        if not overlap.empty:
            overlap_records.append(overlap)

    results = pd.DataFrame(records)
    overlap_results = (
        pd.concat(overlap_records, ignore_index=True)
        if overlap_records
        else pd.DataFrame()
    )

    return results, overlap_results, randomized_targets


# ---------------------------------------------------------------------------
# Candidate Configuration
# ---------------------------------------------------------------------------


def build_candidates() -> list[DgpCandidate]:
    """Build a small interpretable pilot candidate set.

    The anchor is varied along measured- and hidden-confounding dimensions while
    keeping the remaining behavioral parameters fixed. The first sweep is meant to
    reveal which direction needs tuning rather than exhaustively search the space.
    """

    def make_candidate(
        candidate_id: str,
        promo_demand_sensitivity: float,
        promo_shock_sensitivity: float,
        price_demand_sensitivity: float,
        price_shock_sensitivity: float,
        demand_conversion_sensitivity: float,
        shock_conversion_sensitivity: float,
    ) -> DgpCandidate:
        return DgpCandidate(
            candidate_id=candidate_id,
            promo_bias_organic=-1.73,
            promo_bias_affiliate=-0.85,
            promo_bias_paid_search=0.0,
            promo_demand_sensitivity=promo_demand_sensitivity,
            promo_shock_sensitivity=promo_shock_sensitivity,
            price_demand_sensitivity=price_demand_sensitivity,
            price_shock_sensitivity=price_shock_sensitivity,
            baseline_conversion_price_sensitive=0.35,
            baseline_conversion_price_resilient=0.40,
            demand_conversion_sensitivity=demand_conversion_sensitivity,
            shock_conversion_sensitivity=shock_conversion_sensitivity,
            channel_conversion_organic=-0.10,
            channel_conversion_affiliate=0.0,
            channel_conversion_paid_search=0.10,
            demand_std_dev=0.10,
        )

    return [
        make_candidate(
            candidate_id="anchor",
            promo_demand_sensitivity=4.0,
            promo_shock_sensitivity=0.4,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.02,
            demand_conversion_sensitivity=4.0,
            shock_conversion_sensitivity=0.08,
        ),
        make_candidate(
            candidate_id="measured_weaker",
            promo_demand_sensitivity=3.0,
            promo_shock_sensitivity=0.4,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.02,
            demand_conversion_sensitivity=3.0,
            shock_conversion_sensitivity=0.08,
        ),
        make_candidate(
            candidate_id="measured_stronger",
            promo_demand_sensitivity=5.0,
            promo_shock_sensitivity=0.4,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.02,
            demand_conversion_sensitivity=5.0,
            shock_conversion_sensitivity=0.08,
        ),
        make_candidate(
            candidate_id="hidden_weaker",
            promo_demand_sensitivity=4.0,
            promo_shock_sensitivity=0.3,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.015,
            demand_conversion_sensitivity=4.0,
            shock_conversion_sensitivity=0.05,
        ),
        make_candidate(
            candidate_id="hidden_stronger",
            promo_demand_sensitivity=4.0,
            promo_shock_sensitivity=0.5,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.03,
            demand_conversion_sensitivity=4.0,
            shock_conversion_sensitivity=0.12,
        ),
        make_candidate(
            candidate_id="balanced",
            promo_demand_sensitivity=5.0,
            promo_shock_sensitivity=0.45,
            price_demand_sensitivity=0.20,
            price_shock_sensitivity=0.025,
            demand_conversion_sensitivity=5.0,
            shock_conversion_sensitivity=0.10,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the pilot DGP calibration sweep."""
    candidates = build_candidates()

    calibration_seeds: tuple[int, ...] = (11, 22, 33)

    # Pilot target: cheap enough to derive per candidate, but spread over many
    # independent weeks so the target is not dominated by 24 realized shocks.
    pilot_target_seeds: tuple[int, ...] = (101,)

    results, _ = run_calibration(
        candidates=candidates,
        seeds=calibration_seeds,
        n_users=60_000,
        n_weeks=1_000,
        randomization_rate=0.10,
        min_cell_population=100,
        target_total_n_users=200_000,
        target_n_weeks=2_000,
        target_seeds=pilot_target_seeds,
        output_path=Path("artifacts/dgp_calibration.csv"),
        overlap_output_path=Path("artifacts/dgp_overlap_cells.csv"),
    )

    summary = summarize_candidate_passes(results)
    print(summary.to_string(index=False))

    # Full-size confirmation is intentionally separate. For the selected finalist,
    # call confirm_candidate() with >= 1_000_000 target users in total across three
    # target seeds, thousands of target weeks, and the final fixture population /
    # randomization rate. Freeze those values in docs/dgp.md after confirmation.


if __name__ == "__main__":
    main()
