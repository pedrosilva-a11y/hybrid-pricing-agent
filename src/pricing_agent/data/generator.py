"""Synthetic pricing data generation."""

import math
from typing import Final, TypedDict

import numpy as np
from numpy.random import SeedSequence

from pricing_agent.data.config import (
    ACQUISITION_CHANNELS,
    DEMAND_HIGH_BOUND,
    DEMAND_LOW_BOUND,
    DEMAND_MEAN,
    DEMAND_STD_DEV,
    HIDDEN_SHOCK_MEAN,
    HIDDEN_SHOCK_STD_DEV,
    PRICE_DEMAND_SENSITIVITY,
    PRICE_SHOCK_SENSITIVITY,
    PROMO_BIAS_BY_CHANNEL,
    PROMO_DEMAND_SENSITIVITY,
    PROMO_DEPTHS,
    PROMO_SHOCK_SENSITIVITY,
    RANDOMIZED_PRICE_MULTIPLIERS,
    REFERENCE_PRICE_BY_TIER,
    SUBSCRIPTION_TIERS,
    PricingDataConfig,
)

# Random stream identifiers
USER_GENERATION_STREAM: Final = 0
WEEKLY_DEMAND_STREAM: Final = 1
PRICE_ASSIGNMENT_STREAM: Final = 2
CONVERSION_OUTCOME_STREAM: Final = 3
CHURN_OUTCOME_STREAM: Final = 4  # Legacy / reserved
HIDDEN_SHOCK_STREAM: Final = 5
PROMO_ASSIGNMENT_STREAM: Final = 6
PROMO_DEPTH_STREAM: Final = 7
RANDOMIZED_ARM_STREAM: Final = 8

# User Population Global Variables
CHANNEL_KEY: Final = "channel"
SEGMENT_KEY: Final = "segment"
SIGNUP_WEEK_KEY: Final = "signup_week"
TIER_KEY: Final = "tier"
USER_ID_KEY: Final = "user_id"

# Weekly Demand Global Variables
DEMAND_INDEX_KEY: Final = "demand_index"
HIDDEN_SHOCK_KEY: Final = "hidden_shock"
WEEK_KEY: Final = "week"

# Weekly Base Prices Global Variables
PRICE_KEY: Final = "price"
REFERENCE_PRICE: Final = 19.99

# Promotion Assignment Global Variable
PROMO_KEY: Final = "promo"

# Promotion Depth Global Variable
PROMO_DEPTH_KEY: Final = "promo_depth"

# Assign User Prices Global Variables
IS_RANDOMIZED_KEY: Final = "is_randomized"
OBSERVED_PRICE_KEY: Final = "observed_price"

# Conversion Probabilities Global Variables
DEMAND_CONVERSION_SENSITIVITY: Final = 1.0
CONVERSION_PROB_KEY: Final = "conversion_probability"

# Conversion Outcome Global Variable
CONVERSION_OUTCOME_KEY: Final = "conversion_outcome"

# Acquisition Cost Mapping Global Variables
ACQUISITION_COST_KEY: Final = "acquisition_cost"
ACQ_COST_MAPPING: Final = {
    "organic": 3.00,
    "affiliate": 10.00,
    "paid_search": 20.00,
}

# Marginal Cost Mapping Global Variables
MARGINAL_COST_KEY: Final = "marginal_cost"
TIER_MARGINAL_COST_MAPPING: Final = {
    "basic": 4.00,
    "premium": 8.00,
}

# Churn Computation Global Variables
CHURN_PRICE_SENSITIVITY: Final = 1.5
CHURN_PROB_KEY: Final = "churn_probability"
TIER_CHURN_ADJUSTMENT: Final = {
    "basic": 0.0,
    "premium": -0.30,
}

# Churn Outcome Global Variable
CHURN_OUTCOME_KEY: Final = "churn_outcome"


class UserPopulation(TypedDict):
    """Column-oriented synthetic user population.

    Attributes:
        user_id: Unique user identifier.
        signup_week: Week the user enters the simulated population.
        segment: Customer segment assigned to the user.
        channel: Acquisition channel through which the user was acquired.
        tier: Subscription tier assigned to the user.
    """

    user_id: list[int]
    signup_week: list[int]
    segment: list[str]
    channel: list[str]
    tier: list[str]


class WeeklyConditions(TypedDict):
    """Column-oriented synthetic weekly market conditions.

    Attributes:
        week: Unique week index in the simulated period.
        demand_index: Relative level of market demand for a given week.
            A value of 1.0 represents normal reference demand. Values above 1.0 indicate
            stronger-than-normal demand, while values below 1.0 indicate
            weaker-than-normal demand.
        hidden_shock: Unobserved week-level market shock sampled independently from the
            demand index. Positive and negative values represent latent weekly factors
            that affect pricing and conversion but are excluded from the modeling path.
    """

    week: list[int]
    demand_index: list[float]
    hidden_shock: list[float]


class RandomizedArmAssignments(TypedDict):
    """Column-oriented randomized experiment membership assignments.

    Attributes:
        user_id: Unique user identifier.
        is_randomized: Whether the user belongs to the randomized pricing arm.
    """

    user_id: list[int]
    is_randomized: list[bool]


class WeeklyBasePrices(TypedDict):
    """Column-oriented weekly base prices by subscription tier.

    Attributes:
        week: Week associated with the generated base price.
        tier: Subscription tier determining the reference price.
        price: Historical base price implied by the weekly market conditions.
    """

    week: list[int]
    tier: list[str]
    price: list[float]


class PromoAssignments(TypedDict):
    """Column-oriented observational promotion assignments.

    Attributes:
        user_id: Unique user identifier.
        promo: Whether the user receives an observational promotion.
    """

    user_id: list[int]
    promo: list[bool]


class PromoDepthAssignments(TypedDict):
    """Column-oriented observational promotion depth assignments.

    Attributes:
        user_id: Unique user identifier.
        promo_depth: Discount fraction assigned to the user. Users without an
            observational promotion receive a depth of zero.
    """

    user_id: list[int]
    promo_depth: list[float]


class AssignedUserPrices(TypedDict):
    """Column-oriented synthetic user price assignments.

    Attributes:
        user_id: Unique user identifier.
        observed_price: Price assigned to the user. For non-randomized users, this
            matches the tier-specific base price for the user's signup week. For
            randomized users, this is assigned independently of the historical
            pricing policy.
        is_randomized: Whether the user's observed price was assigned through the
            randomized pricing experiment.
    """

    user_id: list[int]
    observed_price: list[float]
    is_randomized: list[bool]


class ConversionProbabilities(TypedDict):
    """Column-oriented synthetic user conversion probabilities.

    Attributes:
        user_id: Unique user identifier.
        conversion_probability: Probability that the user converts under the
            assigned price and weekly demand conditions.
    """

    user_id: list[int]
    conversion_probability: list[float]


class ConversionOutcomes(TypedDict):
    """Column-oriented synthetic user conversion outcomes.

    Attributes:
        user_id: Unique user identifier.
        conversion_outcome: Whether the user converted after receiving the assigned
            price under the simulated market conditions.
    """

    user_id: list[int]
    conversion_outcome: list[bool]


class UserAcquisitionCosts(TypedDict):
    """Column-oriented synthetic user acquisition costs.

    Attributes:
        user_id: Unique user identifier.
        acquisition_cost: Cost incurred to acquire the user.
    """

    user_id: list[int]
    acquisition_cost: list[float]


class UserMarginalCosts(TypedDict):
    """Column-oriented synthetic user marginal costs.

    Attributes:
        user_id: Unique user identifier.
        marginal_cost: Incremental cost of serving the user.
    """

    user_id: list[int]
    marginal_cost: list[float]


class ChurnProbabilities(TypedDict):
    """Column-oriented synthetic user churn probabilities.

    Attributes:
        user_id: Unique user identifier.
        churn_probability: Probability that the user churns under the
            simulated pricing and subscription conditions.
    """

    user_id: list[int]
    churn_probability: list[float]


class ChurnOutcomes(TypedDict):
    """Column-oriented synthetic user churn outcomes.

    Attributes:
        user_id: Unique user identifier.
        churn_outcome: Whether the user churned under the simulated pricing
            and subscription conditions.
    """

    user_id: list[int]
    churn_outcome: list[bool]


def derive_seed(master_seed: int, stream: int) -> int:
    """Derive a deterministic seed for an independent random stream."""
    seed_sequence = SeedSequence([master_seed, stream])
    return int(seed_sequence.generate_state(1)[0])


def generate_users(config: PricingDataConfig) -> UserPopulation:
    """Generate the synthetic user population.

    Args:
        config: Generation configuration to determine conditions.

    Returns:
        Mapping of user attributes to their generated column values.
    """
    rng = np.random.default_rng(derive_seed(config.seed, USER_GENERATION_STREAM))

    synthetic_data: UserPopulation = {
        USER_ID_KEY: [],
        SIGNUP_WEEK_KEY: [],
        SEGMENT_KEY: [],
        CHANNEL_KEY: [],
        TIER_KEY: [],
    }

    segment_names = [segment.name.strip() for segment in config.segments]

    for user_id in range(config.n_users):
        synthetic_data[USER_ID_KEY].append(user_id)
        synthetic_data[SIGNUP_WEEK_KEY].append(int(rng.integers(config.n_weeks)))
        synthetic_data[SEGMENT_KEY].append(str(rng.choice(segment_names)))
        synthetic_data[CHANNEL_KEY].append(str(rng.choice(ACQUISITION_CHANNELS)))
        synthetic_data[TIER_KEY].append(str(rng.choice(SUBSCRIPTION_TIERS)))

    return synthetic_data


def generate_weekly_conditions(config: PricingDataConfig) -> WeeklyConditions:
    """Generate reproducible weekly market conditions.

    Args:
        config: Generation configuration defining the number of simulated weeks
            and the random seed.

    Returns:
        Column-oriented weekly market conditions containing one demand index and
            hidden shock per week.
    """
    n_weeks = config.n_weeks

    demand_rng = np.random.default_rng(derive_seed(config.seed, WEEKLY_DEMAND_STREAM))
    shock_rng = np.random.default_rng(derive_seed(config.seed, HIDDEN_SHOCK_STREAM))

    raw_demand = demand_rng.normal(loc=DEMAND_MEAN, scale=DEMAND_STD_DEV, size=n_weeks)
    simulated_demand = np.clip(raw_demand, DEMAND_LOW_BOUND, DEMAND_HIGH_BOUND)

    hidden_shock = shock_rng.normal(
        loc=HIDDEN_SHOCK_MEAN,
        scale=HIDDEN_SHOCK_STD_DEV,
        size=n_weeks,
    )

    return {
        WEEK_KEY: list(range(n_weeks)),
        DEMAND_INDEX_KEY: simulated_demand.tolist(),
        HIDDEN_SHOCK_KEY: hidden_shock.tolist(),
    }


def calculate_base_price(tier: str, demand_index: float, hidden_shock: float) -> float:
    """Calculate the historical base price for a tier and weekly conditions.

    Args:
        tier: Subscription tier determining the reference price.
        demand_index: Relative weekly market demand.
        hidden_shock: Latent weekly market shock affecting historical pricing.

    Returns:
        Base price implied by the calibrated historical pricing policy.
    """
    reference_price = REFERENCE_PRICE_BY_TIER[tier]

    return reference_price * (
        1.0
        + PRICE_DEMAND_SENSITIVITY * (demand_index - DEMAND_MEAN)
        + PRICE_SHOCK_SENSITIVITY * hidden_shock
    )


def generate_weekly_base_prices(
    weekly_conditions: WeeklyConditions,
) -> WeeklyBasePrices:
    """Generate tier-specific weekly base prices.

    Args:
        weekly_conditions: Weekly market conditions containing observed demand and
            latent market shocks.

    Returns:
        Column-oriented weekly base prices containing one price per week and tier.
    """
    weeks: list[int] = []
    tiers: list[str] = []
    base_prices: list[float] = []

    for week, demand_index, hidden_shock in zip(
        weekly_conditions[WEEK_KEY],
        weekly_conditions[DEMAND_INDEX_KEY],
        weekly_conditions[HIDDEN_SHOCK_KEY],
        strict=True,
    ):
        for tier in SUBSCRIPTION_TIERS:
            weeks.append(week)
            tiers.append(tier)
            base_prices.append(
                calculate_base_price(
                    tier=tier,
                    demand_index=demand_index,
                    hidden_shock=hidden_shock,
                )
            )

    return {
        WEEK_KEY: weeks,
        TIER_KEY: tiers,
        PRICE_KEY: base_prices,
    }


def calculate_promo_probability(
    channel: str,
    demand_index: float,
    hidden_shock: float,
) -> float:
    """Calculate the probability of receiving an observational promotion.

    Args:
        channel: User acquisition channel affecting promotion propensity.
        demand_index: Relative week market demand.
        hidden_shock: Latent weekly market shock affecting promotion propensity.

    Returns:
        Probability that the user receives a promotion.
    """
    logit_promo = (
        PROMO_BIAS_BY_CHANNEL[channel]
        - PROMO_DEMAND_SENSITIVITY * (demand_index - DEMAND_MEAN)
        - PROMO_SHOCK_SENSITIVITY * hidden_shock
    )

    return 1.0 / (1.0 + math.exp(-logit_promo))


def assign_promotions(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    weekly_conditions: WeeklyConditions,
    randomized_arms: RandomizedArmAssignments,
) -> PromoAssignments:
    """Assign observational promotions to synthetic users.

    Args:
        config: Generation configuration containing the master random seed.
        generated_users: Synthetic user population containing the signup weeks and
            acquisition channels.
        weekly_conditions: Weekly market conditions affecting promotion propensity.
        randomized_arms: User-level randomized pricing arm assignments.

    Returns:
        Column-oriented promotion assignments containing each user identifier and
        whether the user receives an observational promotion.
    """
    rng = np.random.default_rng(derive_seed(config.seed, PROMO_ASSIGNMENT_STREAM))

    draws = rng.random(len(generated_users[USER_ID_KEY]))

    promotions: list[bool] = []

    for index, draw in enumerate(draws):
        if randomized_arms[IS_RANDOMIZED_KEY][index]:
            promotions.append(False)
            continue

        signup_week = generated_users[SIGNUP_WEEK_KEY][index]
        channel = generated_users[CHANNEL_KEY][index]

        promo_prob = calculate_promo_probability(
            channel=channel,
            demand_index=weekly_conditions[DEMAND_INDEX_KEY][signup_week],
            hidden_shock=weekly_conditions[HIDDEN_SHOCK_KEY][signup_week],
        )

        promotions.append(bool(draw < promo_prob))

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        PROMO_KEY: promotions,
    }


def assign_promo_depths(
    config: PricingDataConfig,
    promo_assignments: PromoAssignments,
) -> PromoDepthAssignments:
    """Assign promotion depths to users receiving observational promotions.

    Args:
        config: Generation configuration containing the master random seed.
        promo_assignments: User-level observational promotion assignments.

    Returns:
        Column-oriented promotion depths containing one depth per user.
    """
    rng = np.random.default_rng(derive_seed(config.seed, PROMO_DEPTH_STREAM))

    sampled_depths = rng.choice(PROMO_DEPTHS, size=len(promo_assignments[USER_ID_KEY]))

    promo_depths = [
        float(depth) if promo else 0.0
        for promo, depth in zip(
            promo_assignments[PROMO_KEY],
            sampled_depths,
            strict=True,
        )
    ]

    return {
        USER_ID_KEY: promo_assignments[USER_ID_KEY].copy(),
        PROMO_DEPTH_KEY: promo_depths,
    }


def assign_randomized_arms(
    config: PricingDataConfig,
    generated_users: UserPopulation,
) -> RandomizedArmAssignments:
    """Assign users to the randomized pricing arm.

    Args:
        config: Generation configuration defining the randomization rate and
            master random seed.
        generated_users: Synthetic user population containing the users eligible
            for randomized-arm assignment.

    Returns:
        Column-oriented randomized-arm assignments containing each user identifier
        and whether the user belongs to the randomized pricing arm.
    """
    n_users = len(generated_users[USER_ID_KEY])
    n_randomized = math.ceil(config.randomization_rate * n_users)

    rng = np.random.default_rng(derive_seed(config.seed, RANDOMIZED_ARM_STREAM))

    randomized_indices = set(
        rng.choice(n_users, size=n_randomized, replace=False).tolist()
    )

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        IS_RANDOMIZED_KEY: [index in randomized_indices for index in range(n_users)],
    }


def assign_user_prices(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    weekly_base_prices: WeeklyBasePrices,
    randomized_arms: RandomizedArmAssignments,
    promo_depths: PromoDepthAssignments,
) -> AssignedUserPrices:
    """Assign observed prices to synthetic users.

    Args:
        config: Generation configuration containing the master random seed.
        generated_users: Synthetic user population containing signup weeks and
            subscription tiers.
        weekly_base_prices: Tier-specific weekly base prices generated from the
            simulated market conditions.
        randomized_arms: User-level randomized pricing arm assignments.
        promo_depths: User-level observational promotion depth assignments.

    Returns:
        Column-oriented user pricing data containing each user's observed price and
        whether the price was assigned through randomization.
    """
    rng = np.random.default_rng(derive_seed(config.seed, PRICE_ASSIGNMENT_STREAM))

    randomized_multipliers = rng.choice(
        RANDOMIZED_PRICE_MULTIPLIERS,
        size=len(generated_users[USER_ID_KEY]),
    )

    price_by_week_and_tier = {
        (week, tier): price
        for week, tier, price in zip(
            weekly_base_prices[WEEK_KEY],
            weekly_base_prices[TIER_KEY],
            weekly_base_prices[PRICE_KEY],
            strict=True,
        )
    }

    observed_prices: list[float] = []

    for index, randomized in enumerate(randomized_arms[IS_RANDOMIZED_KEY]):
        signup_week = generated_users[SIGNUP_WEEK_KEY][index]
        tier = generated_users[TIER_KEY][index]

        if randomized:
            price_multiplier = float(randomized_multipliers[index])
            observed_price = REFERENCE_PRICE_BY_TIER[tier] * price_multiplier
        else:
            base_price = price_by_week_and_tier[(signup_week, tier)]
            promo_depth = promo_depths[PROMO_DEPTH_KEY][index]

            observed_price = base_price * (1.0 - promo_depth)

        observed_prices.append(observed_price)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        OBSERVED_PRICE_KEY: observed_prices,
        IS_RANDOMIZED_KEY: randomized_arms[IS_RANDOMIZED_KEY].copy(),
    }


def calculate_conversion_probabilities(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    user_pricing: AssignedUserPrices,
    weekly_conditions: WeeklyConditions,
) -> ConversionProbabilities:
    """Calculate the conversion probability per user.

    Args:
        config: Generation configuration containing customer segment parameters.
        generated_users: Synthetic user population containing signup weeks.
        user_pricing: User-level observed pricing assignments.
        weekly_conditions: Weekly market conditions containing observed demand and
            the latent weekly market shock.

    Returns:
        Probability of each user being converted.
    """
    segment_lookup = {segment.name.strip(): segment for segment in config.segments}

    probabilities: list[float] = []

    for index in range(len(generated_users[USER_ID_KEY])):
        user_segment = generated_users[SEGMENT_KEY][index]
        segment = segment_lookup[user_segment]

        baseline_conversion = segment.baseline_conversion
        price_coefficient = segment.price_coefficient

        observed_price = user_pricing[OBSERVED_PRICE_KEY][index]
        signup_week = generated_users[SIGNUP_WEEK_KEY][index]
        demand_index = weekly_conditions[DEMAND_INDEX_KEY][signup_week]

        log_odds = math.log(baseline_conversion / (1 - baseline_conversion))
        price_effect = price_coefficient * math.log(observed_price / REFERENCE_PRICE)
        demand_effect = DEMAND_CONVERSION_SENSITIVITY * (demand_index - DEMAND_MEAN)
        logits = log_odds + price_effect + demand_effect
        conversion_probability = 1 / (1 + math.exp(-logits))

        probabilities.append(conversion_probability)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        CONVERSION_PROB_KEY: probabilities,
    }


def sample_conversions(
    config: PricingDataConfig,
    conversion_probabilities: ConversionProbabilities,
) -> ConversionOutcomes:
    """Sample conversion outcomes from user conversion probabilities.

    Args:
        config: Generation configuration containing the random seed.
        conversion_probabilities: User-level probabilities of conversion.

    Returns:
        Column-oriented user conversion outcomes indicating whether each user
        converted.
    """
    prob_array = np.array(conversion_probabilities[CONVERSION_PROB_KEY])

    rng = np.random.default_rng(derive_seed(config.seed, CONVERSION_OUTCOME_STREAM))

    random_draws = rng.random(size=prob_array.shape)

    conversion_array = random_draws < prob_array

    return {
        USER_ID_KEY: conversion_probabilities[USER_ID_KEY].copy(),
        CONVERSION_OUTCOME_KEY: conversion_array.tolist(),
    }


def assign_acquisition_costs(generated_users: UserPopulation) -> UserAcquisitionCosts:
    """Map each user's acquisition channel to an acquisition cost.

    Args:
        generated_users: Synthetic user population containing acquisition channels.

    Returns:
        Column-oriented user acquisition cost with one cost per user.
    """
    acquisition_costs: list[float] = []

    for channel in generated_users[CHANNEL_KEY]:
        user_acquisition_cost = ACQ_COST_MAPPING[channel]
        acquisition_costs.append(user_acquisition_cost)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        ACQUISITION_COST_KEY: acquisition_costs,
    }


def assign_marginal_costs(generated_users: UserPopulation) -> UserMarginalCosts:
    """Map each user's subscription tier to a marginal cost.

    Args:
        generated_users: Synthetic user population containing the subscription tiers.

    Returns:
        Column-oriented user marginal cost with one cost per user.
    """
    marginal_costs: list[float] = []

    for subscription_tier in generated_users[TIER_KEY]:
        user_marginal_cost = TIER_MARGINAL_COST_MAPPING[subscription_tier]
        marginal_costs.append(user_marginal_cost)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        MARGINAL_COST_KEY: marginal_costs,
    }


def calculate_churn_probabilities(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    user_pricing: AssignedUserPrices,
    conversion_outcomes: ConversionOutcomes,
) -> ChurnProbabilities:
    """Calculate the churn probability for each synthetic user.

    Args:
        config: Generation configuration containing customer segment parameters.
        generated_users: Synthetic user population containing segment and tier.
        user_pricing: User-level observed pricing assignments.
        conversion_outcomes: User-level conversion outcomes.

    Returns:
        Column-oriented churn probabilities with one probability per user.
    """
    segment_lookup = {segment.name.strip(): segment for segment in config.segments}

    churn_probabilities: list[float] = []

    for index in range(len(generated_users[USER_ID_KEY])):
        converted = conversion_outcomes[CONVERSION_OUTCOME_KEY][index]

        if not converted:
            churn_probabilities.append(0.0)
            continue

        user_segment = generated_users[SEGMENT_KEY][index]
        subscription_tier = generated_users[TIER_KEY][index]

        segment = segment_lookup[user_segment]

        baseline_churn = segment.baseline_churn
        observed_price = user_pricing[OBSERVED_PRICE_KEY][index]

        log_odds = math.log(baseline_churn / (1 - baseline_churn))

        price_effect = CHURN_PRICE_SENSITIVITY * math.log(
            observed_price / REFERENCE_PRICE
        )

        tier_effect = TIER_CHURN_ADJUSTMENT[subscription_tier]

        logits = log_odds + price_effect + tier_effect

        churn_probability = 1 / (1 + math.exp(-logits))

        churn_probabilities.append(churn_probability)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        CHURN_PROB_KEY: churn_probabilities,
    }


def sample_churn_outcomes(
    config: PricingDataConfig,
    churn_probabilities: ChurnProbabilities,
) -> ChurnOutcomes:
    """Sample churn outcomes from user churn probabilities.

    Args:
        config: Generation configuration containing the random seed.
        churn_probabilities: User-level probabilities of churn.

    Returns:
        Column-oriented user churn outcomes indicating whether the user churned.
    """
    prob_array = np.array(churn_probabilities[CHURN_PROB_KEY])

    rng = np.random.default_rng(derive_seed(config.seed, CHURN_OUTCOME_STREAM))

    random_draws = rng.random(size=prob_array.shape)

    churn_array = random_draws < prob_array

    return {
        USER_ID_KEY: churn_probabilities[USER_ID_KEY].copy(),
        CHURN_OUTCOME_KEY: churn_array.tolist(),
    }
