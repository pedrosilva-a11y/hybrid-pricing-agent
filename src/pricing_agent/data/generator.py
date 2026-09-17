"""Synthetic pricing data generation."""

import math
import random
from typing import Final, TypedDict

import numpy as np
from numpy.random import SeedSequence

from pricing_agent.data.config import PricingDataConfig

# Random Stream Identifiers
USER_GENERATION_STREAM: Final = 0
WEEKLY_DEMAND_STREAM: Final = 1
PRICE_ASSIGNMENT_STREAM: Final = 2
CONVERSION_OUTCOME_STREAM: Final = 3

# User Population Global Variables
ALLOWED_ACQUISITION_CHANNELS: Final = ("organic", "paid_search", "affiliate")
CHANNEL_KEY: Final = "channel"
SEGMENT_KEY: Final = "segment"
SIGNUP_WEEK_KEY: Final = "signup_week"
USER_ID_KEY: Final = "user_id"

# Weekly Demand Global Variables
DEMAND_INDEX_KEY: Final = "demand_index"
DEMAND_HIGH_BOUND: Final = 1.25
DEMAND_LOW_BOUND: Final = 0.75
DEMAND_MEAN: Final = 1.0
# Generates a moderate spread (~99.7% of values fall between 0.76 and 1.24)
DEMAND_STD_DEV: Final = 0.08
WEEK_KEY: Final = "week"

# Weekly Price Global Variables
PRICE_DEMAND_SENSITIVITY: Final = 0.4
PRICE_KEY: Final = "price"
REFERENCE_PRICE: Final = 19.99

# Assign User Prices Global Variables
EXPERIMENTAL_PRICE_MULTIPLIERS: Final = (0.90, 0.95, 1.00, 1.05, 1.10)
IS_RANDOMIZED_KEY: Final = "is_randomized"
OBSERVED_PRICE_KEY: Final = "observed_price"

# Conversion Probabilities Global Variables
DEMAND_CONVERSION_SENSITIVITY: Final = 1.0
CONVERSION_PROB_KEY: Final = "conversion_probability"

# Conversion Outcome Global Variable
CONVERSION_OUTCOME_KEY: Final = "conversion_outcome"

# Acquisition Cost Mapping Global Variable
ACQUISITION_COST_KEY: Final = "acquisition_cost"
ACQ_COST_MAPPING: Final = {
    "organic": 3.00,
    "affiliate": 10.00,
    "paid_search": 20.00,
}


class UserPopulation(TypedDict):
    """Column-oriented synthetic user population.

    Attributes:
        user_id: Unique user identifier.
        signup_week: Week the user enters the simulated population.
        segment: Customer segment assigned to the user.
        channel: Acquisition channel through which the user was acquired.
    """

    user_id: list[int]
    signup_week: list[int]
    segment: list[str]
    channel: list[str]


class WeeklyDemand(TypedDict):
    """Column-oriented synthetic weekly demand population.

    Attributes:
        week: Unique week index in the simulated period.
        demand_index: Represents the relative level of market demand for a given week.
            A value of 1.0 represents normal reference demand. Values above 1.0 indicate
            stronger-than-normal demand, while values below 1.0 indicate
            weaker-than-normal demand. The farther the value is from 1.0, the stronger
            the deviation from the reference level.
    """

    week: list[int]
    demand_index: list[float]


class WeeklyPrice(TypedDict):
    """Column-oriented synthetic weekly pricing data.

    Attributes:
        week: Unique week index in the simulated period.
        price: Policy price offered during the week, derived from the reference price
            and the week's demand condition.
    """

    week: list[int]
    price: list[float]


class AssignedUserPrices(TypedDict):
    """Column-oriented synthetic user price assignments.

    Attributes:
        user_id: Unique user identifier.
        observed_price: Price assigned to the user. For non-randomized users, this
            matches the policy price for the user's signup week. For randomized users,
            this is assigned independently of the weekly pricing policy.
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
    rng = random.Random(derive_seed(config.seed, USER_GENERATION_STREAM))

    synthetic_data: UserPopulation = {
        USER_ID_KEY: [],
        SIGNUP_WEEK_KEY: [],
        SEGMENT_KEY: [],
        CHANNEL_KEY: [],
    }

    segment_names = [segment.name.strip() for segment in config.segments]

    for user_id in range(config.n_users):
        synthetic_data[USER_ID_KEY].append(user_id)
        synthetic_data[SIGNUP_WEEK_KEY].append(rng.randrange(config.n_weeks))
        synthetic_data[SEGMENT_KEY].append(rng.choice(segment_names))
        synthetic_data[CHANNEL_KEY].append(rng.choice(ALLOWED_ACQUISITION_CHANNELS))

    return synthetic_data


def generate_weekly_demand(config: PricingDataConfig) -> WeeklyDemand:
    """Generate reproducible weekly market demand conditions.

    Args:
        config: Generation configuration defining the number of simulated weeks
            and the random seed.

    Returns:
        Column-oriented weekly demand data containing one demand index per week.
    """
    n_weeks = config.n_weeks

    rng = np.random.default_rng(derive_seed(config.seed, WEEKLY_DEMAND_STREAM))

    raw_demand = rng.normal(loc=DEMAND_MEAN, scale=DEMAND_STD_DEV, size=n_weeks)

    simulated_demand = np.clip(raw_demand, DEMAND_LOW_BOUND, DEMAND_HIGH_BOUND)

    return {
        WEEK_KEY: list(range(n_weeks)),
        DEMAND_INDEX_KEY: simulated_demand.tolist(),
    }


def generate_weekly_price(
    weekly_demand: WeeklyDemand,
) -> WeeklyPrice:
    """Generate weekly policy prices from market demand conditions.

    Args:
        weekly_demand: Weekly market demand conditions used by the pricing policy.

    Returns:
        Column-oriented weekly pricing data containing one policy price per week.
    """
    prices = [
        round(
            REFERENCE_PRICE
            * (1.0 + PRICE_DEMAND_SENSITIVITY * (demand_index - DEMAND_MEAN)),
            2,
        )
        for demand_index in weekly_demand[DEMAND_INDEX_KEY]
    ]

    return {
        WEEK_KEY: weekly_demand[WEEK_KEY].copy(),
        PRICE_KEY: prices,
    }


def assign_user_prices(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    weekly_price: WeeklyPrice,
) -> AssignedUserPrices:
    """Assign observed prices to synthetic users.

    Args:
        config: Generation configuration defining randomization rate and seed.
        generated_users: Synthetic user population containing signup weeks.
        weekly_price: Weekly pricing data containing one policy price per week.

    Returns:
        Column-oriented user pricing data containing each user's observed price and
        whether the price was assigned through randomization.
    """
    n_users = len(generated_users[USER_ID_KEY])
    n_randomized = math.ceil(config.randomization_rate * n_users)

    rng = random.Random(derive_seed(config.seed, PRICE_ASSIGNMENT_STREAM))

    randomized_indices = set(rng.sample(range(n_users), k=n_randomized))

    observed_prices: list[float] = []
    is_randomized: list[bool] = []

    for index in range(n_users):
        if index in randomized_indices:
            price_multiplier = rng.choice(EXPERIMENTAL_PRICE_MULTIPLIERS)
            observed_price = round(REFERENCE_PRICE * price_multiplier, 2)
            randomized = True
        else:
            signup_week = generated_users[SIGNUP_WEEK_KEY][index]
            observed_price = weekly_price[PRICE_KEY][signup_week]
            randomized = False

        observed_prices.append(observed_price)
        is_randomized.append(randomized)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        OBSERVED_PRICE_KEY: observed_prices,
        IS_RANDOMIZED_KEY: is_randomized,
    }


def calculate_conversion_probabilities(
    config: PricingDataConfig,
    generated_users: UserPopulation,
    user_pricing: AssignedUserPrices,
    weekly_demand: WeeklyDemand,
) -> ConversionProbabilities:
    """Calculate the conversion probability per user.

    Args:
        config: Generation configuration containing customer segment parameters.
        generated_users: Synthetic user population containing signup weeks.
        user_pricing: User-level observed pricing assignments.
        weekly_demand: Weekly market demand conditions used by the pricing policy.

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
        demand_index = weekly_demand[DEMAND_INDEX_KEY][signup_week]

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

    for index in range(len(generated_users[USER_ID_KEY])):
        user_channel = generated_users[CHANNEL_KEY][index]
        user_acquisition_cost = ACQ_COST_MAPPING[user_channel]
        acquisition_costs.append(user_acquisition_cost)

    return {
        USER_ID_KEY: generated_users[USER_ID_KEY].copy(),
        ACQUISITION_COST_KEY: acquisition_costs,
    }
