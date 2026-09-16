"""Synthetic pricing data generation."""

import math
import random
from typing import Final, TypedDict

import numpy as np

from pricing_agent.data.config import PricingDataConfig

# User Population Global Variables
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


class UserPopulation(TypedDict):
    """Column-oriented synthetic user population.

    Attributes:
        user_id: Unique user identifier.
        signup_week: Week the user enters the simulated population.
        segment: Customer segment assigned to the user.
    """

    user_id: list[int]
    signup_week: list[int]
    segment: list[str]


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


def generate_users(config: PricingDataConfig) -> UserPopulation:
    """Generate the synthetic user population.

    Args:
        config: Generation configuration to determine conditions.

    Returns:
        Mapping of user attributes to their generated column values.
    """
    rng = random.Random(config.seed)
    synthetic_data: UserPopulation = {
        USER_ID_KEY: [],
        SIGNUP_WEEK_KEY: [],
        SEGMENT_KEY: [],
    }

    segment_names = [segment.name.strip() for segment in config.segments]

    for user_id in range(config.n_users):
        synthetic_data[USER_ID_KEY].append(user_id)
        synthetic_data[SIGNUP_WEEK_KEY].append(rng.randrange(config.n_weeks))
        synthetic_data[SEGMENT_KEY].append(rng.choice(segment_names))

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
    rng = np.random.default_rng(seed=config.seed)

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

    rng = random.Random(config.seed)

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
