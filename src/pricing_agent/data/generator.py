"""Synthetic pricing data generation."""

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
