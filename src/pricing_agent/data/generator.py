"""Synthetic pricing data generation."""

import random
from typing import Final, TypedDict

from pricing_agent.data.config import PricingDataConfig

SEGMENT_KEY: Final = "segment"
SIGNUP_WEEK_KEY: Final = "signup_week"
USER_ID_KEY: Final = "user_id"


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
