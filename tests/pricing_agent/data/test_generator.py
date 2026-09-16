"""Tests for synthetic pricing data generation."""

import pytest

from pricing_agent.data.config import PricingDataConfig, SegmentConfig
from pricing_agent.data.generator import (
    DEMAND_HIGH_BOUND,
    DEMAND_INDEX_KEY,
    DEMAND_LOW_BOUND,
    SEGMENT_KEY,
    SIGNUP_WEEK_KEY,
    USER_ID_KEY,
    WEEK_KEY,
    generate_users,
    generate_weekly_demand,
)


@pytest.fixture
def segments() -> tuple[SegmentConfig, ...]:
    """Provide valid customer segment configurations."""
    return (
        SegmentConfig(name="regular", price_coefficient=-2.0, baseline_conversion=0.40),
        SegmentConfig(name="premium", price_coefficient=-0.8, baseline_conversion=0.55),
    )


@pytest.fixture
def config(segments: tuple[SegmentConfig, ...]) -> PricingDataConfig:
    """Provide valid synthetic pricing data generation configuration."""
    return PricingDataConfig(
        n_users=10,
        n_weeks=4,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )


# User Population


def test_generate_users_information(
    config: PricingDataConfig,
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Create user information as synthetic data."""
    n_users = config.n_users
    n_weeks = config.n_weeks
    users_info = generate_users(config)

    configured_segments = {segment.name.strip() for segment in segments}
    generated_segments = set(users_info[SEGMENT_KEY])

    assert users_info[USER_ID_KEY] == list(range(n_users))
    assert len(users_info[SIGNUP_WEEK_KEY]) == n_users
    assert len(users_info[SEGMENT_KEY]) == n_users
    assert all(
        0 <= signup_week < n_weeks for signup_week in users_info[SIGNUP_WEEK_KEY]
    )
    assert generated_segments <= configured_segments


def test_reproduce_identical_users_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical synthetic users when using the same seed."""
    users_info_1 = generate_users(config)
    users_info_2 = generate_users(config)

    assert users_info_1 == users_info_2


# Weekly Demand


def test_generate_weekly_demand_conditions(config: PricingDataConfig) -> None:
    """Create weekly demand conditions as synthetic data."""
    weekly_demand = generate_weekly_demand(config)

    assert weekly_demand[WEEK_KEY] == list(range(config.n_weeks))
    assert all(
        DEMAND_LOW_BOUND <= demand_index <= DEMAND_HIGH_BOUND
        for demand_index in weekly_demand[DEMAND_INDEX_KEY]
    )
    assert (
        len(weekly_demand[WEEK_KEY])
        == len(weekly_demand[DEMAND_INDEX_KEY])
        == config.n_weeks
    )


def test_reproduce_identical_weekly_demand_conditions_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical weekly demand conditions when using the same seed."""
    weekly_demand_1 = generate_weekly_demand(config)
    weekly_demand_2 = generate_weekly_demand(config)

    assert weekly_demand_1 == weekly_demand_2
