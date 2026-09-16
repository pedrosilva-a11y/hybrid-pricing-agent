"""Tests for synthetic pricing data generation."""

import math

import pytest

from pricing_agent.data.config import PricingDataConfig, SegmentConfig
from pricing_agent.data.generator import (
    DEMAND_HIGH_BOUND,
    DEMAND_INDEX_KEY,
    DEMAND_LOW_BOUND,
    DEMAND_MEAN,
    IS_RANDOMIZED_KEY,
    OBSERVED_PRICE_KEY,
    PRICE_KEY,
    REFERENCE_PRICE,
    SEGMENT_KEY,
    SIGNUP_WEEK_KEY,
    USER_ID_KEY,
    WEEK_KEY,
    WeeklyDemand,
    assign_user_prices,
    generate_users,
    generate_weekly_demand,
    generate_weekly_price,
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


# Weekly Price


def test_generate_weekly_policy_prices(config: PricingDataConfig) -> None:
    """Create weekly policy prices as synthetic data."""
    weekly_demand = generate_weekly_demand(config)
    weekly_prices = generate_weekly_price(weekly_demand=weekly_demand)

    assert weekly_prices[WEEK_KEY] == list(range(config.n_weeks))

    for demand_index, price in zip(
        weekly_demand[DEMAND_INDEX_KEY],
        weekly_prices[PRICE_KEY],
        strict=True,
    ):
        if demand_index < DEMAND_MEAN:
            assert price < REFERENCE_PRICE
        elif demand_index > DEMAND_MEAN:
            assert price > REFERENCE_PRICE
        else:
            assert price == REFERENCE_PRICE

    assert (
        len(weekly_prices[WEEK_KEY]) == len(weekly_prices[PRICE_KEY]) == config.n_weeks
    )

    assert all(price > 0 for price in weekly_prices[PRICE_KEY])


def test_calculate_expected_policy_prices() -> None:
    """Calculate expected prices for known demand conditions."""
    weekly_demand: WeeklyDemand = {
        WEEK_KEY: [0, 1, 2],
        DEMAND_INDEX_KEY: [0.75, 1.0, 1.25],
    }

    weekly_prices = generate_weekly_price(weekly_demand)

    assert weekly_prices[PRICE_KEY] == [17.99, 19.99, 21.99]


# Price assignment


def test_assign_user_observed_prices_without_randomization(
    config: PricingDataConfig,
) -> None:
    """Assign synthetic user observed prices without randomization."""
    no_randomization_config = PricingDataConfig(
        n_users=config.n_users,
        n_weeks=config.n_weeks,
        randomization_rate=0.0,
        seed=config.seed,
        segments=config.segments,
    )
    users_info = generate_users(no_randomization_config)
    weekly_demand = generate_weekly_demand(no_randomization_config)
    weekly_price = generate_weekly_price(weekly_demand)

    assigned_prices = assign_user_prices(
        config=no_randomization_config,
        generated_users=users_info,
        weekly_price=weekly_price,
    )

    assert assigned_prices[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(assigned_prices[OBSERVED_PRICE_KEY]) == no_randomization_config.n_users
    assert len(assigned_prices[IS_RANDOMIZED_KEY]) == no_randomization_config.n_users
    assert not any(assigned_prices[IS_RANDOMIZED_KEY])

    for index, signup_week in enumerate(users_info[SIGNUP_WEEK_KEY]):
        assert (
            assigned_prices[OBSERVED_PRICE_KEY][index]
            == weekly_price[PRICE_KEY][signup_week]
        )


def test_assign_expected_number_of_randomized_user_prices(
    config: PricingDataConfig,
) -> None:
    """Randomize the expected number of user price assignments."""
    users_info = generate_users(config)
    weekly_demand = generate_weekly_demand(config)
    weekly_price = generate_weekly_price(weekly_demand)

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_price=weekly_price,
    )

    expected_randomized_users = math.ceil(config.randomization_rate * config.n_users)

    assert sum(assigned_prices[IS_RANDOMIZED_KEY]) == expected_randomized_users
    assert len(assigned_prices[OBSERVED_PRICE_KEY]) == config.n_users
    assert len(assigned_prices[IS_RANDOMIZED_KEY]) == config.n_users


def test_reproduce_identical_user_price_assignments_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical user price assignments when using the same seed."""
    users_info = generate_users(config)
    weekly_demand = generate_weekly_demand(config)
    weekly_price = generate_weekly_price(weekly_demand)

    assigned_prices_1 = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_price=weekly_price,
    )

    assigned_prices_2 = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_price=weekly_price,
    )

    assert assigned_prices_1 == assigned_prices_2
