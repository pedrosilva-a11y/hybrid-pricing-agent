"""Tests for synthetic pricing data configuration."""

import pytest

from pricing_agent.data.config import PricingDataConfig


def test_create_pricing_data_config_with_valid_values() -> None:
    """Create a pricing data configuration with valid parameters."""
    config = PricingDataConfig(n_users=100, n_weeks=4, randomization_rate=0.1, seed=42)

    assert config.n_users == 100
    assert config.n_weeks == 4
    assert config.randomization_rate == 0.1
    assert config.seed == 42


@pytest.mark.parametrize("n_users", [0, -1])
def test_reject_non_positive_number_of_users(n_users: int) -> None:
    """Reject configurations with zero or negative users."""
    with pytest.raises(ValueError, match="n_users must be greater than zero"):
        PricingDataConfig(
            n_users=n_users,
            n_weeks=4,
            randomization_rate=0.1,
            seed=42,
        )


@pytest.mark.parametrize("n_weeks", [0, -1])
def test_reject_non_positive_number_of_weeks(n_weeks: int) -> None:
    """Reject configurations with zero or negative simulation weeks."""
    with pytest.raises(ValueError, match="n_weeks must be greater than zero"):
        PricingDataConfig(
            n_users=100,
            n_weeks=n_weeks,
            randomization_rate=0.1,
            seed=42,
        )


@pytest.mark.parametrize("randomization_rate", [0.0, 1.0])
def test_accept_randomization_rate_at_range_boundaries(
    randomization_rate: float,
) -> None:
    """Accept randomization rates at the inclusive range boundaries."""
    config = PricingDataConfig(
        n_users=100,
        n_weeks=4,
        randomization_rate=randomization_rate,
        seed=42,
    )

    assert config.randomization_rate == randomization_rate


@pytest.mark.parametrize("randomization_rate", [-0.01, 1.01])
def test_reject_randomization_rate_outside_valid_range(
    randomization_rate: float,
) -> None:
    """Reject randomization rates outside the inclusive zero-to-one range."""
    with pytest.raises(ValueError, match="randomization_rate must be between 0 and 1"):
        PricingDataConfig(
            n_users=100,
            n_weeks=4,
            randomization_rate=randomization_rate,
            seed=42,
        )
