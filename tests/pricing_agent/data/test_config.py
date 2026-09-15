"""Tests for synthetic pricing data configuration."""

import pytest

from pricing_agent.data.config import PricingDataConfig, SegmentConfig

# Segment Config


def test_create_segment_config_with_valid_values() -> None:
    """Create a segment configuration with valid parameters."""
    config = SegmentConfig(
        name="regular",
        price_coefficient=-2.0,
        baseline_conversion=0.40,
    )

    assert config.name == "regular"
    assert config.price_coefficient == -2.0
    assert config.baseline_conversion == 0.40


@pytest.mark.parametrize("name", ["", " ", "   "])
def test_reject_empty_segment_name(name: str) -> None:
    """Reject configurations with empty names."""
    with pytest.raises(ValueError, match="name must not be empty"):
        SegmentConfig(
            name=name,
            price_coefficient=-2.0,
            baseline_conversion=0.40,
        )


@pytest.mark.parametrize("price_coefficient", [0.0, 0.5, 1.0])
def test_reject_non_negative_price_coefficients(price_coefficient: float) -> None:
    """Reject configurations with non-negative price coefficient values."""
    with pytest.raises(ValueError, match="price_coefficient must be negative"):
        SegmentConfig(
            name="regular",
            price_coefficient=price_coefficient,
            baseline_conversion=0.40,
        )


@pytest.mark.parametrize("baseline_conversion", [0.0, 1.0, -0.01, 1.01])
def test_reject_out_of_range_baseline_conversion(baseline_conversion: float) -> None:
    """Reject configurations outside the exclusive zero-to-one range."""
    with pytest.raises(
        ValueError,
        match="baseline_conversion must be greater than 0 and less than 1",
    ):
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=baseline_conversion,
        )


# Pricing Data Config


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
