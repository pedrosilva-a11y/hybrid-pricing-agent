"""Tests for synthetic pricing data configuration."""

import pytest

from pricing_agent.data.config import (
    ACQUISITION_CHANNELS,
    BASELINE_CONVERSION_BY_SEGMENT,
    CHANNEL_CONVERSION_EFFECT_BY_CHANNEL,
    DEMAND_CONVERSION_SENSITIVITY,
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
    SEGMENTS,
    SHOCK_CONVERSION_SENSITIVITY,
    STRUCTURAL_BETA_BY_SEGMENT,
    SUBSCRIPTION_TIERS,
    PricingDataConfig,
    SegmentConfig,
    build_pricing_data_config,
)


@pytest.fixture
def segments() -> tuple[SegmentConfig, ...]:
    """Provide valid customer segment configurations."""
    return (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
        SegmentConfig(
            name="premium",
            price_coefficient=-0.8,
            baseline_conversion=0.55,
            baseline_churn=0.12,
        ),
    )


# Segment Config


def test_create_segment_config_with_valid_values() -> None:
    """Create a segment configuration with valid parameters."""
    config = SegmentConfig(
        name="regular",
        price_coefficient=-2.0,
        baseline_conversion=0.40,
        baseline_churn=0.20,
    )

    assert config.name == "regular"
    assert config.price_coefficient == -2.0
    assert config.baseline_conversion == 0.40
    assert config.baseline_churn == 0.20


@pytest.mark.parametrize("name", ["", " ", "   "])
def test_reject_empty_segment_name(name: str) -> None:
    """Reject configurations with empty names."""
    with pytest.raises(ValueError, match="name must not be empty"):
        SegmentConfig(
            name=name,
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        )


@pytest.mark.parametrize("price_coefficient", [0.0, 0.5, 1.0])
def test_reject_non_negative_price_coefficients(price_coefficient: float) -> None:
    """Reject configurations with non-negative price coefficient values."""
    with pytest.raises(ValueError, match="price_coefficient must be negative"):
        SegmentConfig(
            name="regular",
            price_coefficient=price_coefficient,
            baseline_conversion=0.40,
            baseline_churn=0.20,
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
            baseline_churn=0.20,
        )


@pytest.mark.parametrize("baseline_churn", [0.0, 1.0, -0.01, 1.01])
def test_reject_out_of_range_baseline_churn(baseline_churn: float) -> None:
    """Reject configurations outside the exclusive zero-to-one range."""
    with pytest.raises(
        ValueError,
        match="baseline_churn must be greater than 0 and less than 1",
    ):
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=baseline_churn,
        )


# Pricing Data Config


def test_create_pricing_data_config_with_valid_values(
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Create a pricing data configuration with valid parameters."""
    config = PricingDataConfig(
        n_users=100,
        n_weeks=4,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )

    assert config.n_users == 100
    assert config.n_weeks == 4
    assert config.randomization_rate == 0.1
    assert config.seed == 42
    assert config.segments == segments


@pytest.mark.parametrize("n_users", [0, -1])
def test_reject_non_positive_number_of_users(
    n_users: int,
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Reject configurations with zero or negative users."""
    with pytest.raises(ValueError, match="n_users must be greater than zero"):
        PricingDataConfig(
            n_users=n_users,
            n_weeks=4,
            randomization_rate=0.1,
            seed=42,
            segments=segments,
        )


@pytest.mark.parametrize("n_weeks", [0, -1])
def test_reject_non_positive_number_of_weeks(
    n_weeks: int,
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Reject configurations with zero or negative simulation weeks."""
    with pytest.raises(ValueError, match="n_weeks must be greater than zero"):
        PricingDataConfig(
            n_users=100,
            n_weeks=n_weeks,
            randomization_rate=0.1,
            seed=42,
            segments=segments,
        )


@pytest.mark.parametrize("randomization_rate", [0.0, 1.0])
def test_accept_randomization_rate_at_range_boundaries(
    randomization_rate: float,
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Accept randomization rates at the inclusive range boundaries."""
    config = PricingDataConfig(
        n_users=100,
        n_weeks=4,
        randomization_rate=randomization_rate,
        seed=42,
        segments=segments,
    )

    assert config.randomization_rate == randomization_rate


@pytest.mark.parametrize("randomization_rate", [-0.01, 1.01])
def test_reject_randomization_rate_outside_valid_range(
    randomization_rate: float,
    segments: tuple[SegmentConfig, ...],
) -> None:
    """Reject randomization rates outside the inclusive zero-to-one range."""
    with pytest.raises(ValueError, match="randomization_rate must be between 0 and 1"):
        PricingDataConfig(
            n_users=100,
            n_weeks=4,
            randomization_rate=randomization_rate,
            seed=42,
            segments=segments,
        )


def test_reject_empty_segments_collection() -> None:
    """Reject configurations with an empty segment collection."""
    with pytest.raises(ValueError, match="segments must contain at least one segment"):
        PricingDataConfig(
            n_users=100,
            n_weeks=4,
            randomization_rate=0.1,
            seed=42,
            segments=(),
        )


def test_reject_non_unique_segment_names() -> None:
    """Reject configurations with duplicate segment names."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
        SegmentConfig(
            name="premium",
            price_coefficient=-0.8,
            baseline_conversion=0.55,
            baseline_churn=0.12,
        ),
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    with pytest.raises(ValueError, match="segment names must be unique"):
        PricingDataConfig(
            n_users=100,
            n_weeks=4,
            randomization_rate=0.1,
            seed=42,
            segments=segments,
        )


def test_reject_segment_names_with_duplicate_whitespace_variants() -> None:
    """Reject duplicate segment names that differ only by surrounding whitespace."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
        SegmentConfig(
            name="premium",
            price_coefficient=-0.8,
            baseline_conversion=0.55,
            baseline_churn=0.12,
        ),
        SegmentConfig(
            name=" regular ",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    with pytest.raises(ValueError, match="segment names must be unique"):
        PricingDataConfig(
            n_users=100,
            n_weeks=4,
            randomization_rate=0.1,
            seed=42,
            segments=segments,
        )


def test_build_pricing_data_config_with_default_segments() -> None:
    """Build pricing configuration with provided values and default segments."""
    config = build_pricing_data_config(
        n_users=500,
        n_weeks=52,
        randomization_rate=0.2,
        seed=123,
    )

    assert config.n_users == 500
    assert config.n_weeks == 52
    assert config.randomization_rate == 0.2
    assert config.seed == 123

    assert config.segments == (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
        SegmentConfig(
            name="premium",
            price_coefficient=-0.8,
            baseline_conversion=0.55,
            baseline_churn=0.12,
        ),
    )


# Calibrated constants


def test_define_frozen_calibrated_dgp_constants() -> None:
    """Define production DGP constants from the frozen balanced calibration."""
    assert SEGMENTS == ("price_sensitive", "price_resilient")
    assert ACQUISITION_CHANNELS == ("organic", "affiliate", "paid_search")
    assert SUBSCRIPTION_TIERS == ("basic", "premium")

    assert DEMAND_MEAN == 1.0
    assert DEMAND_STD_DEV == 0.10
    assert DEMAND_LOW_BOUND == 0.75
    assert DEMAND_HIGH_BOUND == 1.25

    assert HIDDEN_SHOCK_MEAN == 0.0
    assert HIDDEN_SHOCK_STD_DEV == 1.0

    assert REFERENCE_PRICE_BY_TIER == {"basic": 19.99, "premium": 29.99}

    assert STRUCTURAL_BETA_BY_SEGMENT == {
        "price_sensitive": -2.0,
        "price_resilient": -0.8,
    }

    assert BASELINE_CONVERSION_BY_SEGMENT == {
        "price_sensitive": 0.35,
        "price_resilient": 0.40,
    }

    assert PRICE_DEMAND_SENSITIVITY == 0.20
    assert PRICE_SHOCK_SENSITIVITY == 0.025

    assert PROMO_BIAS_BY_CHANNEL == {
        "organic": -1.73,
        "affiliate": -0.85,
        "paid_search": 0.0,
    }
    assert PROMO_DEMAND_SENSITIVITY == 5.0
    assert PROMO_SHOCK_SENSITIVITY == 0.45
    assert PROMO_DEPTHS == (0.05, 0.10, 0.20)

    assert RANDOMIZED_PRICE_MULTIPLIERS == (0.85, 0.925, 1.00, 1.075, 1.15)

    assert DEMAND_CONVERSION_SENSITIVITY == 5.0
    assert SHOCK_CONVERSION_SENSITIVITY == 0.10

    assert CHANNEL_CONVERSION_EFFECT_BY_CHANNEL == {
        "organic": -0.10,
        "affiliate": 0.0,
        "paid_search": 0.10,
    }


def test_center_channel_conversion_effects() -> None:
    """Keep channel conversion effects centered around zero."""
    assert sum(CHANNEL_CONVERSION_EFFECT_BY_CHANNEL.values()) == pytest.approx(0.0)
