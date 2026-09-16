"""Tests for synthetic user population."""

import pytest

from pricing_agent.data.config import PricingDataConfig, SegmentConfig
from pricing_agent.data.generator import (
    SEGMENT_KEY,
    SIGNUP_WEEK_KEY,
    USER_ID_KEY,
    generate_users,
)


def test_generate_users_information() -> None:
    """Create user information as synthetic data."""
    segments = (
        SegmentConfig(name="regular", price_coefficient=-2.0, baseline_conversion=0.40),
        SegmentConfig(name="premium", price_coefficient=-0.8, baseline_conversion=0.55),
    )
    n_users = 10
    n_weeks = 4

    config = PricingDataConfig(
        n_users=n_users,
        n_weeks=n_weeks,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )

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


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_reproduce_identical_users_with_same_seed(seed: int) -> None:
    """Reproduce identical synthetic users when using the same seed."""
    segments = (
        SegmentConfig(name="regular", price_coefficient=-2.0, baseline_conversion=0.40),
        SegmentConfig(name="premium", price_coefficient=-0.8, baseline_conversion=0.55),
    )
    n_users = 10
    n_weeks = 4
    randomization_rate = 0.1

    config_1 = PricingDataConfig(
        n_users=n_users,
        n_weeks=n_weeks,
        randomization_rate=randomization_rate,
        seed=seed,
        segments=segments,
    )
    users_info_1 = generate_users(config_1)

    config_2 = PricingDataConfig(
        n_users=n_users,
        n_weeks=n_weeks,
        randomization_rate=randomization_rate,
        seed=seed,
        segments=segments,
    )
    users_info_2 = generate_users(config_2)

    assert users_info_1 == users_info_2
