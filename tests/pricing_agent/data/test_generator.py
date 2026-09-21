"""Tests for synthetic pricing data generation."""

import math
from dataclasses import replace

import pytest

from pricing_agent.data.config import (
    ACQUISITION_CHANNELS,
    SUBSCRIPTION_TIERS,
    PricingDataConfig,
    SegmentConfig,
)
from pricing_agent.data.generator import (
    ACQ_COST_MAPPING,
    ACQUISITION_COST_KEY,
    CHANNEL_KEY,
    CHURN_OUTCOME_KEY,
    CHURN_OUTCOME_STREAM,
    CHURN_PROB_KEY,
    CONVERSION_OUTCOME_KEY,
    CONVERSION_OUTCOME_STREAM,
    CONVERSION_PROB_KEY,
    DEMAND_HIGH_BOUND,
    DEMAND_INDEX_KEY,
    DEMAND_LOW_BOUND,
    DEMAND_MEAN,
    HIDDEN_SHOCK_STREAM,
    IS_RANDOMIZED_KEY,
    MARGINAL_COST_KEY,
    OBSERVED_PRICE_KEY,
    PRICE_ASSIGNMENT_STREAM,
    PRICE_KEY,
    PROMO_ASSIGNMENT_STREAM,
    PROMO_DEPTH_STREAM,
    RANDOMIZED_ARM_STREAM,
    REFERENCE_PRICE,
    SEGMENT_KEY,
    SIGNUP_WEEK_KEY,
    TIER_KEY,
    TIER_MARGINAL_COST_MAPPING,
    USER_GENERATION_STREAM,
    USER_ID_KEY,
    WEEK_KEY,
    WEEKLY_DEMAND_STREAM,
    AssignedUserPrices,
    ChurnProbabilities,
    ConversionOutcomes,
    ConversionProbabilities,
    UserPopulation,
    WeeklyDemand,
    assign_acquisition_costs,
    assign_marginal_costs,
    assign_user_prices,
    calculate_churn_probabilities,
    calculate_conversion_probabilities,
    derive_seed,
    generate_users,
    generate_weekly_demand,
    generate_weekly_price,
    sample_churn_outcomes,
    sample_conversions,
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
    assert len(users_info[CHANNEL_KEY]) == n_users
    assert set(users_info[CHANNEL_KEY]) <= set(ACQUISITION_CHANNELS)
    assert len(users_info[TIER_KEY]) == n_users
    assert set(users_info[TIER_KEY]) <= set(SUBSCRIPTION_TIERS)


def test_reproduce_identical_users_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical synthetic users when using the same seed."""
    users_info_1 = generate_users(config)
    users_info_2 = generate_users(config)

    assert users_info_1 == users_info_2


def test_generate_different_users_with_different_seeds(
    config: PricingDataConfig,
) -> None:
    """Generate different populations when using different master seeds."""
    different_seed_config = replace(config, seed=config.seed + 1)

    users_1 = generate_users(config)
    users_2 = generate_users(different_seed_config)

    users_1 != users_2


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


# Conversion Probability Calculation


def test_calculate_baseline_conversion_probability_under_reference_conditions() -> None:
    """Return the segment baseline conversion under reference conditions."""
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
    )
    config = PricingDataConfig(
        n_users=1,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    users_info: UserPopulation = {
        USER_ID_KEY: [0],
        SIGNUP_WEEK_KEY: [0],
        SEGMENT_KEY: ["regular"],
        CHANNEL_KEY: ["organic"],
        TIER_KEY: ["basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: [0],
        OBSERVED_PRICE_KEY: [REFERENCE_PRICE],
        IS_RANDOMIZED_KEY: [False],
    }

    weekly_demand: WeeklyDemand = {
        WEEK_KEY: [0],
        DEMAND_INDEX_KEY: [DEMAND_MEAN],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_demand=weekly_demand,
    )

    assert conversion_probabilities[USER_ID_KEY] == [0]
    assert conversion_probabilities[CONVERSION_PROB_KEY][0] == pytest.approx(0.40)


def test_decrease_conversion_probability_as_price_increases() -> None:
    """Decrease conversion probability as the observed price increases."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=3,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2],
        SIGNUP_WEEK_KEY: [0, 0, 0],
        SEGMENT_KEY: ["regular", "regular", "regular"],
        CHANNEL_KEY: ["organic", "organic", "organic"],
        TIER_KEY: ["basic", "basic", "basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: [0, 1, 2],
        OBSERVED_PRICE_KEY: [
            round(REFERENCE_PRICE * 0.90, 2),
            REFERENCE_PRICE,
            round(REFERENCE_PRICE * 1.10, 2),
        ],
        IS_RANDOMIZED_KEY: [False, False, False],
    }

    weekly_demand: WeeklyDemand = {
        WEEK_KEY: [0],
        DEMAND_INDEX_KEY: [DEMAND_MEAN],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_demand=weekly_demand,
    )

    lower_price_probability, reference_probability, higher_price_probability = (
        conversion_probabilities[CONVERSION_PROB_KEY]
    )

    assert lower_price_probability > reference_probability > higher_price_probability


def test_increase_conversion_probability_as_demand_increases() -> None:
    """Increase conversion probability as weekly demand increases."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=3,
        n_weeks=3,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2],
        SIGNUP_WEEK_KEY: [0, 1, 2],
        SEGMENT_KEY: ["regular", "regular", "regular"],
        CHANNEL_KEY: ["organic", "organic", "organic"],
        TIER_KEY: ["basic", "basic", "basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: [0, 1, 2],
        OBSERVED_PRICE_KEY: [REFERENCE_PRICE, REFERENCE_PRICE, REFERENCE_PRICE],
        IS_RANDOMIZED_KEY: [False, False, False],
    }

    weekly_demand: WeeklyDemand = {
        WEEK_KEY: [0, 1, 2],
        DEMAND_INDEX_KEY: [
            round(0.80 * DEMAND_MEAN, 2),
            DEMAND_MEAN,
            round(1.20 * DEMAND_MEAN, 2),
        ],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_demand=weekly_demand,
    )

    weak_demand_probability, normal_demand_probability, strong_demand_probability = (
        conversion_probabilities[CONVERSION_PROB_KEY]
    )

    assert (
        weak_demand_probability < normal_demand_probability < strong_demand_probability
    )


def test_generate_valid_conversion_probabilities_for_all_users(
    config: PricingDataConfig,
) -> None:
    """Generate a valid conversion probability for each synthetic user."""
    users_info = generate_users(config)
    weekly_demand = generate_weekly_demand(config)
    weekly_price = generate_weekly_price(weekly_demand)

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_price=weekly_price,
    )

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_demand=weekly_demand,
    )

    assert conversion_probabilities[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(conversion_probabilities[CONVERSION_PROB_KEY]) == config.n_users
    assert all(
        0.0 < probability < 1.0
        for probability in conversion_probabilities[CONVERSION_PROB_KEY]
    )


# Conversion Outcomes


def test_sample_deterministic_conversion_outcomes_at_probability_boundaries() -> None:
    """Sample deterministic outcomes for zero and one conversion probabilities."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )
    config = PricingDataConfig(
        n_users=4,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )
    conversion_probabilities: ConversionProbabilities = {
        USER_ID_KEY: [0, 1, 2, 3],
        CONVERSION_PROB_KEY: [1.0, 0.0, 0.0, 1.0],
    }

    sampled_conversions = sample_conversions(
        config=config,
        conversion_probabilities=conversion_probabilities,
    )

    assert sampled_conversions[USER_ID_KEY] == conversion_probabilities[USER_ID_KEY]
    assert sampled_conversions[CONVERSION_OUTCOME_KEY] == [True, False, False, True]


def test_preserve_users_when_sampling_conversion_outcomes() -> None:
    """Preserve user identifiers and generate one outcome per probability."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=5,
        n_weeks=4,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )

    conversion_probabilities: ConversionProbabilities = {
        USER_ID_KEY: list(range(5)),
        CONVERSION_PROB_KEY: [0.05, 0.15, 0.25, 0.35, 0.45],
    }

    sampled_conversions = sample_conversions(
        config=config,
        conversion_probabilities=conversion_probabilities,
    )

    assert len(sampled_conversions[USER_ID_KEY]) == config.n_users
    assert len(sampled_conversions[CONVERSION_OUTCOME_KEY]) == config.n_users
    assert sampled_conversions[USER_ID_KEY] == conversion_probabilities[USER_ID_KEY]
    assert all(
        isinstance(outcome, bool)
        for outcome in sampled_conversions[CONVERSION_OUTCOME_KEY]
    )


def test_reproduce_identical_conversion_outcomes_with_same_seed() -> None:
    """Reproduce identical conversion outcomes when using the same seed."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=5,
        n_weeks=4,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )

    conversion_probabilities: ConversionProbabilities = {
        USER_ID_KEY: list(range(5)),
        CONVERSION_PROB_KEY: [0.10, 0.30, 0.50, 0.70, 0.90],
    }

    sampled_conversions_1 = sample_conversions(
        config=config,
        conversion_probabilities=conversion_probabilities,
    )

    sampled_conversions_2 = sample_conversions(
        config=config,
        conversion_probabilities=conversion_probabilities,
    )

    assert sampled_conversions_1 == sampled_conversions_2


# Acquisition Costs


def test_assign_expected_acquisition_costs_from_channels() -> None:
    """Assign the expected acquisition cost for each known acquisition channel."""
    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2],
        SIGNUP_WEEK_KEY: [0, 1, 2],
        SEGMENT_KEY: ["regular", "regular", "regular"],
        CHANNEL_KEY: ["organic", "affiliate", "paid_search"],
        TIER_KEY: ["basic", "basic", "basic"],
    }

    acquisition_costs = assign_acquisition_costs(users_info)

    assert acquisition_costs[USER_ID_KEY] == users_info[USER_ID_KEY]
    for index, channel in enumerate(users_info[CHANNEL_KEY]):
        mapped_cost = ACQ_COST_MAPPING[channel]
        assert mapped_cost == acquisition_costs[ACQUISITION_COST_KEY][index]


def test_preserve_users_when_assigning_acquisition_costs() -> None:
    """Preserve user identifiers and assign one acquisition cost per user."""
    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2, 3],
        SIGNUP_WEEK_KEY: [0, 1, 2, 3],
        SEGMENT_KEY: ["regular", "regular", "premium", "premium"],
        CHANNEL_KEY: ["organic", "affiliate", "paid_search", "organic"],
        TIER_KEY: ["basic", "basic", "basic", "basic"],
    }

    acquisition_costs = assign_acquisition_costs(users_info)

    assert acquisition_costs[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(acquisition_costs[ACQUISITION_COST_KEY]) == len(users_info[USER_ID_KEY])
    assert all(
        acquisition_cost >= 0
        for acquisition_cost in acquisition_costs[ACQUISITION_COST_KEY]
    )


# Marginal Costs


def test_assign_expected_marginal_costs_from_subscription_tiers() -> None:
    """Assign the expected marginal cost for each known subscription tier."""
    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1],
        SIGNUP_WEEK_KEY: [0, 1],
        SEGMENT_KEY: ["regular", "regular"],
        CHANNEL_KEY: ["organic", "organic"],
        TIER_KEY: ["basic", "premium"],
    }

    marginal_costs = assign_marginal_costs(users_info)

    assert marginal_costs[USER_ID_KEY] == users_info[USER_ID_KEY]
    for index, tier in enumerate(users_info[TIER_KEY]):
        mapped_cost = TIER_MARGINAL_COST_MAPPING[tier]
        assert mapped_cost == marginal_costs[MARGINAL_COST_KEY][index]


def test_preserve_users_when_assigning_marginal_costs() -> None:
    """Preserve user identifiers and assign one marginal cost per user."""
    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2, 3],
        SIGNUP_WEEK_KEY: [0, 1, 2, 3],
        SEGMENT_KEY: ["regular", "regular", "regular", "regular"],
        CHANNEL_KEY: ["organic", "organic", "organic", "organic"],
        TIER_KEY: ["basic", "premium", "basic", "premium"],
    }

    marginal_costs = assign_marginal_costs(users_info)

    assert marginal_costs[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(marginal_costs[MARGINAL_COST_KEY]) == len(users_info[USER_ID_KEY])
    assert all(
        marginal_cost >= 0 for marginal_cost in marginal_costs[MARGINAL_COST_KEY]
    )


# Churn Probability Calculation


def test_calculate_baseline_churn_probability_under_reference_conditions() -> None:
    """Return baseline churn under reference price and basic tier conditions."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=1,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    users_info: UserPopulation = {
        USER_ID_KEY: [0],
        SIGNUP_WEEK_KEY: [0],
        SEGMENT_KEY: ["regular"],
        CHANNEL_KEY: ["organic"],
        TIER_KEY: ["basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: [0],
        OBSERVED_PRICE_KEY: [REFERENCE_PRICE],
        IS_RANDOMIZED_KEY: [False],
    }

    conversion_outcomes: ConversionOutcomes = {
        USER_ID_KEY: [0],
        CONVERSION_OUTCOME_KEY: [True],
    }

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=conversion_outcomes,
    )

    assert churn_probabilities[USER_ID_KEY] == [0]
    assert churn_probabilities[CHURN_PROB_KEY][0] == pytest.approx(0.20)


def test_increase_churn_probability_as_price_increases() -> None:
    """Increase churn probability as the observed price increases."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=3,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    user_ids = [0, 1, 2]

    users_info: UserPopulation = {
        USER_ID_KEY: user_ids,
        SIGNUP_WEEK_KEY: [0, 0, 0],
        SEGMENT_KEY: ["regular", "regular", "regular"],
        CHANNEL_KEY: ["organic", "organic", "organic"],
        TIER_KEY: ["basic", "basic", "basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: user_ids,
        OBSERVED_PRICE_KEY: [
            round(REFERENCE_PRICE * 0.90, 2),
            REFERENCE_PRICE,
            round(REFERENCE_PRICE * 1.20, 2),
        ],
        IS_RANDOMIZED_KEY: [False, False, False],
    }

    conversion_outcomes: ConversionOutcomes = {
        USER_ID_KEY: user_ids,
        CONVERSION_OUTCOME_KEY: [True, True, True],
    }

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=conversion_outcomes,
    )

    lower_price_probability, reference_probability, higher_price_probability = (
        churn_probabilities[CHURN_PROB_KEY]
    )

    assert lower_price_probability < reference_probability < higher_price_probability


def test_decrease_churn_probability_for_premium_tier() -> None:
    """Decrease churn probability for the premium subscription tier."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=2,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    user_ids = [0, 1]

    users_info: UserPopulation = {
        USER_ID_KEY: user_ids,
        SIGNUP_WEEK_KEY: [0, 0],
        SEGMENT_KEY: ["regular", "regular"],
        CHANNEL_KEY: ["organic", "organic"],
        TIER_KEY: ["basic", "premium"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: user_ids,
        OBSERVED_PRICE_KEY: [REFERENCE_PRICE, REFERENCE_PRICE],
        IS_RANDOMIZED_KEY: [False, False],
    }

    conversion_outcomes: ConversionOutcomes = {
        USER_ID_KEY: user_ids,
        CONVERSION_OUTCOME_KEY: [True, True],
    }

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=conversion_outcomes,
    )

    basic_probability, premium_probability = churn_probabilities[CHURN_PROB_KEY]

    assert premium_probability < basic_probability


def test_return_zero_churn_probability_for_non_converted_users() -> None:
    """Return zero churn probability for users who did not convert."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=1,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    user_id = [0]

    users_info: UserPopulation = {
        USER_ID_KEY: user_id,
        SIGNUP_WEEK_KEY: [0],
        SEGMENT_KEY: ["regular"],
        CHANNEL_KEY: ["organic"],
        TIER_KEY: ["premium"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: user_id,
        OBSERVED_PRICE_KEY: [round(REFERENCE_PRICE * 1.10, 2)],
        IS_RANDOMIZED_KEY: [False],
    }

    conversion_outcomes: ConversionOutcomes = {
        USER_ID_KEY: user_id,
        CONVERSION_OUTCOME_KEY: [False],
    }

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=conversion_outcomes,
    )

    assert churn_probabilities[USER_ID_KEY] == user_id
    assert churn_probabilities[CHURN_PROB_KEY] == [0.0]


def test_generate_valid_churn_probabilities_for_all_users() -> None:
    """Generate a valid churn probability for every synthetic user."""
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
    )

    config = PricingDataConfig(
        n_users=4,
        n_weeks=2,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )

    users_info: UserPopulation = {
        USER_ID_KEY: [0, 1, 2, 3],
        SIGNUP_WEEK_KEY: [0, 0, 1, 1],
        SEGMENT_KEY: ["regular", "premium", "regular", "premium"],
        CHANNEL_KEY: ["organic", "affiliate", "paid_search", "organic"],
        TIER_KEY: ["basic", "premium", "premium", "basic"],
    }

    assigned_prices: AssignedUserPrices = {
        USER_ID_KEY: [0, 1, 2, 3],
        OBSERVED_PRICE_KEY: [
            REFERENCE_PRICE,
            round(REFERENCE_PRICE * 1.05, 2),
            round(REFERENCE_PRICE * 0.95, 2),
            REFERENCE_PRICE,
        ],
        IS_RANDOMIZED_KEY: [False, False, False, False],
    }

    conversion_outcomes: ConversionOutcomes = {
        USER_ID_KEY: [0, 1, 2, 3],
        CONVERSION_OUTCOME_KEY: [True, True, False, True],
    }

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=conversion_outcomes,
    )

    assert churn_probabilities[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(churn_probabilities[CHURN_PROB_KEY]) == config.n_users
    assert all(
        0.0 <= probability < 1.0 for probability in churn_probabilities[CHURN_PROB_KEY]
    )


# Churn Outcome


def test_sample_deterministic_churn_outcomes_at_probability_boundaries() -> None:
    """Sample deterministic outcomes for zero and one churn probabilities."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )
    config = PricingDataConfig(
        n_users=4,
        n_weeks=1,
        randomization_rate=0.0,
        seed=42,
        segments=segments,
    )
    churn_probabilities: ChurnProbabilities = {
        USER_ID_KEY: [0, 1, 2, 3],
        CHURN_PROB_KEY: [1.0, 0.0, 0.0, 1.0],
    }

    sampled_churns = sample_churn_outcomes(
        config=config,
        churn_probabilities=churn_probabilities,
    )

    assert sampled_churns[USER_ID_KEY] == churn_probabilities[USER_ID_KEY]
    assert sampled_churns[CHURN_OUTCOME_KEY] == [True, False, False, True]


def test_preserve_users_when_sampling_churn_outcomes() -> None:
    """Preserve user identifiers and generate one outcome per probability."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )
    config = PricingDataConfig(
        n_users=5,
        n_weeks=4,
        randomization_rate=0.1,
        seed=42,
        segments=segments,
    )
    churn_probabilities: ChurnProbabilities = {
        USER_ID_KEY: [0, 1, 2, 3, 4],
        CHURN_PROB_KEY: [0.05, 0.15, 0.25, 0.35, 0.45],
    }

    sampled_churns = sample_churn_outcomes(
        config=config,
        churn_probabilities=churn_probabilities,
    )

    assert len(sampled_churns[USER_ID_KEY]) == config.n_users
    assert len(sampled_churns[CHURN_OUTCOME_KEY]) == config.n_users
    assert sampled_churns[USER_ID_KEY] == churn_probabilities[USER_ID_KEY]
    assert all(
        isinstance(outcome, bool) for outcome in sampled_churns[CHURN_OUTCOME_KEY]
    )


def test_reproduce_identical_churn_outcomes_with_same_seed() -> None:
    """Reproduce identical churn outcomes when using the same seed."""
    segments = (
        SegmentConfig(
            name="regular",
            price_coefficient=-2.0,
            baseline_conversion=0.40,
            baseline_churn=0.20,
        ),
    )

    config = PricingDataConfig(
        n_users=5,
        n_weeks=4,
        randomization_rate=0.01,
        seed=42,
        segments=segments,
    )

    churn_probabilities: ChurnProbabilities = {
        USER_ID_KEY: list(range(5)),
        CHURN_PROB_KEY: [0.10, 0.30, 0.50, 0.70, 0.90],
    }

    sampled_churns_1 = sample_churn_outcomes(
        config=config,
        churn_probabilities=churn_probabilities,
    )

    sampled_churns_2 = sample_churn_outcomes(
        config=config,
        churn_probabilities=churn_probabilities,
    )

    assert sampled_churns_1 == sampled_churns_2


# Random stream contract


def test_define_frozen_random_stream_identifiers() -> None:
    """Keep random stream identifiers stable after DGP calibration."""
    assert USER_GENERATION_STREAM == 0
    assert WEEKLY_DEMAND_STREAM == 1
    assert PRICE_ASSIGNMENT_STREAM == 2
    assert CONVERSION_OUTCOME_STREAM == 3
    assert CHURN_OUTCOME_STREAM == 4
    assert HIDDEN_SHOCK_STREAM == 5
    assert PROMO_ASSIGNMENT_STREAM == 6
    assert PROMO_DEPTH_STREAM == 7
    assert RANDOMIZED_ARM_STREAM == 8


def test_derive_frozen_seed_for_each_random_stream() -> None:
    """Derive stable independent seeds from the frozen stream identifiers."""
    expected_seeds = {
        0: 3444837047,
        1: 3329053876,
        2: 955475868,
        3: 2541583436,
        4: 964687612,
        5: 2103693821,
        6: 709256125,
        7: 1955881634,
        8: 3117874100,
    }

    assert {
        stream: derive_seed(42, stream) for stream in expected_seeds
    } == expected_seeds
