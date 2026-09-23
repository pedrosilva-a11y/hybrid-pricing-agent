"""Tests for synthetic pricing data generation."""

import math
from dataclasses import replace

import pytest

from pricing_agent.data.config import (
    ACQUISITION_CHANNELS,
    DEMAND_HIGH_BOUND,
    DEMAND_LOW_BOUND,
    DEMAND_MEAN,
    PROMO_DEPTHS,
    RANDOMIZED_PRICE_MULTIPLIERS,
    REFERENCE_PRICE_BY_TIER,
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
    DEMAND_INDEX_KEY,
    HIDDEN_SHOCK_KEY,
    HIDDEN_SHOCK_STREAM,
    IS_RANDOMIZED_KEY,
    MARGINAL_COST_KEY,
    OBSERVED_PRICE_KEY,
    PRICE_ASSIGNMENT_STREAM,
    PRICE_KEY,
    PROMO_ASSIGNMENT_STREAM,
    PROMO_DEPTH_KEY,
    PROMO_DEPTH_STREAM,
    PROMO_KEY,
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
    PromoAssignments,
    UserPopulation,
    WeeklyConditions,
    assign_acquisition_costs,
    assign_marginal_costs,
    assign_promo_depths,
    assign_promotions,
    assign_randomized_arms,
    assign_user_prices,
    calculate_base_price,
    calculate_churn_probabilities,
    calculate_conversion_probabilities,
    calculate_promo_probability,
    derive_seed,
    generate_users,
    generate_weekly_base_prices,
    generate_weekly_conditions,
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


# Weekly Conditions


def test_generate_weekly_conditions(config: PricingDataConfig) -> None:
    """Create weekly demand conditions as synthetic data."""
    weekly_conditions = generate_weekly_conditions(config)

    assert weekly_conditions[WEEK_KEY] == list(range(config.n_weeks))
    assert all(
        DEMAND_LOW_BOUND <= demand_index <= DEMAND_HIGH_BOUND
        for demand_index in weekly_conditions[DEMAND_INDEX_KEY]
    )
    assert (
        len(weekly_conditions[WEEK_KEY])
        == len(weekly_conditions[DEMAND_INDEX_KEY])
        == len(weekly_conditions[HIDDEN_SHOCK_KEY])
        == config.n_weeks
    )


def test_reproduce_identical_weekly_conditions_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical weekly market conditions when using the same seed."""
    weekly_conditions_1 = generate_weekly_conditions(config)
    weekly_conditions_2 = generate_weekly_conditions(config)

    assert weekly_conditions_1 == weekly_conditions_2


# Base Price Calculation


def test_return_reference_price_under_reference_conditions() -> None:
    """Return the tier reference price under neutral market conditions."""
    assert calculate_base_price("basic", 1.0, 0.0) == pytest.approx(19.99)
    assert calculate_base_price("premium", 1.0, 0.0) == pytest.approx(29.99)


def test_increase_base_price_with_demand_and_positive_shock() -> None:
    """Increase historical base price as demand and latent shock increase."""
    reference = calculate_base_price("basic", 1.0, 0.0)
    stronger_demand = calculate_base_price("basic", 1.1, 0.0)
    positive_shock = calculate_base_price("basic", 1.0, 1.0)

    assert stronger_demand > reference
    assert positive_shock > reference


def test_calculate_expected_base_price() -> None:
    """Calculate the expected base price from calibrated pricing parameters."""
    price = calculate_base_price("basic", 1.10, 0.50)

    expected = 19.99 * (1.0 + 0.20 * 0.10 + 0.025 * 0.50)

    assert price == pytest.approx(expected)


# Weekly Base Price


def test_generate_weekly_base_prices(config: PricingDataConfig) -> None:
    """Create tier-specific weekly base prices as synthetic data."""
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(
        weekly_conditions=weekly_conditions,
    )

    assert len(weekly_base_prices[WEEK_KEY]) == config.n_weeks * len(SUBSCRIPTION_TIERS)

    assert weekly_base_prices[WEEK_KEY] == [
        week for week in range(config.n_weeks) for _ in SUBSCRIPTION_TIERS
    ]
    assert weekly_base_prices[TIER_KEY] == list(SUBSCRIPTION_TIERS) * config.n_weeks


def test_calculate_expected_base_prices() -> None:
    """Calculate expected tier-specific base prices for known conditions."""
    weekly_conditions: WeeklyConditions = {
        WEEK_KEY: [0, 1, 2],
        DEMAND_INDEX_KEY: [0.75, 1.0, 1.25],
        HIDDEN_SHOCK_KEY: [0.0, 0.0, 0.0],
    }

    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)

    assert weekly_base_prices[WEEK_KEY] == [0, 0, 1, 1, 2, 2]
    assert weekly_base_prices[TIER_KEY] == [
        "basic",
        "premium",
        "basic",
        "premium",
        "basic",
        "premium",
    ]
    assert weekly_base_prices[PRICE_KEY] == pytest.approx(
        [
            18.9905,
            28.4905,
            19.99,
            29.99,
            20.9895,
            31.4895,
        ]
    )


def test_generate_expected_weekly_conditions_for_frozen_seed(
    config: PricingDataConfig,
) -> None:
    """Generate stable weekly demand values for the frozen random stream."""
    weekly_conditions = generate_weekly_conditions(config)

    assert weekly_conditions[WEEK_KEY] == [0, 1, 2, 3]
    assert weekly_conditions[DEMAND_INDEX_KEY] == pytest.approx(
        [
            1.0054461180,
            0.8781068864,
            1.0137059806,
            1.1192886592,
        ]
    )


# Promotion probability calculation


def test_return_expected_promo_probability_under_neutral_conditions() -> None:
    """Calculate promo probability from the calibrated channel bias."""
    promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=0.0,
    )

    expected_promo_prob = 1.0 / (1.0 + math.exp(0.85))

    assert promo_prob == pytest.approx(expected_promo_prob)


def test_decrease_promo_probability_as_demand_increases() -> None:
    """Decrease promotion probability as market demand strengthens."""
    higher_demand_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.2,
        hidden_shock=0.0,
    )
    neutral_demand_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=0.0,
    )
    lower_demand_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=0.8,
        hidden_shock=0.0,
    )

    assert (
        higher_demand_promo_prob < neutral_demand_promo_prob < lower_demand_promo_prob
    )


def test_decrease_promo_probability_with_positive_hidden_shock() -> None:
    """Decrease promotion probability as the latent market shock increases."""
    neutral_shock_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=0.0,
    )
    positive_shock_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=0.2,
    )
    negative_shock_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=-0.2,
    )

    assert (
        positive_shock_promo_prob < neutral_shock_promo_prob < negative_shock_promo_prob
    )


def test_rank_promo_probability_by_channel_bias() -> None:
    """Rank promo probabilities as paid search, affiliate, then organic."""
    paid_search_promo_prob = calculate_promo_probability(
        channel="paid_search",
        demand_index=1.0,
        hidden_shock=0.0,
    )
    affiliate_promo_prob = calculate_promo_probability(
        channel="affiliate",
        demand_index=1.0,
        hidden_shock=0.0,
    )
    organic_promo_prob = calculate_promo_probability(
        channel="organic",
        demand_index=1.0,
        hidden_shock=0.0,
    )

    assert organic_promo_prob < affiliate_promo_prob < paid_search_promo_prob


# Promotion assignment


def test_exclude_randomized_users_from_promotions(config: PricingDataConfig) -> None:
    """Exclude randomized pricing users from observational promotions."""
    users = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    promo_assignments = assign_promotions(
        config=config,
        generated_users=users,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    for index, randomized in enumerate(randomized_arms[IS_RANDOMIZED_KEY]):
        if randomized:
            assert not promo_assignments[PROMO_KEY][index]


def test_reproduce_identical_promo_assignments_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical promotion assignments with the same seed."""
    users = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    assignments_1 = assign_promotions(
        config=config,
        generated_users=users,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )
    assignments_2 = assign_promotions(
        config=config,
        generated_users=users,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    assert assignments_1 == assignments_2


def test_assign_expected_promotions_for_frozen_seed(config: PricingDataConfig) -> None:
    """Assign the expected promotions for the frozen random seed."""
    users = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    promo_assignments = assign_promotions(
        config=config,
        generated_users=users,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    assert promo_assignments[PROMO_KEY] == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


# Promo Depths Assignment


def test_assign_zero_depth_without_promotion(config: PricingDataConfig) -> None:
    """Assign zero promotion depth to users without a promotion."""
    promo_assignments: PromoAssignments = {
        USER_ID_KEY: [0, 1, 2],
        PROMO_KEY: [False, False, False],
    }

    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assert promo_depths[USER_ID_KEY] == [0, 1, 2]
    assert promo_depths[PROMO_DEPTH_KEY] == [0.0, 0.0, 0.0]


def test_assign_allowed_depth_to_promoted_users(config: PricingDataConfig) -> None:
    """Assign configured promotion depths only to promoted users."""
    promo_assignments: PromoAssignments = {
        USER_ID_KEY: [0, 1, 2, 3],
        PROMO_KEY: [True, False, True, False],
    }

    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    for promo, depth in zip(
        promo_assignments[PROMO_KEY],
        promo_depths[PROMO_DEPTH_KEY],
        strict=True,
    ):
        if promo:
            assert depth in PROMO_DEPTHS
        else:
            assert depth == 0.0


def test_assign_expected_promo_depths_for_frozen_seed(
    config: PricingDataConfig,
) -> None:
    """Assign the expected promotion depths for the frozen random seed."""
    users = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    promo_assignments = assign_promotions(
        config=config,
        generated_users=users,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assert promo_depths[PROMO_DEPTH_KEY] == [
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


# Randomized arm assignment


def test_assign_exact_number_of_users_to_randomized_arm(
    config: PricingDataConfig,
) -> None:
    """Assign exactly the configured number of users to the randomized arm."""
    users = generate_users(config)

    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    expected_count = math.ceil(config.randomization_rate * config.n_users)

    assert randomized_arms[USER_ID_KEY] == users[USER_ID_KEY]
    assert sum(randomized_arms[IS_RANDOMIZED_KEY]) == expected_count


def test_reproduce_randomized_arm_assignments_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical randomized-arm assignments with the same seed."""
    users = generate_users(config)

    assignments_1 = assign_randomized_arms(config=config, generated_users=users)
    assignments_2 = assign_randomized_arms(config=config, generated_users=users)

    assert assignments_1 == assignments_2


def test_assign_expected_randomized_arm_for_frozen_seed(
    config: PricingDataConfig,
) -> None:
    """Assign stable randomized-arm membership for the frozen random stream."""
    users = generate_users(config)

    randomized_arms = assign_randomized_arms(config=config, generated_users=users)

    assert randomized_arms[IS_RANDOMIZED_KEY] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


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
    weekly_conditions = generate_weekly_conditions(no_randomization_config)
    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)
    randomized_arms = assign_randomized_arms(
        config=no_randomization_config,
        generated_users=users_info,
    )
    promo_assignments = assign_promotions(
        config=no_randomization_config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )
    promo_depths = assign_promo_depths(
        config=no_randomization_config,
        promo_assignments=promo_assignments,
    )

    assigned_prices = assign_user_prices(
        config=no_randomization_config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    assert assigned_prices[USER_ID_KEY] == users_info[USER_ID_KEY]
    assert len(assigned_prices[OBSERVED_PRICE_KEY]) == no_randomization_config.n_users
    assert len(assigned_prices[IS_RANDOMIZED_KEY]) == no_randomization_config.n_users
    assert not any(assigned_prices[IS_RANDOMIZED_KEY])

    price_by_week_and_tier = {
        (week, tier): price
        for week, tier, price in zip(
            weekly_base_prices[WEEK_KEY],
            weekly_base_prices[TIER_KEY],
            weekly_base_prices[PRICE_KEY],
            strict=True,
        )
    }

    for index, signup_week in enumerate(users_info[SIGNUP_WEEK_KEY]):
        tier = users_info[TIER_KEY][index]
        base_price = price_by_week_and_tier[(signup_week, tier)]
        promo_depth = promo_depths[PROMO_DEPTH_KEY][index]

        expected_price = base_price * (1.0 - promo_depth)

        assert assigned_prices[OBSERVED_PRICE_KEY][index] == pytest.approx(
            expected_price
        )


def test_assign_expected_number_of_randomized_user_prices(
    config: PricingDataConfig,
) -> None:
    """Randomize the expected number of user price assignments."""
    users_info = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users_info)
    promo_assignments = assign_promotions(
        config=config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )
    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    expected_randomized_users = math.ceil(config.randomization_rate * config.n_users)

    assert sum(assigned_prices[IS_RANDOMIZED_KEY]) == expected_randomized_users
    assert len(assigned_prices[OBSERVED_PRICE_KEY]) == config.n_users
    assert len(assigned_prices[IS_RANDOMIZED_KEY]) == config.n_users


def test_assign_allowed_multiplier_to_randomized_users(
    config: PricingDataConfig,
) -> None:
    """Assign randomized prices using only configured price multipliers."""
    users_info = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(
        weekly_conditions=weekly_conditions,
    )

    randomized_arms = assign_randomized_arms(
        config=config,
        generated_users=users_info,
    )

    promo_assignments = assign_promotions(
        config=config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    for index, randomized in enumerate(randomized_arms[IS_RANDOMIZED_KEY]):
        if not randomized:
            continue

        tier = users_info[TIER_KEY][index]
        reference_price = REFERENCE_PRICE_BY_TIER[tier]
        observed_price = assigned_prices[OBSERVED_PRICE_KEY][index]

        multiplier = observed_price / reference_price

        assert any(
            multiplier == pytest.approx(candidate)
            for candidate in RANDOMIZED_PRICE_MULTIPLIERS
        )


def test_assign_expected_randomized_prices_for_frozen_seed(
    config: PricingDataConfig,
) -> None:
    """Assign the expected randomized prices for the frozen random seed."""
    users_info = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)

    randomized_arms = assign_randomized_arms(
        config=config,
        generated_users=users_info,
    )

    promo_assignments = assign_promotions(
        config=config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )

    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    randomized_prices = [
        price
        for price, randomized in zip(
            assigned_prices[OBSERVED_PRICE_KEY],
            randomized_arms[IS_RANDOMIZED_KEY],
            strict=True,
        )
        if randomized
    ]

    assert randomized_prices == pytest.approx([16.9915])


def test_reproduce_identical_user_price_assignments_with_same_seed(
    config: PricingDataConfig,
) -> None:
    """Reproduce identical user price assignments when using the same seed."""
    users_info = generate_users(config)
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users_info)
    promo_assignments = assign_promotions(
        config=config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )
    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assigned_prices_1 = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    assigned_prices_2 = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
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
        OBSERVED_PRICE_KEY: [REFERENCE_PRICE_BY_TIER["basic"]],
        IS_RANDOMIZED_KEY: [False],
    }

    weekly_conditions: WeeklyConditions = {
        WEEK_KEY: [0],
        DEMAND_INDEX_KEY: [DEMAND_MEAN],
        HIDDEN_SHOCK_KEY: [0.0],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_conditions=weekly_conditions,
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

    weekly_conditions: WeeklyConditions = {
        WEEK_KEY: [0],
        DEMAND_INDEX_KEY: [DEMAND_MEAN],
        HIDDEN_SHOCK_KEY: [0.0],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_conditions=weekly_conditions,
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

    weekly_conditions: WeeklyConditions = {
        WEEK_KEY: [0, 1, 2],
        DEMAND_INDEX_KEY: [
            round(0.80 * DEMAND_MEAN, 2),
            DEMAND_MEAN,
            round(1.20 * DEMAND_MEAN, 2),
        ],
        HIDDEN_SHOCK_KEY: [0.0, 0.0, 0.0],
    }

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_conditions=weekly_conditions,
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
    weekly_conditions = generate_weekly_conditions(config)
    weekly_base_prices = generate_weekly_base_prices(weekly_conditions)
    randomized_arms = assign_randomized_arms(config=config, generated_users=users_info)
    promo_assignments = assign_promotions(
        config=config,
        generated_users=users_info,
        weekly_conditions=weekly_conditions,
        randomized_arms=randomized_arms,
    )
    promo_depths = assign_promo_depths(
        config=config,
        promo_assignments=promo_assignments,
    )

    assigned_prices = assign_user_prices(
        config=config,
        generated_users=users_info,
        weekly_base_prices=weekly_base_prices,
        randomized_arms=randomized_arms,
        promo_depths=promo_depths,
    )

    conversion_probabilities = calculate_conversion_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        weekly_conditions=weekly_conditions,
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
