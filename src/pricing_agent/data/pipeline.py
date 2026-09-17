"""Synthetic data generation pipeline."""

from typing import Final, TypedDict

from pricing_agent.data.config import PricingDataConfig, SegmentConfig
from pricing_agent.data.generator import (
    AssignedUserPrices,
    ChurnOutcomes,
    ChurnProbabilities,
    ConversionOutcomes,
    ConversionProbabilities,
    UserAcquisitionCosts,
    UserMarginalCosts,
    UserPopulation,
    WeeklyDemand,
    WeeklyPrice,
    assign_acquisition_costs,
    assign_marginal_costs,
    assign_user_prices,
    calculate_churn_probabilities,
    calculate_conversion_probabilities,
    generate_users,
    generate_weekly_demand,
    generate_weekly_price,
    sample_churn_outcomes,
    sample_conversions,
)

# Configuration Definitions Global Variables
N_USERS: Final = 100
N_WEEKS: Final = 24
RANDOMIZATION_RATE: Final = 0.1
SEED: Final = 42


class PipelineOutput(TypedDict):
    """Column-oriented outputs produced by the synthetic data pipeline.

    Attributes:
        users: Generated synthetic user population.
        weekly_demand: Simulated weekly market demand conditions.
        weekly_prices: Weekly policy prices derived from market demand.
        assigned_prices: User-level observed price assignments.
        conversion_probabilities: User-level conversion probabilities.
        conversion_outcomes: Sampled user conversion outcomes.
        acquisition_costs: User-level acquisition costs.
        marginal_costs: User-level marginal costs.
        churn_probabilities: User-level churn probabilities.
        churn_outcomes: Sampled user churn outcomes.
    """

    users: UserPopulation
    weekly_demand: WeeklyDemand
    weekly_prices: WeeklyPrice
    assigned_prices: AssignedUserPrices
    conversion_probabilities: ConversionProbabilities
    conversion_outcomes: ConversionOutcomes
    acquisition_costs: UserAcquisitionCosts
    marginal_costs: UserMarginalCosts
    churn_probabilities: ChurnProbabilities
    churn_outcomes: ChurnOutcomes


def run_pipeline() -> PipelineOutput:
    """Run the synthetic data generation pipeline sequentially."""
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
        n_users=N_USERS,
        n_weeks=N_WEEKS,
        randomization_rate=RANDOMIZATION_RATE,
        seed=SEED,
        segments=segments,
    )

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

    sampled_conversions = sample_conversions(
        config=config,
        conversion_probabilities=conversion_probabilities,
    )

    assigned_acquisition_costs = assign_acquisition_costs(users_info)

    assigned_marginal_costs = assign_marginal_costs(users_info)

    churn_probabilities = calculate_churn_probabilities(
        config=config,
        generated_users=users_info,
        user_pricing=assigned_prices,
        conversion_outcomes=sampled_conversions,
    )

    sampled_churn_outcomes = sample_churn_outcomes(
        config=config,
        churn_probabilities=churn_probabilities,
    )

    return PipelineOutput(
        users=users_info,
        weekly_demand=weekly_demand,
        weekly_prices=weekly_price,
        assigned_prices=assigned_prices,
        conversion_probabilities=conversion_probabilities,
        conversion_outcomes=sampled_conversions,
        acquisition_costs=assigned_acquisition_costs,
        marginal_costs=assigned_marginal_costs,
        churn_probabilities=churn_probabilities,
        churn_outcomes=sampled_churn_outcomes,
    )
