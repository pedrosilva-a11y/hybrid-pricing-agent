"""Tests for synthetic data generation pipeline."""

from pricing_agent.data.config import build_pricing_data_config
from pricing_agent.data.generator import USER_ID_KEY
from pricing_agent.data.pipeline import run_pipeline


def test_run_pipeline_end_to_end() -> None:
    """Generate all expected pipeline outputs end to end."""
    config = build_pricing_data_config(
        n_users=100,
        n_weeks=24,
        randomization_rate=0.1,
        seed=42,
    )
    output = run_pipeline(config)

    assert set(output) == {
        "users",
        "weekly_conditions",
        "weekly_prices",
        "assigned_prices",
        "conversion_probabilities",
        "conversion_outcomes",
        "acquisition_costs",
        "marginal_costs",
        "churn_probabilities",
        "churn_outcomes",
    }

    assert len(output["users"][USER_ID_KEY]) == config.n_users
    assert len(output["assigned_prices"][USER_ID_KEY]) == config.n_users
    assert len(output["conversion_outcomes"][USER_ID_KEY]) == config.n_users
    assert len(output["churn_outcomes"][USER_ID_KEY]) == config.n_users
