"""Tests for synthetic data generation pipeline."""

from pricing_agent.data.generator import USER_ID_KEY
from pricing_agent.data.pipeline import N_USERS, run_pipeline


def test_run_pipeline_end_to_end() -> None:
    """Generate all expected pipeline outputs end to end."""
    output = run_pipeline()

    assert set(output) == {
        "users",
        "weekly_demand",
        "weekly_prices",
        "assigned_prices",
        "conversion_probabilities",
        "conversion_outcomes",
        "acquisition_costs",
        "marginal_costs",
        "churn_probabilities",
        "churn_outcomes",
    }

    assert len(output["users"][USER_ID_KEY]) == N_USERS
    assert len(output["assigned_prices"][USER_ID_KEY]) == N_USERS
    assert len(output["conversion_outcomes"][USER_ID_KEY]) == N_USERS
    assert len(output["churn_outcomes"][USER_ID_KEY]) == N_USERS
