"""Tests for synthetic pricing data command-line arguments."""

import pytest

from pricing_agent.data.cli import (
    DEFAULT_N_USERS,
    DEFAULT_N_WEEKS,
    DEFAULT_RANDOMIZATION_RATE,
    DEFAULT_SEED,
    CliArguments,
    parse_args,
)


def test_parse_default_arguments() -> None:
    """Return default values when no command-line arguments are provided."""
    arguments = parse_args([])

    assert arguments == CliArguments(
        n_users=DEFAULT_N_USERS,
        n_weeks=DEFAULT_N_WEEKS,
        randomization_rate=DEFAULT_RANDOMIZATION_RATE,
        seed=DEFAULT_SEED,
    )


def test_parse_custom_arguments() -> None:
    """Return explicitly provided command-line argument values."""
    arguments = parse_args(
        [
            "--n-users",
            "1000",
            "--n-weeks",
            "52",
            "--randomization-rate",
            "0.2",
            "--seed",
            "123",
        ]
    )

    assert arguments == CliArguments(
        n_users=1000,
        n_weeks=52,
        randomization_rate=0.2,
        seed=123,
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("--n-users", "0"),
        ("--n-users", "-1"),
        ("--n-weeks", "0"),
        ("--n-weeks", "-1"),
    ],
)
def test_reject_non_positive_integer_arguments(
    argument: str,
    value: str,
) -> None:
    """Reject non-positive user and week counts."""
    with pytest.raises(SystemExit):
        parse_args([argument, value])


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_reject_randomization_rate_outside_valid_range(value: str) -> None:
    """Reject randomization rates outside the interval from zero to one."""
    with pytest.raises(SystemExit):
        parse_args(["--randomization-rate", value])


@pytest.mark.parametrize("value", ["0", "1"])
def test_accept_randomization_rate_boundaries(value: str) -> None:
    """Accept randomization rates at both valid boundaries."""
    arguments = parse_args(["--randomization-rate", value])

    assert arguments.randomization_rate == float(value)
