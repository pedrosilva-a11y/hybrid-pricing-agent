"""Command-line interface for synthetic pricing data generation."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

DEFAULT_N_USERS: Final = 100
DEFAULT_N_WEEKS: Final = 24
DEFAULT_RANDOMIZATION_RATE: Final = 0.1
DEFAULT_SEED: Final = 42


@dataclass(frozen=True, slots=True)
class CliArguments:
    """Command-line arguments for synthetic data generation.

    Attributes:
        n_users: Number of synthetic users to generate.
        n_weeks: Number of simulated weeks.
        randomization_rate: Fraction of users assigned randomized prices.
        seed: Random seed used for reproducible generation.
    """

    n_users: int
    n_weeks: int
    randomization_rate: float
    seed: int


def _positive_int(value: str) -> int:
    """Parse and validate a positive integer argument.

    Args:
        value: Raw command-line argument value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not positive.
    """
    parsed_value = int(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")

    return parsed_value


def _randomization_rate(value: str) -> float:
    """Parse and validate a randomization rate.

    Args:
        value: Raw command-line argument value.

    Returns:
        Parsed randomization rate.

    Raises:
        argparse.ArgumentTypeError: If the value is outside the interval [0, 1].
    """
    parsed_value = float(value)

    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1.")

    return parsed_value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic subscription pricing data.",
    )

    parser.add_argument(
        "--n-users",
        type=_positive_int,
        default=DEFAULT_N_USERS,
        help="Number of synthetic users to generate.",
    )
    parser.add_argument(
        "--n-weeks",
        type=_positive_int,
        default=DEFAULT_N_WEEKS,
        help="Number of simulated weeks to generate.",
    )
    parser.add_argument(
        "--randomization-rate",
        type=_randomization_rate,
        default=DEFAULT_RANDOMIZATION_RATE,
        help="Fraction of users assigned randomized prices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for reproducible generation.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> CliArguments:
    """Parse command-line arguments.

    Args:
        argv: Optional sequence of command-line arguments.

    Returns:
        Validated command-line arguments.
    """
    namespace = build_parser().parse_args(argv)

    return CliArguments(
        n_users=int(namespace.n_users),
        n_weeks=int(namespace.n_weeks),
        randomization_rate=float(namespace.randomization_rate),
        seed=int(namespace.seed),
    )
