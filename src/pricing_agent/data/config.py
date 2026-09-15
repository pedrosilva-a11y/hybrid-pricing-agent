"""Configuration for synthetic pricing data generation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricingDataConfig:
    """Configuration for synthetic pricing data generation.

    Attributes:
        n_users: Number of synthetic users to generate.
        n_weeks: Number of weeks covered by the simulated dataset.
        randomization_rate: Fraction of users assigned to randomized pricing,
            expressed as a value between 0 and 1.
        seed: Random seed used to make data generation reproducible.
    """

    n_users: int
    n_weeks: int
    randomization_rate: float
    seed: int

    def __post_init__(self) -> None:
        """Validate the pricing configuration values.

        Raises:
            ValueError: If n_users or n_weeks are less than or equal to zero and if
                randomization_rate is outside of the [0, 1].
        """
        if self.n_users <= 0:
            raise ValueError("n_users must be greater than zero.")

        if self.n_weeks <= 0:
            raise ValueError("n_weeks must be greater than zero.")

        if not 0.0 <= self.randomization_rate <= 1.0:
            raise ValueError("randomization_rate must be between 0 and 1.")
