"""Configuration for synthetic pricing data generation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentConfig:
    """Configuration describing a synthetic customer segment.

    Attributes:
        name: Unique name identifying the customer segment.
        price_coefficient: True log-price coefficient controlling how strongly the
            segment's subscription probability responds to changes relative to a
            reference price. Must be negative so higher prices reduce conversion.
            More negative values represent greater price sensitivity.
        baseline_conversion: Probability that a user in the segment subscribes under
            reference conditions, before applying price, demand, channel, or simulated
            effects. For example, 0.40 represents a 40% baseline probability of
            conversion.
        baseline_churn: Probability that a converted user in the segment churns under
            reference pricing and subscription conditions. For example, 0.20 represents
            a 20% baseline probability of churn.
    """

    name: str
    price_coefficient: float
    baseline_conversion: float
    baseline_churn: float

    def __post_init__(self) -> None:
        """Validate the customer segment configuration.

        Raises:
            ValueError: If name is empty, price_coefficient is greater than or equal
                to zero, baseline_conversion is outside the open interval (0, 1),
                or baseline_churn is outside the open interval (0, 1).
        """
        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if self.price_coefficient >= 0.0:
            raise ValueError("price_coefficient must be negative.")

        if not 0.0 < self.baseline_conversion < 1.0:
            raise ValueError(
                "baseline_conversion must be greater than 0 and less than 1.",
            )

        if not 0.0 < self.baseline_churn < 1.0:
            raise ValueError("baseline_churn must be greater than 0 and less than 1.")


@dataclass(frozen=True, slots=True)
class PricingDataConfig:
    """Configuration for synthetic pricing data generation.

    Attributes:
        n_users: Number of synthetic users to generate.
        n_weeks: Number of weeks covered by the simulated dataset.
        randomization_rate: Fraction of users assigned to randomized pricing,
            expressed as a value between 0 and 1.
        seed: Random seed used to make data generation reproducible.
        segments: Customer segment configurations represented in the synthetic
            population.
    """

    n_users: int
    n_weeks: int
    randomization_rate: float
    seed: int
    segments: tuple[SegmentConfig, ...]

    def __post_init__(self) -> None:
        """Validate the pricing configuration values.

        Raises:
            ValueError: If n_users or n_weeks are less than or equal to zero, if
                randomization_rate is outside the closed interval [0, 1], if
                segments is empty, or if segment names are not unique.
        """
        if self.n_users <= 0:
            raise ValueError("n_users must be greater than zero.")

        if self.n_weeks <= 0:
            raise ValueError("n_weeks must be greater than zero.")

        if not 0.0 <= self.randomization_rate <= 1.0:
            raise ValueError("randomization_rate must be between 0 and 1.")

        if not self.segments:
            raise ValueError("segments must contain at least one segment.")

        segment_names = [segment.name.strip() for segment in self.segments]

        if len(segment_names) != len(set(segment_names)):
            raise ValueError("segment names must be unique.")
