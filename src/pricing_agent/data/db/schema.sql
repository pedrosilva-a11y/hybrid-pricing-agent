/*
Weekly synthetic conditions.

    week: Unique week index in the simulated period.
    demand_index: Represents the relative level of market demand for a given week.
        A value of 1.0 represents normal reference demand. Values above 1.0 indicate
        stronger-than-normal demand, while values below 1.0 indicate weaker-than-normal
        demand. The farther the value is from 1.0, the stronger the deviation from the
        reference level.
*/
CREATE TABLE weekly_conditions (
    week INTEGER PRIMARY KEY NOT NULL,
    demand_index DOUBLE NOT NULL,

    CONSTRAINT check_week_positivity
        CHECK (week >= 0),

    CONSTRAINT check_weekly_demand_range
        CHECK (demand_index BETWEEN 0.75 AND 1.25)
);

/*
Users table stores synthetic user attributes

    user_id: Unique user identifier.
    signup_week: Week the user enters the simulated population.
    segment: Customer segment assigned to the user.
    channel: Acquisition channel through which the user was acquired.
    tier: Subscription tier assigned to the user.
*/


CREATE TABLE users (
    user_id INTEGER PRIMARY KEY NOT NULL,
    signup_week INTEGER NOT NULL CHECK (signup_week >= 0),
    segment TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (
        channel IN ('organic', 'affiliate', 'paid_search')
    ),
    tier TEXT NOT NULL CHECK (
        tier IN ('basic', 'premium')
    ),

    CONSTRAINT fk_users_signup_week
        FOREIGN KEY (signup_week) REFERENCES weekly_conditions(week)
);

/*
Tier-specific weekly synthetic base prices.

    week: Week associated with the generated base price.
    tier: Subscription tier determining the reference price.
    price: Historical base price implied by the weekly market conditions.
*/
CREATE TABLE weekly_base_prices (
    week INTEGER NOT NULL,
    tier TEXT NOT NULL CHECK (
        tier IN ('basic', 'premium')
    ),
    price DOUBLE NOT NULL,

    CONSTRAINT pk_weekly_base_prices
        PRIMARY KEY (week, tier),

    CONSTRAINT check_weekly_base_price_positivity
        CHECK (price > 0),

    CONSTRAINT fk_weekly_base_prices_week
        FOREIGN KEY (week) REFERENCES weekly_conditions(week)
);

/*
Assigned synthetic user-level prices.

    user_id: Unique user identifier.
    observed_price: Price assigned to the user. For non-randomized users, this
        matches the tier-specific base price for the user's signup week. For
        randomized users, this is assigned independently of the historical
        pricing policy.
    is_randomized: Whether the user's observed price was assigned through the
        randomized pricing experiment.
*/

CREATE TABLE assigned_prices (
    user_id INTEGER PRIMARY KEY NOT NULL,
    observed_price DECIMAL(10, 2) NOT NULL,
    is_randomized BOOLEAN NOT NULL,

    CONSTRAINT check_assigned_price_positivity
        CHECK (observed_price > 0),

    CONSTRAINT fk_assigned_prices_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
);

/*
Synthetic user-level conversion probabilities.

    user_id: Unique user identifier.
    conversion_probability: Probability that the user converts under the assigned price
        and weekly demand conditions.
*/
CREATE TABLE conversion_probabilities (
    user_id INTEGER PRIMARY KEY NOT NULL,
    conversion_probability DOUBLE NOT NULL,

    CONSTRAINT check_conv_prob_in_range
        CHECK (conversion_probability BETWEEN 0 AND 1),

    CONSTRAINT fk_conv_prob_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
);


/*
Sampled user-level conversion outcomes.

    user_id: Unique user identifier.
    conversion_outcome: Whether the user converted after receiving the assigned
        price under the simulated market conditions.
*/
CREATE TABLE conversion_outcomes (
    user_id INTEGER PRIMARY KEY NOT NULL,
    conversion_outcome BOOLEAN NOT NULL,

    CONSTRAINT fk_conv_outcome_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
);

/*
Synthetic user-level acquisition costs.

    user_id: Unique user identifier.
    acquisition_cost: Cost incurred to acquire the user.
*/

CREATE TABLE acquisition_costs (
    user_id INTEGER PRIMARY KEY NOT NULL,
    acquisition_cost DECIMAL(10, 2) NOT NULL,

    CONSTRAINT fk_acq_cost_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),

    CONSTRAINT check_acquisition_cost_positivity
        CHECK (acquisition_cost > 0)
);

/*
Synthetic user-level marginal costs.

    user_id: Unique user identifier.
    marginal_cost: Incremental cost of serving the user.
*/
CREATE TABLE marginal_costs (
    user_id INTEGER PRIMARY KEY NOT NULL,
    marginal_cost DECIMAL(10, 2) NOT NULL,

    CONSTRAINT fk_mg_cost_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),

    CONSTRAINT check_marginal_cost_positivity
        CHECK (marginal_cost > 0)
);

/*
Synthetic user-level churn probabilities.

    user_id: Unique user identifier.
    churn_probability: Probability that the user churns under the simulated pricing
        and subscription conditions.
*/
CREATE TABLE churn_probabilities (
    user_id INTEGER PRIMARY KEY NOT NULL,
    churn_probability DOUBLE NOT NULL,

    CONSTRAINT fk_churn_prob_user
        FOREIGN KEY (user_id) REFERENCES users(user_id),

    CONSTRAINT check_churn_prob_in_range
        CHECK (churn_probability BETWEEN 0 AND 1)
);

/*
Sampled user-level churn outcomes.

    user_id: Unique user identifier.
    churn_outcome: Whether the user churned under the simulated pricing and subscription
        conditions.
*/
CREATE TABLE churn_outcomes (
    user_id INTEGER PRIMARY KEY NOT NULL,
    churn_outcome BOOLEAN NOT NULL,

    CONSTRAINT fk_churn_outcomes_user
        FOREIGN KEY (user_id) REFERENCES users(user_id)
);
