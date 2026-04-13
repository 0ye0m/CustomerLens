from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(output_path: Path, n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic e-commerce customer dataset.

    Args:
        output_path: Path to write the CSV file.
        n_customers: Number of customers to generate.
        seed: Random seed for reproducibility.

    Returns:
        Generated dataframe.
    """
    rng = np.random.default_rng(seed)

    countries = [
        "United States",
        "Canada",
        "United Kingdom",
        "Germany",
        "France",
        "Australia",
        "India",
        "Brazil",
        "Japan",
        "Singapore",
    ]
    country_probs = np.array([0.26, 0.08, 0.12, 0.09, 0.08, 0.06, 0.12, 0.06, 0.07, 0.06])
    country_probs = country_probs / country_probs.sum()

    cities_by_country = {
        "United States": ["New York", "San Francisco", "Austin", "Chicago", "Seattle"],
        "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary"],
        "United Kingdom": ["London", "Manchester", "Birmingham", "Leeds"],
        "Germany": ["Berlin", "Munich", "Hamburg", "Cologne"],
        "France": ["Paris", "Lyon", "Marseille", "Toulouse"],
        "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth"],
        "India": ["Bengaluru", "Mumbai", "Delhi", "Hyderabad"],
        "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia"],
        "Japan": ["Tokyo", "Osaka", "Nagoya", "Fukuoka"],
        "Singapore": ["Singapore"],
    }

    genders = ["Female", "Male", "Non-binary"]
    channels = ["web", "mobile", "store"]
    channel_probs = [0.46, 0.42, 0.12]
    categories = [
        "Electronics",
        "Fashion",
        "Home",
        "Beauty",
        "Sports",
        "Grocery",
        "Toys",
    ]
    payment_methods = ["card", "paypal", "bank_transfer", "cash_on_delivery", "wallet"]
    discount_sensitivity_levels = ["low", "med", "high"]
    discount_probs = [0.45, 0.4, 0.15]

    customer_id = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
    age = rng.integers(18, 70, size=n_customers)
    gender = rng.choice(genders, size=n_customers, p=[0.48, 0.48, 0.04])

    country = rng.choice(countries, size=n_customers, p=country_probs)
    city = [rng.choice(cities_by_country[c]) for c in country]

    today = pd.Timestamp.today().normalize()
    signup_days_ago = rng.integers(30, 365 * 4, size=n_customers)
    signup_date = pd.Series(today - pd.to_timedelta(signup_days_ago, unit="D"))

    days_since_last_purchase = rng.gamma(shape=2.2, scale=28.0, size=n_customers).astype(int)
    days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365)
    last_purchase_date = pd.Series(today - pd.to_timedelta(days_since_last_purchase, unit="D"))

    last_purchase_date = pd.to_datetime(last_purchase_date)
    signup_date = pd.to_datetime(signup_date)

    invalid_last_purchase = last_purchase_date < signup_date
    if invalid_last_purchase.any():
        last_purchase_date.loc[invalid_last_purchase] = signup_date.loc[invalid_last_purchase] + pd.to_timedelta(
            rng.integers(1, 30, size=invalid_last_purchase.sum()),
            unit="D",
        )

    days_since_last_purchase = (today - last_purchase_date).dt.days

    tenure_years = ((today - signup_date).dt.days / 365.25).clip(lower=0.1)
    base_orders = rng.poisson(lam=2.0 + 2.6 * tenure_years)
    total_orders = np.clip(base_orders, 1, None)

    discount_sensitivity = rng.choice(discount_sensitivity_levels, size=n_customers, p=discount_probs)
    discount_multiplier = np.where(discount_sensitivity == "high", 0.88, np.where(discount_sensitivity == "med", 0.95, 1.03))

    avg_order_value = rng.lognormal(mean=3.35, sigma=0.45, size=n_customers) * discount_multiplier
    total_spend = avg_order_value * total_orders * rng.lognormal(mean=0.0, sigma=0.25, size=n_customers)
    total_spend = np.round(total_spend, 2)

    product_category_preference = rng.choice(categories, size=n_customers)
    channel = rng.choice(channels, size=n_customers, p=channel_probs)

    loyalty_points = (total_spend * 0.08 + total_orders * 6 + rng.normal(0, 15, size=n_customers)).clip(min=0).astype(int)
    support_tickets_raised = rng.poisson(lam=0.6 + 0.015 * days_since_last_purchase).clip(0, 6)
    satisfaction_score = np.clip(rng.normal(loc=7.2, scale=1.4, size=n_customers), 1, 10)
    referral_count = rng.poisson(lam=0.4 + 0.06 * (satisfaction_score - 6)).clip(0, 8)
    payment_method = rng.choice(payment_methods, size=n_customers, p=[0.52, 0.18, 0.12, 0.08, 0.1])

    churn_logit = (
        -2.2
        + 0.018 * days_since_last_purchase
        + 0.55 * (support_tickets_raised >= 2)
        - 0.32 * (satisfaction_score - 6)
        + 0.22 * (discount_sensitivity == "high")
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn_prob = churn_prob * (0.2 / churn_prob.mean())
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn_flag = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "country": country,
            "city": city,
            "signup_date": signup_date,
            "last_purchase_date": last_purchase_date,
            "total_orders": total_orders,
            "total_spend": total_spend,
            "avg_order_value": np.round(avg_order_value, 2),
            "product_category_preference": product_category_preference,
            "channel": channel,
            "loyalty_points": loyalty_points,
            "support_tickets_raised": support_tickets_raised,
            "churn_flag": churn_flag,
            "satisfaction_score": np.round(satisfaction_score, 1),
            "referral_count": referral_count,
            "payment_method": payment_method,
            "discount_sensitivity": discount_sensitivity,
            "days_since_last_purchase": days_since_last_purchase,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """Generate the dataset and persist it to disk."""
    output_path = Path(__file__).resolve().parent / "customers.csv"
    generate_dataset(output_path)


if __name__ == "__main__":
    main()
