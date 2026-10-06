from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DATA_DIR / "demand_data.csv"


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


START_DATE = "2024-01-01"

END_DATE = "2025-12-31"


PRODUCTS = {
    "Laptop": {
        "category": "Electronics",
        "base_demand": 180,
        "base_price": 65000,
    },
    "Monitor": {
        "category": "Electronics",
        "base_demand": 240,
        "base_price": 18000,
    },
    "Keyboard": {
        "category": "Accessories",
        "base_demand": 420,
        "base_price": 2500,
    },
    "Mouse": {
        "category": "Accessories",
        "base_demand": 500,
        "base_price": 1500,
    },
    "Printer": {
        "category": "Office Equipment",
        "base_demand": 150,
        "base_price": 12000,
    },
    "Router": {
        "category": "Networking",
        "base_demand": 280,
        "base_price": 4500,
    },
}


REGIONS = [
    "North",
    "South",
    "East",
    "West",
]


def generate_dates():
    return pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )


def generate_demand_data():
    dates = generate_dates()

    rows = []

    for date in dates:
        month = date.month

        seasonal_factor = 1.0

        if month in [10, 11, 12]:
            seasonal_factor = 1.20

        elif month in [6, 7, 8]:
            seasonal_factor = 0.90

        trend_factor = 1 + (
            (date - dates[0]).days / len(dates)
        ) * 0.15

        for product, details in PRODUCTS.items():

            for region in REGIONS:

                region_factor = {
                    "North": 1.05,
                    "South": 1.15,
                    "East": 0.90,
                    "West": 1.10,
                }[region]

                promotion = np.random.choice(
                    [0, 1],
                    p=[0.80, 0.20],
                )

                promotion_factor = (
                    1.25 if promotion == 1 else 1.0
                )

                price = details["base_price"]

                if promotion == 1:
                    price *= 0.90

                noise = np.random.normal(
                    loc=0,
                    scale=details["base_demand"] * 0.08,
                )

                demand = (
                    details["base_demand"]
                    * seasonal_factor
                    * trend_factor
                    * region_factor
                    * promotion_factor
                    + noise
                )

                demand = max(
                    0,
                    round(demand),
                )

                revenue = demand * price

                rows.append(
                    {
                        "date": date,
                        "product": product,
                        "category": details["category"],
                        "region": region,
                        "promotion": promotion,
                        "price": round(price, 2),
                        "demand": demand,
                        "revenue": round(revenue, 2),
                    }
                )

    return pd.DataFrame(rows)


def main():
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    demand_data = generate_demand_data()

    demand_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Demand dataset generated successfully.")

    print(f"Rows: {len(demand_data):,}")

    print(f"Columns: {len(demand_data.columns)}")

    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFirst 5 rows:")

    print(
        demand_data.head().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()