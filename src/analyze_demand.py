from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_data.csv"
)


def load_data():
    df = pd.read_csv(RAW_DATA_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


def analyze_overall_demand(df):
    print("\n===== OVERALL DEMAND =====")

    print(
        f"Total demand: {df['demand'].sum():,}"
    )

    print(
        f"Total revenue: ₹{df['revenue'].sum():,.2f}"
    )

    print(
        f"Average daily demand: "
        f"{df.groupby('date')['demand'].sum().mean():,.2f}"
    )


def analyze_by_product(df):
    print("\n===== DEMAND BY PRODUCT =====")

    product_summary = (
        df.groupby("product")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("revenue", "sum"),
            average_demand=("demand", "mean"),
        )
        .sort_values(
            "total_demand",
            ascending=False,
        )
    )

    print(
        product_summary.to_string()
    )


def analyze_by_category(df):
    print("\n===== DEMAND BY CATEGORY =====")

    category_summary = (
        df.groupby("category")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .sort_values(
            "total_demand",
            ascending=False,
        )
    )

    print(
        category_summary.to_string()
    )


def analyze_by_region(df):
    print("\n===== DEMAND BY REGION =====")

    region_summary = (
        df.groupby("region")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .sort_values(
            "total_demand",
            ascending=False,
        )
    )

    print(
        region_summary.to_string()
    )


def analyze_monthly_trend(df):
    print("\n===== MONTHLY DEMAND TREND =====")

    monthly_summary = (
        df.assign(
            month=df["date"].dt.to_period("M")
        )
        .groupby("month")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("revenue", "sum"),
        )
    )

    print(
        monthly_summary.to_string()
    )


def analyze_promotion_impact(df):
    print("\n===== PROMOTION IMPACT =====")

    promotion_summary = (
        df.groupby("promotion")
        .agg(
            average_demand=("demand", "mean"),
            total_demand=("demand", "sum"),
            average_revenue=("revenue", "mean"),
        )
    )

    promotion_summary.index = [
        "No Promotion",
        "Promotion",
    ]

    print(
        promotion_summary.to_string()
    )


def analyze_top_products(df):
    print("\n===== TOP PRODUCTS BY REVENUE =====")

    top_products = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(5)
    )

    print(
        top_products.to_string()
    )


def main():
    print("Starting demand analysis...")

    df = load_data()

    print(
        f"Rows analyzed: {len(df):,}"
    )

    analyze_overall_demand(df)

    analyze_by_product(df)

    analyze_by_category(df)

    analyze_by_region(df)

    analyze_monthly_trend(df)

    analyze_promotion_impact(df)

    analyze_top_products(df)

    print(
        "\n========================================"
    )

    print(
        "DEMAND ANALYSIS COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()