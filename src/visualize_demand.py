from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "charts"
)


def load_data():
    df = pd.read_csv(
        RAW_DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def create_monthly_demand_chart(df):
    monthly_demand = (
        df.groupby(
            df["date"].dt.to_period("M")
        )["demand"]
        .sum()
        .reset_index()
    )

    monthly_demand["date"] = (
        monthly_demand["date"]
        .dt.to_timestamp()
    )

    plt.figure(
        figsize=(14, 6)
    )

    sns.lineplot(
        data=monthly_demand,
        x="date",
        y="demand",
        marker="o",
    )

    plt.title(
        "Monthly Demand Trend"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Total Demand"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "monthly_demand_trend.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def create_product_demand_chart(df):
    product_demand = (
        df.groupby("product")["demand"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    plt.figure(
        figsize=(10, 6)
    )

    sns.barplot(
        data=product_demand,
        x="demand",
        y="product",
    )

    plt.title(
        "Total Demand by Product"
    )

    plt.xlabel(
        "Total Demand"
    )

    plt.ylabel(
        "Product"
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "product_demand.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def create_region_demand_chart(df):
    region_demand = (
        df.groupby("region")["demand"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    plt.figure(
        figsize=(10, 6)
    )

    sns.barplot(
        data=region_demand,
        x="region",
        y="demand",
    )

    plt.title(
        "Total Demand by Region"
    )

    plt.xlabel(
        "Region"
    )

    plt.ylabel(
        "Total Demand"
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "region_demand.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def create_promotion_chart(df):
    promotion_demand = (
        df.groupby("promotion")["demand"]
        .mean()
        .reset_index()
    )

    promotion_demand["promotion"] = (
        promotion_demand["promotion"]
        .map(
            {
                0: "No Promotion",
                1: "Promotion",
            }
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.barplot(
        data=promotion_demand,
        x="promotion",
        y="demand",
    )

    plt.title(
        "Average Demand: Promotion vs No Promotion"
    )

    plt.xlabel(
        "Promotion Status"
    )

    plt.ylabel(
        "Average Demand"
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "promotion_impact.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def main():
    print(
        "Creating demand visualizations..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    create_monthly_demand_chart(df)

    create_product_demand_chart(df)

    create_region_demand_chart(df)

    create_promotion_chart(df)

    print(
        "\nDemand visualizations completed."
    )


if __name__ == "__main__":
    main()