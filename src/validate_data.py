from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_data.csv"
)


REQUIRED_COLUMNS = [
    "date",
    "product",
    "category",
    "region",
    "promotion",
    "price",
    "demand",
    "revenue",
]


def load_data():
    return pd.read_csv(
        RAW_DATA_FILE
    )


def validate_columns(df):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("✓ Required columns are present.")


def validate_missing_values(df):
    missing_values = df[
        REQUIRED_COLUMNS
    ].isnull().sum()

    total_missing = missing_values.sum()

    if total_missing > 0:
        print("\nMissing values found:")
        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Dataset contains missing values."
        )

    print("✓ No missing values found.")


def validate_dates(df):
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_dates = df["date"].isnull().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Invalid dates found: {invalid_dates}"
        )

    print("✓ All dates are valid.")


def validate_numeric_values(df):
    if (df["price"] <= 0).any():
        raise ValueError(
            "Price contains zero or negative values."
        )

    if (df["demand"] < 0).any():
        raise ValueError(
            "Demand contains negative values."
        )

    if (df["revenue"] < 0).any():
        raise ValueError(
            "Revenue contains negative values."
        )

    print("✓ Numeric values are valid.")


def validate_promotion_values(df):
    valid_values = {0, 1}

    actual_values = set(
        df["promotion"].unique()
    )

    if not actual_values.issubset(
        valid_values
    ):
        raise ValueError(
            f"Invalid promotion values: {actual_values}"
        )

    print("✓ Promotion values are valid.")


def validate_revenue_calculation(df):
    expected_revenue = (
        df["demand"] * df["price"]
    )

    differences = (
        df["revenue"] - expected_revenue
    ).abs()

    if (differences > 0.01).any():
        raise ValueError(
            "Revenue calculation contains inconsistencies."
        )

    print("✓ Revenue calculations are consistent.")


def validate_duplicate_rows(df):
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate rows found: {duplicate_count}"
        )

    print("✓ No duplicate rows found.")


def main():
    print("Starting demand dataset validation...\n")

    df = load_data()

    print(
        f"Rows loaded: {len(df):,}"
    )

    validate_columns(df)

    validate_missing_values(df)

    validate_dates(df)

    validate_numeric_values(df)

    validate_promotion_values(df)

    validate_revenue_calculation(df)

    validate_duplicate_rows(df)

    print(
        "\n========================================"
    )

    print(
        "DATA VALIDATION PASSED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()