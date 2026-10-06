from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demand_data.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_data.csv"
)


TEST_DAYS = 90


def load_data():
    df = pd.read_csv(
        RAW_DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def create_daily_dataset(df):
    daily_data = (
        df.groupby("date")
        .agg(
            total_demand=("demand", "sum"),
            total_revenue=("revenue", "sum"),
            average_price=("price", "mean"),
            promotion_count=("promotion", "sum"),
        )
        .reset_index()
    )

    daily_data = daily_data.sort_values(
        "date"
    )

    return daily_data


def create_time_features(df):
    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["quarter"] = df["date"].dt.quarter

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_train_test_split(df):
    split_date = (
        df["date"].max()
        - pd.Timedelta(
            days=TEST_DAYS - 1
        )
    )

    train_data = df[
        df["date"] < split_date
    ].copy()

    test_data = df[
        df["date"] >= split_date
    ].copy()

    return train_data, test_data, split_date


def save_datasets(
    df,
    train_data,
    test_data,
):
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    train_file = (
        PROCESSED_DATA_DIR
        / "forecast_train.csv"
    )

    test_file = (
        PROCESSED_DATA_DIR
        / "forecast_test.csv"
    )

    train_data.to_csv(
        train_file,
        index=False,
    )

    test_data.to_csv(
        test_file,
        index=False,
    )

    print(
        f"Saved complete forecast dataset: {OUTPUT_FILE}"
    )

    print(
        f"Saved training dataset: {train_file}"
    )

    print(
        f"Saved testing dataset: {test_file}"
    )


def main():
    print(
        "Preparing forecasting dataset..."
    )

    df = load_data()

    print(
        f"Raw rows loaded: {len(df):,}"
    )

    daily_data = create_daily_dataset(
        df
    )

    daily_data = create_time_features(
        daily_data
    )

    train_data, test_data, split_date = (
        create_train_test_split(
            daily_data
        )
    )

    save_datasets(
        daily_data,
        train_data,
        test_data,
    )

    print(
        "\n===== FORECASTING DATASET ====="
    )

    print(
        f"Total daily observations: {len(daily_data):,}"
    )

    print(
        f"Training observations: {len(train_data):,}"
    )

    print(
        f"Testing observations: {len(test_data):,}"
    )

    print(
        f"Training period: "
        f"{train_data['date'].min().date()} "
        f"to "
        f"{train_data['date'].max().date()}"
    )

    print(
        f"Testing period: "
        f"{test_data['date'].min().date()} "
        f"to "
        f"{test_data['date'].max().date()}"
    )

    print(
        f"Split date: {split_date.date()}"
    )

    print(
        "\nFirst 5 forecasting rows:"
    )

    print(
        daily_data.head().to_string(
            index=False
        )
    )

    print(
        "\nLast 5 forecasting rows:"
    )

    print(
        daily_data.tail().to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "FORECASTING DATA PREPARATION COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()