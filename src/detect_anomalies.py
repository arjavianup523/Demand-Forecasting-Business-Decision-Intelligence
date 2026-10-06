from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORECAST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_anomalies.csv"
)


ROLLING_WINDOW = 30

Z_SCORE_THRESHOLD = 2.5


def load_data():
    df = pd.read_csv(
        FORECAST_DATA_FILE,
        parse_dates=["date"],
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


def detect_anomalies(df):
    df["rolling_mean"] = (
        df["total_demand"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .mean()
    )

    df["rolling_std"] = (
        df["total_demand"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .std()
    )

    df["z_score"] = (
        (
            df["total_demand"]
            - df["rolling_mean"]
        )
        / df["rolling_std"]
    )

    df["anomaly"] = (
        df["z_score"].abs()
        >= Z_SCORE_THRESHOLD
    )

    df["anomaly_type"] = np.where(
        df["z_score"] >= Z_SCORE_THRESHOLD,
        "Demand Spike",
        np.where(
            df["z_score"] <= -Z_SCORE_THRESHOLD,
            "Demand Drop",
            "Normal",
        ),
    )

    return df


def save_anomalies(df):
    anomaly_data = df[
        df["anomaly"]
    ].copy()

    anomaly_data = anomaly_data[
        [
            "date",
            "total_demand",
            "rolling_mean",
            "rolling_std",
            "z_score",
            "anomaly",
            "anomaly_type",
        ]
    ]

    anomaly_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return anomaly_data


def main():
    print(
        "Starting demand anomaly detection..."
    )

    df = load_data()

    print(
        f"Daily observations analyzed: "
        f"{len(df):,}"
    )

    df = detect_anomalies(
        df
    )

    anomaly_data = save_anomalies(
        df
    )

    spike_count = (
        anomaly_data[
            "anomaly_type"
        ]
        .eq("Demand Spike")
        .sum()
    )

    drop_count = (
        anomaly_data[
            "anomaly_type"
        ]
        .eq("Demand Drop")
        .sum()
    )

    print(
        "\n========================================"
    )

    print(
        "ANOMALY DETECTION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Total anomalies: "
        f"{len(anomaly_data):,}"
    )

    print(
        f"Demand spikes: "
        f"{spike_count:,}"
    )

    print(
        f"Demand drops: "
        f"{drop_count:,}"
    )

    print(
        "\nTop anomalies:"
    )

    print(
        anomaly_data
        .sort_values(
            "z_score",
            key=lambda x: x.abs(),
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\n========================================"
    )

    print(
        "ANOMALY DETECTION COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()