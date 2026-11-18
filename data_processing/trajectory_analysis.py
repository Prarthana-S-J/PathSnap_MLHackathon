
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.data_loader import iter_partition


# Calculate the distance between consecutive GPS coordinates.
# Haversine gives us distance in metres using longitude and latitude.
def calculate_distances(gps):

    lon = np.radians(gps["longitude"].to_numpy())
    lat = np.radians(gps["latitude"].to_numpy())

    # Compare every GPS point with the next one.
    dlon = np.diff(lon)
    dlat = np.diff(lat)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:])
        * np.sin(dlon / 2) ** 2
    )

    # Earth's approximate mean radius in metres.
    earth_radius = 6371008.8

    distances = 2 * earth_radius * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    return distances


# Analyse the GPS trajectory of one trip.
def analyse_trip(trip, partition_name):

    gps = trip.gps

    # Calculate the time between consecutive GPS observations.
    time_gaps = gps["timestamp"].diff().dropna().to_numpy()

    # Calculate the distance travelled between consecutive observations.
    distances = calculate_distances(gps)

    # Keep only valid, non-negative time intervals for summary statistics.
    valid_gaps = time_gaps[
        np.isfinite(time_gaps) & (time_gaps >= 0)
    ]

    # Calculate the percentage of intervals that are exactly one second.
    one_second_percentage = (
        100 * np.mean(valid_gaps == 1)
        if len(valid_gaps) else np.nan
    )

    # Calculate the total distance between consecutive GPS observations.
    total_distance = np.sum(distances[np.isfinite(distances)])

    # Store one row of statistics for this trip.
    return {
        "partition": partition_name,
        "trip_id": trip.trip_id,
        "gps_observations": len(gps),
        "trajectory_duration_seconds": (
            gps["timestamp"].max() - gps["timestamp"].min()
        ),
        "median_sampling_interval_seconds": (
            np.median(valid_gaps) if len(valid_gaps) else np.nan
        ),
        "one_second_interval_percentage": one_second_percentage,
        "irregular_intervals": int(np.sum(valid_gaps > 1)),
        "maximum_time_gap_seconds": (
            np.max(valid_gaps) if len(valid_gaps) else np.nan
        ),
        "observed_path_distance_km": total_distance / 1000,
        "longitude_min": gps["longitude"].min(),
        "longitude_max": gps["longitude"].max(),
        "latitude_min": gps["latitude"].min(),
        "latitude_max": gps["latitude"].max()
    }, time_gaps


# Analyse every trip in a partition.
def analyse_partition(partition_path, partition_name):

    records = []
    all_time_gaps = []

    # Process one trip at a time to avoid loading all road networks into RAM.
    for trip in iter_partition(partition_path):

        record, time_gaps = analyse_trip(trip, partition_name)

        records.append(record)

        # Keep the GPS time intervals for the overall distribution plot.
        all_time_gaps.append(time_gaps)

    return records, all_time_gaps


if __name__ == "__main__":

    base_path = Path("data/raw/Student folder")

    output_stats = Path("outputs/statistics")
    output_plots = Path("outputs/plots")

    # Create output directories if they do not exist.
    output_stats.mkdir(parents=True, exist_ok=True)
    output_plots.mkdir(parents=True, exist_ok=True)

    # Analyse training and student-testing data separately.
    train_records, train_gaps = analyse_partition(
        base_path / "train",
        "train"
    )

    test_records, test_gaps = analyse_partition(
        base_path / "student_test",
        "student_test"
    )

    # Combine the statistics into one table.
    statistics = pd.DataFrame(train_records + test_records)

    # Save the trip-wise statistics.
    statistics.to_csv(
        output_stats / "trajectory_statistics.csv",
        index=False
    )

    # Combine time intervals for plotting.
    train_intervals = np.concatenate(train_gaps)
    test_intervals = np.concatenate(test_gaps)

    # Plot 1: Distribution of GPS sampling intervals.
    plt.figure(figsize=(10, 5))

    plt.hist(
        train_intervals,
        bins=50,
        range=(0, 10),
        alpha=0.6,
        label="Training"
    )

    plt.hist(
        test_intervals,
        bins=50,
        range=(0, 10),
        alpha=0.6,
        label="Student testing"
    )

    plt.xlabel("Time between GPS observations (seconds)")
    plt.ylabel("Number of intervals")
    plt.title("GPS Sampling Interval Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plots / "sampling_interval_distribution.png",
        dpi=300
    )

    plt.close()

    # Plot 2: Maximum GPS time gap for each trip.
    plt.figure(figsize=(14, 6))

    for partition_name, color in [
        ("train", "steelblue"),
        ("student_test", "darkorange")
    ]:

        subset = statistics[
            statistics["partition"] == partition_name
        ]

        plt.scatter(
            subset["trip_id"],
            subset["maximum_time_gap_seconds"],
            label=partition_name,
            alpha=0.7
        )

    plt.xlabel("Trip ID")
    plt.ylabel("Maximum time gap (seconds)")
    plt.title("Maximum GPS Time Gap per Trip")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plots / "maximum_gap_by_trip.png",
        dpi=300
    )

    plt.close()

    # Plot 3: GPS coordinate distribution.
    plt.figure(figsize=(10, 7))

    # Use the trip-wise coordinate bounds to show the geographical extent.
    for partition_name, color in [
        ("train", "steelblue"),
        ("student_test", "darkorange")
    ]:

        subset = statistics[
            statistics["partition"] == partition_name
        ]

        plt.scatter(
            (subset["longitude_min"] + subset["longitude_max"]) / 2,
            (subset["latitude_min"] + subset["latitude_max"]) / 2,
            label=partition_name,
            alpha=0.7
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographical Distribution of Trip Centres")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plots / "gps_coordinate_distribution.png",
        dpi=300
    )

    plt.close()

    # Print the overall results.
    print("\n========== TRAJECTORY ANALYSIS ==========")

    print("\nTrip statistics:")
    print(
        statistics.groupby("partition")[
            [
                "gps_observations",
                "median_sampling_interval_seconds",
                "one_second_interval_percentage",
                "maximum_time_gap_seconds",
                "observed_path_distance_km"
            ]
        ].median().round(2)
    )

    print("\nLargest GPS gaps:")
    print(
        statistics.nlargest(
            5,
            "maximum_time_gap_seconds"
        )[
            ["partition", "trip_id", "maximum_time_gap_seconds"]
        ].to_string(index=False)
    )

    print("\nSaved trajectory statistics and plots successfully.")
