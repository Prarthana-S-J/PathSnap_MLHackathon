
from pathlib import Path
import numpy as np
import pandas as pd

from data_processing.data_loader import (
    get_trip_directories,
    load_trip
)


# Validate one trip and return all problems found.
def validate_trip(trip):

    issues = []

    gps = trip.gps
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.reference_route

    # 1. Check whether the files contain data.
    if gps.empty:
        issues.append("Empty GPS file")

    if nodes.empty:
        issues.append("Empty node file")

    if arcs.empty:
        issues.append("Empty arc file")

    # 2. Check for missing values.
    if gps.isnull().any().any():
        issues.append("Missing GPS values")

    if nodes.isnull().any().any():
        issues.append("Missing node coordinates")

    if arcs.isnull().any().any():
        issues.append("Missing arc endpoints")

    # 3. Check whether GPS and node coordinates are valid numbers.
    if not gps.empty:
        gps_values = gps[["longitude", "latitude", "timestamp"]].to_numpy()

        if not np.isfinite(gps_values).all():
            issues.append("Non-finite GPS values")

        if not gps["longitude"].between(-180, 180).all():
            issues.append("GPS longitude outside valid range")

        if not gps["latitude"].between(-90, 90).all():
            issues.append("GPS latitude outside valid range")

    if not nodes.empty:
        node_values = nodes[["longitude", "latitude"]].to_numpy()

        if not np.isfinite(node_values).all():
            issues.append("Non-finite node coordinates")

        if not nodes["longitude"].between(-180, 180).all():
            issues.append("Node longitude outside valid range")

        if not nodes["latitude"].between(-90, 90).all():
            issues.append("Node latitude outside valid range")

    # 4. Check timestamp ordering and calculate time gaps.
    max_time_gap = None
    irregular_intervals = 0

    if len(gps) > 1 and np.isfinite(gps["timestamp"]).all():

        time_gaps = gps["timestamp"].diff().dropna()

        if (time_gaps < 0).any():
            issues.append("Timestamps decrease")

        if (time_gaps == 0).any():
            issues.append("Repeated consecutive timestamps")

        # Count intervals longer than the nominal one-second interval.
        irregular_intervals = int((time_gaps > 1).sum())

        max_time_gap = float(time_gaps.max())

    # 5. Check whether every arc refers to existing node IDs.
    if not arcs.empty:

        endpoints = arcs[["from_node", "to_node"]].to_numpy()

        if not np.isfinite(endpoints).all():
            issues.append("Non-finite arc endpoints")

        elif not np.equal(endpoints, np.floor(endpoints)).all():
            issues.append("Non-integer arc endpoints")

        else:
            endpoints = endpoints.astype(np.int64)

            if (
                (endpoints < 0).any()
                or (endpoints >= len(nodes)).any()
            ):
                issues.append("Arc references invalid node ID")

    # 6. Check whether the reference route uses valid arc IDs.
    if route is None:
        issues.append("Reference route file unavailable")

    elif len(route) == 0:
        issues.append("Reference route is empty")

    elif (
        (route < 0).any()
        or (route >= len(arcs)).any()
    ):
        issues.append("Reference route contains invalid arc ID")

    return issues, irregular_intervals, max_time_gap


# Run validation across one complete partition.
def validate_partition(partition_path, partition_name):

    records = []

    trip_dirs = get_trip_directories(partition_path)

    for trip_dir in trip_dirs:

        trip_id = trip_dir.name

        try:
            # Load only the current trip into memory.
            trip = load_trip(trip_dir)

            issues, irregular, max_gap = validate_trip(trip)

            records.append({
                "partition": partition_name,
                "trip_id": trip_id,
                "status": "PASS" if not issues else "CHECK",
                "issue_count": len(issues),
                "issues": "; ".join(issues),
                "irregular_intervals": irregular,
                "maximum_time_gap_seconds": max_gap
            })

        except Exception as error:

            # Record loading problems instead of stopping the whole analysis.
            records.append({
                "partition": partition_name,
                "trip_id": trip_id,
                "status": "ERROR",
                "issue_count": 1,
                "issues": str(error),
                "irregular_intervals": None,
                "maximum_time_gap_seconds": None
            })

    return records


# Validate both available partitions and save the report.
if __name__ == "__main__":

    base_path = Path("data/raw/Student folder")

    train_path = base_path / "train"
    test_path = base_path / "student_test"

    output_dir = Path("outputs/statistics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyse the two partitions separately.
    all_records = []

    all_records.extend(
        validate_partition(train_path, "train")
    )

    all_records.extend(
        validate_partition(test_path, "student_test")
    )

    # Convert the results into a table.
    report = pd.DataFrame(all_records)

    # Save the complete validation report.
    output_file = output_dir / "data_validation_report.csv"
    report.to_csv(output_file, index=False)

    # Print a compact summary.
    print("\n========== DATA VALIDATION SUMMARY ==========")

    print("\nTrips checked:", len(report))

    print("\nStatus breakdown:")
    print(report["status"].value_counts())

    print("\nTrips requiring attention:")
    print(
        report.loc[
            report["status"] != "PASS",
            ["partition", "trip_id", "status", "issues"]
        ].to_string(index=False)
    )

    print("\nValidation report saved to:", output_file)
