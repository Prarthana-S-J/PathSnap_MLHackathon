
from pathlib import Path
import pandas as pd

from data_processing.data_loader import get_trip_directories


# Find the size of a file in megabytes.
def get_file_size(file_path):

    if file_path.exists():
        return round(file_path.stat().st_size / (1024 * 1024), 4)

    return None


# Collect information about every trip in a partition.
def create_inventory(partition_dir, partition_name):

    partition_dir = Path(partition_dir)

    records = []

    # Process each trip folder separately.
    for trip_dir in get_trip_directories(partition_dir):

        trip_id = trip_dir.name

        # Find the four expected files.
        track_file = trip_dir / f"{trip_id}.track"
        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"
        route_file = trip_dir / f"{trip_id}.route"

        files = {
            "track": track_file,
            "nodes": nodes_file,
            "arcs": arcs_file,
            "route": route_file
        }

        # Record which files are available.
        file_status = {
            name: file.exists()
            for name, file in files.items()
        }

        # Count rows without loading the entire file into memory.
        def count_rows(file_path):

            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as file:
                return sum(1 for line in file)

        # Create one record for this trip.
        record = {
            "partition": partition_name,
            "trip_id": trip_id,

            "track_exists": file_status["track"],
            "nodes_exists": file_status["nodes"],
            "arcs_exists": file_status["arcs"],
            "route_exists": file_status["route"],

            "gps_observations": count_rows(track_file),
            "node_count": count_rows(nodes_file),
            "arc_count": count_rows(arcs_file),
            "route_length": count_rows(route_file),

            "track_size_mb": get_file_size(track_file),
            "nodes_size_mb": get_file_size(nodes_file),
            "arcs_size_mb": get_file_size(arcs_file),
            "route_size_mb": get_file_size(route_file)
        }

        records.append(record)

    return pd.DataFrame(records)


# Run inventory for both available partitions.
if __name__ == "__main__":

    train_path = Path("data/raw/Student folder/train")
    student_test_path = Path("data/raw/Student folder/student_test")

    output_dir = Path("outputs/statistics")

    # Create the output folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyse the training partition.
    train_inventory = create_inventory(
        train_path,
        "train"
    )

    # Analyse the student-testing partition separately.
    test_inventory = create_inventory(
        student_test_path,
        "student_test"
    )

    # Combine the two inventories into one table.
    inventory = pd.concat(
        [train_inventory, test_inventory],
        ignore_index=True
    )

    # Save the results.
    output_file = output_dir / "dataset_inventory.csv"

    inventory.to_csv(output_file, index=False)

    # Display a short summary in the terminal.
    print("\nDATASET INVENTORY")
    print("-----------------")

    print("Total trips:", len(inventory))

    print("\nTrips per partition:")
    print(inventory.groupby("partition")["trip_id"].count())

    print("\nInventory saved to:", output_file)

    print("\nFirst few records:")
    print(inventory.head())
