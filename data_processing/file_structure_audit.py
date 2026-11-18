
from pathlib import Path
import csv
import pandas as pd

BASE = Path("data/raw/Student folder")
OUTPUT = Path("outputs/statistics")
OUTPUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ".track": 3,
    ".nodes": 2,
    ".arcs": 2,
    ".route": 1
}

records = []

# Check every file in every available trip.
for partition in ["train", "student_test"]:
    partition_path = BASE / partition

    for trip_dir in sorted(partition_path.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        for extension, expected_columns in EXPECTED.items():
            file_path = trip_dir / f"{trip_id}{extension}"

            result = {
                "partition": partition,
                "trip_id": trip_id,
                "file_type": extension,
                "exists": file_path.exists(),
                "expected_columns": expected_columns,
                "actual_columns": None,
                "malformed_rows": 0,
                "non_numeric_rows": 0,
                "status": "PASS"
            }

            if not file_path.exists():
                result["status"] = "MISSING"
                records.append(result)
                continue

            actual_columns = set()

            # Read each line and check its structure.
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    values = line.split()
                    actual_columns.add(len(values))

                    if len(values) != expected_columns:
                        result["malformed_rows"] += 1
                        continue

                    # Check whether every value can be interpreted as a number.
                    try:
                        [float(value) for value in values]
                    except ValueError:
                        result["non_numeric_rows"] += 1

            result["actual_columns"] = ",".join(
                map(str, sorted(actual_columns))
            )

            if actual_columns != {expected_columns}:
                result["status"] = "CHECK"

            if result["malformed_rows"] or result["non_numeric_rows"]:
                result["status"] = "CHECK"

            records.append(result)

# Save the audit report.
report = pd.DataFrame(records)
report.to_csv(OUTPUT / "file_structure_audit.csv", index=False)

print("\nFILE STRUCTURE AUDIT")
print("Files checked:", len(report))
print("\nStatus:")
print(report["status"].value_counts().to_string())

print("\nIssues:")
print(
    report[report["status"] != "PASS"][
        ["partition", "trip_id", "file_type",
         "actual_columns", "malformed_rows",
         "non_numeric_rows", "status"]
    ].to_string(index=False)
)

print("\nSaved: outputs/statistics/file_structure_audit.csv")
