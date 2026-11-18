import pandas as pd
import numpy as np

data = pd.read_csv("outputs/training/training_candidates.csv")
audit = pd.read_csv("outputs/training/label_alignment_audit.csv")

features = [
    "distance_m",
    "heading_difference_deg",
    "arc_length_m",
    "position_on_arc"
]

print("\n========== LABEL AND FEATURE CHECK ==========")

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nPositive percentage:")
print(round(data["label"].mean() * 100, 2), "%")

print("\nMissing feature values:")
print(data[features].isna().sum())

print("\nInfinite feature values:")
print(np.isinf(data[features]).sum())

print("\nObservation coverage:")
print(round(audit["candidate_found"].mean() * 100, 2), "%")

# Check whether selected reference-route positions
# generally move forward through each GPS trajectory.
backward_jumps = []

for trip_id, group in audit.groupby("trip_id"):
    positions = group.sort_values("gps_index")[
        "route_position"
    ].to_numpy()

    backward_jumps.append(
        np.sum(np.diff(positions) < -5)
    )

print("\nLarge backward route-order jumps:", sum(backward_jumps))
print("Trips checked:", len(backward_jumps))