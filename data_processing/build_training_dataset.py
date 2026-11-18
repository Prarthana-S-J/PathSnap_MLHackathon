
from pathlib import Path
import numpy as np
import pandas as pd

from data_processing.data_loader import load_trip
from data_processing.candidate_generation import (
    to_xy, point_to_segment
)

TRAIN_DIR = Path("data/raw/Student folder/train")
CANDIDATE_DIR = Path("outputs/candidates/train")
OUTPUT_DIR = Path("outputs/training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_training_rows = []
coverage_records = []

trip_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())

for number, trip_dir in enumerate(trip_dirs, 1):

    trip = load_trip(trip_dir)
    candidate_file = CANDIDATE_DIR / f"{trip.trip_id}_candidates.csv"
    candidates = pd.read_csv(candidate_file)

    gps = trip.gps
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.reference_route

    lon0 = gps.longitude.mean()
    lat0 = gps.latitude.mean()

    gps_xy = to_xy(
        gps.longitude.to_numpy(),
        gps.latitude.to_numpy(), lon0, lat0
    )

    node_xy = to_xy(
        nodes.longitude.to_numpy(),
        nodes.latitude.to_numpy(), lon0, lat0
    )

    starts = node_xy[arcs.from_node.to_numpy()]
    ends = node_xy[arcs.to_node.to_numpy()]

    # Get the road segments in reference-route order.
    route_starts = starts[route]
    route_ends = ends[route]

    # Find the nearest reference-route segment for each GPS point.
    nearest_route_arcs = []

    for point in gps_xy:
        distances = [
            point_to_segment(point, start, end)[0]
            for start, end in zip(route_starts, route_ends)
        ]

        route_position = int(np.argmin(distances))
        nearest_route_arcs.append(
            (int(route[route_position]), route_position,
             float(distances[route_position]))
        )

    candidates["label"] = 0
    candidates["reference_route_position"] = -1

    matched_observations = 0

    # Assign positive labels only when the reference candidate exists.
    for gps_index, (arc_id, route_position, route_distance) in enumerate(
        nearest_route_arcs
    ):
        mask = (
            (candidates["gps_index"] == gps_index) &
            (candidates["arc_id"] == arc_id)
        )

        if mask.any():
            candidates.loc[mask, "label"] = 1
            candidates.loc[mask, "reference_route_position"] = route_position
            matched_observations += 1

        coverage_records.append({
            "trip_id": trip.trip_id,
            "gps_index": gps_index,
            "reference_arc_id": arc_id,
            "route_position": route_position,
            "reference_distance_m": route_distance,
            "candidate_found": bool(mask.any())
        })

    # Keep only GPS observations with a positive candidate.
    valid_indices = {
        i for i, (arc_id, _, _) in enumerate(nearest_route_arcs)
        if (
            (candidates["gps_index"] == i) &
            (candidates["arc_id"] == arc_id)
        ).any()
    }

    valid_candidates = candidates[
        candidates["gps_index"].isin(valid_indices)
    ].copy()

    all_training_rows.append(valid_candidates)

    print(
        f"[{number}/80] Trip {trip.trip_id}: "
        f"{matched_observations}/{len(gps)} observations matched"
    )

# Combine all trips.
training_data = pd.concat(all_training_rows, ignore_index=True)
coverage = pd.DataFrame(coverage_records)

training_data.to_csv(
    OUTPUT_DIR / "training_candidates.csv", index=False
)

coverage.to_csv(
    OUTPUT_DIR / "label_alignment_audit.csv", index=False
)

print("\n========== TRAINING DATASET SUMMARY ==========")
print("Training rows:", len(training_data))
print("Positive labels:", int(training_data.label.sum()))
print("Negative labels:", int((training_data.label == 0).sum()))
print("Unique trips:", training_data.trip_id.nunique())
print(
    "Observation coverage:",
    round(100 * coverage.candidate_found.mean(), 2), "%"
)
print("\nSaved training dataset and alignment audit.")
