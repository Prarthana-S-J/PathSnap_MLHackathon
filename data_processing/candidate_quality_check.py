import numpy as np
import pandas as pd
from pathlib import Path

from data_processing.data_loader import load_trip
from data_processing.candidate_generation import to_xy, point_to_segment


# Load the same trip and its generated candidates.
trip = load_trip("data/raw/Student folder/train/00000001")

candidates = pd.read_csv(
    "outputs/statistics/candidates_00000001.csv"
)

gps = trip.gps
nodes = trip.nodes
arcs = trip.arcs

lon0 = gps["longitude"].mean()
lat0 = gps["latitude"].mean()

gps_xy = to_xy(
    gps.longitude.to_numpy(), gps.latitude.to_numpy(), lon0, lat0
)

node_xy = to_xy(
    nodes.longitude.to_numpy(), nodes.latitude.to_numpy(), lon0, lat0
)

starts = node_xy[arcs.from_node.to_numpy()]
ends = node_xy[arcs.to_node.to_numpy()]

# Get the reference route's actual road-segment IDs.
route_ids = trip.reference_route

route_starts = starts[route_ids]
route_ends = ends[route_ids]

# Measure each GPS point's distance to its nearest reference-route segment.
route_distances = []

for point in gps_xy:
    distances = [
        point_to_segment(point, start, end)[0]
        for start, end in zip(route_starts, route_ends)
    ]
    route_distances.append(min(distances))

# Check how many reference-route arcs appeared in our candidate pool.
candidate_arc_ids = set(candidates["arc_id"].unique())
route_arc_ids = set(route_ids.tolist())

found = candidate_arc_ids.intersection(route_arc_ids)

print("\n========== CANDIDATE QUALITY CHECK ==========")
print("Reference route arcs:", len(route_arc_ids))
print("Reference arcs found in candidate pool:", len(found))
print("Reference arc coverage:",
      round(100 * len(found) / len(route_arc_ids), 2), "%")

print("\nDistance from GPS points to reference route:")
print(pd.Series(route_distances).describe().round(2))

print("\nGPS points within 20m of reference route:",
      sum(d <= 20 for d in route_distances), "/", len(route_distances))