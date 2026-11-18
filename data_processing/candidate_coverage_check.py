import pandas as pd
import numpy as np

from data_processing.data_loader import load_trip
from data_processing.candidate_generation import (
    to_xy, point_to_segment
)

# Load the trip and its generated candidates.
trip = load_trip("data/raw/Student folder/train/00000001")

candidates = pd.read_csv(
    "outputs/statistics/candidates_00000001.csv"
)

gps = trip.gps
nodes = trip.nodes
arcs = trip.arcs
route = trip.reference_route

# Convert GPS and road coordinates into local metres.
lon0 = gps.longitude.mean()
lat0 = gps.latitude.mean()

gps_xy = to_xy(
    gps.longitude.to_numpy(),
    gps.latitude.to_numpy(),
    lon0, lat0
)

node_xy = to_xy(
    nodes.longitude.to_numpy(),
    nodes.latitude.to_numpy(),
    lon0, lat0
)

starts = node_xy[arcs.from_node.to_numpy()]
ends = node_xy[arcs.to_node.to_numpy()]

# Find the closest reference-route arc for each GPS point.
nearest_route_arcs = []

for point in gps_xy:
    distances = [
        point_to_segment(point, starts[a], ends[a])[0]
        for a in route
    ]

    nearest_route_arcs.append(int(route[np.argmin(distances)]))

# Check whether that arc is present in the GPS point's candidates.
candidate_sets = candidates.groupby("gps_index")["arc_id"].apply(set)

found = [
    arc_id in candidate_sets.loc[i]
    for i, arc_id in enumerate(nearest_route_arcs)
]

print("\n========== PER-OBSERVATION COVERAGE ==========")
print("GPS observations:", len(found))
print("Correct nearest route arc found:", sum(found))
print("Missing:", len(found) - sum(found))
print("Coverage:", round(100 * np.mean(found), 2), "%")