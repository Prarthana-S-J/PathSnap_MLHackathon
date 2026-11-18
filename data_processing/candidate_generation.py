from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.data_loader import load_trip


def to_xy(lon, lat, lon0, lat0):
    x = (lon - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110540
    return np.column_stack((x, y))


def point_to_segment(point, start, end):
    vector = end - start
    length_sq = np.dot(vector, vector)

    if length_sq == 0:
        return np.linalg.norm(point - start), 0.0

    t = np.dot(point - start, vector) / length_sq
    t = np.clip(t, 0, 1)

    projection = start + t * vector
    return np.linalg.norm(point - projection), t


def generate_candidates(trip, top_k=20):
    gps = trip.gps
    nodes = trip.nodes
    arcs = trip.arcs

    lon0 = gps.longitude.mean()
    lat0 = gps.latitude.mean()

    gps_xy = to_xy(gps.longitude.to_numpy(),
                   gps.latitude.to_numpy(), lon0, lat0)

    node_xy = to_xy(nodes.longitude.to_numpy(),
                    nodes.latitude.to_numpy(), lon0, lat0)

    from_nodes = arcs.from_node.to_numpy()
    to_nodes = arcs.to_node.to_numpy()

    starts = node_xy[from_nodes]
    ends = node_xy[to_nodes]
    midpoints = (starts + ends) / 2

    midpoint_tree = cKDTree(midpoints)
    node_tree = cKDTree(node_xy)

    # Store every road arc connected to each node.
    node_arcs = [[] for _ in range(len(nodes))]

    for arc_id, (u, v) in enumerate(zip(from_nodes, to_nodes)):
        node_arcs[u].append(arc_id)
        node_arcs[v].append(arc_id)

    rows = []

    for i, point in enumerate(gps_xy):

        # Search a wider area using road midpoints.
        k = min(100, len(arcs))
        _, nearby_midpoints = midpoint_tree.query(point, k=k)
        nearby_midpoints = np.atleast_1d(nearby_midpoints)

        # Also collect roads connected to nearby nodes.
        node_k = min(20, len(nodes))
        _, nearby_nodes = node_tree.query(point, k=node_k)
        nearby_nodes = np.atleast_1d(nearby_nodes)

        possible_arcs = set(nearby_midpoints.tolist())

        for node_id in nearby_nodes:
            possible_arcs.update(node_arcs[int(node_id)])

        # Calculate actual distance to each candidate road.
        scored = []

        for arc_id in possible_arcs:
            distance, position = point_to_segment(
                point, starts[arc_id], ends[arc_id]
            )

            vector = ends[arc_id] - starts[arc_id]
            length = np.linalg.norm(vector)

            road_heading = np.degrees(
                np.arctan2(vector[0], vector[1])
            )

            if i < len(gps_xy) - 1:
                movement = gps_xy[i + 1] - point
                gps_heading = np.degrees(
                    np.arctan2(movement[0], movement[1])
                )

                difference = abs(
                    (gps_heading - road_heading + 180) % 360 - 180
                )
            else:
                difference = np.nan

            scored.append((
                distance, int(arc_id), difference, length, position
            ))

        # Keep the closest actual road segments.
        scored.sort(key=lambda x: x[0])

        for distance, arc_id, difference, length, position in scored[:top_k]:
            rows.append({
                "trip_id": trip.trip_id,
                "gps_index": i,
                "arc_id": arc_id,
                "distance_m": distance,
                "heading_difference_deg": difference,
                "arc_length_m": length,
                "position_on_arc": position
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    trip = load_trip("data/raw/Student folder/train/00000001")

    candidates = generate_candidates(trip)

    output = Path("outputs/statistics")
    output.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(
        output / "candidates_00000001.csv", index=False
    )

    print("GPS observations:", len(trip.gps))
    print("Candidates generated:", len(candidates))
    print("Candidates per GPS point:")
    print(candidates.groupby("gps_index").size().describe())
    print("Nearest candidate distance:")
    print(candidates.groupby("gps_index").distance_m.min().describe())
    print("Saved improved candidates.")