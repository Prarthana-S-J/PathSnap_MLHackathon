
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Store all four files of one trip together.
# This makes it easier to pass trip data between different scripts.
@dataclass
class TripData:
    trip_id: str
    gps: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    reference_route: Optional[np.ndarray]


# Load one trip from its folder.
def load_trip(trip_dir, include_reference_route=True):

    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    # Build the expected file paths using the trip ID.
    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    # The first three files are needed to describe the input.
    required_files = [track_file, nodes_file, arcs_file]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing required file: {file}")

    # GPS observations: longitude, latitude and timestamp.
    gps = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    # Road-network nodes: longitude and latitude.
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    # Road-network arcs: starting node and ending node.
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    # Row positions are the actual node IDs and arc IDs.
    # Do not reset or reorder these indexes.
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    # The reference route contains arc IDs in chronological order.
    # Keep it separate from the GPS observations.
    reference_route = None

    if include_reference_route and route_file.exists():

        route_data = pd.read_csv(
            route_file,
            sep=r"\s+",
            header=None
        )

        # Store the route as a one-dimensional integer array.
        reference_route = pd.to_numeric(
            route_data.iloc[:, 0],
            errors="raise"
        ).to_numpy(dtype=np.int64)

    # Return all loaded information as one TripData object.
    return TripData(
        trip_id=trip_id,
        gps=gps,
        nodes=nodes,
        arcs=arcs,
        reference_route=reference_route
    )


# Find all trip folders inside a partition.
def get_trip_directories(partition_dir):

    partition_dir = Path(partition_dir)

    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Partition directory not found: {partition_dir}"
        )

    # Sort the folders so trips are processed in a predictable order.
    return sorted(
        [folder for folder in partition_dir.iterdir() if folder.is_dir()],
        key=lambda folder: folder.name
    )


# Load trips one by one without keeping every large network in memory.
def iter_partition(partition_dir, include_reference_route=True):

    for trip_dir in get_trip_directories(partition_dir):

        yield load_trip(
            trip_dir,
            include_reference_route=include_reference_route
        )
