
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from data_processing.data_loader import iter_partition


def analyse_network(trip, partition):

    nodes = trip.nodes
    arcs = trip.arcs

    node_count = len(nodes)
    arc_count = len(arcs)

    # Get the starting and ending node IDs for every arc.
    edges = arcs[["from_node", "to_node"]].to_numpy(dtype=int)

    # Count how many arcs enter and leave each node.
    in_degree = np.bincount(edges[:, 1], minlength=node_count)
    out_degree = np.bincount(edges[:, 0], minlength=node_count)

    # A node is isolated if no arc enters or leaves it.
    isolated_nodes = int(
        np.sum((in_degree == 0) & (out_degree == 0))
    )

    # Create a sparse adjacency matrix for the road network.
    graph = csr_matrix(
        (
            np.ones(arc_count, dtype=np.int8),
            (edges[:, 0], edges[:, 1])
        ),
        shape=(node_count, node_count)
    )

    # Find weakly connected components.
    # This treats connections as undirected for connectivity analysis.
    component_count, labels = connected_components(
        graph,
        directed=True,
        connection="weak",
        return_labels=True
    )

    # Count the number of nodes in each component.
    component_sizes = np.bincount(labels)

    largest_component = (
        int(component_sizes.max()) if len(component_sizes) else 0
    )

    # Count duplicate directed arcs.
    unique_arcs = len(np.unique(edges, axis=0))

    return {
        "partition": partition,
        "trip_id": trip.trip_id,
        "node_count": node_count,
        "arc_count": arc_count,
        "isolated_nodes": isolated_nodes,
        "nodes_without_incoming_arcs": int(np.sum(in_degree == 0)),
        "nodes_without_outgoing_arcs": int(np.sum(out_degree == 0)),
        "weakly_connected_components": int(component_count),
        "largest_component_nodes": largest_component,
        "largest_component_percentage": (
            round(100 * largest_component / node_count, 2)
            if node_count else 0
        ),
        "duplicate_directed_arcs": arc_count - unique_arcs
    }


if __name__ == "__main__":

    base = Path("data/raw/Student folder")
    output = Path("outputs/statistics")
    output.mkdir(parents=True, exist_ok=True)

    records = []

    # Analyse both available partitions separately.
    for partition in ["train", "student_test"]:

        for trip in iter_partition(base / partition):

            records.append(
                analyse_network(trip, partition)
            )

    report = pd.DataFrame(records)

    # Save the network statistics for all trips.
    report.to_csv(
        output / "road_network_statistics.csv",
        index=False
    )

    print("\n========== ROAD NETWORK ANALYSIS ==========")
    print("Networks analysed:", len(report))

    print("\nNetwork statistics (median):")
    print(
        report.groupby("partition")[
            [
                "node_count",
                "arc_count",
                "isolated_nodes",
                "weakly_connected_components",
                "largest_component_percentage"
            ]
        ].median().round(2).to_string()
    )

    print("\nNetworks with the most disconnected components:")
    print(
        report.nlargest(5, "weakly_connected_components")[
            ["partition", "trip_id", "weakly_connected_components",
             "isolated_nodes", "largest_component_percentage"]
        ].to_string(index=False)
    )

    print("\nSaved: outputs/statistics/road_network_statistics.csv")
