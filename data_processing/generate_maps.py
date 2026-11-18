
from pathlib import Path
import folium

from data_processing.data_loader import load_trip


# Select one representative training trip.
TRIP_ID = "00000001"

TRIP_PATH = Path("data/raw/Student folder/train") / TRIP_ID
OUTPUT_DIR = Path("outputs/maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the trip data.
trip = load_trip(TRIP_PATH)

gps = trip.gps
nodes = trip.nodes
arcs = trip.arcs
route = trip.reference_route

# Use the average GPS coordinate as the map centre.
centre_lat = gps["latitude"].mean()
centre_lon = gps["longitude"].mean()

# Create the interactive map.
map_view = folium.Map(
    location=[centre_lat, centre_lon],
    zoom_start=14,
    tiles="Esri.WorldStreetMap"
)

# Add the GPS trajectory as a blue line.
gps_coordinates = list(
    zip(gps["latitude"], gps["longitude"])
)

folium.PolyLine(
    gps_coordinates,
    color="blue",
    weight=4,
    opacity=0.8,
    tooltip="Observed GPS trajectory"
).add_to(map_view)

# Add the reference route reconstructed from arc IDs.
# Each arc identifies two node IDs.
route_coordinates = []

if route is not None:

    for arc_id in route:

        # Get the endpoints of the current reference arc.
        from_node = int(arcs.iloc[arc_id]["from_node"])
        to_node = int(arcs.iloc[arc_id]["to_node"])

        # Get the geographic coordinates of both nodes.
        start = nodes.loc[from_node]
        end = nodes.loc[to_node]

        # Folium expects coordinates in latitude, longitude order.
        route_coordinates.append(
            [start["latitude"], start["longitude"]]
        )

        route_coordinates.append(
            [end["latitude"], end["longitude"]]
        )

    # Draw the reference route in red.
    # Individual arc segments are drawn separately so that
    # disconnected transitions are not falsely joined.
    for i in range(0, len(route_coordinates) - 1, 2):

        folium.PolyLine(
            route_coordinates[i:i + 2],
            color="red",
            weight=3,
            opacity=0.7,
            tooltip="Reference route arc"
        ).add_to(map_view)

# Mark the first and last GPS observations.
folium.Marker(
    gps_coordinates[0],
    tooltip="Trip start",
    icon=folium.Icon(color="green")
).add_to(map_view)

folium.Marker(
    gps_coordinates[-1],
    tooltip="Trip end",
    icon=folium.Icon(color="red")
).add_to(map_view)

# Add a layer control so map layers can be inspected.
folium.LayerControl().add_to(map_view)

# Save the interactive map as an HTML file.
output_file = OUTPUT_DIR / f"trajectory_map_{TRIP_ID}.html"

map_view.save(str(output_file))

print("Interactive map generated successfully!")
print("Trip:", TRIP_ID)
print("GPS observations:", len(gps))
print("Reference route arcs:", 0 if route is None else len(route))
print("Saved to:", output_file)
