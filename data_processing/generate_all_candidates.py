from pathlib import Path
from data_processing.data_loader import load_trip
from data_processing.candidate_generation import generate_candidates

TRAIN_DIR = Path("data/raw/Student folder/train")
OUTPUT_DIR = Path("outputs/candidates/train")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

trip_dirs = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())

print("Training trips:", len(trip_dirs))

for i, trip_dir in enumerate(trip_dirs, 1):
    trip = load_trip(trip_dir)
    candidates = generate_candidates(trip)

    output_file = OUTPUT_DIR / f"{trip.trip_id}_candidates.csv"
    candidates.to_csv(output_file, index=False)

    print(
        f"[{i}/{len(trip_dirs)}] Trip {trip.trip_id}: "
        f"{len(candidates)} candidates saved"
    )

print("\nALL TRAINING CANDIDATES GENERATED")