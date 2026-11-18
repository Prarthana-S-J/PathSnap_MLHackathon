
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

# Files and folders
DATA = "outputs/training/training_candidates.csv"
OUT = Path("outputs/models")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "distance_m",
    "heading_difference_deg",
    "arc_length_m",
    "position_on_arc"
]

# Load the labelled training data.
df = pd.read_csv(DATA)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["label", "trip_id"])

X = df[FEATURES]
y = df["label"].astype(int)
groups = df["trip_id"]

print("Dataset:", X.shape)
print("Trips:", groups.nunique())

# Split by trip, never by individual candidate row.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
y_val = y.iloc[val_idx].copy()

print("Training trips:", groups.iloc[train_idx].nunique())
print("Validation trips:", groups.iloc[val_idx].nunique())

# Downsample negatives in TRAINING only.
# Keep all positive examples and up to 3 negatives per positive.
train_frame = X_train.copy()
train_frame["label"] = y_train.values

positive = train_frame[train_frame.label == 1]
negative = train_frame[train_frame.label == 0]

negative_sample = negative.sample(
    n=min(len(negative), 3 * len(positive)),
    random_state=42
)

balanced = pd.concat(
    [positive, negative_sample]
).sample(frac=1, random_state=42)

X_fit = balanced[FEATURES]
y_fit = balanced["label"]

print("\nBalanced training rows:", len(X_fit))
print("Positive:", int(y_fit.sum()))
print("Negative:", int((y_fit == 0).sum()))

# Fit the imputer using training data only.
imputer = SimpleImputer(strategy="median")
X_fit_imputed = imputer.fit_transform(X_fit)

# Select a representative sample to speed up feature selection.
rng = np.random.default_rng(42)

sample_size = min(100000, len(X_fit_imputed))

sample_indices = rng.choice(
    len(X_fit_imputed),
    size=sample_size,
    replace=False
)

# Calculate Mutual Information on the sample.
mi_scores = mutual_info_classif(
    X_fit_imputed[sample_indices],
    y_fit.iloc[sample_indices],
    random_state=42
)

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "mutual_information": mi_scores
}).sort_values("mutual_information", ascending=False)

feature_scores.to_csv(
    OUT / "feature_selection_scores.csv", index=False
)

selected = feature_scores.head(3)["feature"].tolist()

print("\nFeature selection results:")
print(feature_scores.to_string(index=False))
print("\nSelected features:", selected)

X_fit = X_fit[selected]
X_val_selected = X_val[selected]

# Five different model families.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=80, max_depth=18,
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=80, max_depth=18,
        class_weight="balanced",
        n_jobs=-1, random_state=42
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=31, random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    # Scale features for Logistic Regression.
    if name == "Logistic Regression":
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            model
        )
    else:
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            model
        )

    pipeline.fit(X_fit, y_fit)

    predictions = pipeline.predict(X_val_selected)
    probabilities = pipeline.predict_proba(X_val_selected)[:, 1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_val, probabilities
        ),
        "training_seconds": round(time.time() - start, 2)
    })

    joblib.dump(
        pipeline, OUT / (name.lower().replace(" ", "_") + ".joblib")
    )

    print("Completed in", round(time.time() - start, 2), "seconds")

# Save comparison results.
results_df = pd.DataFrame(results).sort_values(
    "average_precision", ascending=False
)

results_df.to_csv(OUT / "model_comparison.csv", index=False)

print("\n========== FIVE MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
print("\nSelected features saved.")
print("Trained models saved in:", OUT)
