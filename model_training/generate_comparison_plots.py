
import os
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(
    BASE_DIR, "outputs", "models", "model_comparison.csv"
)
OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load actual model comparison results
df = pd.read_csv(RESULTS_FILE)

# Metrics to visualize
metrics = {
    "accuracy": "Accuracy Comparison",
    "precision": "Precision Comparison",
    "recall": "Recall Comparison",
    "f1": "F1-Score Comparison",
    "average_precision": "Average Precision Comparison"
}

# Generate one graph for each metric
for metric, title in metrics.items():

    if metric not in df.columns:
        print(f"Column '{metric}' not found. Skipping.")
        continue

    # Sort models for easier comparison
    plot_df = df.sort_values(by=metric, ascending=True)

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        plot_df["model"],
        plot_df[metric]
    )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(metric.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Machine Learning Models")

    plt.xlim(0, 1.08)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.gca().set_axisbelow(True)

    # Display exact metric values
    for bar, value in zip(bars, plot_df[metric]):
        plt.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10
        )

    plt.tight_layout()

    filename = f"{metric}_comparison.png"
    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

print("\nAll comparison plots generated successfully!")
