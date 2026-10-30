from pathlib import Path
import pandas as pd


# Config
ANALYSIS_DIR = Path("analysis/directional")
OUTPUT_FILE = ANALYSIS_DIR / "baseline_comparison_summary.csv"

# Load prediction files
prediction_files = sorted(ANALYSIS_DIR.rglob("*_predictions.csv"))

rows = []

for file in prediction_files:
    df = pd.read_csv(file)

    model_name = file.stem.replace("_predictions", "")
    model_accuracy = (df["actual"] == df["predicted"]).mean()

    majority_class = (df["actual"].mode()[0])
    baseline_accuracy = (df["actual"] == majority_class).mean()
    improvement = (model_accuracy - baseline_accuracy)

    rows.append(
        {
            "model": model_name,
            "model_accuracy": round(model_accuracy, 6),
            "baseline_accuracy": round(baseline_accuracy, 6),
            "improvement": round(improvement, 6),
            "majority_class": majority_class
        }
    )

summary = pd.DataFrame(rows)
summary = summary.sort_values("improvement", ascending=False)

summary.to_csv(OUTPUT_FILE, index=False)

print("\nResults\n")
print(summary)

print("\nSaved:")
print(OUTPUT_FILE)