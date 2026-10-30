from pathlib import Path
import pandas as pd
import re


# Config
ANALYSIS_DIR = Path("analysis/directional")
OUTPUT_FILE = ANALYSIS_DIR / "all_model_results.csv"

# Metric files
metrics_files = sorted(ANALYSIS_DIR.rglob("*_metrics.txt"))

rows = []

# Parse files
for file in metrics_files:
    text = file.read_text(encoding="utf-8")
    target = file.stem.replace("_metrics", "")

    accuracy = re.search(r"Accuracy\s*:\s*([\d.]+)", text)
    precision = re.search(r"Precision\s*:\s*([\d.]+)", text)
    recall = re.search(r"Recall\s*:\s*([\d.]+)", text)
    f1 = re.search(r"F1 Score\s*:\s*([\d.]+)", text)

    rows.append(
        {
            "target": target,
            "accuracy": float(accuracy.group(1)) if accuracy else None,
            "precision": float(precision.group(1)) if precision else None,
            "recall": float(recall.group(1)) if recall else None,
            "f1_score": float(f1.group(1)) if f1 else None,
        }
    )

# Create summary
summary = pd.DataFrame(rows)
summary = summary.sort_values("accuracy", ascending=False)
summary.to_csv(OUTPUT_FILE, index=False)

# Print results
print()
print("Model Summary")
print("=" * 60)

print(summary)

print()
print(f"Saved: {OUTPUT_FILE}")