import pandas as pd
from pathlib import Path


# Config
df = pd.read_csv('analysis/all_model_results.csv')

# Gets best model timeframe and news option
timeframe = df.iloc[0, 0]
news_type = df.iloc[0, 1]
timeframe = timeframe.split("_")[-1]

ANALYSIS_DIR = Path("analysis/directional")

PREDICTIONS_FILE = (ANALYSIS_DIR / timeframe / news_type / "predictions.csv")
FEATURE_FILE = (ANALYSIS_DIR / timeframe / news_type / "feature_importance.csv")
DATA_FILE = (Path("data/targets") /"article_training_dataset.csv")
OUTPUT_FILE = (ANALYSIS_DIR /"winning_model_summary.txt")

NEWS_SOURCES = [
    "Reuters",
    "Benzinga",
    "GlobeNewswire Inc.",
    "Business Wire",
    "PR Newswire"
]

# Load files
preds = pd.read_csv(PREDICTIONS_FILE)
fi = pd.read_csv(FEATURE_FILE)
full_data = pd.read_csv(DATA_FILE)

# Model performance
overall_accuracy = (preds["actual"] == preds["predicted"]).mean()
total_predictions = len(preds)

# Positive predictors
top_positive = (fi.sort_values("coefficient", ascending=False).head(25))

# Negative predictors
top_negative = (fi.sort_values("coefficient", ascending=True).head(25))

# Confidence analysis
confidence_results = []

for threshold in [0.60, 0.70, 0.80, 0.90]:
    subset = preds[
        (preds["prediction_probability"] >= threshold) | (preds["prediction_probability"] <= (1 - threshold))
    ]

    if len(subset) == 0:
        continue

    accuracy = (subset["actual"] == subset["predicted"]).mean()
    coverage = (len(subset) / len(preds))

    confidence_results.append(
        {
            "threshold": threshold,
            "observations": len(subset),
            "coverage": coverage,
            "accuracy": accuracy
        }
    )

confidence_df = pd.DataFrame(confidence_results)

# Source level performance
full_data["published"] = pd.to_datetime(full_data["published"])
full_data = full_data[full_data["source"].isin(NEWS_SOURCES)]
full_data = (full_data.sort_values("published").reset_index(drop=True))

split_idx = int(len(full_data) * 0.80)
test_rows = (full_data.iloc[split_idx:].reset_index(drop=True))

if len(test_rows) == len(preds):
    source_predictions = pd.concat(
        [test_rows[["source", "headline"]].reset_index(drop=True), preds.reset_index(drop=True)], axis=1
    )

    source_summary = (source_predictions.groupby("source").apply(
            lambda x: pd.Series(
                {
                    "count": len(x),
                    "accuracy": (x["actual"] == x["predicted"]).mean()
                }
            )
        ).reset_index().sort_values("accuracy", ascending=False)
    )

else:
    source_summary = pd.DataFrame({"source": ["Could not align source data"]})

# Write report
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("WINNING MODEL ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write("OVERALL PERFORMANCE\n")
    f.write("-" * 80 + "\n")

    f.write(f"Predictions: {total_predictions}\n")
    f.write(f"Accuracy: {overall_accuracy:.4f}\n\n")

    f.write("TOP 25 POSITIVE PREDICTORS\n")
    f.write("-" * 80 + "\n")

    for _, row in top_positive.iterrows():
        f.write(f"{row['feature']}: {row['coefficient']:.6f}\n")

    f.write("\n")

    f.write("TOP 25 NEGATIVE PREDICTORS\n")
    f.write("-" * 80 + "\n")

    for _, row in top_negative.iterrows():
        f.write(f"{row['feature']}: {row['coefficient']:.6f}\n")

    f.write("\n")

    f.write("CONFIDENCE FILTER ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(confidence_df.to_string(index=False))
    f.write("\n\n")

    f.write("SOURCE LEVEL PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(source_summary.to_string(index=False))

print()
print("=" * 80)
print("WINNING MODEL ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Saved: {OUTPUT_FILE}")