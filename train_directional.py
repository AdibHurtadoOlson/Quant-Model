import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Config
DATA_FILE = "data/targets/article_training_dataset.csv"
OUTPUT_DIR = Path("analysis/directional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FEATURES = 15000
MIN_DF = 2
MAX_DF = 0.95
TRAIN_PCT = 0.80

# Targets
TARGETS = [
    "direction_10m",
    "direction_60m",
    "direction_close",
    "direction_1d",
    "direction_5d"
]

# Source modes
SOURCE_MODES = [
    "all",
    "news",
    "exclude_opinion"
]

NEWS_SOURCES = [
    "Reuters",
    "Benzinga",
    "GlobeNewswire Inc.",
    "Business Wire",
    "PR Newswire"
]

OPINION_SOURCES = [
    "The Motley Fool",
    "InvestorPlace",
    "Seeking Alpha"
]

# Results
results = []

# Source filter
def filter_sources(df, source_mode):
    if source_mode == "all":
        return df.copy()

    if source_mode == "news":
        return df[df["source"].isin(NEWS_SOURCES)].copy()

    if source_mode == "exclude_opinion":
        return df[~df["source"].isin(OPINION_SOURCES)].copy()

    raise ValueError(f"Unknown source mode: {source_mode}")

# Model function
def run_model(target, source_mode):
    global results

    print()
    print("=" * 80)
    print(f"TARGET      : {target}")
    print(f"SOURCE MODE : {source_mode}")
    print("=" * 80)

    df = pd.read_csv(DATA_FILE)
    df = filter_sources(df, source_mode)

    if len(df) < 100:
        print(f"Skipping {target}/{source_mode}")
        return

    df["published"] = pd.to_datetime(df["published"])
    df = (df.sort_values("published").reset_index(drop=True))
    df = df.dropna(subset=[target])

    if len(df) < 100:
        print(f"Too few rows after filtering for {target}")
        return

    run_name = f"{target.split('_')[-1]}/{source_mode}/"

    MODEL_FILE = (OUTPUT_DIR / f"{run_name}logistic_regression.joblib")
    VECTORIZER_FILE = (OUTPUT_DIR / f"{run_name}tfidf_vectorizer.joblib")
    FEATURE_IMPORTANCE_FILE = (OUTPUT_DIR / f"{run_name}feature_importance.csv")
    PREDICTIONS_FILE = (OUTPUT_DIR / f"{run_name}predictions.csv")
    METRICS_FILE = (OUTPUT_DIR / f"{run_name}metrics.txt")

    y = df[target].astype(int)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2)
    )

    X_text = vectorizer.fit_transform(df["text"].fillna(""))

    # Extra features
    numeric_cols = [
        "hour",
        "weekday",
        "month",
        "word_count",
        "headline_length",
        "description_length"
    ]

    source_cols = [col for col in df.columns if col.startswith("source_")]
    extra_cols = numeric_cols + source_cols
    X_extra = sparse.csr_matrix(df[extra_cols].fillna(0).values)

    # Combine features
    X = sparse.hstack([X_text, X_extra]).tocsr()

    # Split
    split_idx = int(len(df) * TRAIN_PCT)

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    test_dates = (df["published"].iloc[split_idx:].reset_index(drop=True))

    # Train
    model = LogisticRegression(max_iter=15000, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = (model.predict_proba(X_test)[:, 1])

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, zero_division=0)

    # Feature importance
    feature_names = list(vectorizer.get_feature_names_out())
    feature_names.extend(extra_cols)

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_[0]})
    coef_df["abs_coefficient"] = (coef_df["coefficient"].abs())
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    # Predictions CSV
    prediction_df = pd.DataFrame(
        {
            "published": test_dates,
            "actual": y_test.reset_index(drop=True),
            "predicted": predictions,
            "prediction_probability": probabilities
        }
    )

    prediction_df.to_csv(PREDICTIONS_FILE, index=False)

    # Metrics files
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(f"{run_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Rows : {len(df)}\n\n")

        f.write(f"Accuracy : {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall   : {recall:.6f}\n")
        f.write(f"F1 Score : {f1:.6f}\n\n")

        f.write("Confusion Matrix\n")
        f.write("----------------\n")
        f.write(str(conf_matrix))
        f.write("\n\n")

        f.write("Classification Report\n")
        f.write("---------------------\n")
        f.write(class_report)

    # Save model
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    # Collect results
    results.append(
        {
            "target": target,
            "source_mode": source_mode,
            "rows": len(df),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )

    print(
        f"Accuracy={accuracy:.4f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f}"
    )

# Run all combinations
for target in TARGETS:
    for source_mode in SOURCE_MODES:
        try:
            run_model(target,source_mode)

        except Exception as e:
            print()
            print(f"FAILED: {target} / {source_mode}")
            print(e)

# Final summary
summary = pd.DataFrame(results)
summary = summary.sort_values("accuracy", ascending=False)
summary_file = (OUTPUT_DIR / "all_model_results.csv")

summary.to_csv(summary_file, index=False)

print()
print("=" * 80)
print("FINAL LEADERBOARD")
print("=" * 80)

print(summary)

print()
print(f"Saved summary to: {summary_file}")