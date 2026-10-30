from pathlib import Path
import pandas as pd
import numpy as np


current_dir = Path(__file__).parent
adjusted_dir = current_dir.parent / "data" / "utc_adjusted"
adjusted_dir.mkdir(parents=True, exist_ok=True)

target_dir = current_dir.parent / "data" / "targets"
target_dir.mkdir(parents=True, exist_ok=True)

NEWS_FILE = adjusted_dir / "UTC_adjusted_news.csv"
PRICE_FILE = adjusted_dir / "UTC_adjusted_price_data.csv"

OUTPUT_FILE = target_dir /  "article_training_dataset.csv"

# Load data
print("Loading data...")

news = pd.read_csv(NEWS_FILE)
prices = pd.read_csv(PRICE_FILE)

news["published"] = pd.to_datetime(news["published"], utc=True)
prices["date"] = pd.to_datetime(prices["date"], utc=True)

news = news.sort_values("published").reset_index(drop=True)
prices = prices.sort_values("date").reset_index(drop=True)

# Remove unwanted parts from text
for col in ["headline", "description"]:
    news[col] = (
        news[col]
        .fillna("")
        .astype(str)
        .str.replace("<br>", " ", regex=False)
    )

news["text"] = (news["headline"] + " " + news["description"])

# Create price targets
print("Creating targets...")

# Intra-day timeframe
prices["ret_10m"] = (prices["close"].shift(-10) / prices["close"] - 1)
prices["ret_60m"] = (prices["close"].shift(-60) / prices["close"] - 1)

prices["trade_day"] = prices["date"].dt.date

daily_close = (
    prices.groupby("trade_day")["close"]
    .last()
    .rename("day_close")
)

prices = prices.merge(
    daily_close,
    left_on="trade_day",
    right_index=True,
    how="left"
)

prices["ret_close"] = (prices["day_close"] / prices["close"] - 1)

daily = (
    prices.groupby("trade_day")
    .agg(day_close=("close", "last"))
    .reset_index()
)

daily["next_day_close"] = daily["day_close"].shift(-1)
daily["five_day_close"] = daily["day_close"].shift(-5)

daily["ret_1d"] = (daily["next_day_close"] / daily["day_close"] - 1)
daily["ret_5d"] = (daily["five_day_close"] / daily["day_close"] - 1)

prices = prices.merge(
    daily[
        [
            "trade_day",
            "ret_1d",
            "ret_5d"
        ]
    ],
    on="trade_day",
    how="left"
)

# Direction targets
return_cols = [
    "ret_10m",
    "ret_60m",
    "ret_close",
    "ret_1d",
    "ret_5d"
]

for col in return_cols:
    prices[col.replace("ret_", "direction_")] = (prices[col] > 0).astype("Int64")


print("Matching articles to bars...")

article_dataset = pd.merge_asof(
    news.sort_values("published"),
    prices[
        [
            "date",
            "close",
            "ret_10m",
            "ret_60m",
            "ret_close",
            "ret_1d",
            "ret_5d",
            "direction_10m",
            "direction_60m",
            "direction_close",
            "direction_1d",
            "direction_5d"
        ]
    ].sort_values("date"),
    left_on="published",
    right_on="date",
    direction="forward"
)

# Article features
print("Creating engineered features...")

article_dataset["hour"] = article_dataset["published"].dt.hour
article_dataset["weekday"] = article_dataset["published"].dt.dayofweek
article_dataset["month"] = article_dataset["published"].dt.month

article_dataset["headline_length"] = (article_dataset["headline"].str.len())
article_dataset["description_length"] = (article_dataset["description"].str.len())

article_dataset["word_count"] = (
    article_dataset["text"]
    .str.split()
    .str.len()
)

# Article source as a feature
source_dummies = pd.get_dummies(
    article_dataset["source"],
    prefix="source",
    dtype=int
)

article_dataset = pd.concat([article_dataset, source_dummies], axis=1)


article_dataset = article_dataset.dropna(
    subset=[
        "ret_10m",
        "ret_60m",
        "ret_close",
        "ret_1d",
        "ret_5d"
    ]
)

# Save output
article_dataset.to_csv(OUTPUT_FILE,index=False)

# Print to console summary
print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Articles retained: {len(article_dataset):,}")
print(f"Saved: {OUTPUT_FILE}")

print()
print("Available targets:")
print(
    [
        "ret_10m",
        "ret_60m",
        "ret_close",
        "ret_1d",
        "ret_5d",
        "direction_10m",
        "direction_60m",
        "direction_close",
        "direction_1d",
        "direction_5d"
    ]
)

print()
print("Example source features:")
print(
    [
        col
        for col in article_dataset.columns
        if col.startswith("source_")
    ][:10]
)