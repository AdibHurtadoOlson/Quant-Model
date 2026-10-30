from pathlib import Path
import pandas as pd


current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

adjusted_dir = current_dir.parent / "data" / "utc_adjusted"
adjusted_dir.mkdir(parents=True, exist_ok=True)

NEWS_FILE = data_dir / "news.csv"
PRICE_FILE = data_dir / "price_data.csv"

news = pd.read_csv(NEWS_FILE)
prices = pd.read_csv(PRICE_FILE)

news["published"] = pd.to_datetime(news["published"], utc=True)
prices["date"] = pd.to_datetime(prices["date"], utc=True)

for col in ["headline", "description"]:
    if col in news.columns:
        news[col] = (
            news[col]
            .fillna("")
            .astype(str)
            .str.replace("<br>", " ", regex=False)
        )

news = news.sort_values("published")
prices = prices.sort_values("date")

news.to_csv(adjusted_dir / "UTC_adjusted_news.csv", index=False)
prices.to_csv(adjusted_dir / "UTC_adjusted_price_data.csv", index=False)

print("Done.")
print("Saved:")
print(" - UTC_adjusted_news.csv")
print(" - UTC_adjusted_price_data.csv")

print("\nNews dtype:")
print(news["published"].dtype)

print("\nPrice dtype:")
print(prices["date"].dtype)