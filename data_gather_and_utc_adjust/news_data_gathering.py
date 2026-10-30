from pathlib import Path

import requests
import pandas as pd
import pyarrow
from datetime import datetime, timedelta

API_KEY = "YOUR-OWN-API-KEY"
ticker = "CRWD"

end_date = datetime.now().date()
start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=5)).date()

url = (
    "https://api.polygon.io/v2/reference/news"
    f"?ticker={ticker}"
    f"&published_utc.gte={start_date}"
    f"&published_utc.lte={end_date}"
    f"&limit=1000"
    f"&apiKey={API_KEY}"
)

records = []

while url:
    response = requests.get(url)
    data = response.json()

    for article in data.get("results", []):
        records.append({
            "ticker": ticker,
            "published": pd.to_datetime(article.get("published_utc")),
            "source": article.get("publisher", {}).get("name"),
            "headline": article.get("title"),
            "description": article.get("description")
        })

    next_url = data.get("next_url")

    if next_url:
        url = next_url + f"&apiKey={API_KEY}"
    else:
        url = None

df = pd.DataFrame(records)

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

csv_file = data_dir / f"{ticker}_news_5yrs.csv"
parquet_file = data_dir / f"{ticker}_news_5yrs.parquet"

df.to_csv(csv_file, index=False)
df.to_parquet(parquet_file, index=False)

print(f"\nSaved CSV: {csv_file}")
print(f"Saved Parquet: {parquet_file}")

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head())