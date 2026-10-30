import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock, util


# Config
TICKER = "CRWD"

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = data_dir / f"{TICKER}_1min_5years.csv"
CHECKPOINT_FILE = data_dir / f"{TICKER}_checkpoint.txt"

NORMAL_DELAY = 5
ERROR_DELAY = 60

# Connect to IB Gateway
ib = IB()
ib.connect("127.0.0.1",4001, clientId=10)
contract = Stock(TICKER,"SMART","USD")
ib.qualifyContracts(contract)

# Date range
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5 * 365)

# Checkpointing
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        current_target = datetime.fromisoformat(
            f.read().strip()
        )

    print(f"Resuming from {current_target}")

else:
    current_target = END_DATE

print(f"Downloading {TICKER}")
print(f"Start Date: {START_DATE}")
print(f"End Date:   {END_DATE}")

# CSV header
write_header = not os.path.exists(OUTPUT_CSV)

# Main loop
while current_target > START_DATE:
    request_time = current_target.strftime("%Y%m%d %H:%M:%S")
    print(f"\nRequesting day ending {request_time}")

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=request_time,
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False
        )

        if not bars:
            print("No bars returned.")

            current_target -= timedelta(days=1)
            time.sleep(NORMAL_DELAY)
            continue

        df = util.df(bars)

        if df.empty:
            print("Empty dataframe returned.")

            current_target -= timedelta(days=1)
            time.sleep(NORMAL_DELAY)
            continue

        # Remove timezone information if present
        try:
            df["date"] = (df["date"].dt.tz_convert(None))

        except Exception:
            pass

        df = df.sort_values("date")
        df.insert(0, "ticker", TICKER)

        earliest_bar = df["date"].min()
        latest_bar = df["date"].max()

        print(f"Downloaded {len(df)} bars")
        print(f"Range: {earliest_bar} -> {latest_bar}")

        # Save data
        df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

        write_header = False
        previous_target = current_target
        current_target = (earliest_bar.to_pydatetime() - timedelta(seconds=1))

        print(f"Moving target:")
        print(f"    From: {previous_target}")
        print(f"    To:   {current_target}")

        # Safety check
        if current_target >= previous_target:
            print("\nWARNING: Target did not move backwards.")
            print("Backing up one day manually.")

            current_target = (previous_target - timedelta(days=1))

        # Save checkpoint
        with open(CHECKPOINT_FILE,"w") as f:
            f.write(current_target.isoformat())

        print(f"Sleeping {NORMAL_DELAY} seconds...")

        time.sleep(NORMAL_DELAY)

    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Sleeping {ERROR_DELAY} seconds...")

        time.sleep(ERROR_DELAY)


# Cleanup
ib.disconnect()

print(f"\nFinished downloading {TICKER}")