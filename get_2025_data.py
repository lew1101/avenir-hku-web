import math
import time
import requests  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime, timezone, timedelta
import os

parquet_name_list = os.listdir("kline_data/train_data")
symbol_list = [
    parquet_name.split(".")[0] for parquet_name in parquet_name_list
]

# for i, symbol in enumerate(symbol_list):
#     if symbol == "1000000MOGUSDT":
#         symbol_list[i] = "1000MOGUSDT"  # fix for a specific symbol that has a different format
        
        
# print(f"Symbols in train data: {symbol_list}")

# ---- Config ----
# SYMBOL = "USDTUSD"  # Coinbase product id
GRANULARITY = 900  # seconds: 60(1m), 300(5m), 900(15m), 3600(1h)
START_MS = 1735689600000  # 2025-01-01 00:00:00 UTC
END_MS = 1751336700000  # 2025-06-30 23:45:00 UTC

INTERVAL = "15m"
BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, "2025_data")


os.makedirs(SAVE_DIR, exist_ok=True)  # ensure cache directory exists
# OUT_PATH = os.path.join(SAVE_DIR, "USDT_USD.parquet")
API_URL = "https://api.binance.us"


def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000, sleep=0.2):
    url = f"{API_URL}/api/v3/klines"
    out = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        # advance to just after the last candle's open time to avoid duplicates
        cur = rows[-1][0] + 1
        time.sleep(sleep)
    return out

for symbol in symbol_list:
    try:
        raw = fetch_klines(symbol, INTERVAL, START_MS, END_MS)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        continue

    # Binance kline columns:
    # 0 open_time(ms),1 open,2 high,3 low,4 close,5 volume(base),6 close_time,7 quote_asset_volume,
    # 8 number_of_trades,9 taker_buy_base,10 taker_buy_quote,11 ignore
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Map to your schema; stringify prices/volumes to match your example
    out = pd.DataFrame(
        {
            "timestamp": df["open_time"].astype("int64"),
            "open_price": df["open"].astype(str),
            "high_price": df["high"].astype(str),
            "low_price": df["low"].astype(str),
            "close_price": df["close"].astype(str),
            "volume": df["volume"].astype(str),  # base asset volume
            "amount": df["quote_volume"].astype(str),  # quote asset volume
            "count": df["trades"].astype("int64"),
            "buy_volume": df["taker_buy_base"].astype(str),
            "buy_amount": df["taker_buy_quote"].astype(str),
        }
    )

    # Sort & dedupe just in case
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Save Parquet
    out.to_parquet(os.path.join(SAVE_DIR, f"{symbol}.parquet"), engine="pyarrow", index=False)
    print(f"Saved {len(out):,} rows to {os.path.join(SAVE_DIR, f"{symbol}.parquet")}")
