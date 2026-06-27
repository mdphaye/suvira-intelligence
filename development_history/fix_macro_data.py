"""
fix_macro_data.py
-----------------
One-time script to replace crude_oil_price and natural_gas_price
in FINAL_sentiment_macro.csv with real Yahoo Finance monthly data.

Run from your project root:
    python fix_macro_data.py
"""

import os
import requests
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_CSV = os.path.join(BASE_DIR, "data", "FINAL_sentiment_macro.csv")
BACKUP_CSV    = os.path.join(BASE_DIR, "data", "FINAL_sentiment_macro_BACKUP.csv")


# ── Step 1: Fetch monthly data from Yahoo Finance ─────────────────────────────
def fetch_yahoo_monthly(ticker: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch monthly OHLCV from Yahoo Finance v8 API, no yfinance needed."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": int(pd.Timestamp.now().timestamp()),
        "interval": "1mo",
        "events":   "history",
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DataFixBot/1.0)"}

    print(f"  Fetching {ticker} from Yahoo Finance...")
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()

    result     = r.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes     = result["indicators"]["quote"][0]["close"]

    # Snap to month-start (e.g. 2022-03-15 → 2022-03-01)
    dates = (
        pd.to_datetime(timestamps, unit="s")
          .to_period("M")
          .to_timestamp()
    )

    df = (pd.DataFrame({"date": dates, "value": closes})
            .dropna()
            .reset_index(drop=True))

    print(f"  Got {len(df)} monthly rows for {ticker} "
          f"({df['date'].min().strftime('%b %Y')} → "
          f"{df['date'].max().strftime('%b %Y')})")
    return df


# ── Step 2: Load dataset ───────────────────────────────────────────────────────
print("\n1. Loading FINAL_sentiment_macro.csv...")
df = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
print(f"   {len(df)} rows loaded. "
      f"Date range: {df['date'].min().strftime('%b %Y')} → "
      f"{df['date'].max().strftime('%b %Y')}")

# ── Step 3: Create backup ──────────────────────────────────────────────────────
print("\n2. Creating backup...")
df.to_csv(BACKUP_CSV, index=False)
print(f"   Backup saved → {BACKUP_CSV}")

# ── Step 4: Fetch Yahoo Finance data ──────────────────────────────────────────
print("\n3. Fetching live macro data from Yahoo Finance...")
crude  = fetch_yahoo_monthly("CL=F")   # WTI Crude Oil
natgas = fetch_yahoo_monthly("NG=F")   # Henry Hub Natural Gas

# Build lookup dicts: date → price
crude_map  = dict(zip(crude["date"],  crude["value"]))
natgas_map = dict(zip(natgas["date"], natgas["value"]))

# ── Step 5: Replace values row by row ─────────────────────────────────────────
print("\n4. Replacing crude_oil_price and natural_gas_price values...")

replaced_crude  = 0
replaced_natgas = 0
kept_crude      = 0
kept_natgas     = 0

for idx, row in df.iterrows():
    d = row["date"]

    # Crude oil
    if d in crude_map and pd.notna(crude_map[d]):
        df.at[idx, "crude_oil_price"] = round(crude_map[d], 4)
        replaced_crude += 1
    else:
        kept_crude += 1

    # Natural gas
    if d in natgas_map and pd.notna(natgas_map[d]):
        df.at[idx, "natural_gas_price"] = round(natgas_map[d], 4)
        replaced_natgas += 1
    else:
        kept_natgas += 1

print(f"   Crude Oil    → replaced: {replaced_crude} rows | "
      f"kept original: {kept_crude} rows")
print(f"   Natural Gas  → replaced: {replaced_natgas} rows | "
      f"kept original: {kept_natgas} rows")

# ── Step 6: Save corrected CSV ─────────────────────────────────────────────────
print("\n5. Saving corrected CSV...")
df.to_csv(SENTIMENT_CSV, index=False)
print(f"   Saved → {SENTIMENT_CSV}")

# ── Step 7: Quick sanity check ────────────────────────────────────────────────
print("\n6. Sanity check — sample of replaced values:")
sample = df[["date", "chemical", "crude_oil_price", "natural_gas_price"]].drop_duplicates("date").head(10)
print(sample.to_string(index=False))

print("\n✅ Done. Your dataset now has real Yahoo Finance macro data from 2020 onwards.")
print(f"   Backup of original kept at: {BACKUP_CSV}")