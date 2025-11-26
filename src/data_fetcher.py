import requests
import pandas as pd
from pathlib import Path

def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","num_trades",
                  "taker_base_volume","taker_quote_volume","ignore"]
    return df

df = fetch_binance()
print(df.head())

# save data to csv file (use Path to avoid backslash escape issues and ensure dir exists)
output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "crypto_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved CSV to: {output_path}")
# ...existing code...