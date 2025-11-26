# Crypto Buy/Sell Classification Capstone Project

## End-to-End Machine Learning Pipeline Using Real Binance Data

---

## 1. Project Overview

This capstone project builds a **Buy/Sell/Hold classifier** for cryptocurrency markets using **real data from the Binance API**.  
The goal is to create a fully functioning ML system that:

- Fetches real historical crypto data
- Cleans and engineers features
- Calculates technical indicators
- Generates labels for Buy/Sell/Hold
- Trains a classification model
- Evaluates performance and backtests strategy
- Serializes the trained model
- Deploys prediction logic (optional)

This workflow simulates a realistic applied-finance ML project.

---

## 2. Data Sources

### Binance API

- Endpoint: `/api/v3/klines`
- Provides OHLCV (Open, High, Low, Close, Volume)
- Supports multiple intervals (1m, 1h, 1d)

Example URL: https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000

## 3. Project Structure
crypto-classifier/
│── data/
│ ├── raw/
│ ├── processed/
│── notebooks/
│ ├── 01_fetch_data.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_evaluation.ipynb
│── src/
│ ├── data_fetcher.py
│ ├── feature_generator.py
│ ├── labeler.py
│ ├── train.py
│ ├── predict.py
│── models/
│── README.md
│── requirements.txt


---

## 4. Step-by-Step Documentation

### Step 1 — Fetch Data From Binance

**`data_fetcher.py`**:  
Fetch raw OHLCV data, convert to DataFrame, save CSV in `data/raw/`.

```python
import requests
import pandas as pd

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
    

Step 2 — Data Cleaning & Processing

Convert timestamps → datetime

Numeric columns → float

Drop unused columns
      df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
df["close"] = df["close"].astype(float)

Store cleaned data in data/processed/.



Step 3 — Feature Engineering
Returns

1-day return

7-day return

Rolling volatility

Technical Indicators (using ta or ta-lib)

RSI

MACD

SMA20, SMA50, SMA200

Bollinger Bands

Stochastic Oscillator
     
     import ta

df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
df["sma_20"] = df["close"].rolling(20).mean()


Step 4 — Label Generation (Target Variable)

Rule-based labeling: 
     If next_day_return > +2% → BUY  
If next_day_return < –2% → SELL  
Else → HOLD


     df["future_return"] = df["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1

df["label"] = df.apply(label, axis=1)


Step 5 — Train/Test Split

70% training

15% validation

15% test

No shuffling (time-series data)

Step 6 — Model Training

Test multiple classifiers:

Logistic Regression

Random Forest

XGBoost / LightGBM / CatBoost

LSTM/GRU (optional)
    
    from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)


Step 7 — Evaluation

Metrics:

Accuracy

Macro F1-score

Precision/Recall for BUY class

Confusion matrix

Backtesting: simulate strategy starting with $10,000.

Step 8 — Serialize the Model
      import joblib
joblib.dump(model, "models/buy_sell_classifier.pkl")

Step 9 — Prediction Pipeline
      def predict(features):
    model = joblib.load("models/buy_sell_classifier.pkl")
    return model.predict(features)

