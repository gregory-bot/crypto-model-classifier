# Crypto Buy/Sell Classification — Capstone Project

A reproducible end-to-end pipeline that builds a Buy / Sell / Hold classifier for cryptocurrency markets using historical OHLCV data from the Binance API. The repository includes data ingestion, cleaning, feature engineering, label generation, model training, evaluation, backtesting and model serialization.

## Key Features
- Fetches real Binance OHLCV data (configurable symbol / interval)
- Stores raw and processed datasets under `data/`
- Computes technical indicators and engineered features
- Rule-based label generation (Buy / Sell / Hold)
- Train/evaluate multiple classifiers (Random Forest, XGBoost, LightGBM, etc.)
- Backtesting utilities and model serialization

## Repository structure
```
crypto-classifier/
├── data/
│   ├── raw/             # Raw CSV from API
│   └── processed/       # Cleaned data used for modeling
├── notebooks/           # EDA and experiments
├── src/                 # Scripts and reusable modules
│   ├── data_fetcher.py
│   ├── feature_generator.py
│   ├── labeler.py
│   ├── train.py
│   └── predict.py
├── models/              # Serialized model artifacts
├── README.md
└── requirements.txt
```

## Quickstart

1. Create and activate a virtual environment, then install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Fetch raw data (saves to `data/raw/`):
```powershell
python src\data_fetcher.py
```

3. Clean and generate features (saves to `data/processed/`):
```powershell
python src\feature_generator.py
```

4. Train model:
```powershell
python src\train.py
```

5. Predict / evaluate:
```powershell
python src\predict.py
```

## Data
- Source: Binance `/api/v3/klines` (OHLCV)
- Raw CSV saved to `data/raw/`
- Cleaned/processed CSV saved to `data/processed/` with:
  - timestamps converted to datetime
  - numeric columns cast to float
  - unnecessary columns removed
  - sensible index (e.g., open_time)

## Feature examples
- 1-day and 7-day returns
- Rolling volatility
- SMA20, SMA50, SMA200
- RSI, MACD, Bollinger Bands, Stochastic Oscillator

(Use `ta` or `ta-lib` to compute indicators; see `notebooks/` for examples.)

## Labeling (example rule)
- BUY  = next_day_return > +2%
- SELL = next_day_return < −2%
- HOLD = otherwise

Implementation example:
```python
df["future_return"] = df["close"].pct_change().shift(-1)
df["label"] = df["future_return"].apply(lambda r: 2 if r>0.02 else (0 if r<-0.02 else 1))
```

## Evaluation & Backtest
- Metrics: accuracy, macro F1, precision/recall for BUY, confusion matrix
- Backtesting: simulate strategy (starting capital, position sizing, transaction costs) using model signals; compare vs buy-and-hold baseline

## Contributing
- Keep raw data in `data/raw/` and processed outputs in `data/processed/`
- Add notebooks and experiments under `notebooks/`
- Place trained artifacts under `models/`
- Open issues or PRs with reproducible steps

## License
MIT License — include a LICENSE file if required.

## Contact
Open an issue in this repository for questions, suggestions or reproducibility requests.
```// filepath: c:\Users\HP\Desktop\crypto-classifier\README.md
# Crypto Buy/Sell Classification — Capstone Project

A reproducible end-to-end pipeline that builds a Buy / Sell / Hold classifier for cryptocurrency markets using historical OHLCV data from the Binance API. The repository includes data ingestion, cleaning, feature engineering, label generation, model training, evaluation, backtesting and model serialization.

## Key Features
- Fetches real Binance OHLCV data (configurable symbol / interval)
- Stores raw and processed datasets under `data/`
- Computes technical indicators and engineered features
- Rule-based label generation (Buy / Sell / Hold)
- Train/evaluate multiple classifiers (Random Forest, XGBoost, LightGBM, etc.)
- Backtesting utilities and model serialization

## Repository structure
```
crypto-classifier/
├── data/
│   ├── raw/             # Raw CSV from API
│   └── processed/       # Cleaned data used for modeling
├── notebooks/           # EDA and experiments
├── src/                 # Scripts and reusable modules
│   ├── data_fetcher.py
│   ├── feature_generator.py
│   ├── labeler.py
│   ├── train.py
│   └── predict.py
├── models/              # Serialized model artifacts
├── README.md
└── requirements.txt
```

## Quickstart

1. Create and activate a virtual environment, then install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Fetch raw data (saves to `data/raw/`):
```powershell
python src\data_fetcher.py
```

3. Clean and generate features (saves to `data/processed/`):
```powershell
python src\feature_generator.py
```

4. Train model:
```powershell
python src\train.py
```

5. Predict / evaluate:
```powershell
python src\predict.py
```

## Data
- Source: Binance `/api/v3/klines` (OHLCV)
- Raw CSV saved to `data/raw/`
- Cleaned/processed CSV saved to `data/processed/` with:
  - timestamps converted to datetime
  - numeric columns cast to float
  - unnecessary columns removed
  - sensible index (e.g., open_time)

## Feature examples
- 1-day and 7-day returns
- Rolling volatility
- SMA20, SMA50, SMA200
- RSI, MACD, Bollinger Bands, Stochastic Oscillator

(Use `ta` or `ta-lib` to compute indicators; see `notebooks/` for examples.)

## Labeling (example rule)
- BUY  = next_day_return > +2%
- SELL = next_day_return < −2%
- HOLD = otherwise

Implementation example:
```python
df["future_return"] = df["close"].pct_change().shift(-1)
df["label"] = df["future_return"].apply(lambda r: 2 if r>0.02 else (0 if r<-0.02 else 1))
```

##