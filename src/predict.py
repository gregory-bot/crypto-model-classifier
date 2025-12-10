import joblib
import pandas as pd

# -----------------------------
# Load the saved XGBoost model
# -----------------------------
model = joblib.load("C:/Users/HP/Desktop/crypto-classifier/models/buy_sell_classifier.pkl")

# -----------------------------
# Feature order must match training
# -----------------------------
FEATURE_COLUMNS = [
    "close",
    "volume",
    "num_trades",
    "1_day_return",
    "7_day_volatility",
    "stochastic_oscillator",  # make sure this is in the correct place
    "macd",
    "sma20",
    "bb_high"
]

# -----------------------------
# Preprocess input
# -----------------------------
def preprocess_input(features_dict):
    """
    Convert input dict -> DataFrame -> correct order -> numeric
    """
    df = pd.DataFrame([features_dict])   # single row
    df = df[FEATURE_COLUMNS]             # enforce correct order
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# -----------------------------
# Predict function
# -----------------------------
def predict(features_dict):
    """
    Input: dict of features
    Output: prediction class (0=sell, 1=hold, 2=buy) + probabilities
    """
    X = preprocess_input(features_dict)

    # Convert to numpy array to avoid XGBoost feature name mismatch
    X_np = X.values

    pred = model.predict(X_np)[0]
    proba = model.predict_proba(X_np)[0]

    return {
        "prediction": int(pred),
        "probabilities": {
            "SELL_0": float(proba[0]),
            "HOLD_1": float(proba[1]),
            "BUY_2": float(proba[2])
        }
    }

# -----------------------------
# Local testing
# -----------------------------
if __name__ == "__main__":
    sample = {
        "close": 45000,
        "volume": 1200,
        "num_trades": 450,
        "1_day_return": 0.002,
        "7_day_volatility": 0.05,
        "stochastic_oscillator": 62,
        "macd": -30,
        "sma20": 44800,
        "bb_high": 45500
    }

    result = predict(sample)
    print("Predicted class:", result["prediction"])
    print("Probabilities:", result["probabilities"])
