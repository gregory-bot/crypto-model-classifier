from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load the trained model
model = joblib.load("C:/Users/HP/Desktop/crypto-classifier/models/best.pkl")

# Label mapping
label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Home route
@app.get("/")
def home():
    return {"message": "Crypto Classifier API is running"}

# Input schema for a single data point
class InputData(BaseModel):
    close: float
    volume: float
    num_trades: float
    one_day_return: float
    volatility_7d: float
    macd: float
    sma20: float
    bb_high: float
    stochastic: float

# Output schema for predictions
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

# Prediction route for batch input
@app.post("/predict")
def predict(data: List[InputData]):
    # Prepare all features for batch prediction
    features = np.array([[d.close, d.volume, d.num_trades, d.one_day_return,
                          d.volatility_7d, d.macd, d.sma20, d.bb_high, d.stochastic] 
                         for d in data])

    # Predict numeric labels
    predictions_numeric = model.predict(features)

    # Get probabilities for each prediction
    probabilities = model.predict_proba(features)

    # Create a response list with mapped predictions and their confidence
    predictions = []
    for i, pred_numeric in enumerate(predictions_numeric):
        pred_label = label_map[pred_numeric]
        confidence = float(probabilities[i][pred_numeric])
        predictions.append(PredictionOutput(prediction=pred_label, confidence=confidence))

    return {"predictions": predictions}
