"""
    main.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

from fastapi import FastAPI
from app.models import PredictionRequest, PredictionResponse
from app.views import get_prediction

app = FastAPI(
    title="NLP Text Classification API",
    description="API para clasificar texto con un modelo NLP desde S3",
    version="1.0"
)


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Recibe un texto y devuelve la predicción de su clase junto a las probabilidades.
    """
    return get_prediction(request)
