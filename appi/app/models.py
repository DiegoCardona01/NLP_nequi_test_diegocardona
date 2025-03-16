"""
    models.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
