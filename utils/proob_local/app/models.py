"""
    models.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    text: List[str]


class PredictionResult(BaseModel):
    prediction: str
    probability: float


class PredictionResponse(BaseModel):
    results: List[PredictionResult]
