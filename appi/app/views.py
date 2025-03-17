"""
    views.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

from .models import PredictionRequest, PredictionResponse
from .util import clf, vectorizer
import logging

map_predictions = {
    '0': 'Credit Reporting',
    '1': 'Debt Collection',
    '2': 'Loans',
    '3': 'Credit Card Services',
    '4': 'Bank Accounts and Services'
}


def predecir_texto(texto: list):
    texto_vectorizado = vectorizer.transform(texto)

    predicciones_indices = clf.predict(texto_vectorizado).tolist()

    predicciones_labels = [map_predictions[str(i)] for i in predicciones_indices]

    logging.info(clf.predict(texto_vectorizado))
    logging.info(type(predicciones_labels))
    logging.info(predicciones_labels)

    probabilidades_array = clf.predict_proba(texto_vectorizado)

    logging.info(f"Probabilidades array: {probabilidades_array}")

    probabilidades_predichas = []

    for i, pred_idx in enumerate(predicciones_indices):
        prob = probabilidades_array[i][pred_idx]
        probabilidades_predichas.append(prob)

    return list(zip(predicciones_labels, probabilidades_predichas))


def get_prediction(request: PredictionRequest) -> PredictionResponse:
    texto = request.text
    predicciones_proba = predecir_texto(texto)
    results = [
        {
            'prediction': pred,
            'probability': prob
        } for pred, prob in predicciones_proba
    ]

    return PredictionResponse(results=results)
