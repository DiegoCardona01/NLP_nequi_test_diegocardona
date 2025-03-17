"""
    views.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

from .models import PredictionRequest, PredictionResponse
from .util import clf, vectorizer

map_predictions = {
    '0': 'Credit Reporting',
    '1': 'Debt Collection',
    '2': 'Loans',
    '3': 'Credit Card Services',
    '4': 'Bank Accounts and Services'
}


def predecir_texto(texto: str):
    texto_vectorizado = vectorizer.transform([texto])

    prediccion = int((clf.predict(texto_vectorizado)[0]))
    prediccion = map_predictions[str(prediccion)]
    print(clf.predict(texto_vectorizado))
    print(type(prediccion))
    print(prediccion)

    probabilidades_array = clf.predict_proba(texto_vectorizado)[0]
    clases = clf.classes_
    print(f"Clases: {clases}")
    print(f"Probabilidades array: {probabilidades_array}")

    probabilidades = {int(clase): float(prob) for clase, prob in zip(clases, probabilidades_array)}

    print(f"Probabilidades: {probabilidades}")

    return prediccion, probabilidades


def get_prediction(request: PredictionRequest) -> PredictionResponse:
    texto = request.text
    prediccion, probabilidades = predecir_texto(texto)

    return PredictionResponse(
        prediction=str(prediccion),
        probability=probabilidades[0]
    )
