import joblib
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_PATH = './model_local/model_2025-03-14.pkl'

VECTORIZER_PATH = './model_local/vectorizer_2025-03-14.pkl'


def load_pickle_local(file_path: str):
    with open(file_path, 'rb') as f:
        return joblib.load(f)


clf: BaseEstimator = load_pickle_local(MODEL_PATH)
vectorizer: TransformerMixin = load_pickle_local(VECTORIZER_PATH)
