"""
    util.py
    NLP Prediction API
    Author: Diego Fernando Cardona Pineda
    Date: 15/03/2025
"""

import boto3
import joblib
import os
from io import BytesIO
from sklearn.base import BaseEstimator, TransformerMixin

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = 'nequi-mlops-data-bucket'

# Parámetros del modelo y vectorizer
DATE_MODEL = '2025-03-15'
DATE_VECTORIZER = '2025-03-15_23-40-42'

MODEL_PATH_S3 = f'model/v_{DATE_MODEL}/model_{DATE_MODEL}.pkl'
# VECTORIZER_PATH_S3 = f'model/v_{DATE}/vectorizer_{DATE}.pkl'
VECTORIZER_PATH_S3 = f'model/v_{DATE_VECTORIZER}/vectorizer_{DATE_VECTORIZER}.pkl'

s3_client = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def load_pickle_from_s3(s3_path: str):
    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)
    body = response['Body'].read()
    return joblib.load(BytesIO(body))

# Carga única del modelo y vectorizer
clf: BaseEstimator = load_pickle_from_s3(MODEL_PATH_S3)
vectorizer: TransformerMixin = load_pickle_from_s3(VECTORIZER_PATH_S3)
