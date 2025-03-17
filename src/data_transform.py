# Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import boto3
from io import BytesIO
from datetime import datetime
import logging

S3_BUCKET_NAME = 'nequi-mlops-data-bucket'
s3_client = boto3.client('s3')


def load_data_from_s3(s3_path: str) -> pd.DataFrame:
    """Carga los datos desde S3 y devuelve un DataFrame"""
    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)
    body = response['Body'].read()
    return pd.read_csv(BytesIO(body))


def save_pickle_to_s3(obj, s3_path: str) -> None:
    """Guarda un objeto como pickle en S3"""
    buf = BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    s3_client.upload_fileobj(buf, S3_BUCKET_NAME, s3_path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el DataFrame aplicando limpieza y transformación de texto"""
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english'  # Stopwords de sklearn
    )

    X_tdfidf = tfidf.fit_transform(df['narrative'])

    logging.info(f"TF-IDF matrix shape: {X_tdfidf.shape}")

    return tfidf, X_tdfidf


def main():
    """Función principal que ejecuta el flujo completo de carga, procesamiento y guardado de datos"""
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    logging.info("Cargando los datos desde S3...")
    df_sample_v2 = load_data_from_s3('silver/cleaned_dataset.csv')

    logging.info("Preprocesando los datos...")
    tfidf, X_tdfidf = prepare_data(df_sample_v2)

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vectorizer_dir = f'model/v_{date}'
    data_transform_dir = 'silver'
    vectorizer_path = f'{vectorizer_dir}/vectorizer_{date}.pkl'
    matrix_path = f'{data_transform_dir}/tfidf_matrix_{date}.pkl'

    logging.info(f'Vectorizador guardado en: {vectorizer_path}')
    save_pickle_to_s3(tfidf, vectorizer_path)

    logging.info(f'Matriz TF-IDF guardada en: {matrix_path}')
    save_pickle_to_s3(X_tdfidf, matrix_path)

    logging.info("Proceso completado exitosamente.")


if __name__ == '__main__':
    main()
